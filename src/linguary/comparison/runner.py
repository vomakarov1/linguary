"""Pairwise comparison in isolated copies via existing use cases.

Document read projection limits the experiment to selected chapters. Complete
persisted model, source EPUB, and resources remain immutable.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from pathlib import Path

from linguary.adapters.epub import EpubExportWriter
from linguary.adapters.llm import FakeLLMClient, LoggingLLMClient, RateLimitedLLMClient
from linguary.adapters.validation import Py3LangidDetector
from linguary.adapters.runtime import TaskWorkerPool, WorkerSettings
from linguary.application.context_builder import ContextBuilder, textual_blocks
from linguary.application.retry_policy import RetryPolicy
from linguary.application.task_runner import TaskRunner
from linguary.application.use_cases.analyze_chapter import AnalyzeChapter, AnalyzeChapterCommand
from linguary.application.use_cases.approve_glossary import GlossaryDecision, GlossaryDecisionCommand, StartProcessing, StartProcessingCommand
from linguary.application.use_cases.export_book import ExportBook, ExportBookCommand
from linguary.application.use_cases.process_chapter import ChapterProcessingSequence, ProcessChapter
from linguary.application.use_cases.register_book import RegisterBookCommand
from linguary.comparison.llm import BudgetGuard, CallObservation, ComparisonStopped, ObservationLedger, ObservedLLMClient
from linguary.comparison.protocol import ComparisonManifest, ComparisonProtocol, ValidatedProtocol, build_manifest, file_hash, settings_projection, stable_hash, validate_for_live
from linguary.comparison.workspace import AnalysisArtifact, PartialAnalysisSeed, fork_processing, freeze_analysis, prepare_workspace, load_continuation
from linguary.composition import drain_tasks
from linguary.composition.container import _llm_snapshot, _validator
from linguary.domain import BookEvent, TaskIdentity, TaskType, apply_book_event


@dataclass
class VariantResult:
    variant_id: str
    repeat: int
    analysis_hash: str
    workspace: object
    epubs: dict[str, Path] = field(default_factory=dict)
    epub_chapters: dict[str, list[str]] = field(default_factory=dict)
    completed: bool = False
    failures: list[str] = field(default_factory=list)
    completed_chapters: list[str] = field(default_factory=list)
    llm_snapshots: dict[str, object] = field(default_factory=dict)
    observations: list[CallObservation] = field(default_factory=list)


@dataclass
class ComparisonResult:
    protocol_hash: str
    simulated: bool
    analyses: list[AnalysisArtifact] = field(default_factory=list)
    variants: list[VariantResult] = field(default_factory=list)
    observations: list[CallObservation] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    manifests: dict[str, ComparisonManifest] = field(default_factory=dict)
    stage_wall_seconds: dict[str, float] = field(default_factory=dict)
    stage_wall_segments: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    continuation_provenance: dict[str, object] = field(default_factory=dict)


class _Documents:
    def __init__(self, inner, selection):
        self.inner, self.selection = inner, selection

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def get(self, book_id):
        document = self.inner.get(book_id)
        if document is None or book_id not in self.selection:
            return document
        return replace(document, sections=tuple(s for s in document.sections if s.id in self.selection[book_id]))

    def save(self, *args, **kwargs):
        raise ComparisonStopped('Проекция документа доступна только для чтения')


class _Transaction:
    def __init__(self, inner, selection):
        self.inner = inner
        self.documents = _Documents(inner.documents, selection)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _SelectedUnitOfWork:
    def __init__(self, inner, selection):
        self.inner, self.selection = inner, selection

    @contextmanager
    def transaction(self):
        with self.inner.transaction() as transaction:
            yield _Transaction(transaction, self.selection)


class _ObservedContext(ContextBuilder):
    def __init__(self, ledger, **kwargs):
        super().__init__(**kwargs)
        self.ledger = ledger

    def _chapter_context(self, **kwargs):
        context = super()._chapter_context(**kwargs)
        self.ledger.register_context(context.request, book_id=kwargs['document'].id, chapter_id=kwargs['section'].id, block_ids=context.block_ids_by_alias.values())
        return context

    def _translation_context(self, **kwargs):
        context = super()._translation_context(**kwargs)
        self.ledger.register_context(context.request, book_id=kwargs['document'].id, chapter_id=kwargs['section'].id, block_ids=context.block_ids_by_alias.values())
        return context


class _FixedContext(_ObservedContext):
    def __init__(self, fixed, ledger, **kwargs):
        super().__init__(ledger, **kwargs)
        self.fixed = fixed

    def _chapter_input(self, **kwargs):
        # Replaces the existing previous chapter decisions field; instructions,
        # blocks, schema, and previous fragment results remain standard.
        kwargs['accumulated_candidates'] = ()
        text = super()._chapter_input(**kwargs)
        section = kwargs['section']
        fixed = self.fixed.get(section.id, '')
        return text.replace('Предыдущие главы переводческих решений пока не дали.', fixed or 'Предыдущие главы переводческих решений пока не дали.', 1)


class _SimulatedClient:
    def __init__(self, inner, ledger, model, route):
        if not isinstance(inner, FakeLLMClient):
            raise ComparisonStopped('Имитация требует FakeLLMClient')
        self.inner, self.ledger, self.model, self.route = inner, ledger, model, route

    async def generate(self, request):
        item = self.ledger.begin(request, model=self.model, route=self.route)
        try:
            response = await self.inner.generate(request)
        except Exception as error:
            self.ledger.finish(item, error)
            raise
        item.actual_model, item.actual_provider = response.model, response.provider
        item.cost, item.cost_source = Decimal(0), 'simulation'
        item.full_response_cache = False
        self.ledger.finish(item, response)
        return response


def _stage_client(client, ledger, model, route, simulated, settings, budget, max_call_cost, temperature_policy="set"):
    if simulated:
        return _SimulatedClient(client, ledger, model, route)
    if not isinstance(client, ObservedLLMClient) or client.ledger is not ledger or client.model != model or client.route != route:
        raise ComparisonStopped("Клиент не обеспечивает принятый маршрут и квитанции")
    if client.temperature_policy != temperature_policy:
        ledger.stop("temperature_policy_mismatch")
        raise ComparisonStopped("Клиент изменяет политику temperature")
    if client.budget is not budget or client.maximum != Decimal(str(max_call_cost)):
        ledger.stop("budget_binding_mismatch")
        raise ComparisonStopped("Клиент изменяет общий бюджет или резерв вызова")
    if client.generation_settings != (settings.temperature, settings.max_output_tokens):
        ledger.stop("generation_parameters_mismatch")
        raise ComparisonStopped("Клиент изменяет согласованные параметры генерации")
    return client


def _safe_id(value):
    return re.sub(r'[^A-Za-z0-9_-]', '_', value) + '-' + stable_hash(value)[:8]


def _settings(workspace, model, route):
    workspace.settings = replace(workspace.settings, llm=replace(workspace.settings.llm, model=model, provider=route))
    (workspace.root / "settings.json").write_text(json.dumps(settings_projection(workspace.settings), sort_keys=True))


def _model(application, inner, ledger):
    return RateLimitedLLMClient(inner=LoggingLLMClient(inner=inner, unit_of_work=application.unit_of_work, provider=application.settings.llm.provider, model=application.settings.llm.model, debug_payloads=False, on_logged=ledger.on_logged), limiter=application.rate_limiter)


def _finalize(application, ledger, book_ids):
    statuses = {}
    with application.unit_of_work.transaction() as transaction:
        for book_id in book_ids:
            for item in transaction.llm_calls.list_for_book(book_id):
                statuses[item.id] = item.success
    for operation in {item.operation_id for item in ledger.observations}:
        try:
            ledger.finalize_operation(operation, accepted=None, call_statuses=statuses)
        except ComparisonStopped:
            ledger.stop("observation_finalization_failed")


def _transition(uow, book_id, event):
    with uow.transaction() as transaction:
        book = transaction.books.get(book_id)
        transaction.books.save(replace(book, status=apply_book_event(book.status, event)))


def _selection(application, protocol, sources):
    selection, fixed = {}, {}
    for source_key, book_id in sources.items():
        with application.unit_of_work.transaction() as transaction:
            document = transaction.documents.get(book_id)
        ids = set()
        for chapter in protocol.chapters:
            if file_hash(Path(chapter.source)) != source_key:
                continue
            matches = [s for i, s in enumerate(document.ordered_sections, 1) if chapter.chapter_id in (s.id, s.source_resource_path, str(i))]
            if len(matches) != 1:
                raise ComparisonStopped('Выбранная глава не найдена однозначно')
            section = matches[0]
            context = protocol.previous_context.get(chapter.id, '')
            if section.id in fixed and fixed[section.id] != context:
                raise ComparisonStopped('Разные фиксированные контексты одной главы')
            fixed[section.id] = context
            ids.add(section.id)
        selection[book_id] = ids
    return selection, fixed


def _manifest(workspace, result, key, selection, fixed, analysis_hash=None):
    inputs = json.loads((workspace.root / "inputs.json").read_text())
    manifest = build_manifest(workspace.protocol, {name: workspace.root / value["path"] for name, value in inputs.items()}, workspace.settings)
    manifest = manifest.model_copy(update={"grouping": {"selected_sections": {book: sorted(sections) for book, sections in selection.items()}, "fixed_context_hash": stable_hash(fixed)}, "artifact_hashes": {"analysis": analysis_hash} if analysis_hash else {}, "paths": {"receipts": "receipts", "diagnostics": "diagnostics"}})
    result.manifests[key] = manifest
    (workspace.root / "manifest.json").write_text(manifest.model_dump_json(indent=2))


def _apply_glossary(application, protocol, book_id):
    review = application.approve_glossary.review(book_id)
    decisions = protocol.fixed_glossary
    items = {item.source_text: item for item in review.items}
    for source, decision in decisions.items():
        if source not in items:
            continue
        if not isinstance(decision, dict) or not set(decision) <= {'decision', 'translation', 'note'}:
            raise ComparisonStopped('Формат фиксированного решения непредставим')
        item = items[source]
        action = GlossaryDecision(decision['decision'])
        translation = decision.get('translation')
        if action == GlossaryDecision.ACCEPT and translation != item.suggested_translation:
            raise ComparisonStopped('Предложение модели расходится с фиксированным решением')
        application.approve_glossary(GlossaryDecisionCommand(book_id=book_id, source_text=source, concept_id=item.concept_id, decision=action, translation=translation if action == GlossaryDecision.CORRECT else None, note=decision.get('note')))
    application.approve_glossary.complete_if_ready(book_id)
    if not application.approve_glossary.review(book_id).is_complete:
        raise ComparisonStopped('Глоссарий требует несогласованных решений')


async def _close(client):
    close = getattr(client, 'aclose', None)
    if close is not None:
        await close()


async def run_comparison(protocol, workspace, client_factory) -> ComparisonResult:
    """Factory receives model/stage/settings/ledger/budget; live requires approval.

    For testing, factory explicitly declares simulated=True and returns FakeLLMClient.
    Real factory is called only with a re-checked ValidatedProtocol.
    """
    simulated = getattr(client_factory, 'simulated', False) is True
    if isinstance(protocol, ValidatedProtocol):
        validate_for_live(protocol.protocol, protocol.approval)
        protocol = protocol.protocol
    elif not simulated:
        raise ComparisonStopped('Реальный запуск требует принятого протокола')
    protocol = ComparisonProtocol.model_validate(protocol.model_dump())
    if protocol.content_hash != workspace.protocol.content_hash or not protocol.repeats:
        raise ComparisonStopped('Каталог или число повторов не соответствует протоколу')
    continuation = load_continuation(workspace)
    result = ComparisonResult(protocol.content_hash, simulated)
    budget = BudgetGuard(Decimal(str(protocol.budget)), protocol.max_total_attempts) if protocol.budget else None
    # Copies of identical EPUBs are grouped by content, not file name.
    inputs = json.loads((workspace.root / 'inputs.json').read_text())
    source_paths = {value['sha256']: workspace.root / value['path'] for value in inputs.values()}
    sources = {key: 'book-' + key[:16] for key in source_paths}
    scheduled = protocol.schedule or tuple(v.id for _ in range(protocol.repeats) for v in protocol.variants)
    variants = {v.id: v for v in protocol.variants}
    if len(scheduled) != 4 * protocol.repeats or any(scheduled.count(v.id) != protocol.repeats for v in protocol.variants):
        raise ComparisonStopped('Расписание не соответствует матрице и числу повторов')
    repeats = {v.id: 0 for v in protocol.variants}
    analyses = {}
    terminal = set()
    if continuation is not None:
        artifact = continuation.artifact
        result.analyses.append(artifact)
        result.observations.extend(continuation.observations)
        result.continuation_provenance = continuation.provenance
        result.stage_wall_seconds.update(continuation.stage_wall_seconds)
        stored = continuation.variant
        prior = VariantResult(stored['variant_id'], stored['repeat'], stored['analysis_hash'], None,
                              failures=list(stored['failures']), completed_chapters=list(stored['completed_chapters']),
                              epub_chapters=dict(stored['epub_chapters']),
                              observations=[item for item in continuation.observations if item.variant_id == stored['variant_id']])
        result.variants.append(prior)
        terminal.add((prior.repeat, prior.variant_id))
        analyses[(1, variants[prior.variant_id].analysis_model)] = artifact, continuation.selection
    for variant_id in scheduled:
        variant = variants[variant_id]
        repeats[variant_id] += 1
        repeat = repeats[variant_id]
        if (repeat, variant_id) in terminal:
            continue
        key = repeat, variant.analysis_model
        if key not in analyses:
            seed = continuation.partial_seed if continuation is not None and repeat == 1 and continuation.partial_seed is not None and variant.analysis_model == continuation.partial_seed.observation.assigned_model else None
            stage = f"analysis:{repeat}:{variant.analysis_model}"
            historical = result.stage_wall_seconds.get(stage, 0)
            started = time.monotonic()
            try:
                analyses[key] = await _run_analysis(protocol, workspace, client_factory, result, budget, simulated, sources, repeat, variant.analysis_model, seed=seed)
            finally:
                elapsed = time.monotonic() - started
                result.stage_wall_seconds[stage] = historical + elapsed
                if seed is not None:
                    result.stage_wall_segments[stage] = [{'origin': 'imported', 'seconds': historical}, {'origin': 'current', 'seconds': elapsed}]
        if result.violations and any(not violation.startswith('analysis_failed:') for violation in result.violations):
            break
        prepared = analyses[key]
        if prepared is None:
            result.variants.append(VariantResult(variant.id, repeat, '', None, failures=['analysis_failed']))
            continue
        artifact, selection = prepared
        started = time.monotonic()
        try:
            await _run_processing(protocol, client_factory, result, budget, simulated, sources, repeat, variant, artifact, selection, workspace.root)
        finally:
            result.stage_wall_seconds[f"processing:{repeat}:{variant.id}"] = time.monotonic() - started
        if any(not violation.startswith('analysis_failed:') for violation in result.violations):
            break
    return result


def _import_analysis_seed(seed, application, selection, fixed, ledger, analysis_model, repeat):
    try:
        if not isinstance(seed, PartialAnalysisSeed):
            raise ValueError('Нужен проверенный типизированный анализ')
        fragment, analysis, log, observation = seed.fragment, seed.analysis, seed.log, seed.observation
        book_id, section_id = fragment['book_id'], fragment['section_id']
        if (section_id not in selection.get(book_id, set()) or analysis.section_id != section_id
                or analysis.id != fragment['analysis_id'] or log.book_id != book_id or not log.success
                or observation.assigned_model != analysis_model or observation.repeat != repeat
                or observation.stage != 'chapter_analysis' or log.stage.value != observation.stage or observation.accepted is not True
                or observation.chapter_id != section_id or observation.call_id != log.id
                or log.id != fragment['call_id'] or observation.call_uid != fragment['call_uid']
                or observation.generation_id != fragment['generation_id']
                or observation.operation_id != fragment['operation_id']
                or stable_hash(asdict(analysis)) != fragment['typed_analysis_sha256']):
            raise ValueError('Сохранённый анализ не соответствует доказательству')
        analysis.ensure_ready_for_processing()
        with application.unit_of_work.transaction() as transaction:
            document = transaction.documents.get(book_id)
        section = next(section for section in document.sections if section.id == section_id)
        block_ids = [block.id for block in textual_blocks(section)]
        context_ledger = ObservationLedger(variant_id=ledger.variant_id, repeat=repeat)
        context = _FixedContext(fixed, context_ledger).for_chapter_analysis(document=document, section=section, accumulated_candidates=())
        group = hashlib.sha256(json.dumps([book_id, observation.stage, section_id, block_ids], separators=(',', ':')).encode()).hexdigest()
        operation = hashlib.sha256('\0'.join((ledger.variant_id, str(repeat), book_id, observation.stage, group)).encode()).hexdigest()
        if (block_ids != fragment['block_ids'] or list(analysis.treatment_plan.block_ids) != block_ids
                or hashlib.sha256(context.request.input_text.encode()).hexdigest() != fragment['request_input_sha256']
                or hashlib.sha256(context.request.instructions.encode()).hexdigest() != fragment['request_instructions_sha256']
                or operation != observation.operation_id or group != observation.group_id):
            raise ValueError('Источник или фиксированный контекст изменился')
        with application.unit_of_work.transaction() as transaction:
            transaction.analyses.save(book_id, analysis)
            transaction.llm_calls.add(log)
        return book_id, section_id
    except Exception:
        ledger.stop('partial_seed_mismatch')
        raise


async def _run_analysis(protocol, workspace, client_factory, result, budget, simulated, sources, repeat, analysis_model, seed=None):
    analysis_ws = prepare_workspace(workspace.root / f'analysis-{repeat}-{_safe_id(analysis_model)}', workspace.protected_roots, protocol)
    route = protocol.routes.get(analysis_model, 'fake')
    _settings(analysis_ws, analysis_model, route)
    ledger = ObservationLedger(variant_id=f'analysis-{analysis_model}', repeat=repeat, receipt_root=analysis_ws.root / 'receipts', diagnostic_root=analysis_ws.root / 'diagnostics', retain_rejected=protocol.retain_rejected is True, retention_seconds=protocol.diagnostic_retention_seconds or 0)
    client = application = None
    try:
        client = client_factory(model=analysis_model, stage='analysis', settings=analysis_ws.settings.llm, ledger=ledger, budget=budget)
        inner = _stage_client(client, ledger, analysis_model, route, simulated, analysis_ws.settings.llm, budget, protocol.max_call_cost, protocol.temperature_policy or "set")
        application = analysis_ws.build_application(llm_client=inner, on_logged=ledger.on_logged)
        for source_key, book_id in sources.items():
            local_inputs = json.loads((analysis_ws.root / 'inputs.json').read_text())
            source = next(analysis_ws.root / item['path'] for item in local_inputs.values() if item['sha256'] == source_key)
            application.register_book(RegisterBookCommand(book_id, str(source), source.stem, 'en', 'ru'))
            await application.start_import(book_id)
        imported = await drain_tasks(application)
        if any(task.last_error is not None for task in imported):
            raise ComparisonStopped('Импорт источника не завершён')
        selection, fixed = _selection(application, protocol, sources)
        seeded_section = _import_analysis_seed(seed, application, selection, fixed, ledger, analysis_model, repeat) if seed is not None else None
        _manifest(analysis_ws, result, f"analysis:{repeat}:{analysis_model}", selection, fixed)
        selected_uow = _SelectedUnitOfWork(application.unit_of_work, selection)
        analyze = AnalyzeChapter(unit_of_work=selected_uow, llm_client=_model(application, inner, ledger), language_detector=Py3LangidDetector(), retry_policy=RetryPolicy(max_attempts=application.settings.llm.max_attempts), glossary_confidence_threshold=application.settings.checks.glossary_confidence_threshold, max_fragment_blocks=application.settings.llm.effective_max_fragment_blocks, context_builder=_FixedContext(fixed, ledger))
        async def analyze_task(task):
            await analyze(AnalyzeChapterCommand(task.book_id, task.scope_id, task.id))
        for book_id, section_ids in selection.items():
            _transition(selected_uow, book_id, BookEvent.ANALYSIS_STARTED)
            with selected_uow.transaction() as transaction:
                document = transaction.documents.get(book_id)
            for section in document.ordered_sections:
                if (book_id, section.id) == seeded_section:
                    continue
                task = await application.task_queue.enqueue(TaskIdentity(book_id, TaskType.ANALYZE_CHAPTER, section.id))
                ledger.task_scopes[task.id] = section.id
        tasks = await TaskWorkerPool(TaskRunner(queue=application.task_queue, handlers={TaskType.ANALYZE_CHAPTER: analyze_task}), WorkerSettings(concurrency=1)).drain()
        if any(task.last_error is not None for task in tasks):
            raise ComparisonStopped('Анализ выбранной главы не завершён')
        for book_id in sources.values():
            with selected_uow.transaction() as transaction:
                analyses = transaction.analyses.list_for_book(book_id)
            if {a.section_id for a in analyses} != selection[book_id] or not all(a.is_complete for a in analyses):
                raise ComparisonStopped('Неполное покрытие выбранных глав')
            _transition(selected_uow, book_id, BookEvent.ANALYSIS_COMPLETED)
            _apply_glossary(application, protocol, book_id)
        _finalize(application, ledger, sources.values())
        ledger.ensure_open()
        await application.aclose()
        application = None
        artifact = freeze_analysis(analysis_ws)
        result.analyses.append(artifact)
        return artifact, selection
    except Exception:
        result.violations.append(f'analysis_failed:{repeat}:{analysis_model}')
        if application is not None:
            _finalize(application, ledger, sources.values())
        return None
    finally:
        result.observations.extend(ledger.observations)
        result.violations.extend(ledger.violations)
        if application is not None:
            await application.aclose()
        ledger.cleanup()
        if client is not None:
            await _close(client)


async def _run_processing(protocol, client_factory, result, budget, simulated, sources, repeat, variant, artifact, selection, destination_root=None):
    processing_ws = fork_processing(artifact, _safe_id(f'{repeat}-{variant.id}'), destination_root=destination_root)
    route = protocol.routes.get(variant.processing_model, 'fake')
    _settings(processing_ws, variant.processing_model, route)
    branch = VariantResult(variant.id, repeat, artifact.content_hash, processing_ws)
    result.variants.append(branch)
    ledger = ObservationLedger(variant_id=variant.id, repeat=repeat, receipt_root=processing_ws.root / 'receipts', diagnostic_root=processing_ws.root / 'diagnostics', retain_rejected=protocol.retain_rejected is True, retention_seconds=protocol.diagnostic_retention_seconds or 0)
    client = application = None
    try:
        client = client_factory(model=variant.processing_model, stage='processing', settings=processing_ws.settings.llm, ledger=ledger, budget=budget)
        inner = _stage_client(client, ledger, variant.processing_model, route, simulated, processing_ws.settings.llm, budget, protocol.max_call_cost, protocol.temperature_policy or "set")
        application = processing_ws.build_application(llm_client=inner, on_logged=ledger.on_logged)
        _, fixed = _selection(application, protocol, sources)
        _manifest(processing_ws, result, f"{repeat}:{variant.id}", selection, fixed, artifact.content_hash)
        selected_uow = _SelectedUnitOfWork(application.unit_of_work, selection)
        start = StartProcessing(unit_of_work=selected_uow, glossary_confidence_threshold=application.settings.checks.glossary_confidence_threshold, glossary_review_min_mentions=application.settings.checks.glossary_review_min_mentions, glossary_review_min_chapters=application.settings.checks.glossary_review_min_chapters, llm_settings_snapshot=_llm_snapshot(application.settings))
        sequence = ChapterProcessingSequence(unit_of_work=selected_uow, queue=application.task_queue, process_chapter=ProcessChapter(unit_of_work=selected_uow, llm_client=_model(application, inner, ledger), retry_policy=RetryPolicy(max_attempts=application.settings.llm.max_attempts), validator=_validator(application.settings), max_unit_source_chars=application.settings.llm.max_unit_source_chars, context_builder=_ObservedContext(ledger, language_notes_per_chapter=application.settings.checks.language_notes_per_chapter)))
        for book_id in sources.values():
            run = start(StartProcessingCommand(book_id))
            branch.llm_snapshots[book_id] = run.llm_settings_snapshot
            tasks = await sequence.start(book_id)
            ledger.task_scopes.update({task.id: task.scope_id for task in tasks})
        tasks = await TaskWorkerPool(TaskRunner(queue=application.task_queue, handlers={TaskType.PROCESS_TRANSLATION_UNIT: sequence.run_task}), WorkerSettings(concurrency=application.settings.worker.concurrency)).drain()
        branch.completed_chapters = [task.scope_id for task in tasks if task.last_error is None]
        branch.failures.extend(f'chapter_failed:{task.scope_id}' for task in tasks if task.last_error is not None)
        _finalize(application, ledger, sources.values())
        ledger.ensure_open()
        if branch.failures:
            raise ComparisonStopped('Обработка выбранных глав не завершена')
        export = ExportBook(unit_of_work=selected_uow, writer=EpubExportWriter(), validator=application.epubcheck)
        for book_id in sources.values():
            target = application.paths.export_epub(book_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            export(ExportBookCommand(book_id, str(target)))
            branch.epubs[book_id] = target
            branch.epub_chapters[book_id] = [chapter.id for chapter in protocol.chapters if sources.get(file_hash(Path(chapter.source))) == book_id]
        branch.completed = True
    except Exception:
        branch.failures.append('branch_incomplete')
        if application is not None:
            _finalize(application, ledger, sources.values())
    finally:
        branch.observations = ledger.observations
        result.observations.extend(ledger.observations)
        result.violations.extend(ledger.violations)
        if application is not None:
            await application.aclose()
        ledger.cleanup()
        if client is not None:
            await _close(client)
