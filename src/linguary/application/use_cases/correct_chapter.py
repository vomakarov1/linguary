"""Correction of complete persisted block groups, with atomic stale-safe acceptance."""

import asyncio
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from uuid import uuid4

from linguary.application.block_processing import chapter_number, complete_processing, note_id_factory
from linguary.application.correction import (
    correction_inputs, reconcile_corrections, task_matches_inputs, unchecked_blocks,
)
from linguary.application.correction_dto import ChapterCorrectionResponse, CorrectedBlockResponse
from linguary.application.llm_dto import BlockResultResponse, TranslationResultResponse
from linguary.application.llm_rejections import finalize_response, record_rejection
from linguary.application.ports.llm import LLMContextCapacityError, LLMRequest, LLMResponseFormatError
from linguary.application.reader_text import ReaderTextPolicy, source_quote_fragments
from linguary.application.recovery import RecoveryError
from linguary.application.retry_policy import ResultValidationError, RetryPolicy
from linguary.application.validators import TranslationResultValidator
from linguary.application.validators.content_plan import validate_note_sources
from linguary.domain import LLMCallStage, TaskStatus, TaskType, TranslationUnit
from linguary.domain.book_conditions import WorkStateError
from linguary.domain.correction import CorrectionCheck, correction_task_identity
from linguary.domain.llm_log import CallContext


@dataclass(frozen=True, slots=True)
class CorrectChapterCommand:
    book_id: str
    section_id: str
    task_id: str


@dataclass(frozen=True, slots=True)
class CorrectChapterResult:
    section_id: str
    run_id: str
    checked_blocks: int = 0


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


class CorrectChapter:
    def __init__(self, *, unit_of_work, llm_client, snapshot, validator: TranslationResultValidator,
                 retry_policy=None, sleep=None):
        self.uow, self.client, self.snapshot, self.validator = unit_of_work, llm_client, snapshot, validator
        self.retry = retry_policy or RetryPolicy(max_attempts=snapshot.parameters['max_attempts'])
        self.sleep = sleep or asyncio.sleep

    def _load(self, tx, command, *, expected_task_attempt=None):
        inputs = correction_inputs(tx, command.book_id, command.section_id)
        task = tx.tasks.get(command.task_id)
        binding = tx.work.get_binding('task', command.task_id)
        if (inputs is None or task is None or binding is None or not inputs.translated
                or task.status is not TaskStatus.RUNNING or task.book_id != command.book_id
                or task.task_type is not TaskType.CHAPTER_CORRECTION or task.scope_id != command.section_id
                or binding['book_id'] != command.book_id or binding['conditions_id'] != inputs.conditions.id
                or inputs.snapshot != self.snapshot
                or (expected_task_attempt is not None and task.attempts != expected_task_attempt)):
            raise RecoveryError('interrupted')
        payload = binding['payload_json']
        if (payload.get('run_id') != inputs.run.id or payload.get('section_id') != command.section_id
                or not payload.get('input_generation')
                or task.identity != correction_task_identity(book_id=command.book_id, run_id=inputs.run.id,
                    section_id=command.section_id, input_generation=payload['input_generation'])
                or not task_matches_inputs(tx, task, payload, inputs)):
            raise RecoveryError('interrupted')
        return inputs, task, payload

    def _request(self, inputs, task, block_ids):
        body = dict(inputs.context, blocks=[dict(block_id=block_id,
            original=asdict(next(b for b in inputs.blocks if b.id == block_id)),
            current_result=asdict(inputs.results[block_id])) for block_id in block_ids])
        request = LLMRequest.for_chapter_correction(instructions=self.snapshot.prompt,
            input_text=_json(body), ordered_block_ids=block_ids).with_logging(
                book_id=task.book_id, stage=LLMCallStage.CHAPTER_CORRECTION,
                task_id=task.id, scope_id=task.scope_id, ordered_block_ids=block_ids,
                run_id=inputs.run.id)
        return replace(request, max_completion_tokens=self.snapshot.parameters['max_output_tokens'])

    def _fits(self, request, inputs):
        # Count the full serialized request and reserve room for complete output.
        reserve = len(_json([asdict(inputs.results[b]) for b in request.ordered_block_ids]))
        configured = self.snapshot.parameters['max_output_tokens']
        if configured is not None:
            reserve = max(reserve, configured * 4)
        return len(_json(asdict(request))) + reserve <= self.snapshot.parameters['max_unit_source_chars']

    async def __call__(self, command):
        checked = 0
        while True:
            with self.uow.transaction() as tx:
                inputs, task, _ = self._load(tx, command)
                missing = unchecked_blocks(tx, inputs)
            if not missing:
                return CorrectChapterResult(command.section_id, inputs.run.id, checked)
            group = ()
            for block_id in missing:
                candidate = (*group, block_id)
                if not self._fits(self._request(inputs, task, candidate), inputs):
                    if not group:
                        raise LLMContextCapacityError('целый блок с обязательным контекстом исправления превышает лимит')
                    break
                group = candidate
            checked += await self._group(command, group)

    async def _group(self, command, block_ids):
        response = None
        for ordinal in range(1, self.retry.max_attempts + 1):
            with self.uow.transaction() as tx:
                inputs, task, payload = self._load(tx, command)
                request = self._request(inputs, task, block_ids)
                context = CallContext(conditions_id=inputs.conditions.id,
                    stage=LLMCallStage.CHAPTER_CORRECTION.value, scope_id=command.section_id,
                    ordered_block_ids=block_ids, attempt_id=str(uuid4()), generation=task.attempts,
                    assigned_model=self.snapshot.model,
                    prompt_hash=sha256(self.snapshot.prompt.encode()).hexdigest(),
                    run_id=inputs.run.id, task_id=task.id, policy_version=self.snapshot.policy_version)
                request = replace(request, call_context=context, attempt=ordinal,
                    max_completion_tokens=self.snapshot.parameters['max_output_tokens'])
                expected = inputs.manifest
                tx.work.bind('correction_request', context.attempt_id, book_id=command.book_id,
                    conditions_id=inputs.conditions.id, payload=dict(schema_version=1,
                        run_id=inputs.run.id, section_id=command.section_id, manifest=expected,
                        task_id=task.id, task_attempt=task.attempts,
                        input_generation=payload['input_generation'], block_ids=list(block_ids)))
            try:
                response = await self.client.generate(request)
                if (response.truncated or not isinstance(response.payload, ChapterCorrectionResponse)
                        or response.call_context != context):
                    raise LLMResponseFormatError('неполный ответ исправления или неверная попытка')
                response.payload.validate_for({b: inputs.results[b].treatment for b in block_ids})
                with self.uow.transaction() as tx:
                    current, owned_task, current_payload = self._load(tx, command,
                        expected_task_attempt=task.attempts)
                    attempt = tx.work.get_attempt(context.attempt_id)
                    persisted = tx.work.get_binding('correction_request', context.attempt_id)
                    if (current.manifest != expected or current.run.id != inputs.run.id
                            or current.conditions.id != inputs.conditions.id or persisted is None
                            or persisted['payload_json']['manifest'] != expected
                            or current_payload['input_generation'] != payload['input_generation']
                            or attempt is None or attempt.state != 'validated'
                            or attempt.task_id != owned_task.id or attempt.run_id != current.run.id
                            or attempt.conditions_id != current.conditions.id
                            or any(attempt.context.get(key) != value
                                for key, value in context.to_dict().items()
                                if key != 'ordered_block_ids')
                            or attempt.context.get('task_attempt') != owned_task.attempts
                            or attempt.context.get('operation_id') != owned_task.operation_id
                            or tuple(attempt.context.get('ordered_block_ids', ())) != block_ids):
                        raise RecoveryError('interrupted')
                    candidates = self._candidates(tx, current, response.payload)
                    changed = tuple(r for r in candidates if r != current.results[r.block_id])
                    # Validate the whole group before writing any projection or archive.
                    for result in changed:
                        predecessor = current.bindings[result.block_id]
                        if predecessor is None:
                            raise WorkStateError('исправление не имеет доказанного предшественника')
                        artifact, = tx.work.archive_artifacts([dict(book_id=command.book_id,
                            conditions_id=current.conditions.id, attempt_id=attempt.id, run_id=current.run.id,
                            kind='block_result', scope_id=result.block_id,
                            body=dict(schema_version=1, result=asdict(result),
                                source_revision=predecessor['artifact_revision_id'],
                                source_attempt=predecessor['attempt_id']))])
                        tx.work.replace_result_binding(f'{current.run.id}:{result.block_id}',
                            book_id=command.book_id, conditions_id=current.conditions.id,
                            attempt_id=attempt.id, artifact_revision_id=artifact,
                            expected_attempt_id=predecessor['attempt_id'],
                            expected_artifact_revision_id=predecessor['artifact_revision_id'])
                    if changed:
                        tx.results.replace_results(current.run.id, changed)
                        tx.work.stale_reviews(command.book_id)
                    tx.work.transition_attempt(attempt.id, 'accepted', expected_state='validated')
                    accepted = correction_inputs(tx, command.book_id, command.section_id)
                    checksums = accepted.checksums
                    tx.work.replace_correction_checks(current.run.id, command.section_id,
                        tuple(CorrectionCheck(current.run.id, b, command.section_id, checksums[b], attempt.id)
                              for b in block_ids))
                    finalize_response(self.uow, response, 'accepted', transaction=tx)
                return len(block_ids)
            except LLMContextCapacityError:
                if len(block_ids) == 1:
                    raise LLMContextCapacityError('целый блок с обязательным контекстом исправления не помещается') from None
                middle = len(block_ids) // 2
                return (await self._group(command, block_ids[:middle])
                        + await self._group(command, block_ids[middle:]))
            except RecoveryError:
                if response is not None:
                    with self.uow.transaction() as tx:
                        attempt = tx.work.get_attempt(context.attempt_id)
                        if attempt is not None and attempt.state == 'validated':
                            tx.work.transition_attempt(attempt.id, 'rejected', expected_state='validated')
                    finalize_response(self.uow, response, 'stale')
                raise
            except Exception as error:
                if response is not None:
                    record_rejection(self.uow, response, error)
                    with self.uow.transaction() as tx:
                        attempt = tx.work.get_attempt(context.attempt_id)
                        if attempt is not None and attempt.state == 'validated':
                            tx.work.transition_attempt(attempt.id, 'rejected', expected_state='validated')
                decision = self.retry.decide(error, attempts=ordinal)
                if not decision.should_retry:
                    raise
                if decision.backoff_seconds:
                    await self.sleep(decision.backoff_seconds)
                response = None
        raise ResultValidationError('исчерпаны попытки исправления главы')

    def _candidates(self, tx, inputs, response):
        from linguary.application.use_cases.export_book import _validate_content_plans
        ids = response.block_ids
        all_results = tx.results.list_for_run(inputs.run.id)
        notes = note_id_factory(chapter_number=chapter_number(inputs.document, inputs.section),
            used={note.id for result in all_results for note in result.language_notes})
        originals = {b.id: b for b in inputs.blocks}
        candidates = []
        for block in response.blocks:
            previous = inputs.results[block.block_id]
            if not isinstance(block, CorrectedBlockResponse):
                prepared = ReaderTextPolicy().prepare_results((previous,),
                    source_quote_fragments(b.text for b in inputs.section.ordered_blocks))
                if prepared != (previous,):
                    raise ResultValidationError('неизменённый блок содержит неочищенный читательский текст')
                candidates.append(previous)
                continue
            unit = TranslationUnit.of_blocks(id=str(uuid4()), blocks=(originals[block.block_id],),
                                            plan=inputs.analysis.treatment_plan)
            payload = TranslationResultResponse(
                blocks=(BlockResultResponse(block.block_id, previous.treatment,
                    block.translated_text, block.glossary_hints),), language_notes=block.language_notes)
            validated, = self.validator.validate(payload, unit=unit,
                target_language=inputs.document.target_language, note_id_factory=notes)
            # Preserve identifiers of notes surviving a correction at the same source anchor.
            used = set()
            restored_notes = []
            for raw, note in zip(block.language_notes, validated.language_notes, strict=True):
                old = next((n for n in previous.language_notes
                            if n.original_excerpt == raw.original_excerpt and n.id not in used), None)
                if old is not None:
                    used.add(old.id)
                restored_notes.append(replace(note, id=note.id if old is None else old.id,
                    original_excerpt=raw.original_excerpt, translated_excerpt=raw.translated_excerpt))
            candidate = replace(previous, translated_text=validated.translated_text,
                glossary_hints=tuple(replace(h, original=raw.original)
                    for h, raw in zip(validated.glossary_hints, block.glossary_hints, strict=True)),
                language_notes=tuple(restored_notes))
            candidate, = ReaderTextPolicy().prepare_results((candidate,),
                source_quote_fragments(b.text for b in inputs.section.ordered_blocks))
            candidates.append(candidate)
        validate_note_sources(candidates, {b.id: b.text for b in inputs.blocks})
        merged = tuple(next((candidate for candidate in candidates if candidate.block_id == r.block_id), r)
                       for r in all_results)
        note_ids = [n.id for r in merged for n in r.language_notes]
        if len(set(note_ids)) != len(note_ids):
            raise ResultValidationError('идентификаторы языковых заметок повторяются')
        _validate_content_plans(inputs.document, (inputs.analysis,), merged,
                               english_help=inputs.conditions.payload.get('english_help', False))
        for candidate in candidates:
            for assistance in candidate.fragment_assistances:
                quotes = {anchor.quote for anchor in assistance.anchors}
                if (candidate.translated_text or any(h.original in quotes for h in candidate.glossary_hints)
                        or any(n.original_excerpt in quotes for n in candidate.language_notes)):
                    raise ResultValidationError('исправление дублирует защищённую помощь к фрагменту')
        return tuple(next(r for r in candidates if r.block_id == block_id) for block_id in ids)


class CorrectionSequence:
    """Lease the run-pinned correction role; a correction retry never translates blocks."""

    def __init__(self, *, unit_of_work, queue, runtime_resolver, validator_factory):
        self.uow, self.queue, self.resolver = unit_of_work, queue, runtime_resolver
        self.validator_factory = validator_factory

    def reconcile(self):
        with self.uow.transaction() as tx:
            for book in tx.books.list_all():
                reconcile_corrections(tx, book.id)

    async def start(self, book_id):
        with self.uow.transaction() as tx:
            tasks = reconcile_corrections(tx, book_id)
        for task in tasks:
            if task.status is TaskStatus.PENDING:
                await self.queue.enqueue(task.identity)
        return tasks

    async def run_task(self, task):
        if task.task_type is not TaskType.CHAPTER_CORRECTION:
            raise WorkStateError('неверный тип задачи исправления')
        with self.uow.transaction() as tx:
            binding = tx.work.get_binding('task', task.id)
            if binding is None:
                raise RecoveryError('interrupted')
            run_id = binding['payload_json']['run_id']
        async with self.resolver.lease(run_id) as runtime:
            with self.uow.transaction() as tx:
                conditions = tx.work.get_conditions(runtime.conditions_id)
            await CorrectChapter(unit_of_work=self.uow, llm_client=runtime.llm_client,
                snapshot=runtime.snapshot, validator=self.validator_factory(runtime.settings, conditions))(
                    CorrectChapterCommand(task.book_id, task.scope_id, task.id))
        complete_processing(self.uow, book_id=task.book_id, finished_task_id=task.id,
            task_types={TaskType.PROCESS_TRANSLATION_UNIT, TaskType.REGENERATE_BLOCK,
                        TaskType.CHAPTER_CORRECTION})
