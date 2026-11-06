"""Correction of complete persisted block groups, with atomic stale-safe acceptance."""

import asyncio
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from uuid import uuid4

from linguary.application.block_processing import chapter_number, complete_processing, note_id_factory
from linguary.application.correction import (
    correction_inputs, original_block_payload, reconcile_corrections, task_matches_inputs, unchecked_blocks,
    current_correction_candidates, candidate_is_current, model_correction_blocks,
    correction_input_fingerprint,
)
from linguary.application.correction_dto import (
    ChapterCorrectionResponse, CorrectedBlockResponse, ChapterCorrectionV2Response,
    ChapterCorrectionReviewResponse, CorrectionSnapshotV2, proposed_block_from_payload,
)
from linguary.application.llm_dto import BlockResultResponse, TranslationResultResponse
from linguary.application.llm_rejections import finalize_response, record_rejection
from linguary.application.ports.llm import LLMContextCapacityError, LLMRequest, LLMResponseFormatError
from linguary.application.reader_text import ReaderTextPolicy, source_quote_fragments
from linguary.application.recovery import RecoveryError
from linguary.application.retry_policy import ResultValidationError, RetryPolicy
from linguary.application.validators import TranslationResultValidator
from linguary.application.validators.content_plan import validate_note_sources
from linguary.domain import LLMCallStage, TaskStatus, TaskType, TranslationUnit
from linguary.domain.book_conditions import WorkStateError, thaw
from linguary.domain.content_plan import canonical_sha
from linguary.domain.correction import CorrectionCheck, CorrectionCandidate, correction_task_identity
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
                    section_id=command.section_id, input_generation=payload['input_generation'],
                    policy_version=self.snapshot.policy_version)
                or not task_matches_inputs(tx, task, payload, inputs)):
            raise RecoveryError('interrupted')
        return inputs, task, payload

    def _request(self, inputs, task, block_ids):
        context = inputs.context
        # Dictionary evidence stays local, including in the full input fingerprint.
        conditions = dict(context['book_conditions'])
        conditions.pop('pronunciation_versions', None)
        body = dict(context, book_conditions=conditions, blocks=[dict(block_id=block_id,
            original=original_block_payload(next(b for b in inputs.blocks if b.id == block_id)),
            current_result={key: value for key, value in asdict(inputs.results[block_id]).items()
                            if key != 'pronunciation_assistances'}) for block_id in block_ids])
        request = LLMRequest.for_chapter_correction(instructions=self.snapshot.prompt,
            input_text=_json(body), ordered_block_ids=block_ids,
            contract_version=2 if isinstance(self.snapshot, CorrectionSnapshotV2) else 1).with_logging(
                book_id=task.book_id, stage=LLMCallStage.CHAPTER_CORRECTION,
                task_id=task.id, scope_id=task.scope_id, ordered_block_ids=block_ids,
                run_id=inputs.run.id)
        return replace(request, max_completion_tokens=self.snapshot.parameters['max_output_tokens'])

    async def __call__(self, command):
        checked = 0
        while True:
            with self.uow.transaction() as tx:
                inputs, task, _ = self._load(tx, command)
                missing = unchecked_blocks(tx, inputs)
            if not missing:
                return CorrectChapterResult(command.section_id, inputs.run.id, checked)
            # Source-text limits do not describe a provider's context window.
            # Preserve the full request and let capacity errors split whole blocks.
            checked += await self._group(command, missing)

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
                            provenance=None,
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
        used_note_ids = {note.id for result in all_results for note in result.language_notes}
        if isinstance(inputs.snapshot, CorrectionSnapshotV2):
            used_note_ids.update(n['id'] for c in tx.work.list_correction_candidates(inputs.run.id,
                states={'awaiting_review', 'awaiting_human', 'auto_applied', 'accepted', 'kept'})
                for n in c.payload['result']['language_notes'])
        notes = note_id_factory(chapter_number=chapter_number(inputs.document, inputs.section), used=used_note_ids)
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
                               english_help=inputs.conditions.payload.get('english_help', False),
                               validators_version=inputs.conditions.payload['versions']['validators'],
                               reader_level=inputs.conditions.payload.get('reader_level'))
        for candidate in candidates:
            for assistance in (candidate.fragment_assistances
                               if inputs.conditions.payload.get('english_help', False) else ()):
                quotes = {anchor.quote for anchor in assistance.anchors}
                if (candidate.translated_text or any(h.original in quotes for h in candidate.glossary_hints)
                        or any(n.original_excerpt in quotes for n in candidate.language_notes)):
                    raise ResultValidationError('исправление дублирует защищённую помощь к фрагменту')
        return tuple(next(r for r in candidates if r.block_id == block_id) for block_id in ids)


def prepare_correction_candidates(tx, inputs, candidates, validator):
    """Revalidate allowed fields while keeping the immutable proposal's note IDs."""
    response = ChapterCorrectionV2Response(tuple(proposed_block_from_payload(thaw(c.payload['proposal']))
                                               for c in candidates))
    validated = CorrectChapter(unit_of_work=None, llm_client=None, snapshot=inputs.snapshot,
                               validator=validator)._candidates(tx, inputs, response)
    results = []
    for candidate, result in zip(candidates, validated, strict=True):
        saved_notes = candidate.payload['result']['language_notes']
        if len(saved_notes) != len(result.language_notes):
            raise ResultValidationError('состав заметок кандидата изменился')
        result = replace(result, language_notes=tuple(replace(note, id=saved['id'])
            for note, saved in zip(result.language_notes, saved_notes, strict=True)))
        if canonical_sha(asdict(result)) != canonical_sha(thaw(candidate.payload['result'])):
            raise ResultValidationError('результат не совпадает с неизменяемым предложением')
        results.append(result)
    ids = {r.block_id for r in results}
    all_results = [r for r in tx.results.list_for_run(inputs.run.id) if r.block_id not in ids] + results
    notes = [n.id for r in all_results for n in r.language_notes]
    if len(notes) != len(set(notes)):
        raise ResultValidationError('идентификаторы заметок кандидата заняты')
    return tuple(results)


def apply_correction_candidate(tx, inputs, candidate, result, *, resolution):
    predecessor = inputs.bindings[candidate.block_id]
    if predecessor is None or predecessor != thaw(candidate.payload['source_binding']):
        raise WorkStateError('исходная версия кандидата изменилась')
    artifact, = tx.work.archive_artifacts([dict(book_id=candidate.book_id,
        conditions_id=candidate.conditions_id, attempt_id=candidate.proposal_attempt_id,
        run_id=candidate.run_id, kind='block_result', scope_id=candidate.block_id,
        body=dict(schema_version=1, result=asdict(result),
            source_revision=predecessor['artifact_revision_id'], source_attempt=predecessor['attempt_id'],
            correction_candidate_id=candidate.candidate_id, proposal_sha256=candidate.proposal_sha256,
            resolution=resolution))])
    tx.work.replace_result_binding(f'{candidate.run_id}:{candidate.block_id}',
        book_id=candidate.book_id, conditions_id=candidate.conditions_id,
        attempt_id=candidate.proposal_attempt_id, artifact_revision_id=artifact,
        expected_attempt_id=predecessor['attempt_id'],
        expected_artifact_revision_id=predecessor['artifact_revision_id'], provenance=None)


class CorrectChapterV2(CorrectChapter):
    """Persist proposals before reviewing them; uncertain text never replaces the current text."""

    async def __call__(self, command):
        checked = 0
        while True:
            with self.uow.transaction() as tx:
                inputs, _, _ = self._load(tx, command)
                pending = current_correction_candidates(tx, inputs, states={'awaiting_review'})
                missing = model_correction_blocks(tx, inputs)
            if pending:
                group_id = pending[0].group_id
                checked += await self._review_group(command,
                    tuple(c.candidate_id for c in pending if c.group_id == group_id))
            elif missing:
                checked += await self._propose_group(command, missing)
            else:
                return CorrectChapterResult(command.section_id, inputs.run.id, checked)

    def _call(self, tx, inputs, task, payload, request, ordinal, *, kind, refs=()):
        context = CallContext(conditions_id=inputs.conditions.id, stage=request.stage.value,
            scope_id=inputs.section.id, ordered_block_ids=request.ordered_block_ids,
            attempt_id=str(uuid4()), generation=task.attempts, assigned_model=self.snapshot.model,
            prompt_hash=sha256(request.instructions.encode()).hexdigest(), run_id=inputs.run.id,
            task_id=task.id, policy_version=self.snapshot.policy_version)
        request = replace(request, call_context=context, attempt=ordinal,
                          max_completion_tokens=self.snapshot.parameters['max_output_tokens'])
        tx.work.bind(kind, context.attempt_id, book_id=task.book_id, conditions_id=inputs.conditions.id,
            payload=dict(schema_version=1, run_id=inputs.run.id, section_id=inputs.section.id,
                manifest=inputs.manifest, task_id=task.id, task_attempt=task.attempts,
                input_generation=payload['input_generation'], candidate_refs=[list(r) for r in refs]))
        return request, context

    def _owned_response(self, tx, command, task, expected, context, *, kind):
        current, owned, _ = self._load(tx, command, expected_task_attempt=task.attempts)
        attempt = tx.work.get_attempt(context.attempt_id)
        receipt = tx.work.get_binding(kind, context.attempt_id)
        if (current.manifest != expected or receipt is None or receipt['payload_json']['manifest'] != expected
                or attempt is None or attempt.state != 'validated' or attempt.task_id != owned.id
                or attempt.run_id != current.run.id or attempt.conditions_id != current.conditions.id
                or attempt.context.get('task_attempt') != owned.attempts
                or attempt.context.get('operation_id') != owned.operation_id
                or tuple(attempt.context.get('ordered_block_ids', ())) != context.ordered_block_ids
                or any(attempt.context.get(k) != v for k, v in context.to_dict().items() if k != 'ordered_block_ids')):
            raise RecoveryError('interrupted')
        return current, attempt

    def _archive_response(self, tx, inputs, context, response, kind, request):
        body = dict(schema_version=1, response=asdict(response.payload), raw_response=response.raw_content,
            manifest=inputs.manifest, request=dict(instructions=request.instructions,
                input_text=request.input_text, contract_version=request.contract_version,
                ordered_block_ids=list(request.ordered_block_ids),
                candidate_refs=[list(r) for r in request.correction_candidate_refs]))
        artifact, = tx.work.archive_artifacts([dict(book_id=inputs.document.id,
            conditions_id=inputs.conditions.id, run_id=inputs.run.id, attempt_id=context.attempt_id,
            kind=kind, scope_id=inputs.section.id, body=body)])
        return artifact

    def _reject(self, response, context, error):
        if response is None:
            return
        if isinstance(error, RecoveryError):
            finalize_response(self.uow, response, 'stale')
        else:
            record_rejection(self.uow, response, error)
        with self.uow.transaction() as tx:
            attempt = tx.work.get_attempt(context.attempt_id)
            if attempt is not None and attempt.state == 'validated':
                tx.work.transition_attempt(attempt.id, 'rejected', expected_state='validated')

    async def _propose_group(self, command, block_ids):
        for ordinal in range(1, self.retry.max_attempts + 1):
            response = None
            with self.uow.transaction() as tx:
                inputs, task, payload = self._load(tx, command)
                request, context = self._call(tx, inputs, task, payload,
                    self._request(inputs, task, block_ids), ordinal, kind='correction_request')
                expected = inputs.manifest
            try:
                response = await self.client.generate(request)
                if (response.truncated or not isinstance(response.payload, ChapterCorrectionV2Response)
                        or response.call_context != context):
                    raise LLMResponseFormatError('неполное предложение исправления')
                response.payload.validate_for({b: inputs.results[b].treatment for b in block_ids})
                with self.uow.transaction() as tx:
                    current, attempt = self._owned_response(tx, command, task, expected, context,
                                                           kind='correction_request')
                    prepared = self._candidates(tx, current, response.payload)
                    raw = {b.block_id: b for b in response.payload.blocks}
                    candidates, unchanged = [], []
                    for result in prepared:
                        if result == current.results[result.block_id]:
                            unchanged.append(result.block_id)
                            continue
                        proposal = raw[result.block_id]
                        immutable = json.loads(_json(dict(schema_version=1, book_id=command.book_id,
                            run_id=current.run.id, conditions_id=current.conditions.id,
                            section_id=command.section_id, block_id=result.block_id,
                            input_sha256=current.checksums[result.block_id],
                            input_generation=current.input_generations[result.block_id],
                            group_id=attempt.id, proposal_attempt_id=attempt.id,
                            context_sha256=current.context_sha256,
                            source_binding=current.bindings[result.block_id],
                            original=original_block_payload(next(b for b in current.blocks if b.id == result.block_id)),
                            current_result=asdict(current.results[result.block_id]), result=asdict(result),
                            proposal=asdict(proposal), evidence_valid=all(i.valid_for(
                                next(b.text for b in current.blocks if b.id == result.block_id),
                                asdict(current.results[result.block_id])) for i in proposal.issues))))
                        candidates.append(CorrectionCandidate(str(uuid4()), command.book_id, current.run.id,
                            current.conditions.id, command.section_id, result.block_id,
                            current.checksums[result.block_id], current.input_generations[result.block_id],
                            canonical_sha(immutable), immutable, attempt.id, attempt.id))
                    artifact = self._archive_response(tx, current, context, response, 'correction_proposal', request)
                    tx.work.add_correction_candidates(candidates)
                    tx.work.bind('correction_group', attempt.id, book_id=command.book_id,
                        conditions_id=current.conditions.id, attempt_id=attempt.id, artifact_revision_id=artifact,
                        payload=dict(schema_version=1, run_id=current.run.id, section_id=command.section_id,
                            manifest=expected, block_ids=list(block_ids), unchanged_block_ids=unchanged,
                            candidate_ids=[c.candidate_id for c in candidates]))
                    tx.work.transition_attempt(attempt.id, 'accepted', expected_state='validated')
                    if not candidates:
                        tx.work.replace_correction_checks(current.run.id, command.section_id,
                            tuple(CorrectionCheck(current.run.id, b, command.section_id,
                                current.checksums[b], attempt.id, 'unchanged') for b in unchanged))
                    finalize_response(self.uow, response, 'accepted', transaction=tx)
                return len(block_ids) if not candidates else 0
            except LLMContextCapacityError:
                if len(block_ids) == 1:
                    raise LLMContextCapacityError('блок с контекстом предложения не помещается') from None
                middle = len(block_ids) // 2
                return await self._propose_group(command, block_ids[:middle]) + await self._propose_group(command, block_ids[middle:])
            except Exception as error:
                self._reject(response, context, error)
                if isinstance(error, RecoveryError):
                    raise
                if isinstance(error, LLMResponseFormatError) and error.truncated and len(block_ids) > 1:
                    middle = len(block_ids) // 2
                    return await self._propose_group(command, block_ids[:middle]) + await self._propose_group(command, block_ids[middle:])
                decision = self.retry.decide(error, attempts=ordinal)
                if not decision.should_retry:
                    raise
                if decision.backoff_seconds:
                    await self.sleep(decision.backoff_seconds)
        raise ResultValidationError('исчерпаны попытки предложения исправления')

    def _review_request(self, inputs, task, candidates):
        context = inputs.context
        conditions = dict(context['book_conditions'])
        conditions.pop('pronunciation_versions', None)
        allowed = ('treatment', 'translated_text', 'glossary_hints', 'language_notes')
        body = dict(source_language=inputs.document.source_language, target_language=inputs.document.target_language,
            chapter_originals=context['chapter_originals'], chapter_analysis=context['chapter_analysis'],
            book_conditions=conditions, run_glossary=context['run_glossary'], candidates=[dict(
                candidate_id=c.candidate_id, proposal_sha256=c.proposal_sha256,
                original=thaw(c.payload['original']),
                current_result={k: thaw(c.payload['current_result'][k]) for k in allowed},
                proposed_result={k: thaw(c.payload['result'][k]) for k in allowed},
                issues=thaw(c.payload['proposal']['issues'])) for c in candidates])
        return LLMRequest.for_chapter_correction_review(instructions=self.snapshot.review_prompt,
            input_text=_json(body), ordered_block_ids=tuple(c.block_id for c in candidates),
            candidate_refs=tuple((c.candidate_id, c.proposal_sha256) for c in candidates)).with_logging(
                book_id=task.book_id, stage=LLMCallStage.CHAPTER_CORRECTION_REVIEW,
                task_id=task.id, scope_id=task.scope_id, run_id=inputs.run.id)

    async def _review_group(self, command, candidate_ids):
        for ordinal in range(1, self.retry.max_attempts + 1):
            response = None
            with self.uow.transaction() as tx:
                inputs, task, payload = self._load(tx, command)
                candidates = tuple(tx.work.get_correction_candidate(i) for i in candidate_ids)
                if any(c is None or c.state != 'awaiting_review' or not candidate_is_current(tx, inputs, c)
                       for c in candidates):
                    raise RecoveryError('interrupted')
                refs = tuple((c.candidate_id, c.proposal_sha256) for c in candidates)
                request, context = self._call(tx, inputs, task, payload,
                    self._review_request(inputs, task, candidates), ordinal,
                    kind='correction_review_request', refs=refs)
                expected = inputs.manifest
            try:
                response = await self.client.generate(request)
                if (response.truncated or not isinstance(response.payload, ChapterCorrectionReviewResponse)
                        or response.call_context != context):
                    raise LLMResponseFormatError('неполная проверка кандидатов')
                response.payload.validate_coverage(refs)
                with self.uow.transaction() as tx:
                    current, attempt = self._owned_response(tx, command, task, expected, context,
                                                           kind='correction_review_request')
                    receipt = tx.work.get_binding('correction_review_request', attempt.id)
                    if tuple(tuple(r) for r in receipt['payload_json']['candidate_refs']) != refs:
                        raise RecoveryError('interrupted')
                    live = tuple(tx.work.get_correction_candidate(c.candidate_id) for c in candidates)
                    if any(c is None or c.state != 'awaiting_review' or not candidate_is_current(tx, current, c)
                           or (c.candidate_id, c.proposal_sha256) != ref for c, ref in zip(live, refs, strict=True)):
                        raise RecoveryError('interrupted')
                    verdicts = {v.candidate_id: v for v in response.payload.candidates}
                    accepted = tuple(c for c in live if verdicts[c.candidate_id].confirmed and c.payload['evidence_valid'])
                    results = prepare_correction_candidates(tx, current, accepted, self.validator) if accepted else ()
                    artifact = self._archive_response(tx, current, context, response, 'correction_review', request)
                    tx.work.transition_attempt(attempt.id, 'accepted', expected_state='validated')
                    for candidate in live:
                        verdict = verdicts[candidate.candidate_id]
                        tx.work.record_correction_verdict(candidate.candidate_id,
                            proposal_sha256=candidate.proposal_sha256, input_sha256=candidate.input_sha256,
                            review_attempt_id=attempt.id, review=dict(asdict(verdict), artifact_revision_id=artifact),
                            state='auto_applied' if candidate in accepted else 'awaiting_human')
                    for candidate, result in zip(accepted, results, strict=True):
                        apply_correction_candidate(tx, current, candidate, result, resolution='confirmed')
                    if results:
                        tx.results.replace_results(current.run.id, results)
                        tx.work.stale_reviews(command.book_id)
                    after = correction_inputs(tx, command.book_id, command.section_id)
                    for candidate in accepted:
                        tx.work.advance_correction_input(current.run.id, candidate.block_id,
                            expected_generation=candidate.input_generation,
                            fingerprint=correction_input_fingerprint(after, candidate.block_id))
                    checks = [CorrectionCheck(current.run.id, c.block_id, command.section_id,
                        after.checksums[c.block_id], c.proposal_attempt_id, 'confirmed', c.candidate_id) for c in accepted]
                    for group_id in {c.group_id for c in live}:
                        group = tx.work.get_binding('correction_group', group_id)['payload_json']
                        members = [tx.work.get_correction_candidate(i) for i in group['candidate_ids']]
                        if all(c is not None and c.review is not None for c in members):
                            for block_id in group['unchanged_block_ids']:
                                if (after.checksums[block_id] != group['manifest']['checksums'][block_id]
                                        or after.bindings[block_id] != group['manifest']['bindings'][block_id]):
                                    raise RecoveryError('interrupted')
                                checks.append(CorrectionCheck(current.run.id, block_id, command.section_id,
                                    after.checksums[block_id], group_id, 'unchanged'))
                    tx.work.replace_correction_checks(current.run.id, command.section_id, tuple(checks))
                    finalize_response(self.uow, response, 'accepted', transaction=tx)
                return len(candidate_ids)
            except LLMContextCapacityError:
                if len(candidate_ids) == 1:
                    raise LLMContextCapacityError('кандидат с контекстом проверки не помещается') from None
                middle = len(candidate_ids) // 2
                return await self._review_group(command, candidate_ids[:middle]) + await self._review_group(command, candidate_ids[middle:])
            except Exception as error:
                self._reject(response, context, error)
                if isinstance(error, RecoveryError):
                    raise
                if isinstance(error, LLMResponseFormatError) and error.truncated and len(candidate_ids) > 1:
                    middle = len(candidate_ids) // 2
                    return await self._review_group(command, candidate_ids[:middle]) + await self._review_group(command, candidate_ids[middle:])
                decision = self.retry.decide(error, attempts=ordinal)
                if not decision.should_retry:
                    raise
                if decision.backoff_seconds:
                    await self.sleep(decision.backoff_seconds)
        raise ResultValidationError('исчерпаны попытки проверки кандидатов')


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
            handler = CorrectChapterV2 if isinstance(runtime.snapshot, CorrectionSnapshotV2) else CorrectChapter
            await handler(unit_of_work=self.uow, llm_client=runtime.llm_client,
                snapshot=runtime.snapshot, validator=self.validator_factory(runtime.settings, conditions))(
                    CorrectChapterCommand(task.book_id, task.scope_id, task.id))
        complete_processing(self.uow, book_id=task.book_id, finished_task_id=task.id,
            task_types={TaskType.PROCESS_TRANSLATION_UNIT, TaskType.REGENERATE_BLOCK,
                        TaskType.CHAPTER_CORRECTION})
