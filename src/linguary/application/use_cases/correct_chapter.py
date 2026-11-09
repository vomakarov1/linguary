"""Correction of complete persisted block groups, with atomic stale-safe acceptance."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from typing import Any
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


_STATIC_CHAPTER_PREFIX_KEYS = (
    'schema_version',
    'book_id',
    'section_id',
    'source_language',
    'target_language',
    'book_conditions',
    'run_glossary',
    'chapter_analysis',
    'chapter_blocks',
)


def _normalize_sorted_keys(value: Any) -> Any:
    return json.loads(_json(value))


_STRIPPED_GLOSSARY_KEYS = ('id', 'book_id', 'created_at', 'concept_id')


def _clean_glossary_entry(entry: Any) -> Any:
    if not isinstance(entry, Mapping):
        return entry
    return {
        k: v
        for k, v in entry.items()
        if k not in _STRIPPED_GLOSSARY_KEYS
        and not (k == 'note' and (v is None or (isinstance(v, str) and not v.strip())))
    }


def _llm_run_glossary(glossary: Any) -> Any:
    raw = thaw(glossary)
    if isinstance(raw, Mapping):
        out = {k: v for k, v in raw.items() if k not in _STRIPPED_GLOSSARY_KEYS}
        if isinstance(out.get('entries'), (list, tuple)):
            out['entries'] = [_clean_glossary_entry(e) for e in out['entries']]
        return out
    if isinstance(raw, (list, tuple)):
        return [_clean_glossary_entry(e) for e in raw]
    return raw


def _filtered_chapter_context(
    inputs: Any,
    target_block_ids: Sequence[str] | set[str],
) -> dict[str, Any]:
    context = dict(getattr(inputs, 'context', None) or {})
    document = getattr(inputs, 'document', None)
    if document is not None:
        if 'book_id' not in context and getattr(document, 'id', None) is not None:
            context['book_id'] = document.id
        if 'source_language' not in context and getattr(document, 'source_language', None) is not None:
            context['source_language'] = document.source_language
        if 'target_language' not in context and getattr(document, 'target_language', None) is not None:
            context['target_language'] = document.target_language
    section = getattr(inputs, 'section', None)
    if section is not None and 'section_id' not in context and getattr(section, 'id', None) is not None:
        context['section_id'] = section.id
    if 'run_glossary' in context and context['run_glossary'] is not None:
        context['run_glossary'] = _llm_run_glossary(context['run_glossary'])
    aliases = getattr(inputs, 'block_aliases', {})
    allowed_block_ids: set[str] | None = None
    blocks = getattr(inputs, 'blocks', ())
    target_set = set(target_block_ids)
    if blocks and 0 < len(target_set) < len(blocks):
        ordered_blocks = (
            section.ordered_blocks
            if section is not None and hasattr(section, 'ordered_blocks')
            else blocks
        )
        ordered_ids = [b.id for b in ordered_blocks]
        allowed_block_ids = set(target_set)
        for idx, bid in enumerate(ordered_ids):
            if bid in target_set:
                allowed_block_ids.update(
                    ordered_ids[max(0, idx - 2) : min(len(ordered_ids), idx + 3)]
                )
        allowed_aliases = {aliases.get(bid, bid) for bid in allowed_block_ids} | allowed_block_ids
        if isinstance(context.get('chapter_blocks'), (list, tuple)):
            context['chapter_blocks'] = [
                cb for cb in context['chapter_blocks']
                if isinstance(cb, Mapping) and cb.get('block_id') in allowed_aliases
            ]
    if 'chapter_analysis' in context:
        context['chapter_analysis'] = _llm_chapter_analysis(
            context['chapter_analysis'],
            aliases,
            allowed_block_ids=allowed_block_ids,
        )
    return context


def _request_input_json(
    context: Mapping[str, Any],
    *,
    payload_key: str,
    payload_value: Any,
) -> str:
    ordered_body: dict[str, Any] = {}
    for key in _STATIC_CHAPTER_PREFIX_KEYS:
        if key in context:
            ordered_body[key] = _normalize_sorted_keys(context[key])
    for key in sorted(context):
        if key not in ordered_body and key != payload_key:
            ordered_body[key] = _normalize_sorted_keys(context[key])
    ordered_body[payload_key] = _normalize_sorted_keys(payload_value)
    return json.dumps(
        ordered_body,
        ensure_ascii=False,
        sort_keys=False,
        separators=(',', ':'),
        allow_nan=False,
    )


def _chunk_correction_blocks(
    block_ids: Sequence[str],
    inputs: Any,
    max_chars: int | None,
) -> tuple[tuple[str, ...], ...]:
    ordered = tuple(block_ids)
    if not ordered:
        return ()
    blocks = getattr(inputs, 'blocks', None)
    if not blocks or max_chars is None or max_chars <= 1:
        return (ordered,)
    lengths = {
        block.id: len(block.text)
        for block in blocks
        if hasattr(block, 'id') and hasattr(block, 'text')
    }
    if not lengths:
        return (ordered,)
    chunks: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for block_id in ordered:
        block_len = lengths.get(block_id, 0)
        if current and current_chars + block_len > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(block_id)
        current_chars += block_len
    if current:
        chunks.append(current)
    return tuple(tuple(chunk) for chunk in chunks)


def _needs_plan_id_alias(raw_id: Any, prefix: str) -> bool:
    return isinstance(raw_id, str) and (raw_id.startswith(prefix) or len(raw_id) > 16)


def _difficulty_aliases(analysis: Any) -> dict[str, str]:
    if analysis is None:
        return {}
    raw = thaw(analysis) if isinstance(analysis, Mapping) else analysis
    cp = raw.get('content_plan') if isinstance(raw, Mapping) else getattr(raw, 'content_plan', None)
    if cp is None:
        return {}
    difficulties = cp.get('difficulties') if isinstance(cp, Mapping) else getattr(cp, 'difficulties', None)
    if not isinstance(difficulties, (list, tuple)):
        return {}
    aliases: dict[str, str] = {}
    for idx, d in enumerate(difficulties, start=1):
        diff_id = d.get('id') if isinstance(d, Mapping) else getattr(d, 'id', None)
        if _needs_plan_id_alias(diff_id, 'df-'):
            aliases[diff_id] = f'd{idx}'
    return aliases


def _alias_error_for_retry(
    error: Exception,
    block_aliases: Mapping[str, str] | None = None,
    candidate_aliases: Mapping[str, str] | None = None,
) -> Exception:
    replacements: dict[str, str] = {}
    if block_aliases:
        replacements.update(block_aliases)
    if candidate_aliases:
        replacements.update(candidate_aliases)
    if not replacements:
        return error
    message = str(error)
    for raw_id, alias in replacements.items():
        if raw_id and raw_id in message:
            message = message.replace(raw_id, alias)
    if message != str(error):
        error.args = (message, *error.args[1:])
    expectation = getattr(error, 'expectation', None)
    if expectation is not None and getattr(expectation, 'blocks', None):
        aliased_blocks = tuple(
            replace(b, block_id=replacements.get(b.block_id, b.block_id))
            for b in expectation.blocks
        )
        if aliased_blocks != expectation.blocks:
            error.expectation = replace(expectation, blocks=aliased_blocks)
    return error


def _llm_anchor(anchor: Mapping[str, Any], aliases: Mapping[str, str]) -> dict[str, Any]:
    out = {k: v for k, v in anchor.items() if k not in ('start', 'end')}
    if 'block_id' in out and isinstance(out['block_id'], str):
        out['block_id'] = aliases.get(out['block_id'], out['block_id'])
    return out


def _llm_chapter_analysis(
    analysis: Mapping[str, Any] | None,
    aliases: Mapping[str, str],
    *,
    allowed_block_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    if analysis is None:
        return None
    allowed = (
        ({aliases.get(bid, bid) for bid in allowed_block_ids} | set(allowed_block_ids))
        if allowed_block_ids is not None
        else None
    )
    out = {
        k: thaw(v)
        for k, v in analysis.items()
        if k
        not in (
            'id',
            'section_id',
            'coverage_status',
            'glossary_candidates',
            'name_translation_evidence',
            'before_you_read_evidence',
        )
    }
    treatment_plan = out.get('treatment_plan')
    if isinstance(treatment_plan, Mapping):
        tp = {k: v for k, v in treatment_plan.items() if k != 'section_id'}
        if isinstance(tp.get('items'), (list, tuple)):
            items = [
                {**item, 'block_id': aliases.get(item['block_id'], item['block_id'])}
                if isinstance(item, Mapping) and 'block_id' in item
                else item
                for item in tp['items']
            ]
            if allowed is not None:
                items = [
                    item for item in items
                    if not isinstance(item, Mapping) or 'block_id' not in item or item['block_id'] in allowed
                ]
            tp['items'] = items
        out['treatment_plan'] = tp
    content_plan = out.get('content_plan')
    if isinstance(content_plan, Mapping):
        difficulty_aliases = _difficulty_aliases(out)
        cp = {
            k: v
            for k, v in content_plan.items()
            if k not in ('version', 'profile', 'sha256', 'before_you_read_evidence')
        }
        if isinstance(cp.get('difficulties'), (list, tuple)):
            difficulties = []
            for idx, d in enumerate(cp['difficulties'], start=1):
                if isinstance(d, Mapping):
                    diff = dict(d)
                    diff_id = diff.get('id')
                    if isinstance(diff_id, str):
                        if diff_id in difficulty_aliases:
                            diff['id'] = difficulty_aliases[diff_id]
                        elif _needs_plan_id_alias(diff_id, 'df-'):
                            diff['id'] = f'd{idx}'
                    if 'owner_block_id' in diff:
                        diff['owner_block_id'] = aliases.get(diff['owner_block_id'], diff['owner_block_id'])
                    if isinstance(diff.get('anchors'), (list, tuple)):
                        diff['anchors'] = [
                            _llm_anchor(a, aliases) if isinstance(a, Mapping) else a
                            for a in diff['anchors']
                        ]
                    if allowed is not None:
                        in_window = (
                            diff.get('owner_block_id') in allowed
                            or any(
                                isinstance(a, Mapping) and a.get('block_id') in allowed
                                for a in diff.get('anchors', ())
                            )
                        )
                        if not in_window:
                            continue
                    difficulties.append(diff)
                else:
                    difficulties.append(d)
            cp['difficulties'] = difficulties
        if isinstance(cp.get('context_facts'), (list, tuple)):
            facts = []
            for idx, f in enumerate(cp['context_facts'], start=1):
                if isinstance(f, Mapping):
                    fact = dict(f)
                    fact_id = fact.get('id')
                    if _needs_plan_id_alias(fact_id, 'cf-'):
                        fact['id'] = f'f{idx}'
                    if isinstance(fact.get('anchors'), (list, tuple)):
                        fact['anchors'] = [
                            _llm_anchor(a, aliases) if isinstance(a, Mapping) else a
                            for a in fact['anchors']
                        ]
                    if isinstance(fact.get('subject_anchor'), Mapping):
                        fact['subject_anchor'] = _llm_anchor(fact['subject_anchor'], aliases)
                    if allowed is not None:
                        subj = fact.get('subject_anchor')
                        in_window = (
                            (isinstance(subj, Mapping) and subj.get('block_id') in allowed)
                            or any(
                                isinstance(a, Mapping) and a.get('block_id') in allowed
                                for a in fact.get('anchors', ())
                            )
                        )
                        if not in_window:
                            continue
                    facts.append(fact)
                else:
                    facts.append(f)
            cp['context_facts'] = facts
        out['content_plan'] = cp
    return out


_LLM_RESULT_FIELDS = (
    'treatment',
    'translated_text',
    'glossary_hints',
    'language_notes',
    'fragment_assistances',
)


def _llm_block_result(
    result: Mapping[str, Any],
    aliases: Mapping[str, str],
    difficulty_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _LLM_RESULT_FIELDS:
        if key not in result:
            continue
        value = thaw(result[key])
        if key == 'language_notes' and isinstance(value, (list, tuple)):
            notes = []
            for n in value:
                if isinstance(n, Mapping):
                    target = {k: v for k, v in n.items() if k != 'id'}
                    for bid_key in ('block_id', 'source_block_id'):
                        if bid_key in target and target[bid_key] in aliases:
                            target[bid_key] = aliases[target[bid_key]]
                    notes.append(target)
                else:
                    notes.append(n)
            out[key] = notes
        elif key == 'fragment_assistances' and isinstance(value, (list, tuple)):
            assistances = []
            fallback_diff_aliases: dict[str, str] = {}
            for fa in value:
                if isinstance(fa, Mapping):
                    item = dict(fa)
                    diff_id = item.get('difficulty_id')
                    if isinstance(diff_id, str):
                        if difficulty_aliases and diff_id in difficulty_aliases:
                            item['difficulty_id'] = difficulty_aliases[diff_id]
                        elif diff_id in aliases:
                            item['difficulty_id'] = aliases[diff_id]
                        elif _needs_plan_id_alias(diff_id, 'df-'):
                            if diff_id not in fallback_diff_aliases:
                                fallback_diff_aliases[diff_id] = f'd{len(fallback_diff_aliases) + 1}'
                            item['difficulty_id'] = fallback_diff_aliases[diff_id]
                    if isinstance(item.get('anchors'), (list, tuple)):
                        item['anchors'] = [
                            _llm_anchor(a, aliases) if isinstance(a, Mapping) else a
                            for a in item['anchors']
                        ]
                    assistances.append(item)
                else:
                    assistances.append(fa)
            out[key] = assistances
        else:
            out[key] = value
    return out


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
        raw_analysis = (
            getattr(inputs, 'analysis', None)
            or (inputs.context.get('chapter_analysis') if isinstance(getattr(inputs, 'context', None), Mapping) else None)
        )
        difficulty_aliases = _difficulty_aliases(raw_analysis)
        context = _filtered_chapter_context(inputs, block_ids)
        aliases = getattr(inputs, 'block_aliases', {})
        blocks_payload = []
        for block_id in block_ids:
            alias = aliases.get(block_id, block_id)
            current_result = (
                _llm_block_result(
                    asdict(inputs.results[block_id]),
                    aliases,
                    difficulty_aliases=difficulty_aliases,
                )
                if block_id in inputs.results
                else {}
            )
            blocks_payload.append(dict(
                block_id=alias,
                current_result=current_result,
            ))
        block_aliases = tuple((aliases[b], b) for b in block_ids if b in aliases)
        request = LLMRequest.for_chapter_correction(
            instructions=self.snapshot.prompt,
            input_text=_request_input_json(context, payload_key='blocks', payload_value=blocks_payload),
            ordered_block_ids=block_ids,
            contract_version=2 if isinstance(self.snapshot, CorrectionSnapshotV2) else 1,
            block_aliases=block_aliases,
        ).with_logging(
            book_id=task.book_id,
            stage=LLMCallStage.CHAPTER_CORRECTION,
            task_id=task.id,
            scope_id=task.scope_id,
            ordered_block_ids=block_ids,
            run_id=inputs.run.id,
        )
        return replace(request, max_completion_tokens=self.snapshot.parameters['max_output_tokens'])

    async def __call__(self, command):
        checked = 0
        while True:
            with self.uow.transaction() as tx:
                inputs, task, _ = self._load(tx, command)
                missing = unchecked_blocks(tx, inputs)
            if not missing:
                return CorrectChapterResult(command.section_id, inputs.run.id, checked)
            max_chars = getattr(self.snapshot, 'parameters', {}).get('max_unit_source_chars')
            for chunk in _chunk_correction_blocks(missing, inputs, max_chars):
                checked += await self._group(command, chunk)

    async def _group(self, command, block_ids):
        response = None
        repair_turns = ()
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
                    max_completion_tokens=self.snapshot.parameters['max_output_tokens'],
                    repair_turns=repair_turns)
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
                    raise LLMResponseFormatError(
                        'неполный ответ исправления или неверная попытка',
                        truncated=response.truncated,
                        raw_content=response.raw_content,
                        usage=response.usage,
                        cost=response.cost,
                        provider=response.provider,
                        model=response.model,
                        call_context=response.call_context,
                    )
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
                if isinstance(error, LLMResponseFormatError) and error.truncated and len(block_ids) > 1:
                    middle = len(block_ids) // 2
                    return (await self._group(command, block_ids[:middle])
                            + await self._group(command, block_ids[middle:]))
                _alias_error_for_retry(error, getattr(inputs, 'block_aliases', None))
                decision = self.retry.decide(
                    error,
                    attempts=ordinal,
                    raw_content=response.raw_content if response is not None else None,
                )
                if not decision.should_retry:
                    raise
                if decision.backoff_seconds:
                    await self.sleep(decision.backoff_seconds)
                next_req = decision.next_request(request)
                repair_turns = next_req.repair_turns
                response = None
        raise ResultValidationError('исчерпаны попытки исправления главы')

    def _candidates(self, tx, inputs, response):
        from linguary.application.use_cases.export_book import _validate_content_plans
        ids = response.block_ids
        all_results = tuple(inputs.results.values())
        used_note_ids = {note.id for result in all_results for note in result.language_notes}
        if isinstance(inputs.snapshot, CorrectionSnapshotV2):
            used_note_ids.update(n['id'] for c in tx.work.list_correction_candidates(inputs.run.id,
                section_id=inputs.section.id,
                states={'awaiting_review', 'awaiting_human', 'auto_applied', 'accepted', 'kept'})
                for n in c.payload['result']['language_notes'])
        notes = note_id_factory(chapter_number=chapter_number(inputs.document, inputs.section), used=used_note_ids)
        originals = {b.id: b for b in inputs.blocks}
        section_quote_fragments = source_quote_fragments(b.text for b in inputs.section.ordered_blocks)
        candidates = []
        for block in response.blocks:
            previous = inputs.results[block.block_id]
            if not isinstance(block, CorrectedBlockResponse):
                prepared = ReaderTextPolicy().prepare_results((previous,), section_quote_fragments)
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
            candidate, = ReaderTextPolicy().prepare_results((candidate,), section_quote_fragments)
            candidates.append(candidate)
        validate_note_sources(candidates, {b.id: b.text for b in inputs.section.ordered_blocks})
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
    all_results = [r for r in tuple(inputs.results.values()) if r.block_id not in ids] + results
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
            max_chars = getattr(self.snapshot, 'parameters', {}).get('max_unit_source_chars')
            if pending:
                group_id = pending[0].group_id
                group_pending = tuple(c for c in pending if c.group_id == group_id)
                by_block = {c.block_id: c.candidate_id for c in group_pending}
                for chunk in _chunk_correction_blocks(tuple(by_block), inputs, max_chars):
                    checked += await self._review_group(
                        command, tuple(by_block[b] for b in chunk)
                    )
            elif missing:
                for chunk in _chunk_correction_blocks(missing, inputs, max_chars):
                    checked += await self._propose_group(command, chunk)
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
                candidate_refs=[list(r) for r in request.correction_candidate_refs],
                repair_instructions=list(request.repair_instructions)))
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
        repair_turns = ()
        for ordinal in range(1, self.retry.max_attempts + 1):
            response = None
            with self.uow.transaction() as tx:
                inputs, task, payload = self._load(tx, command)
                base_request = self._request(inputs, task, block_ids)
                if repair_turns:
                    base_request = replace(base_request, repair_turns=repair_turns)
                request, context = self._call(tx, inputs, task, payload,
                    base_request, ordinal, kind='correction_request')
                expected = inputs.manifest
            try:
                response = await self.client.generate(request)
                if (response.truncated or not isinstance(response.payload, ChapterCorrectionV2Response)
                        or response.call_context != context):
                    raise LLMResponseFormatError(
                        'неполное предложение исправления',
                        truncated=response.truncated,
                        raw_content=response.raw_content,
                        usage=response.usage,
                        cost=response.cost,
                        provider=response.provider,
                        model=response.model,
                        call_context=response.call_context,
                    )
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
                        evidence_valid = all(
                            i.valid_for(
                                next(b.text for b in current.blocks if b.id == result.block_id),
                                asdict(current.results[result.block_id]),
                            )
                            for i in proposal.issues
                        )
                        immutable = json.loads(_json(dict(schema_version=1, book_id=command.book_id,
                            run_id=current.run.id, conditions_id=current.conditions.id,
                            section_id=command.section_id, block_id=result.block_id,
                            input_sha256=current.checksums[result.block_id],
                            input_generation=current.input_generations[result.block_id],
                            group_id=attempt.id, proposal_attempt_id=attempt.id,
                            context_sha256=current.context_sha256,
                            source_binding=current.bindings[result.block_id],
                            original=dict(text=next(b.text for b in current.blocks if b.id == result.block_id)),
                            current_result=asdict(current.results[result.block_id]), result=asdict(result),
                            proposal=asdict(proposal), evidence_valid=evidence_valid)))
                        candidate_id = str(uuid4())
                        sha = canonical_sha(immutable)
                        if evidence_valid:
                            candidates.append(CorrectionCandidate(
                                candidate_id, command.book_id, current.run.id,
                                current.conditions.id, command.section_id, result.block_id,
                                current.checksums[result.block_id], current.input_generations[result.block_id],
                                sha, immutable, attempt.id, attempt.id,
                            ))
                        else:
                            candidates.append(CorrectionCandidate(
                                candidate_id, command.book_id, current.run.id,
                                current.conditions.id, command.section_id, result.block_id,
                                current.checksums[result.block_id], current.input_generations[result.block_id],
                                sha, immutable, attempt.id, attempt.id,
                                state='awaiting_human',
                                review={
                                    'candidate_id': candidate_id,
                                    'proposal_sha256': sha,
                                    'verdict': 'needs_human',
                                    'error_supported': False,
                                    'evidence_valid': False,
                                    'minimal': False,
                                    'no_semantic_loss': False,
                                    'reason': 'Цитата или доказательство в предложении не подтверждены текстом блока.',
                                },
                            ))
                    artifact = self._archive_response(tx, current, context, response, 'correction_proposal', request)
                    tx.work.add_correction_candidates(candidates)
                    tx.work.bind('correction_group', attempt.id, book_id=command.book_id,
                        conditions_id=current.conditions.id, attempt_id=attempt.id, artifact_revision_id=artifact,
                        payload=dict(schema_version=1, run_id=current.run.id, section_id=command.section_id,
                            manifest=expected, block_ids=list(block_ids), unchanged_block_ids=unchanged,
                            candidate_ids=[c.candidate_id for c in candidates]))
                    tx.work.transition_attempt(attempt.id, 'accepted', expected_state='validated')
                    has_awaiting_review = any(c.state == 'awaiting_review' for c in candidates)
                    if not has_awaiting_review:
                        tx.work.replace_correction_checks(current.run.id, command.section_id,
                            tuple(CorrectionCheck(current.run.id, b, command.section_id,
                                current.checksums[b], attempt.id, 'unchanged') for b in unchanged))
                    finalize_response(self.uow, response, 'accepted', transaction=tx)
                return len(block_ids) if not has_awaiting_review else 0
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
                _alias_error_for_retry(error, getattr(inputs, 'block_aliases', None))
                decision = self.retry.decide(
                    error,
                    attempts=ordinal,
                    raw_content=response.raw_content if response is not None else None,
                )
                if not decision.should_retry:
                    raise
                if decision.backoff_seconds:
                    await self.sleep(decision.backoff_seconds)
                next_req = decision.next_request(request)
                repair_turns = next_req.repair_turns
                response = None
        raise ResultValidationError('исчерпаны попытки предложения исправления')

    def _review_request(self, inputs, task, candidates):
        raw_analysis = (
            getattr(inputs, 'analysis', None)
            or (inputs.context.get('chapter_analysis') if isinstance(getattr(inputs, 'context', None), Mapping) else None)
        )
        difficulty_aliases = _difficulty_aliases(raw_analysis)
        target_block_ids = {c.block_id for c in candidates}
        context = _filtered_chapter_context(inputs, target_block_ids)
        for key in ('run_glossary', 'chapter_analysis', 'chapter_blocks'):
            context.setdefault(key, None)
        aliases = getattr(inputs, 'block_aliases', {})
        candidates_payload = [
            dict(
                candidate_id=f'c{idx}',
                block_id=aliases.get(c.block_id, c.block_id),
                original=thaw(c.payload['original']),
                current_result=_llm_block_result(
                    thaw(c.payload['current_result']),
                    aliases,
                    difficulty_aliases=difficulty_aliases,
                ),
                proposed_result=_llm_block_result(
                    thaw(c.payload['result']),
                    aliases,
                    difficulty_aliases=difficulty_aliases,
                ),
                issues=thaw(c.payload['proposal']['issues']),
            )
            for idx, c in enumerate(candidates, start=1)
        ]
        return LLMRequest.for_chapter_correction_review(
            instructions=self.snapshot.review_prompt,
            input_text=_request_input_json(context, payload_key='candidates', payload_value=candidates_payload),
            ordered_block_ids=tuple(c.block_id for c in candidates),
            candidate_refs=tuple((c.candidate_id, c.proposal_sha256) for c in candidates),
        ).with_logging(
            book_id=task.book_id,
            stage=LLMCallStage.CHAPTER_CORRECTION_REVIEW,
            task_id=task.id,
            scope_id=task.scope_id,
            run_id=inputs.run.id,
        )

    async def _review_group(self, command, candidate_ids):
        repair_turns = ()
        for ordinal in range(1, self.retry.max_attempts + 1):
            response = None
            with self.uow.transaction() as tx:
                inputs, task, payload = self._load(tx, command)
                candidates = tuple(tx.work.get_correction_candidate(i) for i in candidate_ids)
                if any(c is None or c.state != 'awaiting_review' or not candidate_is_current(tx, inputs, c)
                       for c in candidates):
                    raise RecoveryError('interrupted')
                max_chars = getattr(self.snapshot, 'parameters', {}).get('max_unit_source_chars')
                by_block = {c.block_id: c.candidate_id for c in candidates}
                chunks = _chunk_correction_blocks(tuple(by_block), inputs, max_chars)
            if len(chunks) > 1:
                total = 0
                for chunk in chunks:
                    total += await self._review_group(command, tuple(by_block[b] for b in chunk))
                return total
            with self.uow.transaction() as tx:
                inputs, task, payload = self._load(tx, command)
                candidates = tuple(tx.work.get_correction_candidate(i) for i in candidate_ids)
                if any(c is None or c.state != 'awaiting_review' or not candidate_is_current(tx, inputs, c)
                       for c in candidates):
                    raise RecoveryError('interrupted')
                refs = tuple((c.candidate_id, c.proposal_sha256) for c in candidates)
                base_request = self._review_request(inputs, task, candidates)
                if repair_turns:
                    base_request = replace(base_request, repair_turns=repair_turns)
                request, context = self._call(tx, inputs, task, payload,
                    base_request, ordinal,
                    kind='correction_review_request', refs=refs)
                expected = inputs.manifest
            try:
                response = await self.client.generate(request)
                if (response.truncated or not isinstance(response.payload, ChapterCorrectionReviewResponse)
                        or response.call_context != context):
                    raise LLMResponseFormatError(
                        'неполная проверка кандидатов',
                        truncated=response.truncated,
                        raw_content=response.raw_content,
                        usage=response.usage,
                        cost=response.cost,
                        provider=response.provider,
                        model=response.model,
                        call_context=response.call_context,
                    )
                resolved_payload = response.payload.with_resolved_proposal_shas(refs)
                if resolved_payload != response.payload:
                    response = replace(response, payload=resolved_payload)
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
                candidate_aliases = {
                    cid: f'c{idx}' for idx, cid in enumerate(candidate_ids, start=1)
                }
                _alias_error_for_retry(
                    error,
                    getattr(inputs, 'block_aliases', None),
                    candidate_aliases,
                )
                decision = self.retry.decide(
                    error,
                    attempts=ordinal,
                    raw_content=response.raw_content if response is not None else None,
                )
                if not decision.should_retry:
                    raise
                if decision.backoff_seconds:
                    await self.sleep(decision.backoff_seconds)
                next_req = decision.next_request(request)
                repair_turns = next_req.repair_turns
                response = None
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
