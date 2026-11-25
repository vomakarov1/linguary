"""Shared steps for processing book blocks: model invocation and result storage (ALG-04).

Chapter processing by translation units (`use_cases/process_chapter.py`) and single block
regeneration (`use_cases/regenerate_block.py`) differ in context composition and work granularity.
Everything else is shared and resides here: model call retry loop per policy, blocking response
validations, short save transaction checking active run, learning note numbering, and book completion.

The transaction intentionally does not span the model call (ADR § 3): while the response is in flight,
the user may return to glossary review; saving checks the active run against the initial run, writing
nothing if they diverge.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from contextlib import nullcontext, suppress
from dataclasses import dataclass, replace, asdict
from contextvars import ContextVar
from hashlib import sha256
from typing import Any

from linguary.application.context_builder import TranslationContext
from linguary.application.errors import ApplicationError
from linguary.application.explained_expressions import ExplainedExpressions
from linguary.application.llm_rejections import record_rejection, finalize_response
from linguary.application.ports.llm import (
    LLMClient, LLMConfigurationError, LLMContextCapacityError, LLMProviderError,
    LLMResponse, LLMResponseFormatError, LLMTransportError, TransportOutcome,
)
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.application.ports.pronunciation import PronunciationSource, PronunciationVariant
from linguary.application.validators.pronunciation import (
    PronunciationSelection, PronunciationSkip, preserve_unavailable, verify_candidates,
)
from linguary.application.retry_policy import (
    ResultValidationError, RetryPolicy, _ERROR_TYPES, classify_error, violation_of,
)
from linguary.application.recovery import RecoveryError, RecoverySplit
from linguary.application.book_conditions import ConditionsError
from linguary.application.token_budget import (
    IMPLICIT_REASONING_MIN_TOKENS,
    recommended_unit_max_completion_tokens,
)
from linguary.application.translation_policy import (
    require_policy,
    TranslationPolicyUnavailable,
    validation_policy_level,
)
from linguary.application.reader_text import ReaderTextPolicy, record_reader_warnings, source_quote_fragments
from linguary.application.validators import TranslationResultValidator
from linguary.application.validators.content_plan import resolve_content_coverage, resolve_response_evidence, validate_content_coverage
from linguary.domain.book_conditions import WorkStateError
from linguary.domain.content_plan import ContentEvidence
from linguary.domain.llm_log import Violation
from linguary.domain import (
    LLMCallStage,
    BlockProcessingResult,
    BookEvent,
    BookStatus,
    CanonicalDocument,
    DocumentSection,
    ErrorType,
    GlossarySnapshot,
    GlossarySnapshotEntry,
    ProcessingTask,
    TaskStatus,
    TaskType,
    TokenUsage,
    TranslationUnit,
    apply_book_event,
    build_language_note_id,
)

#: Pause between model call attempts: calculated by retry policy, observed
#: by use case — the application layer does not own the event loop (ALG-05).
type Sleep = Callable[[float], Awaitable[None]]


class BlockProcessingError(ApplicationError):
    """Book blocks cannot be processed in its current status."""


@dataclass(frozen=True, slots=True)
class TranslationCallResult:
    """Validated result of a single call along with its cost.

    `attempts` counts model requests rather than task executions: a single block group
    may cost multiple attempts, and the provider charges for each (PRD § 24).
    """

    results: tuple[BlockProcessingResult, ...]
    attempts: int = 0
    usage: TokenUsage | None = None
    pronunciation_skips: tuple[PronunciationSkip, ...] = ()
    pronunciation_unavailable_reason: str | None = None


class TranslationCall:
    """Executes model call with prepared context and atomically records the result (ALG-04)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        llm_client: LLMClient,
        retry_policy: RetryPolicy,
        validator: TranslationResultValidator,
        sleep: Sleep,
        pronunciation_source: PronunciationSource | None = None,
        pronunciation_help_enabled: bool = False,
        pronunciation_variant: PronunciationVariant | None = None,
        source_texts: Mapping[str, str] | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._llm_client = llm_client
        self._retry_policy = retry_policy
        self._validator = validator
        self._sleep = sleep
        self._pronunciation_source = pronunciation_source
        self._pronunciation_enabled = pronunciation_help_enabled
        self._pronunciation_variant = pronunciation_variant
        self._source_texts = source_texts
        self._validated_response: ContextVar[LLMResponse[Any] | None] = ContextVar(
            f"validated_response_{id(self)}", default=None
        )
        self._expected_bindings: ContextVar[Mapping[str, Mapping[str, object] | None] | None] = ContextVar(
            f"expected_bindings_{id(self)}", default=None
        )
        self._accepted_call_ids: set[str] = set()
        self._implicit_reasoning_lock = threading.Lock()
        self._implicit_reasoning_floor: int = 0

    def _get_implicit_reasoning_floor(self) -> int:
        with self._implicit_reasoning_lock:
            return self._implicit_reasoning_floor

    def _raise_implicit_reasoning_floor(self, candidate: int) -> None:
        if candidate <= 0:
            return
        with self._implicit_reasoning_lock:
            if candidate > self._implicit_reasoning_floor:
                self._implicit_reasoning_floor = candidate

    async def request(
        self,
        context: TranslationContext,
        *,
        unit: TranslationUnit,
        target_language: str,
        note_ids: Callable[[], str],
        recovery: tuple[str, str] | None = None,
        source_texts: Mapping[str, str] | None = None,
        glossary: GlossarySnapshot | None = None,
    ) -> TranslationCallResult:
        """Calls the model and validates the response, retrying per policy (ALG-04, ALG-05).

        Validations run inside the loop: rejecting a response is standard cause for retrying
        with clarification rather than failing the task. Once attempts are exhausted, the call
        raises the underlying error: classifying it and failing the task is the runner's responsibility.

        Hints and notes duplicating chapter `Before You Read` are filtered out of the validated
        result (PRD § 19.3): duplication is not grounds for rejecting the response.
        """
        self._validated_response.set(None)
        request = context.request
        explicit_context = request.call_context
        if recovery is None and explicit_context is not None and explicit_context.policy_version:
            raise RecoveryError('conditions_unavailable', explicit_context.root_id)
        policy = None
        run = None
        loaded_conditions = None
        checked_conditions_id = None
        resolved_source_texts = source_texts if source_texts is not None else self._source_texts
        need_conditions = (
            recovery is None
            and explicit_context is not None
            and explicit_context.conditions_id is not None
        )
        need_texts = (
            resolved_source_texts is None
            and bool(context.request.book_id)
            and (
                request.contract_version == 4
                or (context.content_plan is not None and context.content_plan.version == 3)
            )
        )
        if self._unit_of_work is not None and (recovery is not None or unit is not None or need_conditions or need_texts):
            with self._unit_of_work.transaction() as transaction:
                if recovery is not None:
                    root = transaction.work.get_root(recovery[0])
                    if root is None:
                        raise RecoveryError('conditions_unavailable', recovery[0])
                    try:
                        policy = require_policy(root.policy_version)
                    except TranslationPolicyUnavailable:
                        raise RecoveryError('conditions_unavailable', recovery[0]) from None
                    if explicit_context is not None and (
                        explicit_context.root_id != root.id
                        or explicit_context.policy_version != policy.version
                        or explicit_context.scope_id != recovery[1]
                    ):
                        raise RecoveryError('conditions_unavailable', root.id)
                if unit is not None:
                    run = (
                        transaction.runs.active_for_book(context.request.book_id)
                        if context.request.book_id and hasattr(transaction, 'runs')
                        else None
                    )
                    unit_block_ids = tuple(item.block.id for item in unit.items)
                    if run is not None:
                        run_bindings = _load_result_bindings(
                            transaction.work, run.id, unit_block_ids
                        )
                        self._expected_bindings.set({
                            bid: run_bindings.get(bid)
                            for bid in unit_block_ids
                        })
                    else:
                        self._expected_bindings.set({bid: None for bid in unit_block_ids})
                if need_conditions and hasattr(transaction.work, 'get_conditions'):
                    checked_conditions_id = explicit_context.conditions_id
                    loaded_conditions = transaction.work.get_conditions(checked_conditions_id)
                if need_texts and hasattr(transaction, 'documents'):
                    document = transaction.documents.get(context.request.book_id)
                    if document is not None:
                        resolved_source_texts = {
                            block.id: block.text
                            for section in document.sections
                            for block in section.blocks
                        }
            if unit is not None:
                request = replace(context.request, scope_id=unit.id,
                    ordered_block_ids=tuple(item.block.id for item in unit.items), run_id=None if run is None else run.id)
        resolved_glossary = glossary if glossary is not None else (None if run is None else run.glossary_snapshot)
        explained = ExplainedExpressions(
            context.explained_expressions, explanations=context.explained_entries
        )
        attempts = 0
        capacity_retries = 0
        last_error: Exception | None = None
        while True:
            attempts += 1
            response = None
            physical_context = None
            attempt_root_generation = None
            configured_max_output_tokens: int | None = None
            configured_reasoning_effort: str | None = None
            configured_reasoning_max_tokens: int | None = None
            try:
                if recovery is not None:
                    from linguary.application.diagnostics import diagnostics_for, is_transient_lock_error
                    diagnostics_for(self._unit_of_work).ensure_available()
                    stopped_reason = None
                    persisted_last_error: Exception | None = None
                    try:
                        with self._unit_of_work.transaction() as transaction:
                            root = transaction.work.get_root(recovery[0])
                            if root is None or policy is None or root.policy_version != policy.version:
                                raise RecoveryError('conditions_unavailable', recovery[0])
                            attempt_root_generation = getattr(root, 'generation', None)
                            checked_conditions_id = root.conditions_id
                            loaded_conditions = transaction.work.get_conditions(root.conditions_id)
                            if loaded_conditions is None:
                                raise RecoveryError('conditions_unavailable', recovery[0])
                            stage_value = (
                                request.stage.value
                                if request.stage is not None
                                else LLMCallStage.TRANSLATION_UNIT.value
                            )
                            physical_context = transaction.work.begin_recovery_attempt(*recovery,
                                task_id=request.task_id, run_id=request.run_id, stage=stage_value,
                                assigned_model=loaded_conditions.payload['models']['processing'],
                                prompt_hash=sha256(request.instructions.encode()).hexdigest(),
                                policy_version=policy.version)
                            if physical_context is None:
                                stopped = transaction.work.get_root(recovery[0])
                                stopped_reason = (
                                    getattr(stopped, 'reason', None)
                                    if stopped is not None
                                    else None
                                ) or 'limit'
                                if last_error is None and stopped_reason in {'limit', 'scope_limit'}:
                                    persisted_last_error = _attempt_original_error(
                                        transaction.work, recovery[0], recovery[1]
                                    )
                    except RecoveryError:
                        raise
                    except Exception as error:
                        if is_transient_lock_error(error):
                            raise
                        diagnostics_for(self._unit_of_work).fail()
                    if physical_context is None:
                        raise RecoveryError(
                            stopped_reason or 'limit',
                            recovery[0],
                            original_error=last_error if last_error is not None else persisted_last_error,
                        )
                    request = replace(request, call_context=physical_context,
                                      scope_id=physical_context.scope_id)
                if loaded_conditions is not None and isinstance(loaded_conditions.payload, Mapping):
                    raw_params = loaded_conditions.payload.get("llm_parameters")
                    if isinstance(raw_params, Mapping):
                        raw_max = raw_params.get("max_output_tokens")
                        if type(raw_max) is int and raw_max > 0:
                            configured_max_output_tokens = raw_max
                        raw_effort = raw_params.get("reasoning_effort")
                        if isinstance(raw_effort, str) and raw_effort.strip():
                            configured_reasoning_effort = raw_effort.strip().lower()
                        raw_reasoning_max = raw_params.get("reasoning_max_tokens")
                        if type(raw_reasoning_max) is int:
                            configured_reasoning_max_tokens = raw_reasoning_max
                if (
                    loaded_conditions is not None
                    and unit is not None
                    and request.max_completion_tokens is None
                ):
                    raw_params = loaded_conditions.payload.get("llm_parameters")
                    llm_params = raw_params if isinstance(raw_params, Mapping) else {}
                    limit = recommended_unit_max_completion_tokens(
                        sum(len(item.block.text) for item in unit.items),
                        max_output_tokens=llm_params.get("max_output_tokens"),
                        reasoning_effort=llm_params.get("reasoning_effort"),
                        reasoning_max_tokens=llm_params.get("reasoning_max_tokens"),
                        implicit_reasoning_tokens=self._get_implicit_reasoning_floor(),
                    )
                    request = replace(request, max_completion_tokens=limit)
                response = await self._llm_client.generate(request)
                resp_reasoning = getattr(getattr(response, "usage", None), "reasoning_tokens", None)
                if type(resp_reasoning) is int and resp_reasoning > 0:
                    self._raise_implicit_reasoning_floor(resp_reasoning)
                if physical_context is not None:
                    with self._unit_of_work.transaction() as transaction:
                        owns = transaction.work.owns_recovery_attempt(physical_context.attempt_id)
                    if not owns:
                        finalize_response(self._unit_of_work, response, 'stale')
                        raise RecoveryError('interrupted', None if recovery is None else recovery[0])
                if physical_context is not None and response.call_context != physical_context:
                    raise ConditionsError('conditions_unavailable')
                mapped = context.to_block_ids(response.payload)
                validation_policy = 'v1' if policy is None else policy.validation_policy
                effective_glossary: tuple[GlossarySnapshotEntry, ...] = ()
                if resolved_glossary is not None:
                    if validation_policy_level(validation_policy) >= 3:
                        effective_glossary = tuple(
                            entry
                            for entry in resolved_glossary.entries
                            if entry.status.is_user_decision
                            or entry.type.always_requires_review
                        )
                    else:
                        effective_glossary = resolved_glossary.mandatory_entries
                results = self._validator.validate(
                    mapped,
                    unit=unit,
                    target_language=target_language,
                    # In-memory ids only. save() replaces them when chapter_num is set.
                    note_id_factory=note_ids,
                    truncated=response.truncated,
                    expectation=context.expectation,
                    mandatory_glossary=effective_glossary,
                    validation_policy=validation_policy,
                    protected_quotes=() if unit is None else source_quote_fragments(
                        item.block.text for item in unit.items
                    ),
                )
                conditions = None
                call_context = getattr(response, 'call_context', None) or getattr(request, 'call_context', None)
                if call_context is not None:
                    if checked_conditions_id == call_context.conditions_id:
                        conditions = loaded_conditions
                    else:
                        with self._unit_of_work.transaction() as transaction:
                            conditions = transaction.work.get_conditions(call_context.conditions_id)
                        checked_conditions_id = call_context.conditions_id
                        loaded_conditions = conditions
                    if conditions is None:
                        raise ConditionsError('conditions_unavailable')
                validators_version = 1 if conditions is None else conditions.payload.get('versions', {}).get('validators', 1)
                if type(validators_version) is not int or validators_version not in (1, 3, 4):
                    raise ConditionsError('conditions_unavailable')
                a2_v4 = (validators_version == 4 and conditions is not None
                    and conditions.payload.get('reader_level') == 'a2')
                if validators_version == 4 and not a2_v4:
                    raise ConditionsError('conditions_unavailable')
                if context.content_plan is not None and context.content_plan.version == 3:
                    from linguary.application.analysis_merge import validate_fact_glossary, glossary_context_constraints
                    if resolved_glossary is None and self._unit_of_work is not None and request.book_id:
                        with self._unit_of_work.transaction() as transaction:
                            current_run = transaction.runs.active_for_book(request.book_id)
                        if current_run is not None:
                            resolved_glossary = current_run.glossary_snapshot
                    if resolved_glossary is not None:
                        validate_fact_glossary(context.content_plan, glossary_context_constraints(resolved_glossary.mandatory_entries))
                    restored = []
                    for result in results:
                        raw = next(block for block in mapped.blocks if block.source_block_id == result.block_id)
                        hints = tuple(replace(hint, original=source.original) for hint, source in zip(result.glossary_hints, raw.glossary_hints, strict=True))
                        notes = tuple(replace(note, original_excerpt=source.original_excerpt, translated_excerpt=source.translated_excerpt)
                            for note, source in zip(result.language_notes, mapped.notes_for(result.block_id), strict=True))
                        restored.append(replace(result, glossary_hints=hints, language_notes=notes))
                    checked_results = []
                    for restored_result in restored:
                        try:
                            checked_results.extend(ReaderTextPolicy().prepare_results((restored_result,),
                                protected_quotes=source_quote_fragments(item.block.text for item in unit.items)))
                        except ResultValidationError as error:
                            if error.violation is not None:
                                raise ResultValidationError(str(error), expectation=context.expectation,
                                    violation=replace(error.violation, scope='local', block_ids=(restored_result.block_id,))) from error
                            raise
                    results = tuple(checked_results)
                    policy_level = validation_policy_level(
                        'v1' if policy is None else policy.validation_policy
                    )
                    from linguary.application.validators.content_plan import validate_note_sources
                    validate_note_sources(
                        results,
                        {item.block.id: item.block.text for item in unit.items},
                        blocks={item.block.id: item.block for item in unit.items},
                        check_translated_excerpt=policy_level >= 3,
                    )
                    record_reader_warnings(self._unit_of_work, response,
                        [('translation', result.translated_text or '') for result in results],
                        reader_level=conditions.payload['reader_level'] if conditions else 'b2')
                if request.contract_version == 4:
                    from linguary.application.validators.content_plan import resolve_fragment_assistances
                    owners = tuple(r.block_id for r in results)
                    if resolved_source_texts is None:
                        document = None
                        if request.book_id is not None:
                            with self._unit_of_work.transaction() as transaction:
                                document = transaction.documents.get(request.book_id)
                        call_source_texts = (
                            {block.id: block.text for section in document.sections for block in section.blocks}
                            if document is not None
                            else {item.block.id: item.block.text for item in unit.items}
                        )
                        resolved_source_texts = call_source_texts
                    else:
                        call_source_texts = resolved_source_texts
                    assistance = resolve_fragment_assistances(context.content_plan, mapped, results, call_source_texts,
                        owner_block_ids=owners, before_you_read=context.before_you_read,
                        glossary_entries=() if resolved_glossary is None else resolved_glossary.mandatory_entries,
                        validators_version=validators_version,
                        blocks=tuple(item.block for item in unit.items))
                    results = tuple(replace(r, fragment_assistances=assistance[r.block_id]) for r in results)
                elif context.content_plan is not None:
                    owners = tuple(r.block_id for r in results)
                    if resolved_source_texts is not None:
                        call_source_texts = resolved_source_texts
                    else:
                        document = None
                        if request.book_id is not None and self._unit_of_work is not None:
                            with self._unit_of_work.transaction() as transaction:
                                docs = getattr(transaction, 'documents', None)
                                if docs is not None:
                                    document = docs.get(request.book_id)
                        call_source_texts = (
                            {block.id: block.text for section in document.sections for block in section.blocks}
                            if document is not None
                            else ({item.block.id: item.block.text for item in unit.items} if unit is not None else {})
                        )
                        resolved_source_texts = call_source_texts
                    evidence = resolve_response_evidence(
                        context.content_plan,
                        mapped,
                        results,
                        context.before_you_read,
                        owner_block_ids=owners,
                        source_texts=call_source_texts,
                    )
                    results = tuple(explained.without_repeats_v2(r) for r in results)
                    if a2_v4:
                        checked, unresolved = resolve_content_coverage(context.content_plan, results,
                            context.before_you_read, evidence, owner_block_ids=owners, source_texts=call_source_texts,
                            validators_version=validators_version)
                    else:
                        checked = validate_content_coverage(context.content_plan, results, context.before_you_read, evidence,
                            owner_block_ids=owners, validators_version=validators_version)
                        unresolved = {owner: () for owner in owners}
                    results = tuple(replace(r,
                        content_evidence=ContentEvidence(context.content_plan.sha256,
                            tuple(e for e in checked if e.block_id == r.block_id), version=context.content_plan.version,
                            structural_coverage='partial' if unresolved[r.block_id] else 'complete'),
                        fragment_assistances=unresolved[r.block_id]) for r in results)
                else:
                    results = tuple(explained.without_repeats(r) for r in results)
            except asyncio.CancelledError:
                if recovery is not None and physical_context is not None and self._unit_of_work is not None:
                    with suppress(Exception), self._unit_of_work.transaction() as transaction:
                        _finalize_cancelled_attempt(transaction.work, physical_context)
                raise
            except Exception as error:
                if isinstance(error, RecoveryError):
                    raise
                if response is not None:
                    record_rejection(self._unit_of_work, response, error)
                err_violation = getattr(error, "violation", None)
                is_truncated = (
                    (isinstance(error, LLMResponseFormatError) and error.truncated)
                    or (
                        err_violation is not None
                        and getattr(err_violation, "rule_code", None) == "provider_truncated"
                    )
                )
                if is_truncated:
                    err_reasoning = getattr(getattr(error, "usage", None), "reasoning_tokens", None)
                    observed_reasoning = (
                        err_reasoning if type(err_reasoning) is int and err_reasoning > 0 else 0
                    )
                    if observed_reasoning > 0:
                        self._raise_implicit_reasoning_floor(observed_reasoning)
                    if configured_reasoning_effort is None and configured_reasoning_max_tokens is None:
                        self._raise_implicit_reasoning_floor(
                            max(
                                observed_reasoning,
                                request.max_completion_tokens or IMPLICIT_REASONING_MIN_TOKENS,
                            )
                        )
                can_escalate_tokens = (
                    is_truncated
                    and configured_max_output_tokens is not None
                    and request.max_completion_tokens is not None
                    and request.max_completion_tokens < configured_max_output_tokens
                )
                outcome = _failure_outcome(error, state=None, saw_response=response is not None)
                if recovery is not None and physical_context is not None:
                    violation = violation_of(error, context=physical_context)
                    was_superseded = False
                    was_stale_gen = False
                    existing_reason = None
                    with self._unit_of_work.transaction() as transaction:
                        attempt = _attempt(transaction.work, physical_context.attempt_id)
                        if attempt is not None:
                            ctx_gen = (getattr(attempt, 'context', None) or {}).get('root_generation')
                            if ctx_gen is not None:
                                attempt_root_generation = ctx_gen
                        pre_root = transaction.work.get_root(recovery[0])
                        if pre_root is not None:
                            was_superseded = getattr(pre_root, 'superseded_by', None) is not None
                            was_stale_gen = (
                                attempt_root_generation is not None
                                and getattr(pre_root, 'generation', None) != attempt_root_generation
                            )
                            existing_reason = getattr(pre_root, 'reason', None)
                        outcome = _failure_outcome(
                            error,
                            state=None if attempt is None else attempt.state,
                            saw_response=response is not None,
                        )
                        split = _record_recovery_failure(
                            transaction.work, physical_context.attempt_id, violation, outcome,
                        )
                        if (
                            not split
                            and violation is not None
                            and violation.rule_code == "provider_truncated"
                            and not callable(getattr(transaction.work, "get_attempt", None))
                        ):
                            current_root = transaction.work.get_root(recovery[0])
                            split = _can_split_truncated_scope(
                                transaction.work, current_root, recovery[1], unit
                            )
                    if outcome == 'unknown':
                        if was_superseded:
                            raise RecoveryError('superseded', recovery[0], original_error=error) from error
                        if was_stale_gen and existing_reason != 'unknown_outcome':
                            raise RecoveryError('interrupted', recovery[0], original_error=error) from error
                        raise RecoveryError('unknown_outcome', recovery[0], original_error=error) from error
                    if can_escalate_tokens:
                        stale_reason = None
                        within_budget = False
                        with self._unit_of_work.transaction() as transaction:
                            root = transaction.work.get_root(recovery[0])
                            if root is not None:
                                if getattr(root, "superseded_by", None) is not None:
                                    stale_reason = "superseded"
                                elif (
                                    attempt_root_generation is not None
                                    and getattr(root, "generation", None) != attempt_root_generation
                                ):
                                    stale_reason = "interrupted"
                                elif (
                                    getattr(root, "status", None) == "stopped"
                                    and getattr(root, "reason", None)
                                    in {"unknown_outcome", "interrupted"}
                                ):
                                    stale_reason = getattr(root, "reason", None)
                                else:
                                    policy_map = (
                                        (root.manifest or {}).get("policy")
                                        if isinstance(getattr(root, "manifest", None), Mapping)
                                        else None
                                    )
                                    if isinstance(policy_map, Mapping):
                                        max_calls = policy_map.get("max_calls")
                                        max_scope_calls = policy_map.get("max_scope_calls")
                                        count_fn = getattr(
                                            transaction.work, "scope_attempt_count", None
                                        )
                                        within_budget = (
                                            type(max_calls) is int
                                            and type(max_scope_calls) is int
                                            and root.calls_used < max_calls + root.extra_calls
                                            and (
                                                not callable(count_fn)
                                                or count_fn(recovery[1]) < max_scope_calls
                                            )
                                        )
                        if stale_reason is not None:
                            raise RecoveryError(
                                stale_reason, recovery[0], original_error=error
                            ) from error
                        if within_budget:
                            request = replace(
                                request,
                                max_completion_tokens=configured_max_output_tokens,
                                attempt=request.attempt + 1,
                            )
                            last_error = error
                            continue
                    if split and (unit is None or len(unit.items) > 1) and not isinstance(error, LLMContextCapacityError):
                        failing_ids = (
                            tuple(violation.block_ids)
                            if violation is not None and violation.scope == "local"
                            else ()
                        )
                        raise RecoverySplit(*recovery, failing_block_ids=failing_ids) from None
                elif outcome == 'unknown' and recovery is not None:
                    raise RecoveryError('unknown_outcome', recovery[0], original_error=error) from error
                elif (
                    recovery is None
                    and can_escalate_tokens
                    and attempts < self._retry_policy.max_attempts
                ):
                    request = replace(
                        request,
                        max_completion_tokens=configured_max_output_tokens,
                        attempt=request.attempt + 1,
                    )
                    last_error = error
                    continue
                if isinstance(error, LLMContextCapacityError):
                    can_retry = (policy is not None and context.has_optional_neighbours
                        and capacity_retries < policy.capacity_retries_without_neighbours)
                    if recovery is not None:
                        stale_reason = None
                        with self._unit_of_work.transaction() as transaction:
                            root = transaction.work.get_root(recovery[0])
                            if root is not None:
                                if getattr(root, "superseded_by", None) is not None:
                                    stale_reason = "superseded"
                                elif (
                                    attempt_root_generation is not None
                                    and getattr(root, "generation", None) != attempt_root_generation
                                ):
                                    stale_reason = "interrupted"
                                elif (
                                    getattr(root, "status", None) == "stopped"
                                    and getattr(root, "reason", None) in {"unknown_outcome", "interrupted"}
                                ):
                                    stale_reason = getattr(root, "reason", None)
                                else:
                                    count = transaction.work.scope_attempt_count(recovery[1])
                                    within_budget = (
                                        root.calls_used < root.manifest['policy']['max_calls'] + root.extra_calls
                                        and count < root.manifest['policy']['max_scope_calls']
                                    )
                                    can_retry = can_retry and within_budget
                                    if not can_retry:
                                        try:
                                            _set_root_status(
                                                transaction.work,
                                                root,
                                                'context_capacity',
                                                pause=False,
                                                expected_generation=attempt_root_generation,
                                            )
                                        except WorkStateError:
                                            latest = transaction.work.get_root(recovery[0])
                                            stale_reason = (
                                                "superseded"
                                                if (
                                                    latest is not None
                                                    and getattr(latest, "superseded_by", None) is not None
                                                )
                                                else "interrupted"
                                            )
                            else:
                                can_retry = False
                        if stale_reason is not None:
                            raise RecoveryError(stale_reason, recovery[0], original_error=error) from error
                    if not can_retry:
                        if recovery is not None:
                            raise RecoveryError('context_capacity', recovery[0], original_error=error) from error
                        raise
                    capacity_retries += 1
                    last_error = error
                    context = context.without_optional_neighbours()
                    request = replace(request, input_text=context.request.input_text, attempt=request.attempt + 1)
                    continue
                decision = self._retry_policy.decide(
                    _worded_for_the_model(error, context),
                    attempts=attempts,
                    raw_content=getattr(response, "raw_content", None) if response is not None else None,
                )
                if not decision.should_retry:
                    if recovery is not None:
                        with self._unit_of_work.transaction() as transaction:
                            reason = _stop_or_pause_root(
                                transaction.work,
                                recovery,
                                error,
                                expected_generation=attempt_root_generation,
                            )
                        raise RecoveryError(reason, recovery[0], original_error=error) from error
                    raise
                if decision.backoff_seconds > 0 and self._sleep is not None:
                    await self._sleep(decision.backoff_seconds)
                last_error = error
                request = decision.next_request(request)
                continue
            self._validated_response.set(response)
            results, pronunciation_skips, pronunciation_unavailable_reason = await self._with_pronunciations(
                context, mapped, results, unit=unit, run_id=request.run_id)
            return TranslationCallResult(
                results=results,
                attempts=attempts,
                usage=response.usage,
                pronunciation_skips=pronunciation_skips,
                pronunciation_unavailable_reason=pronunciation_unavailable_reason,
            )

    async def _with_pronunciations(self, context, mapped, results, *, unit, run_id):
        """Source access follows successful semantic validation, outside model repair."""
        if not self._pronunciation_enabled:
            return results, (), None
        texts = {item.block.id: item.block.text for item in unit.items}
        with self._unit_of_work.transaction() as transaction:
            stored_by_id = _load_results_by_block_id(
                transaction.results,
                run_id,
                tuple(result.block_id for result in results),
            )
            previous = tuple(
                assistance
                for result in results
                if (stored := stored_by_id.get(result.block_id)) is not None
                for assistance in stored.pronunciation_assistances
            )
        reason = None
        if context.request.pronunciation_candidates_version is None:
            reason = 'candidate_contract_unavailable'
        elif self._pronunciation_source is None or not self._pronunciation_source.configured:
            reason = 'source_disabled'
        if reason is not None or self._pronunciation_source is None:
            selection = PronunciationSelection(unavailable=True)
        else:
            selection = await verify_candidates(mapped.pronunciation_candidates,
                source=self._pronunciation_source, texts=texts, variant=self._pronunciation_variant,
                scope_id=run_id)
        assistances = preserve_unavailable(selection, previous, texts=texts, variant=self._pronunciation_variant)
        return (tuple(replace(result, pronunciation_assistances=tuple(item for item in assistances
                    if item.anchor.block_id == result.block_id)) for result in results), selection.skipped, reason)

    def save(
        self, *, book_id: str, run_id: str, results: Sequence[BlockProcessingResult],
        pronunciation_unavailable: bool = False, reconcile: bool = True,
        chapter_num: int | None = None, transaction=None,
    ) -> bool:
        """Atomically replaces results of the specified blocks (ALG-04, PRD § 17).

        Returns `False` if there is no active run to write to, or if the run ID does not
        match. This discards stale results of calls started before returning to glossary review.
        A temporary pronunciation failure preserves accepted help; a checked empty result clears it.
        `transaction` joins an already open unit of work. Final language-note ids are reserved
        in that same transaction when `chapter_num` is set, and only after ownership checks pass.
        """
        response = self._validated_response.get()
        context = None if response is None else response.call_context
        saved = False
        block_ids = tuple(result.block_id for result in results)
        expected_bindings = self._expected_bindings.get() or {}
        opened_here = transaction is None
        transaction_scope = nullcontext(transaction) if transaction is not None else self._unit_of_work.transaction()
        with transaction_scope as transaction:
            run = transaction.runs.active_for_book(book_id)
            conditions = transaction.work.latest_conditions(book_id) if context else None
            pointers_match = True
            if context is not None and results:
                current_bindings = _load_result_bindings(
                    transaction.work, run_id, block_ids
                )
                for result in results:
                    previous = expected_bindings.get(result.block_id)
                    current = current_bindings.get(result.block_id)
                    if (None if previous is None else previous['attempt_id']) != (
                        None if current is None else current['attempt_id']
                    ):
                        pointers_match = False
            owns_scope = context is None or context.root_id is None or transaction.work.owns_recovery_attempt(context.attempt_id)
            if run is not None and run.id == run_id and pointers_match and owns_scope and (context is None or (conditions is not None and conditions.id == context.conditions_id)):
                previous_results = (
                    _load_results_by_block_id(transaction.results, run_id, block_ids)
                    if results
                    else {}
                )
                if context is not None:
                    legacy_artifacts = [
                        dict(
                            book_id=book_id,
                            run_id=run_id,
                            kind='block_result',
                            scope_id=result.block_id,
                            body={'schema_version': 1, 'result': asdict(previous_result)},
                        )
                        for result in results
                        if (previous_result := previous_results.get(result.block_id)) is not None
                        and expected_bindings.get(result.block_id) is None
                    ]
                    if legacy_artifacts:
                        transaction.work.archive_artifacts(legacy_artifacts)
                if pronunciation_unavailable:
                    results = tuple(
                        replace(result, pronunciation_assistances=previous.pronunciation_assistances)
                        if (previous := previous_results.get(result.block_id)) is not None
                        else result
                        for result in results
                    )
                results = tuple(
                    _retain_saved_segments(result, previous_results.get(result.block_id))
                    for result in results
                )
                results = _with_allocated_note_ids(
                    transaction, results, run_id=run_id, chapter_num=chapter_num,
                )
                text_changed = any(
                    previous_results.get(result.block_id) != result
                    for result in results
                )
                transaction.results.replace_results(run_id, results)
                if context is not None and context.root_id is not None:
                    scope = transaction.work.get_scope(context.scope_id)
                    if tuple(result.block_id for result in results) != scope.ordered_blocks:
                        raise RecoveryError('unsafe_partition', context.root_id)
                if context is not None and results:
                    provenance = None if context.root_id is None else {
                        'schema_version': 1, 'root_id': context.root_id, 'policy_version': context.policy_version,
                        'prompt_hash': context.prompt_hash, 'attempt_id': context.attempt_id}
                    new_artifacts = [
                        dict(
                            book_id=book_id,
                            conditions_id=context.conditions_id,
                            attempt_id=context.attempt_id,
                            run_id=run_id,
                            kind='block_result',
                            scope_id=result.block_id,
                            body={
                                'schema_version': 1,
                                'result': asdict(result),
                                **({'provenance': provenance} if provenance is not None else {}),
                            },
                        )
                        for result in results
                    ]
                    artifact_ids = transaction.work.archive_artifacts(new_artifacts)
                    for result, artifact_id in zip(results, artifact_ids, strict=True):
                        previous_binding = expected_bindings.get(result.block_id)
                        transaction.work.replace_result_binding(
                            f'{run_id}:{result.block_id}',
                            book_id=book_id,
                            conditions_id=context.conditions_id,
                            attempt_id=context.attempt_id,
                            artifact_revision_id=artifact_id,
                            provenance=provenance,
                            expected_attempt_id=None if previous_binding is None else previous_binding['attempt_id'],
                        )
                if context is not None and context.root_id is not None:
                    transaction.work.mark_recovery_accepted(context.attempt_id)
                finalize_response(self._unit_of_work, response, 'accepted', transaction=transaction)
                if text_changed:
                    transaction.work.stale_reviews(book_id)
                if context is not None and context.root_id is not None:
                    transaction.work.root_coverage_complete(context.root_id)
                from linguary.application.correction import invalidate_correction
                invalidate_correction(
                    transaction,
                    book_id,
                    run_id,
                    (result.block_id for result in results),
                    reconcile=reconcile,
                )
                saved = True
        call_id = getattr(response, "call_id", None)
        if saved and opened_here and call_id is not None:
            self._accepted_call_ids.add(call_id)
        already_accepted = call_id is not None and call_id in self._accepted_call_ids
        if not saved and opened_here and not already_accepted:
            finalize_response(self._unit_of_work, response, 'stale')
        return saved


def note_id_factory(*, chapter_number: int, used: Collection[str]) -> Callable[[], str]:
    """Factory for chapter learning note IDs formatted as `ln-ch03-001` (PRD § 19.3).

    IDs are unique within a run, so numbering continues past already stored notes of this
    chapter: neither re-running a partially processed chapter nor regenerating an individual
    block will collide with existing notes. Gaps in numbering are permitted — IDs must be
    unique, not strictly consecutive.
    """
    ordinal = 0

    def next_id() -> str:
        nonlocal ordinal
        while True:
            ordinal += 1
            note_id = build_language_note_id(
                chapter_number=chapter_number, ordinal=ordinal
            )
            if note_id not in used:
                return note_id

    return next_id


def note_numbering_from_binding(binding: Mapping[str, Any] | None) -> str:
    """Extract `note_numbering` mode from a `run` work binding, defaulting to `'reading-order'`."""
    if not isinstance(binding, Mapping):
        return "reading-order"
    payload = binding.get("payload_json")
    if not isinstance(payload, Mapping):
        return "reading-order"
    value = payload.get("note_numbering")
    return value if isinstance(value, str) and value else "reading-order"


def run_note_numbering(transaction: object, run_id: str) -> str:
    """Read `note_numbering` from the `run` work binding with safe `work` guard."""
    work_repo = getattr(transaction, "work", None)
    get_binding = getattr(work_repo, "get_binding", None)
    if not callable(get_binding):
        return "reading-order"
    return note_numbering_from_binding(get_binding("run", run_id))


def chapter_number(document: CanonicalDocument, section: DocumentSection) -> int:
    """Chapter number in reading order: included in note IDs (PRD § 19.3)."""
    sections = document.ordered_sections
    return next(
        (
            number
            for number, other in enumerate(sections, start=1)
            if other.id == section.id
        ),
        section.order + 1,
    )


def note_chapter_number(
    document: CanonicalDocument,
    section: DocumentSection,
    *,
    note_numbering: str = "reading-order",
) -> int:
    """Chapter number for persistent language note IDs (`ln-chNN-NNN`).

    `"reading-order"` preserves legacy 1-based spine order across all sections.
    `"story-v1"` numbers story chapters (`not is_front_matter and not is_notes_only`)
    as `1..N`, and non-story sections as `N + non_story_index` (`>= 1` even when `N == 0`).
    """
    if note_numbering != "story-v1":
        return chapter_number(document, section)
    sections = document.ordered_sections
    is_front = getattr(document, "is_front_matter", lambda _s: False)
    story: list[DocumentSection] = []
    non_story: list[DocumentSection] = []
    for other in sections:
        if not is_front(other) and not getattr(other, "is_notes_only", False):
            story.append(other)
        else:
            non_story.append(other)
    story_count = len(story)
    for number, other in enumerate(story, start=1):
        if other.id == section.id:
            return number
    for non_story_index, other in enumerate(non_story, start=1):
        if other.id == section.id:
            return story_count + non_story_index
    return max(story_count + len(non_story) + 1, section.order + 1, 1)


def complete_processing(
    unit_of_work: UnitOfWork,
    *,
    book_id: str,
    finished_task_id: str | None,
    task_types: Collection[TaskType],
) -> None:
    """Transitions the book to `READY_TO_EXPORT` when all work is finished (ALG-06).

    A task executing right now is still marked `running`: the runner records its final status
    only after the handler returns. Therefore its success is recorded right here, in the same
    transaction with its siblings: otherwise during concurrent processing (PRD § 25), two
    chapters completing almost simultaneously see each other still running, and the book remains
    in `PROCESSING` forever. Re-recording the same outcome by the runner is a no-op.
    A book that left `PROCESSING` does not change status: while the call was in flight,
    the user could have returned to glossary review.
    """
    with unit_of_work.transaction() as transaction:
        complete_processing_in_transaction(transaction, book_id=book_id,
            finished_task_id=finished_task_id, task_types=task_types)


def complete_processing_in_transaction(transaction, *, book_id, finished_task_id, task_types):
    """Recompute mandatory coverage and readiness within the caller's transaction."""
    finished_task = _record_success(transaction, finished_task_id)
    from linguary.application.correction import reconcile_corrections, correction_coverage_complete
    section_id = finished_task.scope_id if finished_task is not None and finished_task.scope_id else None
    if section_id is not None:
        try:
            reconcile_corrections(transaction, book_id, section_id=section_id)
        except TypeError:
            reconcile_corrections(transaction, book_id)
    else:
        reconcile_corrections(transaction, book_id)
    book = book_record(transaction, book_id)
    if book.status is not BookStatus.PROCESSING:
        return
    unfinished = tuple(
        task
        for task in transaction.tasks.list_for_book(book_id)
        if task.task_type in {*task_types, TaskType.CHAPTER_CORRECTION}
        and task.id != finished_task_id
        and task.status not in {TaskStatus.SUCCEEDED, TaskStatus.SUPERSEDED}
    )
    if unfinished:
        return
    run = transaction.runs.active_for_book(book_id)
    roots = transaction.work.list_active_roots(book_id, run_id=run.id) if run is not None else ()
    # A stopped regeneration is a finished side operation. It must not pin the book in
    # PROCESSING after translation coverage is complete. A stopped translation root still blocks.
    blocking = tuple(
        root for root in roots
        if not (getattr(root, 'operation_kind', 'translation') == 'regeneration' and root.status == 'stopped')
    )
    if any(not transaction.work.root_coverage_complete(root.id) for root in blocking):
        raise RecoveryError('unsafe_partition')
    if not correction_coverage_complete(transaction, book_id):
        return
    document = transaction.documents.get(book_id)
    if run is not None and document is not None:
        from linguary.application.context_builder import textual_blocks
        expected = {block.id for section in document.sections for block in textual_blocks(section)}
        actual = {result.block_id for result in transaction.results.list_for_run(run.id)}
        if not actual <= expected:
            raise RecoveryError('unsafe_partition')
        if expected != actual:
            return
    work_repo = getattr(transaction, "work", None)
    set_book_paused = getattr(work_repo, "set_book_paused", None)
    if callable(set_book_paused):
        from linguary.application.illustrations import has_unfinished_illustration_work

        if not has_unfinished_illustration_work(transaction, book_id):
            set_book_paused(book_id, None, False)
    transaction.books.save(
        replace(
            book,
            status=apply_book_event(book.status, BookEvent.PROCESSING_COMPLETED),
        )
    )



def _record_success(transaction: UnitOfWork, task_id: str | None) -> ProcessingTask | None:
    """Marks running task as successful if it still exists and is running.

    Tasks of a cancelled run are deleted along with it, so a missing row is not an error.
    """
    if task_id is None:
        return None
    task = transaction.tasks.get(task_id)
    if task is not None and task.status is TaskStatus.RUNNING:
        task = task.succeed()
        transaction.tasks.save(task)
    return task


def book_record(transaction: UnitOfWork, book_id: str) -> BookRecord:
    """Book record within an active transaction."""
    book = transaction.books.get(book_id)
    if book is None:
        raise BlockProcessingError(f"книги {book_id} нет в хранилище")
    return book


def total_usage(usages: Iterable[TokenUsage | None]) -> TokenUsage | None:
    """Total token usage: provider tracks each call separately.

    A provider reporting no token counts leaves `None`: zero would signify a free call
    rather than missing data (PRD § 24).
    """
    reported = [usage for usage in usages if usage is not None]
    if not reported:
        return None
    return TokenUsage(
        prompt_tokens=_total_tokens(usage.prompt_tokens for usage in reported),
        completion_tokens=_total_tokens(usage.completion_tokens for usage in reported),
    )


def _total_tokens(values: Iterable[int | None]) -> int | None:
    """Sum of token counts reported by the provider."""
    reported = [value for value in values if value is not None]
    return sum(reported) if reported else None


def _can_split_truncated_scope(
    work_repo,
    root,
    scope_id: str,
    unit: TranslationUnit | None,
) -> bool:
    if (
        root is None
        or getattr(root, "superseded_by", None) is not None
        or unit is None
        or len(unit.items) <= 1
    ):
        return False
    manifest = getattr(root, "manifest", None) or {}
    if "atomic_intervals" in manifest:
        scope_set = set(unit.block_ids)
        atoms = [
            tuple(group)
            for group in manifest["atomic_intervals"]
            if set(group) <= scope_set
        ]
        if len(atoms) < 2:
            return False
    list_scopes = getattr(work_repo, "list_scopes", None)
    policy = manifest.get("policy") or {}
    if callable(list_scopes) and "max_depth" in policy and "max_leaves" in policy:
        scopes = list_scopes(root.id)
        by_id = {item.id: item for item in scopes}
        scope = by_id.get(scope_id)
        if scope is not None:
            depth, parent = 0, scope
            while parent.parent_id is not None:
                depth += 1
                parent = by_id[parent.parent_id]
            leaves = sum(item.status != "split" for item in scopes)
            if depth >= policy["max_depth"] or leaves + 1 > policy["max_leaves"]:
                return False
    return True


def _retain_saved_segments(result, previous):
    """Keep segment fields when a later replace leaves both tuples empty.

    A different treatment or joined translation is stored as given: copying the
    old slots onto it would break the domain join.
    """
    if previous is None:
        return result
    if result.segment_treatments or result.segment_translations:
        return result
    if not previous.segment_treatments and not previous.segment_translations:
        return result
    if result.treatment is not previous.treatment or result.translated_text != previous.translated_text:
        return result
    return replace(
        result,
        segment_treatments=previous.segment_treatments,
        segment_translations=previous.segment_translations,
    )


def _load_result_bindings(
    work_repo, run_id: str, block_ids: Sequence[str]
) -> dict[str, dict | None]:
    """Loads result bindings only for the specified blocks when supported."""
    result_bindings_fn = getattr(work_repo, "result_bindings", None)
    if callable(result_bindings_fn):
        try:
            return dict(result_bindings_fn(run_id, block_ids=block_ids))
        except TypeError:
            return dict(result_bindings_fn(run_id))
    return {
        block_id: work_repo.get_binding("result", f"{run_id}:{block_id}")
        for block_id in dict.fromkeys(block_ids)
    }


def _load_results_by_block_id(
    results_repo, run_id: str, block_ids: Sequence[str]
) -> dict[str, BlockProcessingResult]:
    """Loads existing block results in a single batch when supported by the repository."""
    get_many = getattr(results_repo, "get_many", None)
    if callable(get_many):
        return dict(get_many(run_id, block_ids))
    loaded: dict[str, BlockProcessingResult] = {}
    for block_id in dict.fromkeys(block_ids):
        stored = results_repo.get(run_id, block_id)
        if stored is not None:
            loaded[block_id] = stored
    return loaded


def _accepts_keyword(method, name: str) -> bool:
    """Whether `method` can take `name` without a TypeError from older fakes."""
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    if name in parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _attempt(work, attempt_id):
    getter = getattr(work, "get_attempt", None)
    if not callable(getter):
        return None
    return getter(attempt_id)


def _failure_outcome(error: Exception, *, state: str | None, saw_response: bool) -> str:
    """Classify a failed send without turning a known response into `unknown`."""
    if state == "unknown":
        return "unknown"
    if isinstance(error, LLMTransportError):
        if error.outcome == "not_sent":
            return "not_sent"
        if error.outcome == "unknown":
            return "unknown"
        return "known"
    if isinstance(error, ConditionsError) or any(
        isinstance(error, kind) for kind, _task_type in _ERROR_TYPES
    ):
        return "known"
    sent_states = {"sent", "validated", "rejected", "failed", "accepted"}
    if saw_response or state in sent_states:
        return "unknown"
    return "known"


def _record_recovery_failure(work, attempt_id, violation, outcome: str) -> bool:
    """Record the failure. A rolled-back `not_sent` is already cancelled."""
    attempt = _attempt(work, attempt_id)
    state = None if attempt is None else attempt.state
    if outcome == "not_sent":
        if state == "cancelled":
            return False
        if state not in {None, "created"}:
            outcome = "unknown" if state == "unknown" else "known"
    recorder = work.record_recovery_failure
    try:
        if _accepts_keyword(recorder, "outcome"):
            return recorder(attempt_id, violation, outcome=outcome)
        return recorder(attempt_id, violation)
    except WorkStateError:
        return False


def _finalize_cancelled_attempt(work, physical_context) -> None:
    """Cancel a call that never left `created`; an in-flight send stays unknown."""
    attempt_id = physical_context.attempt_id
    attempt = _attempt(work, attempt_id)
    state = None if attempt is None else attempt.state
    if state == "created":
        work.cancel_unsent_attempt(attempt_id)
        return
    if state in {"sent", "unknown"}:
        violation = violation_of(
            LLMTransportError("вызов отменён после отправки", outcome="unknown"),
            context=physical_context,
        )
        _record_recovery_failure(work, attempt_id, violation, "unknown")


def _set_root_status(
    work, root, reason: str, *, pause: bool, expected_generation: int | None = None
) -> None:
    generation = (
        expected_generation
        if expected_generation is not None
        else getattr(root, "generation", None)
    )
    if pause:
        pause_root = getattr(work, "pause_root", None)
        if callable(pause_root):
            if _accepts_keyword(pause_root, "expected_generation"):
                pause_root(root.id, reason, expected_generation=generation)
            else:
                pause_root(root.id, reason)
            return
    stop_root = work.stop_root
    if _accepts_keyword(stop_root, "expected_generation"):
        stop_root(root.id, reason, expected_generation=generation)
    else:
        stop_root(root.id, reason)


def _pause_reason(error: Exception) -> str:
    if isinstance(error, LLMProviderError) and not isinstance(error, LLMContextCapacityError):
        return "provider_error"
    if isinstance(error, LLMConfigurationError):
        return "configuration_error"
    kind = classify_error(error).type
    if kind in {ErrorType.VALIDATION, ErrorType.MODEL_OUTPUT}:
        return "validation_error"
    if kind is ErrorType.CONFIGURATION:
        return "configuration_error"
    return "transient_error"


def _terminal_reason(error: Exception, *, calls_exhausted: bool, scope_exhausted: bool) -> tuple[str, bool]:
    """Stop reason, or a pause reason while the durable budget still remains."""
    violation = getattr(error, "violation", None)
    rule = getattr(violation, "rule_code", None)
    if isinstance(error, ConditionsError):
        return "conditions_unavailable", False
    truncated = rule == "provider_truncated" or (
        isinstance(error, LLMResponseFormatError) and error.truncated
    )
    if truncated:
        return "output_truncated", False
    if rule in {"provider_refusal", "provider_content_filter"}:
        return "provider_refusal", False
    if calls_exhausted:
        return "limit", False
    if scope_exhausted:
        return "scope_limit", False
    return _pause_reason(error), True


def _stop_or_pause_root(
    work, recovery, error: Exception, *, expected_generation: int | None = None
) -> str:
    root = work.get_root(recovery[0])
    if root is None:
        return "conditions_unavailable"
    if getattr(root, "superseded_by", None) is not None:
        return "superseded"
    if (
        expected_generation is not None
        and getattr(root, "generation", None) != expected_generation
    ):
        return "interrupted"
    root_reason = getattr(root, "reason", None)
    if (
        getattr(root, "status", None) == "stopped"
        and root_reason in {"unknown_outcome", "interrupted"}
    ):
        return root_reason
    count = work.scope_attempt_count(recovery[1])
    limit = root.manifest["policy"]["max_calls"] + root.extra_calls
    reason, pause = _terminal_reason(
        error,
        calls_exhausted=root.calls_used >= limit,
        scope_exhausted=count >= root.manifest["policy"]["max_scope_calls"],
    )
    try:
        _set_root_status(
            work,
            root,
            reason,
            pause=pause,
            expected_generation=expected_generation,
        )
    except WorkStateError:
        latest = work.get_root(recovery[0])
        if latest is not None and getattr(latest, "superseded_by", None) is not None:
            return "superseded"
        return "interrupted"
    return reason


def _with_allocated_note_ids(transaction, results, *, run_id: str, chapter_num: int | None):
    """Replace temporary note ids after ownership checks, in the caller's transaction."""
    if chapter_num is None:
        return results
    allocate = getattr(getattr(transaction, "results", None), "allocate_note_ids", None)
    if not callable(allocate):
        return results
    notes = tuple(note for result in results for note in result.language_notes)
    if not notes:
        return results
    issued = tuple(allocate(run_id, chapter_num, len(notes)))
    if len(issued) != len(notes):
        raise ValueError("note id allocation did not cover every note")
    issued_ids = iter(issued)
    return tuple(
        replace(
            result,
            language_notes=tuple(
                replace(note, id=next(issued_ids)) for note in result.language_notes
            ),
        ) if result.language_notes else result
        for result in results
    )


def _attempt_original_error(work, root_id: str, scope_id: str) -> Exception | None:
    """Reconstruct last failed attempt's error type after restart when in-memory last_error is absent."""
    list_root_attempts = getattr(work, "list_root_attempts", None)
    if not callable(list_root_attempts):
        return None
    attempts = [
        item
        for item in list_root_attempts(root_id)
        if getattr(item, "state", None) in {"rejected", "failed", "unknown"}
    ]
    if not attempts:
        return None
    scope_attempts = [item for item in attempts if getattr(item, "scope_id", None) == scope_id]
    latest = max(scope_attempts or attempts, key=lambda item: getattr(item, "ordinal", 0))
    raw_violation = (getattr(latest, "outcome", None) or {}).get("violation")
    if not isinstance(raw_violation, Mapping):
        if getattr(latest, "state", None) == "rejected":
            return ResultValidationError("Превышен лимит попыток восстановления")
        return None
    rule = str(raw_violation.get("rule_code") or "unknown")
    msg = str(raw_violation.get("safe_message") or "Превышен лимит попыток восстановления")
    stage = str(raw_violation.get("stage") or "translation_unit")
    scope = str(raw_violation.get("scope") or "unknown")
    block_ids = tuple(str(b) for b in (raw_violation.get("block_ids") or ()) if b)
    if scope not in {"local", "unit", "global", "unknown"} or (scope == "local" and not block_ids):
        scope = "unknown"
        block_ids = ()
    violation = Violation(rule, stage, scope=scope, block_ids=block_ids, safe_message=msg)
    if rule == "provider_truncated":
        return LLMResponseFormatError(msg, truncated=True, violation=violation)
    if rule in {
        "response_schema",
        "response_parse",
        "response_format",
        "response_domain",
        "response_coverage",
        "provider_refusal",
        "provider_content_filter",
    }:
        return LLMResponseFormatError(msg, violation=violation)
    if rule.startswith("transport_"):
        t_outcome: TransportOutcome = (
            "unknown"
            if rule == "transport_unknown"
            else ("not_sent" if rule == "transport_not_sent" else "known_transient")
        )
        return LLMTransportError(msg, outcome=t_outcome)
    return ResultValidationError(msg, violation=violation)


def _aliased_violation(violation, context: TranslationContext):
    if violation is not None and violation.block_ids:
        aliases_by_block_id = {
            block_id: alias
            for alias, block_id in (getattr(context, "block_ids_by_alias", None) or {}).items()
        }
        aliased_ids = tuple(aliases_by_block_id.get(b, b) for b in violation.block_ids)
        if aliased_ids != violation.block_ids:
            return replace(violation, block_ids=aliased_ids)
    return violation


def _worded_for_the_model(error: Exception, context: TranslationContext) -> Exception:
    """Same error rephrased using block aliases seen by the model (ALG-05).

    The rejection reason is inserted into the retry clarification instruction, while
    validators formulate it with stable block IDs. The expected block list remains as
    populated by the validator. Outwardly — into logs and task state — the original error
    with real block IDs is propagated.
    """
    if isinstance(error, LLMResponseFormatError):
        violation = getattr(error, "violation", None)
        if violation is None or violation.scope != "local" or not violation.block_ids:
            return error
        cloned = LLMResponseFormatError(
            context.to_aliases(str(error)),
            truncated=error.truncated,
            raw_content=error.raw_content,
            usage=error.usage,
            cost=error.cost,
            provider=error.provider,
            model=error.model,
            violation=_aliased_violation(violation, context),
            body_meta=error.body_meta,
            call_context=error.call_context,
            expectation=getattr(error, "expectation", None) or getattr(context, "expectation", None),
        )
        cloned.call_id = getattr(error, "call_id", None)
        return cloned
    if not isinstance(error, ResultValidationError):
        return error
    return ResultValidationError(
        context.to_aliases(str(error)),
        expectation=error.expectation or getattr(context, "expectation", None),
        violation=_aliased_violation(getattr(error, "violation", None), context),
    )


__all__ = [
    "BlockProcessingError",
    "Sleep",
    "TranslationCall",
    "TranslationCallResult",
    "book_record",
    "chapter_number",
    "complete_processing",
    "note_chapter_number",
    "note_id_factory",
    "note_numbering_from_binding",
    "run_note_numbering",
    "total_usage",
]
