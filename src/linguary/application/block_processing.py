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

from collections.abc import Awaitable, Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, replace, asdict
from contextvars import ContextVar
from hashlib import sha256

from linguary.application.context_builder import TranslationContext
from linguary.application.errors import ApplicationError
from linguary.application.explained_expressions import ExplainedExpressions
from linguary.application.llm_rejections import record_rejection, finalize_response
from linguary.application.ports.llm import LLMClient, LLMContextCapacityError
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.application.ports.pronunciation import PronunciationSource, PronunciationVariant
from linguary.application.validators.pronunciation import (
    PronunciationSelection, PronunciationSkip, preserve_unavailable, verify_candidates,
)
from linguary.application.retry_policy import ResultValidationError, RetryPolicy, violation_of
from linguary.application.recovery import RecoveryError, RecoverySplit
from linguary.application.book_conditions import ConditionsError
from linguary.application.translation_policy import require_policy, TranslationPolicyUnavailable
from linguary.application.reader_text import ReaderTextPolicy, record_reader_warnings, source_quote_fragments
from linguary.application.validators import TranslationResultValidator
from linguary.application.validators.content_plan import resolve_content_coverage, resolve_response_evidence, validate_content_coverage
from linguary.domain.content_plan import ContentEvidence
from linguary.domain import (
    BlockProcessingResult,
    BookEvent,
    BookStatus,
    CanonicalDocument,
    DocumentSection,
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
    ) -> None:
        self._unit_of_work = unit_of_work
        self._llm_client = llm_client
        self._retry_policy = retry_policy
        self._validator = validator
        self._sleep = sleep
        self._pronunciation_source = pronunciation_source
        self._pronunciation_enabled = pronunciation_help_enabled
        self._pronunciation_variant = pronunciation_variant
        self._validated_response = ContextVar(f"validated_response_{id(self)}", default=None)
        self._expected_bindings = ContextVar(f"expected_bindings_{id(self)}", default={})

    async def request(
        self,
        context: TranslationContext,
        *,
        unit: TranslationUnit,
        target_language: str,
        note_ids: Callable[[], str],
        recovery: tuple[str, str] | None = None,
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
        if recovery is not None:
            with self._unit_of_work.transaction() as transaction:
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
        if self._unit_of_work is not None and unit is not None:
            with self._unit_of_work.transaction() as transaction:
                run = transaction.runs.active_for_book(context.request.book_id) if context.request.book_id else None
                self._expected_bindings.set({item.block.id: (
                    transaction.work.get_binding('result', f'{run.id}:{item.block.id}') if run else None
                ) for item in unit.items})
            request = replace(context.request, scope_id=unit.id,
                ordered_block_ids=tuple(item.block.id for item in unit.items), run_id=None if run is None else run.id)
        explained = ExplainedExpressions(
            context.explained_expressions, explanations=context.explained_entries
        )
        attempts = 0
        capacity_retries = 0
        while True:
            attempts += 1
            response = None
            physical_context = None
            try:
                if recovery is not None:
                    from linguary.application.diagnostics import diagnostics_for
                    diagnostics_for(self._unit_of_work).ensure_available()
                    try:
                        with self._unit_of_work.transaction() as transaction:
                            root = transaction.work.get_root(recovery[0])
                            if root is None or root.policy_version != policy.version:
                                raise RecoveryError('conditions_unavailable', recovery[0])
                            conditions = transaction.work.get_conditions(root.conditions_id)
                            if conditions is None:
                                raise RecoveryError('conditions_unavailable', recovery[0])
                            physical_context = transaction.work.begin_recovery_attempt(*recovery,
                                task_id=request.task_id, run_id=request.run_id, stage=request.stage.value,
                                assigned_model=conditions.payload['models']['processing'],
                                prompt_hash=sha256(request.instructions.encode()).hexdigest(),
                                policy_version=policy.version)
                    except RecoveryError:
                        raise
                    except Exception:
                        diagnostics_for(self._unit_of_work).fail()
                    if physical_context is None:
                        raise RecoveryError('limit', recovery[0])
                    request = replace(request, call_context=physical_context,
                                      scope_id=physical_context.scope_id)
                response = await self._llm_client.generate(request)
                if physical_context is not None:
                    with self._unit_of_work.transaction() as transaction:
                        owns = transaction.work.owns_recovery_attempt(physical_context.attempt_id)
                    if not owns:
                        finalize_response(self._unit_of_work, response, 'stale')
                        raise RecoveryError('interrupted', recovery[0])
                if physical_context is not None and response.call_context != physical_context:
                    raise ConditionsError('conditions_unavailable')
                mapped = context.to_block_ids(response.payload)
                results = self._validator.validate(
                    mapped,
                    unit=unit,
                    target_language=target_language,
                    note_id_factory=note_ids,
                    truncated=response.truncated,
                    expectation=context.expectation,
                )
                conditions = None
                current_run = None
                call_context = response.call_context or request.call_context
                if call_context is not None:
                    with self._unit_of_work.transaction() as transaction:
                        conditions = transaction.work.get_conditions(call_context.conditions_id)
                    if conditions is None:
                        raise ConditionsError('conditions_unavailable')
                validators_version = 1 if conditions is None else conditions.payload['versions']['validators']
                if type(validators_version) is not int or validators_version not in (1, 3, 4):
                    raise ConditionsError('conditions_unavailable')
                a2_v4 = (validators_version == 4 and conditions is not None
                    and conditions.payload.get('reader_level') == 'a2')
                if validators_version == 4 and not a2_v4:
                    raise ConditionsError('conditions_unavailable')
                if context.content_plan is not None and context.content_plan.version == 3:
                    from linguary.application.analysis_merge import validate_fact_glossary, glossary_context_constraints
                    with self._unit_of_work.transaction() as transaction:
                        current_run = transaction.runs.active_for_book(request.book_id)
                    if current_run is not None:
                        validate_fact_glossary(context.content_plan, glossary_context_constraints(current_run.glossary_snapshot.mandatory_entries))
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
                    from linguary.application.validators.content_plan import validate_note_sources
                    validate_note_sources(results, {item.block.id: item.block.text for item in unit.items})
                    record_reader_warnings(self._unit_of_work, response,
                        [('translation', result.translated_text or '') for result in results],
                        reader_level=conditions.payload['reader_level'] if conditions else 'b2')
                if request.contract_version == 4:
                    from linguary.application.validators.content_plan import resolve_fragment_assistances
                    owners = tuple(r.block_id for r in results)
                    with self._unit_of_work.transaction() as transaction:
                        document = transaction.documents.get(request.book_id)
                    source_texts = {block.id: block.text for section in document.sections for block in section.blocks}
                    assistance = resolve_fragment_assistances(context.content_plan, mapped, results, source_texts,
                        owner_block_ids=owners, before_you_read=context.before_you_read,
                        glossary_entries=() if current_run is None else current_run.glossary_snapshot.mandatory_entries,
                        validators_version=validators_version)
                    results = tuple(replace(r, fragment_assistances=assistance[r.block_id]) for r in results)
                elif context.content_plan is not None:
                    owners = tuple(r.block_id for r in results)
                    evidence = resolve_response_evidence(context.content_plan, mapped, results, context.before_you_read, owner_block_ids=owners)
                    results = tuple(explained.without_repeats_v2(r) for r in results)
                    document = None
                    if a2_v4:
                        with self._unit_of_work.transaction() as transaction:
                            document = transaction.documents.get(request.book_id)
                    if a2_v4:
                        source_texts = ({block.id: block.text for section in document.sections for block in section.blocks}
                            if document is not None else {item.block.id: item.block.text for item in unit.items})
                        checked, unresolved = resolve_content_coverage(context.content_plan, results,
                            context.before_you_read, evidence, owner_block_ids=owners, source_texts=source_texts,
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
            except Exception as error:
                if isinstance(error, RecoveryError):
                    raise
                if response is not None:
                    record_rejection(self._unit_of_work, response, error)
                if recovery is not None and physical_context is not None:
                    with self._unit_of_work.transaction() as transaction:
                        split = transaction.work.record_recovery_failure(physical_context.attempt_id,
                            violation_of(error, context=physical_context))
                    if split and not isinstance(error, LLMContextCapacityError):
                        raise RecoverySplit(*recovery) from None
                if isinstance(error, LLMContextCapacityError):
                    can_retry = (policy is not None and context.has_optional_neighbours
                        and capacity_retries < policy.capacity_retries_without_neighbours
                        and attempts < self._retry_policy.max_attempts)
                    if recovery is not None:
                        with self._unit_of_work.transaction() as transaction:
                            root = transaction.work.get_root(recovery[0])
                            count = transaction.work.scope_attempt_count(recovery[1])
                            can_retry = (can_retry and root.calls_used < root.manifest['policy']['max_calls'] + root.extra_calls
                                and count < root.manifest['policy']['max_scope_calls'])
                            if not can_retry:
                                transaction.work.stop_root(root.id, 'context_capacity')
                    if not can_retry:
                        if recovery is not None:
                            raise RecoveryError('context_capacity', recovery[0], original_error=error) from error
                        raise
                    capacity_retries += 1
                    context = context.without_optional_neighbours()
                    request = replace(request, input_text=context.request.input_text, attempt=request.attempt + 1)
                    continue
                decision = self._retry_policy.decide(
                    _worded_for_the_model(error, context), attempts=attempts
                )
                if not decision.should_retry:
                    if recovery is not None:
                        with self._unit_of_work.transaction() as transaction:
                            root = transaction.work.get_root(recovery[0])
                            count = transaction.work.scope_attempt_count(recovery[1])
                            reason = ('conditions_unavailable' if isinstance(error, ConditionsError) else
                                'limit' if (root.calls_used >= root.manifest['policy']['max_calls'] + root.extra_calls
                                or count >= root.manifest['policy']['max_scope_calls']) else 'unsafe_partition')
                            transaction.work.stop_root(root.id, reason)
                        raise RecoveryError(reason, recovery[0], original_error=error) from error
                    raise
                if decision.backoff_seconds > 0:
                    await self._sleep(decision.backoff_seconds)
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
            previous = tuple(assistance for result in results
                if (stored := transaction.results.get(run_id, result.block_id)) is not None
                for assistance in stored.pronunciation_assistances)
        reason = None
        if context.request.pronunciation_candidates_version is None:
            reason = 'candidate_contract_unavailable'
        elif self._pronunciation_source is None or not self._pronunciation_source.configured:
            reason = 'source_disabled'
        if reason is not None:
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
        pronunciation_unavailable: bool = False,
    ) -> bool:
        """Atomically replaces results of the specified blocks (ALG-04, PRD § 17).

        Returns `False` if there is no active run to write to, or if the run ID does not
        match. This discards stale results of calls started before returning to glossary review.
        A temporary pronunciation failure preserves accepted help; a checked empty result clears it.
        """
        response = self._validated_response.get()
        context = None if response is None else response.call_context
        saved = False
        with self._unit_of_work.transaction() as transaction:
            run = transaction.runs.active_for_book(book_id)
            conditions = transaction.work.latest_conditions(book_id) if context else None
            pointers_match = True
            if context is not None:
                for result in results:
                    previous = self._expected_bindings.get().get(result.block_id)
                    current = transaction.work.get_binding('result', f'{run_id}:{result.block_id}')
                    if (None if previous is None else previous['attempt_id']) != (None if current is None else current['attempt_id']):
                        pointers_match = False
            owns_scope = context is None or context.root_id is None or transaction.work.owns_recovery_attempt(context.attempt_id)
            if run is not None and run.id == run_id and pointers_match and owns_scope and (context is None or (conditions is not None and conditions.id == context.conditions_id)):
                if context is not None:
                    for result in results:
                        previous_result = transaction.results.get(run_id, result.block_id)
                        previous_binding = self._expected_bindings.get().get(result.block_id)
                        if previous_result is not None and previous_binding is None:
                            transaction.work.archive_artifacts([dict(book_id=book_id, run_id=run_id,
                                kind='block_result', scope_id=result.block_id,
                                body={'schema_version': 1, 'result': asdict(previous_result)})])
                if pronunciation_unavailable:
                    preserved = []
                    for result in results:
                        previous = transaction.results.get(run_id, result.block_id)
                        preserved.append(replace(result, pronunciation_assistances=previous.pronunciation_assistances)
                                         if previous is not None else result)
                    results = tuple(preserved)
                text_changed = any(transaction.results.get(run_id, result.block_id) != result for result in results)
                transaction.results.replace_results(run_id, results)
                if context is not None and context.root_id is not None:
                    scope = transaction.work.get_scope(context.scope_id)
                    if tuple(result.block_id for result in results) != scope.ordered_blocks:
                        raise RecoveryError('unsafe_partition', context.root_id)
                if context is not None:
                    provenance = None if context.root_id is None else {
                        'schema_version': 1, 'root_id': context.root_id, 'policy_version': context.policy_version,
                        'prompt_hash': context.prompt_hash, 'attempt_id': context.attempt_id}
                    for result in results:
                        artifact_id, = transaction.work.archive_artifacts([dict(book_id=book_id,
                            conditions_id=context.conditions_id, attempt_id=context.attempt_id, run_id=run_id,
                            kind='block_result', scope_id=result.block_id,
                            body={'schema_version': 1, 'result': asdict(result),
                                  **({'provenance': provenance} if provenance is not None else {})})])
                        previous_binding = self._expected_bindings.get().get(result.block_id)
                        transaction.work.replace_result_binding(f'{run_id}:{result.block_id}', book_id=book_id,
                            conditions_id=context.conditions_id, attempt_id=context.attempt_id,
                            artifact_revision_id=artifact_id, provenance=provenance,
                            expected_attempt_id=None if previous_binding is None else previous_binding['attempt_id'])
                if context is not None and context.root_id is not None:
                    transaction.work.mark_recovery_accepted(context.attempt_id)
                finalize_response(self._unit_of_work, response, 'accepted', transaction=transaction)
                if text_changed:
                    transaction.work.stale_reviews(book_id)
                if context is not None and context.root_id is not None:
                    transaction.work.root_coverage_complete(context.root_id)
                from linguary.application.correction import invalidate_correction
                invalidate_correction(transaction, book_id, run_id,
                                      (result.block_id for result in results))
                saved = True
        if not saved:
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
    _record_success(transaction, finished_task_id)
    from linguary.application.correction import reconcile_corrections, correction_coverage_complete
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
    if any(not transaction.work.root_coverage_complete(root.id) for root in roots):
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
    transaction.books.save(
        replace(
            book,
            status=apply_book_event(book.status, BookEvent.PROCESSING_COMPLETED),
        )
    )



def _record_success(transaction: UnitOfWork, task_id: str | None) -> None:
    """Marks running task as successful if it still exists and is running.

    Tasks of a cancelled run are deleted along with it, so a missing row is not an error.
    """
    if task_id is None:
        return
    task = transaction.tasks.get(task_id)
    if task is not None and task.status is TaskStatus.RUNNING:
        transaction.tasks.save(task.succeed())


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


def _worded_for_the_model(error: Exception, context: TranslationContext) -> Exception:
    """Same error rephrased using block aliases seen by the model (ALG-05).

    The rejection reason is inserted into the retry clarification instruction, while
    validators formulate it with stable block IDs. The expected block list remains as
    populated by the validator. Outwardly — into logs and task state — the original error
    with real block IDs is propagated.
    """
    if not isinstance(error, ResultValidationError):
        return error
    return ResultValidationError(
        context.to_aliases(str(error)), expectation=error.expectation
    )


__all__ = [
    "BlockProcessingError",
    "Sleep",
    "TranslationCall",
    "TranslationCallResult",
    "book_record",
    "chapter_number",
    "complete_processing",
    "note_id_factory",
    "total_usage",
]
