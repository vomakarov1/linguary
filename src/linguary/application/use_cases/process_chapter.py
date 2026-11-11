"""Chapter processing use case and processing task sequence (ALG-04, PRD § 14-17).

`ProcessChapter` splits a chapter into groups of blocks (`translation_units`), processes each
group with a separate model call, and saves its result atomically. The compute unit is
the group; the storage unit is the block: validators reject the entire group, so a partially
saved group does not exist (PRD § 17).

The ordering for a single group is defined by ALG-04: assemble context, invoke the model,
map block labels back to block IDs, run blocking validators and warnings (`validators`),
then in a short transaction verify that the book's run is still active and is the same run,
and replace block results for the group. Verifying the run inside the transaction discards
stale results: the user may have returned to glossary review while the model call was in flight (P3.T7).

The chapter task is a dispatcher whose binding persists exact root IDs and policy versions.
Translation units are rebuilt from unresolved scopes of those roots; retries preserve the
original operations and skip accepted scopes (PRD § 18.1).

`ChapterProcessingSequence` enqueues chapter tasks and transitions the book to `READY_TO_EXPORT`
when all chapters have been processed. The processing run remains active: full export looks up
results via `ProcessingRunRepository.active_for_book`, and a closed run would leave the book
without translations. The run is closed only when returning to glossary review, which deletes it
along with its results (P3.T7).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from hashlib import sha256
import threading
from uuid import uuid4

from linguary.application.block_processing import (
    BlockProcessingError,
    Sleep,
    TranslationCall,
    chapter_number,
    complete_processing,
    note_id_factory,
    total_usage,
)
from linguary.application.context_builder import ContextBuilder, textual_blocks
from linguary.application.ports.llm import LLMClient, LLMResponse
from linguary.application.ports.pronunciation import PronunciationSource, PronunciationVariant
from linguary.application.ports.persistence import UnitOfWork
from linguary.application.retry_policy import RetryPolicy
from linguary.application.recovery import Recovery, RecoveryError, RecoveryPolicy, RecoverySplit
from linguary.application.translation_compatibility import ensure_legacy_operation
from linguary.application.translation_policy import LEGACY_TRANSLATION_POLICY
from linguary.application.task_runner import TaskQueue
from linguary.application.translation_units import (
    DEFAULT_MAX_UNIT_SOURCE_CHARS,
    plan_translation_units,
)
from linguary.application.validators import TranslationResultValidator
from linguary.domain.content_plan import ContentEvidence
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    CanonicalDocument,
    ChapterAnalysis,
    DocumentSection,
    GlossaryCandidate,
    LLMCallStage,
    ProcessingRun,
    ProcessingTask,
    TaskIdentity,
    TaskStatus,
    TaskType,
    TokenUsage,
    TranslationUnit,
)

#: Factory for translation unit IDs.
type IdFactory = Callable[[], str]


class ProcessChapterError(BlockProcessingError):
    """Chapter cannot be processed in its current state."""


@dataclass(frozen=True, slots=True)
class ProcessChapterCommand:
    """Processing command: which chapter of which book to process."""

    book_id: str
    section_id: str
    task_id: str | None = None
    root_ids: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class ProcessChapterResult:
    """Outcome of chapter processing along with technical call metrics.

    `units` — how many groups the chapter was split into; `processed_units` — how many of them
    were processed by this task execution: retries skip groups whose results are already
    saved. `attempts` and `usage` aggregate model requests across all groups in the chapter:
    a single processing task can cost multiple calls (PRD § 24).
    """

    section_id: str
    run_id: str
    units: int = 0
    processed_units: int = 0
    saved_blocks: int = 0
    attempts: int = 0
    usage: TokenUsage | None = None
    discarded: bool = False


@dataclass(frozen=True, slots=True)
class _ChapterWork:
    """Everything needed for chapter processing, loaded in a single transaction."""

    document: CanonicalDocument
    section: DocumentSection
    analysis: ChapterAnalysis
    run: ProcessingRun
    book_candidates: tuple[GlossaryCandidate, ...]
    processed_block_ids: frozenset[str]
    used_note_ids: frozenset[str]
    task_id: str | None = None
    source_texts: Mapping[str, str] = field(default_factory=dict)
    pronunciation_help_enabled: bool = False

    @property
    def target_language(self) -> str:
        """Book target language."""
        return self.document.target_language

    @property
    def chapter_number(self) -> int:
        """Chapter number in reading order: included in language note IDs (PRD § 19.3)."""
        return chapter_number(self.document, self.section)


class ProcessChapter:
    """Processes a chapter by translation units and saves results of each unit (ALG-04)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        llm_client: LLMClient,
        retry_policy: RetryPolicy,
        validator: TranslationResultValidator,
        max_unit_source_chars: int = DEFAULT_MAX_UNIT_SOURCE_CHARS,
        max_concurrent_units: int = 1,
        context_builder: ContextBuilder | None = None,
        pronunciation_source: PronunciationSource | None = None,
        pronunciation_help_enabled: bool = False,
        pronunciation_variant: PronunciationVariant | None = None,
        id_factory: IdFactory | None = None,
        sleep: Sleep | None = None,
        recovery_policy: RecoveryPolicy | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._call = TranslationCall(
            unit_of_work=unit_of_work,
            llm_client=llm_client,
            retry_policy=retry_policy,
            validator=validator,
            sleep=sleep or asyncio.sleep,
            pronunciation_source=pronunciation_source,
            pronunciation_help_enabled=pronunciation_help_enabled,
            pronunciation_variant=pronunciation_variant,
        )
        self._max_unit_source_chars = max_unit_source_chars
        self._max_concurrent_units = max(1, max_concurrent_units)
        self._context_builder = context_builder or ContextBuilder()
        self._id_factory = id_factory or _new_id
        self._recovery = Recovery(unit_of_work, policy=recovery_policy)
        self._validator = validator

    async def __call__(self, command: ProcessChapterCommand) -> ProcessChapterResult:
        """Processes all translation units of the chapter (PRD § 10, § 14).

        A chapter without textual blocks — cover, image-only page — is handled
        without model calls: it has no units, nothing to save.
        """
        work, roots = self.prepare(command)
        note_ids = note_id_factory(
            chapter_number=work.chapter_number, used=work.used_note_ids
        )
        result = ProcessChapterResult(
            section_id=work.section.id, run_id=work.run.id, units=len(roots)
        )
        try:
            return await self._process_roots(roots, work=work, note_ids=note_ids, result=result)
        except RecoveryError as error:
            if error.reason == 'interrupted' and command.task_id is None:
                return replace(result, processed_units=result.processed_units + 1, discarded=True)
            if command.task_id is None and error.original_error is not None:
                raise error.original_error from error
            raise

    def prepare(
        self, command: ProcessChapterCommand, *, new_dispatcher: bool = False,
        transaction=None,
    ) -> tuple[_ChapterWork, tuple[str, ...]]:
        """Pin a new dispatch once, or resume only its persisted operation references."""
        transaction_context = nullcontext(transaction) if transaction is not None else self._unit_of_work.transaction()
        with transaction_context as tx:
            work = self._load(command, transaction=tx)
            units = plan_translation_units(
                work.section.ordered_blocks,
                work.analysis.treatment_plan,
                id_factory=self._id_factory,
                max_source_chars=self._max_unit_source_chars,
                scenes=work.analysis.scenes,
            )
            refs = None
            conditions = tx.work.latest_conditions(command.book_id)
            if conditions is None:
                raise RecoveryError('conditions_unavailable')
            if command.task_id is not None:
                task = tx.tasks.get(command.task_id)
                if (task is None or task.book_id != command.book_id
                        or task.scope_id != command.section_id
                        or task.task_type is not TaskType.PROCESS_TRANSLATION_UNIT
                        or task.status is TaskStatus.SUPERSEDED
                        or not task.operation_id.startswith('dispatch:') or task.policy_version != ''):
                    raise RecoveryError('conditions_unavailable')
                binding = tx.work.get_binding('task', task.id)
                if binding is None and not new_dispatcher:
                    raise RecoveryError('conditions_unavailable')
                if binding is not None and binding['conditions_id'] != conditions.id:
                    raise RecoveryError('conditions_unavailable')
                if binding is not None:
                    refs = binding['payload_json'].get('operation_refs')
                if refs is None and not new_dispatcher:
                    refs = ensure_legacy_operation(tx, task=task, conditions=conditions)
            if command.root_ids is not None:
                if len(set(command.root_ids)) != len(command.root_ids):
                    raise RecoveryError('conditions_unavailable')
                if refs is None:
                    roots = tuple(tx.work.get_root(root_id) for root_id in command.root_ids)
                    if any(root is None for root in roots):
                        raise RecoveryError('conditions_unavailable')
                    refs = tuple({'root_id': root.id, 'policy_version': root.policy_version} for root in roots)
                else:
                    by_id = {ref['root_id']: ref for ref in refs}
                    if not set(command.root_ids) <= by_id.keys():
                        raise RecoveryError('conditions_unavailable')
                    refs = tuple(by_id[root_id] for root_id in command.root_ids)
            roots = self._recovery.prepare_roots(
                work, units, max_source_chars=self._max_unit_source_chars,
                validator=self._validator, operation_refs=refs, transaction=tx,
            )
            return work, roots

    async def _process_unit(
        self,
        unit: TranslationUnit,
        *,
        work: _ChapterWork,
        note_ids: Callable[[], str],
        result: ProcessChapterResult,
        recovery: tuple[str, str],
        root=None,
    ) -> ProcessChapterResult:
        """One ALG-04 cycle: context assembly, model call, validation, atomic save."""
        all_read_as_is = all(
            work.analysis.treatment_plan.treatment_for(b_id) is BlockTreatment.READ_AS_IS
            for b_id in unit.block_ids
        )
        plan = work.analysis.content_plan
        unit_block_ids = set(unit.block_ids)
        covered_points = (
            set()
            if plan is None or work.section.is_notes_only
            else {(e.difficulty_id, e.point_id) for e in plan.before_you_read_evidence}
        )
        has_uncovered_owned_difficulties = bool(
            plan
            and any(
                d.owner_block_id in unit_block_ids
                and (
                    not d.required_points
                    or any((d.id, p.id) not in covered_points for p in d.required_points)
                )
                for d in plan.difficulties
            )
        )
        if (
            all_read_as_is
            and not has_uncovered_owned_difficulties
            and not work.pronunciation_help_enabled
        ):
            byr_by_block = (
                {
                    b_id: tuple(
                        e for e in plan.before_you_read_evidence if e.block_id == b_id
                    )
                    for b_id in unit.block_ids
                }
                if plan is not None and not work.section.is_notes_only
                else {}
            )
            results = tuple(
                BlockProcessingResult(
                    block_id=b_id,
                    treatment=BlockTreatment.READ_AS_IS,
                    translated_text=None,
                    glossary_hints=(),
                    language_notes=(),
                    content_evidence=(
                        ContentEvidence(
                            plan.sha256,
                            byr_by_block.get(b_id, ()),
                            version=plan.version,
                        )
                        if plan
                        else ()
                    ),
                )
                for b_id in unit.block_ids
            )
            with self._unit_of_work.transaction() as transaction:
                root = transaction.work.get_root(recovery[0])
                if root is None:
                    raise RecoveryError('conditions_unavailable', recovery[0])
                conditions = transaction.work.get_conditions(root.conditions_id)
                if conditions is None:
                    raise RecoveryError('conditions_unavailable', recovery[0])
                physical_context = transaction.work.begin_recovery_attempt(
                    *recovery,
                    task_id=work.task_id,
                    run_id=work.run.id,
                    stage=LLMCallStage.TRANSLATION_UNIT.value,
                    assigned_model=conditions.payload['models']['processing'],
                    prompt_hash=sha256(b"read_as_is").hexdigest(),
                    policy_version=root.policy_version,
                )
                if physical_context is None:
                    raise RecoveryError('limit', recovery[0])
                transaction.work.transition_attempt(physical_context.attempt_id, 'sent', expected_state='created')
                transaction.work.transition_attempt(
                    physical_context.attempt_id,
                    'validated',
                    expected_state='sent',
                    outcome={'schema_version': 1, 'provider': 'local', 'model': 'local', 'call_id': None},
                )
                expected = {
                    b_id: transaction.work.get_binding('result', f'{work.run.id}:{b_id}')
                    for b_id in unit.block_ids
                }
            response = LLMResponse(raw_content="", payload=None, provider="local", model="local", call_context=physical_context, call_id=None)
            self._call._validated_response.set(response)
            self._call._expected_bindings.set(expected)
            saved = self._call.save(
                book_id=work.document.id,
                run_id=work.run.id,
                results=results,
                reconcile=False,
            )
            return replace(
                result,
                processed_units=result.processed_units + 1,
                saved_blocks=result.saved_blocks + (len(results) if saved else 0),
                attempts=result.attempts,
                usage=result.usage,
                discarded=not saved,
            )

        if root is None:
            with self._unit_of_work.transaction() as transaction:
                root = transaction.work.get_root(recovery[0])
        recovery_child = tuple(unit.block_ids) != tuple(root.manifest['ordered_blocks'])
        context = self._context_builder.for_translation_unit(
            document=work.document,
            section=work.section,
            unit=unit,
            analysis=work.analysis,
            glossary=work.run.glossary_snapshot,
            book_candidates=work.book_candidates,
            recovery_child=recovery_child,
            policy_version=root.policy_version,
        )
        context = replace(
            context,
            request=context.request.with_logging(
                book_id=work.document.id,
                stage=LLMCallStage.TRANSLATION_UNIT,
                task_id=work.task_id,
            ),
        )
        if recovery_child and root.policy_version == LEGACY_TRANSLATION_POLICY:
            blocks = {block.id: block for block in work.section.ordered_blocks}
            readonly = '\n'.join(
                f'[read-only context] {blocks[block].text}' for block in root.manifest['ordered_blocks']
            )
            context = replace(context, request=replace(context.request,
                input_text=context.request.input_text + '\n\nКонтекст исходной сцены; эти строки не являются дополнительными целями изменения:\n' + readonly))
        source_texts = work.source_texts or {
            block.id: block.text
            for block in work.section.blocks
        }
        call = await self._call.request(
            context, unit=unit, target_language=work.target_language, note_ids=note_ids,
            recovery=recovery,
            source_texts=source_texts,
        )
        saved = self._call.save(
            book_id=work.document.id,
            run_id=work.run.id,
            results=call.results,
            reconcile=False,
        )
        return replace(
            result,
            processed_units=result.processed_units + 1,
            saved_blocks=result.saved_blocks + (len(call.results) if saved else 0),
            attempts=result.attempts + call.attempts,
            usage=total_usage((result.usage, call.usage)),
            discarded=not saved,
        )

    async def _process_roots(self, roots, *, work, note_ids, result):
        with self._unit_of_work.transaction() as transaction:
            roots_by_id = {root: transaction.work.get_root(root) for root in roots}
            before = sum(r.calls_used for r in roots_by_id.values())
        blocks = {block.id: block for block in work.section.ordered_blocks}
        result = replace(result, units=len(roots))

        semaphore = asyncio.Semaphore(self._max_concurrent_units)
        note_id_lock = threading.Lock()

        def safe_note_ids() -> str:
            with note_id_lock:
                return note_ids()

        async def _process_single_root(root_id: str) -> ProcessChapterResult:
            async with semaphore:
                root_result = ProcessChapterResult(
                    section_id=work.section.id,
                    run_id=work.run.id,
                )
                while scopes := self._recovery.plan_unresolved(root_id):
                    scope = scopes[0]
                    unit = TranslationUnit.of_blocks(
                        id=scope.id,
                        blocks=tuple(blocks[block] for block in scope.ordered_blocks),
                        plan=work.analysis.treatment_plan,
                    )
                    try:
                        root_result = await self._process_unit(
                            unit,
                            work=work,
                            note_ids=safe_note_ids,
                            result=root_result,
                            recovery=(root_id, scope.id),
                            root=roots_by_id.get(root_id),
                        )
                    except RecoverySplit:
                        self._recovery.split(root_id, scope.id)
                        continue
                    if root_result.discarded:
                        return root_result
                self._recovery.assert_complete(root_id)
                return root_result

        tasks = [asyncio.create_task(_process_single_root(root_id)) for root_id in roots]
        try:
            root_results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        processed_units = sum(r.processed_units for r in root_results)
        saved_blocks = sum(r.saved_blocks for r in root_results)
        usages = tuple(r.usage for r in root_results if r.usage is not None)
        discarded = any(r.discarded for r in root_results)

        with self._unit_of_work.transaction() as transaction:
            used = sum(transaction.work.get_root(root).calls_used for root in roots) - before
        return replace(
            result,
            processed_units=processed_units,
            saved_blocks=saved_blocks,
            attempts=used,
            usage=total_usage(usages),
            discarded=discarded,
        )

    def _load(self, command: ProcessChapterCommand, *, transaction=None) -> _ChapterWork:
        """Loads the book, chapter analysis, and active run in a single transaction."""
        transaction_context = nullcontext(transaction) if transaction is not None else self._unit_of_work.transaction()
        with transaction_context as tx:
            document = tx.documents.get(command.book_id)
            if document is None:
                raise ProcessChapterError(f"книги {command.book_id} нет в хранилище")
            section = _section_of(document, command)
            all_analyses = tx.analyses.list_for_book(command.book_id)
            analysis = (
                next((ch for ch in all_analyses if ch.section_id == section.id), None)
                or tx.analyses.get_for_section(command.book_id, command.section_id)
            )
            if analysis is None:
                raise ProcessChapterError(
                    f"глава {command.section_id} не проанализирована: "
                    f"обрабатывать её нельзя"
                )
            analysis.ensure_ready_for_processing()
            run = tx.runs.active_for_book(command.book_id)
            if run is None:
                raise ProcessChapterError(
                    f"у книги {command.book_id} нет активного запуска обработки"
                )
            list_for_section = getattr(tx.results, "list_for_section", None)
            stored = (
                list_for_section(run.id, section.id)
                if callable(list_for_section)
                else tx.results.list_for_run(run.id)
            )
            book_candidates = tuple(
                candidate
                for chapter in all_analyses
                if chapter.section_id != section.id
                for candidate in chapter.glossary_candidates
            )
            work_repo = getattr(tx, "work", None)
            conditions = (
                work_repo.latest_conditions(command.book_id)
                if work_repo is not None
                else None
            )
            call = getattr(self, "_call", None)
            pronunciation_help_enabled = bool(
                (conditions and conditions.payload.get("pronunciation_help_enabled"))
                or getattr(call, "_pronunciation_enabled", False)
            )
            source_texts = {
                block.id: block.text
                for block in section.blocks
            }
        section_block_ids = {block.id for block in section.ordered_blocks}
        return _ChapterWork(
            document=document,
            section=section,
            analysis=analysis,
            run=run,
            book_candidates=book_candidates,
            processed_block_ids=frozenset(
                result.block_id
                for result in stored
                if result.block_id in section_block_ids
            ),
            used_note_ids=frozenset(
                note.id
                for result in stored
                for note in result.language_notes
                if result.block_id in section_block_ids
            ),
            task_id=command.task_id,
            source_texts=source_texts,
            pronunciation_help_enabled=pronunciation_help_enabled,
        )



class ChapterProcessingSequence:
    """Enqueues chapter processing tasks and completes book processing (PRD § 14, § 18)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        queue: TaskQueue,
        process_chapter: ProcessChapter,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._queue = queue
        self._process_chapter = process_chapter

    async def start(self, book_id: str) -> tuple[ProcessingTask, ...]:
        """Enqueues processing tasks for all chapters with text.

        Chapters are processed independently, so all tasks are enqueued at once:
        processing order does not affect results, and worker parallelism remains
        effective (PRD § 25). A chapter without textual blocks receives no task —
        there is nothing to process. Repeated starts return this run's existing
        dispatchers, preserving their operation references and identities (ALG-02).
        """
        section_ids = self._section_ids_with_text(book_id)
        tasks = []
        for section_id in section_ids:
            enqueue_identity = None
            with self._unit_of_work.transaction() as transaction:
                task = self._existing_dispatcher(book_id, section_id, transaction=transaction)
                if task is None:
                    enqueue_identity = TaskIdentity(
                        book_id=book_id,
                        task_type=TaskType.PROCESS_TRANSLATION_UNIT,
                        scope_id=section_id,
                        operation_id='dispatch:' + str(uuid4()),
                    )
                    task = ProcessingTask(
                        id=_new_task_id(self._queue), book_id=enqueue_identity.book_id,
                        task_type=enqueue_identity.task_type, scope_id=enqueue_identity.scope_id,
                        operation_id=enqueue_identity.operation_id,
                    )
                    transaction.tasks.add(task)
                    self._process_chapter.prepare(ProcessChapterCommand(
                        book_id, section_id, task.id,
                    ), new_dispatcher=True, transaction=transaction)
            if enqueue_identity is not None:
                task = await self._queue.enqueue(enqueue_identity)
            tasks.append(task)
        if not tasks:
            self._complete_processing(book_id, finished_task_id=None)
        return tuple(tasks)

    async def run_task(self, task: ProcessingTask) -> None:
        """Task handler for `PROCESS_TRANSLATION_UNIT` for the local runner (ALG-02)."""
        if task.task_type is not TaskType.PROCESS_TRANSLATION_UNIT:
            raise ProcessChapterError(
                f"задача {task.id} типа {task.task_type} не является обработкой главы"
            )
        command = ProcessChapterCommand(
            book_id=task.book_id,
            section_id=task.scope_id,
            task_id=task.id,
        )
        while True:
            try:
                await self._process_chapter(command)
                break
            except RecoveryError as error:
                if error.reason != 'interrupted':
                    raise
                with self._unit_of_work.transaction() as transaction:
                    current = transaction.tasks.get(task.id)
                    binding = transaction.work.get_binding('task', task.id)
                if current is None or current.status is TaskStatus.SUPERSEDED:
                    return
                refs = () if binding is None else binding['payload_json'].get('operation_refs', ())
                if error.root_id is None or any(ref['root_id'] == error.root_id for ref in refs):
                    raise
                # Restart removed this root from a still-active dispatcher. Re-read its
                # saved references so the remaining roots continue under the same task.
                if current.status is not TaskStatus.RUNNING:
                    raise
        self._complete_processing(task.book_id, finished_task_id=task.id, section_id=task.scope_id)

    def _existing_dispatcher(self, book_id: str, section_id: str, *, transaction=None) -> ProcessingTask | None:
        """Find a dispatcher for this run; its roots are resolved only by exact refs."""
        transaction_context = nullcontext(transaction) if transaction is not None else self._unit_of_work.transaction()
        with transaction_context as tx:
            run = tx.runs.active_for_book(book_id)
            conditions = tx.work.latest_conditions(book_id)
            if run is None or conditions is None:
                raise RecoveryError('conditions_unavailable')
            for task in reversed(tx.tasks.list_for_book(book_id)):
                if (task.task_type is not TaskType.PROCESS_TRANSLATION_UNIT
                        or task.scope_id != section_id or task.status is TaskStatus.SUPERSEDED):
                    continue
                binding = tx.work.get_binding('task', task.id)
                if binding is None or binding['conditions_id'] != conditions.id:
                    continue
                refs = binding['payload_json'].get('operation_refs')
                if refs is None:
                    return task
                roots = tuple(tx.work.get_root(ref['root_id']) for ref in refs)
                if all(root is not None and root.manifest.get('active_run_id') == run.id for root in roots):
                    return task
        return None

    def _section_ids_with_text(self, book_id: str) -> tuple[str, ...]:
        """Book chapters that contain textual blocks, in reading order."""
        with self._unit_of_work.transaction() as transaction:
            document = transaction.documents.get(book_id)
            if document is None:
                raise ProcessChapterError(f"книги {book_id} нет в хранилище")
            if transaction.runs.active_for_book(book_id) is None:
                raise ProcessChapterError(
                    f"у книги {book_id} нет активного запуска обработки: "
                    f"сначала завершите проверку глоссария"
                )
        return tuple(
            section.id
            for section in document.ordered_sections
            if textual_blocks(section)
        )

    def _complete_processing(
        self,
        book_id: str,
        *,
        finished_task_id: str | None,
        section_id: str | None = None,
    ) -> None:
        """Transitions the book to `READY_TO_EXPORT` when all chapters are processed (ALG-06)."""
        from linguary.application.correction import _scoped_reconcile_section

        with _scoped_reconcile_section(section_id):
            complete_processing(
                self._unit_of_work,
                book_id=book_id,
                finished_task_id=finished_task_id,
                task_types=(TaskType.PROCESS_TRANSLATION_UNIT, TaskType.CHAPTER_CORRECTION,
                            TaskType.REGENERATE_BLOCK),
            )


def _section_of(
    document: CanonicalDocument, command: ProcessChapterCommand
) -> DocumentSection:
    """Document section requested by the command."""
    for section in document.sections:
        if section.id == command.section_id:
            return section
    raise ProcessChapterError(
        f"главы {command.section_id} нет в книге {command.book_id}"
    )


def _new_id() -> str:
    """Identifier for a new block group."""
    return str(uuid4())


def _new_task_id(queue: TaskQueue) -> str:
    """Reuse the queue's task ID source for atomically created dispatcher rows."""
    factory = getattr(queue, "new_task_id", None)
    return factory() if callable(factory) else _new_id()


__all__ = [
    "ChapterProcessingSequence",
    "IdFactory",
    "ProcessChapter",
    "ProcessChapterCommand",
    "ProcessChapterError",
    "ProcessChapterResult",
    "Sleep",
]
