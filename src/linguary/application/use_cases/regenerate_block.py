"""Regeneration of single block processing (PRD § 13, § 19.3, § 21.5, ALG-04).

The user requests regeneration upon seeing an unsatisfactory result for a specific block:
an isolated issue is fixed here, whereas an undesirable decision affecting the whole book
is resolved by returning to glossary review (`reset_processing`).

The original translation unit is not recalculated. The model receives a separate request:
the block itself with its treatment mode from the saved plan, an excerpt of chapter analysis,
the run's glossary snapshot, and several adjacent paragraphs as scene context. The model sees
adjacent paragraphs but does not process them: they have no aliases, and output for them
is rejected as an alien block. Only the selected block's result is replaced — its translation,
vocabulary hints, and `Language Notes`; adjacent blocks remain untouched (PRD § 19.3).

The ordering and validations are identical to chapter processing (ALG-04): context, model call
with policy-driven retries, blocking validations, short transaction checking that the run
is still active. The shared logic lives in `application/block_processing.py`.

The queue task scope is the block rather than the chapter: regeneration of different blocks
runs independently, and a chapter-scoped key would collapse such requests into a single task.
Each explicit repeat creates an operation with a pinned policy; completed tasks remain in history.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from uuid import uuid4

from linguary.application import recovery
from linguary.application.block_processing import (
    BlockProcessingError,
    Sleep,
    TranslationCall,
    book_record,
    chapter_number,
    complete_processing,
    note_id_factory,
)
from linguary.application.context_builder import ContextBuilder, is_textual
from linguary.application.ports.llm import LLMClient
from linguary.application.ports.pronunciation import PronunciationSource, PronunciationVariant
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.application.recovery import Recovery, RecoveryError, RecoverySplit
from linguary.application.retry_policy import RetryPolicy
from linguary.application.task_runner import TaskQueue
from linguary.application.translation_units import DEFAULT_MAX_UNIT_SOURCE_CHARS
from linguary.application.validators import TranslationResultValidator
from linguary.domain import (
    BookEvent,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    DocumentBlock,
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
    apply_book_event,
)
from linguary.domain.recovery import RecoveryRoot

#: Tasks after which the book returns to READY_TO_EXPORT: chapter processing and block
#: regeneration both keep the book in PROCESSING (ALG-06).
_COMPLETING_TASK_TYPES = (TaskType.PROCESS_TRANSLATION_UNIT, TaskType.REGENERATE_BLOCK)

#: Task statuses where a repeated request does not create a new task: work is already in progress.
_ACTIVE_TASK_STATUSES = (TaskStatus.PENDING, TaskStatus.RUNNING)


class RegenerateBlockError(BlockProcessingError):
    """Block cannot be regenerated in the book's current status."""


class RegenerateBlockNotFoundError(RegenerateBlockError):
    """Book or selected block was not found."""


@dataclass(frozen=True, slots=True)
class RegenerateBlockCommand:
    """Regeneration command: which block of which book to regenerate."""

    book_id: str
    block_id: str
    task_id: str | None = None
    root_id: str | None = None


@dataclass(frozen=True, slots=True)
class RegenerateBlockResult:
    """Regeneration result along with call cost metrics.

    `saved` is `False` if the result was discarded: while the call was in flight,
    the user returned to glossary review and the run this work began with no longer exists.
    """

    block_id: str
    run_id: str
    saved: bool = False
    attempts: int = 0
    usage: TokenUsage | None = None


@dataclass(frozen=True, slots=True)
class _BlockWork:
    """Everything needed for block regeneration, loaded in a single transaction."""

    document: CanonicalDocument
    section: DocumentSection
    block: DocumentBlock
    analysis: ChapterAnalysis
    run: ProcessingRun
    book_candidates: tuple[GlossaryCandidate, ...]
    used_note_ids: frozenset[str]
    task_id: str | None = None

    @property
    def target_language(self) -> str:
        """Book target language."""
        return self.document.target_language

    @property
    def chapter_number(self) -> int:
        """Chapter number of the block in reading order (PRD § 19.3)."""
        return chapter_number(self.document, self.section)


class RegenerateBlock:
    """Regenerates the selected block and replaces only its result (PRD § 21.5)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        llm_client: LLMClient,
        retry_policy: RetryPolicy,
        validator: TranslationResultValidator,
        context_builder: ContextBuilder | None = None,
        pronunciation_source: PronunciationSource | None = None,
        pronunciation_help_enabled: bool = False,
        pronunciation_variant: PronunciationVariant | None = None,
        id_factory: Callable[[], str] | None = None,
        sleep: Sleep | None = None,
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
        self._context_builder = context_builder or ContextBuilder()
        self._id_factory = id_factory or _new_id
        self._validator = validator
        self._recovery = Recovery(unit_of_work)

    async def __call__(self, command: RegenerateBlockCommand) -> RegenerateBlockResult:
        """One ALG-04 cycle for a single block (PRD § 13, § 19.3)."""
        work = self._load(command)
        root = self._prepare_root(work, command)
        scopes = self._recovery.plan_unresolved(root.id)
        if not scopes:
            self._recovery.assert_complete(root.id)
            return RegenerateBlockResult(work.block.id, work.run.id, saved=True)
        (scope,) = scopes
        unit = TranslationUnit.of_blocks(
            id=scope.id,
            blocks=(work.block,),
            plan=work.analysis.treatment_plan,
        )
        context = self._context_builder.for_block_regeneration(
            document=work.document,
            section=work.section,
            unit=unit,
            analysis=work.analysis,
            glossary=work.run.glossary_snapshot,
            book_candidates=work.book_candidates,
            policy_version=root.policy_version,
        )
        context = replace(
            context,
            request=context.request.with_logging(
                book_id=work.document.id,
                stage=LLMCallStage.BLOCK_REGENERATION,
                task_id=work.task_id,
            ),
        )
        try:
            call = await self._call.request(
                context,
                unit=unit,
                target_language=work.target_language,
                note_ids=note_id_factory(
                    chapter_number=work.chapter_number, used=work.used_note_ids
                ),
                recovery=(root.id, scope.id),
                source_texts={
                    block.id: block.text
                    for section in work.document.sections
                    for block in section.blocks
                },
            )
        except RecoverySplit:
            self._recovery.split(root.id, scope.id)
            raise
        except RecoveryError:
            with self._unit_of_work.transaction() as transaction:
                run = transaction.runs.active_for_book(work.document.id)
            if run is None or run.id != work.run.id:
                return RegenerateBlockResult(work.block.id, work.run.id)
            raise
        saved = self._call.save(
            book_id=work.document.id, run_id=work.run.id, results=call.results
        )
        return RegenerateBlockResult(
            block_id=work.block.id,
            run_id=work.run.id,
            saved=saved,
            attempts=call.attempts,
            usage=call.usage,
        )

    def ensure_regeneratable(self, command: RegenerateBlockCommand) -> None:
        """Validates the request before enqueuing a background task."""
        self._load(command)

    def prepare_operation(self, command: RegenerateBlockCommand, *, transaction=None) -> RecoveryRoot:
        """Pin a single-block operation before scheduling its first physical call."""
        return self._prepare_root(self._load(command, transaction=transaction), command, transaction=transaction)

    def _prepare_root(
        self, work: _BlockWork, command: RegenerateBlockCommand, *, transaction=None
    ) -> RecoveryRoot:
        refs = None
        context = lambda: nullcontext(transaction) if transaction is not None else self._unit_of_work.transaction()
        if command.root_id is not None:
            with context() as tx:
                root = tx.work.get_root(command.root_id)
                if root is None:
                    raise RecoveryError('conditions_unavailable', command.root_id)
            refs = ({'root_id': root.id, 'policy_version': root.policy_version},)
        unit = TranslationUnit.of_blocks(
            id=self._id_factory(), blocks=(work.block,), plan=work.analysis.treatment_plan
        )
        (root_id,) = self._recovery.prepare_roots(
            work, (unit,), max_source_chars=DEFAULT_MAX_UNIT_SOURCE_CHARS,
            validator=self._validator, operation_refs=refs, operation_kind='regeneration',
            transaction=transaction
        )
        with context() as tx:
            return tx.work.get_root(root_id)

    def _load(self, command: RegenerateBlockCommand, *, transaction=None) -> _BlockWork:
        """Loads the book, block, chapter analysis, and active run in a single transaction."""
        context = nullcontext(transaction) if transaction is not None else self._unit_of_work.transaction()
        with context as transaction:
            document = transaction.documents.get(command.book_id)
            if document is None:
                raise RegenerateBlockNotFoundError(
                    f"книги {command.book_id} нет в хранилище"
                )
            block = _block_of(document, command)
            section = document.section(block.section_id)
            analysis = transaction.analyses.get_for_section(
                command.book_id, section.id
            )
            if analysis is None:
                raise RegenerateBlockError(
                    f"глава {section.id} не проанализирована: обрабатывать её блоки нельзя"
                )
            analysis.ensure_ready_for_processing()
            _ensure_planned_as_text(analysis, block)
            run = transaction.runs.active_for_book(command.book_id)
            if run is None:
                raise RegenerateBlockError(
                    f"у книги {command.book_id} нет активного запуска обработки"
                )
            if any(
                root.operation_kind == 'translation'
                and not transaction.work.root_coverage_complete(root.id)
                for root in transaction.work.list_active_roots(command.book_id, run_id=run.id)
            ):
                raise RegenerateBlockError('книга ещё обрабатывается: восстановление не завершено')
            list_for_section = getattr(transaction.results, "list_for_section", None)
            stored = (
                list_for_section(run.id, section.id)
                if callable(list_for_section)
                else transaction.results.list_for_run(run.id)
            )
            book_candidates = tuple(
                candidate
                for chapter in transaction.analyses.list_for_book(command.book_id)
                for candidate in chapter.glossary_candidates
            )
        section_block_ids = {item.id for item in section.ordered_blocks}
        return _BlockWork(
            document=document,
            section=section,
            block=block,
            analysis=analysis,
            run=run,
            book_candidates=book_candidates,
            used_note_ids=frozenset(
                note.id
                for result in stored
                for note in result.language_notes
                if result.block_id in section_block_ids
            ),
            task_id=command.task_id,
        )


class BlockRegenerationSequence:
    """Enqueues block regeneration tasks and returns the book to export readiness (PRD § 18, § 21.5)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        queue: TaskQueue,
        regenerate_block: RegenerateBlock,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._queue = queue
        self._regenerate_block = regenerate_block
    async def start(self, book_id: str, block_id: str) -> ProcessingTask:
        """Atomically persist a regeneration operation, then return its queued task."""
        command = RegenerateBlockCommand(book_id=book_id, block_id=block_id)
        with self._unit_of_work.transaction() as transaction:
            from linguary.application.correction import reconcile_corrections
            reconcile_corrections(transaction, book_id)
            book = book_record(transaction, book_id)
            status = _status_after_request(transaction, book)
            self._regenerate_block._load(command, transaction=transaction)
            run = transaction.runs.active_for_book(book_id)
            conditions = transaction.work.latest_conditions(book_id)
            for existing in transaction.tasks.list_for_book(book_id):
                if (existing.task_type is not TaskType.REGENERATE_BLOCK
                        or existing.scope_id != block_id or existing.status not in _ACTIVE_TASK_STATUSES
                        or existing.policy_version != recovery.CURRENT_TRANSLATION_POLICY):
                    continue
                root = transaction.work.get_root(existing.operation_id)
                binding = transaction.work.get_binding('task', existing.id)
                if (root is not None and root.book_id == book_id and root.superseded_by is None
                        and root.status != 'stopped' and root.operation_kind == 'regeneration'
                        and root.policy_version == existing.policy_version
                        and run is not None and root.manifest.get('active_run_id') == run.id
                        and tuple(root.manifest.get('ordered_blocks', ())) == (block_id,)
                        and conditions is not None and root.conditions_id == conditions.id
                        and binding is not None and binding['book_id'] == book_id
                        and binding['conditions_id'] == root.conditions_id
                        and binding['payload_json'].get('operation_refs') == [
                            {'root_id': root.id, 'policy_version': root.policy_version}
                        ]):
                    scopes = transaction.work.list_scopes(root.id)
                    if len(scopes) == 1 and scopes[0].ordered_blocks == (block_id,):
                        return existing
            root = self._regenerate_block.prepare_operation(command, transaction=transaction)
            identity = TaskIdentity(book_id=book_id, task_type=TaskType.REGENERATE_BLOCK,
                                    scope_id=block_id, operation_id=root.id,
                                    policy_version=root.policy_version)
            task = ProcessingTask(id=_new_task_id(self._queue), book_id=book_id, task_type=identity.task_type,
                                  scope_id=block_id, operation_id=root.id,
                                  policy_version=root.policy_version)
            transaction.tasks.add(task)
            transaction.work.bind_operation_refs(
                task.id, book_id=book_id, conditions_id=root.conditions_id,
                operation_refs=({'root_id': root.id, 'policy_version': root.policy_version},)
            )
            if status is not book.status:
                transaction.books.save(replace(book, status=status, export_validation_status=None))
        return await self._queue.enqueue(identity)

    async def run_task(self, task: ProcessingTask) -> None:
        """Task handler for `REGENERATE_BLOCK` for the local runner (ALG-02)."""
        if task.task_type is not TaskType.REGENERATE_BLOCK:
            raise RegenerateBlockError(
                f"задача {task.id} типа {task.task_type} не является повторной генерацией"
            )
        with self._unit_of_work.transaction() as transaction:
            root = transaction.work.get_root(task.operation_id) if task.operation_id else None
            if root is None or root.policy_version != task.policy_version:
                raise RecoveryError('conditions_unavailable', task.operation_id)
        await self._regenerate_block(
            RegenerateBlockCommand(
                book_id=task.book_id,
                block_id=task.scope_id,
                task_id=task.id,
                root_id=task.operation_id,
            )
        )
        complete_processing(
            self._unit_of_work,
            book_id=task.book_id,
            finished_task_id=task.id,
            task_types=_COMPLETING_TASK_TYPES,
        )


def _status_after_request(transaction: UnitOfWork, book: BookRecord) -> BookStatus:
    """Book status after a regeneration request (ALG-06).

    The book is already in `PROCESSING` only when another block regeneration is in flight:
    chapter processing is finished by this point, otherwise it would not have transitioned
    to `PROCESSING`. Such a request does not change the status — there is nothing to change
    it to. Regeneration in the middle of initial chapter processing is rejected: the block
    does not yet have a result worth replacing.
    """
    if book.status is not BookStatus.PROCESSING:
        return apply_book_event(book.status, BookEvent.BLOCK_REGENERATION_REQUESTED)
    unfinished = tuple(
        task
        for task in transaction.tasks.list_for_book(book.id)
        if task.task_type is TaskType.PROCESS_TRANSLATION_UNIT
        and task.status not in (TaskStatus.SUCCEEDED, TaskStatus.SUPERSEDED)
    )
    if unfinished:
        raise RegenerateBlockError(
            f"книга {book.id} ещё обрабатывается: повторная генерация блока доступна "
            f"после завершения обработки"
        )
    return book.status


def _block_of(
    document: CanonicalDocument, command: RegenerateBlockCommand
) -> DocumentBlock:
    """Book block requested by the command.

    A block that is not sent to the model — verbatim markup, illustration, empty block —
    has no processing result and nothing to regenerate (PRD § 10.1).
    """
    block = next(
        (item for item in document.blocks() if item.id == command.block_id), None
    )
    if block is None:
        raise RegenerateBlockNotFoundError(
            f"блока {command.block_id} нет в книге {command.book_id}"
        )
    if not is_textual(block):
        raise RegenerateBlockError(
            f"блок {block.id} не является текстом для чтения: "
            f"обрабатывать его модель не должна"
        )
    return block


def _ensure_planned_as_text(analysis: ChapterAnalysis, block: DocumentBlock) -> None:
    """Ensures the chapter treatment plan assigns this block to the model (PRD § 14)."""
    if block.id not in analysis.treatment_plan.textual_block_ids:
        raise RegenerateBlockError(
            f"блок {block.id} не входит в план обработки главы {analysis.section_id} "
            f"как текст для модели"
        )


def _new_id() -> str:
    """Identifier for a single-block unit: it lives for exactly one call."""
    return str(uuid4())


def _new_task_id(queue: TaskQueue) -> str:
    """Reuse the queue's task ID source for atomically created task rows."""
    factory = getattr(queue, "new_task_id", None)
    return factory() if callable(factory) else _new_id()


__all__ = [
    "BlockRegenerationSequence",
    "RegenerateBlock",
    "RegenerateBlockCommand",
    "RegenerateBlockError",
    "RegenerateBlockNotFoundError",
    "RegenerateBlockResult",
]
