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

The queue task is the entire chapter rather than a single group: `TranslationUnit` is not
persisted, its ID lives for one call, and an idempotent task key cannot be built from it.
Retrying a chapter task is safe: already stored groups are skipped, and storing the remaining
groups replaces previous results of the same blocks (PRD § 18.1).

`ChapterProcessingSequence` enqueues chapter tasks and transitions the book to `READY_TO_EXPORT`
when all chapters have been processed. The processing run remains active: full export looks up
results via `ProcessingRunRepository.active_for_book`, and a closed run would leave the book
without translations. The run is closed only when returning to glossary review, which deletes it
along with its results (P3.T7).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
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
from linguary.application.ports.llm import LLMClient
from linguary.application.ports.persistence import UnitOfWork
from linguary.application.retry_policy import RetryPolicy
from linguary.application.task_runner import TaskQueue
from linguary.application.translation_units import (
    DEFAULT_MAX_UNIT_SOURCE_CHARS,
    plan_translation_units,
)
from linguary.application.validators import TranslationResultValidator
from linguary.domain import (
    CanonicalDocument,
    ChapterAnalysis,
    DocumentSection,
    GlossaryCandidate,
    LLMCallStage,
    ProcessingRun,
    ProcessingTask,
    TaskIdentity,
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
        context_builder: ContextBuilder | None = None,
        id_factory: IdFactory | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._call = TranslationCall(
            unit_of_work=unit_of_work,
            llm_client=llm_client,
            retry_policy=retry_policy,
            validator=validator,
            sleep=sleep or asyncio.sleep,
        )
        self._max_unit_source_chars = max_unit_source_chars
        self._context_builder = context_builder or ContextBuilder()
        self._id_factory = id_factory or _new_id

    async def __call__(self, command: ProcessChapterCommand) -> ProcessChapterResult:
        """Processes all translation units of the chapter (PRD § 10, § 14).

        A chapter without textual blocks — cover, image-only page — is handled
        without model calls: it has no units, nothing to save.
        """
        work = self._load(command)
        units = plan_translation_units(
            work.section.ordered_blocks,
            work.analysis.treatment_plan,
            id_factory=self._id_factory,
            max_source_chars=self._max_unit_source_chars,
        )
        note_ids = note_id_factory(
            chapter_number=work.chapter_number, used=work.used_note_ids
        )
        result = ProcessChapterResult(
            section_id=work.section.id, run_id=work.run.id, units=len(units)
        )
        for unit in units:
            if self._is_processed(unit, work=work):
                continue
            result = await self._process_unit(unit, work=work, note_ids=note_ids, result=result)
            if result.discarded:
                break
        return result

    async def _process_unit(
        self,
        unit: TranslationUnit,
        *,
        work: _ChapterWork,
        note_ids: Callable[[], str],
        result: ProcessChapterResult,
    ) -> ProcessChapterResult:
        """One ALG-04 cycle: context assembly, model call, validation, atomic save."""
        context = self._context_builder.for_translation_unit(
            document=work.document,
            section=work.section,
            unit=unit,
            analysis=work.analysis,
            glossary=work.run.glossary_snapshot,
            book_candidates=work.book_candidates,
        )
        context = replace(
            context,
            request=context.request.with_logging(
                book_id=work.document.id,
                stage=LLMCallStage.TRANSLATION_UNIT,
                task_id=work.task_id,
            ),
        )
        call = await self._call.request(
            context, unit=unit, target_language=work.target_language, note_ids=note_ids
        )
        saved = self._call.save(
            book_id=work.document.id, run_id=work.run.id, results=call.results
        )
        return replace(
            result,
            processed_units=result.processed_units + 1,
            saved_blocks=result.saved_blocks + (len(call.results) if saved else 0),
            attempts=result.attempts + call.attempts,
            usage=total_usage((result.usage, call.usage)),
            discarded=not saved,
        )

    def _load(self, command: ProcessChapterCommand) -> _ChapterWork:
        """Loads the book, chapter analysis, and active run in a single transaction."""
        with self._unit_of_work.transaction() as transaction:
            document = transaction.documents.get(command.book_id)
            if document is None:
                raise ProcessChapterError(f"книги {command.book_id} нет в хранилище")
            section = _section_of(document, command)
            analysis = transaction.analyses.get_for_section(
                command.book_id, command.section_id
            )
            if analysis is None:
                raise ProcessChapterError(
                    f"глава {command.section_id} не проанализирована: "
                    f"обрабатывать её нельзя"
                )
            analysis.ensure_ready_for_processing()
            run = transaction.runs.active_for_book(command.book_id)
            if run is None:
                raise ProcessChapterError(
                    f"у книги {command.book_id} нет активного запуска обработки"
                )
            stored = transaction.results.list_for_run(run.id)
            book_candidates = tuple(
                candidate
                for chapter in transaction.analyses.list_for_book(command.book_id)
                for candidate in chapter.glossary_candidates
            )
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
        )

    def _is_processed(self, unit: TranslationUnit, *, work: _ChapterWork) -> bool:
        """Whether the result of this unit has been saved in full (PRD § 18.1).

        A task retry does not pay for already completed work. A group is considered
        completed only as a whole: partial saved groups do not exist, and matching
        only some blocks would imply a different chapter splitting — requiring reprocessing.
        """
        return set(unit.block_ids) <= work.processed_block_ids


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
        there is nothing to process. Re-enqueuing does not duplicate tasks: enqueuing
        is idempotent by `book_id + task_type + scope_id` (ALG-02).
        """
        section_ids = self._section_ids_with_text(book_id)
        tasks = [
            await self._queue.enqueue(
                TaskIdentity(
                    book_id=book_id,
                    task_type=TaskType.PROCESS_TRANSLATION_UNIT,
                    scope_id=section_id,
                )
            )
            for section_id in section_ids
        ]
        if not tasks:
            self._complete_processing(book_id, finished_task_id=None)
        return tuple(tasks)

    async def run_task(self, task: ProcessingTask) -> None:
        """Task handler for `PROCESS_TRANSLATION_UNIT` for the local runner (ALG-02)."""
        if task.task_type is not TaskType.PROCESS_TRANSLATION_UNIT:
            raise ProcessChapterError(
                f"задача {task.id} типа {task.task_type} не является обработкой главы"
            )
        await self._process_chapter(
            ProcessChapterCommand(
                book_id=task.book_id,
                section_id=task.scope_id,
                task_id=task.id,
            )
        )
        self._complete_processing(task.book_id, finished_task_id=task.id)

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

    def _complete_processing(self, book_id: str, *, finished_task_id: str | None) -> None:
        """Transitions the book to `READY_TO_EXPORT` when all chapters are processed (ALG-06)."""
        complete_processing(
            self._unit_of_work,
            book_id=book_id,
            finished_task_id=finished_task_id,
            task_types=(TaskType.PROCESS_TRANSLATION_UNIT,),
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


__all__ = [
    "ChapterProcessingSequence",
    "IdFactory",
    "ProcessChapter",
    "ProcessChapterCommand",
    "ProcessChapterError",
    "ProcessChapterResult",
    "Sleep",
]
