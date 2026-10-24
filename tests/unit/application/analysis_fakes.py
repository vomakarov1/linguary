"""Storage and task queue fakes for chapter analysis unit tests (P3.T3).

The application use case interacts with UnitOfWork and TaskQueue ports, so unit tests
run purely in-memory: no SQLite, no EPUB, and no network. The module intentionally omits
the 'test_' prefix, otherwise pytest would collect it as a test file.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from linguary.adapters.llm import FakeReply
from linguary.application.analysis_merge import DEFAULT_MAX_FRAGMENT_BLOCKS
from linguary.application.ports.llm import LLMClient
from linguary.application.ports.persistence import BookRecord
from linguary.application.retry_policy import RetryPolicy
from linguary.application.use_cases.analyze_chapter import AnalyzeChapter
from linguary.domain import (
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    DocumentBlock,
    DocumentSection,
    GlossaryCandidate,
    ProcessingRun,
    ProcessingTask,
    SourceReference,
    SupportStatus,
    TaskIdentity,
    TaskStatus,
    TermConcept,
    TokenUsage,
    TreatmentPlan,
    active_run,
)

BOOK_ID = "book-1"
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "ru"

#: Confidence threshold below which a glossary candidate requires review (PRD § 12).
CONFIDENCE_THRESHOLD = 0.8


def analysis_reply(
    *,
    aliases: tuple[str, ...] = ("b1", "b2"),
    treatment: str = "TRANSLATE",
    chapter_summary: str = "Mr. Bennet visits the new neighbour.",
    candidates: list[dict[str, Any]] | None = None,
    usage: TokenUsage | None = None,
    **overrides: Any,
) -> FakeReply:
    """Valid chapter analysis response for the specified block aliases."""
    payload: dict[str, Any] = {
        "glossary_candidates": candidates
        if candidates is not None
        else [
            {
                "source_text": "Mr. Bennet",
                "type": "person",
                "suggested_translation": "мистер Беннет",
                "confidence": 0.9,
                "model_flagged_review": False,
                "note": None,
            }
        ],
        "translation_instructions": "Сохранять иронию рассказчика.",
        "treatment_plan": [
            {
                "source_block_id": alias,
                "treatment": treatment,
            }
            for alias in aliases
        ],
        "before_you_read": [
            {
                "expression": "entailed estate",
                "explanation": "имение с ограничением наследования",
            }
        ],
        "chapter_summary": chapter_summary,
    }
    payload.update(overrides)
    return FakeReply(content=json.dumps(payload, ensure_ascii=False), usage=usage)


def make_analyze_chapter(
    unit_of_work: FakeUnitOfWork,
    client: LLMClient,
    *,
    max_attempts: int = 3,
    max_fragment_blocks: int = DEFAULT_MAX_FRAGMENT_BLOCKS,
) -> tuple[AnalyzeChapter, list[float]]:
    """Chapter analysis use case alongside recorded retry delays."""
    delays: list[float] = []

    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    scenario = AnalyzeChapter(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]  # storage port fake
        llm_client=client,
        retry_policy=RetryPolicy(max_attempts=max_attempts),
        glossary_confidence_threshold=CONFIDENCE_THRESHOLD,
        max_fragment_blocks=max_fragment_blocks,
        id_factory=sequential_ids(),
        sleep=sleep,
    )
    return scenario, delays


def sequential_ids() -> Callable[[], str]:
    """Predictable analysis record identifiers: 'id-1', 'id-2', ..."""
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"id-{counter}"

    return next_id


def make_analysis_block(
    *,
    section_id: str,
    order: int,
    text: str,
    type: BlockType = BlockType.PARAGRAPH,
    support_status: SupportStatus = SupportStatus.SUPPORTED,
) -> DocumentBlock:
    """A chapter block with a stable identifier."""
    return DocumentBlock.create(
        document_id=BOOK_ID,
        section_id=section_id,
        type=type,
        order=order,
        text=text,
        source_reference=SourceReference(
            resource_path=f"{section_id}.xhtml",
            structural_path=f"/body/p[{order + 1}]",
        ),
        support_status=support_status,
    )


def make_analysis_section(
    *,
    section_id: str = "section-1",
    title: str = "Chapter 1",
    order: int = 0,
    texts: Sequence[str] = ("It is a truth universally acknowledged.",),
    blocks: Sequence[DocumentBlock] | None = None,
) -> DocumentSection:
    """A book chapter constructed from plain paragraphs or existing blocks."""
    return DocumentSection(
        id=section_id,
        document_id=BOOK_ID,
        title=title,
        order=order,
        source_resource_path=f"{section_id}.xhtml",
        blocks=tuple(blocks)
        if blocks is not None
        else tuple(
            make_analysis_block(section_id=section_id, order=index, text=text)
            for index, text in enumerate(texts)
        ),
    )


def make_analysis_document(
    sections: Sequence[DocumentSection] | None = None,
) -> CanonicalDocument:
    """A book containing the given chapters."""
    return CanonicalDocument(
        id=BOOK_ID,
        metadata=BookMetadata(title="Pride and Prejudice", authors=("Jane Austen",)),
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        sections=tuple(sections) if sections is not None else (make_analysis_section(),),
    )


def make_analysis_candidate(
    *,
    source_text: str = "Mr. Bennet",
    suggested_translation: str = "мистер Беннет",
    type: ConceptType = ConceptType.PERSON,
    analysis_id: str = "analysis-1",
    confidence: float = 0.9,
    model_flagged_review: bool = False,
    note: str = "",
) -> GlossaryCandidate:
    """A glossary candidate accumulated by a previous chapter."""
    return GlossaryCandidate.create(
        id=f"candidate-{analysis_id}-{source_text}",
        analysis_id=analysis_id,
        source_text=source_text,
        type=type,
        suggested_translation=suggested_translation,
        confidence=confidence,
        model_flagged_review=model_flagged_review,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        note=note,
    )


def make_chapter_analysis(
    *,
    id: str = "analysis-1",
    section_id: str = "section-1",
    candidates: Sequence[GlossaryCandidate] = (),
) -> ChapterAnalysis:
    """Chapter analysis where only glossary candidates matter."""
    return ChapterAnalysis.create(
        id=id,
        section_id=section_id,
        treatment_plan=TreatmentPlan(section_id=section_id),
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(),
        glossary_candidates=candidates,
    )


def make_book_record(status: BookStatus = BookStatus.IMPORTED) -> BookRecord:
    """Book record in the status from which analysis is initiated."""
    return BookRecord(
        id=BOOK_ID,
        metadata=BookMetadata(title="Pride and Prejudice", authors=("Jane Austen",)),
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        status=status,
        source_epub_path="pride.epub",
        created_at=datetime(2026, 8, 29, tzinfo=UTC),
    )


@dataclass(slots=True)
class FakeBookRepository:
    """In-memory books repository."""

    records: dict[str, BookRecord] = field(default_factory=dict)

    def add(self, book: BookRecord) -> None:
        self.records[book.id] = book

    def save(self, book: BookRecord) -> None:
        self.records[book.id] = book

    def get(self, book_id: str) -> BookRecord | None:
        return self.records.get(book_id)

    def list_all(self) -> tuple[BookRecord, ...]:
        return tuple(self.records.values())


@dataclass(slots=True)
class FakeDocumentRepository:
    """In-memory canonical document repository."""

    documents: dict[str, CanonicalDocument] = field(default_factory=dict)

    def save(
        self,
        document: CanonicalDocument,
        *,
        stored_resource_paths: Mapping[str, str] | None = None,
    ) -> None:
        self.documents[document.id] = document

    def get(self, book_id: str) -> CanonicalDocument | None:
        return self.documents.get(book_id)

    def stored_resource_paths(self, book_id: str) -> Mapping[str, str]:
        return {}

    def add_import_warnings(self, book_id: str, warnings: Sequence[Any]) -> None:
        return None

    def import_warnings(self, book_id: str) -> tuple[Any, ...]:
        return ()


@dataclass(slots=True)
class FakeChapterAnalysisRepository:
    """In-memory chapter analyses ordered by book reading sequence.

    The order mirrors ChapterAnalysisRepository.list_for_book contract: analyses
    are returned in document chapter order rather than save timestamp order.
    """

    documents: FakeDocumentRepository
    analyses: dict[tuple[str, str], ChapterAnalysis] = field(default_factory=dict)
    saved_section_ids: list[str] = field(default_factory=list)

    def save(self, book_id: str, analysis: ChapterAnalysis) -> None:
        self.analyses[(book_id, analysis.section_id)] = analysis
        self.saved_section_ids.append(analysis.section_id)

    def get_for_section(self, book_id: str, section_id: str) -> ChapterAnalysis | None:
        return self.analyses.get((book_id, section_id))

    def list_for_book(self, book_id: str) -> tuple[ChapterAnalysis, ...]:
        document = self.documents.get(book_id)
        order = (
            [section.id for section in document.ordered_sections] if document else []
        )
        stored = {
            section_id: analysis
            for (stored_book_id, section_id), analysis in self.analyses.items()
            if stored_book_id == book_id
        }
        return tuple(
            stored[section_id] for section_id in order if section_id in stored
        )


@dataclass(slots=True)
class FakeGlossaryRepository:
    """In-memory book glossary concepts repository."""

    concepts: dict[str, TermConcept] = field(default_factory=dict)

    def save(self, concept: TermConcept) -> None:
        self.concepts[concept.id] = concept

    def get(self, concept_id: str) -> TermConcept | None:
        return self.concepts.get(concept_id)

    def list_for_book(self, book_id: str) -> tuple[TermConcept, ...]:
        return tuple(
            concept for concept in self.concepts.values() if concept.book_id == book_id
        )

    def delete(self, concept_id: str) -> None:
        self.concepts.pop(concept_id, None)


@dataclass(slots=True)
class FakeProcessingRunRepository:
    """In-memory book processing runs repository.

    Active run uniqueness is validated by the same domain active_run function as
    in the SQLAlchemy repository: the fake must not be more lenient than real persistence.
    """

    runs: dict[str, ProcessingRun] = field(default_factory=dict)

    def add(self, run: ProcessingRun) -> None:
        self.runs[run.id] = run

    def save(self, run: ProcessingRun) -> None:
        self.runs[run.id] = run

    def get(self, run_id: str) -> ProcessingRun | None:
        return self.runs.get(run_id)

    def active_for_book(self, book_id: str) -> ProcessingRun | None:
        return active_run(
            run for run in self.runs.values() if run.book_id == book_id
        )

    def delete(self, run_id: str) -> None:
        self.runs.pop(run_id, None)


class FakeUnitOfWork:
    """In-memory transactional boundary: commit and rollback are no-ops."""

    def __init__(
        self,
        *,
        document: CanonicalDocument | None = None,
        book: BookRecord | None = None,
    ) -> None:
        self._books = FakeBookRepository()
        self._documents = FakeDocumentRepository()
        self._analyses = FakeChapterAnalysisRepository(documents=self._documents)
        self._glossary = FakeGlossaryRepository()
        self._runs = FakeProcessingRunRepository()
        if document is not None:
            self._documents.save(document)
        if book is not None:
            self._books.add(book)

    @property
    def books(self) -> FakeBookRepository:
        return self._books

    @property
    def documents(self) -> FakeDocumentRepository:
        return self._documents

    @property
    def analyses(self) -> FakeChapterAnalysisRepository:
        return self._analyses

    @property
    def glossary(self) -> FakeGlossaryRepository:
        return self._glossary

    @property
    def runs(self) -> FakeProcessingRunRepository:
        return self._runs

    @contextmanager
    def transaction(self) -> Iterator[FakeUnitOfWork]:
        yield self


class FakeTaskQueue:
    """In-memory task queue tracking concurrently running tasks.

    Claiming a task mirrors the TaskQueue contract: 'pending -> running' with an incremented
    attempt counter. The queue records claim order and maximum concurrent running tasks.
    """

    def __init__(self) -> None:
        self.tasks: list[ProcessingTask] = []
        self.claimed_scope_ids: list[str] = []
        self.max_running = 0
        self._next_id = 0

    async def enqueue(self, identity: TaskIdentity) -> ProcessingTask:
        existing = self._find(identity)
        if existing is not None:
            return existing
        self._next_id += 1
        task = ProcessingTask(
            id=f"task-{self._next_id}",
            book_id=identity.book_id,
            task_type=identity.task_type,
            scope_id=identity.scope_id,
        )
        self.tasks.append(task)
        return task

    async def claim_next_pending(self) -> ProcessingTask | None:
        for index, task in enumerate(self.tasks):
            if task.status is TaskStatus.PENDING:
                claimed = task.start()
                self.tasks[index] = claimed
                self.claimed_scope_ids.append(claimed.scope_id)
                self.max_running = max(self.max_running, self._running_count())
                return claimed
        return None

    async def save(self, task: ProcessingTask) -> None:
        for index, stored in enumerate(self.tasks):
            if stored.id == task.id:
                self.tasks[index] = task
                return
        self.tasks.append(task)

    async def list_running(self) -> tuple[ProcessingTask, ...]:
        return tuple(
            task for task in self.tasks if task.status is TaskStatus.RUNNING
        )

    @property
    def has_work(self) -> bool:
        """Whether pending or running work remains in the queue."""
        return any(
            task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
            for task in self.tasks
        )

    def _running_count(self) -> int:
        return sum(1 for task in self.tasks if task.status is TaskStatus.RUNNING)

    def _find(self, identity: TaskIdentity) -> ProcessingTask | None:
        return next(
            (task for task in self.tasks if task.identity == identity), None
        )


def make_glossary_book(
    analyses: Sequence[ChapterAnalysis] = (),
    *,
    status: BookStatus = BookStatus.GLOSSARY_REVIEW,
) -> FakeUnitOfWork:
    """Book with stored chapter analyses advanced to glossary review.

    Each analysis is assigned to its own document section: analysis repository returns
    them in reading order, and without a document it would return an empty tuple.
    """
    document = make_analysis_document(
        tuple(
            make_analysis_section(section_id=analysis.section_id, order=order)
            for order, analysis in enumerate(analyses)
        )
    )
    unit_of_work = FakeUnitOfWork(document=document, book=make_book_record(status))
    for analysis in analyses:
        unit_of_work.analyses.save(BOOK_ID, analysis)
    return unit_of_work
