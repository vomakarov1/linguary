"""Persistence ports: repositories and unit of work.

The application layer interacts with storage only through these protocols and knows
nothing about SQLAlchemy or SQLite: concrete implementations live in `linguary.adapters.db`.

Repositories return and accept domain entities. Where there is no domain entity
(such as a book record assembled from metadata, languages, status, and file paths),
the contract is defined by a standard library `dataclass`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from linguary.application.errors import ApplicationError
from linguary.domain import (
    BlockProcessingResult,
    BookImportWarning,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    LLMCallLog,
    ProcessingRun,
    ProcessingTask,
    TaskIdentity,
    TermConcept,
    TaskError,
)


class PersistenceError(ApplicationError):
    """Storage access in an invalid state.

    The adapter uses this error class to report violations of port contracts — for example,
    operating on a repository outside an active transaction. Domain invariant violations
    remain subclasses of `DomainError`.
    """


class ExportValidationStatus(StrEnum):
    """Result of validating exported EPUB with external EPUBCheck (PRD § 20).

    Before the first export, a book has no value. `NOT_VALIDATED` means EPUBCheck
    is unavailable on the user's machine and the book was exported without external validation.
    """

    NOT_VALIDATED = "not_validated"
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BookRecord:
    """Book record: what storage tracks about a book beyond the canonical document.

    The canonical document model (sections, blocks, resources) is stored separately
    in `DocumentRepository`; here lies book status and its relationship with the file system.
    """

    id: str
    metadata: BookMetadata
    source_language: str
    target_language: str
    status: BookStatus
    source_epub_path: str
    created_at: datetime
    export_validation_status: ExportValidationStatus | None = None


class BookRepository(Protocol):
    """Books and their user-facing state."""

    def add(self, book: BookRecord) -> None:
        """Adds a new book."""

    def save(self, book: BookRecord) -> None:
        """Overwrites an existing book entirely."""

    def get(self, book_id: str) -> BookRecord | None:
        """Returns book by ID or `None` if not found."""

    def list_all(self) -> tuple[BookRecord, ...]:
        """All books ordered by creation: Web UI book list."""


class DocumentRepository(Protocol):
    """Canonical document model and import warnings."""

    def save(
        self,
        document: CanonicalDocument,
        *,
        stored_resource_paths: Mapping[str, str] | None = None,
    ) -> None:
        """Saves document entirely, replacing previous sections, blocks, and resources.

        `stored_resource_paths` maps original resource paths to the file system path
        where the importer stored them; domain is unaware of this path.
        """

    def get(self, book_id: str) -> CanonicalDocument | None:
        """Canonical model of the book or `None` if no such book exists.

        Before import completes, the document exists but may have no sections.
        """

    def stored_resource_paths(self, book_id: str) -> Mapping[str, str]:
        """Paths to stored book resources by their original paths."""

    def add_import_warnings(
        self, book_id: str, warnings: Sequence[BookImportWarning]
    ) -> None:
        """Adds import warnings for the book."""

    def import_warnings(self, book_id: str) -> tuple[BookImportWarning, ...]:
        """Import warnings for the book."""


class GlossaryRepository(Protocol):
    """Book glossary concepts along with their language variants."""

    def save(self, concept: TermConcept) -> None:
        """Saves a concept with its variants, replacing the previous entry."""

    def get(self, concept_id: str) -> TermConcept | None:
        """Concept by ID or `None`."""

    def list_for_book(self, book_id: str) -> tuple[TermConcept, ...]:
        """All glossary concepts for the book."""

    def delete(self, concept_id: str) -> None:
        """Deletes concept along with its variants."""


class ChapterAnalysisRepository(Protocol):
    """Chapter analysis: treatment plan, glossary candidates, and chapter learning materials."""

    def save(self, book_id: str, analysis: ChapterAnalysis) -> None:
        """Saves chapter analysis, replacing previous analysis for that section."""

    def get_for_section(self, book_id: str, section_id: str) -> ChapterAnalysis | None:
        """Analysis for a specific section or `None` if not yet analyzed."""

    def list_for_book(self, book_id: str) -> tuple[ChapterAnalysis, ...]:
        """Analyses for all analyzed chapters of the book."""

    def delete_for_book(self, book_id: str) -> None:
        """Remove current analysis projections, preserving archives."""


class ProcessingRunRepository(Protocol):
    """Processing runs for books."""

    def add(self, run: ProcessingRun) -> None:
        """Adds a new processing run."""

    def save(self, run: ProcessingRun) -> None:
        """Overwrites existing run: state transitions and snapshots."""

    def get(self, run_id: str) -> ProcessingRun | None:
        """Run by ID or `None`."""

    def active_for_book(self, book_id: str) -> ProcessingRun | None:
        """The single active run for the book or `None`."""

    def delete(self, run_id: str) -> None:
        """Deletes run along with block results, notes, and warnings."""


class BlockResultRepository(Protocol):
    """Block processing results along with notes and warnings."""

    def replace_results(
        self, run_id: str, results: Sequence[BlockProcessingResult]
    ) -> None:
        """Replaces results for listed blocks in a single operation (PRD § 17).

        Previous results, notes, and warnings for these blocks are removed, and new ones
        are written entirely: partially saved `TranslationUnit` results must not occur.
        """

    def get(self, run_id: str, block_id: str) -> BlockProcessingResult | None:
        """Processing result for a single block within a run."""

    def list_for_run(self, run_id: str) -> tuple[BlockProcessingResult, ...]:
        """All results for the run in original document block order."""

    def count_for_run(self, run_id: str) -> int:
        """Number of run blocks already processed.

        A separate method is used for processing progress: book card is updated via polling,
        so retrieving all block results just for a count is unnecessary.
        """


class TaskRepository(Protocol):
    """Local resumable processing tasks."""

    def add(self, task: ProcessingTask) -> None:
        """Adds task; duplicate idempotency key is rejected by storage."""

    def save(self, task: ProcessingTask) -> None:
        """Saves updated task state after a transition."""

    def get(self, task_id: str) -> ProcessingTask | None:
        """Task by ID or `None`."""

    def find_by_identity(self, identity: TaskIdentity) -> ProcessingTask | None:
        """Task by idempotency key `book_id + task_type + scope_id`."""

    def list_for_book(self, book_id: str) -> tuple[ProcessingTask, ...]:
        """Tasks for the book in order of creation."""

    def delete(self, task_id: str) -> None:
        """Deletes task along with its idempotency key.

        The key `book_id + task_type + scope_id` lives as long as the task lives, and
        resubmitting returns the existing task in its current state. Therefore work that
        must be redone begins with deleting the previous task: returning to glossary review
        removes cancelled run tasks, and block regeneration removes its prior task
        (PRD § 18.1, § 21.5).
        """


class LLMCallLogRepository(Protocol):
    """Technical LLM call log (PRD § 24).

    Response text is stored separately only when debugging is enabled.
    """

    def add(self, entry: LLMCallLog, *, response_text: str | None = None) -> None:
        """Adds a log entry."""

    def mark_rejected(
        self, call_id: str, *, error: TaskError, response_text: str | None
    ) -> None:
        """Marks a response rejected by validation after a successful call."""

    def response_text_for(self, call_id: str) -> str | None:
        """Stored response text for diagnostic purposes or `None`."""

    def list_for_book(self, book_id: str) -> tuple[LLMCallLog, ...]:
        """Log entries for the book ordered by call execution."""

    def diagnostic_record(self, *, call_id=None, attempt_id=None):
        """Internal diagnostic record; body is never part of public book projections."""

    def update_diagnostics(self, call_id, *, metadata) -> None: ...
    def store_diagnostic_body(self, call_id, body) -> None: ...
    def purge_diagnostic_bodies(self, book_id, *, call_id=None) -> int: ...
    def enforce_diagnostics_policy(self, policy, now, *, book_id=None) -> None: ...


class WorkRepository(Protocol):
    """Versioned provenance and atomic work operations, without network calls."""

    def add_conditions(self, conditions) -> None: ...
    def get_conditions(self, conditions_id): ...
    def latest_conditions(self, book_id): ...
    def bind(self, entity_kind, entity_id, *, book_id, conditions_id,
             attempt_id=None, artifact_revision_id=None, payload=None) -> None: ...
    def get_binding(self, entity_kind, entity_id): ...
    def add_root(self, root) -> None: ...
    def get_root(self, root_id): ...
    def add_scope(self, scope) -> None: ...
    def get_scope(self, scope_id): ...
    def claim_scope(self, scope_id, *, expected_generation): ...
    def accept_scope(self, scope_id, *, attempt_id, expected_generation, results) -> None: ...
    def add_attempt(self, attempt) -> None: ...
    def get_attempt(self, attempt_id): ...
    def transition_attempt(self, attempt_id, state, *, expected_state, outcome=None): ...
    def archive_artifacts(self, artifacts): ...
    def get_artifact(self, artifact_id): ...
    def replace_result_binding(self, entity_id, *, book_id, conditions_id, attempt_id,
                               artifact_revision_id, expected_attempt_id) -> None: ...
    def add_review(self, review) -> None: ...
    def get_review(self, review_id): ...
    def reserve(self, reservation, *, expected_revision): ...
    def get_reservation(self, attempt_id): ...
    def settle(self, attempt_id, *, actual): ...
    def purge_diagnostics(self, book_id, *, call_id=None, before=None) -> int: ...


class UnitOfWork(Protocol):
    """Transactional boundary of an application use case.

    Repositories are accessible only within `transaction()`: the use case explicitly
    defines the boundary, and the adapter guarantees all changes within it commit
    or roll back as a unit. Transactions are short and do not span HTTP calls to LLM (ADR § 3).
    """

    @property
    def books(self) -> BookRepository:
        """Book repository."""

    @property
    def documents(self) -> DocumentRepository:
        """Canonical document repository."""

    @property
    def glossary(self) -> GlossaryRepository:
        """Glossary repository."""

    @property
    def analyses(self) -> ChapterAnalysisRepository:
        """Chapter analysis repository."""

    @property
    def runs(self) -> ProcessingRunRepository:
        """Processing run repository."""

    @property
    def results(self) -> BlockResultRepository:
        """Block result repository."""

    @property
    def tasks(self) -> TaskRepository:
        """Local task repository."""

    @property
    def llm_calls(self) -> LLMCallLogRepository:
        """LLM call log repository."""

    @property
    def work(self) -> WorkRepository:
        """Immutable work history and transactional execution state."""

    def transaction(self) -> AbstractContextManager[UnitOfWork]:
        """Opens a transaction and makes repositories accessible.

        Exiting the context manager without exception commits changes; an exception rolls back.
        """


__all__ = [
    "BlockResultRepository",
    "BookRecord",
    "BookRepository",
    "ChapterAnalysisRepository",
    "DocumentRepository",
    "ExportValidationStatus",
    "GlossaryRepository",
    "LLMCallLogRepository",
    "PersistenceError",
    "ProcessingRunRepository",
    "TaskRepository",
    "UnitOfWork",
    "WorkRepository",
]
