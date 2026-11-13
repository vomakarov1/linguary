"""Result chapters carry the same reading layout export projects.

The loader opens its own transaction. A call inside `book_result`'s read
raises the storage error for a transaction that is already open. No document,
and a header that skips chapters, leave the layout unset.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime
import json
from types import SimpleNamespace

from linguary.application.book_overview import book_result
from linguary.application.context_builder import textual_blocks
from linguary.application.correction import CorrectionInputs
from linguary.application.correction_dto import (
    CORRECTION_VERSIONS_V2,
    CorrectionSnapshotV2,
    parse_correction_snapshot,
)
from linguary.application.layout.book import (
    ChapterLayout,
    build_book_layout,
    build_layout_inputs,
)
from linguary.application.ports.persistence import PersistenceError
from linguary.domain import (
    BlockProcessingResult,
    BlockSegment,
    BlockTreatment,
    BlockType,
    BookMetadata,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    GlossaryHint,
    GlossarySnapshot,
    LLMSettingsSnapshot,
    ProcessingRun,
    ProcessingRunStatus,
    SourceReference,
)
from linguary.domain.content_plan import canonical_sha
from linguary.domain.correction import CorrectionCandidate

BOOK_ID = "book-1"
RUN_ID = "run-1"
CONDITIONS_ID = "conditions-1"


class _Gate:
    """One open transaction. A second open while this one is active fails."""

    def __init__(self, stored: _StoredBook) -> None:
        self._stored = stored

    def __enter__(self) -> _StoredBook:
        if self._stored.depth:
            raise PersistenceError(
                "транзакция уже открыта: вложенные транзакции не поддерживаются"
            )
        self._stored.depth += 1
        self._stored.opens += 1
        return self._stored

    def __exit__(self, *args: object) -> bool:
        self._stored.depth -= 1
        return False


class _StoredBook:
    """Repositories `book_result` and the layout loader both read."""

    def __init__(
        self,
        *,
        document: CanonicalDocument | None,
        results: tuple[BlockProcessingResult, ...] = (),
        run: ProcessingRun | None = None,
        conditions: object = None,
        run_binding: dict | None = None,
        result_bindings: dict | None = None,
        generations: dict | None = None,
        candidates: tuple[CorrectionCandidate, ...] = (),
    ) -> None:
        self.document = document
        self.result_rows = results
        self.run = run
        self.conditions = conditions
        self.run_binding = run_binding
        self._result_bindings = {} if result_bindings is None else result_bindings
        self._generations = {} if generations is None else generations
        self._candidates = candidates
        self.depth = 0
        self.opens = 0
        self.book = SimpleNamespace(id=BOOK_ID)
        self.books = self._Books(self)
        self.documents = self._Documents(self)
        self.runs = self._Runs(self)
        self.results = self._Results(self)
        self.analyses = self._Analyses()
        self.tasks = self._Tasks()
        self.glossary = self._Glossary()
        self.work = self._Work(self)

    def read_transaction(self) -> _Gate:
        return _Gate(self)

    def transaction(self) -> _Gate:
        return _Gate(self)

    class _Books:
        def __init__(self, stored: _StoredBook) -> None:
            self._stored = stored

        def get(self, book_id: str) -> object:
            return self._stored.book if book_id == BOOK_ID else None

    class _Documents:
        def __init__(self, stored: _StoredBook) -> None:
            self._stored = stored

        def get(self, book_id: str) -> CanonicalDocument | None:
            return self._stored.document if book_id == BOOK_ID else None

    class _Runs:
        def __init__(self, stored: _StoredBook) -> None:
            self._stored = stored

        def active_for_book(self, book_id: str) -> ProcessingRun | None:
            return self._stored.run if book_id == BOOK_ID else None

    class _Results:
        def __init__(self, stored: _StoredBook) -> None:
            self._stored = stored

        def list_for_run(self, run_id: str) -> tuple[BlockProcessingResult, ...]:
            if self._stored.run is None or run_id != self._stored.run.id:
                return ()
            return self._stored.result_rows

    class _Analyses:
        def list_for_book(self, book_id: str) -> tuple:
            return ()

    class _Tasks:
        def list_for_book(self, book_id: str) -> tuple:
            return ()

    class _Glossary:
        def list_for_book(self, book_id: str) -> tuple:
            return ()

    class _Work:
        def __init__(self, stored: _StoredBook) -> None:
            self._stored = stored

        def get_binding(self, kind: str, entity_id: str) -> dict | None:
            if kind == "run" and self._stored.run is not None and entity_id == self._stored.run.id:
                return self._stored.run_binding
            return None

        def result_bindings(self, run_id: str, block_ids: tuple[str, ...] | None = None) -> dict:
            if self._stored.run is None or run_id != self._stored.run.id:
                return {}
            if block_ids is None:
                return dict(self._stored._result_bindings)
            return {
                block_id: self._stored._result_bindings[block_id]
                for block_id in block_ids
                if block_id in self._stored._result_bindings
            }

        def correction_input_generations(self, run_id: str) -> dict:
            if self._stored.run is None or run_id != self._stored.run.id:
                return {}
            return dict(self._stored._generations)

        def list_correction_candidates(self, run_id, *, section_id=None, states=None):
            return tuple(
                candidate
                for candidate in self._stored._candidates
                if candidate.run_id == run_id
                and (section_id is None or candidate.section_id == section_id)
                and (states is None or candidate.state in states)
            )

        def latest_conditions(self, book_id: str) -> object:
            return self._stored.conditions if book_id == BOOK_ID else None

        def get_conditions(self, conditions_id: str) -> object:
            conditions = self._stored.conditions
            if conditions is None or getattr(conditions, "id", None) != conditions_id:
                return None
            return conditions


def _run() -> ProcessingRun:
    return ProcessingRun(
        RUN_ID,
        BOOK_ID,
        ProcessingRunStatus.ACTIVE,
        GlossarySnapshot(),
        LLMSettingsSnapshot("local", "test-model"),
    )


def _document_with(text: str, *, segments: tuple[BlockSegment, ...] = ()) -> tuple[
    CanonicalDocument, DocumentSection, DocumentBlock
]:
    block = DocumentBlock.create(
        document_id=BOOK_ID,
        section_id="chapter",
        type=BlockType.PARAGRAPH,
        order=0,
        text=text,
        source_reference=SourceReference("chapter.xhtml", "/body/p[1]"),
        segments=segments,
    )
    section = DocumentSection(
        id="chapter",
        document_id=BOOK_ID,
        title="Chapter 1",
        order=0,
        source_resource_path="chapter.xhtml",
        blocks=(block,),
    )
    document = CanonicalDocument(
        id=BOOK_ID,
        metadata=BookMetadata(title="Wonderland", authors=("Carroll",)),
        source_language="en",
        target_language="ru",
        sections=(section,),
    )
    return document, section, block


def _translated_book() -> tuple[_StoredBook, DocumentSection, DocumentBlock]:
    document, section, block = _document_with("A clear day.")
    result = BlockProcessingResult(block.id, BlockTreatment.TRANSLATE, "Ясный день.")
    return (
        _StoredBook(document=document, results=(result,), run=_run()),
        section,
        block,
    )


def test_book_result_chapter_has_layout() -> None:
    """A stored chapter carries its reading layout. A missing document does not.

    Skipping chapters does not call the loader. The loader's transaction is
    refused while the result read is still open.
    """
    missing = _StoredBook(document=None, run=_run())
    without_document = book_result(missing, BOOK_ID, include_quality=False)
    assert without_document.chapters == ()
    assert missing.opens == 1

    stored, _, _ = _translated_book()
    header = book_result(stored, BOOK_ID, include_quality=False, include_chapters=False)
    assert header.chapters == ()
    assert stored.opens == 1

    stored, section, block = _translated_book()
    summary = book_result(stored, BOOK_ID, include_quality=False)
    chapter = summary.chapters[0]
    assert chapter.section_id == section.id
    assert isinstance(chapter.layout, ChapterLayout)
    assert chapter.layout.section_id == section.id
    assert chapter.layout.units[block.id][0].translation == "Ясный день."
    assert stored.opens == 2


def test_layout_matches_export_projection() -> None:
    """The chapter on the result is the chapter export projects for this book."""
    stored, section, block = _translated_book()
    summary = book_result(stored, BOOK_ID, include_quality=False)
    expected = build_book_layout(build_layout_inputs(stored, BOOK_ID))

    assert summary.chapters[0].layout == expected.chapters[section.id]
    assert summary.chapters[0].layout.units[block.id][0].translation == "Ясный день."


def _correction_policy() -> dict:
    return CorrectionSnapshotV2(
        model="fake",
        route={"provider": "fake", "base_url": "http://fake", "proxy": None, "route_id": "fake"},
        parameters={
            "temperature": 0.0,
            "max_output_tokens": 1000,
            "timeout_seconds": 30,
            "max_attempts": 1,
            "max_unit_source_chars": 4000,
            "input_price_per_mtok": 0.0,
            "output_price_per_mtok": 0.0,
        },
        prompt="Correct chapter.",
        review_prompt="Review candidate.",
        versions=CORRECTION_VERSIONS_V2,
    ).to_payload()


def test_correction_candidates_support_segments() -> None:
    """Current and proposed results keep segment translations and the old fields."""
    text = "Alpha sat. Beta ran."
    split = text.index("Beta")
    document, section, block = _document_with(
        text,
        segments=(BlockSegment(0, split, 0), BlockSegment(split, len(text), 1)),
    )
    current = BlockProcessingResult(
        block.id,
        BlockTreatment.TRANSLATE,
        "Альфа.",
        glossary_hints=(GlossaryHint("Alpha", "Альфа"),),
        segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS),
        segment_translations=("Альфа.", None),
    )
    proposed = replace(
        current,
        translated_text="Альфа точнее.",
        segment_translations=("Альфа точнее.", None),
    )
    run = _run()
    conditions = SimpleNamespace(
        id=CONDITIONS_ID,
        payload={"schema_version": 1, "reader_level": "a2", "english_help": False},
    )
    policy = _correction_policy()
    source_binding = {"schema_version": 1}
    inputs = CorrectionInputs(
        document,
        section,
        None,
        run,
        conditions,
        parse_correction_snapshot(policy),
        textual_blocks(section),
        {block.id: current},
        {block.id: source_binding},
        {block.id: 1},
    )
    payload = json.loads(json.dumps({
        "schema_version": 1,
        "book_id": BOOK_ID,
        "run_id": run.id,
        "conditions_id": CONDITIONS_ID,
        "section_id": section.id,
        "block_id": block.id,
        "input_sha256": inputs.checksums[block.id],
        "input_generation": 1,
        "group_id": "group-1",
        "proposal_attempt_id": "group-1",
        "context_sha256": inputs.context_sha256,
        "source_binding": source_binding,
        "original": {"text": block.text},
        "current_result": asdict(current),
        "result": asdict(proposed),
        "proposal": {"issues": []},
        "evidence_valid": True,
    }))
    candidate = CorrectionCandidate(
        candidate_id="cand-1",
        book_id=BOOK_ID,
        run_id=run.id,
        conditions_id=CONDITIONS_ID,
        section_id=section.id,
        block_id=block.id,
        input_sha256=inputs.checksums[block.id],
        input_generation=1,
        proposal_sha256=canonical_sha(payload),
        payload=payload,
        group_id="group-1",
        proposal_attempt_id="group-1",
        state="awaiting_human",
    )
    stored = _StoredBook(
        document=document,
        results=(current,),
        run=run,
        conditions=conditions,
        run_binding={
            "book_id": BOOK_ID,
            "conditions_id": CONDITIONS_ID,
            "payload_json": {"schema_version": 1, "correction": policy},
        },
        result_bindings={block.id: source_binding},
        generations={block.id: 1},
        candidates=(candidate,),
    )

    summary = book_result(stored, BOOK_ID, include_quality=False)
    projected = summary.chapters[0].blocks[0].correction_candidates[0]
    assert projected["current_result"]["segment_translations"] == ["Альфа.", None]
    assert projected["proposed_result"]["segment_translations"] == ["Альфа точнее.", None]
    assert projected["current_result"]["translated_text"] == "Альфа."
    assert projected["proposed_result"]["translated_text"] == "Альфа точнее."
    assert projected["current_result"]["glossary_hints"] == [
        {"original": "Alpha", "translation": "Альфа", "note": None}
    ]
    assert projected["proposed_result"]["glossary_hints"] == [
        {"original": "Alpha", "translation": "Альфа", "note": None}
    ]
    assert projected["current_result"]["language_notes"] == []
    assert projected["proposed_result"]["language_notes"] == []
