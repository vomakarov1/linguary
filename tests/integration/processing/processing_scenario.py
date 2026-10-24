"""Building a book that has reached processing stage and prepared model replies (P3.T6).

Processing begins where a book is already imported, all chapters analyzed,
the glossary reviewed, and a processing run is active. This module constructs that
state in real storage using a single function so that tests only describe what they verify.

The module intentionally lacks a `test_` prefix: otherwise pytest would discover it as a test suite.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from linguary.adapters.config import CheckSettings
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application.context_builder import ContextBuilder, textual_blocks
from linguary.application.ports.llm import (
    LLMClient,
    LLMPayload,
    LLMRequest,
    LLMResponse,
)
from linguary.application.ports.persistence import BookRecord
from linguary.application.retry_policy import RetryPolicy
from linguary.application.translation_units import DEFAULT_MAX_UNIT_SOURCE_CHARS
from linguary.application.use_cases.process_chapter import ProcessChapter
from linguary.application.use_cases.regenerate_block import RegenerateBlock
from linguary.application.validators import TranslationResultValidator
from linguary.domain import (
    BeforeYouReadEntry,
    BlockTreatment,
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    DocumentBlock,
    GlossaryCandidate,
    DocumentSection,
    LLMSettingsSnapshot,
    ProcessingRun,
    ProcessingRunStatus,
    SourceReference,
    SupportStatus,
    TermStatus,
    TokenUsage,
    TreatmentPlan,
)
from linguary.domain.processing import GlossarySnapshot, GlossarySnapshotEntry

BOOK_ID = "book-1"
RUN_ID = "run-1"
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "ru"

#: Source paragraph and its finished Russian translation: texts against which checks pass cleanly.
SOURCE_TEXT = "It is a truth universally acknowledged that a man must want a wife."
TRANSLATED_TEXT = "Общепризнанная истина гласит, что мужчина непременно ищет жену."

#: Fixed point in time: test does not depend on the current clock.
CREATED_AT = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ProcessingScenario:
    """Book in `PROCESSING` status along with its active processing run."""

    document: CanonicalDocument
    run_id: str = RUN_ID

    def section(self, index: int = 0) -> DocumentSection:
        """Chapter of the book in reading order."""
        return self.document.ordered_sections[index]

    def block_ids(self, index: int = 0) -> tuple[str, ...]:
        """Block identifiers of the chapter in original document order."""
        return tuple(block.id for block in self.section(index).ordered_blocks)


def build_document(chapters: Sequence[Sequence[str | None]]) -> CanonicalDocument:
    """Book made of chapters defined by the texts of their paragraphs."""
    return CanonicalDocument(
        id=BOOK_ID,
        metadata=BookMetadata(title="Pride and Prejudice", authors=("Jane Austen",)),
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        sections=tuple(
            _section(order=order, texts=texts) for order, texts in enumerate(chapters)
        ),
    )


def store_processing_book(
    unit_of_work: SqlAlchemyUnitOfWork,
    *,
    chapters: Sequence[Sequence[str | None]] = ((SOURCE_TEXT, SOURCE_TEXT),),
    treatments: Mapping[str, Sequence[BlockTreatment]] | None = None,
    status: BookStatus = BookStatus.PROCESSING,
    locked_terms: bool = True,
    run_id: str = RUN_ID,
    analysed_sections: Sequence[str] | None = None,
    before_you_read: Sequence[tuple[str, str]] = (),
    translation_instructions: str = "Сохранять иронию рассказчика.",
    glossary_candidates: Mapping[str, Sequence[tuple[str, str]]] | None = None,
    glossary_entries: Sequence[GlossarySnapshotEntry] = (),
) -> ProcessingScenario:
    """Stores a book ready for processing along with an active run into persistence.

    `analysed_sections` keeps analysis only for listed chapters: processing a chapter
    without a saved analysis is an explicitly forbidden condition (PRD § 14).
    `before_you_read` — pairs of (expression, explanation) for chapter preparation.
    `glossary_candidates` — pairs of (source_text, model_variant) for per-chapter analysis.
    `glossary_entries` — run glossary snapshot entries beyond the Mr. Bennet decision.
    """
    document = build_document(chapters)
    with unit_of_work.transaction() as uow:
        uow.books.add(_book(status))
        uow.documents.save(document)
        for section in document.ordered_sections:
            if analysed_sections is not None and section.id not in analysed_sections:
                continue
            uow.analyses.save(
                BOOK_ID,
                _analysis(
                    section,
                    treatments=treatments,
                    before_you_read=before_you_read,
                    translation_instructions=translation_instructions,
                    candidates=(glossary_candidates or {}).get(section.id, ()),
                ),
            )
        uow.runs.add(
            _run(run_id, locked_terms=locked_terms, extra_entries=glossary_entries)
        )
    return ProcessingScenario(document=document, run_id=run_id)


#: Book decisions from the third live run: the user corrected the model variant
#: "Даскмир" to "Даскмер", while the note `the marsh gate` mentions the former variant.
REJECTED_VARIANT_GLOSSARY = (
    GlossarySnapshotEntry(
        source_text="Duskmere",
        translation="Даскмер",
        type=ConceptType.PLACE,
        status=TermStatus.APPROVED,
    ),
    GlossarySnapshotEntry(
        source_text="the marsh gate",
        translation="болотные ворота",
        type=ConceptType.PLACE,
        status=TermStatus.APPROVED,
        note="Сторожевой пост при въезде в Даскмир. Множественное число, склоняется.",
    ),
)

#: Chapter 2 instructions of the same run: the rejected variant is named without quotes.
REJECTED_VARIANT_INSTRUCTIONS = (
    "1. Сохранять иронию рассказчика.\n"
    "8. Термины из глоссария (болотные огни, Даскмир) использовать единообразно."
)


def store_book_with_rejected_variant(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> ProcessingScenario:
    """Book with two chapters: variant 'Даскмир' was proposed only in chapter 1 analysis (D-4)."""
    return store_processing_book(
        unit_of_work,
        chapters=((SOURCE_TEXT,), (SOURCE_TEXT,)),
        translation_instructions=REJECTED_VARIANT_INSTRUCTIONS,
        glossary_candidates={"section-1": (("Duskmere", "Даскмир"),)},
        glossary_entries=REJECTED_VARIANT_GLOSSARY,
    )


@dataclass(slots=True)
class InterruptingLLMClient(LLMClient):
    """Client that yields to a user-triggered event before replying.

    This reproduces race condition ALG-04: the model response is in flight while
    the book state changes in the meantime (e.g., user returns to glossary review).
    Otherwise such a sequence would have to be simulated with storage stubs, while
    the active run check actually lives within a real transaction.
    """

    inner: FakeLLMClient
    interrupt: Callable[[], None]

    @property
    def call_count(self) -> int:
        """How many requests the client received."""
        return self.inner.call_count

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        self.interrupt()
        return await self.inner.generate(request)


def make_process_chapter(
    unit_of_work: SqlAlchemyUnitOfWork,
    client: LLMClient,
    *,
    max_attempts: int = 3,
    max_unit_source_chars: int = DEFAULT_MAX_UNIT_SOURCE_CHARS,
    checks: CheckSettings | None = None,
) -> tuple[ProcessChapter, list[float]]:
    """Chapter processing use case along with a list of recorded retry delays."""
    delays: list[float] = []
    scenario = ProcessChapter(
        unit_of_work=unit_of_work,
        llm_client=client,
        retry_policy=RetryPolicy(max_attempts=max_attempts),
        validator=make_validator(checks),
        max_unit_source_chars=max_unit_source_chars,
        id_factory=sequential_ids("unit"),
        sleep=_recording_sleep(delays),
    )
    return scenario, delays


def make_regenerate_block(
    unit_of_work: SqlAlchemyUnitOfWork,
    client: LLMClient,
    *,
    max_attempts: int = 3,
    checks: CheckSettings | None = None,
    context_builder: ContextBuilder | None = None,
) -> tuple[RegenerateBlock, list[float]]:
    """Block regeneration use case along with a list of recorded retry delays."""
    delays: list[float] = []
    scenario = RegenerateBlock(
        unit_of_work=unit_of_work,
        llm_client=client,
        retry_policy=RetryPolicy(max_attempts=max_attempts),
        validator=make_validator(checks),
        context_builder=context_builder,
        id_factory=sequential_ids("regeneration"),
        sleep=_recording_sleep(delays),
    )
    return scenario, delays


def make_validator(checks: CheckSettings | None = None) -> TranslationResultValidator:
    """Result validation checks with application settings thresholds."""
    settings = checks or CheckSettings()
    return TranslationResultValidator(
        target_language_min_share=settings.target_language_min_share,
        language_check_min_length=settings.language_check_min_length,
        repetition_ngram_size=settings.repetition_ngram_size,
        repetition_min_repeats=settings.repetition_min_repeats,
    )


def _recording_sleep(delays: list[float]) -> Callable[[float], Awaitable[None]]:
    """Retry delay that does not block execution but records its duration."""

    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    return sleep


def sequential_ids(prefix: str) -> Callable[[], str]:
    """Predictable identifiers: `unit-1`, `unit-2`, ..."""
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}-{counter}"

    return next_id


def block_reply(
    alias: str,
    *,
    treatment: BlockTreatment = BlockTreatment.TRANSLATE,
    translated_text: str | None = TRANSLATED_TEXT,
    hints: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Result of a single block in the model reply."""
    return {
        "source_block_id": alias,
        "treatment": treatment.value,
        "translated_text": translated_text if treatment.requires_translation else None,
        "glossary_hints": [
            {"original": original, "translation": translation, "note": None}
            for original, translation in hints
        ],
    }


def note_reply(
    alias: str,
    *,
    original_excerpt: str = "a truth universally acknowledged",
    note: str = "Ироническая формула зачина романа.",
    translated_excerpt: str | None = "Общепризнанная истина",
) -> dict[str, Any]:
    """Language note of the reply bound to the request block."""
    return {
        "source_block_id": alias,
        "original_excerpt": original_excerpt,
        "translated_excerpt": translated_excerpt,
        "note": note,
    }


def translation_reply(
    blocks: Sequence[Mapping[str, Any]],
    *,
    notes: Sequence[Mapping[str, Any]] = (),
    truncated: bool = False,
    usage: TokenUsage | None = None,
) -> FakeReply:
    """Prepared provider reply for processing a block group."""
    payload = {"blocks": list(blocks), "language_notes": list(notes)}
    return FakeReply(
        content=json.dumps(payload, ensure_ascii=False),
        truncated=truncated,
        usage=usage,
    )


def translated_reply(
    aliases: Sequence[str] = ("b1", "b2"),
    *,
    notes: Sequence[Mapping[str, Any]] = (),
    truncated: bool = False,
    usage: TokenUsage | None = None,
) -> FakeReply:
    """Valid reply: full translation of each block in the group."""
    return translation_reply(
        [block_reply(alias) for alias in aliases],
        notes=notes,
        truncated=truncated,
        usage=usage,
    )


def _section(*, order: int, texts: Sequence[str | None]) -> DocumentSection:
    """Book chapter from paragraphs; None instead of text yields a non-textual image block."""
    section_id = f"section-{order + 1}"
    return DocumentSection(
        id=section_id,
        document_id=BOOK_ID,
        title=f"Chapter {order + 1}",
        order=order,
        source_resource_path=f"{section_id}.xhtml",
        blocks=tuple(
            DocumentBlock.create(
                document_id=BOOK_ID,
                section_id=section_id,
                type=BlockType.PARAGRAPH if text is not None else BlockType.IMAGE,
                order=index,
                text=text or "",
                source_reference=SourceReference(
                    resource_path=f"{section_id}.xhtml",
                    structural_path=f"/body/p[{index + 1}]",
                ),
                support_status=SupportStatus.SUPPORTED,
            )
            for index, text in enumerate(texts)
        ),
    )


def _analysis(
    section: DocumentSection,
    *,
    treatments: Mapping[str, Sequence[BlockTreatment]] | None,
    before_you_read: Sequence[tuple[str, str]] = (),
    translation_instructions: str = "Сохранять иронию рассказчика.",
    candidates: Sequence[tuple[str, str]] = (),
) -> ChapterAnalysis:
    """Full chapter analysis: treatment plan covers all its textual blocks.

    Non-textual blocks (images, page breaks) are not included in the plan: they are
    classified by the importer and are not passed to the model (PRD § 10.1).
    """
    blocks = textual_blocks(section)
    chosen = (
        tuple((treatments or {}).get(section.id, ()))
        or (BlockTreatment.TRANSLATE,) * len(blocks)
    )
    analysis_id = f"analysis-{section.id}"
    return ChapterAnalysis.create(
        id=analysis_id,
        section_id=section.id,
        treatment_plan=TreatmentPlan.of_treatments(
            section.id,
            zip((block.id for block in blocks), chosen, strict=True),
        ),
        translation_instructions=translation_instructions,
        chapter_summary="Mr. Bennet visits the new neighbour.",
        expected_block_ids=[block.id for block in blocks],
        glossary_candidates=[
            GlossaryCandidate.create(
                id=f"candidate-{section.id}-{source_text}",
                analysis_id=analysis_id,
                source_text=source_text,
                type=ConceptType.PLACE,
                suggested_translation=translation,
                confidence=0.9,
                model_flagged_review=False,
                confidence_threshold=0.5,
            )
            for source_text, translation in candidates
        ],
        before_you_read=[
            BeforeYouReadEntry(expression=expression, explanation=explanation)
            for expression, explanation in before_you_read
        ],
    )


def _book(status: BookStatus) -> BookRecord:
    """Book record that reached processing stage."""
    return BookRecord(
        id=BOOK_ID,
        metadata=BookMetadata(title="Pride and Prejudice", authors=("Jane Austen",)),
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        status=status,
        source_epub_path="/library/pride.epub",
        created_at=CREATED_AT,
    )


def _run(
    run_id: str,
    *,
    locked_terms: bool,
    extra_entries: Sequence[GlossarySnapshotEntry] = (),
) -> ProcessingRun:
    """Active processing run with glossary and LLM settings snapshots."""
    entries = [
        GlossarySnapshotEntry(
            source_text="Mr. Bennet",
            translation="мистер Беннет",
            type=ConceptType.PERSON,
            status=TermStatus.LOCKED if locked_terms else TermStatus.SUGGESTED,
        ),
        *extra_entries,
    ]
    return ProcessingRun(
        id=run_id,
        book_id=BOOK_ID,
        status=ProcessingRunStatus.ACTIVE,
        glossary_snapshot=GlossarySnapshot(entries=tuple(entries)),
        llm_settings_snapshot=LLMSettingsSnapshot(provider="fake", model="fake-model"),
    )
    entries = [
        GlossarySnapshotEntry(
            source_text="Mr. Bennet",
            translation="мистер Беннет",
            type=ConceptType.PERSON,
            status=TermStatus.LOCKED if locked_terms else TermStatus.SUGGESTED,
        ),
        *extra_entries,
    ]
    return ProcessingRun(
        id=run_id,
        book_id=BOOK_ID,
        status=ProcessingRunStatus.ACTIVE,
        glossary_snapshot=GlossarySnapshot(entries=tuple(entries)),
        llm_settings_snapshot=LLMSettingsSnapshot(provider="fake", model="fake-model"),
    )
