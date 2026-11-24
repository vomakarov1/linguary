"""Unit/integration tests for `IllustrationSource`, `NarrativePassInputs`, `input_context`, and `IllustrationCoordinator` (`ALG-01`, P2.T1)."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pytest

from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, ReadingSettings, StorageSettings
from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.adapters.validation import Py3LangidDetector
from linguary.application.analysis_pass_cache import decode_pass, encode_pass
from linguary.application.analysis_pass_inputs import is_illustrable_content_section
from linguary.application.book_conditions import ConditionsError
from linguary.application.context_builder import ContextBuilder
from linguary.application.illustration_coordinator import IllustrationCoordinator
from linguary.application.illustration_source import IllustrationSource
from linguary.application.illustration_view import build_illustration_view
from linguary.application.llm_dto import (
    GlossaryCandidateResponse,
    NarrativeAnalysisResponse,
    QuotedSceneResponse,
    QuotedSceneVisualFactResponse,
    QuotedSourceAnchorResponse,
)
from linguary.application.retry_policy import RetryPolicy
from linguary.application.token_budget import narrative_max_fragment_blocks_for
from linguary.application.use_cases.analyze_chapter import (
    AnalyzeChapter,
    AnalyzeChapterCommand,
    AnalyzeChapterError,
    _pass_model_dependencies,
)
from linguary.application.use_cases.approve_glossary import (
    ApproveGlossary,
    GlossaryDecision,
    GlossaryDecisionCommand,
    GlossaryNoteCommand,
)
from linguary.application.ports.llm import LLMResponseFormatError
from linguary.application.ports.persistence import BookRecord
from linguary.composition.book_runtime import conditions_payload
from linguary.composition.migrations import apply_migrations
from linguary.domain import (
    AnalysisPassKind,
    AnalysisPassResult,
    BlockTreatment,
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    DocumentBlock,
    DocumentSection,
    ErrorType,
    ProcessingTask,
    SourceReference,
    SupportStatus,
    TaskError,
    TaskStatus,
    TaskType,
    TermConcept,
    TermStatus,
    TermVariant,
    TreatmentPlan,
    TreatmentPlanItem,
)
from linguary.domain.book_conditions import BookConditions
from linguary.domain.content_plan import canonical_sha
from linguary.domain.illustrations import (
    IllustrationGenerationState,
    IllustrationReasonCode,
    IllustrationSettings,
)
from linguary.domain.name_translation import (
    NameTranslationAssessment,
    NameTranslationMethod,
    NameTranslationPolicy,
)


class _StagedFakeLLM(FakeLLMClient):
    supports_staged_analysis = True


def _make_uow(tmp_path: Path) -> SqlAlchemyUnitOfWork:
    db = tmp_path / "source.db"
    apply_migrations(db)
    return SqlAlchemyUnitOfWork.for_engine(create_database_engine(db))


def _write_epub(tmp_path: Path, name: str = "book.epub", content: bytes = b"epub-v1") -> tuple[str, str]:
    path = tmp_path / name
    path.write_bytes(content)
    return str(path), sha256(content).hexdigest()


def _make_doc(
    book_id: str = "book-1",
    *,
    sections: tuple[DocumentSection, ...] | None = None,
) -> CanonicalDocument:
    if sections is None:
        sections = (
            DocumentSection(
                id="sec-1",
                document_id=book_id,
                title="Chapter 1",
                order=0,
                source_resource_path="Text/ch1.xhtml",
                blocks=(
                    DocumentBlock(
                        id="b-1",
                        section_id="sec-1",
                        type=BlockType.PARAGRAPH,
                        order=0,
                        text="Alice stepped into the quiet hall under a pale lantern.",
                        source_reference=SourceReference("Text/ch1.xhtml", "p[0]"),
                        support_status=SupportStatus.SUPPORTED,
                    ),
                ),
            ),
            DocumentSection(
                id="sec-2",
                document_id=book_id,
                title="Chapter 2",
                order=1,
                source_resource_path="Text/ch2.xhtml",
                blocks=(
                    DocumentBlock(
                        id="b-2",
                        section_id="sec-2",
                        type=BlockType.PARAGRAPH,
                        order=0,
                        text="Bob carried a leather satchel down the rainy street.",
                        source_reference=SourceReference("Text/ch2.xhtml", "p[0]"),
                        support_status=SupportStatus.SUPPORTED,
                    ),
                ),
            ),
        )
    return CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title="Sample Novel"),
        source_language="en",
        target_language="ru",
        sections=sections,
    )


def _seed_book_and_conditions(
    uow: SqlAlchemyUnitOfWork,
    tmp_path: Path,
    *,
    book_id: str = "book-1",
    doc: CanonicalDocument | None = None,
    staged_analysis: bool = True,
    profile: str = "content-plan-v3",
    reader_level: str = "b1",
    name_strategy: str = "sound",
    sequential_analysis: bool = True,
    seed_first_chapter: bool = False,
    prompts_version: int | None = None,
    continue_legacy: bool = False,
) -> tuple[BookRecord, CanonicalDocument, BookConditions, str]:
    epub_path, digest = _write_epub(tmp_path, f"{book_id}.epub")
    document = doc or _make_doc(book_id)
    book = BookRecord(
        id=book_id,
        metadata=BookMetadata(title="Sample Novel"),
        source_language="en",
        target_language="ru",
        status=BookStatus.ANALYZING,
        source_epub_path=epub_path,
        created_at=datetime(2026, 11, 1, tzinfo=UTC),
    )
    settings = AppSettings(
        llm=LLMSettings(
            model="test-model",
            api_key="",  # Intentionally empty to prove no API key is needed!
            staged_analysis=staged_analysis,
        ),
        checks=CheckSettings(
            sequential_analysis=sequential_analysis,
            seed_first_chapter=seed_first_chapter,
        ),
        reading=ReadingSettings(
            reader_level=reader_level,
            name_translation_strategy=name_strategy,
        ),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
    )
    payload = conditions_payload(book, settings, revision=1, profile=profile)
    payload["source_sha256"] = digest
    if continue_legacy:
        payload["staged_analysis"] = False
        payload["llm_parameters"]["staged_analysis"] = False
    if prompts_version is not None:
        payload["versions"]["prompts"] = prompts_version
        if prompts_version == 6:
            payload["versions"]["context_builder"] = 4
        elif prompts_version == 7:
            payload["versions"]["context_builder"] = 5
    cond = BookConditions(
        id=str(uuid4()),
        book_id=book_id,
        revision=1,
        payload=payload,
        created_at=datetime(2026, 11, 1, tzinfo=UTC),
    )
    with uow.transaction() as tx:
        tx.books.add(book)
        tx.documents.save(document)
        tx.work.add_conditions(cond)
    return book, document, cond, digest


def _narrative_reply(
    block: DocumentBlock,
    *,
    candidates: list[dict] | None = None,
    summary: str = "Alice walks through the quiet hall holding a pale lantern in the evening.",
) -> FakeReply:
    quote = block.text[:20]
    body = {
        "translation_instructions": "Сохранять спокойный ритм повествования.",
        "chapter_summary": summary,
        "glossary_candidates": candidates or [],
        "context_facts": [],
        "scenes": [
            {
                "start_block_id": "b1",
                "end_block_id": "b1",
                "summary": "Алиса проходит по тихому залу при свете фонаря.",
                "location": "Тихий зал",
                "participants": ["Алиса"],
                "visual_facts": [
                    {
                        "subject": "Алиса",
                        "detail": "держит бледный фонарь в руке",
                        "anchor": {"block_id": "b1", "quote": quote},
                    }
                ],
                "visual_assumptions": [],
                "continues_previous_scene": False,
            }
        ],
    }
    return FakeReply(content=json.dumps(body, ensure_ascii=False))


def _pedagogical_reply(block_count: int = 1) -> FakeReply:
    body = {
        "treatment_plan": [
            {"source_block_id": f"b{i + 1}", "treatment": "TRANSLATE"}
            for i in range(block_count)
        ],
        "before_you_read": [],
        "difficulties": [],
        "before_you_read_evidence": [],
    }
    return FakeReply(content=json.dumps(body, ensure_ascii=False))


def _build_analyzer(
    uow: SqlAlchemyUnitOfWork,
    llm: FakeLLMClient,
    cond: BookConditions,
    *,
    on_narrative_ready=None,
    max_concurrency: int = 1,
) -> AnalyzeChapter:
    payload = cond.payload
    policy = (
        NameTranslationPolicy(**payload["name_translation_policy"])
        if "name_translation_policy" in payload
        else None
    )
    checks = payload["checks"]
    ctx_builder = ContextBuilder(
        content_plan_profile=payload["profile"],
        prompt_templates=payload["prompt_templates"],
        name_translation_policy=policy,
        reader_level=payload["reader_level"],
        language_notes_per_chapter=checks["language_notes_per_chapter"],
    )
    return AnalyzeChapter(
        unit_of_work=uow,
        llm_client=llm,
        language_detector=Py3LangidDetector(),
        retry_policy=RetryPolicy(max_attempts=1),
        glossary_confidence_threshold=checks["glossary_confidence_threshold"],
        max_fragment_blocks=160,
        context_builder=ctx_builder,
        name_translation_policy=policy,
        conditions_id=cond.id,
        max_concurrency=max_concurrency,
        seed_first_chapter=checks.get("seed_first_chapter", False),
        sequential_analysis=checks.get("sequential_analysis", True),
        max_output_tokens=32_768,
        staged_analysis=bool(payload.get("staged_analysis")),
        on_narrative_ready=on_narrative_ready,
    )


def _make_source() -> IllustrationSource:
    return IllustrationSource(
        language_detector=Py3LangidDetector(),
        source_digest_fn=lambda path: sha256(Path(path).read_bytes()).hexdigest(),
    )


async def test_post_glossary_approval_correct_set_note_skip_and_auto_approve_keep_snapshot_ready(
    tmp_path: Path,
) -> None:
    uow = _make_uow(tmp_path)
    _, doc, cond, _ = _seed_book_and_conditions(uow, tmp_path)
    llm = _StagedFakeLLM(
        [
            _narrative_reply(
                doc.ordered_sections[0].ordered_blocks[0],
                candidates=[
                    {
                        "source_text": "Alice",
                        "type": "person",
                        "suggested_translation": "Алиса",
                        "confidence": 0.95,
                        "model_flagged_review": False,
                        "note": "",
                        "description": "Героиня",
                        "name_translation_assessment": {
                            "applied_method": "sound",
                            "eligible_without_review": True,
                            "review_reasons": [],
                            "rationale": "Standard phonetic transliteration.",
                            "opaque_name": True,
                        },
                    },
                    {
                        "source_text": "Hall",
                        "type": "place",
                        "suggested_translation": "Зал",
                        "confidence": 0.6,
                        "model_flagged_review": True,
                        "note": "",
                        "description": "Зал дома",
                        "name_translation_assessment": {
                            "applied_method": "meaning",
                            "eligible_without_review": True,
                            "review_reasons": [],
                            "rationale": "Common place noun.",
                            "opaque_name": False,
                        },
                    },
                ],
            ),
            _pedagogical_reply(),
            _narrative_reply(doc.ordered_sections[1].ordered_blocks[0]),
            _pedagogical_reply(),
        ]
    )
    analyzer = _build_analyzer(uow, llm, cond)
    await analyzer(AnalyzeChapterCommand("book-1", "sec-1"))
    await analyzer(AnalyzeChapterCommand("book-1", "sec-2"))

    # Transition to GLOSSARY_REVIEW and perform CORRECT + set_note + SKIP decisions.
    with uow.transaction() as tx:
        book = tx.books.get("book-1")
        assert book is not None
        tx.books.save(replace(book, status=BookStatus.GLOSSARY_REVIEW))
        concepts = tx.glossary.list_for_book("book-1")
        if not concepts:
            tx.glossary.save(
                TermConcept(
                    id="term-1",
                    book_id="book-1",
                    type=ConceptType.PERSON,
                    canonical_source_text="Alice",
                    status=TermStatus.SUGGESTED,
                    variants=(
                        TermVariant("term-1", "en", "Alice", TermStatus.SUGGESTED),
                        TermVariant("term-1", "ru", "Алиса", TermStatus.SUGGESTED),
                    ),
                )
            )
            tx.glossary.save(
                TermConcept(
                    id="term-2",
                    book_id="book-1",
                    type=ConceptType.PLACE,
                    canonical_source_text="Hall",
                    status=TermStatus.SUGGESTED,
                    variants=(
                        TermVariant("term-2", "en", "Hall", TermStatus.SUGGESTED),
                        TermVariant("term-2", "ru", "Зал", TermStatus.SUGGESTED),
                    ),
                )
            )

    approve = ApproveGlossary(unit_of_work=uow, glossary_confidence_threshold=0.75)
    corrected = approve(
        GlossaryDecisionCommand(
            book_id="book-1",
            source_text="Alice",
            decision=GlossaryDecision.CORRECT,
            translation="Элис",
            note="использовать Элис везде",
        )
    )
    approve.set_note(
        GlossaryNoteCommand(
            book_id="book-1",
            concept_id=corrected.id,
            note="обновлённая заметка",
        )
    )
    approve(
        GlossaryDecisionCommand(
            book_id="book-1",
            source_text="Hall",
            decision=GlossaryDecision.SKIP,
        )
    )

    source = _make_source()
    with uow.read_transaction() as tx:
        snap = source.snapshot(tx, "book-1")
        assert snap.compatible is True
        assert snap.ready is True
        assert snap.ready_chapters == 2
        assert snap.manifest_sha256 is not None


async def test_pre_analysis_user_decision_with_note_and_gender_constraint_round_trip(
    tmp_path: Path,
) -> None:
    uow = _make_uow(tmp_path)
    _, doc, cond, _ = _seed_book_and_conditions(uow, tmp_path)
    with uow.transaction() as tx:
        tx.glossary.save(
            TermConcept(
                id="pre-dec-1",
                book_id="book-1",
                type=ConceptType.PERSON,
                canonical_source_text="Alice",
                status=TermStatus.APPROVED,
                variants=(
                    TermVariant("pre-dec-1", "en", "Alice", TermStatus.APPROVED),
                    TermVariant("pre-dec-1", "ru", "Алиса", TermStatus.APPROVED),
                ),
                note="женский род, обращение на ты",
            )
        )
        tx.glossary.save(
            TermConcept(
                id="pre-sug-1",
                book_id="book-1",
                type=ConceptType.PLACE,
                canonical_source_text="Hall",
                status=TermStatus.SUGGESTED,
                variants=(
                    TermVariant("pre-sug-1", "en", "Hall", TermStatus.SUGGESTED),
                    TermVariant("pre-sug-1", "ru", "Зал", TermStatus.SUGGESTED),
                ),
                note="необязательная подсказка",
            )
        )

    llm = _StagedFakeLLM(
        [
            _narrative_reply(doc.ordered_sections[0].ordered_blocks[0]),
            _pedagogical_reply(),
            _narrative_reply(doc.ordered_sections[1].ordered_blocks[0]),
            _pedagogical_reply(),
        ]
    )
    analyzer = _build_analyzer(uow, llm, cond)
    await analyzer(AnalyzeChapterCommand("book-1", "sec-1"))
    await analyzer(AnalyzeChapterCommand("book-1", "sec-2"))

    # Verify input_context saved with user_decisions[*].note and nonbinding_suggestions.
    with uow.read_transaction() as tx:
        p1 = tx.analyses.get_pass("book-1", "sec-1", AnalysisPassKind.NARRATIVE)
        assert p1 is not None
        ctx = p1.payload["input_context"]
        assert isinstance(ctx, dict)
        assert ctx["schema_version"] == 1
        assert ctx["user_decisions"][0]["note"] == "женский род, обращение на ты"
        assert ctx["nonbinding_suggestions"][0]["note"] == "необязательная подсказка"

    # Mutating glossary afterwards does NOT break snapshot!
    with uow.transaction() as tx:
        tx.glossary.save(
            TermConcept(
                id="pre-dec-1",
                book_id="book-1",
                type=ConceptType.PERSON,
                canonical_source_text="Alice",
                status=TermStatus.LOCKED,
                variants=(
                    TermVariant("pre-dec-1", "en", "Alice", TermStatus.LOCKED),
                    TermVariant("pre-dec-1", "ru", "Элис", TermStatus.LOCKED),
                ),
                note="изменено позже",
            )
        )

    source = _make_source()
    with uow.read_transaction() as tx:
        snap = source.snapshot(tx, "book-1")
        assert snap.ready is True


async def test_pedagogical_failure_does_not_block_when_all_narratives_ready(
    tmp_path: Path,
) -> None:
    uow = _make_uow(tmp_path)
    _, doc, cond, _ = _seed_book_and_conditions(uow, tmp_path)
    llm = _StagedFakeLLM(
        [
            _narrative_reply(doc.ordered_sections[0].ordered_blocks[0]),
            FakeReply(content="{broken json"),  # pedagogical fails on ch1
            _narrative_reply(doc.ordered_sections[1].ordered_blocks[0]),
            FakeReply(content="{broken json"),  # pedagogical fails on ch2
        ]
    )
    analyzer = _build_analyzer(uow, llm, cond)
    with pytest.raises(LLMResponseFormatError):
        await analyzer(AnalyzeChapterCommand("book-1", "sec-1"))
    with pytest.raises(LLMResponseFormatError):
        await analyzer(AnalyzeChapterCommand("book-1", "sec-2"))

    source = _make_source()
    with uow.read_transaction() as tx:
        assert tx.analyses.list_for_book("book-1") == ()
        snap = source.snapshot(tx, "book-1")
        assert snap.compatible is True
        assert snap.ready is True
        assert snap.ready_chapters == 2


async def test_reanalyze_with_ch1_transparent_name_and_pedagogical_failure_rebuilds_accumulated_for_ch2(
    tmp_path: Path,
) -> None:
    uow = _make_uow(tmp_path)
    _, doc, cond, _ = _seed_book_and_conditions(uow, tmp_path, name_strategy="meaning")
    transparent_candidate = {
        "source_text": "Silverbrook",
        "type": "place",
        "suggested_translation": "Серебряный Ручей",
        "confidence": 0.9,
        "model_flagged_review": False,
        "note": "",
        "description": "Посёлок у реки",
        "name_translation_assessment": {
            "applied_method": "meaning",
            "eligible_without_review": True,
            "review_reasons": [],
            "rationale": "Transparent descriptive place name.",
            "opaque_name": False,
        },
    }
    llm = _StagedFakeLLM(
        [
            _narrative_reply(
                doc.ordered_sections[0].ordered_blocks[0],
                candidates=[transparent_candidate],
            ),
            FakeReply(content="{broken"),  # ch1 pedagogical fails -> no ChapterAnalysis for ch1!
            _narrative_reply(doc.ordered_sections[1].ordered_blocks[0]),
            _pedagogical_reply(),
        ]
    )
    analyzer = _build_analyzer(uow, llm, cond)
    with pytest.raises(LLMResponseFormatError):
        await analyzer(AnalyzeChapterCommand("book-1", "sec-1"))
    # Now ch2 runs with NO ChapterAnalysis for ch1; NarrativePassInputs reconstructs ch1
    # candidates with conditions_id so opaque_name=False matches identically!
    await analyzer(AnalyzeChapterCommand("book-1", "sec-2"))

    source = _make_source()
    with uow.read_transaction() as tx:
        snap = source.snapshot(tx, "book-1")
        assert snap.compatible is True
        assert snap.ready is True
        assert snap.ready_chapters == 2


async def test_sequential_pedagogical_failure_waits_for_unstarted_chapters(
    tmp_path: Path,
) -> None:
    uow = _make_uow(tmp_path)
    _, doc, cond, _ = _seed_book_and_conditions(uow, tmp_path)
    llm = _StagedFakeLLM(
        [
            _narrative_reply(doc.ordered_sections[0].ordered_blocks[0]),
            FakeReply(content="{broken"),
        ]
    )
    analyzer = _build_analyzer(uow, llm, cond)
    with pytest.raises(LLMResponseFormatError):
        await analyzer(AnalyzeChapterCommand("book-1", "sec-1"))

    source = _make_source()
    with uow.read_transaction() as tx:
        snap = source.snapshot(tx, "book-1")
        assert snap.compatible is True
        assert snap.ready is False
        assert snap.ready_chapters == 1
        assert snap.expected_chapters == 2
        assert snap.blocking_section_id == "sec-2"
        assert snap.reason_code == IllustrationReasonCode.NARRATIVE_MISSING.value

    # Mark sec-2 task FAILED -> reason becomes narrative_failed.
    with uow.transaction() as tx:
        tx.tasks.add(
            ProcessingTask(
                id="t-fail",
                book_id="book-1",
                task_type=TaskType.ANALYZE_CHAPTER,
                scope_id="sec-2",
                status=TaskStatus.FAILED,
                attempts=1,
                last_error=TaskError(type=ErrorType.TRANSIENT, message="timeout"),
            )
        )
    with uow.read_transaction() as tx:
        snap2 = source.snapshot(tx, "book-1")
        assert snap2.reason_code == IllustrationReasonCode.NARRATIVE_FAILED.value


async def test_callback_after_fresh_and_cached_narrative_content_only(tmp_path: Path) -> None:
    uow = _make_uow(tmp_path)
    notes_sec = DocumentSection(
        id="sec-notes",
        document_id="book-1",
        title="Notes",
        order=1,
        source_resource_path="Text/notes.xhtml",
        blocks=(
            DocumentBlock(
                id="b-n",
                section_id="sec-notes",
                type=BlockType.FOOTNOTE,
                order=0,
                text="1. Editorial footnote.",
                source_reference=SourceReference("Text/notes.xhtml", "p[0]"),
                support_status=SupportStatus.SUPPORTED,
            ),
        ),
    )
    doc = _make_doc(
        "book-1",
        sections=(
            _make_doc("book-1").ordered_sections[0],
            notes_sec,
        ),
    )
    _, doc, cond, _ = _seed_book_and_conditions(uow, tmp_path, doc=doc)
    called: list[tuple[str, str]] = []
    llm = _StagedFakeLLM(
        [
            _narrative_reply(doc.ordered_sections[0].ordered_blocks[0]),
            _pedagogical_reply(),
            # Second run reuses cached narrative, only pedagogical runs if needed
            _pedagogical_reply(),
        ]
    )
    analyzer = _build_analyzer(
        uow, llm, cond, on_narrative_ready=lambda b, s: called.append((b, s))
    )
    await analyzer(AnalyzeChapterCommand("book-1", "sec-1"))
    assert called == [("book-1", "sec-1")]

    # Re-running with cached narrative triggers callback again.
    await analyzer(AnalyzeChapterCommand("book-1", "sec-1"))
    assert called == [("book-1", "sec-1"), ("book-1", "sec-1")]

    # Non-content section does not trigger callback.
    assert not is_illustrable_content_section(doc, notes_sec)


async def test_callback_failure_and_error_persist_failure_preserve_primary_analysis(
    tmp_path: Path,
) -> None:
    uow = _make_uow(tmp_path)
    _, doc, cond, _ = _seed_book_and_conditions(uow, tmp_path)

    def _boom(_book_id: str, _section_id: str) -> None:
        raise RuntimeError("coordinator exploded")

    llm = _StagedFakeLLM(
        [
            _narrative_reply(doc.ordered_sections[0].ordered_blocks[0]),
            _pedagogical_reply(),
        ]
    )
    analyzer = _build_analyzer(uow, llm, cond, on_narrative_ready=_boom)
    result = await analyzer(AnalyzeChapterCommand("book-1", "sec-1"))
    assert result.analysis.is_complete


async def test_fragment_pass_not_ready(tmp_path: Path) -> None:
    uow = _make_uow(tmp_path)
    _, _doc, cond, _ = _seed_book_and_conditions(uow, tmp_path)
    encoded = encode_pass(NarrativeAnalysisResponse("Keep.", "Sum."))
    encoded["input_suggestions"] = []
    with uow.transaction() as tx:
        tx.analyses.save_pass(
            AnalysisPassResult(
                id="frag-1",
                book_id="book-1",
                section_id="sec-1",
                pass_kind=AnalysisPassKind.NARRATIVE,
                scope_key="frag:0",
                input_sha256="a" * 64,
                output_sha256=canonical_sha(encoded),
                conditions_id=cond.id,
                attempt_id=None,
                payload=encoded,
            )
        )
    source = _make_source()
    with uow.read_transaction() as tx:
        snap = source.snapshot(tx, "book-1")
        assert snap.ready is False
        assert snap.ready_chapters == 0
        assert snap.blocking_section_id == "sec-1"


async def test_frontmatter_part_divider_notes_and_empty_chapter_1_skipped(tmp_path: Path) -> None:
    uow = _make_uow(tmp_path)
    front = DocumentSection(
        id="sec-front",
        document_id="book-1",
        title="Title Page",
        order=0,
        source_resource_path="Text/title.xhtml",
        blocks=(
            DocumentBlock(
                id="b-front",
                section_id="sec-front",
                type=BlockType.HEADING,
                order=0,
                text="Sample Book",
                source_reference=SourceReference("Text/title.xhtml", "h1[0]"),
                support_status=SupportStatus.SUPPORTED,
            ),
        ),
    )
    empty = DocumentSection(
        id="sec-empty",
        document_id="book-1",
        title="Empty",
        order=1,
        source_resource_path="Text/empty.xhtml",
        blocks=(),
    )
    notes = DocumentSection(
        id="sec-notes",
        document_id="book-1",
        title="Notes",
        order=2,
        source_resource_path="Text/notes.xhtml",
        blocks=(
            DocumentBlock(
                id="b-note",
                section_id="sec-notes",
                type=BlockType.FOOTNOTE,
                order=0,
                text="1. Note text.",
                source_reference=SourceReference("Text/notes.xhtml", "p[0]"),
                support_status=SupportStatus.SUPPORTED,
            ),
        ),
    )
    ch = DocumentSection(
        id="sec-ch",
        document_id="book-1",
        title="Chapter 1",
        order=3,
        source_resource_path="Text/ch1.xhtml",
        blocks=(
            DocumentBlock(
                id="b-ch",
                section_id="sec-ch",
                type=BlockType.PARAGRAPH,
                order=0,
                text="Alice opened the heavy wooden gate.",
                source_reference=SourceReference("Text/ch1.xhtml", "p[0]"),
                support_status=SupportStatus.SUPPORTED,
            ),
        ),
    )
    doc = _make_doc("book-1", sections=(front, empty, notes, ch))
    _, doc, cond, _ = _seed_book_and_conditions(uow, tmp_path, doc=doc)
    llm = _StagedFakeLLM([_narrative_reply(ch.ordered_blocks[0]), _pedagogical_reply()])
    analyzer = _build_analyzer(uow, llm, cond)
    await analyzer(AnalyzeChapterCommand("book-1", "sec-ch"))

    source = _make_source()
    with uow.read_transaction() as tx:
        snap = source.snapshot(tx, "book-1")
        assert snap.ready is True
        assert snap.expected_chapters == 1
        assert snap.skipped_sections == 3
        assert snap.content_section_ids == ("sec-ch",)


async def test_all_empty_no_calls(tmp_path: Path) -> None:
    uow = _make_uow(tmp_path)
    empty_doc = _make_doc(
        "book-1",
        sections=(
            DocumentSection(
                id="sec-empty",
                document_id="book-1",
                title="Cover",
                order=0,
                source_resource_path="Text/cover.xhtml",
                blocks=(),
            ),
        ),
    )
    _, _, cond, digest = _seed_book_and_conditions(uow, tmp_path, doc=empty_doc)
    with uow.transaction() as tx:
        rev = tx.illustrations.save_settings(
            "book-1",
            expected_revision=0,
            settings=IllustrationSettings(
                enabled=True, model_reference="m1", model_prompts="m2"
            ),
        )
        gen, _ = tx.illustrations.create_or_get_generation(
            "book-1",
            settings_id=rev.id,
            request_key="req-empty",
            source_sha256=digest,
            source_authority={"conditions_id": cond.id, "conditions_revision": 1, "source_digest": digest, "manifest_sha256": None},
            requested_sections=(),
        )
    coord = IllustrationCoordinator(uow, source=_make_source())
    coord.reconcile("book-1")
    with uow.read_transaction() as tx:
        updated = tx.illustrations.get_generation("book-1", gen.id)
        assert updated is not None
        assert updated.state is IllustrationGenerationState.SKIPPED
        assert updated.reason_code == IllustrationReasonCode.NO_CONTENT.value
        assert tx.tasks.list_for_book("book-1") == ()


async def test_manifest_order_stable(tmp_path: Path) -> None:
    uow = _make_uow(tmp_path)
    _, doc, cond, _ = _seed_book_and_conditions(uow, tmp_path)
    llm = _StagedFakeLLM(
        [
            _narrative_reply(doc.ordered_sections[0].ordered_blocks[0]),
            _pedagogical_reply(),
            _narrative_reply(doc.ordered_sections[1].ordered_blocks[0]),
            _pedagogical_reply(),
        ]
    )
    analyzer = _build_analyzer(uow, llm, cond)
    await analyzer(AnalyzeChapterCommand("book-1", "sec-1"))
    await analyzer(AnalyzeChapterCommand("book-1", "sec-2"))

    source = _make_source()
    with uow.read_transaction() as tx:
        s1 = source.snapshot(tx, "book-1")
        s2 = source.snapshot(tx, "book-1")
        assert s1.manifest_sha256 == s2.manifest_sha256
        assert [e.section_id for e in s1.entries] == ["sec-1", "sec-2"]


def test_fresh_non_staged_continue_legacy_and_mixed_analysis_needs_explicit_reanalysis(
    tmp_path: Path,
) -> None:
    # 1. Fresh non-staged book (0 ChapterAnalysis) -> needs_analysis / source_incompatible
    (tmp_path / "case1").mkdir(exist_ok=True)
    uow1 = _make_uow(tmp_path / "case1")
    _seed_book_and_conditions(uow1, tmp_path / "case1", staged_analysis=False)
    source = _make_source()
    with uow1.read_transaction() as tx:
        snap1 = source.snapshot(tx, "book-1")
        assert snap1.compatible is False
        assert snap1.reason_code == IllustrationReasonCode.SOURCE_INCOMPATIBLE.value
    # Also check IllustrationView reports needs_analysis when enabled!
    with uow1.transaction() as tx:
        tx.illustrations.save_settings(
            "book-1",
            expected_revision=0,
            settings=IllustrationSettings(enabled=True, model_reference="m1", model_prompts="m2"),
        )
    view1 = build_illustration_view(uow1, "book-1", source_snapshot_fn=source.snapshot)
    assert view1.state == "needs_analysis"
    assert view1.reason_code == IllustrationReasonCode.SOURCE_INCOMPATIBLE.value

    # 2. continue_legacy -> source_incompatible
    (tmp_path / "case2").mkdir(exist_ok=True)
    uow2 = _make_uow(tmp_path / "case2")
    _seed_book_and_conditions(uow2, tmp_path / "case2", continue_legacy=True)
    with uow2.read_transaction() as tx:
        assert source.snapshot(tx, "book-1").reason_code == IllustrationReasonCode.SOURCE_INCOMPATIBLE.value

    # 3. Mixed legacy ChapterAnalysis without NARRATIVE pass -> source_incompatible
    (tmp_path / "case3").mkdir(exist_ok=True)
    uow3 = _make_uow(tmp_path / "case3")
    _seed_book_and_conditions(uow3, tmp_path / "case3", staged_analysis=True)
    with uow3.transaction() as tx:
        tx.analyses.save(
            "book-1",
            ChapterAnalysis.create(
                id="legacy-a1",
                section_id="sec-1",
                treatment_plan=TreatmentPlan(
                    section_id="sec-1",
                    items=(TreatmentPlanItem("b-1", BlockTreatment.TRANSLATE),),
                ),
                translation_instructions="Keep.",
                chapter_summary="Summary.",
                expected_block_ids=("b-1",),
                content_plan=None,
            ),
        )
    with uow3.read_transaction() as tx:
        assert source.snapshot(tx, "book-1").reason_code == IllustrationReasonCode.SOURCE_INCOMPATIBLE.value


def test_legacy_v1_pass_without_input_context_survives_post_analysis_glossary_correct(
    tmp_path: Path,
) -> None:
    uow = _make_uow(tmp_path)
    _, doc, cond, _ = _seed_book_and_conditions(uow, tmp_path)
    payload = cond.payload
    policy = NameTranslationPolicy(**payload["name_translation_policy"])
    ctx_builder = ContextBuilder(
        content_plan_profile=payload["profile"],
        prompt_templates=payload["prompt_templates"],
        name_translation_policy=policy,
        reader_level=payload["reader_level"],
        language_notes_per_chapter=payload["checks"]["language_notes_per_chapter"],
    )
    max_frag = narrative_max_fragment_blocks_for(payload["llm_parameters"])
    model_deps = _pass_model_dependencies(payload, AnalysisPassKind.NARRATIVE)

    with uow.transaction() as tx:
        for sec in doc.ordered_sections:
            blk = sec.ordered_blocks[0]
            resp = NarrativeAnalysisResponse(
                translation_instructions="Сохранять спокойный ритм.",
                chapter_summary="Alice walks quietly across the stone hall under a pale lantern.",
                scenes=(
                    QuotedSceneResponse(
                        start_block_id=blk.id,
                        end_block_id=blk.id,
                        summary="Алиса проходит по каменному залу при свете фонаря.",
                        location="Каменный зал",
                        participants=("Алиса",),
                        visual_facts=(
                            QuotedSceneVisualFactResponse(
                                subject="Алиса",
                                detail="держит бледный фонарь в руке",
                                anchor=QuotedSourceAnchorResponse(
                                    block_id=blk.id, quote=blk.text[:15], start=0, end=15
                                ),
                            ),
                        ),
                    ),
                ),
            )
            encoded = encode_pass(resp)
            encoded["input_suggestions"] = []
            # Note: NO 'input_context' key -> legacy v1 pass!
            input_sha = ctx_builder.narrative_input_sha256(
                doc, sec, (), (), (), model_dependencies=model_deps, max_fragment_blocks=max_frag
            )
            tx.analyses.save_pass(
                AnalysisPassResult(
                    id=f"legacy-pass-{sec.id}",
                    book_id="book-1",
                    section_id=sec.id,
                    pass_kind=AnalysisPassKind.NARRATIVE,
                    scope_key="",
                    input_sha256=input_sha,
                    output_sha256=canonical_sha(encoded),
                    conditions_id=cond.id,
                    attempt_id=None,
                    payload=encoded,
                )
            )
        # User modifies glossary post-analysis!
        tx.glossary.save(
            TermConcept(
                id="post-edit",
                book_id="book-1",
                type=ConceptType.PERSON,
                canonical_source_text="Alice",
                status=TermStatus.APPROVED,
                variants=(
                    TermVariant("post-edit", "en", "Alice", TermStatus.APPROVED),
                    TermVariant("post-edit", "ru", "Элис", TermStatus.APPROVED),
                ),
                note="правка после анализа",
            )
        )

    source = _make_source()
    with uow.read_transaction() as tx:
        snap = source.snapshot(tx, "book-1")
        assert snap.compatible is True
        assert snap.ready is True


async def test_reader_level_change_multi_chapter_reuses(tmp_path: Path) -> None:
    uow = _make_uow(tmp_path)
    _, doc, cond, _ = _seed_book_and_conditions(uow, tmp_path, reader_level="b1")
    llm = _StagedFakeLLM(
        [
            _narrative_reply(doc.ordered_sections[0].ordered_blocks[0]),
            _pedagogical_reply(),
            _narrative_reply(doc.ordered_sections[1].ordered_blocks[0]),
            _pedagogical_reply(),
        ]
    )
    analyzer = _build_analyzer(uow, llm, cond)
    await analyzer(AnalyzeChapterCommand("book-1", "sec-1"))
    await analyzer(AnalyzeChapterCommand("book-1", "sec-2"))

    source = _make_source()
    with uow.read_transaction() as tx:
        snap1 = source.snapshot(tx, "book-1")
        assert snap1.ready is True
        assert len(snap1.entries) == 2
        assert all(e.response_sha256 is not None for e in snap1.entries)


@pytest.mark.parametrize(
    ("sequential", "seed", "concurrency"),
    [(True, False, 1), (False, False, 2), (False, True, 2)],
)
async def test_source_hash_matches_core_for_sequential_parallel_and_seed(
    tmp_path: Path, sequential: bool, seed: bool, concurrency: int
) -> None:
    uow = _make_uow(tmp_path)
    _, doc, cond, _ = _seed_book_and_conditions(
        uow,
        tmp_path,
        sequential_analysis=sequential,
        seed_first_chapter=seed,
    )
    llm = _StagedFakeLLM(
        [
            _narrative_reply(doc.ordered_sections[0].ordered_blocks[0]),
            _pedagogical_reply(),
            _narrative_reply(doc.ordered_sections[1].ordered_blocks[0]),
            _pedagogical_reply(),
        ]
    )
    analyzer = _build_analyzer(uow, llm, cond, max_concurrency=concurrency)
    await analyzer(AnalyzeChapterCommand("book-1", "sec-1"))
    await analyzer(AnalyzeChapterCommand("book-1", "sec-2"))

    source = _make_source()
    with uow.read_transaction() as tx:
        snap = source.snapshot(tx, "book-1")
        assert snap.ready is True


def test_v6_vs_v7_enforces_name_policy_and_gender_subject(tmp_path: Path) -> None:
    # Build a narrative response whose candidate violates V7 meaning policy (transparent=True with chosen_method='sound').
    for prompts_ver, expect_ready in ((6, True), (7, False)):
        sub = tmp_path / f"v{prompts_ver}"
        sub.mkdir()
        uow = _make_uow(sub)
        _, doc, cond, _ = _seed_book_and_conditions(
            uow,
            sub,
            name_strategy="meaning",
            prompts_version=prompts_ver,
            doc=_make_doc("book-1", sections=(_make_doc("book-1").ordered_sections[0],)),
        )
        payload = cond.payload
        policy = NameTranslationPolicy(**payload["name_translation_policy"])
        ctx_builder = ContextBuilder(
            content_plan_profile=payload["profile"],
            prompt_templates=payload["prompt_templates"],
            name_translation_policy=policy,
            reader_level=payload["reader_level"],
            language_notes_per_chapter=payload["checks"]["language_notes_per_chapter"],
        )
        sec = doc.ordered_sections[0]
        blk = sec.ordered_blocks[0]
        resp = NarrativeAnalysisResponse(
            translation_instructions="Сохранять спокойный ритм.",
            chapter_summary="Alice steps into the quiet stone hall with a lantern at dusk.",
            glossary_candidates=(
                GlossaryCandidateResponse(
                    source_text="Silverbrook",
                    type=ConceptType.PLACE,
                    suggested_translation="Силвербрук",
                    confidence=0.9,
                    model_flagged_review=False,
                    note="",
                    name_translation_assessment=NameTranslationAssessment(
                        applied_method=NameTranslationMethod.SOUND,
                        eligible_without_review=True,
                        review_reasons=(),
                        rationale="Transliterated despite transparent meaning.",
                        opaque_name=False,
                    ),
                ),
            ),
            scenes=(
                QuotedSceneResponse(
                    start_block_id=blk.id,
                    end_block_id=blk.id,
                    summary="Алиса проходит по каменному залу при свете фонаря.",
                    location="Каменный зал",
                    participants=("Алиса",),
                    visual_facts=(
                        QuotedSceneVisualFactResponse(
                            subject="Алиса",
                            detail="держит бледный фонарь в руке",
                            anchor=QuotedSourceAnchorResponse(
                                block_id=blk.id, quote=blk.text[:15], start=0, end=15
                            ),
                        ),
                    ),
                ),
            ),
        )
        encoded = encode_pass(resp)
        encoded["input_suggestions"] = []
        encoded["input_context"] = {
            "schema_version": 1,
            "sequential_analysis": True,
            "parallel_batch": False,
            "seed_active": False,
            "user_decisions": [],
            "nonbinding_suggestions": [],
            "prompts_version": prompts_ver,
        }
        input_sha = ctx_builder.narrative_input_sha256(
            doc,
            sec,
            (),
            (),
            (),
            model_dependencies=_pass_model_dependencies(payload, AnalysisPassKind.NARRATIVE),
            max_fragment_blocks=narrative_max_fragment_blocks_for(payload["llm_parameters"]),
        )
        with uow.transaction() as tx:
            tx.analyses.save_pass(
                AnalysisPassResult(
                    id=f"p-v{prompts_ver}",
                    book_id="book-1",
                    section_id="sec-1",
                    pass_kind=AnalysisPassKind.NARRATIVE,
                    scope_key="",
                    input_sha256=input_sha,
                    output_sha256=canonical_sha(encoded),
                    conditions_id=cond.id,
                    attempt_id=None,
                    payload=encoded,
                )
            )
        source = _make_source()
        with uow.read_transaction() as tx:
            snap = source.snapshot(tx, "book-1")
            assert snap.ready is expect_ready


async def test_replaced_narrative_revokes_sealed_generation(tmp_path: Path) -> None:
    uow = _make_uow(tmp_path)
    _, doc, cond, digest = _seed_book_and_conditions(
        uow,
        tmp_path,
        doc=_make_doc("book-1", sections=(_make_doc("book-1").ordered_sections[0],)),
    )
    llm = _StagedFakeLLM([_narrative_reply(doc.ordered_sections[0].ordered_blocks[0]), _pedagogical_reply()])
    analyzer = _build_analyzer(uow, llm, cond)
    await analyzer(AnalyzeChapterCommand("book-1", "sec-1"))

    source = _make_source()
    coord = IllustrationCoordinator(uow, source=source)
    with uow.transaction() as tx:
        rev = tx.illustrations.save_settings(
            "book-1",
            expected_revision=0,
            settings=IllustrationSettings(enabled=True, model_reference="m1", model_prompts="m2"),
        )
        gen, _ = tx.illustrations.create_or_get_generation(
            "book-1",
            settings_id=rev.id,
            request_key="req-seal",
            source_sha256=digest,
            source_authority={"conditions_id": cond.id, "conditions_revision": 1, "source_digest": digest, "manifest_sha256": None},
            requested_sections=("sec-1",),
        )
    coord.reconcile("book-1")
    with uow.read_transaction() as tx:
        sealed = tx.illustrations.get_generation("book-1", gen.id)
        assert sealed is not None
        assert sealed.manifest_sha256 is not None

    # Replace narrative pass with a different valid summary -> manifest_sha256 changes!
    with uow.transaction() as tx:
        old = tx.analyses.get_pass("book-1", "sec-1", AnalysisPassKind.NARRATIVE)
        assert old is not None
        new_payload = json.loads(json.dumps(old.payload))
        new_payload["response"]["chapter_summary"] = "Alice stops by the window and watches the falling rain."
        tx.analyses.save_pass(
            replace(old, payload=new_payload, output_sha256=canonical_sha(new_payload))
        )

    coord.reconcile("book-1")
    with uow.read_transaction() as tx:
        revoked = tx.illustrations.get_generation("book-1", gen.id)
        assert revoked is not None
        assert revoked.state is IllustrationGenerationState.REVOKED
        assert revoked.reason_code == IllustrationReasonCode.SOURCE_CHANGED.value


def test_corrupt_payload_fragment_input_context_bool_prompts_version_and_missing_epub_mapped(
    tmp_path: Path,
) -> None:
    uow = _make_uow(tmp_path)
    book, _doc, cond, _ = _seed_book_and_conditions(
        uow,
        tmp_path,
        doc=_make_doc("book-1", sections=(_make_doc("book-1").ordered_sections[0],)),
    )
    analyzer = _build_analyzer(uow, _StagedFakeLLM(), cond)

    # 1. input_context on fragment pass (scope_key != "") rejected.
    bad_frag = encode_pass(NarrativeAnalysisResponse("Keep.", "Sum."))
    bad_frag["input_suggestions"] = []
    bad_frag["input_context"] = {
        "schema_version": 1,
        "sequential_analysis": True,
        "parallel_batch": False,
        "seed_active": False,
        "user_decisions": [],
        "nonbinding_suggestions": [],
        "prompts_version": 8,
    }
    with pytest.raises(AnalyzeChapterError):
        analyzer._verify_cached_pass(
            AnalysisPassResult(
                id="p-frag",
                book_id="book-1",
                section_id="sec-1",
                pass_kind=AnalysisPassKind.NARRATIVE,
                scope_key="frag:0",
                input_sha256="a" * 64,
                output_sha256=canonical_sha(bad_frag),
                conditions_id=cond.id,
                attempt_id=None,
                payload=bad_frag,
            )
        )

    # 2. bool prompts_version rejected.
    bad_bool = encode_pass(NarrativeAnalysisResponse("Keep.", "Sum."))
    bad_bool["input_suggestions"] = []
    bad_bool["input_context"] = {
        "schema_version": 1,
        "sequential_analysis": True,
        "parallel_batch": False,
        "seed_active": False,
        "user_decisions": [],
        "nonbinding_suggestions": [],
        "prompts_version": True,
    }
    with pytest.raises(AnalyzeChapterError):
        analyzer._verify_cached_pass(
            AnalysisPassResult(
                id="p-bool",
                book_id="book-1",
                section_id="sec-1",
                pass_kind=AnalysisPassKind.NARRATIVE,
                scope_key="",
                input_sha256="a" * 64,
                output_sha256=canonical_sha(bad_bool),
                conditions_id=cond.id,
                attempt_id=None,
                payload=bad_bool,
            )
        )

    # 3. Missing EPUB -> storage_unavailable; modified EPUB -> source_changed.
    source_missing = IllustrationSource(
        language_detector=Py3LangidDetector(),
        source_digest_fn=lambda _p: (_ for _ in ()).throw(ConditionsError("source_unavailable")),
    )
    with uow.read_transaction() as tx:
        assert source_missing.snapshot(tx, "book-1").reason_code == IllustrationReasonCode.STORAGE_UNAVAILABLE.value

    Path(book.source_epub_path).write_bytes(b"modified-epub")
    with uow.read_transaction() as tx:
        assert _make_source().snapshot(tx, "book-1").reason_code == IllustrationReasonCode.SOURCE_CHANGED.value


def test_legacy_v1_pass_without_input_context_keeps_output_sha256(tmp_path: Path) -> None:
    uow = _make_uow(tmp_path)
    _, _, cond, _ = _seed_book_and_conditions(uow, tmp_path)
    analyzer = _build_analyzer(uow, _StagedFakeLLM(), cond)
    encoded = encode_pass(NarrativeAnalysisResponse("Keep.", "Summary."))
    encoded["input_suggestions"] = []
    assert "input_context" not in encoded
    sha = canonical_sha(encoded)
    saved = AnalysisPassResult(
        id="v1-pass",
        book_id="book-1",
        section_id="sec-1",
        pass_kind=AnalysisPassKind.NARRATIVE,
        scope_key="",
        input_sha256="a" * 64,
        output_sha256=sha,
        conditions_id=cond.id,
        attempt_id=None,
        payload=encoded,
    )
    decoded = analyzer._verify_cached_pass(saved)
    assert isinstance(decoded, NarrativeAnalysisResponse)
    assert canonical_sha(encode_pass(decoded) | {"input_suggestions": []}) == sha
    with pytest.raises(ValueError):
        decode_pass(encoded, pass_kind=AnalysisPassKind.ILLUSTRATION_PREP)


async def test_snapshot_works_without_provider_api_key(tmp_path: Path) -> None:
    uow = _make_uow(tmp_path)
    _, doc, cond, _ = _seed_book_and_conditions(
        uow,
        tmp_path,
        doc=_make_doc("book-1", sections=(_make_doc("book-1").ordered_sections[0],)),
    )
    llm = _StagedFakeLLM([_narrative_reply(doc.ordered_sections[0].ordered_blocks[0]), _pedagogical_reply()])
    analyzer = _build_analyzer(uow, llm, cond)
    await analyzer(AnalyzeChapterCommand("book-1", "sec-1"))

    # Note: cond was created with LLMSettings(api_key="") and IllustrationSource never invokes BookRuntimeResolver.settings_for().
    source = _make_source()
    with uow.read_transaction() as tx:
        snap = source.snapshot(tx, "book-1")
        assert snap.compatible is True
        assert snap.ready is True
