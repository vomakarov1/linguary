"""End-to-end Everlost Chapter 2 regressions for analysis validation, fragmentation, and prompts.

Exercises the staged chapter-analysis pipeline through real application composition, Alembic-migrated
SQLite storage, Pydantic schema parsing, `Py3LangidDetector`, and `RetryPolicy` with deterministic
fake model replies and zero network calls.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.config.settings import CheckSettings, ReadingSettings, WorkerPoolSettings
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application.book_conditions import (
    A2_CONTENT_PLAN_V5_VERSIONS,
    A2_CONTENT_PLAN_V6_VERSIONS,
    A2_CONTENT_PLAN_V7_VERSIONS,
    payload_digest,
)
from linguary.application.context_builder import (
    _TRANSLATION_INSTRUCTIONS,
    snapshot_prompt_templates,
    textual_blocks,
)
from linguary.application.ports.llm import LLMRequest, LLMResponse, LLMPayload
from linguary.application.ports.persistence import BookRecord
from linguary.application.reader_text import _A2_NOTE_DISCIPLINE
from linguary.application.use_cases.analyze_chapter import AnalyzeChapterCommand
from linguary.composition import Application, build_application, drain_tasks
from linguary.composition.book_runtime import effective_content_plan_profile
from linguary.domain import (
    AnalysisPassKind,
    BlockSegment,
    BlockTreatment,
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    ConceptType,
    DocumentBlock,
    DocumentSection,
    ErrorType,
    LLMCallStage,
    SourceReference,
    SupportStatus,
    TaskStatus,
    TermConcept,
    TermStatus,
    TermVariant,
)
from linguary.domain.book_conditions import thaw
from linguary.domain.name_translation import NameTranslationReviewReason

BOOK_ID = "book-1"
SECTION_ID = "section-1"
CREATED_AT = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)

_BASELINE_TRANSLATION_SHA256 = (
    "8c39d00a516d9db9a8a0079602c48bcfb6d9fc02714338d5d444be6304acbafa"
)
_BASELINE_WITH_PRON_SHA256 = (
    "3bcf968fbeb3eea72bc191df4360d50ec62f005d45a670b54dbd738ef2fb069a"
)
_BASELINE_DEFAULT_SHA256 = (
    "d6e4b0b6dfa1227f6ab86b34ce7d9cb05b465c4e2ba71a5f322dec1f9125d470"
)


class StagedFakeLLMClient(FakeLLMClient):
    """Fake LLM client supporting staged analysis and preserving raw reply text for retries."""

    supports_staged_analysis = True

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        step = self._next_step()
        response = await super().generate(request)
        if isinstance(step, FakeReply) and response.raw_content is None:
            return replace(response, raw_content=step.content)
        return response


def _everlost_ch2_section(
    *, document_id: str = BOOK_ID, section_id: str = SECTION_ID
) -> DocumentSection:
    """Everlost Chapter 2 profile: 121 short textual blocks, 2 PRESERVE_OPAQUE dividers, 2-block QUOTE."""
    special_texts: dict[int, tuple[BlockType, str]] = {
        0: (BlockType.HEADING, "Chapter 2"),
        1: (BlockType.HEADING, "Deadspots"),
        2: (
            BlockType.PARAGRAPH,
            "“Nick — pale and translucent — stared down at the cracked asphalt,” "
            "Allie whispered, afraid of slipping into the earth.",
        ),
        3: (
            BlockType.PARAGRAPH,
            "“I don’t think we are in New Jersey anymore,” Allie admitted as cold "
            "silver mist curled tightly around both of her boots.",
        ),
        4: (
            BlockType.PARAGRAPH,
            "“Stay strictly on the gray stone path,” warned the strange boy from "
            "the trees, pointing at a faint trail through the fog.",
        ),
        5: (
            BlockType.PARAGRAPH,
            "Lief tapped his wooden walking staff twice against the curb and "
            "glanced warily toward the dark forest behind the lane.",
        ),
        6: (
            BlockType.PARAGRAPH,
            "“Wait here while I check the edge of the clearing,” Nick said softly, "
            "testing the packed gravel beneath his ghostly sneakers.",
        ),
        14: (
            BlockType.PARAGRAPH,
            "A sunken streetlamp leaned crookedly above the overgrown ditch, its "
            "shattered glass glowing faintly with trapped afternoon twilight.",
        ),
        29: (
            BlockType.QUOTE,
            "From The Book of Everlost by Mary Hightower: “Stay on the beaten "
            "track where the living walked firmly before us long ago.”",
        ),
        30: (
            BlockType.QUOTE,
            "“Every deadspot is a silent trap waiting beneath the quiet grass, and "
            "only the mindful traveler crosses safely to the tower.”",
        ),
    }
    filler_lines = (
        "“Keep your eyes on the concrete curb ahead,” Allie called across the "
        "foggy clearing as silver leaves drifted past her shoulders.",
        "“Everything the living forget eventually settles here in the quiet "
        "shadows,” Lief explained while tapping a rusted road sign twice.",
        "Nick shifted his weight from one foot to the other, watching tiny "
        "ripples shimmer across the mossy stones near the forest trail.",
    )

    blocks: list[DocumentBlock] = []
    order = 0
    for t_idx in range(121):
        if t_idx in (96, 110):
            blocks.append(
                DocumentBlock.create(
                    document_id=document_id,
                    section_id=section_id,
                    type=BlockType.OTHER,
                    order=order,
                    text="* * *",
                    source_reference=SourceReference("ch2.xhtml", f"/body/hr[{order + 1}]"),
                    support_status=SupportStatus.PRESERVE_OPAQUE,
                )
            )
            order += 1
        if t_idx in special_texts:
            block_type, text = special_texts[t_idx]
            tag = "h" if block_type is BlockType.HEADING else "blockquote" if block_type is BlockType.QUOTE else "p"
        else:
            block_type = BlockType.PARAGRAPH
            text = f"{filler_lines[t_idx % len(filler_lines)]} [{t_idx + 1}]"
            tag = "p"
        blocks.append(
            DocumentBlock.create(
                document_id=document_id,
                section_id=section_id,
                type=block_type,
                order=order,
                text=text,
                source_reference=SourceReference("ch2.xhtml", f"/body/{tag}[{order + 1}]"),
                support_status=SupportStatus.SUPPORTED,
            )
        )
        order += 1

    return DocumentSection(
        id=section_id,
        document_id=document_id,
        title="Chapter 2: Deadspots",
        order=0,
        source_resource_path="ch2.xhtml",
        blocks=tuple(blocks),
    )


def _make_everlost_app(
    root: Path,
    llm_client: FakeLLMClient,
    *,
    max_fragment_blocks: int | None = None,
    max_attempts: int = 3,
    seed_book: bool = True,
) -> tuple[Application, CanonicalDocument]:
    """Assemble real application with Alembic SQLite and Everlost Chapter 2 stored."""
    settings = AppSettings(
        llm=LLMSettings(
            model="everlost-test",
            api_key="synthetic-secret",
            proxy_mode="direct",
            max_attempts=max_attempts,
            max_fragment_blocks=max_fragment_blocks,
        ),
        reading=ReadingSettings(reader_level="a2", name_translation_strategy="meaning"),
        storage=StorageSettings(data_root=str(root)),
    )
    app = build_application(
        settings=settings,
        llm_client=llm_client,
        content_plan_profile="content-plan-v3",
    )
    section = _everlost_ch2_section()
    document = CanonicalDocument(
        id=BOOK_ID,
        metadata=BookMetadata(title="Everlost", authors=("Neal Shusterman",)),
        source_language="en",
        target_language="ru",
        sections=(section,),
    )
    if seed_book:
        source = app.paths.source_epub(BOOK_ID)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"synthetic everlost epub")
        with app.unit_of_work.transaction() as tx:
            tx.books.add(
                BookRecord(
                    id=BOOK_ID,
                    metadata=document.metadata,
                    source_language="en",
                    target_language="ru",
                    status=BookStatus.IMPORTED,
                    source_epub_path=str(source),
                    created_at=CREATED_AT,
                )
            )
            tx.documents.save(document)
    return app, document


def _narrative_payload(
    *,
    block_count: int,
    chapter_summary: str,
    translation_instructions: str = "Сохранять разговорный тон диалогов.",
    scenes: list[dict[str, Any]] | None = None,
    context_facts: list[dict[str, Any]] | None = None,
    glossary_candidates: list[dict[str, Any]] | None = None,
    location: str = "Everlost Forest",
    participants: Sequence[str] = ("Nick", "Allie"),
    continues_previous_scene: bool = False,
) -> dict[str, Any]:
    return {
        "translation_instructions": translation_instructions,
        "chapter_summary": chapter_summary,
        "glossary_candidates": list(glossary_candidates or ()),
        "context_facts": list(context_facts or ()),
        "scenes": (
            list(scenes)
            if scenes is not None
            else [
                {
                    "start_block_id": "b1",
                    "end_block_id": f"b{block_count}",
                    "summary": f"Сцена, охватывающая {block_count} блоков.",
                    "location": location,
                    "participants": list(participants),
                    "visual_facts": [],
                    "visual_assumptions": [],
                    "continues_previous_scene": continues_previous_scene,
                }
            ]
        ),
    }


def _pedagogical_payload(
    *,
    block_count: int,
    summary: str,
    treatment: str = "TRANSLATE",
    difficulties: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "treatment_plan": [
            {"source_block_id": f"b{idx}", "treatment": treatment}
            for idx in range(1, block_count + 1)
        ],
        "before_you_read": [],
        "difficulties": list(difficulties or ()),
        "before_you_read_evidence": [],
        "reader_chapter_summary": summary,
    }


def _name_assessment(
    *,
    applied_method: str,
    opaque_name: bool,
    rationale: str,
) -> dict[str, Any]:
    return {
        "applied_method": applied_method,
        "eligible_without_review": True,
        "review_reasons": [],
        "rationale": rationale,
        "opaque_name": opaque_name,
    }


async def test_everlost_ch2_single_fragment_and_balanced_explicit_split(
    tmp_path: Path,
) -> None:
    """Everlost Ch2 fits one default fragment, balances explicit limits without 1-block tails or split quotes, and enforces prompt rules."""
    section = _everlost_ch2_section()
    tb = textual_blocks(section)
    assert len(section.ordered_blocks) == 123
    assert len(tb) == 121
    assert sum(1 for b in section.ordered_blocks if b.support_status is SupportStatus.PRESERVE_OPAQUE) == 2
    quote_blocks = tuple(b for b in tb if b.type is BlockType.QUOTE)
    assert len(quote_blocks) == 2
    assert quote_blocks[1].order == quote_blocks[0].order + 1
    assert 14_000 <= sum(len(b.text) for b in tb) <= 16_000

    # Frozen baseline B2 templates and _TRANSLATION_INSTRUCTIONS SHA-256 digests stay untouched.
    assert sha256(_TRANSLATION_INSTRUCTIONS.encode("utf-8")).hexdigest() == _BASELINE_TRANSLATION_SHA256
    baseline_with_pron = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b2",
        pronunciation_help_enabled=True,
    )
    assert payload_digest(baseline_with_pron) == _BASELINE_WITH_PRON_SHA256
    baseline_default = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b2",
    )
    assert payload_digest(baseline_default) == _BASELINE_DEFAULT_SHA256

    full_summary = (
        "Nick and Allie explore the misty forest of Everlost, meet Lief near the "
        "deadspots, and read Mary Hightower's warning before heading toward the towers."
    )
    candidates = [
        {
            "source_text": "Nick",
            "type": "person",
            "suggested_translation": "Ник",
            "confidence": 0.95,
            "model_flagged_review": False,
            "note": "мужской род",
            "description": "Мальчик, проснувшийся в Эверлосте вместе с Элли.",
            "name_translation_assessment": _name_assessment(
                applied_method="sound",
                opaque_name=True,
                rationale="Личное имя передано по звучанию.",
            ),
        },
        {
            "source_text": "Everlost",
            "type": "place",
            "suggested_translation": "Междумирье",
            "confidence": 0.9,
            "model_flagged_review": False,
            "note": None,
            "description": "Потусторонний мир между жизнью и смертью.",
            "name_translation_assessment": _name_assessment(
                applied_method="meaning",
                opaque_name=False,
                rationale="Составное название с прозрачными корнями ever и lost.",
            ),
        },
        {
            "source_text": "Mary Hightower",
            "type": "person",
            "suggested_translation": "Мэри Хайтауэр",
            "confidence": 0.9,
            "model_flagged_review": False,
            "note": "женский род",
            "description": "Автор книги наставлений о мире Эверлост.",
            "name_translation_assessment": _name_assessment(
                applied_method="mixed",
                opaque_name=False,
                rationale="Имя передано по звучанию, фамилия составная.",
            ),
        },
        {
            "source_text": "The Book of Everlost",
            "type": "fictional_concept",
            "suggested_translation": "Книга Междумирья",
            "confidence": 0.9,
            "model_flagged_review": False,
            "note": None,
            "description": "Вымышленный сборник правил выживания в Эверлосте.",
            "name_translation_assessment": None,
        },
    ]

    # 1. Default configuration (max_fragment_blocks=None): narrative splits 121 dialogue blocks into
    # 3 balanced fragments (41, 40, 40) to prevent 32K output truncation while passing intra-chapter
    # glossary candidates forward; pedagogical splits 121 blocks into 2 balanced fragments (61, 60).
    default_nar_sizes = (41, 40, 40)
    default_ped_sizes = (61, 60)
    default_client = StagedFakeLLMClient.replying(
        [
            FakeReply.of_payload(
                _narrative_payload(
                    block_count=default_nar_sizes[0],
                    chapter_summary=full_summary,
                    glossary_candidates=candidates[:2],
                )
            ),
            FakeReply.of_payload(
                _narrative_payload(
                    block_count=default_nar_sizes[1],
                    chapter_summary=full_summary,
                    continues_previous_scene=True,
                    glossary_candidates=candidates[2:3],
                )
            ),
            FakeReply.of_payload(
                _narrative_payload(
                    block_count=default_nar_sizes[2],
                    chapter_summary=full_summary,
                    continues_previous_scene=True,
                    glossary_candidates=candidates[3:],
                )
            ),
            FakeReply.of_payload(
                _pedagogical_payload(block_count=default_ped_sizes[0], summary=full_summary)
            ),
            FakeReply.of_payload(
                _pedagogical_payload(block_count=default_ped_sizes[1], summary=full_summary)
            ),
        ]
    )
    default_app, _ = _make_everlost_app(tmp_path / "default", default_client)
    try:
        await default_app.analysis.start(BOOK_ID)
        completed = await drain_tasks(default_app)
        assert all(task.status is TaskStatus.SUCCEEDED for task in completed), [
            task.last_error for task in completed
        ]
        assert [req.stage for req in default_client.requests] == [
            LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE,
            LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE,
            LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE,
            LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL,
            LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL,
        ]
        nar_req1, nar_req2, nar_req3, ped_req1, ped_req2 = default_client.requests
        nar_req = nar_req1
        ped_req = ped_req1
        assert [len(r.ordered_block_ids) for r in (nar_req1, nar_req2, nar_req3)] == list(default_nar_sizes)
        assert (*nar_req1.ordered_block_ids, *nar_req2.ordered_block_ids, *nar_req3.ordered_block_ids) == tuple(
            b.id for b in tb
        )
        assert [len(r.ordered_block_ids) for r in (ped_req1, ped_req2)] == list(default_ped_sizes)
        assert (*ped_req1.ordered_block_ids, *ped_req2.ordered_block_ids) == tuple(b.id for b in tb)
        # Adjacent QUOTE blocks stay together inside the same pedagogical fragment and note quota splits evenly.
        assert all(q.id in ped_req1.ordered_block_ids for q in quote_blocks)
        assert "не более 7" in ped_req1.input_text
        assert "не более 7" in ped_req2.input_text
        # Intra-chapter glossary continuity: fragment 1 candidates are passed to fragments 2 and 3.
        assert "Everlost" in nar_req2.input_text and "Междумирье" in nar_req2.input_text
        assert "Mary Hightower" in nar_req3.input_text
        # Narrative translation_instructions are forwarded into pedagogical input.
        assert "Указания переводчику:" in ped_req.input_text

        conditions = default_app.book_conditions.require(BOOK_ID)
        frozen_templates = conditions.payload["prompt_templates"]

        # Verify narrative prompt updates in live request and frozen templates.
        for nar_text in (nar_req.instructions, frozen_templates["narrative_analysis"]):
            assert "b1" in nar_text and "b2" in nar_text
            assert "первую сцену" in nar_text and "заголовки" in nar_text
            assert "торговые марки" in nar_text and "бытовые предметы" in nar_text
            assert "в скобках" in nar_text and "слэш" in nar_text and "лишних кавычек" in nar_text
            assert "вымышленные книги" in nar_text.lower() and "эпиграфов" in nar_text
            assert "Не дублируй" in nar_text and "переводы имён" in nar_text
            assert "start=null и end=null" in nar_text
            assert "subject_anchor" in nar_text and "дословн" in nar_text
            assert "opaque_name=true" in nar_text and "opaque_name=false" in nar_text
            for name in ("Nick", "Alexandra", "Everlost", "Hightower"):
                assert name in nar_text

        # Verify pedagogical A2 prompt has no duplicates, no HP examples, and targets full chapter summary.
        note_discipline = _A2_NOTE_DISCIPLINE.strip()
        assert ped_req.instructions.count(note_discipline) == 1
        assert ped_req.instructions.count("Различай роль заметок в GLOSSARY_ONLY и TRANSLATE") == 1
        assert ped_req.instructions.count("Для A2 без english help верни before_you_read: []") == 1
        assert ped_req.instructions.count("might have been") == 1
        for ped_text in (ped_req.instructions, frozen_templates["pedagogical_analysis"]):
            assert "Принятый повествовательный анализ" in ped_text
            assert "всей главы" in ped_text and "3–5 коротких" in ped_text
            for hp_phrase in (
                "The Boy Who Lived",
                "The Vanishing Glass",
                "The Letters from No One",
                "sweetums",
                "Fancy seeing you here",
                "not saying the spell isn’t brilliant",
            ):
                assert hp_phrase not in ped_text

        with default_app.unit_of_work.transaction() as tx:
            analysis, = tx.analyses.list_for_book(BOOK_ID)
            assert analysis.scenes[0].start_block_id == tb[0].id
            assert analysis.scenes[-1].end_block_id == tb[-1].id
            assert analysis.treatment_plan.block_ids == tuple(b.id for b in tb)
            by_source = {c.source_text: c for c in analysis.glossary_candidates}
            assert by_source["Nick"].name_translation_evidence is not None
            assert by_source["Nick"].name_translation_evidence.observations[0].assessment.opaque_name is True
            assert by_source["Everlost"].name_translation_evidence.observations[0].assessment.opaque_name is False
            assert "The Book of Everlost" in by_source and "Mary Hightower" in by_source
    finally:
        await default_app.aclose()

    # 2. Explicit max_fragment_blocks=40 balances into 4 fragments without 1-block tail and keeps QUOTE intact.
    explicit_sizes = (31, 30, 35, 25)
    explicit_client = StagedFakeLLMClient.replying(
        [
            *[
                FakeReply.of_payload(
                    _narrative_payload(
                        block_count=size,
                        chapter_summary=full_summary,
                        continues_previous_scene=idx > 0,
                    )
                )
                for idx, size in enumerate(explicit_sizes)
            ],
            *[
                FakeReply.of_payload(_pedagogical_payload(block_count=size, summary=full_summary))
                for size in explicit_sizes
            ],
        ]
    )
    explicit_app, _ = _make_everlost_app(
        tmp_path / "explicit", explicit_client, max_fragment_blocks=40
    )
    try:
        await explicit_app.analysis.start(BOOK_ID)
        completed = await drain_tasks(explicit_app)
        assert all(task.status is TaskStatus.SUCCEEDED for task in completed), [
            task.last_error for task in completed
        ]
        nar_requests = [
            req for req in explicit_client.requests if req.stage is LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE
        ]
        ped_requests = [
            req for req in explicit_client.requests if req.stage is LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL
        ]
        frag_sizes = [len(req.ordered_block_ids) for req in nar_requests]
        assert frag_sizes == list(explicit_sizes)
        assert [len(req.ordered_block_ids) for req in ped_requests] == list(explicit_sizes)
        assert all(20 <= size <= 40 for size in frag_sizes)
        assert sum(frag_sizes) == 121

        # 2-block QUOTE excerpt stays intact within a single fragment.
        quote_ids = {b.id for b in quote_blocks}
        quote_frag_indices = {
            idx
            for idx, req in enumerate(nar_requests)
            if quote_ids & set(req.ordered_block_ids)
        }
        assert len(quote_frag_indices) == 1
        quote_frag = nar_requests[next(iter(quote_frag_indices))]
        assert quote_ids.issubset(set(quote_frag.ordered_block_ids))

        with explicit_app.unit_of_work.transaction() as tx:
            analysis, = tx.analyses.list_for_book(BOOK_ID)
            assert analysis.treatment_plan.block_ids == tuple(b.id for b in tb)
            assert analysis.scenes[0].start_block_id == tb[0].id
            assert analysis.scenes[-1].end_block_id == tb[-1].id
    finally:
        await explicit_app.aclose()

    # 3. A 60-block chapter stays in a single pedagogical call under default 32K budget.
    short_tb = tb[:60]
    short_section = replace(section, blocks=short_tb)
    short_nar_sizes = (31, 29)
    short_client = StagedFakeLLMClient.replying(
        [
            FakeReply.of_payload(
                _narrative_payload(block_count=short_nar_sizes[0], chapter_summary=full_summary)
            ),
            FakeReply.of_payload(
                _narrative_payload(
                    block_count=short_nar_sizes[1],
                    chapter_summary=full_summary,
                    continues_previous_scene=True,
                )
            ),
            FakeReply.of_payload(_pedagogical_payload(block_count=60, summary=full_summary)),
        ]
    )
    short_app, _ = _make_everlost_app(tmp_path / "short_60", short_client, seed_book=False)
    try:
        short_doc = CanonicalDocument(
            id=BOOK_ID,
            metadata=BookMetadata(title="Everlost", authors=("Neal Shusterman",)),
            source_language="en",
            target_language="ru",
            sections=(short_section,),
        )
        source = short_app.paths.source_epub(BOOK_ID)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"synthetic everlost epub")
        with short_app.unit_of_work.transaction() as tx:
            tx.books.add(
                BookRecord(
                    id=BOOK_ID,
                    metadata=short_doc.metadata,
                    source_language="en",
                    target_language="ru",
                    status=BookStatus.IMPORTED,
                    source_epub_path=str(source),
                    created_at=CREATED_AT,
                )
            )
            tx.documents.save(short_doc)
        await short_app.analysis.start(BOOK_ID)
        completed = await drain_tasks(short_app)
        assert all(task.status is TaskStatus.SUCCEEDED for task in completed), [
            task.last_error for task in completed
        ]
        ped_requests = [
            req for req in short_client.requests if req.stage is LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL
        ]
        assert len(ped_requests) == 1
        assert ped_requests[0].ordered_block_ids == tuple(b.id for b in short_tb)
    finally:
        await short_app.aclose()


async def test_everlost_a2_summary_and_validation_retry_end_to_end(
    tmp_path: Path,
) -> None:
    """Preserves full A2 summary on short tail fragments and exercises aggregated validation + retry end-to-end."""
    frag_sizes = (31, 30, 35, 25)
    full_narrative = (
        "Nick and Allie wake up in Everlost, meet Lief in the misty forest, cross "
        "dangerous deadspots, and walk toward the Twin Towers where Mary Hightower lives."
    )
    full_a2_summary = (
        "Nick and Allie wake up in Everlost, meet Lief in the misty forest, avoid "
        "deadspots, and walk to the Twin Towers where Mary Hightower lives."
    )
    short_tail_summaries = (
        "Allie and Nick follow Lief along the narrow stone path.",
        "They pass dark trees and read a warning about deadspots.",
        "Nick and Allie walk slowly toward the tall Twin Towers.",
    )

    # Fragment 0 narrative attempt 1 triggers aggregated validation in one pass:
    # - scene_bounds gap omitting b15
    # - valid dash-space-tolerant visual_fact + broken visual_fact on b3 ('missing lantern')
    # - broken context_fact on b5 ('missing compass')
    # - attribution conflict on b4 ('Allie': female vs male)
    # - glossary conflict on b6 ('Lief': female vs approved мужской род)
    nar_frag0_attempt1 = _narrative_payload(
        block_count=frag_sizes[0],
        chapter_summary="Nick and Allie enter the forest.",
        scenes=[
            {
                "start_block_id": "b1",
                "end_block_id": "b14",
                "summary": "Прибытие.",
                "location": "Forest",
                "participants": ["Nick", "Allie"],
                "visual_facts": [
                    {
                        "anchor": {"block_id": "b3", "quote": "Nick—pale and translucent"},
                        "subject": "Nick",
                        "detail": "Translucent appearance.",
                    },
                    {
                        "anchor": {"block_id": "b3", "quote": "missing lantern"},
                        "subject": "Allie",
                        "detail": "Lantern.",
                    },
                ],
                "visual_assumptions": [],
            },
            {
                "start_block_id": "b16",
                "end_block_id": f"b{frag_sizes[0]}",
                "summary": "Тропа.",
                "location": "Forest",
                "participants": ["Nick", "Allie", "Lief"],
                "visual_facts": [],
                "visual_assumptions": [],
            },
        ],
        context_facts=[
            {
                "id": "f-bad-cf",
                "kind": "speaker",
                "anchors": [{"block_id": "b5", "quote": "missing compass"}],
                "subject_anchor": {"block_id": "b5", "quote": "missing compass"},
                "value": "Lief",
                "certainty": "explicit",
            },
            {
                "id": "f-conflict-1",
                "kind": "gender",
                "anchors": [{"block_id": "b4", "quote": "Allie"}],
                "subject_anchor": {"block_id": "b4", "quote": "Allie"},
                "value": "female",
                "certainty": "explicit",
            },
            {
                "id": "f-conflict-2",
                "kind": "gender",
                "anchors": [{"block_id": "b4", "quote": "Allie"}],
                "subject_anchor": {"block_id": "b4", "quote": "Allie"},
                "value": "male",
                "certainty": "explicit",
            },
            {
                "id": "f-glossary",
                "kind": "gender",
                "anchors": [{"block_id": "b6", "quote": "Lief"}],
                "subject_anchor": {"block_id": "b6", "quote": "Lief"},
                "value": "female",
                "certainty": "explicit",
            },
        ],
    )

    # Attempt 2 fixes scene_bounds and conflicts, leaving one broken visual_fact on b7
    # to verify analysis-specific unresolved_quote repair without translation template.
    nar_frag0_attempt2 = _narrative_payload(
        block_count=frag_sizes[0],
        chapter_summary="Nick and Allie enter the forest.",
        scenes=[
            {
                "start_block_id": "b1",
                "end_block_id": f"b{frag_sizes[0]}",
                "summary": "Прибытие в лес.",
                "location": "Forest",
                "participants": ["Nick", "Allie", "Lief"],
                "visual_facts": [
                    {
                        "anchor": {"block_id": "b7", "quote": "phantom whistle"},
                        "subject": "Nick",
                        "detail": "Whistle.",
                    }
                ],
                "visual_assumptions": [],
            }
        ],
    )

    # Attempt 3 succeeds with dash-space-tolerant visual_fact matching 'Nick — pale and translucent'.
    nar_frag0_attempt3 = _narrative_payload(
        block_count=frag_sizes[0],
        chapter_summary="Nick and Allie enter the forest.",
        scenes=[
            {
                "start_block_id": "b1",
                "end_block_id": f"b{frag_sizes[0]}",
                "summary": "Прибытие в лес.",
                "location": "Forest",
                "participants": ["Nick", "Allie", "Lief"],
                "visual_facts": [
                    {
                        "anchor": {"block_id": "b3", "quote": "Nick—pale and translucent"},
                        "subject": "Nick",
                        "detail": "Translucent appearance.",
                    }
                ],
                "visual_assumptions": [],
            }
        ],
    )

    ped_frag0_bad_schema = _pedagogical_payload(block_count=frag_sizes[0], summary="")

    client = StagedFakeLLMClient.replying(
        [
            FakeReply.of_payload(nar_frag0_attempt1),
            FakeReply.of_payload(nar_frag0_attempt2),
            FakeReply.of_payload(nar_frag0_attempt3),
            FakeReply.of_payload(
                _narrative_payload(
                    block_count=frag_sizes[1],
                    chapter_summary="They meet Lief and learn about deadspots.",
                    continues_previous_scene=True,
                )
            ),
            FakeReply.of_payload(
                _narrative_payload(
                    block_count=frag_sizes[2],
                    chapter_summary="They cross the forest path carefully.",
                    continues_previous_scene=True,
                )
            ),
            FakeReply.of_payload(
                _narrative_payload(
                    block_count=frag_sizes[3],
                    chapter_summary=full_narrative,
                    continues_previous_scene=False,
                )
            ),
            FakeReply(content=json.dumps(ped_frag0_bad_schema, ensure_ascii=False)),
            FakeReply.of_payload(
                _pedagogical_payload(block_count=frag_sizes[0], summary=full_a2_summary)
            ),
            FakeReply.of_payload(
                _pedagogical_payload(block_count=frag_sizes[1], summary=short_tail_summaries[0])
            ),
            FakeReply.of_payload(
                _pedagogical_payload(block_count=frag_sizes[2], summary=short_tail_summaries[1])
            ),
            FakeReply.of_payload(
                _pedagogical_payload(block_count=frag_sizes[3], summary=short_tail_summaries[2])
            ),
        ]
    )

    data_root = tmp_path / "data"
    app, document = _make_everlost_app(
        data_root, client, max_fragment_blocks=40, max_attempts=3
    )
    try:
        section_blocks = textual_blocks(document.ordered_sections[0])
        with app.unit_of_work.transaction() as tx:
            tx.glossary.save(
                TermConcept(
                    id="approved-lief",
                    book_id=BOOK_ID,
                    type=ConceptType.PERSON,
                    canonical_source_text="Lief",
                    status=TermStatus.APPROVED,
                    note="мужской род",
                    variants=(
                        TermVariant("approved-lief", "ru", "Лиф", TermStatus.APPROVED),
                    ),
                )
            )

        await app.analysis.start(BOOK_ID)
        completed = await drain_tasks(app)
        assert all(task.status is TaskStatus.SUCCEEDED for task in completed), [
            task.last_error for task in completed
        ]

        nar_requests = [
            req for req in client.requests if req.stage is LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE
        ]
        ped_requests = [
            req for req in client.requests if req.stage is LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL
        ]
        assert len(nar_requests) == 6  # 3 attempts on frag 0 + 1 each on frags 1..3
        assert len(ped_requests) == 5  # 2 attempts on frag 0 + 1 each on frags 1..3

        # Narrative retry 1: aggregated violations + scene_bounds repair without translation template.
        repair_1 = nar_requests[1].repair_instructions[-1]
        assert "полный массив всех блоков" not in repair_1
        assert f"b1..b{frag_sizes[0]}" in repair_1
        assert "пропущены блоки: b15" in repair_1
        assert "b3: «missing lantern»" in repair_1
        assert "b5: «missing compass»" in repair_1
        assert "gender@b4('Allie'): male/explicit vs female/explicit" in repair_1
        assert "b6: «Lief»" in repair_1
        assert "Nick—pale and translucent" not in repair_1
        for block in section_blocks:
            assert block.id not in repair_1

        # Narrative retry 2: unresolved_quote analysis repair targets only affected block b7.
        repair_2 = nar_requests[2].repair_instructions[-1]
        assert "полный массив всех блоков" not in repair_2
        assert "Исправь затронутые блоки и верни корректный JSON ответа анализа без изменения остальных блоков:" in repair_2
        assert "- b7" in repair_2
        assert "phantom whistle" in repair_2
        for untouched in ("- b1", "- b2", "- b3", "- b4", "- b5", "- b6"):
            assert untouched not in repair_2

        # Pedagogical retry 1: schema validation failure includes exact field path string_too_short.
        ped_repair = ped_requests[1].repair_instructions[-1]
        assert "полный массив всех блоков" not in ped_repair
        assert "reader_chapter_summary: string_too_short" in ped_repair

        # Canonical full A2 summary is propagated across subsequent pedagogical fragment requests.
        for follow_req in ped_requests[2:]:
            assert f"Пересказ: {full_a2_summary}" in follow_req.input_text
            for partial in short_tail_summaries:
                assert partial not in follow_req.input_text

        with app.unit_of_work.transaction() as tx:
            calls = tx.llm_calls.list_for_book(BOOK_ID)
            schema_failures = [
                c
                for c in calls
                if c.stage is LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL and not c.success
            ]
            assert len(schema_failures) == 1
            assert schema_failures[0].error is not None
            assert schema_failures[0].error.type is ErrorType.MODEL_OUTPUT
            assert "reader_chapter_summary: string_too_short" in schema_failures[0].error.message

            nar_call_1 = calls[0]
            nar_diag_1 = tx.llm_calls.diagnostic_record(call_id=nar_call_1.id)
            assert nar_diag_1["metadata"]["decision"] == "rejected"

            analysis, = tx.analyses.list_for_book(BOOK_ID)
            assert analysis.chapter_summary == full_a2_summary
            # Dash-space-tolerant visual_fact resolved to exact source span on b3.
            assert [vf.anchor.quote for vf in analysis.scenes[0].visual_facts] == [
                "Nick — pale and translucent"
            ]
            assert analysis.scenes[0].visual_facts[0].anchor.block_id == section_blocks[2].id

            saved_ped = tx.analyses.get_pass(BOOK_ID, SECTION_ID, AnalysisPassKind.PEDAGOGICAL)
            assert saved_ped is not None
            assert saved_ped.payload["response"]["reader_chapter_summary"] == full_a2_summary
    finally:
        await app.aclose()

    # Restart application with empty LLM client: cached passes preserve full A2 summary with 0 LLM calls.
    zero_call_client = StagedFakeLLMClient()
    restarted_app, _ = _make_everlost_app(
        data_root, zero_call_client, max_fragment_blocks=40, seed_book=False
    )
    try:
        conditions = restarted_app.book_conditions.require(BOOK_ID)
        async with restarted_app.book_runtime.lease(conditions.id, "analysis") as generation:
            cached = await generation.analysis._analyze_chapter(
                AnalyzeChapterCommand(BOOK_ID, SECTION_ID)
            )
        assert cached.attempts == 0
        assert zero_call_client.call_count == 0
        assert cached.analysis.chapter_summary == full_a2_summary
        with restarted_app.unit_of_work.transaction() as tx:
            cached_analysis, = tx.analyses.list_for_book(BOOK_ID)
            assert cached_analysis.chapter_summary == full_a2_summary
    finally:
        await restarted_app.aclose()


async def test_everlost_scene_location_separator_and_instruction_merge(
    tmp_path: Path,
) -> None:
    """Preserves chained location transitions without tail duplication, respects PRESERVE_OPAQUE separators, and deduplicates instructions semantically."""
    # max_fragment_blocks=30 splits 121 textual blocks into [24, 24, 24, 24, 25]:
    # boundaries 0->1, 1->2, 2->3 have no separator; boundary 3->4 crosses PRESERVE_OPAQUE #1 (order=96).
    frag_sizes = (24, 24, 24, 24, 25)
    summary = "Nick and Allie cross Everlost toward the Twin Towers."

    frag_instructions = (
        "\n".join(
            [
                "- Сохранять иронию рассказчика в диалогах.",
                "- Обращаться к спутнику на ты в диалогах главы.",
                "- Не сглаживать грубую разговорную речь персонажей.",
                "- Передавать ироничный тон реплик Nick в диалогах.",
                "- Передавать восклицание «Deadspot!» естественной русской репликой.",
            ]
        ),
        "\n".join(
            [
                "- В диалогах сохраняйте иронию рассказчика.",
                "- Обращаться к спутнику на вы в диалогах главы.",
                "- Сглаживать грубую разговорную речь персонажей.",
            ]
        ),
        "\n".join(
            [
                "- Передавать ироничный тон реплик Allie в диалогах.",
                "- Передавать восклицание «Wake up!» естественной русской репликой.",
            ]
        ),
        "- Сохранять лёгкую иронию рассказчика в диалогах без потери интонации.",
        "- Передавать живой тон писем без упрощения.",
    )
    frag_locations = (
        ("Everlost Forest", ("Nick", "Allie"), False),
        ("Deadspot Road", ("Nick", "Allie", "Lief"), True),
        ("  deadspot road  ", ("Nick", "Allie", "Lief"), True),
        ("Twin Towers Lobby", ("Nick", "Allie", "Mary"), True),
        ("Twin Towers Lobby", ("Nick", "Allie", "Mary"), True),
    )

    client = StagedFakeLLMClient.replying(
        [
            *[
                FakeReply.of_payload(
                    _narrative_payload(
                        block_count=size,
                        chapter_summary=summary,
                        translation_instructions=frag_instructions[idx],
                        location=frag_locations[idx][0],
                        participants=frag_locations[idx][1],
                        continues_previous_scene=frag_locations[idx][2],
                    )
                )
                for idx, size in enumerate(frag_sizes)
            ],
            *[
                FakeReply.of_payload(_pedagogical_payload(block_count=size, summary=summary))
                for size in frag_sizes
            ],
        ]
    )

    app, document = _make_everlost_app(tmp_path / "merge", client, max_fragment_blocks=30)
    try:
        await app.analysis.start(BOOK_ID)
        completed = await drain_tasks(app)
        assert all(task.status is TaskStatus.SUCCEEDED for task in completed), [
            task.last_error for task in completed
        ]

        tb = textual_blocks(document.ordered_sections[0])
        assert len(client.requests) == 10

        with app.unit_of_work.transaction() as tx:
            analysis, = tx.analyses.list_for_book(BOOK_ID)

        # Boundary 0->1->2->3 merges into Scene 0 with deduplicated chain location;
        # PRESERVE_OPAQUE divider #1 blocks merge into Scene 1 despite continues_previous_scene=True.
        assert len(analysis.scenes) == 2
        scene_0, scene_1 = analysis.scenes
        assert (scene_0.start_block_id, scene_0.end_block_id) == (tb[0].id, tb[95].id)
        assert (scene_1.start_block_id, scene_1.end_block_id) == (tb[96].id, tb[-1].id)
        assert scene_0.location == "Everlost Forest → Deadspot Road → Twin Towers Lobby"
        assert scene_0.participants == ("Nick", "Allie", "Lief", "Mary")
        assert scene_1.location == "Twin Towers Lobby"

        # Semantic deduplication keeps fuller phrasing in-place and preserves guarded distinctions.
        assert analysis.translation_instructions == "\n".join(
            [
                "1. Сохранять лёгкую иронию рассказчика в диалогах без потери интонации.",
                "2. Обращаться к спутнику на ты в диалогах главы.",
                "3. Не сглаживать грубую разговорную речь персонажей.",
                "4. Передавать ироничный тон реплик Nick в диалогах.",
                "5. Передавать восклицание «Deadspot!» естественной русской репликой.",
                "6. Обращаться к спутнику на вы в диалогах главы.",
                "7. Сглаживать грубую разговорную речь персонажей.",
                "8. Передавать ироничный тон реплик Allie в диалогах.",
                "9. Передавать восклицание «Wake up!» естественной русской репликой.",
                "10. Передавать живой тон писем без упрощения.",
            ]
        )
    finally:
        await app.aclose()


def _sentence_segments(sentences: Sequence[str]) -> tuple[str, tuple[BlockSegment, ...]]:
    """Build continuous sentence-aligned BlockSegment spans for a multi-sentence paragraph."""
    chunks: list[str] = []
    spans: list[BlockSegment] = []
    offset = 0
    for idx, sentence in enumerate(sentences):
        chunk = sentence if idx == len(sentences) - 1 else f"{sentence} "
        chunks.append(chunk)
        spans.append(BlockSegment(offset, offset + len(chunk), sentence_index=idx))
        offset += len(chunk)
    return "".join(chunks), tuple(spans)


def _everlost_multi_section_document(*, book_id: str = BOOK_ID) -> CanonicalDocument:
    """Four-section Everlost fixture: PART ONE / Afterlights -> Ch. 1 -> Ch. 2 -> Ch. 3."""
    part_one = DocumentSection(
        id="part-1",
        document_id=book_id,
        title="PART ONE: Afterlights",
        order=0,
        source_resource_path="part1.xhtml",
        blocks=(
            DocumentBlock.create(
                document_id=book_id,
                section_id="part-1",
                type=BlockType.HEADING,
                order=0,
                text="PART ONE",
                source_reference=SourceReference("part1.xhtml", "/body/h1[1]"),
            ),
            DocumentBlock.create(
                document_id=book_id,
                section_id="part-1",
                type=BlockType.HEADING,
                order=1,
                text="Afterlights",
                source_reference=SourceReference("part1.xhtml", "/body/h2[1]"),
            ),
        ),
    )

    ch1_long_text, ch1_long_spans = _sentence_segments(
        (
            "The road looked quiet.",
            "A strange silver mist moved slowly across the street.",
            "Allie felt uneasy near the trees.",
            "She wondered what had happened to the highway after the crash.",
        )
    )
    ch1 = DocumentSection(
        id="ch-1",
        document_id=book_id,
        title="Chapter 1: Into the Light",
        order=1,
        source_resource_path="ch1.xhtml",
        blocks=(
            DocumentBlock.create(
                document_id=book_id,
                section_id="ch-1",
                type=BlockType.HEADING,
                order=0,
                text="Chapter 1",
                source_reference=SourceReference("ch1.xhtml", "/body/h1[1]"),
            ),
            DocumentBlock.create(
                document_id=book_id,
                section_id="ch-1",
                type=BlockType.HEADING,
                order=1,
                text="Into the Light",
                source_reference=SourceReference("ch1.xhtml", "/body/h2[1]"),
            ),
            DocumentBlock.create(
                document_id=book_id,
                section_id="ch-1",
                type=BlockType.PARAGRAPH,
                order=2,
                text="A girl woke up beside the wrecked silver car and looked around in silence.",
                source_reference=SourceReference("ch1.xhtml", "/body/p[1]"),
            ),
            DocumentBlock.create(
                document_id=book_id,
                section_id="ch-1",
                type=BlockType.PARAGRAPH,
                order=3,
                text="“I have never seen mist like this before,” Alexandra whispered, shaking her cold hands.",
                source_reference=SourceReference("ch1.xhtml", "/body/p[2]"),
            ),
            DocumentBlock.create(
                document_id=book_id,
                section_id="ch-1",
                type=BlockType.PARAGRAPH,
                order=4,
                text="“Call me Allie,” she told the boy who sat nearby on the curb.",
                source_reference=SourceReference("ch1.xhtml", "/body/p[3]"),
            ),
            DocumentBlock.create(
                document_id=book_id,
                section_id="ch-1",
                type=BlockType.PARAGRAPH,
                order=5,
                text=ch1_long_text,
                source_reference=SourceReference("ch1.xhtml", "/body/p[4]"),
                segments=ch1_long_spans,
            ),
        ),
    )

    ch2 = DocumentSection(
        id="ch-2",
        document_id=book_id,
        title="Chapter 2: The Boy Named Lief",
        order=2,
        source_resource_path="ch2.xhtml",
        blocks=(
            DocumentBlock.create(
                document_id=book_id,
                section_id="ch-2",
                type=BlockType.HEADING,
                order=0,
                text="Chapter 2",
                source_reference=SourceReference("ch2.xhtml", "/body/h1[1]"),
            ),
            DocumentBlock.create(
                document_id=book_id,
                section_id="ch-2",
                type=BlockType.HEADING,
                order=1,
                text="The Boy Named Lief",
                source_reference=SourceReference("ch2.xhtml", "/body/h2[1]"),
            ),
            DocumentBlock.create(
                document_id=book_id,
                section_id="ch-2",
                type=BlockType.PARAGRAPH,
                order=2,
                text="The forest felt cold and quiet as the two children walked along the mossy stones.",
                source_reference=SourceReference("ch2.xhtml", "/body/p[1]"),
            ),
            DocumentBlock.create(
                document_id=book_id,
                section_id="ch-2",
                type=BlockType.PARAGRAPH,
                order=3,
                text="Allie stopped near a fallen tree and stared at a boy dressed in green.",
                source_reference=SourceReference("ch2.xhtml", "/body/p[2]"),
            ),
            DocumentBlock.create(
                document_id=book_id,
                section_id="ch-2",
                type=BlockType.PARAGRAPH,
                order=4,
                text="“My name is Lief, like a green leaf on a tree,” the boy said softly.",
                source_reference=SourceReference("ch2.xhtml", "/body/p[3]"),
            ),
            DocumentBlock.create(
                document_id=book_id,
                section_id="ch-2",
                type=BlockType.PARAGRAPH,
                order=5,
                text="“What are you waiting for?” he asked, and Allie turned pale.",
                source_reference=SourceReference("ch2.xhtml", "/body/p[4]"),
            ),
        ),
    )

    ch3 = DocumentSection(
        id="ch-3",
        document_id=book_id,
        title="Chapter 3: Shadows on the Path",
        order=3,
        source_resource_path="ch3.xhtml",
        blocks=(
            DocumentBlock.create(
                document_id=book_id,
                section_id="ch-3",
                type=BlockType.HEADING,
                order=0,
                text="Chapter 3",
                source_reference=SourceReference("ch3.xhtml", "/body/h1[1]"),
            ),
            DocumentBlock.create(
                document_id=book_id,
                section_id="ch-3",
                type=BlockType.HEADING,
                order=1,
                text="Shadows on the Path",
                source_reference=SourceReference("ch3.xhtml", "/body/h2[1]"),
            ),
            DocumentBlock.create(
                document_id=book_id,
                section_id="ch-3",
                type=BlockType.PARAGRAPH,
                order=2,
                text="Allie followed the boy down the misty trail toward the sunken streetlamp.",
                source_reference=SourceReference("ch3.xhtml", "/body/p[1]"),
            ),
            DocumentBlock.create(
                document_id=book_id,
                section_id="ch-3",
                type=BlockType.PARAGRAPH,
                order=3,
                text="“I ain't going back,” the boy said.",
                source_reference=SourceReference("ch3.xhtml", "/body/p[2]"),
            ),
        ),
    )

    return CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title="Everlost", authors=("Neal Shusterman",)),
        source_language="en",
        target_language="ru",
        sections=(part_one, ch1, ch2, ch3),
    )


class ScopedStagedFakeLLMClient(FakeLLMClient):
    """Staged fake LLM client dispatching deterministic replies by `(scope_id, stage)`."""

    supports_staged_analysis = True

    def __init__(self, replies_by_key: dict[tuple[str | None, LLMCallStage], list[FakeReply]]) -> None:
        super().__init__(replies=[])
        self._queues = {key: list(items) for key, items in replies_by_key.items()}

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        key = (request.scope_id, request.stage)
        queue = self._queues.get(key)
        if not queue:
            raise AssertionError(f"Unexpected LLM call for {key!r}")
        step = queue.pop(0)
        self.replies = [*self.replies, step]
        response = await super().generate(request)
        if isinstance(step, FakeReply) and response.raw_content is None:
            return replace(response, raw_content=step.content)
        return response


def _make_multi_section_app(
    root: Path,
    llm_client: FakeLLMClient,
    document: CanonicalDocument,
    *,
    concurrency: int = 1,
    seed_first_chapter: bool = False,
    sequential_analysis: bool = True,
    max_attempts: int = 3,
) -> Application:
    """Build real SQLite-backed application populated with the multi-section Everlost document."""
    settings = AppSettings(
        llm=LLMSettings(
            model="everlost-test",
            api_key="synthetic-secret",
            proxy_mode="direct",
            max_attempts=max_attempts,
        ),
        checks=CheckSettings(
            seed_first_chapter=seed_first_chapter,
            sequential_analysis=sequential_analysis,
        ),
        worker=WorkerPoolSettings(concurrency=concurrency),
        reading=ReadingSettings(reader_level="a2", name_translation_strategy="meaning"),
        storage=StorageSettings(data_root=str(root)),
    )
    app = build_application(
        settings=settings,
        llm_client=llm_client,
        content_plan_profile="content-plan-v3",
    )
    source = app.paths.source_epub(document.id)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"synthetic everlost multi-section epub")
    with app.unit_of_work.transaction() as tx:
        tx.books.add(
            BookRecord(
                id=document.id,
                metadata=document.metadata,
                source_language=document.source_language,
                target_language=document.target_language,
                status=BookStatus.IMPORTED,
                source_epub_path=str(source),
                created_at=CREATED_AT,
            )
        )
        tx.documents.save(document)
    return app


async def test_everlost_part_divider_cross_chapter_glossary_segments_and_a2_validations_end_to_end(
    tmp_path: Path,
) -> None:
    """End-to-end PART ONE -> Ch. 1-3 regression covering dividers, cross-chapter glossary, segments, A2 guards, and conditions."""
    document = _everlost_multi_section_document()
    part_one, ch1, ch2, ch3 = document.ordered_sections
    assert document.is_part_divider(part_one) is True
    assert document.is_front_matter(part_one) is True
    assert document.has_meaningful_part_title(part_one) is True

    # Frozen baseline templates (legacy-v1 / B2) and _TRANSLATION_INSTRUCTIONS preserve exact SHA-256 digests.
    assert sha256(_TRANSLATION_INSTRUCTIONS.encode("utf-8")).hexdigest() == _BASELINE_TRANSLATION_SHA256
    assert (
        payload_digest(
            snapshot_prompt_templates(
                source_language="en",
                target_language="ru",
                reader_level="b2",
                pronunciation_help_enabled=True,
            )
        )
        == _BASELINE_WITH_PRON_SHA256
    )
    assert (
        payload_digest(
            snapshot_prompt_templates(
                source_language="en",
                target_language="ru",
                reader_level="b2",
            )
        )
        == _BASELINE_DEFAULT_SHA256
    )

    afterlights_candidate = {
        "source_text": "Afterlights",
        "type": "fictional_concept",
        "suggested_translation": "Послесветы",
        "confidence": 0.92,
        "model_flagged_review": False,
        "note": "множественное число",
        "description": "Дети, застрявшие между жизнью и смертью.",
        "name_translation_assessment": _name_assessment(
            applied_method="meaning",
            opaque_name=False,
            rationale="Прозрачное составное название after + lights.",
        ),
    }
    part1_narrative_reply = FakeReply.of_payload(
        _narrative_payload(
            block_count=2,
            chapter_summary="PART ONE: Afterlights",
            scenes=[],
            glossary_candidates=[afterlights_candidate],
        )
    )

    # Ch. 1 narrative attempt 1 triggers all five narrative validations in one pass:
    # - chapter_summary in Russian (narrative_summary_language)
    # - scenes[].summary in English (scene_summary_language)
    # - conflicting assessment: sound + creative_adaptation (name_assessment_conflict)
    # - spoiler participant "Alexandra" on scene b1..b3 before first mention in b4 (spoiler_name_leak)
    # - non-visual sound detail in visual_facts (non_visual_fact)
    ch1_nar_attempt1 = FakeReply.of_payload(
        _narrative_payload(
            block_count=6,
            chapter_summary="Александра просыпается в тумане возле разбитой машины.",
            scenes=[
                {
                    "start_block_id": "b1",
                    "end_block_id": "b3",
                    "summary": "Alexandra wakes up beside the wrecked silver car in total silence.",
                    "location": "У разбитой машины",
                    "participants": ["Alexandra"],
                    "visual_facts": [],
                    "visual_assumptions": [],
                    "continues_previous_scene": False,
                },
                {
                    "start_block_id": "b4",
                    "end_block_id": "b6",
                    "summary": "Александра разговаривает с мальчиком и смотрит на туманную дорогу.",
                    "location": "Туманная дорога",
                    "participants": ["Alexandra", "the boy"],
                    "visual_facts": [
                        {
                            "anchor": {"block_id": "b4", "quote": "Alexandra whispered"},
                            "subject": "Alexandra",
                            "detail": "Тихий шёпот голоса в серебристом тумане.",
                        }
                    ],
                    "visual_assumptions": [],
                    "continues_previous_scene": False,
                },
            ],
            glossary_candidates=[
                {
                    "source_text": "Alexandra",
                    "type": "person",
                    "suggested_translation": "Александра",
                    "confidence": 0.9,
                    "model_flagged_review": True,
                    "note": "женский род; краткая форма Allie -> Элли",
                    "description": "Девочка, очнувшаяся после аварии; просит звать её Элли.",
                    "name_translation_assessment": {
                        "applied_method": "sound",
                        "eligible_without_review": False,
                        "review_reasons": ["creative_adaptation", "creative_adaptation"],
                        "rationale": "Ошибочно указана противоречащая причина.",
                        "opaque_name": True,
                    },
                }
            ],
        )
    )

    ch1_nar_summary = (
        "Alexandra wakes up beside a wrecked silver car in the mist. "
        "She asks a boy on the curb to call her Allie and wonders what happened after the crash."
    )
    # Ch. 1 narrative attempt 2 succeeds and tests review_reasons deduplication preserving order.
    ch1_nar_attempt2 = FakeReply.of_payload(
        _narrative_payload(
            block_count=6,
            chapter_summary=ch1_nar_summary,
            scenes=[
                {
                    "start_block_id": "b1",
                    "end_block_id": "b3",
                    "summary": "Девушка приходит в себя возле разбитой серебристой машины и оглядывается.",
                    "location": "У разбитой машины",
                    "participants": ["the girl"],
                    "visual_facts": [],
                    "visual_assumptions": [],
                    "continues_previous_scene": False,
                },
                {
                    "start_block_id": "b4",
                    "end_block_id": "b6",
                    "summary": "Александра просит звать её Элли и всматривается в окутанную туманом дорогу.",
                    "location": "Туманная дорога",
                    "participants": ["Alexandra", "the boy"],
                    "visual_facts": [
                        {
                            "anchor": {"block_id": "b4", "quote": "shaking her cold hands"},
                            "subject": "Alexandra",
                            "detail": "Дрожащие руки девушки в серебристом тумане.",
                        }
                    ],
                    "visual_assumptions": [],
                    "continues_previous_scene": False,
                },
            ],
            glossary_candidates=[
                {
                    "source_text": "Alexandra",
                    "type": "person",
                    "suggested_translation": "Александра",
                    "confidence": 0.94,
                    "model_flagged_review": True,
                    "note": "женский род; краткая форма Allie -> Элли",
                    "description": "Девочка, очнувшаяся после аварии; просит звать её Элли.",
                    "name_translation_assessment": {
                        "applied_method": "sound",
                        "eligible_without_review": False,
                        "review_reasons": ["ambiguous", "other", "ambiguous"],
                        "rationale": "Личное имя и краткая форма Allie требуют сверки.",
                        "opaque_name": True,
                    },
                }
            ],
        )
    )

    ch1_a2_summary = (
        "Alexandra wakes up near a broken car in the silver mist. "
        "She meets a boy on the street and asks him to call her Allie."
    )
    ch1_ped_attempt1 = FakeReply.of_payload(
        _pedagogical_payload(
            block_count=6,
            summary="Alexandra stays in the silver mist for a long time. She meets a boy on the street.",
        )
    )
    ch1_ped_attempt2_payload = {
        "treatment_plan": [
            {"source_block_id": "b1", "treatment": "READ_AS_IS"},
            {"source_block_id": "b2", "treatment": "TRANSLATE"},
            {"source_block_id": "b3", "treatment": "TRANSLATE"},
            # Present Perfect in b4 marked READ_AS_IS -> A2 stoplist promotes to TRANSLATE.
            {"source_block_id": "b4", "treatment": "READ_AS_IS"},
            {"source_block_id": "b5", "treatment": "READ_AS_IS"},
            # Mixed 4-segment paragraph: seg1 READ_AS_IS, seg2 GLOSSARY_ONLY + lexical_hint,
            # seg3 READ_AS_IS with "uneasy" -> promoted to TRANSLATE, seg4 TRANSLATE.
            {
                "source_block_id": "b6",
                "treatment": "TRANSLATE",
                "segments": [
                    {"index": 1, "treatment": "READ_AS_IS"},
                    {"index": 2, "treatment": "GLOSSARY_ONLY"},
                    {"index": 3, "treatment": "READ_AS_IS"},
                    {"index": 4, "treatment": "TRANSLATE"},
                ],
            },
        ],
        "before_you_read": [],
        "difficulties": [
            {
                "id": "d-ch1-mist",
                "anchors": [{"block_id": "b6", "quote": "silver mist"}],
                "owner_block_id": "b6",
                "kind": "vocabulary",
                "priority": "useful",
                "help_kind": "lexical_hint",
                "required_points": [
                    {"id": "p1", "description": "объяснить значение silver mist (серебристый туман)"}
                ],
            }
        ],
        "before_you_read_evidence": [],
        "reader_chapter_summary": ch1_a2_summary,
    }
    ch1_ped_attempt2 = FakeReply.of_payload(ch1_ped_attempt2_payload)

    # Ch. 2 narrative: lists Alexandra in scene b1..b3 BEFORE Allie appears in b4 ->
    # allowed because Alexandra is in known_names from Ch. 1!
    # Also extracts Lief with leaf etymology and opaque_name=False.
    ch2_nar_summary = (
        "Allie walks through the cold forest and meets a boy named Lief dressed in green. "
        "Lief explains his name comes from leaf and asks what she is waiting for."
    )
    ch2_nar_reply = FakeReply.of_payload(
        _narrative_payload(
            block_count=6,
            chapter_summary=ch2_nar_summary,
            scenes=[
                {
                    "start_block_id": "b1",
                    "end_block_id": "b3",
                    "summary": "Дети идут по холодному тихому лесу вдоль мшистых камней.",
                    "location": "Холодный лес",
                    "participants": ["Alexandra"],
                    "visual_facts": [],
                    "visual_assumptions": [],
                    "continues_previous_scene": False,
                },
                {
                    "start_block_id": "b4",
                    "end_block_id": "b6",
                    "summary": "Элли встречает мальчика в зелёном по имени Лиф и бледнеет от его вопроса.",
                    "location": "Лесная поляна",
                    "participants": ["Alexandra", "Lief"],
                    "visual_facts": [
                        {
                            "anchor": {"block_id": "b4", "quote": "a boy dressed in green"},
                            "subject": "Lief",
                            "detail": "Мальчик одет в зелёное.",
                        }
                    ],
                    "visual_assumptions": [],
                    "continues_previous_scene": False,
                },
            ],
            glossary_candidates=[
                {
                    "source_text": "Lief",
                    "type": "person",
                    "suggested_translation": "Лиф",
                    "confidence": 0.93,
                    "model_flagged_review": False,
                    "note": "мужской род",
                    "description": "Мальчик в зелёном; имя обыгрывает слово leaf (зелёный лист).",
                    "name_translation_assessment": _name_assessment(
                        applied_method="meaning",
                        opaque_name=False,
                        rationale="В тексте явно обыгрывается этимология Lief / leaf.",
                    ),
                }
            ],
        )
    )

    ch2_a2_summary = (
        "Allie walks in the cold forest and meets a boy in green. "
        "His name is Lief, like a green leaf, and his question makes Allie pale."
    )
    ch2_ped_attempt1 = FakeReply.of_payload(
        _pedagogical_payload(
            block_count=6,
            summary="A baker cooks warm bread in a sunny kitchen. He sells red apples at the market.",
        )
    )
    ch2_ped_attempt2_payload = {
        "treatment_plan": [
            {"source_block_id": "b1", "treatment": "READ_AS_IS"},
            {"source_block_id": "b2", "treatment": "TRANSLATE"},
            {"source_block_id": "b3", "treatment": "TRANSLATE"},
            {"source_block_id": "b4", "treatment": "TRANSLATE"},
            {"source_block_id": "b5", "treatment": "TRANSLATE"},
            # Stranded preposition + state-change idiom ("What are you waiting for?" / "turned pale")
            # marked GLOSSARY_ONLY with only lexical_hint -> promoted to TRANSLATE and hint -> explanation.
            {"source_block_id": "b6", "treatment": "GLOSSARY_ONLY"},
        ],
        "before_you_read": [],
        "difficulties": [
            {
                "id": "d-ch2-mossy",
                "anchors": [{"block_id": "b3", "quote": "mossy stones"}],
                "owner_block_id": "b3",
                "kind": "vocabulary",
                "priority": "useful",
                "help_kind": "explanation",
                "required_points": [
                    {"id": "p1", "description": "объяснить mossy stones (камни, покрытые мхом)"}
                ],
            },
            {
                "id": "d-ch2-pale",
                "anchors": [{"block_id": "b6", "quote": "pale"}],
                "owner_block_id": "b6",
                "kind": "vocabulary",
                "priority": "useful",
                "help_kind": "lexical_hint",
                "required_points": [
                    {"id": "p1", "description": "объяснить pale (бледный)"}
                ],
            },
        ],
        "before_you_read_evidence": [],
        "reader_chapter_summary": ch2_a2_summary,
    }
    ch2_ped_attempt2 = FakeReply.of_payload(ch2_ped_attempt2_payload)

    # Ch. 3 narrative + pedagogical: neither Afterlights nor Lief appears in Ch. 3 text,
    # yet both unconfirmed key candidates and Lief's leaf etymology + opaque_name=False are backfilled.
    ch3_nar_summary = (
        "Allie follows the boy along the misty trail toward a sunken streetlamp. "
        "The boy refuses to go back into the dark woods."
    )
    ch3_nar_reply = FakeReply.of_payload(
        _narrative_payload(
            block_count=4,
            chapter_summary=ch3_nar_summary,
            scenes=[
                {
                    "start_block_id": "b1",
                    "end_block_id": "b4",
                    "summary": "Элли идёт за мальчиком по туманной тропе к фонарю, а он отказывается возвращаться в лес.",
                    "location": "Туманная тропа",
                    "participants": ["Alexandra", "Lief"],
                    "visual_facts": [],
                    "visual_assumptions": [],
                    "continues_previous_scene": False,
                }
            ],
        )
    )
    ch3_a2_summary = (
        "Allie follows the boy down the misty trail to a sunken streetlamp. "
        "The boy says he will not go back into the dark woods."
    )
    ch3_ped_attempt1 = FakeReply.of_payload(
        _pedagogical_payload(
            block_count=4,
            summary=(
                "Allie follows the boy carefully along the narrow misty trail through the quiet "
                "cold forest toward the broken sunken streetlamp while the boy refuses to go back "
                "into the dark shadows again."
            ),
        )
    )
    ch3_ped_attempt2 = FakeReply.of_payload(
        {
            "treatment_plan": [
                {"source_block_id": "b1", "treatment": "READ_AS_IS"},
                {"source_block_id": "b2", "treatment": "TRANSLATE"},
                {"source_block_id": "b3", "treatment": "TRANSLATE"},
                # Short 7-word dialogue with nonstandard "ain't": must stay TRANSLATE (not demoted to READ_AS_IS).
                {"source_block_id": "b4", "treatment": "TRANSLATE"},
            ],
            "before_you_read": [],
            "difficulties": [],
            "before_you_read_evidence": [],
            "reader_chapter_summary": ch3_a2_summary,
        }
    )

    # 1. Sequential multi-section run: PART ONE -> Ch. 1 -> Ch. 2 -> Ch. 3.
    seq_client = StagedFakeLLMClient.replying(
        [
            part1_narrative_reply,
            ch1_nar_attempt1,
            ch1_nar_attempt2,
            ch1_ped_attempt1,
            ch1_ped_attempt2,
            ch2_nar_reply,
            ch2_ped_attempt1,
            ch2_ped_attempt2,
            ch3_nar_reply,
            ch3_ped_attempt1,
            ch3_ped_attempt2,
        ]
    )
    seq_app = _make_multi_section_app(tmp_path / "sequential", seq_client, document, concurrency=1)
    try:
        await seq_app.analysis.start(BOOK_ID)
        completed = await drain_tasks(seq_app)
        assert all(task.status is TaskStatus.SUCCEEDED for task in completed), [
            (task.scope_id, task.last_error) for task in completed
        ]

        # Exact call trace: PART ONE gets narrative only (no pedagogical); Ch. 1-3 execute retries cleanly.
        assert [(req.scope_id, req.stage) for req in seq_client.requests] == [
            ("part-1", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE),
            ("ch-1", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE),
            ("ch-1", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE),
            ("ch-1", LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL),
            ("ch-1", LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL),
            ("ch-2", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE),
            ("ch-2", LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL),
            ("ch-2", LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL),
            ("ch-3", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE),
            ("ch-3", LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL),
            ("ch-3", LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL),
        ]

        part1_nar_req = seq_client.requests[0]
        ch1_nar_req1, ch1_nar_req2 = seq_client.requests[1], seq_client.requests[2]
        ch1_ped_req2 = seq_client.requests[4]
        ch2_nar_req = seq_client.requests[5]
        ch2_ped_req1, ch2_ped_req2 = seq_client.requests[6], seq_client.requests[7]
        ch3_nar_req = seq_client.requests[8]
        ch3_ped_req1, ch3_ped_req2 = seq_client.requests[9], seq_client.requests[10]

        # PART ONE divider receives dedicated part-divider instructions.
        assert "разделитель части с содержательным подзаголовком" in part1_nar_req.instructions

        # Ch. 1 receives unconfirmed Afterlights extracted from PART ONE even though absent in Ch. 1 text.
        assert "Afterlights" in ch1_nar_req1.input_text
        assert "Послесветы" in ch1_nar_req1.input_text
        assert "непрозрачное=false" in ch1_nar_req1.input_text

        # Ch. 1 narrative retry contains all five aggregated violation families.
        ch1_nar_repair = ch1_nar_req2.repair_instructions[-1]
        assert "пересказ главы должен быть на языке оригинала (en)" in ch1_nar_repair
        assert "описание сцены должно быть на целевом языке (ru)" in ch1_nar_repair
        assert "несогласованная оценка перевода имени" in ch1_nar_repair
        assert "спойлер имени до первого упоминания в главе" in ch1_nar_repair
        assert "визуальный факт содержит незрительное описание" in ch1_nar_repair

        # Ch. 1/2/3 pedagogical retries carry exact A2 summary violation diagnostics.
        assert "Present Simple с for + длительность" in ch1_ped_req2.repair_instructions[-1]
        assert "не имеет общих значимых основ с narrative_summary" in ch2_ped_req2.repair_instructions[-1]
        assert "не больше 26" in ch3_ped_req2.repair_instructions[-1]

        # Ch. 2 narrative prompt receives Afterlights + Alexandra; Ch. 2 pedagogical receives Ch. 1 topics.
        assert "Afterlights" in ch2_nar_req.input_text
        assert "Alexandra" in ch2_nar_req.input_text
        assert "краткая форма Allie -> Элли" in ch2_nar_req.input_text
        assert "Уже объяснено в предыдущих главах" in ch2_ped_req1.input_text
        assert "- silver mist" in ch2_ped_req1.input_text

        # Ch. 3 narrative prompt receives backfilled Afterlights, Alexandra, and Lief/leaf etymology
        # with непрозрачное=false even though neither Afterlights nor Lief appears in Ch. 3 text.
        assert "Afterlights" in ch3_nar_req.input_text
        assert "Alexandra" in ch3_nar_req.input_text
        lief_lines = [
            line for line in ch3_nar_req.input_text.splitlines() if "Lief" in line
        ]
        assert lief_lines
        assert any(
            "leaf" in line and "непрозрачное=false" in line for line in lief_lines
        )

        # Ch. 3 pedagogical prompt receives deduplicated prior_chapter_topics from both Ch. 1 and Ch. 2.
        assert "Уже объяснено в предыдущих главах" in ch3_ped_req1.input_text
        assert "- silver mist" in ch3_ped_req1.input_text
        assert "- mossy stones" in ch3_ped_req1.input_text

        # Check persisted SQLite analyses across all 4 sections.
        with seq_app.unit_of_work.transaction() as tx:
            analyses_by_section = {a.section_id: a for a in tx.analyses.list_for_book(BOOK_ID)}
            assert set(analyses_by_section) == {"part-1", "ch-1", "ch-2", "ch-3"}

            a_part1 = analyses_by_section["part-1"]
            assert a_part1.scenes == ()
            assert a_part1.before_you_read == ()
            assert a_part1.chapter_summary == ""
            assert [c.source_text for c in a_part1.glossary_candidates] == ["Afterlights"]

            a_ch1 = analyses_by_section["ch-1"]
            assert a_ch1.chapter_summary == ch1_a2_summary
            alex_candidate = next(c for c in a_ch1.glossary_candidates if c.source_text == "Alexandra")
            assert alex_candidate.name_translation_evidence is not None
            alex_assessment = alex_candidate.name_translation_evidence.observations[0].assessment
            assert alex_assessment is not None
            assert alex_assessment.review_reasons == (
                NameTranslationReviewReason.AMBIGUOUS,
                NameTranslationReviewReason.OTHER,
            )

            ch1_blocks = textual_blocks(ch1)
            ch1_plan = {item.block_id: item for item in a_ch1.treatment_plan.items}
            # Present Perfect b4 promoted from READ_AS_IS -> TRANSLATE.
            assert ch1_plan[ch1_blocks[3].id].treatment is BlockTreatment.TRANSLATE
            # Mixed 4-segment paragraph b6: seg3 ("uneasy") promoted -> TRANSLATE; seg1 READ_AS_IS, seg2 GLOSSARY_ONLY.
            b6_item = ch1_plan[ch1_blocks[5].id]
            assert b6_item.treatment is BlockTreatment.TRANSLATE
            assert b6_item.segment_treatments == (
                BlockTreatment.READ_AS_IS,
                BlockTreatment.GLOSSARY_ONLY,
                BlockTreatment.TRANSLATE,
                BlockTreatment.TRANSLATE,
            )
            # Difficulty on GLOSSARY_ONLY seg2 preserves help_kind="lexical_hint".
            assert a_ch1.content_plan is not None
            mist_diff = next(
                d
                for d in a_ch1.content_plan.difficulties
                if any(anchor.quote == "silver mist" for anchor in d.anchors)
            )
            assert mist_diff.help_kind == "lexical_hint"

            a_ch2 = analyses_by_section["ch-2"]
            assert a_ch2.chapter_summary == ch2_a2_summary
            assert "Alexandra" in a_ch2.scenes[0].participants
            lief_candidate = next(c for c in a_ch2.glossary_candidates if c.source_text == "Lief")
            assert "leaf" in lief_candidate.description
            assert lief_candidate.name_translation_evidence is not None
            assert lief_candidate.name_translation_evidence.observations[0].assessment.opaque_name is False
            # Ch. 2 b6 promoted from GLOSSARY_ONLY -> TRANSLATE and lexical_hint -> explanation.
            ch2_blocks = textual_blocks(ch2)
            ch2_plan = {item.block_id: item for item in a_ch2.treatment_plan.items}
            assert ch2_plan[ch2_blocks[5].id].treatment is BlockTreatment.TRANSLATE
            assert a_ch2.content_plan is not None
            pale_diff = next(
                d
                for d in a_ch2.content_plan.difficulties
                if any(anchor.quote == "pale" for anchor in d.anchors)
            )
            assert pale_diff.help_kind == "explanation"

            a_ch3 = analyses_by_section["ch-3"]
            assert a_ch3.chapter_summary == ch3_a2_summary
            ch3_blocks = textual_blocks(ch3)
            ch3_plan = {item.block_id: item for item in a_ch3.treatment_plan.items}
            # Demotion guard keeps short nonstandard "ain't" line in TRANSLATE.
            assert ch3_plan[ch3_blocks[3].id].treatment is BlockTreatment.TRANSLATE

        # Book conditions version bump (context_builder=5, prompts=7) + backward compatibility for v6 and v5 snapshots.
        conditions = seq_app.book_conditions.require(BOOK_ID)
        assert conditions.payload["profile"] == "content-plan-v3"
        assert conditions.payload["versions"] == A2_CONTENT_PLAN_V7_VERSIONS
        assert conditions.payload["versions"]["context_builder"] == 5
        assert conditions.payload["versions"]["prompts"] == 7
        assert effective_content_plan_profile(conditions) == "content-plan-v3"
        seq_app.book_runtime.settings_for(conditions)

        for legacy_versions in (A2_CONTENT_PLAN_V6_VERSIONS, A2_CONTENT_PLAN_V5_VERSIONS):
            legacy_payload = thaw(conditions.payload)
            legacy_payload["versions"] = dict(legacy_versions)
            legacy_conditions = replace(conditions, payload=legacy_payload)
            assert effective_content_plan_profile(legacy_conditions) == "content-plan-v3"
            seq_app.book_runtime.settings_for(legacy_conditions)
    finally:
        await seq_app.aclose()

    # 2. Prewave parallel seeding (explicit sequential_analysis=False, concurrency=2, seed_first_chapter=True):
    # Wave 1 = PART ONE / Afterlights -> Wave 2 = Ch. 1 (seeded with Afterlights) -> Wave 3 = Ch. 2 & Ch. 3.
    prewave_client = ScopedStagedFakeLLMClient(
        {
            ("part-1", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE): [part1_narrative_reply],
            ("ch-1", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE): [ch1_nar_attempt2],
            ("ch-1", LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL): [ch1_ped_attempt2],
            ("ch-2", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE): [ch2_nar_reply],
            ("ch-2", LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL): [ch2_ped_attempt2],
            ("ch-3", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE): [ch3_nar_reply],
            ("ch-3", LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL): [ch3_ped_attempt2],
        }
    )
    prewave_app = _make_multi_section_app(
        tmp_path / "prewave",
        prewave_client,
        document,
        concurrency=2,
        seed_first_chapter=True,
        sequential_analysis=False,
    )
    try:
        await prewave_app.analysis.start(BOOK_ID)
        completed = await drain_tasks(prewave_app)
        assert all(task.status is TaskStatus.SUCCEEDED for task in completed), [
            (task.scope_id, task.last_error) for task in completed
        ]

        # Prewave order: PART ONE completes before Ch. 1; Ch. 1 completes before Ch. 2 & Ch. 3 start.
        assert [(req.scope_id, req.stage) for req in prewave_client.requests[:3]] == [
            ("part-1", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE),
            ("ch-1", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE),
            ("ch-1", LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL),
        ]
        ch1_prewave_nar = prewave_client.requests[1]
        assert "Afterlights" in ch1_prewave_nar.input_text
        assert "Послесветы" in ch1_prewave_nar.input_text

        wave3_nar = {
            req.scope_id: req
            for req in prewave_client.requests[3:]
            if req.stage is LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE
        }
        wave3_ped = {
            req.scope_id: req
            for req in prewave_client.requests[3:]
            if req.stage is LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL
        }
        assert set(wave3_nar) == {"ch-2", "ch-3"}
        assert set(wave3_ped) == {"ch-2", "ch-3"}
        for sid in ("ch-2", "ch-3"):
            assert "Afterlights" in wave3_nar[sid].input_text
            assert "Alexandra" in wave3_nar[sid].input_text
            assert "Уже объяснено в предыдущих главах" in wave3_ped[sid].input_text
            assert "- silver mist" in wave3_ped[sid].input_text
        # Wave-3 isolation: Ch. 2's candidates and topics do NOT leak into parallel Ch. 3.
        assert "Lief" not in wave3_nar["ch-3"].input_text
        assert "Lief" not in wave3_ped["ch-3"].input_text
        assert "mossy stones" not in wave3_ped["ch-3"].input_text
    finally:
        await prewave_app.aclose()

    # 3. Default sequential mode takes precedence over seed_first_chapter=True even with concurrency=4:
    # PART ONE -> Ch. 1 -> Ch. 2 -> Ch. 3 strictly in reading order; Ch. 3 sees Ch. 2 Lief + topics;
    # PART ONE Afterlights gets GLOSSARY_ONLY while PART ONE stays READ_AS_IS.
    default_seq_client = ScopedStagedFakeLLMClient(
        {
            ("part-1", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE): [part1_narrative_reply],
            ("ch-1", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE): [ch1_nar_attempt2],
            ("ch-1", LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL): [ch1_ped_attempt2],
            ("ch-2", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE): [ch2_nar_reply],
            ("ch-2", LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL): [ch2_ped_attempt2],
            ("ch-3", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE): [ch3_nar_reply],
            ("ch-3", LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL): [ch3_ped_attempt2],
        }
    )
    default_seq_app = _make_multi_section_app(
        tmp_path / "default-sequential-high-concurrency",
        default_seq_client,
        document,
        concurrency=4,
        seed_first_chapter=True,
        sequential_analysis=True,
    )
    try:
        await default_seq_app.analysis.start(BOOK_ID)
        completed = await drain_tasks(default_seq_app)
        assert all(task.status is TaskStatus.SUCCEEDED for task in completed), [
            (task.scope_id, task.last_error) for task in completed
        ]
        assert [(req.scope_id, req.stage) for req in default_seq_client.requests] == [
            ("part-1", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE),
            ("ch-1", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE),
            ("ch-1", LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL),
            ("ch-2", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE),
            ("ch-2", LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL),
            ("ch-3", LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE),
            ("ch-3", LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL),
        ]
        ch3_seq_nar = default_seq_client.requests[5]
        ch3_seq_ped = default_seq_client.requests[6]
        assert "Lief" in ch3_seq_nar.input_text
        assert "mossy stones" in ch3_seq_ped.input_text
        with default_seq_app.unit_of_work.transaction() as tx:
            part1_analysis = {a.section_id: a for a in tx.analyses.list_for_book(BOOK_ID)}["part-1"]
            part1_modes = {item.block_id: item.treatment for item in part1_analysis.treatment_plan.items}
            part1_blocks = textual_blocks(part_one)
            assert part1_modes[part1_blocks[0].id] is BlockTreatment.READ_AS_IS
            assert part1_modes[part1_blocks[1].id] is BlockTreatment.GLOSSARY_ONLY
    finally:
        await default_seq_app.aclose()
