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
from linguary.adapters.config.settings import ReadingSettings
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application.book_conditions import payload_digest
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
from linguary.domain import (
    AnalysisPassKind,
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

    # 1. Default configuration (max_fragment_blocks=None) analyzes all 121 textual blocks in 1 fragment.
    default_client = StagedFakeLLMClient.replying(
        [
            FakeReply.of_payload(
                _narrative_payload(
                    block_count=121,
                    chapter_summary=full_summary,
                    glossary_candidates=candidates,
                )
            ),
            FakeReply.of_payload(_pedagogical_payload(block_count=121, summary=full_summary)),
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
            LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL,
        ]
        nar_req, ped_req = default_client.requests
        assert nar_req.ordered_block_ids == tuple(b.id for b in tb)
        assert ped_req.ordered_block_ids == tuple(b.id for b in tb)

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
