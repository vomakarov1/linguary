"""Reproducible evaluation harness for optional book illustration preparation (`P5.T1`).

Fake mode (`--mode fake`) runs the real application pipeline against a deterministic
synthetic fixture and adversarial replies with zero network calls.
Live mode (`--mode live`) requires explicit `--allow-paid-calls` and configured LLM
credentials; without `--allow-paid-calls` it exits non-zero BEFORE constructing any
HTTP client or issuing any network request.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings, load_settings
from linguary.application.illustration_dto import (
    IllustrationAliasDecisionDTO,
    IllustrationAliasResponse,
    IllustrationCandidateEventDTO,
    IllustrationCandidatesResponse,
    IllustrationDesignResponse,
    IllustrationDesignTraitDTO,
    IllustrationExtractResponse,
    IllustrationFactDTO,
    IllustrationFactScopeResponse,
    IllustrationMentionDTO,
    IllustrationPromptResponse,
    IllustrationQuoteAnchorDTO,
    IllustrationSelectionReasonDTO,
    IllustrationSelectionResponse,
)
from linguary.application.illustration_source import (
    IllustrationSourceSnapshot,
    SectionManifestEntry,
)
from linguary.application.illustration_validation import (
    MONOCHROME_INK_DIRECTIVE,
    IllustrationAppearance,
    ValidatedMoment,
    validate_moment,
)
from linguary.domain.content_plan import SourceAnchor
from linguary.domain.illustrations import (
    DEFAULT_ILLUSTRATION_POLICY,
    IllustrationGenerationState,
    IllustrationRegistryKind,
    IllustrationScopeStage,
    IllustrationSettings,
    IllustrationValidationError,
)
from linguary.application.llm_dto import (
    NarrativeAnalysisResponse,
    PedagogicalAnalysisResponse,
    QuotedSceneResponse,
    TreatmentPlanItemResponse,
)
from linguary.application.ports.llm import LLMRequest, LLMResponse, LLMResponseKind
from linguary.application.ports.persistence import BookRecord
from linguary.composition import build_application, drain_tasks
from linguary.domain import (
    BlockTreatment,
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    SourceReference,
    SupportStatus,
    TokenUsage,
)


def _extract_json_block(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    return json.loads(text[start : end + 1])


class SyntheticEvalLLMClient:
    """Deterministic fake LLM client for the synthetic lighthouse novel fixture."""

    supports_staged_analysis = True

    def __init__(self, fixture: dict[str, Any]) -> None:
        self._sections_by_id = {
            sec["id"]: sec for sec in fixture["book"]["sections"]
        }
        self.call_count = 0
        self.max_input_bytes = 0

    async def generate(self, request: LLMRequest[Any]) -> LLMResponse[Any]:
        self.call_count += 1
        wire_len = len(request.instructions.encode("utf-8")) + len(
            request.input_text.encode("utf-8")
        )
        self.max_input_bytes = max(self.max_input_bytes, wire_len)
        usage = TokenUsage(prompt_tokens=60, completion_tokens=40)

        if request.kind is LLMResponseKind.CHAPTER_ANALYSIS_NARRATIVE:
            aliases = (
                tuple(alias for alias, _ in request.block_aliases)
                if request.block_aliases
                else tuple(f"b{idx}" for idx in range(1, len(request.ordered_block_ids) + 1))
                or ("b1",)
            )
            sec = self._sections_by_id.get(request.scope_id or "", {})
            summary_ru = str(sec.get("scene_summary") or "Сцена главы.")
            location = sec.get("location")
            participants = tuple(sec.get("participants") or ())
            payload: Any = NarrativeAnalysisResponse(
                translation_instructions="Сохраняй выразительный литературный тон.",
                chapter_summary="A pivotal maritime chapter in the synthetic evaluation book.",
                scenes=(
                    QuotedSceneResponse(
                        start_block_id=aliases[0],
                        end_block_id=aliases[-1],
                        summary=summary_ru,
                        location=str(location) if location else None,
                        participants=participants,
                    ),
                ),
            )
            return LLMResponse(
                payload=payload, provider="synthetic_fake", model="fake-eval", usage=usage
            )

        if request.kind is LLMResponseKind.CHAPTER_ANALYSIS_PEDAGOGICAL:
            aliases = (
                tuple(alias for alias, _ in request.block_aliases)
                if request.block_aliases
                else tuple(f"b{idx}" for idx in range(1, len(request.ordered_block_ids) + 1))
                or ("b1",)
            )
            items = tuple(
                TreatmentPlanItemResponse(source_block_id=bid, treatment=BlockTreatment.TRANSLATE)
                for bid in aliases
            )
            return LLMResponse(
                payload=PedagogicalAnalysisResponse(treatment_plan=items),
                provider="synthetic_fake",
                model="fake-eval",
                usage=usage,
            )

        if request.kind is LLMResponseKind.ILLUSTRATION_EXTRACT:
            data = _extract_json_block(request.input_text)
            units = data.get("units", ())
            unit_ids = tuple(str(u["source_unit_id"]) for u in units if isinstance(u, dict))
            mentions: list[IllustrationMentionDTO] = []
            facts: list[IllustrationFactDTO] = []
            for unit in units:
                for blk in unit.get("blocks", ()):
                    bid = str(blk["block_id"])
                    txt = str(blk["text"])
                    if "Young John Miller" in txt:
                        mentions.append(
                            IllustrationMentionDTO(
                                mention_id="m_jm_1",
                                name="John Miller",
                                entity_kind="person",
                                alias_of_hint=None,
                                quote_block_id=bid,
                                quote="Young John Miller",
                            )
                        )
                        facts.append(
                            IllustrationFactDTO(
                                fact_id="f_jm_hair",
                                subject_mention_id="m_jm_1",
                                temporal_scope="permanent",
                                back_applicable=True,
                                reveals_future_spoiler=False,
                                description_ru="Тёмные вьющиеся волосы и шерстяной бушлат.",
                                description_en="Dark curly hair and heavy wool peacoat.",
                                quote_block_id=bid,
                                quote="dark curly hair and wool peacoat",
                            )
                        )
                        mentions.append(
                            IllustrationMentionDTO(
                                mention_id="m_lh_1",
                                name="Old Lighthouse",
                                entity_kind="location",
                                alias_of_hint=None,
                                quote_block_id=bid,
                                quote="Old Lighthouse",
                            )
                        )
                        facts.append(
                            IllustrationFactDTO(
                                fact_id="f_lh_arch",
                                subject_mention_id="m_lh_1",
                                temporal_scope="permanent",
                                back_applicable=True,
                                reveals_future_spoiler=False,
                                description_ru="Каменная арка старого маяка у гавани.",
                                description_en="Weathered stone arch of the old harbor lighthouse.",
                                quote_block_id=bid,
                                quote="stone arch of the Old Lighthouse",
                            )
                        )
                    if "Captain John Vance" in txt and "brass-tipped cane" in txt:
                        mentions.append(
                            IllustrationMentionDTO(
                                mention_id="m_jv_1",
                                name="Captain John Vance",
                                entity_kind="person",
                                alias_of_hint=None,
                                quote_block_id=bid,
                                quote="Captain John Vance",
                            )
                        )
                        facts.append(
                            IllustrationFactDTO(
                                fact_id="f_jv_cane",
                                subject_mention_id="m_jv_1",
                                temporal_scope="permanent",
                                back_applicable=True,
                                reveals_future_spoiler=False,
                                description_ru="Пожилой капитан с тростью с латунным наконечником.",
                                description_en="Elderly captain leaning on a brass-tipped cane.",
                                quote_block_id=bid,
                                quote="brass-tipped cane",
                            )
                        )
                    if "scarlet festival cloak" in txt:
                        mentions.append(
                            IllustrationMentionDTO(
                                mention_id="m_jm_3",
                                name="John Miller",
                                entity_kind="person",
                                alias_of_hint=None,
                                quote_block_id=bid,
                                quote="John Miller",
                            )
                        )
                        facts.append(
                            IllustrationFactDTO(
                                fact_id="f_jm_cloak_temp",
                                subject_mention_id="m_jm_3",
                                temporal_scope="episodic",
                                back_applicable=False,
                                reveals_future_spoiler=False,
                                description_ru="Одолженный праздничный плащ на один вечер.",
                                description_en="Borrowed festival cloak worn for a single evening.",
                                quote_block_id=bid,
                                quote="borrowed scarlet festival cloak",
                            )
                        )
                    if "jagged scar" in txt:
                        mentions.append(
                            IllustrationMentionDTO(
                                mention_id="m_jm_10",
                                name="John Miller",
                                entity_kind="person",
                                alias_of_hint=None,
                                quote_block_id=bid,
                                quote="John Miller",
                            )
                        )
                        facts.append(
                            IllustrationFactDTO(
                                fact_id="f_jm_scar_ch10",
                                subject_mention_id="m_jm_10",
                                temporal_scope="state_change",
                                back_applicable=False,
                                reveals_future_spoiler=True,
                                description_ru="Неровный шрам на левой щеке после бури.",
                                description_en="Jagged scar across the left cheek from the storm.",
                                quote_block_id=bid,
                                quote="jagged scar across John Miller's left cheek",
                            )
                        )
            return LLMResponse(
                payload=IllustrationExtractResponse(
                    mentions=tuple(mentions),
                    facts=tuple(facts),
                    covered_source_unit_ids=unit_ids,
                    complete=True,
                ),
                provider="synthetic_fake",
                model="fake-eval",
                usage=usage,
            )

        if request.kind is LLMResponseKind.ILLUSTRATION_ALIAS:
            data = _extract_json_block(request.input_text)
            decisions = []
            for pair in data.get("pairs", ()):
                pid = str(pair["pair_id"])
                pair_text = json.dumps(pair, ensure_ascii=False)
                verdict = "different" if "Vance" in pair_text else "same_entity"
                decisions.append(
                    IllustrationAliasDecisionDTO(pair_id=pid, verdict=verdict, evidence_fact_ids=())
                )
            return LLMResponse(
                payload=IllustrationAliasResponse(decisions=tuple(decisions)),
                provider="synthetic_fake",
                model="fake-eval",
                usage=usage,
            )

        if request.kind is LLMResponseKind.ILLUSTRATION_DESIGN:
            data = _extract_json_block(request.input_text)
            entity = data.get("entity", {})
            eid = str(entity.get("entity_id") or data.get("entity_id") or "ent-unknown")
            allowed_traits = data.get("allowed_traits") or ["attire_default"]
            trait = str(allowed_traits[0])
            return LLMResponse(
                payload=IllustrationDesignResponse(
                    entity_id=eid,
                    traits=(
                        IllustrationDesignTraitDTO(
                            trait=trait,
                            description_ru="Сдержанный морской облик начала века.",
                            description_en="Weathered early-century maritime silhouette.",
                            assumption_ru="Уточнено для цельной графики тушью.",
                        ),
                    ),
                ),
                provider="synthetic_fake",
                model="fake-eval",
                usage=usage,
            )

        if request.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            data = _extract_json_block(request.input_text)
            scene_id = str(data.get("scene_id") or "s000")
            section_id = str(data.get("section_id") or "")
            blocks = data.get("blocks", ())
            unit_ids = tuple(str(b["source_unit_id"]) for b in blocks if isinstance(b, dict))
            if section_id == "sec-03-conversation":
                return LLMResponse(
                    payload=IllustrationCandidatesResponse(
                        scene_id=scene_id,
                        events=(),
                        zero_reason_ru="Обычный бытовой разговор за чаем без выразительного визуального действия.",
                        covered_source_unit_ids=unit_ids,
                        complete=True,
                    ),
                    provider="synthetic_fake",
                    model="fake-eval",
                    usage=usage,
                )
            first_blk = blocks[0]
            bid = str(first_blk["block_id"])
            txt = str(first_blk["text"])
            quote = txt[:48]
            return LLMResponse(
                payload=IllustrationCandidatesResponse(
                    scene_id=scene_id,
                    events=(
                        IllustrationCandidateEventDTO(
                            event_id=f"ev-{section_id}",
                            title_ru="Встреча у старого маяка"
                            if section_id == "sec-02-harbor"
                            else "Шторм на вершине маяка",
                            event_ru="Напряжённый драматический момент у маяка в шторм.",
                            significance_ru="Ключевой поворот главы.",
                            event_kind="turning_point",
                            quote_anchors=(IllustrationQuoteAnchorDTO(block_id=bid, quote=quote),),
                            insert_after_block_id=bid,
                            participant_names=("John Miller", "John Vance"),
                            location_names=("Old Lighthouse",),
                        ),
                    ),
                    covered_source_unit_ids=unit_ids,
                    complete=True,
                ),
                provider="synthetic_fake",
                model="fake-eval",
                usage=usage,
            )

        if request.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            data = _extract_json_block(request.input_text)
            cands = data.get("candidates", ())
            chosen = tuple(str(c["candidate_id"]) for c in cands[:1] if isinstance(c, dict))
            reasons = tuple(
                IllustrationSelectionReasonDTO(
                    candidate_id=cid, reason_ru="Ключевая выразительная сцена главы."
                )
                for cid in chosen
            )
            return LLMResponse(
                payload=IllustrationSelectionResponse(
                    selected_candidate_ids=chosen,
                    selection_reasons_ru=reasons,
                    rejected_reasons_ru=(),
                ),
                provider="synthetic_fake",
                model="fake-eval",
                usage=usage,
            )

        if request.kind is LLMResponseKind.ILLUSTRATION_FACT_SCOPE:
            data = _extract_json_block(request.input_text)
            ev_id = str(data.get("event_id") or "ev-1")
            facts = data.get("facts", ())
            applicable = tuple(str(f["fact_id"]) for f in facts if isinstance(f, dict))
            return LLMResponse(
                payload=IllustrationFactScopeResponse(
                    event_id=ev_id,
                    applicable_fact_ids=applicable,
                    excluded_fact_ids=(),
                    compact_description_ru="Персонаж в соответствии с фактами до текущей сцены.",
                    compact_description_en="Character matching all facts established up to this scene.",
                    assumptions_ru=(),
                    complete=True,
                ),
                provider="synthetic_fake",
                model="fake-eval",
                usage=usage,
            )

        if request.kind is LLMResponseKind.ILLUSTRATION_PROMPT:
            data = _extract_json_block(request.input_text)
            moment_id = str(data.get("moment_id") or "m1")
            chars = data.get("character_descriptions", ())
            locs = data.get("location_descriptions", ())
            used_facts: list[str] = []
            desc_parts: list[str] = []
            for item in (*chars, *locs):
                if isinstance(item, dict):
                    used_facts.extend(str(fid) for fid in item.get("fact_ids", ()))
                    if item.get("description_en"):
                        desc_parts.append(str(item["description_en"]))
            composition = (
                "Dramatic coastal harbor night in monochrome ink line art: "
                + "; ".join(desc_parts or ["stormy lighthouse arch in driving rain"])
                + ", strong crosshatching and stark ink contrast."
            )
            return LLMResponse(
                payload=IllustrationPromptResponse(
                    moment_id=moment_id,
                    scene_composition_en=composition,
                    used_fact_ids=tuple(dict.fromkeys(used_facts)),
                    used_design_keys=(),
                    assumptions_ru=(),
                ),
                provider="synthetic_fake",
                model="fake-eval",
                usage=usage,
            )

        raise AssertionError(f"unexpected request kind: {request.kind}")


def run_adversarial_checks() -> dict[str, bool]:
    """Prove production validators reject all adversarial model replies."""
    from linguary.application.illustration_selection import IllustrationSelection
    from linguary.domain.illustrations import IllustrationScope, IllustrationScopeState

    results: dict[str, bool] = {}

    blk = DocumentBlock(
        id="b1",
        section_id="sec-2",
        type=BlockType.PARAGRAPH,
        order=0,
        text="Young John Miller stood beneath the stone arch.",
        source_reference=SourceReference("ch2.xhtml", "p[0]"),
        support_status=SupportStatus.SUPPORTED,
    )
    section = DocumentSection(
        id="sec-2",
        document_id="book-adv",
        title="Chapter 2",
        order=1,
        source_resource_path="ch2.xhtml",
        blocks=(blk,),
    )
    snapshot = IllustrationSourceSnapshot(
        book_id="book-adv",
        compatible=True,
        ready=True,
        reason_code=None,
        source_sha256="a" * 64,
        conditions_id="cond-1",
        conditions_revision=1,
        expected_chapters=1,
        ready_chapters=1,
        skipped_sections=0,
        blocking_section_id=None,
        content_section_ids=("sec-2",),
        manifest=None,
        manifest_sha256=None,
        source_authority=None,
        entries=(
            SectionManifestEntry(
                section_id="sec-2",
                order=1,
                title="Chapter 2",
                skipped=False,
                skip_reason=None,
                block_digests={},
            ),
        ),
    )
    registry = {
        "f_jm_hair": {"position_key": "0001:0000", "is_back_applicable": True},
        "f_jm_scar_ch10": {"position_key": "0010:0000", "is_back_applicable": False},
    }
    valid_anchor = SourceAnchor(block_id="b1", start=0, end=17, quote="Young John Miller")
    valid_appearance = IllustrationAppearance(
        entity_id="ent-jm",
        name_ru="Джон Миллер",
        description_ru="Тёмные вьющиеся волосы.",
        description_en="Dark curly hair and heavy wool peacoat.",
        fact_ids=("f_jm_hair",),
    )

    def _expect_rejected(moment: ValidatedMoment) -> bool:
        try:
            validate_moment(
                moment,
                snapshot=snapshot,
                section=section,
                scene_start_block_id="b1",
                scene_end_block_id="b1",
                registry=registry,
                event_position_key="0001:0000",
            )
        except IllustrationValidationError:
            return True
        return False

    # 1. Future spoiler fact_id rejected in Chapter 2
    results["adv_future_spoiler_in_ch2"] = _expect_rejected(
        ValidatedMoment(
            id="m-spoiler",
            scene_id="sec-2:scene:0",
            title_ru="У маяка",
            reason_ru="Ключевой момент.",
            event_ru="Джон стоит под аркой.",
            prompt_en=(
                f"Dark curly hair and heavy wool peacoat. Style: {MONOCHROME_INK_DIRECTIVE}."
            ),
            insert_after_block_id="b1",
            anchors=(valid_anchor,),
            character_descriptions=(
                IllustrationAppearance(
                    entity_id="ent-jm",
                    name_ru="Джон Миллер",
                    description_ru="Со шрамом из главы 10.",
                    description_en="Dark curly hair and heavy wool peacoat.",
                    fact_ids=("f_jm_scar_ch10",),
                ),
            ),
        )
    )

    # 2. ID-only prompt without concrete English visual description rejected
    results["adv_id_only_prompt_without_description_en"] = _expect_rejected(
        ValidatedMoment(
            id="m-id-only",
            scene_id="sec-2:scene:0",
            title_ru="У маяка",
            reason_ru="Ключевой момент.",
            event_ru="Джон стоит под аркой.",
            prompt_en=f"ent-jm stands in the rain. Style: {MONOCHROME_INK_DIRECTIVE}.",
            insert_after_block_id="b1",
            anchors=(valid_anchor,),
            character_descriptions=(valid_appearance,),
        )
    )

    # 3. Foreign block_id rejected
    results["adv_foreign_block_id"] = _expect_rejected(
        ValidatedMoment(
            id="m-foreign",
            scene_id="sec-2:scene:0",
            title_ru="У маяка",
            reason_ru="Ключевой момент.",
            event_ru="Джон стоит под аркой.",
            prompt_en=f"Dark curly hair and heavy wool peacoat. Style: {MONOCHROME_INK_DIRECTIVE}.",
            insert_after_block_id="b1",
            anchors=(SourceAnchor(block_id="foreign-999", start=0, end=5, quote="Young"),),
            character_descriptions=(valid_appearance,),
        )
    )

    # 4. Broken quote (not an exact substring of block) rejected
    results["adv_broken_quote"] = _expect_rejected(
        ValidatedMoment(
            id="m-quote",
            scene_id="sec-2:scene:0",
            title_ru="У маяка",
            reason_ru="Ключевой момент.",
            event_ru="Джон стоит под аркой.",
            prompt_en=f"Dark curly hair and heavy wool peacoat. Style: {MONOCHROME_INK_DIRECTIVE}.",
            insert_after_block_id="b1",
            anchors=(SourceAnchor(block_id="b1", start=0, end=10, quote="NOT IN TXT"),),
            character_descriptions=(valid_appearance,),
        )
    )

    # 5. Duplicate selection candidate rejected by IllustrationSelection._validate_selection
    sel_scope = IllustrationScope(
        id="sc-sel",
        generation_id="gen-1",
        parent_scope_id=None,
        stage=IllustrationScopeStage.SELECTION,
        scope_key="sel:sec-2:r0:g0",
        section_id="sec-2",
        section_order=1,
        section_title="Chapter 2",
        state=IllustrationScopeState.RUNNING,
        task_id=None,
        reason_code=None,
        attempt_id=None,
        input_sha256="a" * 64,
        output_sha256=None,
        payload=None,
        summary={
            "max_select": 2,
            "expected_candidate_ids": ["c1", "c2"],
            "is_final_round": True,
        },
        created_at=datetime(2026, 11, 12, tzinfo=UTC),
        updated_at=datetime(2026, 11, 12, tzinfo=UTC),
    )
    dup_dto = IllustrationSelectionResponse(
        selected_candidate_ids=("c1", "c1"),
        selection_reasons_ru=(
            IllustrationSelectionReasonDTO(candidate_id="c1", reason_ru="Повтор."),
        ),
    )
    dup_outcome = IllustrationSelection.__new__(IllustrationSelection)._validate_selection(
        sel_scope, dup_dto
    )
    results["adv_duplicate_scene_moment"] = not dup_outcome.ok

    return results


def _build_document(fixture: dict[str, Any]) -> CanonicalDocument:
    book_spec = fixture["book"]
    sections: list[DocumentSection] = []
    for sec in book_spec["sections"]:
        blocks = tuple(
            DocumentBlock(
                id=blk["id"],
                section_id=sec["id"],
                type=BlockType.PARAGRAPH,
                order=idx,
                text=blk["text"],
                source_reference=SourceReference(f"Text/{sec['id']}.xhtml", f"p[{idx}]"),
                support_status=SupportStatus.SUPPORTED,
            )
            for idx, blk in enumerate(sec["blocks"])
        )
        sections.append(
            DocumentSection(
                id=sec["id"],
                document_id=book_spec["id"],
                title=sec["title"],
                order=sec["order"],
                source_resource_path=f"Text/{sec['id']}.xhtml",
                blocks=blocks,
            )
        )
    return CanonicalDocument(
        id=book_spec["id"],
        metadata=BookMetadata(title=book_spec["title"]),
        source_language=book_spec["source_language"],
        target_language=book_spec["target_language"],
        sections=tuple(sections),
    )


def run_fake_evaluation(fixture_path: Path, output_dir: Path) -> dict[str, Any]:
    """Execute full deterministic fake evaluation and write JSON + Markdown artifacts."""
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = output_dir / "runtime_data"
    fake_epub = output_dir / "synthetic_source.epub"
    fake_epub.write_bytes(b"PK\x03\x04synthetic-evaluation-epub")
    source_sha = hashlib.sha256(fake_epub.read_bytes()).hexdigest()

    fake_llm = SyntheticEvalLLMClient(fixture)
    app = build_application(
        settings=AppSettings(
            storage=StorageSettings(data_root=str(data_root)),
            llm=LLMSettings(
                api_key="synthetic-fake-key",
                model="fake-eval",
                staged_analysis=True,
                max_attempts=1,
            ),
        ),
        llm_client=fake_llm,
        content_plan_profile="content-plan-v3",
    )
    try:
        doc = _build_document(fixture)
        book_id = doc.id
        book = BookRecord(
            id=book_id,
            metadata=doc.metadata,
            source_language=doc.source_language,
            target_language=doc.target_language,
            status=BookStatus.IMPORTED,
            source_epub_path=str(fake_epub),
            created_at=datetime(2026, 11, 12, tzinfo=UTC),
        )
        with app.unit_of_work.transaction() as tx:
            tx.books.add(book)
            tx.documents.save(doc)
            app.book_conditions.create(tx, book, app.runtime_settings.revision)

        asyncio.run(app.analysis.start(book_id))
        asyncio.run(drain_tasks(app))

        app.illustrations.save_settings(
            book_id,
            expected_revision=0,
            settings=IllustrationSettings(
                enabled=True,
                max_per_chapter=2,
                model_reference="fake-eval",
                model_prompts="fake-eval",
            ),
        )
        asyncio.run(app.illustrations.prepare(book_id, expected_revision=1))
        asyncio.run(drain_tasks(app))

        view = app.illustrations.get(book_id, offset=0, limit=50)
        registry = app.illustrations.get_registry(book_id, offset=0, limit=100)

        # Verify Chapter 2 has NO Chapter 10 scar fact and Chapter 3 has 0 moments.
        ch2_view = app.illustrations.get(book_id, section_id="sec-02-harbor")
        ch3_view = app.illustrations.get(book_id, section_id="sec-03-conversation")
        ch10_view = app.illustrations.get(book_id, section_id="sec-10-storm")

        with app.unit_of_work.read_transaction() as tx:
            assert view.active_generation_id is not None
            gen = tx.illustrations.get_generation(book_id, view.active_generation_id)
            assert gen is not None
            scopes = tx.illustrations.list_scopes(gen.id)
            entity_items, _ = tx.illustrations.query_registry_items(
                gen.id, kind=IllustrationRegistryKind.ENTITY, offset=0, limit=50
            )
            fact_items, _ = tx.illustrations.query_registry_items(
                gen.id, kind=IllustrationRegistryKind.FACT, offset=0, limit=50
            )
            ch2_moments = ch2_view.selected_chapter.moments if ch2_view.selected_chapter else ()
            ch2_fact_ids = {
                fid
                for m in ch2_moments
                for desc in (*m.character_descriptions, *m.location_descriptions)
                for fid in desc.fact_ids
            }
            ch2_prompts = " ".join(m.prompt_en for m in ch2_moments)

            report: dict[str, Any] = {
                "mode": "fake",
                "disclaimer": (
                    "Fake evaluation proves deterministic pipeline mechanics, bounds and "
                    "adversarial rejection only; editorial quality of a real LLM requires "
                    "an explicit --mode live --allow-paid-calls run."
                ),
                "book_id": book_id,
                "generation_id": gen.id,
                "generation_state": gen.state.value,
                "hashes": {
                    "source_sha256": source_sha,
                    "manifest_sha256": gen.manifest_sha256,
                    "registry_sha256": gen.registry_sha256,
                },
                "bounds": {
                    "max_input_bytes_observed": fake_llm.max_input_bytes,
                    "max_request_bytes_limit": DEFAULT_ILLUSTRATION_POLICY.max_request_bytes,
                    "calls_sent": gen.calls_sent,
                    "max_calls_per_generation": DEFAULT_ILLUSTRATION_POLICY.max_calls_per_generation,
                    "total_scopes": len(scopes),
                    "chapter_headers": len(view.chapter_headers),
                },
                "invariants": {
                    "generation_ready": gen.state is IllustrationGenerationState.READY,
                    "empty_chapter_skipped": any(
                        h.section_id == "sec-01-empty" and h.state == "skipped"
                        for h in view.chapter_headers
                    ),
                    "ordinary_chapter_zero_moments": (
                        ch3_view.selected_chapter is not None
                        and len(ch3_view.selected_chapter.moments) == 0
                        and bool(ch3_view.selected_chapter.zero_reason_ru)
                    ),
                    "no_ch10_scar_in_ch2": (
                        "f_jm_scar_ch10" not in ch2_fact_ids and "scar" not in ch2_prompts.lower()
                    ),
                    "no_prompt_injection_leak": "api key" not in ch2_prompts.lower(),
                    "homonymous_entities_separated": len(
                        [
                            e
                            for e in entity_items
                            if "john" in json.dumps(dict(e.payload), ensure_ascii=False).lower()
                        ]
                    )
                    >= 2,
                    "registry_entities": len(entity_items),
                    "registry_facts": len(fact_items),
                    "registry_items_total": registry.pagination.total,
                    "ch2_moments": len(ch2_moments),
                    "ch10_moments": len(ch10_view.selected_chapter.moments)
                    if ch10_view.selected_chapter
                    else 0,
                },
                "adversarial_rejections": run_adversarial_checks(),
                "rubric": fixture["editorial_rubric"],
                "scopes_by_stage": {
                    stage.value: sum(1 for s in scopes if s.stage is stage)
                    for stage in IllustrationScopeStage
                },
            }
    finally:
        asyncio.run(app.aclose())

    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Illustration Evaluation Report (fake mode)",
                "",
                f"- Book: `{report['book_id']}`",
                f"- State: `{report['generation_state']}`",
                f"- Source SHA-256: `{report['hashes']['source_sha256']}`",
                f"- Manifest SHA-256: `{report['hashes']['manifest_sha256']}`",
                f"- Registry SHA-256: `{report['hashes']['registry_sha256']}`",
                f"- Calls sent: `{report['bounds']['calls_sent']}` / `{report['bounds']['max_calls_per_generation']}`",
                f"- Max input bytes: `{report['bounds']['max_input_bytes_observed']}` / `{report['bounds']['max_request_bytes_limit']}`",
                "",
                "## Note",
                report["disclaimer"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate optional book illustration preparation pipeline."
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        required=True,
        help="Path to synthetic_cases.json fixture file.",
    )
    parser.add_argument(
        "--mode",
        choices=("fake", "live"),
        default="fake",
        help="Execution mode: deterministic fake (default) or paid live LLM run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for evaluation artifacts (report.json, summary.md).",
    )
    parser.add_argument(
        "--allow-paid-calls",
        action="store_true",
        help="Explicit opt-in required for --mode live before any HTTP client is created.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional Linguary config.toml path for --mode live.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "live":
        if not args.allow_paid_calls:
            print(
                "Refusing --mode live without explicit --allow-paid-calls flag (0 HTTP calls made).",
                file=sys.stderr,
            )
            return 2
        settings = load_settings(config_path=args.config)
        if not settings.llm.api_key or not settings.llm.api_key.strip():
            print(
                "Refusing --mode live: provider API key is not configured (0 HTTP calls made).",
                file=sys.stderr,
            )
            return 2
        print(
            "Live evaluation requires approved control chapters (ADR §9); "
            "run against the selected EPUB with explicit approval.",
            file=sys.stderr,
        )
        return 2

    report = run_fake_evaluation(args.fixtures, args.output)
    print(json.dumps(report["hashes"], ensure_ascii=False))
    all_ok = (
        report["invariants"]["generation_ready"]
        and report["invariants"]["ch2_moments"] >= 1
        and report["invariants"]["ch10_moments"] >= 1
        and all(report["adversarial_rejections"].values())
        and report["invariants"]["no_ch10_scar_in_ch2"]
        and report["invariants"]["no_prompt_injection_leak"]
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
