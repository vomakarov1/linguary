"""Unit tests covering WP-1, WP-2, WP-3, WP-4, and WP-6 contracts from the reliability SDD."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, ReadingSettings, StorageSettings
from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.validation import Py3LangidDetector
from linguary.application.analysis_pass_cache import encode_pass
from linguary.composition.book_runtime import conditions_payload
from linguary.application.context_builder import (
    _CONTENT_PLAN_PROCESSING_V3,
    _NARRATIVE_ANALYSIS_INSTRUCTIONS,
    _TRANSLATION_CONTEXT_V2,
    ContextBuilder,
)
from linguary.application.illustration_chapter import (
    IllustrationChapter,
    _dedupe_semicolon_parts,
    _dedupe_texts,
    _select_english_name,
)
from linguary.application.illustration_context import (
    compact_safe_detail,
    is_context_limit_error,
    is_incompatible_route_error,
    pack_items_by_wire_budget,
)
from linguary.application.illustration_dto import (
    IllustrationCandidatesResponse,
    IllustrationDesignResponse,
    IllustrationDesignTraitDTO,
    IllustrationExtractResponse,
    IllustrationFactDTO,
    IllustrationFactScopeResponse,
    IllustrationMentionDTO,
    IllustrationPromptResponse,
)
from linguary.application.illustration_reference import (
    IllustrationReference,
    _build_bounded_design_input,
    resolve_appearance_with_design_fallback,
)
from linguary.application.illustration_selection import (
    IllustrationSelection,
    _scene_candidate_units,
)
from linguary.application.illustration_source import (
    IllustrationSource,
    SectionManifestEntry,
    _pass_model_dependencies,
)
from linguary.application.illustration_validation import (
    IllustrationAppearance,
    compose_monochrome_prompt,
    has_cyrillic,
)
from linguary.application.llm_dto import NarrativeAnalysisResponse, QuotedSceneResponse
from linguary.application.ports.illustrations import (
    IllustrationRuntimePort,
    RequestWireSizer,
    RequestWireSnapshot,
)
from linguary.application.ports.llm import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMResponseFormatError,
    LLMResponseKind,
)
from linguary.application.ports.persistence import BookRecord
from linguary.application.token_budget import narrative_max_fragment_blocks_for
from linguary.composition.illustration_runtime import IllustrationWireSizer
from linguary.composition.migrations import apply_migrations
from linguary.domain import (
    AnalysisPassKind,
    AnalysisPassResult,
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    ConceptType,
    DocumentBlock,
    DocumentSection,
    LLMCallStage,
    Scene,
    SourceReference,
    SupportStatus,
    TermConcept,
)
from linguary.domain.book_conditions import BookConditions
from linguary.domain.content_plan import canonical_sha
from linguary.domain.name_translation import NameTranslationPolicy, NameTranslationStrategy
from linguary.domain.illustrations import (
    ILLUSTRATION_STYLE_MONOCHROME_INK,
    IllustrationPolicy,
    IllustrationReasonCode,
    IllustrationScope,
    IllustrationScopeStage,
    IllustrationScopeState,
    IllustrationSettings,
    IllustrationSettingsRevision,
    derive_chapter_state,
    executable_scopes,
)


def _dummy_scope(
    *,
    stage: IllustrationScopeStage,
    state: IllustrationScopeState,
    section_id: str | None = "sec-0",
    reason_code: str | None = None,
    summary: dict[str, Any] | None = None,
) -> IllustrationScope:
    now = datetime(2026, 11, 25, tzinfo=UTC)
    return IllustrationScope(
        id=str(uuid4()),
        generation_id="gen-1",
        parent_scope_id=None,
        stage=stage,
        scope_key=f"{stage.value}:{uuid4().hex[:6]}",
        section_id=section_id,
        section_order=0,
        section_title="Chapter 1",
        task_id=None,
        input_sha256="a" * 64,
        state=state,
        reason_code=reason_code,
        attempt_id=None,
        output_sha256=None,
        payload=None,
        summary=summary or {},
        created_at=now,
        updated_at=now,
    )


# ── WP-2: Pure domain functions ─────────────────────────────────────────────


def test_executable_scopes_excludes_chapter_and_split() -> None:
    ch = _dummy_scope(stage=IllustrationScopeStage.CHAPTER, state=IllustrationScopeState.PENDING)
    split = _dummy_scope(stage=IllustrationScopeStage.CANDIDATES, state=IllustrationScopeState.SPLIT)
    failed = _dummy_scope(
        stage=IllustrationScopeStage.CANDIDATES,
        state=IllustrationScopeState.FAILED,
        reason_code=IllustrationReasonCode.MODEL_OUTPUT_INVALID.value,
    )
    accepted = _dummy_scope(stage=IllustrationScopeStage.EXTRACT, state=IllustrationScopeState.ACCEPTED)
    result = executable_scopes([ch, split, failed, accepted])
    assert [s.id for s in result] == [failed.id, accepted.id]


def test_derive_chapter_state_transitions() -> None:
    assert derive_chapter_state(IllustrationScopeState.ACCEPTED, None, ()) == ("ready", None)
    assert derive_chapter_state(IllustrationScopeState.SKIPPED, "non_content_section", ()) == (
        "skipped",
        "non_content_section",
    )
    assert derive_chapter_state(IllustrationScopeState.FAILED, "provider_error", ()) == (
        "failed",
        "provider_error",
    )
    assert derive_chapter_state(IllustrationScopeState.RUNNING, None, ()) == ("running", None)
    assert derive_chapter_state(IllustrationScopeState.SPLIT, None, ()) == ("running", None)
    # PENDING chapter reflects executable children: running > queued > failed > queued.
    assert derive_chapter_state(
        IllustrationScopeState.PENDING,
        None,
        [
            (IllustrationScopeState.FAILED, "provider_error"),
            (IllustrationScopeState.RUNNING, None),
        ],
    ) == ("running", None)
    assert derive_chapter_state(
        IllustrationScopeState.PENDING,
        None,
        [
            (IllustrationScopeState.FAILED, "provider_error"),
            (IllustrationScopeState.PENDING, None),
        ],
    ) == ("queued", None)
    assert derive_chapter_state(
        IllustrationScopeState.PENDING,
        None,
        [
            (IllustrationScopeState.ACCEPTED, None),
            (IllustrationScopeState.FAILED, "route_unavailable"),
        ],
    ) == ("failed", "route_unavailable")
    assert derive_chapter_state(IllustrationScopeState.PENDING, None, ()) == ("queued", None)


# ── WP-3: Packing max_items, concise source_unit_id, validation, composite split ──


class _TinySizer(RequestWireSizer):
    def measure(self, request: LLMRequest[Any]) -> RequestWireSnapshot:
        raw = request.input_text.encode("utf-8")
        return RequestWireSnapshot(sha256=sha256(raw).hexdigest(), size_bytes=len(raw))


def test_pack_items_by_wire_budget_enforces_max_items_and_validates_argument() -> None:
    items = [f"b{i}" for i in range(45)]

    def _build(batch: Any) -> LLMRequest[Any]:
        return LLMRequest(
            kind=LLMResponseKind.ILLUSTRATION_CANDIDATES,
            instructions="Pick.",
            input_text=json.dumps(list(batch)),
        )

    pages = pack_items_by_wire_budget(
        items,
        build_request=_build,
        sizer=_TinySizer(),
        target_bytes=100_000,
        max_items=24,
    )
    assert [len(p) for p in pages] == [24, 21]
    assert [x for p in pages for x in p] == items

    for bad in (0, -5, True, "24"):
        with pytest.raises(ValueError):
            pack_items_by_wire_budget(
                items,
                build_request=_build,
                sizer=_TinySizer(),
                target_bytes=100_000,
                max_items=bad,  # type: ignore[arg-type]
            )


def test_scene_candidate_units_uses_block_id_as_source_unit_id() -> None:
    blocks = tuple(
        DocumentBlock(
            id=f"blk-{i}",
            section_id="sec-0",
            type=BlockType.PARAGRAPH,
            order=i,
            text=f"Sentence {i} in scene.",
            source_reference=SourceReference("Text/ch1.xhtml", f"p[{i}]"),
            support_status=SupportStatus.SUPPORTED,
        )
        for i in range(3)
    )
    section = DocumentSection(
        id="sec-0",
        document_id="book-1",
        title="Chapter 1",
        order=0,
        source_resource_path="Text/ch1.xhtml",
        blocks=blocks,
    )
    # Branch 1: explicit scenes
    entry_with_scenes = SectionManifestEntry(
        section_id="sec-0",
        order=0,
        title="Chapter 1",
        skipped=False,
        skip_reason=None,
        block_digests={b.id: "a" * 64 for b in blocks},
        scenes=(
            Scene(
                id="sc-0",
                order=0,
                start_block_id="blk-0",
                end_block_id="blk-2",
                summary="Scene 0",
                location="Hall",
                participants=("Alice",),
            ),
        ),
    )
    units_with = _scene_candidate_units(entry_with_scenes, section)
    assert [b["source_unit_id"] for b in units_with[0]["blocks"]] == ["blk-0", "blk-1", "blk-2"]

    # Branch 2: synthetic scene without narrative scenes
    entry_no_scenes = SectionManifestEntry(
        section_id="sec-0",
        order=0,
        title="Chapter 1",
        skipped=False,
        skip_reason=None,
        block_digests={b.id: "a" * 64 for b in blocks},
        scenes=(),
    )
    units_without = _scene_candidate_units(entry_no_scenes, section)
    assert [b["source_unit_id"] for b in units_without[0]["blocks"]] == ["blk-0", "blk-1", "blk-2"]


def test_validate_candidates_legacy_block_id_fallback_and_missing_extra_hint() -> None:
    selector = IllustrationSelection.__new__(IllustrationSelection)
    legacy_scope = _dummy_scope(
        stage=IllustrationScopeStage.CANDIDATES,
        state=IllustrationScopeState.RUNNING,
        summary={
            "scene_id": "sc-0",
            "expected_source_unit_ids": ["sc-0:b:blk-0", "sc-0:b:blk-1"],
            "ordered_block_ids": ["blk-0", "blk-1"],
            "scene_start_block_id": "blk-0",
            "scene_end_block_id": "blk-1",
            "max_candidates": 2,
        },
    )
    # 1-to-1 bare block_ids accepted for backward compatibility
    ok_dto = IllustrationCandidatesResponse(
        scene_id="sc-0",
        events=(),
        zero_reason_ru="Переходный эпизод.",
        covered_source_unit_ids=("blk-0", "blk-1"),
        complete=True,
    )
    ok_res = selector._validate_candidates(legacy_scope, ok_dto, {})
    assert ok_res.ok is True

    # Missing/extra IDs reported with split_eligible=True
    bad_dto = IllustrationCandidatesResponse(
        scene_id="sc-0",
        events=(),
        zero_reason_ru="Переходный эпизод.",
        covered_source_unit_ids=("sc-0:b:blk-0", "unexpected-id"),
        complete=True,
    )
    bad_res = selector._validate_candidates(legacy_scope, bad_dto, {})
    assert bad_res.ok is False
    assert bad_res.split_eligible is True
    assert "пропущены=" in bad_res.repair_reason
    assert "лишние=" in bad_res.repair_reason


@dataclass
class _StubRuntime(IllustrationRuntimePort):
    handler: Callable[[LLMRequest[Any], int], Any]
    calls: int = 0

    def is_provider_ready(self) -> bool:
        return True

    def default_model(self) -> str | None:
        return "gpt-5-mini"

    def route_id(self) -> str:
        return "test-route"

    def sizer_for(self, revision: IllustrationSettingsRevision, stage: LLMCallStage) -> RequestWireSizer:
        del revision, stage
        return IllustrationWireSizer(
            settings=LLMSettings(model="gpt-5-mini", api_key="k", temperature=None, max_output_tokens=16384),
            include_openrouter_provider=False,
        )

    def client_for(self, revision: IllustrationSettingsRevision, stage: LLMCallStage) -> Any:
        del revision, stage
        parent = self

        class _Client:
            async def generate(self, request: LLMRequest[Any]) -> LLMResponse[Any]:
                parent.calls += 1
                result = parent.handler(request, parent.calls)
                if isinstance(result, Exception):
                    raise result
                return LLMResponse(
                    payload=result,
                    provider="fake",
                    model="gpt-5-mini",
                    call_context=request.call_context,
                )

        return _Client()


def _seed_simple_book(
    tmp_path: Path,
    blocks_spec: list[tuple[str, str]],
    *,
    policy: IllustrationPolicy | None = None,
    glossary_terms: Sequence[tuple[str, str]] = (),
):
    db = tmp_path / "rel.db"
    apply_migrations(db)
    uow = SqlAlchemyUnitOfWork.for_engine(create_database_engine(db))
    book_id = "book-rel"
    sec_id = "sec-0"
    blocks = tuple(
        DocumentBlock(
            id=bid,
            section_id=sec_id,
            type=BlockType.PARAGRAPH,
            order=idx,
            text=txt,
            source_reference=SourceReference("Text/ch1.xhtml", f"p[{idx}]"),
            support_status=SupportStatus.SUPPORTED,
        )
        for idx, (bid, txt) in enumerate(blocks_spec)
    )
    doc = CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title="Rel Book"),
        source_language="en",
        target_language="ru",
        sections=(
            DocumentSection(
                id=sec_id,
                document_id=book_id,
                title="Chapter 1",
                order=0,
                source_resource_path="Text/ch1.xhtml",
                blocks=blocks,
            ),
        ),
    )
    epub_file = tmp_path / "rel.epub"
    epub_file.write_bytes(b"fake-epub-content")
    book = BookRecord(
        id=book_id,
        metadata=BookMetadata(title="Rel Book"),
        source_language="en",
        target_language="ru",
        status=BookStatus.GLOSSARY_REVIEW,
        source_epub_path=str(epub_file),
        created_at=datetime(2026, 11, 25, tzinfo=UTC),
    )
    app_settings = AppSettings(
        llm=LLMSettings(
            provider="openai_compatible",
            base_url="https://api.openai.com/v1",
            model="gpt-5-mini",
            api_key="test-key",
            max_output_tokens=16384,
            staged_analysis=True,
        ),
        checks=CheckSettings(sequential_analysis=False),
        reading=ReadingSettings(reader_level="b1", name_translation_strategy="sound"),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
    )
    conditions_id = str(uuid4())
    cond_payload = conditions_payload(book, app_settings, revision=1, profile="content-plan-v3")
    cond_payload["source_sha256"] = "a" * 64
    prompts_version = int(cond_payload["versions"]["prompts"])
    ctx_builder = ContextBuilder(
        content_plan_profile="content-plan-v3",
        name_translation_policy=NameTranslationPolicy(
            version=1, strategy=NameTranslationStrategy.SOUND
        ),
        reader_level="b1",
    )
    model_deps = _pass_model_dependencies(cond_payload)
    max_frag_blocks = narrative_max_fragment_blocks_for(cond_payload["llm_parameters"])

    with uow.transaction() as tx:
        tx.books.add(book)
        tx.documents.save(doc)
        tx.work.add_conditions(
            BookConditions(
                id=conditions_id,
                book_id=book_id,
                revision=1,
                payload=cond_payload,
                created_at=datetime(2026, 11, 25, tzinfo=UTC),
            )
        )
        for src, _trans in glossary_terms:
            tx.glossary.save(
                TermConcept(
                    id=str(uuid4()),
                    book_id=book_id,
                    type=ConceptType.PERSON,
                    canonical_source_text=src,
                )
            )
        sec = doc.ordered_sections[0]
        narrative = NarrativeAnalysisResponse(
            translation_instructions="Сохраняй стиль.",
            chapter_summary="Characters appear in the scene.",
            glossary_candidates=(),
            context_facts=(),
            scenes=(
                QuotedSceneResponse(
                    start_block_id=blocks[0].id,
                    end_block_id=blocks[-1].id,
                    summary="Сцена",
                    location="Дом",
                    participants=("Alice",),
                    visual_facts=(),
                    visual_assumptions=(),
                    continues_previous_scene=False,
                ),
            ),
        )
        encoded = encode_pass(narrative)
        encoded["input_suggestions"] = []
        encoded["input_context"] = {
            "schema_version": 1,
            "sequential_analysis": False,
            "parallel_batch": False,
            "seed_active": False,
            "user_decisions": [],
            "nonbinding_suggestions": [],
            "prompts_version": prompts_version,
        }
        input_sha = ctx_builder.narrative_input_sha256(
            doc,
            sec,
            (),
            (),
            (),
            model_dependencies=model_deps,
            max_fragment_blocks=max_frag_blocks,
        )
        tx.analyses.save_pass(
            AnalysisPassResult(
                id=f"pass-{sec_id}",
                book_id=book_id,
                section_id=sec_id,
                pass_kind=AnalysisPassKind.NARRATIVE,
                scope_key="",
                input_sha256=input_sha,
                output_sha256=canonical_sha(encoded),
                conditions_id=conditions_id,
                attempt_id=None,
                payload=encoded,
            )
        )
        settings = IllustrationSettings(
            enabled=True,
            max_per_chapter=2,
            model_reference="gpt-5-mini",
            model_prompts="gpt-5-mini",
            policy=policy or IllustrationPolicy(max_repair_turns=1, max_calls_per_scope_activation=3),
        )
        rev = tx.illustrations.save_settings(book_id, expected_revision=0, settings=settings)
        source = IllustrationSource(
            language_detector=Py3LangidDetector(),
            source_digest_fn=lambda _p: "a" * 64,
        )
        snap = source.snapshot(tx, book_id)
        assert snap.ready
        gen, _ = tx.illustrations.create_or_get_generation(
            book_id,
            settings_id=rev.id,
            request_key="req-rel",
            source_sha256=snap.source_sha256 or "",
            source_authority=snap.source_authority or {},
            requested_sections=(sec_id,),
        )
        if snap.manifest is not None and snap.manifest_sha256 is not None:
            gen = tx.illustrations.seal_manifest(
                book_id, gen.id, manifest=snap.manifest, manifest_sha256=snap.manifest_sha256
            )
        gen = tx.illustrations.update_generation_state(
            book_id, gen.id, state=gen.state.QUEUED, reason_code=None
        )
    return uow, source, book_id, sec_id, gen.id, rev


async def test_composite_candidates_scope_splits_after_exhausting_repair_turns(
    tmp_path: Path,
) -> None:
    blocks = [(f"b-{i}", f"Block {i} narrative text here.") for i in range(4)]
    uow, source, book_id, sec_id, gen_id, _rev = _seed_simple_book(
        tmp_path,
        blocks,
        policy=IllustrationPolicy(max_repair_turns=1, max_calls_per_scope_activation=3),
    )

    def _handler(req: LLMRequest[Any], call_no: int) -> Any:
        del call_no
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            block_ids = [b["source_unit_id"] for b in data["blocks"]]
            # Fail coverage on 4-block composite chunk (both initial + 1 repair turn),
            # then succeed on 2-block split children!
            if len(block_ids) > 2:
                return IllustrationCandidatesResponse(
                    scene_id=data["scene_id"],
                    events=(),
                    zero_reason_ru="Нет событий.",
                    covered_source_unit_ids=(block_ids[0],),
                    complete=True,
                )
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(),
                zero_reason_ru="Нет событий.",
                covered_source_unit_ids=tuple(block_ids),
                complete=True,
            )
        raise AssertionError(f"unexpected kind: {req.kind}")

    runtime = _StubRuntime(handler=_handler)
    selection = IllustrationSelection(
        uow,
        source=source,
        runtime=runtime,
        sleep_fn=lambda _: None,
    )
    selection.plan_scopes(gen_id, sec_id)

    for _ in range(10):
        with uow.read_transaction() as tx:
            pending = [t for t in tx.tasks.list_for_book(book_id) if t.status.value == "pending"]
        if not pending:
            break
        for task in pending:
            await selection.run_task(task)

    with uow.read_transaction() as tx:
        cand_scopes = tx.illustrations.list_scopes(
            gen_id, stage=IllustrationScopeStage.CANDIDATES, section_id=sec_id
        )
    split_parents = [s for s in cand_scopes if s.state is IllustrationScopeState.SPLIT]
    accepted_children = [
        s for s in cand_scopes if s.state is IllustrationScopeState.ACCEPTED and s.parent_scope_id
    ]
    assert len(split_parents) == 1
    assert len(accepted_children) == 2
    assert all(len(c.summary["ordered_block_ids"]) == 2 for c in accepted_children)
    final_sel = selection.selected_moments(gen_id, sec_id)
    assert final_sel.winners == ()


async def test_provider_404_classified_as_route_unavailable_without_retries(tmp_path: Path) -> None:
    uow, source, book_id, _sec_id, gen_id, _rev = _seed_simple_book(
        tmp_path,
        [("b-0", "Alice stepped into the hall.")],
    )

    def _handler(_req: LLMRequest[Any], _call_no: int) -> Any:
        return LLMProviderError(
            "провайдер отверг запрос: 404 - No endpoints found that can handle the requested parameters"
        )

    runtime = _StubRuntime(handler=_handler)
    ref = IllustrationReference(uow, source=source, runtime=runtime, sleep_fn=lambda _: None)
    with uow.transaction() as tx:
        gen = tx.illustrations.get_generation(book_id, gen_id)
        assert gen is not None
        snap = source.snapshot(tx, book_id)
        ref.plan_scopes(tx, gen, snap)

    with uow.read_transaction() as tx:
        pending = [t for t in tx.tasks.list_for_book(book_id) if t.status.value == "pending"]
    assert len(pending) == 1
    await ref.run_task(pending[0])

    assert runtime.calls == 1
    with uow.read_transaction() as tx:
        scopes = tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.EXTRACT)
        assert scopes[0].state is IllustrationScopeState.FAILED
        assert scopes[0].reason_code == IllustrationReasonCode.ROUTE_UNAVAILABLE.value
        gen_after = tx.illustrations.get_generation(book_id, gen_id)
        assert gen_after is not None
        assert gen_after.reason_code == IllustrationReasonCode.ROUTE_UNAVAILABLE.value


# ── WP-4: DESIGN monochrome_ink, trait suppression, English prompt, dedupe, ALIAS ──


def test_validate_design_rejects_color_words_for_monochrome_ink_and_accepts_tonal() -> None:
    ref = IllustrationReference.__new__(IllustrationReference)
    design_scope = _dummy_scope(
        stage=IllustrationScopeStage.DESIGN,
        state=IllustrationScopeState.RUNNING,
        summary={
            "entity_id": "ent-1",
            "entity_kind": "person",
            "style": ILLUSTRATION_STYLE_MONOCHROME_INK,
        },
    )
    bad_en = IllustrationDesignResponse(
        entity_id="ent-1",
        traits=(
            IllustrationDesignTraitDTO(
                trait="palette_value",
                description_ru="Сдержанные тона",
                description_en="muted beige and green palette",
                assumption_ru="Предположение",
            ),
        ),
    )
    res_en = ref._validate_design(design_scope, bad_en)
    assert res_en.ok is False
    assert "monochrome_ink" in res_en.repair_reason

    bad_ru = IllustrationDesignResponse(
        entity_id="ent-1",
        traits=(
            IllustrationDesignTraitDTO(
                trait="palette_value",
                description_ru="Золотые и зелёные тона",
                description_en="high contrast shading",
                assumption_ru="Предположение",
            ),
        ),
    )
    res_ru = ref._validate_design(design_scope, bad_ru)
    assert res_ru.ok is False

    good = IllustrationDesignResponse(
        entity_id="ent-1",
        traits=(
            IllustrationDesignTraitDTO(
                trait="palette_value",
                description_ru="Контрастная чёрно-белая тушь с чёткой штриховкой",
                description_en="high-contrast black ink with crisp white highlights",
                assumption_ru="Выдержано в монохромном профиле",
            ),
        ),
    )
    res_ok = ref._validate_design(design_scope, good)
    assert res_ok.ok is True


def test_build_bounded_design_input_includes_style() -> None:
    payload = _build_bounded_design_input(
        entity={"entity_id": "ent-1", "canonical_name": "Alice", "entity_kind": "person"},
        facts=(),
        sizer=_TinySizer(),
        book_id="book-1",
        conditions_id=None,
        policy=IllustrationPolicy(),
        style=ILLUSTRATION_STYLE_MONOCHROME_INK,
    )
    assert payload["style"] == ILLUSTRATION_STYLE_MONOCHROME_INK


def test_resolve_appearance_with_design_fallback_word_boundaries_and_no_false_positives() -> None:
    design = [
        {"trait": "hair", "description_en": "Short dark hair", "description_ru": "Короткие волосы"},
        {"trait": "age_appearance", "description_en": "Middle-aged", "description_ru": "Средних лет"},
        {"trait": "build", "description_en": "Average build", "description_ru": "Среднее телосложение"},
        {"trait": "attire_default", "description_en": "Simple coat", "description_ru": "Простое пальто"},
        {"trait": "eyes", "description_en": "Calm gaze", "description_ru": "Спокойный взгляд"},
    ]
    # Positive suppressions (EN & RU)
    _, kept_beard_en = resolve_appearance_with_design_fallback(
        [{"description_en": "shaggy beard", "description_ru": ""}], design
    )
    assert "hair" not in {t["trait"] for t in kept_beard_en}

    _, kept_beard_ru = resolve_appearance_with_design_fallback(
        [{"description_en": "", "description_ru": "густая борода"}], design
    )
    assert "hair" not in {t["trait"] for t in kept_beard_ru}

    _, kept_infant = resolve_appearance_with_design_fallback(
        [{"description_en": "infant", "description_ru": "младенец"}], design
    )
    assert "age_appearance" not in {t["trait"] for t in kept_infant}

    _, kept_giant = resolve_appearance_with_design_fallback(
        [{"description_en": "giant", "description_ru": "великан"}], design
    )
    assert "build" not in {t["trait"] for t in kept_giant}

    # False-positive guards: managed, bundle, parties, permanent, big eyes
    _, kept_fp = resolve_appearance_with_design_fallback(
        [
            {
                "description_en": "managed a bundle at parties with permanent big eyes",
                "description_ru": "управлял делами на вечеринках",
            }
        ],
        design,
    )
    kept_traits = {t["trait"] for t in kept_fp}
    assert "hair" in kept_traits  # "bundle" must NOT match "bun"
    assert "age_appearance" in kept_traits  # "managed" must NOT match "aged"
    assert "attire_default" in kept_traits  # "parties" must NOT match "tie"
    assert "build" in kept_traits  # "big eyes" must NOT match "build"
    assert "eyes" not in kept_traits  # "big eyes" SHOULD match "eyes"


def test_compose_monochrome_prompt_uses_name_en_and_avoids_cyrillic() -> None:
    app_en = IllustrationAppearance(
        entity_id="ent-1",
        name_ru="Алиса",
        name_en="Alice",
        description_ru="В плаще",
        description_en="wearing a travelling cloak",
    )
    assert app_en.to_dict()["name_en"] == "Alice"
    assert IllustrationAppearance.from_dict(app_en.to_dict()).name_en == "Alice"

    app_cyr_only = IllustrationAppearance(
        entity_id="ent-2",
        name_ru="Незнакомец",
        name_en=None,
        description_ru="В капюшоне",
        description_en="hooded figure standing by the gate",
    )
    assert "name_en" not in app_cyr_only.to_dict()
    assert IllustrationAppearance.from_dict(app_cyr_only.to_dict()).name_en is None

    prompt = compose_monochrome_prompt("Wide ink view at dusk.", [app_en, app_cyr_only])
    assert "Alice: wearing a travelling cloak" in prompt
    assert "hooded figure standing by the gate" in prompt
    assert not has_cyrillic(prompt)

    # _select_english_name prefers non-Cyrillic canonical_name or alias
    assert _select_english_name({"canonical_name": "Алиса", "aliases": ["Алиса", "Alice"]}) == "Alice"
    assert _select_english_name({"canonical_name": "Незнакомец", "aliases": ["Путник"]}) is None


def test_validate_render_response_rejects_cyrillic_in_prompt_scene_composition() -> None:
    chapter = IllustrationChapter.__new__(IllustrationChapter)
    prompt_scope = _dummy_scope(
        stage=IllustrationScopeStage.PROMPT,
        state=IllustrationScopeState.RUNNING,
        summary={
            "moment_id": "m1",
            "allowed_fact_ids": [],
            "allowed_design_keys": [],
            "characters": [],
            "locations": [],
        },
    )
    rev = IllustrationSettingsRevision(
        id="rev-1",
        book_id="book-1",
        revision=1,
        settings_sha256="a" * 64,
        settings=IllustrationSettings(enabled=True, model_reference="m", model_prompts="m"),
        created_at=datetime(2026, 11, 25, tzinfo=UTC),
    )
    bad = IllustrationPromptResponse(
        moment_id="m1",
        scene_composition_en="Wide ink drawing of Гарри on the porch.",
        used_fact_ids=(),
        used_design_keys=(),
        assumptions_ru=(),
    )
    res = chapter._validate_render_response(prompt_scope, bad, rev)
    assert res.ok is False
    assert "кириллицы" in res.repair_reason


def test_dedupe_texts_and_semicolon_parts() -> None:
    assert _dedupe_texts(
        ["Тёмные волосы.", "  тёмные   волосы ", "Серый плащ;", "серый плащ"]
    ) == ("Тёмные волосы.", "Серый плащ;")
    assert _dedupe_semicolon_parts(
        ["Dark hair; grey cloak.", "dark hair; lantern in hand"]
    ) == ["Dark hair", "grey cloak.", "lantern in hand"]


async def test_alias_pair_payload_includes_alias_of_hint_and_glossary_concept_id(
    tmp_path: Path,
) -> None:
    uow, source, book_id, _sec_id, gen_id, _rev = _seed_simple_book(
        tmp_path,
        [
            ("b-0", "Petunia looked over the garden fence."),
            ("b-1", "Mrs Dursley poured the tea quietly."),
        ],
        glossary_terms=[("Petunia", "Петунья")],
    )
    alias_requests: list[dict[str, Any]] = []

    def _handler(req: LLMRequest[Any], _call_no: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_EXTRACT:
            u = data["units"][0]
            return IllustrationExtractResponse(
                mentions=(
                    IllustrationMentionDTO(
                        mention_id="m1",
                        name="Petunia",
                        entity_kind="person",
                        alias_of_hint="Mrs Dursley",
                        quote_block_id="b-0",
                        quote="Petunia looked",
                    ),
                    IllustrationMentionDTO(
                        mention_id="m2",
                        name="Mrs Dursley",
                        entity_kind="person",
                        alias_of_hint=None,
                        quote_block_id="b-1",
                        quote="Mrs Dursley poured",
                    ),
                ),
                facts=(),
                covered_source_unit_ids=(u["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_ALIAS:
            alias_requests.append(data)
            raise LLMProviderError("провайдер отверг запрос: 404 - stop after capture")
        raise AssertionError(f"unexpected kind: {req.kind}")

    runtime = _StubRuntime(handler=_handler)
    ref = IllustrationReference(uow, source=source, runtime=runtime, sleep_fn=lambda _: None)
    with uow.transaction() as tx:
        gen = tx.illustrations.get_generation(book_id, gen_id)
        assert gen is not None
        snap = source.snapshot(tx, book_id)
        ref.plan_scopes(tx, gen, snap)

    for _ in range(4):
        with uow.read_transaction() as tx:
            pending = [t for t in tx.tasks.list_for_book(book_id) if t.status.value == "pending"]
        if not pending:
            break
        for task in pending:
            await ref.run_task(task)

    assert len(alias_requests) == 1
    pair = alias_requests[0]["pairs"][0]
    sides = [pair["left"], pair["right"]]
    petunia_side = next(s for s in sides if s["name"] == "Petunia")
    assert petunia_side["alias_of_hint"] == "Mrs Dursley"
    assert isinstance(petunia_side.get("glossary_concept_id"), str)


def test_context_builder_prompt_additions_format_safely() -> None:
    formatted = _NARRATIVE_ANALYSIS_INSTRUCTIONS.format(
        source_language="en", target_language="ru"
    )
    assert 'type="person"' in formatted
    assert "5–25 блоков" in formatted
    assert "Прайвет-драйв" in formatted
    for tmpl in (_TRANSLATION_CONTEXT_V2, _CONTENT_PLAN_PROCESSING_V3):
        assert "speech_style" in tmpl
        assert 'help_kind="explanation"' in tmpl
        assert "glossary_hints.original" in tmpl


# ── WP-6: compact_safe_detail ───────────────────────────────────────────────


def test_compact_safe_detail_trims_utf8_cleanly_without_repair_prefix() -> None:
    msg = "  неполный   список covered_source_unit_ids  "
    assert compact_safe_detail(msg, max_bytes=256) == "неполный список covered_source_unit_ids"
    assert not compact_safe_detail(msg).startswith("Исправь ответ:")

    long_cyr = "Ошибка " * 100
    trimmed = compact_safe_detail(long_cyr, max_bytes=64)
    assert len(trimmed.encode("utf-8")) <= 64
    assert trimmed.endswith("...")
    assert compact_safe_detail("   ") == ""


# ── Code review regression tests ────────────────────────────────────────────


def test_context_limit_error_does_not_false_match_413_substring() -> None:
    err_404_with_413_id = LLMProviderError(
        "провайдер отверг запрос: 404 - No endpoints found (req_94130)"
    )
    assert is_context_limit_error(err_404_with_413_id) is False
    assert is_incompatible_route_error(err_404_with_413_id) is True

    err_real_413 = LLMProviderError("провайдер отверг запрос: 413 - Payload Too Large")
    assert is_context_limit_error(err_real_413) is True
    assert is_incompatible_route_error(err_real_413) is False


def test_select_english_name_ignores_none_values() -> None:
    assert _select_english_name({"canonical_name": None, "aliases": [None, "Дом Дёрсли"]}) is None
    assert _select_english_name({"canonical_name": None, "aliases": [None, "Privet Drive"]}) == "Privet Drive"


def test_trait_patterns_match_compound_haired_and_avoid_ru_false_positive() -> None:
    design = [{"trait": "hair", "description_en": "short neat hair", "description_ru": "короткие волосы", "assumption_ru": ""}]
    _, kept_dark_haired = resolve_appearance_with_design_fallback(
        [{"description_en": "a dark-haired boy", "description_ru": "мальчик"}],
        design,
    )
    assert kept_dark_haired == ()

    _, kept_false_ru = resolve_appearance_with_design_fallback(
        [{"description_en": "sleeping quietly", "description_ru": "усыплённый кот на коврике"}],
        design,
    )
    assert len(kept_false_ru) == 1


def test_extract_design_and_fact_scope_reject_cyrillic_in_english_descriptions(tmp_path: Path) -> None:
    uow, source, _book_id, sec_id, gen_id, rev = _seed_simple_book(
        tmp_path, [("b-0", "Harry looked at the house.")]
    )
    ref = IllustrationReference(uow, source=source, runtime=_StubRuntime(handler=lambda _r, _n: None))
    chapter = IllustrationChapter(uow, source=source, runtime=_StubRuntime(handler=lambda _r, _n: None))
    now = datetime(2026, 11, 25, tzinfo=UTC)

    # EXTRACT rejects Cyrillic in fact.description_en
    with uow.read_transaction() as tx:
        block_lookup = ref._load_block_lookup(tx, _book_id, source.snapshot(tx, _book_id))
    extract_scope = IllustrationScope(
        id="sc-ext",
        generation_id=gen_id,
        parent_scope_id=None,
        stage=IllustrationScopeStage.EXTRACT,
        scope_key="extract:p0000",
        section_id=sec_id,
        section_order=0,
        section_title="Chapter 1",
        task_id=None,
        input_sha256="a" * 64,
        state=IllustrationScopeState.RUNNING,
        reason_code=None,
        attempt_id=None,
        output_sha256=None,
        payload=None,
        summary={
            "expected_source_unit_ids": ["u0"],
            "ordered_block_ids": ["b-0"],
        },
        created_at=now,
        updated_at=now,
    )
    ext_res = ref._validate_extract(
        extract_scope,
        IllustrationExtractResponse(
            mentions=(
                IllustrationMentionDTO(
                    mention_id="m1",
                    name="Harry",
                    entity_kind="person",
                    alias_of_hint="   ",
                    quote_block_id="b-0",
                    quote="Harry looked",
                ),
            ),
            facts=(
                IllustrationFactDTO(
                    fact_id="f1",
                    subject_mention_id="m1",
                    temporal_scope="permanent",
                    back_applicable=True,
                    reveals_future_spoiler=False,
                    description_ru="Мальчик смотрит на дом.",
                    description_en="Мальчик looking at the house.",
                    quote_block_id="b-0",
                    quote="Harry looked",
                ),
            ),
            covered_source_unit_ids=("u0",),
            complete=True,
        ),
        block_lookup,
    )
    assert ext_res.ok is False
    assert "кириллиц" in ext_res.repair_reason

    # DESIGN rejects Cyrillic in trait.description_en
    design_scope = IllustrationScope(
        id="sc-des",
        generation_id=gen_id,
        parent_scope_id=None,
        stage=IllustrationScopeStage.DESIGN,
        scope_key="design:ent-1",
        section_id=None,
        section_order=None,
        section_title=None,
        task_id=None,
        input_sha256="b" * 64,
        state=IllustrationScopeState.RUNNING,
        reason_code=None,
        attempt_id=None,
        output_sha256=None,
        payload=None,
        summary={
            "entity_id": "ent-1",
            "entity_kind": "person",
            "style": "monochrome_ink",
        },
        created_at=now,
        updated_at=now,
    )
    des_res = ref._validate_design(
        design_scope,
        IllustrationDesignResponse(
            entity_id="ent-1",
            traits=(
                IllustrationDesignTraitDTO(
                    trait="hair",
                    description_ru="Короткие волосы",
                    description_en="Короткие dark hair",
                    assumption_ru="Нейтральная причёска",
                ),
            ),
        ),
    )
    assert des_res.ok is False
    assert "кириллиц" in des_res.repair_reason

    # FACT_SCOPE rejects Cyrillic in compact_description_en
    fs_scope = IllustrationScope(
        id="sc-fs",
        generation_id=gen_id,
        parent_scope_id=None,
        stage=IllustrationScopeStage.FACT_SCOPE,
        scope_key="fact_scope:sec-0:m1:ent-1:p000",
        section_id=sec_id,
        section_order=0,
        section_title="Chapter 1",
        task_id=None,
        input_sha256="c" * 64,
        state=IllustrationScopeState.RUNNING,
        reason_code=None,
        attempt_id=None,
        output_sha256=None,
        payload=None,
        summary={
            "moment_id": "m1",
            "valid_fact_ids": ["f1"],
            "running_applicable_ids": [],
            "max_applicable_fact_ids": 32,
        },
        created_at=now,
        updated_at=now,
    )
    fs_res = chapter._validate_render_response(
        fs_scope,
        IllustrationFactScopeResponse(
            event_id="m1",
            applicable_fact_ids=("f1",),
            excluded_fact_ids=(),
            compact_description_ru="Дом Дёрсли.",
            compact_description_en="Дом Дёрсли suburban house.",
            assumptions_ru=(),
            complete=True,
        ),
        rev,
    )
    assert fs_res.ok is False
    assert "кириллиц" in fs_res.repair_reason


async def test_composite_candidates_scope_splits_after_format_error_exhausts_repairs(
    tmp_path: Path,
) -> None:
    blocks = [(f"b-{i}", f"Block {i} narrative text here.") for i in range(4)]
    uow, source, book_id, sec_id, gen_id, _rev = _seed_simple_book(
        tmp_path,
        blocks,
        policy=IllustrationPolicy(max_repair_turns=1, max_calls_per_scope_activation=3),
    )

    def _handler(req: LLMRequest[Any], _call_no: int) -> Any:
        data = json.loads(req.input_text)
        block_ids = [b["source_unit_id"] for b in data["blocks"]]
        if len(block_ids) > 2:
            return LLMResponseFormatError("invalid schema on large chunk")
        return IllustrationCandidatesResponse(
            scene_id=data["scene_id"],
            events=(),
            zero_reason_ru="Нет событий.",
            covered_source_unit_ids=tuple(block_ids),
            complete=True,
        )

    runtime = _StubRuntime(handler=_handler)
    selection = IllustrationSelection(uow, source=source, runtime=runtime, sleep_fn=lambda _: None)
    selection.plan_scopes(gen_id, sec_id)

    for _ in range(10):
        with uow.read_transaction() as tx:
            pending = [t for t in tx.tasks.list_for_book(book_id) if t.status.value == "pending"]
        if not pending:
            break
        for task in pending:
            await selection.run_task(task)

    with uow.read_transaction() as tx:
        cand_scopes = tx.illustrations.list_scopes(
            gen_id, stage=IllustrationScopeStage.CANDIDATES, section_id=sec_id
        )
    assert sum(s.state is IllustrationScopeState.SPLIT for s in cand_scopes) == 1
    assert sum(s.state is IllustrationScopeState.ACCEPTED and bool(s.parent_scope_id) for s in cand_scopes) == 2
