"""Unit and persistence tests for chapter fact scoping, prompt rendering, and acceptance (`P3.T2`)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from linguary.adapters.config import (
    AppSettings,
    CheckSettings,
    LLMSettings,
    ReadingSettings,
    StorageSettings,
)
from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.validation import Py3LangidDetector
from linguary.application.analysis_pass_cache import (
    decode_illustration_pass,
    decode_pass,
    encode_illustration_pass,
    encode_pass,
)
from linguary.application.context_builder import ContextBuilder
from linguary.application.illustration_chapter import (
    IllustrationChapter,
    resolve_moments_with_shortlist_fallback,
)
from linguary.application.illustration_dto import (
    IllustrationCandidateEventDTO,
    IllustrationCandidatesResponse,
    IllustrationFactScopeResponse,
    IllustrationPromptResponse,
    IllustrationQuoteAnchorDTO,
    IllustrationSelectionReasonDTO,
    IllustrationSelectionResponse,
)
from linguary.application.illustration_selection import (
    SelectedAnchor,
    SelectedMomentCard,
    SelectedMoments,
)
from linguary.application.illustration_source import (
    IllustrationSource,
    SectionManifestEntry,
    _pass_model_dependencies,
    compute_chapter_fingerprint,
)
from linguary.application.illustration_validation import (
    MONOCHROME_INK_DIRECTIVE,
    ChapterIllustrationPlan,
    IllustrationAppearance,
    ValidatedMoment,
    compose_monochrome_prompt,
    resolve_insert_after_block_id,
    validate_moment,
)
from linguary.application.llm_dto import (
    NarrativeAnalysisResponse,
    QuotedSceneResponse,
)
from linguary.application.ports.llm import LLMClient, LLMRequest, LLMResponse, LLMResponseKind
from linguary.application.ports.persistence import BookRecord
from linguary.application.token_budget import narrative_max_fragment_blocks_for
from linguary.composition.book_runtime import conditions_payload
from linguary.composition.illustration_runtime import IllustrationRuntime
from linguary.composition.migrations import apply_migrations
from linguary.composition.runtime_settings import RuntimeGeneration, RuntimeSettings
from linguary.domain import (
    AnalysisPassKind,
    AnalysisPassResult,
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    SourceReference,
    SupportStatus,
    TaskStatus,
    TaskType,
)
from linguary.domain.book_conditions import BookConditions
from linguary.domain.content_plan import SourceAnchor, canonical_sha
from linguary.domain.illustrations import (
    TIMELESS_POSITION_KEY,
    IllustrationGenerationState,
    IllustrationPolicy,
    IllustrationRegistryKind,
    IllustrationScopeStage,
    IllustrationScopeState,
    IllustrationSettings,
    IllustrationValidationError,
    RegistryItem,
    format_position_key,
)
from linguary.domain.name_translation import NameTranslationPolicy, NameTranslationStrategy


class _ScriptedClient(LLMClient):
    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.requests: list[LLMRequest[Any]] = []

    async def generate(self, request: LLMRequest[Any]) -> LLMResponse[Any]:
        self.requests.append(request)
        payload = self._handler(request, len(self.requests))
        return LLMResponse(payload=payload, raw_content="", model="gpt-5-mini", provider="fake")


async def _noop_sleep(_seconds: float) -> None:
    return None


@dataclass
class _Harness:
    uow: SqlAlchemyUnitOfWork
    source: IllustrationSource
    runtime: IllustrationRuntime
    client: _ScriptedClient
    chapter: IllustrationChapter
    book_id: str
    conditions_id: str
    source_sha: str
    generation_id: str
    section_ids: tuple[str, ...]
    document: CanonicalDocument


def _seed_book_with_registry(
    tmp_path: Path,
    *,
    sections_spec: list[tuple[str, str, int, list[tuple[str, str]]]],
    registry_items_spec: list[RegistryItem],
    handler: Any,
    max_per_chapter: int = 3,
    model_prompts: str = "gpt-5-mini",
    policy: IllustrationPolicy | None = None,
    book_id: str = "book-ch",
) -> _Harness:
    db = tmp_path / f"{book_id}.db"
    apply_migrations(db)
    uow = SqlAlchemyUnitOfWork.for_engine(create_database_engine(db))
    source_sha = "d" * 64
    conditions_id = "33333333-3333-4333-8333-333333333333"

    doc_sections: list[DocumentSection] = []
    for sec_id, title, order, blocks_spec in sections_spec:
        blocks = tuple(
            DocumentBlock(
                id=b_id,
                section_id=sec_id,
                type=BlockType.PARAGRAPH,
                order=b_idx,
                text=b_text,
                source_reference=SourceReference(f"Text/{sec_id}.xhtml", f"p[{b_idx}]"),
                support_status=SupportStatus.SUPPORTED,
            )
            for b_idx, (b_id, b_text) in enumerate(blocks_spec)
        )
        doc_sections.append(
            DocumentSection(
                id=sec_id,
                document_id=book_id,
                order=order,
                title=title,
                source_resource_path=f"Text/{sec_id}.xhtml",
                blocks=blocks,
            )
        )

    document = CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title="Chapter Novel"),
        source_language="en",
        target_language="ru",
        sections=tuple(doc_sections),
    )

    epub_file = tmp_path / f"{book_id}.epub"
    epub_file.write_bytes(b"fake-epub-ch")
    book = BookRecord(
        id=book_id,
        metadata=BookMetadata(title="Chapter Novel"),
        source_language="en",
        target_language="ru",
        status=BookStatus.GLOSSARY_REVIEW,
        source_epub_path=str(epub_file),
        created_at=datetime(2026, 11, 1, tzinfo=UTC),
    )
    app_settings = AppSettings(
        llm=LLMSettings(
            provider="openai_compatible",
            base_url="https://api.openai.com/v1",
            model="gpt-5-mini",
            api_key="test-key",
            temperature=0.2,
            max_output_tokens=4096,
            staged_analysis=True,
        ),
        checks=CheckSettings(sequential_analysis=False),
        reading=ReadingSettings(reader_level="b1", name_translation_strategy="sound"),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
    )
    cond_payload = conditions_payload(book, app_settings, revision=1, profile="content-plan-v3")
    cond_payload["source_sha256"] = source_sha
    prompts_version = int(cond_payload["versions"]["prompts"])
    ctx_builder = ContextBuilder(
        content_plan_profile="content-plan-v3",
        name_translation_policy=NameTranslationPolicy(
            version=1, strategy=NameTranslationStrategy.SOUND
        ),
        reader_level="b1",
    )

    with uow.transaction() as tx:
        tx.books.add(book)
        tx.documents.save(document)
        tx.work.add_conditions(
            BookConditions(
                id=conditions_id,
                book_id=book_id,
                revision=1,
                payload=cond_payload,
                created_at=datetime(2026, 11, 1, tzinfo=UTC),
            )
        )
        model_deps = _pass_model_dependencies(cond_payload)
        max_frag = narrative_max_fragment_blocks_for(cond_payload["llm_parameters"])
        for sec in doc_sections:
            narrative = NarrativeAnalysisResponse(
                translation_instructions="Сохранять спокойный ритм повествования.",
                chapter_summary=f"In {sec.title} the protagonists face decisive events in the hall.",
                glossary_candidates=(),
                context_facts=(),
                scenes=(
                    QuotedSceneResponse(
                        start_block_id=sec.blocks[0].id,
                        end_block_id=sec.blocks[-1].id,
                        summary=f"Сцена в {sec.title}.",
                        location="Зал",
                        participants=(),
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
                document,
                sec,
                (),
                (),
                (),
                model_dependencies=model_deps,
                max_fragment_blocks=max_frag,
            )
            tx.analyses.save_pass(
                AnalysisPassResult(
                    id=f"pass-{sec.id}",
                    book_id=book_id,
                    section_id=sec.id,
                    pass_kind=AnalysisPassKind.NARRATIVE,
                    scope_key="",
                    input_sha256=input_sha,
                    output_sha256=canonical_sha(encoded),
                    conditions_id=conditions_id,
                    attempt_id=None,
                    payload=encoded,
                )
            )
        tx.illustrations.save_settings(
            book_id,
            expected_revision=0,
            settings=IllustrationSettings(
                enabled=True,
                max_per_chapter=max_per_chapter,
                model_reference="gpt-5-mini",
                model_prompts=model_prompts,
                policy=policy or IllustrationPolicy(),
            ),
        )

    client = _ScriptedClient(handler)

    def _gen_factory(candidate: AppSettings, *, conditions: Any = None) -> RuntimeGeneration:
        del conditions
        return RuntimeGeneration(
            settings=candidate,
            configured_client=None,  # type: ignore[arg-type]
            analysis=None,  # type: ignore[arg-type]
            processing=None,  # type: ignore[arg-type]
            regeneration=None,  # type: ignore[arg-type]
        )

    rt_settings = RuntimeSettings(
        app_settings, _gen_factory(app_settings), revision=1, generation_factory=_gen_factory
    )
    runtime = IllustrationRuntime(
        runtime_settings=rt_settings,
        rate_limiter=None,
        unit_of_work=uow,
        injected_client=client,
    )
    source = IllustrationSource(
        language_detector=Py3LangidDetector(),
        source_digest_fn=lambda _p: source_sha,
    )
    chapter = IllustrationChapter(uow, source=source, runtime=runtime, sleep_fn=_noop_sleep)

    with uow.transaction() as tx:
        head = tx.illustrations.get_head(book_id)
        snap = source.snapshot(tx, book_id)
        assert snap.ready and snap.manifest is not None and snap.manifest_sha256 is not None
        assert head.settings_id is not None
        gen, _ = tx.illustrations.create_or_get_generation(
            book_id,
            settings_id=head.settings_id,
            request_key=f"req-{book_id}-{head.settings_id}",
            source_sha256=source_sha,
            source_authority=snap.source_authority or {},
            requested_sections=snap.content_section_ids,
        )
        gen = tx.illustrations.seal_manifest(
            book_id, gen.id, manifest=snap.manifest, manifest_sha256=snap.manifest_sha256
        )
        # Re-bind generation_id on provided registry items and persist.
        bound_items = [
            RegistryItem(
                generation_id=gen.id,
                kind=item.kind,
                item_key=item.item_key,
                entity_id=item.entity_id,
                position_key=item.position_key,
                is_back_applicable=item.is_back_applicable,
                payload_sha256=item.payload_sha256,
                payload=item.payload,
            )
            for item in registry_items_spec
        ]
        if bound_items:
            tx.illustrations.upsert_registry_items(gen.id, bound_items)
        gen = tx.illustrations.seal_registry(
            book_id, gen.id, registry_sha256=canonical_sha({"items": [i.payload_sha256 for i in bound_items]})
        )
        gen = tx.illustrations.update_generation_state(
            book_id, gen.id, state=IllustrationGenerationState.QUEUED, reason_code=None
        )
        chapter.plan_chapter_scopes(tx, gen, snap)
        gen_id = gen.id

    return _Harness(
        uow=uow,
        source=source,
        runtime=runtime,
        client=client,
        chapter=chapter,
        book_id=book_id,
        conditions_id=conditions_id,
        source_sha=source_sha,
        generation_id=gen_id,
        section_ids=tuple(s.id for s in doc_sections),
        document=document,
    )


async def _drain_chapter_tasks(harness: _Harness) -> int:
    ran = 0
    while True:
        with harness.uow.transaction() as tx:
            pending = [
                t
                for t in tx.tasks.list_for_book(harness.book_id)
                if t.task_type is TaskType.PREPARE_CHAPTER_ILLUSTRATIONS
                and t.status is TaskStatus.PENDING
            ]
        if not pending:
            break
        for task in pending:
            with harness.uow.transaction() as tx:
                running = task.start()
                tx.tasks.save(running)
            await harness.chapter.run_task(running)
            ran += 1
    return ran


def _entity_item(entity_id: str, name: str, kind: str = "person", aliases: list[str] | None = None) -> RegistryItem:
    pay = {
        "entity_id": entity_id,
        "entity_kind": kind,
        "canonical_name": name,
        "aliases": aliases or [name],
        "mention_count": 2,
        "first_position_key": format_position_key(0, 0, 0),
    }
    return RegistryItem(
        generation_id="placeholder",
        kind=IllustrationRegistryKind.ENTITY,
        item_key=entity_id,
        entity_id=entity_id,
        position_key=TIMELESS_POSITION_KEY,
        is_back_applicable=False,
        payload_sha256=canonical_sha(pay),
        payload=pay,
    )


def _fact_item(
    fact_id: str,
    entity_id: str,
    *,
    section_order: int,
    block_order: int = 0,
    char_start: int = 0,
    temporal_scope: str = "state_change",
    back_applicable: bool = False,
    reveals_future_spoiler: bool = False,
    desc_ru: str,
    desc_en: str,
) -> RegistryItem:
    pos = format_position_key(section_order, block_order, char_start)
    is_back = temporal_scope == "permanent" and back_applicable and not reveals_future_spoiler
    pay = {
        "fact_id": fact_id,
        "entity_id": entity_id,
        "temporal_scope": temporal_scope,
        "back_applicable": back_applicable,
        "reveals_future_spoiler": reveals_future_spoiler,
        "is_back_applicable": is_back,
        "description_ru": desc_ru,
        "description_en": desc_en,
        "block_id": "b-ref",
        "char_start": char_start,
        "char_end": char_start + 5,
        "quote": "quote",
        "position_key": pos,
    }
    return RegistryItem(
        generation_id="placeholder",
        kind=IllustrationRegistryKind.FACT,
        item_key=fact_id,
        entity_id=entity_id,
        position_key=pos,
        is_back_applicable=is_back,
        payload_sha256=canonical_sha(pay),
        payload=pay,
    )


async def test_later_scar_not_in_early_prompt(tmp_path: Path) -> None:
    seen_fact_ids_in_ch0: list[str] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-1",
                        title_ru="Вход в зал",
                        event_ru="Алиса входит в тихий зал.",
                        significance_ru="Ключевая встреча",
                        event_kind="arrival",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(block_id=blk["block_id"], quote="Alice entered"),
                        ),
                        participant_names=("Alice",),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cid = data["candidates"][0]["candidate_id"]
            return IllustrationSelectionResponse(
                selected_candidate_ids=(cid,),
                selection_reasons_ru=(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Начало главы"),
                ),
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_FACT_SCOPE:
            fids = [f["fact_id"] for f in data["facts"]]
            if data["event_position_key"].startswith("000000:"):
                seen_fact_ids_in_ch0.extend(fids)
            return IllustrationFactScopeResponse(
                event_id=data["event_id"],
                applicable_fact_ids=tuple(fids),
                excluded_fact_ids=(),
                compact_description_ru=", ".join(f["description_ru"] for f in data["facts"]),
                compact_description_en=", ".join(f["description_en"] for f in data["facts"]),
                assumptions_ru=(),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_PROMPT:
            return IllustrationPromptResponse(
                moment_id=data["moment_id"],
                scene_composition_en="Alice stands at the tall doorway holding a lantern.",
                used_fact_ids=tuple(data["allowed_fact_ids"]),
                used_design_keys=tuple(data["allowed_design_keys"]),
                assumptions_ru=(),
            )
        raise AssertionError(req.kind)

    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[
            ("sec-0", "Chapter 1", 0, [("b-0", "Alice entered the quiet stone hall.")]),
            ("sec-5", "Chapter 6", 5, [("b-5", "Alice entered the courtyard with a fresh scar.")]),
        ],
        registry_items_spec=[
            _entity_item("ent-alice", "Alice"),
            _fact_item(
                "f-early-cloak",
                "ent-alice",
                section_order=0,
                temporal_scope="episodic",
                desc_ru="В сером дорожном плаще",
                desc_en="Wearing a grey travelling cloak",
            ),
            _fact_item(
                "f-later-scar",
                "ent-alice",
                section_order=5,
                temporal_scope="state_change",
                desc_ru="Со шрамом на левой щеке",
                desc_en="With a jagged scar across her left cheek",
            ),
        ],
        handler=_handler,
    )
    await _drain_chapter_tasks(harness)

    assert seen_fact_ids_in_ch0 == ["f-early-cloak"]
    with harness.uow.transaction() as tx:
        ch0 = tx.illustrations.list_scopes(
            harness.generation_id, stage=IllustrationScopeStage.CHAPTER, section_id="sec-0"
        )[0]
        assert ch0.state is IllustrationScopeState.ACCEPTED
        assert ch0.payload is not None
        prompt_en = ch0.payload["moments"][0]["prompt_en"]
        assert "grey travelling cloak" in prompt_en
        assert "scar" not in prompt_en.casefold()


async def test_position_key_numeric_order_blocks_ch10_leak_into_ch2(tmp_path: Path) -> None:
    # Chapter 2 (order=2) MUST NOT see Chapter 10 (order=10) state_change fact even though "10" < "2" lexically!
    seen_in_ch2: list[str] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-1",
                        title_ru="Встреча",
                        event_ru="Алиса идёт по мосту.",
                        significance_ru="Важная сцена",
                        event_kind="journey",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(block_id=blk["block_id"], quote="Alice crossed"),
                        ),
                        participant_names=("Alice",),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cid = data["candidates"][0]["candidate_id"]
            return IllustrationSelectionResponse(
                selected_candidate_ids=(cid,),
                selection_reasons_ru=(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Ключевая сцена"),
                ),
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_FACT_SCOPE:
            fids = [f["fact_id"] for f in data["facts"]]
            if data["event_position_key"].startswith("000002:"):
                seen_in_ch2.extend(fids)
            return IllustrationFactScopeResponse(
                event_id=data["event_id"],
                applicable_fact_ids=tuple(fids),
                excluded_fact_ids=(),
                compact_description_ru="Алиса",
                compact_description_en=", ".join(f["description_en"] for f in data["facts"]),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_PROMPT:
            return IllustrationPromptResponse(
                moment_id=data["moment_id"],
                scene_composition_en="Alice crosses the narrow bridge.",
                used_fact_ids=tuple(data["allowed_fact_ids"]),
                used_design_keys=(),
            )
        raise AssertionError(req.kind)

    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[
            ("sec-2", "Chapter 2", 2, [("b-2", "Alice crossed the wooden footbridge.")]),
            ("sec-10", "Chapter 10", 10, [("b-10", "Alice crossed the iron bridge wearing armor.")]),
        ],
        registry_items_spec=[
            _entity_item("ent-alice", "Alice"),
            _fact_item("f-ch2", "ent-alice", section_order=2, desc_ru="В шерстяной шали", desc_en="In a wool shawl"),
            _fact_item("f-ch10", "ent-alice", section_order=10, desc_ru="В стальных доспехах", desc_en="In steel plate armor"),
        ],
        handler=_handler,
    )
    await _drain_chapter_tasks(harness)

    assert seen_in_ch2 == ["f-ch2"]
    assert "f-ch10" not in seen_in_ch2


async def test_multi_anchor_event_uses_max_position_key(tmp_path: Path) -> None:
    # Event has anchors in block 0 and block 2 -> event_position_key MUST be block 2's key,
    # so a fact appearing in block 1 (between anchors) IS visible!
    seen_facts: list[str] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-multi",
                        title_ru="Переодевание и выход",
                        event_ru="Алиса надевает плащ и выходит во двор.",
                        significance_ru="Ключевой переход",
                        event_kind="action",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(block_id="b-0", quote="Alice grabbed"),
                            IllustrationQuoteAnchorDTO(block_id="b-2", quote="stepped outside"),
                        ),
                        participant_names=("Alice",),
                    ),
                ),
                covered_source_unit_ids=tuple(b["source_unit_id"] for b in data["blocks"]),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cid = data["candidates"][0]["candidate_id"]
            return IllustrationSelectionResponse(
                selected_candidate_ids=(cid,),
                selection_reasons_ru=(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Яркий момент"),
                ),
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_FACT_SCOPE:
            seen_facts.extend(f["fact_id"] for f in data["facts"])
            return IllustrationFactScopeResponse(
                event_id=data["event_id"],
                applicable_fact_ids=tuple(f["fact_id"] for f in data["facts"]),
                excluded_fact_ids=(),
                compact_description_ru="В тёмном плаще",
                compact_description_en="Wearing a dark cloak",
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_PROMPT:
            return IllustrationPromptResponse(
                moment_id=data["moment_id"],
                scene_composition_en="Alice steps outside into the rain.",
                used_fact_ids=tuple(data["allowed_fact_ids"]),
                used_design_keys=(),
            )
        raise AssertionError(req.kind)

    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[
            (
                "sec-0",
                "Chapter 1",
                0,
                [
                    ("b-0", "Alice grabbed her lantern from the hook."),
                    ("b-1", "She fastened a heavy dark cloak over her shoulders."),
                    ("b-2", "Then she stepped outside into the storm."),
                ],
            )
        ],
        registry_items_spec=[
            _entity_item("ent-alice", "Alice"),
            _fact_item("f-b1-cloak", "ent-alice", section_order=0, block_order=1, desc_ru="Тёмный плащ", desc_en="Heavy dark cloak"),
        ],
        handler=_handler,
    )
    await _drain_chapter_tasks(harness)
    assert seen_facts == ["f-b1-cloak"]


async def test_contradictory_facts_preserved_separately(tmp_path: Path) -> None:
    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-1",
                        title_ru="Портрет у окна",
                        event_ru="Алиса смотрит в окно.",
                        significance_ru="Крупный план",
                        event_kind="portrait",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(
                                block_id=blk["block_id"], quote="gazed out the narrow window"
                            ),
                        ),
                        participant_names=("Alice",),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cid = data["candidates"][0]["candidate_id"]
            return IllustrationSelectionResponse(
                selected_candidate_ids=(cid,),
                selection_reasons_ru=(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Выразительный кадр"),
                ),
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_FACT_SCOPE:
            fids = [f["fact_id"] for f in data["facts"]]
            return IllustrationFactScopeResponse(
                event_id=data["event_id"],
                applicable_fact_ids=tuple(fids),
                excluded_fact_ids=(),
                compact_description_ru="Глаза описаны противоречиво (синие и серые)",
                compact_description_en="Eyes described variously in source; render in neutral monochrome value",
                assumptions_ru=("В тексте есть противоречие в цвете глаз; оба факта сохранены отдельно.",),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_PROMPT:
            return IllustrationPromptResponse(
                moment_id=data["moment_id"],
                scene_composition_en="Alice gazes out a tall window.",
                used_fact_ids=tuple(data["allowed_fact_ids"]),
                used_design_keys=(),
            )
        raise AssertionError(req.kind)

    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[("sec-0", "Chapter 1", 0, [("b-0", "Alice gazed out the narrow window.")])],
        registry_items_spec=[
            _entity_item("ent-alice", "Alice"),
            _fact_item("f-blue", "ent-alice", section_order=0, char_start=0, temporal_scope="permanent", desc_ru="Синие глаза", desc_en="Blue eyes"),
            _fact_item("f-grey", "ent-alice", section_order=0, char_start=1, temporal_scope="permanent", desc_ru="Серые глаза", desc_en="Grey eyes"),
        ],
        handler=_handler,
    )
    await _drain_chapter_tasks(harness)
    with harness.uow.transaction() as tx:
        ch = tx.illustrations.list_scopes(
            harness.generation_id, stage=IllustrationScopeStage.CHAPTER, section_id="sec-0"
        )[0]
        assert ch.payload is not None
        char = ch.payload["moments"][0]["character_descriptions"][0]
        assert set(char["fact_ids"]) == {"f-blue", "f-grey"}
        assert any("противоречие" in a for a in ch.payload["moments"][0]["assumptions_ru"])


async def test_backward_propagation_fetched_via_query_registry_items_and_requires_invariance_no_spoiler(
    tmp_path: Path,
) -> None:
    seen_prompt_fact_ids: list[str] = []
    fact_scope_calls = 0

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        nonlocal fact_scope_calls
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-1",
                        title_ru="У камина",
                        event_ru="Алиса сидит у камина.",
                        significance_ru="Знакомство с героиней",
                        event_kind="character",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(block_id=blk["block_id"], quote="Alice sat"),
                        ),
                        participant_names=("Alice",),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cid = data["candidates"][0]["candidate_id"]
            return IllustrationSelectionResponse(
                selected_candidate_ids=(cid,),
                selection_reasons_ru=(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Важный портрет"),
                ),
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_FACT_SCOPE:
            fact_scope_calls += 1
            return IllustrationFactScopeResponse(
                event_id=data["event_id"],
                applicable_fact_ids=tuple(f["fact_id"] for f in data["facts"]),
                excluded_fact_ids=(),
                compact_description_ru="Высокая, с прямыми волосами",
                compact_description_en="Tall with straight dark hair",
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_PROMPT:
            seen_prompt_fact_ids.extend(data["allowed_fact_ids"])
            return IllustrationPromptResponse(
                moment_id=data["moment_id"],
                scene_composition_en="Alice sits beside the fireplace.",
                used_fact_ids=tuple(data["allowed_fact_ids"]),
                used_design_keys=(),
            )
        raise AssertionError(req.kind)

    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[("sec-0", "Chapter 1", 0, [("b-0", "Alice sat quietly near the hearth.")])],
        registry_items_spec=[
            _entity_item("ent-alice", "Alice"),
            # Permanent + back_applicable + NOT spoiler -> INCLUDED in ch0 via deterministic fast-path!
            _fact_item(
                "f-perm-ok",
                "ent-alice",
                section_order=7,
                temporal_scope="permanent",
                back_applicable=True,
                reveals_future_spoiler=False,
                desc_ru="Высокого роста",
                desc_en="Tall stature",
            ),
            # Permanent + back_applicable + SPOILER -> BLOCKED!
            _fact_item(
                "f-perm-spoiler",
                "ent-alice",
                section_order=8,
                temporal_scope="permanent",
                back_applicable=True,
                reveals_future_spoiler=True,
                desc_ru="Королевское родимое пятно-спойлер",
                desc_en="Royal birthmark spoiler",
            ),
            # State change + back_applicable=True -> BLOCKED (not invariant)!
            _fact_item(
                "f-state-late",
                "ent-alice",
                section_order=6,
                temporal_scope="state_change",
                back_applicable=True,
                reveals_future_spoiler=False,
                desc_ru="Повязка на руке",
                desc_en="Bandage on arm",
            ),
        ],
        handler=_handler,
    )
    await _drain_chapter_tasks(harness)
    assert fact_scope_calls == 0
    assert seen_prompt_fact_ids == ["f-perm-ok"]
    with harness.uow.read_transaction() as tx:
        ch = tx.illustrations.list_scopes(
            harness.generation_id, stage=IllustrationScopeStage.CHAPTER, section_id="sec-0"
        )[0]
        assert ch.payload is not None
        assert list(ch.payload["moments"][0]["character_descriptions"][0]["fact_ids"]) == [
            "f-perm-ok"
        ]


def test_ambiguous_identity_remains_separate() -> None:
    entities = [
        {"entity_id": "ent-john-1", "entity_kind": "person", "canonical_name": "John", "aliases": ["John"]},
        {"entity_id": "ent-john-2", "entity_kind": "person", "canonical_name": "John", "aliases": ["John"]},
    ]
    amb_card = SelectedMomentCard(
        candidate_id="c-amb",
        section_id="sec-0",
        scene_id="s1",
        title_ru="Встреча с Джоном",
        event_ru="Джон входит в комнату.",
        significance_ru="Важный вход",
        reason_ru="Выбрано",
        event_kind="entry",
        event_position_key=format_position_key(0, 0, 0),
        anchors=(SelectedAnchor("b-0", 0, 4, "John", format_position_key(0, 0, 0)),),
        insert_after_block_id="b-0",
        participant_names=("John",),
        location_names=(),
        scene_start_block_id="b-0",
        scene_end_block_id="b-0",
    )
    bound, zero_reason = resolve_moments_with_shortlist_fallback(
        SelectedMoments("sec-0", winners=(amb_card,), shortlist=(), zero_reason_ru=None),
        entities=entities,
    )
    assert bound == ()
    assert zero_reason is not None and len(zero_reason) > 0


def test_insert_after_last_event_evidence_within_scene_textual_blocks_default_and_validated() -> None:
    blocks = (
        DocumentBlock(id="b-0", section_id="sec-0", type=BlockType.PARAGRAPH, order=0, text="First paragraph text.", source_reference=SourceReference("ch.xhtml", "p[0]"), support_status=SupportStatus.SUPPORTED),
        DocumentBlock(id="b-1", section_id="sec-0", type=BlockType.PARAGRAPH, order=1, text="Second paragraph evidence.", source_reference=SourceReference("ch.xhtml", "p[1]"), support_status=SupportStatus.SUPPORTED),
        DocumentBlock(id="b-img", section_id="sec-0", type=BlockType.IMAGE, order=2, text="", source_reference=SourceReference("ch.xhtml", "img[0]"), support_status=SupportStatus.SUPPORTED),
        DocumentBlock(id="b-2", section_id="sec-0", type=BlockType.PARAGRAPH, order=3, text="Third paragraph closing scene.", source_reference=SourceReference("ch.xhtml", "p[2]"), support_status=SupportStatus.SUPPORTED),
    )
    section = DocumentSection(id="sec-0", document_id="book-1", order=0, title="Ch 1", source_resource_path="ch.xhtml", blocks=blocks)
    anchors = (SourceAnchor("b-0", 0, 5, "First"), SourceAnchor("b-1", 0, 6, "Second"))

    # None -> deterministic default is last evidence block ("b-1").
    assert resolve_insert_after_block_id(None, anchors=anchors, scene_end_block_id="b-2", section=section) == "b-1"
    # Valid block within [b-1, b-2] -> accepted.
    assert resolve_insert_after_block_id("b-2", anchors=anchors, scene_end_block_id="b-2", section=section) == "b-2"
    # Earlier than last evidence ("b-0") -> rejected!
    with pytest.raises(IllustrationValidationError):
        resolve_insert_after_block_id("b-0", anchors=anchors, scene_end_block_id="b-2", section=section)
    # Non-textual IMAGE block ("b-img") -> rejected!
    with pytest.raises(IllustrationValidationError):
        resolve_insert_after_block_id("b-img", anchors=anchors, scene_end_block_id="b-2", section=section)


def test_foreign_anchor_and_block_alias_rejected(tmp_path: Path) -> None:
    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[
            ("sec-0", "Chapter 1", 0, [("canon-b0", "Alice opened the heavy door.")]),
            ("sec-1", "Chapter 2", 1, [("canon-b1", "Bob waited in the garden.")]),
        ],
        registry_items_spec=[],
        handler=lambda _r, _n: (_ for _ in ()).throw(AssertionError("unused")),
    )
    with harness.uow.transaction() as tx:
        snap = harness.source.snapshot(tx, harness.book_id)
    sec0 = harness.document.ordered_sections[0]
    scene_id = snap.entries[0].scenes[0].id
    base_prompt = compose_monochrome_prompt("Alice opens the heavy oak door.", ())

    # 1. Foreign anchor from sec-1 -> rejected!
    foreign_moment = ValidatedMoment(
        id="m1",
        scene_id=scene_id,
        title_ru="Дверь",
        reason_ru="Причина",
        event_ru="Событие",
        prompt_en=base_prompt,
        insert_after_block_id="canon-b0",
        anchors=(SourceAnchor("canon-b1", 0, 3, "Bob"),),
    )
    with pytest.raises(IllustrationValidationError):
        validate_moment(
            foreign_moment,
            snapshot=snap,
            section=sec0,
            scene_start_block_id="canon-b0",
            scene_end_block_id="canon-b0",
            registry={},
            event_position_key=format_position_key(0, 0, 0),
        )

    # 2. Local block alias "b1" -> rejected!
    alias_moment = ValidatedMoment(
        id="m2",
        scene_id=scene_id,
        title_ru="Дверь",
        reason_ru="Причина",
        event_ru="Событие",
        prompt_en=base_prompt,
        insert_after_block_id="canon-b0",
        anchors=(SourceAnchor("b1", 0, 5, "Alice"),),
    )
    with pytest.raises(IllustrationValidationError):
        validate_moment(
            alias_moment,
            snapshot=snap,
            section=sec0,
            scene_start_block_id="canon-b0",
            scene_end_block_id="canon-b0",
            registry={},
            event_position_key=format_position_key(0, 0, 0),
        )


async def test_paginated_fact_scoping_chronological_folding_resolves_clothing_change_and_caps_32_ids(
    tmp_path: Path,
) -> None:
    fact_scope_pages_seen: list[tuple[int, bool, str]] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            target_blk = data["blocks"][-1]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-1",
                        title_ru="Бал в замке",
                        event_ru="Алиса входит в бальный зал.",
                        significance_ru="Смена наряда",
                        event_kind="entrance",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(
                                block_id=target_blk["block_id"], quote="Alice entered"
                            ),
                        ),
                        participant_names=("Alice",),
                    ),
                ),
                covered_source_unit_ids=tuple(b["source_unit_id"] for b in data["blocks"]),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cid = data["candidates"][0]["candidate_id"]
            return IllustrationSelectionResponse(
                selected_candidate_ids=(cid,),
                selection_reasons_ru=(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Яркий выход"),
                ),
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_FACT_SCOPE:
            fact_scope_pages_seen.append(
                (data["page_index"], data["is_consolidation"], data["running_summary_en"])
            )
            fids = [f["fact_id"] for f in data["facts"]]
            if data["is_consolidation"]:
                # Consolidate down to <=32 keeping latest velvet dress, excluding old travelling coat!
                kept = [fid for fid in fids if fid != "f-000"][:30]
                return IllustrationFactScopeResponse(
                    event_id=data["event_id"],
                    applicable_fact_ids=tuple(kept),
                    excluded_fact_ids=("f-000",) if "f-000" in fids else (),
                    compact_description_ru="В бархатном вечернем платье",
                    compact_description_en="Wearing a dark velvet evening gown",
                    complete=True,
                )
            return IllustrationFactScopeResponse(
                event_id=data["event_id"],
                applicable_fact_ids=tuple(fids[:20]),
                excluded_fact_ids=(),
                compact_description_ru=f"Сводка страницы {data['page_index']}",
                compact_description_en=f"Running summary page {data['page_index']}",
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_PROMPT:
            return IllustrationPromptResponse(
                moment_id=data["moment_id"],
                scene_composition_en="Alice enters the grand ballroom.",
                used_fact_ids=tuple(data["allowed_fact_ids"]),
                used_design_keys=(),
            )
        raise AssertionError(req.kind)

    # Create 40 large facts so they paginate across >=3 pages under standard 24576 B cap AND accumulate >32 IDs -> triggers consolidation!
    many_facts = [_entity_item("ent-alice", "Alice")] + [
        _fact_item(
            f"f-{i:03d}",
            "ent-alice",
            section_order=0,
            block_order=0,
            char_start=i,
            temporal_scope="episodic",
            desc_ru=f"Деталь внешности {i}: " + ("подробное описание " * 30),
            desc_en=f"Appearance detail {i}: " + ("detailed visual note " * 30),
        )
        for i in range(40)
    ]

    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[
            (
                "sec-0",
                "Chapter 1",
                0,
                [
                    ("b-0", "Earlier notes about Alice and her wardrobe."),
                    ("b-1", "Alice entered the ballroom."),
                ],
            )
        ],
        registry_items_spec=many_facts,
        handler=_handler,
        policy=IllustrationPolicy(),
    )
    await _drain_chapter_tasks(harness)

    # Multiple chronological pages with non-empty running_summary passed forward + final consolidation!
    assert len(fact_scope_pages_seen) >= 3
    assert fact_scope_pages_seen[1][2].startswith("Running summary page 0")
    assert any(is_consol for _, is_consol, _ in fact_scope_pages_seen)
    with harness.uow.transaction() as tx:
        ch = tx.illustrations.list_scopes(
            harness.generation_id, stage=IllustrationScopeStage.CHAPTER, section_id="sec-0"
        )[0]
        assert ch.state is IllustrationScopeState.ACCEPTED and ch.payload is not None
        char = ch.payload["moments"][0]["character_descriptions"][0]
        assert len(char["fact_ids"]) <= 32
        assert "f-000" not in char["fact_ids"]
        assert "velvet evening gown" in ch.payload["moments"][0]["prompt_en"]


def test_prompt_contains_actual_scoped_descriptions_and_monochrome_directive() -> None:
    app = IllustrationAppearance(
        entity_id="ent-alice",
        name_ru="Алиса",
        description_ru="Короткие тёмные волосы, дорожный плащ",
        description_en="Short dark hair and weathered travelling cloak",
        fact_ids=("f-1",),
    )
    prompt = compose_monochrome_prompt("Alice raises a storm lantern on the pier.", [app])
    assert "Short dark hair and weathered travelling cloak" in prompt
    assert MONOCHROME_INK_DIRECTIVE in prompt


def test_id_only_prompt_without_description_en_rejected(tmp_path: Path) -> None:
    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[("sec-0", "Chapter 1", 0, [("b-0", "Alice raised a lantern.")])],
        registry_items_spec=[],
        handler=lambda _r, _n: (_ for _ in ()).throw(AssertionError("unused")),
    )
    with harness.uow.transaction() as tx:
        snap = harness.source.snapshot(tx, harness.book_id)
    sec0 = harness.document.ordered_sections[0]
    scene_id = snap.entries[0].scenes[0].id

    # Prompt only references entity ID "ent-alice" instead of actual description_en -> REJECTED!
    bad_moment = ValidatedMoment(
        id="m-bad",
        scene_id=scene_id,
        title_ru="Фонарь",
        reason_ru="Причина",
        event_ru="Событие",
        prompt_en=f"ent-alice raises a lantern. Style: {MONOCHROME_INK_DIRECTIVE}.",
        insert_after_block_id="b-0",
        anchors=(SourceAnchor("b-0", 0, 5, "Alice"),),
        character_descriptions=(
            IllustrationAppearance(
                entity_id="ent-alice",
                name_ru="Алиса",
                description_ru="Высокая, в сером плаще",
                description_en="Tall figure in a grey cloak",
                fact_ids=(),
            ),
        ),
    )
    with pytest.raises(IllustrationValidationError):
        validate_moment(
            bad_moment,
            snapshot=snap,
            section=sec0,
            scene_start_block_id="b-0",
            scene_end_block_id="b-0",
            registry={},
            event_position_key=format_position_key(0, 0, 0),
        )


def test_fallback_to_shortlist_rechecks_diversity_and_reason_ru_or_zero_when_identity_unresolvable() -> None:
    entities = [
        {"entity_id": "ent-alice", "entity_kind": "person", "canonical_name": "Alice", "aliases": ["Alice"]},
        {"entity_id": "ent-john-1", "entity_kind": "person", "canonical_name": "John", "aliases": ["John"]},
        {"entity_id": "ent-john-2", "entity_kind": "person", "canonical_name": "John", "aliases": ["John"]},
    ]
    w_ok = SelectedMomentCard(
        candidate_id="c-ok",
        section_id="sec-0",
        scene_id="s1",
        title_ru="Алиса у ворот",
        event_ru="Алиса открывает ворота.",
        significance_ru="Начало пути",
        reason_ru="Сильный старт",
        event_kind="entry",
        event_position_key=format_position_key(0, 0, 0),
        anchors=(SelectedAnchor("b-0", 0, 5, "Alice", format_position_key(0, 0, 0)),),
        insert_after_block_id="b-0",
        participant_names=("Alice",),
        location_names=(),
        scene_start_block_id="b-0",
        scene_end_block_id="b-0",
    )
    w_amb = SelectedMomentCard(
        candidate_id="c-amb",
        section_id="sec-0",
        scene_id="s2",
        title_ru="Спор с Джоном",
        event_ru="Неясно, какой Джон спорит.",
        significance_ru="Спор",
        reason_ru="Конфликт",
        event_kind="conflict",
        event_position_key=format_position_key(0, 1, 0),
        anchors=(SelectedAnchor("b-1", 0, 4, "John", format_position_key(0, 1, 0)),),
        insert_after_block_id="b-1",
        participant_names=("John",),
        location_names=(),
        scene_start_block_id="b-1",
        scene_end_block_id="b-1",
    )
    # First shortlist candidate conflicts on diversity with w_ok (same scene_id + event_kind)!
    short_dup = SelectedMomentCard(
        candidate_id="c-short-dup",
        section_id="sec-0",
        scene_id="s1",
        title_ru="Алиса снова у ворот",
        event_ru="Повторный кадр у ворот.",
        significance_ru="Повторная значимость",
        reason_ru="старая",
        event_kind="entry",
        event_position_key=format_position_key(0, 0, 10),
        anchors=(SelectedAnchor("b-0", 10, 15, "gates", format_position_key(0, 0, 10)),),
        insert_after_block_id="b-0",
        participant_names=("Alice",),
        location_names=(),
        scene_start_block_id="b-0",
        scene_end_block_id="b-0",
    )
    # Second shortlist candidate is distinct and resolvable -> promoted with significance_ru as reason_ru!
    short_good = SelectedMomentCard(
        candidate_id="c-short-good",
        section_id="sec-0",
        scene_id="s3",
        title_ru="Алиса находит карту",
        event_ru="Алиса разворачивает старую карту.",
        significance_ru="Ключевая находка главы",
        reason_ru="старая",
        event_kind="discovery",
        event_position_key=format_position_key(0, 2, 0),
        anchors=(SelectedAnchor("b-2", 0, 5, "Alice", format_position_key(0, 2, 0)),),
        insert_after_block_id="b-2",
        participant_names=("Alice",),
        location_names=(),
        scene_start_block_id="b-2",
        scene_end_block_id="b-2",
    )

    bound, zero_reason = resolve_moments_with_shortlist_fallback(
        SelectedMoments("sec-0", winners=(w_ok, w_amb), shortlist=(short_dup, short_good), zero_reason_ru=None),
        entities=entities,
    )
    assert zero_reason is None
    assert [c.candidate_id for c, _, _ in bound] == ["c-ok", "c-short-good"]
    assert bound[1][0].reason_ru == "Ключевая находка главы"


async def test_analysis_pass_mirror_uses_book_conditions_id_under_fk_on_and_analyze_chapter_filters_pass_kind(
    tmp_path: Path,
) -> None:
    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(),
                zero_reason_ru="Тихая описательная глава.",
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        raise AssertionError(req.kind)

    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[("sec-0", "Chapter 1", 0, [("b-0", "Rain fell softly all afternoon.")])],
        registry_items_spec=[],
        handler=_handler,
    )
    await _drain_chapter_tasks(harness)

    with harness.uow.transaction() as tx:
        mirror = tx.analyses.get_pass(
            harness.book_id,
            "sec-0",
            AnalysisPassKind.ILLUSTRATION_PREP,
            scope_key=harness.generation_id,
        )
        assert mirror is not None
        assert mirror.conditions_id == harness.conditions_id
        decoded = decode_pass(mirror.payload, pass_kind=AnalysisPassKind.ILLUSTRATION_PREP)
        assert isinstance(decoded, ChapterIllustrationPlan)
        assert decoded.zero_reason_ru == "Тихая описательная глава."
        assert decode_illustration_pass(encode_illustration_pass(decoded)) == decoded

        # Verify list_passes_for_section filters strictly by pass_kind!
        narrative_only = tx.analyses.list_passes_for_section(
            harness.book_id, "sec-0", pass_kind=AnalysisPassKind.NARRATIVE
        )
        assert all(p.pass_kind is AnalysisPassKind.NARRATIVE for p in narrative_only)
        assert len(narrative_only) == 1
        ill_only = tx.analyses.list_passes_for_section(
            harness.book_id, "sec-0", pass_kind=AnalysisPassKind.ILLUSTRATION_PREP
        )
        assert len(ill_only) == 1


async def test_limit_or_model_or_style_profile_change_only_rebuilds_prompt_parts(
    tmp_path: Path,
) -> None:
    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-1",
                        title_ru="Выход",
                        event_ru="Алиса выходит на крыльцо.",
                        significance_ru="Ключевой шаг",
                        event_kind="exit",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(block_id=blk["block_id"], quote="Alice stepped"),
                        ),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cid = data["candidates"][0]["candidate_id"]
            return IllustrationSelectionResponse(
                selected_candidate_ids=(cid,),
                selection_reasons_ru=(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Хороший кадр"),
                ),
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_PROMPT:
            return IllustrationPromptResponse(
                moment_id=data["moment_id"],
                scene_composition_en="Alice steps onto the stone porch.",
                used_fact_ids=(),
                used_design_keys=(),
            )
        raise AssertionError(req.kind)

    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[("sec-0", "Chapter 1", 0, [("b-0", "Alice stepped onto the porch.")])],
        registry_items_spec=[],
        handler=_handler,
        max_per_chapter=3,
    )
    await _drain_chapter_tasks(harness)
    first_kinds = [r.kind for r in harness.client.requests]
    assert LLMResponseKind.ILLUSTRATION_CANDIDATES in first_kinds
    assert LLMResponseKind.ILLUSTRATION_SELECTION in first_kinds
    assert LLMResponseKind.ILLUSTRATION_PROMPT in first_kinds
    calls_after_gen1 = len(harness.client.requests)

    # Change ONLY max_per_chapter from 3 -> 2 (keeping same model_prompts and registry):
    # CANDIDATES and PROMPT fingerprints are UNCHANGED -> reused; only SELECTION re-runs!
    with harness.uow.transaction() as tx:
        tx.illustrations.revoke_generation(
            harness.book_id, harness.generation_id, reason_code="settings_changed"
        )
        rev2 = tx.illustrations.save_settings(
            harness.book_id,
            expected_revision=1,
            settings=IllustrationSettings(
                enabled=True,
                max_per_chapter=2,
                model_reference="gpt-5-mini",
                model_prompts="gpt-5-mini",
            ),
        )
        snap = harness.source.snapshot(tx, harness.book_id)
        assert snap.manifest is not None and snap.manifest_sha256 is not None
        gen2, _ = tx.illustrations.create_or_get_generation(
            harness.book_id,
            settings_id=rev2.id,
            request_key=f"req-gen2-{rev2.id}",
            source_sha256=harness.source_sha,
            source_authority=snap.source_authority or {},
            requested_sections=snap.content_section_ids,
        )
        gen2 = tx.illustrations.seal_manifest(
            harness.book_id, gen2.id, manifest=snap.manifest, manifest_sha256=snap.manifest_sha256
        )
        gen2 = tx.illustrations.seal_registry(
            harness.book_id, gen2.id, registry_sha256=canonical_sha({"items": []})
        )
        gen2 = tx.illustrations.update_generation_state(
            harness.book_id, gen2.id, state=IllustrationGenerationState.QUEUED, reason_code=None
        )
        harness.chapter.plan_chapter_scopes(tx, gen2, snap)

    await _drain_chapter_tasks(harness)
    second_kinds = [r.kind for r in harness.client.requests[calls_after_gen1:]]
    # Reference stages (EXTRACT/ALIAS/DESIGN) were NEVER called; CANDIDATES & PROMPT were reused!
    assert LLMResponseKind.ILLUSTRATION_EXTRACT not in second_kinds
    assert LLMResponseKind.ILLUSTRATION_CANDIDATES not in second_kinds
    assert second_kinds == [LLMResponseKind.ILLUSTRATION_SELECTION]


def _design_item(
    entity_id: str,
    trait: str,
    *,
    desc_ru: str,
    desc_en: str,
    assumption_ru: str,
    entity_kind: str = "person",
) -> RegistryItem:
    pay = {
        "entity_id": entity_id,
        "entity_kind": entity_kind,
        "trait": trait,
        "description_ru": desc_ru,
        "description_en": desc_en,
        "assumption_ru": assumption_ru,
    }
    return RegistryItem(
        generation_id="placeholder",
        kind=IllustrationRegistryKind.DESIGN,
        item_key=f"{entity_id}:{trait}",
        entity_id=entity_id,
        position_key=TIMELESS_POSITION_KEY,
        is_back_applicable=False,
        payload_sha256=canonical_sha(pay),
        payload=pay,
    )


async def test_used_design_keys_controls_final_prompt_appearances_and_assumptions_both_branches(
    tmp_path: Path,
) -> None:
    prompt_inputs_seen: list[dict[str, Any]] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-1",
                        title_ru="Встреча в коридоре",
                        event_ru="Алиса и Боб разговаривают в коридоре.",
                        significance_ru="Важная встреча",
                        event_kind="meeting",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(block_id=blk["block_id"], quote="Alice met Bob"),
                        ),
                        participant_names=("Alice", "Bob"),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cid = data["candidates"][0]["candidate_id"]
            return IllustrationSelectionResponse(
                selected_candidate_ids=(cid,),
                selection_reasons_ru=(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Ключевая встреча"),
                ),
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_FACT_SCOPE:
            assert data["style"] == "monochrome_ink"
            return IllustrationFactScopeResponse(
                event_id=data["event_id"],
                applicable_fact_ids=tuple(f["fact_id"] for f in data["facts"]),
                excluded_fact_ids=(),
                compact_description_ru="В простом шерстяном плаще",
                compact_description_en="In a plain wool cloak",
                assumptions_ru=("Допущений нет.",),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_PROMPT:
            prompt_inputs_seen.append(data)
            # Deliberately select ONLY palette_value for Alice (with facts) and hair for Bob (no facts),
            # omitting Alice's `eyes` and Bob's `attire_default`!
            return IllustrationPromptResponse(
                moment_id=data["moment_id"],
                scene_composition_en="Alice in a plain wool cloak speaks with Bob in the hallway.",
                used_fact_ids=tuple(data["allowed_fact_ids"]),
                used_design_keys=("ent-alice:palette_value", "ent-bob:hair"),
                assumptions_ru=("Нет: указано в фактах.", "Нейтральный ракурс камеры"),
            )
        raise AssertionError(req.kind)

    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[("sec-0", "Chapter 1", 0, [("b-0", "Alice met Bob in the quiet hallway.")])],
        registry_items_spec=[
            _entity_item("ent-alice", "Alice"),
            _entity_item("ent-bob", "Bob"),
            _fact_item(
                "f-alice-cloak",
                "ent-alice",
                section_order=0,
                temporal_scope="episodic",
                desc_ru="В шерстяном плаще",
                desc_en="In a plain wool cloak",
            ),
            _design_item(
                "ent-alice",
                "palette_value",
                desc_ru="Высококонтрастная штриховка тушью",
                desc_en="High-contrast ink crosshatching",
                assumption_ru="Контрастная монохромная подача",
            ),
            _design_item(
                "ent-alice",
                "eyes",
                desc_ru="Закрытые спящие глаза",
                desc_en="Closed sleeping eyes",
                assumption_ru="Лишнее допущение про закрытые глаза",
            ),
            _design_item(
                "ent-bob",
                "hair",
                desc_ru="Короткие аккуратные волосы",
                desc_en="Short neat hair",
                assumption_ru="Нейтральная стрижка Боба",
            ),
            _design_item(
                "ent-bob",
                "attire_default",
                desc_ru="Тяжёлое зимнее пальто",
                desc_en="Heavy winter overcoat",
                assumption_ru="Лишнее допущение про зимнее пальто",
            ),
        ],
        handler=_handler,
    )
    await _drain_chapter_tasks(harness)

    assert len(prompt_inputs_seen) == 1
    avail_keys = {t["key"] for t in prompt_inputs_seen[0]["available_design_traits"]}
    assert avail_keys == {
        "ent-alice:palette_value",
        "ent-alice:eyes",
        "ent-bob:hair",
        "ent-bob:attire_default",
    }
    # Base characters sent to PROMPT must NOT pre-bake unused DESIGN traits.
    alice_pre = next(c for c in prompt_inputs_seen[0]["characters"] if c["entity_id"] == "ent-alice")
    assert "Closed sleeping eyes" not in alice_pre["description_en"]
    bob_pre = next(c for c in prompt_inputs_seen[0]["characters"] if c["entity_id"] == "ent-bob")
    assert "Heavy winter overcoat" not in bob_pre["description_en"]

    with harness.uow.transaction() as tx:
        ch = tx.illustrations.list_scopes(
            harness.generation_id, stage=IllustrationScopeStage.CHAPTER, section_id="sec-0"
        )[0]
        assert ch.state is IllustrationScopeState.ACCEPTED and ch.payload is not None
        moment = ch.payload["moments"][0]
        prompt_en = moment["prompt_en"]
        chars_by_id = {c["entity_id"]: c for c in moment["character_descriptions"]}

        # Selected design traits materialized; unused ones strictly excluded!
        assert "High-contrast ink crosshatching" in chars_by_id["ent-alice"]["description_en"]
        assert "Closed sleeping eyes" not in chars_by_id["ent-alice"]["description_en"]
        assert "Closed sleeping eyes" not in prompt_en

        # Bob had no facts -> fallback "Bob" was replaced by selected `hair` trait, without `attire_default`!
        assert chars_by_id["ent-bob"]["description_en"] == "Short neat hair"
        assert "Heavy winter overcoat" not in chars_by_id["ent-bob"]["description_en"]
        assert "Heavy winter overcoat" not in prompt_en

        # Assumptions contain ONLY real assumptions from PROMPT + used_design_keys, no placeholders!
        assumps = moment["assumptions_ru"]
        assert "Контрастная монохромная подача" in assumps
        assert "Нейтральная стрижка Боба" in assumps
        assert "Нейтральный ракурс камеры" in assumps
        assert all("Допущений нет" not in a for a in assumps)
        assert all("Нет: указано" not in a for a in assumps)
        assert all("закрытые глаза" not in a for a in assumps)
        assert all("зимнее пальто" not in a for a in assumps)


def test_compose_monochrome_prompt_deduplicates_clauses_and_validate_moment_handles_partial_and_full_coverage(
    tmp_path: Path,
) -> None:
    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[("sec-0", "Chapter 1", 0, [("b-0", "Harry stepped into the dim hallway.")])],
        registry_items_spec=[],
        handler=lambda _r, _n: (_ for _ in ()).throw(AssertionError("unused")),
    )
    with harness.uow.transaction() as tx:
        snap = harness.source.snapshot(tx, harness.book_id)
    sec0 = harness.document.ordered_sections[0]
    scene_id = snap.entries[0].scenes[0].id

    app = IllustrationAppearance(
        entity_id="ent-harry",
        name_ru="Гарри",
        name_en="Harry",
        description_ru="Шрам-молния на лбу; контрастная штриховка тушью",
        description_en="lightning-shaped scar on forehead; dark ink crosshatching",
    )

    # 1. Clause 1 covered in scene_composition_en, clause 2 missing -> only clause 2 in Visual details!
    partial_prompt = compose_monochrome_prompt(
        "Harry with a lightning-shaped scar on his forehead stands in the doorway.",
        [app],
    )
    assert "Visual details: Harry: dark ink crosshatching." in partial_prompt
    assert partial_prompt.lower().count("lightning-shaped scar") == 1
    validated_partial = validate_moment(
        ValidatedMoment(
            id="m-partial",
            scene_id=scene_id,
            title_ru="В дверях",
            reason_ru="Причина",
            event_ru="Событие",
            prompt_en=partial_prompt,
            insert_after_block_id="b-0",
            anchors=(SourceAnchor("b-0", 0, 5, "Harry"),),
            character_descriptions=(app,),
        ),
        snapshot=snap,
        section=sec0,
        scene_start_block_id="b-0",
        scene_end_block_id="b-0",
        registry={},
        event_position_key=format_position_key(0, 0, 0),
    )
    assert validated_partial.prompt_en == partial_prompt

    # 2. All clauses covered in scene_composition_en -> no Visual details section at all, still valid!
    full_prompt = compose_monochrome_prompt(
        "Harry with a lightning-shaped scar on his forehead rendered in dark ink crosshatching.",
        [app],
    )
    assert "Visual details:" not in full_prompt
    validate_moment(
        ValidatedMoment(
            id="m-full",
            scene_id=scene_id,
            title_ru="В дверях",
            reason_ru="Причина",
            event_ru="Событие",
            prompt_en=full_prompt,
            insert_after_block_id="b-0",
            anchors=(SourceAnchor("b-0", 0, 5, "Harry"),),
            character_descriptions=(app,),
        ),
        snapshot=snap,
        section=sec0,
        scene_start_block_id="b-0",
        scene_end_block_id="b-0",
        registry={},
        event_position_key=format_position_key(0, 0, 0),
    )

    # 3. Prefix non-match guard: "scarf" must NOT suppress a clause requiring "scar" (multi-word or single-word)!
    scar_app = IllustrationAppearance(
        entity_id="ent-harry",
        name_ru="Гарри",
        name_en="Harry",
        description_ru="Шрам на лбу",
        description_en="jagged scar on forehead",
    )
    scarf_prompt = compose_monochrome_prompt(
        "Harry wears a jagged wool scarf across his forehead.",
        [scar_app],
    )
    assert "Visual details: Harry: jagged scar on forehead." in scarf_prompt

    single_word_scar_app = IllustrationAppearance(
        entity_id="ent-harry",
        name_ru="Гарри",
        name_en="Harry",
        description_ru="Шрам; тёмный плащ",
        description_en="scar; dark cloak",
    )
    single_word_scar_prompt = compose_monochrome_prompt(
        "A boy wearing a scarf and a dark cloak.",
        [single_word_scar_app],
    )
    assert "Visual details: Harry: scar." in single_word_scar_prompt

    # 4. Symmetric 4-letter singular/plural matching ("coat" <-> "coats").
    coat_app = IllustrationAppearance(
        entity_id="ent-harry",
        name_ru="Гарри",
        name_en="Harry",
        description_ru="Тёмное пальто",
        description_en="dark wool coat",
    )
    coats_prompt = compose_monochrome_prompt(
        "Figures wearing dark wool coats stand by the gate.",
        [coat_app],
    )
    assert "Visual details:" not in coats_prompt

    # 5. Semicolon-only description_en is rejected consistently by compose_monochrome_prompt.
    semicolon_only = IllustrationAppearance(
        entity_id="ent-harry",
        name_ru="Гарри",
        name_en="Harry",
        description_ru="Пусто",
        description_en="; ;",
    )
    with pytest.raises(IllustrationValidationError):
        compose_monochrome_prompt("Harry stands in the hallway.", [semicolon_only])


def test_materialize_appearances_handles_legacy_none_and_empty_used_keys_fallback() -> None:
    from linguary.application.illustration_chapter import _materialize_appearances_for_prompt

    base_char = IllustrationAppearance(
        entity_id="ent-1",
        name_ru="Гарри",
        name_en="Harry",
        description_ru="Гарри",
        description_en="Harry",
    )
    # 1. Legacy summary with design_traits_by_key=None returns base appearances untouched.
    legacy_chars, legacy_locs = _materialize_appearances_for_prompt(
        [base_char], [], None, ["ent-1:hair"], base_fallback_entity_ids=("ent-1",)
    )
    assert legacy_chars == (base_char,)
    assert legacy_locs == ()

    # 2. Modern summary with used_design_keys=() keeps fallback_en instead of emptying description_en.
    no_traits_chars, _ = _materialize_appearances_for_prompt(
        [base_char],
        [],
        {
            "ent-1:hair": {
                "entity_id": "ent-1",
                "trait": "hair",
                "description_ru": "Тёмные волосы",
                "description_en": "dark hair",
                "assumption_ru": "Волосы",
            }
        },
        [],
        base_fallback_entity_ids=("ent-1",),
    )
    assert no_traits_chars[0].description_en == "Harry"
    assert no_traits_chars[0].assumptions_ru == ()


def test_compute_chapter_fingerprint_uses_v2_kind() -> None:
    entry = SectionManifestEntry(
        section_id="sec-0",
        order=0,
        title="Chapter 1",
        skipped=False,
        skip_reason=None,
        block_digests={"b-0": "a" * 64},
        input_sha256="b" * 64,
        response_sha256="c" * 64,
    )
    fp = compute_chapter_fingerprint(
        entry,
        registry_sha256="d" * 64,
        policy_payload={"policy_version": 1},
        model_prompts="gpt-5-mini",
        route_id="route-1",
        style="monochrome_ink",
        max_per_chapter=3,
    )
    expected_v2 = canonical_sha(
        {
            "kind": "illustration_chapter_v2",
            "registry_sha256": "d" * 64,
            "section_id": "sec-0",
            "block_digests": {"b-0": "a" * 64},
            "input_sha256": "b" * 64,
            "response_sha256": "c" * 64,
            "model_prompts": "gpt-5-mini",
            "route_id": "route-1",
            "style": "monochrome_ink",
            "max_per_chapter": 3,
            "policy": {"policy_version": 1},
        }
    )
    assert fp == expected_v2


def test_resolve_entity_name_prefers_canonical_over_alias_and_preserves_ambiguous() -> None:
    from linguary.application.illustration_chapter import _resolve_entity_name

    entities = [
        {
            "entity_id": "ent-street",
            "entity_kind": "location",
            "canonical_name": "Privet Drive",
            "aliases": [],
        },
        {
            "entity_id": "ent-house",
            "entity_kind": "location",
            "canonical_name": "number four, Privet Drive",
            "aliases": ["Privet Drive", "4 Privet Drive"],
        },
        {
            "entity_id": "ent-john-1",
            "entity_kind": "person",
            "canonical_name": "John",
            "aliases": ["the tall man"],
        },
        {
            "entity_id": "ent-john-2",
            "entity_kind": "person",
            "canonical_name": "John",
            "aliases": ["the tall man"],
        },
    ]
    # Pass 1 (canonical_name) uniquely resolves "Privet Drive" to the street despite alias on house!
    st, ent = _resolve_entity_name("Privet Drive", expected_kind="location", entities=entities)
    assert st == "resolved" and ent is not None and ent["entity_id"] == "ent-street"

    # Pass 2 (aliases) resolves "4 Privet Drive" uniquely to the house!
    st2, ent2 = _resolve_entity_name("4 Privet Drive", expected_kind="location", entities=entities)
    assert st2 == "resolved" and ent2 is not None and ent2["entity_id"] == "ent-house"

    # Duplicate canonical_name remains ambiguous!
    st3, ent3 = _resolve_entity_name("John", expected_kind="person", entities=entities)
    assert st3 == "ambiguous" and ent3 is None

    # Duplicate alias with no canonical match remains ambiguous!
    st4, ent4 = _resolve_entity_name("the tall man", expected_kind="person", entities=entities)
    assert st4 == "ambiguous" and ent4 is None

    # Unknown name is unmatched!
    st5, ent5 = _resolve_entity_name("Hogwarts", expected_kind="location", entities=entities)
    assert st5 == "unmatched" and ent5 is None


async def test_display_name_at_position_prevents_name_spoilers_in_fact_scope_and_appearances(
    tmp_path: Path,
) -> None:
    fact_scope_entity_names: list[tuple[str, str]] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-early",
                        title_ru="Кошка на стене",
                        event_ru="Полосатая кошка сидит на садовой стене.",
                        significance_ru="Загадочное наблюдение",
                        event_kind="observation",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(block_id="b-0", quote="A tabby cat sat"),
                        ),
                        participant_names=("tabby cat",),
                    ),
                    IllustrationCandidateEventDTO(
                        event_id="ev-late",
                        title_ru="Появление профессора",
                        event_ru="Профессор Макгонагалл обращается к Дамблдору.",
                        significance_ru="Раскрытие личности",
                        event_kind="revelation",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(
                                block_id="b-2", quote="Professor McGonagall spoke"
                            ),
                        ),
                        participant_names=("Professor McGonagall",),
                    ),
                ),
                covered_source_unit_ids=tuple(b["source_unit_id"] for b in data["blocks"]),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cids = tuple(c["candidate_id"] for c in data["candidates"])
            return IllustrationSelectionResponse(
                selected_candidate_ids=cids,
                selection_reasons_ru=tuple(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Важный кадр")
                    for cid in cids
                ),
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_FACT_SCOPE:
            fact_scope_entity_names.append((data["event_position_key"], data["entity_name"]))
            fids = tuple(f["fact_id"] for f in data["facts"])
            return IllustrationFactScopeResponse(
                event_id=data["event_id"],
                applicable_fact_ids=fids,
                excluded_fact_ids=(),
                compact_description_ru="Стройный силуэт",
                compact_description_en="Slender silhouette",
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_PROMPT:
            return IllustrationPromptResponse(
                moment_id=data["moment_id"],
                scene_composition_en="Monochrome scene on the quiet street.",
                depicted_entity_ids=tuple(c["entity_id"] for c in data["characters"]),
                used_fact_ids=tuple(data["allowed_fact_ids"]),
                used_design_keys=(),
            )
        raise AssertionError(req.kind)

    mcgonagall_pay = {
        "entity_id": "ent-mc",
        "entity_kind": "person",
        "canonical_name": "Professor McGonagall",
        "aliases": ["tabby cat"],
        "alias_first_positions": {
            "tabby cat": format_position_key(0, 0, 0),
            "Professor McGonagall": format_position_key(0, 2, 0),
        },
        "section_orders": [0],
        "mention_count": 2,
        "first_position_key": format_position_key(0, 0, 0),
    }
    mcgonagall_item = RegistryItem(
        generation_id="placeholder",
        kind=IllustrationRegistryKind.ENTITY,
        item_key="ent-mc",
        entity_id="ent-mc",
        position_key=TIMELESS_POSITION_KEY,
        is_back_applicable=False,
        payload_sha256=canonical_sha(mcgonagall_pay),
        payload=mcgonagall_pay,
    )

    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[
            (
                "sec-0",
                "Chapter 1",
                0,
                [
                    ("b-0", "A tabby cat sat stiffly on the brick wall."),
                    ("b-1", "The street grew dark as lamps went out."),
                    ("b-2", "Professor McGonagall spoke sharply to the tall wizard."),
                ],
            )
        ],
        registry_items_spec=[
            mcgonagall_item,
            _fact_item(
                "f-cat",
                "ent-mc",
                section_order=0,
                block_order=0,
                temporal_scope="episodic",
                desc_ru="Полосатая кошка",
                desc_en="Tabby cat",
            ),
        ],
        handler=_handler,
        book_id="book-no-name-spoiler",
    )
    await _drain_chapter_tasks(harness)

    fact_scope_entity_names.sort(key=lambda x: x[0])
    assert fact_scope_entity_names[0][1] == "tabby cat"
    assert fact_scope_entity_names[1][1] == "Professor McGonagall"

    with harness.uow.transaction() as tx:
        ch = tx.illustrations.list_scopes(
            harness.generation_id, stage=IllustrationScopeStage.CHAPTER, section_id="sec-0"
        )[0]
        assert ch.state is IllustrationScopeState.ACCEPTED and ch.payload is not None
        moments = ch.payload["moments"]
        assert len(moments) == 2
        early_char = moments[0]["character_descriptions"][0]
        late_char = moments[1]["character_descriptions"][0]
        assert early_char["name_ru"] == "tabby cat"
        assert early_char["name_en"] == "tabby cat"
        assert "McGonagall" not in moments[0]["prompt_en"]
        assert late_char["name_ru"] == "Professor McGonagall"
        assert late_char["name_en"] == "Professor McGonagall"


async def test_depicted_entity_ids_filters_prompt_and_moment_and_validates_owners(
    tmp_path: Path,
) -> None:
    prompt_inputs_seen: list[dict[str, Any]] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-photos",
                        title_ru="Фотографии на каминной полке",
                        event_ru="На каминной полке стоят фотографии Дадли, а Гарри в кадре нет.",
                        significance_ru="Контраст отношения семьи",
                        event_kind="detail",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(
                                block_id=blk["block_id"], quote="photographs of Dudley"
                            ),
                        ),
                        # Absent Harry mistakenly listed by CANDIDATES; PROMPT excludes him via depicted_entity_ids!
                        participant_names=("Dudley", "Harry"),
                        location_names=(),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cid = data["candidates"][0]["candidate_id"]
            return IllustrationSelectionResponse(
                selected_candidate_ids=(cid,),
                selection_reasons_ru=(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Яркая деталь"),
                ),
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_FACT_SCOPE:
            fids = tuple(f["fact_id"] for f in data["facts"])
            desc_en = ", ".join(f["description_en"] for f in data["facts"])
            desc_ru = ", ".join(f["description_ru"] for f in data["facts"])
            return IllustrationFactScopeResponse(
                event_id=data["event_id"],
                applicable_fact_ids=fids,
                excluded_fact_ids=(),
                compact_description_ru=desc_ru,
                compact_description_en=desc_en,
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_PROMPT:
            prompt_inputs_seen.append(data)
            return IllustrationPromptResponse(
                moment_id=data["moment_id"],
                scene_composition_en="Mantelpiece lined with framed photographs of a plump boy.",
                depicted_entity_ids=("ent-dudley",),
                used_fact_ids=("f-dudley",),
                used_design_keys=(),
            )
        raise AssertionError(req.kind)

    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[
            (
                "sec-0",
                "Chapter 1",
                0,
                [("b-0", "Rows of photographs of Dudley stood on the mantelpiece, with no sign of Harry.")],
            )
        ],
        registry_items_spec=[
            _entity_item("ent-dudley", "Dudley"),
            _entity_item("ent-harry", "Harry"),
            _fact_item(
                "f-dudley",
                "ent-dudley",
                section_order=0,
                temporal_scope="permanent",
                desc_ru="Пухлый мальчик",
                desc_en="plump boy",
            ),
            _fact_item(
                "f-harry",
                "ent-harry",
                section_order=0,
                temporal_scope="permanent",
                desc_ru="Шрам-молния на лбу",
                desc_en="lightning-shaped scar on forehead",
            ),
        ],
        handler=_handler,
        book_id="book-depicted",
    )
    await _drain_chapter_tasks(harness)

    # Empty locations -> scene_location_ru reference field passed to PROMPT!
    assert len(prompt_inputs_seen) == 1
    assert prompt_inputs_seen[0]["locations"] == []
    assert prompt_inputs_seen[0]["scene_location_ru"] == "Зал"

    with harness.uow.transaction() as tx:
        ch = tx.illustrations.list_scopes(
            harness.generation_id, stage=IllustrationScopeStage.CHAPTER, section_id="sec-0"
        )[0]
        assert ch.state is IllustrationScopeState.ACCEPTED and ch.payload is not None
        moment = ch.payload["moments"][0]
        assert [c["entity_id"] for c in moment["character_descriptions"]] == ["ent-dudley"]
        assert "Harry" not in moment["prompt_en"]
        assert "scar" not in moment["prompt_en"].casefold()

    # Unit checks for _validate_render_response on depicted_entity_ids invariants & legacy None
    from linguary.domain.illustrations import IllustrationScope, IllustrationSettingsRevision

    chapter = IllustrationChapter.__new__(IllustrationChapter)
    now = datetime(2026, 11, 26, tzinfo=UTC)
    rev = IllustrationSettingsRevision(
        id="rev-1",
        book_id="book-1",
        revision=1,
        settings_sha256="a" * 64,
        settings=IllustrationSettings(enabled=True, model_reference="m", model_prompts="m"),
        created_at=now,
    )
    scope = IllustrationScope(
        id="sc-p",
        generation_id="gen-1",
        parent_scope_id=None,
        stage=IllustrationScopeStage.PROMPT,
        scope_key="prompt:sec-0:m1",
        section_id="sec-0",
        section_order=0,
        section_title="Ch 1",
        task_id=None,
        input_sha256="a" * 64,
        state=IllustrationScopeState.RUNNING,
        reason_code=None,
        attempt_id=None,
        output_sha256=None,
        payload=None,
        summary={
            "moment_id": "m1",
            "allowed_fact_ids": ["f-dudley", "f-harry"],
            "allowed_design_keys": ["ent-harry:hair"],
            "design_traits_by_key": {
                "ent-harry:hair": {
                    "key": "ent-harry:hair",
                    "entity_id": "ent-harry",
                    "trait": "hair",
                    "description_ru": "Взъерошенные волосы",
                    "description_en": "messy hair",
                    "assumption_ru": "",
                }
            },
            "characters": [
                IllustrationAppearance("ent-dudley", "Dudley", "Пухлый", "plump boy", ("f-dudley",), (), "Dudley").to_dict(),
                IllustrationAppearance("ent-harry", "Harry", "Шрам", "scar on forehead", ("f-harry",), (), "Harry").to_dict(),
            ],
            "locations": [],
        },
        created_at=now,
        updated_at=now,
    )

    # Duplicate depicted_entity_ids -> rejected
    assert chapter._validate_render_response(
        scope,
        IllustrationPromptResponse("m1", "A plump boy.", ("f-dudley",), (), (), ("ent-dudley", "ent-dudley")),
        rev,
    ).ok is False
    # Unknown entity_id -> rejected
    assert chapter._validate_render_response(
        scope,
        IllustrationPromptResponse("m1", "A plump boy.", (), (), (), ("ent-unknown",)),
        rev,
    ).ok is False
    # Fact owner not in depicted_entity_ids -> rejected
    assert chapter._validate_render_response(
        scope,
        IllustrationPromptResponse("m1", "A plump boy.", ("f-harry",), (), (), ("ent-dudley",)),
        rev,
    ).ok is False
    # Design key owner not in depicted_entity_ids -> rejected
    assert chapter._validate_render_response(
        scope,
        IllustrationPromptResponse("m1", "A plump boy.", (), ("ent-harry:hair",), (), ("ent-dudley",)),
        rev,
    ).ok is False
    # Empty depicted_entity_ids=() -> valid still-life/object prompt with no Visual details block
    empty_ok = chapter._validate_render_response(
        scope,
        IllustrationPromptResponse("m1", "A narrow mantelpiece lined with framed photographs.", (), (), (), ()),
        rev,
    )
    assert empty_ok.ok is True and empty_ok.payload is not None
    assert empty_ok.payload["depicted_entity_ids"] == []
    assert "Visual details:" not in empty_ok.payload["prompt_en"]

    # Legacy depicted_entity_ids=None -> backward compatible (keeps both characters)
    legacy_ok = chapter._validate_render_response(
        scope,
        IllustrationPromptResponse("m1", "Two boys stand in the room.", ("f-dudley", "f-harry"), (), (), None),
        rev,
    )
    assert legacy_ok.ok is True and legacy_ok.payload is not None
    assert "depicted_entity_ids" not in legacy_ok.payload
    assert "scar on forehead" in legacy_ok.payload["prompt_en"]


def test_select_english_name_at_position_edge_cases() -> None:
    from linguary.application.illustration_chapter import _select_english_name_at_position

    # Trimmed key with surrounding whitespace + proper name preferred over earlier generic alias
    entity_with_spaces = {
        "canonical_name": "Прайвет-драйв",
        "aliases": ["the street", "Privet Drive"],
        "alias_first_positions": {
            " the street ": "0001:000001:00000001",
            " Privet Drive ": "0001:000002:00000001",
            "Hogwarts": "0002:000001:00000001",
        },
    }
    assert (
        _select_english_name_at_position(
            entity_with_spaces, "0001:000005:00000001", "Прайвет-драйв"
        )
        == "Privet Drive"
    )
    # Empty alias_first_positions falls back to _select_english_name(entity)
    entity_empty_map = {
        "canonical_name": "Прайвет-драйв",
        "aliases": ["Privet Drive"],
        "alias_first_positions": {},
    }
    assert (
        _select_english_name_at_position(
            entity_empty_map, "0001:000005:00000001", "Прайвет-драйв"
        )
        == "Privet Drive"
    )


def test_punctuation_cleanup_and_single_word_clause_with_trailing_dot(tmp_path: Path) -> None:
    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[("sec-0", "Chapter 1", 0, [("b-0", "Alice stood at the gate.")])],
        registry_items_spec=[],
        handler=lambda _r, _n: (_ for _ in ()).throw(AssertionError("unused")),
        book_id="book-punct",
    )
    with harness.uow.transaction() as tx:
        snap = harness.source.snapshot(tx, harness.book_id)
    sec0 = harness.document.ordered_sections[0]
    scene_id = snap.entries[0].scenes[0].id

    app = IllustrationAppearance(
        entity_id="ent-alice",
        name_ru="Alice",
        name_en="Alice",
        description_ru="Шрам.; Тёмный плащ.",
        description_en="scar.; dark cloak.",
    )
    prompt = compose_monochrome_prompt("Alice stands beside the gate..", [app])
    assert ".;" not in prompt
    assert ".." not in prompt
    assert "Visual details: Alice: scar; dark cloak." in prompt

    validated = validate_moment(
        ValidatedMoment(
            id="m-punct",
            scene_id=scene_id,
            title_ru="У ворот",
            reason_ru="Причина",
            event_ru="Событие",
            prompt_en=prompt,
            insert_after_block_id="b-0",
            anchors=(SourceAnchor("b-0", 0, 5, "Alice"),),
            character_descriptions=(app,),
        ),
        snapshot=snap,
        section=sec0,
        scene_start_block_id="b-0",
        scene_end_block_id="b-0",
        registry={},
        event_position_key=format_position_key(0, 0, 0),
    )
    assert validated.prompt_en == prompt


def test_display_name_ru_at_position_and_old_entity_compat() -> None:
    from linguary.application.illustration_chapter import (
        _display_name_ru_at_position,
        _resolve_entity_name,
    )

    # Old ENTITY without glossary_translation_ru / alias_translations_ru works without KeyError.
    old_entity = {
        "entity_id": "ent-old",
        "entity_kind": "person",
        "canonical_name": "Nick",
        "aliases": ["the boy"],
    }
    assert _display_name_ru_at_position(old_entity, format_position_key(0, 0, 0), "Nick") == "Nick"

    # New ENTITY with glossary_translation_ru & alias_translations_ru localizes non-spoiler names.
    new_entity = {
        "entity_id": "ent-nick",
        "entity_kind": "person",
        "canonical_name": "Nick",
        "aliases": ["the boy", "Chocolate Boy"],
        "glossary_translation_ru": "Ник",
        "alias_translations_ru": {
            "Nick": "Ник",
            "Chocolate Boy": "Шоколадный мальчик",
        },
    }
    # Early anonymous alias without translation does NOT spoil "Ник"!
    assert _display_name_ru_at_position(new_entity, format_position_key(0, 0, 0), "the boy") == "the boy"
    # Revealed canonical name localizes to "Ник".
    assert _display_name_ru_at_position(new_entity, format_position_key(1, 0, 0), "Nick") == "Ник"
    # Alias translation works too.
    assert (
        _display_name_ru_at_position(new_entity, format_position_key(1, 38, 0), "Chocolate Boy")
        == "Шоколадный мальчик"
    )

    # Defensive stripping of "(aliases: ...)" suffix in _resolve_entity_name.
    st, ent = _resolve_entity_name(
        "Nick (aliases: the boy, Chocolate Boy)",
        expected_kind="person",
        entities=[new_entity],
    )
    assert st == "resolved" and ent is not None and ent["entity_id"] == "ent-nick"


async def test_chapter_glossary_terms_ru_non_spoiler_localized_appearances_and_wire_budget(
    tmp_path: Path,
) -> None:
    from dataclasses import replace as dc_replace
    from types import SimpleNamespace

    from linguary.application.illustration_context import (
        REPAIR_RESERVE_BYTES,
        IllustrationContext,
        _collect_book_glossary_terms_ru,
        fit_glossary_terms_to_wire_budget,
    )
    from linguary.application.llm_dto import GlossaryCandidateResponse
    from linguary.domain.glossary import ConceptType, TermConcept, TermStatus, TermVariant
    from linguary.domain.llm_log import LLMCallStage

    # Defensive check: fake tx without glossary attr returns {}.
    assert _collect_book_glossary_terms_ru(SimpleNamespace(), "book-x") == {}

    fact_scope_inputs: list[dict[str, Any]] = []
    prompt_inputs: list[dict[str, Any]] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            if data["section_id"] == "sec-1":
                return IllustrationCandidatesResponse(
                    scene_id=data["scene_id"],
                    events=(),
                    zero_reason_ru="Тихая глава.",
                    covered_source_unit_ids=tuple(b["source_unit_id"] for b in data["blocks"]),
                    complete=True,
                )
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-1",
                        title_ru="Ник в лесу",
                        event_ru="Ник смотрит на деревья.",
                        significance_ru="Ключевая встреча",
                        event_kind="encounter",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(
                                block_id="b-0", quote="Nick looked at the trees"
                            ),
                        ),
                        participant_names=("Nick", "Allie"),
                    ),
                ),
                covered_source_unit_ids=tuple(b["source_unit_id"] for b in data["blocks"]),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cids = tuple(c["candidate_id"] for c in data["candidates"])
            return IllustrationSelectionResponse(
                selected_candidate_ids=cids,
                selection_reasons_ru=tuple(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Ключевой кадр")
                    for cid in cids
                ),
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_FACT_SCOPE:
            fact_scope_inputs.append(data)
            fids = tuple(f["fact_id"] for f in data["facts"])
            return IllustrationFactScopeResponse(
                event_id=data["event_id"],
                applicable_fact_ids=fids,
                excluded_fact_ids=(),
                compact_description_ru="Мальчик в тёмном пальто",
                compact_description_en="Boy in a dark coat",
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_PROMPT:
            prompt_inputs.append(data)
            return IllustrationPromptResponse(
                moment_id=data["moment_id"],
                scene_composition_en="Nick and Allie stand among tall trees in crisp ink.",
                depicted_entity_ids=tuple(c["entity_id"] for c in data["characters"]),
                used_fact_ids=tuple(data["allowed_fact_ids"]),
                used_design_keys=(),
            )
        raise AssertionError(req.kind)

    nick_pay = {
        "entity_id": "ent-nick",
        "entity_kind": "person",
        "canonical_name": "Nick",
        "aliases": [],
        "alias_first_positions": {"Nick": format_position_key(0, 0, 0)},
        "glossary_translation_ru": "Ник",
        "alias_translations_ru": {"Nick": "Ник"},
        "section_orders": [0],
        "mention_count": 2,
        "first_position_key": format_position_key(0, 0, 0),
    }
    # Allie has 0 facts -> verifies fallback description_ru also uses localized name_ru ("Элли")!
    allie_pay = {
        "entity_id": "ent-allie",
        "entity_kind": "person",
        "canonical_name": "Allie",
        "aliases": [],
        "alias_first_positions": {"Allie": format_position_key(0, 0, 10)},
        "glossary_translation_ru": "Элли",
        "alias_translations_ru": {"Allie": "Элли"},
        "section_orders": [0],
        "mention_count": 1,
        "first_position_key": format_position_key(0, 0, 10),
    }

    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[
            (
                "sec-0",
                "Chapter 1",
                0,
                [("b-0", "Nick looked at the trees in Everlost while Allie waited nearby.")],
            ),
            (
                "sec-1",
                "Chapter 2",
                1,
                [("b-1", "The McGill sailed across the misty sea.")],
            ),
        ],
        registry_items_spec=[
            RegistryItem(
                generation_id="placeholder",
                kind=IllustrationRegistryKind.ENTITY,
                item_key="ent-nick",
                entity_id="ent-nick",
                position_key=TIMELESS_POSITION_KEY,
                is_back_applicable=False,
                payload_sha256=canonical_sha(nick_pay),
                payload=nick_pay,
            ),
            RegistryItem(
                generation_id="placeholder",
                kind=IllustrationRegistryKind.ENTITY,
                item_key="ent-allie",
                entity_id="ent-allie",
                position_key=TIMELESS_POSITION_KEY,
                is_back_applicable=False,
                payload_sha256=canonical_sha(allie_pay),
                payload=allie_pay,
            ),
            _fact_item(
                "f-nick-coat",
                "ent-nick",
                section_order=0,
                block_order=0,
                temporal_scope="episodic",
                desc_ru="В тёмном пальто",
                desc_en="Wearing a dark coat",
            ),
        ],
        handler=_handler,
        book_id="book-ch-gloss",
    )

    # Seed glossary concepts AND verify NARRATIVE pass (without PEDAGOGICAL) confirms ch0 term and blocks ch1 spoiler!
    with harness.uow.transaction() as tx:
        for cid, src, ru, ctype in (
            ("c-nick", "Nick", "Ник", ConceptType.PERSON),
            ("c-allie", "Allie", "Элли", ConceptType.PERSON),
            ("c-everlost", "Everlost", "Эверлост", ConceptType.PLACE),
            ("c-mcgill", "The McGill", "Макгилл", ConceptType.PERSON),
        ):
            tx.glossary.save(
                TermConcept(
                    id=cid,
                    book_id=harness.book_id,
                    type=ctype,
                    canonical_source_text=src,
                    status=TermStatus.APPROVED,
                    variants=(
                        TermVariant(
                            concept_id=cid,
                            language="ru",
                            text=ru,
                            status=TermStatus.APPROVED,
                        ),
                    ),
                )
            )
        # Update NARRATIVE passes' glossary_candidates directly in payload to simulate NARRATIVE-only (no PEDAGOGICAL).
        p0 = tx.analyses.get_pass(harness.book_id, "sec-0", AnalysisPassKind.NARRATIVE, scope_key="")
        p1 = tx.analyses.get_pass(harness.book_id, "sec-1", AnalysisPassKind.NARRATIVE, scope_key="")
        assert p0 is not None and p1 is not None
        dec0 = decode_pass(p0.payload, pass_kind=AnalysisPassKind.NARRATIVE)
        dec1 = decode_pass(p1.payload, pass_kind=AnalysisPassKind.NARRATIVE)
        enc0 = encode_pass(
            NarrativeAnalysisResponse(
                translation_instructions=dec0.translation_instructions,
                chapter_summary=dec0.chapter_summary,
                glossary_candidates=(
                    GlossaryCandidateResponse(
                        source_text="Everlost",
                        type=ConceptType.PLACE,
                        suggested_translation="Эверлост",
                        confidence=0.95,
                        model_flagged_review=False,
                    ),
                ),
                context_facts=dec0.context_facts,
                scenes=dec0.scenes,
            )
        )
        enc0["input_suggestions"] = list(p0.payload["input_suggestions"])
        enc0["input_context"] = dict(p0.payload["input_context"])
        enc1 = encode_pass(
            NarrativeAnalysisResponse(
                translation_instructions=dec1.translation_instructions,
                chapter_summary=dec1.chapter_summary,
                glossary_candidates=(
                    GlossaryCandidateResponse(
                        source_text="The McGill",
                        type=ConceptType.PERSON,
                        suggested_translation="Макгилл",
                        confidence=0.95,
                        model_flagged_review=False,
                    ),
                ),
                context_facts=dec1.context_facts,
                scenes=dec1.scenes,
            )
        )
        enc1["input_suggestions"] = list(p1.payload["input_suggestions"])
        enc1["input_context"] = {**dict(p1.payload["input_context"]), "parallel_batch": True}
        tx.analyses.save_pass(
            AnalysisPassResult(
                id=p0.id,
                book_id=p0.book_id,
                section_id=p0.section_id,
                pass_kind=p0.pass_kind,
                scope_key=p0.scope_key,
                input_sha256=p0.input_sha256,
                output_sha256=canonical_sha(enc0),
                conditions_id=p0.conditions_id,
                attempt_id=p0.attempt_id,
                payload=enc0,
            )
        )
        tx.analyses.save_pass(
            AnalysisPassResult(
                id=p1.id,
                book_id=p1.book_id,
                section_id=p1.section_id,
                pass_kind=p1.pass_kind,
                scope_key=p1.scope_key,
                input_sha256=p1.input_sha256,
                output_sha256=canonical_sha(enc1),
                conditions_id=p1.conditions_id,
                attempt_id=p1.attempt_id,
                payload=enc1,
            )
        )

        # Re-seal a fresh generation matching the updated NARRATIVE passes so tasks stay current.
        snap = harness.source.snapshot(tx, harness.book_id)
        head = tx.illustrations.get_head(harness.book_id)
        assert head.settings_id is not None and snap.ready
        assert snap.source_sha256 is not None and snap.manifest is not None and snap.manifest_sha256 is not None
        tx.illustrations.revoke_generation(
            harness.book_id, harness.generation_id, reason_code="source_changed"
        )
        for existing_task in tx.tasks.list_for_book(harness.book_id):
            if existing_task.status in {TaskStatus.PENDING, TaskStatus.RUNNING}:
                tx.tasks.save(existing_task.supersede())
        gen2, _ = tx.illustrations.create_or_get_generation(
            harness.book_id,
            settings_id=head.settings_id,
            request_key="req-ch-gloss-2",
            source_sha256=snap.source_sha256,
            source_authority=snap.source_authority or {},
            requested_sections=snap.content_section_ids,
        )
        gen2 = tx.illustrations.seal_manifest(
            harness.book_id, gen2.id, manifest=snap.manifest, manifest_sha256=snap.manifest_sha256
        )
        tx.illustrations.upsert_registry_items(
            gen2.id,
            [
                RegistryItem(
                    generation_id=gen2.id,
                    kind=IllustrationRegistryKind.ENTITY,
                    item_key="ent-nick",
                    entity_id="ent-nick",
                    position_key=TIMELESS_POSITION_KEY,
                    is_back_applicable=False,
                    payload_sha256=canonical_sha(nick_pay),
                    payload=nick_pay,
                ),
                RegistryItem(
                    generation_id=gen2.id,
                    kind=IllustrationRegistryKind.ENTITY,
                    item_key="ent-allie",
                    entity_id="ent-allie",
                    position_key=TIMELESS_POSITION_KEY,
                    is_back_applicable=False,
                    payload_sha256=canonical_sha(allie_pay),
                    payload=allie_pay,
                ),
                dc_replace(
                    _fact_item(
                        "f-nick-coat",
                        "ent-nick",
                        section_order=0,
                        block_order=0,
                        temporal_scope="episodic",
                        desc_ru="В тёмном пальто",
                        desc_en="Wearing a dark coat",
                    ),
                    generation_id=gen2.id,
                ),
            ],
        )
        gen2 = tx.illustrations.seal_registry(harness.book_id, gen2.id, registry_sha256="c" * 64)
        gen2 = tx.illustrations.update_generation_state(
            harness.book_id, gen2.id, state=IllustrationGenerationState.QUEUED, reason_code=None
        )
        harness.generation_id = gen2.id
        harness.chapter.plan_chapter_scopes(tx, gen2, snap)

        # Direct check: ch0 sees Everlost/Nick/Allie, NOT The McGill; ch1 sees all 4.
        assert tx.analyses.list_for_book(harness.book_id) == ()
        terms_ch0 = _collect_book_glossary_terms_ru(
            tx, harness.book_id, generation_id=gen2.id, max_section_order=0
        )
        assert terms_ch0 == {"Allie": "Элли", "Everlost": "Эверлост", "Nick": "Ник"}
        terms_ch1 = _collect_book_glossary_terms_ru(
            tx, harness.book_id, generation_id=gen2.id, max_section_order=1
        )
        assert terms_ch1["The McGill"] == "Макгилл"

    await _drain_chapter_tasks(harness)

    # Check ch0 FACT_SCOPE and PROMPT received non-spoiler glossary_terms_ru.
    ch0_fs = [x for x in fact_scope_inputs if x["event_position_key"].startswith("000000:")]
    assert len(ch0_fs) == 1
    assert ch0_fs[0]["glossary_terms_ru"] == {"Allie": "Элли", "Everlost": "Эверлост", "Nick": "Ник"}
    assert "The McGill" not in ch0_fs[0]["glossary_terms_ru"]

    assert len(prompt_inputs) == 1
    p_in = prompt_inputs[0]
    assert p_in["glossary_terms_ru"] == {"Allie": "Элли", "Everlost": "Эверлост", "Nick": "Ник"}
    assert "The McGill" not in p_in["glossary_terms_ru"]

    chars_by_id = {c["entity_id"]: c for c in p_in["characters"]}
    assert chars_by_id["ent-nick"]["name_ru"] == "Ник"
    assert chars_by_id["ent-nick"]["name_en"] == "Nick"
    assert chars_by_id["ent-allie"]["name_ru"] == "Элли"
    assert chars_by_id["ent-allie"]["description_ru"] == "Элли"
    assert chars_by_id["ent-allie"]["name_en"] == "Allie"

    # Wire budget truncation helper truncates glossary_terms_ru smoothly down to {}.
    with harness.uow.read_transaction() as tx:
        gen = tx.illustrations.get_generation(harness.book_id, harness.generation_id)
        assert gen is not None
        rev = tx.illustrations.get_settings(harness.book_id, gen.settings_id)
        assert rev is not None
    sizer = harness.runtime.sizer_for(rev, LLMCallStage.ILLUSTRATION_CHAPTER)
    many = {f"Word{i:02d}": f"Слово{i:02d}" for i in range(40)}

    def _probe(g: dict[str, str]) -> LLMRequest[Any]:
        pay: dict[str, Any] = {"moment_id": "m1", "event_ru": "Событие"}
        if g:
            pay["glossary_terms_ru"] = g
        return IllustrationContext.build_for_kind(
            kind=LLMResponseKind.ILLUSTRATION_PROMPT,
            stage=IllustrationScopeStage.PROMPT,
            input_payload=pay,
            book_id=harness.book_id,
            conditions_id=None,
            policy=IllustrationPolicy(
                max_request_bytes=2000 + REPAIR_RESERVE_BYTES,
                repair_reserve_bytes=REPAIR_RESERVE_BYTES,
            ),
        )

    base_size = sizer.measure(_probe({})).size_bytes
    trimmed = fit_glossary_terms_to_wire_budget(
        many,
        build_request=_probe,
        sizer=sizer,
        target_bytes=base_size + 120,
    )
    assert 0 < len(trimmed) < len(many)
    assert sizer.measure(_probe(trimmed)).size_bytes <= base_size + 120

    # If even empty exceeds target, returns {}.
    assert (
        fit_glossary_terms_to_wire_budget(
            many,
            build_request=_probe,
            sizer=sizer,
            target_bytes=max(1, base_size - 10),
        )
        == {}
    )


async def test_trivial_permanent_fact_scope_fast_path_and_guards(tmp_path: Path) -> None:
    # 1. Single permanent fact without color words -> fast-path skips FACT_SCOPE LLM call
    #    and populates allowed_fact_ids + available_design_traits + IllustrationAppearance.
    fact_scope_entities: list[str] = []
    seen_prompt_inputs: list[dict[str, Any]] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-fp",
                        title_ru="Встреча героев",
                        event_ru="Алиса, Боб и Клара встречаются на площади.",
                        significance_ru="Ключевая встреча",
                        event_kind="encounter",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(
                                block_id=blk["block_id"], quote="Alice, Bob, Clara, David, and Eva met"
                            ),
                        ),
                        participant_names=("Alice", "Bob", "Clara", "David", "Eva"),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cid = data["candidates"][0]["candidate_id"]
            return IllustrationSelectionResponse(
                selected_candidate_ids=(cid,),
                selection_reasons_ru=(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Ключевой кадр"),
                ),
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_FACT_SCOPE:
            fact_scope_entities.append(str(data["entity_id"]))
            fids = tuple(f["fact_id"] for f in data["facts"])
            desc_by_ent = {
                "ent-clara": ("Светлый взгляд", "Bright gaze"),
                "ent-bob": ("В плаще", "In a cloak"),
                "ent-david": ("В тёмном плаще", "In a dark cloak"),
                "ent-eva": ("Высокая с короткими волосами", "Tall with short hair"),
            }
            desc_ru, desc_en = desc_by_ent[str(data["entity_id"])]
            return IllustrationFactScopeResponse(
                event_id=data["event_id"],
                applicable_fact_ids=fids,
                excluded_fact_ids=(),
                compact_description_ru=desc_ru,
                compact_description_en=desc_en,
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_PROMPT:
            seen_prompt_inputs.append(data)
            return IllustrationPromptResponse(
                moment_id=data["moment_id"],
                scene_composition_en="Alice, Bob, Clara, David, and Eva stand on the stone square in monochrome ink.",
                depicted_entity_ids=("ent-alice", "ent-bob", "ent-clara", "ent-david", "ent-eva"),
                used_fact_ids=tuple(data["allowed_fact_ids"]),
                used_design_keys=(),
            )
        raise AssertionError(req.kind)

    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[
            (
                "sec-0",
                "Ch 1",
                0,
                [("b-0", "Alice, Bob, Clara, David, and Eva met on the stone square.")],
            )
        ],
        registry_items_spec=[
            _entity_item("ent-alice", "Alice"),
            _entity_item("ent-bob", "Bob"),
            _entity_item("ent-clara", "Clara"),
            _entity_item("ent-david", "David"),
            _entity_item("ent-eva", "Eva"),
            # Alice: 1 permanent fact, no color words, clean EN/RU -> FAST-PATH!
            _fact_item(
                "f-alice-perm",
                "ent-alice",
                section_order=0,
                temporal_scope="permanent",
                desc_ru="Высокого роста",
                desc_en="Tall stature",
            ),
            _design_item(
                "ent-alice",
                "hair",
                desc_ru="Тёмные короткие волосы",
                desc_en="Short dark hair",
                assumption_ru="нейтральный выбор",
            ),
            # Bob: 1 episodic fact -> MUST call FACT_SCOPE (NO fast-path for episodic)!
            _fact_item(
                "f-bob-ep",
                "ent-bob",
                section_order=0,
                temporal_scope="episodic",
                desc_ru="В дорожном плаще",
                desc_en="Wearing a travelling cloak",
            ),
            # Clara: 1 permanent fact WITH chromatic color word ("Green") in monochrome_ink -> MUST call FACT_SCOPE!
            _fact_item(
                "f-clara-color",
                "ent-clara",
                section_order=0,
                temporal_scope="permanent",
                desc_ru="Яркие зелёные глаза",
                desc_en="Bright green eyes",
            ),
            # David: 1 permanent fact WITH Cyrillic in description_en -> MUST call FACT_SCOPE!
            _fact_item(
                "f-david-cyr",
                "ent-david",
                section_order=0,
                temporal_scope="permanent",
                desc_ru="Тёмный плащ",
                desc_en="Dark плащ",
            ),
            # Eva: 2 permanent facts -> MUST call FACT_SCOPE!
            _fact_item(
                "f-eva-p1",
                "ent-eva",
                section_order=0,
                temporal_scope="permanent",
                desc_ru="Высокого роста",
                desc_en="Tall stature",
            ),
            _fact_item(
                "f-eva-p2",
                "ent-eva",
                section_order=0,
                temporal_scope="permanent",
                desc_ru="Короткие волосы",
                desc_en="Short hair",
            ),
        ],
        handler=_handler,
        book_id="book-fact-fastpath",
    )
    await _drain_chapter_tasks(harness)

    # Alice bypassed FACT_SCOPE; Bob (episodic), Clara (color), David (Cyrillic in EN), Eva (2 permanent) ran FACT_SCOPE!
    assert "ent-alice" not in fact_scope_entities
    assert set(fact_scope_entities) == {"ent-bob", "ent-clara", "ent-david", "ent-eva"}
    assert len(seen_prompt_inputs) == 1
    p_in = seen_prompt_inputs[0]
    assert "f-alice-perm" in p_in["allowed_fact_ids"]
    assert "ent-alice:hair" in p_in["allowed_design_keys"]
    alice_char = next(c for c in p_in["characters"] if c["entity_id"] == "ent-alice")
    assert alice_char["description_en"] == "Tall stature"
    assert alice_char["description_ru"] == "Высокого роста"
    assert alice_char["fact_ids"] == ["f-alice-perm"]


async def test_prompt_truncated_retry_recovers_on_second_attempt(tmp_path: Path) -> None:
    from linguary.application.ports.llm import LLMResponseFormatError

    prompt_calls = 0
    seen_prompt_repairs: list[tuple[tuple[str | None, str], ...]] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        nonlocal prompt_calls
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-1",
                        title_ru="У ворот",
                        event_ru="Алиса открывает ворота.",
                        significance_ru="Начало пути",
                        event_kind="entry",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(block_id=blk["block_id"], quote="Alice opened"),
                        ),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cid = data["candidates"][0]["candidate_id"]
            return IllustrationSelectionResponse(
                selected_candidate_ids=(cid,),
                selection_reasons_ru=(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Ключевой кадр"),
                ),
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_PROMPT:
            prompt_calls += 1
            seen_prompt_repairs.append(req.repair_turns)
            if prompt_calls == 1:
                raise LLMResponseFormatError("truncated prompt response", truncated=True)
            return IllustrationPromptResponse(
                moment_id=data["moment_id"],
                scene_composition_en="Alice pushes open the heavy iron gate in crisp ink.",
                depicted_entity_ids=(),
                used_fact_ids=(),
                used_design_keys=(),
            )
        raise AssertionError(req.kind)

    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[("sec-0", "Ch 1", 0, [("b-0", "Alice opened the heavy iron gate.")])],
        registry_items_spec=[],
        handler=_handler,
        book_id="book-prompt-trunc-retry",
    )
    await _drain_chapter_tasks(harness)

    assert prompt_calls == 2
    assert seen_prompt_repairs[0] == ()
    assert seen_prompt_repairs[1] and "обрезан по длине" in seen_prompt_repairs[1][0][1]
    with harness.uow.read_transaction() as tx:
        ch = tx.illustrations.list_scopes(
            harness.generation_id, stage=IllustrationScopeStage.CHAPTER, section_id="sec-0"
        )[0]
        assert ch.state is IllustrationScopeState.ACCEPTED


async def test_prompt_anchor_quotes_and_technical_assumptions_filtered(tmp_path: Path) -> None:
    from linguary.application.illustration_context import _STAGE_INSTRUCTIONS
    from linguary.application.illustration_validation import filter_non_assumption_placeholders

    # 1. Deterministic filtering of technical keys in assumptions_ru without losing real assumptions.
    filtered = filter_non_assumption_placeholders(
        [
            "Использован ключ attire_default из справочника.",
            "Взято из characters[].description_en для согласованности.",
            "Параметр palette_value задаёт штриховку.",
            "Добавлено по used_design_keys и depicted_entity_ids.",
            "Основано на compact_description_en и description_ru.",
            "Не указан фасон плаща, выбран простой дорожный покрой.",
        ]
    )
    assert filtered == ("Не указан фасон плаща, выбран простой дорожный покрой.",)

    # 2. Prompt stage instructions preserve characters[].description_en AND forbid technical keys in assumptions_ru.
    prompt_instr = _STAGE_INSTRUCTIONS[IllustrationScopeStage.PROMPT]
    assert "characters[].description_en" in prompt_instr
    assert "attire_default" in prompt_instr
    fact_scope_instr = _STAGE_INSTRUCTIONS[IllustrationScopeStage.FACT_SCOPE]
    assert "транспорт" in fact_scope_instr.casefold()
    assert "деформац" in fact_scope_instr.casefold()

    # 3. End-to-end PROMPT receives anchor_quotes (<= 3, <= 240 chars) and strips technical assumptions without repair!
    seen_prompt_inputs: list[dict[str, Any]] = []
    prompt_calls = 0
    long_quote = "Alice stepped across the old bridge " + ("under flickering lanterns " * 12)
    assert len(long_quote) > 260

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        nonlocal prompt_calls
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-q",
                        title_ru="На мосту",
                        event_ru="Алиса идёт по мосту под фонарями.",
                        significance_ru="Ключевой переход",
                        event_kind="crossing",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(block_id=blk["block_id"], quote=long_quote),
                        ),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cid = data["candidates"][0]["candidate_id"]
            return IllustrationSelectionResponse(
                selected_candidate_ids=(cid,),
                selection_reasons_ru=(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Яркий момент"),
                ),
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_PROMPT:
            prompt_calls += 1
            seen_prompt_inputs.append(data)
            return IllustrationPromptResponse(
                moment_id=data["moment_id"],
                scene_composition_en="Alice crosses an old arched stone bridge beneath dim lanterns.",
                depicted_entity_ids=(),
                used_fact_ids=(),
                used_design_keys=(),
                assumptions_ru=(
                    "Поле description_en не уточняет тип фонарей.",
                    "Фонари изображены подвесными коваными.",
                ),
            )
        raise AssertionError(req.kind)

    harness = _seed_book_with_registry(
        tmp_path,
        sections_spec=[("sec-0", "Ch 1", 0, [("b-0", long_quote)])],
        registry_items_spec=[],
        handler=_handler,
        book_id="book-prompt-hygiene",
    )
    await _drain_chapter_tasks(harness)

    assert prompt_calls == 1  # No repair turn wasted on filtering technical assumption!
    assert len(seen_prompt_inputs) == 1
    anchor_quotes = seen_prompt_inputs[0].get("anchor_quotes")
    assert isinstance(anchor_quotes, list) and len(anchor_quotes) == 1
    assert len(anchor_quotes[0]) == 240
    assert anchor_quotes[0] == long_quote.strip()[:240]

    with harness.uow.read_transaction() as tx:
        ch = tx.illustrations.list_scopes(
            harness.generation_id, stage=IllustrationScopeStage.CHAPTER, section_id="sec-0"
        )[0]
        assert ch.payload is not None
        assert list(ch.payload["moments"][0]["assumptions_ru"]) == [
            "Фонари изображены подвесными коваными."
        ]

