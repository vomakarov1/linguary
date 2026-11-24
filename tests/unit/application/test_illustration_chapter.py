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
    _pass_model_dependencies,
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
    seen_in_ch0: list[str] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
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
            seen_in_ch0.extend(f["fact_id"] for f in data["facts"])
            return IllustrationFactScopeResponse(
                event_id=data["event_id"],
                applicable_fact_ids=tuple(f["fact_id"] for f in data["facts"]),
                excluded_fact_ids=(),
                compact_description_ru="Высокая, с прямыми волосами",
                compact_description_en="Tall with straight dark hair",
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_PROMPT:
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
            # Permanent + back_applicable + NOT spoiler -> INCLUDED in ch0!
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
    assert seen_in_ch0 == ["f-perm-ok"]


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
