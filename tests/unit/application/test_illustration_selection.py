"""Unit and persistence tests for bounded chapter moment selection (`P3.T1`, `ALG-03`)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from linguary.application.analysis_pass_cache import encode_pass
from linguary.application.context_builder import ContextBuilder
from linguary.application.illustration_dto import (
    IllustrationCandidateEventDTO,
    IllustrationCandidatesResponse,
    IllustrationQuoteAnchorDTO,
    IllustrationSelectionReasonDTO,
    IllustrationSelectionResponse,
)
from linguary.application.illustration_selection import (
    IllustrationSelection,
    SelectedAnchor,
    SelectedMomentCard,
    deduplicate_candidates,
    enforce_intra_chapter_diversity,
    moments_conflict_diversity,
)
from linguary.application.illustration_source import (
    IllustrationSource,
    _pass_model_dependencies,
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
from linguary.domain.content_plan import canonical_sha
from linguary.domain.illustrations import (
    IllustrationGenerationState,
    IllustrationPolicy,
    IllustrationReasonCode,
    IllustrationScopeStage,
    IllustrationScopeState,
    IllustrationSettings,
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
    selection: IllustrationSelection
    book_id: str
    section_id: str
    generation_id: str


def _setup_chapter(
    tmp_path: Path,
    *,
    scenes_spec: list[tuple[str, list[tuple[str, str]]]],
    handler: Any,
    max_per_chapter: int = 3,
    policy: IllustrationPolicy | None = None,
    book_id: str = "book-sel",
) -> _Harness:
    db = tmp_path / f"{book_id}.db"
    apply_migrations(db)
    uow = SqlAlchemyUnitOfWork.for_engine(create_database_engine(db))
    source_sha = "b" * 64
    conditions_id = "22222222-2222-4222-8222-222222222222"
    section_id = "sec-1"

    blocks: list[DocumentBlock] = []
    quoted_scenes: list[QuotedSceneResponse] = []
    b_order = 0
    for scene_summary, scene_blocks in scenes_spec:
        start_id = scene_blocks[0][0]
        end_id = scene_blocks[-1][0]
        for b_id, b_text in scene_blocks:
            blocks.append(
                DocumentBlock(
                    id=b_id,
                    section_id=section_id,
                    type=BlockType.PARAGRAPH,
                    order=b_order,
                    text=b_text,
                    source_reference=SourceReference("Text/ch1.xhtml", f"p[{b_order}]"),
                    support_status=SupportStatus.SUPPORTED,
                )
            )
            b_order += 1
        quoted_scenes.append(
            QuotedSceneResponse(
                start_block_id=start_id,
                end_block_id=end_id,
                summary=scene_summary,
                location="Замок",
                participants=(),
                visual_facts=(),
                visual_assumptions=(),
                continues_previous_scene=False,
            )
        )

    section = DocumentSection(
        id=section_id,
        document_id=book_id,
        order=0,
        title="Chapter 1",
        source_resource_path="Text/ch1.xhtml",
        blocks=tuple(blocks),
    )
    document = CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title="Selection Novel"),
        source_language="en",
        target_language="ru",
        sections=(section,),
    )

    epub_file = tmp_path / f"{book_id}.epub"
    epub_file.write_bytes(b"fake-epub")
    book = BookRecord(
        id=book_id,
        metadata=BookMetadata(title="Selection Novel"),
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
        narrative = NarrativeAnalysisResponse(
            translation_instructions="Сохранять спокойный ритм повествования.",
            chapter_summary="Characters move through multiple dramatic scenes across the castle.",
            glossary_candidates=(),
            context_facts=(),
            scenes=tuple(quoted_scenes),
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
            section,
            (),
            (),
            (),
            model_dependencies=_pass_model_dependencies(cond_payload),
            max_fragment_blocks=narrative_max_fragment_blocks_for(cond_payload["llm_parameters"]),
        )
        tx.analyses.save_pass(
            AnalysisPassResult(
                id="pass-sec-1",
                book_id=book_id,
                section_id=section_id,
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
                model_prompts="gpt-5-mini",
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
    selection = IllustrationSelection(
        uow, source=source, runtime=runtime, sleep_fn=_noop_sleep
    )

    with uow.transaction() as tx:
        head = tx.illustrations.get_head(book_id)
        snap = source.snapshot(tx, book_id)
        assert snap.ready and snap.manifest is not None and snap.manifest_sha256 is not None
        assert head.settings_id is not None
        gen, _ = tx.illustrations.create_or_get_generation(
            book_id,
            settings_id=head.settings_id,
            request_key=f"req-{book_id}",
            source_sha256=source_sha,
            source_authority=snap.source_authority or {},
            requested_sections=snap.content_section_ids,
        )
        gen = tx.illustrations.seal_manifest(
            book_id, gen.id, manifest=snap.manifest, manifest_sha256=snap.manifest_sha256
        )
        gen = tx.illustrations.seal_registry(book_id, gen.id, registry_sha256="c" * 64)
        gen = tx.illustrations.update_generation_state(
            book_id, gen.id, state=IllustrationGenerationState.QUEUED, reason_code=None
        )
        gen_id = gen.id

    selection.plan_scopes(gen_id, section_id)
    return _Harness(
        uow=uow,
        source=source,
        runtime=runtime,
        client=client,
        selection=selection,
        book_id=book_id,
        section_id=section_id,
        generation_id=gen_id,
    )


async def _drain_selection_tasks(harness: _Harness) -> int:
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
            await harness.selection.run_task(running)
            ran += 1
    return ran


def _make_card(
    cid: str,
    *,
    scene_id: str,
    block_id: str,
    start: int = 0,
    end: int = 10,
    title: str | None = None,
    event: str | None = None,
    event_kind: str = "action",
    significance: str | None = None,
    reason: str | None = None,
    order: int = 0,
) -> SelectedMomentCard:
    pos = format_position_key(0, order, start)
    return SelectedMomentCard(
        candidate_id=cid,
        section_id="sec-1",
        scene_id=scene_id,
        title_ru=title or f"Событие {cid}",
        event_ru=event or f"Описание события {cid}",
        significance_ru=significance or f"Значимость события {cid}",
        reason_ru=reason or f"Причина выбора {cid}",
        event_kind=event_kind,
        event_position_key=pos,
        anchors=(
            SelectedAnchor(
                block_id=block_id,
                start=start,
                end=end,
                quote="quote",
                position_key=pos,
            ),
        ),
        insert_after_block_id=block_id,
        participant_names=("Алиса",),
        location_names=("Зал",),
        scene_start_block_id=block_id,
        scene_end_block_id=block_id,
    )


async def test_candidates_from_all_scenes(tmp_path: Path) -> None:
    seen_scenes: list[str] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            seen_scenes.append(data["scene_id"])
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id=f"ev-{blk['block_id']}",
                        title_ru=f"Сцена {blk['block_id']}",
                        event_ru=f"Событие в блоке {blk['block_id']}",
                        significance_ru=f"Поворотный момент {blk['block_id']}",
                        event_kind="encounter",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(
                                block_id=blk["block_id"], quote=blk["text"][:15]
                            ),
                        ),
                        insert_after_block_id=blk["block_id"],
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cands = data["candidates"]
            chosen = [c["candidate_id"] for c in cands[: data["group_cap"]]]
            return IllustrationSelectionResponse(
                selected_candidate_ids=tuple(chosen),
                selection_reasons_ru=tuple(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Яркий момент")
                    for cid in chosen
                ),
            )
        raise AssertionError(req.kind)

    harness = _setup_chapter(
        tmp_path,
        scenes_spec=[
            ("Первая сцена у ворот", [("b-1", "Alice arrived at the iron gate at dusk.")]),
            ("Вторая сцена в зале", [("b-2", "A lantern flared across the stone hall.")]),
            ("Третья сцена на башне", [("b-3", "Lightning struck the tall north tower.")]),
        ],
        handler=_handler,
    )
    await _drain_selection_tasks(harness)

    assert len(seen_scenes) == 3
    selected = harness.selection.selected_moments(harness.generation_id, harness.section_id)
    assert len(selected.winners) == 3
    assert selected.zero_reason_ru is None


async def test_selection_runs_when_candidates_at_or_below_quota_and_can_return_zero(
    tmp_path: Path,
) -> None:
    selection_called = False

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        nonlocal selection_called
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-weak",
                        title_ru="Обычный разговор",
                        event_ru="Персонажи обсуждают погоду.",
                        significance_ru="Слабый бытовой эпизод",
                        event_kind="dialogue",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(block_id=blk["block_id"], quote="weather"),
                        ),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            selection_called = True
            return IllustrationSelectionResponse(
                selected_candidate_ids=(),
                selection_reasons_ru=(),
                rejected_reasons_ru=(
                    IllustrationSelectionReasonDTO(
                        candidate_id=data["candidates"][0]["candidate_id"],
                        reason_ru="Недостаточно визуального действия",
                    ),
                ),
                zero_reason_ru="В главе только бытовой разговор без яркого визуального образа.",
            )
        raise AssertionError(req.kind)

    harness = _setup_chapter(
        tmp_path,
        scenes_spec=[("Сцена с беседой", [("b-1", "They quietly talked about the weather.")])],
        handler=_handler,
        max_per_chapter=3,
    )
    await _drain_selection_tasks(harness)

    assert selection_called is True
    selected = harness.selection.selected_moments(harness.generation_id, harness.section_id)
    assert selected.winners == ()
    assert selected.zero_reason_ru == "В главе только бытовой разговор без яркого визуального образа."


async def test_tournament_bounded_and_resumable(tmp_path: Path) -> None:
    # 10 scenes -> 10 candidates -> fan-in 8 -> round 1 has group of 8 (cap 3) + group of 2 (cap 1) = 4 winners
    # -> round 2 has group of 4 (cap 3) -> 3 final winners!
    round_groups: list[tuple[int, int, int]] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id=f"ev-{blk['block_id']}",
                        title_ru=f"Момент {blk['block_id']}",
                        event_ru=f"Событие в {blk['block_id']}",
                        significance_ru=f"Важный шаг {blk['block_id']}",
                        event_kind="action",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(
                                block_id=blk["block_id"], quote=blk["text"][:14]
                            ),
                        ),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cands = data["candidates"]
            cap = data["group_cap"]
            round_groups.append((data["round"], len(cands), cap))
            chosen = [c["candidate_id"] for c in cands[:cap]]
            return IllustrationSelectionResponse(
                selected_candidate_ids=tuple(chosen),
                selection_reasons_ru=tuple(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Сильный эпизод")
                    for cid in chosen
                ),
            )
        raise AssertionError(req.kind)

    scenes = [
        (f"Сцена {i}", [(f"b-{i}", f"Dramatic scene {i} unfolded across the courtyard.")])
        for i in range(10)
    ]
    harness = _setup_chapter(tmp_path, scenes_spec=scenes, handler=_handler, max_per_chapter=3)
    await _drain_selection_tasks(harness)

    # Round 1: (8 -> cap 3) and (2 -> cap 1); Round 2: (4 -> cap 3)
    assert all(size <= 8 for _, size, _ in round_groups)
    assert any(r == 1 for r, _, _ in round_groups)
    assert any(r == 2 for r, _, _ in round_groups)
    selected = harness.selection.selected_moments(harness.generation_id, harness.section_id)
    assert len(selected.winners) == 3

    # Re-calling plan_scopes after completion performs 0 additional LLM calls (resumable)!
    calls_before = len(harness.client.requests)
    harness.selection.plan_scopes(harness.generation_id, harness.section_id)
    await _drain_selection_tasks(harness)
    assert len(harness.client.requests) == calls_before


async def test_tournament_byte_split_guarantees_strict_progress(tmp_path: Path) -> None:
    # Tight wire cap forces byte splitting of 4 candidates into [2] + [2];
    # with max_per_chapter=3, group_cap MUST be min(3, 2-1) = 1 -> strictly 4 -> 2!
    observed_caps: list[tuple[int, int]] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id=f"ev-{blk['block_id']}",
                        title_ru=f"Длинное название {blk['block_id']}",
                        event_ru=f"Подробное описание {blk['block_id']} " + ("текст " * 140),
                        significance_ru=f"Высокая значимость {blk['block_id']} " + ("важно " * 140),
                        event_kind="action",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(
                                block_id=blk["block_id"], quote=blk["text"][:14]
                            ),
                        ),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cands = data["candidates"]
            cap = data["group_cap"]
            observed_caps.append((len(cands), cap))
            assert cap <= len(cands) - 1
            chosen = [c["candidate_id"] for c in cands[:cap]]
            return IllustrationSelectionResponse(
                selected_candidate_ids=tuple(chosen),
                selection_reasons_ru=tuple(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Выбрано")
                    for cid in chosen
                ),
            )
        raise AssertionError(req.kind)

    scenes = [
        (f"Сцена {i}", [(f"b-{i}", f"Event {i} unfolded in the stone gallery.")])
        for i in range(4)
    ]
    harness = _setup_chapter(
        tmp_path,
        scenes_spec=scenes,
        handler=_handler,
        max_per_chapter=3,
        policy=IllustrationPolicy(max_request_bytes=12000, repair_reserve_bytes=1200),
    )
    await _drain_selection_tasks(harness)

    # Every group of size 2 got group_cap == 1, guaranteeing termination!
    assert any(size == 2 and cap == 1 for size, cap in observed_caps)
    selected = harness.selection.selected_moments(harness.generation_id, harness.section_id)
    assert 1 <= len(selected.winners) <= 3


async def test_tournament_singleton_and_all_zero_round_with_zero_reason(tmp_path: Path) -> None:
    # 9 scenes -> fan-in 8 -> group of 8 + 1 carried singleton.
    # Model returns 0 for the group of 8 -> carried singleton survives -> final result has 1 winner!
    def _handler_singleton(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id=f"ev-{blk['block_id']}",
                        title_ru=f"Сцена {blk['block_id']}",
                        event_ru=f"Действие {blk['block_id']}",
                        significance_ru=f"Важность {blk['block_id']}",
                        event_kind="action",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(
                                block_id=blk["block_id"], quote=blk["text"][:12]
                            ),
                        ),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            return IllustrationSelectionResponse(
                selected_candidate_ids=(),
                selection_reasons_ru=(),
                zero_reason_ru="Группа не содержит сильных сцен.",
            )
        raise AssertionError(req.kind)

    scenes9 = [
        (f"Сцена {i}", [(f"b-{i}", f"Scene {i} occurred near the river.")])
        for i in range(9)
    ]
    h1 = _setup_chapter(
        tmp_path, scenes_spec=scenes9, handler=_handler_singleton, book_id="book-single"
    )
    await _drain_selection_tasks(h1)
    sel1 = h1.selection.selected_moments(h1.generation_id, h1.section_id)
    assert len(sel1.winners) == 1

    # 4 scenes -> group of 4 -> model returns 0 -> all-zero round with consolidated zero_reason_ru!
    scenes4 = [
        (f"Сцена {i}", [(f"b-{i}", f"Minor moment {i} in the corridor.")])
        for i in range(4)
    ]
    h2 = _setup_chapter(
        tmp_path, scenes_spec=scenes4, handler=_handler_singleton, book_id="book-allzero"
    )
    await _drain_selection_tasks(h2)
    sel2 = h2.selection.selected_moments(h2.generation_id, h2.section_id)
    assert sel2.winners == ()
    assert sel2.zero_reason_ru == "Группа не содержит сильных сцен."


async def test_shortlist_lifo_by_round_and_excludes_winners_and_duplicates(tmp_path: Path) -> None:
    # 10 scenes -> Round 1 eliminates 6 losers; Round 2 (4 -> 3) eliminates 1 loser.
    # Round 2's loser MUST appear BEFORE Round 1's losers in shortlist (LIFO)!
    round2_loser_id: list[str] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id=f"ev-{blk['block_id']}",
                        title_ru=f"Событие {blk['block_id']}",
                        event_ru=f"Описание {blk['block_id']}",
                        significance_ru=f"Значимость {blk['block_id']}",
                        event_kind="discovery",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(
                                block_id=blk["block_id"], quote=blk["text"][:12]
                            ),
                        ),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            cands = data["candidates"]
            cap = data["group_cap"]
            chosen = [c["candidate_id"] for c in cands[:cap]]
            if data["round"] == 2:
                round2_loser_id.extend(c["candidate_id"] for c in cands[cap:])
            return IllustrationSelectionResponse(
                selected_candidate_ids=tuple(chosen),
                selection_reasons_ru=tuple(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Хороший кадр")
                    for cid in chosen
                ),
            )
        raise AssertionError(req.kind)

    scenes = [
        (f"Сцена {i}", [(f"b-{i}", f"Scene {i} revealed a secret map.")])
        for i in range(10)
    ]
    harness = _setup_chapter(tmp_path, scenes_spec=scenes, handler=_handler, max_per_chapter=3)
    await _drain_selection_tasks(harness)

    selected = harness.selection.selected_moments(harness.generation_id, harness.section_id)
    assert len(round2_loser_id) == 1
    assert selected.shortlist[0].candidate_id == round2_loser_id[0]
    winner_ids = {w.candidate_id for w in selected.winners}
    assert all(s.candidate_id not in winner_ids for s in selected.shortlist)


def test_duplicate_event_rejected() -> None:
    c1 = _make_card("c1", scene_id="s1", block_id="b1", start=0, end=12, title="Встреча у ворот")
    c2 = _make_card("c2", scene_id="s1", block_id="b1", start=2, end=10, title="Встреча у ворот")
    kept, rejected = deduplicate_candidates([c1, c2])
    assert [c.candidate_id for c in kept] == ["c1"]
    assert len(rejected) == 1 and rejected[0][0] == "c2"


async def test_hallucinated_and_duplicate_candidate_ids_rejected(tmp_path: Path) -> None:
    selection_attempts = 0

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        nonlocal selection_attempts
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-1",
                        title_ru=f"Событие {blk['block_id']}",
                        event_ru=f"Эпизод {blk['block_id']}",
                        significance_ru="Ключевой эпизод",
                        event_kind="encounter",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(
                                block_id=blk["block_id"], quote=blk["text"][:10]
                            ),
                        ),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            selection_attempts += 1
            cid = data["candidates"][0]["candidate_id"]
            if selection_attempts == 1:
                # Hallucinated ID -> rejected, triggers repair!
                return IllustrationSelectionResponse(
                    selected_candidate_ids=("cand-hallucinated",),
                    selection_reasons_ru=(
                        IllustrationSelectionReasonDTO(
                            candidate_id="cand-hallucinated", reason_ru="Выдуманный"
                        ),
                    ),
                )
            # Duplicate ID -> rejected, exhausts max_repair_turns -> fails model_output_invalid!
            return IllustrationSelectionResponse(
                selected_candidate_ids=(cid, cid),
                selection_reasons_ru=(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Повтор"),
                ),
            )
        raise AssertionError(req.kind)

    harness = _setup_chapter(
        tmp_path,
        scenes_spec=[
            ("Сцена 1", [("b-1", "Alice opened the heavy door.")]),
            ("Сцена 2", [("b-2", "Bob lit a candle on the table.")]),
        ],
        handler=_handler,
        max_per_chapter=3,
        policy=IllustrationPolicy(max_repair_turns=1),
    )
    await _drain_selection_tasks(harness)

    assert selection_attempts == 2
    with harness.uow.transaction() as tx:
        failed = tx.illustrations.list_scopes(
            harness.generation_id,
            stage=IllustrationScopeStage.SELECTION,
            state=IllustrationScopeState.FAILED,
        )
        assert len(failed) == 1
        assert failed[0].reason_code == IllustrationReasonCode.MODEL_OUTPUT_INVALID.value


async def test_zero_has_reason(tmp_path: Path) -> None:
    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        assert req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES
        blk = data["blocks"][0]
        return IllustrationCandidatesResponse(
            scene_id=data["scene_id"],
            events=(),
            zero_reason_ru="Сцена содержит только краткое вступительное размышление.",
            covered_source_unit_ids=(blk["source_unit_id"],),
            complete=True,
        )

    harness = _setup_chapter(
        tmp_path,
        scenes_spec=[("Вступление", [("b-1", "It was a quiet evening of reflection.")])],
        handler=_handler,
    )
    await _drain_selection_tasks(harness)

    selected = harness.selection.selected_moments(harness.generation_id, harness.section_id)
    assert selected.winners == ()
    assert selected.zero_reason_ru == "Сцена содержит только краткое вступительное размышление."


async def test_selection_below_quota(tmp_path: Path) -> None:
    # 3 candidates with quota 3 -> model legitimately picks only 1!
    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id=f"ev-{blk['block_id']}",
                        title_ru=f"Сцена {blk['block_id']}",
                        event_ru=f"Описание {blk['block_id']}",
                        significance_ru=f"Значимость {blk['block_id']}",
                        event_kind="action",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(
                                block_id=blk["block_id"], quote=blk["text"][:10]
                            ),
                        ),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            first_id = data["candidates"][0]["candidate_id"]
            return IllustrationSelectionResponse(
                selected_candidate_ids=(first_id,),
                selection_reasons_ru=(
                    IllustrationSelectionReasonDTO(
                        candidate_id=first_id, reason_ru="Единственный по-настоящему сильный кадр"
                    ),
                ),
            )
        raise AssertionError(req.kind)

    harness = _setup_chapter(
        tmp_path,
        scenes_spec=[
            ("Сцена 1", [("b-1", "The bridge collapsed into the gorge.")]),
            ("Сцена 2", [("b-2", "They walked along the road.")]),
            ("Сцена 3", [("b-3", "They rested beside a tree.")]),
        ],
        handler=_handler,
        max_per_chapter=3,
    )
    await _drain_selection_tasks(harness)

    selected = harness.selection.selected_moments(harness.generation_id, harness.section_id)
    assert len(selected.winners) == 1
    assert selected.winners[0].reason_ru == "Единственный по-настоящему сильный кадр"
    assert len(selected.shortlist) == 2


def test_same_action_does_not_displace_distinct_conflict() -> None:
    w1 = _make_card(
        "c1",
        scene_id="scene-1",
        block_id="b-1",
        title="Схватка на мосту",
        event="Герой отражает удар стражника на узком мосту.",
        event_kind="action",
        order=0,
    )
    w2 = _make_card(
        "c2",
        scene_id="scene-2",
        block_id="b-5",
        title="Побег через крышу",
        event="Героиня перепрыгивает провал между башнями под дождём.",
        event_kind="action",
        order=4,
    )
    assert moments_conflict_diversity(w1, w2) is False
    winners, _, zero_reason = enforce_intra_chapter_diversity([w1, w2], [])
    assert [w.candidate_id for w in winners] == ["c1", "c2"]
    assert zero_reason is None


def test_diversity_replacement_shortlist_exhausted_keeps_grounded_reason_ru() -> None:
    # 1. Replacement from LIFO shortlist uses promoted runner-up's significance_ru as reason_ru.
    w1 = _make_card(
        "c1", scene_id="s1", block_id="b1", start=0, end=10, event_kind="duel", reason="Первый удар"
    )
    w_dup = _make_card(
        "c2", scene_id="s1", block_id="b1", start=5, end=15, event_kind="duel", reason="Повтор"
    )
    runner = _make_card(
        "c3",
        scene_id="s2",
        block_id="b2",
        event_kind="discovery",
        significance="Герой находит скрытый ключ в библиотеке",
        reason="старая причина",
        order=1,
    )
    winners, shortlist, _ = enforce_intra_chapter_diversity([w1, w_dup], [runner])
    assert [w.candidate_id for w in winners] == ["c1", "c3"]
    assert winners[1].reason_ru == "Герой находит скрытый ключ в библиотеке"
    # Final winners and rejected duplicate stripped from shortlist.
    assert shortlist == ()

    # 2. When shortlist is exhausted (or only has conflicting items), duplicate is dropped and w1 keeps grounded reason_ru.
    winners2, shortlist2, _ = enforce_intra_chapter_diversity([w1, w_dup], [])
    assert [w.candidate_id for w in winners2] == ["c1"]
    assert winners2[0].reason_ru == "Первый удар"
    assert shortlist2 == ()


async def test_huge_scene_splits_by_blocks_and_atomic_block_fails(tmp_path: Path) -> None:
    # 1. Huge scene with multiple blocks splits into bounded chunks preserving scene_id & order!
    seen_chunks: list[tuple[str, list[str]]] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            seen_chunks.append((data["scene_id"], [b["block_id"] for b in data["blocks"]]))
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(),
                zero_reason_ru="Описательная часть без отдельного события.",
                covered_source_unit_ids=tuple(b["source_unit_id"] for b in data["blocks"]),
                complete=True,
            )
        raise AssertionError(req.kind)

    big_blocks = [
        (f"b-{i}", f"Block {i}: " + ("dramatic candlelight over ancient books " * 140))
        for i in range(6)
    ]
    h1 = _setup_chapter(
        tmp_path,
        scenes_spec=[("Огромная сцена в библиотеке", big_blocks)],
        handler=_handler,
        book_id="book-huge-scene",
    )
    await _drain_selection_tasks(h1)
    assert len(seen_chunks) >= 2
    assert len({sid for sid, _ in seen_chunks}) == 1
    flattened_ids = [bid for _, bids in seen_chunks for bid in bids]
    assert flattened_ids == [f"b-{i}" for i in range(6)]

    # 2. Single atomic huge block fails atomic_input_too_large before HTTP!
    huge_single = "Alice stood in the hall " + ("holding a glowing lantern " * 1400)
    h2 = _setup_chapter(
        tmp_path,
        scenes_spec=[("Огромный абзац", [("b-huge", huge_single)])],
        handler=lambda _req, _n: (_ for _ in ()).throw(AssertionError("must not call HTTP")),
        book_id="book-huge-atom",
    )
    await _drain_selection_tasks(h2)
    assert len(h2.client.requests) == 0
    with h2.uow.transaction() as tx:
        failed = tx.illustrations.list_scopes(
            h2.generation_id,
            stage=IllustrationScopeStage.CANDIDATES,
            state=IllustrationScopeState.FAILED,
        )
        assert len(failed) == 1
        assert failed[0].reason_code == IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value


async def test_candidates_invalid_insert_after_block_id_triggers_repair_and_fails(
    tmp_path: Path,
) -> None:
    attempts = 0

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        nonlocal attempts
        attempts += 1
        data = json.loads(req.input_text)
        assert req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES
        blk0, blk1 = data["blocks"][0], data["blocks"][1]
        # Anchor is on blk1 (later block), but insert_after_block_id points to blk0 (earlier) or foreign block!
        bad_insert = blk0["block_id"] if attempts == 1 else "foreign-blk-999"
        return IllustrationCandidatesResponse(
            scene_id=data["scene_id"],
            events=(
                IllustrationCandidateEventDTO(
                    event_id="ev-bad-ins",
                    title_ru="Плохая вставка",
                    event_ru="Событие на втором абзаце.",
                    significance_ru="Ключевой момент",
                    event_kind="action",
                    quote_anchors=(
                        IllustrationQuoteAnchorDTO(
                            block_id=blk1["block_id"], quote=blk1["text"][:10]
                        ),
                    ),
                    insert_after_block_id=bad_insert,
                ),
            ),
            covered_source_unit_ids=tuple(b["source_unit_id"] for b in data["blocks"]),
            complete=True,
        )

    harness = _setup_chapter(
        tmp_path,
        scenes_spec=[
            (
                "Сцена 1",
                [
                    ("b-0", "First paragraph sets the scene."),
                    ("b-1", "Second paragraph contains the climax."),
                ],
            )
        ],
        handler=_handler,
        policy=IllustrationPolicy(max_repair_turns=1),
    )
    await _drain_selection_tasks(harness)
    assert attempts == 2
    with harness.uow.transaction() as tx:
        failed = tx.illustrations.list_scopes(
            harness.generation_id,
            stage=IllustrationScopeStage.CANDIDATES,
            state=IllustrationScopeState.FAILED,
        )
        assert len(failed) == 1
        assert failed[0].reason_code == IllustrationReasonCode.MODEL_OUTPUT_INVALID.value


async def test_selection_default_policy_allows_two_repair_turns(tmp_path: Path) -> None:
    selection_attempts = 0

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        nonlocal selection_attempts
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-1",
                        title_ru=f"Событие {blk['block_id']}",
                        event_ru=f"Эпизод {blk['block_id']}",
                        significance_ru="Ключевой эпизод",
                        event_kind="encounter",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(
                                block_id=blk["block_id"], quote=blk["text"][:10]
                            ),
                        ),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            selection_attempts += 1
            cid = data["candidates"][0]["candidate_id"]
            if selection_attempts == 1:
                # Missing selection_reasons_ru -> triggers 1st repair turn
                return IllustrationSelectionResponse(
                    selected_candidate_ids=(cid,),
                    selection_reasons_ru=(),
                )
            if selection_attempts == 2:
                # Hallucinated ID -> triggers 2nd repair turn (within default max_repair_turns=2)
                return IllustrationSelectionResponse(
                    selected_candidate_ids=("cand-hallucinated",),
                    selection_reasons_ru=(
                        IllustrationSelectionReasonDTO(
                            candidate_id="cand-hallucinated", reason_ru="Выдуманный"
                        ),
                    ),
                )
            # 3rd attempt (2nd repair) succeeds!
            return IllustrationSelectionResponse(
                selected_candidate_ids=(cid,),
                selection_reasons_ru=(
                    IllustrationSelectionReasonDTO(
                        candidate_id=cid, reason_ru="Точный выбор после двух ремонтов"
                    ),
                ),
            )
        raise AssertionError(req.kind)

    harness = _setup_chapter(
        tmp_path,
        scenes_spec=[
            ("Сцена 1", [("b-1", "Alice opened the heavy door.")]),
            ("Сцена 2", [("b-2", "Bob lit a candle on the table.")]),
        ],
        handler=_handler,
        max_per_chapter=3,
    )
    await _drain_selection_tasks(harness)

    assert selection_attempts == 3
    selected = harness.selection.selected_moments(harness.generation_id, harness.section_id)
    assert len(selected.winners) == 1
    assert selected.winners[0].reason_ru == "Точный выбор после двух ремонтов"


async def test_candidates_receives_spoiler_free_known_entities_and_preserves_on_split(
    tmp_path: Path,
) -> None:
    from linguary.domain.illustrations import (
        TIMELESS_POSITION_KEY,
        IllustrationRegistryKind,
        RegistryItem,
    )

    seen_inputs: list[dict[str, Any]] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            seen_inputs.append(data)
            if len(data["blocks"]) > 1:
                # Trigger composite split to verify children inherit known_characters / known_locations
                return IllustrationCandidatesResponse(
                    scene_id=data["scene_id"],
                    events=(),
                    covered_source_unit_ids=(),
                    complete=False,
                )
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(),
                zero_reason_ru="Тихая сцена.",
                covered_source_unit_ids=tuple(b["source_unit_id"] for b in data["blocks"]),
                complete=True,
            )
        raise AssertionError(req.kind)

    harness = _setup_chapter(
        tmp_path,
        scenes_spec=[
            (
                "Сцена в доме",
                [
                    ("b-0", "A tabby cat sat on the garden wall."),
                    ("b-1", "Mr Dursley looked out onto Privet Drive."),
                ],
            )
        ],
        handler=_handler,
        book_id="book-known-ent",
    )

    # Clear initial empty-registry scopes and seed ENTITY items spanning ch0 + future ch5, then replan!
    def _ent(
        gen_id: str,
        eid: str,
        kind: str,
        canonical: str,
        first_pos: str,
        section_orders: list[int],
        alias_first_positions: dict[str, str] | None = None,
    ) -> RegistryItem:
        pay: dict[str, Any] = {
            "entity_id": eid,
            "entity_kind": kind,
            "canonical_name": canonical,
            "aliases": [],
            "section_orders": section_orders,
            "mention_count": 1,
            "first_position_key": first_pos,
        }
        if alias_first_positions is not None:
            pay["alias_first_positions"] = alias_first_positions
        return RegistryItem(
            generation_id=gen_id,
            kind=IllustrationRegistryKind.ENTITY,
            item_key=eid,
            entity_id=eid,
            position_key=TIMELESS_POSITION_KEY,
            is_back_applicable=False,
            payload_sha256=canonical_sha(pay),
            payload=pay,
        )

    with harness.uow.transaction() as tx:
        gen = tx.illustrations.get_generation(harness.book_id, harness.generation_id)
        assert gen is not None
        snap = harness.source.snapshot(tx, harness.book_id)
        head = tx.illustrations.get_head(harness.book_id)
        assert head.settings_id is not None
        tx.illustrations.revoke_generation(harness.book_id, gen.id, reason_code="settings_changed")
        gen2, _ = tx.illustrations.create_or_get_generation(
            harness.book_id,
            settings_id=head.settings_id,
            request_key="req-known-ent-2",
            source_sha256=gen.source_sha256,
            source_authority=gen.source_authority,
            requested_sections=(harness.section_id,),
        )
        assert snap.manifest is not None and snap.manifest_sha256 is not None
        gen2 = tx.illustrations.seal_manifest(
            harness.book_id, gen2.id, manifest=snap.manifest, manifest_sha256=snap.manifest_sha256
        )
        tx.illustrations.upsert_registry_items(
            gen2.id,
            [
                _ent(
                    gen2.id,
                    "ent-cat",
                    "person",
                    "Professor McGonagall",
                    format_position_key(0, 0, 0),
                    [0, 5],
                    alias_first_positions={
                        "tabby cat": format_position_key(0, 0, 0),
                        "Professor McGonagall": format_position_key(5, 0, 0),
                    },
                ),
                _ent(
                    gen2.id,
                    "ent-vernon",
                    "person",
                    "Mr Dursley",
                    format_position_key(0, 1, 0),
                    [0],
                ),
                _ent(
                    gen2.id,
                    "ent-future-hero",
                    "person",
                    "Future Villain",
                    format_position_key(5, 0, 0),
                    [5],
                ),
                _ent(
                    gen2.id,
                    "ent-street",
                    "location",
                    "Privet Drive",
                    format_position_key(0, 1, 0),
                    [0],
                ),
                _ent(
                    gen2.id,
                    "ent-future-castle",
                    "location",
                    "Future Fortress",
                    format_position_key(5, 0, 0),
                    [5],
                ),
            ],
        )
        gen2 = tx.illustrations.seal_registry(harness.book_id, gen2.id, registry_sha256="e" * 64)
        gen2 = tx.illustrations.update_generation_state(
            harness.book_id, gen2.id, state=IllustrationGenerationState.QUEUED, reason_code=None
        )
        new_gen_id = gen2.id

    harness.selection.plan_scopes(new_gen_id, harness.section_id)
    # Run tasks for gen2 (gen1 was revoked so only gen2 tasks advance)
    await _drain_selection_tasks(harness)

    # 1 initial composite call + 2 split child calls
    assert len(seen_inputs) == 3
    for payload in seen_inputs:
        assert "tabby cat" in payload["known_characters"]
        assert "Mr Dursley" in payload["known_characters"]
        assert "Professor McGonagall" not in payload["known_characters"]
        assert "Future Villain" not in payload["known_characters"]
        assert payload["known_locations"] == ["Privet Drive"]
        assert "Future Fortress" not in payload["known_locations"]


async def test_candidates_known_entity_aliases_and_glossary_terms_ru_non_spoiler_and_split(
    tmp_path: Path,
) -> None:
    from linguary.application.illustration_selection import (
        _known_entities_context_for_chapter,
        _known_entity_names_for_chapter,
    )
    from linguary.domain.glossary import ConceptType, TermConcept, TermStatus, TermVariant
    from linguary.domain.illustrations import (
        TIMELESS_POSITION_KEY,
        IllustrationRegistryKind,
        RegistryItem,
    )

    seen_inputs: list[dict[str, Any]] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            seen_inputs.append(data)
            if len(data["blocks"]) > 1:
                return IllustrationCandidatesResponse(
                    scene_id=data["scene_id"],
                    events=(),
                    covered_source_unit_ids=(),
                    complete=False,
                )
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(),
                zero_reason_ru="Тихая сцена.",
                covered_source_unit_ids=tuple(b["source_unit_id"] for b in data["blocks"]),
                complete=True,
            )
        raise AssertionError(req.kind)

    harness = _setup_chapter(
        tmp_path,
        scenes_spec=[
            (
                "Сцена в лесу",
                [
                    ("b-0", "Nick looked at the boy in the quiet forest."),
                    ("b-1", "Chocolate Boy stepped onto the mossy path."),
                ],
            )
        ],
        handler=_handler,
        book_id="book-aliases-gloss",
    )

    with harness.uow.transaction() as tx:
        # Seed glossary concepts: Nick (backed by ENTITY in ch0) and Future Relic (not in ch0 -> MUST NOT spoil!).
        tx.glossary.save(
            TermConcept(
                id="c-nick",
                book_id=harness.book_id,
                type=ConceptType.PERSON,
                canonical_source_text="Nick",
                status=TermStatus.APPROVED,
                variants=(
                    TermVariant(
                        concept_id="c-nick",
                        language="ru",
                        text="Ник",
                        status=TermStatus.APPROVED,
                    ),
                ),
            )
        )
        tx.glossary.save(
            TermConcept(
                id="c-future",
                book_id=harness.book_id,
                type=ConceptType.FICTIONAL_CONCEPT,
                canonical_source_text="Future Relic",
                status=TermStatus.APPROVED,
                variants=(
                    TermVariant(
                        concept_id="c-future",
                        language="ru",
                        text="Будущая реликвия",
                        status=TermStatus.APPROVED,
                    ),
                ),
            )
        )

        gen = tx.illustrations.get_generation(harness.book_id, harness.generation_id)
        assert gen is not None
        snap = harness.source.snapshot(tx, harness.book_id)
        head = tx.illustrations.get_head(harness.book_id)
        assert head.settings_id is not None
        tx.illustrations.revoke_generation(harness.book_id, gen.id, reason_code="settings_changed")
        gen2, _ = tx.illustrations.create_or_get_generation(
            harness.book_id,
            settings_id=head.settings_id,
            request_key="req-aliases-gloss-2",
            source_sha256=gen.source_sha256,
            source_authority=gen.source_authority,
            requested_sections=(harness.section_id,),
        )
        assert snap.manifest is not None and snap.manifest_sha256 is not None
        gen2 = tx.illustrations.seal_manifest(
            harness.book_id, gen2.id, manifest=snap.manifest, manifest_sha256=snap.manifest_sha256
        )

        nick_pay = {
            "entity_id": "ent-nick",
            "entity_kind": "person",
            "canonical_name": "Nick",
            "aliases": ["the boy", "Chocolate Boy", "Future King"],
            "alias_first_positions": {
                "the boy": format_position_key(0, 0, 0),
                "Nick": format_position_key(0, 0, 5),
                "Chocolate Boy": format_position_key(0, 1, 0),
                "Future King": format_position_key(5, 0, 0),
            },
            "section_orders": [0, 5],
            "mention_count": 4,
            "first_position_key": format_position_key(0, 0, 0),
        }
        # Legacy ENTITY without alias_first_positions / glossary_translation_ru must work without KeyError.
        legacy_loc_pay = {
            "entity_id": "ent-forest",
            "entity_kind": "location",
            "canonical_name": "Whispering Forest",
            "aliases": ["dark forest"],
            "section_orders": [0],
            "mention_count": 1,
            "first_position_key": format_position_key(0, 0, 10),
        }
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
                    item_key="ent-forest",
                    entity_id="ent-forest",
                    position_key=TIMELESS_POSITION_KEY,
                    is_back_applicable=False,
                    payload_sha256=canonical_sha(legacy_loc_pay),
                    payload=legacy_loc_pay,
                ),
            ],
        )
        gen2 = tx.illustrations.seal_registry(harness.book_id, gen2.id, registry_sha256="f" * 64)
        gen2 = tx.illustrations.update_generation_state(
            harness.book_id, gen2.id, state=IllustrationGenerationState.QUEUED, reason_code=None
        )
        new_gen_id = gen2.id

        # Backward-compatible 2-tuple signature stays clean!
        kc2, kl2 = _known_entity_names_for_chapter(tx, new_gen_id, section_order=0)
        assert kc2 == ("Nick",)
        assert kl2 == ("Whispering Forest",)

        # Extended 3-tuple returns clean names + separate non-spoiler aliases.
        kc3, kl3, alias_map = _known_entities_context_for_chapter(tx, new_gen_id, section_order=0)
        assert kc3 == ("Nick",)
        assert kl3 == ("Whispering Forest",)
        assert alias_map == {"Nick": ["the boy", "Chocolate Boy"]}
        assert "Future King" not in alias_map["Nick"]

    harness.selection.plan_scopes(new_gen_id, harness.section_id)
    await _drain_selection_tasks(harness)

    # 1 initial composite call + 2 split child calls all preserve clean lists + aliases + glossary_terms_ru!
    assert len(seen_inputs) == 3
    for payload in seen_inputs:
        assert payload["known_characters"] == ["Nick"]
        assert payload["known_locations"] == ["Whispering Forest"]
        assert payload["known_entity_aliases"] == {"Nick": ["the boy", "Chocolate Boy"]}
        assert payload["glossary_terms_ru"] == {"Nick": "Ник"}
        assert "Future Relic" not in payload["glossary_terms_ru"]


async def test_selection_non_composite_truncated_retry_recovers(tmp_path: Path) -> None:
    from linguary.application.ports.llm import LLMResponseFormatError

    sel_calls = 0
    seen_sel_repairs: list[tuple[tuple[str | None, str], ...]] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        nonlocal sel_calls
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            blk = data["blocks"][0]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id=f"ev-{blk['block_id']}",
                        title_ru=f"Сцена {blk['block_id']}",
                        event_ru=f"Событие {blk['block_id']}",
                        significance_ru="Важный момент",
                        event_kind="encounter",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(block_id=blk["block_id"], quote=blk["text"][:12]),
                        ),
                    ),
                ),
                covered_source_unit_ids=(blk["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            sel_calls += 1
            seen_sel_repairs.append(req.repair_turns)
            if sel_calls == 1:
                raise LLMResponseFormatError("truncated selection", truncated=True)
            cid = data["candidates"][0]["candidate_id"]
            return IllustrationSelectionResponse(
                selected_candidate_ids=(cid,),
                selection_reasons_ru=(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Ключевой кадр"),
                ),
            )
        raise AssertionError(req.kind)

    # 2 candidates -> 1 non-composite SELECTION scope (is_composite is False when len(cards) <= 2).
    h = _setup_chapter(
        tmp_path,
        scenes_spec=[
            ("Сцена 1", [("b-1", "Alice stepped across the bridge.")]),
            ("Сцена 2", [("b-2", "A lantern glowed in the tower.")]),
        ],
        handler=_handler,
        max_per_chapter=2,
        book_id="book-sel-trunc-retry",
    )
    await _drain_selection_tasks(h)

    assert sel_calls == 2
    assert seen_sel_repairs[0] == ()
    assert seen_sel_repairs[1] and "обрезан по длине" in seen_sel_repairs[1][0][1]
    selected = h.selection.selected_moments(h.generation_id, h.section_id)
    assert len(selected.winners) == 1


async def test_wire_id_aliases_candidates_long_ids_decode_and_repair_rewrite(
    tmp_path: Path,
) -> None:
    from linguary.application.illustration_context import (
        decode_wire_id_aliases,
        encode_wire_id_aliases,
    )

    uuid_b1 = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    cand_calls = 0
    seen_wire_cand: list[dict[str, Any]] = []
    seen_repairs: list[str] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        nonlocal cand_calls
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            cand_calls += 1
            seen_wire_cand.append(data)
            if req.repair_turns:
                seen_repairs.append(req.repair_turns[0][1])
            blk = data["blocks"][0]
            if cand_calls == 1:
                return IllustrationCandidatesResponse(
                    scene_id=data["scene_id"],
                    events=(),
                    zero_reason_ru="Нет событий.",
                    covered_source_unit_ids=(),  # triggers repair naming source_unit_id
                    complete=True,
                )
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-1",
                        title_ru="Момент у моста",
                        event_ru="Алиса пересекает мост.",
                        significance_ru="Ключевой переход",
                        event_kind="crossing",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(
                                block_id=blk["block_id"], quote="Alice crossed the narrow bridge"
                            ),
                        ),
                        insert_after_block_id=blk["block_id"],
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
        raise AssertionError(req.kind)

    h = _setup_chapter(
        tmp_path,
        scenes_spec=[("Сцена моста", [(uuid_b1, "Alice crossed the narrow bridge at dawn.")])],
        handler=_handler,
        book_id="book-wire-cand",
    )
    await _drain_selection_tasks(h)

    assert cand_calls == 2
    assert seen_wire_cand[0]["scene_id"] == "s1"
    assert seen_wire_cand[0]["blocks"][0]["block_id"] == "b1"
    assert seen_wire_cand[0]["blocks"][0]["source_unit_id"] == "b1"
    assert seen_repairs and "b1" in seen_repairs[0]
    assert uuid_b1 not in seen_repairs[0]

    selected = h.selection.selected_moments(h.generation_id, h.section_id)
    assert len(selected.winners) == 1
    assert selected.winners[0].anchors[0].block_id == uuid_b1
    assert selected.winners[0].insert_after_block_id == uuid_b1

    # Direct check: single_scene_id fallback maps "s1" even if canonical scene_id was short.
    short_in = {"scene_id": "sc-short", "blocks": [{"source_unit_id": "u-0", "block_id": "b-0"}]}
    wire_short, a2c_short, _ = encode_wire_id_aliases(IllustrationScopeStage.CANDIDATES, short_in)
    assert wire_short["scene_id"] == "sc-short"
    assert wire_short["blocks"][0]["block_id"] == "b-0"
    assert a2c_short == {}
    decoded_s1 = decode_wire_id_aliases(
        IllustrationCandidatesResponse(scene_id="s1", events=(), covered_source_unit_ids=("u-0",)),
        a2c_short,
        single_scene_id="sc-short",
    )
    assert decoded_s1.scene_id == "sc-short"


async def test_effective_chapter_illustration_cap_and_split_preservation(tmp_path: Path) -> None:
    from linguary.application.illustration_selection import effective_chapter_illustration_cap

    assert effective_chapter_illustration_cap(3, 1, 1) == 1
    assert effective_chapter_illustration_cap(3, 2, 1) == 1
    assert effective_chapter_illustration_cap(3, 3, 1) == 2
    assert effective_chapter_illustration_cap(3, 8, 2) == 2
    assert effective_chapter_illustration_cap(3, 3, 3) == 3
    assert effective_chapter_illustration_cap(3, 9, 1) == 3
    assert effective_chapter_illustration_cap(1, 12, 4) == 1

    # 2-block, 1-scene chapter with max_per_chapter=3 MUST cap selection to 1!
    seen_group_caps: list[int] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            b0, b1 = data["blocks"][0], data["blocks"][1]
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(
                    IllustrationCandidateEventDTO(
                        event_id="ev-a",
                        title_ru="Момент 1",
                        event_ru="Первое событие.",
                        significance_ru="Значимость 1",
                        event_kind="entry",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(block_id=b0["block_id"], quote="First paragraph"),
                        ),
                    ),
                    IllustrationCandidateEventDTO(
                        event_id="ev-b",
                        title_ru="Момент 2",
                        event_ru="Второе событие.",
                        significance_ru="Значимость 2",
                        event_kind="exit",
                        quote_anchors=(
                            IllustrationQuoteAnchorDTO(block_id=b1["block_id"], quote="Second paragraph"),
                        ),
                    ),
                ),
                covered_source_unit_ids=tuple(b["source_unit_id"] for b in data["blocks"]),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            seen_group_caps.append(int(data["group_cap"]))
            chosen = [c["candidate_id"] for c in data["candidates"][: data["group_cap"]]]
            return IllustrationSelectionResponse(
                selected_candidate_ids=tuple(chosen),
                selection_reasons_ru=tuple(
                    IllustrationSelectionReasonDTO(candidate_id=cid, reason_ru="Лучший кадр")
                    for cid in chosen
                ),
            )
        raise AssertionError(req.kind)

    h = _setup_chapter(
        tmp_path,
        scenes_spec=[
            (
                "Короткая глава",
                [
                    ("b-0", "First paragraph in short chapter."),
                    ("b-1", "Second paragraph in short chapter."),
                ],
            )
        ],
        handler=_handler,
        max_per_chapter=3,
        book_id="book-short-cap",
    )
    await _drain_selection_tasks(h)

    assert seen_group_caps == [1]
    selected = h.selection.selected_moments(h.generation_id, h.section_id)
    assert len(selected.winners) == 1
    with h.uow.read_transaction() as tx:
        sel_scopes = [
            s
            for s in tx.illustrations.list_scopes(h.generation_id, stage=IllustrationScopeStage.SELECTION)
            if s.summary.get("kind") == "tournament_group"
        ]
        assert sel_scopes
        assert sel_scopes[0].summary["effective_max_per_chapter"] == 1

        # Verify _split_children_for_scope preserves effective_max_per_chapter from scope.summary!
        from linguary.application.illustration_selection import _build_selection_scope_spec

        gen = tx.illustrations.get_generation(h.book_id, h.generation_id)
        assert gen is not None
        rev = tx.illustrations.get_settings(h.book_id, gen.settings_id)
        assert rev is not None
        cards4 = [
            _make_card(f"c{i}", scene_id=f"s{i}", block_id=f"b-{i}", start=0, end=5, title=f"T{i}")
            for i in range(4)
        ]
        parent_spec = _build_selection_scope_spec(
            section_id=h.section_id,
            section_order=0,
            section_title="Short",
            scope_key="select:test:split:parent",
            round_num=1,
            group_idx=0,
            cards=cards4,
            group_cap=1,
            depth=0,
            revision=rev,
            route_id="route-1",
            chapter_cap=1,
        )
        parent_scope = tx.illustrations.upsert_scopes(h.generation_id, [parent_spec])[0]
        children = h.selection._split_children_for_scope(parent_scope, rev, depth=0)
        assert children is not None and len(children) == 2
        for child in children:
            assert child.summary["effective_max_per_chapter"] == 1
            assert child.summary["group_cap"] == 1


def test_moments_conflict_diversity_min_block_gap() -> None:
    order = {f"b-{i}": i for i in range(15)}
    c_b2 = _make_card("c-b2", scene_id="s1", block_id="b-2", start=0, end=5, title="Событие A", event_kind="a")
    c_b3 = _make_card("c-b3", scene_id="s2", block_id="b-3", start=0, end=5, title="Событие B", event_kind="b")
    c_b5 = _make_card("c-b5", scene_id="s3", block_id="b-5", start=0, end=5, title="Событие C", event_kind="c")

    # Default min_block_gap=0 -> no conflict between adjacent blocks in different scenes.
    assert moments_conflict_diversity(c_b2, c_b3) is False
    # With min_block_gap=2 -> |2 - 3| < 2 conflicts, |2 - 5| >= 2 does NOT conflict.
    assert moments_conflict_diversity(c_b2, c_b3, block_order_by_id=order, min_block_gap=2) is True
    assert moments_conflict_diversity(c_b2, c_b5, block_order_by_id=order, min_block_gap=2) is False

    # enforce_intra_chapter_diversity activates min_block_gap=2 for >= 12 blocks and promotes c_b5!
    winners, _shortlist, _zero = enforce_intra_chapter_diversity(
        [c_b2, c_b3],
        [c_b5],
        block_order_by_id=order,
    )
    assert [w.candidate_id for w in winners] == ["c-b2", "c-b5"]

    # Without block_order_by_id -> legacy behavior (both c_b2 and c_b3 kept).
    winners_legacy, _, _ = enforce_intra_chapter_diversity([c_b2, c_b3], [c_b5])
    assert [w.candidate_id for w in winners_legacy] == ["c-b2", "c-b3"]

