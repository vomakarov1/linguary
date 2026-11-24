"""Integration tests for 3-transaction ownership, `:a{n}` activations, and `TaskRunner._finish` (`P4.T1`, `ALG-04`)."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, StorageSettings
from linguary.application.analysis_pass_cache import encode_pass
from linguary.application.context_builder import ContextBuilder
from linguary.application.illustration_dto import (
    IllustrationDesignResponse,
    IllustrationExtractResponse,
)
from linguary.application.illustration_source import _pass_model_dependencies
from linguary.application.illustrations import revoke_active_in_transaction
from linguary.application.llm_dto import NarrativeAnalysisResponse, QuotedSceneResponse
from linguary.application.ports.llm import (
    LLMClient,
    LLMRequest,
    LLMResponse,
    LLMResponseFormatError,
    LLMResponseKind,
)
from linguary.application.ports.persistence import BookRecord
from linguary.application.token_budget import narrative_max_fragment_blocks_for
from linguary.composition import build_application, drain_tasks
from linguary.composition.book_runtime import conditions_payload
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
)
from linguary.domain.name_translation import NameTranslationPolicy, NameTranslationStrategy


class _AsyncScriptedClient(LLMClient):
    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.calls: int = 0

    async def generate(self, request: LLMRequest[Any]) -> LLMResponse[Any]:
        self.calls += 1
        res = self._handler(request, self.calls)
        if asyncio.iscoroutine(res):
            res = await res
        return LLMResponse(payload=res, raw_content="", model="gpt-5-mini", provider="fake")


def _seed_book(
    app: Any,
    tmp_path: Path,
    *,
    book_id: str,
    text: str = "Alice walked across the stone courtyard.",
    policy: IllustrationPolicy | None = None,
) -> tuple[str, Path]:
    epub_path = tmp_path / f"{book_id}.epub"
    content = f"epub-{book_id}".encode()
    epub_path.write_bytes(content)
    digest = sha256(content).hexdigest()
    conditions_id = str(uuid4())

    sec_id = f"{book_id}-sec-0"
    blk_id = f"{book_id}:blk:0"
    sec = DocumentSection(
        id=sec_id,
        document_id=book_id,
        title="Chapter 1",
        order=0,
        source_resource_path="Text/ch1.xhtml",
        blocks=(
            DocumentBlock(
                id=blk_id,
                section_id=sec_id,
                type=BlockType.PARAGRAPH,
                order=0,
                text=text,
                source_reference=SourceReference("Text/ch1.xhtml", "p[0]"),
                support_status=SupportStatus.SUPPORTED,
            ),
        ),
    )
    doc = CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title=f"Novel {book_id}"),
        source_language="en",
        target_language="ru",
        sections=(sec,),
    )
    book = BookRecord(
        id=book_id,
        metadata=doc.metadata,
        source_language="en",
        target_language="ru",
        status=BookStatus.GLOSSARY_REVIEW,
        source_epub_path=str(epub_path),
        created_at=datetime(2026, 11, 1, tzinfo=UTC),
    )
    cond_payload = conditions_payload(book, app.settings, revision=1, profile="content-plan-v3")
    cond_payload["source_sha256"] = digest
    cond_payload["staged_analysis"] = True
    cond_payload["llm_parameters"]["staged_analysis"] = True
    cond_payload["checks"]["sequential_analysis"] = False
    prompts_version = int(cond_payload["versions"]["prompts"])
    ctx_builder = ContextBuilder(
        content_plan_profile="content-plan-v3",
        name_translation_policy=NameTranslationPolicy(
            version=1, strategy=NameTranslationStrategy(app.settings.reading.name_translation_strategy)
        ),
        reader_level=app.settings.reading.normalized_level,
    )
    with app.unit_of_work.transaction() as tx:
        tx.books.add(book)
        tx.documents.save(doc)
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
            chapter_summary="Alice crosses the quiet courtyard in the evening light.",
            glossary_candidates=(),
            context_facts=(),
            scenes=(
                QuotedSceneResponse(
                    start_block_id=blk_id,
                    end_block_id=blk_id,
                    summary="Сцена во дворе.",
                    location="Двор",
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
            doc,
            sec,
            (),
            (),
            (),
            model_dependencies=_pass_model_dependencies(cond_payload),
            max_fragment_blocks=narrative_max_fragment_blocks_for(cond_payload["llm_parameters"]),
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
        tx.illustrations.save_settings(
            book_id,
            expected_revision=0,
            settings=IllustrationSettings(
                enabled=True,
                max_per_chapter=3,
                model_reference="gpt-5-mini",
                model_prompts="gpt-5-mini",
                policy=policy or IllustrationPolicy(max_repair_turns=2, max_calls_per_scope_activation=3),
            ),
        )
    return sec_id, epub_path


def _make_app(tmp_path: Path, client: LLMClient) -> Any:
    return build_application(
        settings=AppSettings(
            storage=StorageSettings(data_root=str(tmp_path / "data")),
            llm=LLMSettings(
                provider="openai_compatible",
                base_url="https://api.openai.com/v1",
                model="gpt-5-mini",
                api_key="test-key",
                staged_analysis=True,
            ),
            checks=CheckSettings(sequential_analysis=False),
        ),
        llm_client=client,
    )


async def test_task_runner_retry_end_to_end_preserves_failed_task_dispatches_a2_and_increments_ordinal_4_to_6(
    tmp_path: Path,
) -> None:
    def _handler(req: LLMRequest[Any], call_no: int) -> Any:
        data = json.loads(req.input_text)
        if call_no <= 5:
            # First activation uses 3 calls (ordinals 1..3 -> FAILED); second activation uses ordinals 4, 5 (fail) and 6 (succeed)!
            raise LLMResponseFormatError(f"bad format on call {call_no}")
        if req.kind is LLMResponseKind.ILLUSTRATION_EXTRACT:
            return IllustrationExtractResponse(
                mentions=(),
                facts=(),
                covered_source_unit_ids=(data["units"][0]["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_DESIGN:
            return IllustrationDesignResponse(entity_id=data["entity_id"], traits=())
        # For chapter stage after registry seals, return empty candidates cleanly.
        from linguary.application.illustration_dto import IllustrationCandidatesResponse

        return IllustrationCandidatesResponse(
            scene_id=data["scene_id"],
            events=(),
            zero_reason_ru="Тихая сцена.",
            covered_source_unit_ids=tuple(b["source_unit_id"] for b in data["blocks"]),
            complete=True,
        )

    client = _AsyncScriptedClient(_handler)
    app = _make_app(tmp_path, client)
    try:
        _seed_book(app, tmp_path, book_id="book-e2e")
        view1 = await app.illustrations.prepare("book-e2e", expected_revision=1)
        gen_id = view1.active_generation_id
        assert gen_id is not None

        # Drain activation :a1 -> TaskRunner._finish MUST preserve FAILED (not overwrite with SUCCEEDED)!
        finished1 = await drain_tasks(app)
        assert len(finished1) == 1
        assert finished1[0].status is TaskStatus.FAILED
        assert finished1[0].operation_id == f"{gen_id}:a1"

        with app.unit_of_work.read_transaction() as tx:
            from sqlalchemy import select
            from linguary.adapters.db.illustration_models import IllustrationAttemptRow

            scope = tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.EXTRACT)[0]
            assert scope.state is IllustrationScopeState.FAILED
            ordinals1 = list(
                tx.illustrations._session.scalars(
                    select(IllustrationAttemptRow.ordinal)
                    .where(IllustrationAttemptRow.scope_id == scope.id)
                    .order_by(IllustrationAttemptRow.ordinal)
                ).all()
            )
            assert ordinals1 == [1, 2, 3]

        # Explicit retry -> supersedes :a1 task and dispatches :a2!
        await app.illustrations.retry(
            "book-e2e", expected_revision=1, expected_generation_id=gen_id
        )
        with app.unit_of_work.read_transaction() as tx:
            ref_tasks = sorted(
                [
                    t
                    for t in tx.tasks.list_for_book("book-e2e")
                    if t.task_type is TaskType.PREPARE_ILLUSTRATION_REFERENCE
                ],
                key=lambda t: t.operation_id,
            )
            assert [t.operation_id for t in ref_tasks] == [f"{gen_id}:a1", f"{gen_id}:a2"]
            assert ref_tasks[0].status is TaskStatus.SUPERSEDED
            assert ref_tasks[1].status is TaskStatus.PENDING

        # Drain :a2 -> ordinals 4, 5, 6 succeed!
        await drain_tasks(app)
        with app.unit_of_work.read_transaction() as tx:
            from sqlalchemy import select
            from linguary.adapters.db.illustration_models import IllustrationAttemptRow

            scope = tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.EXTRACT)[0]
            assert scope.state is IllustrationScopeState.ACCEPTED
            ordinals2 = list(
                tx.illustrations._session.scalars(
                    select(IllustrationAttemptRow.ordinal)
                    .where(IllustrationAttemptRow.scope_id == scope.id)
                    .order_by(IllustrationAttemptRow.ordinal)
                ).all()
            )
            assert ordinals2 == [1, 2, 3, 4, 5, 6]
            gen = tx.illustrations.get_generation("book-e2e", gen_id)
            assert gen is not None and gen.state is IllustrationGenerationState.READY
    finally:
        await app.aclose()


async def test_call_reservation_release_before_send_and_consume_after_send(tmp_path: Path) -> None:
    huge = "Alice walked " + ("down the stone hallway " * 1400)
    client = _AsyncScriptedClient(
        lambda _r, _n: (_ for _ in ()).throw(AssertionError("HTTP must not be called"))
    )
    app = _make_app(tmp_path, client)
    try:
        _seed_book(app, tmp_path, book_id="book-res", text=huge)
        view = await app.illustrations.prepare("book-res", expected_revision=1)
        gen_id = view.active_generation_id
        assert gen_id is not None
        await drain_tasks(app)
        assert client.calls == 0
        with app.unit_of_work.read_transaction() as tx:
            gen = tx.illustrations.get_generation("book-res", gen_id)
            assert gen is not None
            assert gen.calls_reserved == 0
            assert gen.calls_sent == 0
    finally:
        await app.aclose()


async def test_multi_call_scope_activation_ownership(tmp_path: Path) -> None:
    def _handler(req: LLMRequest[Any], call_no: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_EXTRACT:
            if call_no == 1:
                raise LLMResponseFormatError("first attempt invalid")
            return IllustrationExtractResponse(
                mentions=(),
                facts=(),
                covered_source_unit_ids=(data["units"][0]["source_unit_id"],),
                complete=True,
            )
        from linguary.application.illustration_dto import IllustrationCandidatesResponse

        return IllustrationCandidatesResponse(
            scene_id=data["scene_id"],
            events=(),
            zero_reason_ru="Тихо.",
            covered_source_unit_ids=tuple(b["source_unit_id"] for b in data["blocks"]),
            complete=True,
        )

    client = _AsyncScriptedClient(_handler)
    app = _make_app(tmp_path, client)
    try:
        _seed_book(app, tmp_path, book_id="book-multi")
        view = await app.illustrations.prepare("book-multi", expected_revision=1)
        gen_id = view.active_generation_id
        assert gen_id is not None
        await drain_tasks(app)
        with app.unit_of_work.read_transaction() as tx:
            gen = tx.illustrations.get_generation("book-multi", gen_id)
            assert gen is not None
            assert gen.calls_reserved == 0
            assert gen.calls_sent == 3  # 2 for extract (1 rejected + 1 accepted) + 1 for candidates
            assert gen.state is IllustrationGenerationState.READY
    finally:
        await app.aclose()


async def test_late_answer_cannot_replace_current_and_exits_running_state(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_handler(req: LLMRequest[Any], _n: int) -> Any:
        started.set()
        await release.wait()
        data = json.loads(req.input_text)
        return IllustrationExtractResponse(
            mentions=(),
            facts=(),
            covered_source_unit_ids=(data["units"][0]["source_unit_id"],),
            complete=True,
        )

    client = _AsyncScriptedClient(_slow_handler)
    app = _make_app(tmp_path, client)
    try:
        _seed_book(app, tmp_path, book_id="book-late")
        view = await app.illustrations.prepare("book-late", expected_revision=1)
        gen1 = view.active_generation_id
        assert gen1 is not None

        drain_coro = asyncio.create_task(drain_tasks(app))
        await started.wait()

        # While the call is in flight, change settings -> revokes gen1!
        app.illustrations.save_settings(
            "book-late",
            expected_revision=1,
            settings=IllustrationSettings(
                enabled=True,
                max_per_chapter=2,
                model_reference="gpt-5-mini",
                model_prompts="gpt-5-mini",
            ),
        )
        release.set()
        await drain_coro

        with app.unit_of_work.read_transaction() as tx:
            head = tx.illustrations.get_head("book-late")
            assert head.active_generation_id is None
            sc = tx.illustrations.list_scopes(gen1, stage=IllustrationScopeStage.EXTRACT)[0]
            # Must NOT remain stuck in running and must NOT be accepted!
            assert sc.state is IllustrationScopeState.FAILED
            assert sc.payload is None
    finally:
        await app.aclose()


async def test_disable_and_source_change_during_call_exits_running_state(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_handler(req: LLMRequest[Any], _n: int) -> Any:
        started.set()
        await release.wait()
        data = json.loads(req.input_text)
        return IllustrationExtractResponse(
            mentions=(),
            facts=(),
            covered_source_unit_ids=(data["units"][0]["source_unit_id"],),
            complete=True,
        )

    client = _AsyncScriptedClient(_slow_handler)
    app = _make_app(tmp_path, client)
    try:
        _, epub_path = _seed_book(app, tmp_path, book_id="book-srcchg")
        view = await app.illustrations.prepare("book-srcchg", expected_revision=1)
        gen_id = view.active_generation_id
        assert gen_id is not None

        drain_coro = asyncio.create_task(drain_tasks(app))
        await started.wait()

        # Mutate the source EPUB file on disk while the LLM call is in flight!
        epub_path.write_bytes(b"mutated-epub-bytes-different-sha")
        release.set()
        await drain_coro

        with app.unit_of_work.read_transaction() as tx:
            gen = tx.illustrations.get_generation("book-srcchg", gen_id)
            assert gen is not None
            assert gen.state is IllustrationGenerationState.REVOKED
            assert gen.reason_code == IllustrationReasonCode.SOURCE_CHANGED.value
            sc = tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.EXTRACT)[0]
            assert sc.state is IllustrationScopeState.FAILED
            assert sc.reason_code == IllustrationReasonCode.SOURCE_CHANGED.value
    finally:
        await app.aclose()


async def test_revoke_supersedes_pending_running_and_failed_tasks_and_worker_keeps_superseded(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_handler(req: LLMRequest[Any], _n: int) -> Any:
        started.set()
        await release.wait()
        data = json.loads(req.input_text)
        return IllustrationExtractResponse(
            mentions=(),
            facts=(),
            covered_source_unit_ids=(data["units"][0]["source_unit_id"],),
            complete=True,
        )

    client = _AsyncScriptedClient(_slow_handler)
    app = _make_app(tmp_path, client)
    try:
        _seed_book(app, tmp_path, book_id="book-sup")
        view = await app.illustrations.prepare("book-sup", expected_revision=1)
        gen_id = view.active_generation_id
        assert gen_id is not None

        drain_coro = asyncio.create_task(drain_tasks(app))
        await started.wait()

        # Revoke while task is RUNNING -> task is superseded in DB; worker returning MUST keep SUPERSEDED!
        with app.unit_of_work.transaction() as tx:
            revoke_active_in_transaction(
                tx, "book-sup", reason_code=IllustrationReasonCode.SETTINGS_CHANGED.value
            )
        release.set()
        finished = await drain_coro
        assert finished[0].status is TaskStatus.SUPERSEDED
        with app.unit_of_work.read_transaction() as tx:
            task = tx.tasks.get(finished[0].id)
            assert task is not None and task.status is TaskStatus.SUPERSEDED
    finally:
        await app.aclose()


async def test_disable_mid_call_and_split_or_fail_rejected_on_revoked_or_stale(
    tmp_path: Path,
) -> None:
    import pytest
    from linguary.domain.illustrations import IllustrationOwnershipError, IllustrationScopeSpec

    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_handler(req: LLMRequest[Any], _n: int) -> Any:
        started.set()
        await release.wait()
        data = json.loads(req.input_text)
        return IllustrationExtractResponse(
            mentions=(),
            facts=(),
            covered_source_unit_ids=(data["units"][0]["source_unit_id"],),
            complete=True,
        )

    client = _AsyncScriptedClient(_slow_handler)
    app = _make_app(tmp_path, client)
    try:
        _seed_book(app, tmp_path, book_id="book-dis-mid")
        view = await app.illustrations.prepare("book-dis-mid", expected_revision=1)
        gen_id = view.active_generation_id
        assert gen_id is not None

        drain_coro = asyncio.create_task(drain_tasks(app))
        await started.wait()

        # Disable channel mid-call via save_settings -> revokes gen and marks running scope FAILED!
        app.illustrations.save_settings(
            "book-dis-mid",
            expected_revision=1,
            settings=IllustrationSettings(
                enabled=False,
                max_per_chapter=3,
                model_reference="gpt-5-mini",
                model_prompts="gpt-5-mini",
            ),
        )
        release.set()
        await drain_coro

        with app.unit_of_work.transaction() as tx:
            sc = tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.EXTRACT)[0]
            assert sc.state is IllustrationScopeState.FAILED
            assert sc.payload is None
            # Direct repository calls on a revoked generation MUST raise IllustrationOwnershipError!
            with pytest.raises(IllustrationOwnershipError):
                tx.illustrations.fail_scope(
                    sc.id, attempt_id=None, reason_code=IllustrationReasonCode.PROVIDER_ERROR.value
                )
            with pytest.raises(IllustrationOwnershipError):
                tx.illustrations.split_scope(
                    sc.id,
                    attempt_id=None,
                    children=[
                        IllustrationScopeSpec(
                            stage=IllustrationScopeStage.EXTRACT,
                            scope_key="extract:child",
                            input_sha256="a" * 64,
                            summary={},
                        )
                    ],
                    max_scopes=100,
                )
    finally:
        await app.aclose()


async def test_mixed_scope_failure_and_late_acceptance_transitions_gen_to_failed(
    tmp_path: Path,
) -> None:
    from linguary.application.illustrations import finalize_generation_state_in_tx
    from linguary.domain.illustrations import IllustrationScopeSpec

    client = _AsyncScriptedClient(lambda _r, _n: (_ for _ in ()).throw(AssertionError("unused")))
    app = _make_app(tmp_path, client)
    try:
        _seed_book(app, tmp_path, book_id="book-mix")
        with app.unit_of_work.transaction() as tx:
            head = tx.illustrations.get_head("book-mix")
            assert head.settings_id is not None
            gen, _ = tx.illustrations.create_or_get_generation(
                "book-mix",
                settings_id=head.settings_id,
                request_key="req-mix",
                source_sha256="a" * 64,
                source_authority={},
                requested_sections=("book-mix-sec-0",),
            )
            tx.illustrations.update_generation_state(
                "book-mix", gen.id, state=IllustrationGenerationState.RUNNING
            )
            sc1, sc2 = tx.illustrations.upsert_scopes(
                gen.id,
                [
                    IllustrationScopeSpec(
                        stage=IllustrationScopeStage.EXTRACT,
                        scope_key="extract:p0",
                        input_sha256="1" * 64,
                        summary={},
                    ),
                    IllustrationScopeSpec(
                        stage=IllustrationScopeStage.EXTRACT,
                        scope_key="extract:p1",
                        input_sha256="2" * 64,
                        summary={},
                    ),
                ],
            )
            # First scope fails while second is still pending -> gen stays RUNNING
            tx.illustrations.fail_scope(
                sc1.id, attempt_id=None, reason_code=IllustrationReasonCode.PROVIDER_ERROR.value
            )
            finalize_generation_state_in_tx(tx, "book-mix", gen.id)
            mid_gen = tx.illustrations.get_generation("book-mix", gen.id)
            assert mid_gen is not None and mid_gen.state is IllustrationGenerationState.RUNNING

            # Second scope later accepts -> finalize_generation_state_in_tx transitions gen to FAILED!
            tx.illustrations.accept_scope(
                sc2.id,
                attempt_id=None,
                expected_input_sha256=sc2.input_sha256,
                payload={"mentions": [], "facts": []},
                output_sha256="3" * 64,
                summary={},
            )
            finalize_generation_state_in_tx(tx, "book-mix", gen.id)
            final_gen = tx.illustrations.get_generation("book-mix", gen.id)
            assert final_gen is not None
            assert final_gen.state is IllustrationGenerationState.FAILED
            assert final_gen.reason_code == IllustrationReasonCode.PROVIDER_ERROR.value
    finally:
        await app.aclose()
