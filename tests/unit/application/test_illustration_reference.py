"""Unit and persistence tests for bounded reusable illustration registry (`P2.T3`, `ALG-02`)."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, ReadingSettings, StorageSettings
from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm.openai_compatible import serialize_request
from linguary.adapters.validation import Py3LangidDetector
from linguary.application.analysis_pass_cache import encode_pass
from linguary.application.context_builder import ContextBuilder, source_text_for_model
from linguary.application.illustration_context import (
    REPAIR_RESERVE_BYTES,
    IllustrationContext,
    closed_traits_for_kind,
    compact_repair_summary,
    pack_items_by_wire_budget,
    resolve_quote_anchor,
)
from linguary.application.illustration_dto import (
    IllustrationAliasDecisionDTO,
    IllustrationAliasResponse,
    IllustrationDesignResponse,
    IllustrationDesignTraitDTO,
    IllustrationExtractResponse,
    IllustrationFactDTO,
    IllustrationMentionDTO,
)
from linguary.application.illustration_reference import (
    IllustrationReference,
    _build_bounded_design_input,
    build_alias_candidate_pairs,
    resolve_appearance_with_design_fallback,
    resolve_verdict_graph_clusters,
)
from linguary.application.illustration_source import (
    IllustrationSource,
    _pass_model_dependencies,
)
from linguary.application.token_budget import narrative_max_fragment_blocks_for
from linguary.application.llm_dto import (
    NarrativeAnalysisResponse,
    QuotedSceneResponse,
)
from linguary.application.ports.illustrations import RequestWireSizer, RequestWireSnapshot
from linguary.application.ports.llm import LLMClient, LLMRequest, LLMResponse, LLMResponseKind
from linguary.application.ports.persistence import BookRecord
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
    InlineAnnotation,
    InlineAnnotationType,
    SourceReference,
    SupportStatus,
    TaskStatus,
    TaskType,
)
from linguary.domain.book_conditions import BookConditions
from linguary.domain.content_plan import canonical_sha
from linguary.domain.illustrations import (
    TIMELESS_POSITION_KEY,
    IllustrationGenerationState,
    IllustrationPolicy,
    IllustrationReasonCode,
    IllustrationRegistryKind,
    IllustrationScopeStage,
    IllustrationScopeState,
    IllustrationSettings,
    IllustrationValidationError,
    format_position_key,
)
from linguary.domain.name_translation import NameTranslationPolicy, NameTranslationStrategy


class _ScriptedClient(LLMClient):
    """Deterministic stub client recording every physical `LLMRequest`."""

    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.requests: list[LLMRequest[Any]] = []

    async def generate(self, request: LLMRequest[Any]) -> LLMResponse[Any]:
        self.requests.append(request)
        payload = self._handler(request, len(self.requests))
        return LLMResponse(
            payload=payload,
            raw_content="",
            model="gpt-5-mini",
            provider="fake",
        )


def _real_sizer(model: str = "gpt-5-mini") -> RequestWireSizer:
    settings = LLMSettings(
        provider="openai_compatible",
        base_url="https://api.openai.com/v1",
        model=model,
        api_key="test-key",
        temperature=0.2,
        max_output_tokens=4096,
    )

    class _Sizer:
        def measure(self, request: LLMRequest[Any]) -> RequestWireSnapshot:
            wire = serialize_request(
                request, settings, include_openrouter_provider=False, explicit_cache=False
            )
            return RequestWireSnapshot(
                sha256=sha256(wire).hexdigest(),
                size_bytes=len(wire),
            )

    return _Sizer()


@dataclass
class _Harness:
    uow: SqlAlchemyUnitOfWork
    source: IllustrationSource
    runtime: IllustrationRuntime
    client: _ScriptedClient
    book_id: str
    conditions_id: str
    source_sha: str


def _seed_book_with_narrative(
    tmp_path: Path,
    *,
    sections_blocks: list[tuple[str, str, list[tuple[str, str]]]],
    handler: Any,
    policy: IllustrationPolicy | None = None,
    book_id: str = "book-ref",
) -> _Harness:
    db = tmp_path / f"{book_id}.db"
    apply_migrations(db)
    uow = SqlAlchemyUnitOfWork.for_engine(create_database_engine(db))
    source_sha = "a" * 64
    conditions_id = "11111111-1111-4111-8111-111111111111"

    doc_sections: list[DocumentSection] = []
    for s_idx, (sec_id, title, blocks_spec) in enumerate(sections_blocks):
        blocks = tuple(
            DocumentBlock(
                id=b_id,
                section_id=sec_id,
                type=BlockType.PARAGRAPH,
                order=b_idx,
                text=b_text,
                source_reference=SourceReference(f"Text/ch{s_idx}.xhtml", f"p[{b_idx}]"),
                support_status=SupportStatus.SUPPORTED,
            )
            for b_idx, (b_id, b_text) in enumerate(blocks_spec)
        )
        doc_sections.append(
            DocumentSection(
                id=sec_id,
                document_id=book_id,
                order=s_idx,
                title=title,
                source_resource_path=f"Text/ch{s_idx}.xhtml",
                blocks=blocks,
            )
        )

    document = CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title="Test Novel"),
        source_language="en",
        target_language="ru",
        sections=tuple(doc_sections),
    )

    epub_file = tmp_path / f"{book_id}.epub"
    epub_file.write_bytes(b"fake-epub-content")
    book = BookRecord(
        id=book_id,
        metadata=BookMetadata(title="Test Novel"),
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

        # Save valid full NARRATIVE passes matching IllustrationSource cryptographic checks.
        model_deps = _pass_model_dependencies(cond_payload)
        max_frag_blocks = narrative_max_fragment_blocks_for(
            cond_payload["llm_parameters"]
        )
        for sec in doc_sections:
            first_block_id = sec.blocks[0].id
            last_block_id = sec.blocks[-1].id
            narrative = NarrativeAnalysisResponse(
                translation_instructions="Сохранять спокойный ритм повествования.",
                chapter_summary=f"In {sec.title} the characters appear inside the quiet stone hall.",
                glossary_candidates=(),
                context_facts=(),
                scenes=(
                    QuotedSceneResponse(
                        start_block_id=first_block_id,
                        end_block_id=last_block_id,
                        summary=f"Сцена разворачивается в разделе {sec.title} в тихом зале.",
                        location="Большой зал",
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
            output_sha = canonical_sha(encoded)
            input_sha = ctx_builder.narrative_input_sha256(
                document,
                sec,
                (),
                (),
                (),
                model_dependencies=model_deps,
                max_fragment_blocks=max_frag_blocks,
            )
            tx.analyses.save_pass(
                AnalysisPassResult(
                    id=f"pass-{sec.id}",
                    book_id=book_id,
                    section_id=sec.id,
                    pass_kind=AnalysisPassKind.NARRATIVE,
                    scope_key="",
                    input_sha256=input_sha,
                    output_sha256=output_sha,
                    conditions_id=conditions_id,
                    attempt_id=None,
                    payload=encoded,
                )
            )

        settings = IllustrationSettings(
            enabled=True,
            max_per_chapter=3,
            model_reference="gpt-5-mini",
            model_prompts="gpt-5-mini",
            policy=policy or IllustrationPolicy(),
        )
        tx.illustrations.save_settings(book_id, expected_revision=0, settings=settings)

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
    return _Harness(
        uow=uow,
        source=source,
        runtime=runtime,
        client=client,
        book_id=book_id,
        conditions_id=conditions_id,
        source_sha=source_sha,
    )


def _create_and_plan(harness: _Harness) -> tuple[IllustrationReference, str]:
    ref = IllustrationReference(
        harness.uow,
        source=harness.source,
        runtime=harness.runtime,
        sleep_fn=_noop_sleep,
    )
    with harness.uow.transaction() as tx:
        head = tx.illustrations.get_head(harness.book_id)
        snap = harness.source.snapshot(tx, harness.book_id)
        assert snap.ready and snap.manifest is not None and snap.manifest_sha256 is not None
        assert head.settings_id is not None
        gen, _ = tx.illustrations.create_or_get_generation(
            harness.book_id,
            settings_id=head.settings_id,
            request_key=f"req-{harness.book_id}-{head.settings_id}",
            source_sha256=harness.source_sha,
            source_authority=snap.source_authority or {},
            requested_sections=snap.content_section_ids,
        )
        gen = tx.illustrations.seal_manifest(
            harness.book_id,
            gen.id,
            manifest=snap.manifest,
            manifest_sha256=snap.manifest_sha256,
        )
        gen = tx.illustrations.update_generation_state(
            harness.book_id, gen.id, state=IllustrationGenerationState.QUEUED, reason_code=None
        )
        ref.plan_scopes(tx, gen, snap)
        return ref, gen.id


async def _noop_sleep(_seconds: float) -> None:
    return None


async def _drain_ref_tasks(harness: _Harness, ref: IllustrationReference) -> int:
    ran = 0
    while True:
        with harness.uow.transaction() as tx:
            pending = [
                t
                for t in tx.tasks.list_for_book(harness.book_id)
                if t.task_type is TaskType.PREPARE_ILLUSTRATION_REFERENCE
                and t.status is TaskStatus.PENDING
            ]
        if not pending:
            break
        for task in pending:
            with harness.uow.transaction() as tx:
                running = task.start()
                tx.tasks.save(running)
            await ref.run_task(running)
            ran += 1
    return ran


def test_position_key_numeric_order_ch10_after_ch2_and_bounds_bool_validation() -> None:
    ch2 = format_position_key(2, 15, 40)
    ch10 = format_position_key(10, 1, 0)
    assert ch2 < ch10
    assert ch2 == "000002:000015:00000040"
    assert ch10 == "000010:000001:00000000"
    assert TIMELESS_POSITION_KEY == "000000:000000:00000000"

    for bad_args in (
        (True, 0, 0),
        (0, False, 0),
        (0, 0, True),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (1_000_000, 0, 0),
        (0, 1_000_000, 0),
        (0, 0, 100_000_000),
    ):
        with pytest.raises(IllustrationValidationError):
            format_position_key(*bad_args)  # type: ignore[arg-type]


async def test_huge_single_fact_fails_before_http(tmp_path: Path) -> None:
    huge_text = "Alice wore a vast cloak " + ("with golden embroidery " * 1400)
    harness = _seed_book_with_narrative(
        tmp_path,
        sections_blocks=[("sec-1", "Chapter 1", [("b-huge", huge_text)])],
        handler=lambda _req, _n: (_ for _ in ()).throw(AssertionError("HTTP must not be called")),
    )
    ref, gen_id = _create_and_plan(harness)
    await _drain_ref_tasks(harness, ref)

    assert len(harness.client.requests) == 0
    with harness.uow.transaction() as tx:
        gen = tx.illustrations.get_generation(harness.book_id, gen_id)
        assert gen is not None
        assert gen.calls_reserved == 0
        assert gen.calls_sent == 0
        assert gen.state is IllustrationGenerationState.FAILED
        assert gen.reason_code == IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value
        scopes = tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.EXTRACT)
        assert len(scopes) == 1
        assert scopes[0].state is IllustrationScopeState.FAILED
        assert scopes[0].reason_code == IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value


def test_repair_turn_fits_wire_byte_cap_on_maximally_packed_page_without_raw_echo() -> None:
    policy = IllustrationPolicy()
    sizer = _real_sizer()
    items = [
        {
            "source_unit_id": f"sec-1:scene:{i}",
            "blocks": [{"block_id": f"b-{i}", "text": f"Paragraph {i}: " + ("word " * 120)}],
        }
        for i in range(40)
    ]

    def _build(page: Any) -> LLMRequest[Any]:
        return IllustrationContext.build_for_kind(
            kind=LLMResponseKind.ILLUSTRATION_EXTRACT,
            stage=IllustrationScopeStage.EXTRACT,
            input_payload={"units": list(page)},
            book_id="book-1",
            conditions_id="11111111-1111-4111-8111-111111111111",
            policy=policy,
        )

    pages = pack_items_by_wire_budget(
        items, build_request=_build, sizer=sizer, target_bytes=policy.initial_target_bytes
    )
    assert len(pages) >= 2
    first_page = pages[0]
    base_req = _build(first_page)
    base_bytes = sizer.measure(base_req).size_bytes
    assert base_bytes <= policy.max_request_bytes - REPAIR_RESERVE_BYTES

    long_repair = compact_repair_summary(
        "Цитаты неоднозначны и не совпадают с текстом " * 40,
        details=("повторная цитата " * 30,),
    )
    assert len(long_repair.encode("utf-8")) <= 1024
    repair_req = IllustrationContext.build_for_kind(
        kind=LLMResponseKind.ILLUSTRATION_EXTRACT,
        stage=IllustrationScopeStage.EXTRACT,
        input_payload={"units": list(first_page)},
        book_id="book-1",
        conditions_id="11111111-1111-4111-8111-111111111111",
        policy=policy,
        repair_summary=long_repair,
    )
    assert repair_req.repair_turns == ((None, long_repair),)
    repair_bytes = sizer.measure(repair_req).size_bytes
    assert repair_bytes <= policy.max_request_bytes


async def test_complete_false_splits_composite_and_fails_atomic(tmp_path: Path) -> None:
    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_EXTRACT:
            units = data["units"]
            blocks = [b for u in units for b in u["blocks"]]
            # Return complete=False when >1 block (composite), and also for b-atomic!
            if len(blocks) > 1:
                return IllustrationExtractResponse(
                    mentions=(), facts=(), covered_source_unit_ids=(), complete=False
                )
            if blocks[0]["block_id"] == "b-atomic":
                return IllustrationExtractResponse(
                    mentions=(), facts=(), covered_source_unit_ids=(), complete=False
                )
            return IllustrationExtractResponse(
                mentions=(),
                facts=(),
                covered_source_unit_ids=(units[0]["source_unit_id"],),
                complete=True,
            )
        raise AssertionError(f"unexpected kind {req.kind}")

    # Composite case (2 blocks -> splits into 2 leaves -> succeeds!)
    h1 = _seed_book_with_narrative(
        tmp_path,
        sections_blocks=[
            ("sec-1", "Ch 1", [("b-1", "Alice entered the room."), ("b-2", "Bob waved.")])
        ],
        handler=_handler,
        book_id="book-comp",
    )
    ref1, gen1 = _create_and_plan(h1)
    await _drain_ref_tasks(h1, ref1)
    with h1.uow.transaction() as tx:
        scopes = tx.illustrations.list_scopes(gen1, stage=IllustrationScopeStage.EXTRACT)
        states = [s.state for s in scopes]
        assert IllustrationScopeState.SPLIT in states
        assert states.count(IllustrationScopeState.ACCEPTED) == 2
        g = tx.illustrations.get_generation(h1.book_id, gen1)
        assert g is not None and g.registry_sha256 is not None

    # Atomic case (1 block -> complete=False -> fails atomic_input_too_large)
    h2 = _seed_book_with_narrative(
        tmp_path,
        sections_blocks=[("sec-1", "Ch 1", [("b-atomic", "Single atomic block.")])],
        handler=_handler,
        book_id="book-atom",
    )
    ref2, gen2 = _create_and_plan(h2)
    await _drain_ref_tasks(h2, ref2)
    with h2.uow.transaction() as tx:
        scopes = tx.illustrations.list_scopes(gen2, stage=IllustrationScopeStage.EXTRACT)
        assert len(scopes) == 1
        assert scopes[0].state is IllustrationScopeState.FAILED
        assert scopes[0].reason_code == IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value


async def test_complete_true_missing_covered_units_rejected(tmp_path: Path) -> None:
    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        # Always omits covered_source_unit_ids while claiming complete=True.
        return IllustrationExtractResponse(
            mentions=(), facts=(), covered_source_unit_ids=(), complete=True
        )

    harness = _seed_book_with_narrative(
        tmp_path,
        sections_blocks=[("sec-1", "Ch 1", [("b-1", "Alice walked quietly.")])],
        handler=_handler,
        policy=IllustrationPolicy(max_repair_turns=1, max_calls_per_scope_activation=3),
    )
    ref, gen_id = _create_and_plan(harness)
    await _drain_ref_tasks(harness, ref)

    # 1 initial call + 1 repair call = 2 calls, then fails model_output_invalid.
    assert len(harness.client.requests) == 2
    assert harness.client.requests[1].repair_turns
    with harness.uow.transaction() as tx:
        scopes = tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.EXTRACT)
        assert scopes[0].state is IllustrationScopeState.FAILED
        assert scopes[0].reason_code == IllustrationReasonCode.MODEL_OUTPUT_INVALID.value


async def test_ambiguous_repeated_quote_and_footnote_marker_resolution(tmp_path: Path) -> None:
    # 1. Footnote marker resolution unit check
    footnote_block = DocumentBlock(
        id="b-fn",
        section_id="sec-1",
        type=BlockType.PARAGRAPH,
        order=0,
        text="Alice1 stepped into the quiet stone hall.",
        source_reference=SourceReference("Text/ch1.xhtml", "p[0]"),
        support_status=SupportStatus.SUPPORTED,
        inline_annotations=(
            InlineAnnotation(
                type=InlineAnnotationType.FOOTNOTE_REFERENCE,
                start=5,
                end=6,
                target="fn-1",
            ),
        ),
    )
    assert source_text_for_model(footnote_block) == "Alice stepped into the quiet stone hall."
    resolved = resolve_quote_anchor(
        footnote_block, "stepped into the quiet stone hall", section_order=1, block_order=0
    )
    assert resolved.status == "resolved"
    assert resolved.anchor is not None
    assert footnote_block.text[resolved.anchor.start : resolved.anchor.end] == resolved.anchor.quote
    assert resolved.anchor.start == 7
    assert resolved.position_key == format_position_key(1, 0, 7)

    # 2. Ambiguous repeated quote triggers repair and resolves on unique longer quote
    text = "The red coat hung there. Later Alice put on the red coat by the door."

    def _handler(req: LLMRequest[Any], call_no: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_EXTRACT:
            unit_id = data["units"][0]["source_unit_id"]
            if call_no == 1:
                return IllustrationExtractResponse(
                    mentions=(
                        IllustrationMentionDTO(
                            mention_id="m1",
                            name="Alice",
                            entity_kind="person",
                            alias_of_hint=None,
                            quote_block_id="b-rep",
                            quote="Alice",
                        ),
                    ),
                    facts=(
                        IllustrationFactDTO(
                            fact_id="f1",
                            subject_mention_id="m1",
                            temporal_scope="episodic",
                            back_applicable=False,
                            reveals_future_spoiler=False,
                            description_ru="В красном пальто",
                            description_en="Wearing a red coat",
                            quote_block_id="b-rep",
                            quote="red coat",  # repeated twice -> ambiguous!
                        ),
                    ),
                    covered_source_unit_ids=(unit_id,),
                    complete=True,
                )
            return IllustrationExtractResponse(
                mentions=(
                    IllustrationMentionDTO(
                        mention_id="m1",
                        name="Alice",
                        entity_kind="person",
                        alias_of_hint=None,
                        quote_block_id="b-rep",
                        quote="Alice",
                    ),
                ),
                facts=(
                    IllustrationFactDTO(
                        fact_id="f1",
                        subject_mention_id="m1",
                        temporal_scope="episodic",
                        back_applicable=False,
                        reveals_future_spoiler=False,
                        description_ru="В красном пальто",
                        description_en="Wearing a red coat",
                        quote_block_id="b-rep",
                        quote="put on the red coat by the door",
                    ),
                ),
                covered_source_unit_ids=(unit_id,),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_DESIGN:
            return IllustrationDesignResponse(entity_id=data["entity_id"], traits=())
        raise AssertionError(f"unexpected {req.kind}")

    harness = _seed_book_with_narrative(
        tmp_path,
        sections_blocks=[("sec-1", "Ch 1", [("b-rep", text)])],
        handler=_handler,
    )
    ref, gen_id = _create_and_plan(harness)
    await _drain_ref_tasks(harness, ref)

    assert len(harness.client.requests) >= 2
    assert "повторяется" in harness.client.requests[1].repair_turns[0][1]
    with harness.uow.transaction() as tx:
        gen = tx.illustrations.get_generation(harness.book_id, gen_id)
        assert gen is not None and gen.registry_sha256 is not None
        facts, _ = tx.illustrations.query_registry_items(gen_id, kind=IllustrationRegistryKind.FACT)
        assert len(facts) == 1
        assert facts[0].payload["quote"] == "put on the red coat by the door"


async def test_split_depth_call_limit_and_scope_limit_exhausted(tmp_path: Path) -> None:
    def _always_incomplete(_req: LLMRequest[Any], _n: int) -> Any:
        return IllustrationExtractResponse(
            mentions=(), facts=(), covered_source_unit_ids=(), complete=False
        )

    # 1. split_depth_exhausted (max_split_depth=1 with 4 blocks -> depth 0 splits to depth 1, depth 1 fails)
    h_depth = _seed_book_with_narrative(
        tmp_path,
        sections_blocks=[
            (
                "sec-1",
                "Ch 1",
                [(f"b-{i}", f"Sentence number {i} in the hall.") for i in range(4)],
            )
        ],
        handler=_always_incomplete,
        policy=IllustrationPolicy(max_split_depth=1),
        book_id="book-depth",
    )
    ref_d, gen_d = _create_and_plan(h_depth)
    await _drain_ref_tasks(h_depth, ref_d)
    with h_depth.uow.transaction() as tx:
        gen = tx.illustrations.get_generation(h_depth.book_id, gen_d)
        assert gen is not None and gen.registry_sha256 is None
        failed = tx.illustrations.list_scopes(gen_d, state=IllustrationScopeState.FAILED)
        assert failed and all(
            s.reason_code == IllustrationReasonCode.SPLIT_DEPTH_EXHAUSTED.value for s in failed
        )

    # 2. scope_limit_exhausted (max_scopes_per_generation=2: 1 root + 2 children > 2)
    h_scope = _seed_book_with_narrative(
        tmp_path,
        sections_blocks=[
            ("sec-1", "Ch 1", [("b-1", "First line."), ("b-2", "Second line.")])
        ],
        handler=_always_incomplete,
        policy=IllustrationPolicy(max_scopes_per_generation=2),
        book_id="book-scope-lim",
    )
    ref_s, gen_s = _create_and_plan(h_scope)
    await _drain_ref_tasks(h_scope, ref_s)
    with h_scope.uow.transaction() as tx:
        gen = tx.illustrations.get_generation(h_scope.book_id, gen_s)
        assert gen is not None and gen.registry_sha256 is None
        failed = tx.illustrations.list_scopes(gen_s, state=IllustrationScopeState.FAILED)
        assert len(failed) == 1
        assert failed[0].reason_code == IllustrationReasonCode.SCOPE_LIMIT_EXHAUSTED.value

    # 3. call_limit_exhausted (max_calls_per_generation=1 with 2 scenes that split -> 2nd call blocked)
    h_call = _seed_book_with_narrative(
        tmp_path,
        sections_blocks=[
            ("sec-1", "Ch 1", [("b-1", "First line."), ("b-2", "Second line.")])
        ],
        handler=_always_incomplete,
        policy=IllustrationPolicy(max_calls_per_generation=1),
        book_id="book-call-lim",
    )
    ref_c, gen_c = _create_and_plan(h_call)
    await _drain_ref_tasks(h_call, ref_c)
    with h_call.uow.transaction() as tx:
        gen = tx.illustrations.get_generation(h_call.book_id, gen_c)
        assert gen is not None and gen.registry_sha256 is None
        assert gen.calls_sent == 1
        failed = tx.illustrations.list_scopes(gen_c, state=IllustrationScopeState.FAILED)
        assert failed and all(
            s.reason_code == IllustrationReasonCode.CALL_LIMIT_EXHAUSTED.value for s in failed
        )


async def test_no_unbounded_final_merge(tmp_path: Path) -> None:
    sections = [
        (f"sec-{i}", f"Chapter {i}", [(f"b-{i}", f"Hero{i} stood under arch{i} in silence.")])
        for i in range(1, 5)
    ]

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_EXTRACT:
            mentions = []
            facts = []
            covered = []
            for u in data["units"]:
                covered.append(u["source_unit_id"])
                blk = u["blocks"][0]
                idx = blk["block_id"].split("-")[1]
                mentions.append(
                    IllustrationMentionDTO(
                        mention_id=f"m{idx}",
                        name=f"Hero{idx}",
                        entity_kind="person",
                        alias_of_hint=None,
                        quote_block_id=blk["block_id"],
                        quote=f"Hero{idx}",
                    )
                )
                facts.append(
                    IllustrationFactDTO(
                        fact_id=f"f{idx}",
                        subject_mention_id=f"m{idx}",
                        temporal_scope="permanent",
                        back_applicable=True,
                        reveals_future_spoiler=False,
                        description_ru=f"Герой {idx}",
                        description_en=f"Hero {idx}",
                        quote_block_id=blk["block_id"],
                        quote=f"arch{idx}",
                    )
                )
            return IllustrationExtractResponse(
                mentions=tuple(mentions),
                facts=tuple(facts),
                covered_source_unit_ids=tuple(covered),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_DESIGN:
            return IllustrationDesignResponse(
                entity_id=data["entity_id"],
                traits=(
                    IllustrationDesignTraitDTO(
                        trait="hair",
                        description_ru="Тёмные волосы",
                        description_en="Dark hair",
                        assumption_ru="нейтральное допущение",
                    ),
                ),
            )
        raise AssertionError(f"unexpected kind {req.kind}")

    harness = _seed_book_with_narrative(tmp_path, sections_blocks=sections, handler=_handler)
    ref, gen_id = _create_and_plan(harness)
    await _drain_ref_tasks(harness, ref)

    # Only EXTRACT + per-entity DESIGN calls; NO whole-book final merge call!
    kinds = [r.kind for r in harness.client.requests]
    assert set(kinds) <= {
        LLMResponseKind.ILLUSTRATION_EXTRACT,
        LLMResponseKind.ILLUSTRATION_DESIGN,
    }
    sizer = _real_sizer()
    assert all(sizer.measure(r).size_bytes <= 24576 for r in harness.client.requests)
    with harness.uow.transaction() as tx:
        gen = tx.illustrations.get_generation(harness.book_id, gen_id)
        assert gen is not None and gen.registry_sha256 is not None
        entities, total = tx.illustrations.query_registry_items(
            gen_id, kind=IllustrationRegistryKind.ENTITY
        )
        assert total == 4 and len(entities) == 4


def test_design_large_entity_facts_bounded_deterministically_and_closed_traits_by_kind() -> None:
    assert "hair" in closed_traits_for_kind("person")
    assert "architecture" not in closed_traits_for_kind("person")
    assert "architecture" in closed_traits_for_kind("location")
    assert "hair" not in closed_traits_for_kind("location")

    policy = IllustrationPolicy()
    sizer = _real_sizer()
    entity = {
        "entity_id": "ent-protagonist",
        "canonical_name": "Alice",
        "entity_kind": "person",
    }
    facts = [
        {
            "fact_id": f"perm-{i:03d}",
            "temporal_scope": "permanent",
            "position_key": format_position_key(1, i, 0),
            "description_ru": f"Постоянная черта {i}: длинное описание внешности героя " + ("x" * 180),
            "description_en": f"Permanent feature {i}: detailed character appearance " + ("y" * 180),
        }
        for i in range(15)
    ] + [
        {
            "fact_id": f"epi-{i:03d}",
            "temporal_scope": "episodic",
            "position_key": format_position_key(2, i, 0),
            "description_ru": f"Эпизодическая одежда {i}: " + ("z" * 240),
            "description_en": f"Episodic attire {i}: " + ("w" * 240),
        }
        for i in range(80)
    ]

    bounded1 = _build_bounded_design_input(
        entity=entity,
        facts=facts,
        sizer=sizer,
        book_id="b1",
        conditions_id="11111111-1111-4111-8111-111111111111",
        policy=policy,
    )
    bounded2 = _build_bounded_design_input(
        entity=entity,
        facts=list(reversed(facts)),
        sizer=sizer,
        book_id="b1",
        conditions_id="11111111-1111-4111-8111-111111111111",
        policy=policy,
    )
    assert bounded1 == bounded2
    req = IllustrationContext.build_for_kind(
        kind=LLMResponseKind.ILLUSTRATION_DESIGN,
        stage=IllustrationScopeStage.DESIGN,
        input_payload=bounded1,
        book_id="b1",
        conditions_id="11111111-1111-4111-8111-111111111111",
        policy=policy,
    )
    assert sizer.measure(req).size_bytes <= policy.initial_target_bytes
    included_ids = [f["fact_id"] for f in bounded1["facts"]]
    # All 15 permanent facts are prioritized ahead of episodic facts!
    assert all(f"perm-{i:03d}" in included_ids for i in range(15))
    assert bounded1["omitted_episodic_count"] > 0


async def test_alias_blocking_batched_and_same_name_not_identity(tmp_path: Path) -> None:
    # 7 mentions named "John" -> C(7, 2) = 21 pairs -> must batch into <=16 pairs per ALIAS request.
    # Model returns "different" for all pairs -> same name MUST NOT merge into one entity!
    sections = [
        (f"sec-{i}", f"Chapter {i}", [(f"b-{i}", f"John number {i} spoke quietly.")])
        for i in range(7)
    ]
    alias_batch_sizes: list[int] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_EXTRACT:
            mentions = []
            covered = []
            for u in data["units"]:
                covered.append(u["source_unit_id"])
                blk = u["blocks"][0]
                mentions.append(
                    IllustrationMentionDTO(
                        mention_id=f"m-{blk['block_id']}",
                        name="John",
                        entity_kind="person",
                        alias_of_hint=None,
                        quote_block_id=blk["block_id"],
                        quote="John",
                    )
                )
            return IllustrationExtractResponse(
                mentions=tuple(mentions),
                facts=(),
                covered_source_unit_ids=tuple(covered),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_ALIAS:
            pairs = data["pairs"]
            alias_batch_sizes.append(len(pairs))
            return IllustrationAliasResponse(
                decisions=tuple(
                    IllustrationAliasDecisionDTO(
                        pair_id=p["pair_id"], verdict="different", evidence_fact_ids=()
                    )
                    for p in pairs
                )
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_DESIGN:
            return IllustrationDesignResponse(entity_id=data["entity_id"], traits=())
        raise AssertionError(f"unexpected {req.kind}")

    harness = _seed_book_with_narrative(tmp_path, sections_blocks=sections, handler=_handler)
    ref, gen_id = _create_and_plan(harness)
    await _drain_ref_tasks(harness, ref)

    assert sum(alias_batch_sizes) == 21
    assert len(alias_batch_sizes) == 2
    assert all(1 <= sz <= 16 for sz in alias_batch_sizes)
    with harness.uow.transaction() as tx:
        entities, total = tx.illustrations.query_registry_items(
            gen_id, kind=IllustrationRegistryKind.ENTITY
        )
        # Same name "John" was NOT auto-merged -> all 7 remain distinct entities!
        assert total == 7 and len(entities) == 7

    # Also verify subquadratic blocking: unrelated names produce 0 candidate pairs.
    unrelated = [
        {"mention_key": f"m{i}", "name": f"UniqueName{i}", "entity_kind": "person"}
        for i in range(50)
    ]
    assert build_alias_candidate_pairs(unrelated) == ()


def test_transitive_identity_conflict_triangle_order_independent_deterministic() -> None:
    nodes = ["A", "B", "C", "D"]
    same_base = [("A", "B"), ("B", "C"), ("A", "D")]
    diff_base = [("A", "C")]

    results = set()
    for same_perm in itertools.permutations(same_base):
        clusters = resolve_verdict_graph_clusters(
            nodes,
            same_edges=list(same_perm),
            diff_edges=diff_base,
        )
        results.add(clusters)
        # A and C MUST NEVER be in the same cluster, and bridge node B stays separate!
        for comp in clusters:
            assert not ("A" in comp and "C" in comp)
        assert ("B",) in clusters
        assert ("A", "D") in clusters
        assert ("C",) in clusters

    # Identical across all 6 permutations of edge order!
    assert len(results) == 1


async def test_saved_shards_resume_without_calls(tmp_path: Path) -> None:
    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_EXTRACT:
            u = data["units"][0]
            blk = u["blocks"][0]
            return IllustrationExtractResponse(
                mentions=(
                    IllustrationMentionDTO(
                        mention_id="m1",
                        name="Alice",
                        entity_kind="person",
                        alias_of_hint=None,
                        quote_block_id=blk["block_id"],
                        quote="Alice",
                    ),
                ),
                facts=(
                    IllustrationFactDTO(
                        fact_id="f1",
                        subject_mention_id="m1",
                        temporal_scope="permanent",
                        back_applicable=True,
                        reveals_future_spoiler=False,
                        description_ru="Зелёные глаза",
                        description_en="Green eyes",
                        quote_block_id=blk["block_id"],
                        quote="green eyes",
                    ),
                ),
                covered_source_unit_ids=(u["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_DESIGN:
            return IllustrationDesignResponse(
                entity_id=data["entity_id"],
                traits=(
                    IllustrationDesignTraitDTO(
                        trait="hair",
                        description_ru="Короткие тёмные волосы",
                        description_en="Short dark hair",
                        assumption_ru="нейтральный выбор",
                    ),
                ),
            )
        raise AssertionError(f"unexpected {req.kind}")

    harness = _seed_book_with_narrative(
        tmp_path,
        sections_blocks=[("sec-1", "Ch 1", [("b-1", "Alice had bright green eyes.")])],
        handler=_handler,
    )
    ref, gen1 = _create_and_plan(harness)
    await _drain_ref_tasks(harness, ref)
    calls_first = len(harness.client.requests)
    assert calls_first == 2  # 1 EXTRACT + 1 DESIGN

    # Revoke gen1 and start gen2 under same settings/source -> all shards reused with 0 new calls!
    with harness.uow.transaction() as tx:
        tx.illustrations.revoke_generation(harness.book_id, gen1, reason_code="settings_changed")
        head = tx.illustrations.get_head(harness.book_id)
        snap = harness.source.snapshot(tx, harness.book_id)
        assert head.settings_id is not None and snap.manifest is not None and snap.manifest_sha256 is not None
        gen2, _ = tx.illustrations.create_or_get_generation(
            harness.book_id,
            settings_id=head.settings_id,
            request_key="req-gen2",
            source_sha256=harness.source_sha,
            source_authority=snap.source_authority or {},
            requested_sections=snap.content_section_ids,
        )
        gen2 = tx.illustrations.seal_manifest(
            harness.book_id,
            gen2.id,
            manifest=snap.manifest,
            manifest_sha256=snap.manifest_sha256,
        )
        gen2 = tx.illustrations.update_generation_state(
            harness.book_id, gen2.id, state=IllustrationGenerationState.QUEUED, reason_code=None
        )
        ref.plan_scopes(tx, gen2, snap)
        gen2_after = tx.illustrations.get_generation(harness.book_id, gen2.id)
        assert gen2_after is not None and gen2_after.registry_sha256 is not None

    await _drain_ref_tasks(harness, ref)
    assert len(harness.client.requests) == calls_first  # Zero extra LLM calls!


def test_design_closed_traits_do_not_override_explicit_fact() -> None:
    explicit_facts = [
        {
            "fact_id": "f-eyes",
            "temporal_scope": "permanent",
            "description_ru": "Яркие зелёные глаза",
            "description_en": "Bright green eyes",
        }
    ]
    design_traits = [
        {
            "trait": "eyes",
            "description_ru": "Серые глаза",
            "description_en": "Grey eyes",
            "assumption_ru": "нейтральное допущение",
        },
        {
            "trait": "hair",
            "description_ru": "Тёмные волосы средней длины",
            "description_en": "Medium-length dark hair",
            "assumption_ru": "нейтральное допущение",
        },
    ]
    kept_explicit, kept_design = resolve_appearance_with_design_fallback(
        explicit_facts, design_traits
    )
    assert len(kept_explicit) == 1
    assert kept_explicit[0]["description_en"] == "Bright green eyes"
    # Design trait for "eyes" is suppressed because explicit fact covers eyes; "hair" is retained.
    assert [t["trait"] for t in kept_design] == ["hair"]


def test_build_bounded_design_input_isolates_permanent_facts_and_flags_non_permanent_only() -> None:
    policy = IllustrationPolicy()
    sizer = _real_sizer()
    entity = {"entity_id": "ent-hero", "canonical_name": "Hero", "entity_kind": "person"}
    perm_fact = {
        "fact_id": "f-perm",
        "temporal_scope": "permanent",
        "position_key": format_position_key(0, 0, 0),
        "description_ru": "Шрам на лбу",
        "description_en": "Scar on forehead",
    }
    state_fact = {
        "fact_id": "f-state",
        "temporal_scope": "state_change",
        "position_key": format_position_key(0, 1, 0),
        "description_ru": "Младенец в одеяле",
        "description_en": "Infant wrapped in a blanket",
    }
    epi_fact = {
        "fact_id": "f-epi",
        "temporal_scope": "episodic",
        "position_key": format_position_key(0, 2, 0),
        "description_ru": "Спит на крыльце",
        "description_en": "Sleeping on the doorstep",
    }

    # Permanent facts exist -> ONLY permanent facts sent; non_permanent_context_only is False.
    with_perm = _build_bounded_design_input(
        entity=entity,
        facts=[perm_fact, state_fact, epi_fact],
        sizer=sizer,
        book_id="b1",
        conditions_id=None,
        policy=policy,
    )
    assert with_perm["non_permanent_context_only"] is False
    assert [f["fact_id"] for f in with_perm["facts"]] == ["f-perm"]
    assert with_perm["omitted_episodic_count"] == 2
    assert with_perm["omitted_fact_count"] == 2

    # No permanent facts -> non_permanent_context_only is True and non-permanent facts are passed.
    only_temp = _build_bounded_design_input(
        entity=entity,
        facts=[epi_fact, state_fact],
        sizer=sizer,
        book_id="b1",
        conditions_id=None,
        policy=policy,
    )
    assert only_temp["non_permanent_context_only"] is True
    assert [f["fact_id"] for f in only_temp["facts"]] == ["f-state", "f-epi"]
    assert [f["temporal_scope"] for f in only_temp["facts"]] == ["state_change", "episodic"]

    # No facts at all -> non_permanent_context_only is False.
    empty_facts = _build_bounded_design_input(
        entity=entity,
        facts=(),
        sizer=sizer,
        book_id="b1",
        conditions_id=None,
        policy=policy,
    )
    assert empty_facts["non_permanent_context_only"] is False
    assert empty_facts["facts"] == []


def test_alias_blocking_singularizes_plural_surnames_and_ignores_generic_kin_place_stopwords() -> None:
    # Plural possessive surname matches singular surname ("John Smith" <-> "the Smiths’ son").
    family_mentions = [
        {"mention_key": "m1", "name": "John Smith", "entity_kind": "person"},
        {"mention_key": "m2", "name": "the Smiths’ son", "entity_kind": "person"},
        {"mention_key": "m3", "name": "the Bakers’ son", "entity_kind": "person"},
    ]
    pairs = build_alias_candidate_pairs(family_mentions)
    assert ("m1", "m2") in pairs
    # Generic "son" MUST NOT pair unrelated families!
    assert ("m2", "m3") not in pairs

    # Generic place nouns ("house") MUST NOT pair unrelated locations.
    place_mentions = [
        {"mention_key": "p1", "name": "Old Stone House", "entity_kind": "location"},
        {"mention_key": "p2", "name": "Tall Brick House", "entity_kind": "location"},
    ]
    assert build_alias_candidate_pairs(place_mentions) == ()

    # Names ending in -ss, -us, -is MUST NOT be truncated to broken stems.
    special_endings = [
        {"mention_key": "s1", "name": "Ross", "entity_kind": "person"},
        {"mention_key": "s2", "name": "Ros", "entity_kind": "person"},
        {"mention_key": "s3", "name": "Marcus", "entity_kind": "person"},
        {"mention_key": "s4", "name": "Marcu", "entity_kind": "person"},
        {"mention_key": "s5", "name": "Iris", "entity_kind": "person"},
        {"mention_key": "s6", "name": "Iri", "entity_kind": "person"},
    ]
    assert build_alias_candidate_pairs(special_endings) == ()


def test_compact_repair_summary_idempotent_stage_preamble_and_utf8_cap() -> None:
    from linguary.application.illustration_context import MAX_REPAIR_SUMMARY_BYTES

    default_msg = compact_repair_summary("не хватает selection_reasons_ru")
    assert "ПОЛНЫЙ JSON-ответ целиком" in default_msg
    assert compact_repair_summary(default_msg) == default_msg
    assert compact_repair_summary(default_msg, stage=IllustrationScopeStage.EXTRACT) == default_msg

    extract_msg = compact_repair_summary(
        "пропущен unit-2", stage=IllustrationScopeStage.EXTRACT
    )
    assert "ВСЕХ переданных source_unit_id в covered_source_unit_ids" in extract_msg
    assert compact_repair_summary(extract_msg, stage=IllustrationScopeStage.EXTRACT) == extract_msg

    cand_msg = compact_repair_summary(
        "пропущен b-1", stage=IllustrationScopeStage.CANDIDATES
    )
    assert "ВСЕХ переданных source_unit_id в covered_source_unit_ids" in cand_msg

    long_msg = compact_repair_summary(
        "Длинная ошибка кириллицей " * 80,
        details=("подробность " * 40,),
        stage=IllustrationScopeStage.EXTRACT,
    )
    assert len(long_msg.encode("utf-8")) <= MAX_REPAIR_SUMMARY_BYTES


def test_choose_canonical_name_three_tiers_and_aliases_in_design_input() -> None:
    from linguary.application.illustration_reference import (
        _choose_canonical_name,
        _score_canonical_name,
    )

    assert _score_canonical_name("Professor McGonagall")[0] == 2
    assert _score_canonical_name("McGonagall")[0] == 2
    assert _score_canonical_name("The Leaky Cauldron")[0] == 2
    assert _score_canonical_name("number four, Privet Drive")[0] == 2
    assert _score_canonical_name("Tabby cat")[0] == 1
    assert _score_canonical_name("tabby cat")[0] == 0
    assert _score_canonical_name("the boy")[0] == 0

    # Tier 2 (2 proper words) > Tier 2 (1 proper word) > Tier 1 > Tier 0
    assert (
        _choose_canonical_name(
            [
                {"name": "tabby cat", "position_key": format_position_key(0, 0, 0)},
                {"name": "Tabby cat", "position_key": format_position_key(0, 1, 0)},
                {"name": "McGonagall", "position_key": format_position_key(0, 2, 0)},
                {"name": "Professor McGonagall", "position_key": format_position_key(0, 3, 0)},
            ]
        )
        == "Professor McGonagall"
    )
    assert (
        _choose_canonical_name(
            [
                {"name": "tabby cat", "position_key": format_position_key(0, 0, 0)},
                {"name": "McGonagall", "position_key": format_position_key(0, 2, 0)},
            ]
        )
        == "McGonagall"
    )
    assert (
        _choose_canonical_name(
            [
                {"name": "tabby cat", "position_key": format_position_key(0, 0, 0)},
                {"name": "Tabby cat", "position_key": format_position_key(0, 1, 0)},
            ]
        )
        == "Tabby cat"
    )
    assert (
        _choose_canonical_name(
            [
                {"name": "a dark cupboard", "position_key": format_position_key(0, 0, 0)},
                {"name": "number four, Privet Drive", "position_key": format_position_key(0, 1, 0)},
            ]
        )
        == "number four, Privet Drive"
    )
    assert (
        _choose_canonical_name(
            [
                {"name": "the pub", "position_key": format_position_key(0, 0, 0)},
                {"name": "The Leaky Cauldron", "position_key": format_position_key(0, 1, 0)},
            ]
        )
        == "The Leaky Cauldron"
    )

    # _build_bounded_design_input includes up to 8 aliases
    bounded = _build_bounded_design_input(
        entity={
            "entity_id": "ent-1",
            "canonical_name": "Professor McGonagall",
            "aliases": [f"alias-{i}" for i in range(12)],
            "entity_kind": "person",
        },
        facts=(),
        sizer=_real_sizer(),
        book_id="b1",
        conditions_id=None,
        policy=IllustrationPolicy(),
    )
    assert bounded["aliases"] == [f"alias-{i}" for i in range(8)]


async def test_entity_persists_canonical_name_alias_first_positions_and_section_orders(
    tmp_path: Path,
) -> None:
    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_EXTRACT:
            mentions = []
            covered = []
            for u in data["units"]:
                covered.append(u["source_unit_id"])
                for blk in u["blocks"]:
                    if "tabby cat" in blk["text"]:
                        mentions.append(
                            IllustrationMentionDTO(
                                mention_id=f"m-{blk['block_id']}",
                                name="tabby cat",
                                entity_kind="person",
                                alias_of_hint="Professor McGonagall",
                                quote_block_id=blk["block_id"],
                                quote="tabby cat",
                            )
                        )
                    elif "Professor McGonagall" in blk["text"]:
                        mentions.append(
                            IllustrationMentionDTO(
                                mention_id=f"m-{blk['block_id']}",
                                name="Professor McGonagall",
                                entity_kind="person",
                                alias_of_hint=None,
                                quote_block_id=blk["block_id"],
                                quote="Professor McGonagall",
                            )
                        )
            return IllustrationExtractResponse(
                mentions=tuple(mentions),
                facts=(),
                covered_source_unit_ids=tuple(covered),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_ALIAS:
            return IllustrationAliasResponse(
                decisions=tuple(
                    IllustrationAliasDecisionDTO(
                        pair_id=p["pair_id"], verdict="same_entity", evidence_fact_ids=()
                    )
                    for p in data["pairs"]
                )
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_DESIGN:
            return IllustrationDesignResponse(entity_id=data["entity_id"], traits=())
        raise AssertionError(req.kind)

    harness = _seed_book_with_narrative(
        tmp_path,
        sections_blocks=[
            ("sec-0", "Ch 1", [("b-0", "A tabby cat sat on the brick wall.")]),
            ("sec-1", "Ch 2", [("b-1", "Professor McGonagall stepped forward.")]),
        ],
        handler=_handler,
        book_id="book-canon-meta",
    )
    ref, gen_id = _create_and_plan(harness)
    await _drain_ref_tasks(harness, ref)

    with harness.uow.transaction() as tx:
        entities, total = tx.illustrations.query_registry_items(
            gen_id, kind=IllustrationRegistryKind.ENTITY
        )
        assert total == 1
        pay = entities[0].raw_payload()
        assert pay["canonical_name"] == "Professor McGonagall"
        assert pay["aliases"] == ["tabby cat"]
        assert set(pay["alias_first_positions"].keys()) == {"tabby cat", "Professor McGonagall"}
        assert pay["alias_first_positions"]["tabby cat"] < pay["alias_first_positions"]["Professor McGonagall"]
        assert pay["section_orders"] == [0, 1]


def test_validate_design_strips_trait_prefix_before_checks_and_rejects_empty() -> None:
    from datetime import UTC, datetime

    from linguary.domain.illustrations import IllustrationScope

    ref = IllustrationReference.__new__(IllustrationReference)
    now = datetime(2026, 11, 26, tzinfo=UTC)
    design_scope = IllustrationScope(
        id="sc-des-pref",
        generation_id="gen-1",
        parent_scope_id=None,
        stage=IllustrationScopeStage.DESIGN,
        scope_key="design:ent-1",
        section_id=None,
        section_order=None,
        section_title=None,
        task_id=None,
        input_sha256="a" * 64,
        state=IllustrationScopeState.RUNNING,
        reason_code=None,
        attempt_id=None,
        output_sha256=None,
        payload=None,
        summary={"entity_id": "ent-1", "entity_kind": "person", "style": "monochrome_ink"},
        created_at=now,
        updated_at=now,
    )

    ok_res = ref._validate_design(
        design_scope,
        IllustrationDesignResponse(
            entity_id="ent-1",
            traits=(
                IllustrationDesignTraitDTO(
                    trait="palette_value",
                    description_ru="palette_value: Контрастная тушь",
                    description_en="Palette_Value: high-contrast black ink",
                    assumption_ru="",
                ),
            ),
        ),
    )
    assert ok_res.ok is True and ok_res.payload is not None
    assert ok_res.payload["traits"][0]["description_en"] == "high-contrast black ink"
    assert ok_res.payload["traits"][0]["description_ru"] == "Контрастная тушь"

    empty_after_strip = ref._validate_design(
        design_scope,
        IllustrationDesignResponse(
            entity_id="ent-1",
            traits=(
                IllustrationDesignTraitDTO(
                    trait="palette_value",
                    description_ru="Контрастная тушь",
                    description_en="palette_value:   ",
                    assumption_ru="",
                ),
            ),
        ),
    )
    assert empty_after_strip.ok is False

    color_after_strip = ref._validate_design(
        design_scope,
        IllustrationDesignResponse(
            entity_id="ent-1",
            traits=(
                IllustrationDesignTraitDTO(
                    trait="palette_value",
                    description_ru="Контрастная тушь",
                    description_en="palette_value: vivid green ink",
                    assumption_ru="",
                ),
            ),
        ),
    )
    assert color_after_strip.ok is False
    assert "monochrome_ink" in color_after_strip.repair_reason

    cyr_after_strip = ref._validate_design(
        design_scope,
        IllustrationDesignResponse(
            entity_id="ent-1",
            traits=(
                IllustrationDesignTraitDTO(
                    trait="build",
                    description_ru="Стройный силуэт",
                    description_en="build: Стройный силуэт",
                    assumption_ru="",
                ),
            ),
        ),
    )
    assert cyr_after_strip.ok is False
    assert "кириллиц" in cyr_after_strip.repair_reason


def test_resolve_appearance_state_change_suppresses_anatomical_traits_and_short_token_echo() -> None:
    design = [
        {"trait": "build", "description_en": "Small tabby cat silhouette", "description_ru": "Силуэт кошки"},
        {"trait": "distinctive_mark", "description_en": "Square markings around eyes", "description_ru": "Квадратные отметины"},
        {"trait": "hair", "description_en": "Striped fur", "description_ru": "Полосатая шерсть"},
        {"trait": "eyes", "description_en": "Alert feline gaze", "description_ru": "Настороженный взгляд"},
        {"trait": "age_appearance", "description_en": "Ageless animal", "description_ru": "Взрослая кошка"},
        {"trait": "palette_value", "description_en": "Crisp black ink contrast", "description_ru": "Чёткий контраст туши"},
    ]
    # 1. state_change in explicit_facts suppresses all anatomical traits, preserves palette_value
    _, kept_explicit_sc = resolve_appearance_with_design_fallback(
        [{"temporal_scope": "state_change", "description_en": "Severe woman in a cloak", "description_ru": "Строгая женщина в плаще"}],
        design,
    )
    assert [t["trait"] for t in kept_explicit_sc] == ["palette_value"]

    # 2. state_change in excluded_facts ALSO suppresses all anatomical traits, preserves palette_value
    _, kept_excluded_sc = resolve_appearance_with_design_fallback(
        [{"temporal_scope": "episodic", "description_en": "Standing on the pavement", "description_ru": "Стоит на тротуаре"}],
        design,
        excluded_facts=[
            {"temporal_scope": "state_change", "description_en": "Transformed from a cat", "description_ru": "Превратилась из кошки"}
        ],
    )
    assert [t["trait"] for t in kept_excluded_sc] == ["palette_value"]

    # 3. 3-character lexical echo tokens ("cat", "fur", "tail") suppress matching design trait
    _, kept_short_echo = resolve_appearance_with_design_fallback(
        [],
        [
            {"trait": "attire_default", "description_en": "Cat fur tail collar", "description_ru": "Воротник с мехом кошки"},
            {"trait": "palette_value", "description_en": "Crisp ink linework", "description_ru": "Чёткая линия туши"},
        ],
        excluded_facts=[
            {"temporal_scope": "episodic", "description_en": "Small cat with soft fur and tail", "description_ru": "Кошка с мехом и хвостом"}
        ],
    )
    assert [t["trait"] for t in kept_short_echo] == ["palette_value"]


def test_pack_extract_section_units_respects_scene_block_and_byte_caps() -> None:
    """WP3: _pack_extract_section_units enforces max_units, max_blocks, byte cap, and singleton overflow."""
    from types import SimpleNamespace

    from linguary.application.illustration_reference import _pack_extract_section_units

    class _FakeSizer:
        def measure(self, req: LLMRequest[Any]) -> Any:
            size = len(req.input_text.encode("utf-8"))
            return SimpleNamespace(size_bytes=size)

    def _mk_unit(uid: str, n_blocks: int, text: str = "short") -> dict[str, Any]:
        return {
            "source_unit_id": uid,
            "section_id": "sec-1",
            "scene_id": uid,
            "blocks": [{"block_id": f"{uid}-b{i}", "block_order": i, "text": text} for i in range(n_blocks)],
        }

    def _build(units: Any) -> LLMRequest[Any]:
        return LLMRequest(
            kind=LLMResponseKind.ILLUSTRATION_EXTRACT,
            instructions="test",
            input_text=json.dumps({"units": list(units)}),
        )

    sizer = _FakeSizer()

    # 1. Scene cap = 2, block cap = 16:
    # u1(10 blocks) + u2(10 blocks) = 20 > 16 -> split; u2(10) + u3(5) = 15 <= 16 -> packed; u4(18) -> singleton; u5(2)+u6(2)+u7(2) -> (u5,u6), (u7)
    units = [
        _mk_unit("u1", 10),
        _mk_unit("u2", 10),
        _mk_unit("u3", 5),
        _mk_unit("u4", 18),
        _mk_unit("u5", 2),
        _mk_unit("u6", 2),
        _mk_unit("u7", 2),
    ]
    pages = _pack_extract_section_units(
        units,
        build_request=_build,
        sizer=sizer,  # type: ignore[arg-type]
        target_bytes=12_288,
        max_units=2,
        max_blocks=16,
    )
    assert [[u["source_unit_id"] for u in p] for p in pages] == [
        ["u1"],
        ["u2", "u3"],
        ["u4"],
        ["u5", "u6"],
        ["u7"],
    ]

    # 2. Byte cap splits even when unit/block caps allow packing.
    big_units = [_mk_unit("b1", 1, "x" * 7000), _mk_unit("b2", 1, "y" * 7000)]
    byte_pages = _pack_extract_section_units(
        big_units,
        build_request=_build,
        sizer=sizer,  # type: ignore[arg-type]
        target_bytes=12_288,
        max_units=2,
        max_blocks=16,
    )
    assert [[u["source_unit_id"] for u in p] for p in byte_pages] == [["b1"], ["b2"]]


async def test_plan_scopes_packs_extract_strictly_per_section(tmp_path: Path) -> None:
    """WP3: EXTRACT scopes never span multiple chapters/sections; each has concrete section_id."""
    harness = _seed_book_with_narrative(
        tmp_path,
        sections_blocks=[
            ("sec-1", "Ch 1", [("b-1", "Alice walked down the lane.")]),
            ("sec-2", "Ch 2", [("b-2", "Bob waited by the gate.")]),
        ],
        handler=lambda _req, _n: None,
    )
    _ref, gen_id = _create_and_plan(harness)
    with harness.uow.read_transaction() as tx:
        scopes = tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.EXTRACT)
    assert [s.scope_key for s in scopes] == ["extract:p0000", "extract:p0001"]
    assert [s.section_id for s in scopes] == ["sec-1", "sec-2"]


async def test_composite_extract_auto_splits_after_validation_repairs_exhausted(
    tmp_path: Path,
) -> None:
    """WP3: composite EXTRACT scope auto-splits after exhausting max_repair_turns on validation errors."""
    seen_unit_counts: list[int] = []

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_EXTRACT:
            units = data["units"]
            total_blocks = sum(len(u["blocks"]) for u in units)
            seen_unit_counts.append(total_blocks)
            if total_blocks > 1:
                # Emit unknown subject_mention_id on composite packet so all repair turns fail.
                first_blk = units[0]["blocks"][0]["block_id"]
                return IllustrationExtractResponse(
                    mentions=(),
                    facts=(
                        IllustrationFactDTO(
                            fact_id="f-bad",
                            subject_mention_id="missing-mention",
                            temporal_scope="permanent",
                            back_applicable=True,
                            reveals_future_spoiler=False,
                            description_ru="Описание",
                            description_en="Description",
                            quote_block_id=first_blk,
                            quote=units[0]["blocks"][0]["text"][:10],
                        ),
                    ),
                    covered_source_unit_ids=tuple(u["source_unit_id"] for u in units),
                    complete=True,
                )
            blk = units[0]["blocks"][0]
            return IllustrationExtractResponse(
                mentions=(
                    IllustrationMentionDTO(
                        mention_id="m1",
                        name="Alice",
                        entity_kind="person",
                        alias_of_hint=None,
                        quote_block_id=blk["block_id"],
                        quote="Alice",
                    ),
                ),
                facts=(),
                covered_source_unit_ids=(units[0]["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_ALIAS:
            return IllustrationAliasResponse(
                decisions=tuple(
                    IllustrationAliasDecisionDTO(
                        pair_id=p["pair_id"], verdict="same_entity", evidence_fact_ids=()
                    )
                    for p in data.get("pairs", ())
                )
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_DESIGN:
            return IllustrationDesignResponse(
                entity_id=data["entity_id"],
                traits=(
                    IllustrationDesignTraitDTO(
                        trait="palette_value",
                        description_ru="Контрастная штриховка",
                        description_en="Crisp ink hatching",
                        assumption_ru="нейтральный выбор",
                    ),
                ),
            )
        raise AssertionError(f"unexpected {req.kind}")

    harness = _seed_book_with_narrative(
        tmp_path,
        sections_blocks=[
            (
                "sec-1",
                "Ch 1",
                [
                    ("b-1", "Alice stood at the wooden door."),
                    ("b-2", "Alice looked across the garden."),
                ],
            )
        ],
        handler=_handler,
        policy=IllustrationPolicy(max_repair_turns=1, max_calls_per_scope_activation=3),
    )
    ref, gen_id = _create_and_plan(harness)
    await _drain_ref_tasks(harness, ref)

    with harness.uow.read_transaction() as tx:
        gen = tx.illustrations.get_generation(harness.book_id, gen_id)
        extract_scopes = tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.EXTRACT)
    assert gen is not None and gen.registry_sha256 is not None
    # Parent scope split into 2 single-block children (#L/#R) and both succeeded!
    states_by_key = {s.scope_key: s.state for s in extract_scopes}
    assert states_by_key["extract:p0000"] is IllustrationScopeState.SPLIT
    assert states_by_key["extract:p0000.d1.0"] is IllustrationScopeState.ACCEPTED
    assert states_by_key["extract:p0000.d1.1"] is IllustrationScopeState.ACCEPTED
    # Parent attempted initial + 1 repair (2 calls with 2 blocks), then children ran with 1 block each.
    assert seen_unit_counts == [2, 2, 1, 1]


async def test_composite_extract_auto_splits_after_format_error_repairs_exhausted(
    tmp_path: Path,
) -> None:
    """WP3: composite EXTRACT scope auto-splits after exhausting max_repair_turns on LLMResponseFormatError."""
    from linguary.application.ports.llm import LLMResponseFormatError

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_EXTRACT:
            units = data["units"]
            total_blocks = sum(len(u["blocks"]) for u in units)
            if total_blocks > 1:
                raise LLMResponseFormatError("invalid json schema")
            blk = units[0]["blocks"][0]
            return IllustrationExtractResponse(
                mentions=(
                    IllustrationMentionDTO(
                        mention_id="m1",
                        name="Alice",
                        entity_kind="person",
                        alias_of_hint=None,
                        quote_block_id=blk["block_id"],
                        quote="Alice",
                    ),
                ),
                facts=(),
                covered_source_unit_ids=(units[0]["source_unit_id"],),
                complete=True,
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_ALIAS:
            return IllustrationAliasResponse(
                decisions=tuple(
                    IllustrationAliasDecisionDTO(
                        pair_id=p["pair_id"], verdict="same_entity", evidence_fact_ids=()
                    )
                    for p in data.get("pairs", ())
                )
            )
        if req.kind is LLMResponseKind.ILLUSTRATION_DESIGN:
            return IllustrationDesignResponse(
                entity_id=data["entity_id"],
                traits=(
                    IllustrationDesignTraitDTO(
                        trait="palette_value",
                        description_ru="Контрастная штриховка",
                        description_en="Crisp ink hatching",
                        assumption_ru="нейтральный выбор",
                    ),
                ),
            )
        raise AssertionError(f"unexpected {req.kind}")

    harness = _seed_book_with_narrative(
        tmp_path,
        sections_blocks=[
            (
                "sec-1",
                "Ch 1",
                [
                    ("b-1", "Alice opened the iron gate."),
                    ("b-2", "Alice stepped onto the path."),
                ],
            )
        ],
        handler=_handler,
        policy=IllustrationPolicy(max_repair_turns=1, max_calls_per_scope_activation=3),
    )
    ref, gen_id = _create_and_plan(harness)
    await _drain_ref_tasks(harness, ref)

    with harness.uow.read_transaction() as tx:
        gen = tx.illustrations.get_generation(harness.book_id, gen_id)
        extract_scopes = tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.EXTRACT)
    assert gen is not None and gen.registry_sha256 is not None
    states_by_key = {s.scope_key: s.state for s in extract_scopes}
    assert states_by_key["extract:p0000"] is IllustrationScopeState.SPLIT
    assert states_by_key["extract:p0000.d1.0"] is IllustrationScopeState.ACCEPTED
    assert states_by_key["extract:p0000.d1.1"] is IllustrationScopeState.ACCEPTED


async def test_composite_extract_respects_max_split_depth_on_repeated_validation_error(
    tmp_path: Path,
) -> None:
    """WP3: auto-split stops cleanly with SPLIT_DEPTH_EXHAUSTED once max_split_depth is reached."""
    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        data = json.loads(req.input_text)
        units = data["units"]
        first_blk = units[0]["blocks"][0]["block_id"]
        return IllustrationExtractResponse(
            mentions=(),
            facts=(
                IllustrationFactDTO(
                    fact_id="f-bad",
                    subject_mention_id="missing-mention",
                    temporal_scope="permanent",
                    back_applicable=True,
                    reveals_future_spoiler=False,
                    description_ru="Описание",
                    description_en="Description",
                    quote_block_id=first_blk,
                    quote=units[0]["blocks"][0]["text"][:10],
                ),
            ),
            covered_source_unit_ids=tuple(u["source_unit_id"] for u in units),
            complete=True,
        )

    harness = _seed_book_with_narrative(
        tmp_path,
        sections_blocks=[
            (
                "sec-1",
                "Ch 1",
                [
                    ("b-1", "Alice opened the wooden door."),
                    ("b-2", "Alice walked into the hall."),
                    ("b-3", "Alice looked up the stairs."),
                    ("b-4", "Alice heard a soft whisper."),
                ],
            )
        ],
        handler=_handler,
        policy=IllustrationPolicy(max_repair_turns=0, max_split_depth=1),
    )
    ref, gen_id = _create_and_plan(harness)
    await _drain_ref_tasks(harness, ref)

    with harness.uow.read_transaction() as tx:
        scopes = tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.EXTRACT)
    by_key = {s.scope_key: s for s in scopes}
    assert by_key["extract:p0000"].state is IllustrationScopeState.SPLIT
    assert by_key["extract:p0000.d1.0"].state is IllustrationScopeState.FAILED
    assert by_key["extract:p0000.d1.0"].reason_code == IllustrationReasonCode.SPLIT_DEPTH_EXHAUSTED.value
    assert by_key["extract:p0000.d1.1"].state is IllustrationScopeState.FAILED
    assert by_key["extract:p0000.d1.1"].reason_code == IllustrationReasonCode.SPLIT_DEPTH_EXHAUSTED.value

