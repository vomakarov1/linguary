"""Tests for adaptive unit token budget and `block_processing` token cap wiring."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from linguary.application.block_processing import TranslationCall
from linguary.application.book_conditions import ConditionsError
from linguary.application.llm_dto import TranslationResultResponse
from linguary.application.ports.llm import LLMRequest, LLMResponse, LLMResponseFormatError
from linguary.application.recovery import RecoveryError, RecoverySplit
from linguary.application.retry_policy import ResultValidationError, RetryPolicy
from linguary.application.token_budget import (
    REASONING_EFFORT_MIN_TOKENS,
    REASONING_EFFORT_RESERVE_PCT,
    recommended_unit_max_completion_tokens,
)
from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    BlockType,
    DocumentBlock,
    LLMCallStage,
    SourceReference,
    SupportStatus,
    TranslationUnit,
    TreatmentPlan,
    TreatmentPlanItem,
)
from linguary.domain.llm_log import CallContext, Violation


def test_reasoning_effort_reserve_pct_matches_expected_levels() -> None:
    assert REASONING_EFFORT_RESERVE_PCT == {
        "none": 0,
        "minimal": 15,
        "low": 35,
        "medium": 55,
        "high": 75,
    }
    assert REASONING_EFFORT_MIN_TOKENS == {
        "none": 0,
        "minimal": 1024,
        "low": 4096,
        "medium": 8192,
        "high": 12288,
    }


def test_recommended_unit_max_completion_tokens_without_max_output_tokens() -> None:
    assert (
        recommended_unit_max_completion_tokens(
            0,
            max_output_tokens=None,
            reasoning_effort=None,
            reasoning_max_tokens=None,
        )
        == 2048
    )
    assert (
        recommended_unit_max_completion_tokens(
            4000,
            max_output_tokens=None,
            reasoning_effort=None,
            reasoning_max_tokens=None,
        )
        == 6424
    )


def test_recommended_unit_max_completion_tokens_respects_max_output_tokens_cap() -> None:
    assert (
        recommended_unit_max_completion_tokens(
            1000,
            max_output_tokens=32768,
            reasoning_effort=None,
            reasoning_max_tokens=None,
        )
        == 2374
    )
    assert (
        recommended_unit_max_completion_tokens(
            100_000,
            max_output_tokens=32768,
            reasoning_effort=None,
            reasoning_max_tokens=None,
        )
        == 32768
    )


def test_recommended_unit_max_completion_tokens_reasoning_effort_precedes_reasoning_max_tokens() -> None:
    low_result = recommended_unit_max_completion_tokens(
        0,
        max_output_tokens=32768,
        reasoning_effort="low",
        reasoning_max_tokens=12000,
    )
    assert low_result == 2048 + 4096
    assert low_result == 6144
    assert low_result != 2048 + 12000

    high_result = recommended_unit_max_completion_tokens(
        0,
        max_output_tokens=32768,
        reasoning_effort=" HIGH ",
        reasoning_max_tokens=20000,
    )
    assert high_result == 2048 + 12288
    assert high_result == 14336
    assert high_result != 2048 + 20000

    none_result = recommended_unit_max_completion_tokens(
        0,
        max_output_tokens=32768,
        reasoning_effort="none",
        reasoning_max_tokens=4096,
    )
    assert none_result == 2048


def test_recommended_unit_max_completion_tokens_reasoning_max_tokens_and_clamping() -> None:
    unbounded = recommended_unit_max_completion_tokens(
        0,
        max_output_tokens=None,
        reasoning_effort=None,
        reasoning_max_tokens=4096,
    )
    assert unbounded == 6144
    assert unbounded > 4096

    bounded = recommended_unit_max_completion_tokens(
        0,
        max_output_tokens=32768,
        reasoning_effort=None,
        reasoning_max_tokens=4096,
    )
    assert bounded == 6144
    assert bounded > 4096

    just_above = recommended_unit_max_completion_tokens(
        0,
        max_output_tokens=4097,
        reasoning_effort=None,
        reasoning_max_tokens=4096,
    )
    assert just_above == 4097

    assert (
        recommended_unit_max_completion_tokens(
            0,
            max_output_tokens=4096,
            reasoning_effort=None,
            reasoning_max_tokens=4096,
        )
        == 4096
    )
    assert (
        recommended_unit_max_completion_tokens(
            0,
            max_output_tokens=2048,
            reasoning_effort=None,
            reasoning_max_tokens=4096,
        )
        == 2048
    )


def _sample_unit(text: str = "Hello world.") -> tuple[DocumentBlock, TranslationUnit]:
    block = DocumentBlock.create(
        document_id="book-1",
        section_id="sec-1",
        order=0,
        type=BlockType.PARAGRAPH,
        text=text,
        source_reference=SourceReference(resource_path="ch1.xhtml", structural_path="/body/p[1]"),
        support_status=SupportStatus.SUPPORTED,
    )
    plan = TreatmentPlan(
        section_id="sec-1",
        items=(TreatmentPlanItem(block_id=block.id, treatment=BlockTreatment.TRANSLATE),),
    )
    unit = TranslationUnit.of_blocks(id="unit-1", blocks=(block,), plan=plan)
    return block, unit


def _translation_context(request: LLMRequest[TranslationResultResponse]) -> SimpleNamespace:
    return SimpleNamespace(
        request=request,
        content_plan=None,
        before_you_read=(),
        explained_expressions=(),
        explained_entries=(),
        expectation=None,
        has_optional_neighbours=False,
        to_block_ids=lambda value: value,
        to_aliases=lambda text: text,
    )


def _make_call(fake_uow: object, block: DocumentBlock, seen_requests: list[LLMRequest[TranslationResultResponse]]) -> TranslationCall:
    async def generate(req: LLMRequest[TranslationResultResponse]) -> LLMResponse[TranslationResultResponse]:
        seen_requests.append(req)
        return LLMResponse(
            payload=TranslationResultResponse(blocks=()),
            provider="fake",
            model="m",
            call_context=req.call_context,
        )

    return TranslationCall(
        unit_of_work=fake_uow,  # type: ignore[arg-type]
        llm_client=SimpleNamespace(generate=generate),  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_attempts=1),
        validator=SimpleNamespace(
            validate=lambda payload, **kwargs: (
                BlockProcessingResult(block.id, BlockTreatment.TRANSLATE, "Привет мир."),
            )
        ),  # type: ignore[arg-type]
        sleep=lambda _: None,  # type: ignore[arg-type]
    )


async def test_block_processing_leaves_max_completion_tokens_none_when_conditions_missing() -> None:
    block, unit = _sample_unit("Hello world.")
    seen_requests: list[LLMRequest[TranslationResultResponse]] = []
    fake_tx = SimpleNamespace(
        runs=SimpleNamespace(active_for_book=lambda book_id: None),
        work=SimpleNamespace(
            get_binding=lambda kind, key: None,
            get_conditions=lambda cid: None,
        ),
    )
    fake_uow = SimpleNamespace(transaction=lambda: nullcontext(fake_tx))
    call = _make_call(fake_uow, block, seen_requests)

    request = LLMRequest.for_translation(instructions="Translate.", input_text="b1: Hello world.")
    await call.request(
        _translation_context(request),  # type: ignore[arg-type]
        unit=unit,
        target_language="ru",
        note_ids=lambda: "n1",
    )
    assert len(seen_requests) == 1
    assert seen_requests[0].max_completion_tokens is None

    explicit_none_cond = CallContext(
        conditions_id=str(uuid4()),
        stage=LLMCallStage.TRANSLATION_UNIT.value,
        scope_id=unit.id,
        ordered_block_ids=(block.id,),
        attempt_id=str(uuid4()),
        generation=0,
        assigned_model="m",
        prompt_hash="a" * 64,
        root_id=None,
        policy_version=None,
    )
    with pytest.raises(ConditionsError):
        await call.request(
            _translation_context(replace(request, call_context=explicit_none_cond)),  # type: ignore[arg-type]
            unit=unit,
            target_language="ru",
            note_ids=lambda: "n2",
        )
    assert len(seen_requests) == 2
    assert seen_requests[1].max_completion_tokens is None


async def test_block_processing_sets_adaptive_limit_after_loading_conditions_in_non_recovery_and_recovery() -> None:
    block, unit = _sample_unit("A" * 2000)
    seen_requests: list[LLMRequest[TranslationResultResponse]] = []
    cond_id = str(uuid4())
    root_id = str(uuid4())
    rec_attempt_id = str(uuid4())
    conditions = SimpleNamespace(
        id=cond_id,
        payload={
            "models": {"processing": "pinned-model"},
            "versions": {"validators": 1},
            "llm_parameters": {
                "max_output_tokens": 32768,
                "reasoning_effort": "low",
                "reasoning_max_tokens": None,
            },
        },
    )
    root = SimpleNamespace(
        id=root_id,
        conditions_id=cond_id,
        policy_version=CURRENT_TRANSLATION_POLICY,
        generation=1,
    )
    recovery_context = CallContext(
        conditions_id=cond_id,
        stage=LLMCallStage.TRANSLATION_UNIT.value,
        scope_id="scope-1",
        ordered_block_ids=(block.id,),
        attempt_id=rec_attempt_id,
        generation=0,
        assigned_model="pinned-model",
        prompt_hash="a" * 64,
        root_id=root_id,
        policy_version=CURRENT_TRANSLATION_POLICY,
    )
    fake_tx = SimpleNamespace(
        runs=SimpleNamespace(active_for_book=lambda book_id: None),
        work=SimpleNamespace(
            get_binding=lambda kind, key: None,
            get_conditions=lambda cid: conditions if cid == cond_id else None,
            get_root=lambda rid: root if rid == root_id else None,
            begin_recovery_attempt=lambda *args, **kwargs: recovery_context,
            owns_recovery_attempt=lambda aid: aid == rec_attempt_id,
        ),
    )
    fake_uow = SimpleNamespace(
        transaction=lambda: nullcontext(fake_tx),
        _diagnostics_service=SimpleNamespace(
            ensure_available=lambda: None,
            fail=lambda: None,
        ),
    )
    call = _make_call(fake_uow, block, seen_requests)

    expected_limit = recommended_unit_max_completion_tokens(
        2000,
        max_output_tokens=32768,
        reasoning_effort="low",
        reasoning_max_tokens=None,
    )

    # Non-recovery path (#L201)
    non_rec_context = CallContext(
        conditions_id=cond_id,
        stage=LLMCallStage.TRANSLATION_UNIT.value,
        scope_id=unit.id,
        ordered_block_ids=(block.id,),
        attempt_id=str(uuid4()),
        generation=0,
        assigned_model="pinned-model",
        prompt_hash="a" * 64,
    )
    base_request = LLMRequest.for_translation(instructions="Translate.", input_text="b1: " + "A" * 2000)
    await call.request(
        _translation_context(replace(base_request, call_context=non_rec_context)),  # type: ignore[arg-type]
        unit=unit,
        target_language="ru",
        note_ids=lambda: "n1",
    )
    assert seen_requests[-1].max_completion_tokens == expected_limit

    # Recovery path (#L236)
    await call.request(
        _translation_context(base_request),  # type: ignore[arg-type]
        unit=unit,
        target_language="ru",
        note_ids=lambda: "n2",
        recovery=(root_id, "scope-1"),
    )
    assert seen_requests[-1].max_completion_tokens == expected_limit

    # Explicit max_completion_tokens on request is never overwritten
    await call.request(
        _translation_context(replace(base_request, max_completion_tokens=1500)),  # type: ignore[arg-type]
        unit=unit,
        target_language="ru",
        note_ids=lambda: "n3",
        recovery=(root_id, "scope-1"),
    )
    assert seen_requests[-1].max_completion_tokens == 1500

    # unit=None leaves max_completion_tokens=None even when loaded_conditions is present
    await call.request(
        _translation_context(replace(base_request, call_context=non_rec_context)),  # type: ignore[arg-type]
        unit=None,  # type: ignore[arg-type]
        target_language="ru",
        note_ids=lambda: "n4",
    )
    assert seen_requests[-1].max_completion_tokens is None


def _multi_block_unit(count: int = 2) -> tuple[tuple[DocumentBlock, ...], TranslationUnit]:
    blocks = tuple(
        DocumentBlock.create(
            document_id="book-1",
            section_id="sec-1",
            order=i,
            type=BlockType.PARAGRAPH,
            text=f"Paragraph {i}.",
            source_reference=SourceReference(
                resource_path="ch1.xhtml", structural_path=f"/body/p[{i + 1}]"
            ),
            support_status=SupportStatus.SUPPORTED,
        )
        for i in range(count)
    )
    plan = TreatmentPlan(
        section_id="sec-1",
        items=tuple(
            TreatmentPlanItem(block_id=b.id, treatment=BlockTreatment.TRANSLATE) for b in blocks
        ),
    )
    return blocks, TranslationUnit.of_blocks(id="unit-multi", blocks=blocks, plan=plan)


async def test_block_processing_escalates_tokens_on_provider_truncated_in_recovery_single_and_multi_block() -> None:
    cond_id = str(uuid4())
    root_id = str(uuid4())
    conditions = SimpleNamespace(
        id=cond_id,
        payload={
            "models": {"processing": "deepseek-v4"},
            "versions": {"validators": 1},
            "llm_parameters": {
                "max_output_tokens": 32768,
                "reasoning_effort": "low",
                "reasoning_max_tokens": None,
            },
        },
    )

    # 1. Single-block recovery: first call truncated at 6144 -> escalates to 32768 and succeeds.
    block, unit = _sample_unit("Hello world.")
    seen_limits: list[int | None] = []
    calls_used = {"n": 0}

    def _make_uow(
        *,
        max_calls: int = 10,
        max_scope_calls: int = 4,
        scope_count_fn=None,
        superseded_after_first: bool = False,
        interrupted_after_first: bool = False,
    ):
        root_state = SimpleNamespace(
            id=root_id,
            conditions_id=cond_id,
            policy_version=CURRENT_TRANSLATION_POLICY,
            generation=1,
            calls_used=0,
            extra_calls=0,
            superseded_by=None,
            status="active",
            reason=None,
            manifest={"policy": {"max_calls": max_calls, "max_scope_calls": max_scope_calls}},
        )

        def begin_attempt(*args, **kwargs):
            calls_used["n"] += 1
            root_state.calls_used = calls_used["n"]
            return CallContext(
                conditions_id=cond_id,
                stage=LLMCallStage.TRANSLATION_UNIT.value,
                scope_id="scope-1",
                ordered_block_ids=tuple(item.block.id for item in unit.items),
                attempt_id=str(uuid4()),
                generation=0,
                assigned_model="deepseek-v4",
                prompt_hash="a" * 64,
                root_id=root_id,
                policy_version=CURRENT_TRANSLATION_POLICY,
            )

        def record_failure(*args, **kwargs):
            if superseded_after_first:
                root_state.superseded_by = "new-root"
            if interrupted_after_first:
                root_state.generation = 2
            return True  # would otherwise trigger RecoverySplit!

        work_ns = SimpleNamespace(
            get_binding=lambda kind, key: None,
            get_conditions=lambda cid: conditions if cid == cond_id else None,
            get_root=lambda rid: root_state if rid == root_id else None,
            begin_recovery_attempt=begin_attempt,
            owns_recovery_attempt=lambda aid: True,
            record_recovery_failure=record_failure,
            stop_root=lambda *a, **k: None,
            set_root_status=lambda *a, **k: None,
        )
        if scope_count_fn is not None:
            work_ns.scope_attempt_count = scope_count_fn
        fake_tx = SimpleNamespace(
            runs=SimpleNamespace(active_for_book=lambda book_id: None),
            work=work_ns,
        )
        return SimpleNamespace(
            transaction=lambda: nullcontext(fake_tx),
            _diagnostics_service=SimpleNamespace(
                ensure_available=lambda: None,
                fail=lambda: None,
                finalize_attempt=lambda *a, **k: None,
            ),
        )

    async def gen_recover_once(req: LLMRequest[TranslationResultResponse]) -> LLMResponse[TranslationResultResponse]:
        seen_limits.append(req.max_completion_tokens)
        if len(seen_limits) == 1:
            raise LLMResponseFormatError(
                "truncated",
                truncated=True,
                violation=Violation(
                    rule_code="provider_truncated",
                    stage=LLMCallStage.TRANSLATION_UNIT.value,
                    scope="global",
                    safe_message="truncated",
                ),
            )
        return LLMResponse(
            payload=TranslationResultResponse(blocks=()),
            provider="fake",
            model="deepseek-v4",
            call_context=req.call_context,
        )

    call1 = TranslationCall(
        unit_of_work=_make_uow(),  # type: ignore[arg-type]
        llm_client=SimpleNamespace(generate=gen_recover_once),  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_attempts=1),
        validator=SimpleNamespace(
            validate=lambda payload, **kwargs: (
                BlockProcessingResult(block.id, BlockTreatment.TRANSLATE, "Привет."),
            )
        ),  # type: ignore[arg-type]
        sleep=lambda _: None,  # type: ignore[arg-type]
    )
    req = LLMRequest.for_translation(instructions="Translate.", input_text="b1: Hello world.")
    res = await call1.request(
        _translation_context(req),  # type: ignore[arg-type]
        unit=unit,
        target_language="ru",
        note_ids=lambda: "n1",
        recovery=(root_id, "scope-1"),
    )
    assert res.attempts == 2
    assert seen_limits == [6144, 32768]

    # 2. Multi-block recovery: escalates to 32768 first WITHOUT split, then splits if still truncated.
    blocks2, unit2 = _multi_block_unit(2)
    unit = unit2
    seen_limits2: list[int | None] = []
    calls_used["n"] = 0

    async def gen_always_truncated(req: LLMRequest[TranslationResultResponse]) -> LLMResponse[TranslationResultResponse]:
        seen_limits2.append(req.max_completion_tokens)
        raise LLMResponseFormatError(
            "truncated",
            truncated=True,
            violation=Violation(
                rule_code="provider_truncated",
                stage=LLMCallStage.TRANSLATION_UNIT.value,
                scope="global",
                safe_message="truncated",
            ),
        )

    call2 = TranslationCall(
        unit_of_work=_make_uow(scope_count_fn=lambda sid: calls_used["n"]),  # type: ignore[arg-type]
        llm_client=SimpleNamespace(generate=gen_always_truncated),  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_attempts=1),
        validator=SimpleNamespace(validate=lambda payload, **kwargs: ()),  # type: ignore[arg-type]
        sleep=lambda _: None,  # type: ignore[arg-type]
    )
    with pytest.raises(RecoverySplit):
        await call2.request(
            _translation_context(req),  # type: ignore[arg-type]
            unit=unit2,
            target_language="ru",
            note_ids=lambda: "n1",
            recovery=(root_id, "scope-1"),
        )
    assert seen_limits2 == [6144, 32768]

    # 3. Exhausted max_scope_calls prevents escalation -> immediate RecoverySplit.
    seen_limits3: list[int | None] = []
    calls_used["n"] = 0
    call3 = TranslationCall(
        unit_of_work=_make_uow(max_scope_calls=1, scope_count_fn=lambda sid: 1),  # type: ignore[arg-type]
        llm_client=SimpleNamespace(generate=lambda r: _record_and_truncate(seen_limits3, r)),  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_attempts=1),
        validator=SimpleNamespace(validate=lambda payload, **kwargs: ()),  # type: ignore[arg-type]
        sleep=lambda _: None,  # type: ignore[arg-type]
    )
    with pytest.raises(RecoverySplit):
        await call3.request(
            _translation_context(req),  # type: ignore[arg-type]
            unit=unit2,
            target_language="ru",
            note_ids=lambda: "n1",
            recovery=(root_id, "scope-1"),
        )
    assert seen_limits3 == [6144]

    # 4. Stale root (superseded / interrupted) raises RecoveryError immediately without escalating.
    for flag, expected_reason in (
        ({"superseded_after_first": True}, "superseded"),
        ({"interrupted_after_first": True}, "interrupted"),
    ):
        seen_stale: list[int | None] = []
        calls_used["n"] = 0
        call_stale = TranslationCall(
            unit_of_work=_make_uow(**flag),  # type: ignore[arg-type]
            llm_client=SimpleNamespace(generate=lambda r, s=seen_stale: _record_and_truncate(s, r)),  # type: ignore[arg-type]
            retry_policy=RetryPolicy(max_attempts=1),
            validator=SimpleNamespace(validate=lambda payload, **kwargs: ()),  # type: ignore[arg-type]
            sleep=lambda _: None,  # type: ignore[arg-type]
        )
        with pytest.raises(RecoveryError) as err:
            await call_stale.request(
                _translation_context(req),  # type: ignore[arg-type]
                unit=unit2,
                target_language="ru",
                note_ids=lambda: "n1",
                recovery=(root_id, "scope-1"),
            )
        assert err.value.reason == expected_reason
        assert seen_stale == [6144]


async def _record_and_truncate(
    seen: list[int | None], req: LLMRequest[TranslationResultResponse]
) -> LLMResponse[TranslationResultResponse]:
    seen.append(req.max_completion_tokens)
    raise LLMResponseFormatError(
        "truncated",
        truncated=True,
        violation=Violation(
            rule_code="provider_truncated",
            stage=LLMCallStage.TRANSLATION_UNIT.value,
            scope="global",
            safe_message="truncated",
        ),
    )


async def test_block_processing_escalates_tokens_in_non_recovery_without_unbound_local_error() -> None:
    block, unit = _sample_unit("Hello world.")
    cond_id = str(uuid4())
    conditions = SimpleNamespace(
        id=cond_id,
        payload={
            "models": {"processing": "deepseek-v4"},
            "versions": {"validators": 1},
            "llm_parameters": {
                "max_output_tokens": 32768,
                "reasoning_effort": "low",
                "reasoning_max_tokens": None,
            },
        },
    )
    fake_tx = SimpleNamespace(
        runs=SimpleNamespace(active_for_book=lambda book_id: None),
        work=SimpleNamespace(
            get_binding=lambda kind, key: None,
            get_conditions=lambda cid: conditions if cid == cond_id else None,
        ),
    )
    fake_uow = SimpleNamespace(transaction=lambda: nullcontext(fake_tx))
    seen_limits: list[int | None] = []

    async def gen(req: LLMRequest[TranslationResultResponse]) -> LLMResponse[TranslationResultResponse]:
        seen_limits.append(req.max_completion_tokens)
        if len(seen_limits) == 1:
            raise ResultValidationError(
                "truncated",
                violation=Violation(
                    rule_code="provider_truncated",
                    stage=LLMCallStage.TRANSLATION_UNIT.value,
                    scope="global",
                    safe_message="truncated",
                ),
            )
        return LLMResponse(
            payload=TranslationResultResponse(blocks=()),
            provider="fake",
            model="deepseek-v4",
            call_context=req.call_context,
        )

    call = TranslationCall(
        unit_of_work=fake_uow,  # type: ignore[arg-type]
        llm_client=SimpleNamespace(generate=gen),  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_attempts=2),
        validator=SimpleNamespace(
            validate=lambda payload, **kwargs: (
                BlockProcessingResult(block.id, BlockTreatment.TRANSLATE, "Привет."),
            )
        ),  # type: ignore[arg-type]
        sleep=lambda _: None,  # type: ignore[arg-type]
    )
    non_rec_context = CallContext(
        conditions_id=cond_id,
        stage=LLMCallStage.TRANSLATION_UNIT.value,
        scope_id=unit.id,
        ordered_block_ids=(block.id,),
        attempt_id=str(uuid4()),
        generation=0,
        assigned_model="deepseek-v4",
        prompt_hash="a" * 64,
    )
    req = replace(
        LLMRequest.for_translation(instructions="Translate.", input_text="b1: Hello world."),
        call_context=non_rec_context,
    )
    res = await call.request(
        _translation_context(req),  # type: ignore[arg-type]
        unit=unit,
        target_language="ru",
        note_ids=lambda: "n1",
    )
    assert res.attempts == 2
    assert seen_limits == [6144, 32768]
