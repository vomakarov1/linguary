"""Unit tests for `IllustrationRuntime` and `IllustrationWireSizer` (`P2.T2`)."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from linguary.adapters.config import AppSettings, LLMSettings
from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.adapters.llm.openai_compatible import serialize_request
from linguary.adapters.llm.rate_limit import LLMRateLimiter
from linguary.application.illustration_context import is_incompatible_route_error
from linguary.application.ports.llm import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRequest,
    LLMResponseKind,
)
from linguary.composition.illustration_runtime import IllustrationRuntime
from linguary.composition.migrations import apply_migrations
from linguary.composition.runtime_settings import RuntimeGeneration, RuntimeSettings
from linguary.domain import LLMCallStage
from linguary.domain.illustrations import (
    IllustrationConflictError,
    IllustrationPolicy,
    IllustrationReasonCode,
    IllustrationSettings,
    IllustrationSettingsRevision,
    IllustrationValidationError,
)


class _DummyConfigured:
    async def aclose(self) -> None:
        return None


def _make_runtime_settings(llm: LLMSettings) -> RuntimeSettings:
    app_settings = AppSettings(llm=llm)

    def _factory(candidate: AppSettings, *, conditions=None):
        del conditions
        return RuntimeGeneration(
            settings=candidate,
            configured_client=_DummyConfigured(),  # type: ignore[arg-type]
            analysis=None,  # type: ignore[arg-type]
            processing=None,  # type: ignore[arg-type]
            regeneration=None,  # type: ignore[arg-type]
        )

    return RuntimeSettings(app_settings, _factory(app_settings), revision=1, generation_factory=_factory)


def _revision(
    *,
    model_reference: str | None = "anthropic/claude-sonnet-4",
    model_prompts: str | None = "openai/gpt-5",
    policy: IllustrationPolicy | None = None,
) -> IllustrationSettingsRevision:
    settings = IllustrationSettings(
        enabled=True,
        max_per_chapter=3,
        model_reference=model_reference,
        model_prompts=model_prompts,
        policy=policy or IllustrationPolicy(),
    )
    return IllustrationSettingsRevision(
        id=str(uuid4()),
        book_id="book-1",
        revision=1,
        settings_sha256=settings.sha256(),
        settings=settings,
        created_at=datetime(2026, 11, 1, tzinfo=UTC),
    )


async def test_illustration_runtime_sizer_without_key_route_unavailable_injected_client_dynamic_settings_pool_and_aclose(
    tmp_path: Path,
) -> None:
    db = tmp_path / "rt.db"
    apply_migrations(db)
    uow = SqlAlchemyUnitOfWork.for_engine(create_database_engine(db))
    limiter = LLMRateLimiter(None, max_concurrency=2)

    # 1. Without API key: sizer_for works, client_for raises route_unavailable.
    no_key_llm = LLMSettings(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-5-mini",
        api_key=None,
        reasoning_effort="high",
    )
    rt_settings = _make_runtime_settings(no_key_llm)
    runtime = IllustrationRuntime(
        runtime_settings=rt_settings,
        rate_limiter=limiter,
        unit_of_work=uow,
    )
    rev = _revision()
    assert runtime.is_provider_ready() is False
    assert runtime.default_model() == "openai/gpt-5-mini"

    sizer = runtime.sizer_for(rev, LLMCallStage.ILLUSTRATION_REFERENCE)
    req = LLMRequest(
        kind=LLMResponseKind.ILLUSTRATION_EXTRACT,
        instructions="Extract facts.",
        input_text="Alice stepped into the quiet hall.",
    )
    snap = sizer.measure(req)
    assert snap.size_bytes > 0
    assert len(snap.sha256) == 64

    with pytest.raises(IllustrationConflictError) as err:
        runtime.client_for(rev, LLMCallStage.ILLUSTRATION_REFERENCE)
    assert err.value.reason_code == IllustrationReasonCode.ROUTE_UNAVAILABLE.value

    # 2. Dynamically updating RuntimeSettings with an API key enables client_for, reuses pool,
    #    isolates reasoning_effort, and cleanly closes pooled + evicted clients on aclose().
    with_key_llm = replace(no_key_llm, api_key="sk-test-1")
    rt_settings.publish(rt_settings.prepare(AppSettings(llm=with_key_llm)), revision=2)
    assert runtime.is_provider_ready() is True

    c1 = runtime.client_for(rev, LLMCallStage.ILLUSTRATION_REFERENCE)
    c2 = runtime.client_for(rev, LLMCallStage.ILLUSTRATION_REFERENCE)
    assert len(runtime._pool) == 1
    pooled_client = next(iter(runtime._pool.values()))
    assert pooled_client._allow_explicit_prompt_cache is False
    assert pooled_client._settings.reasoning_effort is None
    assert pooled_client._settings.temperature is None
    assert pooled_client._settings.max_output_tokens == 16384
    assert c1._inner.inner is c2._inner.inner  # type: ignore[attr-defined]

    # Updating transport key rotates pool and tracks retired client for aclose().
    rt_settings.publish(
        rt_settings.prepare(AppSettings(llm=replace(with_key_llm, api_key="sk-test-2"))),
        revision=3,
    )
    runtime.client_for(rev, LLMCallStage.ILLUSTRATION_REFERENCE)
    assert len(runtime._retired) == 1
    await runtime.aclose()
    assert len(runtime._pool) == 0
    assert len(runtime._retired) == 0

    # 3. Injected FakeLLMClient works without api_key.
    fake = FakeLLMClient(
        [
            FakeReply(
                content=json.dumps(
                    {
                        "contract_version": 1,
                        "moment_id": "m1",
                        "scene_composition_en": "Ink drawing.",
                        "depicted_entity_ids": [],
                        "used_fact_ids": [],
                        "used_design_keys": [],
                        "assumptions_ru": [],
                    }
                )
            )
        ]
    )
    injected_rt = IllustrationRuntime(
        runtime_settings=_make_runtime_settings(no_key_llm),
        rate_limiter=limiter,
        unit_of_work=uow,
        injected_client=fake,
    )
    assert injected_rt.is_provider_ready() is True
    client = injected_rt.client_for(rev, LLMCallStage.ILLUSTRATION_CHAPTER)
    resp = await client.generate(
        LLMRequest(
            kind=LLMResponseKind.ILLUSTRATION_PROMPT,
            instructions="Compose.",
            input_text="Details.",
        )
    )
    assert resp.payload.moment_id == "m1"
    await injected_rt.aclose()


def test_illustration_policy_defaults_backward_compat_and_temperature_validation() -> None:
    default_policy = IllustrationPolicy()
    assert default_policy.temperature is None
    assert default_policy.max_completion_tokens == 16384
    assert default_policy.max_repair_turns == 2
    assert default_policy.to_payload()["temperature"] is None
    assert default_policy.to_payload()["max_completion_tokens"] == 16384
    assert default_policy.to_payload()["max_repair_turns"] == 2

    legacy = IllustrationPolicy.from_payload(
        {"temperature": 0.2, "max_completion_tokens": 4096}
    )
    assert legacy.temperature == 0.2
    assert legacy.max_completion_tokens == 4096
    assert legacy.max_repair_turns == 2

    explicit_one = IllustrationPolicy.from_payload({"max_repair_turns": 1})
    assert explicit_one.max_repair_turns == 1

    zero_temp = IllustrationPolicy.from_payload({"temperature": 0.0})
    assert zero_temp.temperature == 0.0
    assert zero_temp.max_completion_tokens == 16384

    for bad_temp in (-0.1, 2.5, True, "0.2", float("nan"), float("inf")):
        with pytest.raises(IllustrationValidationError):
            IllustrationPolicy(temperature=bad_temp)  # type: ignore[arg-type]


async def test_illustration_runtime_model_compatibility_reasoning_inheritance_and_pool_keys(
    tmp_path: Path,
) -> None:
    db = tmp_path / "rt-compat.db"
    apply_migrations(db)
    uow = SqlAlchemyUnitOfWork.for_engine(create_database_engine(db))
    limiter = LLMRateLimiter(None, max_concurrency=2)

    primary_llm = LLMSettings(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-v4.1-flash-1017",
        api_key="sk-test",
        temperature=None,
        max_output_tokens=32768,
        reasoning_effort="low",
    )
    rt_settings = _make_runtime_settings(primary_llm)
    runtime = IllustrationRuntime(
        runtime_settings=rt_settings,
        rate_limiter=limiter,
        unit_of_work=uow,
    )

    # 1. Same model as primary: inherits reasoning_effort and max_output_tokens, omits temperature.
    same_rev = _revision(
        model_reference="deepseek/deepseek-v4.1-flash-1017",
        model_prompts="openai/gpt-4.1-mini",
    )
    req = LLMRequest(
        kind=LLMResponseKind.ILLUSTRATION_EXTRACT,
        instructions="Extract facts.",
        input_text="Alice stepped in.",
    )
    same_sizer = runtime.sizer_for(same_rev, LLMCallStage.ILLUSTRATION_REFERENCE)
    wire_same = json.loads(
        serialize_request(
            req,
            same_sizer.settings,  # type: ignore[attr-defined]
            include_openrouter_provider=True,
            explicit_cache=False,
        )
    )
    assert "temperature" not in wire_same
    assert wire_same["max_completion_tokens"] == 32768
    assert same_sizer.settings.reasoning_effort == "low"  # type: ignore[attr-defined]

    # 2. Different model from primary: reasoning_effort and reasoning_max_tokens are None.
    diff_sizer = runtime.sizer_for(same_rev, LLMCallStage.ILLUSTRATION_CHAPTER)
    assert diff_sizer.settings.reasoning_effort is None  # type: ignore[attr-defined]
    assert diff_sizer.settings.reasoning_max_tokens is None  # type: ignore[attr-defined]
    assert diff_sizer.settings.max_output_tokens == 32768  # type: ignore[attr-defined]

    # 3. Non-OpenRouter URL with reasoning_max_tokens: does not inherit and sizer does not crash.
    non_or_llm = LLMSettings(
        provider="openai_compatible",
        base_url="https://api.openai.com/v1",
        model="deepseek/deepseek-v4.1-flash-1017",
        api_key="sk-test",
        max_output_tokens=32768,
        reasoning_effort=None,
        reasoning_max_tokens=4096,
    )
    rt_settings.publish(rt_settings.prepare(AppSettings(llm=non_or_llm)), revision=2)
    non_or_sizer = runtime.sizer_for(same_rev, LLMCallStage.ILLUSTRATION_REFERENCE)
    assert non_or_sizer.settings.reasoning_max_tokens is None  # type: ignore[attr-defined]
    assert non_or_sizer.measure(req).size_bytes > 0

    # 4. OpenRouter with reasoning_max_tokens >= effective_max_tokens resets reasoning_max_tokens.
    bad_budget_llm = LLMSettings(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-v4.1-flash-1017",
        api_key="sk-test",
        max_output_tokens=None,
        reasoning_effort=None,
        reasoning_max_tokens=16384,
    )
    rt_settings.publish(rt_settings.prepare(AppSettings(llm=bad_budget_llm)), revision=3)
    bad_budget_sizer = runtime.sizer_for(same_rev, LLMCallStage.ILLUSTRATION_REFERENCE)
    assert bad_budget_sizer.settings.reasoning_max_tokens is None  # type: ignore[attr-defined]

    # Valid OpenRouter reasoning_max_tokens < effective_max_tokens IS inherited for same model.
    ok_budget_llm = replace(bad_budget_llm, max_output_tokens=32768, reasoning_max_tokens=4096)
    rt_settings.publish(rt_settings.prepare(AppSettings(llm=ok_budget_llm)), revision=4)
    ok_budget_sizer = runtime.sizer_for(same_rev, LLMCallStage.ILLUSTRATION_REFERENCE)
    assert ok_budget_sizer.settings.reasoning_max_tokens == 4096  # type: ignore[attr-defined]

    # 5. Changing policy.temperature creates separate pool entry; changing base max_output_tokens retires old clients.
    runtime.client_for(same_rev, LLMCallStage.ILLUSTRATION_REFERENCE)
    rev_temp = _revision(
        model_reference="deepseek/deepseek-v4.1-flash-1017",
        model_prompts="openai/gpt-4.1-mini",
        policy=IllustrationPolicy(temperature=0.3),
    )
    runtime.client_for(rev_temp, LLMCallStage.ILLUSTRATION_REFERENCE)
    assert len(runtime._pool) == 2
    assert len(runtime._retired) == 0

    rt_settings.publish(
        rt_settings.prepare(AppSettings(llm=replace(ok_budget_llm, max_output_tokens=16384))),
        revision=5,
    )
    runtime.client_for(same_rev, LLMCallStage.ILLUSTRATION_REFERENCE)
    assert len(runtime._pool) == 1
    assert len(runtime._retired) == 2
    await runtime.aclose()


def test_is_incompatible_route_error_classification() -> None:
    def _provider_err(msg: str, status_code: int | None = None) -> LLMProviderError:
        err = LLMProviderError(msg)
        if status_code is not None:
            err.status_code = status_code  # type: ignore[attr-defined]
        return err

    assert is_incompatible_route_error(LLMConfigurationError("missing key")) is True
    assert is_incompatible_route_error(
        _provider_err(
            "провайдер отверг запрос: 404 - No endpoints found that can handle the requested parameters",
            status_code=404,
        )
    ) is True
    assert is_incompatible_route_error(
        _provider_err("провайдер отверг запрос: 404 - No endpoints found")
    ) is True
    assert is_incompatible_route_error(
        _provider_err("400 - unsupported parameter: temperature", status_code=400)
    ) is True
    # Context limit errors and numeric token counts must NOT false-positive.
    assert is_incompatible_route_error(
        _provider_err("context_length exceeded: 404 tokens over limit", status_code=400)
    ) is False
    assert is_incompatible_route_error(
        _provider_err("rate limit exceeded after 404 tokens", status_code=429)
    ) is False
    assert is_incompatible_route_error(
        _provider_err("generic error with 404 tokens in prompt", status_code=400)
    ) is False
