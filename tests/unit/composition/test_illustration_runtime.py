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
from linguary.adapters.llm.rate_limit import LLMRateLimiter
from linguary.application.ports.llm import LLMRequest, LLMResponseKind
from linguary.composition.illustration_runtime import IllustrationRuntime
from linguary.composition.migrations import apply_migrations
from linguary.composition.runtime_settings import RuntimeGeneration, RuntimeSettings
from linguary.domain import LLMCallStage
from linguary.domain.illustrations import (
    IllustrationConflictError,
    IllustrationReasonCode,
    IllustrationSettings,
    IllustrationSettingsRevision,
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
) -> IllustrationSettingsRevision:
    settings = IllustrationSettings(
        enabled=True,
        max_per_chapter=3,
        model_reference=model_reference,
        model_prompts=model_prompts,
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
    assert pooled_client._settings.temperature == 0.2
    assert pooled_client._settings.max_output_tokens == 4096
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
