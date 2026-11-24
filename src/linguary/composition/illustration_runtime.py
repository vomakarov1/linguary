"""Runtime factory supplying bounded LLM clients and wire sizers for illustrations (`P2.T2`)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from typing import Any

from linguary.adapters.config.settings import LLMSettings
from linguary.adapters.llm.logging import LoggedObserver, LoggingLLMClient
from linguary.adapters.llm.openai_compatible import (
    OpenAICompatibleLLMClient,
    _is_openrouter_url,
    serialize_request,
)
from linguary.adapters.llm.rate_limit import LLMRateLimiter, RateLimitedLLMClient
from linguary.application.ports.illustrations import (
    IllustrationRuntimePort,
    RequestWireSizer,
    RequestWireSnapshot,
)
from linguary.application.ports.llm import LLMClient, LLMRequest
from linguary.application.ports.persistence import UnitOfWork
from linguary.composition.book_runtime import route_snapshot
from linguary.composition.runtime_settings import RuntimeSettings
from linguary.domain.illustrations import (
    IllustrationConflictError,
    IllustrationReasonCode,
    IllustrationSettingsRevision,
)
from linguary.domain.llm_log import LLMCallStage, TokenPricing


def _selected_model_for(revision: IllustrationSettingsRevision, stage: LLMCallStage) -> str | None:
    if stage is LLMCallStage.ILLUSTRATION_REFERENCE:
        return revision.settings.model_reference
    if stage is LLMCallStage.ILLUSTRATION_CHAPTER:
        return revision.settings.model_prompts
    return revision.settings.model_prompts or revision.settings.model_reference


def _clean_llm_settings(
    base_llm: LLMSettings,
    *,
    model: str,
    revision: IllustrationSettingsRevision,
) -> LLMSettings:
    """Build clean illustration `LLMSettings`, isolated from global reasoning/temperature."""
    policy = revision.settings.policy
    return LLMSettings(
        provider=base_llm.provider,
        base_url=base_llm.base_url,
        proxy_mode=base_llm.proxy_mode,
        proxy_url=base_llm.proxy_url,
        api_key=base_llm.api_key,
        model=model,
        timeout_seconds=base_llm.timeout_seconds,
        temperature=policy.temperature,
        max_output_tokens=policy.max_completion_tokens,
        reasoning_effort=None,
        reasoning_max_tokens=None,
        max_attempts=1,
        requests_per_minute=base_llm.requests_per_minute,
        debug_llm_payloads=base_llm.debug_llm_payloads,
        input_price_per_mtok=base_llm.input_price_per_mtok,
        output_price_per_mtok=base_llm.output_price_per_mtok,
    )


@dataclass(frozen=True, slots=True)
class IllustrationWireSizer(RequestWireSizer):
    """Exact UTF-8 JSON wire byte sizer (works without provider API key)."""

    settings: LLMSettings
    include_openrouter_provider: bool
    explicit_cache: bool = False

    def measure(self, request: LLMRequest[Any]) -> RequestWireSnapshot:
        wire_bytes = serialize_request(
            request,
            self.settings,
            include_openrouter_provider=self.include_openrouter_provider,
            explicit_cache=self.explicit_cache,
        )
        return RequestWireSnapshot(
            sha256=sha256(wire_bytes).hexdigest(),
            size_bytes=len(wire_bytes),
        )


class IllustrationRuntime(IllustrationRuntimePort):
    """Dynamically resolves illustration LLM clients and wire sizers from `RuntimeSettings`."""

    def __init__(
        self,
        *,
        runtime_settings: RuntimeSettings,
        rate_limiter: LLMRateLimiter,
        unit_of_work: UnitOfWork,
        on_logged: LoggedObserver | None = None,
        injected_client: LLMClient | None = None,
    ) -> None:
        self._runtime_settings = runtime_settings
        self._rate_limiter = rate_limiter
        self._uow = unit_of_work
        self._on_logged = on_logged
        self._injected_client = injected_client
        self._pool: dict[tuple[tuple[Any, ...], str], OpenAICompatibleLLMClient] = {}
        self._retired: list[OpenAICompatibleLLMClient] = []

    def _current_llm(self) -> LLMSettings:
        return self._runtime_settings.current.settings.llm

    def _transport_fingerprint(self, llm: LLMSettings) -> tuple[Any, ...]:
        return (
            llm.provider,
            llm.base_url,
            llm.proxy_mode,
            llm.proxy_url,
            llm.api_key,
            llm.timeout_seconds,
        )

    def _pricing_for(self, llm: LLMSettings) -> TokenPricing | None:
        if llm.input_price_per_mtok is not None and llm.output_price_per_mtok is not None:
            return TokenPricing(
                input_per_mtok=Decimal(str(llm.input_price_per_mtok)),
                output_per_mtok=Decimal(str(llm.output_price_per_mtok)),
            )
        return None

    def is_provider_ready(self) -> bool:
        if self._injected_client is not None:
            return True
        llm = self._current_llm()
        return bool(llm.api_key and llm.api_key.strip())

    def default_model(self) -> str | None:
        model = self._current_llm().model.strip()
        return model or None

    def route_id(self) -> str:
        return str(route_snapshot(self._current_llm())["route_id"])

    def sizer_for(
        self,
        revision: IllustrationSettingsRevision,
        stage: LLMCallStage,
    ) -> RequestWireSizer:
        llm = self._current_llm()
        model = _selected_model_for(revision, stage) or llm.model or "unconfigured-model"
        clean = _clean_llm_settings(llm, model=model, revision=revision)
        return IllustrationWireSizer(
            settings=clean,
            include_openrouter_provider=_is_openrouter_url(clean.base_url),
            explicit_cache=False,
        )

    def client_for(
        self,
        revision: IllustrationSettingsRevision,
        stage: LLMCallStage,
    ) -> LLMClient:
        llm = self._current_llm()
        model = _selected_model_for(revision, stage)
        if not model:
            raise IllustrationConflictError(
                IllustrationReasonCode.ROUTE_UNAVAILABLE,
                "не указана модель для этапа подготовки иллюстраций",
            )
        if self._injected_client is not None:
            inner: LLMClient = self._injected_client
        else:
            if not (llm.api_key and llm.api_key.strip()):
                raise IllustrationConflictError(
                    IllustrationReasonCode.ROUTE_UNAVAILABLE,
                    "ключ провайдера LLM не настроен",
                )
            clean = _clean_llm_settings(llm, model=model, revision=revision)
            fp = self._transport_fingerprint(clean)
            # Evict any cached clients belonging to an older transport fingerprint.
            for key in list(self._pool.keys()):
                if key[0] != fp:
                    self._retired.append(self._pool.pop(key))
            pool_key = (fp, model)
            cached = self._pool.get(pool_key)
            if cached is None:
                cached = OpenAICompatibleLLMClient(
                    settings=clean,
                    allow_explicit_prompt_cache=False,
                )
                self._pool[pool_key] = cached
            inner = cached

        logged = LoggingLLMClient(
            inner=inner,
            unit_of_work=self._uow,
            provider=llm.provider,
            model=model,
            debug_payloads=llm.debug_llm_payloads,
            pricing=self._pricing_for(llm),
            on_logged=self._on_logged,
        )
        return RateLimitedLLMClient(inner=logged, limiter=self._rate_limiter)

    async def aclose(self) -> None:
        clients = [*self._retired, *self._pool.values()]
        self._retired.clear()
        self._pool.clear()
        for client in clients:
            await client.aclose()


__all__ = ["IllustrationRuntime", "IllustrationWireSizer"]
