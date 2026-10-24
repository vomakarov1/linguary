"""Connecting LLM client to use cases assembled before its configuration.

The application must start up without a configured LLM: the first launch begins that way,
and the user needs to see configuration state and an action to fix it (PRD § 23).
However, `OpenAICompatibleLLMClient` cannot be created without a model and key, while analysis
and processing use cases require a client in their constructor.

Therefore, the client permits explicit preparation without network access, preserving the ability
to assemble an unconfigured generation. An unconfigured LLM upon invocation produces
`LLMConfigurationError` — configuration error: the task stops with a diagnostic
message and is not retried automatically; once settings are fixed, the user
retries it manually.
"""

from __future__ import annotations

from dataclasses import replace

from linguary.adapters.config import LLMSettings
from linguary.adapters.llm import OpenAICompatibleLLMClient
from linguary.application.ports.llm import (
    LLMClient,
    LLMPayload,
    LLMRequest,
    LLMResponse,
)


class ConfiguredLLMClient(LLMClient):
    """Provider client with network-free preparation and safe initial launch."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._clients: dict[str, OpenAICompatibleLLMClient] = {}

    def prepare(self) -> None:
        """Creates own HTTP client without network access; incomplete settings are permitted."""
        if self._settings.is_ready:
            models = {
                self._settings.effective_analysis_model,
                self._settings.effective_processing_model,
            }
            for model in models:
                if model:
                    self._connected(model)

    def client_for(self, model: str) -> LLMClient:
        """Returns client bound to a concrete model."""
        return _ModelBoundLLMClient(self, model)

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        """Executes request with base model, creating provider client on first invocation."""
        return await self._connected().generate(request)

    async def generate_for[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT], *, model: str
    ) -> LLMResponse[PayloadT]:
        """Executes request with explicitly specified model."""
        return await self._connected(model).generate(request)

    async def aclose(self) -> None:
        """Closes provider connections if they were opened."""
        clients = list(self._clients.values())
        self._clients.clear()
        for client in clients:
            await client.aclose()

    @property
    def _client(self) -> OpenAICompatibleLLMClient | None:
        return self._clients.get(self._settings.model)

    def _connected(self, model: str | None = None) -> OpenAICompatibleLLMClient:
        """Provider client; unconfigured LLM rejects creation (PRD § 23)."""
        target_model = model or self._settings.model
        if target_model not in self._clients:
            settings = (
                self._settings
                if target_model == self._settings.model
                else replace(self._settings, model=target_model)
            )
            self._clients[target_model] = OpenAICompatibleLLMClient(settings=settings)
        return self._clients[target_model]


class _ModelBoundLLMClient(LLMClient):
    """Delegate to ConfiguredLLMClient binding a concrete model."""

    def __init__(self, parent: ConfiguredLLMClient, model: str) -> None:
        self._parent = parent
        self._model = model

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        return await self._parent.generate_for(request, model=self._model)


__all__ = ["ConfiguredLLMClient"]
