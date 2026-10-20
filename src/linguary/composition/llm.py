"""Подключение клиента LLM к сценариям, которые собираются до его настройки.

Приложение обязано подниматься без настроенной LLM: первый запуск начинается именно так,
и пользователю нужно увидеть состояние конфигурации и действие для исправления (PRD § 23).
Но `OpenAICompatibleLLMClient` без модели и ключа не создаётся вовсе, а сценарии анализа
и обработки требуют клиент в конструкторе.

Поэтому клиент допускает явную подготовку без сети, сохраняя возможность собрать
ненастроенное поколение. Ненастроенная LLM при обращении даёт
`LLMConfigurationError` — ошибка настройки: задача останавливается с диагностическим
сообщением и автоматически не повторяется, а после исправления настроек пользователь
повторяет её сам.
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
    """Клиент провайдера с подготовкой без сети и безопасным первым запуском."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._clients: dict[str, OpenAICompatibleLLMClient] = {}

    def prepare(self) -> None:
        """Создаёт собственный HTTP-клиент без сети; неполные настройки допустимы."""
        if self._settings.is_ready:
            models = {
                self._settings.effective_analysis_model,
                self._settings.effective_processing_model,
            }
            for model in models:
                if model:
                    self._connected(model)

    def client_for(self, model: str) -> LLMClient:
        """Возвращает клиента, привязанного к конкретной модели."""
        return _ModelBoundLLMClient(self, model)

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        """Выполняет запрос с базовой моделью, создав клиент провайдера при первом обращении."""
        return await self._connected().generate(request)

    async def generate_for[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT], *, model: str
    ) -> LLMResponse[PayloadT]:
        """Выполняет запрос с явно заданной моделью."""
        return await self._connected(model).generate(request)

    async def aclose(self) -> None:
        """Закрывает соединения провайдера, если они открывались."""
        clients = list(self._clients.values())
        self._clients.clear()
        for client in clients:
            await client.aclose()

    @property
    def _client(self) -> OpenAICompatibleLLMClient | None:
        return self._clients.get(self._settings.model)

    def _connected(self, model: str | None = None) -> OpenAICompatibleLLMClient:
        """Клиент провайдера; ненастроенная LLM отвергает создание (PRD § 23)."""
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
    """Делегат к ConfiguredLLMClient с фиксацией конкретной модели."""

    def __init__(self, parent: ConfiguredLLMClient, model: str) -> None:
        self._parent = parent
        self._model = model

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        return await self._parent.generate_for(request, model=self._model)


__all__ = ["ConfiguredLLMClient"]
