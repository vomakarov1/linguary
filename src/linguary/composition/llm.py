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
        self._client: OpenAICompatibleLLMClient | None = None

    def prepare(self) -> None:
        """Создаёт собственный HTTP-клиент без сети; неполные настройки допустимы."""
        if self._settings.is_ready:
            self._connected()

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        """Выполняет запрос, создав клиент провайдера при первом обращении."""
        return await self._connected().generate(request)

    async def aclose(self) -> None:
        """Закрывает соединения провайдера, если они открывались."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _connected(self) -> OpenAICompatibleLLMClient:
        """Клиент провайдера; ненастроенная LLM отвергает создание (PRD § 23)."""
        if self._client is None:
            self._client = OpenAICompatibleLLMClient(settings=self._settings)
        return self._client


__all__ = ["ConfiguredLLMClient"]
