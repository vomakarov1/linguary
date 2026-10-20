"""Тесты ConfiguredLLMClient: подготовка клиентов, выбор моделей и закрытие ресурсов."""

from __future__ import annotations

import pytest

from linguary.adapters.config import LLMSettings
from linguary.composition.llm import ConfiguredLLMClient


def test_configured_client_unready_does_not_prepare() -> None:
    """Ненастроенный клиент при подготовке не создаёт соединений."""
    settings = LLMSettings()
    client = ConfiguredLLMClient(settings)
    client.prepare()
    assert len(client._clients) == 0


def test_configured_client_prepare_prepares_effective_models() -> None:
    """Подготовка инициализирует клиенты для обеих эффективных моделей."""
    settings = LLMSettings(
        model="base-model",
        analysis_model="analysis-model",
        processing_model="processing-model",
        api_key="synthetic-key",
    )
    client = ConfiguredLLMClient(settings)
    client.prepare()
    assert "analysis-model" in client._clients
    assert "processing-model" in client._clients


def test_configured_client_single_model_prepares_once() -> None:
    """Если задана только общая model, создаётся один клиент."""
    settings = LLMSettings(
        model="base-model",
        api_key="synthetic-key",
    )
    client = ConfiguredLLMClient(settings)
    client.prepare()
    assert list(client._clients.keys()) == ["base-model"]


@pytest.mark.asyncio
async def test_configured_client_aclose_closes_all() -> None:
    """Закрытие закрывает все созданные клиенты и очищает реестр."""
    settings = LLMSettings(
        model="base-model",
        analysis_model="analysis-model",
        processing_model="processing-model",
        api_key="synthetic-key",
    )
    client = ConfiguredLLMClient(settings)
    client.prepare()
    assert len(client._clients) == 2
    await client.aclose()
    assert len(client._clients) == 0
