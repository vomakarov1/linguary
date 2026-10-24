"""Tests for ConfiguredLLMClient: client preparation, model selection, and resource closure."""

from __future__ import annotations

import pytest

from linguary.adapters.config import LLMSettings
from linguary.composition.llm import ConfiguredLLMClient


def test_configured_client_unready_does_not_prepare() -> None:
    """An unconfigured client creates no connections during preparation."""
    settings = LLMSettings()
    client = ConfiguredLLMClient(settings)
    client.prepare()
    assert len(client._clients) == 0


def test_configured_client_prepare_prepares_effective_models() -> None:
    """Preparation initializes clients for both effective models."""
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
    """If only the general model is configured, a single client is created."""
    settings = LLMSettings(
        model="base-model",
        api_key="synthetic-key",
    )
    client = ConfiguredLLMClient(settings)
    client.prepare()
    assert list(client._clients.keys()) == ["base-model"]


@pytest.mark.asyncio
async def test_configured_client_aclose_closes_all() -> None:
    """Closing shuts down all created clients and clears the registry."""
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
