"""Адаптеры LLM: схемы структурированных ответов, HTTP-клиент и клиент для тестов.

Здесь проходит граница провайдера: наружу отдаётся только порт
`linguary.application.ports.llm.LLMClient`, а JSON-формат ответа, Pydantic и HTTP
остаются внутри пакета (ADR § 3).
"""

from linguary.adapters.llm.fake import (
    FAKE_MODEL,
    FAKE_PROVIDER,
    FakeLLMClient,
    FakeReply,
    FakeScenarioError,
    FakeScenarioStep,
)
from linguary.adapters.llm.json_schema import strict_json_schema
from linguary.adapters.llm.openai_compatible import OpenAICompatibleLLMClient
from linguary.adapters.llm.payloads import parse_payload, schema_for

__all__ = [
    "FAKE_MODEL",
    "FAKE_PROVIDER",
    "FakeLLMClient",
    "FakeReply",
    "FakeScenarioError",
    "FakeScenarioStep",
    "OpenAICompatibleLLMClient",
    "parse_payload",
    "schema_for",
    "strict_json_schema",
]
