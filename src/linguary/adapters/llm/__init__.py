"""LLM adapters: structured response schemas, HTTP client, and mock client for tests.

Encapsulates provider specifics: only the `linguary.application.ports.llm.LLMClient`
port is exposed externally, while JSON response formatting, Pydantic, and HTTP
remain internal to the package (ADR § 3).
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
from linguary.adapters.llm.logging import LoggingLLMClient
from linguary.adapters.llm.openai_compatible import OpenAICompatibleLLMClient
from linguary.adapters.llm.payloads import parse_payload, schema_for
from linguary.adapters.llm.rate_limit import RateLimitedLLMClient

__all__ = [
    "FAKE_MODEL",
    "FAKE_PROVIDER",
    "FakeLLMClient",
    "FakeReply",
    "FakeScenarioError",
    "FakeScenarioStep",
    "LoggingLLMClient",
    "OpenAICompatibleLLMClient",
    "RateLimitedLLMClient",
    "parse_payload",
    "schema_for",
    "strict_json_schema",
]
