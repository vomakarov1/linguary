"""Deterministic LLM client for tests and local debugging.

Replaces the provider with preconfigured replies and performs no network calls. Needed
across all test levels — unit, integration, end-to-end scenarios (ADR § 3) — and CLI debug mode,
which is why it resides in the package alongside the production adapter rather than in tests.

Replies are configured in the same format as returned by the provider: raw structured response text.
They are parsed by the same `parse_payload` used by the HTTP client, ensuring that scenarios
such as "invalid JSON", "token limit truncation", and "results for unrecognized blocks"
reproduce identically to a live model rather than being simulated via exceptions.

Issued requests are recorded for inspecting context structure, call sequence,
and verifying that retry instructions were properly appended to the original prompt.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from linguary.adapters.llm.payloads import parse_payload
from linguary.application.ports.llm import (
    LLMClient,
    LLMPayload,
    LLMRequest,
    LLMResponse,
)
from linguary.domain import TokenUsage

#: Provider and model in client replies: recorded in call logs to show that execution
#: occurred without calling a live model.
FAKE_PROVIDER = "fake"
FAKE_MODEL = "fake-model"


class FakeScenarioError(Exception):
    """Client scenario configuration error: preconfigured replies exhausted.

    Represents a test setup defect rather than model behavior, so it does not inherit from `LLMError`
    and must not look like a retryable invocation failure.
    """


@dataclass(frozen=True, slots=True)
class FakeReply:
    """A single preconfigured provider response."""

    content: str
    truncated: bool = False
    usage: TokenUsage | None = None
    model: str = FAKE_MODEL

    @classmethod
    def of_payload(
        cls,
        payload: Mapping[str, object],
        *,
        truncated: bool = False,
        usage: TokenUsage | None = None,
    ) -> FakeReply:
        """Constructs a reply from a payload structure: serialized identically to model output."""
        return cls(
            content=json.dumps(payload, ensure_ascii=False),
            truncated=truncated,
            usage=usage,
        )


#: Action performed by the client on each invocation: reply or raise an error. Exceptions
#: represent invocation failures (`LLMTransportError` and other `LLMError` subclasses).
type FakeScenarioStep = FakeReply | Exception


@dataclass(slots=True)
class FakeLLMClient(LLMClient):
    """LLM client returning preconfigured responses.

    Replies are consumed in invocation order. When preconfigured replies are exhausted,
    `default` is used; without it, the scenario is considered exhausted and reports an error
    rather than returning random data.
    """

    replies: Sequence[FakeScenarioStep] = ()
    default: FakeScenarioStep | None = None
    provider: str = FAKE_PROVIDER
    requests: list[LLMRequest[LLMPayload]] = field(default_factory=list)

    @classmethod
    def always(cls, step: FakeScenarioStep, *, provider: str = FAKE_PROVIDER) -> FakeLLMClient:
        """Client returning the same reply on every request."""
        return cls(default=step, provider=provider)

    @classmethod
    def replying(
        cls, steps: Iterable[FakeScenarioStep], *, provider: str = FAKE_PROVIDER
    ) -> FakeLLMClient:
        """Client returning a fixed sequence of steps."""
        return cls(replies=tuple(steps), provider=provider)

    @property
    def call_count(self) -> int:
        """Number of requests received by the client."""
        return len(self.requests)

    @property
    def last_request(self) -> LLMRequest[LLMPayload]:
        """Most recent request received."""
        if not self.requests:
            raise FakeScenarioError("клиент ещё не получал запросов")
        return self.requests[-1]

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        """Returns the next preconfigured response, parsing it as a provider payload."""
        step = self._next_step()
        self.requests.append(request)
        if isinstance(step, Exception):
            raise step
        payload = parse_payload(request.kind, step.content, contract_version=request.contract_version, name_translation_policy_version=request.name_translation_policy_version, truncated=step.truncated)
        return LLMResponse(
            payload=payload,  # type: ignore[arg-type]  # schema selected by request.kind
            provider=self.provider,
            model=step.model,
            truncated=step.truncated,
            usage=step.usage,
        )

    def _next_step(self) -> FakeScenarioStep:
        """Scenario step for the current invocation."""
        index = len(self.requests)
        if index < len(self.replies):
            return self.replies[index]
        if self.default is not None:
            return self.default
        raise FakeScenarioError(
            f"сценарий клиента исчерпан: заготовлено ответов — {len(self.replies)}, "
            f"вызов — {index + 1}"
        )


__all__ = [
    "FAKE_MODEL",
    "FAKE_PROVIDER",
    "FakeLLMClient",
    "FakeReply",
    "FakeScenarioError",
    "FakeScenarioStep",
]
