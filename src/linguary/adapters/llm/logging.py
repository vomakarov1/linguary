"""Logging LLM calls without storing book text.

Wraps any `LLMClient` and records one log entry per actual model call.
Full request and response contents are intentionally omitted to avoid storing book text,
while preserving diagnostics for status codes, token cost, and errors (PRD § 24).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from uuid import uuid4

from linguary.application.diagnostics import bounded_body, diagnostics_for
from linguary.application.retry_policy import violation_of
from linguary.application.ports.llm import (
    LLMClient,
    LLMConfigurationError,
    LLMPayload,
    LLMRequest,
    LLMResponse,
    LLMResponseFormatError,
)
from linguary.application.ports.persistence import UnitOfWork
from linguary.application.retry_policy import classify_error
from linguary.domain import LLMCallLog, TokenUsage
from linguary.domain.llm_log import TokenPricing

type Clock = Callable[[], datetime]
type IdFactory = Callable[[], str]
type LoggedObserver = Callable[[LLMCallLog, LLMRequest, LLMResponse | Exception], None]


@dataclass(slots=True)
class LoggingLLMClient(LLMClient):
    """`LLMClient` decorator recording technical call logs."""

    inner: LLMClient
    unit_of_work: UnitOfWork
    provider: str
    model: str
    id_factory: IdFactory | None = None
    clock: Clock | None = None
    debug_payloads: bool = False
    pricing: TokenPricing | None = None
    on_logged: LoggedObserver | None = None
    _observation_failed: bool = field(default=False, init=False, repr=False)

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        """Invokes the model and logs the call outcome if the request is associated with a book."""
        if self._observation_failed:
            raise LLMConfigurationError("Запись наблюдения недоступна; новые вызовы остановлены")
        if request.call_context is not None:
            request = replace(request, call_context=replace(request.call_context,
                prompt_hash=sha256(request.instructions.encode()).hexdigest()))
        if request.book_id is None or request.stage is None:
            return await self.inner.generate(request)
        diagnostics = diagnostics_for(self.unit_of_work)
        diagnostics.ensure_available()
        policy = diagnostics.policy
        started_at = self._now()
        try:
            response = await self.inner.generate(request)
        except LLMResponseFormatError as error:
            error.raw_content, error.body_meta = _bound_if_needed(
                error.raw_content, error.body_meta, max_bytes=policy.max_body_bytes
            )
            self._log(
                request=request,
                started_at=started_at,
                provider=error.provider if error.provider is not None else self.provider,
                model=error.model if error.model is not None else self.model,
                success=False,
                error=error,
                usage=error.usage,
                cost=self._cost_of(error.usage, provider_cost=error.cost),
                response_text=error.raw_content,
                body_meta=error.body_meta,
                outcome=error,
            )
            raise
        except Exception as error:
            self._log(
                request=request,
                started_at=started_at,
                provider=self.provider,
                model=self.model,
                success=False,
                error=error,
                outcome=error,
            )
            raise
        call_id = self._new_id()
        bounded_text, resolved_body_meta = _bound_if_needed(
            response.raw_content, response.body_meta, max_bytes=policy.max_body_bytes
        )
        self._log(
            request=request,
            started_at=started_at,
            provider=response.provider,
            model=response.model,
            success=True,
            usage=response.usage,
            cost=self._cost_of(response.usage, provider_cost=response.cost),
            call_id=call_id,
            response_text=bounded_text,
            body_meta=resolved_body_meta,
            outcome=response,
        )
        return replace(
            response,
            call_id=call_id,
            raw_content=bounded_text,
            call_context=request.call_context,
            body_meta={**resolved_body_meta, 'debug_retained': self.debug_payloads},
        )

    def _cost_of(
        self, usage: TokenUsage | None, *, provider_cost: Decimal | None
    ) -> Decimal | None:
        """Preserves provider-reported cost or computes token cost from pricing configuration."""
        if provider_cost is not None:
            return provider_cost
        return self.pricing.cost_of(usage) if self.pricing is not None else None

    def _log(
        self,
        *,
        request: LLMRequest[LLMPayload],
        started_at: datetime,
        provider: str,
        model: str,
        success: bool,
        error: Exception | None = None,
        usage: TokenUsage | None = None,
        cost: Decimal | None = None,
        call_id: str | None = None,
        response_text: str | None = None,
        body_meta: dict[str, object] | None = None,
        outcome: LLMResponse | Exception | None = None,
    ) -> None:
        """Records a single log entry within a short transaction."""
        if request.book_id is None or request.stage is None:
            return
        diagnostics = diagnostics_for(self.unit_of_work)
        policy = diagnostics.policy
        if body_meta is not None:
            supplied_meta = getattr(outcome, 'body_meta', None) or body_meta
        else:
            response_text, bounded_meta = bounded_body(
                response_text, max_bytes=policy.max_body_bytes
            )
            supplied_meta = getattr(outcome, 'body_meta', None) or bounded_meta
        context = request.call_context
        metadata = {'schema_version': 1, 'decision': 'unknown', 'body_meta': supplied_meta,
                    'context': None if context is None else context.to_dict()}
        if usage is not None:
            usage_details: dict[str, int] = {}
            if usage.cached_prompt_tokens is not None:
                usage_details['cached_prompt_tokens'] = usage.cached_prompt_tokens
            if usage.reasoning_tokens is not None:
                usage_details['reasoning_tokens'] = usage.reasoning_tokens
            if usage_details:
                metadata['usage_details'] = usage_details
        if request.repair_instructions:
            metadata['repair_instructions'] = list(request.repair_instructions)
            if request.attempt > 1 and self.debug_payloads and policy.bodies_enabled:
                raw_input = request.input_text + "\n\n" + "\n\n".join(request.repair_instructions)
                bounded_input, _ = bounded_body(raw_input, max_bytes=policy.max_body_bytes)
                metadata['input_text'] = bounded_input
        entry = LLMCallLog(
            id=call_id or self._new_id(),
            book_id=request.book_id,
            task_id=request.task_id,
            stage=request.stage,
            provider=provider,
            model=model,
            started_at=started_at,
            duration_ms=self._duration_ms(started_at),
            attempt=request.attempt,
            success=success,
            error=None if error is None else classify_error(error),
            usage=usage,
            cost=cost,
            attempt_id=None if context is None else context.attempt_id,
            diagnostic_meta=metadata,
        )
        finalize_error: Exception | None = None
        try:
            with self.unit_of_work.transaction() as transaction:
                transaction.llm_calls.add(entry)
                try:
                    diagnostics.finalize_attempt(
                        entry.attempt_id,
                        'unknown' if success else ('rejected' if isinstance(error, LLMResponseFormatError) else 'unknown'),
                        violation=None if error is None else violation_of(error, context=context),
                        body=response_text,
                        call_id=entry.id,
                        body_meta=supplied_meta,
                        retain_accepted=self.debug_payloads,
                        transaction=transaction,
                    )
                except Exception as exc:
                    finalize_error = exc
            if finalize_error is not None:
                raise finalize_error
            if isinstance(error, LLMResponseFormatError):
                error.call_id = entry.id
                error.call_context = context
        except Exception:
            self._observation_failed = True
            diagnostics.fail()
        if self.on_logged is not None and outcome is not None:
            try:
                self.on_logged(entry, request, outcome)
            except Exception:
                self._observation_failed = True
                raise LLMConfigurationError("Запись наблюдения недоступна; новые вызовы остановлены") from None

    def _duration_ms(self, started_at: datetime) -> int:
        """Call duration in milliseconds."""
        return max(0, int((self._now() - started_at).total_seconds() * 1000))

    def _new_id(self) -> str:
        """Log entry identifier."""
        return (self.id_factory or _uuid)()

    def _now(self) -> datetime:
        """Current time in UTC."""
        return (self.clock or _utc_now)()


def _bound_if_needed(
    raw_content: str | None,
    body_meta: dict[str, object] | None,
    *,
    max_bytes: int,
) -> tuple[str | None, dict[str, object]]:
    """Skips redundant `bounded_body` when the adapter already bounded and scrubbed `raw_content`."""
    if (
        body_meta is not None
        and "redacted_count" in body_meta
        and (raw_content is None or len(raw_content.encode("utf-8")) <= max_bytes)
    ):
        return raw_content, dict(body_meta)
    bounded_text, bounded_meta = bounded_body(raw_content, max_bytes=max_bytes)
    if body_meta is None:
        return bounded_text, bounded_meta
    return bounded_text, {
        **bounded_meta,
        **body_meta,
        "redacted": bool(body_meta.get("redacted")) or bounded_meta["redacted"],
        "truncated": bool(body_meta.get("truncated")) or bounded_meta["truncated"],
        "original_bytes": body_meta.get("original_bytes") or bounded_meta["original_bytes"],
        "stored_bytes": bounded_meta["stored_bytes"],
    }


def _uuid() -> str:
    """New UUID string."""
    return str(uuid4())


def _utc_now() -> datetime:
    """Current UTC timestamp."""
    return datetime.now(UTC)


__all__ = ["LoggingLLMClient"]
