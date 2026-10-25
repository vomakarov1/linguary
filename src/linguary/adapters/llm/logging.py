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
        if request.book_id is None or request.stage is None:
            return await self.inner.generate(request)
        diagnostics_for(self.unit_of_work).ensure_available()
        started_at = self._now()
        try:
            response = await self.inner.generate(request)
        except LLMResponseFormatError as error:
            error.raw_content, temporary_meta = bounded_body(error.raw_content)
            error.body_meta = error.body_meta or temporary_meta
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
        self._log(
            request=request,
            started_at=started_at,
            provider=response.provider,
            model=response.model,
            success=True,
            usage=response.usage,
            cost=self._cost_of(response.usage, provider_cost=response.cost),
            call_id=call_id,
            response_text=response.raw_content,
            outcome=response,
        )
        return replace(
            response,
            call_id=call_id,
            raw_content=bounded_body(response.raw_content)[0],
            call_context=request.call_context,
            body_meta={**(response.body_meta or bounded_body(response.raw_content)[1]), 'debug_retained': self.debug_payloads},
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
        outcome: LLMResponse | Exception | None = None,
    ) -> None:
        """Records a single log entry within a short transaction."""
        if request.book_id is None or request.stage is None:
            return
        response_text, bounded_meta = bounded_body(response_text)
        supplied_meta = getattr(outcome, 'body_meta', None) or bounded_meta
        context = request.call_context
        metadata = {'schema_version': 1, 'decision': 'unknown', 'body_meta': supplied_meta,
                    'context': None if context is None else context.to_dict()}
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
        try:
            with self.unit_of_work.transaction() as transaction:
                transaction.llm_calls.add(entry)
            diagnostics_for(self.unit_of_work).finalize_attempt(
                entry.attempt_id, 'unknown' if success else ('rejected' if isinstance(error, LLMResponseFormatError) else 'unknown'),
                violation=None if error is None else violation_of(error, context=context),
                body=response_text, call_id=entry.id, body_meta=supplied_meta,
                retain_accepted=self.debug_payloads,
            )
            if isinstance(error, LLMResponseFormatError):
                error.call_id = entry.id
                error.call_context = context
        except Exception:
            self._observation_failed = True
            diagnostics_for(self.unit_of_work).fail()
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


def _uuid() -> str:
    """New UUID string."""
    return str(uuid4())


def _utc_now() -> datetime:
    """Current UTC timestamp."""
    return datetime.now(UTC)


__all__ = ["LoggingLLMClient"]
