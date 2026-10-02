"""Журналирование обращений к LLM без сохранения текста книги.

Адаптер оборачивает любой `LLMClient` и пишет по одной записи на каждый фактический
вызов модели. Полные запросы и ответы намеренно не сохраняются: в них находится текст
книги, а журнал нужен для диагностики статусов, стоимости и ошибок (PRD § 24).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from linguary.application.ports.llm import (
    LLMClient,
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


@dataclass(slots=True)
class LoggingLLMClient(LLMClient):
    """Декоратор `LLMClient`, записывающий технический журнал вызовов."""

    inner: LLMClient
    unit_of_work: UnitOfWork
    provider: str
    model: str
    id_factory: IdFactory | None = None
    clock: Clock | None = None
    debug_payloads: bool = False
    pricing: TokenPricing | None = None

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        """Выполняет вызов модели и фиксирует его итог, если запрос привязан к книге."""
        if request.book_id is None or request.stage is None:
            return await self.inner.generate(request)
        started_at = self._now()
        try:
            response = await self.inner.generate(request)
        except LLMResponseFormatError as error:
            self._log(
                request=request,
                started_at=started_at,
                provider=error.provider if error.provider is not None else self.provider,
                model=error.model if error.model is not None else self.model,
                success=False,
                error=error,
                usage=error.usage,
                cost=self._cost_of(error.usage, provider_cost=error.cost),
                response_text=error.raw_content if self.debug_payloads else None,
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
            response_text=response.raw_content if self.debug_payloads else None,
        )
        return replace(
            response,
            call_id=call_id,
            raw_content=response.raw_content if self.debug_payloads else None,
        )

    def _cost_of(
        self, usage: TokenUsage | None, *, provider_cost: Decimal | None
    ) -> Decimal | None:
        """Сохраняет цену провайдера либо вычисляет стоимость известных токенов."""
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
    ) -> None:
        """Записывает одну строку журнала в короткой транзакции."""
        if request.book_id is None or request.stage is None:
            return
        with self.unit_of_work.transaction() as transaction:
            transaction.llm_calls.add(
                LLMCallLog(
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
                ),
                response_text=response_text,
            )

    def _duration_ms(self, started_at: datetime) -> int:
        """Длительность вызова в миллисекундах."""
        return max(0, int((self._now() - started_at).total_seconds() * 1000))

    def _new_id(self) -> str:
        """Идентификатор записи журнала."""
        return (self.id_factory or _uuid)()

    def _now(self) -> datetime:
        """Текущее время в UTC."""
        return (self.clock or _utc_now)()


def _uuid() -> str:
    """Новый UUID строкой."""
    return str(uuid4())


def _utc_now() -> datetime:
    """Текущий момент в UTC."""
    return datetime.now(UTC)


__all__ = ["LoggingLLMClient"]
