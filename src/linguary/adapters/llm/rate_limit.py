"""Ограничитель частоты запросов к модели, общий для всех воркеров процесса.

Параллельная обработка (PRD § 25) умножает частоту запросов к провайдеру, а у провайдеров
есть предел запросов в минуту: превысив его, приложение получает отказ `429`, и задачи
тратят попытки на ожидание. Декоратор над `LLMClient` держит между запросами шаг
`60 / requests_per_minute` секунд, сколько бы воркеров ни обращалось к модели.

Шаг равномерный, без накопления: простой не копит запас на пачку запросов, потому что
провайдеры считают частоту скользящим окном и пачку всё равно отвергнут.

Отказ по частоте с заголовком `Retry-After` сдвигает следующий свободный момент для всех:
провайдер сказал, когда он снова примет запрос, и остальным воркерам незачем получать
тот же отказ раньше срока. Паузу перед повтором самого отвергнутого запроса выбирает
политика повторов (`RetryPolicy`) по тому же значению.

Очередь моментов защищена блокировкой потоков, а не цикла событий: ограничитель живёт
столько же, сколько приложение, и не должен зависеть от того, в каком цикле событий
или потоке его позвали.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable

from linguary.application.ports.llm import (
    LLMClient,
    LLMPayload,
    LLMRequest,
    LLMResponse,
    LLMTransportError,
)

type Clock = Callable[[], float]
type Sleep = Callable[[float], Awaitable[None]]

#: Секунд в минуте: предел задаётся в запросах в минуту, как его публикуют провайдеры.
_SECONDS_PER_MINUTE = 60


class LLMRateLimiter:
    """Общие резервации запросов и независимая пауза провайдера."""

    def __init__(
        self,
        requests_per_minute: int | None,
        *,
        clock: Clock | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._lock = threading.Lock()
        self._last_numeric_slot: float | None = None
        self._last_unlimited_call: float | None = None
        self._provider_blocked_until = 0.0
        self.set_limit(requests_per_minute)

    @property
    def requests_per_minute(self) -> int | None:
        with self._lock:
            return self._requests_per_minute

    def set_limit(self, value: int | None) -> None:
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError("предел частоты запросов должен быть положительным целым числом")
        # Деление целых не преобразует огромный предел в float до расчёта отношения.
        interval = None if value is None else _SECONDS_PER_MINUTE / value
        with self._lock:
            self._requests_per_minute = value
            self._interval = interval

    async def acquire(self) -> None:
        with self._lock:
            slot = None
            if self._interval is not None:
                times = [self._clock()]
                for previous in (self._last_numeric_slot, self._last_unlimited_call):
                    if previous is not None:
                        times.append(previous + self._interval)
                slot = max(times)
                self._last_numeric_slot = slot
        while True:
            with self._lock:
                now = self._clock()
                deadline = self._provider_blocked_until
                if slot is not None:
                    deadline = max(deadline, slot)
                delay = deadline - now
                if delay <= 0:
                    if slot is None:
                        self._last_unlimited_call = now
                    return
            await self._sleep(delay)

    def hold(self, seconds: float) -> None:
        """Продлевает общий запрет провайдера, не меняя числовые слоты."""
        with self._lock:
            self._provider_blocked_until = max(
                self._provider_blocked_until, self._clock() + seconds
            )


class RateLimitedLLMClient(LLMClient):
    """Декоратор над клиентом с собственным или общим ограничителем."""

    def __init__(
        self,
        *,
        inner: LLMClient,
        requests_per_minute: int | None = None,
        limiter: LLMRateLimiter | None = None,
        clock: Clock | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        self._inner = inner
        self._limiter = limiter if limiter is not None else LLMRateLimiter(
            requests_per_minute, clock=clock, sleep=sleep
        )

    @property
    def requests_per_minute(self) -> int | None:
        return self._limiter.requests_per_minute

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        await self._limiter.acquire()
        try:
            return await self._inner.generate(request)
        except LLMTransportError as error:
            if error.retry_after_seconds is not None:
                self._limiter.hold(error.retry_after_seconds)
            raise


__all__ = ["LLMRateLimiter", "RateLimitedLLMClient"]
