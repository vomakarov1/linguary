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
_SECONDS_PER_MINUTE = 60.0


class RateLimitedLLMClient(LLMClient):
    """Декоратор `LLMClient`, выдерживающий предел запросов в минуту."""

    def __init__(
        self,
        *,
        inner: LLMClient,
        requests_per_minute: int,
        clock: Clock | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        if requests_per_minute < 1:
            raise ValueError("предел частоты запросов к модели должен быть положительным")
        self._inner = inner
        self._interval = _SECONDS_PER_MINUTE / requests_per_minute
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._lock = threading.Lock()
        self._next_free: float | None = None

    @property
    def requests_per_minute(self) -> float:
        """Предел частоты, который выдерживает ограничитель."""
        return _SECONDS_PER_MINUTE / self._interval

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        """Дожидается своего момента и выполняет запрос."""
        delay = self._reserve()
        if delay > 0:
            await self._sleep(delay)
        try:
            return await self._inner.generate(request)
        except LLMTransportError as error:
            if error.retry_after_seconds is not None:
                self._hold(error.retry_after_seconds)
            raise

    def _reserve(self) -> float:
        """Занимает ближайший свободный момент и возвращает, сколько до него ждать."""
        with self._lock:
            now = self._clock()
            slot = now if self._next_free is None else max(now, self._next_free)
            self._next_free = slot + self._interval
            return slot - now

    def _hold(self, seconds: float) -> None:
        """Не выпускает запросы раньше, чем провайдер обещал их принять."""
        with self._lock:
            resume_at = self._clock() + seconds
            if self._next_free is None or self._next_free < resume_at:
                self._next_free = resume_at


__all__ = ["RateLimitedLLMClient"]
