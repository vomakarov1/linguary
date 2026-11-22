"""Model request rate limiter shared across all workers in a process.

Parallel processing (PRD § 25) multiplies provider request frequency, while providers
enforce requests-per-minute limits: exceeding limits results in HTTP `429` errors and wastes
task retry attempts on waiting. This decorator over `LLMClient` maintains an interval of
`60 / requests_per_minute` seconds between requests, regardless of worker count.

Interval pacing is uniform without token bucket bursting: idle periods do not accumulate
request credits, as providers evaluate rates using sliding windows and reject bursts anyway.
Waiters share one FIFO queue and only its head sleeps. The admit time is the moment the head
actually proceeds, so a provider cool-down cannot release the queue in one burst.

A rate limit rejection carrying a `Retry-After` header delays the next available dispatch slot
for all workers: the provider announced when it will accept requests again, so peer workers
should not trigger the same rejection prematurely. Non-finite, non-positive, and over-long
values are ignored or capped. The retry delay for the failed request itself is scheduled by
the retry policy (`RetryPolicy`) using the same duration.

The schedule queue is guarded by a thread lock rather than an event loop lock: the limiter's
lifecycle matches the application process and must not depend on which event loop or thread invokes it.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
import math
import threading
import time

from linguary.application.ports.llm import (
    LLMClient,
    LLMPayload,
    LLMRequest,
    LLMResponse,
    LLMTransportError,
)

type Clock = Callable[[], float]
type Sleep = Callable[[float], Awaitable[None]]

#: Seconds per minute: rate limits are specified in requests per minute (RPM) as published by providers.
_SECONDS_PER_MINUTE = 60


def _retrieve_task_exception(task: asyncio.Task[object]) -> None:
    if not task.cancelled():
        task.exception()


class LLMRateLimiter:
    """Shared request reservations, concurrency cap, and independent provider cool-down delay."""

    def __init__(
        self,
        requests_per_minute: int | None = None,
        *,
        max_concurrency: int | None = None,
        max_hold_seconds: float = 120.0,
        clock: Clock | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._lock = threading.Lock()
        self._last_admitted: float | None = None
        self._provider_blocked_until: float | None = None
        self._max_hold_seconds = max_hold_seconds
        self._max_concurrency: int | None = None
        self._in_flight = 0
        self._concurrency_waiters: deque[list[object]] = deque()
        self._pacing_waiters: deque[asyncio.Future[None]] = deque()
        self._head_wake: asyncio.Future[None] | None = None
        self.set_limit(requests_per_minute)
        self.set_max_concurrency(max_concurrency)

    @property
    def requests_per_minute(self) -> int | None:
        with self._lock:
            return self._requests_per_minute

    @property
    def max_concurrency(self) -> int | None:
        with self._lock:
            return self._max_concurrency

    def set_limit(self, value: int | None) -> None:
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError("предел частоты запросов должен быть положительным целым числом")
        # Integer division avoids casting huge limits to float before ratio calculation.
        interval = None if value is None else _SECONDS_PER_MINUTE / value
        with self._lock:
            self._requests_per_minute = value
            self._interval = interval
            self._wake_pacing_head_locked()

    def set_max_concurrency(self, value: int | None) -> None:
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError("предел параллельных запросов должен быть положительным целым числом")
        with self._lock:
            self._max_concurrency = value
            self._wake_waiters_locked()

    async def acquire_concurrency(self) -> None:
        loop = asyncio.get_running_loop()
        with self._lock:
            if not self._concurrency_waiters and (
                self._max_concurrency is None or self._in_flight < self._max_concurrency
            ):
                self._in_flight += 1
                return
            waiter: asyncio.Future[None] = loop.create_future()
            entry: list[object] = [waiter, False]
            self._concurrency_waiters.append(entry)
        try:
            await waiter
        except BaseException:
            with self._lock:
                if entry in self._concurrency_waiters:
                    self._concurrency_waiters.remove(entry)
                elif entry[1]:
                    self._in_flight -= 1
                    self._wake_waiters_locked()
            raise

    def release_concurrency(self) -> None:
        with self._lock:
            if self._in_flight > 0:
                self._in_flight -= 1
            self._wake_waiters_locked()

    @asynccontextmanager
    async def concurrency_slot(self) -> AsyncIterator[None]:
        await self.acquire_concurrency()
        try:
            yield
        finally:
            self.release_concurrency()

    def _wake_waiters_locked(self) -> None:
        while self._concurrency_waiters and (
            self._max_concurrency is None or self._in_flight < self._max_concurrency
        ):
            entry = self._concurrency_waiters.popleft()
            waiter = entry[0]
            assert isinstance(waiter, asyncio.Future)
            if waiter.done():
                continue
            entry[1] = True
            self._in_flight += 1
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if running_loop is waiter.get_loop():
                waiter.set_result(None)
            else:
                try:
                    waiter.get_loop().call_soon_threadsafe(self._resolve_waiter, waiter)
                except RuntimeError:
                    # The waiter's loop is already closed, so the handoff never ran.
                    entry[1] = False
                    self._in_flight -= 1
                    continue

    @staticmethod
    def _resolve_waiter(waiter: asyncio.Future[None]) -> None:
        if not waiter.done():
            waiter.set_result(None)

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        ticket: asyncio.Future[None] = loop.create_future()
        with self._lock:
            self._pacing_waiters.append(ticket)
        try:
            while True:
                with self._lock:
                    is_head = bool(self._pacing_waiters) and self._pacing_waiters[0] is ticket
                    if is_head:
                        now = self._clock()
                        delay = self._pacing_deadline_locked(now) - now
                        if delay <= 0:
                            self._last_admitted = now
                            self._pacing_waiters.popleft()
                            self._head_wake = None
                            self._signal_next_pacing_locked()
                            return
                        wake = loop.create_future()
                        self._head_wake = wake
                    else:
                        wake = None
                        delay = None
                if not is_head:
                    await ticket
                    continue
                assert wake is not None and delay is not None
                await self._sleep_until_wake(delay, wake)
                with self._lock:
                    if self._head_wake is wake:
                        self._head_wake = None
        except asyncio.CancelledError:
            with self._lock:
                was_head = bool(self._pacing_waiters) and self._pacing_waiters[0] is ticket
                try:
                    self._pacing_waiters.remove(ticket)
                except ValueError:
                    pass
                if was_head:
                    self._head_wake = None
                    self._signal_next_pacing_locked()
            raise

    def _pacing_deadline_locked(self, now: float) -> float:
        if self._interval and self._last_admitted is not None:
            earliest = self._last_admitted + self._interval
        else:
            earliest = now
        blocked = now if self._provider_blocked_until is None else self._provider_blocked_until
        return max(now, earliest, blocked)

    async def _sleep_until_wake(self, delay: float, wake: asyncio.Future[None]) -> None:
        """Sleep for ``delay`` unless ``wake`` fires first. Reaps the sleep task without swallowing cancellation."""
        if wake.done():
            return
        sleep_task = asyncio.create_task(self._sleep(delay))
        try:
            await asyncio.wait(
                {sleep_task, wake},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            if not sleep_task.done():
                sleep_task.cancel()
            sleep_task.add_done_callback(_retrieve_task_exception)
        if sleep_task.done() and not sleep_task.cancelled():
            error = sleep_task.exception()
            if error is not None:
                raise error

    def _signal_next_pacing_locked(self) -> None:
        if self._pacing_waiters:
            self._signal(self._pacing_waiters[0])

    def _wake_pacing_head_locked(self) -> None:
        if self._head_wake is not None:
            self._signal(self._head_wake)

    def _signal(self, waiter: asyncio.Future[None]) -> None:
        if waiter.done():
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is waiter.get_loop():
            waiter.set_result(None)
        else:
            try:
                waiter.get_loop().call_soon_threadsafe(self._resolve_waiter, waiter)
            except RuntimeError:
                return

    def hold(self, seconds: float) -> None:
        """Extend the provider cool-down. Ignore non-numeric, non-finite, and non-positive delays.

        The extension is capped by ``max_hold_seconds``. Queued waiters keep sleeping and
        recheck when they wake, so the cool-down cannot admit more than the head.
        """
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            return
        if not math.isfinite(seconds) or seconds <= 0:
            return
        delay = min(float(seconds), self._max_hold_seconds)
        with self._lock:
            until = self._clock() + delay
            blocked = self._provider_blocked_until
            if blocked is None or until > blocked:
                self._provider_blocked_until = until


class RateLimitedLLMClient(LLMClient):
    """LLMClient decorator backed by an owned or shared rate limiter."""

    def __init__(
        self,
        *,
        inner: LLMClient,
        requests_per_minute: int | None = None,
        max_concurrency: int | None = None,
        limiter: LLMRateLimiter | None = None,
        clock: Clock | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        self._inner = inner
        self._limiter = limiter if limiter is not None else LLMRateLimiter(
            requests_per_minute,
            max_concurrency=max_concurrency,
            clock=clock,
            sleep=sleep,
        )

    @property
    def requests_per_minute(self) -> int | None:
        return self._limiter.requests_per_minute

    @property
    def max_concurrency(self) -> int | None:
        return self._limiter.max_concurrency

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        await self._limiter.acquire_concurrency()
        try:
            await self._limiter.acquire()
            try:
                return await self._inner.generate(request)
            except LLMTransportError as error:
                if error.retry_after_seconds is not None:
                    self._limiter.hold(error.retry_after_seconds)
                raise
        finally:
            self._limiter.release_concurrency()


__all__ = ["LLMRateLimiter", "RateLimitedLLMClient"]
