"""Model request rate limiter shared across all workers in a process.

Parallel processing (PRD § 25) multiplies provider request frequency, while providers
enforce requests-per-minute limits: exceeding limits results in HTTP `429` errors and wastes
task retry attempts on waiting. This decorator over `LLMClient` maintains an interval of
`60 / requests_per_minute` seconds between requests, regardless of worker count.

Interval pacing is uniform without token bucket bursting: idle periods do not accumulate
request credits, as providers evaluate rates using sliding windows and reject bursts anyway.

A rate limit rejection carrying a `Retry-After` header delays the next available dispatch slot
for all workers: the provider announced when it will accept requests again, so peer workers
should not trigger the same rejection prematurely. The retry delay for the failed request itself
is scheduled by the retry policy (`RetryPolicy`) using the same duration.

The schedule queue is guarded by a thread lock rather than an event loop lock: the limiter's
lifecycle matches the application process and must not depend on which event loop or thread invokes it.
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

#: Seconds per minute: rate limits are specified in requests per minute (RPM) as published by providers.
_SECONDS_PER_MINUTE = 60


class LLMRateLimiter:
    """Shared request reservations and independent provider cool-down delay."""

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
        # Integer division avoids casting huge limits to float before ratio calculation.
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
        """Extends provider cool-down period without altering numerical reservation slots."""
        with self._lock:
            self._provider_blocked_until = max(
                self._provider_blocked_until, self._clock() + seconds
            )


class RateLimitedLLMClient(LLMClient):
    """LLMClient decorator backed by an owned or shared rate limiter."""

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
