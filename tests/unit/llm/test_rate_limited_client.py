"""Model request rate limiter: shared limit across all workers."""

from __future__ import annotations

import asyncio

import pytest

from linguary.adapters.llm import FakeLLMClient, FakeReply, RateLimitedLLMClient
from linguary.application.llm_dto import TranslationResultResponse
from linguary.application.ports.llm import LLMRequest, LLMTransportError
from llm_payload_factories import as_content, translation_payload


class ManualClock:
    """Time that advances only when explicitly stepped by a test or sleep."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        await asyncio.sleep(0)
        self.now += seconds


def request() -> LLMRequest[TranslationResultResponse]:
    """Any processing request."""
    return LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")


def reply() -> FakeReply:
    """Valid processing reply."""
    return FakeReply(content=as_content(translation_payload()))


def limited(
    inner: FakeLLMClient, clock: ManualClock, *, requests_per_minute: int = 60
) -> RateLimitedLLMClient:
    """Rate limiter with controlled time."""
    return RateLimitedLLMClient(
        inner=inner,
        requests_per_minute=requests_per_minute,
        clock=clock,
        sleep=clock.sleep,
    )


async def test_first_request_is_not_delayed() -> None:
    """First request is dispatched immediately: nothing to throttle yet."""
    clock = ManualClock()
    client = limited(FakeLLMClient.always(reply()), clock)

    await client.generate(request())

    assert clock.sleeps == []


async def test_simultaneous_requests_are_spaced_by_the_limit() -> None:
    """Simultaneous requests from multiple workers are spaced apart by the rate step."""
    clock = ManualClock()
    inner = FakeLLMClient.always(reply())
    client = limited(inner, clock, requests_per_minute=30)

    await asyncio.gather(*(client.generate(request()) for _ in range(3)))

    assert sorted(clock.sleeps) == [2.0, 4.0]
    assert inner.call_count == 3


async def test_idle_time_is_not_saved_up_for_a_burst() -> None:
    """After idle time, request proceeds without delay, but the next resumes spacing."""
    clock = ManualClock()
    client = limited(FakeLLMClient.always(reply()), clock, requests_per_minute=60)

    await client.generate(request())
    clock.now = 100.0
    await client.generate(request())
    await client.generate(request())

    assert clock.sleeps == [1.0]


async def test_retry_after_pauses_every_worker() -> None:
    """Provider 429 response with Retry-After pauses requests for all workers."""
    clock = ManualClock()
    inner = FakeLLMClient.replying(
        [LLMTransportError("429", retry_after_seconds=20.0), reply()]
    )
    client = limited(inner, clock, requests_per_minute=60)

    with pytest.raises(LLMTransportError):
        await client.generate(request())
    await client.generate(request())

    assert clock.sleeps == [20.0]


def test_limit_must_be_positive() -> None:
    """A limit of zero requests per minute would mean waiting indefinitely."""
    with pytest.raises(ValueError):
        RateLimitedLLMClient(inner=FakeLLMClient(), requests_per_minute=0)


class ControlledClock:
    """Waits are resolved only by explicitly advancing the clock."""

    def __init__(self):
        self.now = 0.0
        self.waiters = []
        self.waiting = asyncio.Queue()

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        future = asyncio.get_running_loop().create_future()
        self.waiters.append((self.now + seconds, future))
        self.waiting.put_nowait(self.now + seconds)
        await future

    def advance(self, now):
        self.now = now
        for deadline, future in self.waiters:
            if deadline <= now and not future.done():
                future.set_result(None)


class RecordingClient:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.inner = FakeLLMClient.always(reply())

    async def generate(self, request):
        self.calls.append(self.clock())
        return await self.inner.generate(request)


async def make_recording_client(clock, limit):
    from linguary.adapters.llm.rate_limit import LLMRateLimiter
    limiter = LLMRateLimiter(limit, clock=clock, sleep=clock.sleep)
    inner = RecordingClient(clock)
    return limiter, inner, RateLimitedLLMClient(inner=inner, limiter=limiter)


@pytest.mark.parametrize('hold', [None, 70.0])
async def test_60_to_1_dispatches_at_60(hold):
    clock = ControlledClock()
    limiter, inner, client = await make_recording_client(clock, 60)
    await client.generate(request())
    clock.advance(0.1)
    limiter.set_limit(1)
    if hold:
        limiter.hold(hold - clock.now)
    pending = asyncio.create_task(client.generate(request()))
    deadline = await clock.waiting.get()
    assert deadline == (hold or 60.0)
    clock.advance(deadline)
    await pending
    assert inner.calls == [0, deadline]


@pytest.mark.parametrize('limit,hold,expected', [(60, None, 121), (60, 130, 130), (None, None, 1), (None, 10, 10)])
async def test_limit_change_preserves_reserved_slots(limit, hold, expected):
    clock = ControlledClock()
    limiter, inner, client = await make_recording_client(clock, 1)
    await client.generate(request())
    old = []
    for deadline in (60, 120):
        old.append(asyncio.create_task(client.generate(request())))
        assert await clock.waiting.get() == deadline
    clock.advance(1)
    limiter.set_limit(limit)
    if hold:
        limiter.hold(hold - clock.now)
    new = asyncio.create_task(client.generate(request()))
    if expected > 1:
        assert await clock.waiting.get() == expected
        clock.advance(expected)
    await new
    assert inner.calls[-1] == expected
    clock.advance(max(120, hold or 0, expected))
    await asyncio.gather(*old)
    assert len(inner.calls) == 4


@pytest.mark.parametrize('future_slot,hold,expected', [(False, None, 1.4), (False, 2, 2), (True, None, 121), (True, 130, 130)])
async def test_null_to_number_uses_last_unlimited_dispatch(future_slot, hold, expected):
    clock = ControlledClock()
    limiter, inner, client = await make_recording_client(clock, None if not future_slot else 1)
    old = []
    if future_slot:
        await client.generate(request())
        for deadline in (60, 120):
            old.append(asyncio.create_task(client.generate(request())))
            assert await clock.waiting.get() == deadline
        limiter.set_limit(None)
    clock.advance(1 if future_slot else 0.4)
    await client.generate(request())
    clock.advance(2 if future_slot else 0.5)
    limiter.set_limit(60)
    if hold:
        limiter.hold(hold - clock.now)
    new = asyncio.create_task(client.generate(request()))
    assert await clock.waiting.get() == expected
    clock.advance(expected)
    await new
    assert inner.calls[-1] == expected
    await asyncio.gather(*old)


async def test_hold_during_wait_is_rechecked_before_dispatch():
    clock = ControlledClock()
    limiter, inner, client = await make_recording_client(clock, 1)
    await client.generate(request())
    pending = asyncio.create_task(client.generate(request()))
    assert await clock.waiting.get() == 60
    clock.advance(50)
    limiter.hold(40)
    clock.advance(60)
    assert await clock.waiting.get() == 90
    clock.advance(80)
    limiter.hold(20)
    clock.advance(90)
    assert await clock.waiting.get() == 100
    clock.advance(100)
    await pending
    assert inner.calls == [0, 100]


async def test_two_clients_share_one_limit():
    clock = ControlledClock()
    limiter, first, client = await make_recording_client(clock, 60)
    second = RecordingClient(clock)
    other = RateLimitedLLMClient(inner=second, limiter=limiter)
    await client.generate(request())
    pending = asyncio.create_task(other.generate(request()))
    assert await clock.waiting.get() == 1
    clock.advance(1)
    await pending
    assert first.calls == [0] and second.calls == [1]


@pytest.mark.parametrize('value', [True, 1.0, 0, -1, '1'])
def test_changed_limit_rejects_invalid_values(value):
    from linguary.adapters.llm.rate_limit import LLMRateLimiter
    limiter = LLMRateLimiter(60)
    with pytest.raises(ValueError):
        limiter.set_limit(value)
    assert limiter.requests_per_minute == 60


@pytest.mark.parametrize("rate", [10**309, 10**400])
@pytest.mark.parametrize("change_limit", [False, True])
async def test_large_positive_integer_limit_is_applied(rate, change_limit):
    from linguary.adapters.llm.rate_limit import LLMRateLimiter

    clock = ManualClock()
    limiter = LLMRateLimiter(60 if change_limit else rate, clock=clock, sleep=clock.sleep)
    if change_limit:
        limiter.set_limit(rate)
    assert limiter.requests_per_minute == rate
    await limiter.acquire()
    await limiter.acquire()
    assert clock.now == 60 / rate
    assert clock.sleeps == ([] if rate == 10**400 else [60 / rate])


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "60"])
async def test_invalid_limit_preserves_state(invalid):
    from linguary.adapters.llm.rate_limit import LLMRateLimiter

    clock = ManualClock()
    limiter = LLMRateLimiter(30, clock=clock, sleep=clock.sleep)
    await limiter.acquire()
    limiter.hold(3)
    with pytest.raises(ValueError):
        limiter.set_limit(invalid)
    assert limiter.requests_per_minute == 30
    await limiter.acquire()
    await limiter.acquire()
    assert clock.sleeps == [3, 1]


async def test_rate_limited_client_caps_concurrent_in_flight_requests() -> None:
    """Total concurrent in-flight inner.generate calls across shared limiter never exceed max_concurrency."""
    from linguary.adapters.llm.rate_limit import LLMRateLimiter

    limiter = LLMRateLimiter(None, max_concurrency=2)
    in_flight = 0
    max_observed = 0
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingInnerClient:
        def __init__(self) -> None:
            self.completed = 0

        async def generate(self, req):
            nonlocal in_flight, max_observed
            in_flight += 1
            max_observed = max(max_observed, in_flight)
            if in_flight == 2:
                started.set()
            try:
                await release.wait()
                return await FakeLLMClient.always(reply()).generate(req)
            finally:
                in_flight -= 1
                self.completed += 1

    inner_a = BlockingInnerClient()
    inner_b = BlockingInnerClient()
    client_a = RateLimitedLLMClient(inner=inner_a, limiter=limiter)  # type: ignore[arg-type]
    client_b = RateLimitedLLMClient(inner=inner_b, limiter=limiter)  # type: ignore[arg-type]

    tasks = [
        asyncio.create_task(client_a.generate(request())),
        asyncio.create_task(client_b.generate(request())),
        asyncio.create_task(client_a.generate(request())),
        asyncio.create_task(client_b.generate(request())),
        asyncio.create_task(client_a.generate(request())),
    ]
    await started.wait()
    await asyncio.sleep(0)
    assert in_flight == 2
    assert max_observed == 2
    assert inner_a.completed + inner_b.completed == 0

    release.set()
    await asyncio.gather(*tasks)
    assert max_observed == 2
    assert in_flight == 0
    assert inner_a.completed + inner_b.completed == 5


async def test_concurrency_slot_acquired_before_rpm_reservation() -> None:
    """Waiting for a concurrency slot must not reserve an RPM slot in the past."""
    from linguary.adapters.llm.rate_limit import LLMRateLimiter

    clock = ControlledClock()
    limiter = LLMRateLimiter(60, max_concurrency=1, clock=clock, sleep=clock.sleep)
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    call_times: list[float] = []

    class SlowInnerClient:
        async def generate(self, req):
            call_times.append(clock())
            if len(call_times) == 1:
                first_entered.set()
                await release_first.wait()
            return await FakeLLMClient.always(reply()).generate(req)

    client = RateLimitedLLMClient(inner=SlowInnerClient(), limiter=limiter)  # type: ignore[arg-type]

    first = asyncio.create_task(client.generate(request()))
    await first_entered.wait()
    assert call_times == [0.0]

    # Second and third requests arrive at t=0.0 while first is still in flight.
    second = asyncio.create_task(client.generate(request()))
    third = asyncio.create_task(client.generate(request()))
    await asyncio.sleep(0)

    # Neither second nor third has entered acquire() yet, so no RPM sleep has been scheduled.
    assert clock.waiting.empty()

    # First finishes at t=5.0.
    clock.advance(5.0)
    release_first.set()
    await first
    await second
    assert call_times == [0.0, 5.0]

    # Third acquires concurrency slot at t=5.0 and reserves RPM slot at 5.0 + 1.0 = 6.0 (not 2.0).
    deadline = await clock.waiting.get()
    assert deadline == 6.0
    clock.advance(6.0)
    await third
    assert call_times == [0.0, 5.0, 6.0]


async def test_system_settings_updates_max_concurrency() -> None:
    """SystemSettingsService._apply updates max_concurrency on the shared LLMRateLimiter."""
    from types import SimpleNamespace
    from linguary.adapters.config.settings import AppSettings, LLMSettings, WorkerPoolSettings
    from linguary.adapters.config.system_settings import SystemSettingsUpdate
    from linguary.adapters.llm.rate_limit import LLMRateLimiter
    from linguary.composition.system_settings import SystemSettingsService

    initial = AppSettings(
        llm=LLMSettings(model="gpt-4o", api_key="secret"),
        worker=WorkerPoolSettings(concurrency=1),
    )
    generation = SimpleNamespace(
        settings=initial,
        configured_client=SimpleNamespace(aclose=lambda: asyncio.sleep(0)),
    )

    class FakeRuntime:
        def __init__(self) -> None:
            self.revision = 0
            self.initial_settings = initial
            self.current = generation

        def prepare(self, candidate: AppSettings):
            return SimpleNamespace(
                settings=candidate,
                configured_client=SimpleNamespace(aclose=lambda: asyncio.sleep(0)),
            )

        def publish(self, gen, revision: int) -> None:
            self.current = gen
            self.revision = revision

    class FakeStore:
        def load(self):
            return None

        def save(self, overrides, api_key, *, expected_revision: int, proxy_url=None):
            return SimpleNamespace(revision=expected_revision + 1)

        def reset(self, *, expected_revision: int):
            return SimpleNamespace(revision=expected_revision + 1)

    class FakeWorkerPool:
        def __init__(self) -> None:
            self.concurrency: int | None = None

        def set_concurrency(self, value: int) -> None:
            self.concurrency = value

    limiter = LLMRateLimiter(60, max_concurrency=1)
    worker_pool = FakeWorkerPool()
    service = SystemSettingsService(
        runtime=FakeRuntime(),  # type: ignore[arg-type]
        store=FakeStore(),  # type: ignore[arg-type]
        limiter=limiter,
        worker_pool=worker_pool,  # type: ignore[arg-type]
    )

    await service.save(
        SystemSettingsUpdate(
            provider="openai_compatible",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            requests_per_minute=120,
            concurrency=4,
        ),
        expected_revision=0,
    )

    assert limiter.requests_per_minute == 120
    assert limiter.max_concurrency == 4
    assert worker_pool.concurrency == 4

