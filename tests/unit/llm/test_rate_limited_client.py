"""Model request rate limiter: shared limit across all workers."""

from __future__ import annotations

import asyncio
import math

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
    admitted: list[float] = []

    class RecordingInner:
        def __init__(self) -> None:
            self._inner = FakeLLMClient.always(reply())

        @property
        def call_count(self) -> int:
            return self._inner.call_count

        async def generate(self, req):
            admitted.append(clock())
            return await self._inner.generate(req)

    inner = RecordingInner()
    client = limited(inner, clock, requests_per_minute=30)

    await asyncio.gather(*(client.generate(request()) for _ in range(3)))

    assert admitted == [0.0, 2.0, 4.0]
    assert clock.sleeps == [2.0, 2.0]
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


async def _await_admission(task: asyncio.Task, clock: ControlledClock, at: float) -> None:
    """Wait until ``task`` is admitted at ``at``, advancing the clock when it is still in the future."""
    if at <= clock.now:
        for _ in range(20):
            if task.done():
                break
            await asyncio.sleep(0)
        await task
        return
    deadline = await clock.waiting.get()
    assert deadline == at
    clock.advance(at)
    await task


@pytest.mark.parametrize(
    ("limit", "hold_until", "head_at", "follower_at"),
    [
        (60, None, 1.0, 2.0),
        (60, 5.0, 5.0, 6.0),
        (None, None, 1.0, 1.0),
        (None, 10.0, 10.0, 10.0),
    ],
)
async def test_limit_change_recalculates_head_from_last_admit(limit, hold_until, head_at, follower_at):
    """set_limit wakes the head and spaces it from the last real admit, not a reserved slot.

    Renamed from test_limit_change_preserves_reserved_slots. Non-heads do not sleep
    until they become the head. An unlimited limit admits the head immediately unless
    a provider hold says otherwise.
    """
    clock = ControlledClock()
    limiter, inner, client = await make_recording_client(clock, 1)
    await client.generate(request())
    assert inner.calls == [0.0]
    head = asyncio.create_task(client.generate(request()))
    assert await clock.waiting.get() == 60.0
    follower = asyncio.create_task(client.generate(request()))
    third: asyncio.Task | None = None
    await asyncio.sleep(0)
    assert clock.waiting.empty()
    try:
        clock.advance(1.0)
        limiter.set_limit(limit)
        if hold_until is not None:
            limiter.hold(hold_until - clock.now)
        await _await_admission(head, clock, head_at)
        assert inner.calls[-1] == head_at
        if follower_at > clock.now:
            assert await clock.waiting.get() == follower_at
            third = asyncio.create_task(client.generate(request()))
            await asyncio.sleep(0)
            assert clock.waiting.empty()
            clock.advance(follower_at)
        await follower
        assert inner.calls[-1] == follower_at
        if third is not None:
            third_at = follower_at + (60 / limit)
            await _await_admission(third, clock, third_at)
            assert inner.calls[-1] == third_at
    finally:
        for task in (head, follower, third):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(head, follower, *( [] if third is None else [third]), return_exceptions=True)


@pytest.mark.parametrize(
    "future_slot,hold,expected",
    [(False, None, 1.4), (False, 2, 2), (True, None, 2.0), (True, 10, 10)],
)
async def test_null_to_number_uses_last_unlimited_dispatch(future_slot, hold, expected):
    """Switching on a numeric limit spaces from the last real admit, not a phantom slot."""
    clock = ControlledClock()
    limiter, inner, client = await make_recording_client(clock, None if not future_slot else 1)
    queued: list[asyncio.Task] = []
    try:
        if future_slot:
            await client.generate(request())
            for _ in range(2):
                queued.append(asyncio.create_task(client.generate(request())))
                await asyncio.sleep(0)
            assert clock.waiting.get_nowait() == 60
            assert clock.waiting.empty()
            limiter.set_limit(None)
            for _ in range(10):
                if all(task.done() for task in queued):
                    break
                await asyncio.sleep(0)
            assert [task.done() for task in queued] == [True, True]
            assert inner.calls[-1] == 0.0
            clock.advance(1)
            await client.generate(request())
            assert inner.calls[-1] == 1
            clock.advance(2)
        else:
            clock.advance(0.4)
            await client.generate(request())
            clock.advance(0.5)
        limiter.set_limit(60)
        if hold:
            limiter.hold(hold - clock.now)
        new = asyncio.create_task(client.generate(request()))
        queued.append(new)
        await asyncio.sleep(0)
        if not new.done():
            assert await clock.waiting.get() == expected
            clock.advance(expected)
            await new
        assert inner.calls[-1] == expected
    finally:
        for task in queued:
            if not task.done():
                task.cancel()
        if queued:
            await asyncio.gather(*queued, return_exceptions=True)


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
    # hold(3) admits the next call late; the one after waits a full 2s interval.
    assert clock.sleeps == [3.0, 2.0]


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


async def _settle(clock: ControlledClock, admitted: list[float]) -> None:
    """Yield until admits and newly scheduled sleeps have been idle for several turns.

    Waking a sleeper takes more than one loop turn (sleep task, then ``asyncio.wait``),
    so two identical snapshots are not yet stable.
    """
    snapshot: tuple[int, int, float] | None = None
    quiet = 0
    for _ in range(40):
        await asyncio.sleep(0)
        current = (len(admitted), clock.waiting.qsize(), clock.now)
        if current == snapshot:
            quiet += 1
            if quiet >= 5:
                return
        else:
            snapshot = current
            quiet = 0


@pytest.mark.asyncio
async def test_hold_spaces_three_waiters_at_20_21_22_and_fourth_at_23():
    """60 RPM and hold(20) at t=0 spaces waiters at 20, 21, 22, and the late fourth at 23."""
    from linguary.adapters.llm.rate_limit import LLMRateLimiter

    clock = ControlledClock()
    limiter = LLMRateLimiter(60, clock=clock, sleep=clock.sleep)
    limiter.hold(20.0)
    admitted: list[float] = []

    async def admit() -> None:
        await limiter.acquire()
        admitted.append(clock())

    waiters = [asyncio.create_task(admit()) for _ in range(3)]
    fourth: asyncio.Task | None = None
    try:
        await _settle(clock, admitted)
        clock.advance(20.0)
        await _settle(clock, admitted)
        clock.advance(20.01)
        fourth = asyncio.create_task(admit())
        await _settle(clock, admitted)
        pending = [*waiters, fourth]
        for _ in range(10):
            if all(task.done() for task in pending):
                break
            await _settle(clock, admitted)
            if clock.waiting.empty():
                continue
            deadline = clock.waiting.get_nowait()
            if deadline > clock.now:
                clock.advance(deadline)
            await _settle(clock, admitted)
        assert admitted == [20.0, 21.0, 22.0, 23.0]
    finally:
        for task in (*waiters, fourth):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*waiters, *([] if fourth is None else [fourth]), return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_leak_slot_or_block_queue():
    """Cancelling a queued or leading waiter does not consume an interval or stall the rest."""
    from linguary.adapters.llm.rate_limit import LLMRateLimiter

    clock = ControlledClock()
    limiter = LLMRateLimiter(60, clock=clock, sleep=clock.sleep)
    admitted: list[float] = []

    async def admit() -> None:
        await limiter.acquire()
        admitted.append(clock())

    await admit()
    first = asyncio.create_task(admit())
    middle: asyncio.Task | None = None
    last: asyncio.Task | None = None
    head: asyncio.Task | None = None
    follower: asyncio.Task | None = None
    try:
        assert await clock.waiting.get() == 1.0
        middle = asyncio.create_task(admit())
        last = asyncio.create_task(admit())
        await asyncio.sleep(0)
        assert clock.waiting.empty()

        middle.cancel()
        with pytest.raises(asyncio.CancelledError):
            await middle

        clock.advance(1.0)
        await first
        assert admitted == [0.0, 1.0]
        assert await clock.waiting.get() == 2.0
        clock.advance(2.0)
        await last
        assert admitted == [0.0, 1.0, 2.0]

        head = asyncio.create_task(admit())
        assert await clock.waiting.get() == 3.0
        follower = asyncio.create_task(admit())
        await asyncio.sleep(0)
        assert clock.waiting.empty()
        head.cancel()
        with pytest.raises(asyncio.CancelledError):
            await head
        assert await clock.waiting.get() == 3.0
        clock.advance(3.0)
        await follower
        assert admitted == [0.0, 1.0, 2.0, 3.0]
    finally:
        for task in (first, middle, last, head, follower):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            first,
            *([middle] if middle is not None else []),
            *([last] if last is not None else []),
            *([head] if head is not None else []),
            *([follower] if follower is not None else []),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_hold_caps_huge_and_ignores_infinite_retry_after():
    """Non-finite, non-positive, and non-numeric Retry-After values do not freeze the queue."""
    from linguary.adapters.llm.rate_limit import LLMRateLimiter

    clock = ControlledClock()
    limiter = LLMRateLimiter(None, clock=clock, sleep=clock.sleep)
    limiter.hold(math.inf)
    immediate = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0)
    if not immediate.done():
        queued = None if clock.waiting.empty() else clock.waiting.get_nowait()
        raise AssertionError(f"Retry-After froze acquire at deadline {queued}")
    await immediate

    for ignored in (-math.inf, math.nan, 0, -3, "20", None, True):
        limiter.hold(ignored)  # type: ignore[arg-type]
    still_open = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0)
    assert still_open.done()
    await still_open
    assert clock.waiting.empty()

    limiter.hold(10**9)
    capped = asyncio.create_task(limiter.acquire())
    assert await clock.waiting.get() == 120.0
    clock.advance(120.0)
    await capped

    narrow = LLMRateLimiter(None, max_hold_seconds=10, clock=clock, sleep=clock.sleep)
    narrow.hold(10**6)
    narrowed = asyncio.create_task(narrow.acquire())
    assert await clock.waiting.get() == 130.0
    clock.advance(130.0)
    await narrowed

    inner = FakeLLMClient.replying([LLMTransportError("429", retry_after_seconds=math.inf), reply()])
    client = RateLimitedLLMClient(
        inner=inner, requests_per_minute=None, clock=clock, sleep=clock.sleep
    )
    with pytest.raises(LLMTransportError):
        await client.generate(request())
    nxt = asyncio.create_task(client.generate(request()))
    await asyncio.sleep(0)
    assert nxt.done()
    await nxt

    huge = FakeLLMClient.replying([LLMTransportError("429", retry_after_seconds=10**9), reply()])
    capped_client = RateLimitedLLMClient(
        inner=huge, requests_per_minute=None, clock=clock, sleep=clock.sleep
    )
    with pytest.raises(LLMTransportError):
        await capped_client.generate(request())
    waiting = asyncio.create_task(capped_client.generate(request()))
    assert await clock.waiting.get() == clock.now + 120.0
    clock.advance(clock.now + 120.0)
    await waiting


@pytest.mark.asyncio
async def test_set_limit_recalculates_waiting_head_immediately():
    """A faster limit wakes the sleeping head now instead of leaving the old deadline in place."""
    from linguary.adapters.llm.rate_limit import LLMRateLimiter

    clock = ControlledClock()
    limiter = LLMRateLimiter(1, clock=clock, sleep=clock.sleep)
    admitted: list[float] = []

    async def admit() -> None:
        await limiter.acquire()
        admitted.append(clock())

    await admit()
    head = asyncio.create_task(admit())
    follower: asyncio.Task | None = None
    try:
        assert await clock.waiting.get() == 60.0
        follower = asyncio.create_task(admit())
        await asyncio.sleep(0)
        assert clock.waiting.empty()
        clock.advance(10.0)
        limiter.set_limit(60)
        for _ in range(20):
            if head.done():
                break
            await asyncio.sleep(0)
        assert head.done()
        await head
        assert admitted == [0.0, 10.0]
        assert await clock.waiting.get() == 11.0
        clock.advance(11.0)
        await follower
        assert admitted == [0.0, 10.0, 11.0]
    finally:
        for task in (head, follower):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(head, *([] if follower is None else [follower]), return_exceptions=True)


@pytest.mark.asyncio
async def test_negative_monotonic_clock_and_closed_loop_wakeup():
    """A negative clock is not blocked until 0, and a closed loop does not leak a concurrency slot."""
    from linguary.adapters.llm.rate_limit import LLMRateLimiter

    clock = ManualClock()
    clock.now = -5.0
    limiter = LLMRateLimiter(60, clock=clock, sleep=clock.sleep)
    await limiter.acquire()
    assert clock.sleeps == []
    assert clock.now == -5.0
    await limiter.acquire()
    assert clock.sleeps == [1.0]
    assert clock.now == -4.0

    capped = LLMRateLimiter(None, max_concurrency=1)
    await capped.acquire_concurrency()
    dead = asyncio.new_event_loop()
    parked: list[asyncio.Task] = []

    def queue_on_dead_loop() -> None:
        async def queue_dead_waiter() -> None:
            task = asyncio.create_task(capped.acquire_concurrency())
            await asyncio.sleep(0)
            parked.append(task)

        dead.run_until_complete(queue_dead_waiter())

    live: asyncio.Task | None = None
    extra: asyncio.Task | None = None
    try:
        await asyncio.to_thread(queue_on_dead_loop)
        dead.set_exception_handler(lambda *_args: None)
        dead.close()
        live = asyncio.create_task(capped.acquire_concurrency())
        await asyncio.sleep(0)
        assert not live.done()
        capped.release_concurrency()
        await asyncio.wait_for(live, timeout=1)
        extra = asyncio.create_task(capped.acquire_concurrency())
        await asyncio.sleep(0)
        assert not extra.done()
        capped.release_concurrency()
        await asyncio.wait_for(extra, timeout=1)
    finally:
        if not dead.is_closed():
            dead.set_exception_handler(lambda *_args: None)
            dead.close()
        for task in (live, extra):
            if task is not None and not task.done():
                task.cancel()
        pending = [task for task in (live, extra) if task is not None]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_pacing_sleep_error_does_not_leave_a_dead_head():
    """A non-cancel sleep failure drops the FIFO ticket and lets the next waiter admit."""
    from linguary.adapters.llm.rate_limit import LLMRateLimiter

    clock = ManualClock()
    failed = False

    async def sleep(seconds: float) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("sleep failed")
        await clock.sleep(seconds)

    limiter = LLMRateLimiter(60, clock=clock, sleep=sleep)
    await limiter.acquire()
    with pytest.raises(RuntimeError, match="sleep failed"):
        await limiter.acquire()
    await asyncio.wait_for(limiter.acquire(), timeout=1)


@pytest.mark.asyncio
async def test_closed_pacing_loop_does_not_block_the_next_waiter():
    """A pacing head whose loop is closed is dropped instead of stalling the queue."""
    from linguary.adapters.llm.rate_limit import LLMRateLimiter

    clock = ManualClock()
    dead = asyncio.new_event_loop()

    async def sleep(seconds: float) -> None:
        if asyncio.get_running_loop() is dead:
            await asyncio.Event().wait()
        await clock.sleep(seconds)

    limiter = LLMRateLimiter(60, clock=clock, sleep=sleep)
    await limiter.acquire()
    parked: list[asyncio.Task] = []

    def queue_on_dead_loop() -> None:
        async def queue_dead_waiter() -> None:
            task = asyncio.create_task(limiter.acquire())
            for _ in range(5):
                await asyncio.sleep(0)
            parked.append(task)

        dead.run_until_complete(queue_dead_waiter())

    live: asyncio.Task | None = None
    try:
        await asyncio.to_thread(queue_on_dead_loop)
        assert parked and not parked[0].done()
        dead.set_exception_handler(lambda *_args: None)
        dead.close()
        live = asyncio.create_task(limiter.acquire())
        limiter.set_limit(60)
        await asyncio.wait_for(live, timeout=1)
    finally:
        if not dead.is_closed():
            dead.set_exception_handler(lambda *_args: None)
            dead.close()
        if live is not None and not live.done():
            live.cancel()
            await asyncio.gather(live, return_exceptions=True)


@pytest.mark.asyncio
async def test_closed_pacing_loop_does_not_block_without_set_limit_or_with_done_head_wake():
    """Closing the head loop without set_limit(), even with a resolved _head_wake, never stalls acquire()."""
    from linguary.adapters.llm.rate_limit import LLMRateLimiter

    clock = ManualClock()
    dead = asyncio.new_event_loop()

    async def sleep(seconds: float) -> None:
        if asyncio.get_running_loop() is dead:
            await asyncio.Event().wait()
        await clock.sleep(seconds)

    limiter = LLMRateLimiter(60, clock=clock, sleep=sleep)
    await limiter.acquire()
    parked: list[asyncio.Task] = []

    def queue_on_dead_loop() -> None:
        async def queue_dead_waiter() -> None:
            task = asyncio.create_task(limiter.acquire())
            for _ in range(5):
                await asyncio.sleep(0)
            if limiter._head_wake is not None and not limiter._head_wake.done():
                limiter._head_wake.set_result(None)
            parked.append(task)

        dead.run_until_complete(queue_dead_waiter())

    try:
        await asyncio.to_thread(queue_on_dead_loop)
        assert parked and not parked[0].done()
        dead.set_exception_handler(lambda *_args: None)
        dead.close()
        await asyncio.wait_for(limiter.acquire(), timeout=1)
    finally:
        if not dead.is_closed():
            dead.set_exception_handler(lambda *_args: None)
            dead.close()

