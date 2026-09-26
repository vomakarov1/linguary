"""Ограничитель частоты запросов к модели: общий предел для всех воркеров."""

from __future__ import annotations

import asyncio

import pytest

from linguary.adapters.llm import FakeLLMClient, FakeReply, RateLimitedLLMClient
from linguary.application.llm_dto import TranslationResultResponse
from linguary.application.ports.llm import LLMRequest, LLMTransportError
from llm_payload_factories import as_content, translation_payload


class ManualClock:
    """Время, которое идёт только тогда, когда тест или пауза его сдвигают."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        await asyncio.sleep(0)


def request() -> LLMRequest[TranslationResultResponse]:
    """Любой запрос обработки."""
    return LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")


def reply() -> FakeReply:
    """Корректный ответ обработки."""
    return FakeReply(content=as_content(translation_payload()))


def limited(
    inner: FakeLLMClient, clock: ManualClock, *, requests_per_minute: int = 60
) -> RateLimitedLLMClient:
    """Ограничитель с управляемым временем."""
    return RateLimitedLLMClient(
        inner=inner,
        requests_per_minute=requests_per_minute,
        clock=clock,
        sleep=clock.sleep,
    )


async def test_first_request_is_not_delayed() -> None:
    """Первый запрос уходит сразу: ограничивать ещё нечего."""
    clock = ManualClock()
    client = limited(FakeLLMClient.always(reply()), clock)

    await client.generate(request())

    assert clock.sleeps == []


async def test_simultaneous_requests_are_spaced_by_the_limit() -> None:
    """Одновременные запросы нескольких воркеров расходятся по времени на шаг предела."""
    clock = ManualClock()
    inner = FakeLLMClient.always(reply())
    client = limited(inner, clock, requests_per_minute=30)

    await asyncio.gather(*(client.generate(request()) for _ in range(3)))

    assert sorted(clock.sleeps) == [2.0, 4.0]
    assert inner.call_count == 3


async def test_idle_time_is_not_saved_up_for_a_burst() -> None:
    """После простоя запрос идёт без ожидания, но следующий снова держит шаг."""
    clock = ManualClock()
    client = limited(FakeLLMClient.always(reply()), clock, requests_per_minute=60)

    await client.generate(request())
    clock.now = 100.0
    await client.generate(request())
    await client.generate(request())

    assert clock.sleeps == [1.0]


async def test_retry_after_pauses_every_worker() -> None:
    """Отказ провайдера по частоте с `Retry-After` придерживает запросы всех воркеров."""
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
    """Предел в ноль запросов в минуту означал бы вечное ожидание."""
    with pytest.raises(ValueError):
        RateLimitedLLMClient(inner=FakeLLMClient(), requests_per_minute=0)
