"""Оценка оставшегося времени анализа и обработки книги (QA-13)."""

from __future__ import annotations

import pytest

from linguary.application.book_overview import estimate_remaining_seconds
from linguary.web.labels import remaining_time_label


def test_no_calls_yet_gives_no_estimate() -> None:
    """До первого вызова модели мерить нечем: оценки нет, а не ноль."""
    assert (
        estimate_remaining_seconds(
            durations_ms=(), completed_blocks=0, remaining_blocks=100
        )
        is None
    )


def test_estimate_is_median_call_times_remaining_calls() -> None:
    """Медиана длительности вызова × оставшиеся вызовы: один вызов на главу из 10 блоков."""
    seconds = estimate_remaining_seconds(
        durations_ms=(40_000, 50_000, 60_000),
        completed_blocks=30,
        remaining_blocks=100,
    )

    assert seconds == 500


def test_one_slow_retry_does_not_stretch_the_estimate() -> None:
    """Медиана, а не среднее: один затянувшийся вызов оценку не раздувает."""
    seconds = estimate_remaining_seconds(
        durations_ms=(50_000, 50_000, 900_000),
        completed_blocks=30,
        remaining_blocks=30,
    )

    assert seconds == 150


def test_parallelism_divides_the_estimate() -> None:
    """Несколько глав одновременно — во столько же раз меньше ждать."""
    seconds = estimate_remaining_seconds(
        durations_ms=(40_000,),
        completed_blocks=10,
        remaining_blocks=40,
        parallelism=4,
    )

    assert seconds == 40


def test_rate_limit_bounds_the_gain_of_parallelism() -> None:
    """Предел частоты запросов не даёт параллелизму ускорить работу сверх него."""
    seconds = estimate_remaining_seconds(
        durations_ms=(10_000,),
        completed_blocks=10,
        remaining_blocks=40,
        parallelism=8,
        requests_per_minute=6,
    )

    assert seconds == 40


def test_finished_stage_needs_no_more_time() -> None:
    """Когда оставшейся работы нет, ждать нечего."""
    assert (
        estimate_remaining_seconds(
            durations_ms=(40_000,), completed_blocks=10, remaining_blocks=0
        )
        == 0
    )


@pytest.mark.parametrize(
    ("seconds", "label"),
    [
        pytest.param(None, "Оценка оставшегося времени появится после первой главы", id="unknown"),
        pytest.param(20, "Осталось меньше минуты", id="under-a-minute"),
        pytest.param(61, "Осталось примерно 2 мин", id="rounded-up"),
        pytest.param(3600, "Осталось примерно 1 ч", id="whole-hour"),
        pytest.param(5000, "Осталось примерно 1 ч 24 мин", id="hours-and-minutes"),
    ],
)
def test_remaining_time_is_named_roughly(seconds: int | None, label: str) -> None:
    """Оценка грубая, поэтому называется «примерно» и округляется до минут вверх."""
    assert remaining_time_label(seconds) == label
