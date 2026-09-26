"""Параллелизм обработки и предел частоты в настоящем composition root (PRD § 25)."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from processing_scenario import BOOK_ID, SOURCE_TEXT, store_processing_book, translated_reply

from linguary.adapters.config import (
    AppSettings,
    CheckSettings,
    LLMSettings,
    StorageSettings,
    WorkerPoolSettings,
)
from linguary.adapters.llm import FakeLLMClient
from linguary.adapters.runtime import WorkerSettings
from linguary.application.ports.llm import LLMClient, LLMPayload, LLMRequest, LLMResponse
from linguary.composition import Application, build_application, drain_tasks
from linguary.domain import BookStatus, TaskStatus

#: Шесть глав по два абзаца: работы хватает на несколько воркеров.
SIX_CHAPTERS = tuple((SOURCE_TEXT, SOURCE_TEXT) for _ in range(6))


@dataclass(slots=True)
class TimedLLMClient(LLMClient):
    """Клиент, запоминающий моменты вызовов модели."""

    inner: FakeLLMClient
    started: list[float] = field(default_factory=list)

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        self.started.append(time.monotonic())
        return await self.inner.generate(request)


def settings(
    tmp_path: Path, *, concurrency: int = 1, requests_per_minute: int | None = None
) -> AppSettings:
    """Настройки приложения с заданным параллелизмом и пределом частоты."""
    return AppSettings(
        llm=LLMSettings(
            model="fake-model",
            api_key="test-key",
            requests_per_minute=requests_per_minute,
        ),
        checks=CheckSettings(),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
        worker=WorkerPoolSettings(concurrency=concurrency),
    )


@pytest.fixture
def build(
    tmp_path: Path, processing_task_ids: Callable[[], str]
) -> Iterator[Callable[..., Application]]:
    """Сборка приложения; всё собранное закрывается после теста."""
    built: list[Application] = []

    def make(app_settings: AppSettings, client: LLMClient, **overrides: object) -> Application:
        application = build_application(
            settings=app_settings,
            llm_client=client,
            task_id_factory=processing_task_ids,
            **overrides,  # type: ignore[arg-type]
        )
        built.append(application)
        return application

    yield make
    for application in built:
        application.dispose()


def test_worker_concurrency_comes_from_settings(
    tmp_path: Path, build: Callable[..., Application]
) -> None:
    """Параллелизм воркера задаётся секцией `[worker]`, а не кодом (PRD § 25)."""
    application = build(settings(tmp_path, concurrency=3), FakeLLMClient())

    assert application.worker_pool.concurrency == 3


def test_explicit_worker_settings_override_configuration(
    tmp_path: Path, build: Callable[..., Application]
) -> None:
    """Явно переданные настройки воркера важнее файла — так их задают тесты и CLI."""
    application = build(
        settings(tmp_path, concurrency=3),
        FakeLLMClient(),
        worker_settings=WorkerSettings(concurrency=2),
    )

    assert application.worker_pool.concurrency == 2


async def test_configured_parallel_processing_finishes_the_book(
    tmp_path: Path, build: Callable[..., Application]
) -> None:
    """Приложение с параллелизмом из настроек доводит обработку книги до конца."""
    application = build(
        settings(tmp_path, concurrency=4),
        FakeLLMClient.always(translated_reply()),
    )
    scenario = store_processing_book(application.unit_of_work, chapters=SIX_CHAPTERS)
    await application.processing.start(BOOK_ID)

    finished = await drain_tasks(application)

    assert len(finished) == len(SIX_CHAPTERS)
    assert all(task.status is TaskStatus.SUCCEEDED for task in finished)
    with application.unit_of_work.transaction() as transaction:
        book = transaction.books.get(BOOK_ID)
        results = transaction.results.list_for_run(scenario.run_id)
    assert book is not None
    assert book.status is BookStatus.READY_TO_EXPORT
    assert len(results) == 2 * len(SIX_CHAPTERS)


async def test_requests_per_minute_spaces_calls_of_all_workers(
    tmp_path: Path, build: Callable[..., Application]
) -> None:
    """Предел частоты общий для всех воркеров: вызовы модели идут не чаще заданного шага."""
    client = TimedLLMClient(inner=FakeLLMClient.always(translated_reply()))
    application = build(
        settings(tmp_path, concurrency=3, requests_per_minute=600), client
    )
    store_processing_book(application.unit_of_work, chapters=SIX_CHAPTERS[:3])
    await application.processing.start(BOOK_ID)

    await drain_tasks(application)

    assert len(client.started) == 3
    gaps = [later - earlier for earlier, later in zip(client.started, client.started[1:])]
    assert all(gap >= 0.09 for gap in gaps), gaps
