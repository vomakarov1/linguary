"""Фоновый воркер веб-процесса: задачи выполняются сами (PRD § 17-18.1).

Локальное приложение обязано доводить загруженную книгу до результата без второго
запущенного терминала, поэтому веб-процесс держит воркер очереди. Здесь проверяется
именно это, а не сама очередь: она покрыта тестами задач фазы 2.

Ожидание результата ограничено по времени: тест не должен висеть, если воркер не
запустился.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import pytest
from conftest import FIXTURE_DIRECTORY
from fastapi.testclient import TestClient
from web_scenario import store_book

from linguary.composition import Application
from linguary.domain import BookStatus, ProcessingTask, TaskStatus, TaskType
from linguary.web import create_app

#: Маленькая синтетическая книга набора фикстур (PRD § 28.1).
SYNTHETIC_BOOK = FIXTURE_DIRECTORY / "synthetic_story.epub"

#: Сколько ждать работы фонового воркера и как часто опрашивать результат.
_TIMEOUT_SECONDS = 10.0
_POLL_SECONDS = 0.02


@pytest.fixture
def working_client(application: Application) -> Iterator[TestClient]:
    """Клиент приложения с работающим фоновым воркером."""
    app = create_app(application=application, run_background_worker=True)
    with TestClient(app, base_url=application.settings.server.base_url) as client:
        yield client


def wait_until[T](probe: Callable[[], T | None]) -> T:
    """Ждёт результата фоновой работы и возвращает его."""
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        result = probe()
        if result is not None:
            return result
        time.sleep(_POLL_SECONDS)
    raise AssertionError("фоновый воркер не довёл задачу до результата")


def test_uploaded_book_is_imported_without_any_further_request(
    working_client: TestClient,
) -> None:
    """Загруженная книга импортируется сама: веб-процесс исполняет свои задачи."""
    with SYNTHETIC_BOOK.open("rb") as file:
        response = working_client.post(
            "/books/upload",
            files={"file": (SYNTHETIC_BOOK.name, file, "application/epub+zip")},
            data={"source_language": "en", "target_language": "ru"},
            follow_redirects=False,
        )
    book_id = response.headers["location"].rsplit("/", 1)[-1]

    def imported() -> dict[str, object] | None:
        detail = working_client.get(f"/books/{book_id}").json()
        failures = detail["progress"]["failures"]
        assert not failures, f"задача импорта неуспешна: {failures}"
        return detail if detail["book"]["status"] == BookStatus.IMPORTED.value else None

    assert wait_until(imported)["progress"]["total_sections"] > 0


def test_interrupted_task_is_recovered_on_startup(
    application: Application,
) -> None:
    """Задача, оставшаяся `running` после аварийного завершения, помечается прерванной.

    Её уже никто не исполняет, поэтому старт приложения переводит её в `failed`
    с возможностью повтора (ALG-02, PRD § 17).
    """
    scenario = store_book(application, analysed_chapters=())
    with application.unit_of_work.transaction() as uow:
        uow.tasks.add(
            ProcessingTask(
                id="task-interrupted",
                book_id=scenario.book_id,
                task_type=TaskType.ANALYZE_CHAPTER,
                scope_id=scenario.section(0).id,
            ).start()
        )

    app = create_app(application=application, run_background_worker=True)
    with TestClient(app):

        def recovered() -> ProcessingTask | None:
            with application.unit_of_work.transaction() as uow:
                task = uow.tasks.get("task-interrupted")
            return task if task is not None and task.status is TaskStatus.FAILED else None

        task = wait_until(recovered)

    assert task.last_error is not None
    assert task.last_error.detail == "Interrupted"
    assert task.is_retryable
