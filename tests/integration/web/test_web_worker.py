"""Background web process worker: tasks run automatically (PRD § 17-18.1).

The local application must bring an uploaded book to completion without requiring a second
running terminal, so the web process maintains a task queue worker. This test focuses
specifically on this capability rather than the task queue itself (which is covered by Phase 2 tests).

Waiting for results is bounded by a timeout so the test will not hang if the worker fails to start.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import pytest
from conftest import FIXTURE_DIRECTORY
from fastapi.testclient import TestClient
from web_scenario import store_book

from linguary.adapters.runtime import ExecutorLock
from linguary.composition import Application, drain_tasks, run_worker
from linguary.domain import BookStatus, ProcessingTask, TaskStatus, TaskType
from linguary.web import create_app

#: Small synthetic book from the fixture suite (PRD § 28.1).
SYNTHETIC_BOOK = FIXTURE_DIRECTORY / "synthetic_story.epub"

#: How long to wait for the background worker and polling frequency.
_TIMEOUT_SECONDS = 10.0
_POLL_SECONDS = 0.02


@pytest.fixture
def working_client(application: Application) -> Iterator[TestClient]:
    """Application client with running background worker."""
    app = create_app(application=application, run_background_worker=True)
    with TestClient(app, base_url=application.settings.server.base_url) as client:
        yield client


def store_running_task(application: Application) -> str:
    """A `running` task as left by an executor mid-execution."""
    scenario = store_book(application, analysed_chapters=())
    with application.unit_of_work.transaction() as uow:
        uow.tasks.add(
            ProcessingTask(
                id="task-running",
                book_id=scenario.book_id,
                task_type=TaskType.ANALYZE_CHAPTER,
                scope_id=scenario.section(0).id,
            ).start()
        )
    return "task-running"


def task_status(application: Application, task_id: str) -> TaskStatus:
    """Current task status in the database."""
    with application.unit_of_work.transaction() as uow:
        task = uow.tasks.get(task_id)
    assert task is not None
    return task.status


def wait_until[T](probe: Callable[[], T | None]) -> T:
    """Waits for background task result and returns it."""
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        result = probe()
        if result is not None:
            return result
        time.sleep(_POLL_SECONDS)
    raise AssertionError("background worker did not complete the task")


def test_uploaded_book_is_imported_without_any_further_request(
    working_client: TestClient,
) -> None:
    """Uploaded book is imported automatically: web process executes its tasks."""
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
    """A task remaining `running` after an abnormal exit is marked interrupted.

    No worker is executing it anymore, so application startup transitions it to `failed`
    with retry enabled (ALG-02, PRD § 17).
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


async def test_worker_started_next_to_another_executor_keeps_its_running_task(
    application: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker started alongside an active executor does not mark its running task interrupted.

    Otherwise, the user could retry it while it is still running, sending the same fragment
    to the model twice.
    """
    task_id = store_running_task(application)
    other = ExecutorLock(application.paths.executor_lock)
    other.join()
    other.share()

    async def stop_at_once() -> None:
        return None

    monkeypatch.setattr(application.worker_pool, "run_forever", stop_at_once)
    try:
        await run_worker(application)
    finally:
        other.leave()

    assert task_status(application, task_id) is TaskStatus.RUNNING


async def test_worker_releases_its_mark_when_it_stops(
    application: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stopped worker is no longer considered an active executor."""

    async def stop_at_once() -> None:
        return None

    monkeypatch.setattr(application.worker_pool, "run_forever", stop_at_once)
    await run_worker(application)

    following = ExecutorLock(application.paths.executor_lock)
    try:
        assert following.join() is True
    finally:
        following.leave()


async def test_draining_marks_itself_as_a_working_executor(
    application: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While drain is in progress, another starting executor observes it as active."""
    observed: list[bool] = []

    async def probe_while_draining() -> tuple[ProcessingTask, ...]:
        probe = ExecutorLock(application.paths.executor_lock)
        observed.append(probe.join())
        probe.leave()
        return ()

    monkeypatch.setattr(application.worker_pool, "drain", probe_while_draining)

    await drain_tasks(application)

    assert observed == [False]
