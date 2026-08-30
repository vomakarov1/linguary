"""Журнал LLM-вызовов в настоящем composition root."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, StorageSettings
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.composition import Application, build_application, drain_tasks
from linguary.domain import ErrorType, LLMCallStage, TaskIdentity, TaskType, TokenUsage
from processing_scenario import BOOK_ID, store_processing_book, translated_reply


@pytest.fixture
def application(
    tmp_path: Path, processing_task_ids: Callable[[], str]
) -> Iterator[Application]:
    """Собранное приложение с журналирующим LLM-клиентом."""
    settings = AppSettings(
        llm=LLMSettings(model="fake-model", api_key="test-key", max_attempts=2),
        checks=CheckSettings(),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
    )
    client = FakeLLMClient.replying(
        [
            FakeReply(content="{"),
            translated_reply(
                usage=TokenUsage(prompt_tokens=10, completion_tokens=20)
            ),
        ]
    )
    built = build_application(
        settings=settings, llm_client=client, task_id_factory=processing_task_ids
    )
    try:
        yield built
    finally:
        built.dispose()


async def test_llm_calls_are_logged_per_attempt(application: Application) -> None:
    """Неуспешная и успешная попытки одного вызова попадают в журнал без payload-ов."""
    scenario = store_processing_book(application.unit_of_work)
    await application.task_queue.enqueue(
        TaskIdentity(
            book_id=BOOK_ID,
            task_type=TaskType.PROCESS_TRANSLATION_UNIT,
            scope_id=scenario.section().id,
        )
    )

    finished = await drain_tasks(application)

    with application.unit_of_work.transaction() as transaction:
        logs = transaction.llm_calls.list_for_book(BOOK_ID)
    assert len(finished) == 1
    assert finished[0].last_error is None
    assert [entry.attempt for entry in logs] == [1, 2]
    assert [entry.success for entry in logs] == [False, True]
    assert logs[0].stage is LLMCallStage.TRANSLATION_UNIT
    assert logs[0].task_id == "task-1"
    assert logs[0].error is not None
    assert logs[0].error.type is ErrorType.MODEL_OUTPUT
    assert logs[1].usage == TokenUsage(prompt_tokens=10, completion_tokens=20)
