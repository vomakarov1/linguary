"""Отбракованный проверкой ответ связывается с вызовом модели."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.adapters.llm.logging import LoggingLLMClient
from linguary.application.ports.llm import LLMRequest, LLMResponse
from linguary.application.retry_policy import RetryPolicy
from linguary.application.use_cases.analyze_chapter import AnalyzeChapter, AnalyzeChapterCommand
from linguary.composition import build_application, drain_tasks
from linguary.domain import ErrorType, TaskIdentity, TaskType
from processing_scenario import (
    BOOK_ID,
    block_reply,
    store_processing_book,
    translated_reply,
    translation_reply,
)


class WithRawContent:
    """Передаёт текст ответа вместе с разобранным ответом тестовой модели."""

    def __init__(self, inner: FakeLLMClient) -> None:
        self.inner = inner

    async def generate(self, request: LLMRequest[Any]) -> LLMResponse[Any]:
        response = await self.inner.generate(request)
        return replace(response, raw_content="сырой ответ модели")


def _analysis_reply(aliases: tuple[str, ...]) -> FakeReply:
    return FakeReply(content=json.dumps({
        "glossary_candidates": [],
        "translation_instructions": "Сохранять стиль.",
        "treatment_plan": [
            {"source_block_id": alias, "treatment": "TRANSLATE"}
            for alias in aliases
        ],
        "before_you_read": [],
        "chapter_summary": "Краткая сводка главы.",
    }, ensure_ascii=False))


@pytest.mark.parametrize("debug", [False, True])
async def test_rejected_translation_logged_unsuccessful_with_reason(
    tmp_path: Path, debug: bool
) -> None:
    client = FakeLLMClient.replying(
        [translated_reply(aliases=("b1",)), translated_reply()]
    )
    application = build_application(
        settings=AppSettings(
            llm=LLMSettings(
                model="fake-model", api_key="test-key", max_attempts=2,
                debug_llm_payloads=debug,
            ),
            storage=StorageSettings(data_root=str(tmp_path / "data")),
        ),
        llm_client=WithRawContent(client),
    )
    try:
        scenario = store_processing_book(application.unit_of_work)
        await application.task_queue.enqueue(
            TaskIdentity(
                book_id=BOOK_ID,
                task_type=TaskType.PROCESS_TRANSLATION_UNIT,
                scope_id=scenario.section().id,
            )
        )
        await drain_tasks(application)
        with application.unit_of_work.transaction() as transaction:
            calls = transaction.llm_calls.list_for_book(BOOK_ID)
            texts = tuple(transaction.llm_calls.response_text_for(call.id) for call in calls)
        assert len(calls) == 2
        assert calls[0].success is False
        assert calls[0].error is not None
        assert calls[0].error.type is ErrorType.VALIDATION
        assert scenario.block_ids()[1] in calls[0].error.message
        assert calls[1].success is True
        assert texts == (("сырой ответ модели", "сырой ответ модели") if debug else (None, None))
    finally:
        application.dispose()


async def test_rejected_analysis_logged_unsuccessful_with_reason(tmp_path: Path) -> None:
    application = build_application(
        settings=AppSettings(
            llm=LLMSettings(model="fake-model", api_key="test-key"),
            storage=StorageSettings(data_root=str(tmp_path / "data")),
        ),
        llm_client=FakeLLMClient.replying([]),
    )
    try:
        scenario = store_processing_book(application.unit_of_work)
        logger = LoggingLLMClient(
            inner=FakeLLMClient.replying([
                _analysis_reply(("b1",)), _analysis_reply(("b1", "b2"))
            ]),
            unit_of_work=application.unit_of_work,
            provider="fake",
            model="fake-model",
        )
        analyze = AnalyzeChapter(
            unit_of_work=application.unit_of_work,
            llm_client=logger,
            retry_policy=RetryPolicy(max_attempts=2),
            glossary_confidence_threshold=0.5,
            sleep=_no_sleep,
        )
        await analyze(AnalyzeChapterCommand(book_id=BOOK_ID, section_id=scenario.section().id))
        with application.unit_of_work.transaction() as transaction:
            calls = transaction.llm_calls.list_for_book(BOOK_ID)
        assert len(calls) == 2
        assert calls[0].success is False
        assert calls[0].error is not None
        assert calls[0].error.type is ErrorType.VALIDATION
        assert scenario.block_ids()[1] in calls[0].error.message
        assert calls[1].success is True
    finally:
        application.dispose()


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.mark.parametrize("debug", [False, True])
async def test_retry_instruction_uses_alias_and_keeps_rejection_log(
    tmp_path: Path, debug: bool
) -> None:
    client = FakeLLMClient.replying(
        [
            translation_reply([block_reply("b1", translated_text="«Мы вернёмся».\nОн ушёл.")]),
            translation_reply([block_reply("b1", translated_text="«Мы вернёмся». Он ушёл.")]),
        ]
    )
    application = build_application(
        settings=AppSettings(
            llm=LLMSettings(
                model="fake-model", api_key="test-key", max_attempts=2,
                debug_llm_payloads=debug,
            ),
            storage=StorageSettings(data_root=str(tmp_path / "data")),
        ),
        llm_client=WithRawContent(client),
    )
    try:
        scenario = store_processing_book(
            application.unit_of_work, chapters=(('"We shall return." He left.',),)
        )
        await application.task_queue.enqueue(
            TaskIdentity(
                book_id=BOOK_ID, task_type=TaskType.PROCESS_TRANSLATION_UNIT,
                scope_id=scenario.section().id,
            )
        )
        await drain_tasks(application)
        assert client.call_count == 2
        retry = client.requests[1].instructions
        assert "перевод блока b1 содержит 2 абзацев, а оригинал — 1" in retry
        assert scenario.block_ids()[0] not in retry
        assert "сохрани исходное число непустых строк (1)" in retry
        assert "весь текст своего блока" in retry
        assert "объедини собственную реплику или авторское окончание" in retry
        assert "не удаляй их" in retry
        assert "не включай соседние" in retry
        with application.unit_of_work.transaction() as transaction:
            calls = transaction.llm_calls.list_for_book(BOOK_ID)
            texts = tuple(transaction.llm_calls.response_text_for(call.id) for call in calls)
        assert len(calls) == 2
        assert calls[0].success is False
        assert calls[0].error is not None
        assert calls[0].error.type is ErrorType.VALIDATION
        assert scenario.block_ids()[0] in calls[0].error.message
        assert calls[1].success is True
        assert texts == (("сырой ответ модели", "сырой ответ модели") if debug else (None, None))
    finally:
        application.dispose()
