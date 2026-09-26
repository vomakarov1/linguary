"""Журнал LLM-вызовов в настоящем composition root."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, StorageSettings
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.adapters.llm.logging import LoggingLLMClient
from linguary.application.ports.llm import LLMRequest, LLMResponse, LLMResponseFormatError
from linguary.composition import Application, build_application, drain_tasks
from linguary.domain.llm_log import TokenPricing
from linguary.domain import ErrorType, LLMCallStage, TaskError, TaskIdentity, TaskType, TokenUsage
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


@pytest.mark.parametrize("debug", [False, True])
async def test_format_error_text_stored_only_with_debug(
    tmp_path: Path, debug: bool
) -> None:
    raw = '{"private":"secret book text"}'
    built = build_application(
        settings=AppSettings(
            llm=LLMSettings(
                model="fake-model", api_key="test-key", debug_llm_payloads=debug
            ),
            storage=StorageSettings(data_root=str(tmp_path / "data")),
        ),
        llm_client=FakeLLMClient.replying([]),
    )
    try:
        store_processing_book(built.unit_of_work)
        request = LLMRequest.for_translation(
            instructions="Переведи", input_text="Исходный текст"
        ).with_logging(book_id=BOOK_ID, stage=LLMCallStage.TRANSLATION_UNIT)
        logger = LoggingLLMClient(
            inner=FakeLLMClient.replying([FakeReply(content=raw)]),
            unit_of_work=built.unit_of_work,
            provider="fake",
            model="fake-model",
            debug_payloads=debug,
        )
        with pytest.raises(LLMResponseFormatError) as caught:
            await logger.generate(request)
        assert raw not in str(caught.value)
        with built.unit_of_work.transaction() as transaction:
            (entry,) = transaction.llm_calls.list_for_book(BOOK_ID)
            stored = transaction.llm_calls.response_text_for(entry.id)
        assert raw not in (entry.error.message if entry.error else "")
        assert stored == (raw if debug else None)
    finally:
        built.dispose()


@pytest.mark.parametrize("debug", [False, True])
async def test_success_response_carries_call_id_and_debug_content(
    tmp_path: Path, debug: bool
) -> None:
    built = build_application(
        settings=AppSettings(
            llm=LLMSettings(model="fake-model", api_key="test-key"),
            storage=StorageSettings(data_root=str(tmp_path / "data")),
        ),
        llm_client=FakeLLMClient.replying([]),
    )
    try:
        store_processing_book(built.unit_of_work)
        raw = "секретный ответ модели"
        inner = AsyncMock()
        inner.generate.return_value = LLMResponse(
            payload=object(), provider="fake", model="fake-model", raw_content=raw
        )
        logger = LoggingLLMClient(
            inner=inner,
            unit_of_work=built.unit_of_work,
            provider="fake",
            model="fake-model",
            debug_payloads=debug,
        )
        request = LLMRequest.for_translation(
            instructions="Переведи", input_text="Исходный текст"
        ).with_logging(book_id=BOOK_ID, stage=LLMCallStage.TRANSLATION_UNIT)
        response = await logger.generate(request)
        assert response.call_id is not None
        assert response.raw_content == (raw if debug else None)
        with built.unit_of_work.transaction() as transaction:
            assert transaction.llm_calls.response_text_for(response.call_id) == (
                raw if debug else None
            )
    finally:
        built.dispose()


async def _logged_cost(
    tmp_path: Path, *, provider_cost: Decimal | None,
    pricing: TokenPricing | None, usage: TokenUsage | None,
) -> Decimal | None:
    built = build_application(
        settings=AppSettings(
            llm=LLMSettings(model="fake-model", api_key="test-key"),
            storage=StorageSettings(data_root=str(tmp_path / "data")),
        ),
        llm_client=FakeLLMClient.replying([]),
    )
    try:
        store_processing_book(built.unit_of_work)
        inner = AsyncMock()
        inner.generate.return_value = LLMResponse(
            payload=object(), provider="fake", model="fake-model",
            usage=usage, cost=provider_cost,
        )
        logger = LoggingLLMClient(
            inner=inner, unit_of_work=built.unit_of_work,
            provider="fake", model="fake-model", pricing=pricing,
        )
        request = LLMRequest.for_translation(
            instructions="Переведи", input_text="Исходный текст"
        ).with_logging(book_id=BOOK_ID, stage=LLMCallStage.TRANSLATION_UNIT)
        await logger.generate(request)
        with built.unit_of_work.transaction() as transaction:
            (entry,) = transaction.llm_calls.list_for_book(BOOK_ID)
        return entry.cost
    finally:
        built.dispose()


async def test_log_prefers_provider_cost(tmp_path: Path) -> None:
    cost = await _logged_cost(
        tmp_path, provider_cost=Decimal("0.007"),
        pricing=TokenPricing(Decimal("3"), Decimal("15")),
        usage=TokenUsage(prompt_tokens=1000, completion_tokens=1000),
    )
    assert cost == Decimal("0.007")


async def test_log_computes_cost_from_pricing(tmp_path: Path) -> None:
    cost = await _logged_cost(
        tmp_path, provider_cost=None,
        pricing=TokenPricing(Decimal("3"), Decimal("15")),
        usage=TokenUsage(prompt_tokens=1000, completion_tokens=1000),
    )
    assert cost == Decimal("0.018")


async def test_log_cost_none_without_data(tmp_path: Path) -> None:
    cost = await _logged_cost(
        tmp_path, provider_cost=None,
        pricing=TokenPricing(Decimal("3"), Decimal("15")), usage=None,
    )
    assert cost is None


async def test_rejected_call_keeps_cost(tmp_path: Path) -> None:
    built = build_application(
        settings=AppSettings(
            llm=LLMSettings(model="fake-model", api_key="test-key"),
            storage=StorageSettings(data_root=str(tmp_path / "data")),
        ),
        llm_client=FakeLLMClient.replying([]),
    )
    try:
        store_processing_book(built.unit_of_work)
        inner = AsyncMock()
        inner.generate.return_value = LLMResponse(
            payload=object(), provider="fake", model="fake-model",
            cost=Decimal("0.007"),
        )
        logger = LoggingLLMClient(
            inner=inner, unit_of_work=built.unit_of_work,
            provider="fake", model="fake-model",
        )
        request = LLMRequest.for_translation(
            instructions="Переведи", input_text="Исходный текст"
        ).with_logging(book_id=BOOK_ID, stage=LLMCallStage.TRANSLATION_UNIT)
        response = await logger.generate(request)
        assert response.call_id is not None
        with built.unit_of_work.transaction() as transaction:
            transaction.llm_calls.mark_rejected(
                response.call_id,
                error=TaskError(type=ErrorType.VALIDATION, message="неверный блок"),
                response_text=None,
            )
        with built.unit_of_work.transaction() as transaction:
            (entry,) = transaction.llm_calls.list_for_book(BOOK_ID)
        assert entry.success is False
        assert entry.cost == Decimal("0.007")
    finally:
        built.dispose()


async def test_reference_pricing_fills_cost_without_provider_cost(tmp_path: Path) -> None:
    built = build_application(
        settings=AppSettings(
            llm=LLMSettings(
                model="anthropic/claude-sonnet-4.6", api_key="test-key",
                input_price_per_mtok=3, output_price_per_mtok=15,
            ),
            storage=StorageSettings(data_root=str(tmp_path / "data")),
        ),
        llm_client=FakeLLMClient.replying([
            translated_reply(usage=TokenUsage(prompt_tokens=1000, completion_tokens=1000))
        ]),
    )
    try:
        scenario = store_processing_book(built.unit_of_work)
        await built.task_queue.enqueue(TaskIdentity(
            book_id=BOOK_ID, task_type=TaskType.PROCESS_TRANSLATION_UNIT,
            scope_id=scenario.section().id,
        ))
        await drain_tasks(built)
        with built.unit_of_work.transaction() as transaction:
            (entry,) = transaction.llm_calls.list_for_book(BOOK_ID)
        assert entry.cost == Decimal("0.018")
    finally:
        built.dispose()
