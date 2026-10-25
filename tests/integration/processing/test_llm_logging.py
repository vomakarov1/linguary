"""LLM call log in real composition root."""

from __future__ import annotations
from test_book_conditions import choose_legacy_fixture

import json
from collections.abc import Callable, Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, StorageSettings
from linguary.adapters.llm import FakeLLMClient, FakeReply, OpenAICompatibleLLMClient
from linguary.adapters.llm.logging import LoggingLLMClient
from linguary.application.ports.llm import LLMRequest, LLMResponse, LLMResponseFormatError
from linguary.composition import Application, build_application, drain_tasks
from linguary.web import create_app
from linguary.domain.llm_log import TokenPricing
from linguary.domain import (
    ErrorType, LLMCallLog, LLMCallStage, TaskError, TaskIdentity, TaskType, TokenUsage,
)
from processing_scenario import BOOK_ID, store_processing_book, translated_reply


@pytest.fixture
def application(
    tmp_path: Path, processing_task_ids: Callable[[], str]
) -> Iterator[Application]:
    """Composed application with logging LLM client."""
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
    """Unsuccessful and successful attempts of a single call are logged without payloads."""
    scenario = store_processing_book(application.unit_of_work)
    await application.task_queue.enqueue(
        TaskIdentity(
            book_id=BOOK_ID,
            task_type=TaskType.PROCESS_TRANSLATION_UNIT,
            scope_id=scenario.section().id,
        )
    )

    choose_legacy_fixture(application)
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
        choose_legacy_fixture(built)
        await drain_tasks(built)
        with built.unit_of_work.transaction() as transaction:
            (entry,) = transaction.llm_calls.list_for_book(BOOK_ID)
        assert entry.cost == Decimal("0.018")
    finally:
        built.dispose()


def _schema_error_document() -> dict[str, Any]:
    """Repeated b1 with empty translation violates active schema."""
    return {
        "model": "gateway/actual-model",
        "choices": [{"message": {"content": json.dumps({
            "blocks": [
                {
                    "source_block_id": "b1", "treatment": "TRANSLATE",
                    "translated_text": "secret book translation", "glossary_hints": [],
                },
                {
                    "source_block_id": "b1", "treatment": "TRANSLATE",
                    "translated_text": "", "glossary_hints": [],
                },
            ],
            "language_notes": [],
        })}}],
        "usage": {"prompt_tokens": 24480, "completion_tokens": 1122, "cost": 0.09027},
    }


async def _logged_format_error(
    application: Application,
    document: dict[str, Any] | None = None,
    *,
    debug: bool = False,
    pricing: TokenPricing | None = None,
    unreadable: bool = False,
) -> tuple[LLMResponseFormatError, LLMCallLog, str | None]:
    """Executes a single local HTTP call through real log and SQLite."""
    store_processing_book(application.unit_of_work)
    document = _schema_error_document() if document is None else document
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if unreadable:
            return httpx.Response(200, text="secret HTTP document")
        return httpx.Response(200, json=document)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        inner = OpenAICompatibleLLMClient(
            settings=LLMSettings(
                provider="http-provider", model="configured-model", api_key="secret-api-key"
            ),
            http_client=http,
        )
        original_generate = inner.generate
        original_errors: list[LLMResponseFormatError] = []

        async def record_error(request: LLMRequest[Any]) -> LLMResponse[Any]:
            try:
                return await original_generate(request)
            except LLMResponseFormatError as error:
                original_errors.append(error)
                raise

        logger = LoggingLLMClient(
            inner=inner, unit_of_work=application.unit_of_work,
            provider="decorator-provider", model="decorator-model",
            debug_payloads=debug, pricing=pricing,
        )
        request = LLMRequest.for_translation(
            instructions="Переведи", input_text="secret request text"
        ).with_logging(book_id=BOOK_ID, stage=LLMCallStage.TRANSLATION_UNIT)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(inner, "generate", record_error)
            with pytest.raises(LLMResponseFormatError) as caught:
                await logger.generate(request)

    assert len(calls) == 1
    assert original_errors == [caught.value]
    with application.unit_of_work.transaction() as transaction:
        (entry,) = transaction.llm_calls.list_for_book(BOOK_ID)
        stored = transaction.llm_calls.response_text_for(entry.id)
    assert entry.success is False
    assert entry.error is not None
    assert entry.error.type is ErrorType.MODEL_OUTPUT
    assert entry.error.message == str(caught.value)
    for message in (str(caught.value), entry.error.message, entry.error.detail or ""):
        for private in (
            "secret book translation", "secret book text", "secret request text",
            "secret-api-key", "secret HTTP document", "TRANSLATE", "b1",
        ):
            assert private not in message
        if caught.value.raw_content is not None:
            assert caught.value.raw_content not in message
    return caught.value, entry, stored


async def test_schema_error_log_keeps_usage_and_provider_cost(application: Application) -> None:
    error, entry, stored = await _logged_format_error(application)

    assert entry.usage == error.usage == TokenUsage(24480, 1122)
    assert entry.cost == error.cost == Decimal("0.09027")
    assert entry.provider == error.provider == "http-provider"
    assert entry.model == error.model == "gateway/actual-model"
    assert "value_error" in str(error)
    assert stored is None


async def test_invalid_json_error_log_keeps_usage_and_cost(application: Application) -> None:
    document = _schema_error_document()
    document["choices"][0]["message"]["content"] = '{"private": "secret book text"'
    error, entry, stored = await _logged_format_error(application, document)

    assert entry.usage == error.usage == TokenUsage(24480, 1122)
    assert entry.cost == error.cost == Decimal("0.09027")
    assert "json_invalid" in str(error)
    assert stored is None


@pytest.mark.parametrize("provider_cost", [0.001, 0])
async def test_format_error_log_prefers_provider_cost_over_pricing(
    application: Application, provider_cost: float
) -> None:
    document = _schema_error_document()
    document["usage"]["cost"] = provider_cost
    error, entry, _ = await _logged_format_error(
        application, document, pricing=TokenPricing(Decimal("7"), Decimal("9")),
    )

    assert entry.cost == error.cost == Decimal(str(provider_cost))
    assert entry.usage == TokenUsage(24480, 1122)


async def test_format_error_log_computes_cost_from_pricing(application: Application) -> None:
    document = _schema_error_document()
    document["usage"].pop("cost")
    error, entry, _ = await _logged_format_error(
        application, document, pricing=TokenPricing(Decimal("3"), Decimal("15")),
    )

    assert error.cost is None
    assert entry.usage == TokenUsage(24480, 1122)
    assert entry.cost == Decimal("0.09027")


async def test_format_error_log_preserves_provider_cost_without_token_counts(
    application: Application,
) -> None:
    document = _schema_error_document()
    document["usage"] = {"cost": 0.09027}
    error, entry, _ = await _logged_format_error(
        application, document, pricing=TokenPricing(Decimal("3"), Decimal("15")),
    )

    assert entry.usage is error.usage is None
    assert entry.cost == error.cost == Decimal("0.09027")


@pytest.mark.parametrize("with_tokens", [False, True])
async def test_format_error_log_cost_none_without_data(
    application: Application, with_tokens: bool
) -> None:
    document = _schema_error_document()
    document["usage"].pop("cost")
    if not with_tokens:
        document.pop("usage")
    error, entry, _ = await _logged_format_error(
        application, document,
        pricing=None if with_tokens else TokenPricing(Decimal("3"), Decimal("15")),
    )

    assert entry.usage == error.usage == (TokenUsage(24480, 1122) if with_tokens else None)
    assert entry.cost is error.cost is None


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("invalid_json", [False, True])
async def test_format_error_log_payload_and_message_obey_debug_setting(
    application: Application, debug: bool, invalid_json: bool
) -> None:
    document = _schema_error_document()
    if invalid_json:
        document["choices"][0]["message"]["content"] = '{"private": "secret book text"'
    raw = document["choices"][0]["message"]["content"]
    error, entry, stored = await _logged_format_error(application, document, debug=debug)

    assert entry.usage == TokenUsage(24480, 1122)
    assert entry.cost == Decimal("0.09027")
    assert error.raw_content == raw
    assert stored == (raw if debug else None)


async def test_format_error_log_uses_configured_metadata_for_unreadable_http_document(
    application: Application,
) -> None:
    error, entry, stored = await _logged_format_error(application, unreadable=True)

    assert error.provider is error.model is None
    assert entry.provider == "decorator-provider"
    assert entry.model == "decorator-model"
    assert entry.usage is entry.cost is stored is None


@pytest.mark.parametrize("failure", ["auth", "format", "document", "transport"])
async def test_task_and_llm_log_errors_do_not_contain_key(
    tmp_path: Path, failure: str,
) -> None:
    secret = "generation-specific-test-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "transport":
            raise httpx.ConnectError(f"connection rejected {secret}")
        if failure == "auth":
            return httpx.Response(401, text=f"invalid key {secret}")
        if failure == "document":
            return httpx.Response(200, text=f"invalid document {secret}")
        document = _schema_error_document()
        document["choices"][0]["message"]["content"] = f'{{"secret":"{secret}"'
        return httpx.Response(200, json=document)

    settings = LLMSettings(
        model="configured-model", api_key=secret, max_attempts=1,
        debug_llm_payloads=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        built = build_application(
            settings=AppSettings(
                llm=settings, storage=StorageSettings(data_root=str(tmp_path / "data")),
            ),
            llm_client=OpenAICompatibleLLMClient(settings=settings, http_client=http),
        )
        try:
            scenario = store_processing_book(built.unit_of_work)
            await built.task_queue.enqueue(TaskIdentity(
                book_id=BOOK_ID, task_type=TaskType.PROCESS_TRANSLATION_UNIT,
                scope_id=scenario.section().id,
            ))
            choose_legacy_fixture(built)
            finished = await drain_tasks(built)
            assert len(finished) == 1
            error = finished[0].last_error
            assert error is not None
            assert (secret not in repr(error)) is True
            with built.unit_of_work.transaction() as transaction:
                (entry,) = transaction.llm_calls.list_for_book(BOOK_ID)
                stored = transaction.llm_calls.response_text_for(entry.id)
            assert entry.error is not None
            assert (secret not in repr(entry)) is True
            assert (secret not in (stored or "")) is True
            expected = {
                "auth": ErrorType.CONFIGURATION, "transport": ErrorType.TRANSIENT,
                "format": ErrorType.MODEL_OUTPUT, "document": ErrorType.MODEL_OUTPUT,
            }
            assert error.type is entry.error.type is expected[failure]
            app = create_app(application=built, run_background_worker=False)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000",
                ) as web:
                    for accept in ("application/json", "text/html"):
                        response = await web.get(f"/books/{BOOK_ID}", headers={"Accept": accept})
                        assert response.status_code == 200
                        assert (secret not in response.text) is True
        finally:
            await built.aclose()


@pytest.mark.parametrize('failure', ['provider', 'format'])
async def test_proxy_secret_not_in_task_llm_logs_or_http_responses(tmp_path, failure):
    secret = 'http://alice:p%40ss@localhost:8123'
    detail = f'{secret} alice:p@ss p%40ss p@ss Basic YWxpY2U6cEBzcw=='
    settings = LLMSettings(model='test', api_key='key', proxy_mode='custom', proxy_url=secret, max_attempts=1, debug_llm_payloads=True)
    def handler(request):
        if failure == 'provider':
            return httpx.Response(403, text=detail)
        return httpx.Response(200, json={'model': detail, 'choices': [{'message': {'content': detail}, 'finish_reason': 'stop'}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        llm = OpenAICompatibleLLMClient(settings=settings, http_client=transport)
        built = build_application(settings=AppSettings(llm=settings, storage=StorageSettings(data_root=str(tmp_path))), llm_client=llm)
        try:
            scenario = store_processing_book(built.unit_of_work)
            await built.task_queue.enqueue(TaskIdentity(book_id=BOOK_ID, task_type=TaskType.PROCESS_TRANSLATION_UNIT, scope_id=scenario.section().id))
            choose_legacy_fixture(built)
            finished = await drain_tasks(built)
            with built.unit_of_work.transaction() as transaction:
                logs = transaction.llm_calls.list_for_book(BOOK_ID)
                raw = [transaction.llm_calls.response_text_for(entry.id) for entry in logs]
            assert len(finished) == len(logs) == 1
            output = repr(finished) + repr(logs) + repr(raw)
            web = create_app(application=built, run_background_worker=False)
            async with web.router.lifespan_context(web):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web), base_url='http://localhost') as http:
                    output += (await http.get('/system/config')).text
            for value in (secret, 'alice', 'p%40ss', 'p@ss', 'YWxpY2U6cEBzcw=='):
                assert value not in output
        finally:
            await built.aclose()


async def test_observer_links_success_and_format_error_call_ids(application):
    store_processing_book(application.unit_of_work)
    events = []
    logger = LoggingLLMClient(inner=FakeLLMClient.replying([FakeReply(content='{'), translated_reply()]), unit_of_work=application.unit_of_work, provider='fake', model='fake', on_logged=lambda *event: events.append(event))
    request = LLMRequest.for_translation(instructions='translate', input_text='text').with_logging(book_id=BOOK_ID, stage=LLMCallStage.TRANSLATION_UNIT)
    with pytest.raises(LLMResponseFormatError):
        await logger.generate(request)
    response = await logger.generate(request)
    with application.unit_of_work.transaction() as transaction:
        logs = transaction.llm_calls.list_for_book(BOOK_ID)
        assert all(transaction.llm_calls.response_text_for(entry.id) is None for entry in logs)
    assert [event[0].id for event in events] == [entry.id for entry in logs]
    assert events[1][0].id == response.call_id
    assert isinstance(events[0][2], LLMResponseFormatError)


async def test_observer_receives_transport_failure_id(application):
    from linguary.application.ports.llm import LLMTransportError
    store_processing_book(application.unit_of_work)
    events = []
    logger = LoggingLLMClient(inner=FakeLLMClient.replying([LLMTransportError('offline')]), unit_of_work=application.unit_of_work, provider='fake', model='fake', on_logged=lambda *event: events.append(event))
    request = LLMRequest.for_translation(instructions='translate', input_text='text').with_logging(book_id=BOOK_ID, stage=LLMCallStage.TRANSLATION_UNIT)
    with pytest.raises(LLMTransportError):
        await logger.generate(request)
    with application.unit_of_work.transaction() as transaction:
        (entry,) = transaction.llm_calls.list_for_book(BOOK_ID)
    assert events[0][0].id == entry.id


async def test_observer_failure_stops_followup_calls(application):
    from linguary.application.ports.llm import LLMConfigurationError
    store_processing_book(application.unit_of_work)
    inner = AsyncMock()
    inner.generate.return_value = LLMResponse(payload=object(), provider='fake', model='fake')
    def fail(*args):
        raise RuntimeError('secret observer detail')
    logger = LoggingLLMClient(inner=inner, unit_of_work=application.unit_of_work, provider='fake', model='fake', on_logged=fail)
    request = LLMRequest.for_translation(instructions='translate', input_text='text').with_logging(book_id=BOOK_ID, stage=LLMCallStage.TRANSLATION_UNIT)
    for _ in range(2):
        with pytest.raises(LLMConfigurationError, match='наблюдения'):
            await logger.generate(request)
    assert inner.generate.await_count == 1


async def test_optional_observer_is_inactive_by_default(application):
    assert LoggingLLMClient(inner=FakeLLMClient(), unit_of_work=application.unit_of_work, provider='fake', model='fake').on_logged is None


async def test_debug_false_never_persists_success_raw(tmp_path):
    events = []
    inner = AsyncMock()
    inner.generate.return_value = LLMResponse(payload=object(), provider='fake', model='fake', raw_content='private accepted answer')
    built = build_application(settings=AppSettings(llm=LLMSettings(model='fake'), storage=StorageSettings(data_root=str(tmp_path / 'data'))), llm_client=inner, on_logged=lambda *event: events.append(event))
    try:
        store_processing_book(built.unit_of_work)
        request = LLMRequest.for_translation(instructions='translate', input_text='text').with_logging(book_id=BOOK_ID, stage=LLMCallStage.TRANSLATION_UNIT)
        # The application stage is accessed through the real processing pipeline.
        logger = LoggingLLMClient(inner=inner, unit_of_work=built.unit_of_work, provider='fake', model='fake', on_logged=lambda *event: events.append(event))
        response = await logger.generate(request)
        assert events[0][2].raw_content == 'private accepted answer'
        assert response.raw_content is None
        with built.unit_of_work.transaction() as transaction:
            assert transaction.llm_calls.response_text_for(response.call_id) is None
    finally:
        await built.aclose()
