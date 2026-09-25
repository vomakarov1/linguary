"""Классификация ошибок и правила повторов вызова модели (P3.T2, ALG-05)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application.analysis_merge import AnalysisMergeError
from linguary.application.llm_dto import LLMPayloadError, TranslationResultResponse
from linguary.application.ports.epub import EpubExportFatalError, EpubImportFatalError
from linguary.application.ports.llm import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRequest,
    LLMResponseFormatError,
    LLMTransportError,
)
from linguary.application.retry_policy import (
    ExpectedBlock,
    ResultExpectation,
    ResultValidationError,
    RetryAction,
    RetryPolicy,
    RetryPolicyError,
    build_repair_instruction,
    classify_error,
)
from linguary.domain import (
    BlockOrderError,
    BlockResultError,
    BlockTreatment,
    ErrorType,
    MissingBlockResultError,
    NonTextualBlockError,
    TreatmentMismatchError,
    TreatmentPlanError,
    UnknownBlockError,
)

TARGET_LANGUAGE = "ru"


def expectation(*, blocks: tuple[ExpectedBlock, ...] | None = None) -> ResultExpectation:
    """Ожидаемый результат группы: два блока с разными режимами обработки."""
    return ResultExpectation(
        target_language=TARGET_LANGUAGE,
        blocks=blocks
        if blocks is not None
        else (
            ExpectedBlock(block_id="block-1", treatment=BlockTreatment.TRANSLATE),
            ExpectedBlock(block_id="block-2", treatment=BlockTreatment.GLOSSARY_ONLY),
        ),
    )


def validation_error(message: str = "в ответе нет блока block-2") -> ResultValidationError:
    """Отбраковка результата блокирующей проверкой."""
    return ResultValidationError(message, expectation=expectation())


def policy(max_attempts: int = 3, **overrides: Any) -> RetryPolicy:
    """Политика повторов с заданным числом попыток."""
    return RetryPolicy(max_attempts=max_attempts, **overrides)


def translation_request() -> LLMRequest[TranslationResultResponse]:
    """Исходный запрос обработки группы блоков."""
    return LLMRequest.for_translation(
        instructions="Переводи художественно, соблюдая глоссарий.",
        input_text="block-1: It is a truth…\nblock-2: My dear Mr. Bennet",
    )


def reply_for(block_ids: tuple[str, ...]) -> FakeReply:
    """Ответ модели с результатом ровно для указанных блоков."""
    payload: dict[str, Any] = {
        "blocks": [
            {
                "source_block_id": block_id,
                "treatment": "TRANSLATE",
                "translated_text": "Общепризнанная истина гласит…",
                "glossary_hints": [],
            }
            for block_id in block_ids
        ],
        "language_notes": [],
    }
    return FakeReply(content=json.dumps(payload, ensure_ascii=False))


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(
            LLMTransportError("429 Too Many Requests"),
            ErrorType.TRANSIENT,
            id="rate-limit",
        ),
        pytest.param(
            LLMTransportError("время ожидания ответа истекло"),
            ErrorType.TRANSIENT,
            id="timeout",
        ),
        pytest.param(
            LLMResponseFormatError("ответ не разбирается как JSON"),
            ErrorType.MODEL_OUTPUT,
            id="invalid-json",
        ),
        pytest.param(
            LLMResponseFormatError("ответ оборван", truncated=True),
            ErrorType.MODEL_OUTPUT,
            id="truncated-output",
        ),
        pytest.param(
            LLMPayloadError("ответ не приводится к внутреннему DTO"),
            ErrorType.MODEL_OUTPUT,
            id="schema-mismatch",
        ),
        pytest.param(
            MissingBlockResultError("в ответе нет блока block-2"),
            ErrorType.VALIDATION,
            id="missing-block",
        ),
        pytest.param(
            UnknownBlockError("блока block-99 нет в группе"),
            ErrorType.VALIDATION,
            id="unknown-block",
        ),
        pytest.param(
            TreatmentMismatchError("режим блока block-1 не совпал с планом"),
            ErrorType.VALIDATION,
            id="wrong-treatment",
        ),
        pytest.param(
            BlockOrderError("порядок блоков не совпадает с исходным"),
            ErrorType.VALIDATION,
            id="wrong-order",
        ),
        pytest.param(
            BlockResultError("результат блока не соответствует режиму"),
            ErrorType.VALIDATION,
            id="wrong-block-result",
        ),
        pytest.param(
            NonTextualBlockError("нетекстовый блок в группе"),
            ErrorType.VALIDATION,
            id="non-textual-block",
        ),
        pytest.param(
            TreatmentPlanError("блок встречается в плане дважды"),
            ErrorType.VALIDATION,
            id="broken-plan",
        ),
        pytest.param(
            ResultValidationError("доля целевого алфавита ниже порога"),
            ErrorType.VALIDATION,
            id="wrong-language",
        ),
        pytest.param(
            AnalysisMergeError("фрагмент 1 вернул режимы для чужих блоков"),
            ErrorType.VALIDATION,
            id="fragments-do-not-merge",
        ),
        pytest.param(
            EpubImportFatalError("EPUB нечитаем"),
            ErrorType.PERMANENT,
            id="invalid-epub",
        ),
        pytest.param(
            EpubExportFatalError("потерян обязательный ресурс книги"),
            ErrorType.PERMANENT,
            id="unsupported-input",
        ),
        pytest.param(
            LLMConfigurationError("LLM не настроена"),
            ErrorType.CONFIGURATION,
            id="missing-llm-config",
        ),
        pytest.param(
            LLMProviderError("провайдер отверг запрос"),
            ErrorType.CONFIGURATION,
            id="provider-rejected-request",
        ),
        pytest.param(
            RuntimeError("что-то пошло не так"),
            ErrorType.TRANSIENT,
            id="unknown-error",
        ),
    ],
)
def test_error_is_classified_by_its_type(error: Exception, expected: ErrorType) -> None:
    """Каждая ветка ALG-05 определяется типом исключения, а не текстом сообщения."""
    assert classify_error(error).type is expected


def test_classified_error_keeps_message_and_technical_detail() -> None:
    """Ошибка задачи несёт короткое сообщение и техническую деталь для интерфейса."""
    task_error = classify_error(LLMTransportError("провайдер недоступен"))

    assert task_error.message == "провайдер недоступен"
    assert task_error.detail == "linguary.application.ports.llm.LLMTransportError"


def test_error_without_message_is_named_by_its_class() -> None:
    """Ошибка без текста не оставляет пользователя с пустым сообщением."""
    assert classify_error(LLMTransportError("")).message == "LLMTransportError"


def test_permanent_error_is_not_retried_on_the_first_attempt() -> None:
    """`PermanentError` повтору не подлежит: нужен другой файл или настройка."""
    decision = policy().decide(EpubImportFatalError("EPUB нечитаем"), attempts=1)

    assert decision.action is RetryAction.GIVE_UP
    assert not decision.should_retry
    assert decision.error.type is ErrorType.PERMANENT


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(LLMConfigurationError("ключ не принят: 401"), id="rejected-key"),
        pytest.param(LLMProviderError("модель не найдена: 404"), id="unknown-model"),
    ],
)
def test_configuration_error_is_not_retried_but_stays_retryable_by_user(
    error: Exception,
) -> None:
    """Ошибка настройки LLM не повторяется сама, но задачу можно повторить (QA-01).

    Новая попытка с тем же ключом получит тот же отказ, поэтому политика её не делает.
    Книга же пригодна: после исправления настроек пользователь повторяет задачу, и
    другой файл для этого не нужен.
    """
    decision = policy().decide(error, attempts=1)

    assert decision.action is RetryAction.GIVE_UP
    assert decision.error.type is ErrorType.CONFIGURATION
    assert decision.error.type.is_retryable


def test_exhausted_attempts_stop_retrying_a_retryable_error() -> None:
    """Число попыток ограничено настройкой: временная ошибка тоже кончается `failed`."""
    decision = policy(max_attempts=3).decide(LLMTransportError("таймаут"), attempts=3)

    assert decision.action is RetryAction.GIVE_UP
    assert decision.error.type is ErrorType.TRANSIENT


def test_single_attempt_setting_disables_retries() -> None:
    """Настройка в одну попытку означает отсутствие автоматических повторов."""
    decision = policy(max_attempts=1).decide(LLMTransportError("таймаут"), attempts=1)

    assert decision.action is RetryAction.GIVE_UP


def test_last_attempt_before_the_limit_is_still_retried() -> None:
    """Предел попыток срабатывает ровно на своём значении, а не раньше."""
    assert policy(max_attempts=3).decide(LLMTransportError("таймаут"), attempts=2).should_retry


def test_transient_error_is_retried_with_growing_backoff() -> None:
    """Временная ошибка повторяется с паузой, которая растёт от попытки к попытке."""
    retry_policy = policy(max_attempts=5, backoff_seconds=1.0, max_backoff_seconds=30.0)

    delays = [
        retry_policy.decide(LLMTransportError("429"), attempts=attempt).backoff_seconds
        for attempt in (1, 2, 3)
    ]

    assert delays == [1.0, 2.0, 4.0]


def test_backoff_does_not_grow_beyond_its_limit() -> None:
    """Пауза перед повтором ограничена: задача не растягивается на минуты ожидания."""
    retry_policy = policy(max_attempts=10, backoff_seconds=1.0, max_backoff_seconds=4.0)

    decision = retry_policy.decide(LLMTransportError("429"), attempts=9)

    assert decision.backoff_seconds == 4.0


def test_model_output_error_is_retried_with_the_same_request() -> None:
    """Невалидный ответ модели повторяется тем же запросом без уточнений."""
    request = translation_request()

    decision = policy().decide(LLMResponseFormatError("ответ не разбирается"), attempts=1)

    assert decision.action is RetryAction.RETRY_SAME_REQUEST
    assert decision.repair_instruction is None
    retried = decision.next_request(request)
    assert retried.instructions == request.instructions
    assert retried.input_text == request.input_text
    assert retried.attempt == 2


def test_output_limit_truncation_is_not_retried() -> None:
    """Обрыв по лимиту вывода повтору не подлежит: тот же запрос даст тот же обрыв."""
    decision = policy(max_attempts=3).decide(
        LLMResponseFormatError("обрыв ответа по лимиту токенов", truncated=True),
        attempts=1,
    )

    assert decision.action is RetryAction.GIVE_UP
    assert not decision.should_retry
    assert decision.error.type is ErrorType.MODEL_OUTPUT
    assert decision.backoff_seconds == 0.0


def test_validation_error_is_retried_with_repair_instruction() -> None:
    """Отбракованный результат повторяется с уточняющей инструкцией (ALG-05)."""
    decision = policy().decide(validation_error(), attempts=1)

    assert decision.action is RetryAction.RETRY_WITH_REPAIR
    assert decision.repair_instruction is not None
    assert decision.backoff_seconds == 0.0


def test_repair_instruction_is_appended_to_the_original_request() -> None:
    """Уточнение дописывается к исходному запросу и не заменяет собранный контекст."""
    request = translation_request()
    decision = policy().decide(validation_error(), attempts=1)

    repaired = decision.next_request(request)

    assert repaired.instructions.startswith(request.instructions)
    assert len(repaired.instructions) > len(request.instructions)
    assert repaired.attempt == 2
    assert repaired.input_text == request.input_text
    assert repaired.kind is request.kind


def test_repair_instruction_names_blocks_treatments_and_language() -> None:
    """Уточнение перечисляет ожидаемые блоки, режимы обработки и целевой язык."""
    instruction = build_repair_instruction(expectation(), violation="в ответе нет блока block-2")

    assert "block-1" in instruction
    assert "block-2" in instruction
    assert "TRANSLATE" in instruction
    assert "GLOSSARY_ONLY" in instruction
    assert TARGET_LANGUAGE in instruction
    assert "в ответе нет блока block-2" in instruction


def test_repair_instruction_omits_treatments_the_model_chooses_itself() -> None:
    """Для анализа главы режимы выбирает модель: уточнение их не диктует."""
    instruction = build_repair_instruction(
        expectation(blocks=(ExpectedBlock(block_id="block-1"),))
    )

    assert "block-1" in instruction
    assert "режим обработки" not in instruction


def test_repair_instruction_without_expectation_still_reports_rejection() -> None:
    """Нарушение без сведений об ожидаемом результате не даёт пустой инструкции."""
    decision = policy().decide(ResultValidationError("не тот язык"), attempts=1)

    assert decision.repair_instruction is not None
    assert "не тот язык" in decision.repair_instruction


def test_request_of_a_forbidden_retry_is_refused() -> None:
    """Решение «не повторять» не даёт запроса следующей попытки."""
    decision = policy().decide(LLMConfigurationError("LLM не настроена"), attempts=1)

    with pytest.raises(RetryPolicyError):
        decision.next_request(translation_request())


@pytest.mark.parametrize(
    "attempts",
    [pytest.param(0, id="zero"), pytest.param(-1, id="negative")],
)
def test_attempt_number_starts_from_one(attempts: int) -> None:
    """Номер попытки — счётчик сделанных вызовов, и он не может быть нулевым."""
    with pytest.raises(RetryPolicyError):
        policy().decide(LLMTransportError("таймаут"), attempts=attempts)


def test_policy_requires_at_least_one_attempt() -> None:
    """Политика без единой попытки бессмысленна и не собирается."""
    with pytest.raises(RetryPolicyError):
        RetryPolicy(max_attempts=0)


def test_expectation_requires_target_language() -> None:
    """Ожидаемый результат без целевого языка не описывает того, что нужно повторить."""
    with pytest.raises(RetryPolicyError):
        ResultExpectation(target_language=" ")


def test_expectation_rejects_duplicate_blocks() -> None:
    """Один блок не может ожидаться дважды: результат сохраняется по `block_id`."""
    with pytest.raises(RetryPolicyError):
        ResultExpectation(
            target_language=TARGET_LANGUAGE,
            blocks=(ExpectedBlock(block_id="block-1"), ExpectedBlock(block_id="block-1")),
        )


def test_expectation_rejects_non_textual_treatment() -> None:
    """Нетекстовый блок модели не передаётся, значит и ожидать его результат нельзя."""
    with pytest.raises(RetryPolicyError):
        ExpectedBlock(block_id="block-1", treatment=BlockTreatment.PRESERVE_ONLY)


async def test_retried_call_carries_the_repair_instruction_to_the_model() -> None:
    """Повторный вызов уходит модели с уточнением поверх исходной инструкции."""
    client = FakeLLMClient.replying(
        [reply_for(("block-99",)), reply_for(("block-1", "block-2"))]
    )
    retry_policy = policy(max_attempts=3)
    request = translation_request()
    expected = expectation()

    for attempt in (1, 2):
        response = await client.generate(request)
        block_ids = response.payload.block_ids
        if block_ids == expected.block_ids:
            break
        decision = retry_policy.decide(
            ResultValidationError(
                f"ответ вернулся для чужих блоков: {', '.join(block_ids)}",
                expectation=expected,
            ),
            attempts=attempt,
        )
        assert decision.should_retry
        request = decision.next_request(request)

    assert client.call_count == 2
    first, second = client.requests
    assert first.instructions == translation_request().instructions
    assert second.instructions.startswith(first.instructions)
    assert "block-1" in second.instructions
    assert "TRANSLATE" in second.instructions
    assert TARGET_LANGUAGE in second.instructions
