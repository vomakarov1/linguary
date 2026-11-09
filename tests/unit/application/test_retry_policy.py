"""Error classification and model invocation retry rules (P3.T2, ALG-05)."""

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
    """Expected unit result: two blocks with different treatments."""
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
    """Result rejection by blocking validation."""
    return ResultValidationError(message, expectation=expectation())


def policy(max_attempts: int = 3, **overrides: Any) -> RetryPolicy:
    """Retry policy with specified number of attempts."""
    return RetryPolicy(max_attempts=max_attempts, **overrides)


def translation_request() -> LLMRequest[TranslationResultResponse]:
    """Initial request for processing a block unit."""
    return LLMRequest.for_translation(
        instructions="Переводи художественно, соблюдая глоссарий.",
        input_text="block-1: It is a truth…\nblock-2: My dear Mr. Bennet",
    )


def reply_for(block_ids: tuple[str, ...]) -> FakeReply:
    """Model reply with results for exactly the specified blocks."""
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
    """Each ALG-05 branch is determined by exception type rather than message text."""
    assert classify_error(error).type is expected


def test_classified_error_keeps_message_and_technical_detail() -> None:
    """A task error carries a short message and technical detail for the interface."""
    task_error = classify_error(LLMTransportError("провайдер недоступен"))

    assert task_error.message == "провайдер недоступен"
    assert task_error.detail == "linguary.application.ports.llm.LLMTransportError"


def test_error_without_message_is_named_by_its_class() -> None:
    """An error without text does not leave the user with an empty message."""
    assert classify_error(LLMTransportError("")).message == "LLMTransportError"


def test_permanent_error_is_not_retried_on_the_first_attempt() -> None:
    """A `PermanentError` cannot be retried: a different file or configuration is required."""
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
    """LLM configuration errors are not retried automatically, but the task can be retried (QA-01).

    A new attempt with the same key would get the same failure, so policy does not retry it.
    The book itself is valid: once settings are fixed, the user retries the task, and no new
    file is required.
    """
    decision = policy().decide(error, attempts=1)

    assert decision.action is RetryAction.GIVE_UP
    assert decision.error.type is ErrorType.CONFIGURATION
    assert decision.error.type.is_retryable


def test_exhausted_attempts_stop_retrying_a_retryable_error() -> None:
    """The number of attempts is bounded by configuration: transient errors also end in `failed`."""
    decision = policy(max_attempts=3).decide(LLMTransportError("таймаут"), attempts=3)

    assert decision.action is RetryAction.GIVE_UP
    assert decision.error.type is ErrorType.TRANSIENT


def test_single_attempt_setting_disables_retries() -> None:
    """A single attempt setting means no automatic retries."""
    decision = policy(max_attempts=1).decide(LLMTransportError("таймаут"), attempts=1)

    assert decision.action is RetryAction.GIVE_UP


def test_last_attempt_before_the_limit_is_still_retried() -> None:
    """The attempt limit triggers at exactly its configured value, not sooner."""
    assert policy(max_attempts=3).decide(LLMTransportError("таймаут"), attempts=2).should_retry


def test_transient_error_is_retried_with_growing_backoff() -> None:
    """A transient error is retried with a backoff that grows from attempt to attempt."""
    retry_policy = policy(max_attempts=5, backoff_seconds=1.0, max_backoff_seconds=30.0)

    delays = [
        retry_policy.decide(LLMTransportError("429"), attempts=attempt).backoff_seconds
        for attempt in (1, 2, 3)
    ]

    assert delays == [1.0, 2.0, 4.0]


def test_backoff_does_not_grow_beyond_its_limit() -> None:
    """Backoff delay is bounded: tasks are not stretched into minutes of waiting."""
    retry_policy = policy(max_attempts=10, backoff_seconds=1.0, max_backoff_seconds=4.0)

    decision = retry_policy.decide(LLMTransportError("429"), attempts=9)

    assert decision.backoff_seconds == 4.0


def test_provider_retry_after_extends_the_backoff() -> None:
    """Retry backoff is no shorter than requested by the provider's `Retry-After` header.

    Retrying earlier would trigger rate limiting again and waste an attempt.
    """
    retry_policy = policy(max_attempts=5, backoff_seconds=1.0, max_backoff_seconds=30.0)

    decision = retry_policy.decide(
        LLMTransportError("429", retry_after_seconds=12.0), attempts=1
    )

    assert decision.action is RetryAction.RETRY_AFTER_BACKOFF
    assert decision.backoff_seconds == 12.0


def test_short_retry_after_does_not_shorten_the_backoff() -> None:
    """A short `Retry-After` does not override growing exponential backoff."""
    retry_policy = policy(max_attempts=5, backoff_seconds=1.0, max_backoff_seconds=30.0)

    decision = retry_policy.decide(
        LLMTransportError("503", retry_after_seconds=0.5), attempts=3
    )

    assert decision.backoff_seconds == 4.0


def test_retry_after_is_bounded() -> None:
    """An excessive `Retry-After` does not turn a single task into hours of waiting."""
    retry_policy = policy(
        max_attempts=5,
        backoff_seconds=1.0,
        max_backoff_seconds=30.0,
        max_retry_after_seconds=120.0,
    )

    decision = retry_policy.decide(
        LLMTransportError("429", retry_after_seconds=3600.0), attempts=1
    )

    assert decision.backoff_seconds == 120.0


def test_model_output_error_is_retried_with_the_same_request() -> None:
    """An invalid model response is retried with the same request without repairs."""
    request = translation_request()

    decision = policy().decide(LLMResponseFormatError("ответ не разбирается"), attempts=1)

    assert decision.action is RetryAction.RETRY_SAME_REQUEST
    assert decision.repair_instruction is None
    retried = decision.next_request(request)
    assert retried.instructions == request.instructions
    assert retried.input_text == request.input_text
    assert retried.attempt == 2


def test_output_limit_truncation_is_not_retried() -> None:
    """Output token limit truncation is not retried: the same request would truncate again."""
    decision = policy(max_attempts=3).decide(
        LLMResponseFormatError("обрыв ответа по лимиту токенов", truncated=True),
        attempts=1,
    )

    assert decision.action is RetryAction.GIVE_UP
    assert not decision.should_retry
    assert decision.error.type is ErrorType.MODEL_OUTPUT
    assert decision.backoff_seconds == 0.0


def test_validation_error_is_retried_with_repair_instruction() -> None:
    """A rejected result is retried with a repair instruction (ALG-05)."""
    decision = policy().decide(validation_error(), attempts=1)

    assert decision.action is RetryAction.RETRY_WITH_REPAIR
    assert decision.repair_instruction is not None
    assert decision.backoff_seconds == 0.0


def test_repair_instruction_is_appended_to_the_original_request() -> None:
    """The repair instruction is appended to the original request and does not replace context."""
    request = translation_request()
    decision = policy().decide(validation_error(), attempts=1)

    repaired = decision.next_request(request)

    assert repaired.instructions == request.instructions
    assert repaired.repair_instructions == (decision.repair_instruction,)
    assert repaired.attempt == 2
    assert repaired.input_text == request.input_text
    assert repaired.kind is request.kind


def test_repair_instruction_names_blocks_treatments_and_language() -> None:
    """The repair instruction names expected blocks, treatments, and target language."""
    instruction = build_repair_instruction(expectation(), violation="в ответе нет блока block-2")

    assert "block-1" in instruction
    assert "block-2" in instruction
    assert "TRANSLATE" in instruction
    assert "GLOSSARY_ONLY" in instruction
    assert TARGET_LANGUAGE in instruction
    assert "в ответе нет блока block-2" in instruction


def test_repair_instruction_omits_treatments_the_model_chooses_itself() -> None:
    """For chapter analysis, treatments are chosen by the model: repairs do not dictate them."""
    instruction = build_repair_instruction(
        expectation(blocks=(ExpectedBlock(block_id="block-1"),))
    )

    assert "block-1" in instruction
    assert "режим обработки" not in instruction


def test_repair_instruction_without_expectation_still_reports_rejection() -> None:
    """A violation without expectation details still yields a non-empty repair instruction."""
    decision = policy().decide(ResultValidationError("не тот язык"), attempts=1)

    assert decision.repair_instruction is not None
    assert "не тот язык" in decision.repair_instruction


def test_request_of_a_forbidden_retry_is_refused() -> None:
    """A 'give up' decision refuses to construct a next request."""
    decision = policy().decide(LLMConfigurationError("LLM не настроена"), attempts=1)

    with pytest.raises(RetryPolicyError):
        decision.next_request(translation_request())


@pytest.mark.parametrize(
    "attempts",
    [pytest.param(0, id="zero"), pytest.param(-1, id="negative")],
)
def test_attempt_number_starts_from_one(attempts: int) -> None:
    """The attempt number counts completed invocations and cannot be zero or negative."""
    with pytest.raises(RetryPolicyError):
        policy().decide(LLMTransportError("таймаут"), attempts=attempts)


def test_policy_requires_at_least_one_attempt() -> None:
    """A policy with zero attempts is invalid and cannot be constructed."""
    with pytest.raises(RetryPolicyError):
        RetryPolicy(max_attempts=0)


def test_expectation_requires_target_language() -> None:
    """An expectation without a target language does not adequately describe retry requirements."""
    with pytest.raises(RetryPolicyError):
        ResultExpectation(target_language=" ")


def test_expectation_rejects_duplicate_blocks() -> None:
    """The same block cannot be expected twice: results are keyed by `block_id`."""
    with pytest.raises(RetryPolicyError):
        ResultExpectation(
            target_language=TARGET_LANGUAGE,
            blocks=(ExpectedBlock(block_id="block-1"), ExpectedBlock(block_id="block-1")),
        )


def test_expectation_rejects_non_textual_treatment() -> None:
    """Non-textual blocks are not sent to the model, so expecting results for them is invalid."""
    with pytest.raises(RetryPolicyError):
        ExpectedBlock(block_id="block-1", treatment=BlockTreatment.PRESERVE_ONLY)


async def test_retried_call_carries_the_repair_instruction_to_the_model() -> None:
    """The retried invocation is sent to the model with repairs layered over the prompt."""
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
    assert second.instructions == first.instructions
    assert len(second.repair_instructions) == 1
    assert "block-1" in second.repair_instructions[0]
    assert "TRANSLATE" in second.repair_instructions[0]
    assert TARGET_LANGUAGE in second.repair_instructions[0]


def test_retry_decision_records_assistant_raw_content_in_repair_turns() -> None:
    """Retry decision records assistant raw content in repair turns for format and validation errors (AC-003)."""
    retry_policy = policy(max_attempts=3)
    request = translation_request()

    format_error = LLMResponseFormatError("не разобран json", raw_content='{"broken":')
    decision1 = retry_policy.decide(format_error, attempts=1)
    assert decision1.action is RetryAction.RETRY_WITH_REPAIR
    assert decision1.assistant_response == '{"broken":'
    assert decision1.repair_instruction is not None

    retried1 = decision1.next_request(request)
    assert retried1.attempt == 2
    assert retried1.repair_turns == (('{"broken":', decision1.repair_instruction),)
    assert retried1.repair_instructions == (decision1.repair_instruction,)

    val_error = validation_error()
    decision2 = retry_policy.decide(val_error, attempts=2, raw_content='{"blocks": []}')
    assert decision2.action is RetryAction.RETRY_WITH_REPAIR
    assert decision2.assistant_response == '{"blocks": []}'

    retried2 = decision2.next_request(retried1)
    assert retried2.attempt == 3
    assert retried2.repair_turns == (
        ('{"broken":', decision1.repair_instruction),
        ('{"blocks": []}', decision2.repair_instruction),
    )
    assert retried2.repair_instructions == (
        decision1.repair_instruction,
        decision2.repair_instruction,
    )


def test_retry_policy_uses_compact_local_repair_instruction_with_conversation_turns() -> None:
    from linguary.domain.llm_log import Violation

    six_blocks = expectation(
        blocks=tuple(
            ExpectedBlock(
                block_id=f"b{idx}",
                treatment=BlockTreatment.TRANSLATE if idx % 2 else BlockTreatment.READ_AS_IS,
            )
            for idx in range(1, 7)
        )
    )
    local_err = ResultValidationError(
        "цитата языковой заметки отсутствует в исходном блоке (b3)",
        expectation=six_blocks,
        violation=Violation(
            "note_quote_missing",
            "translation_unit",
            scope="local",
            block_ids=("b3",),
            safe_message="цитата языковой заметки отсутствует в исходном блоке (b3)",
        ),
    )

    decision_compact = policy(max_attempts=3).decide(
        local_err, attempts=1, raw_content='{"blocks": []}'
    )
    assert decision_compact.action is RetryAction.RETRY_WITH_REPAIR
    assert decision_compact.assistant_response == '{"blocks": []}'
    assert decision_compact.repair_instruction is not None
    assert "- b3: режим обработки TRANSLATE" in decision_compact.repair_instruction
    assert "полный массив всех блоков" in decision_compact.repair_instruction
    assert "без изменений" in decision_compact.repair_instruction
    for untouched in ("- b1:", "- b2:", "- b4:", "- b5:", "- b6:"):
        assert untouched not in decision_compact.repair_instruction

    decision_no_turn = policy(max_attempts=3).decide(local_err, attempts=1)
    assert decision_no_turn.repair_instruction is not None
    for idx in range(1, 7):
        assert f"- b{idx}:" in decision_no_turn.repair_instruction

    unmatched_instruction = build_repair_instruction(
        six_blocks,
        violation="локальная ошибка с каноническим ID",
        local_block_ids=("canonical-uuid-3",),
        has_conversation_turns=True,
    )
    for idx in range(1, 7):
        assert f"- b{idx}:" in unmatched_instruction

    small_blocks = expectation(
        blocks=tuple(
            ExpectedBlock(block_id=f"b{idx}", treatment=BlockTreatment.TRANSLATE)
            for idx in range(1, 4)
        )
    )
    small_err = ResultValidationError(
        "ошибка в b2",
        expectation=small_blocks,
        violation=Violation(
            "note_quote_missing",
            "translation_unit",
            scope="local",
            block_ids=("b2",),
        ),
    )
    decision_small = policy(max_attempts=3).decide(
        small_err, attempts=1, raw_content='{"blocks": []}'
    )
    assert decision_small.repair_instruction is not None
    for idx in range(1, 4):
        assert f"- b{idx}:" in decision_small.repair_instruction


