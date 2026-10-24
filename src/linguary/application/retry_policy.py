"""Error classification and retry policy (PRD § 16, ALG-05).

This module answers two questions and nothing else: what type of error occurred, and what
action to take for the next attempt. It does not execute calls, does not sleep between attempts,
and knows nothing about queue tasks: the retry loop is orchestrated by the use case holding
the model client and original request, while waiting between attempts is executed by the event
loop owner.

Errors do not form a separate hierarchy on top of the domain: classification maps exceptions
to domain `ErrorType`, and retry decisions are based on it. Discrimination is based on exception
type rather than message text — this is why `LLMClient` port types are split (`ports/llm.py`).

Blocking result validation violations (ALG-04) arrive here as `ResultValidationError` along with
`ResultExpectation`: from this same information — expected `block_id`s, expected treatments,
and target language — a retry repair instruction is constructed. It is appended to the original
request instruction rather than replacing it: chapter context, glossary, and block text are
assembled by `ContextBuilder`, and the retry retains them.

A single task failure does not halt book processing: the policy decides the fate of a single
call attempt, while task state is finalized by `TaskRunner` (ALG-02).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from linguary.application.analysis_merge import AnalysisMergeError
from linguary.application.errors import ApplicationError
from linguary.application.llm_dto import LLMPayloadError
from linguary.application.ports.epub import EpubExportFatalError, EpubImportFatalError
from linguary.application.ports.llm import (
    LLMConfigurationError,
    LLMPayload,
    LLMProviderError,
    LLMRequest,
    LLMResponseFormatError,
    LLMTransportError,
)
from linguary.domain import (
    BlockResultError,
    BlockTreatment,
    ErrorType,
    NonTextualBlockError,
    TaskError,
    TranslationUnitError,
    TreatmentPlanError,
    UnknownBlockError,
)

#: Delay before the first retry attempt of a transient error. It doubles thereafter:
#: provider rate limits take time to reset, and rapid retries prolong them.
DEFAULT_BACKOFF_SECONDS = 1.0

#: Maximum backoff delay. Waiting longer is unproductive: the user sees the task in progress
#: rather than results, and it is cleaner to exhaust attempts than stretch one task for minutes.
DEFAULT_MAX_BACKOFF_SECONDS = 30.0

#: Upper bound on the pause requested by the provider via the `Retry-After` header.
#: The policy respects provider requests beyond `DEFAULT_MAX_BACKOFF_SECONDS` — otherwise
#: retries will simply be rejected again — but caps it to prevent waiting for hours.
DEFAULT_MAX_RETRY_AFTER_SECONDS = 120.0


class RetryPolicyError(ApplicationError):
    """Retry policy configured or invoked in an invalid manner."""


class ResultValidationError(ApplicationError):
    """Call result rejected by blocking validation (PRD § 15, ALG-04).

    Describes an unusable model response rather than a transport failure: missing expected block,
    unexpected block, treatment mode mismatch against plan, or non-target language.
    Such responses are retried with repair instructions, which is why the error carries `expectation` —
    the information the model was missing.
    """

    def __init__(self, message: str, *, expectation: ResultExpectation | None = None) -> None:
        super().__init__(message)
        self.expectation = expectation


@dataclass(frozen=True, slots=True)
class ExpectedBlock:
    """A block for which a retry call must return a result.

    `treatment` is unset where the mode is chosen by the model: in chapter analysis, the treatment
    plan is model output rather than a request constraint (PRD § 14).
    """

    block_id: str
    treatment: BlockTreatment | None = None

    def __post_init__(self) -> None:
        if not self.block_id.strip():
            raise RetryPolicyError("идентификатор ожидаемого блока пуст")
        if self.treatment is not None and not self.treatment.is_textual:
            raise RetryPolicyError(
                f"блок {self.block_id} с режимом {self.treatment} модели не передаётся"
            )


@dataclass(frozen=True, slots=True)
class ResultExpectation:
    """Exact expectations for a retry call (ALG-05).

    The exact details enumerated by the repair instruction: expected blocks in order,
    expected treatment modes, and target result language.
    """

    target_language: str
    blocks: tuple[ExpectedBlock, ...] = ()

    def __post_init__(self) -> None:
        if not self.target_language.strip():
            raise RetryPolicyError("целевой язык результата не задан")
        seen: set[str] = set()
        for block in self.blocks:
            if block.block_id in seen:
                raise RetryPolicyError(
                    f"блок {block.block_id} указан в ожидаемом результате дважды"
                )
            seen.add(block.block_id)

    @property
    def block_ids(self) -> tuple[str, ...]:
        """Expected block IDs in original request order."""
        return tuple(block.block_id for block in self.blocks)


class RetryAction(StrEnum):
    """Action to take on a call attempt (ALG-05)."""

    GIVE_UP = "give_up"
    RETRY_AFTER_BACKOFF = "retry_after_backoff"
    RETRY_SAME_REQUEST = "retry_same_request"
    RETRY_WITH_REPAIR = "retry_with_repair"


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """Decision on a failed attempt: whether and how to retry.

    `error` is ready to be recorded on the task: a use case exhausting attempts yields it
    to the runner, transitioning the task to `failed` with type, message, and detail (PRD § 16).
    """

    action: RetryAction
    error: TaskError
    backoff_seconds: float = 0.0
    repair_instruction: str | None = None

    @property
    def should_retry(self) -> bool:
        """Whether another attempt is warranted."""
        return self.action is not RetryAction.GIVE_UP

    def next_request[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMRequest[PayloadT]:
        """Request for the next attempt.

        Repair instructions are appended to the original request instructions:
        assembled context is neither lost nor replaced.
        """
        if not self.should_retry:
            raise RetryPolicyError(
                "политика запретила повтор: следующей попытки у этого вызова нет"
            )
        if self.repair_instruction is None:
            return replace(request, attempt=request.attempt + 1)
        return replace(
            request,
            instructions=f"{request.instructions}\n\n{self.repair_instruction}",
            attempt=request.attempt + 1,
        )


class RetryPolicy:
    """Model call retry rules (ALG-05).

    Max attempts is specified via configuration (`LLMSettings.max_attempts`) rather than hardcoded:
    local models, commercial providers, and debugging demand different thresholds.
    """

    def __init__(
        self,
        *,
        max_attempts: int,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
        max_retry_after_seconds: float = DEFAULT_MAX_RETRY_AFTER_SECONDS,
    ) -> None:
        if max_attempts < 1:
            raise RetryPolicyError("число попыток вызова модели должно быть положительным")
        if backoff_seconds < 0 or max_backoff_seconds < 0 or max_retry_after_seconds < 0:
            raise RetryPolicyError("задержка перед повтором не может быть отрицательной")
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._max_retry_after_seconds = max_retry_after_seconds

    @property
    def max_attempts(self) -> int:
        """Maximum call attempts permitted before transitioning task to `failed`."""
        return self._max_attempts

    def decide(self, error: Exception, *, attempts: int) -> RetryDecision:
        """Determines the fate of a failed attempt.

        `attempts` — number of attempts made so far, including this one. Tracked by the call loop:
        `ProcessingTask.attempts` counts task executions rather than individual model calls.

        Output truncation due to token limits is the sole model error that the policy never retries:
        the exact same request will hit the same ceiling (`ports/llm.py`), and any remaining attempts
        would be paid calls with predetermined outcomes.
        """
        if attempts < 1:
            raise RetryPolicyError("попытка вызова модели нумеруется с единицы")
        task_error = classify_error(error)
        if _truncated_by_output_limit(error):
            return RetryDecision(action=RetryAction.GIVE_UP, error=task_error)
        if not task_error.type.is_retried_automatically:
            return RetryDecision(action=RetryAction.GIVE_UP, error=task_error)
        if attempts >= self._max_attempts:
            return RetryDecision(action=RetryAction.GIVE_UP, error=task_error)
        if task_error.type is ErrorType.TRANSIENT:
            return RetryDecision(
                action=RetryAction.RETRY_AFTER_BACKOFF,
                error=task_error,
                backoff_seconds=self._backoff_for(attempts, error),
            )
        if task_error.type is ErrorType.MODEL_OUTPUT:
            return RetryDecision(action=RetryAction.RETRY_SAME_REQUEST, error=task_error)
        return RetryDecision(
            action=RetryAction.RETRY_WITH_REPAIR,
            error=task_error,
            repair_instruction=build_repair_instruction(
                _expectation_of(error), violation=task_error.message
            ),
        )

    def _backoff_for(self, attempts: int, error: Exception) -> float:
        """Delay before retrying a transient error: exponential backoff up to limit.

        If the provider explicitly supplied a delay (`Retry-After`), the pause is no shorter.
        """
        backoff = min(
            self._backoff_seconds * 2 ** (attempts - 1), self._max_backoff_seconds
        )
        retry_after = (
            error.retry_after_seconds if isinstance(error, LLMTransportError) else None
        )
        if retry_after is None:
            return backoff
        return max(backoff, min(retry_after, self._max_retry_after_seconds))


#: Mapping of exceptions to task error types (ALG-05). Order is significant:
#: subclasses must precede their base classes.
_ERROR_TYPES: tuple[tuple[type[Exception], ErrorType], ...] = (
    # unconfigured or provider rejected request: automatic retry yields same rejection;
    # user retries manually after correcting settings
    (LLMConfigurationError, ErrorType.CONFIGURATION),
    (LLMProviderError, ErrorType.CONFIGURATION),
    # unusable book: requires a different file rather than another attempt
    (EpubImportFatalError, ErrorType.PERMANENT),
    (EpubExportFatalError, ErrorType.PERMANENT),
    # timeout, connection drop, rate limit, provider-side glitch
    (LLMTransportError, ErrorType.TRANSIENT),
    # unparseable JSON, schema mismatch, token output cutoff
    (LLMResponseFormatError, ErrorType.MODEL_OUTPUT),
    (LLMPayloadError, ErrorType.MODEL_OUTPUT),
    # response parsed but invalid: missing block, unknown block, wrong treatment, wrong language
    (ResultValidationError, ErrorType.VALIDATION),
    # chapter fragments do not assemble cleanly: foreign blocks, overlapping numbers
    (AnalysisMergeError, ErrorType.VALIDATION),
    (TranslationUnitError, ErrorType.VALIDATION),
    (UnknownBlockError, ErrorType.VALIDATION),
    (NonTextualBlockError, ErrorType.VALIDATION),
    (BlockResultError, ErrorType.VALIDATION),
    (TreatmentPlanError, ErrorType.VALIDATION),
)


def classify_error(error: Exception) -> TaskError:
    """Typed task error derived from use case exception (ALG-05).

    Unrecognized errors default to transient: the application cannot assume retrying
    is futile, preserving the user's ability to re-run the task.
    """
    for error_type, task_error_type in _ERROR_TYPES:
        if isinstance(error, error_type):
            return _task_error(task_error_type, error)
    return _task_error(ErrorType.TRANSIENT, error)


def build_repair_instruction(
    expectation: ResultExpectation | None, *, violation: str | None = None
) -> str:
    """Refining instruction for retry (ALG-05).

    Lists expected `block_id`s, expected treatment modes, and target language — the aspects
    in which model output diverged from request. Appended to the original prompt.
    """
    lines = ["Уточнение к предыдущему запросу: предыдущий ответ отклонён проверкой."]
    if violation and violation.strip():
        lines.append(f"Причина отклонения: {violation.strip()}")
    if expectation is None:
        return "\n".join(lines)
    if expectation.blocks:
        lines.append(
            "Верни результат ровно для этих блоков, в том же порядке, без пропущенных "
            "и без посторонних:"
        )
        lines.extend(_block_lines(expectation))
    lines.append(f"Целевой язык результата — {expectation.target_language}.")
    return "\n".join(lines)


def _block_lines(expectation: ResultExpectation) -> list[str]:
    """Lines enumerating expected blocks along with treatment modes."""
    return [
        f"- {block.block_id}"
        if block.treatment is None
        else f"- {block.block_id}: режим обработки {block.treatment}"
        for block in expectation.blocks
    ]


def _truncated_by_output_limit(error: Exception) -> bool:
    """Whether the response was truncated by model output limits rather than semantically.

    Reported by provider in `LLMResponseFormatError.truncated`.
    Resolved by configuring a higher output limit or submitting smaller chunks, so retries
    with the same request or repair prompt are useless here.
    """
    return isinstance(error, LLMResponseFormatError) and error.truncated


def _expectation_of(error: Exception) -> ResultExpectation | None:
    """Expected result information if carried by error."""
    if isinstance(error, ResultValidationError):
        return error.expectation
    return None


def _task_error(error_type: ErrorType, error: Exception) -> TaskError:
    """Task error with concise message and technical detail."""
    return TaskError(
        type=error_type,
        message=str(error).strip() or type(error).__name__,
        detail=f"{type(error).__module__}.{type(error).__qualname__}",
    )


__all__ = [
    "DEFAULT_BACKOFF_SECONDS",
    "DEFAULT_MAX_BACKOFF_SECONDS",
    "DEFAULT_MAX_RETRY_AFTER_SECONDS",
    "ExpectedBlock",
    "ResultExpectation",
    "ResultValidationError",
    "RetryAction",
    "RetryDecision",
    "RetryPolicy",
    "RetryPolicyError",
    "build_repair_instruction",
    "classify_error",
]
