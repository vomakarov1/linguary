"""Классификация ошибок и политика повторов (PRD § 16, ALG-05).

Модуль отвечает на два вопроса и ни на что больше: какого типа ошибка случилась и что
делать с очередной попыткой. Он не выполняет вызовов, не ждёт между попытками и не знает
о задачах очереди: цикл повторов ведёт сценарий, у которого есть клиент модели и исходный
запрос, а ожидание между попытками выполняет тот, кто владеет циклом событий.

Ошибки не образуют собственной иерархии поверх доменной: классификация отображает
исключения в `ErrorType` домена, и решение о повторе принимает по нему же. Разбор идёт
по типу исключения, а не по тексту сообщения — типы порта `LLMClient` для этого и
разделены (`ports/llm.py`).

Нарушения блокирующих проверок результата (ALG-04) приходят сюда как `ResultValidationError`
вместе с `ResultExpectation`: из тех же сведений — ожидаемые `block_id`, ожидаемые режимы
обработки и целевой язык — строится уточняющая инструкция повтора. Она дописывается
к инструкции исходного запроса и не заменяет её: контекст главы, глоссарий и текст блоков
собирает `ContextBuilder`, и повтор их не теряет.

Ошибка одной задачи здесь не останавливает обработку книги: политика решает судьбу одной
попытки одного вызова, а состояние задачи фиксирует `TaskRunner` (ALG-02).
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

#: Задержка перед первой повторной попыткой временной ошибки. Дальше она удваивается:
#: ограничение частоты у провайдера снимается не мгновенно, а плотный повтор его продлевает.
DEFAULT_BACKOFF_SECONDS = 1.0

#: Предел задержки. Дальше ждать бессмысленно: пользователь видит задачу в работе, а не
#: результат, и честнее исчерпать попытки, чем растянуть одну задачу на минуты.
DEFAULT_MAX_BACKOFF_SECONDS = 30.0

#: Предел паузы, которую провайдер может попросить заголовком `Retry-After`. Просьбу
#: провайдера политика выполняет и сверх `DEFAULT_MAX_BACKOFF_SECONDS` — иначе повтор
#: снова получит отказ, — но не бесконечно: задача не должна ждать часами.
DEFAULT_MAX_RETRY_AFTER_SECONDS = 120.0


class RetryPolicyError(ApplicationError):
    """Политика повторов настроена или вызвана недопустимым образом."""


class ResultValidationError(ApplicationError):
    """Результат вызова отбракован блокирующей проверкой (PRD § 15, ALG-04).

    Ошибка описывает не сбой вызова, а негодный ответ модели: нет ожидаемого блока, есть
    неизвестный блок, режим обработки не совпал с планом, язык результата не целевой.
    Такой ответ повторяют с уточняющей инструкцией, поэтому ошибка несёт `expectation` —
    сведения, которых модели не хватило.
    """

    def __init__(self, message: str, *, expectation: ResultExpectation | None = None) -> None:
        super().__init__(message)
        self.expectation = expectation


@dataclass(frozen=True, slots=True)
class ExpectedBlock:
    """Блок, результат для которого обязан вернуть повторный вызов.

    `treatment` не задан там, где режим выбирает сама модель: при анализе главы план
    обработки — это её ответ, а не требование запроса (PRD § 14).
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
    """Что именно должен вернуть повторный вызов (ALG-05).

    Ровно те сведения, которые перечисляет уточняющая инструкция: ожидаемые блоки в их
    порядке, ожидаемые режимы обработки и целевой язык результата.
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
        """Ожидаемые идентификаторы блоков в порядке исходного запроса."""
        return tuple(block.block_id for block in self.blocks)


class RetryAction(StrEnum):
    """Что делать с очередной попыткой вызова (ALG-05)."""

    GIVE_UP = "give_up"
    RETRY_AFTER_BACKOFF = "retry_after_backoff"
    RETRY_SAME_REQUEST = "retry_same_request"
    RETRY_WITH_REPAIR = "retry_with_repair"


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """Решение по одной неудачной попытке: повторять ли и чем именно.

    `error` готов к записи в задачу: сценарий, исчерпавший попытки, отдаёт его runner-у,
    и задача получает `failed` с типом, сообщением и технической деталью (PRD § 16).
    """

    action: RetryAction
    error: TaskError
    backoff_seconds: float = 0.0
    repair_instruction: str | None = None

    @property
    def should_retry(self) -> bool:
        """Имеет ли смысл ещё одна попытка."""
        return self.action is not RetryAction.GIVE_UP

    def next_request[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMRequest[PayloadT]:
        """Запрос следующей попытки.

        Уточняющая инструкция дописывается к инструкции исходного запроса: собранный
        контекст повтор не теряет и не подменяет.
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
    """Правила повторов вызова модели (ALG-05).

    Число попыток задаётся настройкой (`LLMSettings.max_attempts`), а не константой кода:
    локальная модель, платный провайдер и отладка требуют разного.
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
        """Сколько попыток вызова допускается до перехода задачи в `failed`."""
        return self._max_attempts

    def decide(self, error: Exception, *, attempts: int) -> RetryDecision:
        """Решает судьбу попытки, завершившейся ошибкой.

        `attempts` — сколько попыток уже сделано, включая эту. Счётчик ведёт цикл вызова:
        `ProcessingTask.attempts` считает запуски задачи, а не обращения к модели.

        Обрыв ответа по лимиту вывода — единственная ошибка модели, которую политика
        не повторяет: тот же запрос упирается в тот же предел (`ports/llm.py`), и каждая
        оставшаяся попытка была бы оплаченным вызовом с заранее известным итогом.
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
        """Задержка перед повтором временной ошибки: удвоение до предела.

        Если провайдер сам назвал паузу (`Retry-After`), задержка не короче её.
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


#: Отображение исключений в типы ошибок задачи (ALG-05). Порядок значим: наследник
#: указывается раньше своего базового класса.
_ERROR_TYPES: tuple[tuple[type[Exception], ErrorType], ...] = (
    # вызывать нечем или провайдер отверг сам запрос: автоматический повтор даст тот же
    # отказ, а после исправления настроек LLM пользователь повторяет задачу сам
    (LLMConfigurationError, ErrorType.CONFIGURATION),
    (LLMProviderError, ErrorType.CONFIGURATION),
    # книга непригодна: нужен другой файл, а не другая попытка
    (EpubImportFatalError, ErrorType.PERMANENT),
    (EpubExportFatalError, ErrorType.PERMANENT),
    # таймаут, обрыв связи, ограничение частоты, сбой на стороне провайдера
    (LLMTransportError, ErrorType.TRANSIENT),
    # неразбираемый JSON, ответ не по схеме, обрыв генерации по лимиту токенов
    (LLMResponseFormatError, ErrorType.MODEL_OUTPUT),
    (LLMPayloadError, ErrorType.MODEL_OUTPUT),
    # ответ разобран, но не годится: нет блока, неизвестный блок, режим, язык
    (ResultValidationError, ErrorType.VALIDATION),
    # фрагменты длинной главы не складываются в главу: чужие блоки, спорные номера
    (AnalysisMergeError, ErrorType.VALIDATION),
    (TranslationUnitError, ErrorType.VALIDATION),
    (UnknownBlockError, ErrorType.VALIDATION),
    (NonTextualBlockError, ErrorType.VALIDATION),
    (BlockResultError, ErrorType.VALIDATION),
    (TreatmentPlanError, ErrorType.VALIDATION),
)


def classify_error(error: Exception) -> TaskError:
    """Типизированная ошибка задачи по исключению сценария (ALG-05).

    Неизвестная ошибка считается временной: приложение не берётся утверждать, что повтор
    заведомо бесполезен, и пользователь сохраняет право запустить задачу заново.
    """
    for error_type, task_error_type in _ERROR_TYPES:
        if isinstance(error, error_type):
            return _task_error(task_error_type, error)
    return _task_error(ErrorType.TRANSIENT, error)


def build_repair_instruction(
    expectation: ResultExpectation | None, *, violation: str | None = None
) -> str:
    """Уточняющая инструкция повтора (ALG-05).

    Перечисляет ожидаемые `block_id`, ожидаемые режимы обработки и целевой язык — то,
    в чём ответ модели разошёлся с запросом. Инструкция дописывается к исходному запросу
    и без него смысла не имеет.
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
    """Строки перечня ожидаемых блоков с режимами обработки."""
    return [
        f"- {block.block_id}"
        if block.treatment is None
        else f"- {block.block_id}: режим обработки {block.treatment}"
        for block in expectation.blocks
    ]


def _truncated_by_output_limit(error: Exception) -> bool:
    """Оборван ли ответ пределом вывода модели, а не самой моделью по смыслу.

    Признак приходит от провайдера и попадает в `LLMResponseFormatError.truncated`.
    Лечится он другим лимитом вывода или меньшим запросом, поэтому повтор — того же
    запроса или с уточняющей инструкцией — здесь смысла не имеет.
    """
    return isinstance(error, LLMResponseFormatError) and error.truncated


def _expectation_of(error: Exception) -> ResultExpectation | None:
    """Сведения об ожидаемом результате, если ошибка их несёт."""
    if isinstance(error, ResultValidationError):
        return error.expectation
    return None


def _task_error(error_type: ErrorType, error: Exception) -> TaskError:
    """Ошибка задачи с коротким сообщением и технической деталью."""
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
