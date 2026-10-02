"""Порт вызова LLM (PRD § 23, ADR § 3).

Прикладной слой знает только `LLMClient.generate`: он собирает текст запроса и получает
разобранный внутренний DTO. Провайдер, HTTP, формат структурированного ответа и Pydantic
остаются за адаптером `linguary.adapters.llm`.

Запрос типизирован видом ответа (`LLMRequest[ChapterAnalysisResponse]`), поэтому сценарий
получает нужный DTO без разбора union-а на своей стороне; адаптер выбирает схему разбора
по `kind`.

Метод асинхронный: вызов модели — это ожидание сети, а локальный runner задач исполняет
обработчики в цикле событий, и его настраиваемый параллелизм (PRD § 25) рассчитан именно
на такое ожидание.

Ошибки порта разделены так, чтобы политика повторов (PRD § 16, ALG-05) могла принять
решение, не разбирая текст сообщения:

- `LLMConfigurationError` — вызывать нечем: ключ или модель не настроены, провайдер
  отклонил ключ. Повтор бессмысленен, пока пользователь не исправит настройку;
- `LLMTransportError` — временная неудача: таймаут, обрыв связи, ограничение частоты,
  ошибка на стороне провайдера;
- `LLMProviderError` — провайдер отверг сам запрос и повторит тот же отказ;
- `LLMResponseFormatError` — ответ не разбирается по схеме, в том числе из-за обрыва
  генерации по лимиту токенов.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

from linguary.application.errors import ApplicationError
from linguary.application.llm_dto import ChapterAnalysisResponse, TranslationResultResponse
from linguary.domain import LLMCallStage, TokenUsage

#: Структурированные ответы, которые умеет возвращать модель (PRD § 23).
type LLMPayload = ChapterAnalysisResponse | TranslationResultResponse


class LLMError(ApplicationError):
    """Вызов модели не дал пригодного результата."""


class LLMConfigurationError(LLMError):
    """LLM не настроена или провайдер не принял настройку.

    Анализ и обработка книги с такой ошибкой не стартуют: Web UI показывает состояние
    конфигурации и действие для исправления, CLI — диагностическое сообщение (PRD § 23).
    Текст ошибки не содержит ключа провайдера.
    """


class LLMTransportError(LLMError):
    """Временная неудача вызова: таймаут, обрыв связи, ограничение частоты, сбой провайдера.

    `retry_after_seconds` — сколько провайдер попросил подождать до следующего запроса
    (заголовок `Retry-After` ответов `429` и `503`), если он это сообщил. Повтор раньше
    срока получит тот же отказ, поэтому политика повторов не делает паузу короче.
    """

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LLMProviderError(LLMError):
    """Провайдер отверг запрос и на повтор ответит тем же."""


class LLMResponseFormatError(LLMError):
    """Ответ модели не разбирается по схеме структурированного ответа.

    `truncated` означает, что провайдер сообщил об обрыве генерации по лимиту токенов:
    ответ оборван не моделью по смыслу, а пределом вывода, и повторять его имеет смысл
    только с другим лимитом.
    """

    def __init__(
        self,
        message: str,
        *,
        truncated: bool = False,
        raw_content: str | None = None,
        usage: TokenUsage | None = None,
        cost: Decimal | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.truncated = truncated
        self.raw_content = raw_content
        self.usage = usage
        self.cost = cost
        self.provider = provider
        self.model = model


class LLMResponseKind(StrEnum):
    """Вид структурированного ответа, который запрашивает сценарий."""

    CHAPTER_ANALYSIS = "chapter_analysis"
    TRANSLATION_RESULT = "translation_result"


@dataclass(frozen=True, slots=True)
class LLMRequest[PayloadT: LLMPayload]:
    """Запрос к модели: постоянная инструкция и собранный контекст.

    Текст запроса целиком собирает `ContextBuilder`: порт не знает ни о главах, ни о
    глоссарии. `kind` определяет, какую схему структурированного ответа адаптер потребует
    от провайдера и какой DTO вернёт сценарию.
    """

    kind: LLMResponseKind
    instructions: str
    input_text: str
    book_id: str | None = None
    stage: LLMCallStage | None = None
    task_id: str | None = None
    attempt: int = 1

    def __post_init__(self) -> None:
        if not self.instructions.strip():
            raise LLMError("инструкция запроса к модели пуста")
        if not self.input_text.strip():
            raise LLMError("содержимое запроса к модели пусто")
        if self.attempt < 1:
            raise LLMError("номер попытки вызова модели начинается с единицы")
        if (self.book_id is None) != (self.stage is None):
            raise LLMError("журналируемый запрос должен содержать книгу и этап")
        if self.book_id is not None and not self.book_id.strip():
            raise LLMError("идентификатор книги для журнала LLM пуст")
        if self.task_id is not None and not self.task_id.strip():
            raise LLMError("идентификатор задачи для журнала LLM пуст")

    def with_logging(
        self,
        *,
        book_id: str,
        stage: LLMCallStage,
        task_id: str | None = None,
    ) -> LLMRequest[PayloadT]:
        """Возвращает тот же запрос с метаданными для журнала вызовов."""
        return replace(
            self, book_id=book_id, stage=stage, task_id=task_id, attempt=1
        )

    @classmethod
    def for_chapter_analysis(
        cls, *, instructions: str, input_text: str
    ) -> LLMRequest[ChapterAnalysisResponse]:
        """Запрос анализа главы или её фрагмента (PRD § 11, § 14)."""
        return LLMRequest[ChapterAnalysisResponse](
            kind=LLMResponseKind.CHAPTER_ANALYSIS,
            instructions=instructions,
            input_text=input_text,
        )

    @classmethod
    def for_translation(
        cls, *, instructions: str, input_text: str
    ) -> LLMRequest[TranslationResultResponse]:
        """Запрос обработки группы блоков или одного блока (PRD § 10, § 14)."""
        return LLMRequest[TranslationResultResponse](
            kind=LLMResponseKind.TRANSLATION_RESULT,
            instructions=instructions,
            input_text=input_text,
        )


@dataclass(frozen=True, slots=True)
class LLMResponse[PayloadT: LLMPayload]:
    """Разобранный ответ модели вместе с техническими данными вызова.

    `truncated` — признак обрыва по лимиту токенов, если провайдер его сообщает: это
    блокирующая проверка результата (PRD § 15). `usage` заполняется, только когда
    провайдер даёт объём токенов; журнал вызовов допускает его отсутствие (PRD § 24).
    """

    payload: PayloadT
    provider: str
    model: str
    truncated: bool = False
    usage: TokenUsage | None = None
    call_id: str | None = None
    raw_content: str | None = field(default=None, repr=False)
    cost: Decimal | None = None


class LLMClient(Protocol):
    """Вызов модели со структурированным ответом."""

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        """Выполняет запрос и возвращает разобранный ответ.

        Реализация обязана приводить любую неудачу к наследнику `LLMError`: политика
        повторов различает их по типу, а не по тексту (PRD § 16).
        """


__all__ = [
    "LLMClient",
    "LLMConfigurationError",
    "LLMError",
    "LLMPayload",
    "LLMProviderError",
    "LLMRequest",
    "LLMResponse",
    "LLMResponseFormatError",
    "LLMResponseKind",
    "LLMTransportError",
]
