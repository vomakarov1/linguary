"""Технический журнал вызовов LLM (PRD § 24).

Журнал нужен для диагностики, оценки стоимости и разбора ошибок. Полные запросы и ответы
в нём не хранятся: они содержат текст книги. Отладочное сохранение payload остаётся
настройкой адаптера и в доменную запись не попадает.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from linguary.domain.errors import LLMCallLogError
from linguary.domain.tasks import TaskError


class LLMCallStage(StrEnum):
    """Этап обработки, на котором выполнен вызов."""

    CHAPTER_ANALYSIS = "chapter_analysis"
    TRANSLATION_UNIT = "translation_unit"
    BLOCK_REGENERATION = "block_regeneration"


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Объём токенов вызова, если провайдер сообщает такие данные."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "completion_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise LLMCallLogError(f"количество токенов не может быть отрицательным: {name}")

    @property
    def total_tokens(self) -> int | None:
        """Суммарный объём токенов либо `None`, если провайдер данных не дал."""
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


@dataclass(frozen=True, slots=True)
class LLMCallLog:
    """Запись журнала об одном вызове модели."""

    id: str
    book_id: str
    stage: LLMCallStage
    provider: str
    model: str
    started_at: datetime
    duration_ms: int
    attempt: int
    success: bool
    task_id: str | None = None
    error: TaskError | None = None
    usage: TokenUsage | None = None
    cost: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise LLMCallLogError("идентификатор записи журнала пуст")
        if not self.book_id.strip():
            raise LLMCallLogError("запись журнала не привязана к книге")
        if not self.provider.strip() or not self.model.strip():
            raise LLMCallLogError("провайдер и модель в записи журнала обязательны")
        if self.duration_ms < 0:
            raise LLMCallLogError("длительность вызова не может быть отрицательной")
        if self.attempt < 1:
            raise LLMCallLogError("номер попытки вызова начинается с единицы")
        if self.success and self.error is not None:
            raise LLMCallLogError(f"успешный вызов {self.id} не может содержать ошибку")
        if not self.success and self.error is None:
            raise LLMCallLogError(f"у неуспешного вызова {self.id} не указана ошибка")
        if self.cost is not None and self.cost < 0:
            raise LLMCallLogError("стоимость вызова не может быть отрицательной")
