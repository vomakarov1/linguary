"""Порт внешней проверки экспортированного EPUB (PRD § 20).

Собственные технические проверки экспортёра выполняются всегда и внешних инструментов
не требуют. Этот порт описывает дополнительную проверку сторонним валидатором — в MVP
это EPUBCheck, которому нужна установленная Java, поэтому на машине пользователя его
может не быть вовсе.

Адаптер сообщает только факты: доступен ли валидатор, какие замечания он нашёл и куда
сохранён его отчёт. Решение по этим фактам принимает прикладной сценарий экспорта:

- валидатор недоступен — экспорт успешен, книга помечается как непроверенная;
- найдены ошибки — экспортная задача неуспешна;
- предупреждения не блокируют экспорт, но в строгом режиме приравниваются к ошибкам;
- техническая ошибка запуска доступного валидатора — `ExportValidationError`, и экспортная
  задача тоже неуспешна.

Разделение сделано ради адаптера: чтобы подключить EPUBCheck, прикладной слой менять
не нужно, а вся политика остаётся в одном месте — в сценарии экспорта.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from linguary.application.errors import ApplicationError


class ExportValidationError(ApplicationError):
    """Доступный валидатор не смог выполнить проверку.

    Это техническая ошибка запуска, а не отрицательный результат проверки: пользователю
    показывается деталь и действие для повтора.
    """


class ExportIssueSeverity(StrEnum):
    """Значимость замечания внешнего валидатора."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ExportValidationIssue:
    """Одно замечание отчёта: значимость, текст и место в книге."""

    severity: ExportIssueSeverity
    message: str
    location: str | None = None

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ExportValidationError("текст замечания внешней проверки пуст")


@dataclass(frozen=True, slots=True)
class ExportValidationRequest:
    """Запрос на внешнюю проверку собранного файла.

    `report_path` — куда положить отчёт валидатора; он сохраняется рядом с экспортным
    артефактом (PRD § 20). `strict` передаётся адаптеру, чтобы тот мог включить строгий
    режим самого инструмента; итоговое решение всё равно принимает сценарий экспорта.
    """

    epub_path: str
    report_path: str | None = None
    strict: bool = False

    def __post_init__(self) -> None:
        if not self.epub_path.strip():
            raise ExportValidationError("путь к проверяемому файлу пуст")


@dataclass(frozen=True, slots=True)
class ExportValidationReport:
    """Результат внешней проверки.

    Недоступный валидатор — обычный, ожидаемый исход: `available` равен `False`, замечаний
    нет, а причина недоступности лежит в `unavailable_reason` и показывается пользователю
    вместе с подсказкой, как включить проверку.
    """

    available: bool
    issues: tuple[ExportValidationIssue, ...] = ()
    report_path: str | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.available and self.issues:
            raise ExportValidationError(
                "недоступный валидатор не может вернуть замечания по книге"
            )

    @property
    def errors(self) -> tuple[ExportValidationIssue, ...]:
        """Замечания, которые делают проверку неуспешной."""
        return self._with_severity(ExportIssueSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ExportValidationIssue, ...]:
        """Замечания, которые сами по себе экспорт не блокируют."""
        return self._with_severity(ExportIssueSeverity.WARNING)

    def _with_severity(
        self, severity: ExportIssueSeverity
    ) -> tuple[ExportValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is severity)


class ExportValidator(Protocol):
    """Внешняя проверка экспортированного EPUB."""

    def validate(self, request: ExportValidationRequest) -> ExportValidationReport:
        """Проверяет собранный файл и отдаёт отчёт.

        Недоступность инструмента — не ошибка: реализация возвращает отчёт с
        `available = False`. `ExportValidationError` поднимается только тогда, когда
        инструмент есть, но проверку выполнить не удалось.
        """


__all__ = [
    "ExportIssueSeverity",
    "ExportValidationError",
    "ExportValidationIssue",
    "ExportValidationReport",
    "ExportValidationRequest",
    "ExportValidator",
]
