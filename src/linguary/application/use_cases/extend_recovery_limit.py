"""Extend the durable call budget of a translation root stopped on ``limit``."""

from __future__ import annotations

from dataclasses import dataclass

from linguary.application.errors import ApplicationError
from linguary.application.ports.persistence import UnitOfWork
from linguary.domain.book_conditions import WorkStateError


class ExtendRecoveryLimitError(ApplicationError):
    """The recovery call limit cannot be extended."""

    def __init__(self, message: str, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(message)


class ExtendRecoveryLimitConflict(ExtendRecoveryLimitError):
    """The root changed, or it is not a stopped ``limit`` root that can accept more calls."""


class ExtendRecoveryLimitNotFoundError(ApplicationError):
    """The book or recovery root is not visible to this caller. HTTP 404."""

    status_code = 404

    def __init__(self, message: str = "корень перевода не найден") -> None:
        self.reason_code = "not_found"
        self.status_code = 404
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ExtendRecoveryLimitCommand:
    """Add calls to one stopped limit root at the generation last shown to the caller."""

    book_id: str
    root_id: str
    additional_calls: int
    expected_generation: int


@dataclass(frozen=True, slots=True)
class ExtendRecoveryLimitResult:
    """Root budget after the extra calls are persisted."""

    root_id: str
    generation: int
    calls_used: int
    extra_calls: int
    limit: int


_CONFLICT_REASONS = frozenset({"generation_conflict", "unknown", "state_conflict"})
_MESSAGES = {
    "generation_conflict": "операция уже изменилась; обновите состояние и повторите",
    "unknown": "нельзя увеличить лимит, пока исход отправленного запроса неизвестен",
    "state_conflict": "лимит можно увеличить только у остановленной операции с причиной limit",
    "invalid_limit": "недопустимый размер увеличения лимита",
    "conditions_unavailable": "условия перевода или запуск книги недоступны",
}


class ExtendRecoveryLimit:
    """Increase ``extra_calls`` only for a root stopped with reason ``limit``.

    ``scope_limit`` stays a restart reason. This use case does not clear an unknown
    outcome and does not call the model.
    """

    def __init__(self, *, unit_of_work: UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def __call__(self, command: ExtendRecoveryLimitCommand) -> ExtendRecoveryLimitResult:
        with self._unit_of_work.transaction() as transaction:
            book = transaction.books.get(command.book_id)
            root = transaction.work.get_root(command.root_id)
            if book is None or root is None or root.book_id != command.book_id:
                raise ExtendRecoveryLimitNotFoundError("корень перевода не найден")
            try:
                updated = transaction.work.extend_recovery_limit(
                    command.root_id,
                    additional_calls=command.additional_calls,
                    expected_generation=command.expected_generation,
                )
            except WorkStateError as error:
                raise _translate(error) from None
        limit = updated.manifest["policy"]["max_calls"] + updated.extra_calls
        return ExtendRecoveryLimitResult(
            root_id=updated.id,
            generation=updated.generation,
            calls_used=updated.calls_used,
            extra_calls=updated.extra_calls,
            limit=limit,
        )


def _translate(error: WorkStateError) -> ApplicationError:
    reason = str(error)
    if reason == "root_not_found":
        return ExtendRecoveryLimitNotFoundError("корень перевода не найден")
    message = _MESSAGES.get(reason, "лимит восстановления нельзя изменить")
    if reason in _CONFLICT_REASONS:
        return ExtendRecoveryLimitConflict(message, reason_code=reason)
    return ExtendRecoveryLimitError(message, reason_code=reason)


__all__ = [
    "ExtendRecoveryLimit",
    "ExtendRecoveryLimitCommand",
    "ExtendRecoveryLimitConflict",
    "ExtendRecoveryLimitError",
    "ExtendRecoveryLimitNotFoundError",
    "ExtendRecoveryLimitResult",
]
