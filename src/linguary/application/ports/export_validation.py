"""Port for external validation of exported EPUB (PRD § 20).

The exporter's own technical checks are always executed and require no external tools.
This port describes optional validation by a third-party validator — in MVP this is
EPUBCheck, which requires an installed Java runtime, so on a user's machine it might
not be available at all.

The adapter reports only facts: whether the validator is available, what issues it found,
and where its report is saved. Decisions based on these facts are made by the export use case:

- validator unavailable — export succeeds, the book is marked as unvalidated;
- errors found — export task fails;
- warnings do not block export, but in strict mode they are treated as errors;
- technical failure when launching an available validator — `ExportValidationError`,
  and the export task fails as well.

The separation exists for the adapter: connecting EPUBCheck requires no application layer
changes, while policy remains consolidated in one place — the export use case.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from linguary.application.errors import ApplicationError


class ExportValidationError(ApplicationError):
    """An available validator failed to execute the check.

    This is a technical launch error, not a negative validation result: details and a retry
    action are displayed to the user.
    """


class ExportIssueSeverity(StrEnum):
    """Severity of an issue reported by the external validator."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ExportValidationIssue:
    """A single report issue: severity, message, and location in the book."""

    severity: ExportIssueSeverity
    message: str
    location: str | None = None

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ExportValidationError("текст замечания внешней проверки пуст")


@dataclass(frozen=True, slots=True)
class ExportValidationRequest:
    """Request for external validation of an assembled file.

    `report_path` — destination for the validator report; it is stored alongside the export
    artifact (PRD § 20). `strict` is passed to the adapter so it can enable the tool's strict
    mode; the final decision is still made by the export use case.
    """

    epub_path: str
    report_path: str | None = None
    strict: bool = False

    def __post_init__(self) -> None:
        if not self.epub_path.strip():
            raise ExportValidationError("путь к проверяемому файлу пуст")


@dataclass(frozen=True, slots=True)
class ExportValidationReport:
    """Result of external validation.

    An unavailable validator is a normal, expected outcome: `available` is `False`,
    there are no issues, and the reason for unavailability is stored in `unavailable_reason`
    and displayed to the user alongside instructions on how to enable validation.
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
        """Issues that cause validation to fail."""
        return self._with_severity(ExportIssueSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ExportValidationIssue, ...]:
        """Issues that do not block export by themselves."""
        return self._with_severity(ExportIssueSeverity.WARNING)

    def _with_severity(
        self, severity: ExportIssueSeverity
    ) -> tuple[ExportValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is severity)


class ExportValidator(Protocol):
    """External validation of exported EPUB."""

    def validate(self, request: ExportValidationRequest) -> ExportValidationReport:
        """Validates the assembled file and returns a report.

        Unavailability of the tool is not an error: the implementation returns a report with
        `available = False`. `ExportValidationError` is raised only when the tool exists,
        but the check failed to run.
        """


__all__ = [
    "ExportIssueSeverity",
    "ExportValidationError",
    "ExportValidationIssue",
    "ExportValidationReport",
    "ExportValidationRequest",
    "ExportValidator",
]
