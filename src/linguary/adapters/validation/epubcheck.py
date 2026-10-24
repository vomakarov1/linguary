"""Validation of built EPUBs using external EPUBCheck (PRD § 20).

Implementation of `linguary.application.ports.export_validation.ExportValidator` port
on top of the official command-line contract:

```text
java -jar epubcheck.jar <file.epub> --json <report.json>
```

EPUBCheck requires a Java installation that may not be present on the user's machine
(ADR § 6, risk 8), so tool unavailability is a normal outcome rather than an error:
the adapter returns a report with `available = False` and the reason why validation was not performed.
Java and `epubcheck.jar` are read from local configuration (`EpubCheckSettings`),
and if not specified, are located automatically.

The adapter reports facts only: whether the tool is available, what issues it identified, and where
its report was saved. Decisions based on these facts — "unverified book", "export failed",
"warnings are non-blocking" — are made by the export use case and are not duplicated here.

The EPUBCheck report is a public contract of a third-party tool, and is parsed as-is:
unrecognized fields and severity levels are ignored, while an incomplete report (the process crashed
before writing the full JSON) raises `ExportValidationError` — a technical error of an available tool.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from linguary.application.ports.export_validation import (
    ExportIssueSeverity,
    ExportValidationError,
    ExportValidationIssue,
    ExportValidationReport,
    ExportValidationRequest,
    ExportValidator,
)

#: Tool filename in the EPUBCheck distribution.
_JAR_NAME = "epubcheck.jar"

#: Report extension if the export use case does not specify a path: the report is placed
#: alongside the exported artifact (PRD § 20).
_REPORT_SUFFIX = ".epubcheck.json"

#: Directories where `epubcheck.jar` is searched for when not specified in configuration.
#: Along with the application home directory, this includes package manager directories
#: as users frequently install the tool via package managers rather than manual extraction.
_SEARCH_DIRECTORIES = (
    "~/.linguary/epubcheck",
    "/opt/homebrew/opt/epubcheck/libexec",
    "/usr/local/opt/epubcheck/libexec",
    "/usr/share/java/epubcheck",
)

#: EPUBCheck message severity levels relevant to export. `INFO`, `USAGE`, and `SUPPRESSED`
#: describe the check itself rather than the book, and are omitted from issues.
_SEVERITIES: Mapping[str, ExportIssueSeverity] = {
    "FATAL": ExportIssueSeverity.ERROR,
    "ERROR": ExportIssueSeverity.ERROR,
    "WARNING": ExportIssueSeverity.WARNING,
}

#: Timeout in seconds for validating a single book. EPUBCheck parses the entire book,
#: so the limit is generous: it exists to prevent a stalled process from halting export indefinitely.
_DEFAULT_TIMEOUT_SECONDS = 600.0

#: Maximum characters from tool output included in technical error details.
_OUTPUT_LIMIT = 500


class _ToolUnavailable(Exception):
    """EPUBCheck was not found on the machine.

    Internal adapter exception: not exposed externally, converted to a report
    with `available = False` and a user-facing explanation.
    """


@dataclass(frozen=True, slots=True)
class EpubCheckSettings:
    """Local configuration for external validation.

    Empty values indicate autodetection: `java` is looked up in `PATH`, and `epubcheck.jar`
    in known installation directories. Explicit paths override autodetection: if a user specifies
    a path and it does not exist, falling back to another file is prohibited.
    """

    java_executable: str = "java"
    jar_path: str | None = None
    search_directories: Sequence[str] = _SEARCH_DIRECTORIES
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.java_executable.strip():
            raise ValueError("имя исполняемого файла Java пусто")
        if self.timeout_seconds <= 0:
            raise ValueError("предел ожидания EPUBCheck должен быть положительным")


@dataclass(frozen=True, slots=True)
class EpubCheckAvailability:
    """Readiness status of external validation on this machine (PRD § 21.6).

    The configuration screen and `doctor` command inspect this before exporting books,
    so the check is decoupled from validation itself: building an EPUB is not required.
    Discovered paths are included in the result so the user can verify the exact executable invoked.
    """

    available: bool
    reason: str | None = None
    java_path: str | None = None
    jar_path: str | None = None


class EpubCheckValidator(ExportValidator):
    """External validation of exported EPUB using the EPUBCheck tool."""

    def __init__(self, *, settings: EpubCheckSettings | None = None) -> None:
        self._settings = settings or EpubCheckSettings()

    def availability(self) -> EpubCheckAvailability:
        """Locate Java and `epubcheck.jar` without performing validation."""
        try:
            tool = _locate(self._settings)
        except _ToolUnavailable as unavailable:
            return EpubCheckAvailability(available=False, reason=str(unavailable))
        return EpubCheckAvailability(
            available=True, java_path=tool.java, jar_path=tool.jar
        )

    def validate(self, request: ExportValidationRequest) -> ExportValidationReport:
        """Validate built file; an unavailable tool is not treated as an error."""
        try:
            tool = _locate(self._settings)
        except _ToolUnavailable as unavailable:
            return ExportValidationReport(
                available=False, unavailable_reason=str(unavailable)
            )
        report_path = _report_path(request)
        completed = self._run(tool, request=request, report_path=report_path)
        issues = _issues(_read_report(report_path, completed=completed))
        return ExportValidationReport(
            available=True,
            issues=issues or _failure_issues(completed),
            report_path=str(report_path),
        )

    def _run(
        self,
        tool: _EpubCheckTool,
        *,
        request: ExportValidationRequest,
        report_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        """Run EPUBCheck; process execution failure is treated as a technical error.

        Any previous report is deleted before execution: a tool that crashes before writing
        its output would otherwise leave an outdated report that the adapter might mistake
        for the current check result.
        """
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.unlink(missing_ok=True)
            return subprocess.run(
                tool.command(request, report_path=report_path),
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=self._settings.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ExportValidationError(
                f"EPUBCheck не завершил проверку книги {request.epub_path} "
                f"за {self._settings.timeout_seconds:g} с"
            ) from error
        except OSError as error:
            raise ExportValidationError(
                f"EPUBCheck не удалось запустить: {error}"
            ) from error


@dataclass(frozen=True, slots=True)
class _EpubCheckTool:
    """Discovered pair of Java executable and tool jar file."""

    java: str
    jar: str

    def command(
        self, request: ExportValidationRequest, *, report_path: Path
    ) -> tuple[str, ...]:
        """Validation command line (PRD § 20).

        Strict mode is enabled via the tool's own flag: users opening the report manually
        should see the same result as the application.
        """
        command = (
            self.java,
            "-jar",
            self.jar,
            request.epub_path,
            "--json",
            str(report_path),
        )
        return (*command, "--failonwarnings") if request.strict else command


def _locate(settings: EpubCheckSettings) -> _EpubCheckTool:
    """Locate Java and `epubcheck.jar` via settings or autodetection."""
    return _EpubCheckTool(java=_locate_java(settings), jar=_locate_jar(settings))


def _locate_java(settings: EpubCheckSettings) -> str:
    """Locate Java executable; its absence indicates validation unavailability."""
    executable = shutil.which(settings.java_executable)
    if executable is None:
        raise _ToolUnavailable(
            f"Java не найдена ({settings.java_executable}): установите Java или укажите "
            f"путь к ней в настройках, чтобы включить проверку EPUBCheck"
        )
    return executable


def _locate_jar(settings: EpubCheckSettings) -> str:
    """Locate tool jar: explicitly configured or discovered in installation directories."""
    if settings.jar_path is not None:
        configured = Path(settings.jar_path).expanduser()
        if not configured.is_file():
            raise _ToolUnavailable(
                f"файл EPUBCheck {settings.jar_path} не найден: проверьте путь "
                f"в настройках"
            )
        return str(configured)
    found = next(_jar_candidates(settings.search_directories), None)
    if found is None:
        raise _ToolUnavailable(
            f"{_JAR_NAME} не найден: распакуйте EPUBCheck в ~/.linguary/epubcheck или "
            f"укажите путь к нему в настройках, чтобы включить проверку"
        )
    return str(found)


def _jar_candidates(directories: Sequence[str]) -> Iterator[Path]:
    """Iterate candidate jar files in search directories, including single-level version subdirectories.

    The EPUBCheck distribution extracts into a versioned subdirectory, so one level of
    nesting beneath search directories is inspected as well.
    """
    for directory in directories:
        root = Path(directory).expanduser()
        direct = root / _JAR_NAME
        if direct.is_file():
            yield direct
        yield from sorted(path for path in root.glob(f"*/{_JAR_NAME}") if path.is_file())


def _report_path(request: ExportValidationRequest) -> Path:
    """Target path where the tool writes its JSON report.

    Specified by the export use case; if omitted, the report is saved adjacent to the
    exported artifact with the same basename (PRD § 20).
    """
    if request.report_path is not None:
        return Path(request.report_path)
    return Path(request.epub_path).with_suffix(_REPORT_SUFFIX)


def _read_report(
    path: Path, *, completed: subprocess.CompletedProcess[str]
) -> Mapping[str, object]:
    """Parse the tool's JSON report; unparseable report is treated as a technical error.

    A tool that executed but left an unparseable report did not complete validation,
    and this cannot be reported as a successful check.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        raise ExportValidationError(
            f"отчёт EPUBCheck {path} нельзя прочитать: {error}"
            f"{_output_detail(completed)}"
        ) from error
    if not isinstance(payload, dict):
        raise ExportValidationError(
            f"отчёт EPUBCheck {path} не похож на отчёт инструмента"
            f"{_output_detail(completed)}"
        )
    return payload


def _issues(payload: Mapping[str, object]) -> tuple[ExportValidationIssue, ...]:
    """Extract issues from the report: messages with significant severity levels in appearance order."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ()
    issues = (_issue(message) for message in messages if isinstance(message, Mapping))
    return tuple(issue for issue in issues if issue is not None)


def _issue(message: Mapping[str, object]) -> ExportValidationIssue | None:
    """Parse a single report message; insignificant severities and empty texts are skipped."""
    severity = _SEVERITIES.get(str(message.get("severity", "")).upper())
    if severity is None:
        return None
    text = _message_text(message)
    if text is None:
        return None
    return ExportValidationIssue(
        severity=severity, message=text, location=_location(message)
    )


def _message_text(message: Mapping[str, object]) -> str | None:
    """Format issue message text with the tool code to facilitate looking up documentation."""
    code = _text(message.get("ID"))
    description = _text(message.get("message"))
    if code is not None and description is not None:
        return f"{code}: {description}"
    return description or code


def _location(message: Mapping[str, object]) -> str | None:
    """Extract issue location in the book: first document with line and column if present."""
    locations = message.get("locations")
    if not isinstance(locations, list) or not locations:
        return None
    first = locations[0]
    if not isinstance(first, Mapping):
        return None
    path = _text(first.get("url")) or _text(first.get("path"))
    if path is None:
        return None
    position = tuple(
        str(value)
        for value in (first.get("line"), first.get("column"))
        if isinstance(value, int)
    )
    return ":".join((path, *position))


def _failure_issues(
    completed: subprocess.CompletedProcess[str],
) -> tuple[ExportValidationIssue, ...]:
    """Create an issue for a non-zero exit code when the report contains no messages.

    A non-zero exit code indicates failed export (PRD § 20). Usually the tool explains it
    via report messages; if absent, the exit code itself becomes the reason — otherwise
    a failed validation would appear successful.
    """
    if not completed.returncode:
        return ()
    return (
        ExportValidationIssue(
            severity=ExportIssueSeverity.ERROR,
            message=(
                f"EPUBCheck завершился с кодом {completed.returncode} "
                f"без замечаний в отчёте{_output_detail(completed)}"
            ),
        ),
    )


def _output_detail(completed: subprocess.CompletedProcess[str]) -> str:
    """Extract tail of tool process output for technical error detail."""
    output = (completed.stderr or completed.stdout or "").strip()
    if not output:
        return ""
    return f": {output[:_OUTPUT_LIMIT]}"


def _text(value: object) -> str | None:
    """Return stripped non-empty string or `None`."""
    return value.strip() or None if isinstance(value, str) else None


__all__ = ["EpubCheckAvailability", "EpubCheckSettings", "EpubCheckValidator"]
