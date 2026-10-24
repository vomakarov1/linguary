"""EPUBCheck external validation adapter: availability, invocation, and report parsing (P2.T6).

Real EPUBCheck is not executed here: Java may be absent on the machine, and actual
book validation is the domain of that external tool rather than these tests. Instead,
an executable stub is used to record the received command line, write a pre-configured
report to `--json`, and exit with the specified code. This covers the exact branches
owned by the adapter (PRD § 20).
"""

from __future__ import annotations

import json
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from linguary.adapters.validation import EpubCheckSettings, EpubCheckValidator
from linguary.application.ports.export_validation import (
    ExportIssueSeverity,
    ExportValidationError,
    ExportValidationRequest,
)

#: Report of a tool run finding no issues in the book.
CLEAN_REPORT: dict[str, object] = {
    "checker": {"checkerVersion": "5.1.0", "filename": "book.epub"},
    "items": [],
    "messages": [],
}

#: Report with messages across all severity levels, including one unknown to the adapter.
REPORT_WITH_MESSAGES: dict[str, object] = {
    "checker": {"checkerVersion": "5.1.0"},
    "messages": [
        {
            "ID": "RSC-005",
            "severity": "ERROR",
            "message": "Ошибка при разборе файла: несоответствие схеме",
            "locations": [
                {"url": "OEBPS/chapter-1.xhtml", "line": 12, "column": 5},
            ],
            "additionalLocations": 3,
        },
        {
            "ID": "PKG-012",
            "severity": "WARNING",
            "message": "Имя файла содержит небезопасные символы",
            "locations": [{"url": "OEBPS/глава.xhtml"}],
        },
        {"ID": "INF-001", "severity": "INFO", "message": "Проверено 4 документа"},
        {"ID": "OPF-096", "severity": "USAGE", "message": "Устаревшее свойство"},
    ],
}


@dataclass(frozen=True, slots=True)
class Stub:
    """Substituted EPUBCheck stub: adapter settings and recorded command line."""

    settings: EpubCheckSettings
    argv_path: Path

    def argv(self) -> list[str]:
        """Arguments with which the adapter executed the tool."""
        return json.loads(self.argv_path.read_text(encoding="utf-8"))


def install_stub(
    directory: Path,
    *,
    report: object | str | None = CLEAN_REPORT,
    exit_code: int = 0,
    jar_directory: Path | None = None,
) -> Stub:
    """Installs a Java stub and tool file into the directory.

    The stub behaves like a real invocation: parses `--json`, writes the report there
    (`report` set to `None` writes nothing, simulating a crash prior to output), and exits
    with the given exit code.
    """
    argv_path = directory / "argv.json"
    payload = report if isinstance(report, str) or report is None else json.dumps(report)
    script = directory / "java"
    script.write_text(
        "\n".join(
            (
                f"#!{sys.executable}",
                "import json, pathlib, sys",
                f"argv_path = pathlib.Path({str(argv_path)!r})",
                "argv = sys.argv[1:]",
                "argv_path.write_text(json.dumps(argv), encoding='utf-8')",
                f"report = {payload!r}",
                "if report is not None and '--json' in argv:",
                "    target = pathlib.Path(argv[argv.index('--json') + 1])",
                "    target.parent.mkdir(parents=True, exist_ok=True)",
                "    target.write_text(report, encoding='utf-8')",
                f"sys.exit({exit_code})",
                "",
            )
        ),
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    jar = (jar_directory or directory) / "epubcheck.jar"
    jar.parent.mkdir(parents=True, exist_ok=True)
    jar.write_bytes(b"")
    return Stub(
        settings=EpubCheckSettings(
            java_executable=str(script),
            jar_path=None if jar_directory else str(jar),
            search_directories=(str(directory),),
        ),
        argv_path=argv_path,
    )


@pytest.fixture
def book(tmp_path: Path) -> Path:
    """Export artifact passed for validation."""
    path = tmp_path / "export" / "book.epub"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    return path


def request_for(book: Path, **overrides: object) -> ExportValidationRequest:
    """Request to validate the generated file."""
    arguments: dict[str, object] = {"epub_path": str(book)}
    arguments.update(overrides)
    return ExportValidationRequest(**arguments)  # type: ignore[arg-type]


def test_missing_java_leaves_the_book_unchecked(tmp_path: Path, book: Path) -> None:
    """Without Java, validation is skipped, and the adapter explains how to enable it."""
    stub = install_stub(tmp_path)
    validator = EpubCheckValidator(
        settings=EpubCheckSettings(
            java_executable=str(tmp_path / "absent-java"),
            jar_path=stub.settings.jar_path,
        )
    )

    report = validator.validate(request_for(book))

    assert report.available is False
    assert report.issues == ()
    assert report.unavailable_reason is not None
    assert "Java не найдена" in report.unavailable_reason


def test_missing_jar_leaves_the_book_unchecked(tmp_path: Path, book: Path) -> None:
    """Java present but tool missing: validation is also skipped."""
    stub = install_stub(tmp_path)
    validator = EpubCheckValidator(
        settings=EpubCheckSettings(
            java_executable=stub.settings.java_executable,
            jar_path=str(tmp_path / "absent" / "epubcheck.jar"),
        )
    )

    report = validator.validate(request_for(book))

    assert report.available is False
    assert report.unavailable_reason is not None
    assert "absent" in report.unavailable_reason


def test_jar_is_found_in_the_distribution_directory(
    tmp_path: Path, book: Path
) -> None:
    """Unconfigured tool is discovered in the distribution directory, including version folder."""
    stub = install_stub(tmp_path, jar_directory=tmp_path / "epubcheck-5.1.0")

    report = EpubCheckValidator(settings=stub.settings).validate(request_for(book))

    assert report.available is True
    assert stub.argv()[1] == str(tmp_path / "epubcheck-5.1.0" / "epubcheck.jar")


def test_clean_report_leaves_the_book_without_issues(
    tmp_path: Path, book: Path
) -> None:
    """Successful validation: no issues, report saved alongside the artifact."""
    stub = install_stub(tmp_path)

    report = EpubCheckValidator(settings=stub.settings).validate(request_for(book))

    assert report.available is True
    assert report.issues == ()
    assert report.report_path == str(book.with_suffix(".epubcheck.json"))
    assert json.loads(Path(report.report_path).read_text(encoding="utf-8")) == (
        CLEAN_REPORT
    )


def test_command_line_follows_the_official_contract(
    tmp_path: Path, book: Path
) -> None:
    """Tool is invoked using `-jar <jar> <file.epub> --json <report>` contract."""
    stub = install_stub(tmp_path)
    target = tmp_path / "reports" / "book.json"

    EpubCheckValidator(settings=stub.settings).validate(
        request_for(book, report_path=str(target))
    )

    assert stub.argv() == [
        "-jar",
        stub.settings.jar_path,
        str(book),
        "--json",
        str(target),
    ]
    assert target.is_file()


def test_strict_mode_turns_on_the_strict_mode_of_the_tool(
    tmp_path: Path, book: Path
) -> None:
    """Strict mode is passed to the tool itself, not only the export use case."""
    stub = install_stub(tmp_path)

    EpubCheckValidator(settings=stub.settings).validate(
        request_for(book, strict=True)
    )

    assert stub.argv()[-1] == "--failonwarnings"


def test_report_messages_become_issues_of_their_severity(
    tmp_path: Path, book: Path
) -> None:
    """Report errors and warnings become issues, informational messages are ignored."""
    stub = install_stub(tmp_path, report=REPORT_WITH_MESSAGES, exit_code=1)

    report = EpubCheckValidator(settings=stub.settings).validate(request_for(book))

    assert report.available is True
    assert [issue.severity for issue in report.issues] == [
        ExportIssueSeverity.ERROR,
        ExportIssueSeverity.WARNING,
    ]
    error, warning = report.issues
    assert error.message.startswith("RSC-005: Ошибка при разборе файла")
    assert error.location == "OEBPS/chapter-1.xhtml:12:5"
    assert warning.location == "OEBPS/глава.xhtml"


def test_nonzero_exit_without_messages_is_reported_as_an_error(
    tmp_path: Path, book: Path
) -> None:
    """Non-zero exit code without report messages is not treated as successful."""
    stub = install_stub(tmp_path, exit_code=1)

    report = EpubCheckValidator(settings=stub.settings).validate(request_for(book))

    assert report.available is True
    assert len(report.errors) == 1
    assert "кодом 1" in report.errors[0].message


def test_unknown_report_fields_do_not_break_the_parsing(
    tmp_path: Path, book: Path
) -> None:
    """Unknown report fields do not break parsing: format belongs to third-party tool."""
    stub = install_stub(
        tmp_path,
        report={
            "unknownSection": {"whatever": True},
            "messages": [
                {
                    "ID": "RSC-001",
                    "severity": "FATAL",
                    "message": "Файл не найден",
                    "unknownField": [1, 2],
                    "locations": [{"url": "OEBPS/lost.xhtml", "line": 1}],
                }
            ],
        },
        exit_code=1,
    )

    report = EpubCheckValidator(settings=stub.settings).validate(request_for(book))

    assert [issue.severity for issue in report.issues] == [ExportIssueSeverity.ERROR]
    assert report.issues[0].location == "OEBPS/lost.xhtml:1"


def test_missing_report_of_an_available_tool_is_a_technical_error(
    tmp_path: Path, book: Path
) -> None:
    """Tool ran but left no report: technical validation error."""
    stub = install_stub(tmp_path, report=None, exit_code=2)

    with pytest.raises(ExportValidationError, match="нельзя прочитать"):
        EpubCheckValidator(settings=stub.settings).validate(request_for(book))


def test_broken_report_of_an_available_tool_is_a_technical_error(
    tmp_path: Path, book: Path
) -> None:
    """Truncated JSON is also a technical error, not an issue-free book."""
    stub = install_stub(tmp_path, report='{"messages": [', exit_code=1)

    with pytest.raises(ExportValidationError, match="нельзя прочитать"):
        EpubCheckValidator(settings=stub.settings).validate(request_for(book))


def test_report_of_a_previous_run_is_not_taken_for_the_current_one(
    tmp_path: Path, book: Path
) -> None:
    """Report from a prior run must not be mistaken for the current result."""
    stub = install_stub(tmp_path, report=None)
    stale = book.with_suffix(".epubcheck.json")
    stale.write_text(json.dumps(CLEAN_REPORT), encoding="utf-8")

    with pytest.raises(ExportValidationError, match="нельзя прочитать"):
        EpubCheckValidator(settings=stub.settings).validate(request_for(book))
