"""Проверка собранного EPUB внешним EPUBCheck (PRD § 20).

Реализация порта `linguary.application.ports.export_validation.ExportValidator` поверх
официального контракта командной строки:

```text
java -jar epubcheck.jar <file.epub> --json <report.json>
```

EPUBCheck требует установленной Java, которой на машине пользователя может не быть
(ADR § 6, риск 8), поэтому недоступность инструмента — обычный исход, а не ошибка:
адаптер возвращает отчёт с `available = False` и причиной, по которой проверка не
выполнялась. Java и `epubcheck.jar` берутся из локальной конфигурации (`EpubCheckSettings`),
а если она их не задаёт — ищутся автоматически.

Адаптер сообщает только факты: доступен ли инструмент, что он нашёл и куда сохранён его
отчёт. Решение по этим фактам — «непроверенная книга», «экспорт неуспешен», «предупреждения
не блокируют» — принимает сценарий экспорта, и здесь оно не повторяется.

Отчёт EPUBCheck — публичный контракт чужого инструмента, поэтому разбирается по факту:
незнакомые поля и уровни игнорируются, а неполный отчёт (процесс упал раньше, чем дописал
JSON) даёт `ExportValidationError` — техническую ошибку доступного инструмента.
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

#: Имя файла инструмента в дистрибутиве EPUBCheck.
_JAR_NAME = "epubcheck.jar"

#: Расширение отчёта, если сценарий экспорта не выбрал путь сам: отчёт кладётся рядом
#: с экспортным артефактом (PRD § 20).
_REPORT_SUFFIX = ".epubcheck.json"

#: Каталоги, в которых `epubcheck.jar` ищется, когда он не задан настройкой. Кроме
#: домашнего каталога приложения сюда входят каталоги пакетных менеджеров: инструмент
#: чаще ставят ими, чем распаковывают вручную.
_SEARCH_DIRECTORIES = (
    "~/.linguary/epubcheck",
    "/opt/homebrew/opt/epubcheck/libexec",
    "/usr/local/opt/epubcheck/libexec",
    "/usr/share/java/epubcheck",
)

#: Уровни сообщений EPUBCheck, значимые для экспорта. `INFO`, `USAGE` и `SUPPRESSED`
#: описывают саму проверку, а не книгу, и в замечания не попадают.
_SEVERITIES: Mapping[str, ExportIssueSeverity] = {
    "FATAL": ExportIssueSeverity.ERROR,
    "ERROR": ExportIssueSeverity.ERROR,
    "WARNING": ExportIssueSeverity.WARNING,
}

#: Сколько ждать проверку одной книги. EPUBCheck разбирает книгу целиком, поэтому предел
#: щедрый: он существует, чтобы зависший процесс не остановил экспорт навсегда.
_DEFAULT_TIMEOUT_SECONDS = 600.0

#: Сколько символов вывода инструмента попадает в текст технической ошибки.
_OUTPUT_LIMIT = 500


class _ToolUnavailable(Exception):
    """EPUBCheck на машине не найден.

    Внутренняя ошибка адаптера: наружу она не выходит, а превращается в отчёт
    с `available = False` и причиной для пользователя.
    """


@dataclass(frozen=True, slots=True)
class EpubCheckSettings:
    """Локальная конфигурация внешней проверки.

    Пустые значения означают автоопределение: `java` ищется в `PATH`, а `epubcheck.jar` —
    в известных каталогах установки. Явно заданный путь автоопределение отменяет: если
    пользователь указал файл, а его нет, подставлять вместо него другой нельзя.
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


class EpubCheckValidator(ExportValidator):
    """Внешняя проверка экспортированного EPUB инструментом EPUBCheck."""

    def __init__(self, *, settings: EpubCheckSettings | None = None) -> None:
        self._settings = settings or EpubCheckSettings()

    def validate(self, request: ExportValidationRequest) -> ExportValidationReport:
        """Проверяет собранный файл; недоступный инструмент — не ошибка."""
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
        """Запускает EPUBCheck; неудача самого запуска — техническая ошибка.

        Прошлый отчёт удаляется до запуска: инструмент, упавший до записи файла, иначе
        оставил бы адаптеру чужой отчёт, который тот принял бы за результат проверки.
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
    """Найденная пара «Java и jar-файл инструмента»."""

    java: str
    jar: str

    def command(
        self, request: ExportValidationRequest, *, report_path: Path
    ) -> tuple[str, ...]:
        """Командная строка проверки (PRD § 20).

        Строгий режим включается флагом самого инструмента: пользователь, открывший отчёт
        руками, должен видеть тот же результат, что и приложение.
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
    """Java и `epubcheck.jar` по настройке или автоопределением."""
    return _EpubCheckTool(java=_locate_java(settings), jar=_locate_jar(settings))


def _locate_java(settings: EpubCheckSettings) -> str:
    """Исполняемый файл Java; его отсутствие — причина недоступности проверки."""
    executable = shutil.which(settings.java_executable)
    if executable is None:
        raise _ToolUnavailable(
            f"Java не найдена ({settings.java_executable}): установите Java или укажите "
            f"путь к ней в настройках, чтобы включить проверку EPUBCheck"
        )
    return executable


def _locate_jar(settings: EpubCheckSettings) -> str:
    """Файл инструмента: заданный настройкой либо найденный в каталогах установки."""
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
    """Файлы инструмента в каталогах поиска, включая каталог версии дистрибутива.

    Дистрибутив EPUBCheck распаковывается каталогом с номером версии, поэтому рядом
    с самим каталогом поиска просматривается и один уровень вложенности.
    """
    for directory in directories:
        root = Path(directory).expanduser()
        direct = root / _JAR_NAME
        if direct.is_file():
            yield direct
        yield from sorted(path for path in root.glob(f"*/{_JAR_NAME}") if path.is_file())


def _report_path(request: ExportValidationRequest) -> Path:
    """Куда инструмент пишет JSON-отчёт.

    Путь выбирает сценарий экспорта; без него отчёт ложится рядом с экспортным
    артефактом под его же именем (PRD § 20).
    """
    if request.report_path is not None:
        return Path(request.report_path)
    return Path(request.epub_path).with_suffix(_REPORT_SUFFIX)


def _read_report(
    path: Path, *, completed: subprocess.CompletedProcess[str]
) -> Mapping[str, object]:
    """Разбирает отчёт инструмента; нечитаемый отчёт — техническая ошибка.

    Инструмент, который запустился, но не оставил разбираемого отчёта, не проверил книгу,
    и выдавать это за успешную проверку нельзя.
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
    """Замечания отчёта: сообщения значимых уровней в порядке их появления."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ()
    issues = (_issue(message) for message in messages if isinstance(message, Mapping))
    return tuple(issue for issue in issues if issue is not None)


def _issue(message: Mapping[str, object]) -> ExportValidationIssue | None:
    """Одно сообщение отчёта; незначимый уровень и пустой текст пропускаются."""
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
    """Текст замечания с кодом инструмента: по коду ищут объяснение в документации."""
    code = _text(message.get("ID"))
    description = _text(message.get("message"))
    if code is not None and description is not None:
        return f"{code}: {description}"
    return description or code


def _location(message: Mapping[str, object]) -> str | None:
    """Место замечания в книге: первый документ со строкой и колонкой, если они есть."""
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
    """Замечание о ненулевом коде выхода при отчёте без единого сообщения.

    Ненулевой код делает экспорт неуспешным (PRD § 20). Обычно инструмент объясняет его
    сообщениями отчёта; если объяснения нет, причиной становится сам код выхода — иначе
    неуспешная проверка выглядела бы как успешная.
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
    """Хвост вывода инструмента для технической детали ошибки."""
    output = (completed.stderr or completed.stdout or "").strip()
    if not output:
        return ""
    return f": {output[:_OUTPUT_LIMIT]}"


def _text(value: object) -> str | None:
    """Непустая строка поля отчёта либо `None`."""
    return value.strip() or None if isinstance(value, str) else None


__all__ = ["EpubCheckSettings", "EpubCheckValidator"]
