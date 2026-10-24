"""Russian labels for statuses, modes, and errors in the user interface (PRD § 21).

The domain speaks identifiers (`GLOSSARY_REVIEW`, `TRANSLATE`, `ModelOutputError`),
while the user needs human-readable phrasing. Translations live in a single place, so
the book list, book card, and processing screen refer to the same things consistently.

Texts are written as explanations of the state of affairs rather than literal translations of technical terms:
the book list displays "waiting for glossary review" rather than "GLOSSARY_REVIEW".
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from linguary.application.ports.persistence import ExportValidationStatus
from linguary.domain import (
    BlockTreatment,
    BookStatus,
    ConceptType,
    ErrorType,
    ImportWarningKind,
    ResultWarningKind,
    TaskStatus,
    TaskType,
    TermStatus,
)

#: Warning about transmitting full book text to external provider (PRD § 4.1, § 21.6).
#: Displayed during book upload and on the configuration screen.
EXTERNAL_LLM_WARNING = (
    "Текст книги целиком отправляется внешнему провайдеру LLM: анализ глав и перевод "
    "выполняются на его серверах. Загружайте только те книги, которые можно передавать "
    "третьей стороне."
)

#: Hint when LLM is not configured (PRD § 23).
LLM_NOT_CONFIGURED_HINT = (
    "Укажите модель и ключ провайдера в файле настроек ~/.linguary/config.toml "
    "или в переменных окружения LINGUARY_LLM_MODEL и LINGUARY_LLM_API_KEY."
)

#: Hint when external EPUBCheck validator is unavailable (PRD § 20, § 21.6).
EPUBCHECK_HINT = (
    "Внешняя проверка EPUBCheck необязательна: без неё книга собирается и помечается "
    "как непроверенная. Чтобы включить проверку, установите Java и распакуйте EPUBCheck "
    "в ~/.linguary/epubcheck."
)

#: Warning when server is exposed beyond this machine (PRD § 4.1).
NON_LOCAL_SERVER_WARNING = (
    "Приложение доступно не только с этой машины. Оно хранит ключ провайдера и полный "
    "текст книг, поэтому открывать его в сеть стоит только осознанно."
)

#: Browser error page titles by response status code (API-01). Displayed
#: instead of raw JSON when a form is submitted from browser.
ERROR_PAGE_TITLES: Mapping[int, str] = {
    400: "Запрос не принят",
    403: "Действие запрещено",
    404: "Не найдено",
    405: "Действие недоступно",
    409: "Действие сейчас недоступно",
    422: "Форма заполнена неверно",
}

#: Error page title for status codes not described in the table.
DEFAULT_ERROR_PAGE_TITLE = "Не удалось выполнить действие"

BOOK_STATUS_LABELS: Mapping[BookStatus, str] = {
    BookStatus.IMPORTING: "Импортируется",
    BookStatus.IMPORTED: "Импортирована",
    BookStatus.ANALYZING: "Анализируется",
    BookStatus.GLOSSARY_REVIEW: "Ожидает проверки глоссария",
    BookStatus.READY_TO_PROCESS: "Готова к обработке",
    BookStatus.PROCESSING: "Обрабатывается",
    BookStatus.READY_TO_EXPORT: "Готова к сборке EPUB",
    BookStatus.COMPLETED: "Обработка завершена",
    BookStatus.FAILED: "Импорт не удался",
}

#: Block treatment mode labels in terms of the processing screen (PRD § 21.4).
TREATMENT_LABELS: Mapping[BlockTreatment, str] = {
    BlockTreatment.TRANSLATE: "Полный перевод",
    BlockTreatment.GLOSSARY_ONLY: "Словарные подсказки",
    BlockTreatment.READ_AS_IS: "Без помощи",
    BlockTreatment.PRESERVE_ONLY: "Сохранено без обработки",
    BlockTreatment.DROP_WITH_WARNING: "Отброшено с предупреждением",
}

TASK_TYPE_LABELS: Mapping[TaskType, str] = {
    TaskType.IMPORT_BOOK: "Импорт книги",
    TaskType.ANALYZE_CHAPTER: "Анализ главы",
    TaskType.PROCESS_TRANSLATION_UNIT: "Обработка главы",
    TaskType.REGENERATE_BLOCK: "Повторная генерация блока",
    TaskType.EXPORT_BOOK: "Сборка EPUB",
}

TASK_STATUS_LABELS: Mapping[TaskStatus, str] = {
    TaskStatus.PENDING: "Ожидает выполнения",
    TaskStatus.RUNNING: "Выполняется",
    TaskStatus.SUCCEEDED: "Готово",
    TaskStatus.FAILED: "Не удалось выполнить",
}

#: What error type means to the user: whether retry is possible and what it depends on
#: (PRD § 16).
ERROR_TYPE_LABELS: Mapping[ErrorType, str] = {
    ErrorType.TRANSIENT: "Временный сбой — попробуйте повторить",
    ErrorType.MODEL_OUTPUT: "Модель вернула негодный ответ",
    ErrorType.VALIDATION: "Результат не прошёл проверку",
    ErrorType.CONFIGURATION: (
        "Провайдер LLM отверг запрос — проверьте ключ, модель и баланс, затем повторите"
    ),
    ErrorType.PERMANENT: "Ошибка, которую повтор не исправит",
}

#: User decision on a glossary item (PRD § 12). A correction is stored under the same
#: `approved` status as an acceptance, differing only in translation text, so
#: it has its own label.
GLOSSARY_DECISION_LABELS: Mapping[TermStatus, str] = {
    TermStatus.APPROVED: "принято",
    TermStatus.LOCKED: "зафиксировано",
    TermStatus.SUGGESTED: "пропущено",
}

#: Label for translation variant corrected by user.
GLOSSARY_CORRECTED_LABEL = "исправлено"

#: Glossary concept type (PRD § 12): in item card and type correction list.
CONCEPT_TYPE_LABELS: Mapping[ConceptType, str] = {
    ConceptType.PERSON: "действующее лицо",
    ConceptType.PLACE: "место",
    ConceptType.ORGANIZATION: "организация",
    ConceptType.FICTIONAL_CONCEPT: "вымышленное понятие",
    ConceptType.TERM: "термин",
}

#: Book processing plan completeness on plan screen (PRD § 14): short label
#: and explanation of what it means.
PLAN_COMPLETE_LABEL = "План полный"
PLAN_COMPLETE_HINT = (
    "Все главы проанализированы, и у каждого текстового блока есть режим обработки."
)
PLAN_INCOMPLETE_LABEL = "План неполный"
PLAN_INCOMPLETE_HINT = (
    "Часть глав ещё не проанализирована или не у всех текстовых блоков есть режим "
    "обработки."
)

IMPORT_WARNING_LABELS: Mapping[ImportWarningKind, str] = {
    ImportWarningKind.DROPPED_BLOCK: "Фрагмент книги пропущен",
    ImportWarningKind.UNSUPPORTED_CONSTRUCT: "Неподдерживаемая конструкция",
    ImportWarningKind.TOC_MISMATCH: "Оглавление расходится с порядком чтения",
    ImportWarningKind.MISSING_RESOURCE: "Ресурс книги не найден",
}

RESULT_WARNING_LABELS: Mapping[ResultWarningKind, str] = {
    ResultWarningKind.REPETITION: "Повторы в переводе",
    ResultWarningKind.DROPPED_BLOCK: "Фрагмент книги пропущен",
}

EXPORT_VALIDATION_STATUS_LABELS: Mapping[ExportValidationStatus, str] = {
    ExportValidationStatus.NOT_VALIDATED: "EPUBCheck не выполнялся",
    ExportValidationStatus.PASSED: "EPUBCheck пройден",
    ExportValidationStatus.FAILED: "EPUBCheck нашёл ошибки",
}


#: What book card displays while stage has no completed model calls.
REMAINING_TIME_PENDING_LABEL = "Оценка оставшегося времени появится после первой главы"


def remaining_time_label(seconds: int | None) -> str:
    """Remaining stage time in words (QA-13).

    Rough estimate — call median, retries, and chapter length skew it, — so it
    is designated 'about' and rounded up to minutes: promising less than it takes is worse.
    """
    if seconds is None:
        return REMAINING_TIME_PENDING_LABEL
    if seconds < 60:
        return "Осталось меньше минуты"
    hours, minutes = divmod(math.ceil(seconds / 60), 60)
    if hours == 0:
        return f"Осталось примерно {minutes} мин"
    if minutes == 0:
        return f"Осталось примерно {hours} ч"
    return f"Осталось примерно {hours} ч {minutes} мин"


def error_page_title(status_code: int) -> str:
    """Error page title for response status code."""
    return ERROR_PAGE_TITLES.get(status_code, DEFAULT_ERROR_PAGE_TITLE)


def book_status_label(status: BookStatus) -> str:
    """Book status label for list and card."""
    return BOOK_STATUS_LABELS[status]


def treatment_label(treatment: BlockTreatment) -> str:
    """Block treatment mode label."""
    return TREATMENT_LABELS[treatment]


def task_type_label(task_type: TaskType) -> str:
    """Background task type label."""
    return TASK_TYPE_LABELS[task_type]


def task_status_label(status: TaskStatus) -> str:
    """Background task status label."""
    return TASK_STATUS_LABELS[status]


def error_type_label(error_type: ErrorType) -> str:
    """What error type means to the user."""
    return ERROR_TYPE_LABELS[error_type]


def concept_type_label(concept_type: ConceptType) -> str:
    """Glossary concept type label."""
    return CONCEPT_TYPE_LABELS[concept_type]


def glossary_decision_label(
    status: TermStatus, *, translation: str | None, suggested_translation: str
) -> str:
    """Glossary item decision in Russian: accepted, corrected, etc."""
    if status is TermStatus.APPROVED and translation not in (None, suggested_translation):
        return GLOSSARY_CORRECTED_LABEL
    return GLOSSARY_DECISION_LABELS[status]


def import_warning_label(kind: ImportWarningKind) -> str:
    """Import warning kind label."""
    return IMPORT_WARNING_LABELS[kind]


def result_warning_label(kind: ResultWarningKind) -> str:
    """Processing result warning kind label."""
    return RESULT_WARNING_LABELS[kind]


def export_validation_status_label(status: ExportValidationStatus) -> str:
    """User-facing outcome of external EPUBCheck validation."""
    return EXPORT_VALIDATION_STATUS_LABELS[status]


__all__ = [
    "BOOK_STATUS_LABELS",
    "CONCEPT_TYPE_LABELS",
    "DEFAULT_ERROR_PAGE_TITLE",
    "EPUBCHECK_HINT",
    "ERROR_PAGE_TITLES",
    "ERROR_TYPE_LABELS",
    "EXPORT_VALIDATION_STATUS_LABELS",
    "EXTERNAL_LLM_WARNING",
    "GLOSSARY_CORRECTED_LABEL",
    "GLOSSARY_DECISION_LABELS",
    "IMPORT_WARNING_LABELS",
    "LLM_NOT_CONFIGURED_HINT",
    "NON_LOCAL_SERVER_WARNING",
    "PLAN_COMPLETE_HINT",
    "PLAN_COMPLETE_LABEL",
    "PLAN_INCOMPLETE_HINT",
    "PLAN_INCOMPLETE_LABEL",
    "RESULT_WARNING_LABELS",
    "TASK_STATUS_LABELS",
    "TASK_TYPE_LABELS",
    "TREATMENT_LABELS",
    "book_status_label",
    "concept_type_label",
    "error_page_title",
    "error_type_label",
    "export_validation_status_label",
    "glossary_decision_label",
    "import_warning_label",
    "result_warning_label",
    "task_status_label",
    "task_type_label",
    "treatment_label",
]


# Prepared settings update messages; raw input is never interpolated into them.
SETTINGS_SAVED = "Настройки сохранены"
SETTINGS_RESET = "Настройки сброшены"
SETTINGS_INVALID = "Проверьте настройки"
SETTINGS_CONFLICT = "Настройки изменены. Загрузите актуальные настройки"
SETTINGS_NOT_APPLIED = "Изменения не применены"
SETTINGS_CLOSING = "Приложение завершает работу. Запрос не принят"
SETTINGS_UNSUPPORTED_MEDIA_TYPE = "Отправьте настройки в формате JSON или формы"
SETTINGS_UNKNOWN_FIELDS = "Запрос содержит неизвестные или повторяющиеся поля"
SETTINGS_REQUIRED_FIELD = "Заполните поле"
SETTINGS_STRING_FIELD = "Ожидается строка"
SETTINGS_INTEGER_FIELD = "Ожидается целое число"
SETTINGS_REVISION_FIELD = "Ожидается целая редакция не меньше нуля"
SETTINGS_SAFE_FIELD = "Укажите значение без ключа и учётных данных"
