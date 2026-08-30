"""Русские названия состояний, режимов и ошибок для интерфейса (PRD § 21).

Домен говорит идентификаторами (`GLOSSARY_REVIEW`, `TRANSLATE`, `ModelOutputError`),
а пользователю нужны человеческие формулировки. Перевод живёт в одном месте, поэтому
список книг, карточка книги и экран обработки называют одно и то же одинаково.

Тексты написаны как объяснение состояния дел, а не как перевод технического термина:
в списке книг читается «ожидает проверки глоссария», а не «GLOSSARY_REVIEW».
"""

from __future__ import annotations

from collections.abc import Mapping

from linguary.application.ports.persistence import ExportValidationStatus
from linguary.domain import (
    BlockTreatment,
    BookStatus,
    ErrorType,
    ImportWarningKind,
    ResultWarningKind,
    TaskStatus,
    TaskType,
)

#: Предупреждение о передаче текста книги внешнему провайдеру (PRD § 4.1, § 21.6).
#: Показывается при загрузке книги и на экране конфигурации.
EXTERNAL_LLM_WARNING = (
    "Текст книги целиком отправляется внешнему провайдеру LLM: анализ глав и перевод "
    "выполняются на его серверах. Загружайте только те книги, которые можно передавать "
    "третьей стороне."
)

#: Подсказка, когда LLM не настроена (PRD § 23).
LLM_NOT_CONFIGURED_HINT = (
    "Укажите модель и ключ провайдера в файле настроек ~/.linguary/config.toml "
    "или в переменных окружения LINGUARY_LLM_MODEL и LINGUARY_LLM_API_KEY."
)

#: Подсказка, когда внешняя проверка EPUB недоступна (PRD § 20, § 21.6).
EPUBCHECK_HINT = (
    "Внешняя проверка EPUBCheck необязательна: без неё книга собирается и помечается "
    "как непроверенная. Чтобы включить проверку, установите Java и распакуйте EPUBCheck "
    "в ~/.linguary/epubcheck."
)

#: Предупреждение о сервере, открытом не только этой машине (PRD § 4.1).
NON_LOCAL_SERVER_WARNING = (
    "Приложение доступно не только с этой машины. Оно хранит ключ провайдера и полный "
    "текст книг, поэтому открывать его в сеть стоит только осознанно."
)

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

#: Названия режимов обработки блоков в терминах экрана обработки (PRD § 21.4).
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

#: Что означает тип ошибки для пользователя: можно ли повторить и от чего это зависит
#: (PRD § 16).
ERROR_TYPE_LABELS: Mapping[ErrorType, str] = {
    ErrorType.TRANSIENT: "Временный сбой — попробуйте повторить",
    ErrorType.MODEL_OUTPUT: "Модель вернула негодный ответ",
    ErrorType.VALIDATION: "Результат не прошёл проверку",
    ErrorType.PERMANENT: "Ошибка, которую повтор не исправит",
}

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


def book_status_label(status: BookStatus) -> str:
    """Название состояния книги для списка и карточки."""
    return BOOK_STATUS_LABELS[status]


def treatment_label(treatment: BlockTreatment) -> str:
    """Название режима обработки блока."""
    return TREATMENT_LABELS[treatment]


def task_type_label(task_type: TaskType) -> str:
    """Название типа фоновой задачи."""
    return TASK_TYPE_LABELS[task_type]


def task_status_label(status: TaskStatus) -> str:
    """Название состояния фоновой задачи."""
    return TASK_STATUS_LABELS[status]


def error_type_label(error_type: ErrorType) -> str:
    """Что тип ошибки означает для пользователя."""
    return ERROR_TYPE_LABELS[error_type]


def import_warning_label(kind: ImportWarningKind) -> str:
    """Название вида предупреждения импорта."""
    return IMPORT_WARNING_LABELS[kind]


def result_warning_label(kind: ResultWarningKind) -> str:
    """Название вида замечания к результату обработки."""
    return RESULT_WARNING_LABELS[kind]


def export_validation_status_label(status: ExportValidationStatus) -> str:
    """Пользовательский итог внешней проверки собранного EPUB."""
    return EXPORT_VALIDATION_STATUS_LABELS[status]


__all__ = [
    "BOOK_STATUS_LABELS",
    "EPUBCHECK_HINT",
    "ERROR_TYPE_LABELS",
    "EXPORT_VALIDATION_STATUS_LABELS",
    "EXTERNAL_LLM_WARNING",
    "IMPORT_WARNING_LABELS",
    "LLM_NOT_CONFIGURED_HINT",
    "NON_LOCAL_SERVER_WARNING",
    "RESULT_WARNING_LABELS",
    "TASK_STATUS_LABELS",
    "TASK_TYPE_LABELS",
    "TREATMENT_LABELS",
    "book_status_label",
    "error_type_label",
    "export_validation_status_label",
    "import_warning_label",
    "result_warning_label",
    "task_status_label",
    "task_type_label",
    "treatment_label",
]
