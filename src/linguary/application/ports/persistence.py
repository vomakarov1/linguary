"""Порты хранения: репозитории и unit of work.

Прикладной слой работает с хранилищем только через эти протоколы и ничего не знает
ни о SQLAlchemy, ни о SQLite: конкретная реализация живёт в `linguary.adapters.db`.

Репозиторий отдаёт и принимает доменные сущности. Там, где доменной сущности нет
(строка книги собирается из метаданных, языков, состояния и путей файловой системы),
контракт описан обычным `dataclass` из стандартной библиотеки.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from linguary.application.errors import ApplicationError
from linguary.domain import (
    BlockProcessingResult,
    BookImportWarning,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    LLMCallLog,
    ProcessingRun,
    ProcessingTask,
    TaskIdentity,
    TermConcept,
)


class PersistenceError(ApplicationError):
    """Обращение к хранилищу вне допустимого состояния.

    Ошибками этого класса адаптер сообщает о нарушении контракта порта — например
    о работе с репозиторием вне открытой транзакции. Нарушения доменных инвариантов
    остаются наследниками `DomainError`.
    """


class ExportValidationStatus(StrEnum):
    """Результат проверки экспортированного EPUB внешним EPUBCheck (PRD § 20).

    До первого экспорта у книги значения нет. `NOT_VALIDATED` означает, что EPUBCheck
    на машине пользователя недоступен и книга экспортирована без внешней проверки.
    """

    NOT_VALIDATED = "not_validated"
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BookRecord:
    """Строка книги: то, что о книге знает хранилище помимо канонического документа.

    Каноническая модель документа (разделы, блоки, ресурсы) хранится отдельно и живёт
    в `DocumentRepository`; здесь — состояние книги и её связь с файловой системой.
    """

    id: str
    metadata: BookMetadata
    source_language: str
    target_language: str
    status: BookStatus
    source_epub_path: str
    created_at: datetime
    export_validation_status: ExportValidationStatus | None = None


class BookRepository(Protocol):
    """Книги и их пользовательское состояние."""

    def add(self, book: BookRecord) -> None:
        """Добавляет новую книгу."""

    def save(self, book: BookRecord) -> None:
        """Перезаписывает существующую книгу целиком."""

    def get(self, book_id: str) -> BookRecord | None:
        """Книга по идентификатору либо `None`, если её нет."""

    def list_all(self) -> tuple[BookRecord, ...]:
        """Все книги в порядке добавления: список книг Web UI."""


class DocumentRepository(Protocol):
    """Каноническая модель документа и предупреждения импорта."""

    def save(
        self,
        document: CanonicalDocument,
        *,
        stored_resource_paths: Mapping[str, str] | None = None,
    ) -> None:
        """Сохраняет документ целиком, заменяя прежние разделы, блоки и ресурсы.

        `stored_resource_paths` сопоставляет исходному пути ресурса путь, по которому
        импортёр сохранил его в файловой системе; домен этого пути не знает.
        """

    def get(self, book_id: str) -> CanonicalDocument | None:
        """Каноническая модель книги либо `None`, если такой книги нет.

        До завершения импорта документ существует, но может быть без разделов.
        """

    def stored_resource_paths(self, book_id: str) -> Mapping[str, str]:
        """Пути сохранённых ресурсов книги по их исходным путям."""

    def add_import_warnings(
        self, book_id: str, warnings: Sequence[BookImportWarning]
    ) -> None:
        """Добавляет предупреждения импорта книги."""

    def import_warnings(self, book_id: str) -> tuple[BookImportWarning, ...]:
        """Предупреждения импорта книги."""


class GlossaryRepository(Protocol):
    """Понятия глоссария книги вместе с их языковыми вариантами."""

    def save(self, concept: TermConcept) -> None:
        """Сохраняет понятие вместе с вариантами, заменяя прежнюю запись."""

    def get(self, concept_id: str) -> TermConcept | None:
        """Понятие по идентификатору либо `None`."""

    def list_for_book(self, book_id: str) -> tuple[TermConcept, ...]:
        """Все понятия глоссария книги."""

    def delete(self, concept_id: str) -> None:
        """Удаляет понятие вместе с его вариантами."""


class ChapterAnalysisRepository(Protocol):
    """Анализ главы: план обработки, кандидаты глоссария и учебные материалы главы."""

    def save(self, book_id: str, analysis: ChapterAnalysis) -> None:
        """Сохраняет анализ главы, заменяя прежний анализ той же главы."""

    def get_for_section(self, book_id: str, section_id: str) -> ChapterAnalysis | None:
        """Анализ конкретной главы либо `None`, если глава ещё не анализировалась."""

    def list_for_book(self, book_id: str) -> tuple[ChapterAnalysis, ...]:
        """Анализы всех проанализированных глав книги."""


class ProcessingRunRepository(Protocol):
    """Запуски обработки книги."""

    def add(self, run: ProcessingRun) -> None:
        """Добавляет новый запуск обработки."""

    def save(self, run: ProcessingRun) -> None:
        """Перезаписывает существующий запуск: смену состояния и снимки."""

    def get(self, run_id: str) -> ProcessingRun | None:
        """Запуск по идентификатору либо `None`."""

    def active_for_book(self, book_id: str) -> ProcessingRun | None:
        """Единственный активный запуск книги либо `None`."""

    def delete(self, run_id: str) -> None:
        """Удаляет запуск вместе с результатами блоков, заметками и замечаниями."""


class BlockResultRepository(Protocol):
    """Результаты обработки блоков вместе с заметками и замечаниями."""

    def replace_results(
        self, run_id: str, results: Sequence[BlockProcessingResult]
    ) -> None:
        """Заменяет результаты перечисленных блоков одной операцией (PRD § 17).

        Прежние результаты, заметки и замечания этих блоков удаляются, новые пишутся
        целиком: частично сохранённого результата `TranslationUnit` быть не должно.
        """

    def get(self, run_id: str, block_id: str) -> BlockProcessingResult | None:
        """Результат обработки одного блока в рамках запуска."""

    def list_for_run(self, run_id: str) -> tuple[BlockProcessingResult, ...]:
        """Все результаты запуска в порядке блоков исходного документа."""


class TaskRepository(Protocol):
    """Локальные возобновляемые задачи обработки."""

    def add(self, task: ProcessingTask) -> None:
        """Добавляет задачу; повтор идемпотентного ключа отвергается хранилищем."""

    def save(self, task: ProcessingTask) -> None:
        """Сохраняет новое состояние задачи после перехода."""

    def get(self, task_id: str) -> ProcessingTask | None:
        """Задача по идентификатору либо `None`."""

    def find_by_identity(self, identity: TaskIdentity) -> ProcessingTask | None:
        """Задача по идемпотентному ключу `book_id + task_type + scope_id`."""

    def list_for_book(self, book_id: str) -> tuple[ProcessingTask, ...]:
        """Задачи книги в порядке их создания."""

    def delete(self, task_id: str) -> None:
        """Удаляет задачу вместе с её идемпотентным ключом.

        Ключ `book_id + task_type + scope_id` живёт, пока живёт задача, и повторная
        постановка возвращает уже существующую задачу в её текущем состоянии. Поэтому
        работа, которую нужно выполнить заново, начинается с удаления прежней задачи:
        возврат к проверке глоссария снимает задачи отменённого запуска, а повторная
        генерация блока — свою прошлую задачу (PRD § 18.1, § 21.5).
        """


class LLMCallLogRepository(Protocol):
    """Технический журнал вызовов LLM (PRD § 24).

    Полные тексты запросов и ответов в журнал не попадают: они содержат текст книги.
    """

    def add(self, entry: LLMCallLog) -> None:
        """Добавляет запись журнала."""

    def list_for_book(self, book_id: str) -> tuple[LLMCallLog, ...]:
        """Записи журнала по книге в порядке выполнения вызовов."""


class UnitOfWork(Protocol):
    """Транзакционная граница прикладного сценария.

    Репозитории доступны только внутри `transaction()`: сценарий явно выбирает границу,
    а адаптер гарантирует, что все изменения внутри неё зафиксируются или откатятся
    целиком. Транзакция короткая и не охватывает HTTP-вызовы к LLM (ADR § 3).
    """

    @property
    def books(self) -> BookRepository:
        """Репозиторий книг."""

    @property
    def documents(self) -> DocumentRepository:
        """Репозиторий канонической модели документа."""

    @property
    def glossary(self) -> GlossaryRepository:
        """Репозиторий глоссария."""

    @property
    def analyses(self) -> ChapterAnalysisRepository:
        """Репозиторий анализа глав."""

    @property
    def runs(self) -> ProcessingRunRepository:
        """Репозиторий запусков обработки."""

    @property
    def results(self) -> BlockResultRepository:
        """Репозиторий результатов обработки блоков."""

    @property
    def tasks(self) -> TaskRepository:
        """Репозиторий локальных задач."""

    @property
    def llm_calls(self) -> LLMCallLogRepository:
        """Репозиторий журнала вызовов LLM."""

    def transaction(self) -> AbstractContextManager[UnitOfWork]:
        """Открывает транзакцию и делает репозитории доступными.

        Выход из блока без исключения фиксирует изменения, исключение — откатывает.
        """


__all__ = [
    "BlockResultRepository",
    "BookRecord",
    "BookRepository",
    "ChapterAnalysisRepository",
    "DocumentRepository",
    "ExportValidationStatus",
    "GlossaryRepository",
    "LLMCallLogRepository",
    "PersistenceError",
    "ProcessingRunRepository",
    "TaskRepository",
    "UnitOfWork",
]
