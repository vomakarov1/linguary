"""Реализации портов хранения поверх SQLAlchemy.

Каждый репозиторий работает с уже открытой сессией: транзакционную границу задаёт
`SqlAlchemyUnitOfWork`, а не отдельная операция. Репозиторий принимает и возвращает
доменные сущности; строки таблиц наружу не выходят.

Технические отметки времени (`created_at`) принадлежат хранилищу: доменные сущности их
не содержат, поэтому репозиторий проставляет их при добавлении и сохраняет при обновлении.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from linguary.adapters.db import mapping
from linguary.adapters.db.models import (
    BlockResultRow,
    BookRow,
    ChapterAnalysisRow,
    DocumentBlockRow,
    GlossaryCandidateRow,
    ImportWarningRow,
    LanguageNoteRow,
    LLMCallLogRow,
    ProcessingRunRow,
    ProcessingTaskRow,
    ResourceRow,
    ResultWarningRow,
    SectionRow,
    TermConceptRow,
    TermVariantRow,
    TreatmentPlanItemRow,
)
from linguary.application.ports.persistence import BookRecord, PersistenceError
from linguary.domain import (
    BlockProcessingResult,
    BookImportWarning,
    CanonicalDocument,
    ChapterAnalysis,
    LLMCallLog,
    ProcessingRun,
    ProcessingRunStatus,
    ProcessingTask,
    TaskIdentity,
    TermConcept,
    TaskError,
)


class SqlAlchemyBookRepository:
    """Книги и их пользовательское состояние."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, book: BookRecord) -> None:
        """Добавляет новую книгу."""
        self._session.add(mapping.to_book_row(book))
        self._session.flush()

    def save(self, book: BookRecord) -> None:
        """Перезаписывает существующую книгу целиком."""
        self._session.merge(mapping.to_book_row(book))
        self._session.flush()

    def get(self, book_id: str) -> BookRecord | None:
        """Книга по идентификатору либо `None`, если её нет."""
        row = self._session.get(BookRow, book_id)
        return None if row is None else mapping.to_book_record(row)

    def list_all(self) -> tuple[BookRecord, ...]:
        """Все книги в порядке добавления."""
        rows = self._session.scalars(
            select(BookRow).order_by(BookRow.created_at, BookRow.id)
        ).all()
        return tuple(mapping.to_book_record(row) for row in rows)


class SqlAlchemyDocumentRepository:
    """Каноническая модель документа и предупреждения импорта."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(
        self,
        document: CanonicalDocument,
        *,
        stored_resource_paths: Mapping[str, str] | None = None,
    ) -> None:
        """Сохраняет документ целиком, заменяя прежние разделы, блоки и ресурсы.

        Разделы записываются отдельным шагом: из-за самоссылки `document_blocks.parent_id`
        SQLAlchemy не может упорядочить вставку таблиц сам, и блоки ушли бы в базу раньше
        своих разделов.
        """
        paths = stored_resource_paths or {}
        self._session.execute(
            delete(SectionRow).where(SectionRow.book_id == document.id)
        )
        self._session.execute(
            delete(ResourceRow).where(ResourceRow.book_id == document.id)
        )
        self._session.flush()
        for section in document.ordered_sections:
            self._session.add(mapping.to_section_row(document.id, section))
        self._session.flush()
        for section in document.ordered_sections:
            for block in section.ordered_blocks:
                self._session.add(mapping.to_block_row(document.id, block))
        for resource in document.resources:
            self._session.add(
                mapping.to_resource_row(
                    document.id, resource, paths.get(resource.source_path)
                )
            )
        self._session.flush()

    def get(self, book_id: str) -> CanonicalDocument | None:
        """Каноническая модель книги либо `None`, если такой книги нет."""
        book = self._session.get(BookRow, book_id)
        if book is None:
            return None
        sections = self._session.scalars(
            select(SectionRow).where(SectionRow.book_id == book_id)
        ).all()
        blocks = self._session.scalars(
            select(DocumentBlockRow).where(DocumentBlockRow.book_id == book_id)
        ).all()
        resources = self._session.scalars(
            select(ResourceRow)
            .where(ResourceRow.book_id == book_id)
            .order_by(ResourceRow.source_path)
        ).all()
        return mapping.to_document(book, sections, blocks, resources)

    def stored_resource_paths(self, book_id: str) -> Mapping[str, str]:
        """Пути сохранённых ресурсов книги по их исходным путям."""
        rows = self._session.execute(
            select(ResourceRow.source_path, ResourceRow.stored_path).where(
                ResourceRow.book_id == book_id, ResourceRow.stored_path.is_not(None)
            )
        ).all()
        return {source_path: stored_path for source_path, stored_path in rows}

    def add_import_warnings(
        self, book_id: str, warnings: Sequence[BookImportWarning]
    ) -> None:
        """Добавляет предупреждения импорта книги."""
        self._session.add_all(
            mapping.to_import_warning_row(book_id, warning) for warning in warnings
        )
        self._session.flush()

    def import_warnings(self, book_id: str) -> tuple[BookImportWarning, ...]:
        """Предупреждения импорта книги."""
        rows = self._session.scalars(
            select(ImportWarningRow)
            .where(ImportWarningRow.book_id == book_id)
            .order_by(ImportWarningRow.id)
        ).all()
        return tuple(mapping.to_import_warning(row) for row in rows)


class SqlAlchemyGlossaryRepository:
    """Понятия глоссария книги вместе с их языковыми вариантами."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, concept: TermConcept) -> None:
        """Сохраняет понятие вместе с вариантами, заменяя прежнюю запись."""
        self._session.execute(
            delete(TermVariantRow).where(TermVariantRow.concept_id == concept.id)
        )
        self._session.merge(mapping.to_concept_row(concept))
        self._session.add_all(
            mapping.to_variant_row(variant) for variant in concept.variants
        )
        self._session.flush()

    def get(self, concept_id: str) -> TermConcept | None:
        """Понятие по идентификатору либо `None`."""
        row = self._session.get(TermConceptRow, concept_id)
        return None if row is None else mapping.to_concept(row, self._variants(concept_id))

    def list_for_book(self, book_id: str) -> tuple[TermConcept, ...]:
        """Все понятия глоссария книги."""
        rows = self._session.scalars(
            select(TermConceptRow)
            .where(TermConceptRow.book_id == book_id)
            .order_by(TermConceptRow.canonical_source_text, TermConceptRow.id)
        ).all()
        return tuple(mapping.to_concept(row, self._variants(row.id)) for row in rows)

    def delete(self, concept_id: str) -> None:
        """Удаляет понятие вместе с его вариантами."""
        self._session.execute(
            delete(TermConceptRow).where(TermConceptRow.id == concept_id)
        )
        self._session.flush()

    def _variants(self, concept_id: str) -> Sequence[TermVariantRow]:
        """Варианты понятия в устойчивом порядке."""
        return self._session.scalars(
            select(TermVariantRow)
            .where(TermVariantRow.concept_id == concept_id)
            .order_by(TermVariantRow.id)
        ).all()


class SqlAlchemyChapterAnalysisRepository:
    """Анализ главы: план обработки, кандидаты глоссария и материалы главы."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, book_id: str, analysis: ChapterAnalysis) -> None:
        """Сохраняет анализ главы, заменяя прежний анализ той же главы."""
        self._session.execute(
            delete(ChapterAnalysisRow).where(
                ChapterAnalysisRow.book_id == book_id,
                ChapterAnalysisRow.section_id == analysis.section_id,
            )
        )
        self._session.flush()
        self._session.add(mapping.to_analysis_row(book_id, analysis))
        self._session.add_all(mapping.to_plan_item_rows(analysis))
        self._session.add_all(
            mapping.to_candidate_row(candidate)
            for candidate in analysis.glossary_candidates
        )
        self._session.flush()

    def get_for_section(self, book_id: str, section_id: str) -> ChapterAnalysis | None:
        """Анализ конкретной главы либо `None`, если глава ещё не анализировалась."""
        row = self._session.scalars(
            select(ChapterAnalysisRow).where(
                ChapterAnalysisRow.book_id == book_id,
                ChapterAnalysisRow.section_id == section_id,
            )
        ).one_or_none()
        return None if row is None else self._to_analysis(row)

    def list_for_book(self, book_id: str) -> tuple[ChapterAnalysis, ...]:
        """Анализы всех проанализированных глав книги в порядке чтения."""
        rows = self._session.scalars(
            select(ChapterAnalysisRow)
            .join(SectionRow, SectionRow.id == ChapterAnalysisRow.section_id)
            .where(ChapterAnalysisRow.book_id == book_id)
            .order_by(SectionRow.order)
        ).all()
        return tuple(self._to_analysis(row) for row in rows)

    def _to_analysis(self, row: ChapterAnalysisRow) -> ChapterAnalysis:
        """Собирает доменный анализ вместе с планом и кандидатами."""
        plan_items = self._session.scalars(
            select(TreatmentPlanItemRow)
            .where(TreatmentPlanItemRow.analysis_id == row.id)
            .order_by(TreatmentPlanItemRow.position)
        ).all()
        candidates = self._session.scalars(
            select(GlossaryCandidateRow)
            .where(GlossaryCandidateRow.analysis_id == row.id)
            .order_by(GlossaryCandidateRow.id)
        ).all()
        return mapping.to_analysis(row, plan_items, candidates)


class SqlAlchemyProcessingRunRepository:
    """Запуски обработки книги."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, run: ProcessingRun) -> None:
        """Добавляет новый запуск обработки."""
        self._session.add(mapping.to_run_row(run, datetime.now(UTC)))
        self._session.flush()

    def save(self, run: ProcessingRun) -> None:
        """Перезаписывает существующий запуск, сохраняя отметку его создания."""
        row = mapping.to_run_row(run, datetime.now(UTC))
        result = self._session.execute(
            update(ProcessingRunRow)
            .where(ProcessingRunRow.id == run.id)
            .values(
                status=row.status,
                glossary_snapshot_json=row.glossary_snapshot_json,
                llm_settings_json=row.llm_settings_json,
            )
        )
        if result.rowcount != 1:
            raise PersistenceError(f"запуск обработки {run.id} отсутствует в хранилище")
        self._session.flush()

    def get(self, run_id: str) -> ProcessingRun | None:
        """Запуск по идентификатору либо `None`."""
        row = self._session.get(ProcessingRunRow, run_id)
        return None if row is None else mapping.to_run(row)

    def active_for_book(self, book_id: str) -> ProcessingRun | None:
        """Единственный активный запуск книги либо `None`.

        Единственность гарантирует частичный уникальный индекс схемы.
        """
        row = self._session.scalars(
            select(ProcessingRunRow).where(
                ProcessingRunRow.book_id == book_id,
                ProcessingRunRow.status == ProcessingRunStatus.ACTIVE.value,
            )
        ).one_or_none()
        return None if row is None else mapping.to_run(row)

    def delete(self, run_id: str) -> None:
        """Удаляет запуск вместе с результатами блоков, заметками и замечаниями."""
        self._session.execute(
            delete(ProcessingRunRow).where(ProcessingRunRow.id == run_id)
        )
        self._session.flush()


class SqlAlchemyBlockResultRepository:
    """Результаты обработки блоков вместе с заметками и замечаниями."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def replace_results(
        self, run_id: str, results: Sequence[BlockProcessingResult]
    ) -> None:
        """Заменяет результаты перечисленных блоков одной операцией (PRD § 17)."""
        block_ids = [result.block_id for result in results]
        if not block_ids:
            return
        for table in (BlockResultRow, LanguageNoteRow, ResultWarningRow):
            self._session.execute(
                delete(table).where(
                    table.run_id == run_id, table.block_id.in_(block_ids)
                )
            )
        self._session.flush()
        for result in results:
            self._session.add(mapping.to_result_row(run_id, result))
            self._session.add_all(mapping.to_note_rows(run_id, result))
            self._session.add_all(mapping.to_result_warning_rows(run_id, result))
        self._session.flush()

    def get(self, run_id: str, block_id: str) -> BlockProcessingResult | None:
        """Результат обработки одного блока в рамках запуска."""
        row = self._session.scalars(
            select(BlockResultRow).where(
                BlockResultRow.run_id == run_id, BlockResultRow.block_id == block_id
            )
        ).one_or_none()
        if row is None:
            return None
        return mapping.to_block_result(
            row, self._notes(run_id, block_id), self._warnings(run_id, block_id)
        )

    def list_for_run(self, run_id: str) -> tuple[BlockProcessingResult, ...]:
        """Все результаты запуска в порядке блоков исходного документа.

        Заметки и замечания читаются двумя запросами на весь запуск и раскладываются
        по блокам в памяти: запрос на каждый результат превращал бы чтение книги
        в тысячи обращений к базе.
        """
        rows = self._session.scalars(
            select(BlockResultRow)
            .join(DocumentBlockRow, DocumentBlockRow.id == BlockResultRow.block_id)
            .join(SectionRow, SectionRow.id == DocumentBlockRow.section_id)
            .where(BlockResultRow.run_id == run_id)
            .order_by(SectionRow.order, DocumentBlockRow.order)
        ).all()
        notes = _by_block(self._run_notes(run_id))
        warnings = _by_block(self._run_warnings(run_id))
        return tuple(
            mapping.to_block_result(
                row,
                notes.get(row.block_id, ()),
                warnings.get(row.block_id, ()),
            )
            for row in rows
        )

    def count_for_run(self, run_id: str) -> int:
        """Сколько блоков запуска уже обработано.

        Карточка книги обновляется опросом и берёт из результатов только их число,
        поэтому счёт выполняется базой, а не сборкой доменных результатов.
        """
        return (
            self._session.scalar(
                select(func.count())
                .select_from(BlockResultRow)
                .where(BlockResultRow.run_id == run_id)
            )
            or 0
        )

    def _notes(self, run_id: str, block_id: str) -> Sequence[LanguageNoteRow]:
        """Заметки блока в устойчивом порядке."""
        return self._session.scalars(
            select(LanguageNoteRow)
            .where(LanguageNoteRow.run_id == run_id, LanguageNoteRow.block_id == block_id)
            .order_by(LanguageNoteRow.id)
        ).all()

    def _warnings(self, run_id: str, block_id: str) -> Sequence[ResultWarningRow]:
        """Замечания к результату блока в устойчивом порядке."""
        return self._session.scalars(
            select(ResultWarningRow)
            .where(
                ResultWarningRow.run_id == run_id, ResultWarningRow.block_id == block_id
            )
            .order_by(ResultWarningRow.id)
        ).all()

    def _run_notes(self, run_id: str) -> Sequence[LanguageNoteRow]:
        """Заметки всех блоков запуска в том же порядке, что и заметки одного блока."""
        return self._session.scalars(
            select(LanguageNoteRow)
            .where(LanguageNoteRow.run_id == run_id)
            .order_by(LanguageNoteRow.id)
        ).all()

    def _run_warnings(self, run_id: str) -> Sequence[ResultWarningRow]:
        """Замечания ко всем блокам запуска в том же порядке, что и замечания блока."""
        return self._session.scalars(
            select(ResultWarningRow)
            .where(ResultWarningRow.run_id == run_id)
            .order_by(ResultWarningRow.id)
        ).all()


class SqlAlchemyTaskRepository:
    """Локальные возобновляемые задачи обработки."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, task: ProcessingTask) -> None:
        """Добавляет задачу; повтор идемпотентного ключа отвергается хранилищем."""
        self._session.add(mapping.to_task_row(task, datetime.now(UTC)))
        self._session.flush()

    def save(self, task: ProcessingTask) -> None:
        """Сохраняет новое состояние задачи, не трогая отметку её создания."""
        row = mapping.to_task_row(task, datetime.now(UTC))
        result = self._session.execute(
            update(ProcessingTaskRow)
            .where(ProcessingTaskRow.id == task.id)
            .values(
                status=row.status,
                attempts=row.attempts,
                error_type=row.error_type,
                error_message=row.error_message,
                error_detail=row.error_detail,
            )
        )
        if result.rowcount != 1:
            raise PersistenceError(f"задача {task.id} отсутствует в хранилище")
        self._session.flush()

    def get(self, task_id: str) -> ProcessingTask | None:
        """Задача по идентификатору либо `None`."""
        row = self._session.get(ProcessingTaskRow, task_id)
        return None if row is None else mapping.to_task(row)

    def find_by_identity(self, identity: TaskIdentity) -> ProcessingTask | None:
        """Задача по идемпотентному ключу `book_id + task_type + scope_id`."""
        row = self._session.scalars(
            select(ProcessingTaskRow).where(
                ProcessingTaskRow.book_id == identity.book_id,
                ProcessingTaskRow.task_type == identity.task_type.value,
                ProcessingTaskRow.scope_id == identity.scope_id,
            )
        ).one_or_none()
        return None if row is None else mapping.to_task(row)

    def list_for_book(self, book_id: str) -> tuple[ProcessingTask, ...]:
        """Задачи книги в порядке их создания."""
        rows = self._session.scalars(
            select(ProcessingTaskRow)
            .where(ProcessingTaskRow.book_id == book_id)
            .order_by(ProcessingTaskRow.created_at, ProcessingTaskRow.id)
        ).all()
        return tuple(mapping.to_task(row) for row in rows)

    def delete(self, task_id: str) -> None:
        """Удаляет задачу вместе с её идемпотентным ключом."""
        self._session.execute(
            delete(ProcessingTaskRow).where(ProcessingTaskRow.id == task_id)
        )
        self._session.flush()


class SqlAlchemyLLMCallLogRepository:
    """Технический журнал вызовов LLM."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, entry: LLMCallLog, *, response_text: str | None = None) -> None:
        """Добавляет запись журнала."""
        row = mapping.to_llm_call_row(entry)
        row.response_text = response_text
        self._session.add(row)
        self._session.flush()

    def mark_rejected(
        self, call_id: str, *, error: TaskError, response_text: str | None
    ) -> None:
        """Помечает отбракованный ответ, сохраняя расход токенов и стоимость."""
        row = self._session.get(LLMCallLogRow, call_id)
        if row is None:
            raise PersistenceError(f"запись вызова LLM {call_id} не найдена")
        row.success = False
        row.error_type = error.type.value
        row.error_message = error.message
        row.error_detail = error.detail
        row.response_text = response_text
        self._session.flush()

    def response_text_for(self, call_id: str) -> str | None:
        """Сохранённый текст ответа, если он был записан."""
        row = self._session.get(LLMCallLogRow, call_id)
        return None if row is None else row.response_text

    def list_for_book(self, book_id: str) -> tuple[LLMCallLog, ...]:
        """Записи журнала по книге в порядке выполнения вызовов."""
        rows = self._session.scalars(
            select(LLMCallLogRow)
            .where(LLMCallLogRow.book_id == book_id)
            .order_by(LLMCallLogRow.started_at, LLMCallLogRow.id)
        ).all()
        return tuple(mapping.to_llm_call(row) for row in rows)


def _by_block[RowT: (LanguageNoteRow, ResultWarningRow)](
    rows: Iterable[RowT],
) -> Mapping[str, Sequence[RowT]]:
    """Строки, разложенные по блокам с сохранением исходного порядка."""
    grouped: dict[str, list[RowT]] = {}
    for row in rows:
        grouped.setdefault(row.block_id, []).append(row)
    return grouped


__all__ = [
    "SqlAlchemyBlockResultRepository",
    "SqlAlchemyBookRepository",
    "SqlAlchemyChapterAnalysisRepository",
    "SqlAlchemyDocumentRepository",
    "SqlAlchemyGlossaryRepository",
    "SqlAlchemyLLMCallLogRepository",
    "SqlAlchemyProcessingRunRepository",
    "SqlAlchemyTaskRepository",
]
