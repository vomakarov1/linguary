"""Обратимость хранения: доменная сущность возвращается из базы без потерь.

Домен и таблицы связаны отдельными функциями преобразования, поэтому потеря поля не
проявится ни как ошибка типов, ни как ошибка запроса — только как расхождение при чтении.
Тесты сравнивают сущность целиком, а не отдельные поля.
"""

from __future__ import annotations

import pytest
from persistence_factories import (
    BLOCK_ID,
    BOOK_ID,
    RUN_ID,
    SECOND_BLOCK_ID,
    SECTION_ID,
    make_analysis,
    make_book,
    make_concept,
    make_document,
    make_glossary_only_result,
    make_llm_call,
    make_run,
    make_task,
    make_translated_result,
)

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.application.ports.persistence import PersistenceError
from linguary.domain import (
    BookImportWarning,
    BookStatus,
    ImportWarningKind,
    TaskType,
)


@pytest.fixture
def stored_book(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    """Книга без канонического документа."""
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())


def test_book_round_trip(unit_of_work: SqlAlchemyUnitOfWork, stored_book: None) -> None:
    """Запись книги читается в точности такой, какой была сохранена."""
    with unit_of_work.transaction() as uow:
        stored = uow.books.get(BOOK_ID)

    assert stored == make_book()


def test_book_status_change_is_stored(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Смена состояния книги перезаписывает существующую строку."""
    with unit_of_work.transaction() as uow:
        uow.books.save(make_book(status=BookStatus.IMPORTED))

    with unit_of_work.transaction() as uow:
        stored = uow.books.get(BOOK_ID)
        all_books = uow.books.list_all()

    assert stored is not None
    assert stored.status is BookStatus.IMPORTED
    assert len(all_books) == 1


def test_document_round_trip(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Каноническая модель документа восстанавливается вместе с блоками и ресурсами."""
    document = make_document()
    with unit_of_work.transaction() as uow:
        uow.documents.save(
            document, stored_resource_paths={"OEBPS/images/cover.jpg": "/data/cover.jpg"}
        )

    with unit_of_work.transaction() as uow:
        stored = uow.documents.get(BOOK_ID)
        paths = dict(uow.documents.stored_resource_paths(BOOK_ID))

    assert stored == document
    assert paths == {"OEBPS/images/cover.jpg": "/data/cover.jpg"}


def test_document_save_replaces_previous_structure(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Повторный импорт заменяет прежние разделы и блоки, а не добавляет их."""
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.documents.save(make_document())

    with unit_of_work.transaction() as uow:
        stored = uow.documents.get(BOOK_ID)

    assert stored is not None
    assert len(stored.sections) == 1
    assert len(stored.section(SECTION_ID).blocks) == 2


def test_import_warnings_round_trip(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Предупреждения импорта сохраняются вместе со своей привязкой."""
    warnings = (
        BookImportWarning(
            kind=ImportWarningKind.TOC_MISMATCH, message="Оглавление не совпало с spine."
        ),
        BookImportWarning(
            kind=ImportWarningKind.DROPPED_BLOCK,
            message="Пустой декоративный блок отброшен.",
            section_id=SECTION_ID,
            block_id=BLOCK_ID,
        ),
    )
    with unit_of_work.transaction() as uow:
        uow.documents.add_import_warnings(BOOK_ID, warnings)

    with unit_of_work.transaction() as uow:
        stored = uow.documents.import_warnings(BOOK_ID)

    assert stored == warnings


def test_glossary_concept_round_trip(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Понятие глоссария читается вместе со всеми языковыми вариантами."""
    concept = make_concept()
    with unit_of_work.transaction() as uow:
        uow.glossary.save(concept)

    with unit_of_work.transaction() as uow:
        stored = uow.glossary.get(concept.id)
        listed = uow.glossary.list_for_book(BOOK_ID)

    assert stored == concept
    assert listed == (concept,)


def test_glossary_save_replaces_variants(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Повторное сохранение понятия не размножает его варианты."""
    with unit_of_work.transaction() as uow:
        uow.glossary.save(make_concept())
        uow.glossary.save(make_concept())

    with unit_of_work.transaction() as uow:
        stored = uow.glossary.get("concept-1")

    assert stored is not None
    assert len(stored.variants) == 2


def test_glossary_delete_removes_concept(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Удаление понятия убирает его вместе с вариантами."""
    with unit_of_work.transaction() as uow:
        uow.glossary.save(make_concept())

    with unit_of_work.transaction() as uow:
        uow.glossary.delete("concept-1")

    with unit_of_work.transaction() as uow:
        assert uow.glossary.get("concept-1") is None
        assert uow.glossary.list_for_book(BOOK_ID) == ()


def test_chapter_analysis_round_trip(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Анализ главы восстанавливается вместе с планом, кандидатами и материалами."""
    analysis = make_analysis()
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.analyses.save(BOOK_ID, analysis)

    with unit_of_work.transaction() as uow:
        stored = uow.analyses.get_for_section(BOOK_ID, SECTION_ID)
        listed = uow.analyses.list_for_book(BOOK_ID)

    assert stored == analysis
    assert listed == (analysis,)


def test_chapter_analysis_save_replaces_previous(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Повторный анализ той же главы заменяет прежний, не нарушая уникальности."""
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.analyses.save(BOOK_ID, make_analysis())
        uow.analyses.save(BOOK_ID, make_analysis(analysis_id="analysis-2"))

    with unit_of_work.transaction() as uow:
        listed = uow.analyses.list_for_book(BOOK_ID)

    assert len(listed) == 1
    assert listed[0].id == "analysis-2"


def test_processing_run_round_trip(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Запуск обработки читается со снимками глоссария и настроек LLM."""
    run = make_run()
    with unit_of_work.transaction() as uow:
        uow.runs.add(run)

    with unit_of_work.transaction() as uow:
        stored = uow.runs.get(RUN_ID)

    assert stored == run


def test_block_results_round_trip(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Результаты блоков читаются вместе с подсказками, заметками и замечаниями."""
    results = [make_translated_result(), make_glossary_only_result()]
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.runs.add(make_run())
        uow.results.replace_results(RUN_ID, results)

    with unit_of_work.transaction() as uow:
        stored = uow.results.list_for_run(RUN_ID)
        single = uow.results.get(RUN_ID, SECOND_BLOCK_ID)

    assert list(stored) == results
    assert single == results[1]


def test_task_round_trip(unit_of_work: SqlAlchemyUnitOfWork, stored_book: None) -> None:
    """Задача читается со своим ключом, счётчиком попыток и ошибкой."""
    task = make_task(task_type=TaskType.PROCESS_TRANSLATION_UNIT, scope_id="unit-1")
    with unit_of_work.transaction() as uow:
        uow.tasks.add(task)

    with unit_of_work.transaction() as uow:
        stored = uow.tasks.get(task.id)

    assert stored == task


def test_llm_call_log_round_trip(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Запись журнала читается вместе с токенами и стоимостью без потери точности."""
    entries = (make_llm_call(call_id="call-1"), make_llm_call(call_id="call-2", success=False))
    with unit_of_work.transaction() as uow:
        for entry in entries:
            uow.llm_calls.add(entry)

    with unit_of_work.transaction() as uow:
        stored = uow.llm_calls.list_for_book(BOOK_ID)

    assert stored == entries


def test_repository_is_unavailable_outside_transaction(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Обращение к репозиторию вне транзакции — ошибка контракта, а не тихая запись."""
    with pytest.raises(PersistenceError):
        unit_of_work.books.get(BOOK_ID)


def test_nested_transaction_is_rejected(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    """Вложенная транзакция запрещена: границу задаёт один сценарий."""
    with unit_of_work.transaction(), pytest.raises(PersistenceError):
        with unit_of_work.transaction():
            pass


def test_failed_transaction_is_rolled_back(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Исключение внутри транзакции откатывает все изменения целиком (PRD § 17)."""
    with pytest.raises(RuntimeError), unit_of_work.transaction() as uow:
        uow.books.add(make_book())
        uow.documents.save(make_document())
        raise RuntimeError("сценарий прерван после записи")

    with unit_of_work.transaction() as uow:
        assert uow.books.get(BOOK_ID) is None
        assert uow.documents.get(BOOK_ID) is None


def test_saving_unknown_task_is_rejected(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Сохранение состояния несуществующей задачи не проходит молча."""
    with pytest.raises(PersistenceError), unit_of_work.transaction() as uow:
        uow.tasks.save(make_task(task_id="нет такой задачи"))
