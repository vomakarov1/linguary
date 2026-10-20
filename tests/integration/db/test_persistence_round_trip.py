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


def make_content_plan():
    from linguary.domain.content_plan import (
        ContentPlan, Difficulty, HelpEvidence, RequiredPoint, SourceAnchor, artifact_key,
    )
    difficulty = Difficulty(
        id='difficulty-1',
        anchors=(SourceAnchor(BLOCK_ID, 0, 6, 'It was'), SourceAnchor(SECOND_BLOCK_ID, 0, 7, 'Winston')),
        owner_block_id=BLOCK_ID, kind='semantic_relation', priority='critical',
        help_kind='explanation', required_points=(
            RequiredPoint('relation', 'Объяснить связь двух фрагментов.'),
            RequiredPoint('tone', 'Пояснить оттенок «сухой речи».'),
        ),
    )
    entry = make_analysis().before_you_read[0]
    evidence = HelpEvidence(
        difficulty.id, 'relation', 'before_you_read.explanation', BLOCK_ID,
        artifact_key('before_you_read.explanation', BLOCK_ID, [entry.expression, entry.explanation]),
        entry.explanation,
    )
    return ContentPlan(difficulties=(difficulty,), before_you_read_evidence=(evidence,))


def test_v2_analysis_and_evidence_round_trip(unit_of_work, engine, stored_book):
    from dataclasses import replace
    from sqlalchemy import text
    from linguary.domain.content_plan import ContentEvidence
    plan = make_content_plan()
    analysis = replace(make_analysis(), content_plan=plan)
    result = replace(make_translated_result(), content_evidence=ContentEvidence(
        plan.sha256, plan.before_you_read_evidence,
    ))
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.analyses.save(BOOK_ID, analysis)
        uow.runs.add(make_run())
        uow.results.replace_results(RUN_ID, [result])
    with unit_of_work.transaction() as uow:
        restored = uow.analyses.get_for_section(BOOK_ID, SECTION_ID)
        assert restored == analysis
        assert restored.content_plan.sha256 == plan.sha256
        assert uow.results.get(RUN_ID, BLOCK_ID) == result
    with engine.connect() as connection:
        import json
        raw = connection.execute(text('SELECT content_plan_json FROM chapter_analyses')).scalar_one()
        assert isinstance(json.loads(raw), dict)
        assert '«сухой речи»' in json.loads(raw)['difficulties'][0]['required_points'][1]['description']


@pytest.mark.parametrize('candidate', [False, True])
def test_empty_v2_is_not_legacy(unit_of_work, engine, stored_book, candidate):
    from dataclasses import replace
    from sqlalchemy import text
    from linguary.domain.content_plan import ContentPlan, ContentEvidence
    plan = ContentPlan() if candidate else None
    evidence = ContentEvidence('a' * 64) if candidate else ()
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.analyses.save(BOOK_ID, replace(make_analysis(), content_plan=plan))
        uow.runs.add(make_run())
        uow.results.replace_results(RUN_ID, [replace(make_translated_result(), content_evidence=evidence)])
    with unit_of_work.transaction() as uow:
        assert uow.analyses.get_for_section(BOOK_ID, SECTION_ID).content_plan == plan
        assert uow.results.get(RUN_ID, BLOCK_ID).content_evidence == evidence
    with engine.connect() as connection:
        assert bool(connection.execute(text('SELECT content_plan_json IS NULL FROM chapter_analyses')).scalar_one()) is (not candidate)
        assert bool(connection.execute(text('SELECT content_evidence_json IS NULL FROM block_results')).scalar_one()) is (not candidate)


PLAN_CORRUPTIONS = [
    ('version', 999), ('version', 2.0), ('version', True), ('profile', 'unknown'),
    ('extra', 'private-book-text'), ('difficulties', None),
    ('difficulties.0.anchors.0.start', True), ('difficulties.0.anchors.0.start', 0.0),
    ('difficulties.0.anchors.0.quote', 42), ('difficulties.0.kind', 'unknown'),
    ('difficulties.0.required_points.0.description', []),
    ('difficulties.0.required_points.0.extra', True),
    ('before_you_read_evidence.0.body_excerpt', ''),
    ('before_you_read_evidence.0.point_id', 'unknown'),
]


@pytest.mark.parametrize('path,value', PLAN_CORRUPTIONS)
def test_corrupt_stored_content_plan_fails_closed(unit_of_work, engine, stored_book, path, value):
    from dataclasses import asdict
    import json
    from sqlalchemy import text
    payload = json.loads(json.dumps(asdict(make_content_plan())))
    keys = path.split('.')
    target = payload
    for key in keys[:-1]:
        target = target[int(key)] if isinstance(target, list) else target[key]
    target[keys[-1]] = value
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.analyses.save(BOOK_ID, make_analysis())
    with engine.begin() as connection:
        connection.execute(text('UPDATE chapter_analyses SET content_plan_json=:payload'), {'payload': json.dumps(payload)})
    with pytest.raises(PersistenceError) as caught, unit_of_work.transaction() as uow:
        uow.analyses.get_for_section(BOOK_ID, SECTION_ID)
    assert 'private-book-text' not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize('corruption', ['missing', 'duplicate_difficulty', 'duplicate_point'])
def test_missing_or_duplicate_plan_fields_fail_closed(unit_of_work, engine, stored_book, corruption):
    from dataclasses import asdict
    import json
    from sqlalchemy import text
    payload = json.loads(json.dumps(asdict(make_content_plan())))
    if corruption == 'missing':
        del payload['difficulties'][0]['owner_block_id']
    elif corruption == 'duplicate_difficulty':
        payload['difficulties'].append(payload['difficulties'][0])
    else:
        points = payload['difficulties'][0]['required_points']
        points.append(points[0])
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.analyses.save(BOOK_ID, make_analysis())
    with engine.begin() as connection:
        connection.execute(text('UPDATE chapter_analyses SET content_plan_json=:payload'), {'payload': json.dumps(payload)})
    with pytest.raises(PersistenceError), unit_of_work.transaction() as uow:
        uow.analyses.get_for_section(BOOK_ID, SECTION_ID)


@pytest.mark.parametrize('field,value', [
    ('version', 999), ('version', 2.0), ('version', True),
    ('plan_sha256', 'invalid'), ('semantic_correctness', 'verified'),
    ('structural_coverage', 'incomplete'), ('extra', 'private-book-text'),
    ('evidence', None),
])
def test_corrupt_stored_evidence_fails_closed(unit_of_work, engine, stored_book, field, value):
    from dataclasses import asdict
    import json
    from sqlalchemy import text
    from linguary.domain.content_plan import ContentEvidence
    payload = asdict(ContentEvidence('a' * 64))
    payload[field] = value
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.runs.add(make_run())
        uow.results.replace_results(RUN_ID, [make_translated_result()])
    with engine.begin() as connection:
        connection.execute(text('UPDATE block_results SET content_evidence_json=:payload'), {'payload': json.dumps(payload)})
    with pytest.raises(PersistenceError) as caught, unit_of_work.transaction() as uow:
        uow.results.get(RUN_ID, BLOCK_ID)
    assert 'private-book-text' not in str(caught.value)


def test_analysis_replacement_rollback_keeps_v2(unit_of_work, stored_book):
    from dataclasses import replace
    original = replace(make_analysis(), content_plan=make_content_plan())
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.analyses.save(BOOK_ID, original)
    with pytest.raises(RuntimeError), unit_of_work.transaction() as uow:
        uow.analyses.save(BOOK_ID, replace(original, content_plan=None))
        raise RuntimeError('отмена замены')
    with unit_of_work.transaction() as uow:
        assert uow.analyses.get_for_section(BOOK_ID, SECTION_ID) == original


@pytest.mark.parametrize('corruption', ['missing', 'extra', 'bad_type'])
def test_nested_evidence_is_strict(unit_of_work, engine, stored_book, corruption):
    from dataclasses import asdict
    import json
    from sqlalchemy import text
    from linguary.domain.content_plan import ContentEvidence
    plan = make_content_plan()
    payload = json.loads(json.dumps(asdict(ContentEvidence(plan.sha256, plan.before_you_read_evidence))))
    if corruption == 'missing':
        del payload['evidence'][0]['point_id']
    elif corruption == 'extra':
        payload['evidence'][0]['unknown'] = 'private-book-text'
    else:
        payload['evidence'][0]['body_excerpt'] = 42
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.runs.add(make_run())
        uow.results.replace_results(RUN_ID, [make_translated_result()])
    with engine.begin() as connection:
        connection.execute(text('UPDATE block_results SET content_evidence_json=:payload'), {'payload': json.dumps(payload)})
    with pytest.raises(PersistenceError) as caught, unit_of_work.transaction() as uow:
        uow.results.list_for_run(RUN_ID)
    assert 'private-book-text' not in str(caught.value)


def test_invalid_domain_scalar_types_cannot_be_saved(unit_of_work, stored_book):
    from dataclasses import replace
    invalid = replace(make_content_plan(), version=2.0)
    with pytest.raises(PersistenceError), unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.analyses.save(BOOK_ID, replace(make_analysis(), content_plan=invalid))
