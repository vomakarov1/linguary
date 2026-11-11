"""Storage round-trip: domain entities are recovered from the database without loss.

Domain entities and tables are connected by dedicated mapping functions, so a lost field
will not manifest as a type error or a query error — only as a discrepancy upon reading.
Tests compare the whole entity rather than individual fields.
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
    """Book without a canonical document."""
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())


def test_book_round_trip(unit_of_work: SqlAlchemyUnitOfWork, stored_book: None) -> None:
    """Book record is read back exactly as it was stored."""
    with unit_of_work.transaction() as uow:
        stored = uow.books.get(BOOK_ID)

    assert stored == make_book()


def test_book_status_change_is_stored(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Book status change overwrites the existing row."""
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
    """Canonical document model is reconstructed along with its blocks and resources."""
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
    """Repeated import replaces previous sections and blocks instead of appending them."""
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
    """Import warnings are stored along with their bindings."""
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
    """Glossary concept is read along with all language variants."""
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
    """Repeated saving of a concept does not duplicate its variants."""
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
    """Deleting a concept removes it along with its variants."""
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
    """Chapter analysis is reconstructed with plan, candidates, and materials."""
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
    """Repeated analysis of the same chapter replaces the previous one without violating uniqueness."""
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
    """Processing run is read with glossary and LLM settings snapshots."""
    run = make_run()
    with unit_of_work.transaction() as uow:
        uow.runs.add(run)

    with unit_of_work.transaction() as uow:
        stored = uow.runs.get(RUN_ID)

    assert stored == run


def test_block_results_round_trip(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Block results are read along with hints, notes, and warnings."""
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
    """Task is read with its key, attempt counter, and error."""
    task = make_task(task_type=TaskType.PROCESS_TRANSLATION_UNIT, scope_id="unit-1")
    with unit_of_work.transaction() as uow:
        uow.tasks.add(task)

    with unit_of_work.transaction() as uow:
        stored = uow.tasks.get(task.id)

    assert stored == task


def test_llm_call_log_round_trip(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Log entry is read with tokens and cost without precision loss."""
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
    """Accessing a repository outside a transaction is a contract violation rather than a silent write."""
    with pytest.raises(PersistenceError):
        unit_of_work.books.get(BOOK_ID)


def test_nested_transaction_is_rejected(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    """Nested transaction is prohibited: the boundary is defined by a single use case."""
    with unit_of_work.transaction(), pytest.raises(PersistenceError):
        with unit_of_work.transaction():
            pass


def test_failed_transaction_is_rolled_back(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Exception within a transaction rolls back all changes completely (PRD § 17)."""
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
    """Saving the state of an unknown task does not succeed silently."""
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


def make_pass_result(**overrides):
    from linguary.domain import AnalysisPassKind, AnalysisPassResult
    return AnalysisPassResult(**{
        'id': 'pass-1', 'book_id': BOOK_ID, 'section_id': SECTION_ID,
        'pass_kind': AnalysisPassKind.NARRATIVE, 'scope_key': '',
        'input_sha256': '1' * 64, 'output_sha256': '2' * 64,
        'conditions_id': None, 'attempt_id': 'attempt-1',
        'payload': {'schema_version': 1, 'chapter_summary': 'Summary', 'scenes': [{'participants': ['Holmes']}]},
        **overrides,
    })


def test_analysis_pass_result_and_chapter_scenes_round_trip(unit_of_work, stored_book):
    from dataclasses import replace
    from linguary.domain import AnalysisPassKind, Scene, SceneVisualFact
    from linguary.domain.content_plan import SourceAnchor
    from linguary.domain.content_plan import ContentPlan
    scene = Scene('scene-1', 0, BLOCK_ID, SECOND_BLOCK_ID, 'Winston enters.',
                  'London', ('Winston',), (SceneVisualFact(SourceAnchor(SECOND_BLOCK_ID, 0, 7, 'Winston'),
                                                          'Winston', 'At the door'),), ('Grey light',))
    plan = ContentPlan(version=3, profile='content-plan-v3', scenes=(scene,))
    analysis = replace(make_analysis(), content_plan=plan, scenes=(scene,))
    result = make_pass_result()
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.analyses.save(BOOK_ID, analysis)
        uow.analyses.save_pass(result)
    with unit_of_work.transaction() as uow:
        assert uow.analyses.get_pass(BOOK_ID, SECTION_ID, AnalysisPassKind.NARRATIVE) == result
        restored = uow.analyses.get_for_section(BOOK_ID, SECTION_ID)
        assert restored == analysis
        assert restored.content_plan.sha256 == plan.sha256


def test_analysis_pass_upsert_uses_full_key_and_preserves_other_results(unit_of_work, engine, stored_book):
    from dataclasses import replace
    from sqlalchemy import text
    from linguary.domain import AnalysisPassKind
    results = (make_pass_result(), make_pass_result(id='scene-pass', scope_key='scene-1'),
               make_pass_result(id='ped-pass', pass_kind=AnalysisPassKind.PEDAGOGICAL))
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        for result in results:
            uow.analyses.save_pass(result)
    with engine.connect() as connection:
        original_created = connection.execute(text("SELECT created_at FROM analysis_pass_results WHERE id='pass-1'")).scalar_one()
    replacement = replace(results[0], id='replacement', output_sha256='3' * 64, payload={'schema_version': 2})
    with unit_of_work.transaction() as uow:
        uow.analyses.save_pass(replacement)
        uow.analyses.save_pass(replacement)
        loaded = uow.analyses.list_passes_for_section(BOOK_ID, SECTION_ID)
        assert len(loaded) == 3
        assert {r.id for r in loaded} == {'replacement', 'scene-pass', 'ped-pass'}
        assert uow.analyses.get_pass(BOOK_ID, SECTION_ID, AnalysisPassKind.NARRATIVE) == replacement
        assert uow.analyses.get_pass(BOOK_ID, SECTION_ID, AnalysisPassKind.NARRATIVE, scope_key='scene-1') == results[1]
        assert uow.analyses.get_pass('other-book', SECTION_ID, AnalysisPassKind.NARRATIVE) is None
        assert uow.analyses.list_passes_for_section(BOOK_ID, 'other-section') == ()
    with engine.connect() as connection:
        created, updated = connection.execute(text("SELECT created_at, updated_at FROM analysis_pass_results WHERE id='replacement'")).one()
        assert created == original_created
        assert updated >= created


def test_analysis_pass_delete_can_filter_kind_and_preserves_chapter_analysis(unit_of_work, stored_book):
    from linguary.domain import AnalysisPassKind
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.analyses.save(BOOK_ID, make_analysis())
        uow.analyses.save_pass(make_pass_result())
        pedagogical = make_pass_result(id='p2', pass_kind=AnalysisPassKind.PEDAGOGICAL)
        uow.analyses.save_pass(pedagogical)
        uow.analyses.delete_passes_for_book('missing-book')
        uow.analyses.delete_passes_for_book(BOOK_ID, pass_kind=AnalysisPassKind.NARRATIVE)
        assert uow.analyses.list_passes_for_section(BOOK_ID, SECTION_ID) == (pedagogical,)
        uow.analyses.delete_passes_for_book(BOOK_ID)
        assert uow.analyses.list_passes_for_section(BOOK_ID, SECTION_ID) == ()
        assert uow.analyses.get_for_section(BOOK_ID, SECTION_ID) == make_analysis()


@pytest.mark.parametrize('table,identity', [('sections', SECTION_ID), ('books', BOOK_ID)])
def test_analysis_passes_cascade_with_book_or_section(unit_of_work, engine, stored_book, table, identity):
    from sqlalchemy import text
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.analyses.save_pass(make_pass_result())
    with engine.begin() as connection:
        connection.execute(text(f'DELETE FROM {table} WHERE id=:id'), {'id': identity})
        assert connection.execute(text('SELECT count(*) FROM analysis_pass_results')).scalar_one() == 0


def test_analysis_pass_conditions_deletion_keeps_result(unit_of_work, engine, stored_book):
    from sqlalchemy import text
    from linguary.domain import AnalysisPassKind
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO book_conditions (id, book_id, revision, schema_version, payload_json, reproducibility, created_at) VALUES ('conditions-1', :book, 0, 1, '{}', 'full', CURRENT_TIMESTAMP)"), {'book': BOOK_ID})
    with unit_of_work.transaction() as uow:
        uow.analyses.save_pass(make_pass_result(conditions_id='conditions-1'))
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM book_conditions WHERE id='conditions-1'"))
    with unit_of_work.transaction() as uow:
        assert uow.analyses.get_pass(BOOK_ID, SECTION_ID, AnalysisPassKind.NARRATIVE) == make_pass_result()


@pytest.mark.parametrize('version', [2, 3])
def test_legacy_content_plan_without_scenes_stays_readable(unit_of_work, engine, stored_book, version):
    from dataclasses import replace
    from linguary.domain.content_plan import ContentPlan
    plan = ContentPlan(version=version, profile=f'content-plan-v{version}')
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.analyses.save(BOOK_ID, replace(make_analysis(), content_plan=plan))
    with unit_of_work.transaction() as uow:
        analysis = uow.analyses.get_for_section(BOOK_ID, SECTION_ID)
        assert analysis.scenes == ()
        assert analysis.content_plan == plan


def test_analysis_passes_are_isolated_by_book_and_section(unit_of_work, stored_book):
    from dataclasses import replace
    from linguary.domain import AnalysisPassKind
    first = make_document()
    second_section = replace(first.sections[0], id='section-2', order=1,
                             blocks=tuple(replace(b, id=f'second-{b.id}', section_id='section-2')
                                          for b in first.sections[0].blocks))
    other = make_document(book_id='book-2')
    other_section = replace(other.sections[0], id='section-other',
                            blocks=tuple(replace(b, id=f'other-{b.id}', section_id='section-other')
                                         for b in other.sections[0].blocks))
    other = replace(other, sections=(other_section,),
                    resources=tuple(replace(r, id=f'other-{r.id}') for r in other.resources))
    results = (make_pass_result(), make_pass_result(id='second-pass', section_id='section-2'),
               make_pass_result(id='other-pass', book_id='book-2', section_id='section-other'))
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book(book_id='book-2'))
        uow.documents.save(replace(first, sections=(*first.sections, second_section)))
        uow.documents.save(other)
        for result in results:
            uow.analyses.save_pass(result)
        assert uow.analyses.list_passes_for_section(BOOK_ID, SECTION_ID) == (results[0],)
        assert uow.analyses.list_passes_for_section(BOOK_ID, 'section-2') == (results[1],)
        uow.analyses.delete_passes_for_book(BOOK_ID)
        assert uow.analyses.get_pass('book-2', 'section-other', AnalysisPassKind.NARRATIVE) == results[2]


def test_failed_analysis_pass_upsert_rolls_back(unit_of_work, stored_book):
    from dataclasses import replace
    from linguary.domain import AnalysisPassKind
    original = make_pass_result()
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.analyses.save_pass(original)
    with pytest.raises(RuntimeError, match='abort transaction'):
        with unit_of_work.transaction() as uow:
            uow.analyses.save_pass(replace(original, id='replacement', payload={'schema_version': 2}))
            raise RuntimeError('abort transaction')
    with unit_of_work.transaction() as uow:
        assert uow.analyses.get_pass(BOOK_ID, SECTION_ID, AnalysisPassKind.NARRATIVE) == original


@pytest.mark.parametrize('field,value', [('order', True), ('visual_facts', None), ('participants', 'Holmes'),
                                        ('unknown', 'field')])
def test_corrupt_stored_scenes_fail_closed(unit_of_work, engine, stored_book, field, value):
    import json
    from dataclasses import replace
    from sqlalchemy import text
    from linguary.domain.content_plan import ContentPlan, Scene
    plan = ContentPlan(version=3, profile='content-plan-v3', scenes=(Scene('s', 0, BLOCK_ID, SECOND_BLOCK_ID, 'Scene'),))
    with unit_of_work.transaction() as uow:
        uow.documents.save(make_document())
        uow.analyses.save(BOOK_ID, replace(make_analysis(), content_plan=plan))
    with engine.begin() as connection:
        raw = json.loads(connection.execute(text('SELECT content_plan_json FROM chapter_analyses')).scalar_one())
        raw['scenes'][0][field] = value
        connection.execute(text('UPDATE chapter_analyses SET content_plan_json=:payload'), {'payload': json.dumps(raw)})
    with unit_of_work.transaction() as uow:
        with pytest.raises(PersistenceError, match='повреждённый сохранённый содержательный план'):
            uow.analyses.get_for_section(BOOK_ID, SECTION_ID)
