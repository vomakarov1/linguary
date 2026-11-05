"""The correction migration removes checks without changing accepted book work."""

from alembic import command
from sqlalchemy import inspect, text

from persistence_factories import make_book, make_document, make_run, make_translated_result


def test_downgrade_preserves_results_and_bindings(alembic_config, engine, unit_of_work):
    with unit_of_work.transaction() as tx:
        tx.books.add(make_book())
        tx.documents.save(make_document())
        tx.runs.add(make_run())
        tx.results.replace_results('run-1', [make_translated_result()])
    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO book_conditions VALUES
                ('conditions', 'book-1', 1, 1, '{"schema_version": 1}', 'complete', '2026-11-04')
        """))
        connection.execute(text("""
            INSERT INTO work_bindings VALUES
                ('result', 'run-1:block-1', 'book-1', 'conditions', NULL, NULL, '{"schema_version": 1}')
        """))
        connection.execute(text("""
            INSERT INTO correction_checks (run_id, block_id, section_id, input_sha256, attempt_id) VALUES
                ('run-1', 'block-1', 'section-1', :checksum, 'attempt')
        """), {'checksum': 'a' * 64})
        assert connection.execute(text('SELECT basis, candidate_id FROM correction_checks')).one() == ('legacy', None)
        correction_tables = {'correction_checks', 'correction_candidates', 'correction_inputs'}
        current_tables = set(inspect(connection).get_table_names())
        assert correction_tables <= current_tables
        tables = current_tables - correction_tables - {'alembic_version'}
        before = {table: connection.execute(text(f'SELECT * FROM {table} ORDER BY 1')).all()
                  for table in tables}

    command.downgrade(alembic_config, '0011')

    with engine.connect() as connection:
        assert set(inspect(connection).get_table_names()) == tables | {'alembic_version'}
        assert connection.execute(text('SELECT version_num FROM alembic_version')).scalar_one() == '0011'
        assert {table: connection.execute(text(f'SELECT * FROM {table} ORDER BY 1')).all()
                for table in tables} == before
        assert connection.execute(text('PRAGMA foreign_key_check')).all() == []

    command.upgrade(alembic_config, '0012')
    with engine.connect() as connection:
        assert set(inspect(connection).get_table_names()) == tables | {'correction_checks', 'alembic_version'}
        assert {column['name'] for column in inspect(connection).get_columns('correction_checks')} == {
            'run_id', 'block_id', 'section_id', 'input_sha256', 'attempt_id',
        }
        assert connection.execute(text('SELECT count(*) FROM correction_checks')).scalar_one() == 0
        assert {table: connection.execute(text(f'SELECT * FROM {table} ORDER BY 1')).all()
                for table in tables} == before
