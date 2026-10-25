import pytest
from alembic import command
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from linguary.adapters.db.models import BookConditionsRow
from persistence_factories import make_book


def test_upgrade_0006_keeps_legacy_rows(alembic_config, engine, unit_of_work):
    command.downgrade(alembic_config, '0006')
    with unit_of_work.transaction():
        unit_of_work.books.add(make_book())
    command.upgrade(alembic_config, 'head')
    with unit_of_work.transaction():
        assert unit_of_work.books.get('book-1') is not None
    with engine.connect() as connection:
        assert connection.scalar(text('select count(*) from book_conditions')) == 0


def test_fk_and_unique_constraints(engine):
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(text("insert into book_conditions values ('x','missing',1,1,'{}','unknown','2026-10-25')"))


def test_downgrade_refuses_nonempty_work(alembic_config, engine, unit_of_work):
    with unit_of_work.transaction():
        unit_of_work.books.add(make_book())
    with engine.begin() as connection:
        connection.execute(text("insert into book_conditions values ('x','book-1',1,1,'{}','unknown','2026-10-25')"))
    with pytest.raises(RuntimeError):
        command.downgrade(alembic_config, '0006')


def test_book_delete_cascades_work(engine, unit_of_work):
    with unit_of_work.transaction():
        unit_of_work.books.add(make_book())
    with engine.begin() as connection:
        connection.execute(text("insert into book_conditions values ('x','book-1',1,1,'{}','unknown','2026-10-25')"))
        connection.execute(text("delete from books where id='book-1'"))
        assert connection.scalar(text('select count(*) from book_conditions')) == 0
