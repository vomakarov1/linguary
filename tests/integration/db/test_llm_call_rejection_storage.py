"""Rejection of a successful call in the technical log."""

from __future__ import annotations

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.application.ports.persistence import PersistenceError
from linguary.domain import ErrorType, TaskError
from persistence_factories import make_book, make_llm_call


def _rejection() -> TaskError:
    return TaskError(type=ErrorType.VALIDATION, message="отвергнут блок b1")


def test_mark_rejected_turns_success_into_validation_error(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    with unit_of_work.transaction() as transaction:
        transaction.books.add(make_book())
        transaction.llm_calls.add(make_llm_call())
    with unit_of_work.transaction() as transaction:
        transaction.llm_calls.mark_rejected("call-1", error=_rejection(), response_text=None)
    with unit_of_work.transaction() as transaction:
        (entry,) = transaction.llm_calls.list_for_book(make_book().id)
    assert entry.success is False
    assert entry.error == _rejection()


def test_mark_rejected_keeps_tokens(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    original = make_llm_call()
    with unit_of_work.transaction() as transaction:
        transaction.books.add(make_book())
        transaction.llm_calls.add(original)
        transaction.llm_calls.mark_rejected("call-1", error=_rejection(), response_text=None)
    with unit_of_work.transaction() as transaction:
        (entry,) = transaction.llm_calls.list_for_book(original.book_id)
    assert entry.usage == original.usage
    assert entry.cost == original.cost


def test_mark_rejected_stores_text_only_when_given(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    from persistence_factories import CREATED_AT
    unit_of_work.diagnostics_clock = lambda: CREATED_AT
    with unit_of_work.transaction() as transaction:
        transaction.books.add(make_book())
        transaction.llm_calls.add(make_llm_call(), response_text="сырой ответ")
        transaction.llm_calls.add(make_llm_call(call_id="call-2"))
        transaction.llm_calls.mark_rejected("call-1", error=_rejection(), response_text="ответ")
        transaction.llm_calls.mark_rejected("call-2", error=_rejection(), response_text=None)
    with unit_of_work.transaction() as transaction:
        assert transaction.llm_calls.response_text_for("call-1") == "ответ"
        assert transaction.llm_calls.response_text_for("call-2") is None


def test_mark_rejected_unknown_call_raises(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    import pytest

    with unit_of_work.transaction() as transaction:
        with pytest.raises(PersistenceError):
            transaction.llm_calls.mark_rejected(
                "missing", error=_rejection(), response_text=None
            )


def test_migration_0003_upgrade_downgrade(
    alembic_config: Config, engine: Engine
) -> None:
    command.downgrade(alembic_config, "0002")
    assert "response_text" not in {
        column["name"] for column in inspect(engine).get_columns("llm_call_logs")
    }
    command.upgrade(alembic_config, "0003")
    assert "response_text" in {
        column["name"] for column in inspect(engine).get_columns("llm_call_logs")
    }
