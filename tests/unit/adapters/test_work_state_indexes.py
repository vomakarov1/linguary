from __future__ import annotations

from datetime import UTC, datetime
import importlib.util
from pathlib import Path
import sqlite3

import pytest
from sqlalchemy.exc import OperationalError

from linguary.adapters.db import unit_of_work as uow_module
from linguary.adapters.db.models import Base
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm import logging as llm_logging_module
from linguary.adapters.llm.logging import LoggingLLMClient
from linguary.application import diagnostics as diagnostics_module
from linguary.application.diagnostics import Diagnostics, bounded_body
from linguary.application.llm_dto import TranslationResultResponse
from linguary.application.ports.llm import (
    LLMClient,
    LLMRequest,
    LLMResponse,
    LLMResponseKind,
)
from linguary.application.ports.persistence import PersistenceError
from linguary.domain import LLMCallStage, TokenUsage


EXPECTED_INDEXES: dict[str, dict[str, tuple[str, ...]]] = {
    "recovery_roots": {
        "ix_recovery_roots_book_id": ("book_id",),
    },
    "recovery_scopes": {
        "ix_recovery_scopes_root_id": ("root_id",),
    },
    "llm_attempts": {
        "ix_llm_attempts_scope_id": ("scope_id",),
        "ix_llm_attempts_task_id": ("task_id",),
        "ix_llm_attempts_book_id": ("book_id",),
    },
    "work_bindings": {
        "ix_work_bindings_book_id_entity_kind": ("book_id", "entity_kind"),
    },
    "artifact_revisions": {
        "ix_artifact_revisions_book_id_run_id": ("book_id", "run_id"),
    },
    "correction_candidates": {
        "ix_correction_candidates_book_id": ("book_id",),
    },
    "processing_runs": {
        "ix_processing_runs_book_id": ("book_id",),
    },
}


def test_work_state_tables_define_expected_indexes() -> None:
    for table_name, expected in EXPECTED_INDEXES.items():
        table = Base.metadata.tables[table_name]
        actual = {
            index.name: tuple(column.name for column in index.columns)
            for index in table.indexes
        }
        for index_name, columns in expected.items():
            assert actual.get(index_name) == columns, (
                f"Таблица {table_name} должна определять индекс {index_name}{columns}, "
                f"найдено: {actual}"
            )

    migration_path = (
        Path(__file__).resolve().parents[3]
        / "alembic"
        / "versions"
        / "0014_work_state_indexes.py"
    )
    assert migration_path.exists(), "Миграция 0014_work_state_indexes.py должна существовать"
    spec = importlib.util.spec_from_file_location("migration_0014", migration_path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert migration.revision == "0014"
    assert migration.down_revision == "0013"

    migration_0015_path = (
        Path(__file__).resolve().parents[3]
        / "alembic"
        / "versions"
        / "0015_book_id_cascade_indexes.py"
    )
    assert (
        migration_0015_path.exists()
    ), "Миграция 0015_book_id_cascade_indexes.py должна существовать"
    spec_0015 = importlib.util.spec_from_file_location("migration_0015", migration_0015_path)
    assert spec_0015 is not None and spec_0015.loader is not None
    migration_0015 = importlib.util.module_from_spec(spec_0015)
    spec_0015.loader.exec_module(migration_0015)
    assert migration_0015.revision == "0015"
    assert migration_0015.down_revision == "0014"


def test_unit_of_work_retries_transient_sqlite_locks(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(uow_module, "sleep", lambda seconds: sleeps.append(round(seconds, 4)))

    attempts = 0

    class _FakeConnection:
        def exec_driver_sql(self, sql: str) -> None:
            nonlocal attempts
            assert sql == "BEGIN IMMEDIATE"
            attempts += 1
            if attempts <= 6:
                raise OperationalError(
                    sql,
                    {},
                    sqlite3.OperationalError("database is locked"),
                )

    class _FakeBegin:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class _FakeSession:
        def begin(self) -> _FakeBegin:
            return _FakeBegin()

        def connection(self) -> _FakeConnection:
            return _FakeConnection()

        def close(self) -> None:
            pass

    uow = SqlAlchemyUnitOfWork(lambda: _FakeSession())  # type: ignore[arg-type]
    with uow.transaction():
        pass

    assert attempts == 7
    assert sleeps == [0.005, 0.01, 0.015, 0.02, 0.02, 0.02]

    sleeps.clear()
    attempts = 0

    class _AlwaysLockedConnection:
        def exec_driver_sql(self, sql: str) -> None:
            nonlocal attempts
            attempts += 1
            raise OperationalError(
                sql,
                {},
                sqlite3.OperationalError("database is locked"),
            )

    class _AlwaysLockedSession(_FakeSession):
        def connection(self) -> _AlwaysLockedConnection:  # type: ignore[override]
            return _AlwaysLockedConnection()

    locked_uow = SqlAlchemyUnitOfWork(lambda: _AlwaysLockedSession())  # type: ignore[arg-type]
    with pytest.raises(PersistenceError, match="база данных занята"):
        with locked_uow.transaction():
            pass

    assert attempts == 10
    assert len(sleeps) == 9
    assert max(sleeps) == 0.02


async def test_logging_llm_client_and_diagnostics_deduplicate_bounded_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def counting_bounded_body(*args, **kwargs):
        nonlocal calls
        calls += 1
        return bounded_body(*args, **kwargs)

    monkeypatch.setattr(llm_logging_module, "bounded_body", counting_bounded_body)
    monkeypatch.setattr(diagnostics_module, "bounded_body", counting_bounded_body)

    stored_records: dict[str, dict] = {}

    class _FakeLLMCalls:
        def add(self, entry, response_text=None):
            stored_records[entry.id] = {
                "call_id": entry.id,
                "book_id": entry.book_id,
                "attempt_id": entry.attempt_id,
                "started_at": entry.started_at,
                "metadata": entry.diagnostic_meta,
                "body": response_text,
            }

        def diagnostic_record(self, *, call_id=None, attempt_id=None):
            return stored_records.get(call_id)

        def update_diagnostics(self, call_id, *, metadata):
            stored_records[call_id]["metadata"] = metadata

        def store_diagnostic_body(self, call_id, body):
            stored_records[call_id]["body"] = body

        def enforce_diagnostics_policy(self, policy, now, *, book_id=None):
            pass

    class _FakeTx:
        def __init__(self) -> None:
            self.llm_calls = _FakeLLMCalls()

    class _FakeUoW:
        def __init__(self) -> None:
            self.tx = _FakeTx()
            self.diagnostics_policy = None
            self.diagnostics_clock = None

        from contextlib import contextmanager

        @contextmanager
        def transaction(self):
            yield self.tx

    class _InnerClient(LLMClient):
        async def generate(self, request):
            return LLMResponse(
                payload=TranslationResultResponse(blocks=()),
                provider="test",
                model="test-model",
                truncated=False,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
                raw_content='{"blocks": []}',
            )

    uow = _FakeUoW()
    Diagnostics(uow, clock=lambda: datetime(2026, 11, 8, 12, tzinfo=UTC))
    client = LoggingLLMClient(
        inner=_InnerClient(),
        unit_of_work=uow,  # type: ignore[arg-type]
        provider="test",
        model="test-model",
        debug_payloads=True,
        clock=lambda: datetime(2026, 11, 8, 12, tzinfo=UTC),
    )

    request = LLMRequest(
        kind=LLMResponseKind.TRANSLATION_RESULT,
        instructions="sys",
        input_text="user",
        book_id="book-1",
        stage=LLMCallStage.TRANSLATION_UNIT,
    )
    await client.generate(request)
    assert calls == 1

    calls = 0
    stored_records["call-direct"] = {
        "call_id": "call-direct",
        "book_id": "book-1",
        "attempt_id": None,
        "started_at": datetime(2026, 11, 8, 12, tzinfo=UTC),
        "metadata": {"schema_version": 1, "decision": "unknown"},
        "body": None,
    }
    diagnostics = Diagnostics(uow, clock=lambda: datetime(2026, 11, 8, 12, tzinfo=UTC))
    diagnostics.finalize_attempt(
        None,
        "rejected",
        body="already-bounded",
        call_id="call-direct",
        body_meta={"redacted_count": 0, "body_status": "available"},
    )
    assert calls == 0
