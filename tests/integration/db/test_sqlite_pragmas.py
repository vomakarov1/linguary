"""SQLite connection settings: WAL mode and enabled foreign keys.

Foreign keys in SQLite are disabled by default. Without `PRAGMA foreign_keys=ON`,
cascading deletes and referential integrity of the schema do not work at all,
making tests that verify them falsely green. Therefore, this mode is verified directly.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from linguary.adapters.db.models import SectionRow


def test_database_opens_in_wal_mode(engine: Engine) -> None:
    """File database is opened in WAL journal mode."""
    with engine.connect() as connection:
        journal_mode = connection.execute(text("PRAGMA journal_mode")).scalar_one()
    assert journal_mode.lower() == "wal"


def test_foreign_keys_are_enabled_on_every_connection(engine: Engine) -> None:
    """Each connection is opened with foreign keys enabled."""
    for _ in range(2):
        with engine.connect() as connection:
            assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1


def test_unknown_foreign_key_is_rejected(engine: Engine) -> None:
    """Reference to a nonexistent book is rejected by the database rather than silently stored."""
    with Session(engine) as session, pytest.raises(IntegrityError):
        session.add(
            SectionRow(
                id="section-orphan",
                book_id="нет такой книги",
                title="Chapter 1",
                order=0,
                source_resource_path="OEBPS/chapter1.xhtml",
            )
        )
        session.commit()
