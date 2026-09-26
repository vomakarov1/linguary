"""Настройки соединения SQLite: WAL и включённые внешние ключи.

Внешние ключи в SQLite по умолчанию выключены. Без `PRAGMA foreign_keys=ON` каскадное
удаление и ссылочная целостность схемы не работают вовсе, а тесты, которые их проверяют,
становятся ложно-зелёными. Поэтому режим проверяется отдельно и прямо.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from linguary.adapters.db.models import SectionRow


def test_database_opens_in_wal_mode(engine: Engine) -> None:
    """Файловая база открыта в режиме журнала WAL."""
    with engine.connect() as connection:
        journal_mode = connection.execute(text("PRAGMA journal_mode")).scalar_one()
    assert journal_mode.lower() == "wal"


def test_foreign_keys_are_enabled_on_every_connection(engine: Engine) -> None:
    """Каждое соединение открывается с включёнными внешними ключами."""
    for _ in range(2):
        with engine.connect() as connection:
            assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1


def test_unknown_foreign_key_is_rejected(engine: Engine) -> None:
    """Ссылка на несуществующую книгу отвергается базой, а не сохраняется молча."""
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
