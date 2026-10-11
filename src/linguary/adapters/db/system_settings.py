"""Хранилище настроек с атомарной проверкой редакции."""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import Engine, insert, select, text, update
from sqlalchemy.exc import SQLAlchemyError

from linguary.adapters.config.settings import SettingsError
from linguary.adapters.db.models import SystemSettingsRow


class SettingsStoreError(SettingsError):
    """Настройки не удалось прочитать или записать."""


class SettingsRevisionConflict(SettingsError):
    """Настройки изменены другой операцией."""


@dataclass(frozen=True, slots=True)
class StoredSystemSettings:
    revision: int
    overrides: dict[str, object] | None = field(repr=False)
    api_key: str | None = field(repr=False)
    proxy_url: str | None = field(default=None, repr=False)


class SystemSettingsStore:
    """Короткая транзакция SQLite сериализует проверку редакции и запись."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def load(self) -> StoredSystemSettings | None:
        try:
            with self._engine.connect() as connection:
                row = (
                    connection.execute(select(SystemSettingsRow.__table__))
                    .mappings()
                    .first()
                )
                if row is None:
                    return None
                return StoredSystemSettings(
                    row['revision'], row['overrides_json'], row['api_key'], row['proxy_url']
                )
        except SQLAlchemyError:
            raise SettingsStoreError('Не удалось прочитать настройки') from None

    def save(
        self,
        overrides: dict[str, object],
        api_key: str | None,
        *,
        expected_revision: int,
        proxy_url: str | None = None,
    ) -> StoredSystemSettings:
        fields = {'provider', 'model', 'base_url', 'requests_per_minute', 'concurrency'}
        if not isinstance(overrides, dict) or set(overrides) not in (fields, fields | {'proxy_mode'}):
            raise SettingsStoreError('Недопустимый состав сохраняемых настроек')
        return self._write(dict(overrides), api_key, proxy_url, expected_revision=expected_revision)

    def reset(self, *, expected_revision: int) -> StoredSystemSettings:
        return self._write(None, None, None, expected_revision=expected_revision)

    def _write(
        self,
        overrides: dict[str, object] | None,
        api_key: str | None,
        proxy_url: str | None,
        *,
        expected_revision: int,
    ) -> StoredSystemSettings:
        if type(expected_revision) is not int or expected_revision < 0:
            raise SettingsRevisionConflict('Редакция настроек недопустима')
        try:
            with self._engine.connect() as connection:
                connection.execute(text('BEGIN IMMEDIATE'))
                revision = connection.execute(
                    select(SystemSettingsRow.revision).where(SystemSettingsRow.id == 1)
                ).scalar_one_or_none()
                if (revision or 0) != expected_revision:
                    raise SettingsRevisionConflict('Настройки изменены другой операцией')
                values = dict(
                    revision=expected_revision + 1,
                    overrides_json=overrides,
                    api_key=api_key,
                    proxy_url=proxy_url,
                )
                if revision is None:
                    connection.execute(insert(SystemSettingsRow).values(id=1, **values))
                else:
                    connection.execute(
                        update(SystemSettingsRow)
                        .where(SystemSettingsRow.id == 1)
                        .values(**values)
                    )
                try:
                    connection.commit()
                except SQLAlchemyError:
                    connection.connection.rollback()
                    raise
            return StoredSystemSettings(
                expected_revision + 1,
                None if overrides is None else dict(overrides),
                api_key,
                proxy_url,
            )
        except SQLAlchemyError:
            raise SettingsStoreError('Не удалось записать настройки') from None
