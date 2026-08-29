"""Регистрация загруженной книги: строка книги появляется до её импорта.

Пользователь видит книгу в списке сразу после загрузки файла, в состоянии `IMPORTING`
(PRD § 18), а разбор EPUB выполняется фоновой задачей. Поэтому запись книги создаёт
отдельный короткий сценарий, а не сам импорт: иначе список книг оставался бы пустым всё
время разбора, а неуспешный импорт вообще не оставил бы следа.

Названия книги до разбора файла ещё нет — его берут из имени загруженного файла. Успешный
импорт заменяет метаданные на настоящие, взятые из EPUB.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from linguary.application.errors import ApplicationError
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.domain import BookEvent, BookMetadata, apply_book_event


class RegisterBookError(ApplicationError):
    """Загруженную книгу нельзя зарегистрировать."""


@dataclass(frozen=True, slots=True)
class RegisterBookCommand:
    """Команда регистрации: куда сохранён файл и с какими языками его читать."""

    book_id: str
    source_epub_path: str
    title: str
    source_language: str
    target_language: str

    def __post_init__(self) -> None:
        if not self.book_id.strip():
            raise RegisterBookError("идентификатор книги пуст")
        if not self.source_epub_path.strip():
            raise RegisterBookError("путь к загруженному файлу книги пуст")
        if not self.title.strip():
            raise RegisterBookError("название книги пусто")
        if not self.source_language.strip():
            raise RegisterBookError("исходный язык книги не выбран")
        if not self.target_language.strip():
            raise RegisterBookError("целевой язык книги не выбран")


class RegisterBook:
    """Заводит книгу в состоянии `IMPORTING` (ALG-06)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._clock = clock or _utc_now

    def __call__(self, command: RegisterBookCommand) -> BookRecord:
        """Создаёт запись книги; повторная регистрация того же идентификатора запрещена."""
        record = BookRecord(
            id=command.book_id,
            metadata=BookMetadata(title=command.title),
            source_language=command.source_language,
            target_language=command.target_language,
            status=apply_book_event(None, BookEvent.FILE_UPLOADED),
            source_epub_path=command.source_epub_path,
            created_at=self._clock(),
        )
        with self._unit_of_work.transaction() as transaction:
            if transaction.books.get(command.book_id) is not None:
                raise RegisterBookError(
                    f"книга {command.book_id} уже зарегистрирована"
                )
            transaction.books.add(record)
        return record


def _utc_now() -> datetime:
    """Текущий момент в UTC: хранилище работает только с осведомлённым временем."""
    return datetime.now(UTC)


__all__ = [
    "RegisterBook",
    "RegisterBookCommand",
    "RegisterBookError",
]
