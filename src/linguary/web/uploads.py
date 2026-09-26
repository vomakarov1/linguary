"""Приём загруженного файла книги.

Файл приходит из браузера, поэтому здесь и только здесь приложение работает с тем, что
прислал пользователь: проверяет расширение, читает поток по частям и следит за пределом
размера. Имя присланного файла в путь не идёт — книга кладётся под своим идентификатором
(`DataPaths.source_epub`), и подставить каталог именем файла невозможно.

Неудачная загрузка не оставляет следов: недописанный файл удаляется.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from linguary.application.errors import ApplicationError

#: Предел размера загружаемой книги. Художественный EPUB с иллюстрациями держится
#: в десятках мегабайт; предел защищает диск от случайного файла другого рода.
MAX_UPLOAD_BYTES = 256 * 1024 * 1024

#: Сколько читается за раз: книга не поднимается в память целиком.
_CHUNK_BYTES = 1024 * 1024

#: Расширение файла книги. MVP поддерживает только EPUB (PRD § 4).
_EPUB_SUFFIX = ".epub"


class UploadRejectedError(ApplicationError):
    """Загруженный файл не принят: не тот формат или слишком большой."""


class UploadStream(Protocol):
    """Поток загруженного файла в том виде, в каком его даёт веб-фреймворк."""

    async def read(self, size: int = -1) -> bytes:
        """Читает очередную часть файла; пустой результат означает конец."""


def book_title_of_filename(filename: str | None) -> str:
    """Название книги до её разбора: имя загруженного файла без расширения.

    Настоящее название приходит из метаданных EPUB при успешном импорте, но книга
    появляется в списке сразу после загрузки, и до тех пор её нужно как-то называть
    (PRD § 18).
    """
    stem = Path(filename or "").stem.strip()
    return stem or "Книга без названия"


def ensure_epub_filename(filename: str | None) -> None:
    """Проверяет, что пользователь прислал EPUB (PRD § 4, § 6)."""
    if not (filename or "").strip():
        raise UploadRejectedError("файл книги не выбран")
    if Path(filename or "").suffix.lower() != _EPUB_SUFFIX:
        raise UploadRejectedError(
            "поддерживается только формат EPUB: выберите файл с расширением .epub"
        )


async def save_upload(
    stream: UploadStream, *, target: Path, max_bytes: int = MAX_UPLOAD_BYTES
) -> int:
    """Сохраняет загруженный файл и возвращает его размер.

    Превышение предела прерывает запись, а не обнаруживается по готовому файлу: иначе
    книга произвольного размера успела бы лечь на диск.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with target.open("wb") as file:
            while chunk := await stream.read(_CHUNK_BYTES):
                written += len(chunk)
                if written > max_bytes:
                    raise UploadRejectedError(
                        f"файл книги больше допустимых "
                        f"{max_bytes // (1024 * 1024)} МБ"
                    )
                file.write(chunk)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    if not written:
        target.unlink(missing_ok=True)
        raise UploadRejectedError("загруженный файл книги пуст")
    return written


__all__ = [
    "MAX_UPLOAD_BYTES",
    "UploadRejectedError",
    "UploadStream",
    "book_title_of_filename",
    "ensure_epub_filename",
    "save_upload",
]
