"""Accepting uploaded book files.

The file arrives from the browser, so here and only here does the application process user-provided
input: validates extension, reads stream in chunks, and enforces the size limit.
The uploaded file name is not used in paths — the book is stored under its own identifier
(`DataPaths.source_epub`), making path traversal via file name impossible.

Failed uploads leave no trace: incomplete files are deleted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from linguary.application.errors import ApplicationError

#: Maximum uploaded book size. Illustrated fiction EPUBs remain within
#: tens of megabytes; the limit protects disk from accidental files of other types.
MAX_UPLOAD_BYTES = 256 * 1024 * 1024

#: Chunk size read at once: the book is not loaded into memory all at once.
_CHUNK_BYTES = 1024 * 1024

#: Book file extension. MVP supports only EPUB (PRD § 4).
_EPUB_SUFFIX = ".epub"


class UploadRejectedError(ApplicationError):
    """Uploaded file rejected: invalid format or too large."""


class UploadStream(Protocol):
    """Uploaded file stream as provided by web framework."""

    async def read(self, size: int = -1) -> bytes:
        """Reads next chunk of file; empty bytes indicates end of stream."""


def book_title_of_filename(filename: str | None) -> str:
    """Book title before parsing: uploaded file name without extension.

    The real title comes from EPUB metadata upon successful import, but the book
    appears in the list immediately after upload and needs a name until then
    (PRD § 18).
    """
    stem = Path(filename or "").stem.strip()
    return stem or "Книга без названия"


def ensure_epub_filename(filename: str | None) -> None:
    """Verifies user uploaded an EPUB (PRD § 4, § 6)."""
    if not (filename or "").strip():
        raise UploadRejectedError("файл книги не выбран")
    if Path(filename or "").suffix.lower() != _EPUB_SUFFIX:
        raise UploadRejectedError(
            "поддерживается только формат EPUB: выберите файл с расширением .epub"
        )


async def save_upload(
    stream: UploadStream, *, target: Path, max_bytes: int = MAX_UPLOAD_BYTES
) -> int:
    """Saves uploaded file and returns its size in bytes.

    Exceeding the limit aborts writing immediately rather than checking after completion:
    otherwise an arbitrarily large book could fill up the disk.
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
