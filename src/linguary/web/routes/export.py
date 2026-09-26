"""Endpoints фоновой сборки и скачивания итогового EPUB (API-01)."""

from __future__ import annotations

import re

from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse, RedirectResponse

from linguary.application.use_cases.export_book import ExportArtifactNotFoundError
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import TaskResponse
from linguary.web.views import wants_html

router = APIRouter(prefix="/books/{book_id}", tags=["export"])

#: Символы, недопустимые в имени файла хотя бы в одной из систем пользователя,
#: и управляющие символы.
_UNSAFE_FILENAME_CHARACTERS = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f]+')

#: Предел длины названия в имени файла: файловые системы ограничивают имя 255 байтами.
_MAX_TITLE_LENGTH = 120

#: Название файла, если от названия книги ничего не осталось.
_FALLBACK_TITLE = "book"


@router.post(
    "/export",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_export(
    request: Request, application: CurrentApplication, book_id: str
) -> TaskResponse | RedirectResponse:
    """Проверяет готовность книги и ставит сборку EPUB в очередь."""
    task = await application.start_export(book_id)
    if wants_html(request):
        return RedirectResponse(
            url=f"/books/{book_id}/result", status_code=status.HTTP_303_SEE_OTHER
        )
    return TaskResponse.of_task(task)


@router.get("/download", response_class=FileResponse)
def download_export(application: CurrentApplication, book_id: str) -> FileResponse:
    """Отдаёт последний успешно собранный EPUB как вложение."""
    application.export_book.downloadable_task(book_id)
    path = application.paths.export_epub(book_id)
    if not path.is_file():
        raise ExportArtifactNotFoundError(
            f"файл экспортированной книги {book_id} не найден"
        )
    with application.unit_of_work.transaction() as transaction:
        book = transaction.books.get(book_id)
    if book is None:
        raise ExportArtifactNotFoundError(f"книги {book_id} нет в хранилище")
    return FileResponse(
        path=path,
        media_type="application/epub+zip",
        filename=download_filename(book.metadata.title, book.target_language),
    )


def download_filename(title: str, target_language: str) -> str:
    """Имя скачиваемого файла: название книги и целевой язык, `Title (ru).epub`.

    Название приходит из метаданных недоверенного EPUB, поэтому символы, которые
    файловая система не примет или истолкует как путь, заменяются пробелом.
    """
    safe = " ".join(_UNSAFE_FILENAME_CHARACTERS.sub(" ", title).split())
    safe = safe[:_MAX_TITLE_LENGTH].strip(" .") or _FALLBACK_TITLE
    language = " ".join(_UNSAFE_FILENAME_CHARACTERS.sub(" ", target_language).split())
    return f"{safe} ({language}).epub" if language else f"{safe}.epub"


__all__ = ["download_filename", "router"]
