"""Background EPUB assembly and download endpoints (API-01)."""

from __future__ import annotations

import re

from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse, RedirectResponse

from linguary.application.use_cases.export_book import ExportArtifactNotFoundError
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import TaskResponse
from linguary.web.views import wants_html

router = APIRouter(prefix="/books/{book_id}", tags=["export"])

#: Characters prohibited in filenames on at least one user OS, and control characters.
_UNSAFE_FILENAME_CHARACTERS = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f]+')

#: Maximum title length in filename: filesystems limit filenames to 255 bytes.
_MAX_TITLE_LENGTH = 120

#: Fallback filename when nothing remains of the book title.
_FALLBACK_TITLE = "book"


@router.post(
    "/export",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_export(
    request: Request, application: CurrentApplication, book_id: str
) -> TaskResponse | RedirectResponse:
    """Verify book readiness and enqueue EPUB assembly."""
    task = await application.start_export(book_id)
    if wants_html(request):
        return RedirectResponse(
            url=f"/books/{book_id}/result", status_code=status.HTTP_303_SEE_OTHER
        )
    return TaskResponse.of_task(task)


@router.get("/download", response_class=FileResponse)
def download_export(application: CurrentApplication, book_id: str) -> FileResponse:
    """Serve the latest successfully assembled EPUB as a file attachment."""
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
    """Name of the downloaded file: book title and target language, `Title (ru).epub`.

    The title comes from the metadata of an untrusted EPUB, so characters that the
    filesystem would reject or interpret as a path are replaced with spaces.
    """
    safe = " ".join(_UNSAFE_FILENAME_CHARACTERS.sub(" ", title).split())
    safe = safe[:_MAX_TITLE_LENGTH].strip(" .") or _FALLBACK_TITLE
    language = " ".join(_UNSAFE_FILENAME_CHARACTERS.sub(" ", target_language).split())
    return f"{safe} ({language}).epub" if language else f"{safe}.epub"


__all__ = ["download_filename", "router"]
