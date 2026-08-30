"""Endpoints фоновой сборки и скачивания итогового EPUB (API-01)."""

from __future__ import annotations

from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse, RedirectResponse

from linguary.application.use_cases.export_book import ExportArtifactNotFoundError
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import TaskResponse
from linguary.web.views import wants_html

router = APIRouter(prefix="/books/{book_id}", tags=["export"])


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
    return FileResponse(
        path=path,
        media_type="application/epub+zip",
        filename=path.name,
    )


__all__ = ["router"]
