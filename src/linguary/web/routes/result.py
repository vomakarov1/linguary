"""Endpoints результата по блокам, перегенерации и повтора задач (API-01)."""

from __future__ import annotations

from fastapi import APIRouter, Request, status
from fastapi.responses import RedirectResponse

from linguary.application.book_overview import book_result, result_block
from linguary.application.use_cases.retry_task import RetryTaskCommand
from linguary.domain import ProcessingTask, TaskIdentity, TaskStatus, TaskType
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import BookResultResponse, ResultBlockResponse, TaskResponse
from linguary.web.views import render, wants_html

router = APIRouter(tags=["result"])


@router.get("/books/{book_id}/result", response_model=BookResultResponse)
def get_book_result(
    request: Request, application: CurrentApplication, book_id: str
) -> BookResultResponse | object:
    """Оригинал и сохранённая помощь модели по блокам книги."""
    response = BookResultResponse.of_result(
        book_result(application.unit_of_work, book_id)
    )
    export_task = _export_task(application, book_id)
    export_task_response = (
        TaskResponse.of_task(export_task) if export_task is not None else None
    )
    if _wants_result_header_fragment(request):
        return render(
            request,
            "partials/result_header.html",
            data=response,
            export_task=export_task_response,
        )
    if wants_html(request):
        return render(
            request,
            "books/result.html",
            page="result",
            data=response,
            export_task=export_task_response,
        )
    return response


@router.get(
    "/books/{book_id}/blocks/{block_id}",
    response_model=ResultBlockResponse,
)
def get_book_result_block(
    request: Request, application: CurrentApplication, book_id: str, block_id: str
) -> ResultBlockResponse | object:
    """Возвращает один блок результата для опроса после перегенерации."""
    response = _result_block_response(application, book_id, block_id)
    if _wants_result_block_fragment(request):
        return render(
            request,
            "partials/result_block.html",
            block=response,
            book_id=book_id,
        )
    if wants_html(request):
        return RedirectResponse(
            url=f"/books/{book_id}/result#{_block_anchor(block_id)}",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    return response


@router.post(
    "/books/{book_id}/blocks/{block_id}/regenerate",
    response_model=ResultBlockResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def regenerate_book_block(
    request: Request, application: CurrentApplication, book_id: str, block_id: str
) -> ResultBlockResponse | object:
    """Ставит перегенерацию и возвращает фрагмент только выбранного блока."""
    task = await application.regeneration.start(book_id, block_id)
    response = _result_block_response(application, book_id, block_id, task=task)
    if _wants_result_block_fragment(request):
        return render(
            request,
            "partials/result_block.html",
            block=response,
            book_id=book_id,
        )
    if wants_html(request):
        return RedirectResponse(
            url=f"/books/{book_id}/result#{_block_anchor(block_id)}",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    return response


@router.post("/tasks/{task_id}/retry", response_model=TaskResponse)
def retry_task(
    request: Request, application: CurrentApplication, task_id: str
) -> TaskResponse | RedirectResponse:
    """Возвращает восстановимую неуспешную задачу в очередь."""
    task = application.retry_task(RetryTaskCommand(task_id=task_id))
    if wants_html(request):
        return RedirectResponse(
            url=f"/books/{task.book_id}", status_code=status.HTTP_303_SEE_OTHER
        )
    return TaskResponse.of_task(task)


def _result_block_response(
    application: CurrentApplication,
    book_id: str,
    block_id: str,
    *,
    task: ProcessingTask | None = None,
) -> ResultBlockResponse:
    """Собирает DTO блока вместе с текущей задачей перегенерации, если она есть."""
    regeneration_task = task or _regeneration_task(application, book_id, block_id)
    return ResultBlockResponse.of_block(
        result_block(application.unit_of_work, book_id, block_id),
        regeneration_task=regeneration_task,
    )


def _regeneration_task(
    application: CurrentApplication, book_id: str, block_id: str
) -> ProcessingTask | None:
    """Активная задача перегенерации выбранного блока для обновления UI."""
    identity = TaskIdentity(
        book_id=book_id,
        task_type=TaskType.REGENERATE_BLOCK,
        scope_id=block_id,
    )
    with application.unit_of_work.transaction() as transaction:
        task = transaction.tasks.find_by_identity(identity)
    if task is None or task.status is TaskStatus.SUCCEEDED:
        return None
    return task


def _export_task(
    application: CurrentApplication, book_id: str
) -> ProcessingTask | None:
    """Активная задача сборки EPUB для шапки результата."""
    identity = TaskIdentity(book_id=book_id, task_type=TaskType.EXPORT_BOOK)
    with application.unit_of_work.transaction() as transaction:
        task = transaction.tasks.find_by_identity(identity)
    if task is None or task.status is TaskStatus.SUCCEEDED:
        return None
    return task


def _wants_result_header_fragment(request: Request) -> bool:
    """Запрос от фронтенда на HTML-фрагмент шапки результата."""
    return request.headers.get("x-linguary-fragment", "").lower() == "result-header"


def _wants_result_block_fragment(request: Request) -> bool:
    """Запрос от фронтенда на HTML-фрагмент одного блока."""
    return request.headers.get("x-linguary-fragment", "").lower() == "result-block"


def _block_anchor(block_id: str) -> str:
    """Якорь блока должен совпадать с `id` в `partials/result_block.html`."""
    return f"block-{block_id.replace(':', '-').replace('/', '-')}"


__all__ = ["router"]
