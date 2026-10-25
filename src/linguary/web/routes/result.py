"""Result blocks, regeneration, and task retry endpoints (API-01)."""

from __future__ import annotations

from fastapi import APIRouter, Request, status
from fastapi.responses import RedirectResponse

from linguary.application.book_overview import book_result, result_block
from linguary.application.use_cases.retry_task import RetryTaskCommand
from linguary.domain import ProcessingTask, TaskIdentity, TaskStatus, TaskType
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import BookResultResponse, ResultBlockResponse, TaskResponse
from linguary.web.views import render, wants_html
from linguary.web.routes.quality import quality_page_context

router = APIRouter(tags=["result"])


@router.get("/books/{book_id}/result", response_model=BookResultResponse)
def get_book_result(
    request: Request, application: CurrentApplication, book_id: str
) -> BookResultResponse | object:
    """Original text and saved model aids by book blocks."""
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
            quality_ui=quality_page_context(request, application, book_id),
        )
    if wants_html(request):
        return render(
            request,
            "books/result.html",
            page="result",
            data=response,
            export_task=export_task_response,
            quality_ui=quality_page_context(request, application, book_id),
        )
    return response


@router.get(
    "/books/{book_id}/blocks/{block_id}",
    response_model=ResultBlockResponse,
)
def get_book_result_block(
    request: Request, application: CurrentApplication, book_id: str, block_id: str
) -> ResultBlockResponse | object:
    """Return a single result block for polling after regeneration."""
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
    """Enqueue block regeneration and return the fragment for the selected block only."""
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
    """Re-enqueue a recoverable failed task."""
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
    """Construct the block DTO along with its current regeneration task, if any."""
    regeneration_task = task or _regeneration_task(application, book_id, block_id)
    return ResultBlockResponse.of_block(
        result_block(application.unit_of_work, book_id, block_id),
        regeneration_task=regeneration_task,
    )


def _regeneration_task(
    application: CurrentApplication, book_id: str, block_id: str
) -> ProcessingTask | None:
    """Active regeneration task for the selected block to update the UI."""
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
    """Active EPUB assembly task for the result header."""
    identity = TaskIdentity(book_id=book_id, task_type=TaskType.EXPORT_BOOK)
    with application.unit_of_work.transaction() as transaction:
        task = transaction.tasks.find_by_identity(identity)
    if task is None or task.status is TaskStatus.SUCCEEDED:
        return None
    return task


def _wants_result_header_fragment(request: Request) -> bool:
    """Frontend request for the HTML fragment of the result header."""
    return request.headers.get("x-linguary-fragment", "").lower() == "result-header"


def _wants_result_block_fragment(request: Request) -> bool:
    """Frontend request for the HTML fragment of a single block."""
    return request.headers.get("x-linguary-fragment", "").lower() == "result-block"


def _block_anchor(block_id: str) -> str:
    """Block anchor must match the `id` in `partials/result_block.html`."""
    return f"block-{block_id.replace(':', '-').replace('/', '-')}"


__all__ = ["router"]
