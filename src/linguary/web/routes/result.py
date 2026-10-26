"""Result blocks, regeneration, and task retry endpoints (API-01)."""

from __future__ import annotations

from collections.abc import Sequence

from fastapi import APIRouter, Request, status
from fastapi.responses import RedirectResponse

from linguary.application.book_overview import book_result, result_block
from linguary.application.use_cases.retry_task import RetryTaskCommand
from linguary.domain import ProcessingTask, TaskStatus, TaskType
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import BookResultResponse, ResultGroupResponse, TaskResponse
from linguary.web.views import render, wants_html
from linguary.web.routes.quality import quality_page_context

router = APIRouter(tags=["result"])


@router.get("/books/{book_id}/result", response_model=BookResultResponse)
def get_book_result(
    request: Request, application: CurrentApplication, book_id: str
) -> BookResultResponse | object:
    """Original text and saved model aids by book blocks."""
    summary = book_result(application.unit_of_work, book_id)
    response = BookResultResponse.of_result(summary)
    header_only = _wants_result_header_fragment(request)
    if not header_only:
        tasks = _regeneration_tasks(summary.tasks)
        for chapter in response.chapters:
            for group in chapter.groups:
                for member in group.member_blocks:
                    current = tasks.get(member.block_id)
                    if current is not None:
                        member.regeneration_task = TaskResponse.of_task(current)
    export_task = next((task for task in summary.tasks
                        if task.task_type is TaskType.EXPORT_BOOK
                        and task.status is not TaskStatus.SUCCEEDED), None)
    export_task_response = (
        TaskResponse.of_task(export_task) if export_task is not None else None
    )
    if header_only:
        return render(
            request,
            "partials/result_header.html",
            data=response,
            export_task=export_task_response,
            quality_ui=quality_page_context(request, application, book_id, tasks=summary.tasks),
        )
    if wants_html(request):
        return render(
            request,
            "books/result.html",
            page="result",
            data=response,
            export_task=export_task_response,
            quality_ui=quality_page_context(request, application, book_id, tasks=summary.tasks),
        )
    return response


@router.get(
    "/books/{book_id}/blocks/{block_id}",
    response_model=ResultGroupResponse,
)
def get_book_result_block(
    request: Request, application: CurrentApplication, book_id: str, block_id: str
) -> ResultGroupResponse | object:
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
    response_model=ResultGroupResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def regenerate_book_block(
    request: Request, application: CurrentApplication, book_id: str, block_id: str
) -> ResultGroupResponse | object:
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
) -> ResultGroupResponse:
    """Construct the block DTO along with its current regeneration task, if any."""
    summaries = result_block(application.unit_of_work, book_id, block_id)
    with application.unit_of_work.transaction() as transaction:
        tasks = _regeneration_tasks(transaction.tasks.list_for_book(book_id))
    if task is not None and task.status is not TaskStatus.SUCCEEDED:
        tasks[task.scope_id] = task
    return ResultGroupResponse.of_group(summaries, tasks=tasks, block_id=block_id)


def _regeneration_tasks(tasks: Sequence[ProcessingTask]) -> dict[str, ProcessingTask]:
    """Index active regeneration tasks from the book's existing batch snapshot."""
    return {task.scope_id: task for task in tasks
            if task.task_type is TaskType.REGENERATE_BLOCK
            and task.status is not TaskStatus.SUCCEEDED}


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
