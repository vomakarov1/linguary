"""Result blocks, regeneration, and task retry endpoints (API-01)."""

from __future__ import annotations

from collections.abc import Sequence

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse, RedirectResponse
from pydantic import ValidationError

from linguary.application.book_overview import book_result, result_groups, ResultBlockNotFoundError
from linguary.application.pronunciation_projection import project_pronunciations
from linguary.application.use_cases.retry_task import RetryTaskCommand
from linguary.application.use_cases.resolve_correction_candidate import (
    ResolveCorrectionCommand, CorrectionCandidateNotFound, CorrectionCandidateConflict,
)
from linguary.application.retry_policy import ResultValidationError
from linguary.domain.book_conditions import WorkStateError
from linguary.domain import ProcessingTask, TaskStatus, TaskType
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import BookResultResponse, ResultGroupResponse, TaskResponse, ResolveCorrectionInput
from linguary.web.views import render, wants_html
from linguary.web.routes.quality import quality_page_context

router = APIRouter(tags=["result"])


@router.get('/pronunciation-sources', name='pronunciation_sources', response_class=PlainTextResponse)
def pronunciation_sources(application: CurrentApplication) -> PlainTextResponse:
    """Serve the fixed local license bundle, without accepting resource paths."""
    try:
        notices = application.pronunciation_source_notices()
    except (OSError, ValueError):
        raise HTTPException(status_code=503, detail='Уведомления об источниках произношения недоступны.') from None
    return PlainTextResponse(notices)


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
    if (
        wants_html(request)
        and request.headers.get("x-linguary-fragment", "").lower() == "result-groups"
    ):
        return render(request, "partials/result_groups.html", data=response)
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
    "/books/{book_id}/blocks",
    response_model=list[ResultGroupResponse],
)
def get_book_result_blocks(
    request: Request,
    application: CurrentApplication,
    book_id: str,
    block_id: list[str] = Query(...),
) -> list[ResultGroupResponse] | object:
    """Return selected result groups from one book snapshot for batch polling."""
    if len(block_id) > 128 or any(len(identifier) > 512 for identifier in block_id):
        raise HTTPException(status_code=422, detail="Некорректный список блоков.")
    responses = _result_block_responses(application, book_id, block_id)
    if _wants_result_blocks_fragment(request):
        return render(
            request,
            "partials/result_blocks.html",
            blocks=responses,
            book_id=book_id,
        )
    return responses


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


@router.post('/books/{book_id}/correction-candidates/{candidate_id}/resolve', response_model=ResultGroupResponse)
async def resolve_correction_candidate(request: Request, application: CurrentApplication,
                                       book_id: str, candidate_id: str):
    try:
        raw = (await request.json() if request.headers.get('content-type', '').split(';')[0] == 'application/json'
               else dict(await request.form()))
        data = ResolveCorrectionInput.model_validate(raw)
    except (ValidationError, ValueError, TypeError):
        raise HTTPException(status_code=422, detail='Проверьте решение и укажите его причину.') from None
    if application.resolve_correction_candidate is None:
        raise HTTPException(status_code=503, detail='Решения по исправлениям недоступны.')
    try:
        block_id = application.resolve_correction_candidate(ResolveCorrectionCommand(
            book_id, candidate_id, data.proposal_sha256, data.input_sha256, data.action, data.reason))
    except CorrectionCandidateNotFound:
        raise HTTPException(status_code=404, detail='Предложение исправления не найдено.') from None
    except (CorrectionCandidateConflict, WorkStateError, ResultValidationError):
        raise HTTPException(status_code=409,
            detail='Предложение или данные книги изменились. Обновите страницу результата.') from None
    response = _result_block_response(application, book_id, block_id)
    if _wants_result_block_fragment(request):
        return render(request, 'partials/result_block.html', block=response, book_id=book_id)
    if wants_html(request):
        return RedirectResponse(url=f'/books/{book_id}/result#{_block_anchor(block_id)}',
                                status_code=status.HTTP_303_SEE_OTHER)
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
    response, = _result_block_responses(application, book_id, (block_id,), task=task)
    return response


def _result_block_responses(
    application: CurrentApplication,
    book_id: str,
    block_ids: Sequence[str],
    *,
    task: ProcessingTask | None = None,
) -> tuple[ResultGroupResponse, ...]:
    """Build selected result groups while taking one book snapshot and pronunciation projection."""
    snapshot = book_result(application.unit_of_work, book_id)
    requested_ids = tuple(dict.fromkeys(block_ids))
    requested_set = set(requested_ids)
    selected_groups = []
    found_ids: set[str] = set()
    for chapter in snapshot.chapters:
        for group in result_groups(chapter.blocks):
            matching_ids = [member.block.id for member in group
                            if member.block.id in requested_set]
            if matching_ids:
                selected_groups.append((group, matching_ids[0]))
                found_ids.update(matching_ids)
    missing_ids = requested_set - found_ids
    if missing_ids:
        missing_id = next(block_id for block_id in requested_ids if block_id in missing_ids)
        raise ResultBlockNotFoundError(f"блока {missing_id} нет в книге {book_id}")

    tasks = _regeneration_tasks(snapshot.tasks)
    if task is not None and task.status is not TaskStatus.SUCCEEDED:
        tasks[task.scope_id] = task
    summaries = tuple(block for chapter in snapshot.chapters for block in chapter.blocks)
    pronunciations = project_pronunciations(
        tuple(summary.block for summary in summaries),
        {summary.block.id: summary.result for summary in summaries if summary.result is not None},
    )
    return tuple(
        ResultGroupResponse.of_group(group, tasks=tasks, block_id=block_id,
                                      source_language=snapshot.book.source_language,
                                      pronunciations=pronunciations)
        for group, block_id in selected_groups
    )


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


def _wants_result_blocks_fragment(request: Request) -> bool:
    """Frontend request for one or more block fragments."""
    return request.headers.get("x-linguary-fragment", "").lower() == "result-blocks"


def _block_anchor(block_id: str) -> str:
    """Block anchor must match the `id` in `partials/result_block.html`."""
    return f"block-{block_id.replace(':', '-').replace('/', '-')}"


__all__ = ["router"]
