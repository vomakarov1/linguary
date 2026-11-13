"""Result blocks, regeneration, and task retry endpoints (API-01)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from urllib.parse import parse_qs, urlsplit

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse, RedirectResponse
from pydantic import ValidationError

from linguary.application.book_overview import book_result, ResultBlockNotFoundError
from linguary.application.pronunciation_projection import project_pronunciations
from linguary.application.use_cases.retry_task import RetryTaskCommand
from linguary.application.use_cases.resolve_correction_candidate import (
    ResolveCorrectionCommand, CorrectionCandidateNotFound, CorrectionCandidateConflict,
)
from linguary.application.retry_policy import ResultValidationError
from linguary.domain.book_conditions import WorkStateError
from linguary.domain import ProcessingTask, TaskStatus, TaskType
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import (
    BookResultResponse, ResultBlockResponse, TaskResponse, ResolveCorrectionInput, reading_units,
)
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
    request: Request,
    application: CurrentApplication,
    book_id: str,
    filter_param: str | None = Query(default=None, alias="filter"),
) -> BookResultResponse | object:
    """Original text and saved model aids by book blocks."""
    result_filter = (
        "pending_corrections" if filter_param == "pending_corrections" else None
    )
    header_only = _wants_result_header_fragment(request)
    try:
        summary = book_result(
            application.unit_of_work,
            book_id,
            include_chapters=not header_only,
        )
    except TypeError:
        summary = book_result(application.unit_of_work, book_id)
    effective_summary = summary
    if not header_only and result_filter == "pending_corrections":
        filtered_chapters = []
        for chapter in summary.chapters:
            kept_blocks = tuple(
                block
                for block in chapter.blocks
                if any(
                    candidate.get("state") == "awaiting_human"
                    for candidate in block.correction_candidates
                )
            )
            if not kept_blocks:
                continue
            filtered_chapters.append(replace(chapter, blocks=kept_blocks))
        effective_summary = replace(summary, chapters=tuple(filtered_chapters))
    response = BookResultResponse.of_result(effective_summary)
    if not header_only:
        tasks = _regeneration_tasks(summary.tasks)
        for chapter in response.chapters:
            for block in chapter.blocks:
                current = tasks.get(block.block_id)
                if current is not None:
                    block.regeneration_task = TaskResponse.of_task(current)
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
            result_filter=result_filter,
            default_reader_level=application.settings.reading.normalized_level,
            default_seed_first_chapter=application.settings.checks.seed_first_chapter,
        )
    if (
        wants_html(request)
        and request.headers.get("x-linguary-fragment", "").lower() == "result-groups"
    ):
        return render(
            request,
            "partials/result_groups.html",
            data=response,
            result_filter=result_filter,
        )
    if wants_html(request):
        return render(
            request,
            "books/result.html",
            page="result",
            data=response,
            export_task=export_task_response,
            quality_ui=quality_page_context(request, application, book_id, tasks=summary.tasks),
            result_filter=result_filter,
            default_reader_level=application.settings.reading.normalized_level,
            default_seed_first_chapter=application.settings.checks.seed_first_chapter,
        )
    return response


@router.get(
    "/books/{book_id}/blocks",
    response_model=list[ResultBlockResponse],
)
def get_book_result_blocks(
    request: Request,
    application: CurrentApplication,
    book_id: str,
    block_id: list[str] = Query(...),
) -> list[ResultBlockResponse] | object:
    """Return selected result blocks from one book snapshot for batch polling."""
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


@router.post('/books/{book_id}/correction-candidates/{candidate_id}/resolve', response_model=ResultBlockResponse)
async def resolve_correction_candidate(request: Request, application: CurrentApplication,
                                       book_id: str, candidate_id: str):
    is_json = request.headers.get('content-type', '').split(';')[0] == 'application/json'
    form_filter: str | None = None
    try:
        if is_json:
            raw = await request.json()
        else:
            raw = dict(await request.form())
            form_filter_val = raw.pop('filter', None)
            if isinstance(form_filter_val, str):
                form_filter = form_filter_val
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
    referer_filters = parse_qs(urlsplit(request.headers.get('referer', '')).query).get('filter', [])
    filter_active = (
        request.query_params.get('filter') == 'pending_corrections'
        or form_filter == 'pending_corrections'
        or 'pending_corrections' in referer_filters
    )
    result_filter = 'pending_corrections' if filter_active else None
    response = _result_block_response(application, book_id, block_id)
    if _wants_result_block_fragment(request):
        return render(
            request,
            'partials/result_block.html',
            block=response,
            book_id=book_id,
            result_filter=result_filter,
        )
    if wants_html(request):
        query_suffix = '?filter=pending_corrections' if filter_active else ''
        return RedirectResponse(
            url=f'/books/{book_id}/result{query_suffix}#{_block_anchor(block_id)}',
            status_code=status.HTTP_303_SEE_OTHER,
        )
    return response


@router.post("/tasks/{task_id}/retry", response_model=TaskResponse)
def retry_task(
    request: Request, application: CurrentApplication, task_id: str
) -> TaskResponse | RedirectResponse:
    """Re-enqueue a recoverable failed task."""
    task = application.retry_task(RetryTaskCommand(task_id=task_id))
    notify = getattr(application.worker_pool, "notify", None)
    if callable(notify):
        notify()
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
    response, = _result_block_responses(application, book_id, (block_id,), task=task)
    return response


def _result_block_responses(
    application: CurrentApplication,
    book_id: str,
    block_ids: Sequence[str],
    *,
    task: ProcessingTask | None = None,
) -> tuple[ResultBlockResponse, ...]:
    """Build selected result blocks from one book snapshot and pronunciation projection."""
    requested_ids = tuple(dict.fromkeys(block_ids))
    try:
        snapshot = book_result(
            application.unit_of_work,
            book_id,
            include_quality=False,
            block_ids=requested_ids,
        )
    except TypeError:
        try:
            snapshot = book_result(
                application.unit_of_work, book_id, include_quality=False
            )
        except TypeError:
            snapshot = book_result(application.unit_of_work, book_id)
    requested_set = set(requested_ids)
    selected_blocks = []
    found_ids: set[str] = set()
    for chapter in snapshot.chapters:
        for block in chapter.blocks:
            if block.block.id in requested_set:
                selected_blocks.append((chapter, block))
                found_ids.add(block.block.id)
    missing_ids = requested_set - found_ids
    if missing_ids:
        missing_id = next(block_id for block_id in requested_ids if block_id in missing_ids)
        raise ResultBlockNotFoundError(f"блока {missing_id} нет в книге {book_id}")

    tasks = _regeneration_tasks(snapshot.tasks)
    if task is not None and task.status is not TaskStatus.SUCCEEDED:
        tasks[task.scope_id] = task
    summaries = tuple(block for chapter in snapshot.chapters for block in chapter.blocks)
    blocks_seq = tuple(summary.block for summary in summaries)
    results_map = {
        summary.block.id: summary.result
        for summary in summaries
        if summary.result is not None
    }
    try:
        pronunciations = project_pronunciations(
            blocks_seq,
            results_map,
            initial_seen=getattr(snapshot, "prior_pronunciation_hints", None),
        )
    except TypeError:
        pronunciations = project_pronunciations(blocks_seq, results_map)
    return tuple(
        ResultBlockResponse.of_block(
            block,
            regeneration_task=tasks.get(block.block.id),
            pronunciations=pronunciations,
            units=reading_units(chapter, block.block.id),
        )
        for chapter, block in selected_blocks
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
