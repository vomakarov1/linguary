"""Endpoints for starting processing and returning to glossary review (API-01)."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from fastapi import APIRouter, Form, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from linguary.application.book_conditions import ConditionsError
from linguary.application.errors import ApplicationError
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.application.use_cases.pause_book import (
    PauseBookCommand,
    PauseBookError,
    ResumeBookCommand,
)
from linguary.application.use_cases.reset_processing import (
    ChooseBookConditions,
    ResetProcessingCommand,
)
from linguary.application.use_cases.restart_translation import (
    RestartTranslationCommand,
    RestartTranslationConflict,
    RestartTranslationError,
)
from linguary.domain import BookStatus, TaskStatus, TaskType
from linguary.domain.book_conditions import WorkStateError
from linguary.web.dependencies import CurrentApplication
from linguary.web.routes.quality import input_error_response, read_input
from linguary.web.schemas import (
    ProcessingStartResponse,
    ResetProcessingRequest,
    ResetProcessingResponse,
    TaskResponse,
)
from linguary.web.views import wants_html

router = APIRouter(prefix="/books/{book_id}", tags=["processing"])


class _InitialAnalysisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reader_level: Literal["a2", "b1", "b2", "c1"] | None = None
    english_help: bool | None = None
    pronunciation_help_enabled: bool | None = None
    pronunciation_variant: Literal["uk", "us"] | None = None
    seed_first_chapter: bool | None = None


class _RestartRecoveryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_generation: int = Field(ge=0)
    confirmed: bool = False


async def _parse_initial_analysis_input(request: Request) -> _InitialAnalysisInput:
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type == "application/json":
        body = (await request.body()).strip()
        if not body:
            return _InitialAnalysisInput()
        try:
            raw = json.loads(body)
        except ValueError:
            raise HTTPException(422, detail={"reason_code": "invalid_request"}) from None
        if not isinstance(raw, dict):
            raise HTTPException(422, detail={"reason_code": "invalid_request"})
    elif content_type in {
        "application/x-www-form-urlencoded",
        "multipart/form-data",
    }:
        form = await request.form()
        raw = {}
        for key, value in form.items():
            if key in {"reader_level", "pronunciation_variant"} and value == "":
                raw[key] = None
            elif key in {
                "english_help",
                "pronunciation_help_enabled",
                "seed_first_chapter",
            }:
                raw[key] = str(value).lower() in {"true", "on", "1"}
            else:
                raw[key] = value
        if raw:
            raw.setdefault("english_help", False)
            raw.setdefault("pronunciation_help_enabled", False)
            raw.setdefault("seed_first_chapter", False)
    else:
        return _InitialAnalysisInput()
    try:
        return _InitialAnalysisInput.model_validate(raw)
    except (ValidationError, ValueError, TypeError):
        raise HTTPException(422, detail={"reason_code": "invalid_request"}) from None


def _conditions_http_error(error: ConditionsError) -> HTTPException:
    code = (
        404
        if error.reason_code == "book_not_found"
        else 422
        if error.reason_code
        in {
            "invalid_conditions_choice",
            "invalid_pronunciation_choice",
            "invalid_pronunciation_variant",
            "pronunciation_requires_english",
            "invalid_reader_level",
        }
        else 409
    )
    return HTTPException(code, detail={"reason_code": error.reason_code})


@router.post(
    "/analyze",
    response_model=TaskResponse | None,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_analysis(
    request: Request, application: CurrentApplication, book_id: str
) -> TaskResponse | RedirectResponse | None:
    """Verify LLM settings, optionally pin initial conditions, and enqueue book analysis."""
    application.settings.llm.ensure_ready()
    try:
        initial = await _parse_initial_analysis_input(request)
    except HTTPException as error:
        return input_error_response(request, error)
    has_initial_options = any(
        value is not None
        for value in (
            initial.reader_level,
            initial.english_help,
            initial.pronunciation_help_enabled,
            initial.pronunciation_variant,
            initial.seed_first_chapter,
        )
    )
    if has_initial_options:
        try:
            application.book_conditions.ensure_initial_for_analysis(
                book_id,
                reader_level=initial.reader_level,
                english_help=initial.english_help,
                pronunciation_help_enabled=initial.pronunciation_help_enabled,
                pronunciation_variant=initial.pronunciation_variant,
                seed_first_chapter=initial.seed_first_chapter,
                default_seed_first_chapter=application.settings.checks.seed_first_chapter,
                validate_conditions=application.book_runtime.settings_for,
            )
        except ConditionsError as error:
            return input_error_response(request, _conditions_http_error(error))
    task = await application.analysis.start(book_id)
    if wants_html(request):
        return RedirectResponse(
            url=f"/books/{book_id}", status_code=status.HTTP_303_SEE_OTHER
        )
    return None if task is None else TaskResponse.of_task(task)


@router.post(
    "/process",
    response_model=ProcessingStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_processing(
    request: Request, application: CurrentApplication, book_id: str
) -> ProcessingStartResponse | RedirectResponse:
    """Freeze the glossary and enqueue processing tasks for all chapters."""
    run = application.start_processing()(StartProcessingCommand(book_id=book_id))
    with application.unit_of_work.transaction() as tx:
        set_book_paused = getattr(tx.work, "set_book_paused", None)
        if callable(set_book_paused):
            set_book_paused(book_id, None, False)
    tasks = await application.processing.start(book_id)
    if wants_html(request):
        return RedirectResponse(
            url=f"/books/{book_id}", status_code=status.HTTP_303_SEE_OTHER
        )
    return ProcessingStartResponse.of_start(run, tasks)


@router.post("/pause")
def pause_book_queue(
    request: Request, application: CurrentApplication, book_id: str
):
    """Pause claiming pending tasks for the book without cancelling or deleting them."""
    if application.pause_book is None:
        raise HTTPException(status_code=503, detail="Пауза очереди недоступна.")
    try:
        result = application.pause_book(PauseBookCommand(book_id=book_id))
    except (PauseBookError, WorkStateError):
        return input_error_response(
            request, HTTPException(409, detail={"reason_code": "state_conflict"})
        )
    if wants_html(request):
        return RedirectResponse(
            url=f"/books/{book_id}", status_code=status.HTTP_303_SEE_OTHER
        )
    return {
        "book_id": result.book_id,
        "pause_requested": result.pause_requested,
        "is_paused": result.is_paused,
        "running_tasks": result.running_tasks,
        "pending_tasks": result.pending_tasks,
    }


@router.post("/resume")
async def resume_book_queue(
    request: Request, application: CurrentApplication, book_id: str
):
    """Clear the pause binding and kick-start missing stage tasks when needed."""
    if application.resume_book is None:
        raise HTTPException(status_code=503, detail="Возобновление очереди недоступно.")
    try:
        result = application.resume_book(ResumeBookCommand(book_id=book_id))
    except (PauseBookError, WorkStateError):
        return input_error_response(
            request, HTTPException(409, detail={"reason_code": "state_conflict"})
        )
    await _kickstart_resumed_stage(application, book_id)
    notify = getattr(application.worker_pool, "notify", None)
    if callable(notify):
        notify()
    if wants_html(request):
        return RedirectResponse(
            url=f"/books/{book_id}", status_code=status.HTTP_303_SEE_OTHER
        )
    return {
        "book_id": result.book_id,
        "pause_requested": result.pause_requested,
        "is_paused": result.is_paused,
        "running_tasks": result.running_tasks,
        "pending_tasks": result.pending_tasks,
    }


async def _kickstart_resumed_stage(
    application: CurrentApplication, book_id: str
) -> None:
    if not application.settings.llm.is_ready:
        return
    read_tx = getattr(
        application.unit_of_work,
        "read_transaction",
        application.unit_of_work.transaction,
    )
    with read_tx() as tx:
        book = tx.books.get(book_id)
        if book is None:
            return
        tasks = tx.tasks.list_for_book(book_id)
        run = tx.runs.active_for_book(book_id)
        stopped_roots = (
            any(
                getattr(root, "superseded_by", None) is None
                and root.status == "stopped"
                for root in tx.work.list_roots(book_id, run_id=run.id)
            )
            if run is not None
            else False
        )
    try:
        if book.status is BookStatus.ANALYZING:
            has_analysis_work = any(
                task.task_type is TaskType.ANALYZE_CHAPTER
                and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.FAILED)
                for task in tasks
            )
            if not has_analysis_work:
                await application.analysis.start(book_id)
        elif book.status is BookStatus.PROCESSING and run is not None and not stopped_roots:
            has_processing_work = any(
                task.task_type
                in (
                    TaskType.PROCESS_TRANSLATION_UNIT,
                    TaskType.CHAPTER_CORRECTION,
                    TaskType.REGENERATE_BLOCK,
                )
                and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.FAILED)
                for task in tasks
            )
            if not has_processing_work:
                await application.processing.start(book_id)
    except (ApplicationError, WorkStateError):
        return


@router.post("/recovery/{root_id}/restart", status_code=status.HTTP_202_ACCEPTED)
async def restart_recovery_root(
    request: Request,
    application: CurrentApplication,
    book_id: str,
    root_id: str,
):
    """Explicitly restart unfinished scopes of a translation recovery root."""
    read_tx = getattr(
        application.unit_of_work,
        "read_transaction",
        application.unit_of_work.transaction,
    )
    with read_tx() as tx:
        if tx.books.get(book_id) is None:
            return input_error_response(
                request, HTTPException(404, detail={"reason_code": "not_found"})
            )
    try:
        payload = await read_input(request, _RestartRecoveryInput)
        if payload.confirmed is not True:
            raise HTTPException(422, detail={"reason_code": "invalid_request"})
    except HTTPException as error:
        return input_error_response(request, error)
    try:
        restarted = application.restart_translation(
            RestartTranslationCommand(
                book_id=book_id,
                root_id=root_id,
                expected_generation=payload.expected_generation,
            )
        )
    except RestartTranslationConflict:
        return input_error_response(
            request, HTTPException(409, detail={"reason_code": "generation_conflict"})
        )
    except RestartTranslationError:
        with read_tx() as tx:
            book_exists = tx.books.get(book_id) is not None
            root = tx.work.get_root(root_id) if book_exists else None
            missing = not book_exists or root is None or root.book_id != book_id
        code = 404 if missing else 409
        reason = "not_found" if missing else "state_conflict"
        return input_error_response(
            request, HTTPException(code, detail={"reason_code": reason})
        )
    notify = getattr(application.worker_pool, "notify", None)
    if callable(notify):
        notify()
    if wants_html(request):
        return RedirectResponse(
            url=f"/books/{book_id}", status_code=status.HTTP_303_SEE_OTHER
        )
    return {
        "book_id": book_id,
        "root_id": root_id,
        "successor_root_id": restarted.successor_root_id,
        "task_id": restarted.task_id,
    }


@router.post("/reset", response_model=ResetProcessingResponse)
def reset_processing(
    http_request: Request,
    application: CurrentApplication,
    book_id: str,
    request: Annotated[ResetProcessingRequest, Form()],
) -> ResetProcessingResponse | RedirectResponse:
    """Delete processing results only after explicit user confirmation.

    The browser returns to the glossary screen: this is the purpose of resetting.
    """
    result = application.reset_processing(ResetProcessingCommand(book_id=book_id))
    if wants_html(http_request):
        return RedirectResponse(
            url=f"/books/{book_id}/glossary", status_code=status.HTTP_303_SEE_OTHER
        )
    return ResetProcessingResponse.of_result(result)


@router.post("/conditions", status_code=status.HTTP_201_CREATED)
async def choose_conditions(
    request: Request, application: CurrentApplication, book_id: str
):
    """Apply the same explicit conditions choice for JSON and HTML forms."""

    class Choice(BaseModel):
        action: Literal["continue_legacy", "reanalyze"]
        expected_revision: int = Field(ge=0)
        settings_revision: int = Field(ge=0)
        confirmed: bool
        english_help: bool = False
        reader_level: Literal["a2", "b1", "b2", "c1"] | None = None
        pronunciation_help_enabled: bool = False
        pronunciation_variant: Literal["uk", "us"] | None = None
        seed_first_chapter: bool | None = None

    try:
        is_json = request.headers.get("content-type", "").startswith("application/json")
        raw = await request.json() if is_json else dict(await request.form())
        if not is_json and isinstance(raw, dict):
            if raw.get("pronunciation_variant") == "":
                raw["pronunciation_variant"] = None
            if raw.get("action") == "reanalyze":
                raw.setdefault("seed_first_chapter", False)
        choice = Choice.model_validate(raw)
        if choice.confirmed is not True:
            if wants_html(request):
                return input_error_response(
                    request, HTTPException(422, detail={"reason_code": "invalid_request"})
                )
            raise HTTPException(422, detail="confirmation_required")
    except (ValidationError, ValueError):
        if wants_html(request):
            return input_error_response(
                request, HTTPException(422, detail={"reason_code": "invalid_request"})
            )
        raise HTTPException(422, detail="invalid_conditions_choice") from None
    try:
        conditions = ChooseBookConditions(
            application.book_conditions, application.book_runtime
        )(book_id, **choice.model_dump())
    except ConditionsError as error:
        return input_error_response(request, _conditions_http_error(error))
    if wants_html(request):
        return RedirectResponse(
            url=f"/books/{book_id}", status_code=status.HTTP_303_SEE_OTHER
        )
    return {
        "conditions_id": conditions.id,
        "revision": conditions.revision,
        "reproducibility": conditions.reproducibility,
    }


__all__ = ["router"]
