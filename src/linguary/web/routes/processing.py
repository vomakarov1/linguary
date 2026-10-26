"""Endpoints for starting processing and returning to glossary review (API-01)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Form, Request, status
from fastapi.responses import RedirectResponse

from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.application.use_cases.reset_processing import ResetProcessingCommand
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import (
    ProcessingStartResponse,
    ResetProcessingRequest,
    ResetProcessingResponse,
    TaskResponse,
)
from linguary.web.views import wants_html

router = APIRouter(prefix="/books/{book_id}", tags=["processing"])


@router.post(
    "/analyze",
    response_model=TaskResponse | None,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_analysis(
    request: Request, application: CurrentApplication, book_id: str
) -> TaskResponse | RedirectResponse | None:
    """Verify LLM settings and enqueue the first book analysis task."""
    application.settings.llm.ensure_ready()
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
    tasks = await application.processing.start(book_id)
    if wants_html(request):
        return RedirectResponse(
            url=f"/books/{book_id}", status_code=status.HTTP_303_SEE_OTHER
        )
    return ProcessingStartResponse.of_start(run, tasks)


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


__all__ = ["router"]


@router.post('/conditions', status_code=status.HTTP_201_CREATED)
async def choose_conditions(request: Request, application: CurrentApplication, book_id: str):
    """Apply the same explicit conditions choice for JSON and HTML forms."""
    from typing import Literal
    from pydantic import BaseModel, Field, ValidationError
    from fastapi import HTTPException
    from linguary.web.routes.quality import input_error_response
    from linguary.application.book_conditions import ConditionsError
    from linguary.application.use_cases.reset_processing import ChooseBookConditions

    class Choice(BaseModel):
        action: Literal['continue_legacy', 'reanalyze']
        expected_revision: int = Field(ge=0)
        settings_revision: int = Field(ge=0)
        confirmed: bool
        english_help: bool = False
        reader_level: Literal["a2", "b1", "b2"] | None = None

    try:
        raw = await request.json() if request.headers.get('content-type', '').startswith('application/json') else dict(await request.form())
        choice = Choice.model_validate(raw)
        if choice.confirmed is not True:
            if wants_html(request):
                return input_error_response(request, HTTPException(422, detail={'reason_code': 'invalid_request'}))
            raise HTTPException(422, detail='confirmation_required')
    except (ValidationError, ValueError):
        if wants_html(request):
            return input_error_response(request, HTTPException(422, detail={'reason_code': 'invalid_request'}))
        raise HTTPException(422, detail='invalid_conditions_choice') from None
    try:
        conditions = ChooseBookConditions(application.book_conditions, application.book_runtime)(book_id, **choice.model_dump())
    except ConditionsError as error:
        return input_error_response(request, HTTPException(404 if error.reason_code == 'book_not_found' else 409,
                            detail={'reason_code': error.reason_code}))
    if wants_html(request):
        return RedirectResponse(url=f'/books/{book_id}', status_code=status.HTTP_303_SEE_OTHER)
    return {'conditions_id': conditions.id, 'revision': conditions.revision, 'reproducibility': conditions.reproducibility}
