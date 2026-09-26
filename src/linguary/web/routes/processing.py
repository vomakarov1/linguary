"""Endpoints запуска обработки и возврата к проверке глоссария (API-01)."""

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
    """Проверяет LLM-настройку и ставит первую задачу анализа книги."""
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
    """Замораживает глоссарий и ставит задачи обработки всех глав."""
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
    """Удаляет результаты только после явного подтверждения пользователя.

    Браузер возвращается на экран глоссария: ради него сброс и выполняется.
    """
    result = application.reset_processing(ResetProcessingCommand(book_id=book_id))
    if wants_html(http_request):
        return RedirectResponse(
            url=f"/books/{book_id}/glossary", status_code=status.HTTP_303_SEE_OTHER
        )
    return ResetProcessingResponse.of_result(result)


__all__ = ["router"]
