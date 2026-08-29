"""Endpoints запуска обработки и возврата к проверке глоссария (API-01)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Form, status

from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.application.use_cases.reset_processing import ResetProcessingCommand
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import (
    ProcessingStartResponse,
    ResetProcessingRequest,
    ResetProcessingResponse,
)

router = APIRouter(prefix="/books/{book_id}", tags=["processing"])


@router.post(
    "/process",
    response_model=ProcessingStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_processing(
    application: CurrentApplication, book_id: str
) -> ProcessingStartResponse:
    """Замораживает глоссарий и ставит задачи обработки всех глав."""
    run = application.start_processing()(StartProcessingCommand(book_id=book_id))
    tasks = await application.processing.start(book_id)
    return ProcessingStartResponse.of_start(run, tasks)


@router.post("/reset", response_model=ResetProcessingResponse)
def reset_processing(
    application: CurrentApplication,
    book_id: str,
    request: Annotated[ResetProcessingRequest, Form()],
) -> ResetProcessingResponse:
    """Удаляет результаты только после явного подтверждения пользователя."""
    result = application.reset_processing(ResetProcessingCommand(book_id=book_id))
    return ResetProcessingResponse.of_result(result)


__all__ = ["router"]
