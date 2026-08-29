"""Локальный Web UI: сборка FastAPI-приложения (PRD § 21, ADR § 3).

Приложение поднимает состав зависимостей один раз при старте, кладёт его в состояние
FastAPI и запускает фоновый воркер очереди. Воркер живёт в том же процессе намеренно:
MVP — локальное однопользовательское приложение, и загруженная книга должна начать
обрабатываться сама, без второго запущенного терминала (PRD § 18.1). Отдельная команда
CLI остаётся возможной: очередь атомарно захватывает задачи, и лишний исполнитель ей
не мешает.

Отказы прикладных сценариев переводятся в коды HTTP здесь, а не в каждом route: набор
ошибок общий для всех экранов, и он же достаётся следующим пакетам endpoints.

Сервер слушает адрес из локальной конфигурации, по умолчанию `127.0.0.1` (PRD § 4.1):
приложение хранит ключ провайдера и полный текст книг.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from linguary.application.book_overview import (
    BookNotFoundError,
    ProcessingPlanUnavailableError,
)
from linguary.application.errors import ApplicationError
from linguary.application.ports.llm import LLMConfigurationError
from linguary.application.use_cases.approve_glossary import GlossaryReviewError
from linguary.application.use_cases.export_book import ExportBookError
from linguary.application.use_cases.import_book import ImportBookError
from linguary.application.use_cases.regenerate_block import RegenerateBlockError
from linguary.application.use_cases.register_book import RegisterBookError
from linguary.application.use_cases.reset_processing import ResetProcessingError
from linguary.application.use_cases.retry_task import RetryTaskError
from linguary.composition import Application, build_application, run_worker
from linguary.domain.errors import (
    BookStatusTransitionError,
    DomainError,
    TaskTransitionError,
)
from linguary.web.dependencies import APPLICATION_STATE_ATTRIBUTE
from linguary.web.routes import books, system
from linguary.web.schemas import ErrorResponse
from linguary.web.uploads import UploadRejectedError

#: Название приложения в документации API.
TITLE = "Linguary"

#: Отказы сценариев и домена в кодах HTTP (API-01).
#:
#: `409` — действие невозможно в текущем состоянии книги: правка глоссария после старта
#: обработки, экспорт с незавершёнными задачами, повтор невосстановимой задачи, запуск
#: без настроенной LLM. `404` — книги или задачи нет. `400` — не принят присланный файл.
#: Порядок значим: наследник указывается раньше своего базового класса.
_ERROR_STATUSES: tuple[tuple[type[Exception], int], ...] = (
    (BookNotFoundError, status.HTTP_404_NOT_FOUND),
    (UploadRejectedError, status.HTTP_400_BAD_REQUEST),
    (RegisterBookError, status.HTTP_400_BAD_REQUEST),
    (ProcessingPlanUnavailableError, status.HTTP_409_CONFLICT),
    (BookStatusTransitionError, status.HTTP_409_CONFLICT),
    (TaskTransitionError, status.HTTP_409_CONFLICT),
    (LLMConfigurationError, status.HTTP_409_CONFLICT),
    (GlossaryReviewError, status.HTTP_409_CONFLICT),
    (RegenerateBlockError, status.HTTP_409_CONFLICT),
    (ResetProcessingError, status.HTTP_404_NOT_FOUND),
    (RetryTaskError, status.HTTP_409_CONFLICT),
    (ExportBookError, status.HTTP_409_CONFLICT),
    (ImportBookError, status.HTTP_409_CONFLICT),
)


def create_app(
    *,
    application: Application | None = None,
    run_background_worker: bool = True,
) -> FastAPI:
    """Собирает FastAPI-приложение локального Web UI.

    `application` задаётся в тестах и в сквозном сценарии; без него состав зависимостей
    собирается по локальной конфигурации при старте процесса. `run_background_worker`
    выключает фоновый воркер там, где задачи выполняются явно.
    """
    app = FastAPI(title=TITLE, lifespan=_lifespan(application, run_background_worker))
    app.include_router(books.router)
    app.include_router(system.router)
    _register_error_handlers(app)
    return app


def _lifespan(
    application: Application | None, run_background_worker: bool
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Жизненный цикл процесса: сборка зависимостей, воркер и закрытие соединений."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        built = application or build_application()
        owns_application = application is None
        setattr(app.state, APPLICATION_STATE_ATTRIBUTE, built)
        worker = (
            asyncio.create_task(run_worker(built)) if run_background_worker else None
        )
        try:
            yield
        finally:
            if worker is not None:
                built.worker_pool.request_stop()
                worker.cancel()
                with suppress(asyncio.CancelledError):
                    await worker
            if owns_application:
                await built.aclose()

    return lifespan


def _register_error_handlers(app: FastAPI) -> None:
    """Переводит отказы сценариев и домена в ответы для пользователя.

    Обработчиков два, по одному на базовый класс: прикладные ошибки и нарушения доменных
    инвариантов. Конкретный код выбирается по таблице, поэтому новый endpoint получает
    её поведение, ничего не регистрируя.
    """

    @app.exception_handler(ApplicationError)
    async def handle_application_error(
        request: Request, error: ApplicationError
    ) -> JSONResponse:
        return _error_response(error)

    @app.exception_handler(DomainError)
    async def handle_domain_error(
        request: Request, error: DomainError
    ) -> JSONResponse:
        return _error_response(error)


def _error_response(error: Exception) -> JSONResponse:
    """Ответ на отказ сценария: код по типу ошибки и её текст для пользователя."""
    return JSONResponse(
        status_code=_status_for(error),
        content=ErrorResponse(detail=str(error)).model_dump(),
    )


def _status_for(error: Exception) -> int:
    """Код HTTP для отказа сценария; незнакомая ошибка приложения — `500`."""
    for error_type, code in _ERROR_STATUSES:
        if isinstance(error, error_type):
            return code
    return status.HTTP_500_INTERNAL_SERVER_ERROR


__all__ = ["TITLE", "create_app"]
