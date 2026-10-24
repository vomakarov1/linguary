"""Local Web UI: FastAPI application assembly (PRD § 21, ADR § 3).

The application sets up dependency composition once at startup, stores it in FastAPI
application state, and launches the background queue worker. The worker lives in the same process intentionally:
the MVP is a local single-user application, and an uploaded book should start processing
automatically without a second terminal session (PRD § 18.1). A separate CLI command
remains possible: the queue claims tasks atomically, and tasks considered interrupted at startup
are only those not executed by any active worker, so an extra worker does not interfere.

Application use case failures are translated to HTTP status codes here rather than in each route:
the error set is shared across all screens and is reused by downstream endpoint packages.

The server listens on the address from local configuration, default `127.0.0.1` (PRD § 4.1):
the application stores the provider key and full book text. It intentionally has no authentication,
so the only distinction between the application page and a foreign page opened in the same browser
is request origin: `Host`, `Origin`, and `Sec-Fetch-Site` are checked here by a single middleware across all endpoints.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress

from fastapi import FastAPI, Request, Response, status
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from linguary.adapters.config import ServerSettings
from linguary.application.book_overview import (
    BookNotFoundError,
    ProcessingPlanUnavailableError,
    ResultBlockNotFoundError,
)
from linguary.application.errors import ApplicationError
from linguary.application.ports.llm import LLMConfigurationError
from linguary.application.glossary_review import (
    GlossaryNotFoundError,
    GlossaryReviewError,
)
from linguary.application.use_cases.export_book import (
    ExportArtifactNotFoundError,
    ExportBookError,
    ExportBookNotFoundError,
)
from linguary.application.use_cases.import_book import ImportBookError
from linguary.application.use_cases.regenerate_block import (
    RegenerateBlockError,
    RegenerateBlockNotFoundError,
)
from linguary.application.use_cases.register_book import RegisterBookError
from linguary.application.use_cases.reset_processing import ResetProcessingError
from linguary.application.use_cases.retry_task import RetryTaskError, TaskNotFoundError
from linguary.composition import Application, build_application, run_worker
from linguary.domain.errors import (
    BookStatusTransitionError,
    DomainError,
    TaskTransitionError,
)
from linguary.web.dependencies import APPLICATION_STATE_ATTRIBUTE
from linguary.web.routes import books, export, glossary, processing, result, system
from linguary.web.schemas import ErrorResponse
from linguary.web.uploads import UploadRejectedError
from linguary.web.views import STATIC_DIRECTORY, render_error, wants_html

#: Application name in API documentation.
TITLE = "Linguary"

#: Methods that do not mutate state: only `Host` is checked for them. Cross-site
#: read is already blocked by the browser from foreign pages — response is invisible to them.
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

#: `Sec-Fetch-Site` values indicating the request came from the application page itself
#: or directly from the user — via bookmark navigation or the address bar.
_SAME_ORIGIN_FETCH_SITES = frozenset({"same-origin", "none"})

#: Use case and domain failures mapped to HTTP status codes (API-01).
#:
#: `409` — action impossible in current book status: glossary edit after processing start,
#: export with pending tasks, retry of non-recoverable task, start without configured LLM.
#: `404` — book or task not found. `400` — uploaded file rejected.
#: Order matters: subclasses must precede their base class.
_ERROR_STATUSES: tuple[tuple[type[Exception], int], ...] = (
    (BookNotFoundError, status.HTTP_404_NOT_FOUND),
    (ResultBlockNotFoundError, status.HTTP_404_NOT_FOUND),
    (GlossaryNotFoundError, status.HTTP_404_NOT_FOUND),
    (TaskNotFoundError, status.HTTP_404_NOT_FOUND),
    (RegenerateBlockNotFoundError, status.HTTP_404_NOT_FOUND),
    (ExportBookNotFoundError, status.HTTP_404_NOT_FOUND),
    (ExportArtifactNotFoundError, status.HTTP_404_NOT_FOUND),
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
    """Assembles FastAPI application of the local Web UI.

    `application` is provided in tests and end-to-end scenarios; without it, dependencies
    are assembled from local configuration at process startup. `run_background_worker`
    disables background worker when tasks are executed explicitly.
    """
    app = FastAPI(title=TITLE, lifespan=_lifespan(application, run_background_worker))
    _register_request_guard(app)
    _register_root_route(app)
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIRECTORY)),
        name="static",
    )
    app.include_router(books.router)
    app.include_router(glossary.router)
    app.include_router(processing.router)
    app.include_router(result.router)
    app.include_router(export.router)
    app.include_router(system.router)
    _register_error_handlers(app)
    return app


def _register_request_guard(app: FastAPI) -> None:
    """Rejects foreign host and foreign origin requests (PRD § 4.1).

    Two checks protect different entry points to the application.

    `Host` prevents DNS rebinding: an attacker's page reaches the server only
    under its own host, while the application responds only to hosts approved by
    the owner (`ServerSettings.allowed_hosts`). Without this check, the browser considers
    the attacker's page same-origin, exposing book reading and actions.

    `Origin` and `Sec-Fetch-Site` protect against state-mutating cross-site requests:
    headers are set by the browser itself and cannot be forged by a page, so a foreign page
    cannot upload books, reset processing, or consume provider balance.
    A request lacking both headers is allowed: this matches non-browser clients (curl, tests),
    while browsers always flag cross-site requests.

    Settings are retrieved from the assembled application on each request:
    middleware is constructed before `lifespan` builds dependencies, and at construction time
    the server address is not yet known.
    """

    @app.middleware("http")
    async def guard_request_origin(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rejection = _request_rejection(request)
        if rejection is not None:
            return rejection
        return await call_next(request)


def _request_rejection(request: Request) -> Response | None:
    """Response rejecting request, or `None` if accepted."""
    server = _server_settings(request)
    if not _host_allowed(request.headers.get("host", ""), server):
        return _rejection(
            request,
            status.HTTP_400_BAD_REQUEST,
            f"приложение отвечает только по адресу {server.base_url}",
        )
    if request.method in _SAFE_METHODS:
        return None
    if not _origin_allowed(request):
        return _rejection(
            request,
            status.HTTP_403_FORBIDDEN,
            "запрос отклонён: действие принимается только со страницы приложения",
        )
    return None


def _server_settings(request: Request) -> ServerSettings:
    """Server address from assembled application; default values before assembly."""
    application: Application | None = getattr(
        request.app.state, APPLICATION_STATE_ATTRIBUTE, None
    )
    return application.settings.server if application is not None else ServerSettings()


def _host_allowed(header: str, server: ServerSettings) -> bool:
    """Whether application responds to host name from `Host` header."""
    allowed = server.allowed_hosts
    if "*" in allowed:
        return True
    return _header_host(header) in {name.strip().lower() for name in allowed}


def _header_host(header: str) -> str:
    """Host name from `Host` header without port; IPv6 address without brackets."""
    value = header.strip().lower()
    if value.startswith("["):
        return value[1:].partition("]")[0]
    return value.partition(":")[0]


def _origin_allowed(request: Request) -> bool:
    """Whether state-mutating request originated from the application page itself."""
    origin = request.headers.get("origin")
    if origin is not None and origin.strip().lower() != _request_origin(request):
        return False
    fetch_site = request.headers.get("sec-fetch-site")
    return fetch_site is None or fetch_site.strip().lower() in _SAME_ORIGIN_FETCH_SITES


def _request_origin(request: Request) -> str:
    """Origin under which application is currently opened by user.

    Host name is taken from `Host` header — the one already verified against allowed hosts.
    This accepts both common local addresses, `127.0.0.1` and `localhost`, and rejects any third.
    """
    host = request.headers.get("host", "").strip().lower()
    return f"{request.url.scheme}://{host}"


def _rejection(request: Request, code: int, detail: str) -> Response:
    """Origin check rejection formatted identically to use case failures."""
    return _error_response(request, code, detail)


def _register_root_route(app: FastAPI) -> None:
    """Opens main screen at server root address."""

    @app.get("/", include_in_schema=False)
    async def redirect_to_books() -> RedirectResponse:
        return RedirectResponse("/books", status_code=status.HTTP_303_SEE_OTHER)


def _lifespan(
    application: Application | None, run_background_worker: bool
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Process lifecycle: dependency assembly, worker, and closing connections."""

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
            built.begin_close()
            try:
                if worker is not None:
                    worker.cancel()
                    with suppress(asyncio.CancelledError):
                        await worker
            finally:
                try:
                    await built.system_settings.aclose()
                finally:
                    if owns_application:
                        await built.aclose()

    return lifespan


def _register_error_handlers(app: FastAPI) -> None:
    """Translates use case and domain failures into responses for the user.

    Two handlers, one per base class: application errors and domain invariant violations.
    The specific code is looked up in the table, so a new endpoint acquires
    its behavior without explicit registration.

    Response format is chosen by the client: a browser submitting a form receives an error page
    with a back link, while a JSON client receives the familiar `{"detail": ...}`. Invalid forms
    (`422`) and unknown routes (`404`) are handled similarly — otherwise the browser would show raw JSON.
    """

    @app.exception_handler(ApplicationError)
    async def handle_application_error(
        request: Request, error: ApplicationError
    ) -> Response:
        return _error_response(request, _status_for(error), str(error))

    @app.exception_handler(DomainError)
    async def handle_domain_error(request: Request, error: DomainError) -> Response:
        return _error_response(request, _status_for(error), str(error))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, error: RequestValidationError
    ) -> Response:
        if not wants_html(request):
            return await request_validation_exception_handler(request, error)
        return render_error(
            request,
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            messages=_validation_messages(error),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        request: Request, error: StarletteHTTPException
    ) -> Response:
        if not wants_html(request):
            return await http_exception_handler(request, error)
        return render_error(
            request, status_code=error.status_code, messages=(str(error.detail),)
        )


def _error_response(request: Request, code: int, detail: str) -> Response:
    """Error response with status code and detail for user: page for browser, else JSON."""
    if wants_html(request):
        return render_error(request, status_code=code, messages=(detail,))
    return JSONResponse(
        status_code=code, content=ErrorResponse(detail=detail).model_dump()
    )


def _validation_messages(error: RequestValidationError) -> tuple[str, ...]:
    """User-friendly reasons why the form was rejected."""
    messages: list[str] = []
    for issue in error.errors():
        field = str(issue.get("loc", ("",))[-1])
        if issue.get("type") == "missing":
            messages.append(f"Не заполнено поле «{field}»")
            continue
        message = str(issue.get("msg", "")).removeprefix("Value error, ").strip()
        messages.append(message or f"Неверное значение поля «{field}»")
    return tuple(messages) or ("Форма заполнена неверно",)


def _status_for(error: Exception) -> int:
    """HTTP status code for use case rejection; unrecognized application error is `500`."""
    for error_type, code in _ERROR_STATUSES:
        if isinstance(error, error_type):
            return code
    return status.HTTP_500_INTERNAL_SERVER_ERROR


__all__ = ["TITLE", "create_app"]
