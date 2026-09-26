"""Локальный Web UI: сборка FastAPI-приложения (PRD § 21, ADR § 3).

Приложение поднимает состав зависимостей один раз при старте, кладёт его в состояние
FastAPI и запускает фоновый воркер очереди. Воркер живёт в том же процессе намеренно:
MVP — локальное однопользовательское приложение, и загруженная книга должна начать
обрабатываться сама, без второго запущенного терминала (PRD § 18.1). Отдельная команда
CLI остаётся возможной: очередь атомарно захватывает задачи, а прерванными при старте
считаются только задачи, которые не выполняет ни один живой исполнитель, поэтому лишний
исполнитель ей не мешает.

Отказы прикладных сценариев переводятся в коды HTTP здесь, а не в каждом route: набор
ошибок общий для всех экранов, и он же достаётся следующим пакетам endpoints.

Сервер слушает адрес из локальной конфигурации, по умолчанию `127.0.0.1` (PRD § 4.1):
приложение хранит ключ провайдера и полный текст книг. Аутентификации у него нет
намеренно, поэтому единственное, чем страница самого приложения отличается от чужой
страницы, открытой в том же браузере, — происхождение запроса: `Host`, `Origin`
и `Sec-Fetch-Site` проверяются здесь одной middleware сразу для всех endpoints.
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

#: Название приложения в документации API.
TITLE = "Linguary"

#: Методы, которые состояния не меняют: у них проверяется только `Host`. Межсайтовое
#: чтение браузер и без того не отдаёт чужой странице — ответ ей не виден.
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

#: Значения `Sec-Fetch-Site`, при которых запрос пришёл со страницы самого приложения
#: либо прямо от пользователя — переходом по закладке или из адресной строки.
_SAME_ORIGIN_FETCH_SITES = frozenset({"same-origin", "none"})

#: Отказы сценариев и домена в кодах HTTP (API-01).
#:
#: `409` — действие невозможно в текущем состоянии книги: правка глоссария после старта
#: обработки, экспорт с незавершёнными задачами, повтор невосстановимой задачи, запуск
#: без настроенной LLM. `404` — книги или задачи нет. `400` — не принят присланный файл.
#: Порядок значим: наследник указывается раньше своего базового класса.
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
    """Собирает FastAPI-приложение локального Web UI.

    `application` задаётся в тестах и в сквозном сценарии; без него состав зависимостей
    собирается по локальной конфигурации при старте процесса. `run_background_worker`
    выключает фоновый воркер там, где задачи выполняются явно.
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
    """Отклоняет обращения по чужому имени и запросы чужого происхождения (PRD § 4.1).

    Проверки две, и закрывают они разные пути к приложению.

    `Host` отвечает за перепривязку DNS: страница атакующего доходит до сервера только
    под своим именем, а приложение отвечает лишь по тем именам, по которым его открывает
    владелец (`ServerSettings.allowed_hosts`). Без этой проверки браузер считает страницу
    атакующего тем же источником, и ей становятся доступны и чтение книг, и действия.

    `Origin` и `Sec-Fetch-Site` отвечают за межсайтовый запрос, меняющий состояние:
    заголовки проставляет сам браузер, страница их не подделает, поэтому чужая страница
    не выполнит ни загрузку книги, ни сброс обработки, ни трату ключа провайдера.
    Запрос, у которого нет обоих заголовков, проходит: так приходит не-браузерный
    клиент (curl, тест), а браузер межсайтовый запрос помечает всегда.

    Настройки берутся из собранного приложения на каждом запросе: набор middleware
    строится раньше, чем `lifespan` собирает состав зависимостей, и на момент сборки
    адрес сервера ещё неизвестен.
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
    """Ответ, которым запрос отклоняется, либо `None`, если он принимается."""
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
    """Адрес сервера из собранного приложения; до его сборки — значения по умолчанию."""
    application: Application | None = getattr(
        request.app.state, APPLICATION_STATE_ATTRIBUTE, None
    )
    return application.settings.server if application is not None else ServerSettings()


def _host_allowed(header: str, server: ServerSettings) -> bool:
    """Отвечает ли приложение по имени из заголовка `Host`."""
    allowed = server.allowed_hosts
    if "*" in allowed:
        return True
    return _header_host(header) in {name.strip().lower() for name in allowed}


def _header_host(header: str) -> str:
    """Имя из заголовка `Host` без порта; адрес IPv6 — без скобок."""
    value = header.strip().lower()
    if value.startswith("["):
        return value[1:].partition("]")[0]
    return value.partition(":")[0]


def _origin_allowed(request: Request) -> bool:
    """Пришёл ли меняющий состояние запрос со страницы самого приложения."""
    origin = request.headers.get("origin")
    if origin is not None and origin.strip().lower() != _request_origin(request):
        return False
    fetch_site = request.headers.get("sec-fetch-site")
    return fetch_site is None or fetch_site.strip().lower() in _SAME_ORIGIN_FETCH_SITES


def _request_origin(request: Request) -> str:
    """Происхождение, которым приложение открыто у пользователя прямо сейчас.

    Имя берётся из заголовка `Host` — того самого, который уже сверен со списком имён
    приложения. Так принимаются оба обычных адреса локального запуска, `127.0.0.1`
    и `localhost`, и не принимается никакой третий.
    """
    host = request.headers.get("host", "").strip().lower()
    return f"{request.url.scheme}://{host}"


def _rejection(request: Request, code: int, detail: str) -> Response:
    """Отказ проверки происхождения в том же виде, что и отказы сценариев."""
    return _error_response(request, code, detail)


def _register_root_route(app: FastAPI) -> None:
    """Открывает главный экран по корневому адресу сервера."""

    @app.get("/", include_in_schema=False)
    async def redirect_to_books() -> RedirectResponse:
        return RedirectResponse("/books", status_code=status.HTTP_303_SEE_OTHER)


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

    Вид ответа выбирает клиент: браузер, отправивший форму, получает страницу ошибки
    со ссылкой назад, а клиент JSON — прежний `{"detail": ...}`. Так же обрабатываются
    неверно заполненная форма (`422`) и неизвестный адрес (`404`) — иначе браузер
    показал бы пользователю сырой JSON.
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
    """Отказ с кодом и текстом для пользователя: страница для браузера, иначе JSON."""
    if wants_html(request):
        return render_error(request, status_code=code, messages=(detail,))
    return JSONResponse(
        status_code=code, content=ErrorResponse(detail=detail).model_dump()
    )


def _validation_messages(error: RequestValidationError) -> tuple[str, ...]:
    """Понятные пользователю причины, по которым форма не принята."""
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
    """Код HTTP для отказа сценария; незнакомая ошибка приложения — `500`."""
    for error_type, code in _ERROR_STATUSES:
        if isinstance(error, error_type):
            return code
    return status.HTTP_500_INTERNAL_SERVER_ERROR


__all__ = ["TITLE", "create_app"]
