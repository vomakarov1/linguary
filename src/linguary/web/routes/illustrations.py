"""HTTP routes for optional book illustration preparation (`API-01`, `API-02`, `API-04`)."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse

from linguary.domain.illustrations import (
    IllustrationConflictError,
    IllustrationError,
    IllustrationLimitError,
    IllustrationNotFoundError,
    IllustrationReasonCode,
    IllustrationValidationError,
)
from linguary.web.dependencies import CurrentApplication
from linguary.web.illustration_schemas import (
    IllustrationRegistryResponse,
    IllustrationViewResponse,
    ParsedCommandInput,
    ParsedSettingsInput,
    parse_api01_query,
    parse_api04_query,
    parse_command_form,
    parse_command_json,
    parse_settings_form,
    parse_settings_json,
)
from linguary.web.labels import ILLUSTRATION_REASON_LABELS
from linguary.web.views import render, render_error, wants_html

router = APIRouter(tags=["illustrations"])

_FORM_MEDIA_TYPES = frozenset({"application/x-www-form-urlencoded", "multipart/form-data"})


def _status_for_error(error: IllustrationError) -> int:
    if isinstance(error, IllustrationNotFoundError):
        return status.HTTP_404_NOT_FOUND
    if isinstance(error, (IllustrationConflictError, IllustrationLimitError)):
        return status.HTTP_409_CONFLICT
    if isinstance(error, IllustrationValidationError):
        return status.HTTP_422_UNPROCESSABLE_CONTENT
    return status.HTTP_409_CONFLICT


def _error_response(
    request: Request,
    error: IllustrationError,
    *,
    json_only: bool = False,
) -> Response:
    status_code = _status_for_error(error)
    code = error.reason_code
    if not json_only and wants_html(request):
        message = ILLUSTRATION_REASON_LABELS.get(code, "Не удалось выполнить действие")
        return render_error(request, status_code=status_code, messages=(message,))
    return JSONResponse(status_code=status_code, content={"detail": {"reason_code": code}})


async def _parse_settings_request(request: Request) -> ParsedSettingsInput:
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type == "application/json":
        try:
            body = await request.json()
        except Exception:
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "некорректный JSON в теле запроса",
            ) from None
        return parse_settings_json(body)
    if content_type in _FORM_MEDIA_TYPES:
        try:
            form = await request.form()
        except Exception:
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "некорректные данные формы",
            ) from None
        return parse_settings_form(list(form.multi_items()))
    raise IllustrationValidationError(
        IllustrationReasonCode.INVALID_REQUEST,
        "неподдерживаемый Content-Type",
    )


@router.get("/books/{book_id}/illustrations")
def get_illustrations(
    request: Request,
    application: CurrentApplication,
    book_id: str,
) -> Response:
    """`API-01`: Read HTML or JSON illustration state, selected chapter, and paginated registry."""
    try:
        params = parse_api01_query(list(request.query_params.multi_items()))
        view = application.illustrations.get(
            book_id,
            offset=params.offset,
            limit=params.limit,
            section_id=params.section_id,
            generation_id=params.generation_id,
        )
        response_dto = IllustrationViewResponse.of_view(view)
        if wants_html(request):
            registry_dto: IllustrationRegistryResponse | None = None
            if view.viewed_generation_id is not None:
                reg_view = application.illustrations.get_registry(
                    book_id,
                    generation_id=view.viewed_generation_id,
                    kind=params.registry_kind,
                    offset=params.registry_offset,
                    limit=params.registry_limit,
                )
                registry_dto = IllustrationRegistryResponse.of_view(reg_view)
            return render(
                request,
                "books/illustrations.html",
                data=response_dto,
                registry=registry_dto,
                requested_generation_id=params.generation_id,
                registry_kind=params.registry_kind.value if params.registry_kind is not None else None,
                book_id=book_id,
                active_tab="illustrations",
                page="illustrations",
            )
        return JSONResponse(status_code=status.HTTP_200_OK, content=response_dto.model_dump(mode="json"))
    except IllustrationError as error:
        return _error_response(request, error)


@router.post("/books/{book_id}/illustrations/settings")
async def save_illustration_settings(
    request: Request,
    application: CurrentApplication,
    book_id: str,
) -> Response:
    """`API-02`: Persist book illustration settings without enqueueing paid preparation."""
    try:
        parsed = await _parse_settings_request(request)
        application.illustrations.save_settings(
            book_id,
            expected_revision=parsed.expected_revision,
            settings=parsed.settings,
        )
        if wants_html(request):
            return RedirectResponse(
                url=parsed.return_params.redirect_location(book_id),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        view = application.illustrations.get(book_id)
        response_dto = IllustrationViewResponse.of_view(view)
        return JSONResponse(status_code=status.HTTP_200_OK, content=response_dto.model_dump(mode="json"))
    except IllustrationError as error:
        return _error_response(request, error)


async def _parse_command_request(
    request: Request,
    *,
    allow_force_new: bool,
    require_generation: bool,
) -> ParsedCommandInput:
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type == "application/json":
        try:
            body = await request.json()
        except Exception:
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "некорректный JSON в теле запроса",
            ) from None
        return parse_command_json(
            body, allow_force_new=allow_force_new, require_generation=require_generation
        )
    if content_type in _FORM_MEDIA_TYPES:
        try:
            form = await request.form()
        except Exception:
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "некорректные данные формы",
            ) from None
        return parse_command_form(
            list(form.multi_items()),
            allow_force_new=allow_force_new,
            require_generation=require_generation,
        )
    raise IllustrationValidationError(
        IllustrationReasonCode.INVALID_REQUEST,
        "неподдерживаемый Content-Type",
    )


@router.post("/books/{book_id}/illustrations/prepare")
async def prepare_illustrations(
    request: Request,
    application: CurrentApplication,
    book_id: str,
) -> Response:
    """`API-03`: Explicitly start or expand optional book illustration preparation."""
    try:
        parsed = await _parse_command_request(
            request, allow_force_new=True, require_generation=False
        )
        view = await application.illustrations.prepare(
            book_id,
            expected_revision=parsed.expected_revision,
            expected_generation_id=parsed.expected_generation_id,
            section_id=parsed.section_id,
            force_new_generation=parsed.force_new_generation,
        )
        if wants_html(request):
            return RedirectResponse(
                url=parsed.return_params.redirect_location(book_id),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        response_dto = IllustrationViewResponse.of_view(view)
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=response_dto.model_dump(mode="json"),
        )
    except IllustrationError as error:
        return _error_response(request, error)


@router.post("/books/{book_id}/illustrations/retry")
async def retry_illustrations(
    request: Request,
    application: CurrentApplication,
    book_id: str,
) -> Response:
    """`API-03`: Retry only retryable failed scopes of the active illustration generation."""
    try:
        parsed = await _parse_command_request(
            request, allow_force_new=False, require_generation=True
        )
        assert parsed.expected_generation_id is not None
        view = await application.illustrations.retry(
            book_id,
            expected_revision=parsed.expected_revision,
            expected_generation_id=parsed.expected_generation_id,
            section_id=parsed.section_id,
        )
        if wants_html(request):
            return RedirectResponse(
                url=parsed.return_params.redirect_location(book_id),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        response_dto = IllustrationViewResponse.of_view(view)
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=response_dto.model_dump(mode="json"),
        )
    except IllustrationError as error:
        return _error_response(request, error)


@router.get("/books/{book_id}/illustrations/registry")
def get_illustration_registry(
    request: Request,
    application: CurrentApplication,
    book_id: str,
) -> Response:
    """`API-04`: JSON-only read-only endpoint for paginated book illustration registry."""
    try:
        params = parse_api04_query(list(request.query_params.multi_items()))
        reg_view = application.illustrations.get_registry(
            book_id,
            generation_id=params.generation_id,
            entity_id=params.entity_id,
            kind=params.kind,
            offset=params.offset,
            limit=params.limit,
        )
        response_dto = IllustrationRegistryResponse.of_view(reg_view)
        return JSONResponse(status_code=status.HTTP_200_OK, content=response_dto.model_dump(mode="json"))
    except IllustrationError as error:
        return _error_response(request, error, json_only=True)


__all__ = ["router"]
