"""Endpoint состояния конфигурации и окружения (API-01, PRD § 21.6).

Экран отвечает на один вопрос: готово ли приложение работать. Настроен ли провайдер LLM
и какая модель активна, доступна ли внешняя проверка EPUB, где лежат данные и по какому
адресу открыт сам интерфейс. Ключ провайдера в ответ не попадает — только признак того,
что он задан (PRD § 23).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

from linguary.adapters.db.system_settings import SettingsRevisionConflict
from linguary.composition.system_settings import SettingsClosingError

from linguary.web import labels
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import (
    EpubCheckConfigResponse,
    LLMConfigResponse,
    ServerConfigResponse,
    StorageConfigResponse,
    SystemConfigResponse,
    WorkerConfigResponse,
    SettingsErrorResponse,
    SafeSettingsFormValues,
)
from linguary.web.views import render, wants_html
from linguary.web.system_settings_input import (
    SettingsInputError, read_settings_update, read_settings_reset, safe_form_values, sanitize_form_values,
)

router = APIRouter(tags=["system"])


@router.get("/system/config", response_model=SystemConfigResponse)
async def get_system_config(
    request: Request, application: CurrentApplication
) -> SystemConfigResponse | object:
    """Единый актуальный снимок настроек и редакции."""
    response = _config_response(application)
    if wants_html(request):
        confirmation = {
            "saved": labels.SETTINGS_SAVED,
            "reset": labels.SETTINGS_RESET,
        }.get(request.query_params.get("result"))
        return _render_config(
            request, application, data=response,
            form_values=safe_form_values(application.settings, expected_revision=response.revision),
            confirmation=confirmation,
        )
    return response


def _config_response(application) -> SystemConfigResponse:
    settings = application.settings
    server = settings.server
    response = SystemConfigResponse(
        llm=LLMConfigResponse.of_settings(settings),
        epubcheck=EpubCheckConfigResponse.of_availability(
            application.epubcheck.availability()
        ),
        storage=StorageConfigResponse(
            data_root=str(application.paths.data_root),
            database_path=str(application.paths.database_path),
            books_root=str(application.paths.books_root),
        ),
        worker=WorkerConfigResponse(concurrency=settings.worker.concurrency),
        revision=application.runtime_settings.revision,
        server=ServerConfigResponse(
            base_url=server.base_url,
            host=server.host,
            port=server.port,
            is_local_only=server.is_local_only,
            warning=(
                None if server.is_local_only else labels.NON_LOCAL_SERVER_WARNING
            ),
        ),
    )
    return response


def _render_config(
    request, application, *, form_values: SafeSettingsFormValues,
    data: SystemConfigResponse | None = None,
    error: SettingsErrorResponse | None = None,
    status_code: int = 200,
    confirmation: str | None = None,
):
    response = render(
        request, "system/config.html", page="system",
        data=data if data is not None else _config_response(application),
        form_values=sanitize_form_values(form_values, application.settings),
        field_errors={} if error is None else error.field_errors,
        settings_error=error,
        confirmation=confirmation,
        is_conflict=status_code == 409,
        can_submit=form_values.expected_revision is not None and status_code not in (409, 503),
        config_url="/system/config",
    )
    response.status_code = status_code
    return response


@router.post("/system/config", response_model=SystemConfigResponse)
async def save_system_config(request: Request, application: CurrentApplication):
    return await _change_config(request, application, reset=False)


@router.post("/system/config/reset", response_model=SystemConfigResponse)
async def reset_system_config(request: Request, application: CurrentApplication):
    return await _change_config(request, application, reset=True)


async def _change_config(request: Request, application, *, reset: bool):
    current_settings = application.settings
    form_values = safe_form_values(current_settings)
    try:
        if reset:
            revision = await read_settings_reset(request, current_settings=current_settings)
            form_values = safe_form_values(current_settings, expected_revision=revision)
            await application.system_settings.reset(expected_revision=revision)
        else:
            update, revision = await read_settings_update(request, current_settings=current_settings)
            form_values = safe_form_values(current_settings, update=update, expected_revision=revision)
            await application.system_settings.save(update, expected_revision=revision)
    except SettingsInputError as error:
        code, message, status_code = error.code, error.message, error.status_code
        field_errors = error.field_errors
        form_values = error.form_values
    except SettingsRevisionConflict:
        code, message, status_code = "revision_conflict", labels.SETTINGS_CONFLICT, 409
        field_errors = {}
    except SettingsClosingError:
        code, message, status_code = "settings_closing", labels.SETTINGS_CLOSING, 503
        field_errors = {}
    except Exception:
        code, message, status_code = "settings_not_applied", labels.SETTINGS_NOT_APPLIED, 500
        field_errors = {}
    else:
        if wants_html(request):
            marker = "reset" if reset else "saved"
            return RedirectResponse(f"/system/config?result={marker}", status_code=303)
        return _config_response(application)
    error_response = SettingsErrorResponse(code=code, message=message, field_errors=field_errors)
    if wants_html(request):
        return _render_config(
            request, application, form_values=form_values,
            error=error_response, status_code=status_code,
        )
    return JSONResponse(status_code=status_code, content=error_response.model_dump())


__all__ = ["router"]
