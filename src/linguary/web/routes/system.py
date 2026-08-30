"""Endpoint состояния конфигурации и окружения (API-01, PRD § 21.6).

Экран отвечает на один вопрос: готово ли приложение работать. Настроен ли провайдер LLM
и какая модель активна, доступна ли внешняя проверка EPUB, где лежат данные и по какому
адресу открыт сам интерфейс. Ключ провайдера в ответ не попадает — только признак того,
что он задан (PRD § 23).
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from linguary.web import labels
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import (
    EpubCheckConfigResponse,
    LLMConfigResponse,
    ServerConfigResponse,
    StorageConfigResponse,
    SystemConfigResponse,
)
from linguary.web.views import render, wants_html

router = APIRouter(tags=["system"])


@router.get("/system/config", response_model=SystemConfigResponse)
def get_system_config(
    request: Request, application: CurrentApplication
) -> SystemConfigResponse | object:
    """Готовность приложения к работе (PRD § 21.6)."""
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
    if wants_html(request):
        return render(request, "system/config.html", page="system", data=response)
    return response


__all__ = ["router"]
