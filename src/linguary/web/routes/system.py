"""Endpoint состояния конфигурации и окружения (API-01, PRD § 21.6).

Экран отвечает на один вопрос: готово ли приложение работать. Настроен ли провайдер LLM
и какая модель активна, доступна ли внешняя проверка EPUB, где лежат данные и по какому
адресу открыт сам интерфейс. Ключ провайдера в ответ не попадает — только признак того,
что он задан (PRD § 23).
"""

from __future__ import annotations

from fastapi import APIRouter

from linguary.web import labels
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import (
    EpubCheckConfigResponse,
    LLMConfigResponse,
    ServerConfigResponse,
    StorageConfigResponse,
    SystemConfigResponse,
)

router = APIRouter(tags=["system"])


@router.get("/system/config", response_model=SystemConfigResponse)
def get_system_config(application: CurrentApplication) -> SystemConfigResponse:
    """Готовность приложения к работе (PRD § 21.6)."""
    settings = application.settings
    server = settings.server
    return SystemConfigResponse(
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


__all__ = ["router"]
