"""Локальный Web UI: FastAPI-приложение, endpoints и внешние DTO (PRD § 21).

Публичный вход слоя — `create_app`. Состав зависимостей слой не собирает: он приходит
из `linguary.composition`, поэтому Web UI, CLI и сквозной тест работают с одним и тем же
приложением (ADR § 3).
"""

from linguary.web.app import create_app

__all__ = ["create_app"]
