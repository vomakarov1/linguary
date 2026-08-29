"""Доступ routes к собранному приложению.

Состав зависимостей собирается один раз при старте процесса и живёт в состоянии
FastAPI-приложения; routes получают его отсюда и ничего не конструируют сами
(ADR § 3). MVP доверяет одному локальному пользователю, но сценарии о нём не знают:
здесь нет ни текущего пользователя, ни глобального состояния книги.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from linguary.composition import Application

#: Ключ приложения в состоянии FastAPI.
APPLICATION_STATE_ATTRIBUTE = "linguary_application"


def get_application(request: Request) -> Application:
    """Собранное приложение текущего процесса."""
    application: Application | None = getattr(
        request.app.state, APPLICATION_STATE_ATTRIBUTE, None
    )
    if application is None:  # приложение собирает `create_app`, минуя её сюда не попасть
        raise RuntimeError("состав зависимостей приложения не собран")
    return application


#: Готовая аннотация для routes: `application: CurrentApplication`.
CurrentApplication = Annotated[Application, Depends(get_application)]


__all__ = ["APPLICATION_STATE_ATTRIBUTE", "CurrentApplication", "get_application"]
