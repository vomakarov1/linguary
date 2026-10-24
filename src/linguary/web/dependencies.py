"""Routes access to assembled application.

Dependency composition is assembled once at process startup and lives in FastAPI
application state; routes obtain it from here and construct nothing themselves
(ADR § 3). The MVP trusts a single local user, but use cases know nothing about them:
there is neither a current user nor global book state here.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from linguary.composition import Application

#: Key for application in FastAPI state.
APPLICATION_STATE_ATTRIBUTE = "linguary_application"


def get_application(request: Request) -> Application:
    """Assembled application of current process."""
    application: Application | None = getattr(
        request.app.state, APPLICATION_STATE_ATTRIBUTE, None
    )
    if application is None:  # create_app builds the application; this cannot be reached bypassing it
        raise RuntimeError("состав зависимостей приложения не собран")
    return application


#: Ready annotation for routes: `application: CurrentApplication`.
CurrentApplication = Annotated[Application, Depends(get_application)]


__all__ = ["APPLICATION_STATE_ATTRIBUTE", "CurrentApplication", "get_application"]
