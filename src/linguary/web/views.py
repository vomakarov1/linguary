"""HTML-представления локального Web UI.

JSON остаётся основным контрактом API-01, а браузерные экраны выбираются через
`Accept: text/html`. Так маршруты используют те же прикладные сценарии и DTO, но не
ломают существующих клиентов JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

PACKAGE_ROOT = Path(__file__).resolve().parent
TEMPLATE_DIRECTORY = PACKAGE_ROOT / "templates"
STATIC_DIRECTORY = PACKAGE_ROOT / "static"

templates = Jinja2Templates(directory=str(TEMPLATE_DIRECTORY))


def wants_html(request: Request) -> bool:
    """Просит ли клиент HTML, а не JSON."""
    accept = request.headers.get("accept", "")
    return "text/html" in accept.lower()


def render(request: Request, template: str, **context: Any) -> HTMLResponse:
    """Рендерит HTML-шаблон с обязательным `request`."""
    return templates.TemplateResponse(
        request=request, name=template, context=context
    )


__all__ = [
    "STATIC_DIRECTORY",
    "TEMPLATE_DIRECTORY",
    "render",
    "templates",
    "wants_html",
]
