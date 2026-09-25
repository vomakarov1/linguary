"""HTML-представления локального Web UI.

JSON остаётся основным контрактом API-01, а браузерные экраны выбираются через
`Accept: text/html`. Так маршруты используют те же прикладные сценарии и DTO, но не
ломают существующих клиентов JSON.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from linguary.web import labels

PACKAGE_ROOT = Path(__file__).resolve().parent
TEMPLATE_DIRECTORY = PACKAGE_ROOT / "templates"
STATIC_DIRECTORY = PACKAGE_ROOT / "static"

templates = Jinja2Templates(directory=str(TEMPLATE_DIRECTORY))

#: Предупреждение о передаче текста книги внешнему провайдеру нужно экранам, которые
#: сами по себе ничего о конфигурации LLM не показывают (PRD § 4.1, § 21.2), поэтому оно
#: доступно любому шаблону, а не приходит отдельным полем каждого ответа.
templates.env.globals["external_llm_warning"] = labels.EXTERNAL_LLM_WARNING


def wants_html(request: Request) -> bool:
    """Просит ли клиент HTML, а не JSON."""
    accept = request.headers.get("accept", "")
    return "text/html" in accept.lower()


def render(request: Request, template: str, **context: Any) -> HTMLResponse:
    """Рендерит HTML-шаблон с обязательным `request`."""
    return templates.TemplateResponse(
        request=request, name=template, context=context
    )


def render_error(
    request: Request, *, status_code: int, messages: Sequence[str]
) -> HTMLResponse:
    """Страница ошибки для браузера: причина отказа и ссылка назад.

    Код ответа остаётся тем же, что получил бы клиент JSON: меняется только вид отказа.
    """
    return templates.TemplateResponse(
        request=request,
        name="error.html",
        context={
            "page": None,
            "title": labels.error_page_title(status_code),
            "status_code": status_code,
            "messages": tuple(_sentence(message) for message in messages),
            "back_url": back_url(request),
        },
        status_code=status_code,
    )


def _sentence(message: str) -> str:
    """Текст отказа как законченное предложение: с заглавной буквы и с точкой.

    Сообщения сценариев написаны для журнала и JSON — со строчной буквы и без точки,
    а на странице они читаются отдельным абзацем.
    """
    text = message.strip()
    if not text:
        return text
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else f"{text}."


#: Страница книги, к которой относится адрес запроса.
_BOOK_PATH = re.compile(r"^/books/(?!upload(?:/|$))([^/]+)")


def back_url(request: Request) -> str:
    """Куда вернуть пользователя со страницы ошибки.

    Предпочтительна страница, с которой отправлена форма: браузер называет её в
    `Referer`. Чужой адрес не принимается — ссылка ведёт только внутрь приложения.
    Без `Referer` действие над книгой возвращает на её карточку, остальное — к списку.
    """
    referer = urlsplit(request.headers.get("referer", ""))
    host = request.headers.get("host", "").strip().lower()
    if (
        referer.netloc.lower() == host
        and referer.path.startswith("/")
        and not referer.path.startswith(("//", "/\\"))
    ):
        return referer.path + (f"?{referer.query}" if referer.query else "")
    match = _BOOK_PATH.match(request.url.path)
    if match is not None and request.method not in {"GET", "HEAD"}:
        return f"/books/{match.group(1)}"
    return "/books"


__all__ = [
    "STATIC_DIRECTORY",
    "TEMPLATE_DIRECTORY",
    "back_url",
    "render",
    "render_error",
    "templates",
    "wants_html",
]
