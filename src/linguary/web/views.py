"""HTML views of the local Web UI.

JSON remains the primary contract of API-01, while browser screens are selected via
`Accept: text/html`. Thus routes reuse the same application use cases and DTOs without
breaking existing JSON clients.
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

#: Warning about transmitting book text to external provider is needed by screens that
#: show nothing about LLM configuration themselves (PRD § 4.1, § 21.2), so it is
#: available to any template rather than passed as a separate field in every response.
templates.env.globals["external_llm_warning"] = labels.EXTERNAL_LLM_WARNING

#: Glossary concept types in Russian: type correction options on glossary screen.
templates.env.globals["concept_types"] = tuple(
    (concept_type.value, label)
    for concept_type, label in labels.CONCEPT_TYPE_LABELS.items()
)


def wants_html(request: Request) -> bool:
    """Whether the client requests HTML rather than JSON."""
    accept = request.headers.get("accept", "")
    return "text/html" in accept.lower()


def render(request: Request, template: str, **context: Any) -> HTMLResponse:
    """Renders HTML template with required `request`."""
    return templates.TemplateResponse(
        request=request, name=template, context=context
    )


def render_error(
    request: Request, *, status_code: int, messages: Sequence[str]
) -> HTMLResponse:
    """Browser error page: rejection reason and back link.

    Status code matches what a JSON client would receive: only the presentation changes.
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
    """Rejection text as a complete sentence: capitalized and ending with a period.

    Use case messages are formatted for logs and JSON — lowercase without trailing periods,
    whereas on the page they are read as standalone paragraphs.
    """
    text = message.strip()
    if not text:
        return text
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else f"{text}."


#: Book page path regex matching request URL.
_BOOK_PATH = re.compile(r"^/books/(?!upload(?:/|$))([^/]+)")


def back_url(request: Request) -> str:
    """Where to return user from error page.

    Preferably the page where the form was submitted from: named by browser in
    `Referer`. Foreign URLs are rejected — link leads only within the application.
    Without `Referer`, actions on a book return to its card, others to book list.
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
