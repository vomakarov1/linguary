"""Book endpoints: list, upload, details, processing plan, and warnings (API-01).

Routes validate incoming requests, invoke application use cases, and transform their
results into responses. Neither book states nor progress rules belong here: the card
and plan summary are computed by `linguary.application.book_overview`, while actions
are executed by use cases (ADR § 3).

Failures in use cases are converted to HTTP status codes by a single set of application
error handlers (`linguary.web.app`), so there is not a single `try` wrapping a use case
call here.
"""

from __future__ import annotations

import asyncio
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, Query, Request, UploadFile, status
from fastapi.responses import RedirectResponse

from linguary.application.book_overview import (
    ProcessingPlanUnavailableError,
    book_overview,
    book_warnings,
    list_books,
    processing_plan,
)
from linguary.application.use_cases.register_book import RegisterBookCommand
from linguary.domain import BookStatus
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import (
    BookDetailResponse,
    BookListResponse,
    BookWarningsResponse,
    ProcessingPlanResponse,
)
from linguary.web.uploads import (
    book_title_of_filename,
    ensure_epub_filename,
    save_upload,
)
from linguary.web import labels
from linguary.web.views import render, wants_html

router = APIRouter(tags=["books"])

#: Book language code: a short string such as `en` or `ru` (PRD § 21.2). MVP supports
#: a single language pair, but the interface does not hardcode it.
_LanguageCode = Annotated[str, Form(min_length=2, max_length=32)]


@router.get("/books", response_model=BookListResponse)
def get_books(
    request: Request,
    application: CurrentApplication,
    book_status: Annotated[
        BookStatus | None,
        Query(alias="status", description="Show only books in this state"),
    ] = None,
) -> BookListResponse | object:
    """List of books and their states (PRD § 21.1)."""
    response = BookListResponse.of_books(
        list_books(application.unit_of_work, status=book_status)
    )
    if wants_html(request):
        return render(request, "books/list.html", page="books", data=response)
    return response


@router.post(
    "/books/upload",
    status_code=status.HTTP_303_SEE_OTHER,
    response_class=RedirectResponse,
)
async def upload_book(
    application: CurrentApplication,
    file: Annotated[UploadFile, File(description="Book file in EPUB format")],
    source_language: _LanguageCode,
    target_language: _LanguageCode,
) -> RedirectResponse:
    """Accept an EPUB, register the book, and enqueue the import task (PRD § 18, § 21.2).

    The book appears in the list immediately in the "Importing" state: file parsing
    runs as a background task, so there is no need to keep the user waiting.
    """
    ensure_epub_filename(file.filename)
    book_id = str(uuid4())
    source_path = application.paths.source_epub(book_id)
    await save_upload(file, target=source_path)
    try:
        await asyncio.to_thread(
            application.register_book,
            RegisterBookCommand(
                book_id=book_id,
                source_epub_path=str(source_path),
                title=book_title_of_filename(file.filename),
                source_language=source_language.strip(),
                target_language=target_language.strip(),
            ),
        )
    except BaseException:
        source_path.unlink(missing_ok=True)
        raise
    await application.start_import(book_id)
    return RedirectResponse(
        url=f"/books/{book_id}", status_code=status.HTTP_303_SEE_OTHER
    )


@router.get("/books/{book_id}", response_model=BookDetailResponse)
def get_book(
    request: Request, application: CurrentApplication, book_id: str
) -> BookDetailResponse | object:
    """Book details: state, progress, treatment counters, and errors (PRD § 21.4)."""
    response = BookDetailResponse.of_overview(
        book_overview(
            application.unit_of_work,
            book_id,
            concurrency=application.worker_pool.concurrency,
            requests_per_minute=application.settings.llm.requests_per_minute,
        )
    )
    if wants_html(request):
        return render(
            request,
            "books/detail.html",
            page="detail",
            data=response,
            auto_refresh=_needs_refresh(response),
            stalled=_is_stalled(response),
            llm_ready=application.settings.llm.is_ready,
            llm_hint=labels.LLM_NOT_CONFIGURED_HINT,
        )
    return response


@router.get("/books/{book_id}/plan", response_model=ProcessingPlanResponse)
def get_book_plan(
    request: Request, application: CurrentApplication, book_id: str
) -> ProcessingPlanResponse | object:
    """Summary of the book processing plan by chapters (PRD § 21, UX notes)."""
    if wants_html(request):
        try:
            response = ProcessingPlanResponse.of_plan(
                processing_plan(application.unit_of_work, book_id)
            )
        except ProcessingPlanUnavailableError:
            return render(
                request, "books/plan.html", page="plan", data=None, book_id=book_id
            )
        return render(
            request,
            "books/plan.html",
            page="plan",
            data=response,
            book_id=book_id,
            plan_complete_label=labels.PLAN_COMPLETE_LABEL,
            plan_complete_hint=labels.PLAN_COMPLETE_HINT,
            plan_incomplete_label=labels.PLAN_INCOMPLETE_LABEL,
            plan_incomplete_hint=labels.PLAN_INCOMPLETE_HINT,
        )
    response = ProcessingPlanResponse.of_plan(
        processing_plan(application.unit_of_work, book_id)
    )
    return response


@router.get("/books/{book_id}/warnings", response_model=BookWarningsResponse)
def get_book_warnings(
    request: Request, application: CurrentApplication, book_id: str
) -> BookWarningsResponse | object:
    """Import warnings and processing result remarks (PRD § 21.6)."""
    response = BookWarningsResponse.of_warnings(
        book_warnings(application.unit_of_work, book_id)
    )
    if wants_html(request):
        return render(
            request,
            "books/warnings.html",
            page="warnings",
            data=response,
            book_id=book_id,
        )
    return response


__all__ = ["router"]


def _needs_refresh(response: BookDetailResponse) -> bool:
    """Whether the book card needs periodic refreshing.

    A book whose progress was stopped by an error will not advance on its own:
    a page with an error is not refreshed so that the user has time to read it.
    """
    if _has_active_tasks(response):
        return True
    return response.book.status in {"IMPORTING", "ANALYZING", "PROCESSING"} and not (
        _is_stalled(response)
    )


def _is_stalled(response: BookDetailResponse) -> bool:
    """Whether work on the book is stalled: no active tasks, but failed tasks exist."""
    return bool(response.progress.failures) and not _has_active_tasks(response)


def _has_active_tasks(response: BookDetailResponse) -> bool:
    """Whether the book has tasks in queue or in progress."""
    return response.progress.pending_tasks + response.progress.running_tasks > 0
