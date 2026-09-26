"""Endpoints проверки глоссария книги (API-01, PRD § 11-12, § 21.3)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from linguary.application.glossary_review import GlossaryNotFoundError
from linguary.application.use_cases.approve_glossary import (
    GlossaryDecisionCommand,
    GlossaryNoteCommand,
)
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import (
    GlossaryDecisionRequest,
    GlossaryItemResponse,
    GlossaryNoteRequest,
    GlossaryReviewResponse,
)
from linguary.web.views import render, wants_html

router = APIRouter(prefix="/books/{book_id}/glossary", tags=["glossary"])


@router.get("", response_model=GlossaryReviewResponse)
def get_glossary(
    request: Request, application: CurrentApplication, book_id: str
) -> GlossaryReviewResponse | object:
    """Глоссарий книги; после старта обработки он доступен только для чтения."""
    review = application.approve_glossary.review(book_id)
    response = GlossaryReviewResponse.of_review(review)
    if wants_html(request):
        return render(
            request,
            "books/glossary.html",
            page="glossary",
            data=response,
            book_status=review.book_status.value if review.book_status else None,
        )
    return response


@router.post("/accept-all", response_model=GlossaryReviewResponse)
def accept_all_required_items(
    request: Request, application: CurrentApplication, book_id: str
) -> GlossaryReviewResponse | RedirectResponse:
    """Принимает предложенные переводы для всех обязательных нерешённых элементов."""
    application.approve_glossary.accept_all_awaiting(book_id)
    if wants_html(request):
        return RedirectResponse(
            url=f"/books/{book_id}/glossary", status_code=303
        )
    return GlossaryReviewResponse.of_review(application.approve_glossary.review(book_id))


@router.post("/{concept_id}", response_model=GlossaryItemResponse)
def decide_glossary_item(
    http_request: Request,
    application: CurrentApplication,
    book_id: str,
    concept_id: str,
    request: Annotated[GlossaryDecisionRequest, Form()],
) -> GlossaryItemResponse | RedirectResponse:
    """Принимает, исправляет, фиксирует или пропускает один найденный термин."""
    review = application.approve_glossary.review(book_id)
    item = review.item_for_concept(concept_id)
    if item is None:
        raise GlossaryNotFoundError(
            f"в глоссарии книги {book_id} нет понятия {concept_id!r}"
        )
    application.approve_glossary(
        GlossaryDecisionCommand(
            book_id=book_id,
            source_text=item.source_text,
            concept_id=concept_id,
            decision=request.action,
            translation=request.text,
            note=request.note,
            type=request.type,
        )
    )
    updated = application.approve_glossary.review(book_id).item_for_concept(concept_id)
    if updated is None:
        raise GlossaryNotFoundError(
            f"решение по понятию {concept_id!r} книги {book_id} не сохранилось"
        )
    if wants_html(http_request):
        return RedirectResponse(
            url=f"/books/{book_id}/glossary", status_code=303
        )
    return GlossaryItemResponse.of_item(updated)


@router.post("/{concept_id}/note", response_model=GlossaryItemResponse)
def set_glossary_note(
    http_request: Request,
    application: CurrentApplication,
    book_id: str,
    concept_id: str,
    request: Annotated[GlossaryNoteRequest, Form()],
) -> GlossaryItemResponse | RedirectResponse:
    """Меняет переводческое указание по элементу, не трогая принятый вариант перевода."""
    application.approve_glossary.set_note(
        GlossaryNoteCommand(
            book_id=book_id, concept_id=concept_id, note=request.note
        )
    )
    updated = application.approve_glossary.review(book_id).item_for_concept(concept_id)
    if updated is None:
        raise GlossaryNotFoundError(
            f"указание по понятию {concept_id!r} книги {book_id} не сохранилось"
        )
    if wants_html(http_request):
        return RedirectResponse(url=f"/books/{book_id}/glossary", status_code=303)
    return GlossaryItemResponse.of_item(updated)


__all__ = ["router"]
