"""Endpoints проверки глоссария книги (API-01, PRD § 11-12, § 21.3)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Form

from linguary.application.glossary_review import GlossaryNotFoundError
from linguary.application.use_cases.approve_glossary import GlossaryDecisionCommand
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import (
    GlossaryDecisionRequest,
    GlossaryItemResponse,
    GlossaryReviewResponse,
)

router = APIRouter(prefix="/books/{book_id}/glossary", tags=["glossary"])


@router.get("", response_model=GlossaryReviewResponse)
def get_glossary(
    application: CurrentApplication, book_id: str
) -> GlossaryReviewResponse:
    """Глоссарий книги; после старта обработки он доступен только для чтения."""
    return GlossaryReviewResponse.of_review(application.approve_glossary.review(book_id))


@router.post("/{concept_id}", response_model=GlossaryItemResponse)
def decide_glossary_item(
    application: CurrentApplication,
    book_id: str,
    concept_id: str,
    request: Annotated[GlossaryDecisionRequest, Form()],
) -> GlossaryItemResponse:
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
        )
    )
    updated = application.approve_glossary.review(book_id).item_for_concept(concept_id)
    if updated is None:
        raise GlossaryNotFoundError(
            f"решение по понятию {concept_id!r} книги {book_id} не сохранилось"
        )
    return GlossaryItemResponse.of_item(updated)


__all__ = ["router"]
