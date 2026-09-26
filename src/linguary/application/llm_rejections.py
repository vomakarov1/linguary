"""Отметка ответа модели, отвергнутого прикладной проверкой."""

from __future__ import annotations

from typing import Any

from linguary.application.ports.llm import LLMResponse
from linguary.application.ports.persistence import UnitOfWork
from linguary.application.retry_policy import classify_error


def record_rejection(
    unit_of_work: UnitOfWork, response: LLMResponse[Any], error: Exception
) -> None:
    """Связывает ошибку проверки с уже записанным вызовом модели."""
    if response.call_id is None:
        return
    with unit_of_work.transaction() as transaction:
        transaction.llm_calls.mark_rejected(
            response.call_id,
            error=classify_error(error),
            response_text=response.raw_content,
        )
