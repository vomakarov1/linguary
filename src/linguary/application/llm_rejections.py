"""Recording model responses rejected by application validation."""

from __future__ import annotations

from typing import Any

from linguary.application.ports.llm import LLMResponse
from linguary.application.ports.persistence import UnitOfWork
from linguary.application.retry_policy import classify_error, violation_of
from linguary.application.diagnostics import diagnostics_for


def record_rejection(
    unit_of_work: UnitOfWork, response: LLMResponse[Any], error: Exception
) -> None:
    """Associates a validation error with an already recorded model call."""
    if response.call_id is None:
        return
    with unit_of_work.transaction() as transaction:
        transaction.llm_calls.mark_rejected(
            response.call_id,
            error=classify_error(error),
            response_text=None,
        )
        diagnostics_for(unit_of_work).finalize_attempt(
            None if response.call_context is None else response.call_context.attempt_id,
            'rejected', violation_of(error, context=response.call_context), response.raw_content,
            call_id=response.call_id, body_meta=response.body_meta, transaction=transaction,
        )



def finalize_response(unit_of_work, response, decision, *, transaction=None):
    if response is None or response.call_id is None:
        return
    diagnostics_for(unit_of_work).finalize_attempt(
        None if response.call_context is None else response.call_context.attempt_id,
        decision, body=response.raw_content, call_id=response.call_id,
        body_meta=response.body_meta, transaction=transaction,
        retain_accepted=bool((response.body_meta or {}).get('debug_retained')),
    )
