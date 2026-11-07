"""Tests for OpenAICompatibleLLMClient repair_instructions handling (P2.T1, AC-007)."""

import pytest
from linguary.adapters.config.settings import LLMSettings
from linguary.adapters.llm.openai_compatible import _body
from linguary.application.ports.llm import LLMRequest


def test_repair_instructions_sent_as_separate_user_messages() -> None:
    """Repair instructions are sent as separate user messages following input_text, leaving system prompt intact (AC-007)."""
    settings = LLMSettings(
        provider="openai",
        api_key="secret",
        model="gpt-4o",
    )
    request = LLMRequest.for_translation(
        instructions="System instructions: translate accurately.",
        input_text="Source text of the chapter.",
    )
    request = request.with_logging(
        book_id="book-1",
        stage="translation",
    )
    # Add repair instructions
    request = LLMRequest(
        kind=request.kind,
        instructions=request.instructions,
        input_text=request.input_text,
        book_id=request.book_id,
        stage=request.stage,
        repair_instructions=("Fix block b1: keep English intact", "Add missing translation for b2"),
    )

    body = _body(request, settings=settings)

    assert len(body["messages"]) == 4
    assert body["messages"][0] == {
        "role": "system",
        "content": "System instructions: translate accurately.",
    }
    assert body["messages"][1] == {
        "role": "user",
        "content": "Source text of the chapter.",
    }
    assert body["messages"][2] == {
        "role": "user",
        "content": "Fix block b1: keep English intact",
    }
    assert body["messages"][3] == {
        "role": "user",
        "content": "Add missing translation for b2",
    }
