"""Готовые ответы модели для тестов схем, разбора и клиентов.

Фабрики собирают структуру ответа в том виде, в каком её отдаёт провайдер, и позволяют
точечно испортить одно поле: тесты отличаются одной строкой, а не копией всего ответа.
"""

from __future__ import annotations

import json
from typing import Any


def analysis_payload(**overrides: Any) -> dict[str, Any]:
    """Корректный ответ анализа главы."""
    payload: dict[str, Any] = {
        "glossary_candidates": [
            {
                "source_text": "Mr. Bennet",
                "type": "person",
                "suggested_translation": "мистер Беннет",
                "confidence": 0.9,
                "model_flagged_review": False,
                "note": None,
            }
        ],
        "translation_instructions": "Сохранять иронию рассказчика.",
        "treatment_plan": [
            {
                "source_block_id": "block-1",
                "treatment": "TRANSLATE",
            },
            {
                "source_block_id": "block-2",
                "treatment": "GLOSSARY_ONLY",
            },
        ],
        "before_you_read": [
            {"expression": "entailed estate", "explanation": "имение с ограничением наследования"}
        ],
        "chapter_summary": "Mr. Bennet visits the new neighbour.",
    }
    payload.update(overrides)
    return payload


def translation_payload(**overrides: Any) -> dict[str, Any]:
    """Корректный ответ обработки группы блоков."""
    payload: dict[str, Any] = {
        "blocks": [
            block_payload(),
            block_payload(
                source_block_id="block-2",
                treatment="GLOSSARY_ONLY",
                translated_text=None,
                glossary_hints=[
                    {"original": "entailed", "translation": "заповедный", "note": None}
                ],
            ),
        ],
        "language_notes": [
            {
                "source_block_id": "block-1",
                "original_excerpt": "It is a truth universally acknowledged",
                "translated_excerpt": "Общепризнанная истина",
                "note": "Ироническая формула зачина.",
            }
        ],
    }
    payload.update(overrides)
    return payload


def block_payload(**overrides: Any) -> dict[str, Any]:
    """Результат одного блока: по умолчанию полный перевод."""
    payload: dict[str, Any] = {
        "source_block_id": "block-1",
        "treatment": "TRANSLATE",
        "translated_text": "Общепризнанная истина гласит…",
        "glossary_hints": [],
    }
    payload.update(overrides)
    return payload


def as_content(payload: dict[str, Any]) -> str:
    """Ответ в том виде, в каком его отдаёт провайдер, — текстом."""
    return json.dumps(payload, ensure_ascii=False)
