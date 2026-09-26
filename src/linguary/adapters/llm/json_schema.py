"""JSON Schema структурированного ответа для провайдера.

Провайдер получает схему ответа, а не описание формата словами: строгий структурированный
вывод — обязательное требование к провайдеру MVP (PRD § 23).

Схема выводится из Pydantic-модели, но перед отправкой очищается от ключевых слов
ограничения значений (`minLength`, `minItems`, `minimum` и подобных). Их поддержка
у совместимых серверов различается, а строгий режим у части из них отвергает запрос
целиком. Ограничения при этом не теряются: их проверяет разбор ответа на нашей стороне,
и результат один — ответ, не соответствующий правилам, отбраковывается.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

#: Ключевые слова, описывающие структуру ответа. Всё остальное из схемы убирается.
_STRUCTURAL_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "anyOf",
        "enum",
        "const",
        "$ref",
        "$defs",
        "title",
        "description",
    }
)


#: Ключевые слова, значение которых — отображение «своё имя → подсхема». Их имена
#: принадлежат ответу, а не схеме, и фильтрации по списку ключевых слов не подлежат.
_SCHEMA_MAPPINGS = frozenset({"properties", "$defs"})


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Схема ответа модели в виде, пригодном для строгого структурированного вывода."""
    return _strip(model.model_json_schema())


def _strip(node: Any) -> Any:
    """Оставляет в узле схемы только структурные ключевые слова."""
    if isinstance(node, dict):
        return {
            key: _strip_mapping(value) if key in _SCHEMA_MAPPINGS else _strip(value)
            for key, value in node.items()
            if key in _STRUCTURAL_KEYWORDS
        }
    if isinstance(node, list):
        return [_strip(item) for item in node]
    return node


def _strip_mapping(node: Any) -> Any:
    """Обрабатывает набор именованных подсхем, сохраняя сами имена."""
    if not isinstance(node, dict):
        return _strip(node)
    return {name: _strip(value) for name, value in node.items()}


__all__ = ["strict_json_schema"]
