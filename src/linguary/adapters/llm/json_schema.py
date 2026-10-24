"""Structured response JSON Schema for the provider.

The provider receives a schema for its response rather than prose format descriptions:
strict structured output is a mandatory provider requirement in the MVP (PRD § 23).

The schema is derived from a Pydantic model, but is stripped of value validation keywords
(`minLength`, `minItems`, `minimum`, etc.) before transmission. Support for these varies
across compatible servers, and strict mode on some rejects the request outright. Validation
constraints are not lost: our response parsing verifies them on arrival, with the same outcome —
an invalid response is rejected and retried.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

#: Keywords defining response structure. Everything else is stripped from the schema.
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


#: Keywords whose values are "custom name -> subschema" mappings. These keys
#: belong to the payload rather than schema keywords, and are exempt from keyword filtering.
_SCHEMA_MAPPINGS = frozenset({"properties", "$defs"})


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Model response schema formatted for strict structured output."""
    return _strip(model.model_json_schema())


def _strip(node: Any) -> Any:
    """Retains only structural keywords in a schema node."""
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
    """Processes a set of named subschemas while preserving the names."""
    if not isinstance(node, dict):
        return _strip(node)
    return {name: _strip(value) for name, value in node.items()}


__all__ = ["strict_json_schema"]
