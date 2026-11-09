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


#: Properties omitted from provider-facing strict schemas (evidence is resolved algorithmically).
_OMITTED_STRICT_PROPERTIES = frozenset({"before_you_read_evidence", "content_evidence"})

#: Model-specific properties omitted from provider-facing strict schemas.
_OMITTED_SCHEMA_PROPERTIES: dict[str, frozenset[str]] = {
    "PronunciationAnchorSchema": frozenset({"start", "end"}),
    "PronunciationCandidateSchema": frozenset({"form"}),
    "ReviewAnchor": frozenset({"start", "end"}),
    "QualityReviewSchema": frozenset({"checked_artifacts", "missing_artifacts"}),
    "ReviewFinding": frozenset({"artifact_sha"}),
    "CorrectionVerdictSchema": frozenset({"proposal_sha256"}),
}


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Model response schema formatted for strict structured output."""
    schema = model.model_json_schema()
    _filter_omitted_properties(schema)
    return _strip(schema)


def _filter_omitted_properties(node: Any) -> None:
    if isinstance(node, dict):
        if "$defs" in node and isinstance(node["$defs"], dict):
            node["$defs"].pop("HelpEvidenceSchema", None)
            for def_name, omitted_props in _OMITTED_SCHEMA_PROPERTIES.items():
                if def_name in node["$defs"] and isinstance(node["$defs"][def_name], dict):
                    target = node["$defs"][def_name]
                    if "properties" in target and isinstance(target["properties"], dict):
                        for prop in omitted_props:
                            target["properties"].pop(prop, None)
                    if "required" in target and isinstance(target["required"], list):
                        target["required"] = [r for r in target["required"] if r not in omitted_props]
        title = node.get("title")
        omitted = set(_OMITTED_STRICT_PROPERTIES)
        if isinstance(title, str) and title in _OMITTED_SCHEMA_PROPERTIES:
            omitted.update(_OMITTED_SCHEMA_PROPERTIES[title])
        if "properties" in node and isinstance(node["properties"], dict):
            props = node["properties"]
            for prop in omitted:
                props.pop(prop, None)
            node.setdefault("additionalProperties", False)
            existing_req = [k for k in node.get("required", []) if k in props and k not in omitted]
            existing_set = set(existing_req)
            node["required"] = existing_req + [k for k in props.keys() if k not in existing_set]
        elif "required" in node and isinstance(node["required"], list):
            node["required"] = [r for r in node["required"] if r not in omitted]
        for v in node.values():
            _filter_omitted_properties(v)
    elif isinstance(node, list):
        for item in node:
            _filter_omitted_properties(item)


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
