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


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Model response schema formatted for strict structured output."""
    schema = model.model_json_schema()
    _filter_omitted_properties(schema)
    return _strip(schema)


def _filter_omitted_properties(node: Any) -> None:
    if isinstance(node, dict):
        if "$defs" in node and isinstance(node["$defs"], dict):
            node["$defs"].pop("HelpEvidenceSchema", None)
            for def_name in ("PronunciationAnchorSchema", "ReviewAnchor"):
                if def_name in node["$defs"] and isinstance(node["$defs"][def_name], dict):
                    target = node["$defs"][def_name]
                    if "properties" in target and isinstance(target["properties"], dict):
                        target["properties"].pop("start", None)
                        target["properties"].pop("end", None)
                    if "required" in target and isinstance(target["required"], list):
                        target["required"] = [r for r in target["required"] if r not in {"start", "end"}]
            if "QualityReviewSchema" in node["$defs"] and isinstance(node["$defs"]["QualityReviewSchema"], dict):
                target = node["$defs"]["QualityReviewSchema"]
                if "properties" in target and isinstance(target["properties"], dict):
                    target["properties"].pop("checked_artifacts", None)
                    target["properties"].pop("missing_artifacts", None)
                if "required" in target and isinstance(target["required"], list):
                    target["required"] = [r for r in target["required"] if r not in {"checked_artifacts", "missing_artifacts"}]
            if "ReviewFinding" in node["$defs"] and isinstance(node["$defs"]["ReviewFinding"], dict):
                target = node["$defs"]["ReviewFinding"]
                if "properties" in target and isinstance(target["properties"], dict):
                    target["properties"].pop("artifact_sha", None)
                if "required" in target and isinstance(target["required"], list):
                    target["required"] = [r for r in target["required"] if r != "artifact_sha"]
            if "CorrectionVerdictSchema" in node["$defs"] and isinstance(node["$defs"]["CorrectionVerdictSchema"], dict):
                target = node["$defs"]["CorrectionVerdictSchema"]
                if "properties" in target and isinstance(target["properties"], dict):
                    target["properties"].pop("proposal_sha256", None)
                if "required" in target and isinstance(target["required"], list):
                    target["required"] = [r for r in target["required"] if r != "proposal_sha256"]
        title = node.get("title")
        if title in ("PronunciationAnchorSchema", "ReviewAnchor"):
            if "properties" in node and isinstance(node["properties"], dict):
                node["properties"].pop("start", None)
                node["properties"].pop("end", None)
            if "required" in node and isinstance(node["required"], list):
                node["required"] = [r for r in node["required"] if r not in {"start", "end"}]
        elif title == "QualityReviewSchema":
            if "properties" in node and isinstance(node["properties"], dict):
                node["properties"].pop("checked_artifacts", None)
                node["properties"].pop("missing_artifacts", None)
            if "required" in node and isinstance(node["required"], list):
                node["required"] = [r for r in node["required"] if r not in {"checked_artifacts", "missing_artifacts"}]
        elif title == "ReviewFinding":
            if "properties" in node and isinstance(node["properties"], dict):
                node["properties"].pop("artifact_sha", None)
            if "required" in node and isinstance(node["required"], list):
                node["required"] = [r for r in node["required"] if r != "artifact_sha"]
        elif title == "CorrectionVerdictSchema":
            if "properties" in node and isinstance(node["properties"], dict):
                node["properties"].pop("proposal_sha256", None)
            if "required" in node and isinstance(node["required"], list):
                node["required"] = [r for r in node["required"] if r != "proposal_sha256"]
        if "properties" in node and isinstance(node["properties"], dict):
            for prop in _OMITTED_STRICT_PROPERTIES:
                node["properties"].pop(prop, None)
            node.setdefault("additionalProperties", False)
        if "required" in node and isinstance(node["required"], list):
            node["required"] = [r for r in node["required"] if r not in _OMITTED_STRICT_PROPERTIES]
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
