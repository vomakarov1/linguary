"""Pure stdlib application DTOs for the 7 illustration LLM operations (`P2.T2`)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IllustrationMentionDTO:
    mention_id: str
    name: str
    entity_kind: str  # "person" | "location"
    alias_of_hint: str | None
    quote_block_id: str
    quote: str


@dataclass(frozen=True, slots=True)
class IllustrationFactDTO:
    fact_id: str
    subject_mention_id: str
    temporal_scope: str  # "permanent" | "episodic" | "state_change"
    back_applicable: bool
    reveals_future_spoiler: bool
    description_ru: str
    description_en: str
    quote_block_id: str
    quote: str


@dataclass(frozen=True, slots=True)
class IllustrationExtractResponse:
    mentions: tuple[IllustrationMentionDTO, ...] = ()
    facts: tuple[IllustrationFactDTO, ...] = ()
    covered_source_unit_ids: tuple[str, ...] = ()
    complete: bool = True
    contract_version: int = 1


@dataclass(frozen=True, slots=True)
class IllustrationAliasDecisionDTO:
    pair_id: str
    verdict: str  # "same_entity" | "different" | "unknown"
    evidence_fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IllustrationAliasResponse:
    decisions: tuple[IllustrationAliasDecisionDTO, ...] = ()
    contract_version: int = 1


@dataclass(frozen=True, slots=True)
class IllustrationDesignTraitDTO:
    trait: str
    description_ru: str
    description_en: str
    assumption_ru: str


@dataclass(frozen=True, slots=True)
class IllustrationDesignResponse:
    entity_id: str
    traits: tuple[IllustrationDesignTraitDTO, ...] = ()
    contract_version: int = 1


@dataclass(frozen=True, slots=True)
class IllustrationFactScopeResponse:
    event_id: str
    applicable_fact_ids: tuple[str, ...] = ()
    excluded_fact_ids: tuple[str, ...] = ()
    compact_description_ru: str = ""
    compact_description_en: str = ""
    assumptions_ru: tuple[str, ...] = ()
    complete: bool = True
    contract_version: int = 1


@dataclass(frozen=True, slots=True)
class IllustrationQuoteAnchorDTO:
    block_id: str
    quote: str


@dataclass(frozen=True, slots=True)
class IllustrationCandidateEventDTO:
    event_id: str
    title_ru: str
    event_ru: str
    significance_ru: str
    event_kind: str
    quote_anchors: tuple[IllustrationQuoteAnchorDTO, ...] = ()
    insert_after_block_id: str | None = None
    participant_names: tuple[str, ...] = ()
    location_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IllustrationCandidatesResponse:
    scene_id: str
    events: tuple[IllustrationCandidateEventDTO, ...] = ()
    zero_reason_ru: str | None = None
    covered_source_unit_ids: tuple[str, ...] = ()
    complete: bool = True
    contract_version: int = 1


@dataclass(frozen=True, slots=True)
class IllustrationSelectionReasonDTO:
    candidate_id: str
    reason_ru: str


@dataclass(frozen=True, slots=True)
class IllustrationSelectionResponse:
    selected_candidate_ids: tuple[str, ...] = ()
    selection_reasons_ru: tuple[IllustrationSelectionReasonDTO, ...] = ()
    rejected_reasons_ru: tuple[IllustrationSelectionReasonDTO, ...] = ()
    zero_reason_ru: str | None = None
    contract_version: int = 1


@dataclass(frozen=True, slots=True)
class IllustrationPromptResponse:
    moment_id: str
    scene_composition_en: str
    used_fact_ids: tuple[str, ...] = ()
    used_design_keys: tuple[str, ...] = ()
    assumptions_ru: tuple[str, ...] = ()
    depicted_entity_ids: tuple[str, ...] | None = None
    contract_version: int = 1


__all__ = [
    "IllustrationAliasDecisionDTO",
    "IllustrationAliasResponse",
    "IllustrationCandidateEventDTO",
    "IllustrationCandidatesResponse",
    "IllustrationDesignResponse",
    "IllustrationDesignTraitDTO",
    "IllustrationExtractResponse",
    "IllustrationFactDTO",
    "IllustrationFactScopeResponse",
    "IllustrationMentionDTO",
    "IllustrationPromptResponse",
    "IllustrationQuoteAnchorDTO",
    "IllustrationSelectionReasonDTO",
    "IllustrationSelectionResponse",
]
