"""Flat strict Pydantic schemas for the 7 illustration LLM operations (`P2.T2`).

Every schema enforces `extra="forbid"` and `strict=True`, avoids `str_strip_whitespace=True`
(preserving exact verbatim quotes), rejects JSON booleans on `contract_version: Literal[1]`,
forbids free `dict[str, str]`, and avoids `$ref` sibling keywords so `strict_json_schema()`
produces valid OpenAI strict structured output schemas.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from linguary.application.illustration_dto import (
    IllustrationAliasDecisionDTO,
    IllustrationAliasResponse,
    IllustrationCandidateEventDTO,
    IllustrationCandidatesResponse,
    IllustrationDesignResponse,
    IllustrationDesignTraitDTO,
    IllustrationExtractResponse,
    IllustrationFactDTO,
    IllustrationFactScopeResponse,
    IllustrationMentionDTO,
    IllustrationPromptResponse,
    IllustrationQuoteAnchorDTO,
    IllustrationSelectionReasonDTO,
    IllustrationSelectionResponse,
)

_MAX_STR = 4000
_MAX_QUOTE = 2000
_MAX_LIST = 32
_MAX_ANCHORS = 16
_MAX_ALIAS_PAIRS = 16


class _IllustrationStrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _VersionedIllustrationBase(_IllustrationStrictBase):
    contract_version: Literal[1] = 1

    @field_validator("contract_version", mode="before")
    @classmethod
    def _reject_bool_contract_version(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("contract_version не может быть логическим значением")
        return value


class IllustrationMentionSchema(_IllustrationStrictBase):
    mention_id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=400)
    entity_kind: Literal["person", "location"]
    alias_of_hint: str | None = Field(default=None, max_length=400)
    quote_block_id: str = Field(min_length=1, max_length=200)
    quote: str = Field(min_length=1, max_length=_MAX_QUOTE)


class IllustrationFactSchema(_IllustrationStrictBase):
    fact_id: str = Field(min_length=1, max_length=200)
    subject_mention_id: str = Field(min_length=1, max_length=200)
    temporal_scope: Literal["permanent", "episodic", "state_change"]
    back_applicable: bool
    reveals_future_spoiler: bool
    description_ru: str = Field(min_length=1, max_length=_MAX_STR)
    description_en: str = Field(min_length=1, max_length=_MAX_STR)
    quote_block_id: str = Field(min_length=1, max_length=200)
    quote: str = Field(min_length=1, max_length=_MAX_QUOTE)


class IllustrationExtractSchema(_VersionedIllustrationBase):
    mentions: list[IllustrationMentionSchema] = Field(max_length=_MAX_LIST)
    facts: list[IllustrationFactSchema] = Field(max_length=_MAX_LIST)
    covered_source_unit_ids: list[str] = Field(max_length=_MAX_LIST)
    complete: bool

    @field_validator("covered_source_unit_ids")
    @classmethod
    def _check_unit_ids(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 200 for item in value):
            raise ValueError("некорректный идентификатор исходного фрагмента")
        return value

    def to_dto(self) -> IllustrationExtractResponse:
        return IllustrationExtractResponse(
            mentions=tuple(
                IllustrationMentionDTO(
                    mention_id=m.mention_id,
                    name=m.name,
                    entity_kind=m.entity_kind,
                    alias_of_hint=m.alias_of_hint,
                    quote_block_id=m.quote_block_id,
                    quote=m.quote,
                )
                for m in self.mentions
            ),
            facts=tuple(
                IllustrationFactDTO(
                    fact_id=f.fact_id,
                    subject_mention_id=f.subject_mention_id,
                    temporal_scope=f.temporal_scope,
                    back_applicable=f.back_applicable,
                    reveals_future_spoiler=f.reveals_future_spoiler,
                    description_ru=f.description_ru,
                    description_en=f.description_en,
                    quote_block_id=f.quote_block_id,
                    quote=f.quote,
                )
                for f in self.facts
            ),
            covered_source_unit_ids=tuple(self.covered_source_unit_ids),
            complete=self.complete,
            contract_version=self.contract_version,
        )


class IllustrationAliasDecisionSchema(_IllustrationStrictBase):
    pair_id: str = Field(min_length=1, max_length=200)
    verdict: Literal["same_entity", "different", "unknown"]
    evidence_fact_ids: list[str] = Field(max_length=_MAX_LIST)

    @field_validator("evidence_fact_ids")
    @classmethod
    def _check_evidence_ids(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 200 for item in value):
            raise ValueError("некорректный идентификатор факта")
        return value


class IllustrationAliasSchema(_VersionedIllustrationBase):
    decisions: list[IllustrationAliasDecisionSchema] = Field(max_length=_MAX_ALIAS_PAIRS)

    def to_dto(self) -> IllustrationAliasResponse:
        return IllustrationAliasResponse(
            decisions=tuple(
                IllustrationAliasDecisionDTO(
                    pair_id=d.pair_id,
                    verdict=d.verdict,
                    evidence_fact_ids=tuple(d.evidence_fact_ids),
                )
                for d in self.decisions
            ),
            contract_version=self.contract_version,
        )


class IllustrationDesignTraitSchema(_IllustrationStrictBase):
    trait: Literal[
        "hair",
        "eyes",
        "build",
        "age_appearance",
        "attire_default",
        "distinctive_mark",
        "palette_value",
        "architecture",
        "atmosphere",
    ]
    description_ru: str = Field(min_length=1, max_length=_MAX_STR)
    description_en: str = Field(min_length=1, max_length=_MAX_STR)
    assumption_ru: str = Field(min_length=1, max_length=_MAX_STR)


class IllustrationDesignSchema(_VersionedIllustrationBase):
    entity_id: str = Field(min_length=1, max_length=200)
    traits: list[IllustrationDesignTraitSchema] = Field(max_length=_MAX_LIST)

    def to_dto(self) -> IllustrationDesignResponse:
        return IllustrationDesignResponse(
            entity_id=self.entity_id,
            traits=tuple(
                IllustrationDesignTraitDTO(
                    trait=t.trait,
                    description_ru=t.description_ru,
                    description_en=t.description_en,
                    assumption_ru=t.assumption_ru,
                )
                for t in self.traits
            ),
            contract_version=self.contract_version,
        )


class IllustrationFactScopeSchema(_VersionedIllustrationBase):
    event_id: str = Field(min_length=1, max_length=200)
    applicable_fact_ids: list[str] = Field(max_length=_MAX_LIST)
    excluded_fact_ids: list[str] = Field(max_length=_MAX_LIST)
    compact_description_ru: str = Field(max_length=_MAX_STR)
    compact_description_en: str = Field(max_length=_MAX_STR)
    assumptions_ru: list[str] = Field(max_length=_MAX_LIST)
    complete: bool

    @field_validator("applicable_fact_ids", "excluded_fact_ids")
    @classmethod
    def _check_fact_ids(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 200 for item in value):
            raise ValueError("некорректный идентификатор факта")
        return value

    @field_validator("assumptions_ru")
    @classmethod
    def _check_assumptions(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > _MAX_STR for item in value):
            raise ValueError("некорректное предположение")
        return value

    def to_dto(self) -> IllustrationFactScopeResponse:
        return IllustrationFactScopeResponse(
            event_id=self.event_id,
            applicable_fact_ids=tuple(self.applicable_fact_ids),
            excluded_fact_ids=tuple(self.excluded_fact_ids),
            compact_description_ru=self.compact_description_ru,
            compact_description_en=self.compact_description_en,
            assumptions_ru=tuple(self.assumptions_ru),
            complete=self.complete,
            contract_version=self.contract_version,
        )


class IllustrationQuoteAnchorSchema(_IllustrationStrictBase):
    block_id: str = Field(min_length=1, max_length=200)
    quote: str = Field(min_length=1, max_length=_MAX_QUOTE)


class IllustrationCandidateEventSchema(_IllustrationStrictBase):
    event_id: str = Field(min_length=1, max_length=200)
    title_ru: str = Field(min_length=1, max_length=400)
    event_ru: str = Field(min_length=1, max_length=_MAX_STR)
    significance_ru: str = Field(min_length=1, max_length=_MAX_STR)
    event_kind: str = Field(min_length=1, max_length=200)
    quote_anchors: list[IllustrationQuoteAnchorSchema] = Field(min_length=1, max_length=_MAX_ANCHORS)
    insert_after_block_id: str | None = Field(default=None, max_length=200)
    participant_names: list[str] = Field(max_length=_MAX_LIST)
    location_names: list[str] = Field(max_length=_MAX_LIST)

    @field_validator("participant_names", "location_names")
    @classmethod
    def _check_names(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 400 for item in value):
            raise ValueError("некорректное имя участника или места")
        return value


class IllustrationCandidatesSchema(_VersionedIllustrationBase):
    scene_id: str = Field(min_length=1, max_length=200)
    events: list[IllustrationCandidateEventSchema] = Field(max_length=_MAX_LIST)
    zero_reason_ru: str | None = Field(default=None, max_length=_MAX_STR)
    covered_source_unit_ids: list[str] = Field(max_length=_MAX_LIST)
    complete: bool

    @field_validator("covered_source_unit_ids")
    @classmethod
    def _check_covered(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 200 for item in value):
            raise ValueError("некорректный идентификатор исходного фрагмента")
        return value

    def to_dto(self) -> IllustrationCandidatesResponse:
        return IllustrationCandidatesResponse(
            scene_id=self.scene_id,
            events=tuple(
                IllustrationCandidateEventDTO(
                    event_id=e.event_id,
                    title_ru=e.title_ru,
                    event_ru=e.event_ru,
                    significance_ru=e.significance_ru,
                    event_kind=e.event_kind,
                    quote_anchors=tuple(
                        IllustrationQuoteAnchorDTO(block_id=a.block_id, quote=a.quote)
                        for a in e.quote_anchors
                    ),
                    insert_after_block_id=e.insert_after_block_id,
                    participant_names=tuple(e.participant_names),
                    location_names=tuple(e.location_names),
                )
                for e in self.events
            ),
            zero_reason_ru=self.zero_reason_ru,
            covered_source_unit_ids=tuple(self.covered_source_unit_ids),
            complete=self.complete,
            contract_version=self.contract_version,
        )


class IllustrationSelectionReasonSchema(_IllustrationStrictBase):
    candidate_id: str = Field(min_length=1, max_length=200)
    reason_ru: str = Field(min_length=1, max_length=_MAX_STR)


class IllustrationSelectionSchema(_VersionedIllustrationBase):
    selected_candidate_ids: list[str] = Field(max_length=_MAX_LIST)
    selection_reasons_ru: list[IllustrationSelectionReasonSchema] = Field(max_length=_MAX_LIST)
    rejected_reasons_ru: list[IllustrationSelectionReasonSchema] = Field(max_length=_MAX_LIST)
    zero_reason_ru: str | None = Field(default=None, max_length=_MAX_STR)

    @field_validator("selected_candidate_ids")
    @classmethod
    def _check_selected_ids(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 200 for item in value):
            raise ValueError("некорректный идентификатор кандидата")
        return value

    def to_dto(self) -> IllustrationSelectionResponse:
        return IllustrationSelectionResponse(
            selected_candidate_ids=tuple(self.selected_candidate_ids),
            selection_reasons_ru=tuple(
                IllustrationSelectionReasonDTO(candidate_id=r.candidate_id, reason_ru=r.reason_ru)
                for r in self.selection_reasons_ru
            ),
            rejected_reasons_ru=tuple(
                IllustrationSelectionReasonDTO(candidate_id=r.candidate_id, reason_ru=r.reason_ru)
                for r in self.rejected_reasons_ru
            ),
            zero_reason_ru=self.zero_reason_ru,
            contract_version=self.contract_version,
        )


class IllustrationPromptSchema(_VersionedIllustrationBase):
    moment_id: str = Field(min_length=1, max_length=200)
    scene_composition_en: str = Field(min_length=1, max_length=_MAX_STR)
    used_fact_ids: list[str] = Field(max_length=_MAX_LIST)
    used_design_keys: list[str] = Field(max_length=_MAX_LIST)
    assumptions_ru: list[str] = Field(max_length=_MAX_LIST)

    @field_validator("used_fact_ids", "used_design_keys")
    @classmethod
    def _check_refs(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 200 for item in value):
            raise ValueError("некорректная ссылка в промте")
        return value

    @field_validator("assumptions_ru")
    @classmethod
    def _check_prompt_assumptions(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > _MAX_STR for item in value):
            raise ValueError("некорректное предположение")
        return value

    def to_dto(self) -> IllustrationPromptResponse:
        return IllustrationPromptResponse(
            moment_id=self.moment_id,
            scene_composition_en=self.scene_composition_en,
            used_fact_ids=tuple(self.used_fact_ids),
            used_design_keys=tuple(self.used_design_keys),
            assumptions_ru=tuple(self.assumptions_ru),
            contract_version=self.contract_version,
        )


ILLUSTRATION_SCHEMA_CLASSES: tuple[type[BaseModel], ...] = (
    IllustrationExtractSchema,
    IllustrationAliasSchema,
    IllustrationDesignSchema,
    IllustrationFactScopeSchema,
    IllustrationCandidatesSchema,
    IllustrationSelectionSchema,
    IllustrationPromptSchema,
)

__all__ = [
    "ILLUSTRATION_SCHEMA_CLASSES",
    "IllustrationAliasDecisionSchema",
    "IllustrationAliasSchema",
    "IllustrationCandidateEventSchema",
    "IllustrationCandidatesSchema",
    "IllustrationDesignSchema",
    "IllustrationDesignTraitSchema",
    "IllustrationExtractSchema",
    "IllustrationFactSchema",
    "IllustrationFactScopeSchema",
    "IllustrationMentionSchema",
    "IllustrationPromptSchema",
    "IllustrationQuoteAnchorSchema",
    "IllustrationSelectionReasonSchema",
    "IllustrationSelectionSchema",
]
