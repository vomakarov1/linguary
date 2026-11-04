"""Application contract for corrections of an explicitly selected block group."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from linguary.application.llm_dto import (
    BlockResultResponse, GlossaryHintResponse, LanguageNoteResponse, LLMPayloadError,
)
from linguary.domain import BlockTreatment
from linguary.domain.book_conditions import freeze, thaw
from linguary.domain.correction import CORRECTION_POLICY_VERSION


CORRECTION_VERSIONS = {'context': 1, 'schema': 1, 'validation': 1, 'grouping': 1}
CORRECTION_PARAMETERS = frozenset({
    'temperature', 'max_output_tokens', 'timeout_seconds', 'max_attempts',
    'max_unit_source_chars', 'input_price_per_mtok', 'output_price_per_mtok',
})


@dataclass(frozen=True, slots=True)
class CorrectionSnapshot:
    """DTO-01: independent, immutable correction role for a new processing run."""

    model: str
    route: Mapping
    parameters: Mapping
    prompt: str
    versions: Mapping
    required: bool = True
    policy_version: str = CORRECTION_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.required is not True or self.policy_version != CORRECTION_POLICY_VERSION:
            raise LLMPayloadError('неподдерживаемая политика исправления главы')
        if not isinstance(self.model, str) or not self.model.strip():
            raise LLMPayloadError('модель исправления не задана')
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise LLMPayloadError('инструкция исправления пуста')
        if (not isinstance(self.versions, Mapping) or dict(self.versions) != CORRECTION_VERSIONS
                or any(type(value) is not int for value in self.versions.values())):
            raise LLMPayloadError('неподдерживаемые версии исправления главы')
        if not isinstance(self.parameters, Mapping) or set(self.parameters) != CORRECTION_PARAMETERS:
            raise LLMPayloadError('неполные параметры исправления главы')
        if not isinstance(self.route, Mapping) or set(self.route) != {'provider', 'base_url', 'proxy', 'route_id'}:
            raise LLMPayloadError('неполный маршрут исправления главы')
        for name in ('route', 'parameters', 'versions'):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    def to_payload(self) -> dict:
        return {name: thaw(getattr(self, name)) for name in self.__dataclass_fields__}

    @classmethod
    def from_payload(cls, payload: Mapping) -> CorrectionSnapshot:
        if not isinstance(payload, Mapping) or set(payload) != set(cls.__dataclass_fields__):
            raise LLMPayloadError('неполный снимок исправления главы')
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class UnchangedBlockResponse:
    block_id: str
    status: Literal['unchanged'] = field(default='unchanged', init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id.strip():
            raise LLMPayloadError('исправление не привязано к блоку')


@dataclass(frozen=True, slots=True)
class CorrectedBlockResponse:
    block_id: str
    translated_text: str | None
    glossary_hints: tuple[GlossaryHintResponse, ...]
    language_notes: tuple[LanguageNoteResponse, ...]
    status: Literal['corrected'] = field(default='corrected', init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id.strip():
            raise LLMPayloadError('исправление не привязано к блоку')
        if self.translated_text is not None and (
            not isinstance(self.translated_text, str) or not self.translated_text.strip()
        ):
            raise LLMPayloadError('исправленный перевод должен быть непустым текстом или null')
        object.__setattr__(self, 'glossary_hints', tuple(self.glossary_hints))
        object.__setattr__(self, 'language_notes', tuple(self.language_notes))
        if any(not isinstance(hint, GlossaryHintResponse) for hint in self.glossary_hints):
            raise LLMPayloadError('неверный тип исправленной словарной подсказки')
        if any(not isinstance(note, LanguageNoteResponse) or note.source_block_id != self.block_id
               for note in self.language_notes):
            raise LLMPayloadError('исправленная заметка относится к чужому блоку')

    def validate_treatment(self, treatment: BlockTreatment) -> None:
        """Use the saved treatment; the model cannot select one."""
        if treatment.is_textual:
            BlockResultResponse(self.block_id, treatment, self.translated_text, self.glossary_hints)
        elif self.translated_text is not None or self.glossary_hints:
            raise LLMPayloadError('нетекстовый блок не допускает перевода и подсказок')
        if self.language_notes and not treatment.allows_language_notes:
            raise LLMPayloadError('режим блока не допускает языковых заметок')


type CorrectionBlockResponse = UnchangedBlockResponse | CorrectedBlockResponse


@dataclass(frozen=True, slots=True)
class ChapterCorrectionResponse:
    blocks: tuple[CorrectionBlockResponse, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'blocks', tuple(self.blocks))
        if any(not isinstance(block, UnchangedBlockResponse | CorrectedBlockResponse)
               for block in self.blocks):
            raise LLMPayloadError('неверный тип исправления блока')
        if len(set(self.block_ids)) != len(self.blocks):
            raise LLMPayloadError('ответ исправления содержит повторяющиеся блоки')

    @property
    def block_ids(self) -> tuple[str, ...]:
        return tuple(block.block_id for block in self.blocks)

    def validate_coverage(self, block_ids: Sequence[str]) -> None:
        expected = tuple(block_ids)
        if len(set(expected)) != len(expected) or set(expected) != set(self.block_ids):
            raise LLMPayloadError('ответ исправления должен содержать ровно все переданные блоки')

    def validate_for(self, treatments: Mapping[str, BlockTreatment]) -> None:
        self.validate_coverage(tuple(treatments))
        for block in self.blocks:
            if isinstance(block, CorrectedBlockResponse):
                block.validate_treatment(treatments[block.block_id])
