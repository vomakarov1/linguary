"""Internal DTOs for LLM responses (DTO-02).

This is the representation of model responses after the adapter parses and validates them
against Pydantic schemas at its boundary. The application layer operates solely with these
dataclasses and knows nothing about Pydantic, JSON, or provider specifics (ADR § 3).

DTOs do not duplicate domain rules: checks for group completeness, block ordering,
treatment matching saved plan, language, and truncated output are performed on these DTOs
by application validators (PRD § 15, ALG-04). Here we only enforce structural integrity:
non-empty IDs and texts, and valid confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from linguary.application.errors import ApplicationError
from linguary.domain import BlockTreatment, ConceptType
from linguary.domain.content_plan import ContextFact, Difficulty, RequiredPoint
from linguary.domain.errors import DomainError
from linguary.domain.name_translation import NameTranslationAssessment


class LLMPayloadError(ApplicationError):
    """Model response cannot be represented as an internal DTO.

    Boundary error: the adapter has already validated the response against its schema,
    so this error indicates a divergence between adapter schema and this contract,
    rather than an ordinary invalid model response.
    """


@dataclass(frozen=True, slots=True)
class GlossaryCandidateResponse:
    """Glossary candidate from chapter analysis (PRD § 11).

    `model_flagged_review` — model's own flag, input signal. Final `requires_review`
    is calculated by the application per DTO-01 (`domain.compute_requires_review`).

    `note` — translation guidance that cannot be expressed as a single translation variant:
    character gender in target language, declension of proper names, speech register (PRD § 11).
    """

    source_text: str
    type: ConceptType
    suggested_translation: str
    confidence: float
    model_flagged_review: bool
    note: str = ""
    name_translation_assessment: NameTranslationAssessment | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.source_text.strip():
            raise LLMPayloadError("исходный текст кандидата глоссария пуст")
        if not self.suggested_translation.strip():
            raise LLMPayloadError(
                f"предложенный перевод кандидата {self.source_text!r} пуст"
            )
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise LLMPayloadError(
                f"уверенность модели вне диапазона 0..1: {self.confidence}"
            )


@dataclass(frozen=True, slots=True)
class SegmentTreatmentResponse:
    """Textual treatment of one segment. `index` starts at 1."""

    index: int
    treatment: BlockTreatment

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 1:
            raise LLMPayloadError("индекс сегмента начинается с 1")
        if not self.treatment.is_textual:
            raise LLMPayloadError(f"режим сегмента {self.index} должен быть текстовым")


@dataclass(frozen=True, slots=True)
class TreatmentPlanItemResponse:
    """Model-suggested treatment mode for a single block (PRD § 10.1).

    The plan becomes the source of truth only after persistence: during processing,
    the `treatment` field in model response confirms this plan rather than overriding it.

    `segments` is empty when `treatment` applies to the whole block.
    """

    source_block_id: str
    treatment: BlockTreatment
    segments: tuple[SegmentTreatmentResponse, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))
        if not self.source_block_id.strip():
            raise LLMPayloadError("элемент плана обработки не привязан к блоку")
        if not self.treatment.is_textual:
            raise LLMPayloadError(
                f"режим {self.treatment} блока {self.source_block_id} не выбирается моделью: "
                "нетекстовые блоки классифицирует импортёр"
            )


@dataclass(frozen=True, slots=True)
class BeforeYouReadEntryResponse:
    """Chapter expression where a simple dictionary hint might be insufficient (PRD § 19.1)."""

    expression: str
    explanation: str

    def __post_init__(self) -> None:
        if not self.expression.strip():
            raise LLMPayloadError("выражение материала Before You Read пусто")
        if not self.explanation.strip():
            raise LLMPayloadError(f"объяснение выражения {self.expression!r} пусто")


@dataclass(frozen=True, slots=True)
class ChapterAnalysisResponse:
    """Model response for analysis of a single chapter or fragment (PRD § 14).

    `chapter_summary` describes events in this chapter and is written in the original language:
    the exporter displays it to the reader before the next chapter (PRD § 19.2).
    """

    translation_instructions: str
    chapter_summary: str
    glossary_candidates: tuple[GlossaryCandidateResponse, ...] = ()
    treatment_plan: tuple[TreatmentPlanItemResponse, ...] = ()
    before_you_read: tuple[BeforeYouReadEntryResponse, ...] = ()

    content_plan: ContentPlanResponse | QuotedContentPlanResponse | None = None

    def __post_init__(self) -> None:
        if not self.translation_instructions.strip():
            raise LLMPayloadError("инструкции перевода главы пусты")
        if not self.chapter_summary.strip():
            raise LLMPayloadError("сводка событий главы пуста")

    @property
    def planned_block_ids(self) -> tuple[str, ...]:
        """Blocks covered by plan in model response order.

        Completeness of coverage with respect to chapter textual blocks is validated
        by the application layer upon saving analysis: the response itself does not know
        the chapter structure.
        """
        return tuple(item.source_block_id for item in self.treatment_plan)


@dataclass(frozen=True, slots=True)
class GlossaryHintResponse:
    """Translation of an individual word or short phrase inside a block (PRD § 10.1)."""

    original: str
    translation: str
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.original.strip():
            raise LLMPayloadError("исходное выражение словарной подсказки пусто")
        if not self.translation.strip():
            raise LLMPayloadError(f"перевод подсказки {self.original!r} пуст")


@dataclass(frozen=True, slots=True)
class BlockResultResponse:
    """Processing result for a single input textual block.

    Composition depends on treatment: full translation is mandatory and allowed only for
    `TRANSLATE`, glossary hints are mandatory for `GLOSSARY_ONLY` and forbidden
    for `READ_AS_IS` (PRD § 15).
    """

    source_block_id: str
    treatment: BlockTreatment
    translated_text: str | None = None
    glossary_hints: tuple[GlossaryHintResponse, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_block_id.strip():
            raise LLMPayloadError("результат обработки не привязан к блоку")
        if not self.treatment.is_textual:
            raise LLMPayloadError(
                f"блок {self.source_block_id} с режимом {self.treatment} модели не передаётся"
            )
        has_translation = bool(self.translated_text and self.translated_text.strip())
        if self.treatment.requires_translation and not has_translation:
            raise LLMPayloadError(
                f"для блока {self.source_block_id} обязателен полный перевод"
            )
        if not self.treatment.allows_translation and self.translated_text is not None:
            raise LLMPayloadError(
                f"для блока {self.source_block_id} с режимом {self.treatment} "
                "полный перевод недопустим"
            )
        if self.treatment.requires_glossary_hints and not self.glossary_hints:
            raise LLMPayloadError(
                f"для блока {self.source_block_id} обязательны словарные подсказки"
            )
        if not self.treatment.allows_glossary_hints and self.glossary_hints:
            raise LLMPayloadError(
                f"для блока {self.source_block_id} с режимом {self.treatment} "
                "словарные подсказки недопустимы"
            )


@dataclass(frozen=True, slots=True)
class LanguageNoteResponse:
    """Language note tied to exactly one block (PRD § 19.3)."""

    source_block_id: str
    original_excerpt: str
    note: str
    translated_excerpt: str | None = None

    def __post_init__(self) -> None:
        if not self.source_block_id.strip():
            raise LLMPayloadError("языковая заметка не привязана к блоку")
        if not self.original_excerpt.strip():
            raise LLMPayloadError(
                f"фрагмент оригинала заметки блока {self.source_block_id} пуст"
            )
        if not self.note.strip():
            raise LLMPayloadError(f"текст заметки блока {self.source_block_id} пуст")


@dataclass(frozen=True, slots=True, init=False)
class PronunciationAnchorResponse:
    block_id: str
    quote: str
    start: int | None = None
    end: int | None = None

    def __init__(
        self,
        block_id: str,
        quote_or_start: str | int | None = None,
        end_or_quote: int | str | None = None,
        quote: str | None = None,
        *,
        start: int | None = None,
        end: int | None = None,
    ):
        object.__setattr__(self, "block_id", block_id)
        if isinstance(quote_or_start, int):
            actual_start = quote_or_start
            actual_end = end_or_quote if isinstance(end_or_quote, int) else None
            actual_quote = quote if quote is not None else ""
        elif isinstance(quote_or_start, str):
            actual_quote = quote_or_start
            if isinstance(end_or_quote, int):
                actual_start = end_or_quote
                actual_end = quote if isinstance(quote, int) else end
            else:
                actual_start = start
                actual_end = end
        else:
            actual_quote = quote if quote is not None else ""
            actual_start = start
            actual_end = end
        object.__setattr__(self, "quote", actual_quote)
        object.__setattr__(self, "start", actual_start)
        object.__setattr__(self, "end", actual_end)


@dataclass(frozen=True, slots=True)
class PronunciationContextResponse:
    meaning_in_context: str | None
    part_of_speech: str | None
    grammatical_form: str | None
    is_proper_name: bool


@dataclass(frozen=True, slots=True)
class PronunciationCandidateResponse:
    anchor: PronunciationAnchorResponse
    form: str
    context: PronunciationContextResponse
    selection_reason: str


@dataclass(frozen=True, slots=True)
class TranslationResultResponse:
    """Model response for processing a single block group (PRD § 10, § 14).

    Matching of block composition and order to the input group is verified by the application layer:
    the response on its own does not know which group requested it.
    """

    blocks: tuple[BlockResultResponse, ...] = ()
    language_notes: tuple[LanguageNoteResponse, ...] = ()
    content_evidence: tuple[HelpEvidenceResponse, ...] = ()
    fragment_assistances: tuple[FragmentAssistanceResponse, ...] = ()
    pronunciation_candidates: tuple[PronunciationCandidateResponse, ...] = ()

    @property
    def block_ids(self) -> tuple[str, ...]:
        """Result blocks in model response order."""
        return tuple(block.source_block_id for block in self.blocks)

    def notes_for(self, block_id: str) -> tuple[LanguageNoteResponse, ...]:
        """Response notes pertaining to the specified block."""
        return tuple(note for note in self.language_notes if note.source_block_id == block_id)


__all__ = [
    "ContentPlanResponse",
    "QuotedSceneVisualFactResponse",
    "QuotedSceneResponse",
    "NarrativeAnalysisResponse",
    "PedagogicalAnalysisResponse",
    "HelpEvidenceResponse",
    "BeforeYouReadEntryResponse",
    "BlockResultResponse",
    "ChapterAnalysisResponse",
    "GlossaryCandidateResponse",
    "GlossaryHintResponse",
    "LLMPayloadError",
    "LanguageNoteResponse",
    "SegmentTreatmentResponse",
    "TranslationResultResponse",
    "TreatmentPlanItemResponse",
]


@dataclass(frozen=True, slots=True)
class HelpEvidenceResponse:
    difficulty_id: str
    point_id: str
    artifact_kind: str
    source_block_id: str
    artifact_index: int
    body_excerpt: str

    def __post_init__(self) -> None:
        if not all(v.strip() for v in (self.difficulty_id, self.point_id, self.source_block_id, self.body_excerpt)) or self.artifact_index < 0:
            raise LLMPayloadError("свидетельство требует аспект, индекс и непустое тело")
        if self.artifact_kind not in {"before_you_read.explanation", "language_note.note", "glossary_hint.note", "glossary_hint.translation"}:
            raise LLMPayloadError("неизвестный канал помощи")


@dataclass(frozen=True, slots=True)
class ContentPlanResponse:
    difficulties: tuple[Difficulty, ...] = ()
    before_you_read_evidence: tuple[HelpEvidenceResponse, ...] = ()
    version: int = 2
    profile: str = "content-plan-v2"
    context_facts: tuple[ContextFact, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "difficulties", tuple(self.difficulties))
        object.__setattr__(self, "before_you_read_evidence", tuple(self.before_you_read_evidence))
        object.__setattr__(self, 'context_facts', tuple(self.context_facts))
        if self.version == 2 and self.context_facts:
            raise LLMPayloadError('контекстные факты требуют версию 3')
        if not ((self.version == 2 and self.profile == 'content-plan-v2')
                or (type(self.version) is int and self.version == 3 and self.profile == 'content-plan-v3')):
            raise LLMPayloadError("неизвестная версия содержательного плана")
        ids = [d.id for d in self.difficulties]
        if len(ids) != len(set(ids)):
            raise LLMPayloadError("повторяющийся идентификатор трудности")


@dataclass(frozen=True, slots=True)
class QuotedSourceAnchorResponse:
    """Unresolved, exact quote returned by the v5 analysis contract."""

    block_id: str
    quote: str

    def __post_init__(self) -> None:
        if not self.block_id.strip() or not self.quote.strip():
            raise LLMPayloadError('цитата требует видимый блок и непустой текст')


@dataclass(frozen=True, slots=True)
class QuotedDifficultyResponse:
    id: str
    anchors: tuple[QuotedSourceAnchorResponse, ...]
    owner_block_id: str
    kind: str
    priority: str
    help_kind: str
    required_points: tuple[RequiredPoint, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'anchors', tuple(self.anchors))
        object.__setattr__(self, 'required_points', tuple(self.required_points))


@dataclass(frozen=True, slots=True)
class QuotedContextFactResponse:
    id: str
    kind: str
    anchors: tuple[QuotedSourceAnchorResponse, ...]
    subject_anchor: QuotedSourceAnchorResponse
    value: str | None = None
    certainty: str = 'unknown'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'anchors', tuple(self.anchors))


@dataclass(frozen=True, slots=True)
class QuotedSceneVisualFactResponse:
    anchor: QuotedSourceAnchorResponse
    subject: str
    detail: str


@dataclass(frozen=True, slots=True)
class QuotedSceneResponse:
    start_block_id: str
    end_block_id: str
    summary: str
    location: str | None = None
    participants: tuple[str, ...] = ()
    visual_facts: tuple[QuotedSceneVisualFactResponse, ...] = ()
    visual_assumptions: tuple[str, ...] = ()
    continues_previous_scene: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "participants", tuple(self.participants))
        object.__setattr__(self, "visual_facts", tuple(self.visual_facts))
        object.__setattr__(self, "visual_assumptions", tuple(self.visual_assumptions))


@dataclass(frozen=True, slots=True)
class NarrativeAnalysisResponse:
    translation_instructions: str
    chapter_summary: str
    glossary_candidates: tuple[GlossaryCandidateResponse, ...] = ()
    context_facts: tuple[QuotedContextFactResponse, ...] = ()
    scenes: tuple[QuotedSceneResponse, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "glossary_candidates", tuple(self.glossary_candidates))
        object.__setattr__(self, "context_facts", tuple(self.context_facts))
        object.__setattr__(self, "scenes", tuple(self.scenes))


@dataclass(frozen=True, slots=True)
class PedagogicalAnalysisResponse:
    treatment_plan: tuple[TreatmentPlanItemResponse, ...] = ()
    before_you_read: tuple[BeforeYouReadEntryResponse, ...] = ()
    difficulties: tuple[QuotedDifficultyResponse, ...] = ()
    before_you_read_evidence: tuple[HelpEvidenceResponse, ...] = ()
    reader_chapter_summary: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "treatment_plan", tuple(self.treatment_plan))
        object.__setattr__(self, "before_you_read", tuple(self.before_you_read))
        object.__setattr__(self, "difficulties", tuple(self.difficulties))
        object.__setattr__(self, "before_you_read_evidence", tuple(self.before_you_read_evidence))


@dataclass(frozen=True, slots=True)
class QuotedContentPlanResponse:
    """v3 plan whose source references have not yet been resolved."""

    difficulties: tuple[QuotedDifficultyResponse, ...] = ()
    before_you_read_evidence: tuple[HelpEvidenceResponse, ...] = ()
    version: int = 3
    profile: str = 'content-plan-v3'
    context_facts: tuple[QuotedContextFactResponse, ...] = ()
    scenes: tuple[QuotedSceneResponse, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'difficulties', tuple(self.difficulties))
        object.__setattr__(self, 'before_you_read_evidence', tuple(self.before_you_read_evidence))
        object.__setattr__(self, 'context_facts', tuple(self.context_facts))
        object.__setattr__(self, 'scenes', tuple(self.scenes))
        if type(self.version) is not int or self.version != 3 or self.profile != 'content-plan-v3':
            raise LLMPayloadError('неизвестная версия содержательного плана')
        ids = [difficulty.id for difficulty in self.difficulties]
        if len(ids) != len(set(ids)):
            raise LLMPayloadError('повторяющийся идентификатор трудности')


@dataclass(frozen=True, slots=True)
class FragmentAssistanceResponse:
    difficulty_id: str
    point_ids: tuple[str, ...]
    kind: str
    text: str | None

    def __post_init__(self):
        object.__setattr__(self, 'point_ids', tuple(self.point_ids))
        from linguary.domain.learning import validate_assistance_body
        try:
            validate_assistance_body(self.kind, self.text)
        except DomainError as error:
            raise LLMPayloadError(str(error)) from error
        if not self.difficulty_id.strip() or not all(p.strip() for p in self.point_ids) or len(set(self.point_ids)) != len(self.point_ids):
            raise LLMPayloadError('неверные идентификаторы помощи')
