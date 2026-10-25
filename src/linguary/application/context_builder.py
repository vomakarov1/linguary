"""Context builder for model requests (PRD § 13, ADR § 3).

`ContextBuilder` is an application policy that determines exactly what the model sees. The
`LLMClient` port knows nothing of chapters, glossary, or blocks: it receives ready instruction
text and ready prompt input text.

Chapter analysis context consists of the textual blocks of that chapter and already approved
decisions of the book — glossary candidates from previous chapters. Text of neighbouring chapters
is not included: chapter analysis depends only on itself and accumulated candidates, so chapters
are analyzed strictly sequentially (PRD § 13, § 25). `PRESERVE_ONLY` and `DROP_WITH_WARNING`
blocks do not enter main or neighbouring context: they remain the responsibility of the importer
and exporter (ADR § 3).

Block group processing context consists of the group itself, the run glossary snapshot,
chapter translation instructions, and the chapter summary — a digest of that chapter's analysis.
Processing results of other groups are not included: this keeps tasks independent and results
reproducible (PRD § 13). Mandatory translation decisions are separated in context from optional
suggestions: approved, corrected, or pinned user decisions must be respected by the model, while
model variants without user decision and skipped items are suggestions (PRD § 11-12). User decisions
override chapter instructions: any instruction item that contradicts them is excluded from context
(`chapter_instructions`), and the context explicitly prioritizes decisions over instructions.

Block regeneration context differs from group context in two ways: it contains a single block
surrounded by several neighbouring paragraphs of the chapter. Neighbours provide scene context;
they do not receive aliases, so the model cannot return results for them (PRD § 13, § 21.5).

Blocks are aliased in prompts with short identifiers `b1..bN` instead of stable UUIDs: in a chapter
with hundreds of paragraphs, UUID keys occupy significant portions of requests and responses, and
the model must echo each verbatim. Aliases cost only a few characters, transfer reliably, and live
for exactly one request — the reverse mapping is held here in `ChapterAnalysisContext`.

Instructions are written in Russian: the retry repair instruction (`retry_policy.build_repair_instruction`)
is written in the same language and is appended to this text.
"""

from __future__ import annotations

import re
import json
from dataclasses import asdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

from linguary.application.analysis_merge import ChapterFragment
from linguary.application.block_labels import without_block_labels
from linguary.application.chapter_instructions import (
    instructions_respecting_glossary,
    note_respecting_glossary,
)
from linguary.application.errors import ApplicationError
from linguary.application.reader_text import learner_instructions
from linguary.application.llm_dto import ChapterAnalysisResponse, TranslationResultResponse
from linguary.application.ports.llm import LLMRequest
from linguary.domain.content_plan import ContentPlan, artifact_key
from linguary.domain import BeforeYouReadEntry
from linguary.application.retry_policy import (
    ExpectedBlock,
    ResultExpectation,
    ResultValidationError,
)
from linguary.domain import (
    BlockType,
    CanonicalDocument,
    ChapterAnalysis,
    DocumentBlock,
    DocumentSection,
    GlossaryCandidate,
    GlossarySnapshot,
    GlossarySnapshotEntry,
    InlineAnnotationType,
    SupportStatus,
    TranslationUnit,
    TreatmentPlan,
    footnote_marker,
    footnote_references,
    glossary_key,
)

#: Maximum number of accumulated decisions from previous chapters passed into chapter analysis context.
#: By the end of a novel, candidates can number in the hundreds, whereas requests only need those
#: that actually recur: the list is capped to prevent context from ballooning with book size
#: (PRD § 13: context limit is configurable).
DEFAULT_MAX_ACCUMULATED_CANDIDATES = 200

#: Number of neighbouring textual blocks on each side seen by the model during single block regeneration.
#: PRD § 13 specifies "several neighbouring paragraphs": they provide scene context without turning
#: a single-block request into a full chapter request.
DEFAULT_REGENERATION_CONTEXT_BLOCKS = 2

#: Block types that are non-textual for reading even when fully supported by the importer.
#: An image is fully supported, but offers an image to the reader rather than text.
_NON_TEXTUAL_BLOCK_TYPES = frozenset({BlockType.IMAGE, BlockType.PAGE_BREAK})


CONTENT_PLAN_PROMPT_VERSION = "content-plan-v2.1"
_CONTENT_PLAN_ANALYSIS = """
Профиль content-plan-v2, инструкция content-plan-v2.1.
Сначала ищи игру слов, смысловые и логические связи, непрозрачную грамматику;
обычная лексика не должна вытеснять эти трудности. Выбирай минимальную помощь.
Верни content_plan с version=2, profile=content-plan-v2, difficulties и
before_you_read_evidence (пустые списки обязательны, если трудностей нет).
Каждая трудность: id, anchors [{block_id, start, end, quote}], owner_block_id,
kind, priority, help_kind, required_points [{id, description}]. Смещения — Unicode
кодовые точки точного исходного текста, end исключён; quote=text[start:end].
Копируй quote ДОСЛОВНО из текста блока, не раскрывай и не меняй сокращения (например, she’s нельзя заменять на she is).
Владелец один, из anchors. Ссылайся только на видимые блоки по b1..bN.
Все включённые аспекты обязательны; не навязывай перевод вопреки глоссарию.
lexical_hint допустим ТОЛЬКО для блоков GLOSSARY_ONLY или при полном покрытии в BYR.
Для блоков TRANSLATE и READ_AS_IS используй help_kind=explanation.
explanation требует объясняющего BYR, языковой заметки либо содержательного glossary_hint.note.
Перевод, цитата и пересказ не заменяют объяснение. Не повышай квоты.
Если BYR уже покрывает аспект, не требуй лишней заметки. Для каждого такого аспекта
верни свидетельство: difficulty_id, point_id, artifact_kind=before_you_read.explanation,
source_block_id владельца, artifact_index, body_excerpt.
artifact_index — строго порядковый номер (начиная с 0) той записи в before_you_read, в чьём explanation находится body_excerpt.
body_excerpt обязан быть точной подстрокой из русского текста explanation соответствующего элемента before_you_read[artifact_index], не цитатой оригинала.
Не выдумывай связи с невидимыми частями. Пустой план допустим.
"""
_CONTENT_PLAN_PROCESSING = """
Профиль content-plan-v2, инструкция content-plan-v2.1.
Выполни все required_points назначенных владельцам трудностей минимальной помощью
в пределах прежних режимов и бюджета. Решения глоссария выше формулировок помощи;
объясни смысл, сохраняя согласованный перевод. Не повторяй уже покрытый BYR аспект.
Верни content_evidence (включая пустой список): difficulty_id, point_id,
artifact_kind (before_you_read.explanation, language_note.note, glossary_hint.note;
для lexical_hint также glossary_hint.translation), source_block_id владельца,
artifact_index с нуля в соответствующем списке (подсказки внутри блока), body_excerpt.
Отрывок должен дословно входить в реальное тело помощи, а не только цитировать
оригинал. Перевод и пересказ не считаются объяснением. Соседние anchors служат
контекстом, не требуют результата. Свидетельства показывают только структуру,
смысловая правильность отдельно не проверена.
"""


_CONTENT_PLAN_ANALYSIS_V3 = _CONTENT_PLAN_ANALYSIS.replace('content-plan-v2.1', 'learner-quality-v3.1').replace('content-plan-v2', 'content-plan-v3').replace('version=2', 'version=3') + """
Верни context_facts: id, kind (speaker, gender, pronoun_referent), anchors,
subject_anchor, value, certainty (explicit, inferred, unknown).
Каждая привязка содержит видимый block_id, точные start/end и дословную quote.
Не исправляй цитату и смещения; subject_anchor проверяется отдельно от anchors.
Не распространяй факт за видимый фрагмент. При certainty=unknown значение не задано (value=null).
Различай подтверждённый текстом факт и предположение; противоречащие атрибуции не выбирай молча.
"""
_CONTENT_PLAN_PROCESSING_V3 = _CONTENT_PLAN_PROCESSING.replace('content-plan-v2.1', 'learner-quality-v3.1').replace('content-plan-v2', 'content-plan-v3') + """
Служебные context_facts применяй только к назначенным субъектам и их точным исходным привязкам.
Не выдавай inferred/unknown за подтверждённый факт; решения глоссария имеют приоритет.
Не превращай эти служебные сведения в обязательную заметку читателю.
"""


class ContextBuildError(ApplicationError):
    """Request context cannot be assembled."""


@dataclass(frozen=True, slots=True)
class FragmentFindings:
    """Findings in previously processed parts of the chapter."""

    summary: str
    instructions: str
    expressions: tuple[str, ...]


def is_textual(block: DocumentBlock) -> bool:
    """Whether the block is passed to the model as part of chapter text (PRD § 10.1).

    The model receives what the reader reads: a fully supported block with non-empty
    text. Everything else — preserved opaque structures (`PRESERVE_OPAQUE`), images,
    page breaks, empty blocks — remains the job of the importer and exporter.
    """
    return (
        block.support_status is SupportStatus.SUPPORTED
        and block.type not in _NON_TEXTUAL_BLOCK_TYPES
        and bool(block.text.strip())
    )


def textual_blocks(section: DocumentSection) -> tuple[DocumentBlock, ...]:
    """Chapter textual blocks in original document order."""
    return tuple(block for block in section.ordered_blocks if is_textual(block))


def notes_budget(*, unit_chars: int, chapter_chars: int, per_chapter: int) -> int:
    """Number of language notes allocated to the portion of the chapter in the request."""
    if chapter_chars <= 0:
        raise ContextBuildError(
            "длина главы для расчёта заметок должна быть положительной"
        )
    return max(1, round(per_chapter * unit_chars / chapter_chars))


class AliasedBlocksContext:
    """Common base for contexts that address blocks via aliases `b1..bN`.

    The alias mapping is populated by subclasses: here we only handle parsing
    the model response, identical for both chapter analysis and block group processing.
    """

    __slots__ = ()

    block_ids_by_alias: Mapping[str, str]

    @property
    def aliases(self) -> tuple[str, ...]:
        """Aliases of provided blocks in original document order."""
        return tuple(self.block_ids_by_alias)

    def block_id_for(self, alias: str) -> str:
        """Stable block identifier for an alias returned by the model."""
        block_id = self.block_ids_by_alias.get(alias)
        if block_id is None:
            raise ContextBuildError(f"блок {alias} в запрос не передавался")
        return block_id


@dataclass(frozen=True, slots=True)
class ChapterAnalysisContext(AliasedBlocksContext):
    """Ready chapter analysis request along with metadata for parsing response.

    `block_ids_by_alias` maps model response aliases back to stable block IDs,
    `expectation` describes expected output as seen by the model: retry repair
    instructions list aliases rather than internal UUIDs.
    """

    section_id: str
    request: LLMRequest[ChapterAnalysisResponse]
    block_ids_by_alias: Mapping[str, str]
    expectation: ResultExpectation
    before_you_read_limit: int = 10
    language_notes_per_chapter: int = 8
    note_limits_by_block: Mapping[str, int] | None = None
    note_group_limits: tuple[tuple[tuple[str, ...], int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "block_ids_by_alias", MappingProxyType(dict(self.block_ids_by_alias))
        )
        if self.note_limits_by_block is not None:
            object.__setattr__(self, "note_limits_by_block", MappingProxyType(dict(self.note_limits_by_block)))


@dataclass(frozen=True, slots=True)
class TranslationContext(AliasedBlocksContext):
    """Ready request for processing a single block group (PRD § 13, ALG-04).

    `expectation` lists aliases along with treatments from the saved plan:
    retry repair instructions identify blocks as the model saw them and repeat
    the required treatment (ALG-05).

    `explained_expressions` — `Before You Read` expressions for this chapter: passed
    in the prompt to prevent duplication, with use case filtering out repeats if returned (PRD § 19.3).
    `explained_entries` also retains full explanations to compare note contents so that
    independent new linguistic facts about the same excerpt are not lost.
    """

    section_id: str
    unit_id: str
    request: LLMRequest[TranslationResultResponse]
    block_ids_by_alias: Mapping[str, str]
    expectation: ResultExpectation
    explained_expressions: tuple[str, ...] = ()
    explained_entries: tuple[tuple[str, str], ...] = ()
    content_plan: ContentPlan | None = None
    before_you_read: tuple[BeforeYouReadEntry, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "block_ids_by_alias", MappingProxyType(dict(self.block_ids_by_alias))
        )

    def to_aliases(self, message: str) -> str:
        """Validation message with internal block IDs replaced by prompt aliases.

        Rejection reasons are sent to the model with repair instructions (ALG-05),
        while validation checks format messages using internal block IDs the model never saw.
        """
        for alias, block_id in self.block_ids_by_alias.items():
            message = message.replace(block_id, alias)
        return message

    def to_block_ids(
        self, payload: TranslationResultResponse
    ) -> TranslationResultResponse:
        """Model response translated from prompt aliases back to group block IDs.

        An alias absent from the request indicates a result for a foreign block (PRD § 15).
        Composition checks operate on group block IDs, so such responses are rejected
        here with the same expectation as other blocking checks: retries get block lists
        rather than bare messages.
        """
        unknown = sorted(
            {
                alias
                for alias in (*payload.block_ids, *(
                    note.source_block_id for note in payload.language_notes
                ), *(e.source_block_id for e in payload.content_evidence))
                if alias not in self.block_ids_by_alias
            }
        )
        if unknown:
            raise ResultValidationError(
                f"ответ группы {self.unit_id} относится к блокам вне запроса: {unknown}",
                expectation=self.expectation,
            )
        return TranslationResultResponse(
            content_evidence=tuple(replace(e, source_block_id=self.block_id_for(e.source_block_id)) for e in payload.content_evidence),
            blocks=tuple(
                replace(block, source_block_id=self.block_id_for(block.source_block_id))
                for block in payload.blocks
            ),
            language_notes=tuple(
                replace(note, source_block_id=self.block_id_for(note.source_block_id))
                for note in payload.language_notes
            ),
        )


@dataclass(frozen=True, slots=True)
class _NeighbourBlocks:
    """Neighbouring source text on one side: paragraphs before or after the block."""

    title: str
    blocks: tuple[DocumentBlock, ...]


class ContextBuilder:
    """Assembles prompt context according to PRD § 13 policy."""

    def __init__(
        self,
        *,
        max_accumulated_candidates: int = DEFAULT_MAX_ACCUMULATED_CANDIDATES,
        regeneration_context_blocks: int = DEFAULT_REGENERATION_CONTEXT_BLOCKS,
        language_notes_per_chapter: int = 8,
        content_plan_profile: str = "legacy-v1",
        max_unit_source_chars: int = 8000,
        reader_level: str = "b2",
        prompt_templates: Mapping[str, str] | None = None,
    ) -> None:
        if content_plan_profile not in {"legacy-v1", "content-plan-v2", "content-plan-v3"}:
            raise ContextBuildError("неизвестный профиль содержательного плана")
        if max_unit_source_chars < 1:
            raise ContextBuildError("предел размера группы должен быть положительным")
        self._max_unit_source_chars = max_unit_source_chars
        self._content_plan_profile = content_plan_profile
        self._reader_level = reader_level.strip().lower()
        self._prompt_templates = MappingProxyType(dict(prompt_templates or {}))
        if max_accumulated_candidates < 0:
            raise ContextBuildError(
                "ограничение числа накопленных кандидатов не может быть отрицательным"
            )
        if regeneration_context_blocks < 0:
            raise ContextBuildError(
                "число соседних блоков контекста не может быть отрицательным"
            )
        if language_notes_per_chapter < 0:
            raise ContextBuildError(
                "число языковых заметок на главу не может быть отрицательным"
            )
        self._max_accumulated_candidates = max_accumulated_candidates
        self._regeneration_context_blocks = regeneration_context_blocks
        self._language_notes_per_chapter = language_notes_per_chapter

    def for_chapter_analysis(
        self,
        *,
        document: CanonicalDocument,
        section: DocumentSection,
        accumulated_candidates: Iterable[GlossaryCandidate] = (),
    ) -> ChapterAnalysisContext:
        """Context for analyzing a single chapter (PRD § 11, § 13, § 14, § 19).

        `accumulated_candidates` — glossary candidates from previous chapters in reading order.
        Text of neighbouring chapters is not passed.
        """
        blocks = textual_blocks(section)
        if not blocks:
            raise ContextBuildError(
                f"глава {section.id} не содержит текстовых блоков: анализировать нечего"
            )
        return self._chapter_context(
            document=document,
            section=section,
            blocks=blocks,
            accumulated_candidates=accumulated_candidates,
            instructions=self._prompt_templates.get("analysis") or chapter_analysis_instructions(
                source_language=document.source_language,
                target_language=document.target_language,
                reader_level=self._reader_level,
            ),
            fragment_label=None,
        )

    def for_chapter_fragment(
        self,
        *,
        document: CanonicalDocument,
        section: DocumentSection,
        fragment: ChapterFragment,
        fragment_count: int,
        accumulated_candidates: Iterable[GlossaryCandidate] = (),
        findings: FragmentFindings | None = None,
    ) -> ChapterAnalysisContext:
        """Context for analyzing a single fragment of a long chapter (PRD § 14, ALG-03).

        Fallback scenario: chapters that do not fit into a single prompt are seen in parts.
        A fragment receives only its blocks — neither the rest of the chapter nor neighbouring
        chapters are in the request. Findings from previous parts help return a cumulative
        summary from chapter start to the end of the current fragment.
        """
        if fragment_count < 1 or fragment.order >= fragment_count:
            raise ContextBuildError(
                f"фрагмент {fragment.order} не входит в главу {section.id} "
                f"из {fragment_count} фрагментов"
            )
        return self._chapter_context(
            document=document,
            section=section,
            blocks=fragment.blocks,
            accumulated_candidates=accumulated_candidates,
            instructions=self._prompt_templates.get("fragment") or chapter_fragment_instructions(
                source_language=document.source_language,
                target_language=document.target_language,
                reader_level=self._reader_level,
            ),
            before_you_read_limit=10 // fragment_count + (fragment.order < 10 % fragment_count),
            language_notes_limit=self._language_notes_per_chapter // fragment_count + (fragment.order < self._language_notes_per_chapter % fragment_count),
            fragment_label=_fragment_label(fragment.order, fragment_count),
            findings=findings if fragment.order > 0 else None,
        )

    def for_translation_unit(
        self,
        *,
        document: CanonicalDocument,
        section: DocumentSection,
        unit: TranslationUnit,
        analysis: ChapterAnalysis,
        glossary: GlossarySnapshot,
        book_candidates: Iterable[GlossaryCandidate] = (),
        recovery_child: bool = False,
    ) -> TranslationContext:
        """Context for processing a single block group (PRD § 10, § 13, § 19.3, ALG-04).

        The request includes the group itself with treatments from the saved plan,
        the run glossary snapshot, chapter translation instructions, and its summary.
        Results from other groups and neighbouring chapter text are excluded. `book_candidates`
        contains candidates from all analyzed chapters: user-rejected model variants are known
        from them, so instructions referencing such variants are omitted.
        """
        return self._translation_context(
            document=document,
            section=section,
            unit=unit,
            analysis=analysis,
            glossary=glossary,
            book_candidates=book_candidates,
            instructions=self._prompt_templates.get("translation") or translation_unit_instructions(
                source_language=document.source_language,
                target_language=document.target_language,
            ),
            neighbours=(),
            recovery_child=recovery_child,
        )

    def for_block_regeneration(
        self,
        *,
        document: CanonicalDocument,
        section: DocumentSection,
        unit: TranslationUnit,
        analysis: ChapterAnalysis,
        glossary: GlossarySnapshot,
        book_candidates: Iterable[GlossaryCandidate] = (),
    ) -> TranslationContext:
        """Context for regenerating a single block (PRD § 13, § 19.3, § 21.5).

        The original block group is not recomputed: the model receives the chosen block
        with its plan treatment, the same chapter digest and glossary snapshot as standard
        processing, plus several neighbouring paragraphs for scene context. Neighbours do
        not receive aliases and no results are expected for them.
        """
        if len(unit.items) != 1:
            raise ContextBuildError(
                f"повторная генерация выполняется по одному блоку, а группа {unit.id} "
                f"содержит {len(unit.items)}"
            )
        return self._translation_context(
            document=document,
            section=section,
            unit=unit,
            analysis=analysis,
            glossary=glossary,
            book_candidates=book_candidates,
            instructions=self._prompt_templates.get("regeneration") or block_regeneration_instructions(
                source_language=document.source_language,
                target_language=document.target_language,
            ),
            neighbours=self._neighbours(section, unit),
        )

    def _translation_context(
        self,
        *,
        document: CanonicalDocument,
        section: DocumentSection,
        unit: TranslationUnit,
        analysis: ChapterAnalysis,
        glossary: GlossarySnapshot,
        book_candidates: Iterable[GlossaryCandidate],
        instructions: str,
        neighbours: Sequence[_NeighbourBlocks],
        recovery_child: bool = False,
    ) -> TranslationContext:
        """Ready translation request for given blocks along with response parsing setup."""
        if analysis.content_plan is not None and self._content_plan_profile != analysis.content_plan.profile:
            raise ContextBuildError("сохранённый план требует соответствующий явный профиль")
        version = analysis.content_plan.version if analysis.content_plan is not None else 1
        if version in (2, 3):
            default = _CONTENT_PLAN_PROCESSING_V3 if version == 3 else _CONTENT_PLAN_PROCESSING
            instructions += "\n" + self._prompt_templates.get("content_processing", default)
        if version == 3:
            instructions += "\n" + self._prompt_templates.get('learner_quality', learner_instructions(self._reader_level))
        if unit.section_id != section.id:
            raise ContextBuildError(
                f"группа {unit.id} собрана из блоков главы {unit.section_id}, "
                f"а не {section.id}"
            )
        aliases = {
            _alias(index): item.block.id
            for index, item in enumerate(unit.items, start=1)
        }
        return TranslationContext(
            section_id=section.id,
            unit_id=unit.id,
            request=LLMRequest.for_translation(
                contract_version=version,
                instructions=instructions,
                input_text=self._translation_input(
                    document=document,
                    section=section,
                    unit=unit,
                    analysis=analysis,
                    glossary=glossary,
                    candidates=(*analysis.glossary_candidates, *book_candidates),
                    neighbours=neighbours,
                    recovery_child=recovery_child,
                ),
            ),
            block_ids_by_alias=aliases,
            expectation=ResultExpectation(
                target_language=document.target_language,
                blocks=tuple(
                    ExpectedBlock(block_id=alias, treatment=item.treatment)
                    for alias, item in zip(aliases, unit.items, strict=True)
                ),
            ),
            explained_expressions=_before_you_read_expressions(analysis, section),
            explained_entries=_before_you_read_entries(analysis, section),
            content_plan=analysis.content_plan,
            before_you_read=() if section.is_notes_only else analysis.before_you_read,
        )

    def _translation_input(
        self,
        *,
        document: CanonicalDocument,
        section: DocumentSection,
        unit: TranslationUnit,
        analysis: ChapterAnalysis,
        glossary: GlossarySnapshot,
        candidates: Sequence[GlossaryCandidate],
        neighbours: Sequence[_NeighbourBlocks],
        recovery_child: bool = False,
    ) -> str:
        """Processing request prompt text: header, chapter digest, glossary, and blocks.

        `candidates` — candidates from book chapter analyses: used to check chapter and glossary
        instructions against user-rejected model suggestions.
        """
        budget = notes_budget(
            unit_chars=sum(len(item.block.text) for item in unit.items),
            chapter_chars=sum(len(block.text) for block in textual_blocks(section)),
            per_chapter=self._language_notes_per_chapter,
        )
        if analysis.content_plan is not None and self._language_notes_per_chapter == 0:
            budget = 0
        parts = [
            _chapter_header(document=document, section=section),
            (f"Ориентир количества языковых заметок: {budget}. "
             "Для необходимого объяснения можно превысить этот ориентир, в том числе дать три заметки одному фрагменту. "
             "Не теряй обязательную помощь из учебного плана ради количества. "
             "Каждая заметка должна быть привязана к исходнику; не повторяй уже объяснённое."
             if recovery_child else f"В этом запросе — не больше {budget} языковых заметок."),
            _chapter_digest(analysis, glossary, candidates),
            _mandatory_decisions(
                _with_agreed_notes(glossary.mandatory_entries, glossary, candidates),
                target_language=document.target_language,
            ),
            _glossary_suggestions(
                _with_agreed_notes(
                    self._suggested_entries(glossary), glossary, candidates
                ),
                target_language=document.target_language,
            ),
            _explained_expressions(analysis, section),
            _neighbour_context(neighbours),
            _unit_blocks(unit),
            _owned_content_plan(analysis, unit),
        ]
        return "\n\n".join(part for part in parts if part)

    def _neighbours(
        self, section: DocumentSection, unit: TranslationUnit
    ) -> tuple[_NeighbourBlocks, ...]:
        """Neighbouring source text around block: several paragraphs before and after (PRD § 13).

        Only textual blocks of the same chapter qualify as neighbours: neighbouring chapters
        represent different scenes, while preserved opaque elements and illustrations are not passed.
        """
        limit = self._regeneration_context_blocks
        if limit == 0:
            return ()
        blocks = textual_blocks(section)
        requested = set(unit.block_ids)
        positions = [
            index for index, block in enumerate(blocks) if block.id in requested
        ]
        if not positions:
            return ()
        before = blocks[max(positions[0] - limit, 0) : positions[0]]
        after = blocks[positions[-1] + 1 : positions[-1] + 1 + limit]
        return tuple(
            _NeighbourBlocks(title=title, blocks=tuple(found))
            for title, found in (("До блока:", before), ("После блока:", after))
            if found
        )

    def _suggested_entries(
        self, glossary: GlossarySnapshot
    ) -> tuple[GlossarySnapshotEntry, ...]:
        """Optional glossary snapshot suggestions within context constraints.

        Mandatory decisions are not subject to this limit: they represent user decisions
        which the application has no right to truncate (PRD § 11).
        """
        mandatory = set(glossary.mandatory_entries)
        suggested = tuple(
            entry for entry in glossary.entries if entry not in mandatory
        )
        return suggested[: self._max_accumulated_candidates]

    def _chapter_context(
        self,
        *,
        document: CanonicalDocument,
        section: DocumentSection,
        blocks: Sequence[DocumentBlock],
        accumulated_candidates: Iterable[GlossaryCandidate],
        instructions: str,
        fragment_label: str | None,
        findings: FragmentFindings | None = None,
        before_you_read_limit: int = 10,
        language_notes_limit: int | None = None,
    ) -> ChapterAnalysisContext:
        """Ready request for chapter blocks along with response parsing metadata."""
        aliases = {_alias(index): block.id for index, block in enumerate(blocks, start=1)}
        candidate = self._content_plan_profile in {"content-plan-v2", "content-plan-v3"}
        version = 3 if self._content_plan_profile == "content-plan-v3" else (2 if candidate else 1)
        note_limit = self._language_notes_per_chapter if language_notes_limit is None else language_notes_limit
        note_groups = self._analysis_note_groups(section, blocks) if candidate else ()
        if candidate and section.is_notes_only:
            before_you_read_limit = 0
        if candidate:
            instructions += "\n" + self._prompt_templates.get("content_analysis", _CONTENT_PLAN_ANALYSIS_V3 if version == 3 else _CONTENT_PLAN_ANALYSIS)
            if version == 3:
                instructions += "\n" + self._prompt_templates.get("learner_quality", learner_instructions(self._reader_level))
            instructions += f"\nВ этом ответе не больше {before_you_read_limit} BYR."
            instructions += f"\nБюджет обязательных заметок этого анализа: {note_limit}."
            instructions += "\nПределы обязательных заметок по группам обработки: " + json.dumps([dict(blocks=[alias for alias, block_id in aliases.items() if block_id in ids], limit=limit) for ids, limit in note_groups], ensure_ascii=False)
            instructions += "\nСначала используй существующий BYR или допустимый hint.note; READ_AS_IS требует отдельной заметки, если BYR не покрывает все аспекты."
        return ChapterAnalysisContext(
            before_you_read_limit=before_you_read_limit,
            note_group_limits=note_groups,
            language_notes_per_chapter=note_limit,
            note_limits_by_block={b.id: notes_budget(unit_chars=len(b.text), chapter_chars=sum(len(t.text) for t in textual_blocks(section)), per_chapter=self._language_notes_per_chapter) if self._language_notes_per_chapter else 0 for b in blocks},
            section_id=section.id,
            request=LLMRequest.for_chapter_analysis(
                contract_version=version,
                instructions=instructions,
                input_text=self._chapter_input(
                    document=document,
                    section=section,
                    blocks=blocks,
                    accumulated_candidates=accumulated_candidates,
                    fragment_label=fragment_label,
                    findings=findings,
                ),
            ),
            block_ids_by_alias=aliases,
            expectation=ResultExpectation(
                target_language=document.target_language,
                blocks=tuple(ExpectedBlock(block_id=alias) for alias in aliases),
            ),
        )

    def _analysis_note_groups(
        self, section: DocumentSection, blocks: Sequence[DocumentBlock],
    ) -> tuple[tuple[tuple[str, ...], int], ...]:
        # Grouping depends on source blocks rather than textual treatment mode.
        from linguary.application.translation_units import plan_translation_units
        from linguary.domain import BlockTreatment

        chapter = textual_blocks(section)
        plan = TreatmentPlan.of_treatments(
            section.id, ((b.id, BlockTreatment.READ_AS_IS) for b in chapter),
        )
        units = plan_translation_units(
            chapter, plan, id_factory=lambda: "budget",
            max_source_chars=self._max_unit_source_chars,
        )
        visible = {b.id for b in blocks}
        chapter_chars = sum(len(b.text) for b in chapter)
        result = []
        for unit in units:
            size = sum(len(item.block.text) for item in unit.items)
            capacity = notes_budget(unit_chars=size, chapter_chars=chapter_chars,
                                    per_chapter=self._language_notes_per_chapter)
            if self._language_notes_per_chapter == 0:
                capacity = 0
            cursor = 0
            share = 0
            owners = []
            for item in unit.items:
                end = cursor + len(item.block.text)
                if item.block.id in visible:
                    owners.append(item.block.id)
                    share += capacity * end // size - capacity * cursor // size
                cursor = end
            if owners:
                result.append((tuple(owners), share))
        return tuple(result)

    def _chapter_input(
        self,
        *,
        document: CanonicalDocument,
        section: DocumentSection,
        blocks: Sequence[DocumentBlock],
        accumulated_candidates: Iterable[GlossaryCandidate],
        fragment_label: str | None,
        findings: FragmentFindings | None = None,
    ) -> str:
        """Prompt text: chapter header, accepted book decisions, and aliased blocks."""
        parts = [
            _chapter_header(
                document=document, section=section, fragment_label=fragment_label
            ),
            _known_decisions(
                self._selected_candidates(accumulated_candidates),
                target_language=document.target_language,
            ),
        ]
        if findings is not None:
            parts.append(
                "Уже найдено в предыдущих частях главы:\n"
                f"Сводка: {findings.summary}\n"
                f"Указания: {findings.instructions}\n"
                f"Выражения: {', '.join(findings.expressions)}"
            )
        parts.append(_chapter_blocks(blocks))
        return "\n\n".join(parts)

    def _selected_candidates(
        self, candidates: Iterable[GlossaryCandidate]
    ) -> tuple[GlossaryCandidate, ...]:
        """Decisions from previous chapters that enter context.

        Duplicates of the same source text are collapsed, including obvious spelling
        variants (`glossary_key`): context needs the term itself, not how many chapters
        it appeared in. When overflowing, earlier chapter decisions are retained: key characters
        and places appear early and recur, while a one-off term from the latest chapter adds little.
        """
        selected: dict[str, GlossaryCandidate] = {}
        for candidate in candidates:
            if len(selected) >= self._max_accumulated_candidates:
                break
            selected.setdefault(glossary_key(candidate.source_text), candidate)
        return tuple(selected.values())


def chapter_analysis_instructions(
    *, source_language: str, target_language: str, reader_level: str = "b2"
) -> str:
    """Invariant chapter analysis instructions (PRD § 10.1, § 11, § 14, § 19)."""
    return _instructions(
        source_language=source_language,
        target_language=target_language,
        scope=_WHOLE_CHAPTER_SCOPE,
        summary=_WHOLE_CHAPTER_SUMMARY,
        reader_level=reader_level,
    )


def chapter_fragment_instructions(
    *, source_language: str, target_language: str, reader_level: str = "b2"
) -> str:
    """Instructions for analyzing a fragment of a long chapter (PRD § 14, ALG-03).

    Differs from whole chapter instructions only in adapting to the fallback scenario:
    the model sees part of the chapter, and `chapter_summary` covers events from chapter start.
    Other rules — treatment plan, glossary candidates, translation instructions, and learning
    materials — remain identical.
    """
    return _instructions(
        source_language=source_language,
        target_language=target_language,
        scope=_FRAGMENT_SCOPE,
        summary=_FRAGMENT_SUMMARY,
        reader_level=reader_level,
    )


def translation_unit_instructions(*, source_language: str, target_language: str) -> str:
    """Invariant instructions for processing a block group (PRD § 10.1, § 15, § 19.3)."""
    return _TRANSLATION_INSTRUCTIONS.format(
        source_language=source_language,
        target_language=target_language,
        scope=_UNIT_SCOPE,
    )


def block_regeneration_instructions(
    *, source_language: str, target_language: str
) -> str:
    """Instructions for regenerating a single block (PRD § 13, § 19.3, § 21.5).

    Differs from group instructions only in adapting to single block regeneration: the model
    processes one block and sees neighbouring paragraphs that must not be replaced. Rules
    for treatment modes, translation, and notes remain the same.
    """
    return _TRANSLATION_INSTRUCTIONS.format(
        source_language=source_language,
        target_language=target_language,
        scope=_REGENERATION_SCOPE,
    )


def _reader_level_treatment_guidance(reader_level: str) -> str:
    """Instructions for selecting treatment modes tailored to target reader level (PRD § 30)."""
    level = reader_level.strip().lower()
    if level == "a2":
        return """\
Читатель владеет изучаемым языком на начальном уровне (A2). Ему необходима высокая степень языковой поддержки.

Верни ровно один элемент на каждый блок, который тебе дали: с тем же идентификатором \
и в том же порядке. Не пропускай блоки, не добавляй чужих, не объединяй и не дели их.

- TRANSLATE — основной режим: выбирай щедро (ориентир 65–80% блоков главы) для любого \
предложения с временами сложнее Simple/Continuous, придаточными предложениями, редкой лексикой, \
витиеватыми диалогами или развёрнутыми описаниями.
- GLOSSARY_ONLY — выбирай только для коротких абзацев с прозрачной базовой грамматикой, \
подсказывай любые слова выше уровня A2.
- READ_AS_IS — выбирай только для самых элементарных реплик и базовых коротких фраз \
(приветствия, простые утверждения без незнакомых слов).

Короткая прозрачная реплика диалога длиной 4–8 слов уровня A1–A2 не требует полного \
перевода только из-за того, что это реплика: выбирай READ_AS_IS или GLOSSARY_ONLY \
по незнакомой лексике и понятности смысла.
Не оставляй без достаточной помощи грамматически скрытые конструкции уровня выше A2: \
модальный перфект (например, «might have been»), придаточное с «what» в значении \
«то, что» и отрыв предлога в конце придаточного или вопроса. Для каждой такой \
конструкции выбирай TRANSLATE либо обеспечь отдельную ясную грамматическую заметку; \
словарной подсказки без объяснения грамматики недостаточно.
Если встречается двойное отрицание вида «not saying ... isn’t ...», сохрани его \
утвердительный смысл: это может быть признание качества персонажа или события, \
а не отрицание этого качества. Объяснение такого выражения в Before You Read \
формулируй утвердительно. Например, «not saying the spell isn’t brilliant» означает \
«не отрицаю, что заклинание блестящее», а объяснение звучит так: «Говорящий признаёт, \
что заклинание блестящее».

Длина блока не определяет режим. Если сомневаешься между режимами, выбирай более сильную помощь (TRANSLATE). \
Перед ответом в этом же запросе проверь план: ориентир TRANSLATE — 65–80% блоков; GLOSSARY_ONLY — до 25% блоков.\
"""
    if level == "b1":
        return """\
Читатель владеет изучаемым языком на среднем уровне (B1). Ему необходима повышенная языковая поддержка при чтении художественной литературы в оригинале.

Верни ровно один элемент на каждый блок, который тебе дали: с тем же идентификатором \
и в том же порядке. Не пропускай блоки, не добавляй чужих, не объединяй и не дели их.

- TRANSLATE — выбирай для большинства сюжетных, описательных и синтаксически сложных абзацев \
(ориентир 50–70% блоков главы). Если есть сомнение в прозрачности синтаксиса, времени глаголов \
или идиоматики для уровня B1 — выбирай TRANSLATE, а не бросай читателя в сложный оригинал.
- GLOSSARY_ONLY — выбирай, когда синтаксис прост и последовательность действий ясна, \
но встречаются незнакомые слова уровня B2+; подсказывай их щедро.
- READ_AS_IS — выбирай для прозрачных диалогов и простых повествовательных предложений без трудной лексики.

Длина блока не определяет режим. Если сомневаешься между режимами, отдавай предпочтение более доступному пониманию (TRANSLATE). \
Перед ответом в этом же запросе проверь план: ориентиры TRANSLATE — 50–70% блоков главы; GLOSSARY_ONLY — не меньше 20% блоков.\
"""
    if level == "c1":
        return """\
Читатель владеет изучаемым языком на продвинутом уровне (C1). Ему требуется минимальная помощь, сохраняй максимальное погружение в оригинал.

Верни ровно один элемент на каждый блок, который тебе дали: с тем же идентификатором \
и в том же порядке. Не пропускай блоки, не добавляй чужих, не объединяй и не дели их.

- READ_AS_IS — основной режим (ориентир 65–85% блоков главы): всё, что может быть понято продвинутым читателем, оставляй в оригинале без перевода и без подсказок.
- GLOSSARY_ONLY — для редких авторских терминов, архаизмов, узких реалий и необычных идиом.
- TRANSLATE — только для исключительных мест с предельно непрозрачным, запутанным или архаичным синтаксисом (ориентир 10–25% блоков главы).

Длина блока не определяет режим. Если сомневаешься между режимами, выбирай более лёгкую помощь (READ_AS_IS). \
Перед ответом в этом же запросе проверь план: ориентиры TRANSLATE — не больше 25% блоков; READ_AS_IS — не меньше 65% блоков.\
"""
    return """\
Читатель владеет изучаемым языком на уровне B1–B2. Выбирай помощь по тому, сможет ли \
он понять смысл блока в оригинале, а не по желанию перевести красивый текст.

Верни ровно один элемент на каждый блок, который тебе дали: с тем же идентификатором \
и в том же порядке. Не пропускай блоки, не добавляй чужих, не объединяй и не дели их.

- TRANSLATE — выбирай, когда без полного перевода читатель потеряет смысл из-за \
непрозрачного синтаксиса или смысловых связей, которые подсказки не прояснят. \
Это может быть сюжетный абзац, сложный диалог или важное описание.
- GLOSSARY_ONLY — выбирай, когда общий смысл и грамматика доступны, но одно-три \
трудных слова или выражения требуют подсказки: идиома, реалия, редкое слово. Например, \
длинный абзац с ясным ходом мысли и двумя незнакомыми словами читается с подсказками.
- READ_AS_IS — выбирай, когда лексика и грамматика прозрачны для B1–B2 и смысл \
понятен без помощи. Например, длинный абзац с привычными словами и простыми связями \
оставь в оригинале: перевод помешает тренировке чтения.

Длина блока не определяет режим. Если сомневаешься между режимами, выбирай более лёгкую \
помощь. Полный перевод обычно нужен примерно половине блоков главы; это мягкий ориентир, \
а не квота, и решение по каждому блоку зависит от его реальной трудности.

При понятных связях число незнакомых слов само по себе не требует TRANSLATE: \
три уже найденных слова покрываются GLOSSARY_ONLY. Если предложение распадается на \
понятные цепочки «кто сделал что где», не переводи весь абзац, даже если в нём встретились \
два-три редких слова. Например, длинный сюжетный абзац \
«He entered the courtyard, put down his valise and waited by the portcullis» имеет \
ясную последовательность действий; courtyard, valise, portcullis достаточно объяснить \
подсказками. Развёрнутое описание обстановки или пейзажа с простым синтаксисом и редкими \
словами (например: «The oak-panelled parlour was lit by a single sconce, and a cold \
draught stirred the faded damask curtains») тоже оставляй GLOSSARY_ONLY с подсказками \
к sconce, damask: ход мысли и связи слов прозрачны, полный перевод здесь избыточен. \
Длинный доступный диалог «We will meet at the station at six. Bring \
your valise and wait near the carriage. If I am late, ask the porter to keep your \
bag. The train will take us home before dark» тоже может остаться GLOSSARY_ONLY \
с подсказками valise, carriage, porter: намерения и условия ясны. Такой диалог без \
незнакомых слов — READ_AS_IS. Напротив, \
короткий диалог с непрозрачной смысловой связью «Were it not for what he left unsaid, \
I should never have suspected her» требует TRANSLATE: отдельные слова не объясняют \
условность и причину подозрения.

Перед ответом в этом же запросе проверь план: ориентиры TRANSLATE — не больше \
60% блоков и 70% объёма исходного текста; GLOSSARY_ONLY — не меньше 15% блоков. \
При большом охвате пересмотри сомнительные решения по смыслу и синтаксису: \
не переключай самые длинные блоки ради процентов и не лишай помощи сложные блоки. \
Это самопроверка, а не жёсткие квоты: длина и порог 200 символов не определяют режим, \
дополнительного запроса для достижения долей не требуется.\
"""


def _instructions(
    *,
    source_language: str,
    target_language: str,
    scope: str,
    summary: str,
    reader_level: str = "b2",
) -> str:
    """Analysis instructions compiled for a whole chapter or a chapter fragment."""
    languages = {"source_language": source_language, "target_language": target_language}
    treatment_guidance = _reader_level_treatment_guidance(reader_level)
    return _ANALYSIS_INSTRUCTIONS.format(
        scope=scope.format(**languages),
        summary=summary.format(**languages),
        treatment_guidance=treatment_guidance,
        **languages,
    )


#: What the model sees in the request: the whole chapter or a sequential fragment.
_WHOLE_CHAPTER_SCOPE = """\
Тебе дают одну главу, разбитую на пронумерованные текстовые блоки. Анализируй только эту \
главу: соседних глав ты не видишь и догадок о них не строишь."""

_FRAGMENT_SCOPE = """\
Тебе дают часть одной главы — последовательный фрагмент её текста, разбитый на \
пронумерованные текстовые блоки. Анализируй только этот фрагмент: ни остальной главы, \
ни соседних глав ты не видишь и догадок о них не строишь."""

#: What `chapter_summary` describes. For a fragment — events from chapter start up to it.
_WHOLE_CHAPTER_SUMMARY = """\
5. chapter_summary — что произошло в этой главе.

Пиши на языке оригинала книги ({source_language}), три-шесть предложений. Только события \
самой этой главы: что случилось, с кем, что изменилось. Предыдущие главы не пересказывай, \
о будущих не рассуждай, текст не оценивай. Этот пересказ увидит читатель перед следующей \
главой, чтобы вспомнить, на чём остановился."""

_FRAGMENT_SUMMARY = """\
5. chapter_summary — что произошло в главе от начала до конца этого фрагмента.

Пиши на языке оригинала книги ({source_language}), три-шесть предложений. Верни сводку \
всей главы от начала до конца своего фрагмента, включая уже найденное в предыдущих \
частях. Не рассуждай о том, что будет после. Текст не оценивай. Указания переводчику \
и выражения before_you_read возвращай только новые, не повторяя предыдущие части."""


#: Chapter analysis instructions. A single request covers treatment plan, glossary candidates,
#: translation instructions, Before You Read, and chapter summary (PRD § 14): no separate
#: preliminary model calls are made.
_ANALYSIS_INSTRUCTIONS = """\
Ты готовишь учебное издание художественной книги. Издание помогает читателю, который \
владеет языком {target_language} и учит язык {source_language}, читать эту книгу \
в оригинале.

{scope} Переводить текст сейчас не \
нужно — перевод блоков выполняется отдельными запросами позже.

1. treatment_plan — режим обработки каждого блока.

{treatment_guidance}

2. glossary_candidates — переводческие решения, которые обязаны совпадать во всей книге.

Это имена персонажей, названия мест, организации, важные вымышленные понятия и \
повторяющиеся специальные термины — но лишь там, где перевод нужно выбирать: \
вымышленные названия, малоизвестные топонимы, всё, что можно передать по-разному. \
Реальное место или учреждение с единственной устоявшейся формой на языке \
{target_language} решения не требует и в кандидаты не идёт; имена действующих лиц \
нужны всегда, даже очевидные. Обычную лексику, не требующую единообразия, не добавляй.

Очевидные альтернативные имена одной сущности объединяй только тогда, когда это \
бесспорно; неоднозначные случаи оставляй отдельными кандидатами.

- type — person, place, organization, fictional_concept или term; person — только \
тот, кто действует в сюжете. Народы, племена, сословия и исторические имена из справок \
и отступлений — term; блюда, сладости, напитки, предметы и марки, названные в честь \
людей, — тоже term, а не person, и указание рода им не нужно;
- suggested_translation — перевод именно этой формы, как она стоит в source_text, ровно \
один вариант на языке {target_language}, без пояснений в скобках и без вторых вариантов \
через слэш: этот текст подставляется в перевод как есть. Если в тексте стоит только имя \
или только фамилия, переводи их, не подставляй полное имя из других упоминаний;
- confidence — твоя уверенность в этом варианте от 0 до 1;
- model_flagged_review — true, если решение спорное, зависит от дальнейших глав или \
требует человека;
- note — короткое переводческое указание на языке {target_language}, если одного \
варианта перевода мало, иначе пустая строка.

Указание нужно там, где выбор придётся повторять в каждом предложении, а сам вариант \
перевода его не несёт. Главный случай — грамматический род персонажа: если язык \
{source_language} рода не показывает, а {target_language} требует его в глаголах и \
определениях, назови род прямо («женский род») — иначе одна глава сделает персонажа \
мужчиной, другая женщиной. Так же указывай несклоняемость названия, обращение на «ты» \
или «вы», устойчивый регистр речи. Общих слов о стиле в указании не пиши.

Если такое решение по персонажу или названию уже принято в предыдущих главах, повтори \
его указание дословно и своего не выдумывай.

Если термин уже есть в принятых решениях книги, повтори принятый перевод и нового \
варианта не предлагай. Это касается той же формы термина: если в решениях есть полное \
имя, а в главе встречается его часть, переведи эту часть и не подставляй полное имя.

3. translation_instructions — короткие указания переводчику этой главы на языке \
{target_language}: голос рассказчика, речевые особенности персонажей, что обязательно \
сохранить (иронию, архаику, диалект, игру слов), где легко ошибиться. Пиши только то, \
что видно в этой главе, три-восемь пунктов. Переводы имён и терминов здесь не назначай: \
их выбирает глоссарий — при необходимости просто сошлись на него. Общие слова \
о качественном художественном переводе не нужны.

Авторские особенности речи сохраняй естественными понятными средствами целевого \
языка, без искусственной имитации: опирайся на наблюдаемую речь конкретного персонажа, \
не предписывай всем героям приёмы одного персонажа или одной книги.

В указаниях и сводке не ссылайся на блоки по идентификаторам (например, b1).

4. before_you_read — пять-десять ключевых понятий и выражений главы, которые критически \
важны для понимания происходящего перед началом чтения главы (advance organizer):
- Тематические и смысловые опоры главы (thematic / setting anchors) — специфическая \
терминология, ключевые предметные понятия главы или историко-бытовые реалии, которые задают \
сцену, действие или конфликт и повторяются в главе.
- Семантические ловушки и непрозрачные смыслы (semantic traps) — идиомы, непрозрачные \
фразовые глаголы, устойчивые сочетания, необычные контекстуальные или архаичные значения \
внешне знакомых слов, где буквальный перевод вводит читателя в заблуждение.

Имена персонажей и уникальные географические названия сюда не попадают: их место — \
glossary_candidates, и дублировать их здесь не нужно. Банальные общеупотребительные слова не включай.

Поле expression — слово или выражение так, как оно стоит в тексте главы; explanation — \
короткое, понятное и ёмкое объяснение на языке {target_language}. Это предварительная опора \
перед чтением, а не пересказ: сюжет и спойлеры здесь не раскрывают. Меньше пяти выражений — \
нормально, если больше в главе нет; добирать список ради количества не нужно.

{summary}

Отвечай строго по схеме структурированного ответа, без пояснений вне её полей."""


#: What the model processes: a group of consecutive blocks or a single selected block
#: where regeneration provides neighbouring paragraphs as scene context (PRD § 13, § 21.5).
_UNIT_SCOPE = """\
Тебе дают несколько идущих подряд текстовых блоков одной главы. У каждого блока указан \
режим обработки: он выбран заранее, и менять его нельзя. Соседних глав и результатов \
по другим блокам ты не видишь."""

_REGENERATION_SCOPE = """\
Тебе дают один текстовый блок одной главы и несколько соседних абзацев как контекст \
сцены. Обработать нужно только этот блок: соседние абзацы даны для понимания \
происходящего, переводить их и возвращать в ответе не нужно. У блока указан режим \
обработки: он выбран заранее, и менять его нельзя. Соседних глав и результатов \
по другим блокам ты не видишь."""


#: Block processing instructions. A single request produces translation, glossary hints,
#: and language notes (PRD § 14, § 19.3): no separate pass is needed for notes because the model
#: already sees the original, the block mode, and its output.
_TRANSLATION_INSTRUCTIONS = """\
Ты готовишь учебное издание художественной книги. Издание помогает читателю, который \
владеет языком {target_language} и учит язык {source_language}, читать эту книгу \
в оригинале.

{scope}

1. blocks — результат по каждому блоку.

Верни ровно один элемент на каждый блок, который тебе дали: с тем же идентификатором \
и в том же порядке. Не пропускай блоки, не добавляй чужих, не объединяй и не дели их. \
Поле treatment повторяет режим блока из запроса — это подтверждение, а не твой выбор: \
другой режим отклоняет весь ответ.

- TRANSLATE — верни в translated_text полный литературный перевод блока на язык \
{target_language}. Список glossary_hints для такого блока обычно пуст: перевод уже дан.
- GLOSSARY_ONLY — полного перевода не давай, translated_text оставь пустым. Верни \
в glossary_hints одно-три слова или выражения этого блока, без которых его трудно \
читать в оригинале: original — как в тексте, translation — на языке {target_language}, \
note — короткое пояснение, если оно нужно.
- READ_AS_IS — блок читается в оригинале без помощи: ни перевода, ни словарных \
подсказок. Оставь translated_text пустым, а glossary_hints — пустым списком.

Перевод пиши целиком на языке {target_language} и доводи до конца: оборванная фраза, \
незакрытая кавычка или скобка отклоняют весь ответ. Абзац переводи как одно целое — \
не дели его и не склеивай с соседним. Обязательные переводческие решения книги соблюдай \
дословно; менять допустимо только грамматическую форму. Эти решения принял пользователь: \
они сильнее указаний переводчику главы, и при расхождении действует решение.

В каждом переводе сохрани ровно исходное число непустых строк: строка здесь — текст \
между переносами, содержащий хотя бы один непробельный символ. Передай весь собственный текст блока, \
включая слова рассказчика после реплики и авторское окончание. Не добавляй соседей \
и не удаляй окончание ради числа строк; не вводи новые переносы для оформления диалога.

Особенности речи персонажа, названные в указаниях главы, передавай средствами языка \
{target_language}: ломаную грамматику, акцент и диалект не выправляй автоматически \
в гладкую речь. Сохраняй авторскую особенность естественными понятными средствами, \
без искусственной имитации; не переноси приёмы одного персонажа на остальных. \
Не утверждай в заметке, что особенность речи сохранена, если перевод её не передаёт.
Заметка может описывать особенность оригинала. Для заявления о её сохранении нужно \
наблюдаемое основание в самом переводе: назови конкретную переданную конструкцию \
или речевой приём. Гладкий перевод без этой особенности не даёт такого основания; \
наличие указания главы или обещания сохранить речь тоже ничего не доказывает.

Метка вида [^1] в тексте блока — ссылка на сноску книги. Перенеси каждую такую метку \
в translated_text без изменений, в то место перевода, к которому она относится. Цифрой \
или словом её не заменяй, в словарные подсказки и языковые заметки не включай.

2. language_notes — языковые заметки для читателя (PRD: учебный материал).

Заметка объясняет то, что реально помогает читать оригинал или понять переводческое \
решение: идиому, сленг, игру слов, нестандартную грамматику, диалект, культурную \
особенность, необычное употребление.

Заметка не пересказывает сюжет и не объясняет обычные слова блока TRANSLATE, которые \
уже переданы переводом. Исключение — идиома, игра слов или само переводческое решение, \
если их невозможно понять по готовому переводу.

Заметка отвечает на вопрос «как устроен язык здесь», а не «какое здесь настроение». \
Разбор образа, тона, символики и авторского замысла — это литературоведение, и \
читателю оригинала оно не помогает: такие заметки не пиши. «Сравнение создаёт \
тревожное ощущение» — не заметка; «have no business + Ving — идиома со значением \
„не положено, незачем“» — заметка.

Одно и то же место объясняется читателю один раз. Не повторяй заметкой то, что уже \
сказано словарной подсказкой этого блока или подготовкой к чтению этой главы: если \
подсказка выражение уже перевела, заметка добавляет к ней новое или её не будет.

Перед включением заметки сравни все её объясняющие утверждения с полными объяснениями \
Before You Read и подсказкой своего блока. Полный смысловой повтор запрещён даже \
при другой форме выражения или перефразировании: made/make a grab at с объяснением \
«попытаться схватить» и sheath/armour с объяснением «ножны скрывают, броня защищает» \
не объясняй повторно. Но новый языковой или культурный факт того же отрывка допустим: \
например, нестандартное without to, эллипсис или игра слов, которых в прежнем \
объяснении не было. Не давай словарный пересказ готового перевода. В смешанной \
заметке культурная справка не оправдывает пересказ сюжета: убери описание действий \
персонажей и оставь только новое объяснение языка или реалии.

Каждая заметка привязана ровно к одному блоку из этого запроса: source_block_id — его \
идентификатор, original_excerpt — фрагмент оригинала, translated_excerpt — его перевод, \
если он уместен, note — объяснение на языке {target_language}. Заметки допустимы при \
любом режиме, включая READ_AS_IS. Соблюдай бюджет заметок, указанный в запросе; \
если объяснять нечего, верни пустой список.

Служебных сообщений, извинений и рассуждений о самой задаче в ответе быть не должно. \
Отвечай строго по схеме структурированного ответа, без пояснений вне её полей."""


def _chapter_digest(
    analysis: ChapterAnalysis,
    glossary: GlossarySnapshot,
    candidates: Sequence[GlossaryCandidate],
) -> str:
    """Chapter analysis digest: summary and translation instructions (PRD § 13).

    Glossary candidates are omitted here: approved book decisions arrive via the run snapshot,
    and unapproved candidates no longer exist after processing begins. Instructions were written
    prior to glossary review, so items contradicting user decisions are pruned to prevent stale
    model variants from overriding corrections (PRD § 11). Rejected variants are gathered from
    the whole book's candidates: a chapter analysis does not return candidates already decided
    in earlier chapters.
    """
    lines: list[str] = []
    if summary := analysis.chapter_summary.strip():
        lines.append(f"О чём эта глава:\n{summary}")
    instructions = instructions_respecting_glossary(
        analysis.translation_instructions,
        decisions=glossary.mandatory_entries,
        chapter_candidates=candidates,
    )
    if instructions:
        lines.append(
            "Указания переводчику этой главы:\n"
            "Особенности речи передавай естественными понятными средствами, без "
            "искусственной имитации; сохранённые указания главы подчинены этому принципу. "
            "Применяй их только к персонажам и местам, где особенность видна в оригинале.\n"
            f"{without_block_labels(instructions)}"
        )
    return "\n\n".join(lines)


def _with_agreed_notes(
    entries: Sequence[GlossarySnapshotEntry],
    glossary: GlossarySnapshot,
    candidates: Sequence[GlossaryCandidate],
) -> tuple[GlossarySnapshotEntry, ...]:
    """Snapshot entries with notes reconciled against decisions for other concepts (PRD § 11).

    The note of an unedited approved entry was generated by the model during analysis and may
    mention variants of other concepts that the user rejected. Rejected variants are known
    from candidates across all book chapter analyses.
    """
    return tuple(
        replace(
            entry,
            note=note_respecting_glossary(
                entry,
                decisions=glossary.mandatory_entries,
                chapter_candidates=candidates,
            ),
        )
        for entry in entries
    )


def _explained_expressions(analysis: ChapterAnalysis, section: DocumentSection) -> str:
    """Chapter `Before You Read` expressions and explanations (PRD § 19.1, § 19.3).

    Advance organizer material was compiled during chapter analysis, whereas hints and notes
    are generated later. Without this list, a phrase might be explained to the reader three times —
    in `Before You Read`, in a block glossary hint, and in a language note — and § 19.3 explicitly
    forbids notes from repeating hints.
    """
    entries = _before_you_read_entries(analysis, section)
    if not entries:
        return ""
    lines = [
        "Эти выражения главы читателю уже объяснены до начала чтения. Не объясняй их "
        "заново ни словарной подсказкой, ни языковой заметкой. Сравни содержание "
        "заметки с полным объяснением; самостоятельный новый факт допустим:"
    ]
    lines.extend(f"- {expression}: {explanation}" for expression, explanation in entries)
    return "\n".join(lines)


def _before_you_read_entries(
    analysis: ChapterAnalysis, section: DocumentSection
) -> tuple[tuple[str, str], ...]:
    """Full advance reading explanations accessible to the model and note filtering."""
    if section.is_notes_only:
        return ()
    return tuple(
        (entry.expression.strip(), entry.explanation.strip())
        for entry in analysis.before_you_read
        if entry.expression.strip()
    )


def _before_you_read_expressions(
    analysis: ChapterAnalysis, section: DocumentSection
) -> tuple[str, ...]:
    """Non-empty `Before You Read` expressions for the chapter in analysis order.

    A footnotes section does not receive `Before You Read`, so there are no expressions to repeat.
    """
    if section.is_notes_only:
        return ()
    return tuple(
        entry.expression.strip()
        for entry in analysis.before_you_read
        if entry.expression.strip()
    )


def _mandatory_decisions(
    entries: Sequence[GlossarySnapshotEntry], *, target_language: str
) -> str:
    """Mandatory translation guidance for the book: user decisions (PRD § 11-12)."""
    if not entries:
        return "Обязательных переводческих указаний по этой книге нет."
    lines = [
        f"Обязательные переводческие решения книги ({target_language}). Их принял "
        f"пользователь, и они сильнее указаний переводчику главы: если указание называет "
        f"другой вариант перевода или другой род, следуй решению. Соблюдай их дословно, "
        f"меняя только грамматическую форму:"
    ]
    lines.extend(_glossary_line(entry) for entry in entries)
    lines.append(_notes_legend(entries))
    return "\n".join(line for line in lines if line)


def _glossary_suggestions(
    entries: Sequence[GlossarySnapshotEntry], *, target_language: str
) -> str:
    """Optional glossary suggestions for the book (PRD § 11-12)."""
    if not entries:
        return ""
    lines = [
        f"Необязательные подсказки глоссария ({target_language}). При прочих равных "
        f"держись их, но точность важнее:"
    ]
    lines.extend(_glossary_line(entry) for entry in entries)
    lines.append(_notes_legend(entries))
    return "\n".join(line for line in lines if line)


def _notes_legend(entries: Sequence[GlossarySnapshotEntry]) -> str:
    """Legend for list brackets if at least one decision contains guidance.

    Guidance is a book-wide decision, not a single-request hint: character gender,
    once assigned, must hold across all chapters (PRD § 11).
    """
    if not any(entry.note.strip() for entry in entries):
        return ""
    return (
        "В скобках — переводческое указание по решению. Оно принято для всей книги: "
        "соблюдай его в каждом предложении, где оно применимо."
    )


def _glossary_line(entry: GlossarySnapshotEntry) -> str:
    """Glossary snapshot entry line with its translation guidance."""
    line = f"- {entry.source_text} [{entry.type}] -> {entry.translation}"
    return f"{line} ({entry.note.strip()})" if entry.note.strip() else line


def _neighbour_context(neighbours: Sequence[_NeighbourBlocks]) -> str:
    """Neighbouring source text: scene context that is not processed (PRD § 13)."""
    if not neighbours:
        return ""
    lines = [
        "Соседние абзацы главы — только для понимания сцены. Их не переводи, "
        "словарных подсказок к ним не давай и в ответ не включай:"
    ]
    for side in neighbours:
        lines.append(side.title)
        lines.extend(source_text_for_model(block) for block in side.blocks)
    return "\n\n".join(lines)


def source_text_for_model(block: DocumentBlock, *, footnote_markers: bool = False) -> str:
    """Block text as presented to the model.

    Differs from `block.text` in two ways. First, footnote back links — '↩', 'back', and similar
    tails that return readers to caller locations — are stripped. For the book this is navigation
    rather than footnote body text: the exporter recreates it from block markup. The model would see
    the back link as part of the text and transfer it into translation, creating a dead symbol.
    Back links are detected by markup rather than character matching: link at the end of footnote text
    pointing to an anchor within the book.

    Second, footnote call numbers in the middle of paragraphs are not shown as bare numbers to the model;
    otherwise they carry over as raw digits. For blocks being processed by the model, calls are passed
    as `[^N]` markers (`footnote_markers`): the model reproduces them in translation, and the exporter
    inserts original links in their place. In other cases — chapter analysis, neighbouring paragraphs —
    the call reference is simply omitted.
    """
    text = block.text
    end = _back_link_start(block) if block.type is BlockType.FOOTNOTE else len(text)
    pieces: list[str] = []
    position = 0
    for ordinal, reference in enumerate(footnote_references(block), start=1):
        if reference.start < position or reference.end > end:
            continue
        before = text[position : reference.start]
        if not footnote_markers and before.endswith(" ") and (
            reference.end == end or text[reference.end] == " "
        ):
            before = before[:-1]
        pieces.append(before)
        if footnote_markers:
            pieces.append(footnote_marker(ordinal))
        position = reference.end
    pieces.append(text[position:end])
    return "".join(pieces).strip()


def _back_link_start(block: DocumentBlock) -> int:
    """Start position of back link at end of footnote, or text length if none."""
    text = block.text
    tail = max(
        (
            annotation.start
            for annotation in block.inline_annotations
            if annotation.type is InlineAnnotationType.LINK
            and annotation.end >= len(text.rstrip())
            and text[: annotation.start].strip()
            and _points_to_book_anchor(annotation.target)
        ),
        default=None,
    )
    return len(text) if tail is None else tail


#: External URI scheme: `http:`, `mailto:`, etc. Importer preserves them as-is,
#: while internal book links are mapped to resource paths with anchors.
_EXTERNAL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _points_to_book_anchor(target: str | None) -> bool:
    """Whether link points to an internal book anchor within the same or another resource.

    Return links from chapter footnotes point to the same file, while returns from dedicated
    footnote pages point to the chapter file where the footnote was referenced. External URIs
    and anchorless file links are never return links: they are footnote content.
    """
    if not target or _EXTERNAL_SCHEME.match(target):
        return False
    _, separator, fragment = target.partition("#")
    return bool(separator and fragment)


def _unit_blocks(unit: TranslationUnit) -> str:
    """Group blocks with aliases along with their treatment modes."""
    lines = ["Блоки для обработки:"]
    lines.extend(
        f"[{_alias(index)}] режим {item.treatment}\n"
        f"{source_text_for_model(item.block, footnote_markers=True)}"
        for index, item in enumerate(unit.items, start=1)
    )
    return "\n\n".join(lines)


def _alias(index: int) -> str:
    """Short block alias in requests."""
    return f"b{index}"


def _fragment_label(order: int, fragment_count: int) -> str:
    """Header label specifying fragment position within chapter."""
    return (
        f"Фрагмент главы: {order + 1}-й из {fragment_count}. "
        f"Остальные фрагменты главы тебе не показаны."
    )


def _chapter_header(
    *,
    document: CanonicalDocument,
    section: DocumentSection,
    fragment_label: str | None = None,
) -> str:
    """Prompt header: book, chapter, fragment position, and edition languages."""
    sections = document.ordered_sections
    position = next(
        (
            number
            for number, other in enumerate(sections, start=1)
            if other.id == section.id
        ),
        section.order + 1,
    )
    authors = ", ".join(document.metadata.authors)
    lines = [
        f"Книга: {document.metadata.title}."
        + (f" Автор: {authors}." if authors else ""),
        f"Глава: {section.title} ({position}-я по порядку чтения из {len(sections)}).",
    ]
    if fragment_label is not None:
        lines.append(fragment_label)
    lines.append(
        f"Язык оригинала: {document.source_language}. "
        f"Язык перевода: {document.target_language}."
    )
    return "\n".join(lines)


def _known_decisions(
    candidates: Sequence[GlossaryCandidate], *, target_language: str
) -> str:
    """Already accepted book decisions: candidates of prior chapters and user decisions."""
    if not candidates:
        return "Предыдущие главы переводческих решений пока не дали."
    lines = [
        f"Принятые переводческие решения книги ({target_language}). "
        f"Повторяй их дословно:"
    ]
    lines.extend(_known_decision_line(candidate) for candidate in candidates)
    return "\n".join(lines)


def _known_decision_line(candidate: GlossaryCandidate) -> str:
    """Accepted book decision line with its translation guidance."""
    line = (
        f"- {candidate.source_text} [{candidate.type}] -> "
        f"{candidate.suggested_translation}"
    )
    return f"{line} ({candidate.note.strip()})" if candidate.note.strip() else line


def _chapter_blocks(blocks: Sequence[DocumentBlock]) -> str:
    """Textual blocks of chapter under aliases."""
    lines = ["Текстовые блоки главы:"]
    lines.extend(
        f"[{_alias(index)}]\n{source_text_for_model(block)}"
        for index, block in enumerate(blocks, start=1)
    )
    return "\n\n".join(lines)


__all__ = [
    "DEFAULT_MAX_ACCUMULATED_CANDIDATES",
    "DEFAULT_REGENERATION_CONTEXT_BLOCKS",
    "AliasedBlocksContext",
    "ChapterAnalysisContext",
    "ContextBuildError",
    "ContextBuilder",
    "TranslationContext",
    "block_regeneration_instructions",
    "chapter_analysis_instructions",
    "chapter_fragment_instructions",
    "is_textual",
    "source_text_for_model",
    "textual_blocks",
    "translation_unit_instructions",
]


def _owned_content_plan(analysis: ChapterAnalysis, unit: TranslationUnit) -> str:
    if analysis.content_plan is None:
        return ""
    aliases = {item.block.id: _alias(i) for i, item in enumerate(unit.items, 1)}
    difficulties = []
    related = {}
    for difficulty in analysis.content_plan.difficulties:
        if difficulty.owner_block_id not in aliases:
            continue
        value = asdict(difficulty)
        value["owner_block_id"] = aliases[difficulty.owner_block_id]
        for anchor in value["anchors"]:
            source_id = anchor["block_id"]
            if source_id not in aliases:
                related.setdefault(source_id, f"related-source-{len(related) + 1}")
            anchor["block_id"] = aliases.get(source_id, related.get(source_id))
        difficulties.append(value)
    evidence = []
    for item in analysis.content_plan.before_you_read_evidence:
        if item.block_id in aliases:
            value = asdict(item)
            value["block_id"] = aliases[item.block_id]
            for index, entry in enumerate(analysis.before_you_read):
                if artifact_key(item.artifact_kind, item.block_id, {"expression": entry.expression, "explanation": entry.explanation}) == item.artifact_key:
                    value["artifact_index"] = index
                    break
            evidence.append(value)
    facts = []
    for fact in analysis.content_plan.context_facts:
        if fact.subject_anchor.block_id not in aliases:
            continue
        value = fact.to_dict()
        for anchor in (*value['anchors'], value['subject_anchor']):
            source_id = anchor['block_id']
            if source_id not in aliases:
                related.setdefault(source_id, f'related-source-{len(related) + 1}')
            anchor['block_id'] = aliases.get(source_id, related.get(source_id))
        facts.append(value)
    return "Обязательные аспекты и точные связанные привязки:\n" + json.dumps({
        "version": analysis.content_plan.version, "difficulties": difficulties, "before_you_read_evidence": evidence,
        "before_you_read": [dict(artifact_index=i, expression=e.expression, explanation=e.explanation) for i, e in enumerate(analysis.before_you_read)],
        **({'context_facts': facts} if analysis.content_plan.version == 3 else {}),
    }, ensure_ascii=False)


def snapshot_prompt_templates(*, source_language, target_language, reader_level, profile="legacy-v1"):
    """Render immutable instruction templates without book text or secrets."""
    languages = dict(source_language=source_language, target_language=target_language)
    result = {
        'analysis': chapter_analysis_instructions(**languages, reader_level=reader_level),
        'fragment': chapter_fragment_instructions(**languages, reader_level=reader_level),
        'translation': translation_unit_instructions(**languages),
        'regeneration': block_regeneration_instructions(**languages),
        'content_analysis': _CONTENT_PLAN_ANALYSIS,
        'content_processing': _CONTENT_PLAN_PROCESSING,
    }
    if profile == 'content-plan-v3':
        result.update(content_analysis=_CONTENT_PLAN_ANALYSIS_V3, content_processing=_CONTENT_PLAN_PROCESSING_V3,
                      learner_quality=learner_instructions(reader_level))
    return result
