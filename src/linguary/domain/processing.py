"""Book processing: computational block unit, results, and processing runs.

`TranslationUnit` is a transient grouping of sequential textual blocks for a single LLM call.
Results are persisted separately for each input block, but only after successful validation
of the entire group (PRD § 10, § 17; ADR § 3).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from linguary.domain.content_plan import ContentEvidence
from linguary.domain.document import DocumentBlock
from linguary.domain.errors import (
    ActiveRunConflictError,
    BlockOrderError,
    BlockResultError,
    MissingBlockResultError,
    NonTextualBlockError,
    ProcessingRunError,
    TranslationUnitError,
    TreatmentMismatchError,
    UnknownBlockError,
)
from linguary.domain.glossary import ConceptType, TermStatus
from linguary.domain.learning import LanguageNote
from linguary.domain.treatment import BlockTreatment, TreatmentPlan


@dataclass(frozen=True, slots=True)
class GlossaryHint:
    """Translation of an individual word or short expression within a block."""

    original: str
    translation: str
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.original.strip():
            raise BlockResultError("исходное выражение словарной подсказки пусто")
        if not self.translation.strip():
            raise BlockResultError(f"перевод подсказки {self.original!r} пуст")


class ResultWarningKind(StrEnum):
    """Kinds of non-blocking warnings attached to processing results (PRD § 15)."""

    REPETITION = "repetition"
    DROPPED_BLOCK = "dropped_block"


@dataclass(frozen=True, slots=True)
class ResultWarning:
    """Warning persisted alongside the result that does not block processing."""

    kind: ResultWarningKind
    message: str

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise BlockResultError("текст замечания к результату пуст")


@dataclass(frozen=True, slots=True)
class BlockProcessingResult:
    """Processing result for a single input block.

    Composition rules are determined by the treatment from the persisted plan (PRD § 14, § 15):
    full translation is required and permitted only for `TRANSLATE`; for `GLOSSARY_ONLY`
    glossary hints are mandatory; for `READ_AS_IS` there is neither translation nor hints.
    """

    block_id: str
    treatment: BlockTreatment
    translated_text: str | None = None
    glossary_hints: tuple[GlossaryHint, ...] = ()
    language_notes: tuple[LanguageNote, ...] = ()
    warnings: tuple[ResultWarning, ...] = ()
    content_evidence: ContentEvidence | tuple[()] = ()

    def __post_init__(self) -> None:
        if not self.block_id.strip():
            raise BlockResultError("результат не привязан к блоку")
        if not self.treatment.is_textual:
            raise NonTextualBlockError(
                f"блок {self.block_id} с режимом {self.treatment} не имеет результата обработки"
            )
        has_translation = bool(self.translated_text and self.translated_text.strip())
        if self.treatment.requires_translation and not has_translation:
            raise BlockResultError(f"для блока {self.block_id} обязателен полный перевод")
        if not self.treatment.allows_translation and self.translated_text is not None:
            raise BlockResultError(
                f"для блока {self.block_id} с режимом {self.treatment} полный перевод недопустим"
            )
        if self.treatment.requires_glossary_hints and not self.glossary_hints:
            raise BlockResultError(
                f"для блока {self.block_id} обязательны словарные подсказки"
            )
        if not self.treatment.allows_glossary_hints and self.glossary_hints:
            raise BlockResultError(
                f"для блока {self.block_id} с режимом {self.treatment} "
                "словарные подсказки недопустимы"
            )
        for note in self.language_notes:
            if note.block_id != self.block_id:
                raise BlockResultError(
                    f"заметка {note.id} привязана к другому блоку: {note.block_id}"
                )


@dataclass(frozen=True, slots=True)
class TranslationUnitItem:
    """A unit block alongside its treatment mode from the persisted plan."""

    block: DocumentBlock
    treatment: BlockTreatment

    def __post_init__(self) -> None:
        if not self.treatment.is_textual:
            raise NonTextualBlockError(
                f"блок {self.block.id} с режимом {self.treatment} не передаётся модели"
            )


@dataclass(frozen=True, slots=True)
class TranslationUnit:
    """A group of sequential textual blocks within a chapter for a single LLM call.

    A computational unit rather than storage: results of individual blocks are persisted.
    """

    id: str
    section_id: str
    items: tuple[TranslationUnitItem, ...]

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise TranslationUnitError("идентификатор группы блоков пуст")
        if not self.items:
            raise TranslationUnitError(f"группа блоков {self.id} пуста")
        seen: set[str] = set()
        previous_order: int | None = None
        for item in self.items:
            if item.block.section_id != self.section_id:
                raise TranslationUnitError(
                    f"блок {item.block.id} принадлежит другой главе: {item.block.section_id}"
                )
            if item.block.id in seen:
                raise TranslationUnitError(
                    f"блок {item.block.id} встречается в группе {self.id} дважды"
                )
            if previous_order is not None and item.block.order <= previous_order:
                raise BlockOrderError(
                    f"блоки группы {self.id} идут не в порядке исходного документа"
                )
            seen.add(item.block.id)
            previous_order = item.block.order

    @classmethod
    def of_blocks(
        cls, *, id: str, blocks: Iterable[DocumentBlock], plan: TreatmentPlan
    ) -> TranslationUnit:
        """Constructs a unit from chapter blocks, reading treatments from the persisted plan.

        `PRESERVE_ONLY` and `DROP_WITH_WARNING` blocks do not enter the unit: attempting
        to include such a block is an invariant violation rather than a filtering trigger.
        """
        ordered = tuple(blocks)
        if not ordered:
            raise TranslationUnitError(f"группа блоков {id} пуста")
        return cls(
            id=id,
            section_id=ordered[0].section_id,
            items=tuple(
                TranslationUnitItem(block=block, treatment=plan.treatment_for(block.id))
                for block in ordered
            ),
        )

    @property
    def blocks(self) -> tuple[DocumentBlock, ...]:
        """Unit blocks in source document order."""
        return tuple(item.block for item in self.items)

    @property
    def block_ids(self) -> tuple[str, ...]:
        """Unit block identifiers in source document order."""
        return tuple(item.block.id for item in self.items)

    def treatment_for(self, block_id: str) -> BlockTreatment:
        """Block treatment mode for a unit block per the persisted plan."""
        for item in self.items:
            if item.block.id == block_id:
                return item.treatment
        raise UnknownBlockError(f"блок {block_id} отсутствует в группе {self.id}")

    def validate_results(self, results: Sequence[BlockProcessingResult]) -> None:
        """Blocking validations for results of the entire unit (PRD § 15).

        Validates presence of all expected blocks, absence of unknown blocks, order
        preservation, and treatment match with the plan. Any violation aborts persistence
        of the entire unit: partially persisted results are disallowed.
        """
        expected = self.block_ids
        returned = tuple(result.block_id for result in results)
        unknown = set(returned) - set(expected)
        if unknown:
            raise UnknownBlockError(
                f"группа {self.id} получила результат для чужих блоков: {sorted(unknown)}"
            )
        missing = set(expected) - set(returned)
        if missing:
            raise MissingBlockResultError(
                f"в результате группы {self.id} нет блоков: {sorted(missing)}"
            )
        if returned != expected:
            raise BlockOrderError(
                f"порядок блоков в результате группы {self.id} не совпадает с исходным"
            )
        for result in results:
            planned = self.treatment_for(result.block_id)
            if result.treatment is not planned:
                raise TreatmentMismatchError(
                    f"режим блока {result.block_id} в ответе модели ({result.treatment}) "
                    f"не совпадает с планом ({planned})"
                )


class ProcessingRunStatus(StrEnum):
    """State of a book processing run."""

    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class GlossarySnapshotEntry:
    """Approved translation decision at the time of launching a processing run.

    `note` is translation guidance for this decision: what cannot be expressed by a single
    translation variant but must remain consistent throughout the book (PRD § 11).
    """

    source_text: str
    translation: str
    type: ConceptType
    status: TermStatus
    note: str = ""

    def __post_init__(self) -> None:
        if not self.source_text.strip():
            raise ProcessingRunError("исходный текст записи снимка глоссария пуст")
        if not self.translation.strip():
            raise ProcessingRunError(f"перевод записи {self.source_text!r} пуст")


@dataclass(frozen=True, slots=True)
class GlossarySnapshot:
    """Glossary snapshot with which a processing run is executed."""

    entries: tuple[GlossarySnapshotEntry, ...] = ()

    def __post_init__(self) -> None:
        texts = [entry.source_text for entry in self.entries]
        if len(set(texts)) != len(texts):
            raise ProcessingRunError("снимок глоссария содержит повторяющиеся исходные тексты")

    @property
    def mandatory_entries(self) -> tuple[GlossarySnapshotEntry, ...]:
        """User decisions passed to the model as mandatory."""
        return tuple(entry for entry in self.entries if entry.status.is_user_decision)


@dataclass(frozen=True, slots=True)
class LLMSettingsSnapshot:
    """LLM settings captured at the time of launching a processing run."""

    provider: str
    model: str
    temperature: float | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ProcessingRunError("провайдер LLM в снимке настроек не задан")
        if not self.model.strip():
            raise ProcessingRunError("модель LLM в снимке настроек не задана")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ProcessingRunError("лимит выходных токенов должен быть положительным")


@dataclass(frozen=True, slots=True)
class ProcessingRun:
    """Book processing run with a snapshot of glossary and LLM settings.

    A book has at most one active run at any given time: concurrent result sets
    never accumulate (ADR § 3).
    """

    id: str
    book_id: str
    status: ProcessingRunStatus
    glossary_snapshot: GlossarySnapshot
    llm_settings_snapshot: LLMSettingsSnapshot

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ProcessingRunError("идентификатор запуска обработки пуст")
        if not self.book_id.strip():
            raise ProcessingRunError("запуск обработки не привязан к книге")

    @property
    def is_active(self) -> bool:
        """Whether this run is the currently active run for the book."""
        return self.status is ProcessingRunStatus.ACTIVE


def active_run(runs: Iterable[ProcessingRun]) -> ProcessingRun | None:
    """The sole active run of the book or `None` if no run is active."""
    active = [run for run in runs if run.is_active]
    if len(active) > 1:
        raise ActiveRunConflictError(
            f"у книги больше одного активного запуска обработки: {[run.id for run in active]}"
        )
    return active[0] if active else None
    active = [run for run in runs if run.is_active]
    if len(active) > 1:
        raise ActiveRunConflictError(
            f"у книги больше одного активного запуска обработки: {[run.id for run in active]}"
        )
    return active[0] if active else None
