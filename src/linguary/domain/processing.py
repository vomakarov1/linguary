"""Обработка книги: вычислительная группа блоков, результаты и запуск обработки.

`TranslationUnit` — временная группа последовательных текстовых блоков для одного вызова
LLM. Результат сохраняется отдельно для каждого входного блока, но только после успешной
проверки всей группы (PRD § 10, § 17; ADR § 3).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

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
    """Перевод отдельного слова или короткого выражения внутри блока."""

    original: str
    translation: str
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.original.strip():
            raise BlockResultError("исходное выражение словарной подсказки пусто")
        if not self.translation.strip():
            raise BlockResultError(f"перевод подсказки {self.original!r} пуст")


class ResultWarningKind(StrEnum):
    """Виды непреградных замечаний к результату обработки (PRD § 15)."""

    REPETITION = "repetition"
    DROPPED_BLOCK = "dropped_block"


@dataclass(frozen=True, slots=True)
class ResultWarning:
    """Замечание, которое сохраняется вместе с результатом и не блокирует обработку."""

    kind: ResultWarningKind
    message: str

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise BlockResultError("текст замечания к результату пуст")


@dataclass(frozen=True, slots=True)
class BlockProcessingResult:
    """Результат обработки одного входного блока.

    Правила состава определяются режимом из сохранённого плана (PRD § 14, § 15):
    полный перевод обязателен и допустим только для `TRANSLATE`; для `GLOSSARY_ONLY`
    обязательны словарные подсказки; для `READ_AS_IS` нет ни перевода, ни подсказок.
    """

    block_id: str
    treatment: BlockTreatment
    translated_text: str | None = None
    glossary_hints: tuple[GlossaryHint, ...] = ()
    language_notes: tuple[LanguageNote, ...] = ()
    warnings: tuple[ResultWarning, ...] = ()

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
    """Блок группы вместе с его режимом обработки из сохранённого плана."""

    block: DocumentBlock
    treatment: BlockTreatment

    def __post_init__(self) -> None:
        if not self.treatment.is_textual:
            raise NonTextualBlockError(
                f"блок {self.block.id} с режимом {self.treatment} не передаётся модели"
            )


@dataclass(frozen=True, slots=True)
class TranslationUnit:
    """Группа последовательных текстовых блоков одной главы для одного вызова LLM.

    Единица вычисления, а не хранения: сохраняются результаты отдельных блоков.
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
        """Собирает группу из блоков главы, взяв режимы из сохранённого плана.

        Блоки `PRESERVE_ONLY` и `DROP_WITH_WARNING` в группу не попадают: попытка
        включить такой блок — нарушение инварианта, а не повод его отфильтровать.
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
        """Блоки группы в порядке исходного документа."""
        return tuple(item.block for item in self.items)

    @property
    def block_ids(self) -> tuple[str, ...]:
        """Идентификаторы блоков группы в порядке исходного документа."""
        return tuple(item.block.id for item in self.items)

    def treatment_for(self, block_id: str) -> BlockTreatment:
        """Режим обработки блока группы по сохранённому плану."""
        for item in self.items:
            if item.block.id == block_id:
                return item.treatment
        raise UnknownBlockError(f"блок {block_id} отсутствует в группе {self.id}")

    def validate_results(self, results: Sequence[BlockProcessingResult]) -> None:
        """Блокирующие проверки результата всей группы (PRD § 15).

        Проверяются наличие всех ожидаемых блоков, отсутствие неизвестных, сохранение
        порядка и совпадение режима с планом. Любое нарушение отменяет сохранение всей
        группы: частично сохранённого результата быть не должно.
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
    """Состояние запуска обработки книги."""

    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class GlossarySnapshotEntry:
    """Утверждённое переводческое решение на момент запуска обработки.

    `note` — переводческое указание по этому решению: то, что нельзя выразить одним
    вариантом перевода, но обязано совпадать во всей книге (PRD § 11).
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
    """Снимок глоссария, с которым выполняется запуск обработки."""

    entries: tuple[GlossarySnapshotEntry, ...] = ()

    def __post_init__(self) -> None:
        texts = [entry.source_text for entry in self.entries]
        if len(set(texts)) != len(texts):
            raise ProcessingRunError("снимок глоссария содержит повторяющиеся исходные тексты")

    @property
    def mandatory_entries(self) -> tuple[GlossarySnapshotEntry, ...]:
        """Решения, которые передаются модели как обязательные."""
        return tuple(entry for entry in self.entries if entry.status.is_mandatory)


@dataclass(frozen=True, slots=True)
class LLMSettingsSnapshot:
    """Настройки LLM, зафиксированные на момент запуска обработки."""

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
    """Запуск обработки книги со снимком глоссария и настроек LLM.

    У книги одновременно существует не более одного активного запуска: параллельные
    наборы результатов не накапливаются (ADR § 3).
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
        """Является ли запуск текущим для книги."""
        return self.status is ProcessingRunStatus.ACTIVE


def active_run(runs: Iterable[ProcessingRun]) -> ProcessingRun | None:
    """Единственный активный запуск книги либо `None`, если активного нет."""
    active = [run for run in runs if run.is_active]
    if len(active) > 1:
        raise ActiveRunConflictError(
            f"у книги больше одного активного запуска обработки: {[run.id for run in active]}"
        )
    return active[0] if active else None
