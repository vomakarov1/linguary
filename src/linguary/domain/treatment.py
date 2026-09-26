"""Режимы обработки блоков и план обработки главы.

План (`TreatmentPlan`) получают при анализе главы и сохраняют до запуска основной обработки.
Он — единственный источник истины о режиме блока: поле `treatment` в ответе модели только
подтверждает план и не может его переопределить (PRD § 14, ADR § 3).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from linguary.domain.errors import TreatmentPlanError, UnknownBlockError


class BlockTreatment(StrEnum):
    """Режимы обработки блока (PRD § 10.1)."""

    TRANSLATE = "TRANSLATE"
    GLOSSARY_ONLY = "GLOSSARY_ONLY"
    READ_AS_IS = "READ_AS_IS"
    PRESERVE_ONLY = "PRESERVE_ONLY"
    DROP_WITH_WARNING = "DROP_WITH_WARNING"

    @property
    def is_textual(self) -> bool:
        """Блок является частью текста для чтения и передаётся модели."""
        return self in (
            BlockTreatment.TRANSLATE,
            BlockTreatment.GLOSSARY_ONLY,
            BlockTreatment.READ_AS_IS,
        )

    @property
    def requires_translation(self) -> bool:
        """Для режима обязателен полный русский перевод блока."""
        return self is BlockTreatment.TRANSLATE

    @property
    def allows_translation(self) -> bool:
        """Полный перевод допустим только там, где он обязателен (PRD § 15)."""
        return self.requires_translation

    @property
    def requires_glossary_hints(self) -> bool:
        """Для режима обязательны словарные подсказки."""
        return self is BlockTreatment.GLOSSARY_ONLY

    @property
    def allows_glossary_hints(self) -> bool:
        """Словарные подсказки запрещены только для `READ_AS_IS`."""
        return self in (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY)

    @property
    def allows_language_notes(self) -> bool:
        """`Language Notes` возможны при любом текстовом режиме (PRD § 10.1)."""
        return self.is_textual


class CoverageStatus(StrEnum):
    """Полнота покрытия текстовых блоков главы планом обработки (PRD § 14)."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class TreatmentPlanItem:
    """Режим обработки одного блока."""

    block_id: str
    treatment: BlockTreatment

    def __post_init__(self) -> None:
        if not self.block_id.strip():
            raise TreatmentPlanError("идентификатор блока в плане обработки пуст")


@dataclass(frozen=True, slots=True)
class TreatmentPlan:
    """План обработки блоков одной главы.

    Каждый блок встречается в плане ровно один раз: при слиянии фрагментов анализа
    приложение собирает план по `block_id`, а повтор означает конфликт фрагментов.
    """

    section_id: str
    items: tuple[TreatmentPlanItem, ...] = ()

    def __post_init__(self) -> None:
        if not self.section_id.strip():
            raise TreatmentPlanError("план обработки не привязан к главе")
        seen: set[str] = set()
        for item in self.items:
            if item.block_id in seen:
                raise TreatmentPlanError(
                    f"блок {item.block_id} встречается в плане главы {self.section_id} дважды"
                )
            seen.add(item.block_id)

    @classmethod
    def of_treatments(
        cls, section_id: str, treatments: Iterable[tuple[str, BlockTreatment]]
    ) -> TreatmentPlan:
        """Собирает план из пар «идентификатор блока — режим»."""
        return cls(
            section_id=section_id,
            items=tuple(
                TreatmentPlanItem(block_id=block_id, treatment=treatment)
                for block_id, treatment in treatments
            ),
        )

    @property
    def block_ids(self) -> tuple[str, ...]:
        """Идентификаторы всех блоков плана в порядке их добавления."""
        return tuple(item.block_id for item in self.items)

    def treatment_for(self, block_id: str) -> BlockTreatment:
        """Режим обработки блока по плану."""
        for item in self.items:
            if item.block_id == block_id:
                return item.treatment
        raise UnknownBlockError(
            f"блок {block_id} отсутствует в плане обработки главы {self.section_id}"
        )

    def block_ids_with(self, *treatments: BlockTreatment) -> tuple[str, ...]:
        """Блоки плана с одним из перечисленных режимов."""
        wanted = frozenset(treatments)
        return tuple(item.block_id for item in self.items if item.treatment in wanted)

    @property
    def textual_block_ids(self) -> tuple[str, ...]:
        """Блоки, которые передаются модели."""
        return tuple(item.block_id for item in self.items if item.treatment.is_textual)

    def coverage_for(self, expected_block_ids: Iterable[str]) -> CoverageStatus:
        """Полнота плана относительно ожидаемых текстовых блоков главы.

        Анализ считается полным, только если каждый ожидаемый блок покрыт планом
        ровно один раз; лишние блоки в плане тоже делают его неполным.
        """
        expected = tuple(expected_block_ids)
        if len(set(expected)) != len(expected):
            raise TreatmentPlanError("ожидаемые блоки главы содержат повторы")
        return (
            CoverageStatus.COMPLETE
            if set(expected) == set(self.block_ids)
            else CoverageStatus.INCOMPLETE
        )
