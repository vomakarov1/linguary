"""Block treatment modes and chapter treatment plan.

The plan (`TreatmentPlan`) is obtained during chapter analysis and persisted before primary
processing begins. It is the single source of truth for block treatment: the `treatment`
field in the model response merely confirms the plan and cannot override it (PRD § 14, ADR § 3).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from linguary.domain.errors import TreatmentPlanError, UnknownBlockError


class BlockTreatment(StrEnum):
    """Block treatment modes (PRD § 10.1)."""

    TRANSLATE = "TRANSLATE"
    GLOSSARY_ONLY = "GLOSSARY_ONLY"
    READ_AS_IS = "READ_AS_IS"
    PRESERVE_ONLY = "PRESERVE_ONLY"
    DROP_WITH_WARNING = "DROP_WITH_WARNING"

    @property
    def is_textual(self) -> bool:
        """Block is part of readable text and passed to the model."""
        return self in (
            BlockTreatment.TRANSLATE,
            BlockTreatment.GLOSSARY_ONLY,
            BlockTreatment.READ_AS_IS,
        )

    @property
    def requires_translation(self) -> bool:
        """The mode requires a full Russian translation of the block."""
        return self is BlockTreatment.TRANSLATE

    @property
    def allows_translation(self) -> bool:
        """Full translation is permitted only where required (PRD § 15)."""
        return self.requires_translation

    @property
    def requires_glossary_hints(self) -> bool:
        """The mode requires glossary hints."""
        return self is BlockTreatment.GLOSSARY_ONLY

    @property
    def allows_glossary_hints(self) -> bool:
        """Glossary hints are forbidden only for `READ_AS_IS`."""
        return self in (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY)

    @property
    def allows_language_notes(self) -> bool:
        """`Language Notes` are allowed for any textual mode (PRD § 10.1)."""
        return self.is_textual


class CoverageStatus(StrEnum):
    """Completeness of chapter textual block coverage by the treatment plan (PRD § 14)."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class TreatmentPlanItem:
    """Treatment mode for a single block."""

    block_id: str
    treatment: BlockTreatment

    def __post_init__(self) -> None:
        if not self.block_id.strip():
            raise TreatmentPlanError("идентификатор блока в плане обработки пуст")


@dataclass(frozen=True, slots=True)
class TreatmentPlan:
    """Treatment plan for blocks within a single chapter.

    Each block occurs in the plan exactly once: when merging analysis chunks,
    the application indexes the plan by `block_id`, and duplicates signify chunk conflicts.
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
        """Constructs a plan from block id and treatment pairs."""
        return cls(
            section_id=section_id,
            items=tuple(
                TreatmentPlanItem(block_id=block_id, treatment=treatment)
                for block_id, treatment in treatments
            ),
        )

    @property
    def block_ids(self) -> tuple[str, ...]:
        """Identifiers of all planned blocks in insertion order."""
        return tuple(item.block_id for item in self.items)

    def treatment_for(self, block_id: str) -> BlockTreatment:
        """Block treatment mode according to the plan."""
        for item in self.items:
            if item.block_id == block_id:
                return item.treatment
        raise UnknownBlockError(
            f"блок {block_id} отсутствует в плане обработки главы {self.section_id}"
        )

    def block_ids_with(self, *treatments: BlockTreatment) -> tuple[str, ...]:
        """Plan blocks with one of the specified treatments."""
        wanted = frozenset(treatments)
        return tuple(item.block_id for item in self.items if item.treatment in wanted)

    @property
    def textual_block_ids(self) -> tuple[str, ...]:
        """Blocks that are passed to the model."""
        return tuple(item.block_id for item in self.items if item.treatment.is_textual)

    def coverage_for(self, expected_block_ids: Iterable[str]) -> CoverageStatus:
        """Completeness of the plan relative to expected textual chapter blocks.

        Analysis is considered complete only if every expected block is covered by the plan
        exactly once; extraneous blocks in the plan also render it incomplete.
        """
        expected = tuple(expected_block_ids)
        if len(set(expected)) != len(expected):
            raise TreatmentPlanError("ожидаемые блоки главы содержат повторы")
        return (
            CoverageStatus.COMPLETE
            if set(expected) == set(self.block_ids)
            else CoverageStatus.INCOMPLETE
        )
