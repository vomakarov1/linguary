"""Chapter analysis result: a single LLM request before primary processing begins.

`AnalyzeChapter` returns glossary candidates, chapter translation instructions, block
treatment plan, `Before You Read`, and a concise chapter summary in a single call (PRD § 14).
An incomplete analysis prevents launching primary processing for that chapter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from linguary.domain.content_plan import ContentPlan
from linguary.domain.errors import AnalysisIncompleteError, DomainError
from linguary.domain.glossary import GlossaryCandidate
from linguary.domain.learning import BeforeYouReadEntry
from linguary.domain.treatment import CoverageStatus, TreatmentPlan


@dataclass(frozen=True, slots=True)
class ChapterAnalysis:
    """Persisted analysis result for a single chapter."""

    id: str
    section_id: str
    coverage_status: CoverageStatus
    treatment_plan: TreatmentPlan
    translation_instructions: str
    chapter_summary: str
    glossary_candidates: tuple[GlossaryCandidate, ...] = ()
    before_you_read: tuple[BeforeYouReadEntry, ...] = ()

    content_plan: ContentPlan | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise DomainError("идентификатор анализа главы пуст")
        if not self.section_id.strip():
            raise DomainError("анализ не привязан к главе")
        if self.treatment_plan.section_id != self.section_id:
            raise DomainError(
                f"план обработки относится к другой главе: {self.treatment_plan.section_id}"
            )
        for candidate in self.glossary_candidates:
            if candidate.analysis_id != self.id:
                raise DomainError(
                    f"кандидат {candidate.source_text!r} принадлежит другому анализу: "
                    f"{candidate.analysis_id}"
                )

    @classmethod
    def create(
        cls,
        *,
        id: str,
        section_id: str,
        treatment_plan: TreatmentPlan,
        translation_instructions: str,
        chapter_summary: str,
        expected_block_ids: Iterable[str],
        glossary_candidates: Iterable[GlossaryCandidate] = (),
        before_you_read: Iterable[BeforeYouReadEntry] = (),
        content_plan: ContentPlan | None = None,
    ) -> ChapterAnalysis:
        """Creates an analysis, computing plan coverage completeness for chapter blocks."""
        return cls(
            id=id,
            section_id=section_id,
            coverage_status=treatment_plan.coverage_for(expected_block_ids),
            treatment_plan=treatment_plan,
            translation_instructions=translation_instructions,
            chapter_summary=chapter_summary,
            glossary_candidates=tuple(glossary_candidates),
            before_you_read=tuple(before_you_read),
            content_plan=content_plan,
        )

    @property
    def is_complete(self) -> bool:
        """Whether all chapter blocks are covered by the plan."""
        return self.coverage_status is CoverageStatus.COMPLETE

    @property
    def candidates_requiring_review(self) -> tuple[GlossaryCandidate, ...]:
        """Candidates on which the user must make a decision (PRD § 12)."""
        return tuple(
            candidate for candidate in self.glossary_candidates if candidate.requires_review
        )

    def ensure_ready_for_processing(self) -> None:
        """Verifies that the chapter is ready for processing."""
        if not self.is_complete:
            raise AnalysisIncompleteError(
                f"анализ главы {self.section_id} неполон: обработку запускать нельзя"
            )
