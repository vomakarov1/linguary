"""Результат анализа главы: один LLM-запрос до начала обработки.

`AnalyzeChapter` за один вызов возвращает кандидатов глоссария, инструкции перевода главы,
план обработки блоков, `Before You Read` и краткую сводку событий главы (PRD § 14).
Неполный анализ не допускает запуск основной обработки этой главы.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from linguary.domain.errors import AnalysisIncompleteError, DomainError
from linguary.domain.glossary import GlossaryCandidate
from linguary.domain.learning import BeforeYouReadEntry
from linguary.domain.treatment import CoverageStatus, TreatmentPlan


@dataclass(frozen=True, slots=True)
class ChapterAnalysis:
    """Сохранённый результат анализа одной главы."""

    id: str
    section_id: str
    coverage_status: CoverageStatus
    treatment_plan: TreatmentPlan
    translation_instructions: str
    chapter_summary: str
    glossary_candidates: tuple[GlossaryCandidate, ...] = ()
    before_you_read: tuple[BeforeYouReadEntry, ...] = ()

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
    ) -> ChapterAnalysis:
        """Создаёт анализ, вычислив полноту покрытия блоков главы планом."""
        return cls(
            id=id,
            section_id=section_id,
            coverage_status=treatment_plan.coverage_for(expected_block_ids),
            treatment_plan=treatment_plan,
            translation_instructions=translation_instructions,
            chapter_summary=chapter_summary,
            glossary_candidates=tuple(glossary_candidates),
            before_you_read=tuple(before_you_read),
        )

    @property
    def is_complete(self) -> bool:
        """Покрыты ли планом все блоки главы."""
        return self.coverage_status is CoverageStatus.COMPLETE

    @property
    def candidates_requiring_review(self) -> tuple[GlossaryCandidate, ...]:
        """Кандидаты, по которым пользователь обязан принять решение (PRD § 12)."""
        return tuple(
            candidate for candidate in self.glossary_candidates if candidate.requires_review
        )

    def ensure_ready_for_processing(self) -> None:
        """Проверяет, что главу можно обрабатывать."""
        if not self.is_complete:
            raise AnalysisIncompleteError(
                f"анализ главы {self.section_id} неполон: обработку запускать нельзя"
            )
