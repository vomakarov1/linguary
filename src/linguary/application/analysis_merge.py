"""Резервный сценарий длинной главы: дробление и слияние фрагментов (ALG-03, PRD § 14).

Анализ главы одним запросом — основной сценарий. Глава, которая не помещается в лимиты
модели или технически представлена слишком крупным XHTML-документом, делится на
последовательные фрагменты: каждый фрагмент анализируется отдельным запросом и возвращает
кандидатов глоссария, новые инструкции и план обработки только для своего диапазона
блоков. Сводка описывает главу от начала до конца фрагмента. Этот модуль отвечает за
обе границы резервного сценария — как глава распадается
на фрагменты и как ответы фрагментов снова становятся одним `ChapterAnalysis`.

Слияние ничего не решает за домен: конфликт вариантов перевода передаётся в
`GlossaryCandidate.create` признаком `has_translation_conflict`, а итоговое
`requires_review` вычисляет `compute_requires_review` по DTO-01. Полнота покрытия тоже
остаётся доменным вычислением `TreatmentPlan.coverage_for`: модуль лишь собирает план
и перечень ожидаемых блоков главы.

Неполный анализ обработку главы блокирует: `ChapterAnalysis.ensure_ready_for_processing`
отвергает главу с покрытием `incomplete`, и потерянный фрагмент лечится повтором анализа,
а не частичной обработкой.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from linguary.application.block_labels import without_block_labels
from linguary.application.errors import ApplicationError
from linguary.application.llm_dto import (
    BeforeYouReadEntryResponse,
    ChapterAnalysisResponse,
    GlossaryCandidateResponse,
)
from linguary.domain import (
    BeforeYouReadEntry,
    BlockTreatment,
    ChapterAnalysis,
    DocumentBlock,
    GlossaryCandidate,
    TreatmentPlan,
    TreatmentPlanItem,
    glossary_key,
    spellings_by_preference,
    translations_conflict,
)

#: Резервный предел, если провайдер не сообщает доступный объём вывода.
DEFAULT_MAX_FRAGMENT_BLOCKS = 40
#: Верхняя оценка расхода ответа на один анализируемый блок из живого прогона.
TOKENS_PER_ANALYSED_BLOCK = 200
MAX_BEFORE_YOU_READ = 10


def fragment_blocks_for(max_output_tokens: int | None) -> int:
    """Число блоков анализа, которое допускает предел ответа модели."""
    if max_output_tokens is None:
        return DEFAULT_MAX_FRAGMENT_BLOCKS
    return max(1, max_output_tokens // TOKENS_PER_ANALYSED_BLOCK)


class AnalysisMergeError(ApplicationError):
    """Фрагменты анализа главы нельзя разобрать или собрать вместе."""


@dataclass(frozen=True, slots=True)
class ChapterFragment:
    """Последовательный фрагмент длинной главы: часть её текстовых блоков.

    `order` задаёт место фрагмента в порядке чтения главы.
    """

    order: int
    blocks: tuple[DocumentBlock, ...]

    def __post_init__(self) -> None:
        if self.order < 0:
            raise AnalysisMergeError("порядковый номер фрагмента главы отрицателен")
        if not self.blocks:
            raise AnalysisMergeError("фрагмент главы без текстовых блоков не анализируется")

    @property
    def block_ids(self) -> tuple[str, ...]:
        """Идентификаторы блоков фрагмента в порядке исходного документа."""
        return tuple(block.id for block in self.blocks)


@dataclass(frozen=True, slots=True)
class FragmentAnalysis:
    """Разобранный ответ модели по одному фрагменту главы.

    Псевдонимы запроса здесь уже возвращены к стабильным идентификаторам блоков: слияние
    работает с блоками главы, а не с нумерацией отдельного запроса.
    """

    order: int
    translation_instructions: str
    chapter_summary: str
    treatments: tuple[TreatmentPlanItem, ...] = ()
    glossary_candidates: tuple[GlossaryCandidateResponse, ...] = ()
    before_you_read: tuple[BeforeYouReadEntryResponse, ...] = ()

    def __post_init__(self) -> None:
        if self.order < 0:
            raise AnalysisMergeError("порядковый номер фрагмента главы отрицателен")

    @classmethod
    def of_response(
        cls,
        payload: ChapterAnalysisResponse,
        *,
        order: int,
        block_ids_by_alias: Mapping[str, str],
    ) -> FragmentAnalysis:
        """Ответ фрагмента с планом, переведённым в идентификаторы блоков главы.

        План собирается в порядке исходного документа: `block_ids_by_alias` перечисляет
        блоки фрагмента так, как их получила модель. Блок, о котором фрагмент ничего не
        сказал, в план не попадает и проявится неполным покрытием главы при слиянии.
        """
        treatments = {
            item.source_block_id: item.treatment for item in payload.treatment_plan
        }
        unknown = sorted(set(treatments) - set(block_ids_by_alias))
        if unknown:
            raise AnalysisMergeError(
                f"фрагмент {order} вернул режимы для чужих блоков: {unknown}"
            )
        return cls(
            order=order,
            translation_instructions=without_block_labels(payload.translation_instructions),
            chapter_summary=without_block_labels(payload.chapter_summary),
            treatments=tuple(
                TreatmentPlanItem(block_id=block_id, treatment=treatments[alias])
                for alias, block_id in block_ids_by_alias.items()
                if alias in treatments
            ),
            glossary_candidates=payload.glossary_candidates,
            before_you_read=payload.before_you_read,
        )


def plan_fragments(
    blocks: Sequence[DocumentBlock],
    *,
    max_blocks: int = DEFAULT_MAX_FRAGMENT_BLOCKS,
) -> tuple[ChapterFragment, ...]:
    """Делит текстовые блоки главы на последовательные фрагменты анализа (PRD § 14).

    Глава, помещающаяся в один запрос, остаётся единственным фрагментом: дробление —
    резервный сценарий, а не обычный путь. Длинная глава делится на равные по числу
    блоков части, а не отрезается кусками предельного размера: хвост в один абзац дал бы
    модели фрагмент без контекста и лишний вызов.
    """
    if max_blocks < 1:
        raise AnalysisMergeError("размер фрагмента главы должен быть положительным")
    total = len(blocks)
    if total == 0:
        return ()
    count = -(-total // max_blocks)
    size, remainder = divmod(total, count)
    fragments: list[ChapterFragment] = []
    start = 0
    for order in range(count):
        length = size + (1 if order < remainder else 0)
        fragments.append(
            ChapterFragment(order=order, blocks=tuple(blocks[start : start + length]))
        )
        start += length
    return tuple(fragments)


def merge_chapter_analysis(
    fragments: Iterable[FragmentAnalysis],
    *,
    id: str,
    section_id: str,
    expected_block_ids: Iterable[str],
    glossary_confidence_threshold: float,
    id_factory: Callable[[], str],
) -> ChapterAnalysis:
    """Собирает анализ разбитой главы из ответов её фрагментов (ALG-03).

    `expected_block_ids` — все текстовые блоки главы в порядке документа: относительно них
    и считается покрытие. Фрагмент, который не выполнен, просто отсутствует в `fragments`,
    и его блоки остаются непокрытыми — анализ главы получает статус `incomplete`.
    """
    ordered = _ordered_fragments(fragments)
    expected = tuple(expected_block_ids)
    return ChapterAnalysis.create(
        id=id,
        section_id=section_id,
        treatment_plan=_merged_plan(ordered, section_id=section_id, expected=expected),
        translation_instructions=_merged_instructions(ordered),
        chapter_summary=_merged_summary(ordered),
        expected_block_ids=expected,
        glossary_candidates=_merged_candidates(
            ordered,
            analysis_id=id,
            confidence_threshold=glossary_confidence_threshold,
            id_factory=id_factory,
        ),
        before_you_read=_merged_before_you_read(ordered),
    )


def _ordered_fragments(
    fragments: Iterable[FragmentAnalysis],
) -> tuple[FragmentAnalysis, ...]:
    """Выполненные фрагменты в порядке чтения главы."""
    ordered = tuple(sorted(fragments, key=lambda fragment: fragment.order))
    orders = [fragment.order for fragment in ordered]
    if len(set(orders)) != len(orders):
        raise AnalysisMergeError(
            f"фрагменты главы получили повторяющиеся номера: {sorted(orders)}"
        )
    return ordered


def _merged_plan(
    fragments: Sequence[FragmentAnalysis], *, section_id: str, expected: Sequence[str]
) -> TreatmentPlan:
    """План обработки главы, собранный по `block_id` (ALG-03).

    Два фрагмента, назвавшие для одного блока разные режимы, решения не дают: приложение
    не выбирает между ними, блок остаётся без режима и глава оказывается покрытой неполно.
    Повтор одного и того же режима конфликтом не считается — фрагменты согласны.
    """
    decided: dict[str, BlockTreatment] = {}
    conflicting: set[str] = set()
    for fragment in fragments:
        for item in fragment.treatments:
            previous = decided.get(item.block_id)
            if previous is None:
                decided[item.block_id] = item.treatment
            elif previous is not item.treatment:
                conflicting.add(item.block_id)
    for block_id in conflicting:
        del decided[block_id]
    known = set(expected)
    order = [block_id for block_id in expected if block_id in decided]
    order.extend(block_id for block_id in decided if block_id not in known)
    return TreatmentPlan.of_treatments(
        section_id, ((block_id, decided[block_id]) for block_id in order)
    )


def _merged_instructions(fragments: Sequence[FragmentAnalysis]) -> str:
    """Инструкции перевода главы: локальные указания фрагментов в порядке чтения.

    Одинаковые указания разных фрагментов не повторяются: голос рассказчика и речевые
    особенности персонажей модель замечает в каждом фрагменте, а переводчику главы этот
    пункт нужен один раз.
    """
    points: dict[str, str] = {}
    for fragment in fragments:
        for point in _instruction_points(fragment.translation_instructions):
            cleaned = without_block_labels(point)
            key = re.sub(r"\s+", " ", cleaned).casefold().strip(" \t.,;:!?")
            if key:
                points.setdefault(key, cleaned)
    return "\n".join(
        f"{index}. {point}" for index, point in enumerate(points.values(), 1)
    )


def _instruction_points(text: str) -> tuple[str, ...]:
    """Разбирает пункты с маркерами и обычные абзацы."""
    points: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        lines = paragraph.splitlines()
        if len(lines) > 1 and any(re.match(r"^\s*(?:[-•]|\d+\.)\s+", line) for line in lines):
            points.extend(
                re.sub(r"^\s*(?:[-•]|\d+\.)\s+", "", line).strip()
                for line in lines
                if line.strip()
            )
        elif paragraph.strip():
            points.append(re.sub(r"^\s*(?:[-•]|\d+\.)\s+", "", paragraph).strip())
    return tuple(points)


def _merged_summary(fragments: Sequence[FragmentAnalysis]) -> str:
    """Последняя непустая сводка охватывает главу до конца своего фрагмента."""
    return next(
        (
            summary
            for fragment in reversed(fragments)
            if (summary := fragment.chapter_summary.strip())
        ),
        "",
    )


def _merged_candidates(
    fragments: Sequence[FragmentAnalysis],
    *,
    analysis_id: str,
    confidence_threshold: float,
    id_factory: Callable[[], str],
) -> tuple[GlossaryCandidate, ...]:
    """Кандидаты глоссария главы, объединённые по исходному тексту (PRD § 11-12).

    Очевидные варианты написания одного текста — регистр, типографика, точка после
    `Mr` — объединяются ключом `glossary_key`: иначе каждое написание имени стало бы
    отдельным обязательным решением, а решения по ним могли бы разойтись.
    """
    groups: dict[str, list[GlossaryCandidateResponse]] = {}
    for fragment in fragments:
        for candidate in fragment.glossary_candidates:
            groups.setdefault(glossary_key(candidate.source_text), []).append(candidate)
    return tuple(
        _merged_candidate(
            group,
            id=id_factory(),
            analysis_id=analysis_id,
            confidence_threshold=confidence_threshold,
        )
        for group in groups.values()
    )


def _merged_candidate(
    group: Sequence[GlossaryCandidateResponse],
    *,
    id: str,
    analysis_id: str,
    confidence_threshold: float,
) -> GlossaryCandidate:
    """Один кандидат главы из вариантов, предложенных её фрагментами.

    Вариант перевода берётся у самого уверенного фрагмента, а сигналы проверки
    объединяются: пометка модели хотя бы от одного фрагмента, тип, требующий решения
    всегда, и расхождение вариантов перевода делают кандидата требующим внимания.
    Согласие фрагментов на одном варианте, наоборот, само по себе внимания не требует.

    Исходный текст — самое частое написание среди фрагментов (`spellings_by_preference`).

    Переводческое указание берётся у самого уверенного фрагмента, который его дал:
    пустое указание не должно вытеснять непустое, потому что фрагменты видят разные
    части главы и признак, требующий указания, встречается не в каждой из них.
    """
    leading = max(group, key=lambda candidate: candidate.confidence)
    return GlossaryCandidate.create(
        id=id,
        analysis_id=analysis_id,
        source_text=spellings_by_preference(
            (candidate.source_text, candidate.confidence) for candidate in group
        )[0],
        type=next(
            (
                candidate.type
                for candidate in group
                if candidate.type.always_requires_review
            ),
            leading.type,
        ),
        suggested_translation=leading.suggested_translation,
        confidence=leading.confidence,
        model_flagged_review=any(candidate.model_flagged_review for candidate in group),
        confidence_threshold=confidence_threshold,
        has_translation_conflict=translations_conflict(
            candidate.suggested_translation for candidate in group
        ),
        note=next(
            (
                candidate.note
                for candidate in sorted(
                    group, key=lambda candidate: candidate.confidence, reverse=True
                )
                if candidate.note.strip()
            ),
            "",
        ),
    )


def _merged_before_you_read(
    fragments: Sequence[FragmentAnalysis],
) -> tuple[BeforeYouReadEntry, ...]:
    """Материал `Before You Read` главы: выражения фрагментов без повторов (PRD § 19.1)."""
    return before_you_read_entries(
        entry for fragment in fragments for entry in fragment.before_you_read
    )


def before_you_read_entries(
    entries: Iterable[BeforeYouReadEntryResponse],
) -> tuple[BeforeYouReadEntry, ...]:
    """Оставляет первые десять различных выражений главы."""
    selected: dict[str, BeforeYouReadEntry] = {}
    for entry in entries:
        key = (
            re.sub(r"\s+", " ", entry.expression)
            .casefold()
            .strip(" \t.,;:!?\"'«»()[]")
        )
        if key and key not in selected:
            selected[key] = BeforeYouReadEntry(
                expression=entry.expression, explanation=entry.explanation
            )
        if len(selected) >= MAX_BEFORE_YOU_READ:
            break
    return tuple(selected.values())


__all__ = [
    "DEFAULT_MAX_FRAGMENT_BLOCKS",
    "MAX_BEFORE_YOU_READ",
    "TOKENS_PER_ANALYSED_BLOCK",
    "before_you_read_entries",
    "fragment_blocks_for",
    "AnalysisMergeError",
    "ChapterFragment",
    "FragmentAnalysis",
    "merge_chapter_analysis",
    "plan_fragments",
]
