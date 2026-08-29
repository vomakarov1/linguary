"""Резервный сценарий длинной главы: дробление и слияние фрагментов (ALG-03, PRD § 14).

Анализ главы одним запросом — основной сценарий. Глава, которая не помещается в лимиты
модели или технически представлена слишком крупным XHTML-документом, делится на
последовательные фрагменты: каждый фрагмент анализируется отдельным запросом и возвращает
кандидатов глоссария, локальные инструкции и план обработки только для своего диапазона
блоков. Этот модуль отвечает за обе границы резервного сценария — как глава распадается
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

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

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
    translations_conflict,
)

#: Сколько текстовых блоков главы уходит в один запрос анализа. Ориентир взят из живого
#: прогона: глава в 40 текстовых блоков стоила 4 000-7 900 токенов ответа, то есть порядка
#: 100-200 токенов на блок. При обычном пределе структурированного ответа около 8 000
#: токенов больше сорока блоков за один запрос модель не дописывает, и ответ обрывается
#: по лимиту вместо того, чтобы покрыть главу.
DEFAULT_MAX_FRAGMENT_BLOCKS = 40


class AnalysisMergeError(ApplicationError):
    """Фрагменты анализа главы нельзя разобрать или собрать вместе."""


@dataclass(frozen=True, slots=True)
class ChapterFragment:
    """Последовательный фрагмент длинной главы: часть её текстовых блоков.

    `order` задаёт место фрагмента в порядке чтения главы: в этом же порядке собирается
    сводка разбитой главы.
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
            translation_instructions=payload.translation_instructions,
            chapter_summary=payload.chapter_summary,
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
    return "\n\n".join(
        dict.fromkeys(
            instructions
            for fragment in fragments
            if (instructions := fragment.translation_instructions.strip())
        )
    )


def _merged_summary(fragments: Sequence[FragmentAnalysis]) -> str:
    """Сводка разбитой главы: сводки фрагментов подряд в порядке чтения (ALG-03).

    Отдельного запроса ради общей сводки не выполняется: читатель увидит события главы
    в том порядке, в каком они в ней произошли (PRD § 19.2).
    """
    return " ".join(
        summary for fragment in fragments if (summary := fragment.chapter_summary.strip())
    )


def _merged_candidates(
    fragments: Sequence[FragmentAnalysis],
    *,
    analysis_id: str,
    confidence_threshold: float,
    id_factory: Callable[[], str],
) -> tuple[GlossaryCandidate, ...]:
    """Кандидаты глоссария главы, объединённые по исходному тексту (PRD § 11-12)."""
    groups: dict[str, list[GlossaryCandidateResponse]] = {}
    for fragment in fragments:
        for candidate in fragment.glossary_candidates:
            groups.setdefault(candidate.source_text, []).append(candidate)
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
    """
    leading = max(group, key=lambda candidate: candidate.confidence)
    return GlossaryCandidate.create(
        id=id,
        analysis_id=analysis_id,
        source_text=leading.source_text,
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
    )


def _merged_before_you_read(
    fragments: Sequence[FragmentAnalysis],
) -> tuple[BeforeYouReadEntry, ...]:
    """Материал `Before You Read` главы: выражения фрагментов без повторов (PRD § 19.1)."""
    entries: dict[str, BeforeYouReadEntry] = {}
    for fragment in fragments:
        for entry in fragment.before_you_read:
            entries.setdefault(
                entry.expression,
                BeforeYouReadEntry(
                    expression=entry.expression, explanation=entry.explanation
                ),
            )
    return tuple(entries.values())


__all__ = [
    "DEFAULT_MAX_FRAGMENT_BLOCKS",
    "AnalysisMergeError",
    "ChapterFragment",
    "FragmentAnalysis",
    "merge_chapter_analysis",
    "plan_fragments",
]
