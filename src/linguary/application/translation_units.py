"""Формирование `TranslationUnit`: что уходит модели одним вызовом (PRD § 10, ALG-04).

Группа — единица вычисления, а не хранения: модель видит её целиком ради контекста, но
возвращает отдельный результат для каждого входного блока, и сохраняются результаты
блоков (ADR § 3). Задача этого модуля — решить, где проходят границы групп.

Размер группы считается символами исходного текста, а не числом блоков. У обработки
профиль расхода не такой, как у анализа главы: анализ отвечает режимом и короткой
причиной, то есть примерно одинаковым числом токенов на блок любой длины, а обработка
возвращает полный перевод, объём которого пропорционален объёму оригинала. Сорок реплик
диалога и сорок страничных абзацев — запросы разной цены, поэтому предел
`analysis_merge.DEFAULT_MAX_FRAGMENT_BLOCKS` сюда не переносится: здесь свой предел
`DEFAULT_MAX_UNIT_SOURCE_CHARS`, и он настраивается (`LLMSettings.max_unit_source_chars`).

Блоки `PRESERVE_ONLY` и `DROP_WITH_WARNING` в группы не попадают и соседний контекст не
разрывают: они остаются работой импортёра и экспортёра (PRD § 10.1).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from linguary.application.context_builder import is_textual
from linguary.domain import (
    BlockType,
    DocumentBlock,
    TranslationUnit,
    TranslationUnitError,
    TreatmentPlan,
)

#: Сколько символов оригинала уходит в один запрос обработки. PRD § 10 задаёт ориентир
#: 2000-5000 токенов исходного текста; 8000 символов — это около 2000 токенов английского
#: оригинала (примерно четыре символа на токен), то есть нижняя граница ориентира.
#: Выход при этом остаётся в типовом пределе структурированного ответа: полный русский
#: перевод такого объёма вместе со словарными подсказками, языковыми заметками и
#: JSON-обвязкой занимает порядка 5000-6000 токенов.
DEFAULT_MAX_UNIT_SOURCE_CHARS = 8000

#: Типы блоков, идущие подряд как одно целое: список, стихотворный фрагмент, цитата.
#: Разорванный посередине список модель видит без начала, а стихи — без строфы.
_COHESIVE_BLOCK_TYPES = frozenset(
    {BlockType.LIST_ITEM, BlockType.POEM, BlockType.QUOTE, BlockType.TABLE}
)


def plan_translation_units(
    blocks: Iterable[DocumentBlock],
    plan: TreatmentPlan,
    *,
    id_factory: Callable[[], str],
    max_source_chars: int = DEFAULT_MAX_UNIT_SOURCE_CHARS,
) -> tuple[TranslationUnit, ...]:
    """Делит блоки главы на группы для вызовов модели (PRD § 10).

    Принимает блоки главы в порядке исходного документа; нетекстовые отсеиваются здесь же
    (`context_builder.is_textual`), режимы берутся из сохранённого плана. Текстовый блок,
    которого в плане нет, — не повод молча его пропустить: обработка запускается только
    после полного анализа главы, поэтому такой блок поднимает `UnknownBlockError`.

    Глава без текстовых блоков (обложка, страница с иллюстрацией) даёт пустой результат:
    обрабатывать нечего, и вызова модели не будет.
    """
    if max_source_chars < 1:
        raise TranslationUnitError("предел размера группы блоков должен быть положительным")
    ordered = tuple(
        block
        for block in blocks
        if is_textual(block) and plan.treatment_for(block.id).is_textual
    )
    if not ordered:
        return ()
    return tuple(
        TranslationUnit.of_blocks(id=id_factory(), blocks=group, plan=plan)
        for group in _grouped(ordered, max_source_chars=max_source_chars)
    )


def _grouped(
    blocks: Sequence[DocumentBlock], *, max_source_chars: int
) -> tuple[tuple[DocumentBlock, ...], ...]:
    """Границы групп: связные куски целиком, размер — ровными долями главы.

    Глава делится не кусками предельного размера, а `ceil(объём / предел)` долями примерно
    одного объёма: хвост в один абзац стоил бы отдельного вызова и остался бы без контекста
    сцены. Предел при этом остаётся жёстким — доля никогда его не превышает, кроме случая,
    когда предела не хватает одному блоку: `DocumentBlock` не делится (ADR § 3).
    """
    total = sum(len(block.text) for block in blocks)
    count = max(1, -(-total // max_source_chars))
    groups: list[list[DocumentBlock]] = []
    current: list[DocumentBlock] = []
    current_size = 0
    placed = 0
    for cohesive in _cohesive_runs(blocks, max_source_chars=max_source_chars):
        size = sum(len(block.text) for block in cohesive)
        if current and _closes_group(
            position=placed + current_size,
            size=size,
            current_size=current_size,
            filled=len(groups),
            count=count,
            total=total,
            max_source_chars=max_source_chars,
        ):
            groups.append(current)
            placed += current_size
            current, current_size = [], 0
        current.extend(cohesive)
        current_size += size
    groups.append(current)
    return tuple(tuple(group) for group in groups)


def _closes_group(
    *,
    position: int,
    size: int,
    current_size: int,
    filled: int,
    count: int,
    total: int,
    max_source_chars: int,
) -> bool:
    """Закрывается ли текущая группа перед очередным связным куском.

    Два правила: жёсткий предел размера запроса и близость к границе ровной доли главы.
    Второе правило и удерживает хвост главы от превращения в отдельный крошечный запрос —
    кусок берётся в группу, пока он приближает её к границе доли, а не удаляет от неё.
    """
    if current_size + size > max_source_chars:
        return True
    if filled >= count - 1:
        return False
    boundary = total * (filled + 1) / count
    return abs(position + size - boundary) > abs(position - boundary)


def _cohesive_runs(
    blocks: Sequence[DocumentBlock], *, max_source_chars: int
) -> tuple[tuple[DocumentBlock, ...], ...]:
    """Куски главы, которые нельзя разрывать между запросами (PRD § 10).

    Связными считаются соседние блоки одного составного целого — с общим родителем — и
    идущие подряд блоки списка, стихов, цитаты или таблицы. Диалоги и логически связанные
    абзацы структурного признака не имеют: их удерживает вместе сам порядок блоков внутри
    группы. Кусок, который сам не помещается в запрос, разбирается на отдельные блоки:
    иначе он утащил бы группу за предел размера.
    """
    runs: list[list[DocumentBlock]] = []
    for block in blocks:
        if runs and _is_cohesive_with(runs[-1][-1], block):
            runs[-1].append(block)
        else:
            runs.append([block])
    return tuple(
        run
        for group in runs
        for run in _split_oversized(group, max_source_chars=max_source_chars)
    )


def _is_cohesive_with(previous: DocumentBlock, block: DocumentBlock) -> bool:
    """Составляют ли соседние блоки одно целое."""
    if previous.parent_id is not None and previous.parent_id == block.parent_id:
        return True
    return previous.type is block.type and block.type in _COHESIVE_BLOCK_TYPES


def _split_oversized(
    run: Sequence[DocumentBlock], *, max_source_chars: int
) -> tuple[tuple[DocumentBlock, ...], ...]:
    """Связный кусок целиком либо, если он длиннее предела, его отдельные блоки."""
    if sum(len(block.text) for block in run) <= max_source_chars:
        return (tuple(run),)
    return tuple((block,) for block in run)


__all__ = [
    "DEFAULT_MAX_UNIT_SOURCE_CHARS",
    "plan_translation_units",
]
