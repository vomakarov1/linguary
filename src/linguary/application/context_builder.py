"""Сборка контекста запросов к модели (PRD § 13, ADR § 3).

`ContextBuilder` — прикладная политика, которая решает, что именно видит модель. Порт
`LLMClient` о главах, глоссарии и блоках не знает: он получает готовый текст инструкции
и готовый текст запроса.

Контекст анализа главы состоит из текстовых блоков этой главы и уже принятых решений
книги — кандидатов глоссария из предыдущих глав. Текст соседних глав в него не попадает:
анализ главы зависит только от неё самой и от накопленных кандидатов, поэтому главы
анализируются строго последовательно (PRD § 13, § 25). Блоки `PRESERVE_ONLY` и
`DROP_WITH_WARNING` не попадают ни в основной контекст, ни в соседний: они остаются
ответственностью импортёра и экспортёра (ADR § 3).

Контекст обработки группы блоков состоит из самой группы, снимка глоссария запуска,
инструкций перевода главы и её сводки — выжимки анализа этой главы. Результаты обработки
других групп в него не входят: так задачи остаются независимыми, а результат
воспроизводимым (PRD § 13). Обязательные переводческие указания стоят в контексте
отдельно от необязательных подсказок: принятое, исправленное или зафиксированное
решение пользователя модель обязана соблюдать, вариант модели без решения и пропущенный
элемент — подсказка (PRD § 11-12). Решение пользователя сильнее указаний переводчику
главы: пункт указаний, который ему противоречит, в контекст не попадает
(`chapter_instructions`), а контекст прямо ставит решения выше указаний.

Контекст повторной генерации отличается от контекста группы двумя вещами: блок в нём
один и рядом с ним стоят несколько соседних абзацев главы. Соседи нужны, чтобы понять
сцену; псевдонимов они не получают, поэтому результат по ним модель вернуть не может
(PRD § 13, § 21.5).

Блоки нумеруются в запросе короткими псевдонимами `b1..bN`, а не своими стабильными
идентификаторами. Идентификатор блока — UUID: в главе на две сотни абзацев такие ключи
занимают заметную часть и запроса, и ответа, а модель обязана вернуть каждый из них
дословно. Псевдоним стоит несколько символов, переносится точнее и живёт ровно один
запрос — обратное отображение хранится здесь же, в `ChapterAnalysisContext`.

Инструкции написаны по-русски: тем же языком написана уточняющая инструкция повтора
(`retry_policy.build_repair_instruction`), которая дописывается к этому же тексту.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

from linguary.application.analysis_merge import ChapterFragment
from linguary.application.chapter_instructions import instructions_respecting_glossary
from linguary.application.errors import ApplicationError
from linguary.application.llm_dto import ChapterAnalysisResponse, TranslationResultResponse
from linguary.application.ports.llm import LLMRequest
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
    footnote_marker,
    footnote_references,
    glossary_key,
)

#: Сколько уже принятых решений книги попадает в контекст анализа очередной главы.
#: К концу романа кандидатов накапливаются сотни, а запросу нужны те, что действительно
#: повторяются: список ограничивается сверху, чтобы контекст не рос вместе с книгой
#: (PRD § 13: ограничение контекста настраивается).
DEFAULT_MAX_ACCUMULATED_CANDIDATES = 200

#: Сколько соседних текстовых блоков с каждой стороны видит модель при повторной генерации
#: одного блока. PRD § 13 просит «несколько соседних абзацев»: они нужны, чтобы понять сцену,
#: и не должны превращать запрос одного блока в запрос главы.
DEFAULT_REGENERATION_CONTEXT_BLOCKS = 2

#: Типы блоков, которые не являются текстом для чтения даже при полной поддержке импортёром.
#: Изображение поддержано целиком, но читателю оно даёт не текст, а картинку.
_NON_TEXTUAL_BLOCK_TYPES = frozenset({BlockType.IMAGE, BlockType.PAGE_BREAK})


class ContextBuildError(ApplicationError):
    """Контекст запроса собрать нельзя."""


def is_textual(block: DocumentBlock) -> bool:
    """Передаётся ли блок модели как часть текста главы (PRD § 10.1).

    Модели передаётся то, что читатель читает: полностью поддержанный блок с непустым
    текстом. Всё остальное — конструкции, перенесённые как есть (`PRESERVE_OPAQUE`),
    изображения, разрывы страниц, пустые блоки — остаётся работой импортёра и экспортёра.
    """
    return (
        block.support_status is SupportStatus.SUPPORTED
        and block.type not in _NON_TEXTUAL_BLOCK_TYPES
        and bool(block.text.strip())
    )


def textual_blocks(section: DocumentSection) -> tuple[DocumentBlock, ...]:
    """Текстовые блоки главы в порядке исходного документа."""
    return tuple(block for block in section.ordered_blocks if is_textual(block))


class AliasedBlocksContext:
    """Общая часть контекстов, адресующих блоки псевдонимами `b1..bN`.

    Отображение псевдонимов заполняет наследник: здесь только разбор ответа модели,
    одинаковый и для анализа главы, и для обработки группы блоков.
    """

    __slots__ = ()

    block_ids_by_alias: Mapping[str, str]

    @property
    def aliases(self) -> tuple[str, ...]:
        """Псевдонимы переданных блоков в порядке исходного документа."""
        return tuple(self.block_ids_by_alias)

    def block_id_for(self, alias: str) -> str:
        """Стабильный идентификатор блока по псевдониму из ответа модели."""
        block_id = self.block_ids_by_alias.get(alias)
        if block_id is None:
            raise ContextBuildError(f"блок {alias} в запрос не передавался")
        return block_id


@dataclass(frozen=True, slots=True)
class ChapterAnalysisContext(AliasedBlocksContext):
    """Готовый запрос анализа главы вместе с тем, что нужно для разбора ответа.

    `block_ids_by_alias` возвращает ответ модели к стабильным идентификаторам блоков,
    `expectation` описывает ожидаемый результат в том виде, в каком его видит модель:
    уточняющая инструкция повтора перечисляет псевдонимы, а не внутренние UUID.
    """

    section_id: str
    request: LLMRequest[ChapterAnalysisResponse]
    block_ids_by_alias: Mapping[str, str]
    expectation: ResultExpectation

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "block_ids_by_alias", MappingProxyType(dict(self.block_ids_by_alias))
        )


@dataclass(frozen=True, slots=True)
class TranslationContext(AliasedBlocksContext):
    """Готовый запрос обработки одной группы блоков (PRD § 13, ALG-04).

    `expectation` перечисляет псевдонимы вместе с режимами обработки из сохранённого
    плана: уточняющая инструкция повтора называет блоки так, как их видела модель,
    и повторяет требование режима (ALG-05).

    `explained_expressions` — выражения `Before You Read` этой главы: модель получает их
    в запросе как запрет повтора, а сценарий отсеивает повтор, если модель его всё же
    вернула (PRD § 19.3).
    """

    section_id: str
    unit_id: str
    request: LLMRequest[TranslationResultResponse]
    block_ids_by_alias: Mapping[str, str]
    expectation: ResultExpectation
    explained_expressions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "block_ids_by_alias", MappingProxyType(dict(self.block_ids_by_alias))
        )

    def to_aliases(self, message: str) -> str:
        """Сообщение проверки, в котором идентификаторы блоков заменены псевдонимами.

        Причина отбраковки уходит модели вместе с уточняющей инструкцией (ALG-05), а
        проверки формулируют её идентификаторами блоков группы — тех имён модель
        не видела.
        """
        for alias, block_id in self.block_ids_by_alias.items():
            message = message.replace(block_id, alias)
        return message

    def to_block_ids(
        self, payload: TranslationResultResponse
    ) -> TranslationResultResponse:
        """Ответ модели, переведённый с псевдонимов на идентификаторы блоков группы.

        Псевдоним, которого в запросе не было, — это результат для чужого блока
        (PRD § 15). Проверки состава работают уже с идентификаторами группы, поэтому
        такой ответ отбраковывается здесь и с тем же ожиданием, что и остальные
        блокирующие проверки: повтор получит перечень блоков, а не голое сообщение.
        """
        unknown = sorted(
            {
                alias
                for alias in (*payload.block_ids, *(
                    note.source_block_id for note in payload.language_notes
                ))
                if alias not in self.block_ids_by_alias
            }
        )
        if unknown:
            raise ResultValidationError(
                f"ответ группы {self.unit_id} относится к блокам вне запроса: {unknown}",
                expectation=self.expectation,
            )
        return TranslationResultResponse(
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
    """Соседний исходный текст одной стороны: абзацы до блока или после него."""

    title: str
    blocks: tuple[DocumentBlock, ...]


class ContextBuilder:
    """Собирает контекст запросов к модели по политике PRD § 13."""

    def __init__(
        self,
        *,
        max_accumulated_candidates: int = DEFAULT_MAX_ACCUMULATED_CANDIDATES,
        regeneration_context_blocks: int = DEFAULT_REGENERATION_CONTEXT_BLOCKS,
    ) -> None:
        if max_accumulated_candidates < 0:
            raise ContextBuildError(
                "ограничение числа накопленных кандидатов не может быть отрицательным"
            )
        if regeneration_context_blocks < 0:
            raise ContextBuildError(
                "число соседних блоков контекста не может быть отрицательным"
            )
        self._max_accumulated_candidates = max_accumulated_candidates
        self._regeneration_context_blocks = regeneration_context_blocks

    def for_chapter_analysis(
        self,
        *,
        document: CanonicalDocument,
        section: DocumentSection,
        accumulated_candidates: Iterable[GlossaryCandidate] = (),
    ) -> ChapterAnalysisContext:
        """Контекст анализа одной главы (PRD § 11, § 13, § 14, § 19).

        `accumulated_candidates` — кандидаты глоссария предыдущих глав книги в порядке
        чтения. Текст соседних глав в контекст не передаётся.
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
            instructions=chapter_analysis_instructions(
                source_language=document.source_language,
                target_language=document.target_language,
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
    ) -> ChapterAnalysisContext:
        """Контекст анализа одного фрагмента длинной главы (PRD § 14, ALG-03).

        Резервный сценарий: главу, которая не помещается в один запрос, модель видит
        по частям. Фрагмент получает только свои блоки — ни остальной главы, ни соседних
        глав в запросе нет, — и знает своё место в главе, потому что сводки фрагментов
        соединяются в сводку главы в порядке чтения.
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
            instructions=chapter_fragment_instructions(
                source_language=document.source_language,
                target_language=document.target_language,
            ),
            fragment_label=_fragment_label(fragment.order, fragment_count),
        )

    def for_translation_unit(
        self,
        *,
        document: CanonicalDocument,
        section: DocumentSection,
        unit: TranslationUnit,
        analysis: ChapterAnalysis,
        glossary: GlossarySnapshot,
    ) -> TranslationContext:
        """Контекст обработки одной группы блоков (PRD § 10, § 13, § 19.3, ALG-04).

        В запрос идут сама группа с режимами из сохранённого плана, снимок глоссария
        запуска, инструкции перевода главы и её сводка. Результатов других групп и текста
        соседних глав в контексте нет.
        """
        return self._translation_context(
            document=document,
            section=section,
            unit=unit,
            analysis=analysis,
            glossary=glossary,
            instructions=translation_unit_instructions(
                source_language=document.source_language,
                target_language=document.target_language,
            ),
            neighbours=(),
        )

    def for_block_regeneration(
        self,
        *,
        document: CanonicalDocument,
        section: DocumentSection,
        unit: TranslationUnit,
        analysis: ChapterAnalysis,
        glossary: GlossarySnapshot,
    ) -> TranslationContext:
        """Контекст повторной генерации одного блока (PRD § 13, § 19.3, § 21.5).

        Исходная группа блоков не пересчитывается: модель получает выбранный блок с его
        режимом из плана, ту же выжимку анализа главы и глоссарий, что и обычная обработка,
        и вдобавок несколько соседних абзацев — они нужны, чтобы понять сцену, псевдонимов
        не получают и результата по ним не ждут.
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
            instructions=block_regeneration_instructions(
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
        instructions: str,
        neighbours: Sequence[_NeighbourBlocks],
    ) -> TranslationContext:
        """Готовый запрос обработки переданных блоков вместе с разбором ответа."""
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
                instructions=instructions,
                input_text=self._translation_input(
                    document=document,
                    section=section,
                    unit=unit,
                    analysis=analysis,
                    glossary=glossary,
                    neighbours=neighbours,
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
        )

    def _translation_input(
        self,
        *,
        document: CanonicalDocument,
        section: DocumentSection,
        unit: TranslationUnit,
        analysis: ChapterAnalysis,
        glossary: GlossarySnapshot,
        neighbours: Sequence[_NeighbourBlocks],
    ) -> str:
        """Текст запроса обработки: шапка, выжимка анализа главы, глоссарий и блоки."""
        parts = [
            _chapter_header(document=document, section=section),
            _chapter_digest(analysis, glossary),
            _mandatory_decisions(
                glossary.mandatory_entries, target_language=document.target_language
            ),
            _glossary_suggestions(
                self._suggested_entries(glossary),
                target_language=document.target_language,
            ),
            _explained_expressions(analysis, section),
            _neighbour_context(neighbours),
            _unit_blocks(unit),
        ]
        return "\n\n".join(part for part in parts if part)

    def _neighbours(
        self, section: DocumentSection, unit: TranslationUnit
    ) -> tuple[_NeighbourBlocks, ...]:
        """Соседний исходный текст вокруг блока: несколько абзацев до и после (PRD § 13).

        В соседи идут только текстовые блоки той же главы: соседняя глава — уже другая
        сцена, а перенесённые как есть конструкции и иллюстрации модели не передаются.
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
        """Необязательные подсказки снимка глоссария в пределах ограничения контекста.

        Обязательные указания в это ограничение не входят: они решение пользователя,
        и урезать их приложение не вправе (PRD § 11).
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
    ) -> ChapterAnalysisContext:
        """Готовый запрос по переданным блокам главы вместе с разбором ответа."""
        aliases = {_alias(index): block.id for index, block in enumerate(blocks, start=1)}
        return ChapterAnalysisContext(
            section_id=section.id,
            request=LLMRequest.for_chapter_analysis(
                instructions=instructions,
                input_text=self._chapter_input(
                    document=document,
                    section=section,
                    blocks=blocks,
                    accumulated_candidates=accumulated_candidates,
                    fragment_label=fragment_label,
                ),
            ),
            block_ids_by_alias=aliases,
            expectation=ResultExpectation(
                target_language=document.target_language,
                blocks=tuple(ExpectedBlock(block_id=alias) for alias in aliases),
            ),
        )

    def _chapter_input(
        self,
        *,
        document: CanonicalDocument,
        section: DocumentSection,
        blocks: Sequence[DocumentBlock],
        accumulated_candidates: Iterable[GlossaryCandidate],
        fragment_label: str | None,
    ) -> str:
        """Текст запроса: шапка главы, принятые решения книги и блоки под псевдонимами."""
        parts = [
            _chapter_header(
                document=document, section=section, fragment_label=fragment_label
            ),
            _known_decisions(
                self._selected_candidates(accumulated_candidates),
                target_language=document.target_language,
            ),
            _chapter_blocks(blocks),
        ]
        return "\n\n".join(parts)

    def _selected_candidates(
        self, candidates: Iterable[GlossaryCandidate]
    ) -> tuple[GlossaryCandidate, ...]:
        """Решения предыдущих глав, которые попадут в контекст.

        Повторы одного исходного текста схлопываются, в том числе его очевидные
        варианты написания (`glossary_key`): в контексте нужен сам термин, а не то,
        в скольких главах и какими написаниями он встретился. При переполнении остаются решения
        ранних глав: главные имена и названия появляются в начале книги и повторяются
        дальше, а разовый термин последней главы следующей главе ничего не даёт.
        """
        selected: dict[str, GlossaryCandidate] = {}
        for candidate in candidates:
            if len(selected) >= self._max_accumulated_candidates:
                break
            selected.setdefault(glossary_key(candidate.source_text), candidate)
        return tuple(selected.values())


def chapter_analysis_instructions(*, source_language: str, target_language: str) -> str:
    """Постоянная инструкция анализа главы (PRD § 10.1, § 11, § 14, § 19)."""
    return _instructions(
        source_language=source_language,
        target_language=target_language,
        scope=_WHOLE_CHAPTER_SCOPE,
        summary=_WHOLE_CHAPTER_SUMMARY,
    )


def chapter_fragment_instructions(*, source_language: str, target_language: str) -> str:
    """Инструкция анализа фрагмента длинной главы (PRD § 14, ALG-03).

    От инструкции целой главы отличается ровно тем, что меняет резервный сценарий:
    модель видит часть главы, а её `chapter_summary` описывает события только этой части.
    Остальные правила — план обработки, кандидаты глоссария, инструкции перевода и
    учебный материал — те же самые.
    """
    return _instructions(
        source_language=source_language,
        target_language=target_language,
        scope=_FRAGMENT_SCOPE,
        summary=_FRAGMENT_SUMMARY,
    )


def translation_unit_instructions(*, source_language: str, target_language: str) -> str:
    """Постоянная инструкция обработки группы блоков (PRD § 10.1, § 15, § 19.3)."""
    return _TRANSLATION_INSTRUCTIONS.format(
        source_language=source_language,
        target_language=target_language,
        scope=_UNIT_SCOPE,
    )


def block_regeneration_instructions(
    *, source_language: str, target_language: str
) -> str:
    """Инструкция повторной генерации одного блока (PRD § 13, § 19.3, § 21.5).

    От инструкции группы отличается ровно тем, что меняет повторная генерация: модель
    обрабатывает один блок и видит соседние абзацы, которые заменять не нужно. Правила
    режимов, перевода и заметок остаются те же.
    """
    return _TRANSLATION_INSTRUCTIONS.format(
        source_language=source_language,
        target_language=target_language,
        scope=_REGENERATION_SCOPE,
    )


def _instructions(
    *, source_language: str, target_language: str, scope: str, summary: str
) -> str:
    """Инструкция анализа, собранная под целую главу или под её фрагмент."""
    languages = {"source_language": source_language, "target_language": target_language}
    return _ANALYSIS_INSTRUCTIONS.format(
        scope=scope.format(**languages), summary=summary.format(**languages), **languages
    )


#: Что именно модель видит в запросе: главу целиком или её последовательный фрагмент.
_WHOLE_CHAPTER_SCOPE = """\
Тебе дают одну главу, разбитую на пронумерованные текстовые блоки. Анализируй только эту \
главу: соседних глав ты не видишь и догадок о них не строишь."""

_FRAGMENT_SCOPE = """\
Тебе дают часть одной главы — последовательный фрагмент её текста, разбитый на \
пронумерованные текстовые блоки. Анализируй только этот фрагмент: ни остальной главы, \
ни соседних глав ты не видишь и догадок о них не строишь."""

#: Что описывает `chapter_summary`. У фрагмента — события только его части: сводки
#: фрагментов соединяются в сводку главы в порядке чтения (ALG-03).
_WHOLE_CHAPTER_SUMMARY = """\
5. chapter_summary — что произошло в этой главе.

Пиши на языке оригинала книги ({source_language}), три-шесть предложений. Только события \
самой этой главы: что случилось, с кем, что изменилось. Предыдущие главы не пересказывай, \
о будущих не рассуждай, текст не оценивай. Этот пересказ увидит читатель перед следующей \
главой, чтобы вспомнить, на чём остановился."""

_FRAGMENT_SUMMARY = """\
5. chapter_summary — что произошло в этом фрагменте.

Пиши на языке оригинала книги ({source_language}), два-четыре предложения. Только события \
самого этого фрагмента: что случилось, с кем, что изменилось. Сводки фрагментов главы \
соединяются подряд в один пересказ главы, поэтому не повторяй то, что было до твоего \
фрагмента, и не рассуждай о том, что будет после. Текст не оценивай."""


#: Инструкция анализа главы. Один запрос отвечает сразу за план обработки, кандидатов
#: глоссария, инструкции перевода, `Before You Read` и сводку главы (PRD § 14): отдельные
#: подготовительные запросы к модели не выполняются.
_ANALYSIS_INSTRUCTIONS = """\
Ты готовишь учебное издание художественной книги. Издание помогает читателю, который \
владеет языком {target_language} и учит язык {source_language}, читать эту книгу \
в оригинале.

{scope} Переводить текст сейчас не \
нужно — перевод блоков выполняется отдельными запросами позже.

1. treatment_plan — режим обработки каждого блока.

Верни ровно один элемент на каждый блок, который тебе дали: с тем же идентификатором \
и в том же порядке. Не пропускай блоки, не добавляй чужих, не объединяй и не дели их.

- TRANSLATE — без полного перевода читатель потеряет смысл: сюжетные абзацы, сложные \
диалоги, важные описания, значимые заголовки, содержательные сноски.
- GLOSSARY_ONLY — блок читается в оригинале, но одно-три слова или выражения требуют \
перевода: фразовый глагол, идиома, реалия, редкое или устаревшее слово.
- READ_AS_IS — простая фраза с прозрачной грамматикой, понятная без перевода; перевод \
здесь мешает тренировке чтения сильнее, чем помогает.

Решай по реальной трудности блока для изучающего язык, а не по его длине.

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

4. before_you_read — пять-десять выражений главы, для которых обычного словаря мало: \
идиомы, непрозрачные фразовые глаголы, сленг, разговорные конструкции, устойчивые \
выражения, необычное употребление знакомых слов. Имена, названия и реалии сюда не \
попадают: их место — glossary_candidates, и повторять кандидатов здесь не нужно.

Поле expression — выражение так, как оно стоит в тексте главы; explanation — короткое \
объяснение на языке {target_language}. Это учебный материал, а не пересказ: сюжет здесь \
не объясняют. Меньше пяти выражений — нормально, если больше в главе нет; добирать \
список ради количества не нужно.

{summary}

Отвечай строго по схеме структурированного ответа, без пояснений вне её полей."""


#: Что именно модель обрабатывает: группу идущих подряд блоков или один выбранный блок,
#: которому повторная генерация даёт соседние абзацы как контекст сцены (PRD § 13, § 21.5).
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


#: Инструкция обработки блоков. Один запрос отвечает и за перевод, и за словарные
#: подсказки, и за языковые заметки (PRD § 14, § 19.3): отдельного прохода модели ради
#: заметок не выполняется — модель уже видит оригинал, режим блока и свой результат.
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

Метка вида [^1] в тексте блока — ссылка на сноску книги. Перенеси каждую такую метку \
в translated_text без изменений, в то место перевода, к которому она относится. Цифрой \
или словом её не заменяй, в словарные подсказки и языковые заметки не включай.

2. language_notes — языковые заметки для читателя (PRD: учебный материал).

Заметка объясняет то, что реально помогает читать оригинал или понять переводческое \
решение: идиому, сленг, игру слов, нестандартную грамматику, диалект, культурную \
особенность, необычное употребление.

Заметка отвечает на вопрос «как устроен язык здесь», а не «какое здесь настроение». \
Разбор образа, тона, символики и авторского замысла — это литературоведение, и \
читателю оригинала оно не помогает: такие заметки не пиши. «Сравнение создаёт \
тревожное ощущение» — не заметка; «have no business + Ving — идиома со значением \
„не положено, незачем“» — заметка.

Одно и то же место объясняется читателю один раз. Не повторяй заметкой то, что уже \
сказано словарной подсказкой этого блока или подготовкой к чтению этой главы: если \
подсказка выражение уже перевела, заметка добавляет к ней новое или её не будет.

Каждая заметка привязана ровно к одному блоку из этого запроса: source_block_id — его \
идентификатор, original_excerpt — фрагмент оригинала, translated_excerpt — его перевод, \
если он уместен, note — объяснение на языке {target_language}. Заметки допустимы при \
любом режиме, включая READ_AS_IS. Ориентир — три-восемь заметок на главу, а не на \
каждый блок; если объяснять нечего, верни пустой список.

Служебных сообщений, извинений и рассуждений о самой задаче в ответе быть не должно. \
Отвечай строго по схеме структурированного ответа, без пояснений вне её полей."""


def _chapter_digest(analysis: ChapterAnalysis, glossary: GlossarySnapshot) -> str:
    """Выжимка анализа главы: её сводка и указания переводчику (PRD § 13).

    Кандидаты глоссария сюда не идут: утверждённые решения книги приходят снимком
    запуска, а неутверждённых после старта обработки не существует. Указания написаны
    до проверки глоссария, поэтому пункты, противоречащие решениям пользователя,
    отбрасываются: иначе прежний вариант модели перекрывал бы исправление (PRD § 11).
    """
    lines: list[str] = []
    if summary := analysis.chapter_summary.strip():
        lines.append(f"О чём эта глава:\n{summary}")
    instructions = instructions_respecting_glossary(
        analysis.translation_instructions,
        decisions=glossary.mandatory_entries,
        chapter_candidates=analysis.glossary_candidates,
    )
    if instructions:
        lines.append(f"Указания переводчику этой главы:\n{instructions}")
    return "\n\n".join(lines)


def _explained_expressions(analysis: ChapterAnalysis, section: DocumentSection) -> str:
    """Выражения главы, уже объяснённые в `Before You Read` (PRD § 19.1, § 19.3).

    Материал подготовки к чтению собран на анализе главы, а подсказки и заметки
    рождаются позже и о нём не знают. Без этого списка один и тот же оборот
    объясняется читателю трижды — в `Before You Read`, в словарной подсказке блока
    и в языковой заметке, — а § 19.3 прямо запрещает заметке повторять подсказку.
    """
    expressions = _before_you_read_expressions(analysis, section)
    if not expressions:
        return ""
    lines = [
        "Эти выражения главы читателю уже объяснены до начала чтения. Не объясняй их "
        "заново ни словарной подсказкой, ни языковой заметкой:"
    ]
    lines.extend(f"- {expression}" for expression in expressions)
    return "\n".join(lines)


def _before_you_read_expressions(
    analysis: ChapterAnalysis, section: DocumentSection
) -> tuple[str, ...]:
    """Непустые выражения `Before You Read` главы в порядке анализа.

    Страница сносок `Before You Read` в книге не получает, поэтому и повторять
    её подсказкам и заметкам нечего.
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
    """Обязательные переводческие указания книги: решения пользователя (PRD § 11-12)."""
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
    """Необязательные подсказки глоссария книги (PRD § 11-12)."""
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
    """Пояснение к скобкам списка, если хоть одно решение несёт указание.

    Указание — решение уровня всей книги, а не подсказка для этого запроса: род
    персонажа, названный один раз, обязан держаться во всех главах (PRD § 11).
    """
    if not any(entry.note.strip() for entry in entries):
        return ""
    return (
        "В скобках — переводческое указание по решению. Оно принято для всей книги: "
        "соблюдай его в каждом предложении, где оно применимо."
    )


def _glossary_line(entry: GlossarySnapshotEntry) -> str:
    """Строка записи снимка глоссария вместе с её переводческим указанием."""
    line = f"- {entry.source_text} [{entry.type}] -> {entry.translation}"
    return f"{line} ({entry.note.strip()})" if entry.note.strip() else line


def _neighbour_context(neighbours: Sequence[_NeighbourBlocks]) -> str:
    """Соседний исходный текст блока: контекст сцены, который не обрабатывается (PRD § 13)."""
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
    """Текст блока в том виде, в каком его стоит показывать модели.

    Отличается от `block.text` двумя вещами. Во-первых, у сноски отброшена ссылка
    возврата к месту вызова — «↩», «back» и подобный хвост, которым книга уводит
    читателя обратно в текст. Для книги это навигация, а не текст сноски: экспортёр
    восстанавливает её из разметки блока сам. Модель же видит хвост как часть текста
    и переносит его в перевод, где он становится мёртвым символом — вторым значком
    возврата, который никуда не ведёт. Ссылку возврата опознаём по разметке, а не по
    символам: это ссылка в конце текста сноски, ведущая на якорь внутри книги — в том же
    ресурсе или в другом: у отдельной страницы сносок возврат всегда уходит в главу.

    Во-вторых, номер ссылки на сноску посреди абзаца модели как текст не показывается:
    иначе он переходит в перевод голой цифрой. Блоку, который модель обрабатывает,
    ссылка передаётся меткой `[^N]` (`footnote_markers`): модель переносит её в перевод,
    и экспортёр ставит на её место ссылку оригинала. В остальных случаях — анализ главы,
    соседние абзацы — ссылка просто убирается.
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
    """Начало ссылки возврата в конце сноски либо длина текста, если её нет."""
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


#: Схема внешнего адреса: `http:`, `mailto:` и подобные. Импорт оставляет такие ссылки
#: как есть, а ссылки внутри книги приводит к пути ресурса с якорем.
_EXTERNAL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _points_to_book_anchor(target: str | None) -> bool:
    """Ведёт ли ссылка на якорь внутри книги: в том же ресурсе или в другом.

    Возврат из сноски главы ведёт в тот же файл, а возврат со страницы сносок — в файл
    главы, где стоит ссылка на сноску. Внешний адрес и ссылка на файл без якоря
    возвратом не бывают: это текст сноски.
    """
    if not target or _EXTERNAL_SCHEME.match(target):
        return False
    _, separator, fragment = target.partition("#")
    return bool(separator and fragment)


def _unit_blocks(unit: TranslationUnit) -> str:
    """Блоки группы под псевдонимами вместе с их режимом обработки."""
    lines = ["Блоки для обработки:"]
    lines.extend(
        f"[{_alias(index)}] режим {item.treatment}\n"
        f"{source_text_for_model(item.block, footnote_markers=True)}"
        for index, item in enumerate(unit.items, start=1)
    )
    return "\n\n".join(lines)


def _alias(index: int) -> str:
    """Короткий псевдоним блока в запросе."""
    return f"b{index}"


def _fragment_label(order: int, fragment_count: int) -> str:
    """Строка шапки о месте фрагмента в главе."""
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
    """Шапка запроса: книга, глава, место фрагмента и языки издания."""
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
    """Уже принятые решения книги: кандидаты глоссария предыдущих глав."""
    if not candidates:
        return "Предыдущие главы переводческих решений пока не дали."
    lines = [
        f"Принятые переводческие решения книги ({target_language}). "
        f"Повторяй их дословно:"
    ]
    lines.extend(_known_decision_line(candidate) for candidate in candidates)
    return "\n".join(lines)


def _known_decision_line(candidate: GlossaryCandidate) -> str:
    """Строка принятого решения книги вместе с его переводческим указанием."""
    line = (
        f"- {candidate.source_text} [{candidate.type}] -> "
        f"{candidate.suggested_translation}"
    )
    return f"{line} ({candidate.note.strip()})" if candidate.note.strip() else line


def _chapter_blocks(blocks: Sequence[DocumentBlock]) -> str:
    """Текстовые блоки главы под псевдонимами."""
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
