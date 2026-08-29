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

Блоки нумеруются в запросе короткими псевдонимами `b1..bN`, а не своими стабильными
идентификаторами. Идентификатор блока — UUID: в главе на две сотни абзацев такие ключи
занимают заметную часть и запроса, и ответа, а модель обязана вернуть каждый из них
дословно. Псевдоним стоит несколько символов, переносится точнее и живёт ровно один
запрос — обратное отображение хранится здесь же, в `ChapterAnalysisContext`.

Инструкции написаны по-русски: тем же языком написана уточняющая инструкция повтора
(`retry_policy.build_repair_instruction`), которая дописывается к этому же тексту.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from linguary.application.analysis_merge import ChapterFragment
from linguary.application.errors import ApplicationError
from linguary.application.llm_dto import ChapterAnalysisResponse
from linguary.application.ports.llm import LLMRequest
from linguary.application.retry_policy import ExpectedBlock, ResultExpectation
from linguary.domain import (
    BlockType,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    GlossaryCandidate,
    SupportStatus,
)

#: Сколько уже принятых решений книги попадает в контекст анализа очередной главы.
#: К концу романа кандидатов накапливаются сотни, а запросу нужны те, что действительно
#: повторяются: список ограничивается сверху, чтобы контекст не рос вместе с книгой
#: (PRD § 13: ограничение контекста настраивается).
DEFAULT_MAX_ACCUMULATED_CANDIDATES = 200

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


@dataclass(frozen=True, slots=True)
class ChapterAnalysisContext:
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

    @property
    def aliases(self) -> tuple[str, ...]:
        """Псевдонимы блоков главы в порядке исходного документа."""
        return tuple(self.block_ids_by_alias)

    def block_id_for(self, alias: str) -> str:
        """Стабильный идентификатор блока по псевдониму из ответа модели."""
        block_id = self.block_ids_by_alias.get(alias)
        if block_id is None:
            raise ContextBuildError(f"блок {alias} в запрос главы не передавался")
        return block_id


class ContextBuilder:
    """Собирает контекст запросов к модели по политике PRD § 13."""

    def __init__(
        self, *, max_accumulated_candidates: int = DEFAULT_MAX_ACCUMULATED_CANDIDATES
    ) -> None:
        if max_accumulated_candidates < 0:
            raise ContextBuildError(
                "ограничение числа накопленных кандидатов не может быть отрицательным"
            )
        self._max_accumulated_candidates = max_accumulated_candidates

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

        Повторы одного исходного текста схлопываются: в контексте нужен сам термин,
        а не то, в скольких главах он встретился. При переполнении остаются решения
        ранних глав: главные имена и названия появляются в начале книги и повторяются
        дальше, а разовый термин последней главы следующей главе ничего не даёт.
        """
        selected: dict[str, GlossaryCandidate] = {}
        for candidate in candidates:
            if len(selected) >= self._max_accumulated_candidates:
                break
            selected.setdefault(candidate.source_text, candidate)
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

Решай по реальной трудности блока для изучающего язык, а не по его длине. Поле reason — \
одна короткая фраза на языке {target_language} о том, почему выбран этот режим.

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
действующее лицо повествования, а народы, племена, сословия и исторические имена из \
справок и отступлений — term;
- suggested_translation — ровно один вариант на языке {target_language}, без пояснений \
в скобках и без вторых вариантов через слэш: этот текст подставляется в перевод как есть;
- confidence — твоя уверенность в этом варианте от 0 до 1;
- model_flagged_review — true, если решение спорное, зависит от дальнейших глав или \
требует человека.

Если термин уже есть в принятых решениях книги, повтори принятый перевод и нового \
варианта не предлагай.

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
    lines.extend(
        f"- {candidate.source_text} [{candidate.type}] -> "
        f"{candidate.suggested_translation}"
        for candidate in candidates
    )
    return "\n".join(lines)


def _chapter_blocks(blocks: Sequence[DocumentBlock]) -> str:
    """Текстовые блоки главы под псевдонимами."""
    lines = ["Текстовые блоки главы:"]
    lines.extend(
        f"[{_alias(index)}]\n{block.text.strip()}"
        for index, block in enumerate(blocks, start=1)
    )
    return "\n\n".join(lines)


__all__ = [
    "DEFAULT_MAX_ACCUMULATED_CANDIDATES",
    "ChapterAnalysisContext",
    "ContextBuildError",
    "ContextBuilder",
    "chapter_analysis_instructions",
    "chapter_fragment_instructions",
    "is_textual",
    "textual_blocks",
]
