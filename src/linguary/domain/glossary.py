"""Глоссарий книги: понятия, языковые варианты и кандидаты из анализа глав.

Глоссарий — единственный источник обязательных переводческих решений (PRD § 11).
Кандидат приходит из анализа главы с собственной пометкой модели `model_flagged_review`,
а итоговый признак `requires_review` вычисляет приложение по правилам PRD § 12.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from linguary.domain.errors import GlossaryError


class ConceptType(StrEnum):
    """Тип понятия глоссария (PRD § 12)."""

    PERSON = "person"
    PLACE = "place"
    ORGANIZATION = "organization"
    FICTIONAL_CONCEPT = "fictional_concept"
    TERM = "term"

    @property
    def always_requires_review(self) -> bool:
        """Типы, решение по которым пользователь принимает всегда."""
        return self is not ConceptType.TERM


class TermStatus(StrEnum):
    """Статус переводческого решения по понятию и его варианту (PRD § 11)."""

    SUGGESTED = "suggested"
    APPROVED = "approved"
    LOCKED = "locked"

    @property
    def is_mandatory(self) -> bool:
        """`locked` передаётся модели как обязательное переводческое указание."""
        return self is TermStatus.LOCKED


@dataclass(frozen=True, slots=True)
class TermVariant:
    """Языковой вариант одного понятия."""

    concept_id: str
    language: str
    text: str
    status: TermStatus = TermStatus.SUGGESTED

    def __post_init__(self) -> None:
        if not self.concept_id.strip():
            raise GlossaryError("вариант термина не привязан к понятию")
        if not self.language.strip():
            raise GlossaryError("язык варианта термина не задан")
        if not self.text.strip():
            raise GlossaryError("текст варианта термина пуст")


@dataclass(frozen=True, slots=True)
class TermConcept:
    """Понятие глоссария, объединяющее варианты одного смысла.

    `note` — переводческое указание по этому понятию, которое передаётся модели вместе
    с вариантом перевода: одного варианта мало там, где решение уровня книги влияет
    на согласование в тексте (PRD § 11).
    """

    id: str
    book_id: str
    type: ConceptType
    canonical_source_text: str
    status: TermStatus = TermStatus.SUGGESTED
    variants: tuple[TermVariant, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise GlossaryError("идентификатор понятия глоссария пуст")
        if not self.book_id.strip():
            raise GlossaryError("понятие глоссария не привязано к книге")
        if not self.canonical_source_text.strip():
            raise GlossaryError("исходный текст понятия глоссария пуст")
        for variant in self.variants:
            if variant.concept_id != self.id:
                raise GlossaryError(
                    f"вариант {variant.text!r} принадлежит другому понятию: {variant.concept_id}"
                )
        if self.status.is_mandatory and not any(
            variant.status.is_mandatory for variant in self.variants
        ):
            raise GlossaryError(
                f"понятие {self.id} зафиксировано, но обязательного варианта перевода нет"
            )

    def variant_for(self, language: str) -> TermVariant | None:
        """Вариант перевода на указанный язык с наиболее сильным статусом."""
        candidates = [variant for variant in self.variants if variant.language == language]
        if not candidates:
            return None
        priority = {TermStatus.LOCKED: 0, TermStatus.APPROVED: 1, TermStatus.SUGGESTED: 2}
        return min(candidates, key=lambda variant: priority[variant.status])


@dataclass(frozen=True, slots=True)
class GlossaryCandidate:
    """Кандидат глоссария из анализа главы.

    `model_flagged_review` — собственная пометка модели, входной сигнал. Итоговое
    `requires_review` вычисляет приложение и после сохранения модель его не редактирует.

    `note` — переводческое указание по этому решению, которое нельзя выразить одним
    вариантом перевода: грамматический род персонажа, склонение названия, регистр
    речи. Без него решение уровня книги распадается на решения отдельных глав
    (PRD § 11): имя приходит из глоссария одинаковым, а согласуемые с ним слова
    расходятся от главы к главе.
    """

    id: str
    analysis_id: str
    source_text: str
    type: ConceptType
    suggested_translation: str
    confidence: float
    model_flagged_review: bool
    requires_review: bool
    note: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise GlossaryError("идентификатор кандидата глоссария пуст")
        if not self.analysis_id.strip():
            raise GlossaryError("кандидат глоссария не привязан к анализу главы")
        if not self.source_text.strip():
            raise GlossaryError("исходный текст кандидата глоссария пуст")
        if not self.suggested_translation.strip():
            raise GlossaryError(f"предложенный перевод кандидата {self.source_text!r} пуст")
        if not 0.0 <= self.confidence <= 1.0:
            raise GlossaryError(
                f"уверенность модели вне диапазона 0..1: {self.confidence}"
            )

    @classmethod
    def create(
        cls,
        *,
        id: str,
        analysis_id: str,
        source_text: str,
        type: ConceptType,
        suggested_translation: str,
        confidence: float,
        model_flagged_review: bool,
        confidence_threshold: float,
        has_translation_conflict: bool = False,
        note: str = "",
    ) -> GlossaryCandidate:
        """Создаёт кандидата с вычисленным признаком `requires_review`."""
        return cls(
            id=id,
            analysis_id=analysis_id,
            source_text=source_text,
            type=type,
            suggested_translation=suggested_translation,
            confidence=confidence,
            model_flagged_review=model_flagged_review,
            note=note,
            requires_review=compute_requires_review(
                type=type,
                confidence=confidence,
                model_flagged_review=model_flagged_review,
                confidence_threshold=confidence_threshold,
                has_translation_conflict=has_translation_conflict,
            ),
        )


def compute_requires_review(
    *,
    type: ConceptType,
    confidence: float,
    model_flagged_review: bool,
    confidence_threshold: float,
    has_translation_conflict: bool = False,
) -> bool:
    """Требует ли кандидат внимания пользователя (PRD § 12).

    Достаточно одного условия: тип понятия — имя, место, организация или вымышленное
    понятие; модель пометила кандидата; уверенность ниже настраиваемого порога;
    при слиянии фрагментов анализа для одного исходного текста получены разные варианты.
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        raise GlossaryError(
            f"порог уверенности вне диапазона 0..1: {confidence_threshold}"
        )
    return (
        type.always_requires_review
        or model_flagged_review
        or confidence < confidence_threshold
        or has_translation_conflict
    )


#: Типографские апострофы и кавычки, которые в ключе становятся прямыми.
_STRAIGHT_QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u02bc": "'",
        "\u2032": "'",
        "`": "'",
        "\u00b4": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2033": '"',
    }
)

#: Точка после сокращённого обращения: `Mr. Wonka` и `Mr Wonka` — одно имя.
_TITLE_DOT = re.compile(r"\b(mr|mrs|ms|dr|st)\.\s*")

_WHITESPACE = re.compile(r"\s+")


def glossary_key(text: str) -> str:
    """Ключ, по которому написания одного понятия собираются в один элемент (PRD § 11).

    Объединяются только очевидные варианты: регистр, совместимые формы Unicode,
    типографские апострофы и кавычки, лишние пробелы и точка после `Mr`, `Mrs`, `Ms`,
    `Dr`, `St`. Притяжательная форма и части имени ключ не меняют: `Wonka's factory`
    и `Wonka factory`, `Charlie` и `Charlie Bucket` — разные элементы, потому что
    неоднозначное автоматически не объединяется.
    """
    normalized = unicodedata.normalize("NFKC", text).translate(_STRAIGHT_QUOTES)
    normalized = _WHITESPACE.sub(" ", normalized).strip().casefold()
    return _WHITESPACE.sub(" ", _TITLE_DOT.sub(r"\1 ", normalized)).strip()


def spellings_by_preference(mentions: Iterable[tuple[str, float]]) -> tuple[str, ...]:
    """Различные написания одного понятия, начиная с того, которое показывается.

    `mentions` — пары «написание, уверенность модели» по каждому упоминанию. Первым
    идёт самое частое написание, при равенстве — самое уверенное, затем — встреченное
    раньше: так на экране проверки стоит та форма, которой книга пользуется чаще всего.
    """
    counts: dict[str, int] = {}
    confidence: dict[str, float] = {}
    for text, mention_confidence in mentions:
        spelling = text.strip()
        counts[spelling] = counts.get(spelling, 0) + 1
        confidence[spelling] = max(confidence.get(spelling, 0.0), mention_confidence)
    order = {spelling: index for index, spelling in enumerate(counts)}
    return tuple(
        sorted(
            counts,
            key=lambda spelling: (-counts[spelling], -confidence[spelling], order[spelling]),
        )
    )


def translations_conflict(translations: Iterable[str]) -> bool:
    """Дают ли фрагменты анализа разные варианты перевода одного понятия.

    Варианты сравниваются тем же ключом, что и написания исходного текста: регистр
    и типографика перевода решением не являются.
    """
    return len({glossary_key(translation) for translation in translations}) > 1
