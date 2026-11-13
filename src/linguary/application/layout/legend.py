"""Legend lines for the layers, labels, and sections a book actually uses.

The lines are the mockup's wording. A chapter dictionary, a self-check,
indexes, and the introduction that names the author are not included.
Nothing is read from a book and nothing is written.
"""

from __future__ import annotations

from dataclasses import dataclass

_ORIGINAL = "Абзац оригинала всегда идёт первым."
_TRANSLATION = (
    "Строка с линией слева — перевод этой части абзаца. "
    "Переведены не все абзацы: остальные вы читаете сами, с подсказками или без них."
)
_NOTES = (
    "Маленькая цифра после слова — номер примечания. "
    "Примечания стоят сразу под абзацем, и в каждом абзаце нумерация начинается с единицы."
)
_IPA = (
    "Транскрипция дана в косых чертах. "
    "UK и US — британское и американское произношение. Ключ к знакам — на следующей странице."
)
_SPEAKERS = (
    "Имя перед переводом реплики — кто говорит. "
    "Если перевода нет, в примечании стоит «Говорит …»."
)
_POEM = "Перевод стихов в книге — подстрочный, без рифмы: он передаёт смысл каждой строки."
_LABEL_ORDER: tuple[tuple[str, str], ...] = (
    ("wordplay", "Игра слов."),
    ("grammar", "Грамматика."),
    ("idiom", "Идиома."),
    ("cultural_reference", "Реалия."),
    ("semantic_relation", "Смысл."),
    ("false_friend", "Ложный друг."),
    ("speech_style", "Речь персонажа."),
    ("parody", "Пародия."),
)
_SECTION_ORDER: tuple[tuple[str, str], ...] = (
    ("previously", "«Previously in…» — что было раньше"),
    ("who_is_who", "«Кто есть кто» — новые персонажи"),
    ("review", "«Повторим» — знакомые слова, которые снова встретятся"),
    ("before_you_read", "«Before You Read» — трудные места главы"),
)


@dataclass(frozen=True, slots=True)
class LayoutFeatures:
    """Layers, note labels, and front-matter sections present in one book.

    `labels` holds Russian note labels (`Игра слов.`) or difficulty kind ids
    (`wordplay`). `vocabulary` has no legend label. `sections` holds
    `previously`, `who_is_who`, `review`, and `before_you_read`.
    """

    has_translation: bool
    has_notes: bool
    labels: frozenset[str]
    has_ipa: bool
    has_poems: bool
    has_speakers: bool
    sections: frozenset[str]


def legend_lines(features: LayoutFeatures) -> tuple[str, ...]:
    """Legend sentences for the layers `features` actually contains.

    Every flag false and both sets empty returns an empty tuple. Otherwise
    the original-paragraph line is first. Each other line is omitted when
    its flag is off or its set has none of the known labels or sections.
    Labels and sections keep a fixed order, not the set's order.
    """
    labels = _labels(features.labels)
    sections = _sections(features.sections)
    if not (
        features.has_translation
        or features.has_notes
        or features.has_ipa
        or features.has_poems
        or features.has_speakers
        or features.labels
        or features.sections
    ):
        return ()
    lines = [_ORIGINAL]
    if features.has_translation:
        lines.append(_TRANSLATION)
    if features.has_notes:
        lines.append(_NOTES)
    if labels:
        lines.append(
            "Слово в начале примечания называет его вид: "
            f"{', '.join(labels)} Примечание без такого слова — просто значение."
        )
    if features.has_ipa:
        lines.append(_IPA)
    if features.has_speakers:
        lines.append(_SPEAKERS)
    if features.has_poems:
        lines.append(_POEM)
    if sections:
        lines.append("Перед главой: " + "; ".join(sections))
    return tuple(lines)


def _labels(selected: frozenset[str]) -> tuple[str, ...]:
    """Known labels in legend order. A kind id and its text count once."""
    return tuple(text for kind, text in _LABEL_ORDER if kind in selected or text in selected)


def _sections(selected: frozenset[str]) -> tuple[str, ...]:
    """Known section phrases in reading order."""
    return tuple(phrase for key, phrase in _SECTION_ORDER if key in selected)
