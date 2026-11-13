"""Legend lines and the IPA key for the signs a book actually uses.

The legend is the mockup wording for layers that are present. It has no
chapter dictionary, self-check, indexes, or the introduction that names
Lewis Carroll. The IPA key is the static reference, filtered to signs that
occur.
"""

from __future__ import annotations

from linguary.application.layout.ipa_reference import IPA_REFERENCE, IpaSymbol, ipa_key
from linguary.application.layout.legend import LayoutFeatures, legend_lines

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
_SPEAKERS = "Имя перед переводом реплики — кто говорит. Если перевода нет, в примечании стоит «Говорит …»."
_POEM = "Перевод стихов в книге — подстрочный, без рифмы: он передаёт смысл каждой строки."
_LABELS = (
    "Игра слов.",
    "Грамматика.",
    "Идиома.",
    "Реалия.",
    "Смысл.",
    "Ложный друг.",
    "Речь персонажа.",
    "Пародия.",
)
_SECTIONS = (
    ("previously", "«Previously in…» — что было раньше"),
    ("who_is_who", "«Кто есть кто» — новые персонажи"),
    ("review", "«Повторим» — знакомые слова, которые снова встретятся"),
    ("before_you_read", "«Before You Read» — трудные места главы"),
)
_MOCKUP_SIGNS = (
    ("ˈ", "/ˈweɪskəʊt/", "ударение на следующем слоге"),
    ("ː", "see /siː/", "звук долгий"),
    ("ɪ", "sit /sɪt/", "краткий, между «и» и «ы»"),
    ("æ", "cat /kæt/", "между «э» и «а», рот открыт шире"),
    ("ʌ", "cup /kʌp/", "краткое «а»"),
    ("ɜː", "bird /bɜːd/", "похоже на «ё» без «й», губы не округлять"),
    ("ə", "about /əˈbaʊt/", "безударный неясный звук"),
    ("eɪ", "tale /teɪl/", "«эй»"),
    ("əʊ", "go /ɡəʊ/ (UK)", "от неясного звука к «у»; в US — /oʊ/"),
    ("θ", "think /θɪŋk/", "глухой: кончик языка между зубами"),
    ("ð", "this /ðɪs/", "то же, но звонкий"),
    ("ŋ", "sing /sɪŋ/", "«н» задней частью языка, без «г»"),
    ("w", "well /wel/", "губы трубочкой, краткое «у»"),
    ("tʃ", "Cheshire /ˈtʃeʃə/", "твёрдое «ч»"),
)


def _features(
    *,
    has_translation: bool = False,
    has_notes: bool = False,
    labels: frozenset[str] = frozenset(),
    has_ipa: bool = False,
    has_poems: bool = False,
    has_speakers: bool = False,
    sections: frozenset[str] = frozenset(),
) -> LayoutFeatures:
    return LayoutFeatures(
        has_translation,
        has_notes,
        labels,
        has_ipa,
        has_poems,
        has_speakers,
        sections,
    )


def _symbols(ipas: list[str]) -> tuple[str, ...]:
    return tuple(item.symbol for item in ipa_key(ipas))


def _label_line(*labels: str) -> str:
    listed = ", ".join(labels)
    return (
        "Слово в начале примечания называет его вид: "
        f"{listed} Примечание без такого слова — просто значение."
    )


def _section_line(*keys: str) -> str:
    text = {key: phrase for key, phrase in _SECTIONS}
    return "Перед главой: " + "; ".join(text[key] for key in keys)


def test_ipa_key_lists_only_used_symbols() -> None:
    """Used signs keep the mockup example and description, in reference order."""
    by_symbol = {item.symbol: item for item in IPA_REFERENCE}
    assert len(by_symbol) == len(IPA_REFERENCE)
    for symbol, example, description in _MOCKUP_SIGNS:
        assert by_symbol[symbol] == IpaSymbol(symbol, example, description)
    order = tuple(item.symbol for item in IPA_REFERENCE)
    for longer, shorter in (("tʃ", "t"), ("tʃ", "ʃ"), ("eɪ", "e"), ("eɪ", "ɪ"),
                            ("ɜː", "ɜ"), ("ɜː", "ː"), ("əʊ", "ə"), ("əʊ", "ʊ")):
        assert order.index(longer) < order.index(shorter)

    used = _symbols(["/ˈweɪskəʊt/", "/kæt/", "/ˈweɪskəʊt/"])
    assert used == tuple(symbol for symbol in order if symbol in {"ˈ", "w", "eɪ", "əʊ", "æ", "t"})
    assert "θ" not in used
    assert "ɪ" not in used
    assert "ə" not in used
    assert "ʊ" not in used
    assert ipa_key([]) == ()
    assert ipa_key(["", "/"]) == ()


def test_multichar_symbols_matched_greedily() -> None:
    """The longest sign wins. Its parts are listed only when they occur alone."""
    order = tuple(item.symbol for item in IPA_REFERENCE)
    cheshire = _symbols(["/ˈtʃeʃə/"])
    assert "tʃ" in cheshire
    assert "t" not in cheshire
    assert "ʃ" in cheshire
    assert "e" in cheshire
    assert cheshire == tuple(symbol for symbol in order if symbol in set(cheshire))

    assert _symbols(["/teɪl/"]) == tuple(symbol for symbol in order if symbol in {"t", "eɪ"})
    assert "e" not in _symbols(["/teɪl/"])
    assert "ɪ" not in _symbols(["/teɪl/"])
    assert _symbols(["/bɜːd/"]) == ("ɜː",)
    assert _symbols(["/ɡəʊ/"]) == ("əʊ",)
    assert "ʊ" not in _symbols(["/əˈbaʊt/"])
    assert "aʊ" in _symbols(["/əˈbaʊt/"])
    assert _symbols(["/maɪ/"]) == ("aɪ",)
    assert _symbols(["/oʊ/"]) == ("oʊ",)
    assert "ɪ" not in _symbols(["/nɪə/"])
    assert "ə" not in _symbols(["/nɪə/"])

    both = _symbols(["/teɪl/", "/sɪt/", "/siː/"])
    assert "eɪ" in both and "ɪ" in both and "ː" in both and "t" in both
    assert both.index("eɪ") < both.index("ɪ")
    assert both.index("eɪ") < both.index("ː")
    combined = _symbols(["/bɜːd siː/"])
    assert combined.index("ɜː") < combined.index("ː")
    assert _symbols(["xɪ"]) == ("ɪ",)


def test_legend_lists_only_present_labels() -> None:
    """Kind names that occur are listed in label order. Absent layers are omitted."""
    assert legend_lines(_features()) == ()
    lines = legend_lines(_features(
        has_notes=True,
        labels=frozenset({"Пародия.", "Игра слов.", "vocabulary", "не метка"}),
    ))
    assert lines == (
        _ORIGINAL,
        _NOTES,
        _label_line("Игра слов.", "Пародия."),
    )
    by_kind = legend_lines(_features(
        labels=frozenset({"false_friend", "grammar", "wordplay"}),
    ))
    assert by_kind == (
        _ORIGINAL,
        _label_line("Игра слов.", "Грамматика.", "Ложный друг."),
    )
    assert _TRANSLATION not in lines
    assert _IPA not in lines
    assert _SPEAKERS not in lines
    assert _POEM not in lines


def test_legend_omits_dictionary_self_check_and_indexes() -> None:
    """A full legend still has no dictionary, self-check, index, or author introduction."""
    lines = legend_lines(_features(
        has_translation=True,
        has_notes=True,
        labels=frozenset(_LABELS),
        has_ipa=True,
        has_poems=True,
        has_speakers=True,
        sections=frozenset(key for key, _phrase in _SECTIONS),
    ))
    assert lines == (
        _ORIGINAL,
        _TRANSLATION,
        _NOTES,
        _label_line(*_LABELS),
        _IPA,
        _SPEAKERS,
        _POEM,
        _section_line("previously", "who_is_who", "review", "before_you_read"),
    )
    text = "\n".join(lines)
    for phrase in (
        "словарь",
        "самопроверк",
        "указател",
        "Кэрролл",
        "Льюис",
        "После главы",
        "В конце книги",
        "листать",
    ):
        assert phrase not in text
    partial = legend_lines(_features(
        sections=frozenset({"before_you_read", "who_is_who", "index", "dictionary"}),
    ))
    assert partial == (
        _ORIGINAL,
        _section_line("who_is_who", "before_you_read"),
    )


def test_poem_line_only_with_poems() -> None:
    """The subscript line follows the poem flag, not the parody label."""
    without = legend_lines(_features(
        has_translation=True,
        has_speakers=True,
        has_poems=False,
        labels=frozenset({"Пародия."}),
        sections=frozenset({"review"}),
    ))
    assert _POEM not in without
    assert _SPEAKERS in without
    assert _label_line("Пародия.") in without

    with_poems = legend_lines(_features(
        has_translation=True,
        has_speakers=True,
        has_poems=True,
        sections=frozenset({"previously", "review"}),
    ))
    assert with_poems.index(_SPEAKERS) < with_poems.index(_POEM)
    assert with_poems.index(_POEM) < with_poems.index(
        _section_line("previously", "review")
    )
    assert legend_lines(_features(has_poems=True)) == (_ORIGINAL, _POEM)
