"""Unit tests for the built-in CEFR A1-A2 core vocabulary filter."""

from __future__ import annotations

import pytest

from linguary.application.a2_core_vocab import (
    A2_CORE_VOCABULARY,
    _NON_CORE_OVERRIDE,
    is_core_a2_word,
    non_core_a2_words,
)


REQUIRED_NON_CORE_WORDS = frozenset({
    "actually",
    "hardly",
    "barely",
    "merely",
    "scarcely",
    "likely",
    "unlikely",
    "mostly",
    "roughly",
    "survivor",
    "survivors",
    "bump",
    "bumped",
    "rip",
    "rips",
    "scratch",
    "scratches",
    "whoa",
    "wink",
    "winked",
    "groan",
    "groaned",
    "chuckle",
    "chuckled",
    "peer",
    "peered",
    "nod",
    "nodded",
    "snigger",
    "sniggered",
    "moan",
    "moaned",
    "uneasy",
    "nervous",
    "demand",
    "demanded",
    "murmur",
    "murmured",
    "snap",
    "snapped",
    "mutter",
    "muttered",
    "whisper",
    "whispered",
    "gasp",
    "gasped",
    "stammer",
    "stammered",
    "budge",
    "cannon",
    "sweetums",
    "sweetheart",
    "suppose",
    "freckle",
    "freckled",
    "wizard",
    "wizards",
    "afterlight",
    "afterlights",
    "skinjacker",
    "windshield",
    "collision",
    "highway",
    "eternity",
    "frown",
    "frowned",
    "shrug",
    "shrugged",
    "glare",
    "glared",
    "stare",
    "stared",
    "tremble",
    "trembled",
    "shiver",
    "shivered",
    "gulp",
    "gulped",
    "blink",
    "blinked",
    "sigh",
    "sighed",
    "grin",
    "grinned",
    "scowl",
    "scowled",
    "sneer",
    "sneered",
    "yell",
    "yelled",
    "scream",
    "screamed",
    "sob",
    "sobbed",
})


def test_vocabulary_size_and_normalization() -> None:
    assert isinstance(A2_CORE_VOCABULARY, frozenset)
    assert 2_500 <= len(A2_CORE_VOCABULARY) <= 3_000
    for word in A2_CORE_VOCABULARY:
        assert word
        assert word == word.lower()
        assert word == word.strip()
        assert "\u2019" not in word
        assert "\u2018" not in word


def test_non_core_override_is_complete_and_disjoint_from_core_vocabulary() -> None:
    assert isinstance(_NON_CORE_OVERRIDE, frozenset)
    assert REQUIRED_NON_CORE_WORDS <= _NON_CORE_OVERRIDE
    overlap = A2_CORE_VOCABULARY & _NON_CORE_OVERRIDE
    assert not overlap, f"non-core override words leaked into A2_CORE_VOCABULARY: {sorted(overlap)}"


def test_every_non_core_override_word_is_rejected() -> None:
    for word in _NON_CORE_OVERRIDE:
        assert not is_core_a2_word(word), f"override word should not be core: {word}"
        assert not is_core_a2_word(word.capitalize()), f"capitalized override should not be core: {word}"
        assert not is_core_a2_word(f"{word}'s"), f"possessive override should not be core: {word}'s"


@pytest.mark.parametrize(
    "word",
    [
        # Function words, pronouns, contractions
        "the",
        "because",
        "through",
        "themselves",
        "everything",
        "don't",
        "don\u2019t",
        "I'm",
        "I\u2019m",
        "won't",
        "can't",
        "let's",
        "o'clock",
        # Numbers and irregular verb forms
        "twenty",
        "hundred",
        "second",
        "went",
        "gone",
        "bought",
        "caught",
        "taught",
        "thought",
        "written",
        "spoken",
        "broken",
        "eaten",
        "slept",
        "understood",
        "forgotten",
        "began",
        "begun",
        # Explicitly required A1-A2 words, interjections, and internationalisms
        "shoo",
        "sharp",
        "nearly",
        "major",
        "telescope",
        "problem",
        "idea",
        "radio",
        "photo",
        "hotel",
        "park",
        "bank",
        "bus",
        "taxi",
        "police",
        "chocolate",
        "coffee",
        "planet",
        "robot",
        "rocket",
        "magic",
        "energy",
        "system",
        "moment",
        "secret",
        "captain",
        "professor",
        "president",
        "director",
        "doctor",
        "normal",
        "special",
        "popular",
        "serious",
    ],
)
def test_core_a2_direct_words(word: str) -> None:
    assert is_core_a2_word(word)


@pytest.mark.parametrize(
    "word",
    [
        # Possessives
        "cat's",
        "teacher's",
        "friend\u2019s",
        "world's",
        # Plurals / 3rd person singular
        "books",
        "boxes",
        "watches",
        "dishes",
        "classes",
        "potatoes",
        "cities",
        "stories",
        "tries",
        "dies",
        "knives",
        "leaves",
        # Past tense / participles
        "walked",
        "liked",
        "stopped",
        "planned",
        "travelled",
        "traveled",
        "carried",
        "studied",
        "tried",
        "died",
        "cried",
        # Present participles / gerunds
        "walking",
        "making",
        "using",
        "running",
        "sitting",
        "stopping",
        "swimming",
        "dying",
        "lying",
        "tying",
        "carrying",
        # Comparatives, superlatives, agentives
        "faster",
        "fastest",
        "bigger",
        "biggest",
        "happier",
        "happiest",
        "nicer",
        "nicest",
        "sharper",
        "sharpest",
        "teacher",
        "player",
        "runner",
        "driver",
        # Adverbs in -ly
        "loudly",
        "sharply",
        "quickly",
        "slowly",
        "quietly",
        "happily",
        "easily",
        "carefully",
        "simply",
        "possibly",
        "basically",
        "magically",
        "fully",
    ],
)
def test_core_a2_morphological_inflections(word: str) -> None:
    assert is_core_a2_word(word)


@pytest.mark.parametrize(
    "word",
    [
        # Override stems and their other inflections
        "actually",
        "hardly",
        "barely",
        "merely",
        "scarcely",
        "likely",
        "unlikely",
        "mostly",
        "roughly",
        "survivor",
        "survivors",
        "bumping",
        "ripped",
        "scratching",
        "winking",
        "whispering",
        "shrugging",
        "staring",
        "trembling",
        "supposed",
        "freckles",
        "wizards",
        # Morphological false-positive traps
        "shed",
        "wither",
        "solder",
        "digest",
        "earnest",
        "ponder",
        "cower",
        "fares",
        "canes",
        # Dialect / non-standard / empty
        "shan't",
        "dunno",
        "codswallop",
        "",
        "   ",
    ],
)
def test_non_core_words_rejected(word: str) -> None:
    assert not is_core_a2_word(word)


@pytest.mark.parametrize(
    ("text", "known_names"),
    [
        ("Nearly.", ()),
        ("\u201cShoo!\u201d said Mr Dursley loudly.", ("Nick", "Mr Dursley")),
        ("\u201cShoo!\u201d said Mr Dursley loudly.", ()),
        ("\u201cShoo!\u201d said Mr. Dursley loudly.", ()),
        ("\u201cNo,\u201d she said sharply. \u201cWhy?\u201d", ()),
        ("1 telescope", ()),
        ("Come here, Nick.", ("Nick", "Mr Dursley")),
        ("Come here, Nick.", ()),
        ("Nick is my friend.", ("Nick",)),
        ("Nick\u2019s coat is warm.", ("Nick",)),
        ("\u201cNo,\u201d Nick said.", ()),
        ('"No," Nick said.', ()),
        ("Part IV, Chapter XII: 2nd page, item b, 100 years.", ()),
        ("He is a well-known twenty-one-year-old teacher.", ()),
        ("She lives in Hogwarts near London.", ()),
    ],
)
def test_non_core_a2_words_allows_a1_a2_and_hp_exceptions(
    text: str, known_names: tuple[str, ...]
) -> None:
    assert non_core_a2_words(text, known_names=known_names) == ()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("We could be the only survivors?", ("survivors",)),
        ("I bumped into you.", ("bumped",)),
        ("No rips, no scratches.", ("rips", "scratches")),
        ("Major whoa.", ("whoa",)),
        ("Actually, I hardly knew him.", ("actually", "hardly")),
        ("She barely looked up and winked.", ("barely", "winked")),
        ("The wizard felt uneasy.", ("wizard", "uneasy")),
        ('He said, "Whoa!"', ("whoa",)),
        ("Stop! Whoa!", ("whoa",)),
        ("A half-wizard stood near the ice-cold window.", ("half-wizard",)),
        ("Whoa! I bumped into a wizard, whoa, and bumped again.", ("whoa", "bumped", "wizard")),
    ],
)
def test_non_core_a2_words_catches_everlost_leaks_and_overrides(
    text: str, expected: tuple[str, ...]
) -> None:
    assert non_core_a2_words(text) == expected


def test_non_core_a2_words_hyphenated_with_known_names() -> None:
    assert non_core_a2_words(
        "A freckle-faced boy looked at Mary-Jane.",
        known_names=("Mary-Jane",),
    ) == ("freckle-faced",)
