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
    "presently",
    "deadly",
    "shortly",
    "willing",
})


def test_vocabulary_size_and_normalization() -> None:
    assert isinstance(A2_CORE_VOCABULARY, frozenset)
    assert 2_500 <= len(A2_CORE_VOCABULARY) <= 3_100
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
        "given",
        "sung",
        "rode",
        "sunk",
        "bored",
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
        "road",
        "mom",
        "mommy",
        "mum",
        "mummy",
        "granny",
    ],
)
def test_core_a2_direct_words(word: str) -> None:
    assert is_core_a2_word(word)


def test_non_core_a2_words_skips_road_and_title_case_proper_names() -> None:
    assert non_core_a2_words("The road looked quiet.") == ()
    assert non_core_a2_words("Boa Constrictor, Brazil.", known_names=("Boa Constrictor",)) == ()
    assert non_core_a2_words("Mary Hightower wrote a book.", known_names=("Mary Hightower",)) == ()
    assert non_core_a2_words("Major whoa.") == ("whoa",)
    assert non_core_a2_words("Major Whoa!") == ("whoa",)
    assert non_core_a2_words("Wretched Muggle!") == ("wretched", "muggle")
    assert non_core_a2_words("Good Grief!") == ("grief",)
    assert non_core_a2_words("STOP, YOU WRETCHED COWARD!") == ("wretched", "coward")


@pytest.mark.parametrize(
    "word",
    [
        "literally",
        "gradually",
        "meanwhile",
        "therefore",
        "otherwise",
        "eventually",
        "seldom",
        "somewhat",
        "independence",
        "confidence",
        "conclusion",
        "destination",
        "disability",
        "elementary",
        "enormous",
        "exception",
        "finance",
        "fortunate",
        "gravity",
        "management",
        "relief",
        "threat",
        "typical",
        "variety",
        "deserve",
        "punish",
        "recommend",
        "recover",
        "suffer",
        "surround",
        "presently",
        "deadly",
        "shortly",
        "willing",
    ],
)
def test_removed_b1_b2_and_new_override_words_are_not_core(word: str) -> None:
    assert word not in A2_CORE_VOCABULARY
    assert not is_core_a2_word(word)


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


def test_spoken_hesitation_interjections_are_core_a2() -> None:
    assert non_core_a2_words("\u2018Er \u2013 no,\u2019 said Harry.", known_names=("Harry",)) == ()
    assert non_core_a2_words("Erm, phew, huh, mhm!", known_names=()) == ()
    assert non_core_a2_words("Whoa!", known_names=()) == ("whoa",)


def test_title_case_minor_words_and_author_attribution_boundaries() -> None:
    from linguary.application.a2_core_vocab import non_core_a2_word_spans

    assert non_core_a2_words(
        "A Beginner\u2019s Guide to Transfiguration by Emeric Switch",
        known_names=(),
    ) == ("transfiguration",)
    assert non_core_a2_words(
        "Magical Drafts and Potions by Arsenius Jigger",
        known_names=(),
    ) == ("drafts", "potions")
    assert non_core_a2_words(
        "The Standard Book of Incantations (Grade 1) by Miranda Goshawk",
        known_names=(),
    ) == ("incantations",)
    assert non_core_a2_words(
        "The Dark Forces: A Guide to Self-Protection",
        known_names=(),
    ) == ("self-protection",)
    # Negative controls: pairs of names, comma-separated names, titled person phrases.
    assert non_core_a2_words("Harry and Ron", known_names=("Harry",)) == ()
    assert non_core_a2_words("Harry, Ron and Hermione", known_names=("Harry",)) == ()
    assert non_core_a2_words("Mr. Dursley of Privet Drive", known_names=()) == ()
    # Exact span extraction skips earlier proper-name occurrence of the same word.
    text = "He met Cannon. Wait, cannon!"
    spans = non_core_a2_word_spans(text, known_names=())
    assert len(spans) == 1
    word, start, end = spans[0]
    assert word == "cannon"
    assert text[start:end] == "cannon"


def test_lowercase_article_a_does_not_bypass_title_case_guard() -> None:
    assert non_core_a2_words("Look, a Severus Snape", known_names=()) == ()
    assert non_core_a2_words("Wait, a Harry Dursley", known_names=()) == ()


def test_onomatopoeia_screams_and_added_a2_tokens() -> None:
    from linguary.application.a2_core_vocab import non_core_a2_word_spans

    assert is_core_a2_word("hamburger") is True
    assert is_core_a2_word("goblin") is True
    assert is_core_a2_word("mmm") is True
    assert is_core_a2_word("ouch") is True

    assert non_core_a2_word_spans("BOOM. SMASH! WHAM! THUD. WHOOSH!", known_names=()) == ()
    assert non_core_a2_word_spans("AAAAARRRGH! A-A-AAAARGH!", known_names=()) == ()
    assert non_core_a2_word_spans("Mmm! Ouch! A goblin ate a hamburger.", known_names=()) == ()

    for word in ("boom", "boomed", "Boom", "BOOM", "smash", "wham", "thud", "whoosh"):
        assert is_core_a2_word(word) is False

    lower_words = non_core_a2_words("A loud thud, smash, and wham whooshed as it boomed.")
    assert lower_words == ("thud", "smash", "wham", "whooshed", "boomed")


def test_a1_trivial_hint_words_compact_and_disjoint_from_borderline_and_overrides() -> None:
    from linguary.application.a2_core_vocab import A1_TRIVIAL_HINT_WORDS, _A1_IRREGULAR_TO_BASE

    assert isinstance(A1_TRIVIAL_HINT_WORDS, frozenset)
    assert A1_TRIVIAL_HINT_WORDS <= A2_CORE_VOCABULARY
    assert A1_TRIVIAL_HINT_WORDS.isdisjoint(_NON_CORE_OVERRIDE)
    assert set(_A1_IRREGULAR_TO_BASE.values()) <= A1_TRIVIAL_HINT_WORDS

    internationalisms = {
        "hamburger", "telescope", "taxi", "hotel", "film", "music", "doctor", "bank",
        "park", "sport", "coffee", "chocolate", "banana", "computer", "internet", "telephone",
    }
    assert internationalisms <= A1_TRIVIAL_HINT_WORDS

    borderline_and_excluded = {
        "handkerchief", "overcoat", "crept", "swore", "stung", "thud", "smash",
        "hug", "lantern", "cloak", "quill", "cauldron", "goblet", "wand",
    }
    assert A1_TRIVIAL_HINT_WORDS.isdisjoint(borderline_and_excluded)


@pytest.mark.parametrize(
    "original",
    [
        "hamburger",
        "telescope",
        "smiled",
        "to smile",
        "To Smile",
        "  telescopes  ",
        "telescopes",
        "smiling",
        "smiles",
        "taxi",
        "hotel",
        "film",
        "music",
        "doctor",
        "bank",
        "park",
        "sport",
        "coffee",
        "chocolate",
        "banana",
        "computer",
        "internet",
        "telephone",
        "buses",
        "boxes",
        "stopped",
        "carried",
        "walked",
        "walking",
        "children",
        "man",
        "men",
        "mouse",
        "mice",
        "went",
        "bought",
        "saw",
    ],
)
def test_is_trivial_a1_hint_accepts_core_a1_inflections_and_internationalisms(
    original: str,
) -> None:
    from linguary.application.a2_core_vocab import is_trivial_a1_hint

    assert is_trivial_a1_hint(original) is True


@pytest.mark.parametrize(
    "original",
    [
        "handkerchief",
        "overcoat",
        "crept",
        "swore",
        "stung",
        "thud",
        "smash",
        "wink",
        "groaned",
        "chuckled",
        "willing",
        "eyed",
        "handed",
        "to walk off",
        "to his surprise",
        "to make a grab for",
        "piece of cake",
        "break a leg",
        "to saw",
        "to went",
        "to him",
        "well-known",
        "123",
        "",
        "   ",
    ],
)
def test_is_trivial_a1_hint_rejects_borderline_a2_non_core_and_multiword(
    original: str,
) -> None:
    from linguary.application.a2_core_vocab import is_trivial_a1_hint

    assert is_trivial_a1_hint(original) is False


def test_going_is_core_a2_and_two_letter_ing_stems_allowed() -> None:
    """WP7: `going` is in A2_CORE_VOCABULARY and `_matches_ing` accepts 2-letter stems {go, do, be}."""
    from linguary.application.a2_core_vocab import _matches_ing, is_trivial_a1_hint

    assert "going" in A2_CORE_VOCABULARY
    assert is_core_a2_word("going") is True
    assert is_trivial_a1_hint("going") is True
    assert non_core_a2_words("Where are you going now?") == ()
    assert _matches_ing("going", frozenset({"go"})) is True
    assert _matches_ing("doing", frozenset({"do"})) is True
    assert _matches_ing("being", frozenset({"be"})) is True
    assert _matches_ing("axing", frozenset({"ax"})) is False


def test_is_trivial_a1_hint_strips_leading_articles() -> None:
    """WP5: leading a/an/the do not bypass the A1 trivial hint filter."""
    from linguary.application.a2_core_vocab import is_trivial_a1_hint

    assert is_trivial_a1_hint("a glass") is True
    assert is_trivial_a1_hint("the paper") is True
    assert is_trivial_a1_hint("the glasses") is True
    assert is_trivial_a1_hint("a wand") is False
    assert is_trivial_a1_hint("a went") is False
    for homograph in (
        "a can",
        "a will",
        "the will",
        "a spell",
        "a watch",
        "a fall",
        "the rest",
        "a study",
        "a well",
        "a fly",
        "a kind",
    ):
        assert is_trivial_a1_hint(homograph) is False, f"false A1 homograph: {homograph}"

