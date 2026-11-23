"""One book projection and the inputs shared by export and the web.

The projection calls the chapter and block helpers. It does not call a model
and it does not write. A missing active run is an empty result map, not an
error. Export readiness is not required.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from linguary.application.a2_core_vocab import is_trivial_a1_hint
from linguary.application.context_builder import textual_blocks
from linguary.application.errors import ApplicationError
from linguary.application.explained_expressions import expression_words
from linguary.application.layout.ipa_reference import IpaSymbol, ipa_key
from linguary.application.layout.legend import LayoutFeatures, legend_lines
from linguary.application.layout.notes import (
    _evidences,
    _find_end,
    _hint_evidence_keys,
    _hint_only_dropped,
    _ids_for,
    _note_evidence_key,
    _note_only_dropped,
    block_notes,
    vocabulary_dedup_key,
)
from linguary.application.layout.parts import LayoutPart, compose_parts
from linguary.application.layout.review import ReviewEntry
from linguary.application.layout.units import LayoutUnit, _whole_block_translation, block_units
from linguary.application.layout.who_is_who import WhoEntry, _content_chapter_numbers
from linguary.application.pronunciation_projection import project_pronunciations
from linguary.application.verse import detect_verse_stanza_runs
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    DocumentBlock,
    GlossaryHint,
    LanguageNote,
    TermConcept,
    glossary_key,
)
from linguary.domain.content_plan import (
    ContentPlan,
    ContextFact,
    Difficulty,
    HelpEvidence,
    SourceAnchor,
)

_SHOWN_CERTAINTY = frozenset({"explicit", "inferred"})
_UNIT_SPEAKER_CERTAINTY = frozenset({"inferred"})
_WORD_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_FADING_PROGRESSION_WORDS_THRESHOLD = 25_000
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_OPEN_QUOTES = {"“": "”", "‘": "’", "«": "»", '"': '"'}
_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "st", "prof"})
_REMARK_VERBS = frozenset({
    "say", "says", "said", "saying",
    "ask", "asks", "asked", "asking",
    "reply", "replies", "replied", "replying",
    "answer", "answers", "answered", "answering",
    "tell", "tells", "told", "telling",
    "add", "adds", "added", "adding",
    "continue", "continues", "continued", "continuing",
    "repeat", "repeats", "repeated", "repeating",
    "explain", "explains", "explained", "explaining",
    "insist", "insists", "insisted", "insisting",
    "suggest", "suggests", "suggested", "suggesting",
    "agree", "agrees", "agreed", "agreeing",
    "order", "orders", "ordered", "ordering",
    "command", "commands", "commanded", "commanding",
    "beg", "begs", "begged", "begging",
    "urge", "urges", "urged", "urging",
    "remind", "reminds", "reminded", "reminding",
    "demand", "demands", "demanded", "demanding",
    "murmur", "murmurs", "murmured", "murmuring",
    "whisper", "whispers", "whispered", "whispering",
    "snap", "snaps", "snapped", "snapping",
    "mutter", "mutters", "muttered", "muttering",
    "shout", "shouts", "shouted", "shouting",
    "cry", "cries", "cried", "crying",
    "stammer", "stammers", "stammered", "stammering",
    "gasp", "gasps", "gasped", "gasping",
    "bellow", "bellows", "bellowed", "bellowing",
    "hiss", "hisses", "hissed", "hissing",
    "growl", "growls", "growled", "growling",
    "exclaim", "exclaims", "exclaimed", "exclaiming",
    "sob", "sobs", "sobbed", "sobbing",
    "plead", "pleads", "pleaded", "pleading",
    "retort", "retorts", "retorted", "retorting",
    "bark", "barks", "barked", "barking",
    "croak", "croaks", "croaked", "croaking",
    "splutter", "splutters", "spluttered", "spluttering",
    "drawl", "drawls", "drawled", "drawling",
    "sneer", "sneers", "sneered", "sneering",
    "sigh", "sighs", "sighed", "sighing",
    "groan", "groans", "groaned", "groaning",
    "yell", "yells", "yelled", "yelling",
    "shriek", "shrieks", "shrieked", "shrieking",
    "whimper", "whimpers", "whimpered", "whimpering",
    "thunder", "thunders", "thundered", "thundering",
    "roar", "roars", "roared", "roaring",
    "inquire", "inquires", "inquired", "inquiring",
    "enquire", "enquires", "enquired", "enquiring",
    "protest", "protests", "protested", "protesting",
    "confess", "confesses", "confessed", "confessing",
    "admit", "admits", "admitted", "admitting",
    "announce", "announces", "announced", "announcing",
    "declare", "declares", "declared", "declaring",
    "remark", "remarks", "remarked", "remarking",
    "interrupt", "interrupts", "interrupted", "interrupting",
    "warn", "warns", "warned", "warning",
    "mumble", "mumbles", "mumbled", "mumbling",
    "snarl", "snarls", "snarled", "snarling",
    "wail", "wailed", "wailing", "wails",
    "scream", "screams", "screamed", "screaming",
    "squeak", "squeaks", "squeaked", "squeaking",
    "rasp", "rasps", "rasped", "rasping",
    "blurt", "blurts", "blurted", "blurting",
    "scoff", "scoffs", "scoffed", "scoffing",
    "falter", "falters", "faltered", "faltering",
    "grunt", "grunts", "grunted", "grunting",
    "snort", "snorts", "snorted", "snorting",
    "whine", "whines", "whined", "whining",
    "boom", "booms", "boomed", "booming",
    "spit", "spits", "spat", "spitting",
    "moan", "moans", "moaned", "moaning",
})
_REMARK_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.IGNORECASE)
_WORD_FINAL_APOSTROPHE = re.compile(
    r"’(?=(?:\s+\w|[,;:])(?:[^‘’]|’(?=\w))*’(?!\w))"
)
_TITLE_TRANSLATIONS = {
    "mr": "мистер",
    "mrs": "миссис",
    "miss": "мисс",
    "ms": "мис",
    "uncle": "дядя",
    "aunt": "тётя",
    "professor": "профессор",
    "dr": "доктор",
    "sir": "сэр",
    "madam": "мадам",
    "lord": "лорд",
}
_RUSSIAN_TITLES = {
    glossary_key(translation): translation
    for translation in _TITLE_TRANSLATIONS.values()
}
_SPEECH_STYLE_FIRST_SCENE_KEEP = 2
_SPEECH_STYLE_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})
_SPEECH_STYLE_TOKEN = re.compile(r"[a-z]+(?:'[a-z]*)*|'[a-z]+", re.IGNORECASE)
_CONTRACTION_SUFFIXES = frozenset({"ll", "re", "ve", "d", "m", "s"})
_SPEECH_STYLE_STOPWORDS = frozenset(
    {
        "a", "an", "the", "i", "you", "he", "she", "it", "we", "they",
        "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
        "to", "of", "in", "on", "at", "by", "for", "with", "from", "up", "out",
        "is", "am", "are", "was", "were", "be", "been",
        "do", "does", "did", "have", "has", "had",
        "will", "would", "can", "could", "not", "no",
        "and", "or", "but", "if", "that", "this", "so", "as", "all",
    }
)
_HINT_LOCAL_WINDOW_BLOCKS = 20
_HINT_REINTRODUCE_CHAPTER_GAP = 3
_GRAMMAR_MIN_CHAPTER_GAP = 2
_GRAMMAR_PATTERN_RULES: tuple[tuple[str, re.Pattern[str], int, int], ...] = (
    (
        "used_to",
        re.compile(r"\bused\s+to\b|\bdid(?:n't|\s+not)\s+use\s+to\b", re.IGNORECASE),
        2,
        _GRAMMAR_MIN_CHAPTER_GAP,
    ),
    (
        "modal_must_have",
        re.compile(r"\bmust(?:\s+not|n't)?(?:\s+have|'ve)\b", re.IGNORECASE),
        3,
        _GRAMMAR_MIN_CHAPTER_GAP,
    ),
    (
        "modal_would_have",
        re.compile(r"\bwould(?:\s+not|n't)?(?:\s+have|'ve)\b", re.IGNORECASE),
        3,
        _GRAMMAR_MIN_CHAPTER_GAP,
    ),
    (
        "modal_could_have",
        re.compile(r"\bcould(?:\s+not|n't)?(?:\s+have|'ve)\b", re.IGNORECASE),
        3,
        _GRAMMAR_MIN_CHAPTER_GAP,
    ),
    (
        "modal_should_have",
        re.compile(r"\bshould(?:\s+not|n't)?(?:\s+have|'ve)\b", re.IGNORECASE),
        3,
        _GRAMMAR_MIN_CHAPTER_GAP,
    ),
    (
        "modal_might_have",
        re.compile(r"\bmight(?:\s+not|n't)?(?:\s+have|'ve)\b", re.IGNORECASE),
        3,
        _GRAMMAR_MIN_CHAPTER_GAP,
    ),
    (
        "wish_past",
        re.compile(
            r"\bwish(?:es|ed)?\s*\+\s*(?:past|прошедш|were|had)"
            r"|\bwish(?:es|ed)?\s+(?:i|he|she|we|they|you|it)\s+"
            r"(?:were|was|had|could|would|didn't|hadn't|weren't|knew|lived|spoke|went|came|did)\b"
            r"|\bwish\b.{0,45}(?:прошедш\w*\s+врем|past\s+simple|past\s+perfect|сослагательн|нереальн\w*\s+желан|were\s+вместо\s+was)",
            re.IGNORECASE,
        ),
        3,
        _GRAMMAR_MIN_CHAPTER_GAP,
    ),
    (
        "double_negative",
        re.compile(
            r"\bdouble\s+negat(?:ive|ion)\b"
            r"|двойн\w*\s+отрицан|два\s+отрицани"
            r"|\b(?:don't|doesn't|didn't|ain't|won't|can't|couldn't|wouldn't|shouldn't|wasn't|weren't|haven't|hasn't|hadn't|never)\b"
            r"[^.!?]{0,25}\b(?:nothing|nobody|no\s+one|nowhere|neither|none|no\s+[a-z]+)\b",
            re.IGNORECASE,
        ),
        3,
        _GRAMMAR_MIN_CHAPTER_GAP,
    ),
    (
        "preposition_stranding",
        re.compile(
            r"\bpreposition\s+stranding\b|\bstranded\s+preposition\b"
            r"|предлог\w*(?:\s+[«\"'(]?[a-z]{1,12}[»\"')]*?)?\s+"
            r"(?:(?:часто|обычно)\s+)?"
            r"(?:(?:стоит|уходит|вынесен\w*|оста[ёе]тся|переносится|ставится|оказывается)\s+)?"
            r"в\s+кон(?:ец|це)"
            r"|в\s+кон(?:ец|це)\s+(?:вопроса|предложения|фразы|придаточного)[^.!?]{0,30}предлог",
            re.IGNORECASE,
        ),
        3,
        _GRAMMAR_MIN_CHAPTER_GAP,
    ),
    (
        "passive_infinitive",
        re.compile(
            r"\bpassive\s+infinitive\b|(?:пассивн\w*|страдательн\w*)\s+инфинитив"
            r"|\bto\s+be\s*\+\s*(?:v3|3-?я|треть|причаст|past\s+participle)"
            r"|\bto\s+be\s+(?:seen|done|told|found|made|taken|given|kept|left|sent|heard|known|called|held|put|set|shown|written|read|built|bought|caught|chosen|drawn|driven|eaten|fed|felt|forgotten|forgiven|frozen|grown|hidden|hit|hurt|led|lent|lost|meant|met|paid|ridden|rung|said|shaken|shut|sold|spent|spoken|stolen|struck|swept|taught|thought|thrown|understood|woken|worn|won|[a-z]{3,}ed)\b",
            re.IGNORECASE,
        ),
        3,
        _GRAMMAR_MIN_CHAPTER_GAP,
    ),
    (
        "past_perfect",
        re.compile(
            r"\bpast\s+perfect\b|предпрошедш"
            r"|\bhad\s*\+\s*(?:v3|3-?я|треть\w*\s+форм|причаст|глагол|past\s+participle)"
            r"|(?:ранее|раньше|до)\s+(?:другого\s+)?(?:события|действия|момента)\s+в\s+прошлом",
            re.IGNORECASE,
        ),
        3,
        _GRAMMAR_MIN_CHAPTER_GAP,
    ),
)


@dataclass(slots=True)
class _HintSeen:
    count: int
    last_shown_block: int
    last_shown_chapter: int
    last_counted_block: int


@dataclass(slots=True)
class _GrammarSeen:
    count: int
    last_chapter: int


@dataclass(frozen=True, slots=True)
class LayoutInputs:
    """Stored data one projection reads. Nothing here is computed from a model."""

    document: CanonicalDocument
    results: Mapping[str, BlockProcessingResult]
    analyses: Sequence[ChapterAnalysis]
    concepts: Sequence[TermConcept]
    reader_level: str | None
    english_help: bool = False


@dataclass(frozen=True, slots=True)
class ChapterLayout:
    """One section: reading units, and extras only when the section is content.

    `number` is none for front matter and for a notes-only section. Those
    sections still carry a unit for every block.
    """

    section_id: str
    number: int | None
    units: Mapping[str, tuple[LayoutUnit, ...]]
    who_is_who: tuple[WhoEntry, ...]
    review: tuple[ReviewEntry, ...]


@dataclass(frozen=True, slots=True)
class BookLayout:
    """The finished book: chapters in reading order, then the book-level pages."""

    chapters: Mapping[str, ChapterLayout]
    legend: tuple[str, ...]
    ipa_key: tuple[IpaSymbol, ...]
    who_is_who: tuple[WhoEntry, ...]
    features: LayoutFeatures


def build_layout_inputs(unit_of_work, book_id: str) -> LayoutInputs:
    """Load the document, active-run results, analyses, glossary, and conditions.

    `read_transaction` is used when the unit of work has it, otherwise
    `transaction`. No active run leaves `results` empty. No conditions leaves
    `reader_level` unset and `english_help` false. A missing book or document
    is an error. The run does not have to be finished, and chapters do not
    have to have a content plan.
    """
    with getattr(unit_of_work, "read_transaction", unit_of_work.transaction)() as transaction:
        if transaction.books.get(book_id) is None:
            raise ApplicationError(f"книги {book_id} нет в хранилище")
        document = transaction.documents.get(book_id)
        if document is None:
            raise ApplicationError(f"книга {book_id} не импортирована")
        run = transaction.runs.active_for_book(book_id)
        if run is None:
            results: dict[str, BlockProcessingResult] = {}
        else:
            results = {
                item.block_id: item for item in transaction.results.list_for_run(run.id)
            }
        analyses = tuple(transaction.analyses.list_for_book(book_id))
        concepts = tuple(transaction.glossary.list_for_book(book_id))
        reader_level, english_help = _reader_settings(transaction.work.latest_conditions(book_id))
    return LayoutInputs(document, results, analyses, concepts, reader_level, english_help)


def build_book_layout(inputs: LayoutInputs) -> BookLayout:
    """Project `inputs` into chapters, legend lines, and the book-level lists.

    Unit labels use inferred speaker facts only. Repeated speech styles still
    count explicit and inferred facts: the first 2 speech-style difficulties
    in a speaker's first scene/chapter and later difficulties with non-overlapping
    dialect lexemes are kept; repeated ones are removed from the plan copy and
    passed to `block_notes` as `dropped_difficulty_ids`. Glossary hints undergo
    book-wide 3-stage deduplication and grammar `language_notes` are capped by
    11 closed regex patterns via suppression/stripping index sets without
    mutating stored results. `previously` is included in `features.sections`
    when there are >= 2 content chapters and a non-last content chapter has a
    non-empty summary. Chapter review and the book who-is-who list stay empty.
    """
    document = inputs.document
    numbers = _content_chapter_numbers(document)
    facts = _speaker_facts(inputs.analyses)
    inferred_by_block = _inferred_facts_by_block(facts)
    speaker_index = (
        _speaker_index(
            inputs.concepts,
            inputs.analyses,
            document.target_language,
            document,
        )
        if inferred_by_block
        else None
    )
    dropped_by_section = _repeated_speech_style_ids(document, inputs.analyses, facts)
    plans = _plans_by_section(inputs.analyses)
    pronunciations = project_pronunciations(tuple(document.blocks()), inputs.results)
    chapters: dict[str, ChapterLayout] = {}
    hint_history: dict[str, _HintSeen] = {}
    grammar_history: dict[str, _GrammarSeen] = {}
    block_seq = 0
    current_chapter_num = 0
    for section in document.ordered_sections:
        section_number = numbers.get(section.id)
        if section_number is not None:
            current_chapter_num = section_number
        dropped = dropped_by_section.get(section.id, frozenset())
        plan = _plan_without(plans.get(section.id), dropped)
        poem_block_ids = {
            poem_block.id
            for run in detect_verse_stanza_runs(section.ordered_blocks)
            for poem_block in run
        }
        units: dict[str, tuple[LayoutUnit, ...]] = {}
        for block in section.ordered_blocks:
            result = inputs.results.get(block.id)
            treatments = _treatments(block, result)
            notes = ()
            if result is not None:
                suppressed_hints, stripped_hints = _deduplicate_glossary_hints(
                    block,
                    result,
                    dropped_difficulty_ids=dropped,
                    block_seq=block_seq,
                    chapter_num=current_chapter_num,
                    history=hint_history,
                    plan=plan,
                    reader_level=inputs.reader_level,
                )
                suppressed_notes = _deduplicate_grammar_notes(
                    result,
                    dropped_difficulty_ids=dropped,
                    chapter_num=current_chapter_num,
                    history=grammar_history,
                    plan=plan,
                )
                notes = block_notes(
                    block,
                    result,
                    plan,
                    pronunciations,
                    english_help=inputs.english_help,
                    dropped_difficulty_ids=dropped,
                    suppressed_hint_indices=suppressed_hints,
                    stripped_hint_indices=stripped_hints,
                    suppressed_note_indices=suppressed_notes,
                )
            block_seq += 1
            block_facts = inferred_by_block.get(block.id, ())
            speakers = (
                _speakers_for_block(
                    block,
                    block_facts,
                    inputs.concepts,
                    inputs.analyses,
                    document.target_language,
                    treatments,
                    result,
                    document=document,
                    _index=speaker_index,
                )
                if block_facts
                else {}
            )
            units[block.id] = block_units(
                block,
                result,
                treatments,
                notes,
                speakers,
                target_language=document.target_language,
                is_poem=block.id in poem_block_ids,
            )
        chapters[section.id] = ChapterLayout(
            section_id=section.id,
            number=section_number,
            units=units,
            who_is_who=(),
            review=(),
        )
    features = _features(
        document,
        chapters,
        inputs.analyses,
        reader_level=inputs.reader_level,
        english_help=inputs.english_help,
    )
    return BookLayout(
        chapters=chapters,
        legend=legend_lines(features),
        ipa_key=ipa_key(_ipa_strings(chapters)),
        who_is_who=(),
        features=features,
    )


def _reader_settings(conditions: object) -> tuple[str | None, bool]:
    """Reader level and English help from conditions. Both default when absent."""
    if conditions is None:
        return None, False
    payload = getattr(conditions, "payload", None)
    if not isinstance(payload, Mapping):
        return None, False
    level = payload.get("reader_level")
    reader_level = level if isinstance(level, str) else None
    return reader_level, bool(payload.get("english_help", False))


def _treatments(
    block: DocumentBlock,
    result: BlockProcessingResult | None,
) -> tuple[BlockTreatment | None, ...]:
    """One treatment per span. No result is all none. An empty result repeats the block mode."""
    count = len(block.segment_spans)
    if result is None:
        return (None,) * count
    if not result.segment_treatments:
        return (result.treatment,) * count
    return result.segment_treatments


def _plans_by_section(analyses: Sequence[ChapterAnalysis]) -> dict[str, ContentPlan]:
    """Latest content plan for each section. An analysis without one is skipped."""
    found: dict[str, ContentPlan] = {}
    for analysis in analyses:
        if analysis.content_plan is not None:
            found[analysis.section_id] = analysis.content_plan
    return found


def _speaker_facts(analyses: Sequence[ChapterAnalysis]) -> tuple[ContextFact, ...]:
    """Speaker facts with a name. `unknown` and other kinds are not shown."""
    found: list[ContextFact] = []
    for analysis in analyses:
        plan = analysis.content_plan
        if plan is None:
            continue
        for fact in plan.context_facts:
            if fact.kind != "speaker" or fact.certainty not in _SHOWN_CERTAINTY:
                continue
            if not isinstance(fact.value, str) or not fact.value.strip():
                continue
            found.append(fact)
    return tuple(found)


@dataclass(frozen=True, slots=True)
class _SpeakerIndex:
    """Pre-indexed user decisions and person candidates for speaker resolution."""

    language: str
    user_by_key: Mapping[str, str]
    candidate_by_key: Mapping[str, str]


def _inferred_facts_by_block(
    facts: Sequence[ContextFact],
) -> dict[str, tuple[ContextFact, ...]]:
    grouped: dict[str, list[ContextFact]] = {}
    for fact in facts:
        if fact.certainty in _SHOWN_CERTAINTY and isinstance(fact.value, str):
            grouped.setdefault(fact.subject_anchor.block_id, []).append(fact)
    return {
        block_id: tuple(items)
        for block_id, items in grouped.items()
        if any(item.certainty in _UNIT_SPEAKER_CERTAINTY for item in items)
    }


def _speaker_index(
    concepts: Sequence[TermConcept],
    analyses: Sequence[ChapterAnalysis],
    language: str,
    document: CanonicalDocument | None,
) -> _SpeakerIndex:
    normalized = _normalized_language(language)
    user_by_key: dict[str, str] = {}
    for concept in concepts:
        key = glossary_key(concept.canonical_source_text)
        if not key or key in user_by_key:
            continue
        chosen = concept.variant_for(normalized)
        if chosen is not None and chosen.status.is_user_decision and chosen.text.strip():
            chosen_text = chosen.text.strip()
            user_by_key[key] = chosen_text
            target_key = glossary_key(chosen_text)
            if target_key:
                user_by_key.setdefault(target_key, chosen_text)
    russian = normalized == "ru"
    allowed = _content_section_ids(document)
    candidate_by_key: dict[str, str] = {}
    for analysis in _analyses_in_reading_order(analyses, document, allowed):
        for candidate in analysis.glossary_candidates:
            if candidate.type is not ConceptType.PERSON:
                continue
            key = glossary_key(candidate.source_text)
            if not key or key in candidate_by_key:
                continue
            translation = candidate.suggested_translation.strip()
            if not translation:
                continue
            if russian and _CYRILLIC.search(translation) is None:
                continue
            candidate_by_key[key] = translation
            target_key = glossary_key(translation)
            if target_key:
                candidate_by_key.setdefault(target_key, translation)
    return _SpeakerIndex(normalized, user_by_key, candidate_by_key)


def _speaker_name(
    value: str,
    concepts: Sequence[TermConcept],
    analyses: Sequence[ChapterAnalysis],
    language: str,
    *,
    document: CanonicalDocument | None = None,
    _index: _SpeakerIndex | None = None,
) -> str | None:
    """Russian (or other target) name for `value`, or none when it stays unresolved.

    Keys are the whole value, the text before `(`, and the text inside `(...)`.
    A user decision wins, then a person candidate from a content section, then
    for Russian a title or kinship prefix plus the same lookup on the remainder.
    The English source is never returned on its own.
    """
    variants = _name_variants(value)
    if not variants:
        return None
    index = _index or _speaker_index(concepts, analyses, language, document)
    for variant in variants:
        found = index.user_by_key.get(glossary_key(variant))
        if found is not None:
            return found
    for variant in variants:
        found = index.candidate_by_key.get(glossary_key(variant))
        if found is not None:
            return found
    if index.language == "ru":
        for variant in variants:
            found = _titled_name_from_index(variant, index)
            if found is not None:
                return found
    return None


def _speakers_for_block(
    block: DocumentBlock,
    facts: Sequence[ContextFact],
    concepts: Sequence[TermConcept],
    analyses: Sequence[ChapterAnalysis],
    language: str,
    treatments: Sequence[BlockTreatment | None],
    result: BlockProcessingResult | None = None,
    *,
    document: CanonicalDocument | None = None,
    _index: _SpeakerIndex | None = None,
) -> dict[tuple[int, int], str]:
    """Inferred subject anchors whose shown part opens the line or continues its quote.

    Parts follow `compose_parts`, including the single unit used when a whole-block
    translation replaces those parts. Secondary anchors are ignored. A part that
    starts as narrative, with a quote only later, is absent. The first resolved
    name keeps a repeated part.
    """
    block_facts = [
        fact
        for fact in facts
        if fact.certainty in _UNIT_SPEAKER_CERTAINTY
        and isinstance(fact.value, str)
        and fact.subject_anchor.block_id == block.id
    ]
    if not block_facts:
        return {}
    explicit_facts = [
        fact
        for fact in facts
        if fact.certainty == "explicit"
        and isinstance(fact.value, str)
        and fact.subject_anchor.block_id == block.id
    ]
    index = _index or _speaker_index(concepts, analyses, language, document)
    parts = _shown_parts(block, treatments, result)
    explicit_parts = {
        (part.start, part.end)
        for fact in explicit_facts
        for part in parts
        if _part_shows_speaker(block.text, part, fact.subject_anchor.start)
    }
    speakers: dict[tuple[int, int], str] = {}
    for fact in block_facts:
        if fact.value is None:
            continue
        name = _speaker_name(
            fact.value,
            concepts,
            analyses,
            language,
            document=document,
            _index=index,
        )
        if name is None:
            continue
        anchor = fact.subject_anchor
        for part in parts:
            bounds = (part.start, part.end)
            if bounds not in explicit_parts and _part_shows_speaker(block.text, part, anchor.start):
                speakers.setdefault(bounds, name)
    return speakers


def _name_variants(value: str) -> tuple[str, ...]:
    """Whole name, the part before parentheses, and the part inside them."""
    text = value.strip()
    if not text:
        return ()
    found = [text]
    if "(" in text and ")" in text:
        before, _, rest = text.partition("(")
        inside, _, _ = rest.partition(")")
        if before.strip():
            found.append(before.strip())
        if inside.strip():
            found.append(inside.strip())
    variants: list[str] = []
    seen: set[str] = set()
    for variant in found:
        key = glossary_key(variant)
        if not key or key in seen:
            continue
        seen.add(key)
        variants.append(variant)
    return tuple(variants)


def _user_translation(
    variant: str,
    concepts: Sequence[TermConcept],
    language: str,
) -> str | None:
    """Approved or locked variant for this key. A suggestion does not count."""
    key = glossary_key(variant)
    if not key:
        return None
    normalized = _normalized_language(language)
    for concept in concepts:
        chosen = concept.variant_for(normalized)
        if chosen is None or not chosen.status.is_user_decision or not chosen.text.strip():
            continue
        chosen_text = chosen.text.strip()
        if key in {glossary_key(concept.canonical_source_text), glossary_key(chosen_text)}:
            return chosen_text
    return None


def _candidate_translation(
    variant: str,
    analyses: Sequence[ChapterAnalysis],
    language: str,
    document: CanonicalDocument | None,
) -> str | None:
    """Person suggestion from a content section. Russian text must use Cyrillic."""
    key = glossary_key(variant)
    if not key:
        return None
    russian = _normalized_language(language) == "ru"
    allowed = _content_section_ids(document)
    for analysis in _analyses_in_reading_order(analyses, document, allowed):
        for candidate in analysis.glossary_candidates:
            if candidate.type is not ConceptType.PERSON:
                continue
            translation = candidate.suggested_translation.strip()
            if not translation:
                continue
            if russian and _CYRILLIC.search(translation) is None:
                continue
            if key in {glossary_key(candidate.source_text), glossary_key(translation)}:
                return translation
    return None


def _titled_name_from_index(variant: str, index: _SpeakerIndex) -> str | None:
    """Russian title or kinship word plus a resolved remainder. No English fallback."""
    if index.language != "ru":
        return None
    token, separator, rest = glossary_key(variant).partition(" ")
    title = _TITLE_TRANSLATIONS.get(token) or _RUSSIAN_TITLES.get(token)
    if not separator or title is None or not rest.strip():
        return None
    key = glossary_key(rest)
    translated = index.user_by_key.get(key) or index.candidate_by_key.get(key)
    if translated is None:
        return None
    return f"{title} {translated}"


def _titled_name(
    variant: str,
    concepts: Sequence[TermConcept],
    analyses: Sequence[ChapterAnalysis],
    language: str,
    document: CanonicalDocument | None,
) -> str | None:
    """Russian title or kinship word plus a resolved remainder. No English fallback."""
    return _titled_name_from_index(
        variant,
        _speaker_index(concepts, analyses, language, document),
    )


def _normalized_language(language: str) -> str:
    return language.strip().lower()


def _content_section_ids(document: CanonicalDocument | None) -> frozenset[str] | None:
    """Content section ids, or none when the document was not passed in."""
    if document is None:
        return None
    return frozenset(
        section.id
        for section in document.ordered_sections
        if not document.is_front_matter(section) and not section.is_notes_only
    )


def _analyses_in_reading_order(
    analyses: Sequence[ChapterAnalysis],
    document: CanonicalDocument | None,
    allowed: frozenset[str] | None,
) -> tuple[ChapterAnalysis, ...]:
    """Content analyses in section order. Without a document, keep the given order."""
    if document is None:
        return tuple(analyses)
    grouped: dict[str, list[ChapterAnalysis]] = {}
    for analysis in analyses:
        if allowed is not None and analysis.section_id not in allowed:
            continue
        grouped.setdefault(analysis.section_id, []).append(analysis)
    ordered: list[ChapterAnalysis] = []
    for section in document.ordered_sections:
        ordered.extend(grouped.get(section.id, ()))
    return tuple(ordered)


def _shown_parts(
    block: DocumentBlock,
    treatments: Sequence[BlockTreatment | None],
    result: BlockProcessingResult | None,
) -> tuple[LayoutPart, ...]:
    """Parts `block_units` will build, so a collapsed translation is one part."""
    chosen = tuple(treatments)
    parts = compose_parts(block, chosen)
    if _whole_block_translation(result, chosen) is None:
        return parts
    spans = block.segment_spans
    return (
        LayoutPart(
            start=spans[0].start,
            end=spans[-1].end,
            treatment=BlockTreatment.TRANSLATE,
            segments=tuple(range(len(spans))),
        ),
    )


def _part_shows_speaker(text: str, part: LayoutPart, anchor_start: int) -> bool:
    """Whether `anchor_start` belongs to the dialogue that opens this part."""
    opening_end = _opening_speech_end(text, part.start, part.end)
    if opening_end is not None and part.start <= anchor_start < opening_end:
        return True
    continued = _continued_speech_bounds(text, part.start)
    if continued is None:
        return False
    start, end = continued
    return start <= anchor_start < end


def _opening_speech_end(text: str, part_start: int, part_end: int) -> int | None:
    """End of the opening replica and its remark, when the part starts with one."""
    open_at = _first_nonspace(text, part_start, part_end)
    if open_at is None or not _is_dialogue_opener(text, open_at, part_end):
        return None
    limit = len(text)
    if text[open_at] not in _OPEN_QUOTES:
        return _sentence_end(text, open_at, limit)
    close = _matching_close(text, open_at, limit)
    if close is None:
        return limit
    return _remark_end(text, close, limit)


def _continued_speech_bounds(text: str, part_start: int) -> tuple[int, int] | None:
    """Quote opened before this part and still open there, through its remark."""
    if part_start <= 0:
        return None
    open_at = _outer_unclosed_open(text, part_start)
    if open_at is None:
        return None
    close = _matching_close(text, open_at, len(text))
    if close is None:
        return (open_at, len(text))
    if close < part_start:
        return None
    return (open_at, _remark_end(text, close, len(text)))


def _is_apostrophe(text: str, index: int) -> bool:
    """Whether `’` at `index` is an apostrophe inside a replica rather than its closer."""
    if text[index] != "’":
        return False
    if index + 1 < len(text) and text[index + 1].isalnum():
        return True
    if index > 0 and text[index - 1].isalnum():
        return _WORD_FINAL_APOSTROPHE.match(text, index) is not None
    return False


def _outer_unclosed_open(text: str, limit: int | None = None) -> int | None:
    """Index of the outermost quote that `text[:limit]` does not close."""
    end = len(text) if limit is None else min(limit, len(text))
    stack: list[tuple[int, str]] = []
    for index in range(end):
        char = text[index]
        if char in _OPEN_QUOTES:
            if char == '"' and stack and stack[-1][1] == '"':
                stack.pop()
            else:
                stack.append((index, _OPEN_QUOTES[char]))
        elif stack and char == stack[-1][1]:
            if char == "’" and _is_apostrophe(text, index):
                continue
            stack.pop()
    if not stack:
        return None
    return stack[0][0]


def _matching_close(text: str, open_index: int, limit: int) -> int | None:
    """Index of the quote that closes the opener at `open_index`, inside `limit`."""
    opener = text[open_index]
    if opener not in _OPEN_QUOTES:
        return None
    stack = [_OPEN_QUOTES[opener]]
    for index in range(open_index + 1, limit):
        char = text[index]
        if char in _OPEN_QUOTES:
            if char == '"' and stack and stack[-1] == '"':
                stack.pop()
            else:
                stack.append(_OPEN_QUOTES[char])
        elif stack and char == stack[-1]:
            if char == "’" and _is_apostrophe(text, index):
                continue
            stack.pop()
        if not stack:
            return index
    return None


def _remark_end(text: str, close_index: int, limit: int) -> int:
    """End of the reporting clause after a closing quote, before the next sentence.

    A period before the closing quote ends the sentence. A comma before the
    closing quote always continues into the reporting clause. After `?!…—`, a
    capital right after the quote stays in the remark when the clause before the
    next quote has a speech verb (`Harry said`, `I gasped`, `Dudley yelled`).
    """
    if (
        close_index > 0
        and text[close_index - 1] == "."
        and not text[:close_index].endswith("...")
    ):
        return close_index + 1
    index = close_index + 1
    while index < limit and text[index].isspace():
        index += 1
    if index >= limit:
        return limit
    if text[index] in _OPEN_QUOTES:
        return close_index + 1
    sentence_end = _sentence_end(text, close_index + 1, limit)
    clause_end = _next_open_quote(text, index, sentence_end)
    if close_index > 0 and text[close_index - 1] == ",":
        return clause_end
    if text[index].isupper() and not _speech_verb_in(text, index, clause_end):
        return close_index + 1
    return clause_end


def _next_open_quote(text: str, start: int, end: int) -> int:
    for index in range(start, end):
        if text[index] in _OPEN_QUOTES:
            return index
    return end


def _speech_verb_in(text: str, start: int, end: int) -> bool:
    """Whether the sentence slice contains a finite speech verb."""
    return any(
        match.group(0).casefold() in _REMARK_VERBS
        for match in _REMARK_WORD.finditer(text, start, end)
    )


def _sentence_end(text: str, start: int, limit: int) -> int:
    """Exclusive end of the sentence containing `start`, capped at `limit`."""
    index = start
    while index < limit:
        if text[index] in ".!?" and _ends_sentence(text, index, limit):
            return index + 1
        index += 1
    return limit


def _ends_sentence(text: str, index: int, limit: int) -> bool:
    """A terminator that is not an abbreviation dot and is followed by a break."""
    if text[index] == "." and _abbreviation_dot(text, index):
        return False
    next_index = index + 1
    return next_index >= limit or text[next_index].isspace()


def _abbreviation_dot(text: str, index: int) -> bool:
    word_start = index
    while word_start > 0 and text[word_start - 1].isalpha():
        word_start -= 1
    return text[word_start:index].casefold() in _ABBREVIATIONS


def _first_nonspace(text: str, start: int, end: int) -> int | None:
    index = start
    while index < end and text[index].isspace():
        index += 1
    if index >= end:
        return None
    return index


def _is_dialogue_opener(text: str, index: int, end: int) -> bool:
    """Opening quote or a dialogue dash, not a hyphen glued to a word."""
    char = text[index]
    if char in _OPEN_QUOTES or char in "—–":
        return True
    return char == "-" and index + 1 < end and text[index + 1] in " —–"


def _hint_segment_index(
    block: DocumentBlock,
    result: BlockProcessingResult,
    hint: GlossaryHint,
) -> int | None:
    """Segment index containing `hint`, or None for whole-block or unmatched hints."""
    spans = block.segment_spans
    treatments = result.segment_treatments
    if not treatments or len(treatments) != len(spans):
        return None
    marker_end = _find_end(block.text, hint.original)
    if marker_end is None:
        return None
    for index, span in enumerate(spans):
        if span.start < marker_end <= span.end:
            return index
    return None


def _hint_is_glossary_only(
    block: DocumentBlock,
    result: BlockProcessingResult,
    hint: GlossaryHint,
    *,
    requires_hints: bool,
) -> bool:
    """Whether `hint` serves an untranslated `GLOSSARY_ONLY` block or segment."""
    if not requires_hints:
        return False
    if result.treatment is BlockTreatment.GLOSSARY_ONLY and not result.segment_treatments:
        return True
    seg_index = _hint_segment_index(block, result, hint)
    if seg_index is None:
        return True
    return result.segment_treatments[seg_index].requires_glossary_hints


def _glossary_only_bucket(
    block: DocumentBlock,
    result: BlockProcessingResult,
    hint: GlossaryHint,
) -> int | None:
    """Bucket key (`None` for whole block, segment index for mixed `GLOSSARY_ONLY` segments)."""
    if result.treatment is BlockTreatment.GLOSSARY_ONLY and not result.segment_treatments:
        return None
    treatments = result.segment_treatments
    if not treatments or len(treatments) != len(block.segment_spans):
        return None
    glossary_segments = [
        index for index, mode in enumerate(treatments) if mode.requires_glossary_hints
    ]
    if len(glossary_segments) == 1:
        return glossary_segments[0]
    return _hint_segment_index(block, result, hint)


def _hint_is_protected_a1(
    hint: GlossaryHint,
    block_id: str,
    evidences: Sequence[HelpEvidence],
    known: Mapping[str, Difficulty],
    dropped_difficulty_ids: frozenset[str],
) -> bool:
    """True when an A1 hint carries a non-empty note or links to a non-`vocabulary` difficulty."""
    if hint.note and hint.note.strip():
        return True
    if not known:
        return False
    raw_ids = _ids_for(evidences, _hint_evidence_keys(hint, block_id))
    return any(
        item_id not in dropped_difficulty_ids
        and item_id in known
        and known[item_id].kind != "vocabulary"
        for item_id in raw_ids
    )


def _a1_pre_suppressed_indices(
    block: DocumentBlock,
    result: BlockProcessingResult,
    *,
    requires_hints: bool,
    evidences: Sequence[HelpEvidence],
    known: Mapping[str, Difficulty],
    dropped_difficulty_ids: frozenset[str],
    only_dropped_by_index: Mapping[int, bool],
    glossary_only_by_index: Mapping[int, bool],
    bucket_by_index: Mapping[int, int | None],
) -> set[int]:
    """Select unprotected trivial A1 hint indices to drop before history/rescue updates."""
    droppable_a1: dict[int, bool] = {}
    glossary_buckets: dict[int | None, list[int]] = {}
    pre_suppressed: set[int] = set()
    for index, hint in enumerate(result.glossary_hints):
        is_droppable = is_trivial_a1_hint(hint.original) and not _hint_is_protected_a1(
            hint, result.block_id, evidences, known, dropped_difficulty_ids
        )
        droppable_a1[index] = is_droppable
        if not glossary_only_by_index[index]:
            if is_droppable and not (only_dropped_by_index[index] and not requires_hints):
                pre_suppressed.add(index)
        else:
            glossary_buckets.setdefault(bucket_by_index[index], []).append(index)
    for indices in glossary_buckets.values():
        if any(not droppable_a1[idx] for idx in indices):
            for idx in indices:
                if droppable_a1[idx]:
                    pre_suppressed.add(idx)
        elif indices:
            keep_idx = next(
                (idx for idx in indices if not only_dropped_by_index[idx]),
                indices[0],
            )
            for idx in indices:
                if idx != keep_idx:
                    pre_suppressed.add(idx)
    return pre_suppressed


def _deduplicate_glossary_hints(
    block: DocumentBlock,
    result: BlockProcessingResult,
    *,
    dropped_difficulty_ids: frozenset[str],
    block_seq: int,
    chapter_num: int,
    history: dict[str, _HintSeen],
    plan: ContentPlan | None = None,
    reader_level: str | None = None,
) -> tuple[frozenset[int], frozenset[int]]:
    """Book-wide 3-stage deduplication for glossary hints without mutating `result`.

    - For `reader_level == "a2"`, unprotected trivial A1 hints are filtered before updating
      `history` or `glossary_only_rescue`, keeping 1 A1 fallback when a `GLOSSARY_ONLY`
      block or segment has only A1 hints.
    - After a gap > 3 chapters since last shown: reset and reintroduce cleanly.
    - Within a 20-block local window: suppress without inflating occurrence count.
    - 2nd–3rd occurrences: suppress on `TRANSLATE`, strip explanatory note on `GLOSSARY_ONLY`.
    - 4th+ occurrences: suppress until a gap > 3 chapters since last shown.
    - Untranslated `GLOSSARY_ONLY` blocks or segments always retain at least one compact hint.
    """
    if not result.glossary_hints:
        return frozenset(), frozenset()
    evidences = _evidences(result)
    requires_hints = result.treatment.requires_glossary_hints or any(
        mode.requires_glossary_hints for mode in result.segment_treatments
    )
    only_dropped_by_index: dict[int, bool] = {}
    glossary_only_by_index: dict[int, bool] = {}
    bucket_by_index: dict[int, int | None] = {}
    for index, hint in enumerate(result.glossary_hints):
        only_dropped_by_index[index] = _hint_only_dropped(
            hint, result.block_id, evidences, dropped_difficulty_ids
        )
        hint_glossary_only = _hint_is_glossary_only(
            block, result, hint, requires_hints=requires_hints
        )
        glossary_only_by_index[index] = hint_glossary_only
        if hint_glossary_only:
            bucket_by_index[index] = _glossary_only_bucket(block, result, hint)

    is_a2 = isinstance(reader_level, str) and reader_level.strip().lower() == "a2"
    known = {} if plan is None else {item.id: item for item in plan.difficulties}
    pre_suppressed_a1 = (
        _a1_pre_suppressed_indices(
            block,
            result,
            requires_hints=requires_hints,
            evidences=evidences,
            known=known,
            dropped_difficulty_ids=dropped_difficulty_ids,
            only_dropped_by_index=only_dropped_by_index,
            glossary_only_by_index=glossary_only_by_index,
            bucket_by_index=bucket_by_index,
        )
        if is_a2
        else set()
    )

    suppressed: set[int] = set(pre_suppressed_a1)
    stripped: set[int] = set()
    glossary_only_indices_by_bucket: dict[int | None, set[int]] = {}
    glossary_only_rescue_by_bucket: dict[int | None, list[tuple[int, int, int, str]]] = {}
    for index, hint in enumerate(result.glossary_hints):
        if index in pre_suppressed_a1:
            continue
        only_dropped = only_dropped_by_index[index]
        if only_dropped and not requires_hints:
            continue
        hint_glossary_only = glossary_only_by_index[index]
        bucket = bucket_by_index.get(index)
        if hint_glossary_only:
            glossary_only_indices_by_bucket.setdefault(bucket, set()).add(index)
        key = vocabulary_dedup_key(hint.original)
        if not key:
            continue
        state = history.get(key)
        if state is None:
            history[key] = _HintSeen(
                count=1,
                last_shown_block=block_seq,
                last_shown_chapter=chapter_num,
                last_counted_block=block_seq,
            )
            continue
        if (chapter_num - state.last_shown_chapter) > _HINT_REINTRODUCE_CHAPTER_GAP:
            history[key] = _HintSeen(
                count=1,
                last_shown_block=block_seq,
                last_shown_chapter=chapter_num,
                last_counted_block=block_seq,
            )
            continue
        window_anchor = state.last_shown_block if hint_glossary_only else state.last_counted_block
        if (block_seq - window_anchor) <= _HINT_LOCAL_WINDOW_BLOCKS:
            suppressed.add(index)
            if hint_glossary_only:
                glossary_only_rescue_by_bucket.setdefault(bucket, []).append(
                    (1 if only_dropped else 0, state.count, index, key)
                )
            continue
        if state.count >= 3:
            suppressed.add(index)
            if hint_glossary_only:
                glossary_only_rescue_by_bucket.setdefault(bucket, []).append(
                    (1 if only_dropped else 0, state.count, index, key)
                )
            continue
        # 2nd or 3rd occurrence outside the 20-block local window.
        state.count += 1
        state.last_counted_block = block_seq
        if hint_glossary_only:
            stripped.add(index)
            state.last_shown_block = block_seq
            state.last_shown_chapter = chapter_num
        else:
            suppressed.add(index)
    for bucket, bucket_indices in glossary_only_indices_by_bucket.items():
        bucket_rescue = glossary_only_rescue_by_bucket.get(bucket)
        if bucket_indices and bucket_indices <= suppressed and bucket_rescue:
            _, _, rescue_index, rescue_key = min(bucket_rescue)
            suppressed.discard(rescue_index)
            stripped.add(rescue_index)
            state = history.get(rescue_key)
            if state is not None:
                state.last_shown_block = block_seq
                state.last_shown_chapter = chapter_num
                state.last_counted_block = block_seq
    return frozenset(suppressed), frozenset(stripped)


def _grammar_note_rule(note: LanguageNote) -> tuple[str, int, int] | None:
    """Classify a `LanguageNote` into one of the 11 closed grammar keys, if matched."""
    haystack = f"{note.original_excerpt or ''} {note.note}".translate(_SPEECH_STYLE_APOSTROPHES)
    for key, pattern, cap, min_gap in _GRAMMAR_PATTERN_RULES:
        if pattern.search(haystack) is not None:
            return key, cap, min_gap
    return None


_CULTURAL_OR_TERM_CAP = 1


def _is_cultural_note(
    note: LanguageNote,
    evidences: Sequence[HelpEvidence],
    known: Mapping[str, Difficulty],
) -> bool:
    if not known:
        return False
    ids = _ids_for(evidences, {_note_evidence_key(note)})
    return any(
        item_id in known and known[item_id].kind == "cultural_reference"
        for item_id in ids
    )


def _deduplicate_grammar_notes(
    result: BlockProcessingResult,
    *,
    dropped_difficulty_ids: frozenset[str],
    chapter_num: int,
    history: dict[str, _GrammarSeen],
    plan: ContentPlan | None = None,
) -> frozenset[int]:
    """Suppress repeated grammar, cultural-reference, and short-term `LanguageNote`s."""
    if not result.language_notes:
        return frozenset()
    evidences = _evidences(result)
    known = {} if plan is None else {item.id: item for item in plan.difficulties}
    suppressed: set[int] = set()
    for index, note in enumerate(result.language_notes):
        if _note_only_dropped(note, evidences, dropped_difficulty_ids):
            continue
        matched = _grammar_note_rule(note)
        if matched is not None:
            key, cap, min_gap = matched
        else:
            is_cultural = _is_cultural_note(note, evidences, known)
            words = expression_words(note.original_excerpt)
            if not is_cultural and not (0 < len(words) <= 4):
                continue
            term_key = vocabulary_dedup_key(note.original_excerpt)
            if not term_key:
                continue
            key, cap, min_gap = f"term:{term_key}", _CULTURAL_OR_TERM_CAP, 0
        state = history.get(key)
        if state is None:
            history[key] = _GrammarSeen(count=1, last_chapter=chapter_num)
            continue
        if state.count >= cap or (chapter_num - state.last_chapter) < min_gap:
            suppressed.add(index)
            continue
        state.count += 1
        state.last_chapter = chapter_num
    return frozenset(suppressed)


def _speech_style_lexemes(difficulty: Difficulty) -> frozenset[str]:
    """Normalized dialect lexemes/anchors for `difficulty`."""
    lexemes: set[str] = set()
    for anchor in difficulty.anchors:
        normalized = anchor.quote.translate(_SPEECH_STYLE_APOSTROPHES).casefold()
        tokens = _SPEECH_STYLE_TOKEN.findall(normalized)
        if not tokens:
            fallback = normalized.strip()
            if fallback:
                lexemes.add(fallback)
            continue
        significant: set[str] = set()
        for token in tokens:
            if token.endswith("'") or token.startswith("'"):
                significant.add(token)
                continue
            if "'" in token:
                base, _, suffix = token.partition("'")
                if suffix in _CONTRACTION_SUFFIXES:
                    if base and base not in _SPEECH_STYLE_STOPWORDS and len(base) >= 2:
                        significant.add(base)
                        significant.add(token)
                    continue
            if token not in _SPEECH_STYLE_STOPWORDS:
                significant.add(token)
        if significant:
            lexemes.update(significant)
        else:
            lexemes.add(" ".join(tokens))
    return frozenset(lexemes)


def _difficulty_scene_key(
    section_id: str,
    plan: ContentPlan,
    position: tuple[int, int, int, str],
    positions: Mapping[str, tuple[int, int]],
) -> tuple[str, str | None]:
    """Scene identifier containing `position`, or `(section_id, None)` without scenes."""
    if not plan.scenes:
        return (section_id, None)
    block_pos = (position[0], position[1])
    for scene in plan.scenes:
        start = positions.get(scene.start_block_id)
        end = positions.get(scene.end_block_id)
        if start is not None and end is not None and start <= block_pos <= end:
            return (section_id, scene.id)
    return (section_id, None)


def _repeated_speech_style_ids(
    document: CanonicalDocument,
    analyses: Sequence[ChapterAnalysis],
    facts: Sequence[ContextFact],
) -> dict[str, frozenset[str]]:
    """Repeated speech-style difficulties of one speaker, by owning section.

    The same source name is one speaker across chapters, whether the fact is
    explicit or inferred. The earliest anchor wins, not the difficulty id.
    The speaker's first 2 speech-style difficulties in their first scene/chapter
    are always kept, and any later difficulty whose dialect anchors/lexemes do
    not overlap already-shown ones is also kept.
    """
    positions = _block_positions(document)
    anchors_by_speaker: dict[str, list[SourceAnchor]] = {}
    for fact in facts:
        if not isinstance(fact.value, str):
            continue
        anchors_by_speaker.setdefault(glossary_key(fact.value), []).extend(fact.anchors)
    grouped: dict[
        str,
        list[tuple[tuple[int, int, int, str], str, Difficulty, tuple[str, str | None]]],
    ] = {}
    for analysis in analyses:
        plan = analysis.content_plan
        if plan is None:
            continue
        for difficulty in plan.difficulties:
            if difficulty.kind != "speech_style":
                continue
            position = _appearance(difficulty, positions)
            scene_key = _difficulty_scene_key(analysis.section_id, plan, position, positions)
            for key, anchors in anchors_by_speaker.items():
                if not _overlaps(difficulty.anchors, anchors):
                    continue
                grouped.setdefault(key, []).append(
                    (position, analysis.section_id, difficulty, scene_key)
                )
    kept: set[tuple[str, str]] = set()
    later: set[tuple[str, str]] = set()
    for items in grouped.values():
        items.sort(key=lambda entry: entry[0])
        first_scene = items[0][3]
        first_scene_kept = 0
        shown_lexemes: set[str] = set()
        for _, section_id, difficulty, scene_key in items:
            lexemes = _speech_style_lexemes(difficulty)
            if scene_key == first_scene and first_scene_kept < _SPEECH_STYLE_FIRST_SCENE_KEEP:
                kept.add((section_id, difficulty.id))
                first_scene_kept += 1
                shown_lexemes.update(lexemes)
            elif lexemes and not (lexemes & shown_lexemes):
                kept.add((section_id, difficulty.id))
                shown_lexemes.update(lexemes)
            else:
                later.add((section_id, difficulty.id))
    dropped: dict[str, set[str]] = {}
    for section_id, difficulty_id in later - kept:
        dropped.setdefault(section_id, set()).add(difficulty_id)
    return {section_id: frozenset(ids) for section_id, ids in dropped.items()}


def _block_positions(document: CanonicalDocument) -> dict[str, tuple[int, int]]:
    """Section index and block index for every block, in reading order."""
    positions: dict[str, tuple[int, int]] = {}
    for section_index, section in enumerate(document.ordered_sections):
        for block_index, block in enumerate(section.ordered_blocks):
            positions[block.id] = (section_index, block_index)
    return positions


def _appearance(
    difficulty: Difficulty,
    positions: Mapping[str, tuple[int, int]],
) -> tuple[int, int, int, str]:
    """Earliest anchor of a difficulty. An anchor outside the document sorts last."""
    ranked: list[tuple[int, int, int]] = []
    for anchor in difficulty.anchors:
        located = positions.get(anchor.block_id)
        if located is None:
            continue
        ranked.append((located[0], located[1], anchor.start))
    if not ranked:
        return (10**9, 10**9, 10**9, difficulty.id)
    section_index, block_index, start = min(ranked)
    return (section_index, block_index, start, difficulty.id)


def _overlaps(
    left: Sequence[SourceAnchor],
    right: Sequence[SourceAnchor],
) -> bool:
    """Whether two half-open anchor ranges share a block and an offset."""
    for anchor in left:
        for other in right:
            if (
                anchor.block_id == other.block_id
                and anchor.start < other.end
                and other.start < anchor.end
            ):
                return True
    return False


def _plan_without(plan: ContentPlan | None, dropped: frozenset[str]) -> ContentPlan | None:
    """Copy of `plan` without the dropped difficulties. The stored plan is not changed.

    Evidence that pointed at a dropped difficulty is left off the copy so the
    plan stays valid. Other difficulties and the speaker facts stay.
    """
    if plan is None or not dropped:
        return plan
    kept = tuple(item for item in plan.difficulties if item.id not in dropped)
    if len(kept) == len(plan.difficulties):
        return plan
    evidence = tuple(
        item for item in plan.before_you_read_evidence if item.difficulty_id not in dropped
    )
    return replace(plan, difficulties=kept, before_you_read_evidence=evidence)


def _features(
    document: CanonicalDocument,
    chapters: Mapping[str, ChapterLayout],
    analyses: Sequence[ChapterAnalysis],
    *,
    reader_level: str | None,
    english_help: bool,
) -> LayoutFeatures:
    """Layers and section keys that the finished chapters actually contain.

    `previously` is set only when there are at least 2 content chapters and at
    least one non-last content chapter has a non-empty `chapter_summary`.
    Chapter who-is-who lists are empty, so that section key is not set. Before
    You Read is a section key when a content chapter's analysis has entries,
    except on A2 without English help. Labels are the strings written on notes,
    not kind ids that lost their label. `has_notes` is true only when some note
    has a number. An unnumbered «Говорит …» note sets `has_speakers` and does
    not turn on the numbered-notes line.
    """
    units = [
        unit
        for chapter in chapters.values()
        for grouped in chapter.units.values()
        for unit in grouped
    ]
    notes = [item.note for unit in units for item in unit.notes]
    ordered_content = sorted(
        (chapter for chapter in chapters.values() if chapter.number is not None),
        key=lambda chapter: chapter.number or 0,
    )
    content_ids = {chapter.section_id for chapter in ordered_content}
    sections: set[str] = set()
    if len(ordered_content) >= 2:
        earlier_content_ids = {chapter.section_id for chapter in ordered_content[:-1]}
        if any(
            analysis.section_id in earlier_content_ids
            and isinstance(analysis.chapter_summary, str)
            and analysis.chapter_summary.strip()
            for analysis in analyses
        ):
            sections.add("previously")
    if any(chapter.who_is_who for chapter in chapters.values()):
        sections.add("who_is_who")
    if any(chapter.review for chapter in chapters.values()):
        sections.add("review")
    if (reader_level != "a2" or english_help) and any(
        analysis.before_you_read and analysis.section_id in content_ids for analysis in analyses
    ):
        sections.add("before_you_read")
    content_sections = tuple(
        section
        for section in document.ordered_sections
        if not document.is_front_matter(section) and not section.is_notes_only
    )
    is_a2_translation_book = (
        isinstance(reader_level, str)
        and reader_level.strip().lower() == "a2"
        and not english_help
    )
    words_before_last = (
        sum(
            len(_WORD_TOKEN_RE.findall(block.text))
            for section in content_sections[:-1]
            for block in textual_blocks(section)
        )
        if is_a2_translation_book and len(content_sections) >= 2
        else 0
    )
    fades_translation = (
        is_a2_translation_book
        and len(content_sections) >= 2
        and words_before_last >= _FADING_PROGRESSION_WORDS_THRESHOLD
    )
    return LayoutFeatures(
        has_translation=any(unit.translation for unit in units),
        has_notes=any(item.number is not None for unit in units for item in unit.notes),
        labels=frozenset(label for note in notes for label in note.labels),
        has_ipa=any(note.ipa for note in notes),
        has_poems=any(unit.lines for unit in units),
        has_speakers=any(
            item.note.origin == "speaker" for unit in units for item in unit.notes
        ),
        sections=frozenset(sections),
        fades_translation=fades_translation,
    )


def _ipa_strings(chapters: Mapping[str, ChapterLayout]) -> tuple[str, ...]:
    """IPA strings carried by notes, in reading order."""
    return tuple(
        item.note.ipa
        for chapter in chapters.values()
        for grouped in chapter.units.values()
        for unit in grouped
        for item in unit.notes
        if item.note.ipa
    )
