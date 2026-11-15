"""Who-is-who entries for one chapter and for the whole book.

The lists are a projection of glossary candidates. Nothing is stored and no
model is called. A content chapter is a section that is neither front matter
nor notes-only. Numbering of those chapters starts at 1.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from linguary.domain import (
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    GlossaryCandidate,
    TermConcept,
    glossary_key,
)
from linguary.domain.name_translation import NameTranslationMethod

NAME_METHOD_LABELS: dict[str, str] = {
    "sound": "по звучанию",
    "meaning": "по смыслу",
    "mixed": "звучание и смысл",
    "creative": "вольный перевод",
    "uncertain": "способ неясен",
}

_SHOWN_METHODS: frozenset[NameTranslationMethod] = frozenset()
_LISTED_TYPES = (ConceptType.PERSON, ConceptType.PLACE)
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_THE_FAMILY = re.compile(r"^(?:[Tt]he) [A-Z][A-Za-z]+s$")
_BARE_PLURAL = re.compile(r"^([A-Z][A-Za-z]+)s$")
_FAMILY_SURNAME = re.compile(r"surname of the .+ family", re.IGNORECASE)
_FAMILY_TRANSLATIONS = frozenset({"поттеры", "дерсли"})
_SURNAME_TITLES = frozenset({"mr", "mrs", "miss", "ms"})
_GIVEN_TITLES = frozenset({"uncle", "aunt"})
_TITLE_GENDER = {
    "mr": "m",
    "mrs": "f",
    "miss": "f",
    "ms": "f",
    "uncle": "m",
    "aunt": "f",
    "sir": "m",
    "lord": "m",
    "madam": "f",
}
_ROLE_MARKERS = (
    ("жена мистера дёрсли", "f", "wife:dursley"),
    ("сестра лили", "f", "sister:lily"),
    ("мать гарри", "f", "mother:harry"),
)
_NON_CHARACTER = (
    "неуверенно называет",
    "пытается вспомнить имя",
    "ошибочно называет",
    "распространённое мужское имя",
    "телеведущий",
    "ведущий новостей",
    "диктор",
    "синоптик",
    "кличка кошки",
)


@dataclass(frozen=True, slots=True)
class WhoEntry:
    """One person or place, from the chapter where the key is first seen.

    `name_method` is always none. `NAME_METHOD_LABELS` still names every
    method, but `_SHOWN_METHODS` is empty, so no method is written on the entry.
    """

    source: str
    translation: str
    type: ConceptType
    description: str
    name_method: str | None
    first_chapter: int


def chapter_who_is_who(
    document: CanonicalDocument,
    analyses: Sequence[ChapterAnalysis],
    concepts: Sequence[TermConcept],
    *,
    section_id: str,
) -> tuple[WhoEntry, ...]:
    """People and places whose key is first seen in this content chapter.

    Persons come first, then places. Each group keeps candidate order.
    A key seen in an earlier content chapter is absent. Other concept types
    are absent. Front matter and notes-only sections return an empty tuple.
    """
    document.section(section_id)
    chapter = _content_chapter_numbers(document).get(section_id)
    if chapter is None:
        return ()
    mentions = tuple(item for item in _mentions(document, analyses) if item[0] == chapter)
    return _entries(mentions, concepts, document.target_language)


def book_who_is_who(
    document: CanonicalDocument,
    analyses: Sequence[ChapterAnalysis],
    concepts: Sequence[TermConcept],
) -> tuple[WhoEntry, ...]:
    """One entry per key, from its first content chapter in the book."""
    return _entries(_mentions(document, analyses), concepts, document.target_language)


def _content_chapter_numbers(document: CanonicalDocument) -> dict[str, int]:
    """Content-chapter numbers by section id. The first content section is 1."""
    numbers: dict[str, int] = {}
    number = 0
    for section in document.ordered_sections:
        if document.is_front_matter(section) or section.is_notes_only:
            continue
        number += 1
        numbers[section.id] = number
    return numbers


def _mentions(
    document: CanonicalDocument,
    analyses: Sequence[ChapterAnalysis],
) -> tuple[tuple[int, GlossaryCandidate], ...]:
    """First person or place candidate of each key, in document order."""
    by_section: dict[str, list[GlossaryCandidate]] = {}
    for analysis in analyses:
        by_section.setdefault(analysis.section_id, []).extend(analysis.glossary_candidates)
    numbers = _content_chapter_numbers(document)
    seen: set[str] = set()
    found: list[tuple[int, GlossaryCandidate]] = []
    for section in document.ordered_sections:
        chapter = numbers.get(section.id)
        if chapter is None:
            continue
        for candidate in by_section.get(section.id, ()):
            if candidate.type not in _LISTED_TYPES:
                continue
            key = glossary_key(candidate.source_text)
            if key in seen:
                continue
            seen.add(key)
            found.append((chapter, candidate))
    return tuple(found)


def _entries(
    mentions: Sequence[tuple[int, GlossaryCandidate]],
    concepts: Sequence[TermConcept],
    language: str,
) -> tuple[WhoEntry, ...]:
    """Persons, then places. Families and non-characters are left out.

    Person rows that name one character are one entry: the fullest source,
    the earliest chapter, and the first description that is already present.
    """
    person_rows = [
        (chapter, candidate)
        for chapter, candidate in mentions
        if candidate.type == ConceptType.PERSON
    ]
    people = [candidate for _chapter, candidate in person_rows]
    kept = [
        (chapter, candidate)
        for chapter, candidate in person_rows
        if not _excluded_person(candidate, people, concepts, language)
    ]
    persons = _dedupe_persons([
        _person_ref(
            _make_entry(chapter, candidate, concepts, language),
            index,
            candidate.description,
        )
        for index, (chapter, candidate) in enumerate(kept)
    ])
    places = tuple(
        _make_entry(chapter, candidate, concepts, language)
        for chapter, candidate in mentions
        if candidate.type == ConceptType.PLACE
    )
    return persons + places


def _make_entry(
    chapter: int,
    candidate: GlossaryCandidate,
    concepts: Sequence[TermConcept],
    language: str,
) -> WhoEntry:
    return WhoEntry(
        source=candidate.source_text,
        translation=_translation(candidate, concepts, language),
        type=candidate.type,
        description=_target_description(candidate.description, language),
        name_method=_name_method(candidate),
        first_chapter=chapter,
    )


@dataclass(frozen=True, slots=True)
class _PersonRef:
    """One person mention plus the name parts used to merge it."""

    entry: WhoEntry
    index: int
    outside: str
    inside: str
    given: str
    surname: str
    gender: str
    alias_given: str
    alias_surname: str
    title_surnames: tuple[tuple[str, str], ...]
    roles: frozenset[str]


def _excluded_person(
    candidate: GlossaryCandidate,
    people: Sequence[GlossaryCandidate],
    concepts: Sequence[TermConcept],
    language: str,
) -> bool:
    """Plural families, a bare family surname, and descriptions that are not characters."""
    if _marked_non_character(candidate.description):
        return True
    shown = _translation(candidate, concepts, language)
    if _family_translation(candidate.suggested_translation) or _family_translation(shown):
        return True
    source = candidate.source_text.strip()
    if _THE_FAMILY.fullmatch(source):
        return True
    bare = _BARE_PLURAL.fullmatch(source)
    if bare is not None and _stem_is_family_surname(bare.group(1), candidate, people):
        return True
    return _bare_family_surname(candidate, people)


def _family_translation(translation: str) -> bool:
    key = glossary_key(translation)
    return key in _FAMILY_TRANSLATIONS or key.startswith(("семья ", "супруги "))


def _marked_non_character(description: str) -> bool:
    text = glossary_key(description)
    return any(glossary_key(marker) in text for marker in _NON_CHARACTER)


def _stem_is_family_surname(
    stem: str,
    candidate: GlossaryCandidate,
    people: Sequence[GlossaryCandidate],
) -> bool:
    key = glossary_key(stem)
    return any(
        other is not candidate and key in _surnames_of(other)
        for other in people
    )


def _bare_family_surname(
    candidate: GlossaryCandidate,
    people: Sequence[GlossaryCandidate],
) -> bool:
    """`Dursley` with a family-surname description, once a member is already listed."""
    source = candidate.source_text.strip()
    if " " in source or "(" in source or not source[:1].isupper():
        return False
    if not _family_surname_description(candidate.description):
        return False
    key = glossary_key(source)
    return any(other is not candidate and _mentions_surname(other, key) for other in people)


def _family_surname_description(description: str) -> bool:
    if "фамилия семьи" in glossary_key(description):
        return True
    return _FAMILY_SURNAME.search(description) is not None


def _mentions_surname(candidate: GlossaryCandidate, surname: str) -> bool:
    if surname in _surnames_of(candidate):
        return True
    return surname in glossary_key(candidate.source_text).split() or (
        surname in glossary_key(candidate.description).split()
    )


def _surnames_of(candidate: GlossaryCandidate) -> set[str]:
    outside, inside = _split_called(candidate.source_text)
    found: set[str] = set()
    for part in (outside, inside):
        _title, _given, surname, _gender = _parse_called(part)
        if surname:
            found.add(glossary_key(surname))
    return found


def _person_ref(entry: WhoEntry, index: int, description: str) -> _PersonRef:
    outside, inside = _split_called(entry.source)
    _title, given, surname, gender = _parse_called(outside)
    _alias_title, alias_given, alias_surname, alias_gender = _parse_called(inside)
    roles, described_gender = _roles(description)
    title_surnames = _title_surnames(outside) + _title_surnames(inside)
    return _PersonRef(
        entry=entry,
        index=index,
        outside=outside,
        inside=inside,
        given=given,
        surname=surname,
        gender=gender or alias_gender or described_gender,
        alias_given=alias_given,
        alias_surname=alias_surname,
        title_surnames=title_surnames,
        roles=roles,
    )


def _dedupe_persons(refs: list[_PersonRef]) -> tuple[WhoEntry, ...]:
    """Merge one character. Different people who only share a surname stay apart."""
    parent = list(range(len(refs)))
    # A bare given name may join one surnamed component. It must not bridge two.
    surnames = [_ref_surnames(ref) for ref in refs]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        left_surnames = surnames[left_root]
        right_surnames = surnames[right_root]
        if left_surnames and right_surnames and left_surnames.isdisjoint(right_surnames):
            return
        parent[right_root] = left_root
        left_surnames |= right_surnames

    for left in range(len(refs)):
        for right in range(left + 1, len(refs)):
            if _same_person(refs[left], refs[right]):
                union(left, right)
    _union_unique_titles(refs, union)
    _union_roles(refs, union)
    groups: dict[int, list[_PersonRef]] = {}
    order: list[int] = []
    for index, ref in enumerate(refs):
        root = find(index)
        if root not in groups:
            groups[root] = []
            order.append(root)
        groups[root].append(ref)
    return tuple(_merge_person(groups[root]) for root in order)


def _same_person(left: _PersonRef, right: _PersonRef) -> bool:
    if _source_keys(left) & _source_keys(right):
        return True
    shared = _given_names(left) & _given_names(right)
    if not shared:
        return False
    left_surnames = _ref_surnames(left)
    right_surnames = _ref_surnames(right)
    return not left_surnames or not right_surnames or not left_surnames.isdisjoint(right_surnames)


def _union_unique_titles(refs: list[_PersonRef], union) -> None:
    """`Mrs Potter` joins the only woman with that surname, not every Potter."""
    for index, ref in enumerate(refs):
        for gender, surname in ref.title_surnames:
            known: list[int] = []
            matches: list[int] = []
            for other_index, other in enumerate(refs):
                if other_index == index or not other.given or not other.surname:
                    continue
                if glossary_key(other.surname) != surname:
                    continue
                if other.gender == gender:
                    known.append(other_index)
                    matches.append(other_index)
                elif not other.gender:
                    matches.append(other_index)
            if len(known) == 1 and len(matches) == 1:
                union(index, known[0])


def _union_roles(refs: list[_PersonRef], union) -> None:
    for left in range(len(refs)):
        for right in range(left + 1, len(refs)):
            if refs[left].roles & refs[right].roles or _role_names_title(refs[left], refs[right]):
                union(left, right)


def _role_names_title(left: _PersonRef, right: _PersonRef) -> bool:
    return "wife:dursley" in left.roles and "mrs dursley" in _source_keys(right) or (
        "wife:dursley" in right.roles and "mrs dursley" in _source_keys(left)
    )


def _merge_person(group: list[_PersonRef]) -> WhoEntry:
    winner = max(group, key=_completeness)
    ordered = sorted(group, key=lambda item: (item.entry.first_chapter, item.index))
    earliest = ordered[0]
    if earliest.entry.description.strip():
        description = earliest.entry.description
    else:
        description = next(
            (item.entry.description for item in ordered if item.entry.description.strip()),
            "",
        )
    return WhoEntry(
        source=winner.entry.source,
        translation=winner.entry.translation,
        type=ConceptType.PERSON,
        description=description,
        name_method=winner.entry.name_method,
        first_chapter=min(item.entry.first_chapter for item in group),
    )


def _completeness(ref: _PersonRef) -> tuple[int, int]:
    score = 0
    if ref.given and ref.surname:
        score += 4
    if ref.inside:
        score += 3
    elif ref.surname and not ref.given:
        score += 2
    elif ref.given:
        score += 1
    return (score, len(ref.entry.source))


def _source_keys(ref: _PersonRef) -> set[str]:
    keys = {glossary_key(ref.outside)}
    if ref.inside:
        keys.add(glossary_key(ref.inside))
    return {key for key in keys if key}


def _given_names(ref: _PersonRef) -> set[str]:
    names = {glossary_key(ref.given), glossary_key(ref.alias_given)}
    return {name for name in names if name}


def _ref_surnames(ref: _PersonRef) -> set[str]:
    names = {glossary_key(ref.surname), glossary_key(ref.alias_surname)}
    return {name for name in names if name}


def _split_called(source: str) -> tuple[str, str]:
    text = source.strip()
    if "(" not in text or ")" not in text:
        return text, ""
    before, _, rest = text.partition("(")
    inside, _, _ = rest.partition(")")
    return before.strip(), inside.strip()


def _parse_called(text: str) -> tuple[str, str, str, str]:
    """Title, given name, surname, and gender. Empty strings when that part is absent."""
    tokens = [token for token in text.replace(".", " ").split() if token]
    if not tokens:
        return "", "", "", ""
    first = tokens[0].casefold()
    gender = _TITLE_GENDER.get(first, "")
    if first in _TITLE_GENDER and len(tokens) >= 2:
        rest = tokens[1:]
        if first in _GIVEN_TITLES or (first not in _SURNAME_TITLES and len(rest) == 1):
            return first, " ".join(rest), "", gender
        if first in _SURNAME_TITLES:
            return first, "", " ".join(rest), gender
        return first, rest[0], " ".join(rest[1:]), gender
    if len(tokens) == 1:
        return "", tokens[0], "", ""
    return "", tokens[0], " ".join(tokens[1:]), ""


def _title_surnames(text: str) -> tuple[tuple[str, str], ...]:
    title, _given, surname, gender = _parse_called(text)
    if title not in _SURNAME_TITLES or not surname or not gender:
        return ()
    return ((gender, glossary_key(surname)),)


def _roles(description: str) -> tuple[frozenset[str], str]:
    text = glossary_key(description)
    roles: list[str] = []
    gender = ""
    for marker, marker_gender, role in _ROLE_MARKERS:
        if glossary_key(marker) in text:
            roles.append(role)
            gender = gender or marker_gender
    return frozenset(roles), gender


def _target_description(description: str, language: str) -> str:
    """Strip surrounding space. A Russian book keeps the text only with Cyrillic."""
    cleaned = description.strip()
    if language.strip().lower() == "ru" and _CYRILLIC.search(cleaned) is None:
        return ""
    return cleaned


def _translation(
    candidate: GlossaryCandidate,
    concepts: Sequence[TermConcept],
    language: str,
) -> str:
    """User decision for this key, otherwise the first candidate's suggestion."""
    key = glossary_key(candidate.source_text)
    for concept in concepts:
        if glossary_key(concept.canonical_source_text) != key:
            continue
        variant = concept.variant_for(language)
        if variant is not None and variant.status.is_user_decision:
            return variant.text
    return candidate.suggested_translation


def _name_method(candidate: GlossaryCandidate) -> str | None:
    """No label while `_SHOWN_METHODS` is empty.

    Several observations can share the highest confidence. The first of them
    would be used. Another method, or an observation with no assessment, stays
    unlabeled. The label texts remain in `NAME_METHOD_LABELS`.
    """
    evidence = candidate.name_translation_evidence
    if evidence is None:
        return None
    observation = max(evidence.observations, key=lambda item: item.confidence)
    assessment = observation.assessment
    if assessment is None or assessment.applied_method not in _SHOWN_METHODS:
        return None
    return NAME_METHOD_LABELS[assessment.applied_method.value]
