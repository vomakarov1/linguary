"""Who-is-who entries for one chapter and for the whole book.

The lists are a projection of glossary candidates. Nothing is stored and no
model is called. A content chapter is a section that is neither front matter
nor notes-only. Numbering of those chapters starts at 1.
"""

from __future__ import annotations

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

_SHOWN_METHODS = frozenset({NameTranslationMethod.MIXED, NameTranslationMethod.CREATIVE})
_LISTED_TYPES = (ConceptType.PERSON, ConceptType.PLACE)


@dataclass(frozen=True, slots=True)
class WhoEntry:
    """One person or place, from the chapter where the key is first seen.

    `name_method` is set only for mixed and creative translations. Other
    methods stay in `NAME_METHOD_LABELS` but are not written on the entry.
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
    """Persons, then places. Order inside each group follows `mentions`."""
    entries = tuple(
        WhoEntry(
            source=candidate.source_text,
            translation=_translation(candidate, concepts, language),
            type=candidate.type,
            description=candidate.description,
            name_method=_name_method(candidate),
            first_chapter=chapter,
        )
        for chapter, candidate in mentions
    )
    persons = tuple(entry for entry in entries if entry.type == ConceptType.PERSON)
    places = tuple(entry for entry in entries if entry.type == ConceptType.PLACE)
    return persons + places


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
    """Label for the first assessment, and only when it is mixed or creative."""
    evidence = candidate.name_translation_evidence
    if evidence is None:
        return None
    for observation in evidence.observations:
        assessment = observation.assessment
        if assessment is None:
            continue
        method = assessment.applied_method
        if method not in _SHOWN_METHODS:
            return None
        return NAME_METHOD_LABELS[method.value]
    return None
