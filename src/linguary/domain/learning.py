"""Educational materials: `Before You Read`, chapter summary, and `Language Notes`.

`LearningMaterial` is a domain representation of data physically residing in chapter analysis
and block notes; it has no dedicated table of its own (PRD § 19, SDD SQL-01).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from linguary.domain.errors import DomainError
from linguary.domain.content_plan import SourceAnchor


@dataclass(frozen=True, slots=True)
class BeforeYouReadEntry:
    """Expression for which an ordinary dictionary may be insufficient (PRD § 19.1)."""

    expression: str
    explanation: str

    def __post_init__(self) -> None:
        if not self.expression.strip():
            raise DomainError("выражение материала Before You Read пусто")
        if not self.explanation.strip():
            raise DomainError(f"объяснение выражения {self.expression!r} пусто")


@dataclass(frozen=True, slots=True)
class LanguageNote:
    """Language note bound to exactly one block (PRD § 19.3).

    Group notes are not supported in MVP: if an explanation is relevant to multiple
    neighboring blocks, it is attached to the most relevant one.
    """

    id: str
    block_id: str
    original_excerpt: str
    note: str
    translated_excerpt: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise DomainError("идентификатор учебной заметки пуст")
        if not self.block_id.strip():
            raise DomainError("учебная заметка не привязана к блоку")
        if not self.original_excerpt.strip():
            raise DomainError(f"фрагмент оригинала заметки {self.id} пуст")
        if not self.note.strip():
            raise DomainError(f"текст заметки {self.id} пуст")


@dataclass(frozen=True, slots=True)
class LearningMaterial:
    """Educational materials for a single chapter in the form required by the exporter and UI."""

    section_id: str
    before_you_read: tuple[BeforeYouReadEntry, ...] = ()
    chapter_summary: str = ""
    language_notes: tuple[LanguageNote, ...] = ()

    def __post_init__(self) -> None:
        if not self.section_id.strip():
            raise DomainError("учебные материалы не привязаны к главе")
        note_ids = [note.id for note in self.language_notes]
        if len(set(note_ids)) != len(note_ids):
            raise DomainError(f"идентификаторы заметок главы {self.section_id} повторяются")

    def notes_for(self, block_id: str) -> tuple[LanguageNote, ...]:
        """Notes placed immediately after the specified block."""
        return tuple(note for note in self.language_notes if note.block_id == block_id)

    @classmethod
    def for_chapter(
        cls,
        *,
        section_id: str,
        before_you_read: Iterable[BeforeYouReadEntry],
        chapter_summary: str,
        language_notes: Iterable[LanguageNote],
    ) -> LearningMaterial:
        """Assembles chapter materials from analysis results and block processing results."""
        return cls(
            section_id=section_id,
            before_you_read=tuple(before_you_read),
            chapter_summary=chapter_summary,
            language_notes=tuple(language_notes),
        )


def validate_assistance_body(kind, text):
    if kind not in {'lexical_hint', 'sentence_rephrase', 'russian_explanation', 'unresolved'}:
        raise DomainError('неизвестный вид помощи')
    if (kind == 'unresolved' and text is not None) or (kind != 'unresolved' and (not isinstance(text, str) or not text.strip())):
        raise DomainError('вид помощи не соответствует её тексту')


@dataclass(frozen=True, slots=True)
class FragmentAssistance:
    difficulty_id: str
    point_ids: tuple[str, ...]
    kind: str
    text: str | None
    anchors: tuple[SourceAnchor, ...]

    def __post_init__(self):
        object.__setattr__(self, 'point_ids', tuple(self.point_ids))
        object.__setattr__(self, 'anchors', tuple(self.anchors))
        validate_assistance_body(self.kind, self.text)
        if not self.difficulty_id.strip() or not self.anchors or not self.point_ids or len(set(self.point_ids)) != len(self.point_ids):
            raise DomainError('помощь требует трудность, точные якоря и смысловые пункты')

    @property
    def resolved(self):
        return self.kind != 'unresolved'
