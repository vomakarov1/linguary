"""Учебные материалы: `Before You Read`, сводка главы и `Language Notes`.

`LearningMaterial` — доменное представление данных, которые физически лежат в анализе
главы и в заметках блоков; собственной таблицы у него нет (PRD § 19, SDD SQL-01).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from linguary.domain.errors import DomainError


@dataclass(frozen=True, slots=True)
class BeforeYouReadEntry:
    """Выражение, для которого обычного словаря может быть недостаточно (PRD § 19.1)."""

    expression: str
    explanation: str

    def __post_init__(self) -> None:
        if not self.expression.strip():
            raise DomainError("выражение материала Before You Read пусто")
        if not self.explanation.strip():
            raise DomainError(f"объяснение выражения {self.expression!r} пусто")


@dataclass(frozen=True, slots=True)
class LanguageNote:
    """Языковой комментарий, привязанный ровно к одному блоку (PRD § 19.3).

    Групповые заметки в MVP не поддерживаются: если объяснение полезно нескольким
    соседним блокам, оно привязывается к наиболее релевантному из них.
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
    """Учебные материалы одной главы в виде, нужном экспортёру и интерфейсу."""

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
        """Заметки, которые размещаются сразу после указанного блока."""
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
        """Собирает материалы главы из результатов анализа и обработки блоков."""
        return cls(
            section_id=section_id,
            before_you_read=tuple(before_you_read),
            chapter_summary=chapter_summary,
            language_notes=tuple(language_notes),
        )
