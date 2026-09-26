"""Стабильные идентификаторы доменных сущностей.

Идентификатор блока (PRD § 8) зависит только от положения блока в исходном документе:
`document_id`, пути к исходному ресурсу и структурного пути внутри него. Он не зависит
от текста блока, состава `TranslationUnit`, модели и номера запуска обработки, поэтому
повторный импорт той же книги даёт те же идентификаторы, а сохранённые результаты
остаются привязанными к своим блокам.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import NAMESPACE_URL, UUID, uuid5

from linguary.domain.errors import IdentifierError

#: Пространство имён для идентификаторов блоков. Значение зафиксировано навсегда:
#: его изменение сделает идентификаторы всех ранее импортированных книг другими.
BLOCK_ID_NAMESPACE: UUID = uuid5(NAMESPACE_URL, "https://linguary.local/document-block")

#: Разделитель составляющих ключа. Перевод строки не встречается ни в пути к ресурсу,
#: ни в структурном пути, поэтому разные наборы составляющих не дают один ключ.
_KEY_SEPARATOR = "\n"


@dataclass(frozen=True, slots=True)
class BlockIdentity:
    """Составляющие стабильного идентификатора блока."""

    document_id: str
    source_resource_path: str
    structural_path: str

    def __post_init__(self) -> None:
        for name in ("document_id", "source_resource_path", "structural_path"):
            if not getattr(self, name).strip():
                raise IdentifierError(f"составляющая идентификатора блока пуста: {name}")

    @property
    def key(self) -> str:
        """Канонический ключ, из которого вычисляется идентификатор."""
        return _KEY_SEPARATOR.join(
            (self.document_id, self.source_resource_path, self.structural_path)
        )

    @property
    def block_id(self) -> str:
        """Стабильный идентификатор блока."""
        return str(uuid5(BLOCK_ID_NAMESPACE, self.key))


def build_block_id(
    *, document_id: str, source_resource_path: str, structural_path: str
) -> str:
    """Вычисляет стабильный идентификатор блока по его положению в документе."""
    return BlockIdentity(
        document_id=document_id,
        source_resource_path=source_resource_path,
        structural_path=structural_path,
    ).block_id


def build_language_note_id(*, chapter_number: int, ordinal: int) -> str:
    """Идентификатор учебной заметки вида `ln-ch03-001` (PRD § 19.3).

    Заметки живут в отдельном пространстве идентификаторов, чтобы не конфликтовать
    с оригинальными `footnote` id книги и не ломать её внутренние ссылки.
    """
    if chapter_number < 1:
        raise IdentifierError("номер главы для идентификатора заметки должен быть положительным")
    if ordinal < 1:
        raise IdentifierError("порядковый номер заметки должен быть положительным")
    return f"ln-ch{chapter_number:02d}-{ordinal:03d}"
