"""Stable identifiers for domain entities.

The block identifier (PRD § 8) depends solely on the block's position in the source document:
`document_id`, source resource path, and structural path within it. It does not depend
on the block text, `TranslationUnit` composition, model, or processing run number; therefore,
reimporting the same book yields the same identifiers, and saved results remain bound
to their blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import NAMESPACE_URL, UUID, uuid5

from linguary.domain.errors import IdentifierError

#: Namespace for block identifiers. Fixed permanently:
#: changing it would alter identifiers for all previously imported books.
BLOCK_ID_NAMESPACE: UUID = uuid5(NAMESPACE_URL, "https://linguary.local/document-block")

#: Separator for key components. Newline appears neither in resource paths nor
#: in structural paths, preventing distinct component tuples from producing identical keys.
_KEY_SEPARATOR = "\n"


@dataclass(frozen=True, slots=True)
class BlockIdentity:
    """Components of a stable block identifier."""

    document_id: str
    source_resource_path: str
    structural_path: str

    def __post_init__(self) -> None:
        for name in ("document_id", "source_resource_path", "structural_path"):
            if not getattr(self, name).strip():
                raise IdentifierError(f"составляющая идентификатора блока пуста: {name}")

    @property
    def key(self) -> str:
        """Canonical key from which the identifier is computed."""
        return _KEY_SEPARATOR.join(
            (self.document_id, self.source_resource_path, self.structural_path)
        )

    @property
    def block_id(self) -> str:
        """Stable block identifier."""
        return str(uuid5(BLOCK_ID_NAMESPACE, self.key))


def build_block_id(
    *, document_id: str, source_resource_path: str, structural_path: str
) -> str:
    """Computes a stable block identifier from its position within the document."""
    return BlockIdentity(
        document_id=document_id,
        source_resource_path=source_resource_path,
        structural_path=structural_path,
    ).block_id


def build_language_note_id(*, chapter_number: int, ordinal: int) -> str:
    """Learning note identifier of the form `ln-ch03-001` (PRD § 19.3).

    Notes reside in a separate identifier space to avoid colliding with original
    book `footnote` ids and breaking internal document links.
    """
    if chapter_number < 1:
        raise IdentifierError("номер главы для идентификатора заметки должен быть положительным")
    if ordinal < 1:
        raise IdentifierError("порядковый номер заметки должен быть положительным")
    return f"ln-ch{chapter_number:02d}-{ordinal:03d}"
