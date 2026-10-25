"""Immutable, versioned provenance of book work."""
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from uuid import UUID

from linguary.domain.errors import DomainError


class WorkStateError(DomainError):
    """Invalid work state or transition."""


class UnsupportedWorkVersion(WorkStateError):
    """A persisted work schema cannot be interpreted."""


def identifier(value: str) -> None:
    try:
        UUID(value)
    except (ValueError, TypeError, AttributeError):
        raise WorkStateError('некорректный UUID работы') from None


def nonnegative(value: int) -> None:
    if type(value) is not int or value < 0:
        raise WorkStateError('ожидается неотрицательное целое число')


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise WorkStateError('некорректное значение JSON')


def thaw(value):
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def version(value: int) -> None:
    if type(value) is not int or value != 1:
        raise UnsupportedWorkVersion('неподдерживаемая версия данных работы')


@dataclass(frozen=True, slots=True)
class BookConditions:
    id: str
    book_id: str
    revision: int
    payload: Mapping
    created_at: datetime
    reproducibility: str = 'complete'
    schema_version: int = 1

    def __post_init__(self):
        identifier(self.id)
        version(self.schema_version)
        nonnegative(self.revision)
        if self.revision == 0 or not self.book_id:
            raise WorkStateError('некорректная версия условий')
        if self.created_at.tzinfo is None or self.created_at.utcoffset() != UTC.utcoffset(None):
            raise WorkStateError('время условий должно быть UTC')
        if self.reproducibility not in {'complete', 'legacy_partial', 'unknown'}:
            raise WorkStateError('неизвестная воспроизводимость')
        version(self.payload.get('schema_version'))
        object.__setattr__(self, 'payload', freeze(self.payload))

    def to_dict(self):
        return {'id': self.id, 'book_id': self.book_id, 'revision': self.revision,
                'payload': thaw(self.payload), 'created_at': self.created_at.isoformat(),
                'reproducibility': self.reproducibility, 'schema_version': self.schema_version}

    @classmethod
    def from_dict(cls, value):
        version(value.get('schema_version'))
        return cls(**{**value, 'created_at': datetime.fromisoformat(value['created_at'])})


class WorkSerialization:
    """Versioned primitive representation shared by durable work entities."""
    __slots__ = ()

    def to_dict(self):
        return {name: thaw(getattr(self, name)) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value):
        version(value.get('schema_version'))
        return cls(**value)
