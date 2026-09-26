"""Матрица поддержки конструкций EPUB (PRD § 6).

Импортёр обязан явно ответить, что он делает с каждой конструкцией книги:

- `SUPPORTED` — конструкция разобрана и полностью поддержана;
- `PRESERVE_OPAQUE` — конструкция переносится в итоговый EPUB без внутренней обработки;
- `DROP_WITH_WARNING` — конструкция пропущена, пользователь видит предупреждение;
- `FATAL` — книгу импортировать нельзя; такой случай поднимает `EpubImportFatalError`
  и до классификации отдельной конструкции не доходит.

Модуль знает только доменные типы: ни lxml, ни EbookLib, ни файловой системы здесь нет,
поэтому решения матрицы одинаковы для читателя книги и для сохранения ресурсов.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from linguary.domain import (
    BlockType,
    BookImportWarning,
    ImportWarningKind,
    SupportStatus,
)

#: Типы блоков, которые переносятся в итоговый EPUB без внутренней обработки: сложные
#: таблицы, нестандартные контейнеры и декоративные конструкции (PRD § 6).
_PRESERVE_OPAQUE_BLOCK_TYPES = frozenset(
    {BlockType.TABLE, BlockType.OTHER, BlockType.PAGE_BREAK}
)

#: Типы содержимого ресурсов, которые в итоговой книге не нужны: сценарии не влияют
#: на чтение, а переносить исполняемый код в учебный EPUB незачем.
_DROPPED_MEDIA_TYPES = frozenset(
    {
        "application/javascript",
        "application/ecmascript",
        "application/x-javascript",
        "text/javascript",
        "text/ecmascript",
    }
)

#: Число примеров структурных путей в тексте предупреждения об отброшенных блоках.
_EXAMPLES_IN_WARNING = 3


class DropReason(StrEnum):
    """Причина, по которой конструкция книги не попала в каноническую модель.

    Список закрыт границами PRD § 6: пропускать разрешено пустой мусор, дублирующие
    разрывы страницы и неподдержанные декоративные элементы, не влияющие на чтение.
    """

    EMPTY_DECORATION = "empty_decoration"
    DUPLICATE_PAGE_BREAK = "duplicate_page_break"

    @property
    def description(self) -> str:
        """Человекочитаемое описание причины для предупреждения импорта."""
        return _DROP_REASON_DESCRIPTIONS[self]


_DROP_REASON_DESCRIPTIONS = {
    DropReason.EMPTY_DECORATION: (
        "пустые декоративные элементы без текста и внутренних ссылок"
    ),
    DropReason.DUPLICATE_PAGE_BREAK: "повторные разрывы страницы печатного оригинала",
}


def block_support_status(block_type: BlockType) -> SupportStatus:
    """Статус поддержки блока по его типу.

    Абзацы, заголовки, цитаты, пункты списка, изображения, сноски и стихи поддержаны
    полностью; таблица, разрыв страницы и нераспознанная конструкция сохраняются целиком.
    """
    return (
        SupportStatus.PRESERVE_OPAQUE
        if block_type in _PRESERVE_OPAQUE_BLOCK_TYPES
        else SupportStatus.SUPPORTED
    )


def resource_support_status(media_type: str) -> SupportStatus:
    """Статус поддержки ресурса книги по типу его содержимого.

    Изображения поддержаны полностью: экспортёр переносит их вместе с блоками `Image`.
    Стили, шрифты, мультимедиа и незнакомые файлы переносятся как есть — разбирать их
    импортёру незачем, а потерять оформление книги нельзя.
    """
    normalized = media_type.strip().lower().partition(";")[0].strip()
    if normalized in _DROPPED_MEDIA_TYPES:
        return SupportStatus.DROP_WITH_WARNING
    if normalized.startswith("image/"):
        return SupportStatus.SUPPORTED
    return SupportStatus.PRESERVE_OPAQUE


def dropped_resource_warning(*, source_path: str, media_type: str) -> BookImportWarning:
    """Предупреждение о ресурсе, который не попал в книгу (`DROP_WITH_WARNING`)."""
    return BookImportWarning(
        kind=ImportWarningKind.UNSUPPORTED_CONSTRUCT,
        message=(
            f"ресурс {source_path} типа {media_type} не переносится в книгу: "
            f"он не влияет на чтение и не поддержан приложением"
        ),
    )


@dataclass(slots=True)
class DroppedBlockLog:
    """Отброшенные блоки одного документа и предупреждения о них.

    `DROP_WITH_WARNING` обязан оставить предупреждение, видимое пользователю (PRD § 6),
    но отдельная строка на каждый пустой `<span>` превращает список предупреждений
    в шум: в изданиях Gutenberg таких элементов сотни. Поэтому отбросы копятся по
    причинам, а пользователь получает по одному предупреждению на причину — с числом
    пропущенных конструкций и примерами их положения в исходном документе.
    """

    resource_path: str
    section_id: str
    _paths: dict[DropReason, list[str]] = field(default_factory=dict)

    def record(self, reason: DropReason, structural_path: str) -> None:
        """Запоминает пропущенную конструкцию и её положение в документе."""
        self._paths.setdefault(reason, []).append(structural_path)

    def warnings(self) -> tuple[BookImportWarning, ...]:
        """Предупреждения импорта по накопленным отбросам."""
        return tuple(
            BookImportWarning(
                kind=ImportWarningKind.DROPPED_BLOCK,
                message=(
                    f"в документе {self.resource_path} пропущены "
                    f"{reason.description} — {len(paths)}. {_examples(paths)}"
                ),
                section_id=self.section_id,
            )
            for reason, paths in self._paths.items()
        )


def _examples(paths: Sequence[str]) -> str:
    """Примеры положения пропущенных конструкций в исходном документе."""
    shown = ", ".join(paths[:_EXAMPLES_IN_WARNING])
    if len(paths) > _EXAMPLES_IN_WARNING:
        return f"Например: {shown} и другие"
    return f"Это: {shown}"


__all__ = [
    "DropReason",
    "DroppedBlockLog",
    "block_support_status",
    "dropped_resource_warning",
    "resource_support_status",
]
