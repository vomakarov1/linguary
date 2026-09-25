"""Сохранение ресурсов книги в локальное хранилище (PRD § 22).

Реализация порта `linguary.application.ports.epub.EpubResourceStore`. Каноническая модель
знает ресурс по его пути внутри EPUB, а экспортёру нужен настоящий файл, поэтому импорт
достаёт из архива всё, что переносится в итоговую книгу, и складывает под каталогом книги.

Архив здесь — недоверенный вход. Путь ресурса из манифеста и имя записи в архиве могут
вести куда угодно, поэтому оба нормализуются, а запись разрешена только внутри каталога
книги. Размер распакованного содержимого и число файлов ограничены: превышение лимита —
`FATAL`, книга не импортируется, и в хранилище не остаётся ни одной копии её ресурсов.
"""

from __future__ import annotations

import posixpath
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from lxml import etree

from linguary.adapters.epub.inline import local_name
from linguary.application.ports.epub import (
    EpubImportFatalError,
    EpubResourceStore,
    StoredResources,
    StoreResourcesRequest,
)
from linguary.domain import CanonicalDocument, SupportStatus

#: Статусы поддержки, при которых ресурс переносится в итоговую книгу и потому сохраняется.
_STORED_STATUSES = frozenset({SupportStatus.SUPPORTED, SupportStatus.PRESERVE_OPAQUE})

#: Подкаталог книги, в котором лежат копии ресурсов.
_RESOURCES_DIRECTORY = "resources"


@dataclass(frozen=True, slots=True)
class ExtractionLimits:
    """Пределы распаковки архива книги.

    Значения по умолчанию рассчитаны на художественную книгу с иллюстрациями и заведомо
    больше любой книги набора; они существуют не ради экономии диска, а чтобы архив,
    распаковывающийся в десятки гигабайт, не занял машину пользователя.

    `max_document_bytes` ограничивает одну запись архива, которую импорт поднимает
    в память: глава художественной книги заведомо меньше десятков мегабайт, и запись
    такого размера читать незачем даже в пределах общего объёма.
    """

    max_entries: int = 10_000
    max_total_bytes: int = 256 * 1024 * 1024
    max_document_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        if (
            self.max_entries <= 0
            or self.max_total_bytes <= 0
            or self.max_document_bytes <= 0
        ):
            raise ValueError("лимиты распаковки должны быть положительными")


class FilesystemEpubResourceStore(EpubResourceStore):
    """Хранилище ресурсов книги в каталоге локальной файловой системы.

    Ресурсы книги лежат в `<корень хранилища>/<book_id>/resources/<путь внутри EPUB>`.
    Повторный импорт книги начинает каталог заново: копии ресурсов производны от архива,
    и держать в хранилище файлы прошлого импорта незачем.
    """

    def __init__(self, *, root: Path, limits: ExtractionLimits | None = None) -> None:
        self._root = root
        self._limits = limits or ExtractionLimits()

    def store(self, request: StoreResourcesRequest) -> StoredResources:
        """Сохраняет ресурсы книги; нарушение матрицы поддержки даёт `FATAL`."""
        book_directory = self._book_directory(request.book_id)
        with _ResourceArchive(request.source_path, limits=self._limits) as archive:
            planned = _plan(
                document=request.document,
                archive=archive,
                book_directory=book_directory,
                book_id=request.book_id,
            )
            _reset_directory(book_directory)
            paths = archive.extract(planned, cleanup=book_directory)
        return StoredResources(paths=paths)

    def _book_directory(self, book_id: str) -> Path:
        """Каталог ресурсов книги внутри корня хранилища."""
        directory = _contained_path(
            self._root, book_id, message=f"идентификатор книги {book_id!r}"
        )
        return directory / _RESOURCES_DIRECTORY


@dataclass(frozen=True, slots=True)
class _PlannedResource:
    """Ресурс, найденная запись архива и путь, по которому он будет сохранён."""

    source_path: str
    archive_name: str
    target: Path


class _ResourceArchive:
    """Архив книги: поиск записей по пути ресурса и извлечение их в хранилище."""

    def __init__(self, source_path: str, *, limits: ExtractionLimits) -> None:
        self._source_path = source_path
        self._limits = limits
        try:
            self._archive = zipfile.ZipFile(source_path)
        except (OSError, zipfile.BadZipFile) as error:
            raise EpubImportFatalError(
                f"EPUB {source_path} нельзя прочитать: {error}"
            ) from error
        try:
            self._check_limits()
            self._directory = posixpath.dirname(package_path(self._archive))
        except (OSError, KeyError, zipfile.BadZipFile, etree.XMLSyntaxError) as error:
            self._archive.close()
            raise EpubImportFatalError(
                f"EPUB {source_path} нельзя прочитать: {error}"
            ) from error
        except EpubImportFatalError:
            self._archive.close()
            raise
        self._names = frozenset(
            posixpath.normpath(name) for name in self._archive.namelist()
        )

    def __enter__(self) -> _ResourceArchive:
        return self

    def __exit__(self, *exception: object) -> None:
        self._archive.close()

    def locate(self, resource_path: str) -> str | None:
        """Имя записи архива для ресурса книги либо `None`, если её там нет.

        Путь ресурса берётся из манифеста и доверия не заслуживает: `../` в нём выводит
        за каталог пакета, а то и за сам архив, поэтому такой путь — `FATAL`.
        """
        name = self._archive_name(resource_path)
        decoded = unquote(name)
        for candidate in (name, decoded) if decoded != name else (name,):
            if candidate in self._names:
                return candidate
        return None

    def extract(
        self, planned: Sequence[_PlannedResource], *, cleanup: Path
    ) -> Mapping[str, str]:
        """Пишет запланированные ресурсы и отдаёт пути их копий.

        Фактический размер считается по мере записи: заголовки архива объявляют размер
        файла, но обязать архив говорить правду нельзя. Превышение лимита оставляет
        книгу неимпортированной, поэтому уже записанные копии удаляются.
        """
        paths: dict[str, str] = {}
        written = 0
        for item in planned:
            payload = self._read(item.archive_name)
            written += len(payload)
            if written > self._limits.max_total_bytes:
                shutil.rmtree(cleanup, ignore_errors=True)
                raise EpubImportFatalError(
                    f"распакованное содержимое EPUB {self._source_path} превышает "
                    f"допустимые {self._limits.max_total_bytes} байт"
                )
            item.target.parent.mkdir(parents=True, exist_ok=True)
            item.target.write_bytes(payload)
            paths[item.source_path] = str(item.target)
        return paths

    def _archive_name(self, resource_path: str) -> str:
        """Имя записи архива для пути ресурса; выход за корень архива — `FATAL`."""
        joined = posixpath.join(self._directory, resource_path)
        normalized = posixpath.normpath(joined)
        if normalized.startswith(("..", "/")):
            raise EpubImportFatalError(
                f"ресурс {resource_path} книги {self._source_path} ведёт за пределы "
                f"архива: {joined!r}"
            )
        return normalized

    def _read(self, name: str) -> bytes:
        """Содержимое записи архива; нечитаемая запись — `FATAL`."""
        try:
            return self._archive.read(name)
        except (KeyError, OSError, zipfile.BadZipFile) as error:
            raise EpubImportFatalError(
                f"ресурс {name} книги {self._source_path} нельзя прочитать: {error}"
            ) from error

    def _check_limits(self) -> None:
        """Проверяет объявленные архивом число файлов и размер распаковки."""
        check_extraction_limits(
            self._archive, source_path=self._source_path, limits=self._limits
        )


def check_extraction_limits(
    archive: zipfile.ZipFile, *, source_path: str, limits: ExtractionLimits
) -> None:
    """Сверяет объявленные архивом число файлов и объём распаковки с пределами.

    Проверка идёт по оглавлению архива, до чтения хотя бы одной его записи: архив,
    распаковывающийся в десятки гигабайт, не должен успеть занять память машины.
    Объявленному размеру доверия нет, поэтому тот, кто читает записи, считает
    прочитанные байты и сам.
    """
    entries = archive.infolist()
    if len(entries) > limits.max_entries:
        raise EpubImportFatalError(
            f"EPUB {source_path} содержит {len(entries)} файлов "
            f"при допустимых {limits.max_entries}"
        )
    declared = sum(entry.file_size for entry in entries)
    if declared > limits.max_total_bytes:
        raise EpubImportFatalError(
            f"распакованное содержимое EPUB {source_path} заявлено "
            f"как {declared} байт при допустимых {limits.max_total_bytes}"
        )


def package_path(archive: zipfile.ZipFile) -> str:
    """Путь к файлу пакета OPF внутри архива по `META-INF/container.xml`.

    Контейнер приходит из недоверенного архива, поэтому разбирается тем же ужесточённым
    парсером, что и остальной XML книги: внешние сущности не подставляются, DTD не
    загружается, в сеть парсер не ходит.
    """
    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
    container = etree.fromstring(archive.read("META-INF/container.xml"), parser=parser)
    for element in container.iter():
        if local_name(element) == "rootfile" and element.get("full-path"):
            return element.get("full-path", "")
    raise EpubImportFatalError("в EPUB не объявлен файл пакета OPF")


def _plan(
    *,
    document: CanonicalDocument,
    archive: _ResourceArchive,
    book_directory: Path,
    book_id: str,
) -> tuple[_PlannedResource, ...]:
    """Что и куда будет записано.

    План строится целиком до первой записи: путь, ведущий за пределы каталога книги,
    обязан остановить импорт до того, как в хранилище появится хоть один файл.

    Ресурс манифеста, потерянный в архиве, — тоже `FATAL` (PRD § 6). До этой проверки
    книга обычно не доходит: EbookLib читает содержимое манифеста при открытии книги
    и отвергает её как нечитаемую раньше, — но целостность хранилища зависит не от того,
    какая библиотека открыла архив.
    """
    planned: list[_PlannedResource] = []
    for resource in document.resources:
        if resource.support_status not in _STORED_STATUSES:
            continue
        target = _contained_path(
            book_directory,
            resource.source_path,
            message=f"ресурс {resource.source_path} книги {book_id}",
        )
        archive_name = archive.locate(resource.source_path)
        if archive_name is None:
            raise EpubImportFatalError(
                f"ресурс {resource.source_path} объявлен в манифесте книги {book_id}, "
                f"но отсутствует в архиве"
            )
        planned.append(
            _PlannedResource(
                source_path=resource.source_path,
                archive_name=archive_name,
                target=target,
            )
        )
    return tuple(planned)


def _reset_directory(directory: Path) -> None:
    """Очищает каталог ресурсов книги перед новым импортом."""
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)


def _contained_path(base: Path, relative: str, *, message: str) -> Path:
    """Путь внутри `base`; выход за его пределы — `FATAL` (защита от zip slip)."""
    normalized = relative.replace("\\", "/").strip()
    parts = tuple(part for part in normalized.split("/") if part and part != ".")
    candidate = Path(base, *parts)
    base_resolved = base.resolve()
    if (
        not parts
        or normalized.startswith("/")
        or ".." in parts
        or base_resolved not in candidate.resolve().parents
    ):
        raise EpubImportFatalError(
            f"{message} ведёт за пределы каталога хранилища книги: {relative!r}"
        )
    return candidate


__all__ = [
    "ExtractionLimits",
    "FilesystemEpubResourceStore",
    "check_extraction_limits",
    "package_path",
]
