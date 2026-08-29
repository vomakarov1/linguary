"""Раскладка каталога данных приложения (PRD § 22).

Всё, что приложение держит на диске, лежит под одним корнем: база, загруженные книги,
их распакованные ресурсы и собранные EPUB. Раскладку знает единственный модуль, поэтому
Web, CLI и сквозной тест находят один и тот же файл по одному и тому же правилу, а
перенос приложения на другую машину сводится к переносу каталога.

Каталог книги переживает повторный импорт: хранилище ресурсов чистит только свой
подкаталог, а исходный файл и собранный EPUB лежат рядом с ним.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from linguary.adapters.config import DEFAULT_DATABASE_NAME, StorageSettings

#: Каталог книг внутри корня данных. Он же — корень хранилища ресурсов EPUB:
#: `FilesystemEpubResourceStore` раскладывает их как `<корень>/<book_id>/resources/...`.
BOOKS_DIRECTORY = "books"

#: Имя исходного файла книги внутри её каталога. Имя загруженного файла в путь не идёт:
#: оно приходит из браузера и участвовать в построении пути не должно.
SOURCE_FILE_NAME = "source.epub"

#: Подкаталог и имя собранного EPUB внутри каталога книги.
EXPORT_DIRECTORY = "export"
EXPORT_FILE_NAME = "translated.epub"


@dataclass(frozen=True, slots=True)
class DataPaths:
    """Пути приложения, вычисленные из настроек хранения."""

    data_root: Path
    database_path: Path

    @classmethod
    def of_settings(cls, settings: StorageSettings) -> DataPaths:
        """Пути по локальной конфигурации; пустой путь базы означает каталог данных."""
        root = Path(settings.data_root).expanduser()
        configured = settings.database_path.strip()
        return cls(
            data_root=root,
            database_path=(
                Path(configured).expanduser()
                if configured
                else root / DEFAULT_DATABASE_NAME
            ),
        )

    @property
    def books_root(self) -> Path:
        """Корень каталогов книг и хранилища их ресурсов."""
        return self.data_root / BOOKS_DIRECTORY

    def book_directory(self, book_id: str) -> Path:
        """Каталог одной книги."""
        return self.books_root / book_id

    def source_epub(self, book_id: str) -> Path:
        """Куда кладётся загруженный пользователем файл книги."""
        return self.book_directory(book_id) / SOURCE_FILE_NAME

    def export_epub(self, book_id: str) -> Path:
        """Куда собирается итоговый EPUB книги (PRD § 20).

        Путь фиксирован: экспортная задача знает о книге только её идентификатор, а
        скачивание должно находить последний собранный файл по тому же правилу.
        """
        return self.book_directory(book_id) / EXPORT_DIRECTORY / EXPORT_FILE_NAME

    def prepare(self) -> None:
        """Создаёт каталоги приложения, если их ещё нет."""
        self.books_root.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


__all__ = [
    "BOOKS_DIRECTORY",
    "EXPORT_DIRECTORY",
    "EXPORT_FILE_NAME",
    "SOURCE_FILE_NAME",
    "DataPaths",
]
