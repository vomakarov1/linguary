"""Application data directory layout (PRD § 22).

Everything the application keeps on disk resides under a single root: database, uploaded books,
their unpacked resources, and assembled EPUBs. The layout is known to a single module, so
Web, CLI, and end-to-end tests locate the same file according to the same rule, and migrating
the application to another machine reduces to copying the directory.

A book's directory survives repeated imports: the resource store cleans only its own
subdirectory, while the source file and assembled EPUB sit alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from linguary.adapters.config import DEFAULT_DATABASE_NAME, StorageSettings

#: Books directory inside data root. It is also the root of the EPUB resource store:
#: `FilesystemEpubResourceStore` arranges them as `<root>/<book_id>/resources/...`.
BOOKS_DIRECTORY = "books"

#: Source book file name inside its directory. The uploaded file name is not used in the path:
#: it comes from the browser and must not participate in path construction.
SOURCE_FILE_NAME = "source.epub"

#: Subdirectory and file name of the assembled EPUB inside the book directory.
EXPORT_DIRECTORY = "export"
EXPORT_FILE_NAME = "translated.epub"

#: Suffix for the queue executor lock file next to the database.
EXECUTOR_LOCK_SUFFIX = ".executors.lock"


@dataclass(frozen=True, slots=True)
class DataPaths:
    """Application paths computed from storage settings."""

    data_root: Path
    database_path: Path

    @classmethod
    def of_settings(cls, settings: StorageSettings) -> DataPaths:
        """Paths from local configuration; empty database path defaults to data directory."""
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
        """Root of book directories and their resource stores."""
        return self.data_root / BOOKS_DIRECTORY

    def book_directory(self, book_id: str) -> Path:
        """Directory of a single book."""
        return self.books_root / book_id

    def source_epub(self, book_id: str) -> Path:
        """Where the user-uploaded book file is placed."""
        return self.book_directory(book_id) / SOURCE_FILE_NAME

    def export_epub(self, book_id: str) -> Path:
        """Where the final book EPUB is assembled (PRD § 20).

        The path is fixed: the export task knows only the book's identifier, and
        downloading must locate the latest assembled file by the same rule.
        """
        return self.book_directory(book_id) / EXPORT_DIRECTORY / EXPORT_FILE_NAME

    @property
    def executor_lock(self) -> Path:
        """Queue executor lock file.

        Resides next to the database rather than data root: executors share the task queue,
        i.e. the database, and its path can be specified separately from data root.
        """
        return self.database_path.with_name(
            f"{self.database_path.name}{EXECUTOR_LOCK_SUFFIX}"
        )

    def prepare(self) -> None:
        """Creates application directories if they do not yet exist."""
        self.books_root.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


__all__ = [
    "BOOKS_DIRECTORY",
    "EXECUTOR_LOCK_SUFFIX",
    "EXPORT_DIRECTORY",
    "EXPORT_FILE_NAME",
    "SOURCE_FILE_NAME",
    "DataPaths",
]
