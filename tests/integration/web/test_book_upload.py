"""EPUB upload: receiving file, registering book, and enqueuing import task (API-01, PRD § 21.2).

The test exercises the entire upload flow: browser request, saving file into data directory,
registering book in "Importing" state, queueing task, and executing it via the real
importer. This also verifies that the composition root connected resource storage
to the importer: without it, the book would have no stored resource copies.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FIXTURE_DIRECTORY
from fastapi.testclient import TestClient

from linguary.composition import Application, drain_tasks
from linguary.domain import BookStatus, TaskStatus, TaskType

#: Small synthetic book from the fixture suite (PRD § 28.1).
SYNTHETIC_BOOK = FIXTURE_DIRECTORY / "synthetic_story.epub"


def upload(client: TestClient, *, path: Path, filename: str | None = None):
    """Submits a book via the upload form, simulating a browser."""
    with path.open("rb") as file:
        return client.post(
            "/books/upload",
            files={"file": (filename or path.name, file, "application/epub+zip")},
            data={"source_language": "en", "target_language": "ru"},
            follow_redirects=False,
        )


def test_upload_redirects_to_the_book_card(client: TestClient) -> None:
    """Upload redirects to the book card (API-01)."""
    response = upload(client, path=SYNTHETIC_BOOK)

    assert response.status_code == 303
    assert response.headers["location"].startswith("/books/")


def test_uploaded_book_appears_in_the_list_as_importing(client: TestClient) -> None:
    """Uploaded book appears in the list immediately in importing state (PRD § 18)."""
    upload(client, path=SYNTHETIC_BOOK, filename="Синтетическая история.epub")

    books = client.get("/books").json()["books"]

    assert len(books) == 1
    assert books[0]["status"] == BookStatus.IMPORTING.value
    assert books[0]["status_label"] == "Импортируется"
    assert books[0]["title"] == "Синтетическая история"
    assert books[0]["source_language"] == "en"
    assert books[0]["target_language"] == "ru"


def test_upload_saves_the_file_into_the_data_directory(
    client: TestClient, application: Application
) -> None:
    """Book file is stored in the data directory under the book ID (PRD § 22)."""
    location = upload(client, path=SYNTHETIC_BOOK).headers["location"]
    book_id = location.rsplit("/", 1)[-1]

    stored = application.paths.source_epub(book_id)

    assert stored.is_file()
    assert stored.read_bytes() == SYNTHETIC_BOOK.read_bytes()


def test_upload_enqueues_the_import_task(
    client: TestClient, application: Application
) -> None:
    """Upload enqueues an import task instead of parsing in the request handler (PRD § 18.1)."""
    book_id = upload(client, path=SYNTHETIC_BOOK).headers["location"].rsplit("/", 1)[-1]

    with application.unit_of_work.transaction() as uow:
        tasks = uow.tasks.list_for_book(book_id)

    assert [(task.task_type, task.status) for task in tasks] == [
        (TaskType.IMPORT_BOOK, TaskStatus.PENDING)
    ]


async def test_import_task_imports_the_uploaded_book(
    client: TestClient, application: Application
) -> None:
    """Import task parses the uploaded book using the real importer."""
    book_id = upload(client, path=SYNTHETIC_BOOK).headers["location"].rsplit("/", 1)[-1]

    finished = await drain_tasks(application)

    assert [task.status for task in finished] == [TaskStatus.SUCCEEDED]
    detail = client.get(f"/books/{book_id}").json()
    assert detail["book"]["status"] == BookStatus.IMPORTED.value
    assert detail["progress"]["total_sections"] > 0


async def test_import_task_stores_book_resources(
    client: TestClient, application: Application
) -> None:
    """Import task persists book resources: storage wired via dependency composition."""
    book_id = upload(client, path=SYNTHETIC_BOOK).headers["location"].rsplit("/", 1)[-1]

    await drain_tasks(application)

    with application.unit_of_work.transaction() as uow:
        stored = uow.documents.stored_resource_paths(book_id)
    assert stored
    for path in stored.values():
        assert Path(path).is_file()
        assert Path(path).is_relative_to(application.paths.book_directory(book_id))


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        pytest.param("book.txt", "только формат EPUB", id="wrong-suffix"),
        pytest.param("book", "только формат EPUB", id="no-suffix"),
    ],
)
def test_upload_rejects_files_that_are_not_epub(
    client: TestClient, filename: str, expected: str
) -> None:
    """Non-EPUB file is rejected with `400` and an explanation (PRD § 4)."""
    response = upload(client, path=SYNTHETIC_BOOK, filename=filename)

    assert response.status_code == 400
    assert expected in response.json()["detail"]


def test_upload_without_languages_is_rejected(client: TestClient) -> None:
    """Form without chosen languages is rejected by input validation (PRD § 21.2)."""
    with SYNTHETIC_BOOK.open("rb") as file:
        response = client.post(
            "/books/upload",
            files={"file": (SYNTHETIC_BOOK.name, file, "application/epub+zip")},
        )

    assert response.status_code == 422


def test_rejected_upload_leaves_no_file_behind(
    client: TestClient, application: Application
) -> None:
    """Rejected upload leaves neither a file nor a book record."""
    upload(client, path=SYNTHETIC_BOOK, filename="book.txt")

    assert client.get("/books").json() == {"books": []}
    assert not list(application.paths.books_root.glob("*/source.epub"))


def test_empty_file_is_rejected(client: TestClient, tmp_path: Path) -> None:
    """Empty book file is rejected: nothing to import."""
    empty = tmp_path / "empty.epub"
    empty.write_bytes(b"")

    response = upload(client, path=empty)

    assert response.status_code == 400
    assert "пуст" in response.json()["detail"]
