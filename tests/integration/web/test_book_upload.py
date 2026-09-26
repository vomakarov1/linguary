"""Загрузка EPUB: приём файла, регистрация книги и задача импорта (API-01, PRD § 21.2).

Тест проходит весь стык загрузки: запрос браузера, сохранение файла в каталоге данных,
запись книги в состоянии «Импортируется», задача очереди и её выполнение настоящим
импортёром. Так проверяется и то, что composition root подключил импорту хранилище
ресурсов: без него у книги не оказалось бы сохранённых копий.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FIXTURE_DIRECTORY
from fastapi.testclient import TestClient

from linguary.composition import Application, drain_tasks
from linguary.domain import BookStatus, TaskStatus, TaskType

#: Маленькая синтетическая книга набора фикстур (PRD § 28.1).
SYNTHETIC_BOOK = FIXTURE_DIRECTORY / "synthetic_story.epub"


def upload(client: TestClient, *, path: Path, filename: str | None = None):
    """Отправляет книгу формой загрузки, как это делает браузер."""
    with path.open("rb") as file:
        return client.post(
            "/books/upload",
            files={"file": (filename or path.name, file, "application/epub+zip")},
            data={"source_language": "en", "target_language": "ru"},
            follow_redirects=False,
        )


def test_upload_redirects_to_the_book_card(client: TestClient) -> None:
    """Загрузка отвечает переходом на карточку книги (API-01)."""
    response = upload(client, path=SYNTHETIC_BOOK)

    assert response.status_code == 303
    assert response.headers["location"].startswith("/books/")


def test_uploaded_book_appears_in_the_list_as_importing(client: TestClient) -> None:
    """Книга появляется в списке сразу после загрузки, в состоянии импорта (PRD § 18)."""
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
    """Файл книги ложится в каталог данных под идентификатором книги (PRD § 22)."""
    location = upload(client, path=SYNTHETIC_BOOK).headers["location"]
    book_id = location.rsplit("/", 1)[-1]

    stored = application.paths.source_epub(book_id)

    assert stored.is_file()
    assert stored.read_bytes() == SYNTHETIC_BOOK.read_bytes()


def test_upload_enqueues_the_import_task(
    client: TestClient, application: Application
) -> None:
    """Загрузка ставит задачу импорта, а не выполняет разбор в запросе (PRD § 18.1)."""
    book_id = upload(client, path=SYNTHETIC_BOOK).headers["location"].rsplit("/", 1)[-1]

    with application.unit_of_work.transaction() as uow:
        tasks = uow.tasks.list_for_book(book_id)

    assert [(task.task_type, task.status) for task in tasks] == [
        (TaskType.IMPORT_BOOK, TaskStatus.PENDING)
    ]


async def test_import_task_imports_the_uploaded_book(
    client: TestClient, application: Application
) -> None:
    """Задача импорта разбирает загруженную книгу настоящим импортёром."""
    book_id = upload(client, path=SYNTHETIC_BOOK).headers["location"].rsplit("/", 1)[-1]

    finished = await drain_tasks(application)

    assert [task.status for task in finished] == [TaskStatus.SUCCEEDED]
    detail = client.get(f"/books/{book_id}").json()
    assert detail["book"]["status"] == BookStatus.IMPORTED.value
    assert detail["progress"]["total_sections"] > 0


async def test_import_task_stores_book_resources(
    client: TestClient, application: Application
) -> None:
    """Импорт сохраняет ресурсы книги: хранилище подключено составом зависимостей."""
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
    """Файл не в формате EPUB отвергается с `400` и объяснением (PRD § 4)."""
    response = upload(client, path=SYNTHETIC_BOOK, filename=filename)

    assert response.status_code == 400
    assert expected in response.json()["detail"]


def test_upload_without_languages_is_rejected(client: TestClient) -> None:
    """Без выбранных языков форма отвергается проверкой входа (PRD § 21.2)."""
    with SYNTHETIC_BOOK.open("rb") as file:
        response = client.post(
            "/books/upload",
            files={"file": (SYNTHETIC_BOOK.name, file, "application/epub+zip")},
        )

    assert response.status_code == 422


def test_rejected_upload_leaves_no_file_behind(
    client: TestClient, application: Application
) -> None:
    """Отвергнутая загрузка не оставляет ни файла, ни книги в списке."""
    upload(client, path=SYNTHETIC_BOOK, filename="book.txt")

    assert client.get("/books").json() == {"books": []}
    assert not list(application.paths.books_root.glob("*/source.epub"))


def test_empty_file_is_rejected(client: TestClient, tmp_path: Path) -> None:
    """Пустой файл книги отвергается: импортировать нечего."""
    empty = tmp_path / "empty.epub"
    empty.write_bytes(b"")

    response = upload(client, path=empty)

    assert response.status_code == 400
    assert "пуст" in response.json()["detail"]
