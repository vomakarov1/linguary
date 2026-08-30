"""Происхождение запроса: чужая страница и чужое имя сервера (PRD § 4.1, ADR § 3).

Аутентификации у локального приложения нет намеренно, поэтому единственное, что отделяет
его от любой страницы, открытой в том же браузере, — заголовки происхождения. Тест
закрепляет обе проверки: обращение по чужому `Host` (перепривязка DNS) и межсайтовый
запрос, меняющий состояние.
"""

from __future__ import annotations

import pytest
from conftest import FIXTURE_DIRECTORY
from fastapi.testclient import TestClient

from linguary.composition import Application

#: Маленькая синтетическая книга набора фикстур (PRD § 28.1).
SYNTHETIC_BOOK = FIXTURE_DIRECTORY / "synthetic_story.epub"

#: Страница, открытая в том же браузере рядом с приложением.
FOREIGN_ORIGIN = "http://attacker.example"


def upload(client: TestClient, *, headers: dict[str, str] | None = None):
    """Отправляет книгу формой загрузки с заданными заголовками происхождения."""
    with SYNTHETIC_BOOK.open("rb") as file:
        return client.post(
            "/books/upload",
            files={"file": (SYNTHETIC_BOOK.name, file, "application/epub+zip")},
            data={"source_language": "en", "target_language": "ru"},
            headers=headers,
            follow_redirects=False,
        )


def test_upload_from_a_foreign_page_is_rejected(
    client: TestClient, application: Application
) -> None:
    """Загрузка книги с чужой страницы отклоняется и файла на диске не оставляет."""
    response = upload(client, headers={"origin": FOREIGN_ORIGIN})

    assert response.status_code == 403
    assert "только со страницы приложения" in response.json()["detail"]
    assert not list(application.paths.books_root.glob("*/source.epub"))


def test_upload_from_the_application_page_is_accepted(
    client: TestClient, application: Application
) -> None:
    """Загрузка со страницы самого приложения проходит как обычно (API-01)."""
    response = upload(
        client,
        headers={
            "origin": application.settings.server.base_url,
            "sec-fetch-site": "same-origin",
        },
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/books/")


def test_upload_without_origin_headers_is_accepted(client: TestClient) -> None:
    """Клиент без заголовков происхождения — не браузер, и его запрос принимается."""
    response = upload(client)

    assert response.status_code == 303


@pytest.mark.parametrize(
    "fetch_site",
    [
        pytest.param("cross-site", id="cross-site"),
        pytest.param("same-site", id="same-site"),
    ],
)
def test_state_changing_request_from_another_site_is_rejected(
    client: TestClient, fetch_site: str
) -> None:
    """Сброс обработки с чужой страницы отклоняется до разбора формы подтверждения."""
    response = client.post(
        "/books/book-1/reset",
        data={"confirmed": "true"},
        headers={"sec-fetch-site": fetch_site},
    )

    assert response.status_code == 403


def test_reading_from_another_page_is_not_blocked(client: TestClient) -> None:
    """Чтение чужой странице не запрещается: ответ ей всё равно не отдаст браузер."""
    response = client.get("/books", headers={"origin": FOREIGN_ORIGIN})

    assert response.status_code == 200


def test_request_by_a_foreign_host_is_rejected(
    client: TestClient, application: Application
) -> None:
    """Обращение под чужим именем отклоняется: так закрывается перепривязка DNS."""
    response = client.get("/books", headers={"host": "attacker.example"})

    assert response.status_code == 400
    assert application.settings.server.base_url in response.json()["detail"]


def test_request_by_localhost_is_accepted(client: TestClient) -> None:
    """Локальный запуск открывается и по адресу, и по имени `localhost`."""
    response = client.get("/books", headers={"host": "localhost:8000"})

    assert response.status_code == 200
