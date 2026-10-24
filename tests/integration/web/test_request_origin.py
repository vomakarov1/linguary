"""Request origin: foreign page and foreign server host (PRD § 4.1, ADR § 3).

The local application intentionally has no authentication, so the only barrier
separating it from any page opened in the same browser is request origin headers.
This test verifies both checks: foreign Host header (DNS rebinding) and cross-site
state-changing requests.
"""

from __future__ import annotations

import pytest
from conftest import FIXTURE_DIRECTORY
from fastapi.testclient import TestClient

from linguary.composition import Application

#: Small synthetic book from the fixture suite (PRD § 28.1).
SYNTHETIC_BOOK = FIXTURE_DIRECTORY / "synthetic_story.epub"

#: Page opened in the same browser alongside the application.
FOREIGN_ORIGIN = "http://attacker.example"


def upload(client: TestClient, *, headers: dict[str, str] | None = None):
    """Uploads book via upload form with specified origin headers."""
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
    """Book upload from a foreign page is rejected and leaves no file on disk."""
    response = upload(client, headers={"origin": FOREIGN_ORIGIN})

    assert response.status_code == 403
    assert "только со страницы приложения" in response.json()["detail"]
    assert not list(application.paths.books_root.glob("*/source.epub"))


def test_upload_from_the_application_page_is_accepted(
    client: TestClient, application: Application
) -> None:
    """Upload from the application's own origin proceeds normally (API-01)."""
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
    """Client without origin headers is not a browser, and its request is accepted."""
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
    """State-changing request from another site is rejected before parsing confirmation form."""
    response = client.post(
        "/books/book-1/reset",
        data={"confirmed": "true"},
        headers={"sec-fetch-site": fetch_site},
    )

    assert response.status_code == 403


def test_reading_from_another_page_is_not_blocked(client: TestClient) -> None:
    """Reading from another page is not blocked: the browser will not deliver the response to it anyway."""
    response = client.get("/books", headers={"origin": FOREIGN_ORIGIN})

    assert response.status_code == 200


def test_request_by_a_foreign_host_is_rejected(
    client: TestClient, application: Application
) -> None:
    """Request with a foreign host header is rejected to prevent DNS rebinding."""
    response = client.get("/books", headers={"host": "attacker.example"})

    assert response.status_code == 400
    assert application.settings.server.base_url in response.json()["detail"]


def test_request_by_localhost_is_accepted(client: TestClient) -> None:
    """Local access works both by IP address and by `localhost` hostname."""
    response = client.get("/books", headers={"host": "localhost:8000"})

    assert response.status_code == 200
