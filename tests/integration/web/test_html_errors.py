"""HTML form rejections are rendered as a page for browsers rather than raw JSON (QA-05).

A browser submitting a form asks for `text/html`. Such a request receives an error page
with the same status code, human-readable reason, and a back link. A JSON client still
receives `{"detail": ...}`: the API-01 contract is preserved.
"""

from __future__ import annotations

import pytest
from conftest import FIXTURE_DIRECTORY
from fastapi.testclient import TestClient
from httpx import Response
from web_scenario import glossary_candidate, start_run, store_book

from linguary.composition import Application
from linguary.domain import BookStatus

#: Header sent by the browser during regular form submissions.
BROWSER_ACCEPT = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

SYNTHETIC_BOOK = FIXTURE_DIRECTORY / "synthetic_story.epub"


def assert_error_page(response: Response, *, status_code: int, back_url: str) -> str:
    """Verifies that rejection arrived as a page with status code and back link."""
    assert response.status_code == status_code
    assert response.headers["content-type"].startswith("text/html")
    assert '{"detail"' not in response.text
    assert f'href="{back_url}"' in response.text
    assert "Вернуться назад" in response.text
    return response.text


def upload_text_file(client: TestClient, headers: dict[str, str]) -> Response:
    """Submits the upload form with a non-EPUB file."""
    with SYNTHETIC_BOOK.open("rb") as file:
        return client.post(
            "/books/upload",
            files={"file": ("book.txt", file, "text/plain")},
            data={"source_language": "en", "target_language": "ru"},
            headers=headers,
        )


def test_rejected_upload_shows_an_error_page_with_a_link_back(
    client: TestClient,
) -> None:
    """Non-EPUB file is explained via a page with a link to the book list."""
    response = upload_text_file(
        client, {**BROWSER_ACCEPT, "referer": "http://127.0.0.1:8000/books"}
    )

    page = assert_error_page(response, status_code=400, back_url="/books")
    assert "Поддерживается только формат EPUB" in page


def test_rejected_upload_still_answers_json_to_an_api_client(
    client: TestClient,
) -> None:
    """JSON client receives the standard rejection `{"detail": ...}`."""
    response = upload_text_file(client, {"accept": "application/json"})

    assert response.status_code == 400
    assert "только формат EPUB" in response.json()["detail"]


def test_incomplete_upload_form_shows_an_error_page(client: TestClient) -> None:
    """Form missing languages (`422`) is explained via a page, not a schema validation error list."""
    with SYNTHETIC_BOOK.open("rb") as file:
        response = client.post(
            "/books/upload",
            files={"file": (SYNTHETIC_BOOK.name, file, "application/epub+zip")},
            headers=BROWSER_ACCEPT,
        )

    page = assert_error_page(response, status_code=422, back_url="/books")
    assert "Не заполнено поле «source_language»." in page


def test_unconfirmed_reset_shows_the_reason_on_a_page(
    client: TestClient, application: Application
) -> None:
    """Unconfirmed reset returns to the page from which the form was submitted."""
    scenario = store_book(application, status=BookStatus.PROCESSING)
    start_run(application, scenario)
    result_page = f"/books/{scenario.book_id}/result"

    response = client.post(
        f"/books/{scenario.book_id}/reset",
        data={"confirmed": "false"},
        headers={**BROWSER_ACCEPT, "referer": f"http://127.0.0.1:8000{result_page}"},
    )

    page = assert_error_page(response, status_code=422, back_url=result_page)
    assert "Возврат к проверке глоссария нужно подтвердить." in page


def test_analysis_without_llm_shows_a_conflict_page(
    client: TestClient, application: Application
) -> None:
    """`409` without configured LLM is rendered as a page and links back to the book card."""
    scenario = store_book(
        application, status=BookStatus.IMPORTED, analysed_chapters=()
    )

    response = client.post(
        f"/books/{scenario.book_id}/analyze", headers=BROWSER_ACCEPT
    )

    page = assert_error_page(
        response, status_code=409, back_url=f"/books/{scenario.book_id}"
    )
    assert "LLM не настроена" in page


def test_glossary_decision_after_processing_started_shows_a_conflict_page(
    client: TestClient, application: Application
) -> None:
    """Glossary edit after processing start is explained via a page (`409`)."""
    candidate = glossary_candidate()
    scenario = store_book(
        application,
        status=BookStatus.PROCESSING,
        glossary_candidates={0: (candidate,)},
    )
    glossary_page = f"/books/{scenario.book_id}/glossary"

    response = client.post(
        f"{glossary_page}/{candidate.id}",
        data={"action": "accept"},
        headers={**BROWSER_ACCEPT, "referer": f"http://127.0.0.1:8000{glossary_page}"},
    )

    page = assert_error_page(response, status_code=409, back_url=glossary_page)
    assert "только для чтения" in page


@pytest.mark.parametrize(
    ("method", "path"),
    [
        pytest.param("post", "/tasks/unknown/retry", id="retry-unknown-task"),
        pytest.param("get", "/books/unknown", id="unknown-book"),
        pytest.param("get", "/books/unknown/download", id="download-unknown-book"),
        pytest.param("get", "/no-such-page", id="unknown-route"),
    ],
)
def test_not_found_shows_an_error_page(
    client: TestClient, method: str, path: str
) -> None:
    """`404` for browser: error page with a link to the book list."""
    response = client.request(method, path, headers=BROWSER_ACCEPT)

    assert_error_page(response, status_code=404, back_url="/books")


def test_not_found_still_answers_json_to_an_api_client(client: TestClient) -> None:
    """Unknown book for JSON client returns the standard `{"detail": ...}`."""
    response = client.get("/books/unknown")

    assert response.status_code == 404
    assert "detail" in response.json()


@pytest.mark.parametrize(
    "referer",
    [
        pytest.param("https://example.com/phishing", id="foreign-site"),
        pytest.param("http://127.0.0.1:8000//example.com/phishing", id="scheme-relative"),
    ],
)
def test_foreign_referer_is_not_offered_as_the_way_back(
    client: TestClient, referer: str
) -> None:
    """Back link points strictly inside the application, even if `Referer` is external."""
    response = client.post(
        "/tasks/unknown/retry", headers={**BROWSER_ACCEPT, "referer": referer}
    )

    page = assert_error_page(response, status_code=404, back_url="/books")
    assert "example.com" not in page
