"""Glossary, processing start, and return to review (API-01, P4.T2)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from web_scenario import (
    glossary_candidate,
    start_run,
    store_book,
    translated_result,
)

from linguary.composition import Application
from linguary.domain import BookStatus


def test_glossary_review_exposes_pending_items(
    client: TestClient, application: Application
) -> None:
    """Glossary screen displays the identifier and pending decision first."""
    candidate = glossary_candidate()
    scenario = store_book(
        application, glossary_candidates={0: (candidate,)},
        chapters=(("Mr. Bennet " * 5,),),
    )

    response = client.get(f"/books/{scenario.book_id}/glossary")

    assert response.status_code == 200
    assert response.json() == {
        "book_id": scenario.book_id,
        "read_only": False,
        "is_complete": False,
        "pending_count": 1,
        "items": [
            {
                "concept_id": candidate.id,
                "source_text": "Mr. Bennet",
                "type": "person",
                "suggested_translation": "мистер Беннет",
                "translation": None,
                "status": None,
                "status_label": None,
                "confidence": 0.95,
                "model_flagged_review": False,
                "model_doubts": False,
                "has_translation_conflict": False,
                "requires_review": True,
                "is_decided": False,
                "note": "",
                "other_spellings": [],
                "has_decision_conflict": False,
            }
        ],
    }


@pytest.mark.parametrize(
    ("action", "text", "expected_translation", "expected_status", "expected_label"),
    [
        pytest.param(
            "accept", None, "мистер Беннет", "approved", "принято", id="accept"
        ),
        pytest.param(
            "correct",
            "господин Беннет",
            "господин Беннет",
            "approved",
            "исправлено",
            id="correct",
        ),
        pytest.param(
            "lock", None, "мистер Беннет", "locked", "зафиксировано", id="lock"
        ),
        pytest.param(
            "skip", None, "мистер Беннет", "suggested", "пропущено", id="skip"
        ),
    ],
)
def test_glossary_decision_returns_updated_item(
    client: TestClient,
    application: Application,
    action: str,
    text: str | None,
    expected_translation: str,
    expected_status: str,
    expected_label: str,
) -> None:
    """Each action saves the decision and returns the updated item."""
    candidate = glossary_candidate()
    scenario = store_book(application, glossary_candidates={0: (candidate,)})
    form = {"action": action}
    if text is not None:
        form["text"] = text

    response = client.post(
        f"/books/{scenario.book_id}/glossary/{candidate.id}",
        data=form,
    )

    assert response.status_code == 200
    assert response.json()["concept_id"] == candidate.id
    assert response.json()["translation"] == expected_translation
    assert response.json()["status"] == expected_status
    assert response.json()["status_label"] == expected_label
    assert response.json()["is_decided"] is True
    with application.unit_of_work.transaction() as uow:
        assert uow.books.get(scenario.book_id).status is BookStatus.READY_TO_PROCESS


def test_glossary_decision_redirects_browser_back_to_review(
    client: TestClient, application: Application
) -> None:
    """HTML action on a single item keeps the user on the glossary screen."""
    candidate = glossary_candidate()
    scenario = store_book(application, glossary_candidates={0: (candidate,)})

    response = client.post(
        f"/books/{scenario.book_id}/glossary/{candidate.id}",
        data={"action": "accept"},
        headers={"accept": "text/html"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == f"/books/{scenario.book_id}/glossary"


def test_accept_all_required_glossary_items(
    client: TestClient, application: Application
) -> None:
    """Bulk action accepts all required pending items."""
    first = glossary_candidate(candidate_id="candidate-first")
    second = glossary_candidate(
        candidate_id="candidate-second",
        source_text="Longbourn",
        suggested_translation="Лонгборн",
    )
    scenario = store_book(
        application,
        glossary_candidates={0: (first, second)},
        chapters=(("Mr. Bennet Longbourn " * 5,),),
    )

    response = client.post(f"/books/{scenario.book_id}/glossary/accept-all")

    assert response.status_code == 200
    assert response.json()["pending_count"] == 0
    assert {
        (item["source_text"], item["translation"], item["status"])
        for item in response.json()["items"]
    } == {
        ("Mr. Bennet", "мистер Беннет", "approved"),
        ("Longbourn", "Лонгборн", "approved"),
    }
    with application.unit_of_work.transaction() as uow:
        assert uow.books.get(scenario.book_id).status is BookStatus.READY_TO_PROCESS


def test_accept_all_redirects_browser_back_to_review(
    client: TestClient, application: Application
) -> None:
    """HTML bulk action returns the user to the glossary screen."""
    candidate = glossary_candidate()
    scenario = store_book(application, glossary_candidates={0: (candidate,)})

    response = client.post(
        f"/books/{scenario.book_id}/glossary/accept-all",
        headers={"accept": "text/html"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == f"/books/{scenario.book_id}/glossary"


def test_glossary_correction_requires_text(
    client: TestClient, application: Application
) -> None:
    """Correction without replacement text is rejected as an incomplete HTTP command."""
    candidate = glossary_candidate()
    scenario = store_book(application, glossary_candidates={0: (candidate,)})

    response = client.post(
        f"/books/{scenario.book_id}/glossary/{candidate.id}",
        data={"action": "correct"},
    )

    assert response.status_code == 422


def test_glossary_correction_changes_the_concept_type(
    client: TestClient, application: Application
) -> None:
    """Correction accepts the concept type and returns it in the item (QA-11, T2)."""
    candidate = glossary_candidate(
        source_text="Charlotte Russe", suggested_translation="Шарлотта Русс"
    )
    scenario = store_book(application, glossary_candidates={0: (candidate,)})

    response = client.post(
        f"/books/{scenario.book_id}/glossary/{candidate.id}",
        data={"action": "correct", "text": "шарлотка по-русски", "type": "term"},
    )

    assert response.status_code == 200
    assert response.json()["type"] == "term"
    assert response.json()["translation"] == "шарлотка по-русски"


def test_glossary_type_is_rejected_without_correction(
    client: TestClient, application: Application
) -> None:
    """Concept type, like translation text, is accepted only via correction."""
    candidate = glossary_candidate()
    scenario = store_book(application, glossary_candidates={0: (candidate,)})

    response = client.post(
        f"/books/{scenario.book_id}/glossary/{candidate.id}",
        data={"action": "accept", "type": "term"},
    )

    assert response.status_code == 422


def test_glossary_note_is_saved_and_shown(
    client: TestClient, application: Application
) -> None:
    """Translator note is saved as a separate action and returned in the item."""
    candidate = glossary_candidate()
    scenario = store_book(application, glossary_candidates={0: (candidate,)})

    response = client.post(
        f"/books/{scenario.book_id}/glossary/{candidate.id}/note",
        data={"note": "женский род"},
    )

    assert response.status_code == 200
    assert response.json()["note"] == "женский род"
    assert response.json()["translation"] == "мистер Беннет"


def test_glossary_note_is_rejected_after_processing_started(
    client: TestClient, application: Application
) -> None:
    """Once processing starts, the note is also frozen: glossary is read-only."""
    candidate = glossary_candidate()
    scenario = store_book(
        application,
        status=BookStatus.PROCESSING,
        glossary_candidates={0: (candidate,)},
    )

    response = client.post(
        f"/books/{scenario.book_id}/glossary/{candidate.id}/note",
        data={"note": "женский род"},
    )

    assert response.status_code == 409


def test_glossary_edit_is_rejected_after_processing_started(
    client: TestClient, application: Application
) -> None:
    """Once processing starts, the glossary is read-only with status code `409`."""
    candidate = glossary_candidate()
    scenario = store_book(
        application,
        status=BookStatus.PROCESSING,
        glossary_candidates={0: (candidate,)},
    )
    review = client.get(f"/books/{scenario.book_id}/glossary")

    response = client.post(
        f"/books/{scenario.book_id}/glossary/{candidate.id}",
        data={"action": "accept"},
    )

    assert review.status_code == 200
    assert review.json()["read_only"] is True
    assert response.status_code == 409
    assert "только для чтения" in response.json()["detail"]


def test_processing_start_is_rejected_while_review_is_incomplete(
    configured_client: TestClient, configured_application: Application
) -> None:
    """An unreviewed required term prevents initiating processing."""
    candidate = glossary_candidate()
    scenario = store_book(
        configured_application, glossary_candidates={0: (candidate,)},
        chapters=(("Mr. Bennet " * 5,),),
    )

    response = configured_client.post(f"/books/{scenario.book_id}/process")

    assert response.status_code == 409
    assert "не завершена" in response.json()["detail"]
    with configured_application.unit_of_work.transaction() as uow:
        assert uow.runs.active_for_book(scenario.book_id) is None
        assert uow.books.get(scenario.book_id).status is BookStatus.GLOSSARY_REVIEW


def test_processing_start_creates_run_and_chapter_tasks(
    configured_client: TestClient, configured_application: Application
) -> None:
    """Completed review permits processing start and schedules chapter tasks."""
    candidate = glossary_candidate()
    scenario = store_book(
        configured_application, glossary_candidates={0: (candidate,)}
    )
    accepted = configured_client.post(
        f"/books/{scenario.book_id}/glossary/{candidate.id}",
        data={"action": "accept"},
    )
    assert accepted.status_code == 200

    response = configured_client.post(f"/books/{scenario.book_id}/process")

    assert response.status_code == 202
    assert response.json()["book_id"] == scenario.book_id
    assert response.json()["task_ids"] == ["task-1"]
    with configured_application.unit_of_work.transaction() as uow:
        assert uow.runs.active_for_book(scenario.book_id) is not None
        assert uow.books.get(scenario.book_id).status is BookStatus.PROCESSING


def test_processing_start_without_configured_model_is_rejected(
    client: TestClient, application: Application
) -> None:
    """Without local model configuration, starting processing returns `409`."""
    scenario = store_book(application)

    response = client.post(f"/books/{scenario.book_id}/process")

    assert response.status_code == 409
    assert "LLM не настроена" in response.json()["detail"]


@pytest.mark.parametrize(
    "form",
    [
        pytest.param({}, id="missing"),
        pytest.param({"confirmed": "false"}, id="negative"),
    ],
)
def test_reset_requires_explicit_confirmation(
    client: TestClient, application: Application, form: dict[str, str]
) -> None:
    """Missing or negative confirmation does not alter the book."""
    scenario = store_book(application, status=BookStatus.PROCESSING)
    start_run(application, scenario)

    response = client.post(f"/books/{scenario.book_id}/reset", data=form)

    assert response.status_code == 422
    with application.unit_of_work.transaction() as uow:
        assert uow.runs.active_for_book(scenario.book_id) is not None
        assert uow.books.get(scenario.book_id).status is BookStatus.PROCESSING


def test_confirmed_reset_returns_book_to_glossary_review(
    client: TestClient, application: Application
) -> None:
    """Confirmed reset removes results and returns book to glossary review."""
    scenario = store_book(application, status=BookStatus.PROCESSING)
    start_run(
        application,
        scenario,
        results=(translated_result(scenario.block_ids()[0]),),
    )

    response = client.post(
        f"/books/{scenario.book_id}/reset", data={"confirmed": "true"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "book_id": scenario.book_id,
        "status": BookStatus.GLOSSARY_REVIEW.value,
        "removed_run_id": scenario.run_id,
        "removed_results": 1,
        "removed_tasks": 0,
    }
    with application.unit_of_work.transaction() as uow:
        assert uow.runs.active_for_book(scenario.book_id) is None
        assert uow.books.get(scenario.book_id).status is BookStatus.GLOSSARY_REVIEW


def test_unknown_glossary_resource_returns_404(
    client: TestClient, application: Application
) -> None:
    """Unknown book and concept return `404` rather than a state conflict."""
    scenario = store_book(application, glossary_candidates={0: (glossary_candidate(),)})

    assert client.get("/books/unknown/glossary").status_code == 404
    assert (
        client.post(
            f"/books/{scenario.book_id}/glossary/unknown",
            data={"action": "accept"},
        ).status_code
        == 404
    )
