"""Глоссарий, запуск обработки и возврат к проверке (API-01, P4.T2)."""

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
    """Экран глоссария показывает идентификатор и ожидающее решение первым."""
    candidate = glossary_candidate()
    scenario = store_book(application, glossary_candidates={0: (candidate,)})

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
                "confidence": 0.95,
                "model_flagged_review": False,
                "has_translation_conflict": False,
                "requires_review": True,
                "is_decided": False,
            }
        ],
    }


@pytest.mark.parametrize(
    ("action", "text", "expected_translation", "expected_status"),
    [
        pytest.param("accept", None, "мистер Беннет", "approved", id="accept"),
        pytest.param(
            "correct", "господин Беннет", "господин Беннет", "approved", id="correct"
        ),
        pytest.param("lock", None, "мистер Беннет", "locked", id="lock"),
        pytest.param("skip", None, "мистер Беннет", "suggested", id="skip"),
    ],
)
def test_glossary_decision_returns_updated_item(
    client: TestClient,
    application: Application,
    action: str,
    text: str | None,
    expected_translation: str,
    expected_status: str,
) -> None:
    """Каждое действие сохраняет решение и возвращает обновлённый элемент."""
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
    assert response.json()["is_decided"] is True


def test_glossary_correction_requires_text(
    client: TestClient, application: Application
) -> None:
    """Исправление без нового текста отклоняется как неполная HTTP-команда."""
    candidate = glossary_candidate()
    scenario = store_book(application, glossary_candidates={0: (candidate,)})

    response = client.post(
        f"/books/{scenario.book_id}/glossary/{candidate.id}",
        data={"action": "correct"},
    )

    assert response.status_code == 422


def test_glossary_edit_is_rejected_after_processing_started(
    client: TestClient, application: Application
) -> None:
    """После старта обработки глоссарий доступен только для чтения с кодом `409`."""
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
    """Необработанный важный термин не даёт открыть запуск обработки."""
    candidate = glossary_candidate()
    scenario = store_book(
        configured_application, glossary_candidates={0: (candidate,)}
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
    """Завершённая проверка открывает запуск и ставит задачи обработки глав."""
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
    """Без локальной настройки модели запуск обработки возвращает `409`."""
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
    """Отсутствующее или отрицательное подтверждение не изменяет книгу."""
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
    """Подтверждённый сброс удаляет результаты и возвращает проверку глоссария."""
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
    """Неизвестные книга и понятие дают `404`, а не конфликт состояния."""
    scenario = store_book(application, glossary_candidates={0: (glossary_candidate(),)})

    assert client.get("/books/unknown/glossary").status_code == 404
    assert (
        client.post(
            f"/books/{scenario.book_id}/glossary/unknown",
            data={"action": "accept"},
        ).status_code
        == 404
    )
