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
    assert response.json()["status_label"] == expected_label
    assert response.json()["is_decided"] is True
    with application.unit_of_work.transaction() as uow:
        assert uow.books.get(scenario.book_id).status is BookStatus.READY_TO_PROCESS


def test_glossary_decision_redirects_browser_back_to_review(
    client: TestClient, application: Application
) -> None:
    """HTML-действие по одному элементу оставляет пользователя на экране глоссария."""
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
    """Bulk-действие принимает все обязательные нерешённые элементы."""
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
    """HTML bulk-действие возвращает пользователя к глоссарию."""
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
    """Исправление без нового текста отклоняется как неполная HTTP-команда."""
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
    """Исправление принимает тип понятия и возвращает его в элементе (QA-11, T2)."""
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
    """Тип понятия, как и текст варианта, принимается только исправлением."""
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
    """Указание переводчику сохраняется отдельным действием и возвращается в элементе."""
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
    """После старта обработки указание тоже заморожено: глоссарий только для чтения."""
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
