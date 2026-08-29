"""Список книг и карточка книги: состояние, прогресс и ошибки (API-01, PRD § 21.1, 21.4)."""

from __future__ import annotations

from fastapi.testclient import TestClient
from web_scenario import (
    dropped_block_warning,
    failed_task,
    hinted_result,
    running_task,
    section_id_of,
    start_run,
    store_book,
    store_task,
    translated_result,
)

from linguary.composition import Application
from linguary.domain import BlockTreatment, BookStatus, ErrorType


def test_books_list_is_empty_before_the_first_upload(client: TestClient) -> None:
    """Пустое хранилище даёт пустой список книг, а не отказ."""
    response = client.get("/books")

    assert response.status_code == 200
    assert response.json() == {"books": []}


def test_books_list_shows_status_in_russian(
    client: TestClient, application: Application
) -> None:
    """Список книг называет состояние по-русски (PRD § 21.1)."""
    store_book(application, status=BookStatus.GLOSSARY_REVIEW)

    books = client.get("/books").json()["books"]

    assert len(books) == 1
    assert books[0]["status"] == BookStatus.GLOSSARY_REVIEW.value
    assert books[0]["status_label"] == "Ожидает проверки глоссария"
    assert books[0]["title"] == "Pride and Prejudice"


def test_books_list_filters_by_status(
    client: TestClient, application: Application
) -> None:
    """Фильтр `status` оставляет только книги в этом состоянии."""
    store_book(application, book_id="book-a", status=BookStatus.GLOSSARY_REVIEW)
    store_book(application, book_id="book-b", status=BookStatus.COMPLETED)

    completed = client.get("/books", params={"status": BookStatus.COMPLETED.value})

    assert [book["id"] for book in completed.json()["books"]] == ["book-b"]


def test_unknown_status_filter_is_rejected(client: TestClient) -> None:
    """Несуществующее состояние в запросе отвергается проверкой входа."""
    response = client.get("/books", params={"status": "SLEEPING"})

    assert response.status_code == 422


def test_book_detail_reports_progress_and_treatment_counts(
    client: TestClient, application: Application
) -> None:
    """Карточка книги показывает прогресс и счётчики режимов (PRD § 21.4)."""
    scenario = store_book(
        application,
        status=BookStatus.PROCESSING,
        chapters=(("Первый абзац.", "Второй абзац."),),
        treatments={0: (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY)},
    )
    block_ids = scenario.block_ids()
    start_run(
        application,
        scenario,
        results=(translated_result(block_ids[0]),),
    )

    progress = client.get(f"/books/{scenario.book_id}").json()["progress"]

    assert progress["planned_blocks"] == 2
    assert progress["processed_blocks"] == 1
    assert progress["percent_complete"] == 50
    assert progress["treatment_counts"] == [
        {"treatment": BlockTreatment.TRANSLATE.value, "label": "Полный перевод", "count": 1},
        {
            "treatment": BlockTreatment.GLOSSARY_ONLY.value,
            "label": "Словарные подсказки",
            "count": 1,
        },
    ]


def test_book_detail_counts_processed_blocks_of_every_treatment(
    client: TestClient, application: Application
) -> None:
    """Обработанным считается и блок со словарными подсказками (PRD § 10.1)."""
    scenario = store_book(
        application,
        status=BookStatus.PROCESSING,
        treatments={0: (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY)},
    )
    block_ids = scenario.block_ids()
    start_run(
        application,
        scenario,
        results=(translated_result(block_ids[0]), hinted_result(block_ids[1])),
    )

    progress = client.get(f"/books/{scenario.book_id}").json()["progress"]

    assert progress["processed_blocks"] == 2
    assert progress["percent_complete"] == 100


def test_book_detail_names_the_current_chapter(
    client: TestClient, application: Application
) -> None:
    """Выполняющаяся задача главы показывается как текущая глава (PRD § 21.4)."""
    scenario = store_book(
        application,
        status=BookStatus.PROCESSING,
        chapters=(("Первый абзац.",), ("Второй абзац.",)),
    )
    store_task(application, task=running_task(scenario, section_index=1))

    progress = client.get(f"/books/{scenario.book_id}").json()["progress"]

    assert progress["current_chapter"] == 2
    assert progress["current_chapter_title"] == "Chapter 2"
    assert progress["running_tasks"] == 1


def test_book_detail_reports_failed_task_with_retry_flag(
    client: TestClient, application: Application
) -> None:
    """Неуспешная задача показывается с русским объяснением и признаком повтора."""
    scenario = store_book(application, status=BookStatus.PROCESSING)
    store_task(application, task=failed_task(scenario))

    failures = client.get(f"/books/{scenario.book_id}").json()["progress"]["failures"]

    assert len(failures) == 1
    assert failures[0]["task_type_label"] == "Обработка главы"
    assert failures[0]["error_type"] == ErrorType.TRANSIENT.value
    assert failures[0]["error_type_label"] == "Временный сбой — попробуйте повторить"
    assert failures[0]["message"] == "провайдер не ответил"
    assert failures[0]["is_retryable"] is True


def test_permanent_failure_is_not_retryable(
    client: TestClient, application: Application
) -> None:
    """Постоянная ошибка повтору не подлежит: нужен другой файл (PRD § 16)."""
    scenario = store_book(application, status=BookStatus.PROCESSING)
    store_task(
        application,
        task=failed_task(
            scenario, error_type=ErrorType.PERMANENT, message="книга нечитаема"
        ),
    )

    failures = client.get(f"/books/{scenario.book_id}").json()["progress"]["failures"]

    assert failures[0]["is_retryable"] is False
    assert failures[0]["error_type_label"] == "Ошибка, которую повтор не исправит"


def test_book_detail_counts_import_warnings(
    client: TestClient, application: Application
) -> None:
    """Карточка книги сообщает, сколько предупреждений оставил импорт (PRD § 21.6)."""
    scenario = store_book(
        application,
        status=BookStatus.PROCESSING,
        import_warnings=(dropped_block_warning(section_id_of("book-web-1", 0)),),
    )

    detail = client.get(f"/books/{scenario.book_id}").json()

    assert detail["warning_count"] == 1
    assert detail["book"]["source_language"] == "en"
    assert detail["book"]["target_language"] == "ru"


def test_unknown_book_gives_404(client: TestClient) -> None:
    """Карточка несуществующей книги — `404`, а не пустой ответ."""
    response = client.get("/books/unknown")

    assert response.status_code == 404
    assert "unknown" in response.json()["detail"]
