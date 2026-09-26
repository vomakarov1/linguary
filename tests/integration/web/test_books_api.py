"""Список книг и карточка книги: состояние, прогресс и ошибки (API-01, PRD § 21.1, 21.4)."""

from __future__ import annotations

from datetime import UTC, datetime

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
from linguary.domain import (
    BlockTreatment,
    BookStatus,
    ErrorType,
    LLMCallLog,
    LLMCallStage,
    ProcessingTask,
    TaskType,
)

HTML_HEADERS = {"accept": "text/html"}


def store_call(
    application: Application,
    *,
    task_id: str,
    duration_ms: int,
    stage: LLMCallStage = LLMCallStage.TRANSLATION_UNIT,
    call_id: str = "call-1",
) -> None:
    """Запись журнала об успешном вызове модели задачей книги."""
    with application.unit_of_work.transaction() as uow:
        uow.llm_calls.add(
            LLMCallLog(
                id=call_id,
                book_id="book-web-1",
                stage=stage,
                provider="fake",
                model="fake-model",
                started_at=datetime(2026, 9, 26, 12, 0, tzinfo=UTC),
                duration_ms=duration_ms,
                attempt=1,
                success=True,
                task_id=task_id,
            )
        )


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


def test_book_detail_lists_every_running_chapter(
    client: TestClient, application: Application
) -> None:
    """При параллельной обработке карточка показывает все выполняющиеся главы (QA-13)."""
    scenario = store_book(
        application,
        status=BookStatus.PROCESSING,
        chapters=(("Первый абзац.",), ("Второй абзац.",), ("Третий абзац.",)),
    )
    store_task(application, task=running_task(scenario, task_id="t-3", section_index=2))
    store_task(application, task=running_task(scenario, task_id="t-1", section_index=0))

    progress = client.get(f"/books/{scenario.book_id}").json()["progress"]
    page = client.get(f"/books/{scenario.book_id}", headers=HTML_HEADERS).text

    assert progress["running_chapters"] == [
        {"chapter_number": 1, "title": "Chapter 1"},
        {"chapter_number": 3, "title": "Chapter 3"},
    ]
    assert "Chapter 1" in page
    assert "Chapter 3" in page


def test_book_detail_promises_an_estimate_after_the_first_chapter(
    client: TestClient, application: Application
) -> None:
    """До первого вызова модели оценки нет — карточка говорит, когда она появится."""
    scenario = store_book(
        application, status=BookStatus.ANALYZING, analysed_chapters=()
    )
    store_task(
        application,
        task=running_task(scenario, task_type=TaskType.ANALYZE_CHAPTER),
    )

    progress = client.get(f"/books/{scenario.book_id}").json()["progress"]
    page = client.get(f"/books/{scenario.book_id}", headers=HTML_HEADERS).text

    assert progress["remaining_seconds"] is None
    assert progress["remaining_time_label"] == (
        "Оценка оставшегося времени появится после первой главы"
    )
    assert "появится после первой главы" in page


def test_book_detail_estimates_remaining_processing_time(
    client: TestClient, application: Application
) -> None:
    """Оценка по медиане вызовов книги × оставшаяся работа ÷ параллелизм (QA-13)."""
    scenario = store_book(
        application,
        status=BookStatus.PROCESSING,
        chapters=(("Первый абзац.", "Второй абзац."), ("Третий абзац.", "Четвёртый.")),
    )
    start_run(
        application,
        scenario,
        results=tuple(translated_result(block_id) for block_id in scenario.block_ids(0)),
    )
    done = ProcessingTask(
        id="t-done",
        book_id=scenario.book_id,
        task_type=TaskType.PROCESS_TRANSLATION_UNIT,
        scope_id=scenario.section(0).id,
    ).start().succeed()
    store_task(application, task=done)
    store_task(application, task=running_task(scenario, section_index=1))
    store_call(application, task_id="t-done", duration_ms=150_000)

    progress = client.get(f"/books/{scenario.book_id}").json()["progress"]
    page = client.get(f"/books/{scenario.book_id}", headers=HTML_HEADERS).text

    assert progress["remaining_seconds"] == 150
    assert progress["remaining_time_label"] == "Осталось примерно 3 мин"
    assert "Осталось примерно 3 мин" in page


def test_book_detail_ignores_calls_of_a_cancelled_run(
    client: TestClient, application: Application
) -> None:
    """Вызовы задач, удалённых возвратом к проверке глоссария, в оценку не входят."""
    scenario = store_book(application, status=BookStatus.PROCESSING)
    start_run(application, scenario)
    store_task(application, task=running_task(scenario))
    store_call(application, task_id="task-of-a-cancelled-run", duration_ms=60_000)

    progress = client.get(f"/books/{scenario.book_id}").json()["progress"]

    assert progress["remaining_seconds"] is None


def test_finished_book_has_no_estimate(
    client: TestClient, application: Application
) -> None:
    """Книга, над которой ничего не выполняется, оценки не получает."""
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)

    progress = client.get(f"/books/{scenario.book_id}").json()["progress"]

    assert progress["remaining_time_label"] is None


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
