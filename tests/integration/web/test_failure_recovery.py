"""Recovery after a failed task: user-facing representation and available actions.

LLM configuration errors can be fixed without re-uploading the file: analysis task is
retried via the retry button or by restarting analysis (QA-01). Books that cannot be
imported transition to "Import failed", the message explains the cause without internal
paths, and prompts to upload another file instead of retrying (QA-02).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from web_scenario import store_book

from linguary.adapters.llm import FakeLLMClient
from linguary.application.ports.llm import LLMConfigurationError
from linguary.composition import Application, drain_tasks
from linguary.domain import BookStatus, ErrorType, TaskStatus, TaskType

HTML_HEADERS = {"accept": "text/html"}


@pytest.fixture
def llm_client() -> FakeLLMClient:
    """Provider rejecting the key: simulates an expired or invalid API key."""
    return FakeLLMClient.always(
        LLMConfigurationError("провайдер fake не принял ключ или доступ к модели: 401")
    )


async def analysis_stopped_by_rejected_key(
    client: TestClient, application: Application
) -> str:
    """Launches book analysis where the first chapter fails on provider key error."""
    scenario = store_book(application, status=BookStatus.IMPORTED, analysed_chapters=())
    started = client.post(f"/books/{scenario.book_id}/analyze")
    assert started.status_code == 202
    finished = await drain_tasks(application)
    assert [task.status for task in finished] == [TaskStatus.FAILED]
    return scenario.book_id


async def test_rejected_key_fails_analysis_with_a_retryable_configuration_error(
    configured_client: TestClient, configured_application: Application
) -> None:
    """Key error is a configuration error: same book, retry is available (QA-01)."""
    book_id = await analysis_stopped_by_rejected_key(
        configured_client, configured_application
    )

    detail = configured_client.get(f"/books/{book_id}").json()

    assert detail["book"]["status"] == BookStatus.ANALYZING.value
    [failure] = detail["progress"]["failures"]
    assert failure["error_type"] == ErrorType.CONFIGURATION.value
    assert failure["is_retryable"] is True
    assert "проверьте ключ, модель и баланс" in failure["error_type_label"]


async def test_configuration_failure_is_retried_by_the_retry_button(
    configured_client: TestClient, configured_application: Application
) -> None:
    """Retry button puts task back in queue instead of returning 409 (QA-01)."""
    book_id = await analysis_stopped_by_rejected_key(
        configured_client, configured_application
    )
    [failure] = configured_client.get(f"/books/{book_id}").json()["progress"]["failures"]

    response = configured_client.post(f"/tasks/{failure['task_id']}/retry")

    assert response.status_code == 200
    assert response.json()["status"] == TaskStatus.PENDING.value


async def test_repeated_analysis_start_requeues_the_failed_chapter(
    configured_client: TestClient, configured_application: Application
) -> None:
    """Repeated "Start analysis" does not return stale failed task (QA-01)."""
    book_id = await analysis_stopped_by_rejected_key(
        configured_client, configured_application
    )

    response = configured_client.post(f"/books/{book_id}/analyze")

    assert response.status_code == 202
    assert response.json()["task_type"] == TaskType.ANALYZE_CHAPTER.value
    assert response.json()["status"] == TaskStatus.PENDING.value


async def test_card_offers_retry_and_configuration_after_rejected_key(
    configured_client: TestClient, configured_application: Application
) -> None:
    """Card does not pretend analysis is running and links to LLM settings (QA-01)."""
    book_id = await analysis_stopped_by_rejected_key(
        configured_client, configured_application
    )

    page = configured_client.get(f"/books/{book_id}", headers=HTML_HEADERS).text

    assert "Анализ остановлен ошибкой" in page
    assert "выполняется в фоне" not in page
    assert 'http-equiv="refresh"' not in page
    assert "disabled" not in page
    assert "Повторить" in page
    assert 'href="/system/config"' in page


def upload_file(client: TestClient, path: Path) -> str:
    """Uploads a file via upload form and returns book ID."""
    with path.open("rb") as file:
        response = client.post(
            "/books/upload",
            files={"file": (path.name, file, "application/epub+zip")},
            data={"source_language": "en", "target_language": "ru"},
            follow_redirects=False,
        )
    assert response.status_code == 303
    return response.headers["location"].rsplit("/", 1)[-1]


async def rejected_import(
    client: TestClient, application: Application, tmp_path: Path
) -> str:
    """Uploads an EPUB-named file that is not a zip archive and imports it."""
    broken = tmp_path / "broken.epub"
    broken.write_bytes(b"this is not a zip archive")
    book_id = upload_file(client, broken)
    finished = await drain_tasks(application)
    assert [task.status for task in finished] == [TaskStatus.FAILED]
    return book_id


async def test_unreadable_book_moves_to_import_failed(
    client: TestClient, application: Application, tmp_path: Path
) -> None:
    """Non-retryable import error transitions book to "Import failed" (QA-02)."""
    book_id = await rejected_import(client, application, tmp_path)

    detail = client.get(f"/books/{book_id}").json()

    assert detail["book"]["status"] == BookStatus.FAILED.value
    assert detail["book"]["status_label"] == "Импорт не удался"
    [failure] = detail["progress"]["failures"]
    assert failure["is_retryable"] is False


async def test_import_failure_message_hides_the_data_directory(
    client: TestClient, application: Application, tmp_path: Path
) -> None:
    """Import error message does not expose application data directory path (QA-02)."""
    book_id = await rejected_import(client, application, tmp_path)

    [failure] = client.get(f"/books/{book_id}").json()["progress"]["failures"]

    assert failure["message"].startswith("EPUB нельзя прочитать")
    assert str(application.paths.data_root) not in failure["message"]
    assert "source.epub" not in failure["message"]


async def test_import_failed_card_offers_another_file_instead_of_retry(
    client: TestClient, application: Application, tmp_path: Path
) -> None:
    """Instead of inactive retry button, card offers uploading another file (QA-02)."""
    book_id = await rejected_import(client, application, tmp_path)

    page = client.get(f"/books/{book_id}", headers=HTML_HEADERS).text

    assert "Импорт не удался" in page
    assert "Импорт выполняется в фоне" not in page
    assert "Повторить" not in page
    assert "/retry" not in page
    assert "Загрузить другой файл" in page
    assert 'http-equiv="refresh"' not in page
