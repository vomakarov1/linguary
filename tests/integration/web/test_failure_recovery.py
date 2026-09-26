"""Восстановление после неуспешной задачи: что видит пользователь и что может сделать.

Ошибка настройки LLM исправляется без нового файла: задача анализа повторяется кнопкой
или повторным запуском анализа (QA-01). Книга, которую импортировать нельзя, переходит
в «Импорт не удался», сообщение объясняет причину без внутренних путей, а вместо повтора
предлагается загрузить другой файл (QA-02).
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
    """Провайдер, отвергающий ключ: так выглядит просроченный или неверный ключ."""
    return FakeLLMClient.always(
        LLMConfigurationError("провайдер fake не принял ключ или доступ к модели: 401")
    )


async def analysis_stopped_by_rejected_key(
    client: TestClient, application: Application
) -> str:
    """Запускает анализ книги, первая глава которого падает на ключе провайдера."""
    scenario = store_book(application, status=BookStatus.IMPORTED, analysed_chapters=())
    started = client.post(f"/books/{scenario.book_id}/analyze")
    assert started.status_code == 202
    finished = await drain_tasks(application)
    assert [task.status for task in finished] == [TaskStatus.FAILED]
    return scenario.book_id


async def test_rejected_key_fails_analysis_with_a_retryable_configuration_error(
    configured_client: TestClient, configured_application: Application
) -> None:
    """Ошибка ключа — ошибка настройки: книга та же, повтор доступен (QA-01)."""
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
    """Кнопка «Повторить» возвращает задачу в очередь, а не отвечает 409 (QA-01)."""
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
    """Повторное «Начать анализ» не возвращает старую упавшую задачу (QA-01)."""
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
    """Карточка не притворяется, что анализ идёт, и ведёт к настройкам LLM (QA-01)."""
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
    """Загружает файл формой загрузки и возвращает идентификатор книги."""
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
    """Загружает файл с расширением EPUB, который архивом не является, и импортирует."""
    broken = tmp_path / "broken.epub"
    broken.write_bytes(b"this is not a zip archive")
    book_id = upload_file(client, broken)
    finished = await drain_tasks(application)
    assert [task.status for task in finished] == [TaskStatus.FAILED]
    return book_id


async def test_unreadable_book_moves_to_import_failed(
    client: TestClient, application: Application, tmp_path: Path
) -> None:
    """Неповторяемая ошибка импорта переводит книгу в «Импорт не удался» (QA-02)."""
    book_id = await rejected_import(client, application, tmp_path)

    detail = client.get(f"/books/{book_id}").json()

    assert detail["book"]["status"] == BookStatus.FAILED.value
    assert detail["book"]["status_label"] == "Импорт не удался"
    [failure] = detail["progress"]["failures"]
    assert failure["is_retryable"] is False


async def test_import_failure_message_hides_the_data_directory(
    client: TestClient, application: Application, tmp_path: Path
) -> None:
    """Сообщение об ошибке импорта не раскрывает путь к каталогу данных (QA-02)."""
    book_id = await rejected_import(client, application, tmp_path)

    [failure] = client.get(f"/books/{book_id}").json()["progress"]["failures"]

    assert failure["message"].startswith("EPUB нельзя прочитать")
    assert str(application.paths.data_root) not in failure["message"]
    assert "source.epub" not in failure["message"]


async def test_import_failed_card_offers_another_file_instead_of_retry(
    client: TestClient, application: Application, tmp_path: Path
) -> None:
    """Вместо неактивного «Повторить» карточка предлагает загрузить другой файл (QA-02)."""
    book_id = await rejected_import(client, application, tmp_path)

    page = client.get(f"/books/{book_id}", headers=HTML_HEADERS).text

    assert "Импорт не удался" in page
    assert "Импорт выполняется в фоне" not in page
    assert "Повторить" not in page
    assert "/retry" not in page
    assert "Загрузить другой файл" in page
    assert 'http-equiv="refresh"' not in page
