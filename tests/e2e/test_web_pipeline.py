"""Сквозной сценарий MVP целиком через Web UI (PRD § 5, § 21, § 28.2).

Прикладной сквозной тест (`test_mvp_pipeline.py`) вызывает сценарии напрямую и проверяет
их связность. Здесь проверяется то, чего он не касается: пользователь проходит весь путь
только запросами к локальному приложению — загружает EPUB, запускает анализ, утверждает
глоссарий, запускает обработку, смотрит результат, собирает книгу и скачивает её. Ни одно
состояние в хранилище руками не кладётся: всё, что видит тест, получено предыдущим шагом
самого сценария.

Отдельно закрывается шаг `Validate` из PRD § 28.2: EPUBCheck подменён заглушкой, поэтому
экспорт доходит до внешней проверки и её разобранного отчёта, а не останавливается на
`not_validated`, как при отсутствующей Java.

Модель заменена `FakeLLMClient`, фоновый воркер выключен: задачи выполняются явным
`drain_tasks`, иначе результат зависел бы от того, кто успел первым.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from pipeline_support import (
    SYNTHETIC_EPUB,
    exported_text,
    install_epubcheck_stub,
    llm_replies,
    sequential_ids,
)

from linguary.adapters.config import (
    AppSettings,
    CheckSettings,
    LLMSettings,
    ServerSettings,
    StorageSettings,
)
from linguary.adapters.llm import FakeLLMClient
from linguary.adapters.validation import EpubCheckSettings, EpubCheckValidator
from linguary.application.context_builder import textual_blocks
from linguary.composition import Application, build_application, drain_tasks
from linguary.web import create_app

#: Языки книги: пара MVP (PRD § 4).
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "ru"


async def test_web_pipeline_from_upload_to_download(tmp_path: Path) -> None:
    """Пользователь проходит весь сценарий MVP через Web UI и скачивает готовый EPUB."""
    model = FakeLLMClient()
    stub = install_epubcheck_stub(tmp_path / "epubcheck")
    application = _application(tmp_path, model, epubcheck=stub.settings)
    try:
        with _http_client(application) as http:
            book_id = _upload(http)
            assert _book(http, book_id)["book"]["status"] == "IMPORTING"

            await drain_tasks(application)
            assert _book(http, book_id)["book"]["status"] == "IMPORTED"
            document = _document(application, book_id)
            model.replies = llm_replies(document)

            analysis = http.post(f"/books/{book_id}/analyze")
            assert analysis.status_code == 202
            await drain_tasks(application)
            assert _book(http, book_id)["book"]["status"] == "GLOSSARY_REVIEW"

            pending = http.get(f"/books/{book_id}/glossary").json()
            assert pending["is_complete"] is False
            assert pending["pending_count"] > 0
            assert "Wren Calder" in {item["source_text"] for item in pending["items"]}

            approved = http.post(f"/books/{book_id}/glossary/accept-all")
            assert approved.status_code == 200
            assert approved.json()["is_complete"] is True
            assert approved.json()["pending_count"] == 0

            started = http.post(f"/books/{book_id}/process")
            assert started.status_code == 202
            assert len(started.json()["task_ids"]) == len(document.ordered_sections)
            await drain_tasks(application)

            result = http.get(f"/books/{book_id}/result").json()
            assert result["status"] == "READY_TO_EXPORT"
            assert len(result["chapters"]) == len(document.ordered_sections)
            blocks = [
                block for chapter in result["chapters"] for block in chapter["blocks"]
            ]
            assert len(blocks) == sum(
                len(section.ordered_blocks) for section in document.ordered_sections
            )
            translated = [block for block in blocks if block["translated_text"]]
            assert len(translated) == sum(
                len(textual_blocks(section)) for section in document.ordered_sections
            )
            assert any(block["glossary_hints"] for block in translated)
            assert any(block["language_notes"] for block in translated)

            export = http.post(f"/books/{book_id}/export")
            assert export.status_code == 202
            await drain_tasks(application)

            validated = http.get(f"/books/{book_id}/result").json()
            assert validated["status"] == "COMPLETED"
            assert validated["export_validation_status"] == "passed"
            assert validated["epubcheck_hint"] is None
            assert stub.was_called
            assert str(application.paths.export_epub(book_id)) in stub.argv()

            download = http.get(f"/books/{book_id}/download")
            assert download.status_code == 200
            assert download.headers["content-type"] == "application/epub+zip"
            assert "attachment" in download.headers["content-disposition"]
            assert download.content.startswith(b"PK")

        book_text = exported_text(application.paths.export_epub(book_id))
        assert "Русский перевод блока" in book_text
        assert "Составное существительное передаёт образ света и стекла." in book_text
        assert "Оборот «не должно было быть»" in book_text
        assert "Previously in Chapter" in book_text
    finally:
        application.dispose()


async def test_web_pipeline_refuses_to_run_ahead_of_the_user(tmp_path: Path) -> None:
    """Шаги сценария нельзя переставить: каждый следующий закрыт до готовности книги."""
    model = FakeLLMClient()
    application = _application(tmp_path, model)
    try:
        with _http_client(application) as http:
            book_id = _upload(http)

            assert http.post(f"/books/{book_id}/process").status_code == 409
            assert http.post(f"/books/{book_id}/export").status_code == 409

            await drain_tasks(application)
            assert http.post(f"/books/{book_id}/process").status_code == 409

            document = _document(application, book_id)
            model.replies = llm_replies(document)
            http.post(f"/books/{book_id}/analyze")
            await drain_tasks(application)

            assert http.post(f"/books/{book_id}/process").status_code == 409
            assert http.get(f"/books/{book_id}/download").status_code == 409

            http.post(f"/books/{book_id}/glossary/accept-all")
            assert http.post(f"/books/{book_id}/process").status_code == 202
    finally:
        application.dispose()


def _application(
    tmp_path: Path, model: FakeLLMClient, *, epubcheck: EpubCheckSettings | None = None
) -> Application:
    """Приложение поверх временного каталога данных с настроенной фиктивной моделью."""
    settings = AppSettings(
        llm=LLMSettings(
            model="fake-model",
            api_key="test-key",
            max_attempts=2,
            max_unit_source_chars=1_000_000,
        ),
        checks=CheckSettings(),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
        server=ServerSettings(),
    )
    return build_application(
        settings=settings,
        llm_client=model,
        task_id_factory=sequential_ids("task"),
        epubcheck=(
            EpubCheckValidator(settings=epubcheck) if epubcheck is not None else None
        ),
    )


def _http_client(application: Application) -> TestClient:
    """Клиент локального Web UI без фонового воркера.

    Обращается по адресу самого приложения: сервер отвечает только по своим именам.
    """
    return TestClient(
        create_app(application=application, run_background_worker=False),
        base_url=application.settings.server.base_url,
    )


def _upload(http: TestClient) -> str:
    """Загружает синтетическую книгу через форму Web UI и отдаёт её идентификатор."""
    response = http.post(
        "/books/upload",
        files={
            "file": (
                SYNTHETIC_EPUB.name,
                SYNTHETIC_EPUB.read_bytes(),
                "application/epub+zip",
            )
        },
        data={
            "source_language": SOURCE_LANGUAGE,
            "target_language": TARGET_LANGUAGE,
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/books/")
    return location.removeprefix("/books/")


def _book(http: TestClient, book_id: str) -> dict:
    """Карточка книги, как её видит пользователь."""
    response = http.get(f"/books/{book_id}")
    assert response.status_code == 200
    return response.json()


def _document(application: Application, book_id: str):
    """Каноническая модель импортированной книги.

    Единственное обращение теста к хранилищу мимо HTTP: по числу глав и текстовых блоков
    готовятся ответы модели, и Web UI такого разреза книги не отдаёт.
    """
    with application.unit_of_work.transaction() as transaction:
        document = transaction.documents.get(book_id)
    assert document is not None
    return document
