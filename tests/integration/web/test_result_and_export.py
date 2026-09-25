"""Результат по блокам, повторы и экспорт EPUB (API-01, P4.T3)."""

from __future__ import annotations

from dataclasses import replace
from urllib.parse import quote

import pytest

from fastapi.testclient import TestClient
from web_scenario import (
    SOURCE_TEXT,
    failed_task,
    hinted_result,
    repetition_warning,
    start_run,
    store_book,
    store_task,
    translated_result,
)

from linguary.application.ports.persistence import ExportValidationStatus
from linguary.composition import Application, drain_tasks
from linguary.web.routes.export import download_filename
from linguary.domain import (
    BlockTreatment,
    BookStatus,
    ProcessingTask,
    TaskStatus,
    TaskType,
)


def test_result_exposes_blocks_warnings_and_epubcheck_status(
    client: TestClient, application: Application
) -> None:
    """Результат сохраняет порядок блоков и показывает замечания и EPUBCheck."""
    scenario = store_book(
        application,
        status=BookStatus.COMPLETED,
        treatments={
            0: (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY),
        },
    )
    block_ids = scenario.block_ids()
    warning = repetition_warning()
    start_run(
        application,
        scenario,
        results=(
            translated_result(block_ids[0], warnings=(warning,)),
            hinted_result(block_ids[1]),
        ),
    )
    with application.unit_of_work.transaction() as uow:
        book = uow.books.get(scenario.book_id)
        assert book is not None
        uow.books.save(
            replace(
                book,
                export_validation_status=ExportValidationStatus.PASSED,
            )
        )

    response = client.get(f"/books/{scenario.book_id}/result")

    assert response.status_code == 200
    body = response.json()
    assert body["book_id"] == scenario.book_id
    assert body["status"] == BookStatus.COMPLETED.value
    assert body["export_validation_status"] == "passed"
    assert body["export_validation_status_label"] == "EPUBCheck пройден"
    assert body["epubcheck_hint"] is None
    assert body["warnings"]["total"] == 1
    assert body["warnings"]["result_warnings"][0]["block_id"] == block_ids[0]
    assert len(body["chapters"]) == 1
    blocks = body["chapters"][0]["blocks"]
    assert [block["block_id"] for block in blocks] == list(block_ids)
    assert blocks[0]["original_text"] == scenario.section().ordered_blocks[0].text
    assert blocks[0]["translated_text"] is not None
    assert blocks[0]["warnings"][0]["kind"] == "repetition"
    assert blocks[1]["translated_text"] is None
    assert blocks[1]["glossary_hints"][0]["original"] == "truth"


def test_result_block_keeps_its_chapter_number_and_planned_treatment(
    client: TestClient, application: Application
) -> None:
    """Отдельный блок читается со своим номером главы, режимом и сохранённым результатом."""
    scenario = store_book(
        application,
        status=BookStatus.COMPLETED,
        chapters=((SOURCE_TEXT, SOURCE_TEXT), (SOURCE_TEXT,)),
        treatments={1: (BlockTreatment.GLOSSARY_ONLY,)},
    )
    block_id = scenario.block_ids(1)[0]
    start_run(application, scenario, results=(hinted_result(block_id),))

    response = client.get(f"/books/{scenario.book_id}/blocks/{block_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["block_id"] == block_id
    assert body["section_id"] == scenario.section(1).id
    assert body["chapter_number"] == 2
    assert body["treatment"] == BlockTreatment.GLOSSARY_ONLY.value
    assert body["translated_text"] is None
    assert body["glossary_hints"][0]["original"] == "truth"


def test_result_block_without_stored_result_shows_only_its_plan(
    client: TestClient, application: Application
) -> None:
    """Ещё не обработанный блок отдаётся с режимом из плана и без результата."""
    scenario = store_book(application, status=BookStatus.PROCESSING)
    block_id = scenario.block_ids()[0]
    start_run(application, scenario)

    response = client.get(f"/books/{scenario.book_id}/blocks/{block_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["treatment"] == BlockTreatment.TRANSLATE.value
    assert body["original_text"] == SOURCE_TEXT
    assert body["translated_text"] is None
    assert body["language_notes"] == []
    assert body["warnings"] == []


def test_unknown_book_result_returns_404(client: TestClient) -> None:
    """Неизвестная книга не превращается в пустой результат."""
    response = client.get("/books/unknown/result")

    assert response.status_code == 404


def test_failed_task_retry_returns_pending_task(
    client: TestClient, application: Application
) -> None:
    """Повтор снимает ошибку задачи и возвращает её новое состояние."""
    scenario = store_book(application, status=BookStatus.PROCESSING)
    task = store_task(application, task=failed_task(scenario))

    response = client.post(f"/tasks/{task.id}/retry")

    assert response.status_code == 200
    assert response.json() == {
        "id": task.id,
        "book_id": scenario.book_id,
        "task_type": TaskType.PROCESS_TRANSLATION_UNIT.value,
        "task_type_label": "Обработка главы",
        "scope_id": scenario.section().id,
        "status": TaskStatus.PENDING.value,
        "status_label": "Ожидает выполнения",
        "attempts": 1,
        "is_retryable": False,
    }


def test_unknown_task_retry_returns_404(client: TestClient) -> None:
    """Неизвестная задача отличается от конфликта состояния."""
    response = client.post("/tasks/unknown/retry")

    assert response.status_code == 404


def test_regeneration_returns_only_the_target_block(
    client: TestClient, application: Application
) -> None:
    """Перегенерация ставит задачу и возвращает фрагмент выбранного блока."""
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    block_ids = scenario.block_ids()
    start_run(
        application,
        scenario,
        results=tuple(translated_result(block_id) for block_id in block_ids),
    )

    response = client.post(
        f"/books/{scenario.book_id}/blocks/{block_ids[0]}/regenerate"
    )

    assert response.status_code == 202
    body = response.json()
    assert body["block_id"] == block_ids[0]
    assert "blocks" not in body
    assert body["regeneration_task"] == {
        "id": "task-1",
        "book_id": scenario.book_id,
        "task_type": TaskType.REGENERATE_BLOCK.value,
        "task_type_label": "Повторная генерация блока",
        "scope_id": block_ids[0],
        "status": TaskStatus.PENDING.value,
        "status_label": "Ожидает выполнения",
        "attempts": 0,
        "is_retryable": False,
    }
    with application.unit_of_work.transaction() as uow:
        run = uow.runs.active_for_book(scenario.book_id)
        assert run is not None
        assert [result.block_id for result in uow.results.list_for_run(run.id)] == list(
            block_ids
        )


def test_regeneration_makes_the_previous_export_stale(
    client: TestClient, application: Application
) -> None:
    """После изменения блока старая отметка EPUBCheck больше не считается актуальной."""
    scenario = store_book(application, status=BookStatus.COMPLETED)
    block_ids = scenario.block_ids()
    start_run(
        application,
        scenario,
        results=tuple(translated_result(block_id) for block_id in block_ids),
    )
    with application.unit_of_work.transaction() as uow:
        book = uow.books.get(scenario.book_id)
        assert book is not None
        uow.books.save(
            replace(book, export_validation_status=ExportValidationStatus.PASSED)
        )

    response = client.post(
        f"/books/{scenario.book_id}/blocks/{block_ids[0]}/regenerate"
    )

    assert response.status_code == 202
    with application.unit_of_work.transaction() as uow:
        book = uow.books.get(scenario.book_id)
        assert book is not None
        assert book.status is BookStatus.PROCESSING
        assert book.export_validation_status is None


def test_html_regeneration_redirects_back_to_the_result_tab(
    client: TestClient, application: Application
) -> None:
    """Браузерная форма не открывает голый HTML-фрагмент или JSON вместо страницы."""
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    block_ids = scenario.block_ids()
    start_run(
        application,
        scenario,
        results=tuple(translated_result(block_id) for block_id in block_ids),
    )

    response = client.post(
        f"/books/{scenario.book_id}/blocks/{block_ids[0]}/regenerate",
        headers={"accept": "text/html"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == (
        f"/books/{scenario.book_id}/result#block-{block_ids[0]}"
    )


def test_fragment_regeneration_returns_the_updated_result_block(
    client: TestClient, application: Application
) -> None:
    """Фронтенд получает один оформленный блок без перезагрузки всей страницы."""
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    block_ids = scenario.block_ids()
    start_run(
        application,
        scenario,
        results=tuple(translated_result(block_id) for block_id in block_ids),
    )

    response = client.post(
        f"/books/{scenario.book_id}/blocks/{block_ids[0]}/regenerate",
        headers={
            "accept": "text/html",
            "x-linguary-fragment": "result-block",
        },
    )

    assert response.status_code == 200
    assert 'data-result-block' in response.text
    assert 'data-regenerate-form' in response.text
    assert "Повторная генерация блока" in response.text
    assert "Ожидает выполнения" in response.text
    assert 'data-active-task="true"' in response.text
    assert "<!doctype html>" not in response.text


def test_regenerating_block_shows_progress_right_after_refresh_is_requested(
    client: TestClient, application: Application
) -> None:
    """Сразу после «Обновить» блок показывает, что перегенерация идёт (QA-14).

    Блок помечен занятым, кнопка неактивна и называет происходящее, а не остаётся
    обычным «Обновить», по которому легко нажать ещё раз.
    """
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    block_ids = scenario.block_ids()
    start_run(
        application,
        scenario,
        results=tuple(translated_result(block_id) for block_id in block_ids),
    )

    response = client.post(
        f"/books/{scenario.book_id}/blocks/{block_ids[0]}/regenerate",
        headers={"accept": "text/html", "x-linguary-fragment": "result-block"},
    )

    assert 'aria-busy="true"' in response.text
    assert "<button type=\"submit\" disabled>Обновляется</button>" in response.text
    assert "Блок обновится автоматически" in response.text
    assert 'class="spinner"' in response.text


def test_fragment_result_block_polling_returns_the_current_block(
    client: TestClient, application: Application
) -> None:
    """Опрос одного блока возвращает тот же частичный шаблон без создания задачи."""
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    block_ids = scenario.block_ids()
    start_run(
        application,
        scenario,
        results=tuple(translated_result(block_id) for block_id in block_ids),
    )

    response = client.get(
        f"/books/{scenario.book_id}/blocks/{block_ids[0]}",
        headers={
            "accept": "text/html",
            "x-linguary-fragment": "result-block",
        },
    )

    assert response.status_code == 200
    assert f'id="block-{block_ids[0]}"' in response.text
    assert 'data-regenerate-form' in response.text
    assert "Задача:" not in response.text
    with application.unit_of_work.transaction() as uow:
        assert uow.tasks.list_for_book(scenario.book_id) == ()


def test_unknown_block_regeneration_returns_404(
    client: TestClient, application: Application
) -> None:
    """Перегенерация неизвестного блока не создаёт фоновую задачу."""
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    start_run(application, scenario)

    response = client.post(
        f"/books/{scenario.book_id}/blocks/unknown/regenerate"
    )

    assert response.status_code == 404
    with application.unit_of_work.transaction() as uow:
        assert uow.tasks.list_for_book(scenario.book_id) == ()


def test_export_is_rejected_with_unfinished_mandatory_task(
    client: TestClient, application: Application
) -> None:
    """Обязательная незавершённая задача блокирует постановку экспорта с `409`."""
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    start_run(
        application,
        scenario,
        results=tuple(
            translated_result(block_id) for block_id in scenario.block_ids()
        ),
    )
    store_task(
        application,
        task=ProcessingTask(
            id="task-processing",
            book_id=scenario.book_id,
            task_type=TaskType.PROCESS_TRANSLATION_UNIT,
            scope_id=scenario.section().id,
        ),
    )

    response = client.post(f"/books/{scenario.book_id}/export")

    assert response.status_code == 409
    assert "незавершённые обязательные задачи" in response.json()["detail"]
    with application.unit_of_work.transaction() as uow:
        export_tasks = [
            task
            for task in uow.tasks.list_for_book(scenario.book_id)
            if task.task_type is TaskType.EXPORT_BOOK
        ]
    assert export_tasks == []


async def test_html_result_header_tracks_export_until_download_is_available(
    client: TestClient, application: Application
) -> None:
    """Шапка результата показывает сборку EPUB и затем ссылку на скачивание."""
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    start_run(
        application,
        scenario,
        results=tuple(
            translated_result(block_id) for block_id in scenario.block_ids()
        ),
    )

    started = client.post(f"/books/{scenario.book_id}/export")
    pending = client.get(
        f"/books/{scenario.book_id}/result",
        headers={
            "accept": "text/html",
            "x-linguary-fragment": "result-header",
        },
    )
    await drain_tasks(application)
    completed = client.get(
        f"/books/{scenario.book_id}/result",
        headers={
            "accept": "text/html",
            "x-linguary-fragment": "result-header",
        },
    )

    assert started.status_code == 202
    assert "Сборка EPUB: Ожидает выполнения" in pending.text
    assert "Скачать EPUB" not in pending.text
    assert "Скачать EPUB" in completed.text
    assert f"/books/{scenario.book_id}/download" in completed.text


async def test_download_serves_epub_only_after_successful_export(
    client: TestClient, application: Application
) -> None:
    """До готовности скачивание запрещено, после неё возвращается EPUB-вложение."""
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    start_run(
        application,
        scenario,
        results=tuple(
            translated_result(block_id) for block_id in scenario.block_ids()
        ),
    )

    started = client.post(f"/books/{scenario.book_id}/export")
    pending_download = client.get(f"/books/{scenario.book_id}/download")
    finished = await drain_tasks(application)
    download = client.get(f"/books/{scenario.book_id}/download")

    assert started.status_code == 202
    assert started.json()["task_type"] == TaskType.EXPORT_BOOK.value
    assert pending_download.status_code == 409
    assert [task.status for task in finished] == [TaskStatus.SUCCEEDED]
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/epub+zip"
    assert "attachment" in download.headers["content-disposition"]
    assert download.content.startswith(b"PK")


async def test_repeated_export_replaces_the_completed_task(
    client: TestClient, application: Application
) -> None:
    """Завершённый идемпотентный ключ не мешает собрать EPUB повторно."""
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    start_run(
        application,
        scenario,
        results=tuple(
            translated_result(block_id) for block_id in scenario.block_ids()
        ),
    )
    first = client.post(f"/books/{scenario.book_id}/export")
    await drain_tasks(application)

    second = client.post(f"/books/{scenario.book_id}/export")

    assert first.json()["id"] == "task-1"
    assert second.status_code == 202
    assert second.json()["id"] == "task-2"
    assert client.get(f"/books/{scenario.book_id}/download").status_code == 409


def test_unknown_book_download_returns_404(client: TestClient) -> None:
    """Скачивание неизвестной книги возвращает `404`."""
    response = client.get("/books/unknown/download")

    assert response.status_code == 404


async def test_download_is_named_after_the_book_and_target_language(
    client: TestClient, application: Application
) -> None:
    """Скачанная книга называется по названию и целевому языку, а не `translated.epub`."""
    scenario = store_book(
        application, status=BookStatus.READY_TO_EXPORT, title="The Keeper's Light"
    )
    start_run(
        application,
        scenario,
        results=tuple(
            translated_result(block_id) for block_id in scenario.block_ids()
        ),
    )
    client.post(f"/books/{scenario.book_id}/export")
    await drain_tasks(application)

    download = client.get(f"/books/{scenario.book_id}/download")

    assert download.status_code == 200
    disposition = download.headers["content-disposition"]
    expected = quote("The Keeper's Light (ru).epub")
    assert f"filename*=utf-8''{expected}" in disposition
    assert "translated.epub" not in disposition


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        pytest.param("The Keeper's Light", "The Keeper's Light (ru).epub", id="plain"),
        pytest.param("../../etc/passwd", "etc passwd (ru).epub", id="path-like"),
        pytest.param('A: "B"?\n', "A B (ru).epub", id="reserved-characters"),
        pytest.param(" ... ", "book (ru).epub", id="nothing-left"),
    ],
)
def test_download_filename_is_safe_for_the_file_system(title: str, expected: str) -> None:
    """Название из метаданных EPUB не выводит имя файла за пределы одного имени."""
    assert download_filename(title, "ru") == expected
