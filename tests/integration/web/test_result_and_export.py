"""Block-level results, retries, and EPUB export (API-01, P4.T3)."""

from __future__ import annotations

from dataclasses import replace
from urllib.parse import quote

import pytest

from fastapi.testclient import TestClient
from web_scenario import (
    continue_legacy,
    SOURCE_TEXT,
    failed_task,
    hinted_result,
    repetition_warning,
    start_run,
    store_book,
    store_task,
    translated_result,
)

from linguary.application.book_overview import ResultBlockSummary
from linguary.application.ports.persistence import ExportValidationStatus
from linguary.composition import Application, drain_tasks
from linguary.web.routes.export import download_filename
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    BlockType,
    BookStatus,
    DocumentBlock,
    InlineAnnotation,
    InlineAnnotationType,
    ProcessingTask,
    SourceReference,
    TaskStatus,
    TaskType,
)
from linguary.web.schemas import ResultBlockResponse


def test_result_exposes_blocks_warnings_and_epubcheck_status(
    client: TestClient, application: Application
) -> None:
    """Result preserves block order and displays warnings and EPUBCheck status."""
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
    """Individual block is retrieved with its chapter number, treatment, and stored result."""
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
    """Unprocessed block is returned with planned treatment and without result."""
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
    """Unknown book does not resolve into an empty result."""
    response = client.get("/books/unknown/result")

    assert response.status_code == 404


def test_failed_task_retry_returns_pending_task(
    client: TestClient, application: Application
) -> None:
    """Retry clears task error and returns its new state."""
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
    """Unknown task retry is distinguished from a state conflict."""
    response = client.post("/tasks/unknown/retry")

    assert response.status_code == 404


def test_regeneration_returns_only_the_target_block(
    client: TestClient, application: Application
) -> None:
    """Regeneration schedules a task and returns fragment of the target block."""
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    block_ids = scenario.block_ids()
    start_run(
        application,
        scenario,
        results=tuple(translated_result(block_id) for block_id in block_ids),
    )

    continue_legacy(application, scenario.book_id)
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
    """After block modification, prior EPUBCheck status is invalidated."""
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

    continue_legacy(application, scenario.book_id)
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
    """Browser form does not render a raw HTML fragment or JSON instead of a page."""
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    block_ids = scenario.block_ids()
    start_run(
        application,
        scenario,
        results=tuple(translated_result(block_id) for block_id in block_ids),
    )

    continue_legacy(application, scenario.book_id)
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
    """Frontend receives a single formatted block without reloading the full page."""
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    block_ids = scenario.block_ids()
    start_run(
        application,
        scenario,
        results=tuple(translated_result(block_id) for block_id in block_ids),
    )

    continue_legacy(application, scenario.book_id)
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
    """Immediately after refresh request, block indicates that regeneration is running (QA-14).

    The block is marked busy, the button is disabled and reflects the in-progress state
    instead of remaining an active button that could easily be clicked again.
    """
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    block_ids = scenario.block_ids()
    start_run(
        application,
        scenario,
        results=tuple(translated_result(block_id) for block_id in block_ids),
    )

    continue_legacy(application, scenario.book_id)
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
    """Polling a single block returns the same partial template without scheduling a task."""
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
    """Regeneration of an unknown block does not schedule a background task."""
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    start_run(application, scenario)

    continue_legacy(application, scenario.book_id)
    response = client.post(
        f"/books/{scenario.book_id}/blocks/unknown/regenerate"
    )

    assert response.status_code == 404
    with application.unit_of_work.transaction() as uow:
        assert uow.tasks.list_for_book(scenario.book_id) == ()


def test_export_is_rejected_with_unfinished_mandatory_task(
    client: TestClient, application: Application
) -> None:
    """Pending mandatory task prevents scheduling export with `409`."""
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
    """Result header displays EPUB assembly and subsequently a download link."""
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
    """Before completion download is rejected; once ready, an EPUB attachment is served."""
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
    """Completed idempotency key does not prevent building the EPUB again."""
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
    """Downloading an unknown book returns `404`."""
    response = client.get("/books/unknown/download")

    assert response.status_code == 404


async def test_download_is_named_after_the_book_and_target_language(
    client: TestClient, application: Application
) -> None:
    """Downloaded book is named after book title and target language rather than `translated.epub`."""
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
    """Title from EPUB metadata does not break the filename beyond a single filename."""
    assert download_filename(title, "ru") == expected


def test_result_block_shows_footnote_marker_as_the_original_label() -> None:
    """Result screen displays footnote reference identically to the original label (QA-08)."""
    text = "Her face.[1] A pity."
    block = DocumentBlock.create(
        document_id="book-1",
        section_id="section-1",
        type=BlockType.PARAGRAPH,
        order=0,
        text=text,
        source_reference=SourceReference(
            resource_path="chapter1.xhtml", structural_path="/body/p[1]"
        ),
        inline_annotations=(
            InlineAnnotation(
                type=InlineAnnotationType.FOOTNOTE_REFERENCE,
                start=text.index("[1]"),
                end=text.index("[1]") + 3,
                target="chapter1.xhtml#fn1",
            ),
        ),
    )
    result = BlockProcessingResult(
        block_id=block.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Её лицо.[^1] Жаль.",
    )

    response = ResultBlockResponse.of_block(
        ResultBlockSummary(
            block=block,
            chapter_number=1,
            treatment=BlockTreatment.TRANSLATE,
            result=result,
        )
    )

    assert response.translated_text == "Её лицо.[1] Жаль."


def _split_result_scenario(application):
    from linguary.domain.learning import FragmentAssistance
    from linguary.domain.content_plan import SourceAnchor

    parts = ("  Hard phrase. ", "Simple sentence. Hard phrase.\n", "Separate.", "Other.")
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT, chapters=(parts,))
    section = scenario.section()
    blocks = tuple(replace(block, attributes={"split_group": "paragraph"}) if index in (0, 1, 3)
                   else block for index, block in enumerate(section.blocks))
    document = replace(scenario.document, sections=(replace(section, blocks=blocks),))
    with application.unit_of_work.transaction() as uow:
        analysis = uow.analyses.get_for_section(scenario.book_id, section.id)
        assert analysis is not None
        uow.documents.save(document)
        uow.analyses.save(scenario.book_id, analysis)
    scenario = replace(scenario, document=document)
    results = []
    for index, block in enumerate(blocks):
        items = ()
        if index < 2:
            start = block.text.index("Hard phrase")
            items = (FragmentAssistance(difficulty_id=f"difficulty-{index}", point_ids=(f"point-{index}",),
                      kind="lexical_hint", text=f"Help {index}",
                      anchors=(SourceAnchor(block.id, start, start + 11, "Hard phrase"),)),)
        results.append(replace(translated_result(block.id), fragment_assistances=items))
    start_run(application, scenario, results=results)
    return scenario, tuple(results), "".join(parts[:2])


def test_result_groups_only_adjacent_parts_and_keeps_exact_occurrences(client, application):
    scenario, results, original = _split_result_scenario(application)
    ids = scenario.block_ids()
    body = client.get(f"/books/{scenario.book_id}/result").json()
    groups = body["chapters"][0]["groups"]
    assert [group["member_block_ids"] for group in groups] == [list(ids[:2]), [ids[2]], [ids[3]]]
    assert groups[0]["original_text"] == original
    assert groups[0]["fragment_assistance"][0]["anchors"][0] == {
        "block_id": ids[0], "start": 2, "end": 13, "quote": "Hard phrase"}
    assert groups[0]["fragment_assistance"][1]["anchors"][0] == {
        "block_id": ids[1], "start": 17, "end": 28, "quote": "Hard phrase"}
    html = client.get(f"/books/{scenario.book_id}/result", headers={"accept": "text/html"}).text
    assert html.count('class="fragment-assistance"') == 1
    assert html.index(original) < html.index('class="fragment-assistance"')
    for block_id in ids[:2]:
        polled = client.get(f"/books/{scenario.book_id}/blocks/{block_id}")
        assert polled.json()["member_block_ids"] == list(ids[:2])
        assert polled.json()["original_text"] == original
        assert polled.json()["fragment_assistance"] == groups[0]["fragment_assistance"]


def test_group_marks_unresolved_help_without_changing_the_source(client, application):
    scenario, results, original = _split_result_scenario(application)
    item = replace(results[1].fragment_assistances[0], kind="unresolved", text=None)
    with application.unit_of_work.transaction() as uow:
        uow.results.replace_results(scenario.run_id, [results[0], replace(results[1], fragment_assistances=(item,)), *results[2:]])
    response = client.get(f"/books/{scenario.book_id}/blocks/{scenario.block_ids()[1]}")
    assert response.json()["original_text"] == original
    assert response.json()["fragment_assistance"][1]["kind"] == "unresolved"
    assert response.json()["fragment_assistance"][1]["text"] is None
    html = client.get(f"/books/{scenario.book_id}/result", headers={"accept": "text/html"}).text
    assert 'data-kind="unresolved"' in html
    assert "Трудность пока не объяснена" in html


@pytest.mark.parametrize("failed", [False, True], ids=["updated", "failed"])
def test_group_polling_preserves_siblings_during_member_regeneration(client, application, failed):
    scenario, results, original = _split_result_scenario(application)
    ids = scenario.block_ids()
    continue_legacy(application, scenario.book_id)
    headers = {"accept": "text/html", "x-linguary-fragment": "result-block"}
    pending = client.post(f"/books/{scenario.book_id}/blocks/{ids[1]}/regenerate", headers=headers)
    assert pending.status_code == 200, pending.text
    assert original in pending.text
    assert pending.text.count('class="fragment-assistance"') == 1
    assert 'data-active-task="true"' in pending.text
    with application.unit_of_work.transaction() as uow:
        task = next(task for task in uow.tasks.list_for_book(scenario.book_id)
                    if task.task_type is TaskType.REGENERATE_BLOCK)
        uow.tasks.save(replace(task, status=TaskStatus.FAILED if failed else TaskStatus.SUCCEEDED,
                               last_error=failed_task(scenario).last_error if failed else None))
        if not failed:
            updated = replace(results[1], fragment_assistances=(replace(results[1].fragment_assistances[0], text="Updated help"),))
            uow.results.replace_results(scenario.run_id, [results[0], updated, *results[2:]])
    for block_id in ids[:2]:
        response = client.get(f"/books/{scenario.book_id}/blocks/{block_id}", headers=headers)
        assert response.status_code == 200
        assert original in response.text
        assert 'data-active-task="true"' not in response.text
        assert response.text.count('class="fragment-assistance"') == 1
        assert "Help 0" in response.text
        assert ("Help 1" if failed else "Updated help") in response.text
        assert all(f'id="block-{member}"' in response.text for member in ids[:2])
        group = client.get(f"/books/{scenario.book_id}/blocks/{block_id}").json()
        assert group["member_block_ids"] == list(ids[:2])
        assert group["original_text"] == original
        assert [item["anchors"] for item in group["fragment_assistance"]] == [
            [{"block_id": anchor.block_id, "start": anchor.start, "end": anchor.end,
              "quote": anchor.quote} for anchor in result.fragment_assistances[0].anchors]
            for result in results[:2]
        ]


@pytest.fixture
def application(configured_application):
    return configured_application


@pytest.fixture
def client(configured_client):
    return configured_client
