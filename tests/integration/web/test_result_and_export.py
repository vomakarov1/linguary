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


@pytest.mark.parametrize('fragment', ['book', 'html', 'block', 'result-header'])
def test_result_tasks_are_loaded_once_without_per_member_reads(
    client, application, monkeypatch, fragment,
):
    """Result and group polling share one task batch; header polling skips block states."""
    from linguary.web.routes import result as result_route

    scenario, _, _ = _split_result_scenario(application)
    ids = scenario.block_ids()
    for index, task_status in enumerate((TaskStatus.PENDING, TaskStatus.FAILED, TaskStatus.SUCCEEDED)):
        store_task(application, task=ProcessingTask(
            id=f'regeneration-{index}', book_id=scenario.book_id,
            task_type=TaskType.REGENERATE_BLOCK, scope_id=ids[index], status=task_status,
            last_error=failed_task(scenario).last_error if task_status is TaskStatus.FAILED else None,
        ))
    with application.unit_of_work.transaction() as tx:
        repository_type = type(tx.tasks)
    original_list = repository_type.list_for_book
    batches = []

    def list_once(repository, book_id):
        batches.append(book_id)
        return original_list(repository, book_id)

    def no_individual_read(*args, **kwargs):
        raise AssertionError('Result must not load task states per block')

    monkeypatch.setattr(repository_type, 'list_for_book', list_once)
    monkeypatch.setattr(repository_type, 'find_by_identity', no_individual_read)
    headers = {}
    endpoint = f'/books/{scenario.book_id}/result'
    if fragment == 'result-header':
        monkeypatch.setattr(result_route, '_regeneration_tasks', no_individual_read)
        headers = {'accept': 'text/html', 'x-linguary-fragment': fragment}
    elif fragment == 'block':
        endpoint = f'/books/{scenario.book_id}/blocks/{ids[0]}'
    elif fragment == 'html':
        headers = {'accept': 'text/html'}
    response = client.get(endpoint, headers=headers)
    assert response.status_code == 200
    assert batches == [scenario.book_id]
    if fragment == 'result-header':
        assert 'data-result-header' in response.text
        assert 'data-result-block' not in response.text
    elif fragment == 'html':
        assert 'data-result-block' in response.text
    else:
        groups = [response.json()] if fragment == 'block' else response.json()['chapters'][0]['groups']
        members = [member for group in groups for member in group['member_blocks']]
        tasks = {member['block_id']: member['regeneration_task'] for member in members}
        assert tasks[ids[0]]['status'] == 'pending'
        assert tasks[ids[1]]['status'] == 'failed'
        if fragment == 'book':
            assert tasks[ids[2]] is None
            assert tasks[ids[3]] is None


def _pronunciation_results(application):
    from linguary.domain.content_plan import SourceAnchor
    from linguary.domain.pronunciation import PronunciationAssistance, PronunciationEvidence

    scenario = store_book(application, status=BookStatus.COMPLETED, chapters=(('zorp',), ('zorp',)))
    results = []
    for section in scenario.document.ordered_sections:
        block, = section.ordered_blocks
        item = PronunciationAssistance(SourceAnchor(block.id, 0, 4, 'zorp'), 'zorp', 'zɔːp', 'uk', f'ph-synthetic-{block.id}',
            PronunciationEvidence('private-source', 'private-version', 'entry-1', 'private-sense', 'verb',
                                  'zorp', 'zɔːp', 'Invented test data'))
        results.append(BlockProcessingResult(block_id=block.id, treatment=BlockTreatment.READ_AS_IS,
                                             pronunciation_assistances=(item,)))
    start_run(application, scenario, results=tuple(reversed(results)))
    return scenario


def test_pronunciation_notices_route_is_offline_and_allowlisted(client, monkeypatch):
    import socket
    from linguary.adapters.pronunciation.source import SOURCE_VERSION
    from linguary.composition import container

    def no_network(*args, **kwargs):
        raise AssertionError('Source notices must not use the network')

    monkeypatch.setattr(socket, 'create_connection', no_network)
    response = client.get('/pronunciation-sources')
    assert response.status_code == 200
    assert response.headers['content-type'] == 'text/plain; charset=utf-8'
    assert SOURCE_VERSION in response.text
    assert all(text in response.text for text in ('MIT License', 'GNU GENERAL PUBLIC LICENSE', 'en_UK.txt', 'en_US.txt'))
    assert client.get('/pronunciation-sources?filename=/etc/passwd&url=https://example.com').text == response.text
    assert client.get('/pronunciation-sources/manifest.json').status_code == 404
    from linguary.adapters.epub import pronunciation_notices
    with monkeypatch.context() as corrupt:
        corrupt.setitem(pronunciation_notices._NOTICE_HASHES, 'NOTICE.md', '0' * 64)
        assert client.get('/pronunciation-sources').status_code == 503

    def unavailable():
        raise OSError('/private/internal/LICENSES.txt')

    monkeypatch.setattr(container, 'load_pronunciation_source_notices', unavailable, raising=False)
    failed = client.get('/pronunciation-sources')
    assert failed.status_code == 503
    assert '/private/' not in failed.text


def test_pronunciation_notice_link_and_escaped_attribution(client, application):
    from lxml import html

    scenario = _pronunciation_results(application)
    block_id = scenario.block_ids()[0]
    with application.unit_of_work.transaction() as tx:
        result = tx.results.get(scenario.run_id, block_id)
        item, = result.pronunciation_assistances
        tx.results.replace_results(scenario.run_id, [replace(result, pronunciation_assistances=(replace(
            item, evidence=replace(item.evidence, attribution='<script id="source-injection">bad</script>'),
        ),))])
    response = client.get(f'/books/{scenario.book_id}/result', headers={'accept': 'text/html'})
    assert response.status_code == 200
    tree = html.fromstring(response.text)
    assert not tree.xpath('//script[@id="source-injection"]')
    assert '<script id="source-injection">bad</script>' in ''.join(tree.itertext())
    links = tree.xpath('//details[contains(@class,"ipa-guide")]/following-sibling::p[1]/a[@class="pronunciation-source-notices"]')
    assert links and all(link.get('href').endswith('/pronunciation-sources') for link in links)


def test_legacy_pronunciation_attribution_still_visible(client, application):
    scenario = _pronunciation_results(application)
    response = client.get(f'/books/{scenario.book_id}/result', headers={'accept': 'text/html'})
    assert response.status_code == 200
    assert 'Invented test data' in response.text and 'entry-1' in response.text


def test_pronunciation_result_and_fragment_contract(client, application, monkeypatch):
    from linguary.web.routes import result as route
    captured = []
    original_render = route.render

    def render(*args, **kwargs):
        captured.append(kwargs.get('block') or kwargs.get('data'))
        return original_render(*args, **kwargs)

    monkeypatch.setattr(route, 'render', render)
    scenario = _pronunciation_results(application)
    full = client.get(f'/books/{scenario.book_id}/result').json()
    primary = full['chapters'][0]['blocks'][0]['pronunciation_assistances'][0]
    repeated = full['chapters'][1]['blocks'][0]['pronunciation_assistances'][0]
    assert primary['is_primary'] is True and repeated['is_primary'] is False
    assert primary['hint_id'] == repeated['hint_id']
    assert primary['source_ref'] == 'entry-1' and primary['source_label'] == 'Invented test data'
    assert 'private-' not in str(full)
    identifier = scenario.block_ids(1)[0]
    fragment = client.get(f'/books/{scenario.book_id}/blocks/{identifier}').json()
    assert fragment['pronunciation_assistances'] == [repeated]
    assert fragment['member_blocks'][0]['pronunciation_assistances'] == [repeated]
    response = client.get(f'/books/{scenario.book_id}/blocks/{identifier}',
                          headers={'accept': 'text/html', 'x-linguary-fragment': 'result-block'})
    assert response.status_code == 200
    assert captured[-1].model_dump(mode='json')['pronunciation_assistances'] == [repeated]
    assert fragment['original_text'] == 'zorp'


def test_pronunciation_fragment_uses_book_snapshot(client, application, monkeypatch):
    from linguary.web.routes import result as route
    scenario = _pronunciation_results(application)
    identifier = scenario.block_ids(1)[0]
    store_task(application, task=ProcessingTask(id='pronunciation-regeneration', book_id=scenario.book_id,
        task_type=TaskType.REGENERATE_BLOCK, scope_id=identifier))
    original = route.book_result
    snapshots = []

    def snapshot(*args):
        summary = original(*args)
        snapshots.append(summary)
        with application.unit_of_work.transaction() as tx:
            tx.results.replace_results(scenario.run_id, [BlockProcessingResult(
                block_id=scenario.block_ids()[0], treatment=BlockTreatment.READ_AS_IS)])
        return summary

    monkeypatch.setattr(route, 'book_result', snapshot)
    response = client.get(f'/books/{scenario.book_id}/blocks/{identifier}')
    assert response.status_code == 200 and len(snapshots) == 1
    assert response.json()['pronunciation_assistances'][0]['is_primary'] is False
    assert response.json()['regeneration_task']['id'] == 'pronunciation-regeneration'


GROUPS_HEADERS = {'accept': 'text/html', 'x-linguary-fragment': 'result-groups'}


def _groups_tree(client, scenario):
    from lxml import html
    response = client.get(f'/books/{scenario.book_id}/result', headers=GROUPS_HEADERS)
    assert response.status_code == 200
    tree = html.fromstring(response.text)
    assert tree.get('data-result-groups') is not None
    assert tree.get('data-result-url') == f'/books/{scenario.book_id}/result'
    for link in tree.xpath('.//a[@class="pronunciation-repeat"]'):
        assert len(tree.xpath('//*[@id=$target]', target=link.get('href')[1:])) == 1
    return tree


def test_result_groups_fragment_matches_full_book_snapshot(client, application, monkeypatch):
    from lxml import html
    from linguary.web.routes import result as route
    from linguary.web import schemas

    scenario = _pronunciation_results(application)
    endpoint = f'/books/{scenario.book_id}/result'
    full = html.fromstring(client.get(endpoint, headers={'accept': 'text/html'}).text)
    expected, = full.xpath('//*[@data-result-groups]')
    original = route.book_result
    original_projection = schemas.project_pronunciations
    snapshots, projections = [], []

    def snapshot(*args):
        summary = original(*args)
        snapshots.append(summary)
        with application.unit_of_work.transaction() as tx:
            results = tx.results.list_for_run(scenario.run_id)
            tx.results.replace_results(scenario.run_id, [
                replace(result, pronunciation_assistances=()) if result.block_id == scenario.block_ids()[0]
                else result for result in results])
        return summary

    def project(*args, **kwargs):
        projections.append(1)
        return original_projection(*args, **kwargs)

    monkeypatch.setattr(route, 'book_result', snapshot)
    monkeypatch.setattr(schemas, 'project_pronunciations', project)
    actual = _groups_tree(client, scenario)
    assert len(snapshots) == len(projections) == 1
    assert html.tostring(actual, with_tail=False) == html.tostring(expected, with_tail=False)
    refreshed = _groups_tree(client, scenario)
    assert not refreshed.xpath('.//a[@class="pronunciation-repeat"]')
    assert len(refreshed.xpath('.//*[@class="pronunciation"]')) == 1


def test_pronunciation_groups_refresh_promotes_later_primary(client, application):
    scenario = _pronunciation_results(application)
    before = _groups_tree(client, scenario)
    old_id = before.xpath('.//*[@class="pronunciation"]')[0].get('id')
    with application.unit_of_work.transaction() as tx:
        results = tx.results.list_for_run(scenario.run_id)
        tx.results.replace_results(scenario.run_id, [
            replace(result, pronunciation_assistances=()) if result.block_id == scenario.block_ids()[0]
            else result for result in results])
    after = _groups_tree(client, scenario)
    primary, = after.xpath('.//*[@class="pronunciation"]')
    assert primary.get('id') != old_id
    assert primary.xpath('ancestor::article')[0].get('id') == f'block-{scenario.block_ids(1)[0]}'
    assert not after.xpath('.//a[@class="pronunciation-repeat"]')


def test_pronunciation_groups_refresh_demotes_old_primary(client, application):
    scenario = _pronunciation_results(application)
    with application.unit_of_work.transaction() as tx:
        results = tx.results.list_for_run(scenario.run_id)
        tx.results.replace_results(scenario.run_id, [
            replace(result, pronunciation_assistances=()) if result.block_id == scenario.block_ids()[0]
            else result for result in results])
    before = _groups_tree(client, scenario)
    assert before.xpath('.//*[@class="pronunciation"]')[0].xpath('ancestor::article')[0].get('id') == f'block-{scenario.block_ids(1)[0]}'
    with application.unit_of_work.transaction() as tx:
        tx.results.replace_results(scenario.run_id, list(results))
    after = _groups_tree(client, scenario)
    primary, = after.xpath('.//*[@class="pronunciation"]')
    repeated, = after.xpath('.//a[@class="pronunciation-repeat"]')
    assert primary.xpath('ancestor::article')[0].get('id') == f'block-{scenario.block_ids()[0]}'
    assert repeated.get('href') == f'#{primary.get("id")}'


def test_pronunciation_groups_refresh_splits_changed_ipa(client, application):
    scenario = _pronunciation_results(application)
    with application.unit_of_work.transaction() as tx:
        results = tx.results.list_for_run(scenario.run_id)
        updated = []
        for result in results:
            if result.block_id == scenario.block_ids()[0]:
                item, = result.pronunciation_assistances
                result = replace(result, pronunciation_assistances=(replace(
                    item, ipa='zɒp', hint_id='ph-changed-synthetic', evidence=replace(item.evidence, dictionary_ipa='zɒp')),))
            updated.append(result)
        tx.results.replace_results(scenario.run_id, updated)
    after = _groups_tree(client, scenario)
    assert len(after.xpath('.//*[@class="pronunciation"]')) == 2
    assert len(set(after.xpath('//*[@class="pronunciation"]/@id'))) == 2
    assert not after.xpath('.//a[@class="pronunciation-repeat"]')
    assert '/zɒp/' in after.text_content() and '/zɔːp/' in after.text_content()


def test_result_groups_fragment_keeps_group_members_and_tasks(client, application):
    scenario, _, original = _split_result_scenario(application)
    ids = scenario.block_ids()
    for index, state in enumerate((TaskStatus.FAILED, TaskStatus.RUNNING, TaskStatus.SUCCEEDED, TaskStatus.PENDING)):
        store_task(application, task=ProcessingTask(
            id=f'group-task-{index}', book_id=scenario.book_id, scope_id=ids[index],
            task_type=TaskType.REGENERATE_BLOCK, status=state,
            last_error=failed_task(scenario).last_error if state is TaskStatus.FAILED else None))
    tree = _groups_tree(client, scenario)
    assert len(tree.xpath('.//*[@data-result-block]')) == 3
    assert len(tree.xpath('.//*[@data-active-task="true"]')) == 2
    assert tree.xpath('.//*[@class="source"]')[0].text == original
    assert tree.xpath('//form[@data-regenerate-form]/@action') == [
        f'/books/{scenario.book_id}/blocks/{identifier}/regenerate' for identifier in ids]
    assert tree.xpath('//form[@data-regenerate-form]/@data-block-id') == list(ids)
    assert len(tree.xpath('.//*[@class="task-state"]')) == 3
    assert 'Задача: Повторная генерация блока /' in tree.text_content()


@pytest.mark.parametrize('empty', [False, True], ids=['legacy', 'empty'])
def test_result_groups_fragment_empty_and_legacy(client, application, empty):
    scenario = store_book(application, status=BookStatus.COMPLETED, chapters=() if empty else (('Legacy text.',),))
    if not empty:
        start_run(application, scenario, results=(translated_result(scenario.block_ids()[0]),))
    tree = _groups_tree(client, scenario)
    assert len(tree.xpath('.//*[@data-result-block]')) == (0 if empty else 1)
    assert not tree.xpath('.//*[@class="pronunciation"]')
    assert not tree.xpath('.//*[@data-result-header] | .//*[@class="book-tabs"]')
    assert '<!doctype' not in client.get(f'/books/{scenario.book_id}/result', headers=GROUPS_HEADERS).text
