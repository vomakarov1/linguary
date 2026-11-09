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


def test_batch_result_block_fragment_uses_one_snapshot_for_requested_groups(
    client: TestClient, application: Application, monkeypatch,
) -> None:
    from lxml import html
    from linguary.web.routes import result as result_route

    scenario, _, _ = _split_result_scenario(application)
    block_ids = scenario.block_ids()
    original_book_result = result_route.book_result
    original_projection = result_route.project_pronunciations
    snapshots = []
    projections = []

    def snapshot(*args):
        snapshots.append(1)
        return original_book_result(*args)

    def project(*args):
        projections.append(1)
        return original_projection(*args)

    monkeypatch.setattr(result_route, "book_result", snapshot)
    monkeypatch.setattr(result_route, "project_pronunciations", project)
    response = client.get(
        f"/books/{scenario.book_id}/blocks",
        params=[("block_id", block_ids[0]), ("block_id", block_ids[1]),
                ("block_id", block_ids[2])],
        headers={"accept": "text/html", "x-linguary-fragment": "result-blocks"},
    )

    assert response.status_code == 200
    tree = html.fromstring(response.text)
    assert tree.get("data-result-blocks") is not None
    assert len(tree.xpath('./article[@data-result-block]')) == 2
    returned_ids = set(tree.xpath('.//form[@data-regenerate-form]/@data-block-id'))
    assert returned_ids == set(block_ids[:3])
    assert len(snapshots) == len(projections) == 1


def test_batch_result_block_fragment_rejects_more_than_128_ids_without_reflection(
    client: TestClient,
) -> None:
    marker = "private-block-id"
    response = client.get(
        "/books/unknown/blocks",
        params=[("block_id", marker)] * 129,
        headers={"accept": "text/html", "x-linguary-fragment": "result-blocks"},
    )

    assert response.status_code == 422
    assert marker not in response.text
    assert "Некорректный список блоков." in response.text


def test_batch_result_block_fragment_rejects_id_over_512_characters_without_reflection(
    client: TestClient,
) -> None:
    marker = "x" * 513
    response = client.get(
        "/books/unknown/blocks",
        params={"block_id": marker},
        headers={"accept": "text/html", "x-linguary-fragment": "result-blocks"},
    )

    assert response.status_code == 422
    assert marker not in response.text
    assert "Некорректный список блоков." in response.text


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
    assert tree.get('data-blocks-url') == f'/books/{scenario.book_id}/blocks'
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


def test_analyze_initial_conditions_pause_resume_and_restart_routes_and_pending_filter_polling(
    client: TestClient, application: Application
) -> None:
    from dataclasses import asdict
    import json
    from uuid import uuid4
    from lxml import html
    from linguary.application.book_conditions import payload_digest
    from linguary.domain.book_conditions import thaw
    from linguary.application.correction import CorrectionInputs, correction_input_fingerprint
    from linguary.application.correction_dto import (
        CORRECTION_VERSIONS_V2,
        CorrectionSnapshotV2,
        parse_correction_snapshot,
    )
    from linguary.application.recovery import NOTES_POLICY, RecoveryPolicy, root_manifest
    from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
    from linguary.domain.content_plan import canonical_sha
    from linguary.domain.correction import CORRECTION_POLICY_VERSION_V2, CorrectionCandidate
    from linguary.domain.recovery import CallAttempt, RecoveryRoot, RecoveryScope

    # 1. Initial conditions form on IMPORTED book & POST /books/{book_id}/analyze with options + empty body compatibility
    imported = store_book(
        application,
        book_id="book-imported-init",
        status=BookStatus.IMPORTED,
        analysed_chapters=(),
        chapters=((SOURCE_TEXT, "Second sentence."),),
    )
    source_path = application.paths.source_epub(imported.book_id)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"synthetic epub bytes")
    with application.unit_of_work.transaction() as tx:
        book_rec = tx.books.get(imported.book_id)
        tx.books.save(replace(book_rec, source_epub_path=str(source_path)))

    detail_html = client.get(
        f"/books/{imported.book_id}", headers={"accept": "text/html"}
    ).text
    assert 'name="reader_level"' in detail_html
    assert 'name="english_help"' in detail_html
    assert 'name="pronunciation_help_enabled"' in detail_html
    assert 'name="seed_first_chapter"' in detail_html

    analyzed_resp = client.post(
        f"/books/{imported.book_id}/analyze",
        json={
            "reader_level": "a2",
            "english_help": True,
            "pronunciation_help_enabled": True,
            "pronunciation_variant": "uk",
            "seed_first_chapter": True,
        },
    )
    assert analyzed_resp.status_code == 202, analyzed_resp.text
    with application.unit_of_work.transaction() as tx:
        cond = tx.work.latest_conditions(imported.book_id)
        assert cond is not None
        assert cond.payload["reader_level"] == "a2"
        assert cond.payload["english_help"] is True
        assert cond.pronunciation_help_enabled is True
        assert cond.pronunciation_variant == "uk"
        assert cond.payload["checks"]["seed_first_chapter"] is True

    empty_imported = store_book(
        application,
        book_id="book-imported-empty",
        status=BookStatus.IMPORTED,
        analysed_chapters=(),
        chapters=((SOURCE_TEXT,),),
    )
    empty_path = application.paths.source_epub(empty_imported.book_id)
    empty_path.parent.mkdir(parents=True, exist_ok=True)
    empty_path.write_bytes(b"synthetic empty body epub")
    with application.unit_of_work.transaction() as tx:
        book_rec2 = tx.books.get(empty_imported.book_id)
        tx.books.save(replace(book_rec2, source_epub_path=str(empty_path)))
    assert client.post(f"/books/{empty_imported.book_id}/analyze").status_code == 202

    # 2. Pause & resume routes and detail/quality_review template states
    pause_resp = client.post(f"/books/{imported.book_id}/pause")
    assert pause_resp.status_code == 200, pause_resp.text
    assert pause_resp.json()["pause_requested"] is True
    assert pause_resp.json()["is_paused"] is True

    paused_html = client.get(
        f"/books/{imported.book_id}", headers={"accept": "text/html"}
    ).text
    assert f'/books/{imported.book_id}/resume' in paused_html
    # Prefilled reanalyze form reflects saved conditions (a2, english_help, seed_first_chapter)
    paused_tree = html.fromstring(paused_html)
    reanalyze_form = paused_tree.xpath(
        f'//form[@action="/books/{imported.book_id}/conditions"][.//input[@name="action" and @value="reanalyze"]]'
    )[0]
    assert reanalyze_form.xpath('.//select[@name="reader_level"]/option[@value="a2" and @selected]')
    assert reanalyze_form.xpath('.//input[@name="english_help" and @checked]')
    assert reanalyze_form.xpath('.//input[@name="seed_first_chapter" and @checked]')

    resume_resp = client.post(f"/books/{imported.book_id}/resume")
    assert resume_resp.status_code == 200, resume_resp.text
    assert resume_resp.json()["is_paused"] is False

    # 3. RestartTranslation route (200/202, 409 on generation conflict, 404 on missing root/book)
    restart_scenario = store_book(
        application,
        book_id="book-restart-web",
        status=BookStatus.PROCESSING,
        chapters=((SOURCE_TEXT, "Another block to translate."),),
    )
    restart_source = application.paths.source_epub(restart_scenario.book_id)
    restart_source.parent.mkdir(parents=True, exist_ok=True)
    restart_source.write_bytes(b"synthetic restart epub")
    with application.unit_of_work.transaction() as tx:
        b_rec = tx.books.get(restart_scenario.book_id)
        tx.books.save(replace(b_rec, source_epub_path=str(restart_source)))
    continue_legacy(application, restart_scenario.book_id)
    start_run(application, restart_scenario)

    ordered_blocks = tuple(restart_scenario.block_ids(0))
    with application.unit_of_work.transaction() as tx:
        r_cond = tx.work.latest_conditions(restart_scenario.book_id)
        r_run = tx.runs.active_for_book(restart_scenario.book_id)
        r_analysis = tx.analyses.get_for_section(
            restart_scenario.book_id, restart_scenario.section(0).id
        )
        policy = RecoveryPolicy(**asdict(application.settings.recovery))
        manifest = root_manifest(
            book_id=restart_scenario.book_id,
            conditions_id=r_cond.id,
            glossary_sha=payload_digest(asdict(r_run.glossary_snapshot)),
            analysis_sha=payload_digest(asdict(r_analysis)),
            plan_sha=None,
            ordered_blocks=ordered_blocks,
            treatments=tuple("TRANSLATE" for _ in ordered_blocks),
            boundaries=tuple((bid,) for bid in ordered_blocks),
            atomic_intervals=tuple((bid,) for bid in ordered_blocks),
            max_source_chars=4000,
            policy=policy,
            block_chars={bid: 20 for bid in ordered_blocks},
        )
        manifest.update(
            notes_policy=NOTES_POLICY,
            active_run_id=r_run.id,
            historical_coverage=[],
        )
        root = RecoveryRoot(
            str(uuid4()),
            restart_scenario.book_id,
            r_cond.id,
            str(uuid4()),
            manifest,
            calls_used=policy.max_calls,
            status="stopped",
            reason="limit",
            operation_kind="translation",
            policy_version=CURRENT_TRANSLATION_POLICY,
        )
        tx.work.add_root(root)
        for bid in ordered_blocks:
            tx.work.add_scope(
                RecoveryScope(str(uuid4()), root.id, (bid,))
            )
    store_task(application, task=failed_task(restart_scenario))

    limit_detail_html = client.get(
        f"/books/{restart_scenario.book_id}", headers={"accept": "text/html"}
    ).text
    assert f"/books/{restart_scenario.book_id}/recovery/{root.id}/restart" in limit_detail_html
    # Retry forms are suppressed when recovery stopped due to limit
    assert f"/tasks/" not in limit_detail_html

    assert (
        client.post(
            f"/books/missing/recovery/{root.id}/restart",
            json={"expected_generation": 0, "confirmed": True},
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/books/{restart_scenario.book_id}/recovery/{uuid4()}/restart",
            json={"expected_generation": 0, "confirmed": True},
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/books/{restart_scenario.book_id}/recovery/{root.id}/restart",
            json={"expected_generation": 99, "confirmed": True},
        ).status_code
        == 409
    )
    restarted = client.post(
        f"/books/{restart_scenario.book_id}/recovery/{root.id}/restart",
        json={"expected_generation": 0, "confirmed": True},
    )
    assert restarted.status_code in (200, 202), restarted.text
    assert restarted.json()["successor_root_id"]
    # Second restart on superseded root conflicts with 409
    assert (
        client.post(
            f"/books/{restart_scenario.book_id}/recovery/{root.id}/restart",
            json={"expected_generation": 0, "confirmed": True},
        ).status_code
        == 409
    )

    # 4. Pending corrections filter on result page and ?filter=pending_corrections in data-result-url
    corr_scenario = store_book(
        application,
        book_id="book-pending-filter",
        status=BookStatus.READY_TO_EXPORT,
        chapters=(("First chapter block.",), ("Second chapter clean block.",)),
    )
    corr_source = application.paths.source_epub(corr_scenario.book_id)
    corr_source.parent.mkdir(parents=True, exist_ok=True)
    corr_source.write_bytes(b"synthetic corr epub")
    b1 = corr_scenario.block_ids(0)[0]
    b2 = corr_scenario.block_ids(1)[0]
    with application.unit_of_work.transaction() as tx:
        c_rec = tx.books.get(corr_scenario.book_id)
        tx.books.save(replace(c_rec, source_epub_path=str(corr_source)))
    continue_legacy(application, corr_scenario.book_id)
    start_run(
        application,
        corr_scenario,
        results=(translated_result(b1), translated_result(b2)),
    )
    with application.unit_of_work.transaction() as tx:
        c_cond = tx.work.latest_conditions(corr_scenario.book_id)
        c_run = tx.runs.active_for_book(corr_scenario.book_id)
        c_doc = tx.documents.get(corr_scenario.book_id)
        c_sec = corr_scenario.section(0)
        c_analysis = tx.analyses.get_for_section(corr_scenario.book_id, c_sec.id)
        corr_policy = CorrectionSnapshotV2(
            model="fake",
            route={"provider": "fake", "base_url": "http://fake", "proxy": None, "route_id": "fake"},
            parameters={
                "temperature": 0.0,
                "max_output_tokens": 1000,
                "timeout_seconds": 30,
                "max_attempts": 1,
                "max_unit_source_chars": 4000,
                "input_price_per_mtok": 0.0,
                "output_price_per_mtok": 0.0,
            },
            prompt="Correct chapter.",
            review_prompt="Review candidate.",
            versions=CORRECTION_VERSIONS_V2,
        ).to_payload()
        tx.work.bind(
            "run",
            c_run.id,
            book_id=corr_scenario.book_id,
            conditions_id=c_cond.id,
            payload={"schema_version": 1, "correction": corr_policy},
        )
        res_b1 = tx.results.get(c_run.id, b1)
        tx.work.bind(
            "result",
            f"{c_run.id}:{b1}",
            book_id=corr_scenario.book_id,
            conditions_id=c_cond.id,
            payload={"schema_version": 1},
        )
        bindings = tx.work.result_bindings(c_run.id)
        snapshot = parse_correction_snapshot(corr_policy)
        inputs = CorrectionInputs(
            c_doc,
            c_sec,
            c_analysis,
            c_run,
            c_cond,
            snapshot,
            tuple(c_sec.ordered_blocks),
            {b1: res_b1},
            {b1: bindings.get(b1)},
            {},
        )
        gen = tx.work.observe_correction_input(
            c_run.id, b1, correction_input_fingerprint(inputs, b1)
        )
        inputs = inputs.with_updated_state(input_generations={b1: gen})
        proposed = replace(res_b1, translated_text="Исправленный вариант перевода.")
        cand_id = str(uuid4())
        attempt_id = str(uuid4())
        tx.work.add_attempt(
            CallAttempt(
                attempt_id,
                corr_scenario.book_id,
                state="accepted",
                conditions_id=c_cond.id,
                run_id=c_run.id,
                context={
                    "schema_version": 1,
                    "stage": "chapter_correction",
                    "policy_version": CORRECTION_POLICY_VERSION_V2,
                    "ordered_block_ids": [b1],
                },
            )
        )
        cand_payload = {
            "schema_version": 1,
            "book_id": corr_scenario.book_id,
            "run_id": c_run.id,
            "conditions_id": c_cond.id,
            "section_id": c_sec.id,
            "block_id": b1,
            "input_sha256": inputs.checksums[b1],
            "input_generation": gen,
            "group_id": attempt_id,
            "proposal_attempt_id": attempt_id,
            "context_sha256": inputs.context_sha256,
            "source_binding": thaw(bindings.get(b1)),
            "original": {"text": c_sec.ordered_blocks[0].text},
            "current_result": json.loads(json.dumps(asdict(res_b1))),
            "result": json.loads(json.dumps(asdict(proposed))),
            "proposal": {
                "issues": [
                    {
                        "source_quote": "First chapter",
                        "current_evidence": {"kind": "quote", "quote": "Перевод"},
                        "problem": "Неточность смысла.",
                        "minimal_fix_reason": "Уточнение по контексту.",
                    }
                ]
            },
            "evidence_valid": False,
        }
        tx.work.add_correction_candidates(
            (
                CorrectionCandidate(
                    candidate_id=cand_id,
                    run_id=c_run.id,
                    book_id=corr_scenario.book_id,
                    section_id=c_sec.id,
                    block_id=b1,
                    conditions_id=c_cond.id,
                    group_id=attempt_id,
                    proposal_attempt_id=attempt_id,
                    input_generation=gen,
                    input_sha256=inputs.checksums[b1],
                    proposal_sha256=canonical_sha(cand_payload),
                    state="awaiting_human",
                    payload=cand_payload,
                    review={"verdict": "uncertain", "reason": "Нужна проверка человеком"},
                ),
            )
        )

    filtered_page = client.get(
        f"/books/{corr_scenario.book_id}/result?filter=pending_corrections",
        headers={"accept": "text/html"},
    )
    assert filtered_page.status_code == 200
    assert "First chapter block." in filtered_page.text
    assert "Second chapter clean block." not in filtered_page.text

    filtered_groups = client.get(
        f"/books/{corr_scenario.book_id}/result?filter=pending_corrections",
        headers=GROUPS_HEADERS,
    )
    assert filtered_groups.status_code == 200
    fg_tree = html.fromstring(filtered_groups.text)
    assert (
        fg_tree.get("data-result-url")
        == f"/books/{corr_scenario.book_id}/result?filter=pending_corrections"
    )
    assert len(fg_tree.xpath(".//*[@data-result-block]")) == 1


async def test_pause_lifecycle_restricts_inactive_statuses_clears_on_stage_end_and_manual_start_and_shows_in_result_header(
    configured_client: TestClient, configured_application: Application
) -> None:
    from linguary.application.block_processing import complete_processing
    from linguary.application.use_cases.analyze_chapter import ChapterAnalysisSequence

    client = configured_client
    application = configured_application

    # 1. PauseBook rejects statuses outside active queues (`ANALYZING`, `PROCESSING`)
    ready_book = store_book(
        application,
        book_id="book-pause-ready",
        status=BookStatus.READY_TO_EXPORT,
        chapters=((SOURCE_TEXT,),),
    )
    ready_src = application.paths.source_epub(ready_book.book_id)
    ready_src.parent.mkdir(parents=True, exist_ok=True)
    ready_src.write_bytes(b"synthetic ready epub")
    with application.unit_of_work.transaction() as tx:
        rec = tx.books.get(ready_book.book_id)
        tx.books.save(replace(rec, source_epub_path=str(ready_src)))
    continue_legacy(application, ready_book.book_id)
    start_run(
        application,
        ready_book,
        results=tuple(translated_result(bid) for bid in ready_book.block_ids()),
    )

    assert client.post(f"/books/{ready_book.book_id}/pause").status_code == 409

    # 2. If a book has an active pause binding (e.g. paused in PROCESSING or leftover in DB),
    # result header shows pause status + resume action, and manual POST /export clears pause so export runs.
    with application.unit_of_work.transaction() as tx:
        cond = tx.work.latest_conditions(ready_book.book_id)
        tx.work.set_book_paused(ready_book.book_id, cond.id, True)

    header_paused = client.get(
        f"/books/{ready_book.book_id}/result",
        headers={"accept": "text/html", "x-linguary-fragment": "result-header"},
    )
    assert header_paused.status_code == 200
    assert f'/books/{ready_book.book_id}/resume' in header_paused.text

    export_resp = client.post(f"/books/{ready_book.book_id}/export")
    assert export_resp.status_code == 202
    with application.unit_of_work.transaction() as tx:
        assert tx.work.is_book_paused(ready_book.book_id) is False
    finished_export = await drain_tasks(application)
    assert [t.status for t in finished_export] == [TaskStatus.SUCCEEDED]

    # 3. Manual POST /analyze clears pause; completing analysis (without auto_start_processing) clears pause,
    # while auto_start_processing preserves pause; manual POST /process clears pause;
    # completing processing (`PROCESSING -> READY_TO_EXPORT`) clears pause.
    analysing_book = store_book(
        application,
        book_id="book-pause-stages",
        status=BookStatus.ANALYZING,
        chapters=((SOURCE_TEXT,),),
    )
    stage_src = application.paths.source_epub(analysing_book.book_id)
    stage_src.parent.mkdir(parents=True, exist_ok=True)
    stage_src.write_bytes(b"synthetic stage epub")
    with application.unit_of_work.transaction() as tx:
        rec = tx.books.get(analysing_book.book_id)
        tx.books.save(replace(rec, source_epub_path=str(stage_src)))
    continue_legacy(application, analysing_book.book_id)

    assert client.post(f"/books/{analysing_book.book_id}/pause").status_code == 200
    with application.unit_of_work.transaction() as tx:
        assert tx.work.is_book_paused(analysing_book.book_id) is True

    # Manual POST /analyze clears pause
    assert client.post(f"/books/{analysing_book.book_id}/analyze").status_code == 202
    with application.unit_of_work.transaction() as tx:
        assert tx.work.is_book_paused(analysing_book.book_id) is False
        cond = tx.work.latest_conditions(analysing_book.book_id)
        tx.books.save(replace(tx.books.get(analysing_book.book_id), status=BookStatus.ANALYZING))
        tx.work.set_book_paused(analysing_book.book_id, cond.id, True)

    seq_manual = ChapterAnalysisSequence(
        unit_of_work=application.unit_of_work,
        queue=application.task_queue,
        analyze_chapter=None,  # type: ignore[arg-type]
        auto_start_processing=False,
    )
    assert seq_manual._try_complete_analysis(analysing_book.book_id) is False
    with application.unit_of_work.transaction() as tx:
        assert tx.books.get(analysing_book.book_id).status is BookStatus.GLOSSARY_REVIEW
        assert tx.work.is_book_paused(analysing_book.book_id) is False
        # When auto_approve_glossary + auto_start_processing triggers automatic transition, pause stays set
        tx.books.save(replace(tx.books.get(analysing_book.book_id), status=BookStatus.ANALYZING))
        tx.work.set_book_paused(analysing_book.book_id, cond.id, True)

    async def _noop_start(_bid: str) -> None:
        return None

    with application.unit_of_work.transaction() as tx:
        # Enable auto_approve_glossary + auto_start_processing on conditions for auto-transition test
        auto_cond = application.book_conditions.create(
            tx,
            tx.books.get(analysing_book.book_id),
            application.runtime_settings.revision,
        )
        # Update checks via new conditions with auto_approve_glossary
    seq_auto = ChapterAnalysisSequence(
        unit_of_work=application.unit_of_work,
        queue=application.task_queue,
        analyze_chapter=None,  # type: ignore[arg-type]
        auto_start_processing=True,
        start_processing=_noop_start,
    )
    with application.unit_of_work.transaction() as tx:
        from linguary.domain.book_conditions import BookConditions, thaw
        from uuid import uuid4
        latest = tx.work.latest_conditions(analysing_book.book_id)
        payload = thaw(latest.payload)
        payload.setdefault("checks", {})["auto_approve_glossary"] = True
        payload["checks"]["auto_start_processing"] = True
        cond_auto = BookConditions(
            str(uuid4()),
            analysing_book.book_id,
            latest.revision + 1,
            payload,
            latest.created_at,
        )
        tx.work.add_conditions(cond_auto)
        tx.work.set_book_paused(analysing_book.book_id, cond_auto.id, True)

    assert seq_auto._try_complete_analysis(analysing_book.book_id) is True
    with application.unit_of_work.transaction() as tx:
        assert tx.work.is_book_paused(analysing_book.book_id) is True

    # Manual POST /process clears pause
    proc_resp = client.post(f"/books/{analysing_book.book_id}/process")
    assert proc_resp.status_code == 202, proc_resp.text
    with application.unit_of_work.transaction() as tx:
        assert tx.work.is_book_paused(analysing_book.book_id) is False

    # Now pause during PROCESSING while last task finishes -> complete_processing clears pause
    assert client.post(f"/books/{analysing_book.book_id}/pause").status_code == 200
    with application.unit_of_work.transaction() as tx:
        run = tx.runs.active_for_book(analysing_book.book_id)
        for t in tx.tasks.list_for_book(analysing_book.book_id):
            tx.tasks.delete(t.id)
        tx.runs.delete(run.id)
    start_run(
        application,
        analysing_book,
        results=tuple(translated_result(bid) for bid in analysing_book.block_ids()),
    )
    with application.unit_of_work.transaction() as tx:
        cond_latest = tx.work.latest_conditions(analysing_book.book_id)
        tx.books.save(replace(tx.books.get(analysing_book.book_id), status=BookStatus.PROCESSING))
        tx.work.set_book_paused(analysing_book.book_id, cond_latest.id, True)

    complete_processing(
        application.unit_of_work,
        book_id=analysing_book.book_id,
        finished_task_id=None,
        task_types=(TaskType.PROCESS_TRANSLATION_UNIT,),
    )
    with application.unit_of_work.transaction() as tx:
        assert tx.books.get(analysing_book.book_id).status is BookStatus.READY_TO_EXPORT
        assert tx.work.is_book_paused(analysing_book.book_id) is False


def test_web_analyze_and_reanalyze_forms_handle_unchecked_seed_and_existing_conditions_revisions(
    settings, llm_client, web_task_ids
) -> None:
    from lxml import html
    from linguary.adapters.config import CheckSettings, LLMSettings
    from linguary.composition import build_application
    from linguary.web import create_app

    seeded_settings = replace(
        settings,
        llm=LLMSettings(model="fake-model", api_key="test-key"),
        checks=CheckSettings(seed_first_chapter=True),
    )
    app = build_application(
        settings=seeded_settings,
        llm_client=llm_client,
        task_id_factory=web_task_ids,
        content_plan_profile="legacy-v1",
    )
    try:
        scenario = store_book(
            app,
            book_id="book-seed-form",
            status=BookStatus.IMPORTED,
            analysed_chapters=(),
            chapters=((SOURCE_TEXT,),),
        )
        src = app.paths.source_epub(scenario.book_id)
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"synthetic seed form epub")
        with app.unit_of_work.transaction() as tx:
            rec = tx.books.get(scenario.book_id)
            tx.books.save(replace(rec, source_epub_path=str(src)))

        with TestClient(
            create_app(application=app, run_background_worker=False),
            base_url=app.settings.server.base_url,
        ) as client:
            # Initially unchecked conditions_id falls back to default_seed_first_chapter=True
            init_tree = html.fromstring(
                client.get(f"/books/{scenario.book_id}", headers={"accept": "text/html"}).text
            )
            assert init_tree.xpath('//form[contains(@action,"/analyze")]//input[@name="seed_first_chapter" and @checked]')

            # 1. Submitting HTML form without `seed_first_chapter` (unchecked box) disables it despite default True
            r1 = client.post(
                f"/books/{scenario.book_id}/analyze",
                data={"reader_level": "b1"},
                headers={"accept": "text/html"},
                follow_redirects=False,
            )
            assert r1.status_code == 303
            with app.unit_of_work.transaction() as tx:
                c1 = tx.work.latest_conditions(scenario.book_id)
                assert c1.revision == 1
                assert c1.payload["reader_level"] == "b1"
                assert c1.payload["english_help"] is False
                assert c1.payload["checks"]["seed_first_chapter"] is False

            # 2. Divergent options on non-IMPORTED book (`ANALYZING`) fail with 409
            conflict_json = client.post(
                f"/books/{scenario.book_id}/analyze",
                json={"reader_level": "c1"},
            )
            assert conflict_json.status_code == 409
            conflict_html = client.post(
                f"/books/{scenario.book_id}/analyze",
                data={"reader_level": "c1"},
                headers={"accept": "text/html"},
            )
            assert conflict_html.status_code == 409
            assert "reason_code" not in conflict_html.text

            # 3. `reanalyze` HTML form with checked seed -> revision 2 (seed=True), then unchecked seed -> revision 3 (seed=False)
            rev = app.runtime_settings.revision
            r2 = client.post(
                f"/books/{scenario.book_id}/conditions",
                data={
                    "action": "reanalyze",
                    "reader_level": "a2",
                    "english_help": "true",
                    "seed_first_chapter": "true",
                    "expected_revision": "1",
                    "settings_revision": str(rev),
                    "confirmed": "true",
                },
                headers={"accept": "text/html"},
                follow_redirects=False,
            )
            assert r2.status_code == 303
            with app.unit_of_work.transaction() as tx:
                c2 = tx.work.latest_conditions(scenario.book_id)
                assert c2.revision == 2
                assert c2.payload["checks"]["seed_first_chapter"] is True

            r3 = client.post(
                f"/books/{scenario.book_id}/conditions",
                data={
                    "action": "reanalyze",
                    "reader_level": "a2",
                    "expected_revision": "2",
                    "settings_revision": str(rev),
                    "confirmed": "true",
                },
                headers={"accept": "text/html"},
                follow_redirects=False,
            )
            assert r3.status_code == 303
            with app.unit_of_work.transaction() as tx:
                c3 = tx.work.latest_conditions(scenario.book_id)
                assert c3.revision == 3
                assert c3.payload["english_help"] is False
                assert c3.payload["checks"]["seed_first_chapter"] is False

            # Saved seed_first_chapter=False is not overridden by default_seed_first_chapter=True in detail.html
            after_reanalyze_tree = html.fromstring(
                client.get(f"/books/{scenario.book_id}", headers={"accept": "text/html"}).text
            )
            assert not after_reanalyze_tree.xpath(
                '//form[contains(@action,"/analyze")]//input[@name="seed_first_chapter" and @checked]'
            )

            # 4. Submitting modified options in `POST /analyze` on clean IMPORTED book creates revision 4
            r4 = client.post(
                f"/books/{scenario.book_id}/analyze",
                data={
                    "reader_level": "c1",
                    "english_help": "true",
                    "pronunciation_help_enabled": "true",
                    "pronunciation_variant": "us",
                    "seed_first_chapter": "true",
                },
                headers={"accept": "text/html"},
                follow_redirects=False,
            )
            assert r4.status_code == 303
            with app.unit_of_work.transaction() as tx:
                c4 = tx.work.latest_conditions(scenario.book_id)
                assert c4.revision == 4
                assert c4.payload["reader_level"] == "c1"
                assert c4.payload["english_help"] is True
                assert c4.pronunciation_help_enabled is True
                assert c4.pronunciation_variant == "us"
                assert c4.payload["checks"]["seed_first_chapter"] is True
    finally:
        app.dispose()


def test_pending_corrections_filter_recomputes_primary_pronunciations_and_preserves_header_filter_url(
    client: TestClient, application: Application
) -> None:
    from dataclasses import asdict
    import json
    from uuid import uuid4
    from lxml import html
    from linguary.application.correction import CorrectionInputs, correction_input_fingerprint
    from linguary.application.correction_dto import (
        CORRECTION_VERSIONS_V2,
        CorrectionSnapshotV2,
        parse_correction_snapshot,
    )
    from linguary.domain.book_conditions import thaw
    from linguary.domain.content_plan import SourceAnchor, canonical_sha
    from linguary.domain.correction import CORRECTION_POLICY_VERSION_V2, CorrectionCandidate
    from linguary.domain.pronunciation import PronunciationAssistance, PronunciationEvidence
    from linguary.domain.recovery import CallAttempt

    ch1_text = "The colonel arrived early."
    ch2_b1_text = "Later the colonel spoke softly."
    ch2_b2_text = "The colonel nodded again."
    scenario = store_book(
        application,
        book_id="book-filter-pronunciation",
        status=BookStatus.READY_TO_EXPORT,
        chapters=((ch1_text,), (ch2_b1_text, ch2_b2_text)),
    )
    src = application.paths.source_epub(scenario.book_id)
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"synthetic filter pronunciation epub")
    with application.unit_of_work.transaction() as tx:
        rec = tx.books.get(scenario.book_id)
        tx.books.save(replace(rec, source_epub_path=str(src)))
    continue_legacy(application, scenario.book_id)

    b_ch1 = scenario.block_ids(0)[0]
    b_ch2_1, b_ch2_2 = scenario.block_ids(1)
    evidence = PronunciationEvidence(
        source_id="synthetic-dict",
        source_version="v1",
        entry_id="entry-colonel",
        sense_id="sense-1",
        part_of_speech="noun",
        dictionary_form="colonel",
        dictionary_ipa="ˈkɜː.nəl",
        attribution="Synthetic Dictionary",
    )

    def _occ(bid: str, text: str, hint_id: str) -> PronunciationAssistance:
        pos = text.index("colonel")
        return PronunciationAssistance(
            anchor=SourceAnchor(bid, pos, pos + len("colonel"), "colonel"),
            form="colonel",
            ipa="ˈkɜː.nəl",
            variant="uk",
            hint_id=hint_id,
            evidence=evidence,
        )

    start_run(
        application,
        scenario,
        results=(
            BlockProcessingResult(
                block_id=b_ch1,
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Полковник прибыл рано.",
                pronunciation_assistances=(_occ(b_ch1, ch1_text, "ph-ch1-colonel"),),
            ),
            BlockProcessingResult(
                block_id=b_ch2_1,
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Позже полковник заговорил тихо.",
                pronunciation_assistances=(_occ(b_ch2_1, ch2_b1_text, "ph-ch2-colonel"),),
            ),
            BlockProcessingResult(
                block_id=b_ch2_2,
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Полковник снова кивнул.",
                pronunciation_assistances=(_occ(b_ch2_2, ch2_b2_text, "ph-ch2-repeat"),),
            ),
        ),
    )

    with application.unit_of_work.transaction() as tx:
        cond = tx.work.latest_conditions(scenario.book_id)
        run = tx.runs.active_for_book(scenario.book_id)
        doc = tx.documents.get(scenario.book_id)
        sec2 = scenario.section(1)
        analysis2 = tx.analyses.get_for_section(scenario.book_id, sec2.id)
        corr_policy = CorrectionSnapshotV2(
            model="fake",
            route={"provider": "fake", "base_url": "http://fake", "proxy": None, "route_id": "fake"},
            parameters={
                "temperature": 0.0,
                "max_output_tokens": 1000,
                "timeout_seconds": 30,
                "max_attempts": 1,
                "max_unit_source_chars": 4000,
                "input_price_per_mtok": 0.0,
                "output_price_per_mtok": 0.0,
            },
            prompt="Correct chapter.",
            review_prompt="Review candidate.",
            versions=CORRECTION_VERSIONS_V2,
        ).to_payload()
        tx.work.bind(
            "run",
            run.id,
            book_id=scenario.book_id,
            conditions_id=cond.id,
            payload={"schema_version": 1, "correction": corr_policy},
        )
        for bid in (b_ch2_1, b_ch2_2):
            tx.work.bind(
                "result",
                f"{run.id}:{bid}",
                book_id=scenario.book_id,
                conditions_id=cond.id,
                payload={"schema_version": 1},
            )
        bindings = tx.work.result_bindings(run.id)
        snapshot = parse_correction_snapshot(corr_policy)
        res_map = {bid: tx.results.get(run.id, bid) for bid in (b_ch2_1, b_ch2_2)}
        inputs = CorrectionInputs(
            doc,
            sec2,
            analysis2,
            run,
            cond,
            snapshot,
            tuple(sec2.ordered_blocks),
            res_map,
            {bid: bindings.get(bid) for bid in (b_ch2_1, b_ch2_2)},
            {},
        )
        gens = {
            bid: tx.work.observe_correction_input(
                run.id, bid, correction_input_fingerprint(inputs, bid)
            )
            for bid in (b_ch2_1, b_ch2_2)
        }
        inputs = inputs.with_updated_state(input_generations=gens)
        attempt_id = str(uuid4())
        tx.work.add_attempt(
            CallAttempt(
                attempt_id,
                scenario.book_id,
                state="accepted",
                conditions_id=cond.id,
                run_id=run.id,
                context={
                    "schema_version": 1,
                    "stage": "chapter_correction",
                    "policy_version": CORRECTION_POLICY_VERSION_V2,
                    "ordered_block_ids": [b_ch2_1, b_ch2_2],
                },
            )
        )
        candidates_to_add = []
        for idx, bid in enumerate((b_ch2_1, b_ch2_2)):
            cur = res_map[bid]
            prop = replace(cur, translated_text=f"Исправленный перевод {idx}.")
            payload = {
                "schema_version": 1,
                "book_id": scenario.book_id,
                "run_id": run.id,
                "conditions_id": cond.id,
                "section_id": sec2.id,
                "block_id": bid,
                "input_sha256": inputs.checksums[bid],
                "input_generation": gens[bid],
                "group_id": attempt_id,
                "proposal_attempt_id": attempt_id,
                "context_sha256": inputs.context_sha256,
                "source_binding": thaw(bindings.get(bid)),
                "original": {"text": sec2.ordered_blocks[idx].text},
                "current_result": json.loads(json.dumps(asdict(cur))),
                "result": json.loads(json.dumps(asdict(prop))),
                "proposal": {
                    "issues": [
                        {
                            "source_quote": "colonel",
                            "current_evidence": {"kind": "quote", "quote": "полковник"},
                            "problem": "Оттенок.",
                            "minimal_fix_reason": "Точнее.",
                        }
                    ]
                },
                "evidence_valid": False,
            }
            candidates_to_add.append(
                CorrectionCandidate(
                    candidate_id=str(uuid4()),
                    run_id=run.id,
                    book_id=scenario.book_id,
                    section_id=sec2.id,
                    block_id=bid,
                    conditions_id=cond.id,
                    group_id=attempt_id,
                    proposal_attempt_id=attempt_id,
                    input_generation=gens[bid],
                    input_sha256=inputs.checksums[bid],
                    proposal_sha256=canonical_sha(payload),
                    state="awaiting_human",
                    payload=payload,
                    review={"verdict": "uncertain", "reason": "Нужна проверка"},
                )
            )
        tx.work.add_correction_candidates(tuple(candidates_to_add))

    filtered_json = client.get(
        f"/books/{scenario.book_id}/result?filter=pending_corrections"
    ).json()
    assert len(filtered_json["chapters"]) == 1
    groups = filtered_json["chapters"][0]["groups"]
    assert len(groups) == 2
    assert groups[0]["pronunciation_assistances"][0]["is_primary"] is True
    assert groups[0]["pronunciation_assistances"][0]["hint_id"] == "ph-ch2-colonel"
    assert groups[1]["pronunciation_assistances"][0]["is_primary"] is False
    assert groups[1]["pronunciation_assistances"][0]["hint_id"] == "ph-ch2-colonel"

    filtered_html = client.get(
        f"/books/{scenario.book_id}/result?filter=pending_corrections",
        headers={"accept": "text/html"},
    )
    assert filtered_html.status_code == 200
    assert 'id="ph-ch2-colonel"' in filtered_html.text
    assert "/ˈkɜː.nəl/" in filtered_html.text
    assert 'href="#ph-ch2-colonel"' in filtered_html.text
    assert "ph-ch1-colonel" not in filtered_html.text
    page_tree = html.fromstring(filtered_html.text)
    assert (
        page_tree.xpath("//*[@data-result-header]")[0].get("data-result-url")
        == f"/books/{scenario.book_id}/result?filter=pending_corrections"
    )

    header_frag = client.get(
        f"/books/{scenario.book_id}/result?filter=pending_corrections",
        headers={"accept": "text/html", "x-linguary-fragment": "result-header"},
    )
    assert header_frag.status_code == 200
    frag_tree = html.fromstring(header_frag.text)
    assert (
        frag_tree.get("data-result-url")
        == f"/books/{scenario.book_id}/result?filter=pending_corrections"
    )
    unfiltered_frag = client.get(
        f"/books/{scenario.book_id}/result",
        headers={"accept": "text/html", "x-linguary-fragment": "result-header"},
    )
    assert (
        html.fromstring(unfiltered_frag.text).get("data-result-url")
        == f"/books/{scenario.book_id}/result"
    )


