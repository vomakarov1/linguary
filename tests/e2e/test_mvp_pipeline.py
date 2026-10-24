"""MVP end-to-end pipeline on synthetic EPUB and scripted LLM replies."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from pipeline_support import (
    SYNTHETIC_EPUB,
    exported_text,
    llm_replies,
    sequential_ids,
)

from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, StorageSettings
from linguary.adapters.llm import FakeLLMClient
from linguary.adapters.validation import EpubCheckSettings, EpubCheckValidator
from linguary.application.context_builder import textual_blocks
from linguary.application.ports.llm import LLMClient, LLMPayload, LLMRequest, LLMResponse
from linguary.application.ports.persistence import ExportValidationStatus
from linguary.application.use_cases.approve_glossary import (
    GlossaryDecision,
    GlossaryDecisionCommand,
    StartProcessingCommand,
)
from linguary.application.use_cases.register_book import RegisterBookCommand
from linguary.composition import Application, build_application, drain_tasks
from linguary.domain import BookStatus, ErrorType, LLMCallStage, TaskType


@dataclass(slots=True)
class _RecordingLLMClient(LLMClient):
    """`FakeLLMClient` whose replies are prepared after EPUB import."""

    inner: FakeLLMClient = field(default_factory=FakeLLMClient)
    requests: list[str] = field(default_factory=list)

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        self.requests.append(request.input_text)
        return await self.inner.generate(request)


def test_full_mvp_pipeline_exports_translated_epub(tmp_path: Path) -> None:
    """`import -> analyze -> glossary -> process -> export` runs without real LLM."""
    client = _RecordingLLMClient()
    application = _application(tmp_path, client)
    try:
        _register_and_import(application, tmp_path)
        document = _document(application)
        client.inner.replies = llm_replies(
            document, broken_reply_before_processing=True
        )

        first_analysis = _run(application.analysis.start("book-e2e"))
        assert first_analysis is not None
        assert _run(drain_tasks(application))

        review = application.approve_glossary.review("book-e2e")
        for item in review.awaiting_decision:
            application.approve_glossary(
                GlossaryDecisionCommand(
                    book_id="book-e2e",
                    source_text=item.source_text,
                    concept_id=item.concept_id,
                    decision=GlossaryDecision.ACCEPT,
                )
            )
        review = application.approve_glossary.review("book-e2e")
        assert review.is_complete
        wren = review.item_for("Wren Calder")
        assert wren is not None
        assert wren.note == "женский род"

        run = application.start_processing()(StartProcessingCommand(book_id="book-e2e"))
        assert run.book_id == "book-e2e"
        processing_tasks = _run(application.processing.start("book-e2e"))
        assert len(processing_tasks) == len(document.ordered_sections)
        processed = _run(drain_tasks(application))
        assert all(task.last_error is None for task in processed)

        export_task = _run(application.start_export("book-e2e"))
        assert export_task.task_type is TaskType.EXPORT_BOOK
        exported = _run(drain_tasks(application))
        assert all(task.last_error is None for task in exported)

        with application.unit_of_work.transaction() as transaction:
            book = transaction.books.get("book-e2e")
            logs = transaction.llm_calls.list_for_book("book-e2e")
            result_count = len(transaction.results.list_for_run(run.id))

        assert book is not None
        assert book.status is BookStatus.COMPLETED
        assert book.export_validation_status is ExportValidationStatus.NOT_VALIDATED
        assert result_count == sum(
            len(textual_blocks(section)) for section in document.ordered_sections
        )
        assert any(not entry.success for entry in logs)
        assert {entry.stage for entry in logs} >= {
            LLMCallStage.CHAPTER_ANALYSIS,
            LLMCallStage.TRANSLATION_UNIT,
        }
        assert logs[3].error is not None
        assert logs[3].error.type is ErrorType.MODEL_OUTPUT
        assert application.paths.export_epub("book-e2e").is_file()
        book_text = exported_text(application.paths.export_epub("book-e2e"))
        assert "Русский перевод блока" in book_text
        assert "Причастие humming описывает непрерывный звук предмета." in book_text
        assert "Previously in Chapter" in book_text
        assert any(
            "Рен Колдер (женский род)" in text for text in client.requests
        ), "glossary translation instruction was not passed to processing request"
    finally:
        application.dispose()


def _application(tmp_path: Path, client: LLMClient) -> Application:
    """Builds application with disabled EPUBCheck."""
    settings = AppSettings(
        llm=LLMSettings(
            model="fake-model",
            api_key="test-key",
            max_attempts=2,
            max_unit_source_chars=1_000_000,
        ),
        checks=CheckSettings(),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
    )
    return build_application(
        settings=settings,
        llm_client=client,
        task_id_factory=sequential_ids("task"),
        epubcheck=EpubCheckValidator(
            settings=EpubCheckSettings(java_executable="missing-linguary-java")
        ),
    )


def _register_and_import(application: Application, tmp_path: Path) -> None:
    """Copies EPUB to application data directory and imports it via queue."""
    source = application.paths.source_epub("book-e2e")
    source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SYNTHETIC_EPUB, source)
    application.register_book(
        RegisterBookCommand(
            book_id="book-e2e",
            source_epub_path=str(source),
            title="The Lamp at Duskmere",
            source_language="en",
            target_language="ru",
        )
    )
    _run(application.start_import("book-e2e"))
    tasks = _run(drain_tasks(application))
    assert len(tasks) == 1
    assert tasks[0].last_error is None


def _document(application: Application):
    """Imported canonical document model of the book."""
    with application.unit_of_work.transaction() as transaction:
        document = transaction.documents.get("book-e2e")
    assert document is not None
    return document


def _run(awaitable):
    """Synchronously executes an async scenario inside the test."""
    import asyncio

    return asyncio.run(awaitable)
