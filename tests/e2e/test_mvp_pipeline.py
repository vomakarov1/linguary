"""Сквозной сценарий MVP на синтетическом EPUB и заготовленных ответах LLM."""

from __future__ import annotations

import json
import shutil
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, StorageSettings
from linguary.adapters.llm import FakeLLMClient, FakeReply
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
from linguary.domain import (
    BookStatus,
    ErrorType,
    LLMCallStage,
    TaskType,
    TokenUsage,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"
SYNTHETIC_EPUB = FIXTURE_ROOT / "epub" / "synthetic_story.epub"
LLM_FIXTURE_ROOT = FIXTURE_ROOT / "llm"


@dataclass(slots=True)
class _RecordingLLMClient(LLMClient):
    """`FakeLLMClient`, ответы которого подготавливаются после импорта EPUB."""

    inner: FakeLLMClient = field(default_factory=FakeLLMClient)
    requests: list[str] = field(default_factory=list)

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        self.requests.append(request.input_text)
        return await self.inner.generate(request)


def test_full_mvp_pipeline_exports_translated_epub(tmp_path: Path) -> None:
    """`import -> analyze -> glossary -> process -> export` проходит без реального LLM."""
    client = _RecordingLLMClient()
    application = _application(tmp_path, client)
    try:
        _register_and_import(application, tmp_path)
        document = _document(application)
        client.inner.replies = _llm_replies(document)

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
        exported_text = "\n".join(
            "".join(root.itertext())
            for root in _documents_of(application.paths.export_epub("book-e2e")).values()
        )
        assert "Русский перевод блока" in exported_text
        assert "Составное существительное передаёт образ света и стекла." in exported_text
        assert "Previously in Chapter" in exported_text
        assert any(
            "Рен Колдер (женский род)" in text for text in client.requests
        ), "переводческое указание глоссария не дошло до запроса обработки"
    finally:
        application.dispose()


def _application(tmp_path: Path, client: LLMClient) -> Application:
    """Сборка приложения с отключённым EPUBCheck."""
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
        task_id_factory=_ids("task"),
        epubcheck=EpubCheckValidator(
            settings=EpubCheckSettings(java_executable="missing-linguary-java")
        ),
    )


def _register_and_import(application: Application, tmp_path: Path) -> None:
    """Копирует EPUB в каталог приложения и импортирует его через очередь."""
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
    """Импортированная каноническая модель книги."""
    with application.unit_of_work.transaction() as transaction:
        document = transaction.documents.get("book-e2e")
    assert document is not None
    return document


def _llm_replies(document) -> tuple[FakeReply | Exception, ...]:
    """Ответы анализа и обработки под реальные размеры глав."""
    analysis = _read_fixture("analysis-template.json")
    translation = _read_fixture("translation-template.json")
    replies: list[FakeReply | Exception] = []
    for section in document.ordered_sections:
        aliases = [f"b{index}" for index, _ in enumerate(textual_blocks(section), start=1)]
        replies.append(
            FakeReply.of_payload(
                {
                    "translation_instructions": analysis["translation_instructions"],
                    "chapter_summary": (
                        f"{analysis['chapter_summary']} Chapter: {section.title}."
                    ),
                    "glossary_candidates": analysis["glossary_candidates"],
                    "before_you_read": analysis["before_you_read"],
                    "treatment_plan": [
                        {
                            "source_block_id": alias,
                            "treatment": "TRANSLATE",
                            "reason": analysis["treatment_reason"],
                        }
                        for alias in aliases
                    ],
                }
            )
        )
    replies.append(FakeReply(content="{"))
    for section in document.ordered_sections:
        aliases = [f"b{index}" for index, _ in enumerate(textual_blocks(section), start=1)]
        replies.append(_translation_reply(aliases, translation))
    return tuple(replies)


def _translation_reply(aliases: list[str], fixture: dict[str, object]) -> FakeReply:
    """Ответ обработки одной главы с переводом, подсказками и заметкой."""
    hint = fixture["glossary_hint"]
    note = fixture["language_note"]
    assert isinstance(hint, dict)
    assert isinstance(note, dict)
    return FakeReply.of_payload(
        {
            "blocks": [
                {
                    "source_block_id": alias,
                    "treatment": "TRANSLATE",
                    "translated_text": f"{fixture['translated_text']} ({alias}).",
                    "glossary_hints": [hint],
                }
                for alias in aliases
            ],
            "language_notes": [
                {
                    "source_block_id": aliases[0],
                    "original_excerpt": note["original_excerpt"],
                    "translated_excerpt": note["translated_excerpt"],
                    "note": note["note"],
                }
            ],
        },
        usage=TokenUsage(prompt_tokens=100, completion_tokens=200),
    )


def _read_fixture(name: str) -> dict[str, object]:
    """JSON-fixture ответа модели."""
    return json.loads((LLM_FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _ids(prefix: str) -> Callable[[], str]:
    """Предсказуемые идентификаторы задач."""
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}-{counter}"

    return next_id


def _documents_of(path: Path) -> dict[str, etree._Element]:
    """XHTML-документы собранного EPUB."""
    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
    with zipfile.ZipFile(path) as archive:
        return {
            name: etree.fromstring(archive.read(name), parser=parser)
            for name in archive.namelist()
            if name.endswith((".xhtml", ".html", ".htm"))
        }


def _run(awaitable):
    """Синхронный вызов асинхронного сценария внутри теста."""
    import asyncio

    return asyncio.run(awaitable)
