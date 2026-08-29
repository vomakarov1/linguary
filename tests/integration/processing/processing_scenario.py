"""Сборка книги, дошедшей до обработки, и заготовленных ответов модели (P3.T6).

Обработка начинается там, где книга уже импортирована, все главы проанализированы,
глоссарий проверен и запуск открыт. Модуль собирает это состояние в настоящем хранилище
одной функцией, чтобы тест описывал только то, что проверяет.

Модуль намеренно без префикса `test_`: иначе pytest соберёт его как тестовый.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from linguary.adapters.config import CheckSettings
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm import FakeReply
from linguary.application.context_builder import textual_blocks
from linguary.application.ports.llm import LLMClient
from linguary.application.ports.persistence import BookRecord
from linguary.application.retry_policy import RetryPolicy
from linguary.application.translation_units import DEFAULT_MAX_UNIT_SOURCE_CHARS
from linguary.application.use_cases.process_chapter import ProcessChapter
from linguary.application.validators import TranslationResultValidator
from linguary.domain import (
    BlockTreatment,
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    DocumentBlock,
    DocumentSection,
    LLMSettingsSnapshot,
    ProcessingRun,
    ProcessingRunStatus,
    SourceReference,
    SupportStatus,
    TermStatus,
    TokenUsage,
    TreatmentPlan,
)
from linguary.domain.processing import GlossarySnapshot, GlossarySnapshotEntry

BOOK_ID = "book-1"
RUN_ID = "run-1"
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "ru"

#: Абзац оригинала и его законченный русский перевод: тексты, на которых проверки молчат.
SOURCE_TEXT = "It is a truth universally acknowledged that a man must want a wife."
TRANSLATED_TEXT = "Общепризнанная истина гласит, что мужчина непременно ищет жену."

#: Фиксированный момент времени: тест не зависит от текущих часов.
CREATED_AT = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ProcessingScenario:
    """Книга в состоянии `PROCESSING` вместе с её активным запуском."""

    document: CanonicalDocument
    run_id: str = RUN_ID

    def section(self, index: int = 0) -> DocumentSection:
        """Глава книги по порядку чтения."""
        return self.document.ordered_sections[index]

    def block_ids(self, index: int = 0) -> tuple[str, ...]:
        """Идентификаторы блоков главы в порядке исходного документа."""
        return tuple(block.id for block in self.section(index).ordered_blocks)


def build_document(chapters: Sequence[Sequence[str | None]]) -> CanonicalDocument:
    """Книга из глав, заданных текстами их абзацев."""
    return CanonicalDocument(
        id=BOOK_ID,
        metadata=BookMetadata(title="Pride and Prejudice", authors=("Jane Austen",)),
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        sections=tuple(
            _section(order=order, texts=texts) for order, texts in enumerate(chapters)
        ),
    )


def store_processing_book(
    unit_of_work: SqlAlchemyUnitOfWork,
    *,
    chapters: Sequence[Sequence[str | None]] = ((SOURCE_TEXT, SOURCE_TEXT),),
    treatments: Mapping[str, Sequence[BlockTreatment]] | None = None,
    status: BookStatus = BookStatus.PROCESSING,
    locked_terms: bool = True,
    run_id: str = RUN_ID,
    analysed_sections: Sequence[str] | None = None,
) -> ProcessingScenario:
    """Кладёт в хранилище книгу, готовую к обработке, вместе с активным запуском.

    `analysed_sections` оставляет анализ только у перечисленных глав: обработка главы
    без сохранённого анализа — отдельный запрещённый случай (PRD § 14).
    """
    document = build_document(chapters)
    with unit_of_work.transaction() as uow:
        uow.books.add(_book(status))
        uow.documents.save(document)
        for section in document.ordered_sections:
            if analysed_sections is not None and section.id not in analysed_sections:
                continue
            uow.analyses.save(BOOK_ID, _analysis(section, treatments=treatments))
        uow.runs.add(_run(run_id, locked_terms=locked_terms))
    return ProcessingScenario(document=document, run_id=run_id)


def make_process_chapter(
    unit_of_work: SqlAlchemyUnitOfWork,
    client: LLMClient,
    *,
    max_attempts: int = 3,
    max_unit_source_chars: int = DEFAULT_MAX_UNIT_SOURCE_CHARS,
    checks: CheckSettings | None = None,
) -> tuple[ProcessChapter, list[float]]:
    """Сценарий обработки главы вместе со списком выдержанных пауз повторов."""
    delays: list[float] = []

    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    settings = checks or CheckSettings()
    scenario = ProcessChapter(
        unit_of_work=unit_of_work,
        llm_client=client,
        retry_policy=RetryPolicy(max_attempts=max_attempts),
        validator=TranslationResultValidator(
            target_language_min_share=settings.target_language_min_share,
            language_check_min_length=settings.language_check_min_length,
            repetition_ngram_size=settings.repetition_ngram_size,
            repetition_min_repeats=settings.repetition_min_repeats,
        ),
        max_unit_source_chars=max_unit_source_chars,
        id_factory=sequential_ids("unit"),
        sleep=sleep,
    )
    return scenario, delays


def sequential_ids(prefix: str) -> Callable[[], str]:
    """Предсказуемые идентификаторы: `unit-1`, `unit-2`, ..."""
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}-{counter}"

    return next_id


def block_reply(
    alias: str,
    *,
    treatment: BlockTreatment = BlockTreatment.TRANSLATE,
    translated_text: str | None = TRANSLATED_TEXT,
    hints: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Результат одного блока в ответе модели."""
    return {
        "source_block_id": alias,
        "treatment": treatment.value,
        "translated_text": translated_text if treatment.requires_translation else None,
        "glossary_hints": [
            {"original": original, "translation": translation, "note": None}
            for original, translation in hints
        ],
    }


def note_reply(
    alias: str,
    *,
    original_excerpt: str = "a truth universally acknowledged",
    note: str = "Ироническая формула зачина романа.",
    translated_excerpt: str | None = "Общепризнанная истина",
) -> dict[str, Any]:
    """Языковая заметка ответа, привязанная к блоку запроса."""
    return {
        "source_block_id": alias,
        "original_excerpt": original_excerpt,
        "translated_excerpt": translated_excerpt,
        "note": note,
    }


def translation_reply(
    blocks: Sequence[Mapping[str, Any]],
    *,
    notes: Sequence[Mapping[str, Any]] = (),
    truncated: bool = False,
    usage: TokenUsage | None = None,
) -> FakeReply:
    """Заготовленный ответ провайдера на обработку группы блоков."""
    payload = {"blocks": list(blocks), "language_notes": list(notes)}
    return FakeReply(
        content=json.dumps(payload, ensure_ascii=False),
        truncated=truncated,
        usage=usage,
    )


def translated_reply(
    aliases: Sequence[str] = ("b1", "b2"),
    *,
    notes: Sequence[Mapping[str, Any]] = (),
    truncated: bool = False,
    usage: TokenUsage | None = None,
) -> FakeReply:
    """Корректный ответ: полный перевод каждого блока группы."""
    return translation_reply(
        [block_reply(alias) for alias in aliases],
        notes=notes,
        truncated=truncated,
        usage=usage,
    )


def _section(*, order: int, texts: Sequence[str | None]) -> DocumentSection:
    """Глава книги из абзацев; `None` вместо текста даёт нетекстовый блок-иллюстрацию."""
    section_id = f"section-{order + 1}"
    return DocumentSection(
        id=section_id,
        document_id=BOOK_ID,
        title=f"Chapter {order + 1}",
        order=order,
        source_resource_path=f"{section_id}.xhtml",
        blocks=tuple(
            DocumentBlock.create(
                document_id=BOOK_ID,
                section_id=section_id,
                type=BlockType.PARAGRAPH if text is not None else BlockType.IMAGE,
                order=index,
                text=text or "",
                source_reference=SourceReference(
                    resource_path=f"{section_id}.xhtml",
                    structural_path=f"/body/p[{index + 1}]",
                ),
                support_status=SupportStatus.SUPPORTED,
            )
            for index, text in enumerate(texts)
        ),
    )


def _analysis(
    section: DocumentSection,
    *,
    treatments: Mapping[str, Sequence[BlockTreatment]] | None,
) -> ChapterAnalysis:
    """Полный анализ главы: план обработки покрывает все её текстовые блоки.

    Нетекстовые блоки (изображение, разрыв страницы) в план не идут: их классифицирует
    импортёр, и модели они не передаются (PRD § 10.1).
    """
    blocks = textual_blocks(section)
    chosen = (
        tuple((treatments or {}).get(section.id, ()))
        or (BlockTreatment.TRANSLATE,) * len(blocks)
    )
    return ChapterAnalysis.create(
        id=f"analysis-{section.id}",
        section_id=section.id,
        treatment_plan=TreatmentPlan.of_treatments(
            section.id,
            zip((block.id for block in blocks), chosen, strict=True),
        ),
        translation_instructions="Сохранять иронию рассказчика.",
        chapter_summary="Mr. Bennet visits the new neighbour.",
        expected_block_ids=[block.id for block in blocks],
    )


def _book(status: BookStatus) -> BookRecord:
    """Запись книги, дошедшей до обработки."""
    return BookRecord(
        id=BOOK_ID,
        metadata=BookMetadata(title="Pride and Prejudice", authors=("Jane Austen",)),
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        status=status,
        source_epub_path="/library/pride.epub",
        created_at=CREATED_AT,
    )


def _run(run_id: str, *, locked_terms: bool) -> ProcessingRun:
    """Активный запуск обработки со снимком глоссария и настроек LLM."""
    entries = [
        GlossarySnapshotEntry(
            source_text="Mr. Bennet",
            translation="мистер Беннет",
            type=ConceptType.PERSON,
            status=TermStatus.LOCKED if locked_terms else TermStatus.SUGGESTED,
        )
    ]
    return ProcessingRun(
        id=run_id,
        book_id=BOOK_ID,
        status=ProcessingRunStatus.ACTIVE,
        glossary_snapshot=GlossarySnapshot(entries=tuple(entries)),
        llm_settings_snapshot=LLMSettingsSnapshot(provider="fake", model="fake-model"),
    )
