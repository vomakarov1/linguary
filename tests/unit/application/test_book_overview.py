"""Book overview warning counts and repository fallback behavior (P2.T4)."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.models import Base
from linguary.adapters.db.repositories import SqlAlchemyBlockResultRepository
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.application.book_overview import book_overview, book_warnings
from linguary.application.ports.persistence import BlockResultRepository, BookRecord
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    BlockType,
    BookImportWarning,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    GlossarySnapshot,
    ImportWarningKind,
    LLMSettingsSnapshot,
    ProcessingRun,
    ProcessingRunStatus,
    ResultWarning,
    ResultWarningKind,
    SourceReference,
    SupportStatus,
)


def _sample_document() -> CanonicalDocument:
    blocks = (
        DocumentBlock(
            id="blk-1",
            section_id="sec-1",
            order=0,
            type=BlockType.PARAGRAPH,
            text="First paragraph.",
            support_status=SupportStatus.SUPPORTED,
            source_reference=SourceReference("ch1.xhtml", "/html/body/p[1]"),
        ),
        DocumentBlock(
            id="blk-2",
            section_id="sec-1",
            order=1,
            type=BlockType.PARAGRAPH,
            text="Second paragraph.",
            support_status=SupportStatus.SUPPORTED,
            source_reference=SourceReference("ch1.xhtml", "/html/body/p[2]"),
        ),
    )
    section = DocumentSection(
        id="sec-1",
        document_id="book-1",
        order=0,
        title="Chapter 1",
        source_resource_path="ch1.xhtml",
        blocks=blocks,
    )
    return CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(title="Sample", authors=("Author",)),
        source_language="en",
        target_language="ru",
        sections=(section,),
    )


def _book_record() -> BookRecord:
    return BookRecord(
        id="book-1",
        metadata=BookMetadata(title="Sample", authors=("Author",)),
        source_language="en",
        target_language="ru",
        status=BookStatus.PROCESSING,
        source_epub_path="/tmp/book.epub",
        created_at=datetime(2026, 11, 23, 12, 0, tzinfo=UTC),
    )


def _processing_run() -> ProcessingRun:
    return ProcessingRun(
        id="run-1",
        book_id="book-1",
        status=ProcessingRunStatus.ACTIVE,
        glossary_snapshot=GlossarySnapshot(entries=()),
        llm_settings_snapshot=LLMSettingsSnapshot(
            provider="openai",
            model="gpt-test",
            temperature=0.2,
            max_output_tokens=4096,
        ),
    )


def test_book_overview_uses_count_warnings_for_run_without_list_for_run() -> None:
    """`book_overview` sums `import_warnings` and `count_warnings_for_run` without calling `list_for_run`."""
    assert hasattr(BlockResultRepository, "count_warnings_for_run")

    document = _sample_document()
    book = _book_record()
    run = _processing_run()
    import_warnings = (
        BookImportWarning(
            kind=ImportWarningKind.DROPPED_BLOCK,
            message="Пропущен блок.",
            section_id="sec-1",
        ),
    )
    count_calls: list[str] = []

    def _count_warnings_for_run(run_id: str) -> int:
        count_calls.append(run_id)
        return 3

    def _unexpected_list_for_run(_run_id: str):
        raise AssertionError(
            "list_for_run must not be called when count_warnings_for_run is available"
        )

    tx = SimpleNamespace(
        books=SimpleNamespace(get=lambda book_id: book),
        documents=SimpleNamespace(
            get=lambda book_id: document,
            import_warnings=lambda book_id: import_warnings,
        ),
        analyses=SimpleNamespace(list_for_book=lambda book_id: ()),
        tasks=SimpleNamespace(list_for_book=lambda book_id: ()),
        runs=SimpleNamespace(active_for_book=lambda book_id: run),
        results=SimpleNamespace(
            count_for_run=lambda run_id: 1,
            block_ids_for_run=lambda run_id: frozenset({"blk-1"}),
            count_warnings_for_run=_count_warnings_for_run,
            list_for_run=_unexpected_list_for_run,
        ),
        llm_calls=SimpleNamespace(list_for_book=lambda book_id: ()),
        work=SimpleNamespace(
            latest_conditions=lambda book_id: None,
            list_roots=lambda book_id, run_id=None: (),
            list_reviews=lambda book_id: (),
        ),
    )

    class _StubUoW:
        @contextmanager
        def read_transaction(self):
            yield tx

        @contextmanager
        def transaction(self):
            yield tx

    overview = book_overview(_StubUoW(), "book-1")  # type: ignore[arg-type]
    assert count_calls == ["run-1"]
    assert overview.warning_count == 4


def test_book_overview_warning_count_falls_back_when_count_warnings_for_run_is_missing() -> None:
    """Partial repositories without `count_warnings_for_run` fall back to `stored_results` or `list_for_run`."""
    document = _sample_document()
    book = _book_record()
    run = _processing_run()
    results = (
        BlockProcessingResult(
            block_id="blk-1",
            treatment=BlockTreatment.TRANSLATE,
            translated_text="Первый абзац.",
            warnings=(
                ResultWarning(kind=ResultWarningKind.REPETITION, message="Повтор 1."),
                ResultWarning(kind=ResultWarningKind.REPETITION, message="Повтор 2."),
            ),
        ),
    )
    list_calls = 0

    def _list_for_run(_run_id: str):
        nonlocal list_calls
        list_calls += 1
        return results

    def _make_uow(with_block_ids: bool):
        results_ns = SimpleNamespace(
            count_for_run=lambda run_id: 1,
            list_for_run=_list_for_run,
        )
        if with_block_ids:
            results_ns.block_ids_for_run = lambda run_id: frozenset({"blk-1"})
        tx = SimpleNamespace(
            books=SimpleNamespace(get=lambda book_id: book),
            documents=SimpleNamespace(
                get=lambda book_id: document,
                import_warnings=lambda book_id: (
                    BookImportWarning(
                        kind=ImportWarningKind.DROPPED_BLOCK,
                        message="Пропущен блок.",
                        section_id="sec-1",
                    ),
                ),
            ),
            analyses=SimpleNamespace(list_for_book=lambda book_id: ()),
            tasks=SimpleNamespace(list_for_book=lambda book_id: ()),
            runs=SimpleNamespace(active_for_book=lambda book_id: run),
            results=results_ns,
            llm_calls=SimpleNamespace(list_for_book=lambda book_id: ()),
            work=SimpleNamespace(
                latest_conditions=lambda book_id: None,
                list_roots=lambda book_id, run_id=None: (),
                list_reviews=lambda book_id: (),
            ),
        )

        class _StubUoW:
            @contextmanager
            def read_transaction(self):
                yield tx

            @contextmanager
            def transaction(self):
                yield tx

        return _StubUoW()

    # Without block_ids_for_run, stored_results is reused without a second list_for_run call.
    overview_reused = book_overview(_make_uow(with_block_ids=False), "book-1")  # type: ignore[arg-type]
    assert overview_reused.warning_count == 3
    assert list_calls == 1

    # With block_ids_for_run but without count_warnings_for_run, fallback calls list_for_run.
    list_calls = 0
    overview_fallback = book_overview(_make_uow(with_block_ids=True), "book-1")  # type: ignore[arg-type]
    assert overview_fallback.warning_count == 3
    assert list_calls == 1


def test_sqlalchemy_book_overview_warning_count_matches_book_warnings_total(
    tmp_path, monkeypatch
) -> None:
    """`SqlAlchemyBlockResultRepository.count_warnings_for_run` matches `BookWarnings.total` 1:1 without `list_for_run`."""
    engine = create_database_engine(tmp_path / "overview.db")
    Base.metadata.create_all(engine)
    uow = SqlAlchemyUnitOfWork.for_engine(engine)
    try:
        document = _sample_document()
        import_warnings = (
            BookImportWarning(
                kind=ImportWarningKind.DROPPED_BLOCK,
                message="Пропущен блок.",
                section_id="sec-1",
            ),
        )
        run = _processing_run()
        results = (
            BlockProcessingResult(
                block_id="blk-1",
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Первый абзац.",
                warnings=(
                    ResultWarning(kind=ResultWarningKind.REPETITION, message="Повтор."),
                ),
            ),
            BlockProcessingResult(
                block_id="blk-2",
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Второй абзац.",
                warnings=(
                    ResultWarning(kind=ResultWarningKind.REPETITION, message="Стиль."),
                    ResultWarning(kind=ResultWarningKind.QUALITY, message="Имя."),
                ),
            ),
        )
        with uow.transaction() as tx:
            tx.books.add(_book_record())
            tx.documents.save(document)
            tx.documents.add_import_warnings("book-1", import_warnings)
            tx.runs.add(run)
            tx.results.replace_results(run.id, results)

        expected_warnings = book_warnings(uow, "book-1")
        assert expected_warnings.total == 4

        monkeypatch.setattr(
            SqlAlchemyBlockResultRepository,
            "list_for_run",
            lambda self, run_id: (_ for _ in ()).throw(
                AssertionError("book_overview must not call list_for_run while processing")
            ),
        )
        overview = book_overview(uow, "book-1")
        assert overview.warning_count == expected_warnings.total
    finally:
        engine.dispose()
