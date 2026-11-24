"""Integration tests for primary queue priority and full product isolation (`P4.T2`)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.runtime.task_store import SqliteTaskStore
from linguary.application.book_overview import book_overview, book_result
from linguary.application.ports.persistence import BookRecord
from linguary.application.use_cases.export_book import (
    ExportBook,
    ExportBookCommand,
    ExportMode,
)
from linguary.application.use_cases.retry_task import (
    RetryTask,
    RetryTaskCommand,
    RetryTaskError,
    evaluate_task_retryable,
    is_task_retryable_for_recovery,
)
from linguary.cli.app import main
from linguary.composition import build_application
from linguary.composition.migrations import apply_migrations
from linguary.domain import (
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    ErrorType,
    LLMCallLog,
    LLMCallStage,
    ProcessingTask,
    SourceReference,
    SupportStatus,
    TaskError,
    TaskIdentity,
    TaskStatus,
    TaskType,
    TokenUsage,
)
from linguary.domain.book_conditions import BookConditions


def _seed_basic_book(uow: SqlAlchemyUnitOfWork, book_id: str = "book-iso") -> tuple[str, str]:
    cond_id = str(uuid4())
    sec = DocumentSection(
        id=f"{book_id}-sec-0",
        document_id=book_id,
        title="Chapter 1",
        order=0,
        source_resource_path="Text/ch1.xhtml",
        blocks=(
            DocumentBlock(
                id=f"{book_id}:b0",
                section_id=f"{book_id}-sec-0",
                type=BlockType.PARAGRAPH,
                order=0,
                text="Alice opened the heavy door.",
                source_reference=SourceReference("Text/ch1.xhtml", "p[0]"),
                support_status=SupportStatus.SUPPORTED,
            ),
        ),
    )
    doc = CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title="Isolation Novel"),
        source_language="en",
        target_language="ru",
        sections=(sec,),
    )
    with uow.transaction() as tx:
        tx.books.add(
            BookRecord(
                id=book_id,
                metadata=doc.metadata,
                source_language="en",
                target_language="ru",
                status=BookStatus.READY_TO_EXPORT,
                source_epub_path="/tmp/fake.epub",
                created_at=datetime(2026, 11, 1, tzinfo=UTC),
            )
        )
        tx.documents.save(doc)
        tx.work.add_conditions(
            BookConditions(
                id=cond_id,
                book_id=book_id,
                revision=1,
                payload={"schema_version": 1},
                created_at=datetime(2026, 11, 1, tzinfo=UTC),
            )
        )
    return sec.id, cond_id


def test_primary_tasks_claimed_before_illustrations(tmp_path: Path) -> None:
    db = tmp_path / "prio.db"
    apply_migrations(db)
    engine = create_database_engine(db)
    uow = SqlAlchemyUnitOfWork.for_engine(engine)
    store = SqliteTaskStore.for_engine(engine)
    _seed_basic_book(uow, "book-prio")

    # Enqueue illustration tasks FIRST (earlier created_at), then primary task LATER.
    store.enqueue(
        TaskIdentity(
            book_id="book-prio",
            task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
            scope_id="scope-ref",
            operation_id="gen1:a1",
            policy_version="illustration-v1",
        )
    )
    store.enqueue(
        TaskIdentity(
            book_id="book-prio",
            task_type=TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
            scope_id="scope-ch",
            operation_id="gen1:a1",
            policy_version="illustration-v1",
        )
    )
    store.enqueue(
        TaskIdentity(
            book_id="book-prio",
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="book-prio-sec-0",
        )
    )

    first = store.claim_next_pending()
    second = store.claim_next_pending()
    third = store.claim_next_pending()
    assert first is not None and first.task_type is TaskType.ANALYZE_CHAPTER
    assert second is not None and second.task_type is TaskType.PREPARE_ILLUSTRATION_REFERENCE
    assert third is not None and third.task_type is TaskType.PREPARE_CHAPTER_ILLUSTRATIONS


def test_book_overview_excludes_illustration_tasks_and_ordinary_cost(tmp_path: Path) -> None:
    db = tmp_path / "ov.db"
    apply_migrations(db)
    uow = SqlAlchemyUnitOfWork.for_engine(create_database_engine(db))
    _seed_basic_book(uow, "book-ov")

    with uow.transaction() as tx:
        # Pending, running, and failed illustration tasks
        tx.tasks.add(
            ProcessingTask(
                id="t-ill-pend",
                book_id="book-ov",
                task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
                scope_id="sc-1",
                status=TaskStatus.PENDING,
                operation_id="g1:a1",
                policy_version="illustration-v1",
            )
        )
        tx.tasks.add(
            ProcessingTask(
                id="t-ill-fail",
                book_id="book-ov",
                task_type=TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
                scope_id="sc-2",
                status=TaskStatus.FAILED,
                attempts=1,
                last_error=TaskError(type=ErrorType.TRANSIENT, message="provider_error"),
                operation_id="g1:a1",
                policy_version="illustration-v1",
            )
        )
        # LLM call logs: 1 ordinary ($0.10) + 2 illustration ($2.00 + $3.00)
        tx.llm_calls.add(
            LLMCallLog(
                id="c-ord",
                book_id="book-ov",
                stage=LLMCallStage.CHAPTER_ANALYSIS,
                provider="openai_compatible",
                model="gpt-5-mini",
                started_at=datetime(2026, 11, 1, tzinfo=UTC),
                duration_ms=100,
                attempt=1,
                success=True,
                usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
                cost=Decimal("0.10"),
            )
        )
        tx.llm_calls.add(
            LLMCallLog(
                id="c-ill-ref",
                book_id="book-ov",
                stage=LLMCallStage.ILLUSTRATION_REFERENCE,
                provider="openai_compatible",
                model="gpt-5-mini",
                started_at=datetime(2026, 11, 1, tzinfo=UTC),
                duration_ms=100,
                attempt=1,
                success=True,
                usage=TokenUsage(prompt_tokens=1000, completion_tokens=500),
                cost=Decimal("2.00"),
            )
        )
        tx.llm_calls.add(
            LLMCallLog(
                id="c-ill-ch",
                book_id="book-ov",
                stage=LLMCallStage.ILLUSTRATION_CHAPTER,
                provider="openai_compatible",
                model="gpt-5-mini",
                started_at=datetime(2026, 11, 1, tzinfo=UTC),
                duration_ms=100,
                attempt=1,
                success=True,
                usage=TokenUsage(prompt_tokens=1000, completion_tokens=500),
                cost=Decimal("3.00"),
            )
        )

    ov = book_overview(uow, "book-ov")
    assert ov.progress.pending_tasks == 0
    assert ov.progress.running_tasks == 0
    assert ov.progress.failures == ()
    assert Decimal(ov.ordinary_cost["known_spent_usd"]) == Decimal("0.10")


def test_book_overview_and_result_is_paused_consistent_while_illustration_running(
    tmp_path: Path,
) -> None:
    db = tmp_path / "pause.db"
    apply_migrations(db)
    uow = SqlAlchemyUnitOfWork.for_engine(create_database_engine(db))
    _, cond_id = _seed_basic_book(uow, "book-p")

    with uow.transaction() as tx:
        tx.work.set_book_paused("book-p", cond_id, True)
        tx.tasks.add(
            ProcessingTask(
                id="t-ill-run",
                book_id="book-p",
                task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
                scope_id="sc-run",
                status=TaskStatus.RUNNING,
                attempts=1,
                operation_id="g1:a1",
                policy_version="illustration-v1",
            )
        )

    ov = book_overview(uow, "book-p")
    res = book_result(uow, "book-p")
    # While an illustration task is still RUNNING, pause_requested=True but is_paused=False in BOTH views!
    assert ov.pause_requested is True
    assert ov.is_paused is False
    assert res.pause_requested is True
    assert res.is_paused is False


def test_generic_retry_task_and_helpers_reject_illustration_types(tmp_path: Path) -> None:
    db = tmp_path / "retry.db"
    apply_migrations(db)
    uow = SqlAlchemyUnitOfWork.for_engine(create_database_engine(db))
    _seed_basic_book(uow, "book-rt")

    for idx, ttype in enumerate(
        (TaskType.PREPARE_ILLUSTRATION_REFERENCE, TaskType.PREPARE_CHAPTER_ILLUSTRATIONS)
    ):
        task = ProcessingTask(
            id=f"t-ill-{idx}",
            book_id="book-rt",
            task_type=ttype,
            scope_id=f"sc-{idx}",
            status=TaskStatus.FAILED,
            attempts=1,
            last_error=TaskError(type=ErrorType.TRANSIENT, message="provider_error"),
            operation_id="g1:a1",
            policy_version="illustration-v1",
        )
        with uow.transaction() as tx:
            tx.tasks.add(task)

        allowed_eval, reason_eval, _ = evaluate_task_retryable(
            task, active_run=None, conditions=None, binding=None, roots_by_id={}
        )
        assert allowed_eval is False and reason_eval is not None

        with uow.read_transaction() as tx:
            allowed_rec, reason_rec = is_task_retryable_for_recovery(tx, task)
            assert allowed_rec is False and reason_rec is not None

        with pytest.raises(RetryTaskError):
            RetryTask(unit_of_work=uow)(RetryTaskCommand(task_id=task.id))


def test_failed_illustration_does_not_block_export(tmp_path: Path) -> None:
    db = tmp_path / "exp.db"
    apply_migrations(db)
    uow = SqlAlchemyUnitOfWork.for_engine(create_database_engine(db))
    _seed_basic_book(uow, "book-exp")

    with uow.transaction() as tx:
        tx.tasks.add(
            ProcessingTask(
                id="t-ill-failed",
                book_id="book-exp",
                task_type=TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
                scope_id="sc-fail",
                status=TaskStatus.FAILED,
                attempts=1,
                last_error=TaskError(type=ErrorType.PERMANENT, message="provider_error"),
                operation_id="g1:a1",
                policy_version="illustration-v1",
            )
        )

    class _StubWriter:
        def write(self, request: Any) -> Any:
            from linguary.application.ports.epub import ExportedBook

            Path(request.target_path).write_bytes(b"epub-ok")
            return ExportedBook(path=request.target_path, warnings=())

    class _StubValidator:
        def validate(self, _request: Any) -> Any:
            from linguary.application.ports.export_validation import ExportValidationReport

            return ExportValidationReport(available=False, unavailable_reason="stub")

    out_path = tmp_path / "out.epub"
    exporter = ExportBook(unit_of_work=uow, writer=_StubWriter(), validator=_StubValidator())  # type: ignore[arg-type]
    result = exporter(
        ExportBookCommand(
            book_id="book-exp",
            target_path=str(out_path),
            mode=ExportMode.STRUCTURE_ONLY,
        )
    )
    assert result.book_id == "book-exp"
    assert out_path.exists()


def test_cli_worker_once_isolated_from_optional_illustration_failure(
    tmp_path: Path, capsys: Any
) -> None:
    data_root = tmp_path / "data"
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"""
[storage]
data_root = "{data_root.as_posix()}"

[llm]
provider = "openai_compatible"
base_url = "https://api.openai.com/v1"
model = "gpt-5-mini"
api_key = ""
""".strip()
        + "\n",
        encoding="utf-8",
    )

    app = build_application(
        settings=AppSettings(
            storage=StorageSettings(data_root=str(data_root)),
            llm=LLMSettings(api_key=None, model="gpt-5-mini"),
        )
    )
    try:
        _seed_basic_book(app.unit_of_work, "book-cli-w")
        # Enqueue an illustration task whose scope fails with route_unavailable (api_key is empty)
        from linguary.domain.illustrations import (
            IllustrationGenerationState,
            IllustrationScopeSpec,
            IllustrationScopeStage,
            IllustrationSettings,
        )

        with app.unit_of_work.transaction() as tx:
            rev = tx.illustrations.save_settings(
                "book-cli-w",
                expected_revision=0,
                settings=IllustrationSettings(
                    enabled=True,
                    max_per_chapter=3,
                    model_reference="gpt-5-mini",
                    model_prompts="gpt-5-mini",
                ),
            )
            gen, _ = tx.illustrations.create_or_get_generation(
                "book-cli-w",
                settings_id=rev.id,
                request_key="req-cli-w",
                source_sha256="a" * 64,
                source_authority={},
                requested_sections=("book-cli-w-sec-0",),
            )
            tx.illustrations.update_generation_state(
                "book-cli-w", gen.id, state=IllustrationGenerationState.QUEUED
            )
            sc = tx.illustrations.upsert_scopes(
                gen.id,
                [
                    IllustrationScopeSpec(
                        stage=IllustrationScopeStage.EXTRACT,
                        scope_key="extract:p0",
                        input_sha256="b" * 64,
                        summary={"request_input": {"units": []}},
                    )
                ],
            )[0]
            from linguary.application.illustrations import dispatch_scope_task

            dispatch_scope_task(
                tx,
                sc,
                generation_id=gen.id,
                task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
                id_factory=app.task_queue.new_task_id,
                book_id="book-cli-w",
            )
    finally:
        asyncio.run(app.aclose())

    rc = main(["--config", str(cfg), "worker", "--once"])
    out = capsys.readouterr().out
    # Exit code MUST be 0 because primary_failed == 0!
    assert rc == 0
    assert "Tasks failed: 0" in out
    assert "Optional illustration tasks failed:" in out
