"""Synthetic local attestations, never real provider tariffs."""
from dataclasses import replace
from decimal import Decimal
import json
import pytest
from linguary.adapters.config import LLMSettings
from linguary.adapters.config.quality_routes import load_quality_routes, PROTOCOL
from linguary.application.ports.llm import LLMRequest, LLMResponseKind
from linguary.domain.book_conditions import WorkStateError


def fixture_route():
    return dict(route_id='synthetic', model='review-test', base_url='https://example.invalid/v1',
        price_version='synthetic-only-v1', currency='USD', input_per_mtok='1.00000000000001',
        output_per_mtok='2', max_fee_usd='0.000001', max_output_tokens=80,
        max_reasoning_tokens=20, max_completion_tokens=100, supports_max_completion_tokens=True,
        completion_includes_reasoning=True, input_bound=dict(protocol=PROTOCOL,
            max_request_bytes=100000, max_input_tokens=200000,
            proof_method='Synthetic test assertion; not valid for a real provider',
            proof_version='fixture-v1', evidence_uri='urn:linguary:synthetic-test'))


def load(tmp_path, monkeypatch, item):
    path = tmp_path / 'quality.json'
    path.write_text(json.dumps(dict(schema_version=1, routes=[item])))
    monkeypatch.setenv('LINGUARY_QUALITY_ROUTES', str(path))
    return load_quality_routes(LLMSettings(base_url='https://example.invalid/v1'))


def test_exact_prices_and_full_wire_bound(tmp_path, monkeypatch):
    item = fixture_route()
    registry = load(tmp_path, monkeypatch, item)
    route = registry['synthetic', 'review-test']
    assert route.input_per_mtok == Decimal('1.00000000000001')
    request = LLMRequest(kind=LLMResponseKind.QUALITY_REVIEW, instructions='Review', input_text='{}', max_completion_tokens=100)
    assert route.estimate(request) > 0
    # Even tiny user content includes the strict schema and message framing.
    item['input_bound']['max_request_bytes'] = 100
    route = load(tmp_path, monkeypatch, item)['synthetic', 'review-test']
    with pytest.raises(WorkStateError, match='полный запрос'):
        route.estimate(request)
    with pytest.raises(WorkStateError):
        route.estimate(replace(request, max_completion_tokens=101))


@pytest.mark.parametrize('field', sorted(fixture_route()))
def test_every_route_attestation_is_required(tmp_path, monkeypatch, field):
    item = fixture_route()
    del item[field]
    assert load(tmp_path, monkeypatch, item) == {}


@pytest.mark.parametrize('field', sorted(fixture_route()['input_bound']))
def test_every_input_proof_field_is_required(tmp_path, monkeypatch, field):
    item = fixture_route()
    del item['input_bound'][field]
    assert load(tmp_path, monkeypatch, item) == {}


@pytest.mark.parametrize('field,value', [('input_per_mtok', 1.0), ('max_fee_usd', 'NaN'),
    ('output_per_mtok', '-1'), ('base_url', 'https://attacker.invalid/v1'),
    ('supports_max_completion_tokens', False), ('completion_includes_reasoning', False),
    ('max_completion_tokens', 80), ('currency', 'EUR'), ('max_reasoning_tokens', True)])
def test_invalid_or_untrusted_routes_disabled(tmp_path, monkeypatch, field, value):
    item = fixture_route()
    item[field] = value
    assert load(tmp_path, monkeypatch, item) == {}


def test_default_off_and_unreadable_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv('LINGUARY_QUALITY_ROUTES', raising=False)
    assert load_quality_routes(LLMSettings()) == {}
    monkeypatch.setenv('LINGUARY_QUALITY_ROUTES', str(tmp_path / 'missing'))
    assert load_quality_routes(LLMSettings()) == {}


def test_money_rounds_up_without_decimal_context_loss():
    from linguary.application.quality_budget import microusd, QualityRoute
    assert microusd(Decimal('0.0000010000000000000000000000000000000000001')) == 2
    route = QualityRoute('model', 'route', 'price',
        Decimal('1.0000000000000000000000000000000000001'), Decimal('0'), Decimal('0'),
        1, lambda request: 1, 'synthetic', True)
    assert route.charge(1, 0) == 2


def test_quality_monitor_counts_retries_across_all_pipeline_task_types():
    from contextlib import contextmanager
    from types import SimpleNamespace
    from linguary.domain import ErrorType, ProcessingTask, TaskError, TaskStatus, TaskType
    from linguary.web.labels import task_type_label
    from linguary.web.routes.quality import quality_page_context

    assert task_type_label(TaskType.QUALITY_REVIEW) == "Вычитка качества"

    pipeline_types = (
        TaskType.ANALYZE_CHAPTER,
        TaskType.PROCESS_TRANSLATION_UNIT,
        TaskType.REGENERATE_BLOCK,
        TaskType.CHAPTER_CORRECTION,
        TaskType.EXPORT_BOOK,
        TaskType.QUALITY_REVIEW,
    )
    tasks = tuple(
        ProcessingTask(
            id=f"task-{task_type.value}",
            book_id="book-1",
            task_type=task_type,
            scope_id="scope-1",
            status=TaskStatus.FAILED,
            attempts=1,
            last_error=TaskError(type=ErrorType.TRANSIENT, message="temporary failure"),
        )
        for task_type in (*pipeline_types, TaskType.IMPORT_BOOK)
    )

    tx = SimpleNamespace(
        documents=SimpleNamespace(get=lambda book_id: None),
        llm_calls=SimpleNamespace(list_for_book=lambda book_id: ()),
        tasks=SimpleNamespace(list_for_book=lambda book_id: tasks),
        work=SimpleNamespace(latest_conditions=lambda book_id: None),
        analyses=SimpleNamespace(list_for_book=lambda book_id: ()),
        runs=SimpleNamespace(active_for_book=lambda book_id: None),
    )

    class _StubUoW:
        @contextmanager
        def transaction(self):
            yield tx

    app = SimpleNamespace(
        unit_of_work=_StubUoW(),
        runtime_settings=SimpleNamespace(revision=1),
        settings=SimpleNamespace(llm=SimpleNamespace(is_ready=True)),
        quality_review=SimpleNamespace(
            preview=lambda book_id, scope: None,
            routes={},
        ),
    )
    request = SimpleNamespace(
        headers={},
        query_params={},
        client=SimpleNamespace(host="127.0.0.1"),
    )

    ctx = quality_page_context(request, app, "book-1", tasks=tasks)
    assert [item["id"] for item in ctx["retry_tasks"]] == [
        f"task-{task_type.value}" for task_type in pipeline_types
    ]
    assert ctx["retry_tasks"][-1]["label"] == "Вычитка качества"


def test_quality_page_context_skips_preview_when_no_routes_and_pause_preserves_pending_tasks(tmp_path):
    from datetime import UTC, datetime
    from types import SimpleNamespace
    from uuid import uuid4
    from linguary.adapters.db.engine import create_database_engine
    from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
    from linguary.adapters.runtime.task_store import SqliteTaskStore
    from linguary.application.book_overview import book_overview
    from linguary.application.ports.persistence import BookRecord
    from linguary.application.use_cases.pause_book import PauseBook, PauseBookCommand, ResumeBook, ResumeBookCommand
    from linguary.application.use_cases.reset_processing import ResetProcessing, ResetProcessingCommand
    from linguary.composition.migrations import apply_migrations
    from linguary.domain import (
        BookMetadata,
        BookStatus,
        CanonicalDocument,
        DocumentBlock,
        DocumentSection,
        BlockType,
        SourceReference,
        SupportStatus,
        TaskIdentity,
        TaskStatus,
        TaskType,
    )
    from linguary.domain.book_conditions import BookConditions
    from linguary.web.routes.books import _needs_refresh
    from linguary.web.routes.quality import quality_page_context
    from linguary.web.schemas import BookDetailResponse

    db_path = tmp_path / "pause_and_quality.db"
    apply_migrations(db_path)
    engine = create_database_engine(db_path)
    try:
        uow = SqlAlchemyUnitOfWork.for_engine(engine)
        store = SqliteTaskStore.for_engine(engine)

        now = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
        for bid in ("book-paused", "book-other"):
            epub_path = tmp_path / f"{bid}.epub"
            epub_path.write_bytes(b"synthetic")
            section = DocumentSection(
                id=f"{bid}:s1",
                document_id=bid,
                order=0,
                title="Chapter 1",
                source_resource_path=f"{bid}-ch1.xhtml",
                blocks=(
                    DocumentBlock.create(
                        document_id=bid,
                        section_id=f"{bid}:s1",
                        order=0,
                        type=BlockType.PARAGRAPH,
                        text="Hello world paragraph text for estimation.",
                        source_reference=SourceReference(
                            resource_path=f"{bid}-ch1.xhtml",
                            structural_path="/body/p[1]",
                        ),
                        support_status=SupportStatus.SUPPORTED,
                    ),
                ),
            )
            doc = CanonicalDocument(
                id=bid,
                source_language="en",
                target_language="ru",
                metadata=BookMetadata(title=f"Title {bid}", authors=("Author",)),
                sections=(section,),
            )
            with uow.transaction() as tx:
                tx.books.add(
                    BookRecord(
                        id=bid,
                        source_epub_path=str(epub_path),
                        source_language="en",
                        target_language="ru",
                        status=BookStatus.PROCESSING,
                        metadata=doc.metadata,
                        created_at=now,
                    )
                )
                tx.documents.save(doc)
                tx.work.add_conditions(
                    BookConditions(
                        id=str(uuid4()),
                        book_id=bid,
                        revision=1,
                        schema_version=1,
                        payload={
                            "schema_version": 1,
                            "reader_level": "b2",
                            "english_help": False,
                            "profile": "legacy-v1",
                            "models": {"analysis": "m1", "processing": "m1"},
                            "versions": {"validators": 1},
                        },
                        reproducibility="complete",
                        created_at=now,
                    )
                )

        paused_task = store.enqueue(
            TaskIdentity(
                book_id="book-paused",
                task_type=TaskType.PROCESS_TRANSLATION_UNIT,
                scope_id="book-paused:s1",
            )
        )
        other_task = store.enqueue(
            TaskIdentity(
                book_id="book-other",
                task_type=TaskType.PROCESS_TRANSLATION_UNIT,
                scope_id="book-other:s1",
            )
        )

        pause_uc = PauseBook(unit_of_work=uow)
        resume_uc = ResumeBook(unit_of_work=uow)
        pause_result = pause_uc(PauseBookCommand(book_id="book-paused"))
        assert pause_result.pause_requested is True
        assert pause_result.is_paused is True

        # Claiming skips the paused book while preserving its PENDING status, and claims the other book.
        claimed = store.claim_next_pending()
        assert claimed is not None and claimed.id == other_task.id
        assert store.claim_next_pending() is None
        with uow.read_transaction() as tx:
            stored_paused = tx.tasks.get(paused_task.id)
            assert stored_paused is not None and stored_paused.status is TaskStatus.PENDING

        overview = book_overview(
            uow,
            "book-paused",
            input_price_per_mtok=1.0,
            output_price_per_mtok=2.0,
        )
        assert overview.pause_requested is True
        assert overview.is_paused is True
        assert set(overview.stage_estimate) == {
            "stage",
            "sections_count",
            "blocks_count",
            "characters_count",
            "input_tokens",
            "output_tokens",
            "estimated_cost_usd",
        }
        assert overview.stage_estimate["estimated_cost_usd"] is not None
        detail_response = BookDetailResponse.of_overview(overview)
        assert detail_response.is_paused is True
        assert _needs_refresh(detail_response) is False

        preview_called = False

        def fail_preview(book_id, scope="all"):
            nonlocal preview_called
            preview_called = True
            raise AssertionError("preview must not be called when routes is empty")

        tx_opened = 0

        class _CountingUoW:
            def read_transaction(self):
                nonlocal tx_opened
                tx_opened += 1
                return uow.read_transaction()

            def transaction(self):
                nonlocal tx_opened
                tx_opened += 1
                return uow.transaction()

        app = SimpleNamespace(
            unit_of_work=_CountingUoW(),
            runtime_settings=SimpleNamespace(revision=1),
            settings=SimpleNamespace(llm=SimpleNamespace(is_ready=True)),
            quality_review=SimpleNamespace(preview=fail_preview, routes={}),
        )
        request = SimpleNamespace(
            headers={},
            query_params={},
            client=SimpleNamespace(host="127.0.0.1"),
        )
        ctx = quality_page_context(request, app, "book-paused", overview=overview)
        assert preview_called is False
        assert tx_opened == 0
        assert ctx["forms"] == []

        # Resume unblocks claiming the preserved pending task.
        resume_result = resume_uc(ResumeBookCommand(book_id="book-paused"))
        assert resume_result.pause_requested is False
        assert resume_result.is_paused is False
        claimed_after_resume = store.claim_next_pending()
        assert claimed_after_resume is not None and claimed_after_resume.id == paused_task.id

        # Pause again and verify ResetProcessing clears the pause binding.
        pause_uc(PauseBookCommand(book_id="book-paused"))
        ResetProcessing(unit_of_work=uow)(ResetProcessingCommand(book_id="book-paused"))
        with uow.read_transaction() as tx:
            assert tx.work.is_book_paused("book-paused") is False
    finally:
        engine.dispose()


