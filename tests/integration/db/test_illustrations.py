"""Storage integration tests for optional book illustration persistence (SQL-01, P1.T1)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from persistence_factories import make_book
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError

from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.models import Base
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.domain import (
    BlockType,
    BookMetadata,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    ErrorType,
    ProcessingTask,
    SourceReference,
    SupportStatus,
    TaskError,
    TaskStatus,
    TaskType,
)
from linguary.domain.content_plan import canonical_sha
from linguary.domain.illustrations import (
    IllustrationAttemptState,
    IllustrationConflictError,
    IllustrationGenerationState,
    IllustrationLimitError,
    IllustrationNotFoundError,
    IllustrationOwnershipError,
    IllustrationReasonCode,
    IllustrationRegistryKind,
    IllustrationScopeSpec,
    IllustrationScopeStage,
    IllustrationScopeState,
    IllustrationSettings,
    RegistryItem,
    format_position_key,
)
from linguary.web.labels import TASK_STATUS_LABELS, TASK_TYPE_LABELS


def _enabled_settings(
    *,
    max_per_chapter: int = 3,
    model_reference: str = "openai/gpt-5-mini",
    model_prompts: str = "openai/gpt-5",
) -> IllustrationSettings:
    return IllustrationSettings(
        enabled=True,
        max_per_chapter=max_per_chapter,
        model_reference=model_reference,
        model_prompts=model_prompts,
    )


def _seed_book(uow: SqlAlchemyUnitOfWork, book_id: str = "book-1") -> None:
    section_id = f"{book_id}-sec-1"
    block_id = f"{book_id}-blk-1"
    doc = CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title=f"Book {book_id}"),
        source_language="en",
        target_language="ru",
        sections=(
            DocumentSection(
                id=section_id,
                document_id=book_id,
                title="Chapter 1",
                order=0,
                source_resource_path="Text/ch1.xhtml",
                blocks=(
                    DocumentBlock(
                        id=block_id,
                        section_id=section_id,
                        type=BlockType.PARAGRAPH,
                        order=0,
                        text="Sample chapter paragraph.",
                        source_reference=SourceReference("Text/ch1.xhtml", "p[0]"),
                        support_status=SupportStatus.SUPPORTED,
                    ),
                ),
            ),
        ),
    )
    with uow.transaction() as tx:
        tx.books.add(make_book(book_id=book_id))
        tx.documents.save(doc)


def _seed_generation(
    uow: SqlAlchemyUnitOfWork,
    *,
    book_id: str = "book-1",
    request_key: str = "req-1",
    source_sha: str = "a" * 64,
    sections: tuple[str, ...] = ("section-1",),
):
    with uow.transaction() as tx:
        head = tx.illustrations.get_head(book_id)
        rev = tx.illustrations.save_settings(
            book_id,
            expected_revision=head.revision_number,
            settings=_enabled_settings(),
        )
        gen, created = tx.illustrations.create_or_get_generation(
            book_id,
            settings_id=rev.id,
            request_key=request_key,
            source_sha256=source_sha,
            source_authority={"conditions_id": str(uuid4()), "conditions_revision": 1, "source_digest": source_sha, "manifest_sha256": None},
            requested_sections=sections,
        )
        return rev, gen, created


def test_default_disabled_no_tasks(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    _seed_book(unit_of_work)
    with unit_of_work.read_transaction() as tx:
        head = tx.illustrations.get_head("book-1")
        assert head.settings_id is None
        assert head.active_generation_id is None
        assert head.revision_number == 0
        assert head.effective_settings.enabled is False
        assert head.effective_settings.max_per_chapter == 3
        assert head.effective_settings.style == "monochrome_ink"
        assert head.effective_settings.model_reference is None
        assert head.effective_settings.model_prompts is None
        assert tx.tasks.list_for_book("book-1") == ()


def test_settings_compare_and_swap(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    _seed_book(unit_of_work)
    first = _enabled_settings(max_per_chapter=2)
    second = _enabled_settings(max_per_chapter=3)

    with unit_of_work.transaction() as tx:
        rev1 = tx.illustrations.save_settings("book-1", expected_revision=0, settings=first)
        assert rev1.revision == 1
        assert rev1.settings == first

        # Same payload at current revision -> no-op, no new revision.
        noop = tx.illustrations.save_settings("book-1", expected_revision=1, settings=first)
        assert noop.id == rev1.id
        assert noop.revision == 1

        # Same payload at current - 1 -> idempotent replay.
        replay = tx.illustrations.save_settings("book-1", expected_revision=0, settings=first)
        assert replay.id == rev1.id
        assert replay.revision == 1

        # Conflicting payload at stale or future revision -> revision_conflict.
        with pytest.raises(IllustrationConflictError) as stale_err:
            tx.illustrations.save_settings("book-1", expected_revision=0, settings=second)
        assert stale_err.value.reason_code == IllustrationReasonCode.REVISION_CONFLICT

        with pytest.raises(IllustrationConflictError) as future_err:
            tx.illustrations.save_settings("book-1", expected_revision=5, settings=second)
        assert future_err.value.reason_code == IllustrationReasonCode.REVISION_CONFLICT

        rev2 = tx.illustrations.save_settings("book-1", expected_revision=1, settings=second)
        assert rev2.revision == 2
        assert rev2.id != rev1.id


def test_migration_0018_to_0019_round_trip(alembic_config: Config, database_path: Path) -> None:
    command.upgrade(alembic_config, "0018")
    engine = create_database_engine(database_path)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO books (id, title, authors_json, source_language, "
                    "target_language, status, source_epub_path, created_at) "
                    "VALUES ('keep-book', 'Keep', '[]', 'en', 'ru', 'imported', "
                    "'source.epub', '2026-10-10 00:00:00')"
                )
            )
            before = conn.execute(text("SELECT * FROM books")).all()
        command.upgrade(alembic_config, "0019")
        engine.dispose()
        tables = set(inspect(engine).get_table_names())
        assert {
            "illustration_settings",
            "illustration_generations",
            "illustration_heads",
            "illustration_scopes",
            "illustration_attempts",
            "illustration_registry_items",
        } <= tables
        with engine.connect() as conn:
            assert conn.execute(text("SELECT * FROM books")).all() == before
        command.downgrade(alembic_config, "0018")
        engine.dispose()
        tables_after = set(inspect(engine).get_table_names())
        assert "illustration_settings" not in tables_after
        assert "illustration_generations" not in tables_after
        with engine.connect() as conn:
            assert conn.execute(text("SELECT * FROM books")).all() == before
    finally:
        engine.dispose()


def test_migration_0019_downgrade_with_populated_rows(
    alembic_config: Config, engine: Engine, unit_of_work: SqlAlchemyUnitOfWork
) -> None:
    _seed_book(unit_of_work)
    _rev, gen, _ = _seed_generation(unit_of_work)
    with unit_of_work.transaction() as tx:
        scopes = tx.illustrations.upsert_scopes(
            gen.id,
            [
                IllustrationScopeSpec(
                    stage=IllustrationScopeStage.EXTRACT,
                    scope_key="extract-0",
                    input_sha256="b" * 64,
                    section_id="section-1",
                    section_order=0,
                    section_title="Chapter 1",
                )
            ],
        )
        tx.illustrations.claim_scope(
            scopes[0].id,
            task_id="task-1",
            expected_generation_id=gen.id,
            max_calls=3,
        )
        tx.illustrations.upsert_registry_items(
            gen.id,
            [
                RegistryItem(
                    generation_id=gen.id,
                    kind=IllustrationRegistryKind.ENTITY,
                    item_key="ent-1",
                    entity_id="ent-1",
                    position_key=format_position_key(0, 0, 0),
                    is_back_applicable=False,
                    payload_sha256=canonical_sha({"name": "Alice"}),
                    payload={"name": "Alice"},
                )
            ],
        )

    command.downgrade(alembic_config, "0018")
    engine.dispose()
    tables = set(inspect(engine).get_table_names())
    assert "illustration_registry_items" not in tables
    assert "illustration_settings" not in tables
    assert "books" in tables

    command.upgrade(alembic_config, "head")


def test_nullable_json_columns_store_sql_null(
    engine: Engine, unit_of_work: SqlAlchemyUnitOfWork
) -> None:
    _seed_book(unit_of_work)
    _rev, gen, _ = _seed_generation(unit_of_work)
    with unit_of_work.transaction() as tx:
        scopes = tx.illustrations.upsert_scopes(
            gen.id,
            [
                IllustrationScopeSpec(
                    stage=IllustrationScopeStage.EXTRACT,
                    scope_key="extract-null",
                    input_sha256="c" * 64,
                )
            ],
        )
        tx.illustrations.claim_scope(
            scopes[0].id,
            task_id="task-1",
            expected_generation_id=gen.id,
            max_calls=3,
        )

    with engine.connect() as conn:
        assert conn.execute(
            text("SELECT manifest_json IS NULL, typeof(manifest_json) FROM illustration_generations")
        ).one() == (1, "null")
        assert conn.execute(
            text("SELECT payload_json IS NULL, typeof(payload_json) FROM illustration_scopes")
        ).one() == (1, "null")
        assert conn.execute(
            text("SELECT outcome_json IS NULL, typeof(outcome_json) FROM illustration_attempts")
        ).one() == (1, "null")


def test_two_claims_one_owner(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    _seed_book(unit_of_work)
    _rev, gen, _ = _seed_generation(unit_of_work)
    with unit_of_work.transaction() as tx:
        (scope,) = tx.illustrations.upsert_scopes(
            gen.id,
            [
                IllustrationScopeSpec(
                    stage=IllustrationScopeStage.EXTRACT,
                    scope_key="extract-claim",
                    input_sha256="d" * 64,
                )
            ],
        )
        attempt = tx.illustrations.claim_scope(
            scope.id,
            task_id="task-1",
            expected_generation_id=gen.id,
            max_calls=3,
        )
        assert attempt.ordinal == 1
        assert attempt.state is IllustrationAttemptState.RESERVED

        with pytest.raises(IllustrationOwnershipError):
            tx.illustrations.claim_scope(
                scope.id,
                task_id="task-2",
                expected_generation_id=gen.id,
                max_calls=3,
            )


def test_db_rejects_two_active_attempts_per_scope(
    engine: Engine, unit_of_work: SqlAlchemyUnitOfWork
) -> None:
    _seed_book(unit_of_work)
    _rev, gen, _ = _seed_generation(unit_of_work)
    with unit_of_work.transaction() as tx:
        (scope,) = tx.illustrations.upsert_scopes(
            gen.id,
            [
                IllustrationScopeSpec(
                    stage=IllustrationScopeStage.EXTRACT,
                    scope_key="extract-idx",
                    input_sha256="e" * 64,
                )
            ],
        )
        tx.illustrations.claim_scope(
            scope.id,
            task_id="task-1",
            expected_generation_id=gen.id,
            max_calls=3,
        )
        scope_id = scope.id

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO illustration_attempts "
                "(id, scope_id, task_id, ordinal, state, request_meta_json, created_at, updated_at) "
                "VALUES (:id, :scope_id, 'task-2', 2, 'reserved', '{}', :now, :now)"
            ),
            {"id": str(uuid4()), "scope_id": scope_id, "now": now},
        )


def test_scope_unique_rejects_duplicate_key_different_input_sha(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    _seed_book(unit_of_work)
    _rev, gen, _ = _seed_generation(unit_of_work)
    with unit_of_work.transaction() as tx:
        tx.illustrations.upsert_scopes(
            gen.id,
            [
                IllustrationScopeSpec(
                    stage=IllustrationScopeStage.EXTRACT,
                    scope_key="same-key",
                    input_sha256="1" * 64,
                )
            ],
        )
        with pytest.raises(IllustrationConflictError) as exc:
            tx.illustrations.upsert_scopes(
                gen.id,
                [
                    IllustrationScopeSpec(
                        stage=IllustrationScopeStage.EXTRACT,
                        scope_key="same-key",
                        input_sha256="2" * 64,
                    )
                ],
            )
        assert exc.value.reason_code == IllustrationReasonCode.GENERATION_CONFLICT


def test_split_scope_and_max_scopes_limit(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    _seed_book(unit_of_work)
    _rev, gen, _ = _seed_generation(unit_of_work)
    with unit_of_work.transaction() as tx:
        (parent,) = tx.illustrations.upsert_scopes(
            gen.id,
            [
                IllustrationScopeSpec(
                    stage=IllustrationScopeStage.EXTRACT,
                    scope_key="root-page",
                    input_sha256="3" * 64,
                    section_id="section-1",
                    section_order=0,
                    section_title="Chapter 1",
                )
            ],
        )
        attempt = tx.illustrations.claim_scope(
            parent.id,
            task_id="task-1",
            expected_generation_id=gen.id,
            max_calls=3,
        )
        children = [
            IllustrationScopeSpec(
                stage=IllustrationScopeStage.EXTRACT,
                scope_key="root-page:0",
                input_sha256="4" * 64,
            ),
            IllustrationScopeSpec(
                stage=IllustrationScopeStage.EXTRACT,
                scope_key="root-page:1",
                input_sha256="5" * 64,
            ),
        ]
        # Exceeding max_scopes raises SCOPE_LIMIT_EXHAUSTED before mutating parent.
        with pytest.raises(IllustrationLimitError) as limit_err:
            tx.illustrations.split_scope(
                parent.id,
                attempt_id=attempt.id,
                children=children,
                max_scopes=2,
            )
        assert limit_err.value.reason_code == IllustrationReasonCode.SCOPE_LIMIT_EXHAUSTED

        created = tx.illustrations.split_scope(
            parent.id,
            attempt_id=attempt.id,
            children=children,
            max_scopes=10,
        )
        assert len(created) == 2
        assert all(child.parent_scope_id == parent.id for child in created)
        refreshed_parent = tx.illustrations.get_scope(parent.id)
        assert refreshed_parent is not None
        assert refreshed_parent.state is IllustrationScopeState.SPLIT
        updated_gen = tx.illustrations.get_generation("book-1", gen.id)
        assert updated_gen is not None
        assert updated_gen.calls_reserved == 0


def test_cross_book_reuse_isolated(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    _seed_book(unit_of_work, "book-1")
    _seed_book(unit_of_work, "book-2")
    _rev1, gen1, _ = _seed_generation(unit_of_work, book_id="book-1")
    shared_input = "f" * 64
    shared_registry = "9" * 64

    with unit_of_work.transaction() as tx:
        (scope1,) = tx.illustrations.upsert_scopes(
            gen1.id,
            [
                IllustrationScopeSpec(
                    stage=IllustrationScopeStage.EXTRACT,
                    scope_key="extract-0",
                    input_sha256=shared_input,
                )
            ],
        )
        tx.illustrations.accept_scope(
            scope1.id,
            attempt_id=None,
            expected_input_sha256=shared_input,
            payload={"mentions": []},
            output_sha256="8" * 64,
            summary={"mentions": 0},
        )
        tx.illustrations.seal_registry("book-1", gen1.id, registry_sha256=shared_registry)

        assert tx.illustrations.find_reusable_scope(
            "book-1", stage=IllustrationScopeStage.EXTRACT, input_sha256=shared_input
        ) is not None
        assert tx.illustrations.find_reusable_scope(
            "book-2", stage=IllustrationScopeStage.EXTRACT, input_sha256=shared_input
        ) is None
        assert tx.illustrations.find_reusable_registry("book-1", registry_sha256=shared_registry) == gen1.id
        assert tx.illustrations.find_reusable_registry("book-2", registry_sha256=shared_registry) is None


def test_back_applicable_registry_query_includes_permanent_non_spoiler_facts(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    _seed_book(unit_of_work)
    _rev, gen, _ = _seed_generation(unit_of_work)
    early_pos = format_position_key(1, 0, 0)
    later_pos = format_position_key(10, 0, 0)

    items = [
        RegistryItem(
            generation_id=gen.id,
            kind=IllustrationRegistryKind.FACT,
            item_key="fact-early",
            entity_id="hero",
            position_key=early_pos,
            is_back_applicable=False,
            payload_sha256=canonical_sha({"text": "early"}),
            payload={"text": "early"},
        ),
        RegistryItem(
            generation_id=gen.id,
            kind=IllustrationRegistryKind.FACT,
            item_key="fact-late-scar",
            entity_id="hero",
            position_key=later_pos,
            is_back_applicable=False,
            payload_sha256=canonical_sha({"text": "scar in ch 10"}),
            payload={"text": "scar in ch 10"},
        ),
        RegistryItem(
            generation_id=gen.id,
            kind=IllustrationRegistryKind.FACT,
            item_key="fact-late-permanent-eyes",
            entity_id="hero",
            position_key=later_pos,
            is_back_applicable=True,
            payload_sha256=canonical_sha({"text": "green eyes"}),
            payload={"text": "green eyes"},
        ),
    ]
    with unit_of_work.transaction() as tx:
        tx.illustrations.upsert_registry_items(gen.id, items)

        strict_rows, strict_total = tx.illustrations.query_registry_items(
            gen.id,
            entity_id="hero",
            kind=IllustrationRegistryKind.FACT,
            max_position_key=format_position_key(2, 0, 0),
            include_back_applicable=False,
        )
        assert strict_total == 1
        assert [row.item_key for row in strict_rows] == ["fact-early"]

        back_rows, back_total = tx.illustrations.query_registry_items(
            gen.id,
            entity_id="hero",
            kind=IllustrationRegistryKind.FACT,
            max_position_key=format_position_key(2, 0, 0),
            include_back_applicable=True,
        )
        assert back_total == 2
        assert [row.item_key for row in back_rows] == [
            "fact-early",
            "fact-late-permanent-eyes",
        ]


def test_book_delete_cascades(engine: Engine, unit_of_work: SqlAlchemyUnitOfWork) -> None:
    _seed_book(unit_of_work)
    _rev, gen, _ = _seed_generation(unit_of_work)
    with unit_of_work.transaction() as tx:
        (scope,) = tx.illustrations.upsert_scopes(
            gen.id,
            [
                IllustrationScopeSpec(
                    stage=IllustrationScopeStage.EXTRACT,
                    scope_key="extract-0",
                    input_sha256="a" * 64,
                )
            ],
        )
        tx.illustrations.claim_scope(
            scope.id, task_id="task-1", expected_generation_id=gen.id, max_calls=3
        )
        tx.illustrations.upsert_registry_items(
            gen.id,
            [
                RegistryItem(
                    generation_id=gen.id,
                    kind=IllustrationRegistryKind.ENTITY,
                    item_key="e1",
                    entity_id="e1",
                    position_key=format_position_key(0, 0, 0),
                    is_back_applicable=False,
                    payload_sha256="a" * 64,
                    payload={"name": "Hero"},
                )
            ],
        )

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM books WHERE id = 'book-1'"))

    with engine.connect() as conn:
        for table in (
            "illustration_settings",
            "illustration_generations",
            "illustration_heads",
            "illustration_scopes",
            "illustration_attempts",
            "illustration_registry_items",
        ):
            assert conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one() == 0


def test_document_resave_preserves_historical_chapter_headers_order_and_titles(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    _seed_book(unit_of_work)
    _rev, gen, _ = _seed_generation(unit_of_work, sections=("old-sec-1", "old-sec-2"))
    with unit_of_work.transaction() as tx:
        scopes = tx.illustrations.upsert_scopes(
            gen.id,
            [
                IllustrationScopeSpec(
                    stage=IllustrationScopeStage.CHAPTER,
                    scope_key="chapter:old-sec-2",
                    input_sha256="2" * 64,
                    section_id="old-sec-2",
                    section_order=1,
                    section_title="Chapter Two Old",
                    state=IllustrationScopeState.ACCEPTED,
                    summary={"moment_count": 2},
                ),
                IllustrationScopeSpec(
                    stage=IllustrationScopeStage.CHAPTER,
                    scope_key="chapter:old-sec-1",
                    input_sha256="1" * 64,
                    section_id="old-sec-1",
                    section_order=0,
                    section_title="Chapter One Old",
                    state=IllustrationScopeState.ACCEPTED,
                    summary={"moment_count": 0},
                ),
            ],
        )
        assert len(scopes) == 2

    # Reimport document with completely new section IDs.
    reimported = CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(title="Reimported"),
        source_language="en",
        target_language="ru",
        sections=(
            DocumentSection(
                id="new-sec-1",
                document_id="book-1",
                title="Brand New Section",
                order=0,
                source_resource_path="Text/new1.xhtml",
                blocks=(
                    DocumentBlock(
                        id="new-block-1",
                        section_id="new-sec-1",
                        type=BlockType.PARAGRAPH,
                        order=0,
                        text="Updated chapter text.",
                        source_reference=SourceReference("Text/new1.xhtml", "p[0]"),
                        support_status=SupportStatus.SUPPORTED,
                    ),
                ),
            ),
        ),
    )
    with unit_of_work.transaction() as tx:
        tx.documents.save(reimported)

    with unit_of_work.read_transaction() as tx:
        headers, total = tx.illustrations.get_chapter_header_page(
            "book-1", gen.id, offset=0, limit=10
        )
        assert total == 2
        assert [(h.section_id, h.order, h.title, h.state, h.moment_count) for h in headers] == [
            ("old-sec-1", 0, "Chapter One Old", "ready", 0),
            ("old-sec-2", 1, "Chapter Two Old", "ready", 2),
        ]


def test_history_ordered_newest_first_capped_at_10(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    _seed_book(unit_of_work)
    with unit_of_work.transaction() as tx:
        rev = tx.illustrations.save_settings(
            "book-1", expected_revision=0, settings=_enabled_settings()
        )
        for idx in range(1, 13):
            gen, created = tx.illustrations.create_or_get_generation(
                "book-1",
                settings_id=rev.id,
                request_key=f"req-{idx}",
                source_sha256="a" * 64,
                source_authority={"conditions_id": str(uuid4()), "conditions_revision": 1, "source_digest": "a" * 64, "manifest_sha256": None},
                requested_sections=("section-1",),
            )
            assert created is True
            assert gen.generation_number == idx
            if idx < 12:
                tx.illustrations.revoke_generation(
                    "book-1", gen.id, reason_code=IllustrationReasonCode.SETTINGS_CHANGED.value
                )

    with unit_of_work.read_transaction() as tx:
        history = tx.illustrations.list_history("book-1", limit=10)
        assert len(history) == 10
        assert [item.generation_number for item in history] == list(range(12, 2, -1))
        assert history[0].state is IllustrationGenerationState.WAITING_SOURCE
        assert all(item.state is IllustrationGenerationState.REVOKED for item in history[1:])


def test_cross_book_ids_rejected(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    _seed_book(unit_of_work, "book-1")
    _seed_book(unit_of_work, "book-2")
    rev1, gen1, _ = _seed_generation(unit_of_work, book_id="book-1")

    with unit_of_work.transaction() as tx:
        assert tx.illustrations.get_settings("book-2", rev1.id) is None
        assert tx.illustrations.get_generation("book-2", gen1.id) is None

        with pytest.raises(IllustrationNotFoundError) as gen_err:
            tx.illustrations.revoke_generation(
                "book-2", gen1.id, reason_code=IllustrationReasonCode.SETTINGS_CHANGED.value
            )
        assert gen_err.value.reason_code == IllustrationReasonCode.GENERATION_NOT_FOUND

        with pytest.raises(Exception) as settings_err:
            tx.illustrations.create_or_get_generation(
                "book-2",
                settings_id=rev1.id,
                request_key="req-foreign",
                source_sha256="a" * 64,
                source_authority={},
                requested_sections=(),
            )
        assert getattr(settings_err.value, "reason_code", None) == IllustrationReasonCode.INVALID_REQUEST


def test_task_supersede_clears_last_error() -> None:
    failed = ProcessingTask(
        id="task-1",
        book_id="book-1",
        task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
        scope_id="scope-1",
        status=TaskStatus.FAILED,
        attempts=1,
        last_error=TaskError(type=ErrorType.TRANSIENT, message="temporary error"),
        operation_id="gen-1:a1",
        policy_version="illustration-v1",
    )
    superseded = failed.supersede()
    assert superseded.status is TaskStatus.SUPERSEDED
    assert superseded.last_error is None
    assert superseded.supersede() == superseded


def test_labels_cover_all_task_types_and_statuses() -> None:
    for task_type in TaskType:
        assert task_type in TASK_TYPE_LABELS
        assert TASK_TYPE_LABELS[task_type].strip()
    for status in TaskStatus:
        assert status in TASK_STATUS_LABELS
        assert TASK_STATUS_LABELS[status].strip()


def test_metadata_matches_migration(engine: Engine) -> None:
    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        differences = compare_metadata(context, Base.metadata)
    assert differences == []
