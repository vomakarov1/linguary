"""Batch query and diagnostics policy tests for SQLAlchemy repositories (P3.T1)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from persistence_factories import (
    BOOK_ID,
    CREATED_AT,
    RUN_ID,
    make_analysis,
    make_block,
    make_book,
    make_concept,
    make_document,
    make_llm_call,
    make_run,
    make_task,
)
from sqlalchemy import Engine, event

import linguary.application.diagnostics as diagnostics_module
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.application.diagnostics import DiagnosticsPolicy
from linguary.domain import (
    BlockTreatment,
    ConceptType,
    DocumentSection,
    GlossaryCandidate,
    TermStatus,
    TermVariant,
    TreatmentPlan,
    TreatmentPlanItem,
)
from linguary.domain.book_conditions import BookConditions
from linguary.domain.recovery import CallAttempt, RecoveryRoot, RecoveryScope


def _count_statements(engine: Engine, prefix: str = "SELECT") -> tuple[list[str], object]:
    """Attach a listener recording SQL statements starting with `prefix`."""
    statements: list[str] = []
    upper_prefix = prefix.upper()

    def _listener(conn, cursor, statement, parameters, context, executemany) -> None:
        if statement.lstrip().upper().startswith(upper_prefix):
            statements.append(statement)

    event.listen(engine, "before_cursor_execute", _listener)
    return statements, _listener


def test_list_for_book_repositories_avoid_n_plus_one_queries(
    engine: Engine,
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """`list_for_book` on analyses and glossary executes a constant number of SELECTs."""
    sections = tuple(
        DocumentSection(
            id=f"section-{idx}",
            document_id=BOOK_ID,
            title=f"Chapter {idx}",
            order=idx - 1,
            source_resource_path=f"OEBPS/ch{idx}.xhtml",
            blocks=(
                make_block(block_id=f"b-{idx}-1", section_id=f"section-{idx}", order=0),
                make_block(block_id=f"b-{idx}-2", section_id=f"section-{idx}", order=1),
            ),
        )
        for idx in range(1, 5)
    )
    document = replace(make_document(), sections=sections)

    expected_analyses = tuple(
        replace(
            make_analysis(analysis_id=f"analysis-{idx}"),
            section_id=f"section-{idx}",
            treatment_plan=TreatmentPlan(
                section_id=f"section-{idx}",
                items=(
                    TreatmentPlanItem(
                        block_id=f"b-{idx}-1", treatment=BlockTreatment.TRANSLATE
                    ),
                    TreatmentPlanItem(
                        block_id=f"b-{idx}-2", treatment=BlockTreatment.GLOSSARY_ONLY
                    ),
                ),
            ),
            glossary_candidates=(
                GlossaryCandidate(
                    id=f"cand-{idx}-1",
                    analysis_id=f"analysis-{idx}",
                    source_text=f"Term {idx} A",
                    type=ConceptType.PERSON,
                    suggested_translation=f"Термин {idx} А",
                    confidence=0.9,
                    model_flagged_review=False,
                    requires_review=False,
                ),
                GlossaryCandidate(
                    id=f"cand-{idx}-2",
                    analysis_id=f"analysis-{idx}",
                    source_text=f"Term {idx} B",
                    type=ConceptType.PERSON,
                    suggested_translation=f"Термин {idx} Б",
                    confidence=0.8,
                    model_flagged_review=False,
                    requires_review=False,
                ),
            ),
        )
        for idx in range(1, 5)
    )

    expected_concepts = tuple(
        replace(
            make_concept(concept_id=f"concept-{idx}"),
            canonical_source_text=f"Concept {idx}",
            variants=(
                TermVariant(
                    concept_id=f"concept-{idx}",
                    language="ru",
                    text=f"Понятие {idx}",
                    status=TermStatus.LOCKED,
                ),
                TermVariant(
                    concept_id=f"concept-{idx}",
                    language="en",
                    text=f"Concept {idx}",
                    status=TermStatus.APPROVED,
                ),
            ),
        )
        for idx in range(1, 5)
    )

    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
        uow.documents.save(document)
        for analysis in expected_analyses:
            uow.analyses.save(BOOK_ID, analysis)
        for concept in expected_concepts:
            uow.glossary.save(concept)

    selects, listener = _count_statements(engine, "SELECT")
    try:
        with unit_of_work.transaction() as uow:
            selects.clear()
            loaded_analyses = uow.analyses.list_for_book(BOOK_ID)
            analysis_select_count = len(selects)

            selects.clear()
            loaded_concepts = uow.glossary.list_for_book(BOOK_ID)
            glossary_select_count = len(selects)
    finally:
        event.remove(engine, "before_cursor_execute", listener)

    assert loaded_analyses == expected_analyses
    assert analysis_select_count == 3
    assert loaded_concepts == expected_concepts
    assert glossary_select_count == 2


def test_list_task_and_root_attempts_fetch_rows_in_single_query(
    engine: Engine,
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """`list_task_attempts` and `list_root_attempts` fetch `LLMAttemptRow` in one SELECT."""
    conditions = BookConditions(
        id=str(uuid4()),
        book_id=BOOK_ID,
        revision=1,
        payload={"schema_version": 1},
        created_at=CREATED_AT,
    )
    root = RecoveryRoot(
        id=str(uuid4()),
        book_id=BOOK_ID,
        conditions_id=conditions.id,
        root_key=str(uuid4()),
        manifest={"schema_version": 1},
    )
    scope = RecoveryScope(
        id=str(uuid4()),
        root_id=root.id,
        ordered_blocks=("block-1",),
    )
    attempts = tuple(
        CallAttempt(
            id=str(uuid4()),
            book_id=BOOK_ID,
            conditions_id=conditions.id,
            task_id="task-1",
            run_id=RUN_ID,
            root_id=root.id,
            scope_id=scope.id,
            generation=0,
            ordinal=idx,
            state="failed",
        )
        for idx in range(4)
    )

    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
        uow.documents.save(make_document())
        uow.runs.add(make_run())
        uow.tasks.add(make_task(task_id="task-1"))
        uow.work.add_conditions(conditions)
        uow.work.add_root(root)
        uow.work.add_scope(scope)
        for attempt in attempts:
            uow.work.add_attempt(attempt)

    selects, listener = _count_statements(engine, "SELECT")
    try:
        with unit_of_work.transaction() as uow:
            selects.clear()
            task_attempts = uow.work.list_task_attempts("task-1")
            task_select_count = len(selects)

        with unit_of_work.transaction() as uow:
            selects.clear()
            root_attempts = uow.work.list_root_attempts(root.id)
            root_select_count = len(selects)
    finally:
        event.remove(engine, "before_cursor_execute", listener)

    assert task_attempts == attempts
    assert task_select_count == 1
    assert root_attempts == attempts
    assert root_select_count == 1


def test_enforce_diagnostics_policy_skips_rebounding_already_bounded_rows_while_enforcing_book_quota(
    engine: Engine,
    unit_of_work: SqlAlchemyUnitOfWork,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Already bounded rows skip `bounded_body` and redundant UPDATEs but still count toward book quota."""
    now = datetime(2026, 10, 25, 12, tzinfo=UTC)
    policy = DiagnosticsPolicy(max_body_bytes=100, max_book_bytes=15, retention_seconds=3600)

    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
        uow.llm_calls.add(
            replace(make_llm_call(call_id="call-1"), started_at=now),
            response_text="012345",
        )
        uow.llm_calls.add(
            replace(make_llm_call(call_id="call-2"), started_at=now + timedelta(seconds=1)),
            response_text="abcdef",
        )
        uow.llm_calls.enforce_diagnostics_policy(policy, now, book_id=BOOK_ID)

    bounded_calls: list[str] = []
    orig_bounded_body = diagnostics_module.bounded_body

    def tracked_bounded_body(body, *, max_bytes=64 * 1024, secrets=()):
        bounded_calls.append(body)
        return orig_bounded_body(body, max_bytes=max_bytes, secrets=secrets)

    monkeypatch.setattr(diagnostics_module, "bounded_body", tracked_bounded_body)

    updates, listener = _count_statements(engine, "UPDATE")
    try:
        with unit_of_work.transaction() as uow:
            updates.clear()
            uow.llm_calls.enforce_diagnostics_policy(policy, now, book_id=BOOK_ID)
            noop_update_count = len(updates)
    finally:
        event.remove(engine, "before_cursor_execute", listener)

    assert bounded_calls == []
    assert noop_update_count == 0

    with unit_of_work.transaction() as uow:
        uow.llm_calls.add(
            replace(make_llm_call(call_id="call-3"), started_at=now + timedelta(seconds=2)),
            response_text="ghijkl",
        )
        uow.llm_calls.enforce_diagnostics_policy(policy, now, book_id=BOOK_ID)

    assert bounded_calls == ["ghijkl"]

    with unit_of_work.transaction() as uow:
        rec1 = uow.llm_calls.diagnostic_record(call_id="call-1")
        rec2 = uow.llm_calls.diagnostic_record(call_id="call-2")
        rec3 = uow.llm_calls.diagnostic_record(call_id="call-3")

    assert rec1 is not None and rec1["body"] is None
    assert rec1["metadata"]["body_meta"]["body_status"] == "deleted"
    assert rec2 is not None and rec2["body"] == "abcdef"
    assert rec3 is not None and rec3["body"] == "ghijkl"
