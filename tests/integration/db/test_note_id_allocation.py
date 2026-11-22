"""Atomic per-run, per-chapter language-note id allocation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
import threading

import pytest
from persistence_factories import (
    BLOCK_ID,
    BOOK_ID,
    RUN_ID,
    SECOND_BLOCK_ID,
    SECTION_ID,
    make_book,
    make_document,
    make_run,
    make_translated_result,
)
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from linguary.adapters.db.models import CorrectionCandidateRow, NoteNumberCounterRow
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.domain.book_conditions import BookConditions
from linguary.domain.learning import LanguageNote
from linguary.domain.processing import BlockProcessingResult, ProcessingRunStatus

_OTHER_RUN_ID = "run-2"
_CONDITIONS_ID = "11111111-1111-4111-8111-111111111111"
_REVISION_ID = "22222222-2222-4222-8222-222222222222"
_OTHER_REVISION_ID = "33333333-3333-4333-8333-333333333333"
_CREATED_AT = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def test_allocate_note_ids_monotonic_unique_across_concurrent_transactions(
    unit_of_work: SqlAlchemyUnitOfWork, engine: Engine
) -> None:
    """Parallel transactions on one chapter issue contiguous ids and no duplicates."""
    _prepare(unit_of_work)
    workers = 6
    per_worker = 4
    barrier = threading.Barrier(workers)

    def allocate() -> tuple[str, ...]:
        barrier.wait()
        with unit_of_work.transaction() as uow:
            return uow.results.allocate_note_ids(RUN_ID, 1, per_worker)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(allocate) for _ in range(workers)]
        batches = [future.result() for future in futures]

    issued = [note_id for batch in batches for note_id in batch]
    expected = [f"ln-ch01-{ordinal:03d}" for ordinal in range(1, workers * per_worker + 1)]
    assert sorted(issued) == expected
    assert len(set(issued)) == len(issued)
    for batch in batches:
        ordinals = [int(note_id.rsplit("-", 1)[1]) for note_id in batch]
        assert ordinals == list(range(ordinals[0], ordinals[0] + per_worker))
        assert list(batch) == [f"ln-ch01-{ordinal:03d}" for ordinal in ordinals]
    assert _counter_rows(engine) == [(RUN_ID, 1, workers * per_worker)]


def test_allocate_note_ids_lazy_seeds_from_existing_notes_and_candidates(
    unit_of_work: SqlAlchemyUnitOfWork, engine: Engine
) -> None:
    """A missing counter starts after this run and chapter, then stays authoritative."""
    _prepare(unit_of_work, second_run=True)
    _seed_issued_ids(unit_of_work, engine)
    assert _counter_rows(engine) == []

    with unit_of_work.transaction() as uow:
        assert uow.results.allocate_note_ids(RUN_ID, 1, 2) == (
            "ln-ch01-013",
            "ln-ch01-014",
        )
        uow.results.replace_results(
            RUN_ID, [_result_with_notes(SECOND_BLOCK_ID, "ln-ch01-040")]
        )
        assert uow.results.allocate_note_ids(RUN_ID, 1, 1) == ("ln-ch01-015",)
        assert uow.results.allocate_note_ids(RUN_ID, 2, 1) == ("ln-ch02-005",)
        assert uow.results.allocate_note_ids(RUN_ID, 3, 1) == ("ln-ch03-007",)
        assert uow.results.allocate_note_ids(_OTHER_RUN_ID, 1, 1) == ("ln-ch01-100",)

    assert _counter_rows(engine) == [
        (RUN_ID, 1, 15),
        (RUN_ID, 2, 5),
        (RUN_ID, 3, 7),
        (_OTHER_RUN_ID, 1, 100),
    ]


def test_allocate_note_ids_zero_count_and_cascade_on_run_delete(
    unit_of_work: SqlAlchemyUnitOfWork, engine: Engine
) -> None:
    """Zero does not touch the counter; deleting the run removes the row."""
    _prepare(unit_of_work)
    with unit_of_work.transaction() as uow:
        assert uow.results.allocate_note_ids(RUN_ID, 1, 0) == ()
        assert uow.results.allocate_note_ids(RUN_ID, 0, 0) == ()
        with pytest.raises(ValueError):
            uow.results.allocate_note_ids(RUN_ID, 1, -1)
        with pytest.raises(ValueError):
            uow.results.allocate_note_ids(RUN_ID, 0, 1)
        with pytest.raises(ValueError):
            uow.results.allocate_note_ids(RUN_ID, -5, 2)
    assert _counter_rows(engine) == []

    with unit_of_work.transaction() as uow:
        assert uow.results.allocate_note_ids(RUN_ID, 1, 2) == (
            "ln-ch01-001",
            "ln-ch01-002",
        )
        assert uow.results.allocate_note_ids(RUN_ID, 1, 0) == ()
        assert uow.results.allocate_note_ids(RUN_ID, 1, 1) == ("ln-ch01-003",)
    assert _counter_rows(engine) == [(RUN_ID, 1, 3)]

    with unit_of_work.transaction() as uow:
        uow.runs.delete(RUN_ID)
        assert uow.runs.get(RUN_ID) is None
        assert uow.books.get(BOOK_ID) is not None
    assert _counter_rows(engine) == []


def _prepare(unit_of_work: SqlAlchemyUnitOfWork, *, second_run: bool = False) -> None:
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
        uow.documents.save(make_document())
        uow.runs.add(make_run())
        if second_run:
            uow.runs.add(make_run(run_id=_OTHER_RUN_ID, status=ProcessingRunStatus.COMPLETED))
        uow.work.add_conditions(
            BookConditions(
                _CONDITIONS_ID,
                BOOK_ID,
                1,
                {"schema_version": 1},
                _CREATED_AT,
            )
        )


def _seed_issued_ids(unit_of_work: SqlAlchemyUnitOfWork, engine: Engine) -> None:
    """Notes, candidate payloads, and revisions. Only whole ids for this run count."""
    with unit_of_work.transaction() as uow:
        uow.results.replace_results(
            RUN_ID,
            [
                _result_with_notes(
                    BLOCK_ID,
                    "ln-ch01-001",
                    "ln-ch01-007",
                    "ln-ch02-002",
                    "ln-ch03-006",
                )
            ],
        )
        uow.results.replace_results(
            _OTHER_RUN_ID, [_result_with_notes(BLOCK_ID, "ln-ch01-077")]
        )
        uow.work.archive_artifacts(
            [
                {
                    "id": _REVISION_ID,
                    "book_id": BOOK_ID,
                    "run_id": RUN_ID,
                    "kind": "block_result",
                    "scope_id": BLOCK_ID,
                    "body": {
                        "schema_version": 1,
                        "result": {
                            "language_notes": [
                                {"id": "ln-ch01-012"},
                                {"id": "custom-note"},
                                {"id": "ln-ch01-012-extra"},
                            ],
                            "label": "see ln-ch01-040",
                        },
                    },
                },
                {
                    "id": _OTHER_REVISION_ID,
                    "book_id": BOOK_ID,
                    "run_id": _OTHER_RUN_ID,
                    "kind": "block_result",
                    "scope_id": BLOCK_ID,
                    "body": {
                        "schema_version": 1,
                        "result": {"language_notes": [{"id": "ln-ch01-099"}]},
                    },
                },
            ]
        )
    with Session(engine) as session, session.begin():
        session.add(
            CorrectionCandidateRow(
                candidate_id="candidate-1",
                book_id=BOOK_ID,
                run_id=RUN_ID,
                conditions_id=_CONDITIONS_ID,
                section_id=SECTION_ID,
                block_id=BLOCK_ID,
                input_sha256="a" * 64,
                input_generation=1,
                proposal_sha256="b" * 64,
                payload_json={
                    "result": {"language_notes": [{"id": "ln-ch01-009"}]},
                    "current_result": {"language_notes": [{"id": "ln-ch02-004"}]},
                    "label": "see ln-ch01-099",
                    "ignored": "custom-note",
                },
                group_id="group-1",
                proposal_attempt_id="attempt-1",
                state="proposed",
            )
        )
        session.add(
            CorrectionCandidateRow(
                candidate_id="candidate-2",
                book_id=BOOK_ID,
                run_id=_OTHER_RUN_ID,
                conditions_id=_CONDITIONS_ID,
                section_id=SECTION_ID,
                block_id=BLOCK_ID,
                input_sha256="c" * 64,
                input_generation=1,
                proposal_sha256="d" * 64,
                payload_json={"result": {"language_notes": [{"id": "ln-ch01-080"}]}},
                group_id="group-2",
                proposal_attempt_id="attempt-2",
                state="proposed",
            )
        )


def _result_with_notes(block_id: str, *note_ids: str) -> BlockProcessingResult:
    notes = tuple(
        LanguageNote(
            id=note_id,
            block_id=block_id,
            original_excerpt="bright cold day",
            note=f"Note {note_id}.",
        )
        for note_id in note_ids
    )
    return replace(make_translated_result(block_id=block_id), language_notes=notes)


def _counter_rows(engine: Engine) -> list[tuple[str, int, int]]:
    with Session(engine) as session:
        return list(
            session.execute(
                select(
                    NoteNumberCounterRow.run_id,
                    NoteNumberCounterRow.chapter_number,
                    NoteNumberCounterRow.last_ordinal,
                ).order_by(NoteNumberCounterRow.run_id, NoteNumberCounterRow.chapter_number)
            ).all()
        )
