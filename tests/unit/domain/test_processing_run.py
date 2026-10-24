"""Book processing run: snapshots and uniqueness of active run (ADR § 3)."""

from __future__ import annotations

import pytest

from linguary.domain import (
    ActiveRunConflictError,
    ConceptType,
    GlossarySnapshot,
    GlossarySnapshotEntry,
    LLMSettingsSnapshot,
    ProcessingRun,
    ProcessingRunError,
    ProcessingRunStatus,
    TermStatus,
    active_run,
)

SETTINGS = LLMSettingsSnapshot(provider="openai", model="gpt-test")


def make_run(
    run_id: str, status: ProcessingRunStatus = ProcessingRunStatus.ACTIVE
) -> ProcessingRun:
    """A book processing run in the given status."""
    return ProcessingRun(
        id=run_id,
        book_id="book-1",
        status=status,
        glossary_snapshot=GlossarySnapshot(),
        llm_settings_snapshot=SETTINGS,
    )


def test_single_active_run_is_returned() -> None:
    """There is strictly one active run for a book."""
    runs = [make_run("run-1", ProcessingRunStatus.CANCELLED), make_run("run-2")]
    current = active_run(runs)
    assert current is not None
    assert current.id == "run-2"


def test_book_without_active_run() -> None:
    """Before processing starts, the book has no active run."""
    assert active_run([make_run("run-1", ProcessingRunStatus.COMPLETED)]) is None


def test_two_active_runs_are_rejected() -> None:
    """Concurrent result sets are not accumulated."""
    with pytest.raises(ActiveRunConflictError):
        active_run([make_run("run-1"), make_run("run-2")])


def test_glossary_snapshot_selects_mandatory_entries() -> None:
    """User decisions are passed to the model as mandatory; model suggestions are not.

    An approved or corrected decision ('approved') is as much a user decision as a
    locked one: otherwise chapter instructions with the model's previous variant would override it (D-1).
    """
    snapshot = GlossarySnapshot(
        entries=(
            GlossarySnapshotEntry(
                source_text="Winston Smith",
                translation="Уинстон Смит",
                type=ConceptType.PERSON,
                status=TermStatus.LOCKED,
            ),
            GlossarySnapshotEntry(
                source_text="doublethink",
                translation="двоемыслие",
                type=ConceptType.TERM,
                status=TermStatus.APPROVED,
            ),
            GlossarySnapshotEntry(
                source_text="Newspeak",
                translation="новояз",
                type=ConceptType.TERM,
                status=TermStatus.SUGGESTED,
            ),
        )
    )
    assert tuple(entry.source_text for entry in snapshot.mandatory_entries) == (
        "Winston Smith",
        "doublethink",
    )


def test_glossary_snapshot_rejects_duplicate_source_text() -> None:
    """A glossary snapshot does not contain duplicate decisions for the same source text."""
    entry = GlossarySnapshotEntry(
        source_text="Winston Smith",
        translation="Уинстон Смит",
        type=ConceptType.PERSON,
        status=TermStatus.LOCKED,
    )
    with pytest.raises(ProcessingRunError):
        GlossarySnapshot(entries=(entry, entry))


def test_llm_settings_snapshot_requires_provider_and_model() -> None:
    """A settings snapshot records which model produced the result."""
    with pytest.raises(ProcessingRunError):
        LLMSettingsSnapshot(provider="", model="gpt-test")
