"""Запуск обработки книги: снимки и единственность активного run (ADR § 3)."""

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
    """Запуск обработки книги в заданном состоянии."""
    return ProcessingRun(
        id=run_id,
        book_id="book-1",
        status=status,
        glossary_snapshot=GlossarySnapshot(),
        llm_settings_snapshot=SETTINGS,
    )


def test_single_active_run_is_returned() -> None:
    """Активный запуск книги ровно один."""
    runs = [make_run("run-1", ProcessingRunStatus.CANCELLED), make_run("run-2")]
    current = active_run(runs)
    assert current is not None
    assert current.id == "run-2"


def test_book_without_active_run() -> None:
    """До запуска обработки активного run у книги нет."""
    assert active_run([make_run("run-1", ProcessingRunStatus.COMPLETED)]) is None


def test_two_active_runs_are_rejected() -> None:
    """Параллельные наборы результатов не накапливаются."""
    with pytest.raises(ActiveRunConflictError):
        active_run([make_run("run-1"), make_run("run-2")])


def test_glossary_snapshot_selects_mandatory_entries() -> None:
    """Модели как обязательные передаются решения пользователя, подсказки модели — нет.

    Принятое и исправленное решение (`approved`) — такое же решение пользователя, как
    зафиксированное: иначе указание главы с прежним вариантом модели перекрывало его (Д-1).
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
    """Снимок глоссария не содержит двух решений для одного исходного текста."""
    entry = GlossarySnapshotEntry(
        source_text="Winston Smith",
        translation="Уинстон Смит",
        type=ConceptType.PERSON,
        status=TermStatus.LOCKED,
    )
    with pytest.raises(ProcessingRunError):
        GlossarySnapshot(entries=(entry, entry))


def test_llm_settings_snapshot_requires_provider_and_model() -> None:
    """Снимок настроек фиксирует, какой моделью получен результат."""
    with pytest.raises(ProcessingRunError):
        LLMSettingsSnapshot(provider="", model="gpt-test")
