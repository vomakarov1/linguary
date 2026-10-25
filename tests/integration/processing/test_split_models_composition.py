"""Tests for separate LLM model configuration for analysis and translation."""

from __future__ import annotations

from pathlib import Path

import pytest

from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, StorageSettings
from linguary.application.ports.llm import LLMConfigurationError
from linguary.composition import build_application, drain_tasks
from linguary.domain import BookStatus, LLMCallStage
from processing_scenario import BOOK_ID, store_processing_book


class FailingClient:
    """Client rejecting requests with a configuration error to record the call in logs."""

    async def generate(self, request):
        raise LLMConfigurationError("Тестовая ошибка")


def test_llm_snapshot_uses_effective_processing_model(tmp_path: Path) -> None:
    """Settings snapshot when starting book processing records effective_processing_model."""
    settings = AppSettings(
        llm=LLMSettings(
            model="base-model",
            analysis_model="analysis-model",
            processing_model="processing-model",
            api_key="synthetic-key",
        ),
        checks=CheckSettings(),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
    )
    app = build_application(settings=settings, llm_client=FailingClient())
    try:
        snapshot = app.start_processing()._llm_settings_snapshot
        assert snapshot.model == "processing-model"
    finally:
        app.dispose()


def test_llm_snapshot_falls_back_to_general_model(tmp_path: Path) -> None:
    """If processing_model is not set, snapshot falls back to general model."""
    settings = AppSettings(
        llm=LLMSettings(
            model="base-model",
            api_key="synthetic-key",
        ),
        checks=CheckSettings(),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
    )
    app = build_application(settings=settings, llm_client=FailingClient())
    try:
        snapshot = app.start_processing()._llm_settings_snapshot
        assert snapshot.model == "base-model"
    finally:
        app.dispose()


async def test_analysis_logs_analysis_model(tmp_path: Path) -> None:
    """Analysis logs calls under analysis_model."""
    settings = AppSettings(
        llm=LLMSettings(
            model="base-model",
            analysis_model="deep-analysis-model",
            processing_model="fast-processing-model",
            api_key="synthetic-key",
        ),
        checks=CheckSettings(),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
    )
    app = build_application(settings=settings, llm_client=FailingClient())
    try:
        store_processing_book(app.unit_of_work, status=BookStatus.IMPORTED, analysed_sections=())
        choose_fixture_conditions(app, tmp_path)
        await app.analysis.start(BOOK_ID)
        await drain_tasks(app)

        with app.unit_of_work.transaction() as transaction:
            analysis_logs = transaction.llm_calls.list_for_book(BOOK_ID)
        assert len(analysis_logs) >= 1
        assert analysis_logs[0].model == "deep-analysis-model"
    finally:
        await app.aclose()


async def test_processing_logs_processing_model(tmp_path: Path) -> None:
    """Processing (translation) logs calls under processing_model."""
    settings = AppSettings(
        llm=LLMSettings(
            model="base-model",
            analysis_model="deep-analysis-model",
            processing_model="fast-processing-model",
            api_key="synthetic-key",
        ),
        checks=CheckSettings(),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
    )
    app = build_application(settings=settings, llm_client=FailingClient())
    try:
        store_processing_book(app.unit_of_work, status=BookStatus.PROCESSING)
        choose_fixture_conditions(app, tmp_path)
        await app.processing.start(BOOK_ID)
        await drain_tasks(app)

        with app.unit_of_work.transaction() as transaction:
            logs = transaction.llm_calls.list_for_book(BOOK_ID)
        processing_logs = [log for log in logs if log.stage == LLMCallStage.TRANSLATION_UNIT]
        assert len(processing_logs) >= 1
        assert processing_logs[0].model == "fast-processing-model"
    finally:
        await app.aclose()


def choose_fixture_conditions(application, tmp_path):
    from dataclasses import replace
    from linguary.application.use_cases.reset_processing import ChooseBookConditions
    source = tmp_path / 'isolated-source.epub'
    source.write_bytes(b'synthetic isolated fixture source')
    with application.unit_of_work.transaction() as transaction:
        book = transaction.books.get(BOOK_ID)
        transaction.books.save(replace(book, source_epub_path=str(source)))
    ChooseBookConditions(application.book_conditions, application.book_runtime)(BOOK_ID,
        action='continue_legacy', expected_revision=0,
        settings_revision=application.runtime_settings.revision, confirmed=True)
