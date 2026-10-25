"""Application reconstruction preserves reservations, policies and historical choices."""
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.llm import FakeLLMClient
from linguary.application.quality_budget import QualityBudget
from linguary.composition import build_application, drain_tasks
from linguary.domain import LLMCallLog, LLMCallStage
from test_book_conditions import store_imported
from test_quality_review import make_review_app, command


async def test_startup_reconciles_attempts_not_cost_zero(tmp_path):
    app, fake, _, _ = await make_review_app(tmp_path)
    created = await app.quality_review.start(command(app))
    budget = QualityBudget(app.unit_of_work)
    attempt = budget.reserve(created['review_id'], 800, price={'schema_version': 1}, expected_revision=0)
    budget.mark_sent(attempt.id)
    settings = app.settings
    await app.aclose()
    new_fake = FakeLLMClient()
    restarted = build_application(settings=settings, llm_client=new_fake, quality_routes={})
    try:
        await restarted.runner.recover_interrupted()
        await restarted.runner.recover_interrupted()
        await drain_tasks(restarted)
        with restarted.unit_of_work.transaction() as tx:
            review = tx.work.get_review(created['review_id'])
            assert review.state == 'unknown' and review.reserved == 800 and review.spent == 0
            assert tx.work.get_attempt(attempt.id).state == 'unknown'
            assert tx.work.get_reservation(attempt.id).state == 'unknown'
        assert new_fake.requests == []
    finally:
        await restarted.aclose()


@pytest.mark.parametrize('profile', ['legacy-v1', 'content-plan-v2', 'content-plan-v3'])
async def test_restart_rebuilds_supported_policies(tmp_path, monkeypatch, profile):
    settings = AppSettings(llm=LLMSettings(model='synthetic', api_key='synthetic-key'),
        storage=StorageSettings(data_root=str(tmp_path / 'data')))
    app = build_application(settings=settings, llm_client=FakeLLMClient(), content_plan_profile=profile)
    store_imported(app)
    conditions_id = app.book_conditions.freeze_before_analysis('book-1', 0)
    before = app.book_conditions.require('book-1')
    await app.aclose()
    monkeypatch.setenv('LINGUARY_CONTENT_PLAN_PROFILE', 'content-plan-v3')
    new_settings = replace(settings, recovery=replace(settings.recovery, max_calls=100))
    restarted = build_application(settings=new_settings, llm_client=FakeLLMClient())
    try:
        async with restarted.book_runtime.lease(conditions_id, 'processing') as generation:
            processor = generation.processing._process_chapter
            assert processor._context_builder._content_plan_profile == profile
            assert generation.settings.recovery == settings.recovery
        assert restarted.book_conditions.require('book-1') == before
    finally:
        await restarted.aclose()


def test_legacy_profile_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv('LINGUARY_CONTENT_PLAN_PROFILE', 'content-plan-v3')
    app = build_application(settings=AppSettings(storage=StorageSettings(data_root=str(tmp_path / 'data'))),
        llm_client=FakeLLMClient(), content_plan_profile='legacy-v1')
    try:
        store_imported(app)
        app.book_conditions.freeze_before_analysis('book-1', 0)
        assert app.book_conditions.require('book-1').payload['profile'] == 'legacy-v1'
    finally:
        app.dispose()


async def test_review_off_after_update(tmp_path, monkeypatch):
    monkeypatch.delenv('LINGUARY_QUALITY_ROUTES', raising=False)
    app, _, _, _ = await make_review_app(tmp_path)
    created = await app.quality_review.start(command(app))
    settings = app.settings
    await app.aclose()
    fake = FakeLLMClient()
    restarted = build_application(settings=settings, llm_client=fake)
    try:
        await restarted.runner.recover_interrupted()
        await drain_tasks(restarted)
        projection = restarted.quality_review.get('book-1', created['review_id'])
        assert projection['reason'] == 'price_unavailable'
        assert not restarted.quality_review.routes and fake.requests == []
    finally:
        await restarted.aclose()


async def test_retention_maintenance_without_calls(tmp_path):
    fake = FakeLLMClient()
    app = build_application(settings=AppSettings(storage=StorageSettings(data_root=str(tmp_path / 'data'))), llm_client=fake)
    try:
        store_imported(app)
        now = datetime.now(UTC)
        with app.unit_of_work.transaction() as tx:
            tx.llm_calls.add(LLMCallLog('old', 'book-1', LLMCallStage.CHAPTER_ANALYSIS, 'fake', 'fake', now,
                1, 1, True), response_text='synthetic private body')
        app.diagnostics.clock = lambda: now + timedelta(days=8)
        assert await app.runner.run_next() is None
        with app.unit_of_work.transaction() as tx:
            assert tx.llm_calls.response_text_for('old') is None
            assert tx.llm_calls.list_for_book('book-1')
        assert fake.requests == []
    finally:
        await app.aclose()


def test_application_opens_without_llm(tmp_path):
    from fastapi.testclient import TestClient
    from linguary.web import create_app
    app = build_application(settings=AppSettings(storage=StorageSettings(data_root=str(tmp_path / 'data'))))
    try:
        with TestClient(create_app(application=app, run_background_worker=False), base_url=app.settings.server.base_url) as client:
            assert client.get('/books').status_code == 200
        assert not app.quality_review.routes
    finally:
        app.dispose()
