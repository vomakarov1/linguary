"""Pinned name strategy through real composition and persistence."""

import pytest

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.config.settings import ReadingSettings
from linguary.adapters.config.system_settings import SystemSettingsUpdate
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application.book_conditions import ConditionsError
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.application.use_cases.reset_processing import ChooseBookConditions, ResetProcessingCommand
from linguary.composition import build_application, drain_tasks
from linguary.domain import ConceptType, TermConcept, TermStatus, TermVariant
from processing_scenario import block_reply, translation_reply

from test_book_conditions import store_imported


def analysis_reply(*, treatment="READ_AS_IS", assessment=True):
    candidate = {"source_text": "Alice", "type": "person",
        "suggested_translation": "Элис", "confidence": 0.9,
        "model_flagged_review": False, "note": None}
    if assessment:
        candidate["name_translation_assessment"] = {"applied_method": "sound",
            "eligible_without_review": True, "review_reasons": [],
            "rationale": "Однозначное имя."}
    return FakeReply.of_payload({
        "treatment_plan": [{"source_block_id": "b1", "treatment": treatment}],
        "glossary_candidates": [candidate],
        "translation_instructions": "Сохраняй имя.", "chapter_summary": "Alice appeared.",
        "before_you_read": [],
    })


@pytest.mark.parametrize("strategy", ["sound", "meaning"])
async def test_full_book_uses_pinned_policy_after_global_save_restart(tmp_path, strategy):
    fake = FakeLLMClient.always(analysis_reply())
    initial_settings = AppSettings(
        llm=LLMSettings(model="pinned-model", api_key="synthetic-secret", proxy_mode="direct"),
        reading=ReadingSettings(name_translation_strategy=strategy),
        storage=StorageSettings(data_root=str(tmp_path / "data")))
    app = build_application(settings=initial_settings, llm_client=fake)
    try:
        store_imported(app, (("Alice Alice Alice Alice Alice appeared.",),))
        await app.analysis.start("book-1")
        await app.system_settings.save(SystemSettingsUpdate(
            provider="openai_compatible", model="pinned-model",
            base_url="https://api.openai.com/v1", requests_per_minute=None,
            concurrency=1, name_translation_strategy="meaning" if strategy == "sound" else "sound",
        ), expected_revision=0)
        await app.aclose()
        app = build_application(settings=initial_settings, llm_client=fake)
        completed = await drain_tasks(app)
        assert all(task.last_error is None for task in completed)
        conditions = app.book_conditions.require("book-1")
        with app.unit_of_work.transaction() as transaction:
            candidate = transaction.analyses.list_for_book("book-1")[0].glossary_candidates[0]
        assert candidate.name_translation_evidence is not None
        assert candidate.name_translation_evidence.conditions_id == conditions.id
        assert candidate.name_translation_evidence.strategy.value == strategy
        assert fake.last_request.name_translation_policy_version == 1
        assert not app.approve_glossary.review("book-1").items[0].requires_review
        app.start_processing()(StartProcessingCommand("book-1"))
    finally:
        await app.aclose()


async def test_first_request_after_reanalysis_contains_saved_decisions(tmp_path):
    fake = FakeLLMClient(replies=(analysis_reply(), analysis_reply(treatment="TRANSLATE"),
        translation_reply((block_reply("b1", translated_text="Алиса появилась."),)),
        translation_reply((block_reply("b1", translated_text="Алиса снова появилась."),))))
    app = build_application(settings=AppSettings(
        llm=LLMSettings(model="pinned-model", api_key="synthetic-secret", proxy_mode="direct"),
        storage=StorageSettings(data_root=str(tmp_path / "data"))), llm_client=fake)
    try:
        document = store_imported(app, (("Alice appeared.",),))
        await app.analysis.start("book-1")
        await drain_tasks(app)
        with app.unit_of_work.transaction() as transaction:
            transaction.glossary.save(TermConcept(id="saved-1", book_id="book-1",
                type=ConceptType.PERSON, canonical_source_text="Alice", status=TermStatus.APPROVED,
                variants=(TermVariant(concept_id="saved-1", language="ru",
                    text="Алиса", status=TermStatus.APPROVED),)))
            transaction.glossary.save(TermConcept(id="suggested-1", book_id="book-1",
                type=ConceptType.PERSON, canonical_source_text="Bob", status=TermStatus.SUGGESTED,
                variants=(TermVariant(concept_id="suggested-1", language="ru",
                    text="Боб", status=TermStatus.SUGGESTED),)))
        ChooseBookConditions(app.book_conditions, app.book_runtime)(
            "book-1", action="reanalyze", expected_revision=1,
            settings_revision=0, confirmed=True)
        await app.analysis.start("book-1")
        await drain_tasks(app)
        assert "Алиса" in fake.requests[1].input_text
        assert "Принятые решения пользователя" in fake.requests[1].input_text
        mandatory, optional = fake.requests[1].input_text.split("Необязательные подсказки", 1)
        assert "Боб" not in mandatory
        assert "Боб" in optional
        run = app.start_processing()(StartProcessingCommand("book-1"))
        assert run.glossary_snapshot.entries[0].translation == "Алиса"
        await app.processing.start("book-1")
        processed = await drain_tasks(app)
        assert all(task.last_error is None for task in processed)
        assert "Алиса" in fake.requests[2].input_text
        await app.regeneration.start("book-1", document.ordered_sections[0].ordered_blocks[0].id)
        regenerated = await drain_tasks(app)
        assert all(task.last_error is None for task in regenerated)
        assert "Алиса" in fake.requests[3].input_text
        conditions = app.book_conditions.require("book-1")
        app.reset_processing(ResetProcessingCommand("book-1"))
        assert app.book_conditions.require("book-1").id == conditions.id
        assert app.approve_glossary.review("book-1").item_for("Alice").decision.status is TermStatus.APPROVED
    finally:
        await app.aclose()


async def test_legacy_conditions_resume_with_old_schema(tmp_path):
    fake = FakeLLMClient.always(analysis_reply(assessment=False))
    app = build_application(settings=AppSettings(
        llm=LLMSettings(model="pinned-model", api_key="synthetic-secret", proxy_mode="direct"),
        storage=StorageSettings(data_root=str(tmp_path / "data"))), llm_client=fake)
    try:
        store_imported(app, (("Alice Alice Alice Alice Alice appeared.",),))
        ChooseBookConditions(app.book_conditions, app.book_runtime)("book-1",
            action="continue_legacy", expected_revision=0, settings_revision=0, confirmed=True)
        await app.analysis.start("book-1")
        completed = await drain_tasks(app)
        assert all(task.last_error is None for task in completed)
        assert fake.last_request.name_translation_policy_version is None
        with app.unit_of_work.transaction() as transaction:
            candidate = transaction.analyses.list_for_book("book-1")[0].glossary_candidates[0]
        assert candidate.name_translation_evidence is None
        assert app.approve_glossary.review("book-1").items[0].requires_review
    finally:
        await app.aclose()


async def test_reanalysis_replaces_policy_and_rejects_old_task(tmp_path):
    app = build_application(settings=AppSettings(
        llm=LLMSettings(model="pinned-model", api_key="synthetic-secret", proxy_mode="direct"),
        storage=StorageSettings(data_root=str(tmp_path / "data"))),
        llm_client=FakeLLMClient.always(analysis_reply()))
    try:
        store_imported(app)
        old_task = await app.analysis.start("book-1")
        old_conditions = app.book_conditions.require("book-1")
        await app.system_settings.save(SystemSettingsUpdate(
            provider="openai_compatible", model="pinned-model",
            base_url="https://api.openai.com/v1", requests_per_minute=None,
            concurrency=1, name_translation_strategy="meaning",
        ), expected_revision=0)
        new_conditions = ChooseBookConditions(app.book_conditions, app.book_runtime)(
            "book-1", action="reanalyze", expected_revision=1,
            settings_revision=1, confirmed=True)
        assert new_conditions.id != old_conditions.id
        assert new_conditions.payload["name_translation_policy"]["strategy"] == "meaning"
        with pytest.raises(ConditionsError):
            await app.analysis.run_task(old_task)
    finally:
        await app.aclose()
