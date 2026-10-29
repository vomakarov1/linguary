"""Persistence of name translation provenance across migration and analysis reloads."""

from dataclasses import asdict, replace

import pytest
from alembic import command
from sqlalchemy import text

from linguary.domain.name_translation import (
    NameTranslationAssessment,
    NameTranslationEvidence,
    NameTranslationObservation,
    NameTranslationPolicy,
    NameTranslationStrategy,
)
from linguary.application.ports.persistence import PersistenceError
from persistence_factories import make_analysis, make_book, make_document


def _seed(unit_of_work, *, evidence=None) -> None:
    analysis = make_analysis()
    if evidence is not None:
        candidate = replace(analysis.glossary_candidates[0], name_translation_evidence=evidence)
        analysis = replace(analysis, glossary_candidates=(candidate,))
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
        uow.documents.save(make_document())
        uow.analyses.save("book-1", analysis)


def _evidence() -> NameTranslationEvidence:
    return NameTranslationEvidence(
        1, "conditions-1", 1, NameTranslationStrategy.SOUND,
        (
            NameTranslationObservation("Уинстон Смит", 0.93, False,
                NameTranslationAssessment("sound", True, (), "Звучание имени", opaque_name=True)),
            NameTranslationObservation("Уинстон Смит", 0.68, True,
                NameTranslationAssessment("uncertain", False, ("ambiguous",), "Есть сомнение")),
        ),
    )


def test_upgrade_keeps_old_candidates(alembic_config, engine, unit_of_work) -> None:
    _seed(unit_of_work)
    command.downgrade(alembic_config, "0008")
    with engine.connect() as connection:
        old = connection.execute(text("SELECT id, suggested_translation FROM glossary_candidates")).one()
    command.upgrade(alembic_config, "head")
    with engine.connect() as connection:
        row = connection.execute(text("SELECT id, suggested_translation, name_translation_evidence_json FROM glossary_candidates")).one()
    assert row[:2] == old
    assert row[2] is None


def test_evidence_round_trip(unit_of_work) -> None:
    evidence = _evidence()
    _seed(unit_of_work, evidence=evidence)
    with unit_of_work.transaction() as uow:
        loaded = uow.analyses.get_for_section("book-1", make_analysis().section_id)
    assert loaded.glossary_candidates[0].name_translation_evidence == evidence
    assert not loaded.glossary_candidates[0].name_translation_evidence.supports(
        NameTranslationPolicy(1, NameTranslationStrategy.SOUND), "conditions-1"
    )


def test_old_assessment_without_opaque_name_defaults_false(unit_of_work, engine) -> None:
    _seed(unit_of_work, evidence=_evidence())
    with engine.begin() as connection:
        connection.execute(text("UPDATE glossary_candidates SET name_translation_evidence_json = "
                                "json_remove(name_translation_evidence_json, '$.observations[0].assessment.opaque_name')"))
    with unit_of_work.transaction() as uow:
        loaded = uow.analyses.get_for_section("book-1", make_analysis().section_id)
    assessment = loaded.glossary_candidates[0].name_translation_evidence.observations[0].assessment
    assert assessment.opaque_name is False


def test_null_is_legacy(unit_of_work) -> None:
    _seed(unit_of_work)
    with unit_of_work.transaction() as uow:
        loaded = uow.analyses.get_for_section("book-1", make_analysis().section_id)
    assert loaded.glossary_candidates[0].name_translation_evidence is None


def test_corrupt_evidence_fails_read(engine, unit_of_work) -> None:
    _seed(unit_of_work)
    with engine.begin() as connection:
        connection.execute(text("UPDATE glossary_candidates SET name_translation_evidence_json = '{}'"))
    with pytest.raises(PersistenceError, match="свидетельства"):
        with unit_of_work.transaction() as uow:
            uow.analyses.get_for_section("book-1", make_analysis().section_id)


def test_archived_observations_keep_policy_and_conditions() -> None:
    archived = asdict(_evidence())
    assert archived["conditions_id"] == "conditions-1"
    assert archived["policy_version"] == 1
    assert archived["strategy"] == "sound"
    assert len(archived["observations"]) == 2


def test_downgrade_refuses_saved_evidence(alembic_config, unit_of_work) -> None:
    _seed(unit_of_work, evidence=_evidence())
    with pytest.raises(RuntimeError, match="свидетельств"):
        command.downgrade(alembic_config, "0008")
