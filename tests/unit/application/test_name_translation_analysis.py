"""Evidence retained from accepted chapter analysis responses."""

from analysis_fakes import BOOK_ID, FakeUnitOfWork, analysis_reply, make_analysis_document, make_analysis_section, make_analyze_chapter
from linguary.adapters.llm import FakeLLMClient
from linguary.application.context_builder import ContextBuilder
from linguary.application.glossary_review import candidates_with_decisions
from linguary.application.use_cases.analyze_chapter import AnalyzeChapterCommand
from linguary.domain import ConceptType, TermConcept, TermStatus, TermVariant
from linguary.domain.name_translation import NameTranslationPolicy, NameTranslationStrategy


async def test_whole_chapter_retains_observation() -> None:
    unit_of_work = FakeUnitOfWork(document=make_analysis_document((make_analysis_section(texts=("Alice appears.",)),)))
    reply = analysis_reply(aliases=("b1",), candidates=[{
        "source_text": "Alice", "type": "person", "suggested_translation": "Элис",
        "confidence": 0.9, "model_flagged_review": False, "note": None,
        "name_translation_assessment": {"applied_method": "sound", "eligible_without_review": True,
            "review_reasons": [], "rationale": "Однозначное имя.", "opaque_name": True},
    }])
    scenario, _ = make_analyze_chapter(unit_of_work, FakeLLMClient.always(reply))
    scenario._context_builder = ContextBuilder(name_translation_policy=NameTranslationPolicy(1, NameTranslationStrategy.SOUND))
    scenario._name_translation_policy = scenario._context_builder.name_translation_policy
    scenario._conditions_id = "conditions-1"

    result = await scenario(AnalyzeChapterCommand(BOOK_ID, "section-1"))

    evidence = result.analysis.glossary_candidates[0].name_translation_evidence
    assert evidence is not None
    assert evidence.conditions_id == "conditions-1"
    assert evidence.observations[0].suggested_translation == "Элис"


async def test_one_missing_assessment_prevents_exemption() -> None:
    unit_of_work = FakeUnitOfWork(document=make_analysis_document((make_analysis_section(texts=("Alice appears.",)),)))
    reply = analysis_reply(aliases=("b1",), candidates=[{
        "source_text": "Alice", "type": "person", "suggested_translation": "Элис",
        "confidence": 0.9, "model_flagged_review": False, "note": None,
    }])
    scenario, _ = make_analyze_chapter(unit_of_work, FakeLLMClient.always(reply))
    scenario._context_builder = ContextBuilder(name_translation_policy=NameTranslationPolicy(1, NameTranslationStrategy.SOUND))
    scenario._name_translation_policy = scenario._context_builder.name_translation_policy
    scenario._conditions_id = "conditions-1"

    result = await scenario(AnalyzeChapterCommand(BOOK_ID, "section-1"))

    evidence = result.analysis.glossary_candidates[0].name_translation_evidence
    assert evidence is not None
    assert evidence.observations[0].assessment is None


async def test_user_decision_does_not_forge_assessment() -> None:
    unit_of_work = FakeUnitOfWork(document=make_analysis_document((make_analysis_section(texts=("Alice appears.",)),)))
    reply = analysis_reply(aliases=("b1",), candidates=[{
        "source_text": "Alice", "type": "person", "suggested_translation": "Элис",
        "confidence": 0.9, "model_flagged_review": False, "note": None,
    }])
    scenario, _ = make_analyze_chapter(unit_of_work, FakeLLMClient.always(reply))
    scenario._context_builder = ContextBuilder(name_translation_policy=NameTranslationPolicy(1, NameTranslationStrategy.SOUND))
    scenario._name_translation_policy = scenario._context_builder.name_translation_policy
    scenario._conditions_id = "conditions-1"
    candidate = (await scenario(AnalyzeChapterCommand(BOOK_ID, "section-1"))).analysis.glossary_candidates[0]
    decision = TermConcept(id="decision-1", book_id=BOOK_ID, type=ConceptType.PERSON,
        canonical_source_text="Alice", status=TermStatus.APPROVED,
        variants=(TermVariant(concept_id="decision-1", language="ru", text="Алиса",
            status=TermStatus.APPROVED),))

    adjusted = candidates_with_decisions((candidate,), (decision,), language="ru")[0]

    assert adjusted.suggested_translation == "Алиса"
    assert adjusted.name_translation_evidence == candidate.name_translation_evidence
    assert adjusted.name_translation_evidence.observations[0].assessment is None
