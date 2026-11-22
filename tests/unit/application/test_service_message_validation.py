"""Service-message checks across reader fields and correction snapshot v3."""

from __future__ import annotations

from dataclasses import replace

from processing_fakes import make_block, make_block_result, make_note, make_response

from linguary.adapters.config.settings import AppSettings, LLMSettings
from linguary.application.llm_dto import (
    BlockResultResponse,
    FragmentAssistanceResponse,
    GlossaryHintResponse,
    LanguageNoteResponse,
    SegmentTranslationResponse,
    TranslationResultResponse,
)
from linguary.application.reader_text import source_quote_fragments
from linguary.application.validators.service_messages import service_message_violation
from linguary.composition.correction_runtime import correction_snapshot
from linguary.domain import BlockProcessingResult, BlockTreatment, GlossaryHint, LanguageNote
from linguary.domain.content_plan import SourceAnchor
from linguary.domain.correction import (
    CORRECTION_POLICY_VERSION_V2,
    CORRECTION_POLICY_VERSION_V3,
    is_candidate_correction_policy,
)
from linguary.domain.learning import FragmentAssistance

_OLD_SEGMENT = "Извините, но я не могу выполнить этот запрос."
_CLEAN_SEGMENT = "Вторая строка остаётся чистой."
_OLD_NOTE = "Старая заметка: извините, но я не могу переписать этот оборот."
_OLD_EXCERPT = "I’m sorry, but I cannot keep this excerpt."
_OLD_HINT = "Извините, но я не могу"
_OLD_HINT_NOTE = "Старая подсказка без нового отказа."
_OLD_HELP = "I'm sorry, but I cannot repeat this help."
_NEW_NOTE = "Новая заметка объясняет только время глагола."
_NEW_EXCERPT = "Чистый перевод нового фрагмента."


def test_service_messages_rejects_refusal_in_english_help_and_translated_excerpt() -> None:
    """Strict checks catch a refusal in English Help and in translated_excerpt."""
    block = make_block(order=0)
    help_payload = TranslationResultResponse(
        blocks=(make_block_result(block),),
        fragment_assistances=(
            FragmentAssistanceResponse(
                difficulty_id="diff-en-1",
                point_ids=("point-1",),
                kind="lexical_hint",
                text="I'm sorry, but I cannot fulfill this request.",
            ),
        ),
    )

    help_violation = service_message_violation(
        help_payload, strict_fields=True, include_assistances=True
    )

    assert help_violation is not None
    assert "diff-en-1" in help_violation
    assert block.id not in help_violation
    assert service_message_violation(
        help_payload, strict_fields=True, include_assistances=False
    ) is None

    excerpt_payload = make_response(
        (make_block_result(block),),
        language_notes=(
            make_note(
                block,
                note="Спокойное объяснение оборота.",
                translated_excerpt="I can't fulfill this translation.",
            ),
        ),
    )

    excerpt_violation = service_message_violation(excerpt_payload, strict_fields=True)

    assert excerpt_violation is not None
    assert block.id in excerpt_violation
    assert "i can't fulfill" in excerpt_violation
    assert service_message_violation(excerpt_payload) is None


def test_service_messages_normalizes_curly_apostrophes() -> None:
    """Typographic apostrophes still match the straight-apostrophe markers."""
    sorry = "I’m sorry, but I cannot..."
    cant = "I can’t fulfill..."
    assert "\u2019" in sorry
    assert "\u2019" in cant
    block = make_block(order=0)

    sorry_violation = service_message_violation(
        make_response((make_block_result(block, translated_text=sorry),))
    )
    cant_violation = service_message_violation(
        make_response((make_block_result(block, translated_text=cant),))
    )

    assert sorry_violation is not None
    assert "i'm sorry, but i cannot" in sorry_violation
    assert cant_violation is not None
    assert "i can't fulfill" in cant_violation


def test_service_messages_allows_verbatim_source_quotes() -> None:
    """A service-message phrase copied from the source is not a model refusal."""
    source = 'She said "I\'m sorry, but I cannot fulfill this request."'
    protected = source_quote_fragments((source,))
    assert any("I'm sorry, but I cannot fulfill this request." in quote for quote in protected)
    block = make_block(order=0, text=source)
    quoted = make_response(
        (
            make_block_result(
                block,
                translated_text='Она сказала: "I\'m sorry, but I cannot fulfill this request."',
            ),
        )
    )

    assert service_message_violation(quoted, protected_quotes=protected) is None
    assert service_message_violation(quoted) is not None

    outside = make_response(
        (
            make_block_result(
                block,
                translated_text='Она сказала: "hello." I\'m sorry, but I cannot fulfill this request.',
            ),
        )
    )
    assert service_message_violation(
        outside, protected_quotes=source_quote_fragments(('She said "hello."',))
    ) is not None


def test_correction_checks_only_changed_texts_and_parses_v2_and_v3_snapshots() -> None:
    """Unchanged notes and segments stay, and both correction snapshots parse."""
    from linguary.application.correction_dto import (
        CORRECTION_VERSIONS_V2,
        CORRECTION_VERSIONS_V3,
        parse_correction_snapshot,
        unchanged_reader_texts,
    )

    block = make_block(order=0, text="The clocks were striking thirteen today.")
    anchor = SourceAnchor(block_id=block.id, start=0, end=5, quote="The c")
    previous = BlockProcessingResult(
        block_id=block.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text=f"{_OLD_SEGMENT} {_CLEAN_SEGMENT}",
        glossary_hints=(GlossaryHint("sorry", _OLD_HINT, _OLD_HINT_NOTE),),
        language_notes=(
            LanguageNote("ln-1", block.id, "a truth", _OLD_NOTE, _OLD_EXCERPT),
        ),
        fragment_assistances=(
            FragmentAssistance(
                "diff-1", ("p1",), "lexical_hint", _OLD_HELP, (anchor,),
            ),
        ),
        segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
        segment_translations=(_OLD_SEGMENT, _CLEAN_SEGMENT),
    )
    carried = unchanged_reader_texts(previous)
    assert _OLD_SEGMENT in carried
    assert _OLD_NOTE in carried
    assert _OLD_EXCERPT in carried
    assert _OLD_HINT in carried
    assert _OLD_HELP in carried
    assert _NEW_NOTE not in carried

    kept = TranslationResultResponse(
        blocks=(
            BlockResultResponse(
                block.id,
                BlockTreatment.TRANSLATE,
                glossary_hints=(GlossaryHintResponse("sorry", _OLD_HINT, _OLD_HINT_NOTE),),
                segments=(
                    SegmentTranslationResponse(1, _OLD_SEGMENT),
                    SegmentTranslationResponse(2, _CLEAN_SEGMENT),
                ),
            ),
        ),
        language_notes=(
            LanguageNoteResponse(block.id, "a truth", _OLD_NOTE, _OLD_EXCERPT),
            LanguageNoteResponse(block.id, "another phrase", _NEW_NOTE, _NEW_EXCERPT),
        ),
        fragment_assistances=(
            FragmentAssistanceResponse("diff-1", ("p1",), "lexical_hint", _OLD_HELP),
        ),
    )
    excluded = {block.id: carried}

    assert service_message_violation(kept, strict_fields=True) is not None
    assert service_message_violation(
        kept, strict_fields=True, excluded_texts=excluded
    ) is None

    changed_note = replace(
        kept,
        language_notes=(
            kept.language_notes[0],
            LanguageNoteResponse(
                block.id,
                "another phrase",
                "Извините, но я не могу принять новую заметку.",
                _NEW_EXCERPT,
            ),
        ),
    )
    changed_segment = replace(
        kept,
        blocks=(
            BlockResultResponse(
                block.id,
                BlockTreatment.TRANSLATE,
                glossary_hints=kept.blocks[0].glossary_hints,
                segments=(
                    SegmentTranslationResponse(1, _OLD_SEGMENT),
                    SegmentTranslationResponse(
                        2, "Извините, но я не могу выполнить этот запрос сейчас."
                    ),
                ),
            ),
        ),
    )
    assert service_message_violation(
        changed_note, strict_fields=True, excluded_texts=excluded
    ) is not None
    assert service_message_violation(
        changed_segment, strict_fields=True, excluded_texts=excluded
    ) is not None

    settings = AppSettings(
        llm=LLMSettings(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="gpt-test",
            api_key="sk-corr-1",
            proxy_mode="direct",
        )
    )
    snapshot = correction_snapshot(settings)
    assert snapshot.policy_version == CORRECTION_POLICY_VERSION_V3
    assert is_candidate_correction_policy(snapshot.policy_version) is True
    assert dict(snapshot.versions) == CORRECTION_VERSIONS_V3
    assert CORRECTION_VERSIONS_V3 != CORRECTION_VERSIONS_V2

    parsed_v3 = parse_correction_snapshot(snapshot.to_payload())
    assert parsed_v3.policy_version == CORRECTION_POLICY_VERSION_V3
    assert dict(parsed_v3.versions) == CORRECTION_VERSIONS_V3

    v2_payload = dict(snapshot.to_payload())
    v2_payload["policy_version"] = CORRECTION_POLICY_VERSION_V2
    v2_payload["versions"] = dict(CORRECTION_VERSIONS_V2)
    parsed_v2 = parse_correction_snapshot(v2_payload)
    assert parsed_v2.policy_version == CORRECTION_POLICY_VERSION_V2
    assert dict(parsed_v2.versions) == CORRECTION_VERSIONS_V2
    assert is_candidate_correction_policy(parsed_v2.policy_version) is True


def test_validator_v2_preserves_segments_for_excluded_service_texts_and_masks_apostrophes() -> None:
    """v2 validation keeps block.segments for service-message exclusions and normalizes apostrophes in quotes."""
    import pytest
    from linguary.application.correction_dto import unchanged_reader_texts
    from linguary.application.retry_policy import ResultValidationError
    from linguary.application.validators import TranslationResultValidator
    from linguary.domain import (
        BlockSegment,
        TreatmentPlan,
        TreatmentPlanItem,
        TranslationUnit,
    )

    part1 = "First line of English prose."
    part2 = " Second line of English prose."
    block = replace(
        make_block(order=0, text=part1 + part2),
        segments=(
            BlockSegment(0, len(part1), 0),
            BlockSegment(len(part1), len(part1) + len(part2), 1),
        ),
    )
    plan = TreatmentPlan(
        section_id=block.section_id,
        items=(
            TreatmentPlanItem(
                block_id=block.id,
                treatment=BlockTreatment.TRANSLATE,
                segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
                sentence_indexes=(0, 1),
            ),
        ),
    )
    unit = TranslationUnit.of_blocks(id="u1", blocks=(block,), plan=plan)
    validator = TranslationResultValidator(
        target_language_min_share=0.5,
        language_check_min_length=10,
        repetition_ngram_size=5,
        repetition_min_repeats=4,
    )

    previous = BlockProcessingResult(
        block_id=block.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text=f"{_OLD_SEGMENT} Старый перевод второй строки.",
        segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
        segment_translations=(_OLD_SEGMENT, "Старый перевод второй строки."),
    )
    excluded = {block.id: unchanged_reader_texts(previous)}

    # Segment 1 untouched (contains marker), segment 2 changed to clean text -> passes under v2
    clean_edit = TranslationResultResponse(
        blocks=(
            BlockResultResponse(
                block.id,
                BlockTreatment.TRANSLATE,
                segments=(
                    SegmentTranslationResponse(1, _OLD_SEGMENT),
                    SegmentTranslationResponse(2, "Обновлённый чистый перевод второй строки."),
                ),
            ),
        ),
    )
    results = validator.validate(
        clean_edit,
        unit=unit,
        target_language="ru",
        note_id_factory=lambda: "ln-ch01-001",
        validation_policy="v2",
        excluded_service_texts=excluded,
    )
    assert len(results) == 1

    # Marker introduced in changed segment 2 -> rejected under v2
    dirty_edit = TranslationResultResponse(
        blocks=(
            BlockResultResponse(
                block.id,
                BlockTreatment.TRANSLATE,
                segments=(
                    SegmentTranslationResponse(1, _OLD_SEGMENT),
                    SegmentTranslationResponse(2, "Извините, но я не могу выполнить этот запрос снова."),
                ),
            ),
        ),
    )
    with pytest.raises(ResultValidationError) as caught:
        validator.validate(
            dirty_edit,
            unit=unit,
            target_language="ru",
            note_id_factory=lambda: "ln-ch01-001",
            validation_policy="v2",
            excluded_service_texts=excluded,
        )
    assert caught.value.violation is not None
    assert caught.value.violation.rule_code == "service_message_leak"

    # Curly apostrophe in protected source quote matches straight apostrophe in model output
    curly_source = 'She whispered “I’m sorry, but I cannot help you.”'
    protected = source_quote_fragments((curly_source,))
    quoted_block = make_block(order=1, text=curly_source)
    quoted_resp = make_response(
        (
            make_block_result(
                quoted_block,
                translated_text="Она прошептала: \"I'm sorry, but I cannot help you.\"",
            ),
        )
    )
    assert service_message_violation(quoted_resp, protected_quotes=protected) is None
