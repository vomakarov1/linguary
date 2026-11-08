"""Correction fingerprints preserve the immutable source block representation."""

from dataclasses import replace
import json
from types import SimpleNamespace
from contextlib import nullcontext

import pytest

from analysis_fakes import make_analysis_document, make_analysis_section

from linguary.application.correction import CorrectionInputs
from linguary.application.correction_dto import (
    CORRECTION_VERSIONS, CorrectionSnapshot, parse_correction_snapshot,
)
from linguary.application.ports.llm import LLMResponseKind
from linguary.application.use_cases.correct_chapter import CorrectChapter
from linguary.domain import BlockProcessingResult, BlockTreatment, GlossarySnapshot


def test_pinned_v1_correction_snapshot_keeps_v1_request_contract():
    snapshot = CorrectionSnapshot(
        model='legacy-model',
        route={'provider': 'openai_compatible', 'base_url': 'https://example.test/v1',
               'proxy': '', 'route_id': 'legacy-route'},
        parameters={'temperature': 0, 'max_output_tokens': None, 'timeout_seconds': 30,
                    'max_attempts': 1, 'max_unit_source_chars': 1000,
                    'input_price_per_mtok': 0, 'output_price_per_mtok': 0},
        prompt='Check the chapter translation.', versions=CORRECTION_VERSIONS,
    )
    saved = parse_correction_snapshot(snapshot.to_payload())
    section = make_analysis_section()
    block = section.blocks[0]
    result = BlockProcessingResult(block_id=block.id, treatment=BlockTreatment.READ_AS_IS)
    inputs = SimpleNamespace(context={'book_conditions': {}}, blocks=(block,),
                             results={block.id: result}, run=SimpleNamespace(id='run'))
    task = SimpleNamespace(book_id='book', id='task', scope_id=section.id)
    correct = CorrectChapter(unit_of_work=None, llm_client=None, snapshot=saved, validator=None)

    request = correct._request(inputs, task, (block.id,))

    assert isinstance(saved, CorrectionSnapshot)
    assert saved.policy_version == 'chapter-correction-v1'
    assert request.kind is LLMResponseKind.CHAPTER_CORRECTION
    assert request.contract_version == 1


def test_correction_fingerprints_include_source_attributes():
    section = make_analysis_section()
    block = replace(section.blocks[0], attributes={'lang': 'en'})
    section = replace(section, blocks=(block,))
    inputs = CorrectionInputs(
        document=make_analysis_document((section,)), section=section, analysis=None,
        run=SimpleNamespace(id='run', glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(payload={'schema_version': 1,
            'pronunciation_versions': {'source': 'private-dictionary-source'}}),
        snapshot=SimpleNamespace(to_payload=lambda: {'model': 'correction'}),
        blocks=(block,), results={}, bindings={block.id: None},
    )

    chapter_block, = inputs.context['chapter_blocks']
    assert chapter_block['text'] == block.text
    assert chapter_block['block_id'] == 'b1'
    assert 'chapter_originals' not in inputs.context
    assert 'book_conditions' not in inputs.context
    assert 'correction_snapshot' not in inputs.context
    assert inputs.checksums == inputs.checksums

    changed = replace(block, attributes={'lang': 'fr'})
    changed_section = replace(section, blocks=(changed,))
    changed_inputs = replace(inputs, section=changed_section, blocks=(changed,))
    assert changed_inputs.checksums[block.id] != inputs.checksums[block.id]
    assert dict(block.attributes) == {'lang': 'en'}

    snapshot = SimpleNamespace(prompt='Correct the chapter.',
        parameters={'max_attempts': 1, 'max_output_tokens': None})
    correct = CorrectChapter(unit_of_work=None, llm_client=None, snapshot=snapshot, validator=None)
    result = BlockProcessingResult(block_id=block.id, treatment=BlockTreatment.READ_AS_IS)
    task = SimpleNamespace(book_id=inputs.document.id, id='task', scope_id=section.id)
    request = correct._request(replace(inputs, results={block.id: result}), task, (block.id,))
    assert json.loads(request.input_text)['blocks'][0]['original'] == {'text': block.text}
    assert json.loads(request.input_text)['blocks'][0]['block_id'] == 'b1'
    assert 'book_conditions' not in json.loads(request.input_text)
    assert 'private-dictionary-source' not in request.input_text
    assert 'pronunciation_assistances' not in json.loads(request.input_text)['blocks'][0]['current_result']
    changed_conditions = replace(inputs, conditions=SimpleNamespace(payload={'schema_version': 1,
        'pronunciation_versions': {'source': 'different-dictionary-source'}}))
    assert changed_conditions.checksums == inputs.checksums


def test_correction_inputs_omits_book_conditions_and_filters_glossary_by_chapter():
    section = make_analysis_section(texts=("Netherfield is mentioned here.", "Second line without terms."))
    doc = make_analysis_document((section,))
    from linguary.domain import GlossarySnapshotEntry, ConceptType, TermStatus
    glossary = GlossarySnapshot(entries=(
        GlossarySnapshotEntry(source_text="Netherfield", translation="Незерфилд", type=ConceptType.PLACE, status=TermStatus.LOCKED),
        GlossarySnapshotEntry(source_text="Pemberley", translation="Пемберли", type=ConceptType.PLACE, status=TermStatus.LOCKED),
    ))
    inputs = CorrectionInputs(
        document=doc, section=section, analysis=None,
        run=SimpleNamespace(id='run', glossary_snapshot=glossary),
        conditions=SimpleNamespace(payload={'schema_version': 1}),
        snapshot=SimpleNamespace(to_payload=lambda: {'model': 'correction'}),
        blocks=section.blocks, results={}, bindings={b.id: None for b in section.blocks},
    )

    context = inputs.context
    assert 'book_conditions' not in context
    assert 'correction_snapshot' not in context
    assert 'chapter_originals' not in context
    assert 'chapter_blocks' in context
    assert len(context['chapter_blocks']) == 2
    assert context['chapter_blocks'][0] == {'block_id': 'b1', 'text': section.blocks[0].text}
    assert context['chapter_blocks'][1] == {'block_id': 'b2', 'text': section.blocks[1].text}

    run_glossary = context['run_glossary']
    entry_sources = [e['source_text'] for e in run_glossary['entries']]
    assert "Netherfield" in entry_sources
    assert "Pemberley" not in entry_sources
    assert inputs.block_aliases[section.blocks[0].id] == 'b1'
    assert inputs.block_aliases[section.blocks[1].id] == 'b2'
    assert inputs.alias_to_block_id['b1'] == section.blocks[0].id
    assert inputs.alias_to_block_id['b2'] == section.blocks[1].id


def test_correct_chapter_uses_block_aliases_and_resolves_uuids_before_coverage_check():
    from linguary.application.correction_dto import ChapterCorrectionResponse, UnchangedBlockResponse
    section = make_analysis_section(texts=("First.", "Second."))
    doc = make_analysis_document((section,))
    inputs = CorrectionInputs(
        document=doc, section=section, analysis=None,
        run=SimpleNamespace(id='run', glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(payload={'schema_version': 1}),
        snapshot=SimpleNamespace(to_payload=lambda: {'model': 'correction'}),
        blocks=section.blocks, results={
            b.id: BlockProcessingResult(block_id=b.id, treatment=BlockTreatment.TRANSLATE, translated_text="Перевод")
            for b in section.blocks
        },
        bindings={b.id: None for b in section.blocks},
    )
    snapshot = SimpleNamespace(prompt='Correct the chapter.',
        parameters={'max_attempts': 1, 'max_output_tokens': None})
    correct = CorrectChapter(unit_of_work=None, llm_client=None, snapshot=snapshot, validator=None)
    task = SimpleNamespace(book_id=doc.id, id='task', scope_id=section.id)

    target_ids = (section.blocks[0].id, section.blocks[1].id)
    request = correct._request(inputs, task, target_ids)

    body = json.loads(request.input_text)
    assert body['blocks'][0]['block_id'] == 'b1'
    assert body['blocks'][1]['block_id'] == 'b2'
    assert body['blocks'][0]['original'] == {'text': section.blocks[0].text}
    assert dict(request.block_aliases) == {'b1': section.blocks[0].id, 'b2': section.blocks[1].id}
    assert request.ordered_block_ids == target_ids

    # Model returns aliases b1 and b2
    raw_response = ChapterCorrectionResponse(blocks=(
        UnchangedBlockResponse(block_id='b1'),
        UnchangedBlockResponse(block_id='b2'),
    ))
    resolved = raw_response.with_resolved_block_ids(dict(request.block_aliases))
    assert resolved.block_ids == target_ids
    # Passes coverage check with canonical UUIDs
    resolved.validate_coverage(request.ordered_block_ids)


def test_unchecked_blocks_skips_pure_read_as_is_blocks_in_story_chapters():
    from linguary.application.correction import unchecked_blocks
    from linguary.domain import ChapterAnalysis, TreatmentPlan, LanguageNote
    section = make_analysis_section(texts=("Block 1 read as is.", "Block 2 translate.", "Block 3 read as is with note."))
    doc = make_analysis_document((section,))
    b1, b2, b3 = section.blocks

    plan = TreatmentPlan.of_treatments(section.id, (
        (b1.id, BlockTreatment.READ_AS_IS),
        (b2.id, BlockTreatment.TRANSLATE),
        (b3.id, BlockTreatment.READ_AS_IS),
    ))
    analysis = ChapterAnalysis.create(
        id='analysis-1', section_id=section.id, treatment_plan=plan,
        translation_instructions='', chapter_summary='', expected_block_ids=(b1.id, b2.id, b3.id),
    )
    note = LanguageNote(
        id='ln-1', block_id=b3.id, original_excerpt='Block 3', note='Заметка',
    )
    results = {
        b1.id: BlockProcessingResult(block_id=b1.id, treatment=BlockTreatment.READ_AS_IS),
        b2.id: BlockProcessingResult(block_id=b2.id, treatment=BlockTreatment.TRANSLATE, translated_text="Блок 2 перевод"),
        b3.id: BlockProcessingResult(block_id=b3.id, treatment=BlockTreatment.READ_AS_IS, language_notes=(note,)),
    }
    inputs = CorrectionInputs(
        document=doc, section=section, analysis=analysis,
        run=SimpleNamespace(id='run', glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(payload={'schema_version': 1}),
        snapshot=SimpleNamespace(to_payload=lambda: {'model': 'correction'}),
        blocks=section.blocks, results=results, bindings={b.id: None for b in section.blocks},
    )

    class FakeTx:
        class work:
            @staticmethod
            def correction_checks(run_id, section_id):
                return ()

    missing = unchecked_blocks(FakeTx(), inputs)
    # b1 is pure READ_AS_IS without notes/difficulties, so it must be skipped
    assert b1.id not in missing
    assert b2.id in missing
    assert b3.id in missing



@pytest.mark.asyncio
async def test_source_text_limit_does_not_reject_complete_correction_context(monkeypatch):
    import linguary.application.use_cases.correct_chapter as module
    missing = [('one', 'two'), ()]
    monkeypatch.setattr(module, 'unchecked_blocks', lambda tx, inputs: missing.pop(0))
    snapshot = SimpleNamespace(parameters={'max_attempts': 1, 'max_unit_source_chars': 1})
    correct = CorrectChapter(unit_of_work=SimpleNamespace(transaction=lambda: nullcontext(None)),
        llm_client=None, snapshot=snapshot, validator=None)
    correct._load = lambda tx, command: (SimpleNamespace(run=SimpleNamespace(id='run')), None, None)
    groups = []
    async def group(command, blocks):
        groups.append(blocks)
        return len(blocks)
    correct._group = group
    result = await correct(SimpleNamespace(section_id='chapter'))
    assert groups == [('one', 'two')]
    assert result.checked_blocks == 2


def test_correct_chapter_request_aliases_analysis_and_strips_internal_result_fields():
    from dataclasses import asdict
    from analysis_fakes import make_analysis_candidate
    from linguary.application.use_cases.correct_chapter import (
        CorrectChapter, _llm_block_result, _llm_chapter_analysis,
    )
    from linguary.domain import (
        BeforeYouReadEntry, ChapterAnalysis, GlossaryHint, LanguageNote,
        ResultWarning, ResultWarningKind, TreatmentPlan,
    )
    from linguary.domain.content_plan import (
        ContentEvidence, ContentPlan, ContextFact, Difficulty, HelpEvidence,
        RequiredPoint, SourceAnchor,
    )
    from linguary.domain.learning import FragmentAssistance

    section = make_analysis_section(
        texts=("First sentence with idiom.", "Second sentence he spoke."),
    )
    doc = make_analysis_document((section,))
    b1, b2 = section.blocks

    plan = TreatmentPlan.of_treatments(
        section.id,
        ((b1.id, BlockTreatment.TRANSLATE), (b2.id, BlockTreatment.TRANSLATE)),
    )
    content_plan = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(
            Difficulty(
                id="d1",
                anchors=(SourceAnchor(block_id=b1.id, start=0, end=5, quote="First"),),
                owner_block_id=b1.id,
                kind="idiom",
                priority="critical",
                help_kind="explanation",
                required_points=(RequiredPoint(id="p1", description="meaning"),),
            ),
        ),
        before_you_read_evidence=(
            HelpEvidence(
                difficulty_id="d1",
                point_id="p1",
                artifact_kind="before_you_read.explanation",
                block_id=b1.id,
                artifact_key="a" * 64,
                body_excerpt="идиома",
            ),
        ),
        context_facts=(
            ContextFact(
                id="cf1",
                kind="speaker",
                anchors=(SourceAnchor(block_id=b2.id, start=0, end=6, quote="Second"),),
                subject_anchor=SourceAnchor(block_id=b2.id, start=16, end=18, quote="he"),
                value="Narrator",
                certainty="explicit",
            ),
        ),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="Keep tone.",
        chapter_summary="Summary.",
        expected_block_ids=(b1.id, b2.id),
        glossary_candidates=(make_analysis_candidate(analysis_id="analysis-1"),),
        before_you_read=(BeforeYouReadEntry(expression="idiom", explanation="идиома"),),
        content_plan=content_plan,
    )
    res1 = BlockProcessingResult(
        block_id=b1.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Первое предложение с идиомой.",
        glossary_hints=(GlossaryHint(original="First", translation="Первое"),),
        language_notes=(
            LanguageNote(
                id="ln-ch01-001",
                block_id=b1.id,
                original_excerpt="First",
                note="Заметка",
            ),
        ),
        warnings=(ResultWarning(kind=ResultWarningKind.REPETITION, message="повтор"),),
        content_evidence=ContentEvidence(
            plan_sha256="b" * 64,
            evidence=(
                HelpEvidence(
                    difficulty_id="d1",
                    point_id="p1",
                    artifact_kind="language_note.note",
                    block_id=b1.id,
                    artifact_key="c" * 64,
                    body_excerpt="Заметка",
                ),
            ),
            version=3,
        ),
        fragment_assistances=(
            FragmentAssistance(
                difficulty_id="d1",
                point_ids=("p1",),
                kind="russian_explanation",
                text="Пояснение",
                anchors=(SourceAnchor(block_id=b1.id, start=0, end=5, quote="First"),),
            ),
        ),
    )
    res2 = BlockProcessingResult(
        block_id=b2.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Второе предложение сказал он.",
    )
    inputs = CorrectionInputs(
        document=doc,
        section=section,
        analysis=analysis,
        run=SimpleNamespace(id="run", glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(payload={"schema_version": 1}),
        snapshot=SimpleNamespace(to_payload=lambda: {"model": "correction"}),
        blocks=section.blocks,
        results={b1.id: res1, b2.id: res2},
        bindings={b1.id: None, b2.id: None},
    )
    checksums_before = dict(inputs.checksums)
    context_sha_before = inputs.context_sha256

    snapshot = SimpleNamespace(
        prompt="Correct the chapter.",
        parameters={"max_attempts": 1, "max_output_tokens": None},
    )
    correct = CorrectChapter(unit_of_work=None, llm_client=None, snapshot=snapshot, validator=None)
    task = SimpleNamespace(book_id=doc.id, id="task", scope_id=section.id)

    request = correct._request(inputs, task, (b1.id, b2.id))
    body = json.loads(request.input_text)

    assert inputs.checksums == checksums_before
    assert inputs.context_sha256 == context_sha_before
    assert "glossary_candidates" in inputs.context["chapter_analysis"]

    llm_analysis = body["chapter_analysis"]
    assert "glossary_candidates" not in llm_analysis
    assert "name_translation_evidence" not in llm_analysis
    assert [item["block_id"] for item in llm_analysis["treatment_plan"]["items"]] == ["b1", "b2"]

    llm_cp = llm_analysis["content_plan"]
    assert "before_you_read_evidence" not in llm_cp
    assert llm_cp["difficulties"][0]["owner_block_id"] == "b1"
    assert llm_cp["difficulties"][0]["anchors"] == [{"block_id": "b1", "quote": "First"}]
    assert llm_cp["context_facts"][0]["anchors"] == [{"block_id": "b2", "quote": "Second"}]
    assert llm_cp["context_facts"][0]["subject_anchor"] == {"block_id": "b2", "quote": "he"}

    with_name_ev = {**asdict(analysis), "name_translation_evidence": [{"term": "Bennet"}]}
    cleaned_direct = _llm_chapter_analysis(with_name_ev, inputs.block_aliases)
    assert "name_translation_evidence" not in cleaned_direct
    assert _llm_chapter_analysis(None, inputs.block_aliases) is None

    current_result = body["blocks"][0]["current_result"]
    assert "block_id" not in current_result
    assert "content_evidence" not in current_result
    assert "warnings" not in current_result
    assert "pronunciation_assistances" not in current_result
    assert current_result["language_notes"] == [
        {
            "block_id": "b1",
            "original_excerpt": "First",
            "note": "Заметка",
            "translated_excerpt": None,
        }
    ]
    assert current_result["fragment_assistances"][0]["anchors"] == [
        {"block_id": "b1", "quote": "First"}
    ]
    assert b1.id not in request.input_text
    assert b2.id not in request.input_text
    assert _llm_block_result(asdict(res1), inputs.block_aliases) == current_result


def test_correct_chapter_v2_review_request_includes_block_alias_and_aliases_notes():
    from dataclasses import asdict
    from analysis_fakes import make_analysis_candidate
    from linguary.application.use_cases.correct_chapter import CorrectChapterV2
    from linguary.domain import ChapterAnalysis, LanguageNote, ResultWarning, ResultWarningKind, TreatmentPlan
    from linguary.domain.correction import CorrectionCandidate

    section = make_analysis_section(texts=("First block.", "Second block."))
    doc = make_analysis_document((section,))
    b1, b2 = section.blocks

    plan = TreatmentPlan.of_treatments(
        section.id,
        ((b1.id, BlockTreatment.TRANSLATE), (b2.id, BlockTreatment.TRANSLATE)),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="Keep tone.",
        chapter_summary="Summary.",
        expected_block_ids=(b1.id, b2.id),
        glossary_candidates=(make_analysis_candidate(analysis_id="analysis-1"),),
    )
    current_res = BlockProcessingResult(
        block_id=b1.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Первый блок.",
        language_notes=(
            LanguageNote(
                id="ln-ch01-001",
                block_id=b1.id,
                original_excerpt="First",
                note="Старая заметка",
            ),
        ),
        warnings=(ResultWarning(kind=ResultWarningKind.REPETITION, message="повтор"),),
    )
    proposed_res = BlockProcessingResult(
        block_id=b1.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Первый блок уточнён.",
        language_notes=(
            LanguageNote(
                id="ln-ch01-002",
                block_id=b1.id,
                original_excerpt="First",
                note="Новая заметка",
            ),
        ),
    )
    res2 = BlockProcessingResult(
        block_id=b2.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Второй блок.",
    )
    inputs = CorrectionInputs(
        document=doc,
        section=section,
        analysis=analysis,
        run=SimpleNamespace(id="run", glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(payload={"schema_version": 1}),
        snapshot=SimpleNamespace(to_payload=lambda: {"model": "correction"}),
        blocks=section.blocks,
        results={b1.id: current_res, b2.id: res2},
        bindings={b1.id: None, b2.id: None},
    )
    from linguary.domain.content_plan import canonical_sha
    candidate_payload = json.loads(json.dumps({
        "schema_version": 1,
        "book_id": doc.id,
        "run_id": "run",
        "conditions_id": "cond-1",
        "section_id": section.id,
        "block_id": b1.id,
        "input_sha256": inputs.checksums[b1.id],
        "input_generation": 1,
        "group_id": "group-1",
        "proposal_attempt_id": "att-1",
        "original": {"text": b1.text},
        "current_result": asdict(current_res),
        "result": asdict(proposed_res),
        "proposal": {"issues": [{"kind": "note", "quote": "First", "explanation": "Уточнить"}]},
    }))
    candidate = CorrectionCandidate(
        candidate_id="cand-1",
        book_id=doc.id,
        run_id="run",
        conditions_id="cond-1",
        section_id=section.id,
        block_id=b1.id,
        input_sha256=inputs.checksums[b1.id],
        input_generation=1,
        proposal_sha256=canonical_sha(candidate_payload),
        payload=candidate_payload,
        group_id="group-1",
        proposal_attempt_id="att-1",
    )
    snapshot = SimpleNamespace(
        prompt="Propose.",
        review_prompt="Review proposals.",
        parameters={"max_attempts": 1, "max_output_tokens": None},
    )
    correct_v2 = CorrectChapterV2(unit_of_work=None, llm_client=None, snapshot=snapshot, validator=None)
    task = SimpleNamespace(book_id=doc.id, id="task", scope_id=section.id)

    request = correct_v2._review_request(inputs, task, (candidate,))
    body = json.loads(request.input_text)

    assert "glossary_candidates" not in body["chapter_analysis"]
    assert [item["block_id"] for item in body["chapter_analysis"]["treatment_plan"]["items"]] == ["b1", "b2"]

    cand_payload, = body["candidates"]
    assert cand_payload["block_id"] == "b1"
    assert cand_payload["original"] == {"text": b1.text}
    assert "block_id" not in cand_payload["current_result"]
    assert "warnings" not in cand_payload["current_result"]
    assert "content_evidence" not in cand_payload["current_result"]
    assert cand_payload["current_result"]["language_notes"] == [
        {
            "block_id": "b1",
            "original_excerpt": "First",
            "note": "Старая заметка",
            "translated_excerpt": None,
        }
    ]
    assert cand_payload["proposed_result"]["language_notes"] == [
        {
            "block_id": "b1",
            "original_excerpt": "First",
            "note": "Новая заметка",
            "translated_excerpt": None,
        }
    ]
    assert b1.id not in request.input_text
    assert b2.id not in request.input_text


def test_candidates_and_prepare_correction_candidates_use_section_results_without_list_for_run():
    from dataclasses import asdict
    from processing_fakes import make_validator
    from linguary.application.correction_dto import (
        CORRECTION_VERSIONS_V2, ChapterCorrectionV2Response, CorrectionSnapshotV2, UnchangedBlockResponse,
    )
    from linguary.application.use_cases.correct_chapter import CorrectChapter, prepare_correction_candidates
    from linguary.domain import ChapterAnalysis, LanguageNote, TreatmentPlan
    from linguary.domain.content_plan import canonical_sha
    from linguary.domain.correction import CorrectionCandidate

    section = make_analysis_section(texts=("First block.", "Second block."))
    doc = make_analysis_document((section,))
    b1, b2 = section.blocks

    plan = TreatmentPlan.of_treatments(
        section.id,
        ((b1.id, BlockTreatment.TRANSLATE), (b2.id, BlockTreatment.TRANSLATE)),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(b1.id, b2.id),
    )
    res1 = BlockProcessingResult(
        block_id=b1.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Первый блок.",
        language_notes=(
            LanguageNote(id="ln-ch01-001", block_id=b1.id, original_excerpt="First", note="Заметка."),
        ),
    )
    res2 = BlockProcessingResult(
        block_id=b2.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Второй блок.",
    )
    snapshot = CorrectionSnapshotV2(
        model="gpt-test",
        route={"provider": "openai_compatible", "base_url": "https://example.test/v1", "proxy": "", "route_id": "r1"},
        parameters={
            "temperature": 0.2,
            "max_output_tokens": 1000,
            "timeout_seconds": 30,
            "max_attempts": 1,
            "max_unit_source_chars": 1000,
            "input_price_per_mtok": 0,
            "output_price_per_mtok": 0,
        },
        prompt="Propose.",
        review_prompt="Review.",
        versions=CORRECTION_VERSIONS_V2,
    )
    inputs = CorrectionInputs(
        document=doc,
        section=section,
        analysis=analysis,
        run=SimpleNamespace(id="run-1", glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(payload={"schema_version": 1, "versions": {"validators": 3}}),
        snapshot=snapshot,
        blocks=section.blocks,
        results={b1.id: res1, b2.id: res2},
        bindings={b1.id: None, b2.id: None},
    )

    list_candidates_calls: list[tuple[str, str | None, set[str] | None]] = []

    def forbidden_list_for_run(run_id: str):
        raise AssertionError("tx.results.list_for_run must not be called when inputs.results is available")

    def fake_list_correction_candidates(run_id: str, *, section_id: str | None = None, states=None):
        list_candidates_calls.append((run_id, section_id, set(states) if states is not None else None))
        return ()

    tx = SimpleNamespace(
        results=SimpleNamespace(list_for_run=forbidden_list_for_run),
        work=SimpleNamespace(list_correction_candidates=fake_list_correction_candidates),
    )

    validator = make_validator()
    correct = CorrectChapter(unit_of_work=None, llm_client=None, snapshot=snapshot, validator=validator)
    response = ChapterCorrectionV2Response(blocks=(UnchangedBlockResponse(block_id=b1.id),))
    validated = correct._candidates(tx, inputs, response)
    assert validated == (res1,)
    assert list_candidates_calls == [
        ("run-1", section.id, {"awaiting_review", "awaiting_human", "auto_applied", "accepted", "kept"})
    ]

    proposed_res = BlockProcessingResult(
        block_id=b1.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Первый блок уточнён.",
        language_notes=(
            LanguageNote(id="ln-ch01-001", block_id=b1.id, original_excerpt="First", note="Заметка."),
        ),
    )
    candidate_payload = json.loads(json.dumps({
        "schema_version": 1,
        "book_id": doc.id,
        "run_id": "run-1",
        "conditions_id": "cond-1",
        "section_id": section.id,
        "block_id": b1.id,
        "input_sha256": inputs.checksums[b1.id],
        "input_generation": 1,
        "group_id": "group-1",
        "proposal_attempt_id": "att-1",
        "original": {"text": b1.text},
        "current_result": asdict(res1),
        "result": asdict(proposed_res),
        "proposal": {
            "block_id": b1.id,
            "status": "corrected",
            "translated_text": "Первый блок уточнён.",
            "glossary_hints": [],
            "language_notes": [
                {"source_block_id": b1.id, "original_excerpt": "First", "note": "Заметка.", "translated_excerpt": None}
            ],
            "issues": [
                {
                    "field_path": "translated_text",
                    "source_quote": "First",
                    "current_evidence": {"kind": "quote", "field_path": "translated_text", "quote": "Первый"},
                    "problem": "Неточность",
                    "minimal_fix_reason": "Уточнить",
                }
            ],
        },
    }))
    candidate = CorrectionCandidate(
        candidate_id="cand-1",
        book_id=doc.id,
        run_id="run-1",
        conditions_id="cond-1",
        section_id=section.id,
        block_id=b1.id,
        input_sha256=inputs.checksums[b1.id],
        input_generation=1,
        proposal_sha256=canonical_sha(candidate_payload),
        payload=candidate_payload,
        group_id="group-1",
        proposal_attempt_id="att-1",
    )
    list_candidates_calls.clear()
    prepared = prepare_correction_candidates(tx, inputs, (candidate,), validator)
    assert prepared == (proposed_res,)
    assert list_candidates_calls == [
        ("run-1", section.id, {"awaiting_review", "awaiting_human", "auto_applied", "accepted", "kept"})
    ]


@pytest.mark.asyncio
async def test_correct_chapter_pre_chunks_blocks_by_max_unit_source_chars(monkeypatch):
    import linguary.application.use_cases.correct_chapter as module
    from linguary.application.use_cases.correct_chapter import CorrectChapterV2, _chunk_correction_blocks

    section = make_analysis_section(
        texts=("A" * 60, "B" * 60, "C" * 60, "D" * 60),
    )
    b1, b2, b3, b4 = section.blocks
    inputs = SimpleNamespace(
        run=SimpleNamespace(id="run-1"),
        blocks=section.blocks,
    )

    chunks = _chunk_correction_blocks((b1.id, b2.id, b3.id, b4.id), inputs, 130)
    assert chunks == ((b1.id, b2.id), (b3.id, b4.id))

    missing_v1 = [(b1.id, b2.id, b3.id, b4.id), ()]
    monkeypatch.setattr(module, "unchecked_blocks", lambda tx, inp: missing_v1.pop(0))
    snapshot = SimpleNamespace(parameters={"max_attempts": 1, "max_unit_source_chars": 130})
    correct_v1 = CorrectChapter(
        unit_of_work=SimpleNamespace(transaction=lambda: nullcontext(None)),
        llm_client=None,
        snapshot=snapshot,
        validator=None,
    )
    correct_v1._load = lambda tx, command: (inputs, None, None)
    v1_groups: list[tuple[str, ...]] = []

    async def fake_group(command, block_ids):
        v1_groups.append(tuple(block_ids))
        return len(block_ids)

    correct_v1._group = fake_group
    res_v1 = await correct_v1(SimpleNamespace(section_id=section.id))
    assert v1_groups == [(b1.id, b2.id), (b3.id, b4.id)]
    assert res_v1.checked_blocks == 4

    missing_v2 = [(b1.id, b2.id, b3.id, b4.id), ()]
    monkeypatch.setattr(module, "model_correction_blocks", lambda tx, inp: missing_v2.pop(0))
    monkeypatch.setattr(module, "current_correction_candidates", lambda tx, inp, states=None: ())
    correct_v2 = CorrectChapterV2(
        unit_of_work=SimpleNamespace(transaction=lambda: nullcontext(None)),
        llm_client=None,
        snapshot=snapshot,
        validator=None,
    )
    correct_v2._load = lambda tx, command: (inputs, None, None)
    v2_propose_groups: list[tuple[str, ...]] = []

    async def fake_propose(command, block_ids):
        v2_propose_groups.append(tuple(block_ids))
        return len(block_ids)

    correct_v2._propose_group = fake_propose
    res_v2 = await correct_v2(SimpleNamespace(section_id=section.id))
    assert v2_propose_groups == [(b1.id, b2.id), (b3.id, b4.id)]
    assert res_v2.checked_blocks == 4

    pending_candidates = [
        (
            SimpleNamespace(candidate_id=f"c-{idx}", block_id=b.id, group_id="g-1")
            for idx, b in enumerate(section.blocks, start=1)
        ),
        (),
    ]
    monkeypatch.setattr(module, "model_correction_blocks", lambda tx, inp: ())
    monkeypatch.setattr(
        module,
        "current_correction_candidates",
        lambda tx, inp, states=None: tuple(pending_candidates.pop(0)),
    )
    v2_review_groups: list[tuple[str, ...]] = []

    async def fake_review(command, candidate_ids):
        v2_review_groups.append(tuple(candidate_ids))
        return len(candidate_ids)

    correct_v2._review_group = fake_review
    res_v2_review = await correct_v2(SimpleNamespace(section_id=section.id))
    assert v2_review_groups == [("c-1", "c-2"), ("c-3", "c-4")]
    assert res_v2_review.checked_blocks == 4


def test_request_filters_chapter_context_for_chunk_without_changing_checksums():
    from linguary.domain import ChapterAnalysis, TreatmentPlan
    from linguary.domain.content_plan import (
        ContentPlan, ContextFact, Difficulty, RequiredPoint, SourceAnchor,
    )

    section = make_analysis_section(
        texts=("Block one.", "Block two.", "Block three.", "Block four.", "Block five.", "Block six."),
    )
    doc = make_analysis_document((section,))
    b1, b2, b3, b4, b5, b6 = section.blocks

    plan = TreatmentPlan.of_treatments(
        section.id,
        tuple((b.id, BlockTreatment.TRANSLATE) for b in section.blocks),
    )
    content_plan = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(
            Difficulty(
                id="d1",
                anchors=(SourceAnchor(block_id=b1.id, start=0, end=5, quote="Block"),),
                owner_block_id=b1.id,
                kind="idiom",
                priority="critical",
                help_kind="explanation",
                required_points=(RequiredPoint(id="p1", description="meaning 1"),),
            ),
            Difficulty(
                id="d3",
                anchors=(SourceAnchor(block_id=b3.id, start=0, end=5, quote="Block"),),
                owner_block_id=b3.id,
                kind="idiom",
                priority="critical",
                help_kind="explanation",
                required_points=(RequiredPoint(id="p3", description="meaning 3"),),
            ),
            Difficulty(
                id="d6",
                anchors=(SourceAnchor(block_id=b6.id, start=0, end=5, quote="Block"),),
                owner_block_id=b6.id,
                kind="idiom",
                priority="critical",
                help_kind="explanation",
                required_points=(RequiredPoint(id="p6", description="meaning 6"),),
            ),
        ),
        context_facts=(
            ContextFact(
                id="cf1",
                kind="speaker",
                anchors=(SourceAnchor(block_id=b1.id, start=0, end=5, quote="Block"),),
                subject_anchor=SourceAnchor(block_id=b1.id, start=6, end=9, quote="one"),
                value="Speaker 1",
                certainty="explicit",
            ),
            ContextFact(
                id="cf3",
                kind="speaker",
                anchors=(SourceAnchor(block_id=b3.id, start=0, end=5, quote="Block"),),
                subject_anchor=SourceAnchor(block_id=b3.id, start=6, end=11, quote="three"),
                value="Speaker 3",
                certainty="explicit",
            ),
            ContextFact(
                id="cf6",
                kind="speaker",
                anchors=(SourceAnchor(block_id=b6.id, start=0, end=5, quote="Block"),),
                subject_anchor=SourceAnchor(block_id=b6.id, start=6, end=9, quote="six"),
                value="Speaker 6",
                certainty="explicit",
            ),
        ),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="Keep tone.",
        chapter_summary="Summary.",
        expected_block_ids=tuple(b.id for b in section.blocks),
        content_plan=content_plan,
    )
    results = {
        b.id: BlockProcessingResult(
            block_id=b.id,
            treatment=BlockTreatment.TRANSLATE,
            translated_text=f"Перевод {idx}.",
        )
        for idx, b in enumerate(section.blocks, start=1)
    }
    inputs = CorrectionInputs(
        document=doc,
        section=section,
        analysis=analysis,
        run=SimpleNamespace(id="run", glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(payload={"schema_version": 1}),
        snapshot=SimpleNamespace(to_payload=lambda: {"model": "correction"}),
        blocks=section.blocks,
        results=results,
        bindings={b.id: None for b in section.blocks},
    )
    checksums_before = dict(inputs.checksums)
    context_sha_before = inputs.context_sha256
    chapter_blocks_before = list(inputs.context["chapter_blocks"])

    snapshot = SimpleNamespace(
        prompt="Correct the chapter.",
        parameters={"max_attempts": 1, "max_output_tokens": None},
    )
    correct = CorrectChapter(unit_of_work=None, llm_client=None, snapshot=snapshot, validator=None)
    task = SimpleNamespace(book_id=doc.id, id="task", scope_id=section.id)

    request = correct._request(inputs, task, (b1.id,))
    body = json.loads(request.input_text)

    assert [cb["block_id"] for cb in body["chapter_blocks"]] == ["b1", "b2", "b3"]
    assert [item["block_id"] for item in body["chapter_analysis"]["treatment_plan"]["items"]] == [
        "b1", "b2", "b3"
    ]
    assert [d["id"] for d in body["chapter_analysis"]["content_plan"]["difficulties"]] == [
        "d1", "d3"
    ]
    assert [f["id"] for f in body["chapter_analysis"]["content_plan"]["context_facts"]] == [
        "cf1", "cf3"
    ]
    assert inputs.checksums == checksums_before
    assert inputs.context_sha256 == context_sha_before
    assert inputs.context["chapter_blocks"] == chapter_blocks_before



