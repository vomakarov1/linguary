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
    assert 'original' not in json.loads(request.input_text)['blocks'][0]
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
    assert 'original' not in body['blocks'][0]
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


def test_correction_requests_place_static_chapter_prefix_before_blocks_and_candidates():
    from dataclasses import asdict
    from linguary.application.use_cases.correct_chapter import CorrectChapterV2
    from linguary.domain import ChapterAnalysis, TreatmentPlan
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
        translation_instructions="Keep tone.",
        chapter_summary="Summary.",
        expected_block_ids=(b1.id, b2.id),
    )
    res1 = BlockProcessingResult(
        block_id=b1.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Первый блок.",
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
        results={b1.id: res1, b2.id: res2},
        bindings={b1.id: None, b2.id: None},
    )
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
        "current_result": asdict(res1),
        "result": asdict(replace(res1, translated_text="Первый блок уточнён.")),
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
    task = SimpleNamespace(book_id=doc.id, id="task", scope_id=section.id)
    correct = CorrectChapter(unit_of_work=None, llm_client=None, snapshot=snapshot, validator=None)
    correct_v2 = CorrectChapterV2(unit_of_work=None, llm_client=None, snapshot=snapshot, validator=None)

    propose_req = correct._request(inputs, task, (b1.id, b2.id))
    review_req = correct_v2._review_request(inputs, task, (candidate,))

    expected_prefix_keys = [
        "schema_version",
        "book_id",
        "section_id",
        "source_language",
        "target_language",
        "run_glossary",
        "chapter_analysis",
        "chapter_blocks",
    ]
    assert list(json.loads(propose_req.input_text).keys()) == [*expected_prefix_keys, "blocks"]
    assert list(json.loads(review_req.input_text).keys()) == [*expected_prefix_keys, "candidates"]
    assert propose_req.input_text.index('"chapter_blocks":') < propose_req.input_text.index('"blocks":')
    assert review_req.input_text.index('"chapter_blocks":') < review_req.input_text.index('"candidates":')


def test_llm_chapter_analysis_and_quality_artifact_strip_internal_ids_and_char_offsets_while_keeping_contracts():
    from dataclasses import asdict
    from linguary.application.quality_context import _llm_artifact, request_text
    from linguary.application.use_cases.correct_chapter import _llm_chapter_analysis
    from linguary.composition.correction_runtime import CORRECTION_PROMPT, CORRECTION_PROMPT_V2
    from linguary.domain import ChapterAnalysis, LanguageNote, TreatmentPlan
    from linguary.domain.content_plan import (
        ContentPlan,
        ContextFact,
        Difficulty,
        RequiredPoint,
        SourceAnchor,
        canonical_sha,
    )
    from linguary.domain.learning import FragmentAssistance

    assert "условиям книги" not in CORRECTION_PROMPT
    assert "условия книги" not in CORRECTION_PROMPT
    assert "условиям книги" not in CORRECTION_PROMPT_V2
    assert "условия книги" not in CORRECTION_PROMPT_V2

    section = make_analysis_section(texts=("First sentence.", "Second sentence he said."))
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
        id="analysis-internal-id",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="Keep tone.",
        chapter_summary="Summary.",
        expected_block_ids=(b1.id, b2.id),
        content_plan=content_plan,
    )
    aliases = {b1.id: "b1", b2.id: "b2"}
    llm_analysis = _llm_chapter_analysis(asdict(analysis), aliases)
    assert llm_analysis is not None
    assert "id" not in llm_analysis
    assert "section_id" not in llm_analysis
    assert "coverage_status" not in llm_analysis
    assert "section_id" not in llm_analysis["treatment_plan"]
    llm_cp = llm_analysis["content_plan"]
    assert "version" not in llm_cp
    assert "profile" not in llm_cp
    assert "sha256" not in llm_cp
    assert llm_cp["difficulties"][0]["anchors"] == [{"block_id": "b1", "quote": "First"}]
    assert llm_cp["context_facts"][0]["anchors"] == [{"block_id": "b2", "quote": "Second"}]
    assert llm_cp["context_facts"][0]["subject_anchor"] == {"block_id": "b2", "quote": "he"}

    result = BlockProcessingResult(
        block_id=b1.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text=" Первое предложение.",
        language_notes=(
            LanguageNote(
                id="ln-ch01-001",
                block_id=b1.id,
                original_excerpt="First",
                note="Заметка",
            ),
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
    orig_env = {"artifact_kind": "original", "block_ids": [b1.id], "body": {"text": b1.text}}
    analysis_env = {"artifact_kind": "analysis", "block_ids": [b1.id, b2.id], "body": asdict(analysis)}
    result_env = {"artifact_kind": "result", "block_ids": [b1.id], "body": asdict(result)}
    manifest = {
        "schema_version": 1,
        "book_id": "book-1",
        "book_manifest_sha": "a" * 64,
        "manifest_sha": "b" * 64,
        "scope": [b1.id],
        "artifacts": [
            {**orig_env, "artifact_sha": canonical_sha(orig_env)},
            {**analysis_env, "artifact_sha": canonical_sha(analysis_env)},
            {**result_env, "artifact_sha": canonical_sha(result_env)},
        ],
    }
    parsed = json.loads(request_text(manifest))
    assert "manifest_sha" not in parsed
    assert "book_manifest_sha" not in parsed
    assert parsed["scope"] == ["b1"]
    assert parsed["artifacts"][0]["body"] == {"text": b1.text}
    assert "artifact_sha" not in parsed["artifacts"][0]

    qual_analysis = parsed["artifacts"][1]["body"]
    assert parsed["artifacts"][1]["block_ids"] == ["b1"]
    assert "id" not in qual_analysis
    assert "section_id" not in qual_analysis
    assert "coverage_status" not in qual_analysis
    assert "section_id" not in qual_analysis["treatment_plan"]
    assert [item["block_id"] for item in qual_analysis["treatment_plan"]["items"]] == ["b1"]
    assert "version" not in qual_analysis["content_plan"]
    assert "profile" not in qual_analysis["content_plan"]
    assert "sha256" not in qual_analysis["content_plan"]
    assert qual_analysis["content_plan"]["difficulties"][0]["anchors"] == [
        {"block_id": "b1", "quote": "First"}
    ]
    assert qual_analysis["content_plan"]["context_facts"] == []

    full_scope_parsed = json.loads(request_text({**manifest, "scope": [b1.id, b2.id]}))
    assert full_scope_parsed["scope"] == ["b1", "b2"]
    assert full_scope_parsed["artifacts"][1]["block_ids"] == ["b1", "b2"]
    full_qual_analysis = full_scope_parsed["artifacts"][1]["body"]
    assert [item["block_id"] for item in full_qual_analysis["treatment_plan"]["items"]] == [
        "b1",
        "b2",
    ]
    assert full_qual_analysis["content_plan"]["difficulties"][0]["anchors"] == [
        {"block_id": "b1", "quote": "First"}
    ]
    assert full_qual_analysis["content_plan"]["context_facts"][0]["anchors"] == [
        {"block_id": "b2", "quote": "Second"}
    ]
    assert full_qual_analysis["content_plan"]["context_facts"][0]["subject_anchor"] == {
        "block_id": "b2",
        "quote": "he",
    }

    qual_result = _llm_artifact(manifest["artifacts"][2], aliases)["body"]
    assert "block_id" not in qual_result
    assert "id" not in qual_result["language_notes"][0]
    assert qual_result["fragment_assistances"][0]["anchors"] == [
        {"block_id": "b1", "quote": "First"}
    ]


def test_correct_chapter_v2_populates_omitted_proposal_sha256_from_candidate_ref():
    from linguary.adapters.llm.correction_payloads import (
        ChapterCorrectionReviewSchema,
        CorrectionVerdictSchema,
    )
    from linguary.adapters.llm.json_schema import strict_json_schema
    from linguary.adapters.llm.payloads import parse_payload
    from linguary.application.correction_dto import (
        ChapterCorrectionReviewResponse,
        CorrectionVerdict,
    )
    from linguary.application.llm_dto import LLMPayloadError

    review_schema = strict_json_schema(ChapterCorrectionReviewSchema)
    verdict_def = review_schema["$defs"]["CorrectionVerdictSchema"]
    assert "proposal_sha256" not in verdict_def["properties"]
    assert "proposal_sha256" not in verdict_def.get("required", [])

    direct_verdict_schema = strict_json_schema(CorrectionVerdictSchema)
    assert "proposal_sha256" not in direct_verdict_schema["properties"]
    assert "proposal_sha256" not in direct_verdict_schema.get("required", [])

    parsed = parse_payload(
        LLMResponseKind.CHAPTER_CORRECTION_REVIEW,
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "cand-1",
                        "verdict": "confirmed",
                        "error_supported": True,
                        "evidence_valid": True,
                        "minimal": True,
                        "no_semantic_loss": True,
                        "reason": "Подтверждено оригиналом.",
                    }
                ]
            }
        ),
    )
    assert isinstance(parsed, ChapterCorrectionReviewResponse)
    assert parsed.candidates[0].proposal_sha256 is None

    expected_sha = "a" * 64
    refs = (("cand-1", expected_sha),)
    resolved = parsed.with_resolved_proposal_shas(refs)
    assert resolved.candidates[0].proposal_sha256 == expected_sha
    resolved.validate_coverage(refs)

    mismatched = ChapterCorrectionReviewResponse(
        candidates=(
            CorrectionVerdict(
                candidate_id="cand-1",
                proposal_sha256="b" * 64,
                verdict="confirmed",
                error_supported=True,
                evidence_valid=True,
                minimal=True,
                no_semantic_loss=True,
                reason="Неверный хеш.",
            ),
        )
    )
    kept_mismatch = mismatched.with_resolved_proposal_shas(refs)
    assert kept_mismatch.candidates[0].proposal_sha256 == "b" * 64
    with pytest.raises(LLMPayloadError):
        kept_mismatch.validate_coverage(refs)


def test_reconcile_corrections_with_section_id_scopes_work_to_target_chapter():
    from linguary.application.correction import (
        correction_coverage_complete,
        reconcile_corrections,
    )
    from linguary.application.correction_dto import (
        CORRECTION_VERSIONS_V2,
        CorrectionSnapshotV2,
    )
    from linguary.domain import ChapterAnalysis, TreatmentPlan

    s1 = make_analysis_section(section_id="section-1", order=0, texts=("Chapter 1 text.",))
    s2 = make_analysis_section(section_id="section-2", order=1, texts=("Chapter 2 text.",))
    document = make_analysis_document((s1, s2))
    b1 = s1.blocks[0]
    b2 = s2.blocks[0]

    snapshot = CorrectionSnapshotV2(
        model="gpt-test",
        route={"provider": "openai_compatible", "base_url": "https://example.test/v1", "proxy": "", "route_id": "r1"},
        parameters={
            "temperature": 0.1,
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
    run = SimpleNamespace(id="run-1", book_id=document.id, glossary_snapshot=GlossarySnapshot())
    conditions = SimpleNamespace(id="cond-1", payload={"schema_version": 1})

    analyses = {
        s.id: ChapterAnalysis.create(
            id=f"analysis-{s.id}",
            section_id=s.id,
            treatment_plan=TreatmentPlan.of_treatments(
                s.id, ((s.blocks[0].id, BlockTreatment.TRANSLATE),)
            ),
            translation_instructions="",
            chapter_summary="",
            expected_block_ids=(s.blocks[0].id,),
        )
        for s in (s1, s2)
    }
    queried_sections: list[str] = []
    added_tasks: list[object] = []

    class FakeResults:
        def list_for_section(self, run_id: str, section_id: str):
            queried_sections.append(section_id)
            if section_id == "section-1":
                return (
                    BlockProcessingResult(
                        block_id=b1.id,
                        treatment=BlockTreatment.TRANSLATE,
                        translated_text="Перевод главы 1.",
                    ),
                )
            return ()

    class FakeWork:
        def get_binding(self, kind: str, entity_id: str):
            if kind == "run" and entity_id == run.id:
                return {
                    "book_id": document.id,
                    "conditions_id": conditions.id,
                    "payload_json": {"correction": snapshot.to_payload()},
                }
            return None

        def latest_conditions(self, book_id: str):
            return conditions

        def result_bindings(self, run_id: str):
            return {
                b1.id: {
                    "book_id": document.id,
                    "conditions_id": conditions.id,
                    "attempt_id": "att-1",
                    "artifact_revision_id": "art-1",
                }
            }

        def correction_input_generations(self, run_id: str):
            return {b1.id: 1, b2.id: 0}

        def observe_correction_input(self, run_id: str, block_id: str, fingerprint: str) -> int:
            if block_id != b1.id:
                raise AssertionError(f"observe_correction_input called for non-target block {block_id}")
            return 1

        def list_correction_candidates(self, run_id: str, *, section_id: str | None = None, states=None):
            if section_id != "section-1":
                raise AssertionError(f"list_correction_candidates called for {section_id}")
            return ()

        def correction_checks(self, run_id: str, section_id: str):
            if section_id != "section-1":
                raise AssertionError(f"correction_checks called for {section_id}")
            return ()

        def bind(self, kind: str, entity_id: str, **kwargs):
            pass

    tx = SimpleNamespace(
        runs=SimpleNamespace(active_for_book=lambda _bid: run),
        documents=SimpleNamespace(get=lambda _bid: document),
        analyses=SimpleNamespace(
            get_for_section=lambda _bid, sid: analyses[sid],
            list_for_book=lambda _bid: tuple(analyses.values()),
        ),
        results=FakeResults(),
        work=FakeWork(),
        tasks=SimpleNamespace(
            list_for_book=lambda _bid: (),
            find_by_identity=lambda _id: None,
            add=lambda task: added_tasks.append(task),
        ),
        books=SimpleNamespace(get=lambda _bid: None),
    )

    pending = reconcile_corrections(tx, document.id, section_id="section-1")
    assert len(pending) == 1
    assert pending[0].scope_id == "section-1"
    assert queried_sections == ["section-1"]

    queried_sections.clear()
    assert correction_coverage_complete(tx, document.id) is False
    assert queried_sections == []


def test_correct_chapter_omits_duplicate_original_and_aliases_plan_ids():
    from dataclasses import asdict
    from linguary.application.use_cases.correct_chapter import CorrectChapter, CorrectChapterV2
    from linguary.domain import ChapterAnalysis, TreatmentPlan
    from linguary.domain.content_plan import (
        ContentPlan,
        ContextFact,
        Difficulty,
        RequiredPoint,
        SourceAnchor,
        canonical_sha,
    )
    from linguary.domain.correction import CorrectionCandidate
    from linguary.domain.learning import FragmentAssistance

    section = make_analysis_section(
        texts=("First sentence with idiom.", "Second sentence he spoke."),
    )
    doc = make_analysis_document((section,))
    b1, b2 = section.blocks

    long_diff_id = "df-" + "a" * 64
    long_fact_id = "cf-" + "b" * 64

    plan = TreatmentPlan.of_treatments(
        section.id,
        ((b1.id, BlockTreatment.TRANSLATE), (b2.id, BlockTreatment.TRANSLATE)),
    )
    content_plan = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(
            Difficulty(
                id=long_diff_id,
                anchors=(SourceAnchor(block_id=b1.id, start=0, end=5, quote="First"),),
                owner_block_id=b1.id,
                kind="idiom",
                priority="critical",
                help_kind="explanation",
                required_points=(RequiredPoint(id="p1", description="meaning"),),
            ),
        ),
        context_facts=(
            ContextFact(
                id=long_fact_id,
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
        content_plan=content_plan,
    )
    res1 = BlockProcessingResult(
        block_id=b1.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Первое предложение.",
        fragment_assistances=(
            FragmentAssistance(
                difficulty_id=long_diff_id,
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
        translated_text="Второе предложение.",
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
    snapshot = SimpleNamespace(
        prompt="Correct the chapter.",
        review_prompt="Review proposals.",
        parameters={"max_attempts": 1, "max_output_tokens": None},
    )
    task = SimpleNamespace(book_id=doc.id, id="task", scope_id=section.id)
    correct = CorrectChapter(unit_of_work=None, llm_client=None, snapshot=snapshot, validator=None)
    correct_v2 = CorrectChapterV2(unit_of_work=None, llm_client=None, snapshot=snapshot, validator=None)

    request = correct._request(inputs, task, (b1.id, b2.id))
    body = json.loads(request.input_text)

    assert "original" not in body["blocks"][0]
    assert "original" not in body["blocks"][1]
    assert body["chapter_analysis"]["content_plan"]["difficulties"][0]["id"] == "d1"
    assert body["chapter_analysis"]["content_plan"]["context_facts"][0]["id"] == "f1"
    assert body["blocks"][0]["current_result"]["fragment_assistances"][0]["difficulty_id"] == "d1"
    assert long_diff_id not in request.input_text
    assert long_fact_id not in request.input_text

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
        "current_result": asdict(res1),
        "result": asdict(replace(res1, translated_text="Первое предложение уточнено.")),
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
    review_req = correct_v2._review_request(inputs, task, (candidate,))
    review_body = json.loads(review_req.input_text)
    assert review_body["candidates"][0]["candidate_id"] == "c1"
    assert review_body["candidates"][0]["current_result"]["fragment_assistances"][0]["difficulty_id"] == "d1"
    assert review_body["candidates"][0]["proposed_result"]["fragment_assistances"][0]["difficulty_id"] == "d1"
    assert long_diff_id not in review_req.input_text
    assert long_fact_id not in review_req.input_text


def test_section_correction_inputs_uses_batch_result_bindings_and_incremental_state_update(monkeypatch):
    import linguary.application.correction as correction_module
    from linguary.application.book_overview import _quality_projection
    from linguary.application.correction import (
        CorrectionInputs,
        _observe_correction_inputs,
        _section_correction_inputs,
    )
    from linguary.application.correction_dto import CORRECTION_VERSIONS_V2, CorrectionSnapshotV2
    from linguary.application.quality_context import QualityReviewResponse, snapshot as quality_snapshot, validate_response
    from linguary.domain import BookStatus, ChapterAnalysis, TreatmentPlan

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
    snapshot = CorrectionSnapshotV2(
        model="gpt-test",
        route={"provider": "openai_compatible", "base_url": "https://example.test/v1", "proxy": "", "route_id": "r1"},
        parameters={
            "temperature": 0.0,
            "max_output_tokens": 1000,
            "timeout_seconds": 30,
            "max_attempts": 1,
            "max_unit_source_chars": 8000,
            "input_price_per_mtok": 0,
            "output_price_per_mtok": 0,
        },
        prompt="Propose.",
        review_prompt="Review.",
        versions=CORRECTION_VERSIONS_V2,
    )
    res1 = BlockProcessingResult(block_id=b1.id, treatment=BlockTreatment.TRANSLATE, translated_text="Первый.")
    res2 = BlockProcessingResult(block_id=b2.id, treatment=BlockTreatment.TRANSLATE, translated_text="Второй.")
    binding1 = {"book_id": doc.id, "conditions_id": "cond-1", "attempt_id": "att-1", "artifact_revision_id": "art-1"}
    binding2 = {"book_id": doc.id, "conditions_id": "cond-1", "attempt_id": "att-2", "artifact_revision_id": "art-2"}
    run = SimpleNamespace(id="run-1", glossary_snapshot=GlossarySnapshot())
    conditions = SimpleNamespace(id="cond-1", revision=1, payload={"schema_version": 1, "versions": {"validators": 3}},
                                 pronunciation_help_enabled=False, pronunciation_variant=None, reproducibility="strict")

    batch_calls: list[str] = []

    class FakeWork:
        def result_bindings(self, run_id: str):
            batch_calls.append(run_id)
            return {b1.id: binding1, b2.id: binding2}

        def get_binding(self, kind: str, entity_id: str):
            if kind == "result":
                raise AssertionError("Per-block get_binding('result', ...) must not be called when result_bindings is available")
            return None

        def correction_input_generations(self, run_id: str):
            return {b1.id: 1, b2.id: 1}

        def observe_correction_input(self, run_id: str, block_id: str, fingerprint: str) -> int:
            return 2 if block_id == b1.id else 1

        def stale_correction_candidates(self, run_id: str, *, block_ids=None, candidate_ids=None):
            pass

    tx = SimpleNamespace(
        results=SimpleNamespace(list_for_section=lambda _rid, _sid: (res1, res2)),
        work=FakeWork(),
        analyses=SimpleNamespace(get_for_section=lambda _bid, _sid: analysis),
    )

    inputs = _section_correction_inputs(
        tx,
        document=doc,
        section=section,
        run=run,
        conditions=conditions,
        snapshot=snapshot,
    )
    assert batch_calls == ["run-1"]
    assert inputs.bindings == {b1.id: binding1, b2.id: binding2}

    payload_calls: list[str] = []
    orig_payload_fn = correction_module.original_block_payload

    def spy_payload(block):
        payload_calls.append(block.id)
        return orig_payload_fn(block)

    monkeypatch.setattr(correction_module, "original_block_payload", spy_payload)

    observed = _observe_correction_inputs(tx, inputs)
    assert observed.input_generations == {b1.id: 2, b2.id: 1}
    assert payload_calls == []

    updated_res1 = replace(res1, translated_text="Первый обновлённый.")
    updated_binding1 = {**binding1, "attempt_id": "att-1-new", "artifact_revision_id": "art-1-new"}
    updated = inputs.with_updated_state(
        updated_results=(updated_res1,),
        updated_bindings={b1.id: updated_binding1},
        input_generations={b1.id: 3, b2.id: 1},
    )
    assert payload_calls == [b1.id]
    assert updated.context is inputs.context
    assert updated.context_sha256 == inputs.context_sha256
    assert updated.checksums[b2.id] == inputs.checksums[b2.id]
    assert updated.checksums[b1.id] != inputs.checksums[b1.id]
    assert updated.bindings[b1.id] == updated_binding1
    assert updated.bindings[b2.id] == binding2
    assert updated.input_generations == {b1.id: 3, b2.id: 1}

    fresh = CorrectionInputs(
        document=doc,
        section=section,
        analysis=analysis,
        run=run,
        conditions=conditions,
        snapshot=snapshot,
        blocks=section.blocks,
        results={b1.id: updated_res1, b2.id: res2},
        bindings={b1.id: updated_binding1, b2.id: binding2},
        input_generations={b1.id: 3, b2.id: 1},
    )
    assert updated.checksums == fresh.checksums

    # Verify quality_context.snapshot single glossary read when run is None and multi-block finding artifact_sha
    glossary_calls = 0

    def count_glossary(_book_id: str):
        nonlocal glossary_calls
        glossary_calls += 1
        return ()

    q_tx = SimpleNamespace(
        documents=SimpleNamespace(get=lambda _bid: doc),
        work=SimpleNamespace(latest_conditions=lambda _bid: conditions),
        runs=SimpleNamespace(active_for_book=lambda _bid: None),
        results=SimpleNamespace(list_for_run=lambda _rid: ()),
        analyses=SimpleNamespace(list_for_book=lambda _bid: (analysis,)),
        glossary=SimpleNamespace(list_for_book=count_glossary),
    )
    manifest = quality_snapshot(q_tx, doc.id, "all")
    assert glossary_calls == 1
    orig_b2_sha = next(
        a["artifact_sha"]
        for a in manifest["artifacts"]
        if a["artifact_kind"] == "original" and a["block_ids"] == [b2.id]
    )
    validated_qr = validate_response(
        manifest,
        QualityReviewResponse(
            findings=(
                {
                    "artifact_kind": "original",
                    "artifact_sha": orig_b2_sha,
                    "block_ids": ["b1", "b2"],
                    "evidence_anchors": [{"block_id": "b2", "quote": "Second"}],
                    "problem": "Issue across blocks",
                },
            ),
            checked_artifacts=(),
            missing_artifacts=(),
        ),
    )
    assert validated_qr.findings[0]["artifact_sha"] == orig_b2_sha

    # Verify book_overview._quality_projection checks correction_coverage_complete
    monkeypatch.setattr(correction_module, "correction_coverage_complete", lambda _tx, _bid: False)
    book_record = SimpleNamespace(id=doc.id, status=BookStatus.READY_TO_EXPORT, source_language="en")
    overview_tx = SimpleNamespace(
        work=SimpleNamespace(
            latest_conditions=lambda _bid: conditions,
            list_roots=lambda _bid, run_id=None: (),
            list_reviews=lambda _bid: (),
        ),
        results=SimpleNamespace(list_for_run=lambda _rid: (res1, res2)),
        llm_calls=SimpleNamespace(list_for_book=lambda _bid: ()),
    )
    projection = _quality_projection(
        overview_tx,
        book_record,
        doc,
        (analysis,),
        run,
        (),
        stored_results=(res1, res2),
        stored_calls=(),
    )
    assert projection["epub_ready"] is False


def test_quality_projection_reuses_preloaded_inputs_in_correction_coverage_complete():
    from uuid import uuid4
    from linguary.application.book_overview import _quality_projection
    from linguary.application.correction import CorrectionInputs
    from linguary.application.correction_dto import CORRECTION_VERSIONS_V2, CorrectionSnapshotV2
    from linguary.domain import BookStatus, ChapterAnalysis, TreatmentPlan
    from linguary.domain.correction import CorrectionCheck
    from linguary.domain.recovery import CallAttempt, RecoveryRoot, RecoveryScope

    sections = tuple(
        make_analysis_section(
            section_id=f"00000000-0000-0000-0000-00000000010{idx}",
            title=f"Chapter {idx}",
            order=idx - 1,
            texts=(f"Chapter {idx} line one.", f"Chapter {idx} line two."),
        )
        for idx in (1, 2)
    )
    doc = make_analysis_document(sections)
    snapshot = CorrectionSnapshotV2(
        model="gpt-test",
        route={"provider": "openai_compatible", "base_url": "https://example.test/v1", "proxy": "", "route_id": "r1"},
        parameters={
            "temperature": 0.0,
            "max_output_tokens": 1000,
            "timeout_seconds": 30,
            "max_attempts": 1,
            "max_unit_source_chars": 8000,
            "input_price_per_mtok": 0,
            "output_price_per_mtok": 0,
        },
        prompt="Propose.",
        review_prompt="Review.",
        versions=CORRECTION_VERSIONS_V2,
    )
    cond_id = "00000000-0000-0000-0000-000000000001"
    run_id = "00000000-0000-0000-0000-000000000002"
    conditions = SimpleNamespace(
        id=cond_id,
        revision=1,
        payload={"schema_version": 1, "versions": {"validators": 3}},
        pronunciation_help_enabled=False,
        pronunciation_variant=None,
        reproducibility="strict",
    )
    run = SimpleNamespace(id=run_id, glossary_snapshot=GlossarySnapshot())
    book_record = SimpleNamespace(id=doc.id, status=BookStatus.READY_TO_EXPORT, source_language="en")

    analyses = []
    all_results = []
    bindings_by_block = {}
    generations_by_block = {}
    attempts_by_id = {}
    checks_by_section = {}
    correction_groups = {}

    for section in sections:
        block_ids = tuple(b.id for b in section.ordered_blocks)
        plan = TreatmentPlan.of_treatments(
            section.id,
            tuple((b_id, BlockTreatment.TRANSLATE) for b_id in block_ids),
        )
        analysis = ChapterAnalysis.create(
            id=str(uuid4()),
            section_id=section.id,
            treatment_plan=plan,
            translation_instructions="",
            chapter_summary="",
            expected_block_ids=block_ids,
        )
        analyses.append(analysis)
        sec_results = {}
        sec_bindings = {}
        for b in section.ordered_blocks:
            res = BlockProcessingResult(
                block_id=b.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text=f"Перевод {b.id}.",
            )
            all_results.append(res)
            sec_results[b.id] = res
            binding = {
                "book_id": doc.id,
                "conditions_id": conditions.id,
                "attempt_id": str(uuid4()),
                "artifact_revision_id": str(uuid4()),
            }
            bindings_by_block[b.id] = binding
            sec_bindings[b.id] = binding
            generations_by_block[b.id] = 1
        sec_inputs = CorrectionInputs(
            document=doc,
            section=section,
            analysis=analysis,
            run=run,
            conditions=conditions,
            snapshot=snapshot,
            blocks=section.ordered_blocks,
            results=sec_results,
            bindings=sec_bindings,
            input_generations={b.id: 1 for b in section.ordered_blocks},
        )
        corr_attempt_id = str(uuid4())
        attempts_by_id[corr_attempt_id] = CallAttempt(
            id=corr_attempt_id,
            book_id=doc.id,
            state="accepted",
            generation=1,
            ordinal=1,
            conditions_id=conditions.id,
            run_id=run.id,
            context={
                "schema_version": 1,
                "stage": "chapter_correction",
                "ordered_block_ids": list(block_ids),
            },
        )
        correction_groups[corr_attempt_id] = {
            "payload_json": {
                "unchanged_block_ids": list(block_ids),
                "manifest": sec_inputs.manifest,
            }
        }
        checks_by_section[section.id] = tuple(
            CorrectionCheck(
                run_id=run.id,
                block_id=b.id,
                section_id=section.id,
                input_sha256=sec_inputs.checksums[b.id],
                attempt_id=corr_attempt_id,
                basis="unchanged",
            )
            for b in section.ordered_blocks
        )

    all_block_ids = tuple(r.block_id for r in all_results)
    root = RecoveryRoot(
        id=str(uuid4()),
        book_id=doc.id,
        conditions_id=conditions.id,
        root_key="rk-1",
        manifest={
            "schema_version": 1,
            "ordered_blocks": list(all_block_ids),
            "atomic_intervals": [[b_id] for b_id in all_block_ids],
            "policy": {"max_calls": 10, "max_scope_calls": 3},
        },
        status="complete",
    )
    scope = RecoveryScope(
        id=str(uuid4()),
        root_id=root.id,
        ordered_blocks=all_block_ids,
        status="accepted",
        accepted_attempt_id=str(uuid4()),
    )

    counts = {
        "latest_conditions": 0,
        "result_bindings": 0,
        "list_for_run": 0,
    }

    class GuardedDocuments:
        def get(self, _book_id: str):
            raise AssertionError("documents.get must not be called when document is preloaded")

    class GuardedAnalyses:
        def list_for_book(self, _book_id: str):
            raise AssertionError("analyses.list_for_book must not be called when analyses are preloaded")

        def get_for_section(self, _book_id: str, _section_id: str):
            raise AssertionError("analyses.get_for_section must not be called when analyses are preloaded")

    class GuardedRuns:
        def active_for_book(self, _book_id: str):
            raise AssertionError("runs.active_for_book must not be called when run is preloaded")

    class GuardedResults:
        def block_ids_for_run(self, _run_id: str):
            return frozenset(all_block_ids)

        def list_for_run(self, _run_id: str):
            counts["list_for_run"] += 1
            return tuple(all_results)

        def list_for_section(self, _run_id: str, _section_id: str):
            raise AssertionError("results.list_for_section must not be called when run_results is preloaded")

    class GuardedWork:
        def latest_conditions(self, _book_id: str):
            counts["latest_conditions"] += 1
            return conditions

        def list_roots(self, _book_id: str, run_id=None):
            return (root,)

        def list_scopes_for_roots(self, _root_ids):
            return {root.id: (scope,)}

        def list_attempts_for_roots(self, _root_ids):
            return {root.id: ()}

        def result_bindings(self, _run_id: str):
            counts["result_bindings"] += 1
            return dict(bindings_by_block)

        def root_coverage_complete(self, _root_id: str, **_kwargs):
            return True

        def get_binding(self, kind: str, entity_id: str):
            if kind == "run" and entity_id == run.id:
                return {
                    "book_id": doc.id,
                    "conditions_id": conditions.id,
                    "payload_json": {"correction": snapshot.to_payload()},
                }
            if kind == "correction_group":
                return correction_groups.get(entity_id)
            if kind == "result":
                raise AssertionError("per-block get_binding('result', ...) must not be called")
            return None

        def correction_input_generations(self, _run_id: str):
            return dict(generations_by_block)

        def correction_checks(self, _run_id: str, section_id: str):
            return checks_by_section.get(section_id, ())

        def get_attempt(self, attempt_id: str):
            return attempts_by_id.get(attempt_id)

        def list_correction_candidates(self, _run_id: str, *, section_id=None, states=None):
            return ()

        def list_reviews(self, _book_id: str):
            return ()

    tx = SimpleNamespace(
        documents=GuardedDocuments(),
        analyses=GuardedAnalyses(),
        runs=GuardedRuns(),
        results=GuardedResults(),
        work=GuardedWork(),
        llm_calls=SimpleNamespace(list_for_book=lambda _bid: ()),
    )

    # Case 1: book_overview path (stored_results=None -> loads list_for_run once, 0 list_for_section)
    projection_overview = _quality_projection(
        tx,
        book_record,
        doc,
        tuple(analyses),
        run,
        (),
        stored_results=None,
        stored_calls=(),
    )
    assert projection_overview["epub_ready"] is True
    assert counts == {"latest_conditions": 1, "result_bindings": 1, "list_for_run": 1}

    # Case 2: book_result path (stored_results provided -> 0 list_for_run, 0 list_for_section)
    counts = {"latest_conditions": 0, "result_bindings": 0, "list_for_run": 0}
    projection_result = _quality_projection(
        tx,
        book_record,
        doc,
        tuple(analyses),
        run,
        (),
        stored_results=tuple(all_results),
        stored_calls=(),
    )
    assert projection_result["epub_ready"] is True
    assert counts == {"latest_conditions": 1, "result_bindings": 1, "list_for_run": 0}

    # Case 3: missing check on second chapter -> epub_ready is False without extra queries
    checks_by_section[sections[1].id] = ()
    counts = {"latest_conditions": 0, "result_bindings": 0, "list_for_run": 0}
    projection_incomplete = _quality_projection(
        tx,
        book_record,
        doc,
        tuple(analyses),
        run,
        (),
        stored_results=tuple(all_results),
        stored_calls=(),
    )
    assert projection_incomplete["epub_ready"] is False
    assert counts == {"latest_conditions": 1, "result_bindings": 1, "list_for_run": 0}


def test_is_pure_read_as_is_skips_english_help_and_byr_covered_blocks():
    from linguary.application.correction import _is_pure_read_as_is, unchecked_blocks
    from linguary.domain import (
        BeforeYouReadEntry,
        ChapterAnalysis,
        GlossaryHint,
        LanguageNote,
        TreatmentPlan,
    )
    from linguary.domain.content_plan import (
        ContentPlan,
        Difficulty,
        HelpEvidence,
        RequiredPoint,
        SourceAnchor,
        artifact_key,
    )
    from linguary.domain.learning import FragmentAssistance

    class FakeTx:
        class work:
            @staticmethod
            def correction_checks(run_id, section_id):
                return ()

    # 1. english_help=True: READ_AS_IS without translation/hints/notes is pure even with fragment_assistances & difficulties
    eh_section = make_analysis_section(
        texts=(
            "Echo. The cat sat.",
            "Second original block with note.",
            "Third original block with hint.",
            "Fourth original block with translation.",
            "Fifth translated block.",
        )
    )
    eh_doc = make_analysis_document((eh_section,))
    eh_b1, eh_b2, eh_b3, eh_b4, eh_b5 = eh_section.blocks
    eh_anchor = SourceAnchor(block_id=eh_b1.id, start=0, end=4, quote="Echo")
    eh_plan = TreatmentPlan.of_treatments(
        eh_section.id,
        (
            (eh_b1.id, BlockTreatment.READ_AS_IS),
            (eh_b2.id, BlockTreatment.READ_AS_IS),
            (eh_b3.id, BlockTreatment.READ_AS_IS),
            (eh_b4.id, BlockTreatment.READ_AS_IS),
            (eh_b5.id, BlockTreatment.TRANSLATE),
        ),
    )
    eh_content_plan = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(
            Difficulty(
                id="d-eh",
                anchors=(eh_anchor,),
                owner_block_id=eh_b1.id,
                kind="vocabulary",
                priority="critical",
                help_kind="lexical_hint",
                required_points=(RequiredPoint(id="p1", description="meaning"),),
            ),
        ),
    )
    eh_analysis = ChapterAnalysis.create(
        id="analysis-eh",
        section_id=eh_section.id,
        treatment_plan=eh_plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=tuple(b.id for b in eh_section.blocks),
        content_plan=eh_content_plan,
    )
    eh_results = {
        eh_b1.id: BlockProcessingResult(
            block_id=eh_b1.id,
            treatment=BlockTreatment.READ_AS_IS,
            fragment_assistances=(
                FragmentAssistance(
                    difficulty_id="d-eh",
                    point_ids=("p1",),
                    kind="lexical_hint",
                    text="a returning sound",
                    anchors=(eh_anchor,),
                ),
            ),
        ),
        eh_b2.id: BlockProcessingResult(
            block_id=eh_b2.id,
            treatment=BlockTreatment.READ_AS_IS,
            language_notes=(
                LanguageNote(
                    id="ln-ch01-001",
                    block_id=eh_b2.id,
                    original_excerpt="Second",
                    note="Заметка",
                ),
            ),
        ),
        eh_b3.id: BlockProcessingResult(
            block_id=eh_b3.id,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(GlossaryHint(original="Third", translation="Третий"),),
        ),
        eh_b4.id: BlockProcessingResult(
            block_id=eh_b4.id,
            treatment=BlockTreatment.TRANSLATE,
            translated_text="Четвёртый блок.",
        ),
        eh_b5.id: BlockProcessingResult(
            block_id=eh_b5.id,
            treatment=BlockTreatment.TRANSLATE,
            translated_text="Пятый блок.",
        ),
    }
    eh_inputs = CorrectionInputs(
        document=eh_doc,
        section=eh_section,
        analysis=eh_analysis,
        run=SimpleNamespace(id="run-eh", glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(payload={"schema_version": 1, "english_help": True}),
        snapshot=SimpleNamespace(to_payload=lambda: {"model": "correction"}),
        blocks=eh_section.blocks,
        results=eh_results,
        bindings={b.id: None for b in eh_section.blocks},
    )
    assert _is_pure_read_as_is(eh_b1, eh_inputs) is True
    assert _is_pure_read_as_is(eh_b2, eh_inputs) is False
    assert _is_pure_read_as_is(eh_b3, eh_inputs) is False
    assert _is_pure_read_as_is(eh_b4, eh_inputs) is False
    assert _is_pure_read_as_is(eh_b5, eh_inputs) is False
    assert unchecked_blocks(FakeTx(), eh_inputs) == (
        eh_b2.id,
        eh_b3.id,
        eh_b4.id,
        eh_b5.id,
    )

    # 2. english_help=False: BYR-covered owned difficulties and foreign-owner anchor overlaps are pure;
    # partially/uncovered owned difficulties are not pure.
    std_section = make_analysis_section(
        texts=(
            "Break a leg tonight.",
            "Hello there, friend.",
            "Spill the beans now.",
            "They carried on talking.",
        )
    )
    std_doc = make_analysis_document((std_section,))
    b_byr, b_foreign, b_partial, b_trans = std_section.blocks
    std_plan = TreatmentPlan.of_treatments(
        std_section.id,
        (
            (b_byr.id, BlockTreatment.READ_AS_IS),
            (b_foreign.id, BlockTreatment.READ_AS_IS),
            (b_partial.id, BlockTreatment.READ_AS_IS),
            (b_trans.id, BlockTreatment.TRANSLATE),
        ),
    )
    byr_entry_1 = BeforeYouReadEntry(expression="Break a leg", explanation="Пожелание удачи перед выступлением.")
    byr_entry_2 = BeforeYouReadEntry(expression="Spill the beans", explanation="Выдать секрет.")
    key_b_byr = artifact_key(
        "before_you_read.explanation",
        b_byr.id,
        {"expression": byr_entry_1.expression, "explanation": byr_entry_1.explanation},
    )
    key_b_partial = artifact_key(
        "before_you_read.explanation",
        b_partial.id,
        {"expression": byr_entry_2.expression, "explanation": byr_entry_2.explanation},
    )
    std_content_plan = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(
            Difficulty(
                id="d-byr",
                anchors=(SourceAnchor(block_id=b_byr.id, start=0, end=11, quote="Break a leg"),),
                owner_block_id=b_byr.id,
                kind="idiom",
                priority="critical",
                help_kind="explanation",
                required_points=(
                    RequiredPoint(id="p1", description="idiomatic meaning"),
                    RequiredPoint(id="p2", description="theatre usage"),
                ),
            ),
            Difficulty(
                id="d-foreign",
                anchors=(
                    SourceAnchor(block_id=b_foreign.id, start=0, end=11, quote="Hello there"),
                    SourceAnchor(block_id=b_trans.id, start=5, end=15, quote="carried on"),
                ),
                owner_block_id=b_trans.id,
                kind="semantic_relation",
                priority="useful",
                help_kind="explanation",
                required_points=(RequiredPoint(id="p1", description="continuity"),),
            ),
            Difficulty(
                id="d-partial",
                anchors=(SourceAnchor(block_id=b_partial.id, start=0, end=15, quote="Spill the beans"),),
                owner_block_id=b_partial.id,
                kind="idiom",
                priority="critical",
                help_kind="explanation",
                required_points=(
                    RequiredPoint(id="p1", description="literal image"),
                    RequiredPoint(id="p2", description="uncovered nuance"),
                ),
            ),
        ),
        before_you_read_evidence=(
            HelpEvidence(
                difficulty_id="d-byr",
                point_id="p1",
                artifact_kind="before_you_read.explanation",
                block_id=b_byr.id,
                artifact_key=key_b_byr,
                body_excerpt="Пожелание удачи",
            ),
            HelpEvidence(
                difficulty_id="d-byr",
                point_id="p2",
                artifact_kind="before_you_read.explanation",
                block_id=b_byr.id,
                artifact_key=key_b_byr,
                body_excerpt="перед выступлением",
            ),
            HelpEvidence(
                difficulty_id="d-partial",
                point_id="p1",
                artifact_kind="before_you_read.explanation",
                block_id=b_partial.id,
                artifact_key=key_b_partial,
                body_excerpt="Выдать секрет",
            ),
        ),
    )
    std_analysis = ChapterAnalysis.create(
        id="analysis-std",
        section_id=std_section.id,
        treatment_plan=std_plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=tuple(b.id for b in std_section.blocks),
        before_you_read=(byr_entry_1, byr_entry_2),
        content_plan=std_content_plan,
    )
    std_results = {
        b_byr.id: BlockProcessingResult(block_id=b_byr.id, treatment=BlockTreatment.READ_AS_IS),
        b_foreign.id: BlockProcessingResult(block_id=b_foreign.id, treatment=BlockTreatment.READ_AS_IS),
        b_partial.id: BlockProcessingResult(block_id=b_partial.id, treatment=BlockTreatment.READ_AS_IS),
        b_trans.id: BlockProcessingResult(
            block_id=b_trans.id,
            treatment=BlockTreatment.TRANSLATE,
            translated_text="Они продолжали разговаривать.",
        ),
    }
    std_inputs = CorrectionInputs(
        document=std_doc,
        section=std_section,
        analysis=std_analysis,
        run=SimpleNamespace(id="run-std", glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(payload={"schema_version": 1, "english_help": False}),
        snapshot=SimpleNamespace(to_payload=lambda: {"model": "correction"}),
        blocks=std_section.blocks,
        results=std_results,
        bindings={b.id: None for b in std_section.blocks},
    )
    assert _is_pure_read_as_is(b_byr, std_inputs) is True
    assert _is_pure_read_as_is(b_foreign, std_inputs) is True
    assert _is_pure_read_as_is(b_partial, std_inputs) is False
    assert _is_pure_read_as_is(b_trans, std_inputs) is False
    assert unchecked_blocks(FakeTx(), std_inputs) == (b_partial.id, b_trans.id)








