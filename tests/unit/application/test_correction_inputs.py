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
