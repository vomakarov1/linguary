"""Correction fingerprints preserve the immutable source block representation."""

from dataclasses import replace
import json
from types import SimpleNamespace

from analysis_fakes import make_analysis_document, make_analysis_section

from linguary.application.correction import CorrectionInputs
from linguary.application.use_cases.correct_chapter import CorrectChapter
from linguary.domain import BlockProcessingResult, BlockTreatment, GlossarySnapshot


def test_correction_fingerprints_include_source_attributes():
    section = make_analysis_section()
    block = replace(section.blocks[0], attributes={'lang': 'en'})
    section = replace(section, blocks=(block,))
    inputs = CorrectionInputs(
        document=make_analysis_document((section,)), section=section, analysis=None,
        run=SimpleNamespace(id='run', glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(payload={'schema_version': 1}),
        snapshot=SimpleNamespace(to_payload=lambda: {'model': 'correction'}),
        blocks=(block,), results={}, bindings={block.id: None},
    )

    original, = inputs.context['chapter_originals']
    assert original['attributes'] == {'lang': 'en'}
    assert original['source_reference']['resource_path'] == block.source_reference.resource_path
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
    assert json.loads(request.input_text)['blocks'][0]['original'] == original
