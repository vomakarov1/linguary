"""One accepted book snapshot projects identically for every presentation."""
from dataclasses import asdict, replace

from linguary.application.pronunciation_projection import project_pronunciations
from linguary.domain import BlockProcessingResult, BlockTreatment, BlockType, DocumentBlock, SourceReference
from linguary.domain.content_plan import SourceAnchor
from linguary.domain.pronunciation import PronunciationAssistance, PronunciationEvidence


def block(identifier, order=0, text='zorp'):
    return DocumentBlock(id=identifier, section_id='s', order=order, type=BlockType.PARAGRAPH,
                         text=text, source_reference=SourceReference('s.xhtml', '/html/body/p'))


def assistance(identifier, *, start=0, ipa='zɔːp', hint='hint'):
    return PronunciationAssistance(SourceAnchor(identifier, start, start + 4, 'zorp'), 'zorp', ipa, 'uk', hint,
        PronunciationEvidence('synthetic-private', 'private-version', 'entry-1', 'private-sense', 'verb',
                              'zorp', ipa, 'Invented test data'))


def result(identifier, *items):
    return BlockProcessingResult(block_id=identifier, treatment=BlockTreatment.READ_AS_IS,
                                 pronunciation_assistances=items)


def test_pronunciation_projection_stable_original_order():
    blocks = [block('first', 0, 'zorp zorp'), block('later', 1)]
    results = {'later': result('later', assistance('later')), 'first': result('first', assistance('first', start=5), assistance('first'))}
    projected = project_pronunciations(blocks, results)
    assert [(p.block_id, p.start, p.is_primary) for p in projected] == [('first', 0, True), ('first', 5, False), ('later', 0, False)]
    assert project_pronunciations(blocks, dict(reversed(list(results.items())))) == projected


def test_pronunciation_projection_rejects_wrong_owner_or_anchor():
    blocks = [block('first'), block('later', 1)]
    assert project_pronunciations(blocks, {'first': result('later', assistance('later'))}) == ()
    corrupted = result('first')
    object.__setattr__(corrupted, 'pronunciation_assistances', (assistance('later'),))
    assert project_pronunciations(blocks, {'first': corrupted}) == ()
    assert project_pronunciations([block('first', text='nope')], {'first': result('first', assistance('first'))}) == ()


def test_pronunciation_projection_primary_and_repeat_identity():
    blocks = [block('first'), block('later', 1), block('other', 2)]
    projected = project_pronunciations(blocks, {'first': result('first', assistance('first')),
        'later': result('later', assistance('later', hint='different-stored-id')),
        'other': result('other', assistance('other', ipa='zɛp', hint='other-hint'))})
    assert [p.hint_id for p in projected] == ['hint', 'hint', 'other-hint']
    assert [p.is_primary for p in projected] == [True, False, True]


def test_pronunciation_projection_exposes_only_allowed_attribution():
    item, = project_pronunciations([block('first')], {'first': result('first', assistance('first'))})
    assert set(asdict(item)) == {'hint_id', 'block_id', 'start', 'end', 'quote', 'ipa', 'variant', 'source_label', 'source_ref', 'is_primary'}
    assert item.source_label == 'Invented test data' and item.source_ref == 'entry-1'
    assert 'private' not in repr(asdict(item))
