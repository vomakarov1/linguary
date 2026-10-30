"""Pronunciation provenance distinguishes senses from an exact lexicon form."""

from dataclasses import replace

import pytest

from linguary.application.ports.pronunciation import InflectionRule, SourceEntry
from linguary.domain.content_plan import SourceAnchor
from linguary.domain.errors import DomainError
from linguary.domain.pronunciation import PronunciationAssistance, PronunciationEvidence


SNAPSHOT = '972cd67dd5f96bc5660e28070cde26e3928abcd7'


def lexicon_evidence(**changes):
    evidence = PronunciationEvidence(
        source_id='ipa-dict', source_version=f'ipa-dict@{SNAPSHOT}:parser-1',
        entry_id='en_UK.txt:1:queue', sense_id=None, part_of_speech=None,
        dictionary_form='queue', dictionary_ipa='kjuː', attribution='IPA-Dict',
        kind='lexicon_form', license_id='GPL-3.0',
        source_uri=f'https://raw.githubusercontent.com/open-dict-data/ipa-dict/{SNAPSHOT}/data/en_UK.txt',
    )
    return replace(evidence, **changes)


@pytest.mark.parametrize(('variant', 'filename', 'license_id'), [
    ('uk', 'en_UK.txt', 'GPL-3.0'), ('us', 'en_US.txt', 'MIT'),
])
def test_lexicon_form_evidence_has_no_semantic_claim(variant, filename, license_id):
    evidence = lexicon_evidence(
        license_id=license_id,
        source_uri=f'https://raw.githubusercontent.com/open-dict-data/ipa-dict/{SNAPSHOT}/data/{filename}',
    )
    entry = SourceEntry('queue', 'kjuː', variant, None, None, evidence)

    assert evidence.kind == 'lexicon_form'
    assert evidence.sense_id is None
    assert evidence.part_of_speech is None
    assert evidence.lexicon_variant == variant
    assert entry.sense is None
    assert entry.grammatical_form is None
    assert not entry.proper_name_confirmed


def test_dictionary_sense_evidence_stays_strict():
    evidence = PronunciationEvidence(
        'synthetic', 'invented-v1', 'entry', 'sense', 'noun', 'queue', 'kjuː', 'Invented data',
    )
    entry = SourceEntry('queue', 'kjuː', 'uk', 'a line of people', 'singular', evidence)

    assert evidence.kind == 'dictionary_sense'
    assert evidence.license_id is None
    assert evidence.source_uri is None
    assert entry.sense == 'a line of people'
    for changes in ({'sense_id': None}, {'part_of_speech': None}, {'sense_id': ''}):
        with pytest.raises(DomainError):
            replace(evidence, **changes)
    with pytest.raises(ValueError):
        replace(entry, sense=None)
    with pytest.raises(ValueError):
        replace(entry, grammatical_form=None)


@pytest.mark.parametrize('changes', [
    {'kind': 'other'}, {'kind': True}, {'kind': []},
    {'license_id': None}, {'license_id': ''}, {'license_id': True},
    {'source_id': 'other-lexicon'}, {'source_version': 'current'},
    {'source_uri': None}, {'source_uri': ''},
    {'source_uri': 'https://example.com/' + SNAPSHOT + '/data/en_UK.txt'},
    {'source_uri': 'http://raw.githubusercontent.com/open-dict-data/ipa-dict/' + SNAPSHOT + '/data/en_UK.txt'},
    {'source_uri': 'https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/en_UK.txt'},
    {'source_uri': 'https://raw.githubusercontent.com/open-dict-data/ipa-dict/' + 'a' * 40 + '/data/en_UK.txt'},
    {'source_uri': 'https://raw.githubusercontent.com/open-dict-data/ipa-dict/' + SNAPSHOT + '/data/en_UK.txt?next=other'},
    {'license_id': 'MIT'},
])
def test_lexicon_form_requires_license_and_pinned_source(changes):
    with pytest.raises(DomainError):
        lexicon_evidence(**changes)


@pytest.mark.parametrize('changes', [
    {'sense_id': 'a line of people'}, {'part_of_speech': 'noun'},
    {'rule_id': 'plural', 'rule_version': '1'},
])
def test_lexicon_form_evidence_rejects_semantic_and_inflection_claims(changes):
    with pytest.raises(DomainError):
        lexicon_evidence(**changes)


@pytest.mark.parametrize('changes', [
    {'sense': 'a line of people'}, {'grammatical_form': 'singular'},
    {'proper_name_confirmed': True}, {'proper_name_confirmed': 1},
    {'inflection_rule': InflectionRule('plural', '1', 'plural', 's', 'z')},
    {'form': 'queues'}, {'ipa': 'kjuːz'}, {'variant': 'us'},
])
def test_lexicon_form_rejects_inflection_and_proper_name_claim(changes):
    entry = SourceEntry('queue', 'kjuː', 'uk', None, None, lexicon_evidence())
    with pytest.raises(ValueError):
        replace(entry, **changes)


def test_lexicon_assistance_variant_matches_its_source_file():
    help = PronunciationAssistance(
        SourceAnchor('block', 0, 5, 'queue'), 'queue', 'kjuː', 'uk', 'hint', lexicon_evidence(),
    )
    with pytest.raises(DomainError):
        replace(help, variant='us')
