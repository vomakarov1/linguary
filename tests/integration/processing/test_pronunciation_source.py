"""Source contracts use invented words and never load dictionary data."""
from dataclasses import replace

import pytest

from linguary.adapters.pronunciation.source import DisabledPronunciationSource
from linguary.application.ports.pronunciation import InflectionRule, SourceEntry, SourceLookupResult, valid_lookup_form
from linguary.domain.pronunciation import PronunciationEvidence


def entry(*, form='zorp', ipa='zɔːp', variant='uk', sense='invented action', grammar='present'):
    return SourceEntry(form=form, ipa=ipa, variant=variant, sense=sense,
        grammatical_form=grammar, evidence=PronunciationEvidence(
            source_id='synthetic', source_version='test-1', entry_id='entry-1',
            sense_id=sense, part_of_speech='verb', dictionary_form=form,
            dictionary_ipa=ipa, attribution='Invented test data'))


def test_pronunciation_source_variant_and_sense():
    uk = entry()
    us = replace(uk, variant='us', ipa='zɔp', evidence=replace(uk.evidence, dictionary_ipa='zɔp'))
    other = replace(uk, sense='invented object', evidence=replace(uk.evidence, sense_id='sense-2', part_of_speech='noun'))
    found = SourceLookupResult.found((uk, us, other))
    assert found.status == 'found'
    assert {item.variant for item in found.entries} == {'uk', 'us'}
    assert {item.evidence.part_of_speech for item in found.entries} == {'verb', 'noun'}
    assert len({item.sense for item in found.entries}) == 2


def test_pronunciation_source_missing_vs_unavailable():
    assert SourceLookupResult.missing().status == 'missing'
    assert SourceLookupResult.unavailable('temporary_failure').status == 'unavailable'
    with pytest.raises(ValueError):
        SourceLookupResult(status='found')
    with pytest.raises(ValueError):
        SourceLookupResult(status='missing', entries=(entry(),))


@pytest.mark.parametrize('form', ['https://untrusted.invalid/entry', '//127.0.0.1', 'file:///tmp/source', 'zorp?url=x', 'two words', ''])
async def test_pronunciation_source_rejects_untrusted_url(form):
    assert not valid_lookup_form(form)
    assert (await DisabledPronunciationSource().lookup(form, 'uk')).status == 'unavailable'


async def test_pronunciation_disabled_source_skips():
    assert valid_lookup_form('zorp')
    assert valid_lookup_form("zorp’s")
    result = await DisabledPronunciationSource().lookup('zorp', 'us')
    assert result.status == 'unavailable'
    assert result.reason == 'source_disabled'
    assert result.entries == ()


def test_pronunciation_source_inflection_rule_checks_form_ipa_and_version():
    base = entry()
    rule = InflectionRule(id='test-past', version='1', grammatical_form='past',
                          form_suffix='ed', ipa_suffix='t')
    derived = replace(base, form='zorped', ipa='zɔːpt', grammatical_form='past',
        evidence=replace(base.evidence, rule_id=rule.id, rule_version=rule.version),
        inflection_rule=rule)
    assert rule.confirms(derived)
    assert not rule.confirms(replace(derived, ipa='zɔːpd'))
    assert not rule.confirms(replace(derived, form='zorps'))
    assert not rule.confirms(replace(derived, evidence=replace(derived.evidence, rule_version='2')))
    assert not replace(rule, base_form_suffix='x').confirms(derived)
