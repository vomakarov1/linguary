"""Source contracts and the pinned local pronunciation snapshot."""
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


async def test_ipa_dict_uk_us_exact_lookup():
    from linguary.adapters.pronunciation.source import IpaDictPronunciationSource, SOURCE_VERSION
    source = IpaDictPronunciationSource()
    uk = await source.lookup('colonel', 'uk')
    us = await source.lookup('colonel', 'us')
    assert [item.ipa for item in uk.entries] == ['kˈɜːnəl']
    assert [item.ipa for item in us.entries] == ['ˈkɝnəɫ']
    assert all(item.evidence.kind == 'lexicon_form' and item.sense is None
               and item.evidence.source_version == SOURCE_VERSION for item in (*uk.entries, *us.entries))
    assert (await source.lookup('Colonel', 'uk')).status == 'missing'
    assert (await source.lookup('colonel', 'au')).status == 'missing'


@pytest.mark.parametrize('field, expected', [
    ('/ðə, ði/', ('ðə', 'ði')),
    ('/ˈɹɛd/, /ˈɹid/', ('ˈɹɛd', 'ˈɹid')),
    ('/ðə/, /ðə/', ('ðə',)),
])
def test_ipa_dict_all_alternatives_in_one_or_many_wrappers(field, expected):
    from linguary.adapters.pronunciation.source import _parse_pronunciations
    assert _parse_pronunciations(field) == expected


def test_ipa_dict_duplicate_rows_union_before_selection():
    from linguary.adapters.pronunciation.source import _parse_rows
    entries = _parse_rows('zorp\t/zɔːp/\nzorp\t/zɔːp/, /zɛp/\n', 'uk')['zorp']
    assert [ipa for ipa, line in entries] == ['zɔːp', 'zɛp']
    assert [line for ipa, line in entries] == [1, 2]


@pytest.mark.parametrize('field', ['/zɔːp/, /?/', '/zɔːp, /', '/zɔːp/ garbage', '/zɔː(p)/'])
def test_ipa_dict_invalid_alternative_never_becomes_unique(field):
    from linguary.adapters.pronunciation.source import _parse_rows
    rows = _parse_rows(f'zorp\t/zɔːp/\nzorp\t{field}\n', 'uk')
    assert rows['zorp'] == ()


@pytest.mark.parametrize('missing', [False, True])
async def test_ipa_dict_corrupt_or_missing_resource_is_unavailable(monkeypatch, missing):
    from linguary.adapters.pronunciation import source as module
    original = module._read_resource
    calls = []
    def damaged(filename):
        if filename == 'en_UK.txt':
            calls.append(filename)
            if missing:
                raise FileNotFoundError(filename)
            return b'corrupted'
        return original(filename)
    module._load_index.cache_clear()
    monkeypatch.setattr(module, '_read_resource', damaged)
    try:
        source = module.IpaDictPronunciationSource()
        assert (await source.lookup('queue', 'uk')).status == 'unavailable'
        assert (await source.lookup('queue', 'uk')).status == 'unavailable'
        assert calls == ['en_UK.txt', 'en_UK.txt']
    finally:
        module._load_index.cache_clear()


async def test_ipa_dict_no_network(monkeypatch):
    import socket
    from linguary.adapters.pronunciation import source as module
    def no_network(*args, **kwargs):
        raise AssertionError('dictionary lookup must stay local')
    monkeypatch.setattr(socket, 'create_connection', no_network)
    module._load_index.cache_clear()
    assert (await module.IpaDictPronunciationSource().lookup('queue', 'uk')).status == 'found'
