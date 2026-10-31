"""Pinned local pronunciations; a lexicon row makes no semantic claim."""
import asyncio
from functools import cache
from hashlib import sha256
from importlib import resources
import json
import re
from types import MappingProxyType

from linguary.application.ports.pronunciation import SourceEntry, SourceLookupResult, _valid_ipa, valid_lookup_form
from linguary.domain.pronunciation import PronunciationEvidence


SNAPSHOT = '972cd67dd5f96bc5660e28070cde26e3928abcd7'
SOURCE_VERSION = f'ipa-dict@{SNAPSHOT}:parser-1'
_FILES = {
    'uk': ('en_UK.txt', 1684124, 65119, '221394caef0cf723b4f2df81a98ac33191293257b88aed5b1fb89466d3a0dc77', 'GPL-3.0'),
    'us': ('en_US.txt', 3180267, 125927, '2af6f154a5c363275f052d1f85acedef38ed185ca9745aa4314be77f6b70de67', 'MIT'),
}


class DisabledPronunciationSource:
    configured = False

    async def lookup(self, form, variant):
        return SourceLookupResult.unavailable('source_disabled')


def _read_resource(filename):
    return resources.files('linguary.adapters.pronunciation').joinpath('data', SNAPSHOT, filename).read_bytes()


def _parse_pronunciations(field):
    wrappers = tuple(re.finditer(r'/([^/]*)/', field))
    if not wrappers or field[:wrappers[0].start()].strip() or field[wrappers[-1].end():].strip():
        return ()
    previous = wrappers[0]
    for wrapper in wrappers[1:]:
        if field[previous.end():wrapper.start()].strip() != ',':
            return ()
        previous = wrapper
    alternatives = tuple(part.strip() for wrapper in wrappers for part in wrapper[1].split(','))
    if any(not _valid_ipa(part) or any(char.isspace() or char in '()|/' for char in part)
           for part in alternatives):
        return ()
    return tuple(dict.fromkeys(alternatives))


def _parse_rows(text, variant):
    rows, invalid = {}, set()
    for number, line in enumerate(text.splitlines(), 1):
        if line.count('\t') != 1:
            raise ValueError('invalid snapshot row')
        form, field = line.split('\t')
        if not form or not field:
            raise ValueError('empty snapshot row')
        if not valid_lookup_form(form):
            continue
        alternatives = _parse_pronunciations(field)
        if not alternatives:
            invalid.add(form)
        items = rows.setdefault(form, {})
        for ipa in alternatives:
            items.setdefault(ipa, number)
    return MappingProxyType({form: () if form in invalid else tuple(items.items())
                             for form, items in rows.items()})


@cache
def _load_index(variant):
    filename, size, lines, digest, license_id = _FILES[variant]
    manifest = json.loads(_read_resource('manifest.json'))
    expected = dict(variant=variant, bytes=size, lines=lines, sha256=digest,
        license_id=license_id, upstream_path=f'data/{filename}',
        source_uri=f'https://raw.githubusercontent.com/open-dict-data/ipa-dict/{SNAPSHOT}/data/{filename}')
    if (not isinstance(manifest, dict)
            or manifest.get('repository') != 'https://github.com/open-dict-data/ipa-dict'
            or manifest.get('commit') != SNAPSHOT or manifest.get('parser_version') != 1
            or manifest.get('source_version') != SOURCE_VERSION
            or not isinstance(manifest.get('files'), dict)
            or manifest['files'].get(filename) != expected):
        raise ValueError('invalid snapshot manifest')
    data = _read_resource(filename)
    if len(data) != size or sha256(data).hexdigest() != digest:
        raise ValueError('invalid snapshot checksum')
    text = data.decode('utf-8')
    if len(text.splitlines()) != lines:
        raise ValueError('invalid snapshot length')
    return _parse_rows(text, variant)


class IpaDictPronunciationSource:
    configured = True

    async def lookup(self, form, variant):
        if variant not in _FILES or not valid_lookup_form(form):
            return SourceLookupResult.missing()
        try:
            index = await asyncio.to_thread(_load_index, variant)
        except (OSError, ValueError, TypeError, KeyError, UnicodeError):
            return SourceLookupResult.unavailable('source_unavailable')
        pronunciations = index.get(form, ())
        if not pronunciations:
            return SourceLookupResult.missing()
        filename, _, _, _, license_id = _FILES[variant]
        source_uri = f'https://raw.githubusercontent.com/open-dict-data/ipa-dict/{SNAPSHOT}/data/{filename}'
        entries = tuple(SourceEntry(form, ipa, variant, None, None, PronunciationEvidence(
            source_id='ipa-dict', source_version=SOURCE_VERSION,
            entry_id=f'{SNAPSHOT}:{filename}:{number}:{form}:{sha256(ipa.encode()).hexdigest()[:16]}',
            sense_id=None, part_of_speech=None, dictionary_form=form, dictionary_ipa=ipa,
            attribution=f'ipa-dict {filename} ({license_id}), snapshot {SNAPSHOT}',
            kind='lexicon_form', license_id=license_id, source_uri=source_uri))
            for ipa, number in pronunciations)
        return SourceLookupResult.found(entries)
