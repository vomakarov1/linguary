"""Integrity of the locally supplied ipa-dict snapshot."""

import hashlib
import json
from importlib import resources

import pytest


COMMIT = '972cd67dd5f96bc5660e28070cde26e3928abcd7'
SOURCE_VERSION = f'ipa-dict@{COMMIT}:parser-1'
SNAPSHOT = resources.files('linguary.adapters.pronunciation').joinpath('data', COMMIT)
EXPECTED_FILES = {
    'en_UK.txt': {
        'variant': 'uk',
        'bytes': 1684124,
        'lines': 65119,
        'sha256': '221394caef0cf723b4f2df81a98ac33191293257b88aed5b1fb89466d3a0dc77',
        'license_id': 'GPL-3.0',
    },
    'en_US.txt': {
        'variant': 'us',
        'bytes': 3180267,
        'lines': 125927,
        'sha256': '2af6f154a5c363275f052d1f85acedef38ed185ca9745aa4314be77f6b70de67',
        'license_id': 'MIT',
    },
}


def test_ipa_dict_snapshot_hashes():
    manifest = json.loads(SNAPSHOT.joinpath('manifest.json').read_text(encoding='utf-8'))
    assert manifest['repository'] == 'https://github.com/open-dict-data/ipa-dict'
    assert manifest['commit'] == COMMIT
    assert manifest['parser_version'] == 1
    assert manifest['source_version'] == SOURCE_VERSION
    assert set(manifest['files']) == set(EXPECTED_FILES)
    for filename, expected in EXPECTED_FILES.items():
        record = manifest['files'][filename]
        assert record == {
            **expected,
            'upstream_path': f'data/{filename}',
            'source_uri': f'https://raw.githubusercontent.com/open-dict-data/ipa-dict/{COMMIT}/data/{filename}',
        }
        content = SNAPSHOT.joinpath(filename).read_bytes()
        assert len(content) == expected['bytes']
        assert len(content.splitlines()) == expected['lines']
        assert hashlib.sha256(content).hexdigest() == expected['sha256']


@pytest.mark.parametrize('filename', EXPECTED_FILES)
def test_ipa_dict_snapshot_utf8_and_tab_format(filename):
    lines = SNAPSHOT.joinpath(filename).read_text(encoding='utf-8').splitlines()
    for line in lines:
        form, ipa = line.split('\t')
        assert form
        assert ipa
        assert '\ufffd' not in line


def test_ipa_dict_snapshot_packaged_resources():
    assert SNAPSHOT.is_dir()
    assert {item.name for item in SNAPSHOT.iterdir()} >= {
        'manifest.json', *EXPECTED_FILES,
    }


def test_ipa_dict_notice_covers_each_locale():
    notice = SNAPSHOT.joinpath('NOTICE.md').read_text(encoding='utf-8')
    assert f'https://github.com/open-dict-data/ipa-dict/tree/{COMMIT}' in notice
    assert 'en_UK.txt — GPL-3.0' in notice
    assert 'en_US.txt — MIT' in notice
    for source in ('ipa-dict', 'cmudict-ipa', 'syllabify', 'CMUdict', 'ipacards', 'eSpeak'):
        assert source in notice
    assert 'без изменений' in notice
    assert 'омограф' in notice
    assert 'ошибк' in notice


def test_ipa_dict_license_texts_and_sources_present():
    licenses = SNAPSHOT.joinpath('LICENSES.txt').read_text(encoding='utf-8')
    for copyright_notice in (
        'Copyright (c) 2016 dohliam',
        'Copyright (c) 2016 Lingliang Zhang',
        'Copyright (c) 2012-2013 Kyle Gorman <gormanky@ohsu.edu>',
        'Copyright (C) 1993-2015 Carnegie Mellon University. All rights reserved.',
        'Copyright (C) 2007 Free Software Foundation, Inc.',
    ):
        assert copyright_notice in licenses
    assert licenses.count('Permission is hereby granted, free of charge') == 3
    assert 'The contents of this file are deemed to be source code.' in licenses
    assert 'GNU GENERAL PUBLIC LICENSE' in licenses
    assert 'Version 3, 29 June 2007' in licenses
    assert 'END OF TERMS AND CONDITIONS' in licenses
    assert 'If the program does terminal interaction, make it output a short' in licenses
    for project in ('open-dict-data/ipa-dict', 'lingz/cmudict-ipa',
                    'kylebgorman/syllabify', 'cmusphinx/cmudict', 'melissaboiko/ipacards'):
        assert f'https://github.com/{project}/blob/' in licenses
