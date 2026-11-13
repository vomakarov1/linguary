"""A small invented book exercises the real pinned lexicon without a network."""

from dataclasses import asdict, replace
import json
import socket
import zipfile

from fastapi.testclient import TestClient
from lxml import etree

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.epub import EpubExportWriter, verify_exported_epub
from correction_fakes import CorrectionAwareFakeLLMClient as FakeLLMClient
from linguary.adapters.pronunciation.source import SOURCE_VERSION
from linguary.application.ports.epub import EpubWriteRequest
from linguary.application.pronunciation_projection import project_pronunciations
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.application.use_cases.reset_processing import ChooseBookConditions
from linguary.composition import build_application, drain_tasks
from linguary.domain import TaskStatus
from linguary.web import create_app
from test_book_conditions import store_imported
from test_pronunciation_selection import analysis_reply, candidate, processing_reply, regenerate


TEXTS = ('the colonel joined the queue by the quay.', 'I read by the queue. Xyzzy saw the queue.')
TRANSLATIONS = ('Полковник встал в очередь у причала.', 'Я читал у очереди. Икси увидел очередь.')
EXPECTED = {
    'uk': {'colonel': 'kˈɜːnəl', 'queue': 'kjˈuː', 'quay': 'kˈiː'},
    'us': {'colonel': 'ˈkɝnəɫ', 'queue': 'ˈkju'},
}


def _candidates(text):
    items = []
    for form in ('colonel', 'queue', 'quay', 'read', 'the', 'Xyzzy'):
        start = text.find(form)
        while start >= 0:
            proposed = candidate(form, start=start, proper=form == 'Xyzzy')
            if form == 'read':
                proposed = replace(proposed, selection_reason='context_dependent')
            items.append(proposed)
            start = text.find(form, start + len(form))
    return sorted(items, key=lambda item: item.anchor.start)


async def _small_book(tmp_path, variant):
    replies = [analysis_reply(treatment='TRANSLATE'), analysis_reply(treatment='TRANSLATE')]
    replies += [processing_reply(_candidates(text), treatment='TRANSLATE', translated_text=translation)
                for text, translation in zip(TEXTS, TRANSLATIONS)]
    replies += [processing_reply(_candidates(TEXTS[0]), treatment='TRANSLATE', translated_text=TRANSLATIONS[0])] * 2
    fake = FakeLLMClient(replies)
    app = build_application(settings=AppSettings(
        llm=LLMSettings(model='small-book', api_key='synthetic', proxy_mode='direct'),
        storage=StorageSettings(data_root=str(tmp_path / 'data')),
    ), llm_client=fake)
    store_imported(app, tuple((text,) for text in TEXTS))
    with app.unit_of_work.transaction() as tx:
        document = tx.documents.get('book-1')
    conditions = ChooseBookConditions(app.book_conditions, app.book_runtime)('book-1',
        action='reanalyze', expected_revision=0, settings_revision=0, confirmed=True,
        pronunciation_help_enabled=True, pronunciation_variant=variant)
    assert conditions.payload['pronunciation_versions']['source'] == SOURCE_VERSION
    await app.analysis.start('book-1')
    tasks = await drain_tasks(app)
    assert all(task.status is TaskStatus.SUCCEEDED for task in tasks), [task.last_error for task in tasks]
    run = app.start_processing()(StartProcessingCommand('book-1'))
    await app.processing.start('book-1')
    tasks = await drain_tasks(app)
    assert all(task.status is TaskStatus.SUCCEEDED for task in tasks), [task.last_error for task in tasks]
    return app, fake, document, run


def _no_network(*args, **kwargs):
    raise AssertionError('Small-book processing and export must stay offline')


async def _assert_small_book(tmp_path, monkeypatch, variant):
    monkeypatch.setattr(socket, 'create_connection', _no_network)
    app, fake, document, run = await _small_book(tmp_path, variant)
    try:
        blocks = tuple(document.blocks())
        with app.unit_of_work.transaction() as tx:
            stored = {item.block_id: item for item in tx.results.list_for_run(run.id)}
            assert tx.documents.get('book-1') == document
            for block in blocks:
                result = stored[block.id]
                binding = tx.work.get_binding('result', f'{run.id}:{block.id}')
                archive = tx.work.get_artifact(binding['artifact_revision_id'])
                assert archive['body']['result']['pronunciation_assistances'] == [asdict(item) for item in result.pronunciation_assistances]
        projected = project_pronunciations(blocks, stored)
        expected = EXPECTED[variant]
        assert {item.quote: item.ipa for item in projected} == expected
        assert len(projected) == (5 if variant == 'uk' else 4)
        assert all(item.variant == variant and item.quote not in ('read', 'the', 'Xyzzy') for item in projected)
        queues = [item for item in projected if item.quote == 'queue']
        assert len(queues) == 3 and sum(item.is_primary for item in queues) == 1
        assert len({item.hint_id for item in queues}) == 1
        assert [stored[block.id].translated_text for block in blocks] == list(TRANSLATIONS)
        for result in stored.values():
            for item in result.pronunciation_assistances:
                assert item.evidence.kind == 'lexicon_form' and item.evidence.source_version == SOURCE_VERSION
                assert item.evidence.sense_id is item.evidence.part_of_speech is None
        with TestClient(create_app(application=app, run_background_worker=False), base_url=app.settings.server.base_url) as client:
            whole = client.get('/books/book-1/result').json()
            actual = [item for chapter in whole['chapters'] for block in chapter['blocks']
                      for item in block['pronunciation_assistances']]
            assert actual == [asdict(item) for item in projected]
            repeated = client.get(f'/books/book-1/blocks/{blocks[1].id}').json()
            assert repeated['pronunciation_assistances'] == [asdict(item) for item in projected if item.block_id == blocks[1].id]
            assert client.get('/pronunciation-sources').status_code == 200
        target = tmp_path / f'ipa-dict-{variant}.epub'
        from linguary.application.layout.book import LayoutInputs, build_book_layout
        layout = build_book_layout(LayoutInputs(document, stored, (), (), None))
        EpubExportWriter().write(EpubWriteRequest(
            document=document, target_path=str(target), block_results=stored, layout=layout,
        ))
        assert verify_exported_epub(target) == ()
        parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
        with zipfile.ZipFile(target) as archive:
            roots = [etree.fromstring(archive.read(name), parser=parser) for name in archive.namelist() if name.endswith('.xhtml')]
            provenance, = [json.loads(archive.read(name)) for name in archive.namelist() if name.endswith('pronunciation-sources.json')]
        references = [node for root in roots for node in root.iter() if 'lg-ref' in (node.get('class') or '').split()]
        notes = [node for root in roots for node in root.iter() if 'lg-ipa' in (node.get('class') or '').split()]
        assert len(references) == len(projected) and len(notes) == len(projected)
        assert sum(node.get('{http://www.idpf.org/2007/ops}type') == 'backlink' for root in roots for node in root.iter()) == 0
        assert {entry['form']: entry['ipa'] for entry in provenance['entries']} == expected
        assert all(entry['license_id'] == ('GPL-3.0' if variant == 'uk' else 'MIT') for entry in provenance['entries'])
        for request in fake.requests:
            outgoing = request.instructions + request.input_text
            assert SOURCE_VERSION not in outgoing and 'raw.githubusercontent.com' not in outgoing
            assert all(ipa not in outgoing for ipa in expected.values())
    finally:
        await app.aclose()


async def test_ipa_dict_small_book_uk(tmp_path, monkeypatch):
    await _assert_small_book(tmp_path, monkeypatch, 'uk')


async def test_ipa_dict_small_book_us(tmp_path, monkeypatch):
    await _assert_small_book(tmp_path, monkeypatch, 'us')


async def test_ipa_dict_small_book_reprocessing_and_source_failure(tmp_path, monkeypatch):
    from linguary.adapters.pronunciation import source

    monkeypatch.setattr(socket, 'create_connection', _no_network)
    app, _, document, run = await _small_book(tmp_path, 'uk')
    try:
        block = document.sections[0].blocks[0]
        with app.unit_of_work.transaction() as tx:
            before = tx.results.get(run.id, block.id)
            old_binding = tx.work.get_binding('result', f'{run.id}:{block.id}')
        await regenerate(app, block.id)
        with app.unit_of_work.transaction() as tx:
            regenerated = tx.results.get(run.id, block.id)
            assert regenerated == before
            new_binding = tx.work.get_binding('result', f'{run.id}:{block.id}')
            assert new_binding['artifact_revision_id'] != old_binding['artifact_revision_id']
            old = tx.work.get_artifact(old_binding['artifact_revision_id'])
            assert old['body']['result']['pronunciation_assistances'] == [asdict(item) for item in before.pronunciation_assistances]
        source._load_index.cache_clear()
        monkeypatch.setattr(source, '_read_resource', lambda filename: (_ for _ in ()).throw(OSError('unavailable')))
        await regenerate(app, block.id)
        with app.unit_of_work.transaction() as tx:
            assert tx.results.get(run.id, block.id) == before
            assert tx.documents.get('book-1') == document
    finally:
        source._load_index.cache_clear()
        await app.aclose()
