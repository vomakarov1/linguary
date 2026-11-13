"""Synthetic acceptance covers contracts; independent language review remains required."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from linguary.application.ports.pronunciation import InflectionRule
from linguary.application.validators.pronunciation import verify_candidates
from test_pronunciation_selection import SyntheticSource, analysis_reply, candidate, processing_reply, regenerate
from test_pronunciation_source import entry


CASES = json.loads((Path(__file__).resolve().parents[2] / 'fixtures/pronunciation/acceptance_cases.json').read_text())


def acceptance_source():
    base = entry()
    rule = InflectionRule('synthetic-plural', '1', 'plural', 's', 's')
    return SyntheticSource((base, entry(variant='us', ipa='zɔp'), entry(ipa='zɛp', grammar='past'),
        entry(form='zorple', ipa='ˈzɔːpəl'), entry(form='Xyzzy', ipa='zɪzi'),
        replace(base, form='zorps', ipa='zɔːps', grammatical_form='plural', inflection_rule=rule,
                evidence=replace(base.evidence, rule_id=rule.id, rule_version=rule.version))))


@pytest.mark.parametrize('case', CASES['cases'], ids=lambda case: case['id'])
async def test_pronunciation_acceptance_positive_and_negative(case):
    source = acceptance_source()
    if case.get('ambiguous'):
        source.entries += (entry(ipa='zɪp'),)
    proposed = replace(candidate(case['form'], sense=case['context'], grammar=case['grammar'],
                                 proper=case.get('proper', False)),
                       selection_reason=case.get('reason', 'unpredictable_spelling'))
    selection = await verify_candidates([proposed], source=source, texts={'b1': case['text']},
                                         variant=case['variant'], scope_id='synthetic-run')
    assert bool(selection.accepted) is case['expected']
    assert not selection.unavailable and not selection.unavailable_anchors
    if case['expected']:
        item, = selection.accepted
        assert item.ipa == case['ipa'] and item.variant == case['variant']
        assert item.evidence.entry_id == case['entry_id']
        assert item.evidence.rule_id == case.get('rule_id')
        assert item.evidence.rule_version == case.get('rule_version')
        item.anchor.validate({'b1': case['text']})
    else:
        assert selection.skipped


def test_pronunciation_quality_threshold_requires_agreement():
    # A perfect synthetic contract run does not supply an agreed linguistic threshold.
    assert CASES['quality_threshold'] is None
    assert CASES['independent_review'] is None
    assert CASES['release_ready'] is False
    assert 'invented' in CASES['provenance']
    assert {case['expected'] for case in CASES['cases']} == {True, False}


async def test_pronunciation_partial_failure_preserves_materials(tmp_path):
    from linguary.domain import LanguageNote, BlockProcessingResult, BlockTreatment, GlossaryHint
    from linguary.domain.content_plan import SourceAnchor
    from linguary.domain.learning import FragmentAssistance
    source = SyntheticSource((entry(), entry(form='bip', ipa='bɪp')))
    candidates = [candidate(), candidate('bip', start=5)]
    from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
    from correction_fakes import CorrectionAwareFakeLLMClient as FakeLLMClient
    from linguary.application.use_cases.reset_processing import ChooseBookConditions
    from linguary.application.use_cases.approve_glossary import StartProcessingCommand
    from linguary.composition import build_application, drain_tasks
    from test_book_conditions import store_imported
    fake = FakeLLMClient([analysis_reply(2, treatment='TRANSLATE'),
        processing_reply(candidates, blocks=2, treatment='TRANSLATE', translated_text='Вымышленное действие.'),
        processing_reply(candidates, treatment='TRANSLATE', translated_text='Новое вымышленное действие.')])
    app = build_application(settings=AppSettings(llm=LLMSettings(model='synthetic', api_key='synthetic', proxy_mode='direct'),
        storage=StorageSettings(data_root=str(tmp_path / 'data'))), llm_client=fake, pronunciation_source=source)
    document = store_imported(app, (('zorp bip', 'A separate paragraph.'),))
    ChooseBookConditions(app.book_conditions, app.book_runtime)('book-1', action='reanalyze', expected_revision=0,
        settings_revision=0, confirmed=True, pronunciation_help_enabled=True, pronunciation_variant='uk')
    await app.analysis.start('book-1')
    await drain_tasks(app)
    run = app.start_processing()(StartProcessingCommand('book-1'))
    await app.processing.start('book-1')
    await drain_tasks(app)
    block, sibling = document.sections[0].blocks
    note = LanguageNote('note', sibling.id, 'separate', 'Смысл вымышленного действия.')
    fragment = FragmentAssistance('difficulty', ('point',), 'russian_explanation',
                                  'Самостоятельное смысловое пояснение.', (SourceAnchor(sibling.id, 2, 10, 'separate'),))
    semantic = BlockProcessingResult(block_id=sibling.id, treatment=BlockTreatment.TRANSLATE,
        translated_text='Ранее принятый перевод.', glossary_hints=(GlossaryHint('zorp', 'выдумка'),),
        language_notes=(note,), fragment_assistances=(fragment,))
    try:
        with app.unit_of_work.transaction() as tx:
            tx.results.replace_results(run.id, [semantic])
        # Both lookup failures must leave prior accepted IPA and all other stored materials usable.
        source.failures = {'zorp', 'bip'}
        with app.unit_of_work.transaction() as tx:
            before = tx.results.get(run.id, block.id)
            document_before = tx.documents.get('book-1')
        await regenerate(app, block.id)
        with app.unit_of_work.transaction() as tx:
            after = tx.results.get(run.id, block.id)
            assert tx.documents.get('book-1') == document_before
        assert after.pronunciation_assistances == before.pronunciation_assistances
        assert after.translated_text == 'Новое вымышленное действие.'
        with app.unit_of_work.transaction() as tx:
            assert tx.results.get(run.id, sibling.id) == semantic
    finally:
        await app.aclose()


async def test_pronunciation_repeat_and_export_regression(tmp_path, monkeypatch):
    """Small synthetic book: SQLite -> whole/fragment web -> autonomous EPUB."""
    from datetime import UTC, datetime
    import socket
    from fastapi.testclient import TestClient
    for directory in ('epub', 'web'):
        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / directory))
    from test_epub_export_content import pronunciation_book, pronunciation_nodes
    from export_pipeline import documents_of, elements
    from test_book_conditions import make_app
    from linguary.adapters.epub import EpubExportWriter, verify_exported_epub
    from linguary.application.book_overview import book_result
    from linguary.application.ports.epub import EpubWriteRequest
    from linguary.application.ports.persistence import BookRecord
    from linguary.application.pronunciation_projection import project_pronunciations
    from linguary.domain import BlockProcessingResult, BlockTreatment, BookMetadata, BookStatus
    from linguary.web import create_app
    from linguary.web.schemas import BookResultResponse
    from web_scenario import WebScenario, start_run

    source_path, document, _ = pronunciation_book(tmp_path)
    blocks = tuple(block for section in document.ordered_sections for block in section.ordered_blocks)
    texts = {block.id: block.text for block in blocks}
    source = acceptance_source()
    results = {}
    for block in reversed(blocks):
        form = 'zorp' if 'zorp' in block.text else 'Xyzzy' if 'Xyzzy' in block.text else None
        if form is None:
            continue
        proposed = candidate(form, block=block.id, start=block.text.index(form),
                             grammar='past' if block.text.startswith('Yesterday') else 'present',
                             proper=form == 'Xyzzy')
        selected = await verify_candidates([proposed], source=source, texts=texts, variant='uk', scope_id='synthetic-run')
        results[block.id] = BlockProcessingResult(block_id=block.id, treatment=BlockTreatment.READ_AS_IS,
                                                 pronunciation_assistances=selected.accepted)
    calls = tuple(source.calls)
    app = make_app(tmp_path / 'application')
    try:
        with app.unit_of_work.transaction() as tx:
            tx.books.add(BookRecord('book-1', BookMetadata('Synthetic pronunciation'), 'en', 'ru',
                BookStatus.COMPLETED, str(source_path), datetime.now(UTC)))
            tx.documents.save(document)
        scenario = WebScenario(document=document, book_id='book-1', run_id='synthetic-run')
        start_run(app, scenario, results=tuple(results.values()))
        with app.unit_of_work.transaction() as tx:
            stored = {item.block_id: item for item in tx.results.list_for_run('synthetic-run')}
        assert stored == results
        projected = project_pronunciations(blocks, stored)
        assert [p.ipa for p in projected] == ['zɔːp', 'zɔːp', 'zɛp']
        assert [p.is_primary for p in projected] == [True, False, True]
        assert projected[0].hint_id == projected[1].hint_id != projected[2].hint_id
        assert all(texts[p.block_id][p.start:p.end] == p.quote == 'zorp' for p in projected)
        web = BookResultResponse.of_result(book_result(app.unit_of_work, 'book-1'))
        assert [p.model_dump() for chapter in web.chapters for block in chapter.blocks
                for p in block.pronunciation_assistances] == [asdict(p) for p in projected]
        def no_network(*args, **kwargs):
            raise AssertionError('Reading/export must not use the network')
        monkeypatch.setattr(socket, 'create_connection', no_network)
        with TestClient(create_app(application=app, run_background_worker=False), base_url=app.settings.server.base_url) as client:
            whole = client.get('/books/book-1/result').json()
            assert [p for chapter in whole['chapters'] for block in chapter['blocks']
                    for p in block['pronunciation_assistances']] == [asdict(p) for p in projected]
            repeated = client.get(f'/books/book-1/blocks/{projected[1].block_id}').json()
            assert repeated['pronunciation_assistances'] == [asdict(projected[1])]
            target = tmp_path / 'synthetic-export.epub'
            from linguary.application.layout.book import LayoutInputs, build_book_layout
            layout = build_book_layout(LayoutInputs(document, stored, (), (), None))
            EpubExportWriter().write(EpubWriteRequest(document=document, target_path=str(target),
                source_epub_path=str(source_path), block_results=stored, layout=layout))
            assert verify_exported_epub(target) == ()
            notes = pronunciation_nodes(target, 'span', 'lg-ipa')
            references = pronunciation_nodes(target, 'a', 'lg-ref')
            assert len(notes) == 3 and len(references) == 3
            assert [node.text for _, node in notes] == [f'/{ipa}/' for ipa in ('zɔːp', 'zɔːp', 'zɛp')]
            assert all(node.get('href', '').startswith('#') for _, node in references)
            assert all('UK' in (node.tail or '') for _, node in notes)
            epub_type = '{http://www.idpf.org/2007/ops}type'
            assert not any(
                link.get(epub_type) == 'backlink'
                for root in documents_of(target).values()
                for link in elements(root, 'a')
            )
            assert tuple(source.calls) == calls
    finally:
        await app.aclose()
