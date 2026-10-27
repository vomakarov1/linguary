"""Real analysis validation and persisted processing for English lexical help."""

import json

import pytest

from test_content_plan_pipeline import BOOK, finish, import_story, make_app, run
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.adapters.llm.payloads import parse_payload
from linguary.application.analysis_merge import AnalysisMergeError, validate_content_plan
from linguary.application.ports.llm import LLMResponseKind
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.application.use_cases.export_book import ExportBookCommand
from linguary.domain import BlockTreatment
from linguary.web.labels import treatment_label


def lexical_analysis():
    return dict(
        translation_instructions='Preserve the original text.',
        chapter_summary='The cat sits quietly and hears a sound that comes back.\n\nКошка тихо сидит и слышит возвращающийся звук.',
        treatment_plan=[dict(source_block_id='b1', treatment='READ_AS_IS')],
        glossary_candidates=[], before_you_read=[],
        content_plan=dict(version=3, profile='content-plan-v3', context_facts=[],
            before_you_read_evidence=[], difficulties=[dict(
                id='echo', owner_block_id='b1', kind='vocabulary', priority='critical',
                help_kind='lexical_hint',
                anchors=[dict(block_id='b1', start=0, end=4, quote='Echo')],
                required_points=[dict(id='meaning', description='Meaning in context')])]),
    )


def test_read_as_is_label_describes_original_instead_of_claiming_no_help():
    assert treatment_label(BlockTreatment.READ_AS_IS) == 'В оригинале'


@pytest.mark.parametrize('level', ['a2', 'b1', 'b2'])
def test_english_lexical_analysis_processes_and_stores_without_byr(tmp_path, level):
    client = FakeLLMClient()
    app = make_app(tmp_path / 'data', client, {})
    try:
        _, section, blocks, _ = import_story(app, {'blocks': ['Echo. The cat sat.']})
        with app.unit_of_work.transaction() as tx:
            app.book_conditions.create(tx, tx.books.get(BOOK), app.runtime_settings.revision,
                                       english_help=True, reader_level=level)
        client.replies = (FakeReply.of_payload(lexical_analysis()),)
        run(app.analysis.start(BOOK))
        finish(app)
        analysis_instructions = client.requests[0].instructions
        assert 'пять-десять ключевых понятий' not in analysis_instructions
        assert '65–80% блоков главы' not in analysis_instructions
        assert 'before_you_read' in analysis_instructions and 'всегда пустой список' in analysis_instructions
        with app.unit_of_work.transaction() as tx:
            analysis = tx.analyses.get_for_section(BOOK, section.id)
        assert analysis.before_you_read == ()
        assert analysis.content_plan.version == 3
        assert analysis.content_plan.before_you_read_evidence == ()
        assert analysis.treatment_plan.items[0].treatment is BlockTreatment.READ_AS_IS
        difficulty = analysis.content_plan.difficulties[0]
        assert difficulty.help_kind == 'lexical_hint'
        assert difficulty.anchors[0].block_id == blocks[0].id
        processing_run = app.start_processing()(StartProcessingCommand(BOOK))
        client.replies = (*client.replies, FakeReply.of_payload(dict(
            blocks=[dict(source_block_id='b1', treatment='READ_AS_IS',
                         translated_text=None, glossary_hints=[])],
            language_notes=[], content_evidence=[], fragment_assistances=[dict(
                difficulty_id=difficulty.id, point_ids=['meaning'],
                kind='lexical_hint', text='a sound that comes back')],
        )))
        run(app.processing.start(BOOK))
        finish(app)
        processing_instructions = client.requests[1].instructions
        assert 'fragment_assistances' in processing_instructions
        assert 'content_evidence верни пустыми списками' in processing_instructions
        assert {
            'a2': 'Для A2 используй частотные слова',
            'b1': 'Для B1 используй знакомую лексику',
            'b2': 'Для B2 используй естественный английский',
        }[level] in processing_instructions
        assert 'Служебные context_facts применяй' not in processing_instructions
        assert [request.contract_version for request in client.requests] == [3, 4]
        with app.unit_of_work.transaction() as tx:
            results = tx.results.list_for_run(processing_run.id)
        assert len(results) == 1
        assert results[0].translated_text is None
        assert results[0].glossary_hints == ()
        assert results[0].language_notes == ()
        assert len(results[0].fragment_assistances) == 1
        assistance = results[0].fragment_assistances[0]
        assert assistance.kind == 'lexical_hint'
        assert assistance.text == 'a sound that comes back'
        assert assistance.anchors == difficulty.anchors
        exported = app.export_book(ExportBookCommand(
            book_id=BOOK, target_path=str(tmp_path / f'{level}-english-help.epub')))
        assert exported.path == str(tmp_path / f'{level}-english-help.epub')
        assert (tmp_path / f'{level}-english-help.epub').is_file()
    finally:
        app.dispose()


def test_english_help_with_no_difficulties_can_be_exported(tmp_path):
    client = FakeLLMClient()
    app = make_app(tmp_path / 'data', client, {})
    try:
        _, section, blocks, _ = import_story(app, {'blocks': ['A quiet cat slept.']})
        with app.unit_of_work.transaction() as tx:
            app.book_conditions.create(tx, tx.books.get(BOOK), app.runtime_settings.revision,
                                       english_help=True, reader_level='a2')
        client.replies = (FakeReply.of_payload(dict(
            translation_instructions='Preserve the original text.',
            chapter_summary='A cat sleeps quietly.\n\nКошка тихо спит.',
            treatment_plan=[dict(source_block_id='b1', treatment='READ_AS_IS')],
            glossary_candidates=[], before_you_read=[],
            content_plan=dict(version=3, profile='content-plan-v3', context_facts=[],
                              before_you_read_evidence=[], difficulties=[]),
        )),)
        run(app.analysis.start(BOOK))
        finish(app)
        with app.unit_of_work.transaction() as tx:
            analysis = tx.analyses.get_for_section(BOOK, section.id)
        processing_run = app.start_processing()(StartProcessingCommand(BOOK))
        client.replies = (*client.replies, FakeReply.of_payload(dict(
            blocks=[dict(source_block_id='b1', treatment='READ_AS_IS',
                         translated_text=None, glossary_hints=[])],
            language_notes=[], content_evidence=[], fragment_assistances=[],
        )))
        run(app.processing.start(BOOK))
        finish(app)
        with app.unit_of_work.transaction() as tx:
            results = tx.results.list_for_run(processing_run.id)
        assert analysis.content_plan.difficulties == ()
        assert len(results) == len(blocks) == 1
        assert results[0].content_evidence == ()
        assert results[0].fragment_assistances == ()
        target = tmp_path / 'empty-plan-english-help.epub'
        exported = app.export_book(ExportBookCommand(book_id=BOOK, target_path=str(target)))
        assert exported.path == str(target)
        assert target.is_file()
    finally:
        app.dispose()


@pytest.mark.parametrize('english_help,version,treatment,message', [
    (False, 3, 'READ_AS_IS', 'лексическая помощь требует'),
    (True, 2, 'READ_AS_IS', 'план версии 3'),
    (True, 3, 'TRANSLATE', 'чтение блоков в оригинале'),
])
def test_lexical_fragment_exception_is_limited_to_enabled_v3_original(
    english_help, version, treatment, message,
):
    raw = lexical_analysis()
    raw['content_plan'].update(version=version, profile=f'content-plan-v{version}')
    if version == 2:
        raw['content_plan'].pop('context_facts')
    raw['treatment_plan'][0]['treatment'] = treatment
    payload = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, json.dumps(raw), contract_version=version)
    with pytest.raises(AnalysisMergeError, match=message):
        validate_content_plan(payload, {'b1': 'block'}, {'block': 'Echo. The cat sat.'},
                              english_help=english_help)


def test_english_help_analysis_rejects_before_you_read_materials():
    raw = lexical_analysis()
    raw['before_you_read'] = [dict(expression='Echo', explanation='A returning sound.')]
    payload = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, json.dumps(raw), contract_version=3)
    with pytest.raises(AnalysisMergeError, match='не использует материалы BYR'):
        validate_content_plan(payload, {'b1': 'block'}, {'block': 'Echo. The cat sat.'},
                              english_help=True)


def test_english_help_allows_explanation_difficulties_for_fragment_help():
    raw = lexical_analysis()
    raw['content_plan']['difficulties'][0]['help_kind'] = 'explanation'
    payload = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, json.dumps(raw), contract_version=3)
    validate_content_plan(payload, {'b1': 'block'}, {'block': 'Echo. The cat sat.'},
                          english_help=True)


def test_english_help_analysis_rejects_two_difficulties_for_same_occurrence():
    raw = lexical_analysis()
    duplicate = dict(raw['content_plan']['difficulties'][0])
    duplicate.update(
        id='echo-idiom', kind='idiom',
        required_points=[dict(id='idiomatic-meaning', description='Meaning of the expression')],
    )
    raw['content_plan']['difficulties'].append(duplicate)
    payload = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, json.dumps(raw), contract_version=3)
    with pytest.raises(AnalysisMergeError, match='одно место'):
        validate_content_plan(payload, {'b1': 'block'}, {'block': 'Echo. The cat sat.'},
                              english_help=True)
