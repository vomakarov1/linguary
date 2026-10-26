"""Real analysis validation and persisted processing for English lexical help."""

import json

import pytest

from test_content_plan_pipeline import BOOK, finish, import_story, make_app, run
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.adapters.llm.payloads import parse_payload
from linguary.application.analysis_merge import AnalysisMergeError, validate_content_plan
from linguary.application.ports.llm import LLMResponseKind
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.domain import BlockTreatment


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
    finally:
        app.dispose()


@pytest.mark.parametrize('english_help,version,treatment', [
    (False, 3, 'READ_AS_IS'), (True, 2, 'READ_AS_IS'), (True, 3, 'TRANSLATE'),
])
def test_lexical_fragment_exception_is_limited_to_enabled_v3_original(
    english_help, version, treatment,
):
    raw = lexical_analysis()
    raw['content_plan'].update(version=version, profile=f'content-plan-v{version}')
    if version == 2:
        raw['content_plan'].pop('context_facts')
    raw['treatment_plan'][0]['treatment'] = treatment
    payload = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, json.dumps(raw), contract_version=version)
    with pytest.raises(AnalysisMergeError, match='лексическая помощь требует'):
        validate_content_plan(payload, {'b1': 'block'}, {'block': 'Echo. The cat sat.'},
                              english_help=english_help)
