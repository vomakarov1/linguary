"""The explicit v3 profile runs through the real pinned runtime and storage."""
from dataclasses import replace
import json
from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.config.settings import ReadingSettings
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.composition import build_application, drain_tasks
from linguary.domain import TaskStatus
from test_book_conditions import store_imported


SOURCE = 'Mara snapped, "Shoo!"'
EXPLANATION = 'Здесь snapped описывает резкую манеру речи.'


def analysis_reply():
    return FakeReply.of_payload(dict(translation_instructions='Сохраняй согласованные имена.', chapter_summary='Mara speaks sharply.',
        glossary_candidates=[], treatment_plan=[dict(source_block_id='b1', treatment='TRANSLATE')],
        before_you_read=[dict(expression='snapped', explanation='[b1] ' + EXPLANATION)],
        content_plan=dict(version=3, profile='content-plan-v3',
            difficulties=[dict(id='d', anchors=[dict(block_id='b1', start=5, end=12, quote='snapped')], owner_block_id='b1',
                kind='vocabulary', priority='useful', help_kind='explanation', required_points=[dict(id='p', description='Explain the manner of speaking')])],
            before_you_read_evidence=[dict(difficulty_id='d', point_id='p', artifact_kind='before_you_read.explanation', source_block_id='b1', artifact_index=0, body_excerpt='[b1] ' + EXPLANATION)],
            context_facts=[dict(id='speaker', kind='speaker', anchors=[dict(block_id='b1', start=0, end=4, quote='Mara')],
                subject_anchor=dict(block_id='b1', start=0, end=4, quote='Mara'), value='Mara', certainty='explicit')])) )


def processing_reply(*, marker=False):
    return FakeReply.of_payload(dict(blocks=[dict(source_block_id='b1', treatment='TRANSLATE',
        translated_text=('[b1] ' if marker else '') + 'Мара резко сказала: «Кыш!»',
        glossary_hints=[dict(original='Shoo', translation='Кыш', note='[b1] Короткий окрик, чтобы прогнать.')])],
        language_notes=[dict(source_block_id='b1', original_excerpt='Shoo', translated_excerpt='Кыш',
            note='[b1] Языковая справка: короткий окрик, чтобы прогнать.')], content_evidence=[]))


def make_v3(root, fake):
    return build_application(settings=AppSettings(
        llm=LLMSettings(model='assigned-v3', api_key='synthetic-key', proxy_mode='direct', max_attempts=1),
        reading=ReadingSettings(reader_level='a2'), storage=StorageSettings(data_root=str(root))),
        llm_client=fake, content_plan_profile='content-plan-v3')


async def test_v3_pinned_runtime_clean_bodies_and_evidence_survive_restart(tmp_path):
    fake = FakeLLMClient([analysis_reply(), processing_reply()])
    app = make_v3(tmp_path / 'data', fake)
    store_imported(app, ((SOURCE,),))
    await app.analysis.start('book-1')
    completed = await drain_tasks(app)
    assert all(task.status == TaskStatus.SUCCEEDED for task in completed), [task.last_error for task in completed]
    with app.unit_of_work.transaction() as transaction:
        analysis, = transaction.analyses.list_for_book('book-1')
        assert analysis.content_plan.version == 3
        assert analysis.before_you_read[0].explanation == EXPLANATION
        assert analysis.content_plan.context_facts[0].subject_anchor.quote == 'Mara'
        assert '[b1]' not in json.dumps(analysis.content_plan.before_you_read_evidence[0].body_excerpt)
    await app.aclose()
    restarted = make_v3(tmp_path / 'data', fake)
    candidate = replace(restarted.settings, llm=replace(restarted.settings.llm, model='changed-model'),
                        reading=ReadingSettings(reader_level='c1'))
    restarted.runtime_settings.publish(restarted.runtime_settings.prepare(candidate), 1)
    try:
        run = restarted.start_processing()(StartProcessingCommand('book-1'))
        await restarted.processing.start('book-1')
        completed = await drain_tasks(restarted)
        assert all(task.status == TaskStatus.SUCCEEDED for task in completed), [task.last_error for task in completed]
        assert all(request.contract_version == 3 for request in fake.requests)
        assert all('learner-quality-v3.1' in request.instructions and 'Для A2' in request.instructions for request in fake.requests)
        assert 'context_facts' in fake.requests[-1].input_text
        with restarted.unit_of_work.transaction() as transaction:
            result, = transaction.results.list_for_run(run.id)
            assert result.translated_text == 'Мара резко сказала: «Кыш!»'
            assert result.glossary_hints[0].original == 'Shoo'
            assert result.language_notes[0].original_excerpt == 'Shoo'
            assert not any('[b1]' in body for body in (result.glossary_hints[0].note, result.language_notes[0].note))
            assert result.content_evidence.version == 3
            assert result.content_evidence.semantic_correctness == 'not_evaluated'
            assert transaction.documents.get('book-1').sections[0].blocks[0].text == SOURCE
            assert run.llm_settings_snapshot.model == 'assigned-v3'
    finally:
        await restarted.aclose()


async def test_v3_translation_service_marker_is_rejected(tmp_path):
    app = make_v3(tmp_path / 'data', FakeLLMClient([analysis_reply(), processing_reply(marker=True)]))
    try:
        store_imported(app, ((SOURCE,),))
        await app.analysis.start('book-1')
        await drain_tasks(app)
        run = app.start_processing()(StartProcessingCommand('book-1'))
        await app.processing.start('book-1')
        completed = await drain_tasks(app)
        assert completed[0].status == TaskStatus.FAILED
        with app.unit_of_work.transaction() as transaction:
            assert transaction.results.list_for_run(run.id) == ()
    finally:
        await app.aclose()
