"""Durable counters, disjoint acceptance, and failure boundaries in the real pipeline."""
from dataclasses import replace
import json
import pytest
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.composition import drain_tasks
from linguary.domain import TaskStatus, LLMCallStage
from test_book_conditions import make_app, store_imported


def analysis_reply(count):
    return FakeReply.of_payload(dict(translation_instructions='Сохраняй смысл.', chapter_summary='They returned home.',
        glossary_candidates=[], treatment_plan=[dict(source_block_id=f'b{index}', treatment='TRANSLATE') for index in range(1, count + 1)], before_you_read=[]))


def translation_reply(count, *, bad=False):
    return FakeReply.of_payload(dict(blocks=[dict(source_block_id=f'b{index}', treatment='TRANSLATE',
        translated_text='«Он вернулся домой.' if bad and index == count else 'Он вернулся домой.', glossary_hints=[])
        for index in range(1, count + 1)], language_notes=[]))


async def prepare(app, *, blocks=2):
    doc = store_imported(app, (tuple('He returned home.' for _ in range(blocks)),))
    await app.analysis.start('book-1')
    completed = await drain_tasks(app)
    assert all(task.status == TaskStatus.SUCCEEDED for task in completed), [task.last_error for task in completed]
    run = app.start_processing()(StartProcessingCommand('book-1'))
    tasks = await app.processing.start('book-1')
    return doc, run, tasks[0]


async def test_rejected_original_never_partially_accepted_and_children_union_complete(tmp_path):
    class CheckingFake(FakeLLMClient):
        async def generate(self, request):
            if request.stage == LLMCallStage.TRANSLATION_UNIT and len(request.ordered_block_ids) == 2:
                with app.unit_of_work.transaction() as transaction:
                    assert transaction.results.list_for_run(run.id) == ()
            return await super().generate(request)
    fake = CheckingFake([analysis_reply(2), translation_reply(2, bad=True), translation_reply(2, bad=True), translation_reply(1), translation_reply(1)])
    app = make_app(tmp_path, fake)
    try:
        doc, run, task = await prepare(app)
        completed = await drain_tasks(app)
        assert completed[-1].status == TaskStatus.SUCCEEDED, completed[-1].last_error
        with app.unit_of_work.transaction() as transaction:
            root, = transaction.work.list_roots('book-1', run_id=run.id)
            assert root.calls_used == 4 and root.status == 'complete'
            scopes = transaction.work.list_scopes(root.id)
            assert sorted(scope.status for scope in scopes) == ['accepted', 'accepted', 'split']
            assert transaction.work.root_coverage_complete(root.id)
            assert len(transaction.results.list_for_run(run.id)) == 2
            calls = [call for call in transaction.llm_calls.list_for_book('book-1') if call.stage == LLMCallStage.TRANSLATION_UNIT]
            assert len(calls) == 4
            assert len({call.attempt_id for call in calls}) == 4
            assert [call.diagnostic_meta['decision'] for call in calls] == ['rejected', 'rejected', 'accepted', 'accepted']
        child_requests = [request for request in fake.requests if request.stage == LLMCallStage.TRANSLATION_UNIT and len(request.ordered_block_ids) == 1]
        assert len(child_requests) == 2
        assert all('Контекст исходной сцены' in request.input_text for request in child_requests)
    finally:
        await app.aclose()


async def test_all_transport_and_format_retries_share_durable_counter(tmp_path):
    from linguary.application.ports.llm import LLMTransportError
    fake = FakeLLMClient([analysis_reply(2), LLMTransportError('HTTP 503'), FakeReply(content='{'), translation_reply(2)])
    app = make_app(tmp_path, fake)
    try:
        doc, run, task = await prepare(app)
        completed = await drain_tasks(app)
        assert completed[0].status == TaskStatus.SUCCEEDED, completed[0].last_error
        with app.unit_of_work.transaction() as transaction:
            root, = transaction.work.list_roots('book-1', run_id=run.id)
            assert root.calls_used == 3
            assert len(transaction.work.list_scopes(root.id)) == 1
            calls = [call for call in transaction.llm_calls.list_for_book('book-1') if call.stage == LLMCallStage.TRANSLATION_UNIT]
            attempts = [transaction.work.get_attempt(call.attempt_id) for call in calls]
            assert [attempt.ordinal for attempt in attempts] == [1, 2, 3]
            assert calls[0].cost is None
    finally:
        await app.aclose()


async def test_save_failure_is_atomic(tmp_path, monkeypatch):
    from linguary.adapters.db.work_state import SqlAlchemyWorkRepository
    fake = FakeLLMClient([analysis_reply(2), translation_reply(2)])
    app = make_app(tmp_path, fake)
    try:
        doc, run, task = await prepare(app)
        original = SqlAlchemyWorkRepository.replace_result_binding
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('failure after binding')
        monkeypatch.setattr(SqlAlchemyWorkRepository, 'replace_result_binding', fail)
        completed = await drain_tasks(app)
        assert completed[0].status == TaskStatus.FAILED
        with app.unit_of_work.transaction() as transaction:
            assert transaction.results.list_for_run(run.id) == ()
            root, = transaction.work.list_roots('book-1', run_id=run.id)
            assert root.calls_used == 1
            assert transaction.work.list_scopes(root.id)[0].status == 'claimed'
            assert transaction.work.get_binding('result', f'{run.id}:{doc.sections[0].blocks[0].id}') is None
            calls = [call for call in transaction.llm_calls.list_for_book('book-1') if call.stage == LLMCallStage.TRANSLATION_UNIT]
            assert calls[0].diagnostic_meta['decision'] == 'unknown'
    finally:
        await app.aclose()


async def test_children_notes_may_exceed_original_group_target(tmp_path):
    from linguary.application.context_builder import notes_budget
    replies = [analysis_reply(2), translation_reply(2, bad=True), translation_reply(2, bad=True)]
    for _ in range(2):
        body = json.loads(translation_reply(1).content)
        body['language_notes'] = [dict(source_block_id='b1', original_excerpt=quote,
            translated_excerpt=None, note=explanation) for quote, explanation in (
                ('He', 'Местоимение обозначает говорящего в третьем лице.'),
                ('returned', 'Прошедшее время: действие возвращения уже произошло.'),
                ('home', 'Направление домой передано без предлога.'))]
        replies.append(FakeReply.of_payload(body))
    fake = FakeLLMClient(replies)
    app = make_app(tmp_path, fake)
    try:
        settings = replace(app.settings, checks=replace(app.settings.checks, language_notes_per_chapter=2))
        app.runtime_settings.publish(app.runtime_settings.prepare(settings), 1)
        doc, run, task = await prepare(app)
        completed = await drain_tasks(app)
        assert completed[0].status == TaskStatus.SUCCEEDED, completed[0].last_error
        children = [r for r in fake.requests if r.stage == LLMCallStage.TRANSLATION_UNIT and len(r.ordered_block_ids) == 1]
        assert len(children) == 2
        assert all('Ориентир количества языковых заметок' in r.input_text for r in children)
        assert all('не больше' not in r.input_text for r in children)
        with app.unit_of_work.transaction() as transaction:
            results = transaction.results.list_for_run(run.id)
            assert [len(r.language_notes) for r in results] == [3, 3]
            assert len({n.id for r in results for n in r.language_notes}) == 6
            root, = transaction.work.list_roots('book-1', run_id=run.id)
            assert sum(len(r.language_notes) for r in results) > notes_budget(unit_chars=32, chapter_chars=32, per_chapter=2)
            assert transaction.work.root_coverage_complete(root.id)
            assert root.calls_used == 4
    finally:
        await app.aclose()


def planned_analysis_reply():
    body = json.loads(analysis_reply(2).content)
    body['content_plan'] = dict(version=3, profile='content-plan-v3',
        difficulties=[dict(id=f'd{i}', anchors=[dict(block_id=f'b{i}', start=0, end=2, quote='He')],
            owner_block_id=f'b{i}', kind='grammar', priority='critical', help_kind='explanation',
            required_points=[dict(id=f'p{j}', description=description) for j, description in enumerate(
                ('Explain the pronoun', 'Explain the past tense', 'Explain the direction'), 1)]) for i in (1, 2)],
        before_you_read_evidence=[], context_facts=[])
    return FakeReply.of_payload(body)


def planned_child_reply(difficulty, *, defect=None):
    body = json.loads(translation_reply(1).content)
    body['language_notes'] = [dict(source_block_id='b1', original_excerpt=quote, translated_excerpt=None,
        note=explanation) for quote, explanation in (
            ('He', 'Местоимение третьего лица обозначает мужчину.'),
            ('returned', 'Прошедшее время: возвращение уже произошло.'),
            ('home', 'Направление домой выражено без предлога.'))]
    body['content_evidence'] = [dict(difficulty_id=difficulty['id'], point_id=difficulty['required_points'][j]['id'],
        artifact_kind='language_note.note', source_block_id='b1', artifact_index=j,
        body_excerpt=note['note']) for j, note in enumerate(body['language_notes'])]
    if defect == 'foreign_difficulty':
        body['content_evidence'][0]['difficulty_id'] = 'unrelated-difficulty'
    elif defect == 'missing':
        body['content_evidence'].pop()
    elif defect == 'foreign':
        body['language_notes'][0]['original_excerpt'] = 'never appeared in source'
    elif defect == 'duplicate':
        body['language_notes'][0]['original_excerpt'] = 'returned'
        body['language_notes'][0]['note'] = body['language_notes'][1]['note']
        body['content_evidence'][0]['body_excerpt'] = body['language_notes'][1]['note']
    return FakeReply.of_payload(body)


@pytest.mark.parametrize('defect', [None, 'missing', 'foreign', 'duplicate', 'foreign_difficulty'])
async def test_excess_notes_do_not_bypass_grounding_dedup_or_coverage(tmp_path, defect):
    from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
    from linguary.adapters.config.settings import CheckSettings
    from linguary.composition import build_application
    bad = json.loads(translation_reply(2, bad=True).content)
    bad['content_evidence'] = []
    replies = [planned_analysis_reply(), FakeReply.of_payload(bad), FakeReply.of_payload(bad)]
    class PlannedFake(FakeLLMClient):
        async def generate(self, request):
            if request.stage == LLMCallStage.TRANSLATION_UNIT and len(request.ordered_block_ids) == 1:
                # The analysis replaces provider IDs with canonical IDs; child prompts
                # keep those IDs while rebinding their source owner to b1.
                marker = 'Обязательные аспекты и точные связанные привязки:\n'
                plan, _ = json.JSONDecoder().raw_decode(request.input_text.split(marker, 1)[1])
                difficulty, = plan['difficulties']
                assert difficulty['owner_block_id'] == 'b1'
                assert difficulty['id'] not in {'d1', 'd2'}
                self.replies = [*self.replies, planned_child_reply(difficulty, defect=defect)]
            return await super().generate(request)
    fake = PlannedFake(replies)
    app = build_application(settings=AppSettings(
        llm=LLMSettings(model='pinned-v3', api_key='synthetic-secret', proxy_mode='direct'),
        checks=CheckSettings(language_notes_per_chapter=2), storage=StorageSettings(data_root=str(tmp_path / 'data'))),
        llm_client=fake, content_plan_profile='content-plan-v3')
    try:
        doc, run, task = await prepare(app)
        completed = await drain_tasks(app)
        with app.unit_of_work.transaction() as transaction:
            root, = transaction.work.list_roots('book-1', run_id=run.id)
            results = transaction.results.list_for_run(run.id)
            if defect:
                assert completed[0].status == TaskStatus.FAILED
                assert not transaction.work.root_coverage_complete(root.id)
                assert results == ()
                errors = [call.error.message if call.error else '' for call in transaction.llm_calls.list_for_book('book-1')]
                expected = {
                    'missing': 'не все обязательные аспекты',
                    'foreign': 'цитата языковой заметки отсутствует',
                    'duplicate': 'повтор языковой заметки',
                    'foreign_difficulty': 'свидетельство относится к чужой трудности',
                }[defect]
                assert any(expected in error for error in errors), errors
            else:
                assert completed[0].status == TaskStatus.SUCCEEDED, [(c.diagnostic_meta, c.error) for c in transaction.llm_calls.list_for_book('book-1')]
                assert transaction.work.root_coverage_complete(root.id)
                assert [len(r.language_notes) for r in results] == [3, 3]
                assert all(len(r.content_evidence.evidence) == 3 for r in results)
                assert all(r.content_evidence.semantic_correctness == 'not_evaluated' for r in results)
    finally:
        await app.aclose()
