from dataclasses import replace
from decimal import Decimal
import json
import pytest
from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application.ports.llm import LLMResponseKind, LLMResponseFormatError, LLMTransportError
from linguary.application.quality_budget import QualityRoute
from linguary.application.use_cases.quality_review import StartQualityReview
from linguary.composition import build_application, drain_tasks
from linguary.domain import LLMCallStage, TaskStatus, TokenUsage
from linguary.domain.book_conditions import WorkStateError
from test_recovery_commit import prepare, analysis_reply, translation_reply


def route():
    return QualityRoute('review-model', 'test-route', 'fixture-v1', Decimal('1'), Decimal('2'), Decimal('0'),
        100, lambda request: 1000 + len(request.input_text.encode()), 'fixture-verification-v1', True)


class ReviewFake(FakeLLMClient):
    mode = 'valid'
    review_calls = 0
    callback = None

    async def generate(self, request):
        if request.kind != LLMResponseKind.QUALITY_REVIEW:
            return await super().generate(request)
        self.review_calls += 1
        manifest = json.loads(request.input_text)
        if self.callback:
            self.callback()
        if self.mode == 'transport':
            self.requests.append(request)
            raise LLMTransportError('unknown send')
        if self.mode == 'schema' and self.review_calls == 1:
            self.requests.append(request)
            raise LLMResponseFormatError('invalid schema', usage=TokenUsage(5, 5), cost=Decimal('0.0001'))
        payload = dict(findings=[], checked_artifacts=[a['artifact_sha'] for a in manifest['artifacts']], missing_artifacts=[])
        if self.mode == 'missing':
            payload['missing_artifacts'].append(payload['checked_artifacts'].pop())
        if self.mode == 'omitted':
            payload['checked_artifacts'].pop()
        if self.mode in ('finding', 'bad-anchor'):
            artifact = next(a for a in manifest['artifacts'] if a['artifact_kind'] == 'result')
            payload['findings'] = [dict(block_ids=artifact['block_ids'], artifact_kind='result',
                artifact_sha=artifact['artifact_sha'], severity='major', issue='Possible tense error',
                evidence_anchors=[dict(block_id=artifact['block_ids'][0], start=0, end=2,
                    quote='XX' if self.mode == 'bad-anchor' else 'He')], suggestion='Check tense', uncertainty='probable')]
        self.default = FakeReply.of_payload(payload)
        response = await super().generate(request)
        return replace(response, cost=Decimal('0.0001'), usage=TokenUsage(5, 5))


async def make_review_app(tmp_path, *, complete=True, priced=True):
    fake = ReviewFake([analysis_reply(2), translation_reply(2)])
    app = build_application(settings=AppSettings(llm=LLMSettings(model='source-model', api_key='synthetic-secret'),
        storage=StorageSettings(data_root=str(tmp_path / 'data'))), llm_client=fake,
        quality_routes={('test-route', 'review-model'): route()} if priced else {})
    doc, run, task = await prepare(app)
    if complete:
        assert (await drain_tasks(app))[0].status == TaskStatus.SUCCEEDED
    else:
        with app.unit_of_work.transaction() as tx:
            tx.tasks.delete(task.id)
        # Keep the fake sequence aligned when processing is deliberately absent.
        fake.replies = fake.replies[:1]
    return app, fake, doc, run


def command(app, *, scope='all', key='review-1', budget=100000):
    manifest = app.quality_review.preview('book-1', scope)
    return StartQualityReview('book-1', key, 'review-model', 'test-route', budget, scope, manifest['manifest_sha'])


async def test_default_off_no_calls_and_explicit_queue_is_idempotent(tmp_path):
    app, fake, doc, run = await make_review_app(tmp_path)
    try:
        count = len(fake.requests)
        assert app.quality_review.get('book-1')['state'] == 'not-requested'
        assert await drain_tasks(app) == ()
        assert len(fake.requests) == count
        cmd = command(app)
        first = await app.quality_review.start(cmd)
        assert (await app.quality_review.start(cmd))['review_id'] == first['review_id']
        with pytest.raises(WorkStateError, match='request_key_conflict'):
            await app.quality_review.start(replace(cmd, budget_microusd=cmd.budget_microusd + 1))
        completed = await drain_tasks(app)
        assert completed[0].status == TaskStatus.SUCCEEDED, completed[0].last_error
        projection = app.quality_review.get('book-1', first['review_id'])
        assert projection['state'] == 'completed-without-findings'
        assert projection['spent'] == 100 and projection['reserved'] == 0
        assert fake.review_calls == 1
        with app.unit_of_work.transaction() as tx:
            call, = [c for c in tx.llm_calls.list_for_book('book-1') if c.stage == LLMCallStage.QUALITY_REVIEW]
            attempt = tx.work.get_attempt(call.attempt_id)
            assert attempt.review_id == first['review_id'] and attempt.state == 'accepted'
            assert call.diagnostic_meta['decision'] == 'accepted'
            assert tx.work.get_reservation(attempt.id).state == 'settled'
    finally:
        await app.aclose()


@pytest.mark.parametrize('mode,state', [('finding', 'completed-with-findings'), ('missing', 'partial'),
    ('omitted', 'partial'), ('bad-anchor', 'partial'), ('transport', 'unknown')])
async def test_review_only_returns_findings_and_validates_coverage(tmp_path, mode, state):
    app, fake, doc, run = await make_review_app(tmp_path)
    try:
        with app.unit_of_work.transaction() as tx:
            before_results = tx.results.list_for_run(run.id)
            before_analysis = tx.analyses.list_for_book('book-1')
            before_glossary = tx.glossary.list_for_book('book-1')
        fake.mode = mode
        created = await app.quality_review.start(command(app))
        finished = await drain_tasks(app)
        assert all(t.status == TaskStatus.SUCCEEDED for t in finished), [t.last_error for t in finished]
        result = app.quality_review.get('book-1', created['review_id'])
        assert result['state'] == state, result
        assert result['semantic_correctness'] == 'not_evaluated'
        with app.unit_of_work.transaction() as tx:
            assert tx.results.list_for_run(run.id) == before_results
            assert tx.analyses.list_for_book('book-1') == before_analysis
            assert tx.glossary.list_for_book('book-1') == before_glossary
        if mode == 'transport':
            assert fake.review_calls == 1 and result['reserved'] > 0
        if mode in ('omitted', 'bad-anchor'):
            assert fake.review_calls == 3 and not result['findings']
    finally:
        await app.aclose()


async def test_explicit_scope_and_partial_manifest(tmp_path):
    app, fake, doc, run = await make_review_app(tmp_path, complete=False)
    try:
        block_id = doc.sections[0].blocks[0].id
        created = await app.quality_review.start(command(app, scope=(block_id,)))
        finished = await drain_tasks(app)
        assert all(t.status == TaskStatus.SUCCEEDED for t in finished), [t.last_error for t in finished]
        result = app.quality_review.get('book-1', created['review_id'])
        assert result['state'] == 'partial'
        assert result['coverage']['checked']
        assert json.loads(fake.requests[-1].input_text)['scope'] == [block_id]
    finally:
        await app.aclose()


async def test_every_retry_reserves_and_format_error_settles_usage(tmp_path):
    app, fake, doc, run = await make_review_app(tmp_path)
    try:
        fake.mode = 'schema'
        created = await app.quality_review.start(command(app))
        finished = await drain_tasks(app)
        assert all(t.status == TaskStatus.SUCCEEDED for t in finished), [t.last_error for t in finished]
        result = app.quality_review.get('book-1', created['review_id'])
        assert result['state'] == 'completed-without-findings' and result['spent'] == 200
        calls = [r for r in fake.requests if r.kind == LLMResponseKind.QUALITY_REVIEW]
        assert len({r.call_context.attempt_id for r in calls}) == 2
        with app.unit_of_work.transaction() as tx:
            assert all(tx.work.get_reservation(r.call_context.attempt_id).state == 'settled' for r in calls)
    finally:
        await app.aclose()


async def test_unpriced_model_has_no_request(tmp_path):
    app, fake, doc, run = await make_review_app(tmp_path, priced=False)
    try:
        created = await app.quality_review.start(command(app))
        finished = await drain_tasks(app)
        assert all(t.status == TaskStatus.SUCCEEDED for t in finished), [t.last_error for t in finished]
        assert fake.review_calls == 0
        assert app.quality_review.get('book-1', created['review_id'])['reason'] == 'price_unavailable'
    finally:
        await app.aclose()


async def test_regeneration_during_review_stales(tmp_path):
    app, fake, doc, run = await make_review_app(tmp_path)
    try:
        def change_body():
            with app.unit_of_work.transaction() as tx:
                current = tx.results.list_for_run(run.id)[0]
                tx.results.replace_results(run.id, (replace(current, translated_text='Он снова вернулся домой.'),))
                tx.work.stale_reviews('book-1')
        fake.callback = change_body
        created = await app.quality_review.start(command(app))
        finished = await drain_tasks(app)
        assert all(t.status == TaskStatus.SUCCEEDED for t in finished), [t.last_error for t in finished]
        result = app.quality_review.get('book-1', created['review_id'])
        assert result['state'] == 'stale' and result['spent'] == 100
    finally:
        await app.aclose()


async def test_local_registry_is_used_by_build_application(tmp_path, monkeypatch):
    from linguary.adapters.config.quality_routes import PROTOCOL
    path = tmp_path / 'routes.json'
    path.write_text(json.dumps(dict(schema_version=1, routes=[dict(route_id='test-route', model='review-model',
        base_url=LLMSettings().base_url, price_version='synthetic-v1', currency='USD',
        input_per_mtok='1', output_per_mtok='2', max_fee_usd='0', max_output_tokens=80,
        max_reasoning_tokens=20, max_completion_tokens=100, supports_max_completion_tokens=True,
        completion_includes_reasoning=True, input_bound=dict(protocol=PROTOCOL,
            max_request_bytes=1000000, max_input_tokens=100000, proof_method='Synthetic test proof only',
            proof_version='fixture-v1', evidence_uri='urn:test:quality'))])))
    monkeypatch.setenv('LINGUARY_QUALITY_ROUTES', str(path))
    fake = ReviewFake([analysis_reply(2), translation_reply(2)])
    app = build_application(settings=AppSettings(llm=LLMSettings(model='source-model', api_key='synthetic-secret'),
        storage=StorageSettings(data_root=str(tmp_path / 'data'))), llm_client=fake)
    try:
        await prepare(app)
        await drain_tasks(app)
        created = await app.quality_review.start(command(app, budget=1000000))
        finished = await drain_tasks(app)
        assert all(t.status == TaskStatus.SUCCEEDED for t in finished)
        assert app.quality_review.get('book-1', created['review_id'])['state'] == 'completed-without-findings'
        assert fake.review_calls == 1 and fake.requests[-1].max_completion_tokens == 100
    finally:
        await app.aclose()


async def test_changed_route_attestation_cannot_send_queued_review(tmp_path):
    app, fake, doc, run = await make_review_app(tmp_path)
    try:
        created = await app.quality_review.start(command(app))
        app.quality_review.routes['test-route', 'review-model'] = replace(route(), price_version='changed')
        await drain_tasks(app)
        assert fake.review_calls == 0
        assert app.quality_review.get('book-1', created['review_id'])['reason'] == 'price_unavailable'
    finally:
        await app.aclose()


async def test_limit_before_send_and_over_estimate_stop(tmp_path):
    app, fake, doc, run = await make_review_app(tmp_path)
    try:
        created = await app.quality_review.start(command(app, budget=1))
        await drain_tasks(app)
        assert fake.review_calls == 0
        assert app.quality_review.get('book-1', created['review_id'])['state'] == 'limit'
        app.quality_review.routes['test-route', 'review-model'] = replace(route(), input_per_mtok=Decimal('0'),
            output_per_mtok=Decimal('0'), fee=Decimal('0.000001'))
        created = await app.quality_review.start(command(app, key='over-estimate'))
        await drain_tasks(app)
        result = app.quality_review.get('book-1', created['review_id'])
        assert fake.review_calls == 1 and result['state'] == 'limit'
        assert result['spent'] == 100 and result['reserved'] == 0
    finally:
        await app.aclose()


async def test_stale_detects_unselected_artifact_change_without_explicit_marker(tmp_path):
    app, fake, doc, run = await make_review_app(tmp_path)
    try:
        def change_body():
            with app.unit_of_work.transaction() as tx:
                current = tx.results.list_for_run(run.id)[1]
                tx.results.replace_results(run.id, (replace(current, translated_text='Иной текст.'),))
        fake.callback = change_body
        block = doc.sections[0].blocks[0].id
        created = await app.quality_review.start(command(app, scope=(block,)))
        await drain_tasks(app)
        result = app.quality_review.get('book-1', created['review_id'])
        assert result['state'] == 'stale' and result['spent'] == 100
    finally:
        await app.aclose()


@pytest.mark.parametrize('mode,budget,state', [('transport', 100000, 'unknown'),
    ('valid', 1, 'limit'), ('valid', 100000, 'completed-without-findings')])
async def test_repeated_terminal_review_task_does_not_send_or_reserve(tmp_path, mode, budget, state):
    app, fake, doc, run = await make_review_app(tmp_path)
    try:
        fake.mode = mode
        created = await app.quality_review.start(command(app, budget=budget))
        finished = await drain_tasks(app)
        assert app.quality_review.get('book-1', created['review_id'])['state'] == state
        before_calls = fake.review_calls
        with app.unit_of_work.transaction() as tx:
            before = tx.work.get_review(created['review_id'])
            before_attempts = tx.work.review_attempt_count(created['review_id'])
        await app.quality_review.run_task(finished[0])
        assert fake.review_calls == before_calls
        with app.unit_of_work.transaction() as tx:
            assert tx.work.get_review(created['review_id']) == before
            assert tx.work.review_attempt_count(created['review_id']) == before_attempts
    finally:
        await app.aclose()
