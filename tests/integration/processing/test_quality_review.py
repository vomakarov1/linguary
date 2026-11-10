from dataclasses import replace
from decimal import Decimal
import json
import pytest
from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.llm import FakeReply
from correction_fakes import CorrectionAwareFakeLLMClient as FakeLLMClient
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
            raise LLMResponseFormatError('invalid schema', raw_content='{"bad": true}', usage=TokenUsage(5, 5), cost=Decimal('0.0001'))
        payload = dict(findings=[], checked_artifacts=[], missing_artifacts=[])
        if self.mode == 'missing':
            payload['missing_artifacts'] = ['a' * 64]
        if self.mode == 'omitted':
            payload['checked_artifacts'] = ['a' * 64]
        if self.mode in ('finding', 'bad-anchor'):
            artifact = next(a for a in manifest['artifacts'] if a['artifact_kind'] == 'result')
            payload['findings'] = [dict(block_ids=artifact['block_ids'], artifact_kind='result',
                severity='major', issue='Possible tense error',
                evidence_anchors=[dict(block_id=artifact['block_ids'][0],
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
        assert json.loads(fake.requests[-1].input_text)['scope'] == ['b1']
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


async def test_regeneration_after_reservation_cancels_unsent_attempt(tmp_path, monkeypatch):
    app, fake, doc, run = await make_review_app(tmp_path)
    try:
        created = await app.quality_review.start(command(app))
        reserve = app.quality_review.budget.reserve
        attempts = []

        def reserve_then_regenerate(*args, **kwargs):
            attempt = reserve(*args, **kwargs)
            attempts.append(attempt.id)
            with app.unit_of_work.transaction() as tx:
                current = tx.results.list_for_run(run.id)[0]
                tx.results.replace_results(run.id, (replace(current, translated_text='Другой перевод.'),))
                tx.work.stale_reviews('book-1')
            return attempt

        monkeypatch.setattr(app.quality_review.budget, 'reserve', reserve_then_regenerate)
        finished = await drain_tasks(app)
        assert finished[0].status == TaskStatus.SUCCEEDED, finished[0].last_error
        result = app.quality_review.get('book-1', created['review_id'])
        assert result['state'] == 'stale' and result['spent'] == result['reserved'] == 0
        assert fake.review_calls == 0
        with app.unit_of_work.transaction() as tx:
            assert len(attempts) == 1
            assert tx.work.get_attempt(attempts[0]).state == 'cancelled'
            assert tx.work.get_reservation(attempts[0]).state == 'released'
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


async def test_quality_review_passes_previous_validation_error_into_repair_turn(tmp_path):
    from hashlib import sha256
    from linguary.application.quality_context import INSTRUCTIONS

    app, fake, doc, run = await make_review_app(tmp_path)
    try:
        fake.mode = 'schema'
        created = await app.quality_review.start(command(app))
        finished = await drain_tasks(app)
        assert all(t.status == TaskStatus.SUCCEEDED for t in finished), [t.last_error for t in finished]
        review_requests = [r for r in fake.requests if r.kind == LLMResponseKind.QUALITY_REVIEW]
        assert len(review_requests) == 2
        first_req, second_req = review_requests
        assert first_req.instructions == INSTRUCTIONS
        assert second_req.instructions == INSTRUCTIONS
        assert first_req.repair_turns == ()
        assert first_req.repair_instructions == ()
        assert len(second_req.repair_turns) == 1
        assert second_req.repair_turns[0][0] == '{"bad": true}'
        assert 'invalid schema' in second_req.repair_turns[0][1]
        assert len(second_req.repair_instructions) == 1
        assert 'invalid schema' in second_req.repair_instructions[0]
        assert second_req.call_context.prompt_hash == sha256(INSTRUCTIONS.encode()).hexdigest()
        assert json.loads(second_req.input_text) == json.loads(first_req.input_text)
    finally:
        await app.aclose()


async def test_quality_review_executes_chunked_manifest_and_keeps_succeeded_task_with_unknown_state_on_transport_error(tmp_path):
    from linguary.application.use_cases.approve_glossary import StartProcessingCommand
    from test_book_conditions import store_imported

    class ChunkAwareReviewFake(ReviewFake):
        fail_transport_on_review_call: int | None = None

        async def generate(self, request):
            if (
                request.kind == LLMResponseKind.QUALITY_REVIEW
                and self.fail_transport_on_review_call is not None
                and self.review_calls + 1 == self.fail_transport_on_review_call
            ):
                self.review_calls += 1
                self.requests.append(request)
                raise LLMTransportError('network dropped on second chunk')
            return await super().generate(request)

    def bounded_tokens(request):
        parsed = json.loads(request.input_text)
        if len(parsed.get('scope') or ()) > 1:
            raise WorkStateError('полный запрос превышает подтверждённую границу')
        return 1000

    small_route = QualityRoute(
        'review-model',
        'test-route',
        'fixture-v1',
        Decimal('1'),
        Decimal('2'),
        Decimal('0'),
        100,
        bounded_tokens,
        'fixture-verification-v1',
        True,
    )

    fake = ChunkAwareReviewFake(
        [
            analysis_reply(2),
            analysis_reply(1),
            translation_reply(2),
            translation_reply(1),
        ]
    )
    app = build_application(
        settings=AppSettings(
            llm=LLMSettings(model='source-model', api_key='synthetic-secret'),
            storage=StorageSettings(data_root=str(tmp_path / 'data')),
        ),
        llm_client=fake,
        quality_routes={('test-route', 'review-model'): small_route},
    )
    try:
        store_imported(
            app,
            (
                ('He returned home.', 'He waited outside.'),
                ('He opened the door.',),
            ),
        )
        await app.analysis.start('book-1')
        analysed = await drain_tasks(app)
        assert all(t.status == TaskStatus.SUCCEEDED for t in analysed)
        app.start_processing()(StartProcessingCommand('book-1'))
        await app.processing.start('book-1')
        translated = await drain_tasks(app)
        assert all(t.status == TaskStatus.SUCCEEDED for t in translated)

        manifest = app.quality_review.preview('book-1')
        chunks = app.quality_review._chunks_for_route(manifest, small_route)
        assert len(chunks) == 3
        assert sum(len(c['scope']) for c in chunks) == 3

        impossible_route = replace(
            small_route,
            input_token_bound=lambda _req: (_ for _ in ()).throw(
                WorkStateError('полный запрос превышает подтверждённую границу')
            ),
        )
        with pytest.raises(WorkStateError, match='полный запрос превышает подтверждённую границу'):
            app.quality_review._chunks_for_route(manifest, impossible_route)

        # Sum of 3 chunks (each 1000 in @ $1/M + 100 out @ $2/M = 1200 microusd -> 3 * 1200 = 3600 microusd)
        assert manifest['route_estimates']['test-route:review-model'] == 3600
        with pytest.raises(WorkStateError, match='budget'):
            await app.quality_review.start(
                command(app, key='too-small-budget', budget=3599),
                require_available=True,
            )

        # Successful chunked execution deduplicates shared chapter analysis artifact_sha
        fake.mode = 'finding'
        created = await app.quality_review.start(
            command(app, key='chunked-ok', budget=10000),
            require_available=True,
        )
        finished = await drain_tasks(app)
        assert all(t.status == TaskStatus.SUCCEEDED for t in finished), [t.last_error for t in finished]
        assert fake.review_calls == 3

        result = app.quality_review.get('book-1', created['review_id'])
        assert result['state'] == 'completed-with-findings'
        assert result['spent'] == 300 and result['reserved'] == 0
        assert len(result['findings']) == 3
        checked = result['coverage']['checked']
        assert len(checked) == len(set(checked)) == len(manifest['artifacts'])

        # Transport error after mark_sent on chunk 2 finishes task as SUCCEEDED with 'unknown' review state
        fake.review_calls = 0
        fake.fail_transport_on_review_call = 2
        created_unknown = await app.quality_review.start(
            command(app, key='chunked-transport', budget=10000),
            require_available=True,
        )
        finished_unknown = await drain_tasks(app)
        assert len(finished_unknown) == 1
        assert finished_unknown[0].status == TaskStatus.SUCCEEDED
        assert finished_unknown[0].last_error is None
        assert fake.review_calls == 2

        unknown_state = app.quality_review.get('book-1', created_unknown['review_id'])
        assert unknown_state['state'] == 'unknown'
        assert unknown_state['spent'] == 100
        assert unknown_state['reserved'] > 0
    finally:
        await app.aclose()


async def test_chunked_quality_review_resumes_without_rebilling_checked_chunks_and_populates_missing_on_partial_stop(tmp_path, monkeypatch):
    from linguary.application.use_cases.approve_glossary import StartProcessingCommand
    from linguary.application.use_cases.retry_task import RetryTaskCommand
    from test_book_conditions import store_imported

    class SelectiveReviewFake(ReviewFake):
        fail_mode_on_call: dict[int, str]

        def __init__(self, replies):
            super().__init__(replies)
            self.fail_mode_on_call = {}

        async def generate(self, request):
            if request.kind != LLMResponseKind.QUALITY_REVIEW:
                return await super().generate(request)
            call_no = self.review_calls + 1
            override = self.fail_mode_on_call.get(call_no)
            prev_mode = self.mode
            try:
                if override is not None:
                    self.mode = override
                return await super().generate(request)
            finally:
                self.mode = prev_mode

    def bounded_tokens(request):
        parsed = json.loads(request.input_text)
        if len(parsed.get('scope') or ()) > 1:
            raise WorkStateError('полный запрос превышает подтверждённую границу')
        return 1000

    small_route = QualityRoute(
        'review-model',
        'test-route',
        'fixture-v1',
        Decimal('1'),
        Decimal('2'),
        Decimal('0'),
        100,
        bounded_tokens,
        'fixture-verification-v1',
        True,
    )

    fake = SelectiveReviewFake(
        [
            analysis_reply(2),
            analysis_reply(1),
            translation_reply(2),
            translation_reply(1),
        ]
    )
    app = build_application(
        settings=AppSettings(
            llm=LLMSettings(model='source-model', api_key='synthetic-secret'),
            storage=StorageSettings(data_root=str(tmp_path / 'data')),
        ),
        llm_client=fake,
        quality_routes={('test-route', 'review-model'): small_route},
    )
    try:
        store_imported(
            app,
            (
                ('He returned home.', 'He waited outside.'),
                ('He opened the door.',),
            ),
        )
        await app.analysis.start('book-1')
        assert all(t.status == TaskStatus.SUCCEEDED for t in await drain_tasks(app))
        app.start_processing()(StartProcessingCommand('book-1'))
        await app.processing.start('book-1')
        assert all(t.status == TaskStatus.SUCCEEDED for t in await drain_tasks(app))

        manifest = app.quality_review.preview('book-1')
        all_manifest_shas = {a['artifact_sha'] for a in manifest['artifacts']}
        fake.mode = 'finding'

        # 1. Resume after interruption following chunk 1 without rebilling or losing findings
        orig_reserve = app.quality_review.budget.reserve
        reserve_calls = 0

        def interrupt_on_second_chunk(*args, **kwargs):
            nonlocal reserve_calls
            reserve_calls += 1
            if reserve_calls == 2:
                raise ConnectionError('simulated transient worker interruption before chunk 2')
            return orig_reserve(*args, **kwargs)

        monkeypatch.setattr(app.quality_review.budget, 'reserve', interrupt_on_second_chunk)
        resumable = await app.quality_review.start(command(app, key='resume-chunked', budget=2500))
        interrupted_tasks = await drain_tasks(app)
        assert len(interrupted_tasks) == 1
        assert interrupted_tasks[0].status == TaskStatus.FAILED
        assert fake.review_calls == 1

        mid_state = app.quality_review.get('book-1', resumable['review_id'])
        assert mid_state['state'] == 'running'
        assert mid_state['spent'] == 100
        assert len(mid_state['findings']) == 1
        assert len(mid_state['coverage']['checked']) == 3

        monkeypatch.setattr(app.quality_review.budget, 'reserve', orig_reserve)
        app.retry_task(RetryTaskCommand(task_id=interrupted_tasks[0].id))
        resumed_tasks = await drain_tasks(app)
        assert len(resumed_tasks) == 1 and resumed_tasks[0].status == TaskStatus.SUCCEEDED
        # Only chunks 2 and 3 were sent on retry (total 3 calls, not 4; budget 2500 would limit if chunk 1 were rebilled)
        assert fake.review_calls == 3
        final_resumed = app.quality_review.get('book-1', resumable['review_id'])
        assert final_resumed['state'] == 'completed-with-findings'
        assert final_resumed['spent'] == 300
        assert len(final_resumed['findings']) == 3
        assert set(final_resumed['coverage']['checked']) == all_manifest_shas
        assert final_resumed['coverage']['missing'] == []

        # 2. Partial stop on `limit` after chunk 1 populates `coverage.missing` with remaining artifacts
        fake.review_calls = 0
        limited = await app.quality_review.start(command(app, key='partial-limit', budget=1250))
        assert all(t.status == TaskStatus.SUCCEEDED for t in await drain_tasks(app))
        limit_state = app.quality_review.get('book-1', limited['review_id'])
        assert limit_state['state'] == 'limit'
        assert len(limit_state['findings']) == 1
        assert len(limit_state['coverage']['checked']) == 3
        assert len(limit_state['coverage']['missing']) > 0
        assert set(limit_state['coverage']['checked']) & set(limit_state['coverage']['missing']) == set()
        assert set(limit_state['coverage']['checked']) | set(limit_state['coverage']['missing']) == all_manifest_shas

        # 3. Partial stop on `unknown` (transport error on chunk 2) preserves chunk 1 and populates `missing`
        fake.review_calls = 0
        fake.fail_mode_on_call = {2: 'transport'}
        unknown_rev = await app.quality_review.start(command(app, key='partial-unknown', budget=10000))
        assert all(t.status == TaskStatus.SUCCEEDED for t in await drain_tasks(app))
        unk_state = app.quality_review.get('book-1', unknown_rev['review_id'])
        assert unk_state['state'] == 'unknown'
        assert len(unk_state['findings']) == 1
        assert len(unk_state['coverage']['checked']) == 3
        assert len(unk_state['coverage']['missing']) > 0
        assert set(unk_state['coverage']['checked']) | set(unk_state['coverage']['missing']) == all_manifest_shas

        # 4. Partial stop on `failed` (3 invalid responses on chunk 2) preserves chunk 1 and populates `missing`
        fake.review_calls = 0
        fake.fail_mode_on_call = {2: 'bad-anchor', 3: 'bad-anchor', 4: 'bad-anchor'}
        failed_rev = await app.quality_review.start(command(app, key='partial-failed', budget=10000))
        assert all(t.status == TaskStatus.SUCCEEDED for t in await drain_tasks(app))
        fail_state = app.quality_review.get('book-1', failed_rev['review_id'])
        assert fail_state['state'] == 'partial'
        assert fail_state['reason'] == 'invalid_response'
        assert len(fail_state['findings']) == 1
        assert len(fail_state['coverage']['checked']) == 3
        assert len(fail_state['coverage']['missing']) > 0
        assert set(fail_state['coverage']['checked']) | set(fail_state['coverage']['missing']) == all_manifest_shas
    finally:
        await app.aclose()


async def test_quality_review_populates_missing_after_reconciled_unknown_and_preserves_unknown_reason(tmp_path, monkeypatch):
    app, fake, doc, run = await make_review_app(tmp_path)
    try:
        manifest = app.quality_review.preview('book-1')
        all_manifest_shas = [a['artifact_sha'] for a in manifest['artifacts']]

        # 1. When budget.reserve fails because the review was reconciled to `unknown`,
        #    run_task checks _stopped_state first, preserves state='unknown' and reason=None (never 'budget'),
        #    and ensures coverage.missing has all unchecked manifest artifact_shas with partial=True.
        created = await app.quality_review.start(command(app, key='reconciled-during-reserve', budget=10000))
        review_id = created['review_id']
        orig_reserve = app.quality_review.budget.reserve

        def reconcile_to_unknown_then_fail(rev_id, estimate, **kwargs):
            att = orig_reserve(rev_id, estimate, **kwargs)
            assert app.quality_review.budget.mark_sent(att.id) is True
            app.quality_review.budget.reconcile()
            raise WorkStateError('бюджет исчерпан или состояние вычитки изменилось')

        monkeypatch.setattr(app.quality_review.budget, 'reserve', reconcile_to_unknown_then_fail)
        finished = await drain_tasks(app)
        assert len(finished) == 1 and finished[0].status == TaskStatus.SUCCEEDED
        monkeypatch.setattr(app.quality_review.budget, 'reserve', orig_reserve)

        state = app.quality_review.get('book-1', review_id)
        assert state['state'] == 'unknown'
        assert state['reason'] is None
        assert state['coverage']['checked'] == []
        assert state['coverage']['missing'] == all_manifest_shas
        with app.unit_of_work.transaction() as tx:
            progress = tx.work.review_progress(review_id)
            assert progress['partial'] is True
            assert progress['reason'] is None
            rev_after_stop = tx.work.get_review(review_id)

        # Calling run_task again on the already-accounted `unknown` review does not bump revision or mutate it
        await app.quality_review.run_task(finished[0])
        with app.unit_of_work.transaction() as tx:
            assert tx.work.get_review(review_id) == rev_after_stop

        # 2. Entering run_task on an already stopped `unknown` review that still has unaccounted
        #    manifest artifact_sha items after a crash populates `missing` with partial=True,
        #    and subsequent run_task calls do not bump revision.
        created2 = await app.quality_review.start(command(app, key='crashed-after-settle-unknown', budget=10000))
        review2_id = created2['review_id']
        with app.unit_of_work.transaction() as tx:
            rev2 = tx.work.get_review(review2_id)
        att2 = app.quality_review.budget.reserve(
            review2_id,
            1200,
            price=route().snapshot(),
            expected_revision=rev2.revision,
        )
        assert app.quality_review.budget.mark_sent(att2.id) is True
        app.quality_review.budget.settle(att2.id, None)
        app.quality_review._finish_attempt(att2.id, 'unknown')
        with app.unit_of_work.transaction() as tx:
            assert tx.work.get_review(review2_id).state == 'unknown'
            assert tx.work.review_progress(review2_id).get('missing') is None

        finished2 = await drain_tasks(app)
        assert len(finished2) == 1 and finished2[0].status == TaskStatus.SUCCEEDED
        state2 = app.quality_review.get('book-1', review2_id)
        assert state2['state'] == 'unknown'
        assert state2['reason'] is None
        assert state2['coverage']['checked'] == []
        assert state2['coverage']['missing'] == all_manifest_shas
        with app.unit_of_work.transaction() as tx:
            assert tx.work.review_progress(review2_id)['partial'] is True
            rev2_accounted = tx.work.get_review(review2_id)

        await app.quality_review.run_task(finished2[0])
        with app.unit_of_work.transaction() as tx:
            assert tx.work.get_review(review2_id) == rev2_accounted

        # 3. Stopping due to `stale` also populates `missing` with remaining unchecked artifact_sha items and partial=True
        created3 = await app.quality_review.start(command(app, key='stale-populates-missing', budget=10000))
        review3_id = created3['review_id']
        with app.unit_of_work.transaction() as tx:
            current_res = tx.results.list_for_run(run.id)[0]
            tx.results.replace_results(run.id, (replace(current_res, translated_text='Изменённый перевод до запуска.'),))
        finished3 = await drain_tasks(app)
        assert len(finished3) == 1 and finished3[0].status == TaskStatus.SUCCEEDED
        state3 = app.quality_review.get('book-1', review3_id)
        assert state3['state'] == 'stale'
        assert state3['reason'] == 'manifest_changed'
        assert state3['coverage']['checked'] == []
        assert state3['coverage']['missing'] == all_manifest_shas
        with app.unit_of_work.transaction() as tx:
            assert tx.work.review_progress(review3_id)['partial'] is True
            rev3_accounted = tx.work.get_review(review3_id)

        await app.quality_review.run_task(finished3[0])
        with app.unit_of_work.transaction() as tx:
            assert tx.work.get_review(review3_id) == rev3_accounted
    finally:
        await app.aclose()

