"""Network-free book/recovery/review/export contract, without a semantic quality claim."""
from dataclasses import replace
from decimal import Decimal
import asyncio
import hashlib
import json
from pathlib import Path
import socket

import httpx
import pytest
from pipeline_support import exported_documents
from test_content_plan_pipeline import import_story, analysis_payload, processing_payload, BOOK

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.config.settings import ReadingSettings
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.adapters.validation import EpubCheckSettings, EpubCheckValidator
from linguary.application.ports.llm import LLMResponseKind
from linguary.application.recovery import dependency_intervals, local_failure_signature
from linguary.application.reader_text import ReaderTextPolicy, ReaderTextError
from linguary.application.use_cases.approve_glossary import GlossaryDecision, GlossaryDecisionCommand, StartProcessingCommand
from linguary.application.use_cases.export_book import ExportBookError
from linguary.application.use_cases.quality_review import StartQualityReview
from linguary.application.quality_budget import QualityRoute
from linguary.composition import build_application, drain_tasks
from linguary.domain import LLMCallStage, TaskStatus, TokenUsage
from linguary.domain.content_plan import ContentPlan, Difficulty, RequiredPoint, SourceAnchor
from linguary.domain.llm_log import Violation


FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures/learner_quality'
DATA = json.loads((FIXTURES / 'stories.json').read_text())
STORY = DATA['scene']


@pytest.fixture(autouse=True)
def network_forbidden(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Learner quality tests must not use the network')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(httpx.Client, 'send', forbidden)
    monkeypatch.setattr(httpx.AsyncClient, 'send', forbidden)
    monkeypatch.delenv('LINGUARY_CONTENT_PLAN_PROFILE', raising=False)
    monkeypatch.delenv('LINGUARY_QUALITY_ROUTES', raising=False)


def analysis_for(profile, *, indivisible=False):
    payload = analysis_payload(STORY, list(range(len(STORY['blocks']))), candidate=profile != 'legacy-v1')
    payload['chapter_summary'] = 'Independent synthetic scene.\n\nНезависимая вымышленная сцена.'
    if profile == 'content-plan-v3':
        lamp_text = STORY['blocks'][1]
        lamp_start, it_start = lamp_text.index('lamp'), lamp_text.index('It')
        payload['content_plan'].update(version=3, profile=profile, context_facts=[dict(id='gender', kind='gender',
            anchors=[dict(block_id='b1', start=0, end=4, quote='Mara')],
            subject_anchor=dict(block_id='b1', start=0, end=4, quote='Mara'), value='female', certainty='explicit'),
            dict(id='speaker', kind='speaker', anchors=[dict(block_id='b1', start=0, end=9, quote='Mara said')],
                subject_anchor=dict(block_id='b1', start=0, end=4, quote='Mara'), value='Mara', certainty='explicit'),
            dict(id='referent', kind='pronoun_referent', anchors=[dict(block_id='b2', start=lamp_start, end=lamp_start + 4, quote='lamp')],
                subject_anchor=dict(block_id='b2', start=it_start, end=it_start + 2, quote='It'), value='lamp', certainty='explicit')])
    if indivisible:
        payload['content_plan']['difficulties'][0]['anchors'].append(dict(block_id='b4', start=1, end=6, quote='Shoo!'))
    return payload


class SceneFake(FakeLLMClient):
    def __init__(self, profile='content-plan-v3', *, reject=False, defect=None, indivisible=False):
        super().__init__()
        self.profile, self.reject, self.defect, self.indivisible = profile, reject, defect, indivisible
        self.blocks, self.plan = (), None
        self.review_mode = 'valid'
        self.after_response = None

    async def generate(self, request):
        if request.kind == LLMResponseKind.QUALITY_REVIEW:
            manifest = json.loads(request.input_text)
            artifacts = manifest['artifacts']
            findings = []
            if self.review_mode == 'finding':
                artifact = next(a for a in artifacts if a['artifact_kind'] == 'result')
                findings = [dict(block_ids=artifact['block_ids'], artifact_kind='result', artifact_sha=artifact['artifact_sha'],
                    severity='major', issue='Проверить род говорящего', evidence_anchors=[dict(block_id=self.blocks[0].id,
                        start=0, end=4, quote='Mara')], suggestion='Проверить по оригиналу', uncertainty='probable')]
            payload = dict(findings=findings, checked_artifacts=[a['artifact_sha'] for a in artifacts], missing_artifacts=[])
            self.default = FakeReply.of_payload(payload, usage=TokenUsage(5, 5))
        elif request.stage == LLMCallStage.CHAPTER_ANALYSIS:
            self.default = FakeReply.of_payload(analysis_for(self.profile, indivisible=self.indivisible))
        else:
            indices = [i for i, b in enumerate(self.blocks) if b.id in request.ordered_block_ids]
            if self.profile == 'legacy-v1':
                payload = dict(blocks=[dict(source_block_id=f'b{j}', treatment='TRANSLATE',
                    translated_text=STORY['translations'][i], glossary_hints=[]) for j, i in enumerate(indices, 1)], language_notes=[])
            else:
                payload = processing_payload(STORY, self.blocks, self.plan, indices=indices)
            if self.reject and len(indices) > 1:
                payload['blocks'][-1]['translated_text'] = '«Незаконченная цитата.'
            if self.defect == 'erased_help':
                payload['language_notes'][0]['note'] = '[b17]'
                payload['content_evidence'][0]['body_excerpt'] = '[b17]'
            elif self.defect == 'ambiguous_error':
                self.default = FakeReply(content='{')
                return await super().generate(request)
            self.default = FakeReply.of_payload(payload)
        response = await super().generate(request)
        if self.after_response is not None and request.kind == LLMResponseKind.QUALITY_REVIEW:
            self.after_response()
        return response


def route():
    return QualityRoute('fake-review', 'synthetic-only', 'test-v1', Decimal('1'), Decimal('2'), Decimal('0'),
        100, lambda request: 100000, 'synthetic-fixture-not-provider-proof', True)


def make_app(path, fake, profile='content-plan-v3'):
    return build_application(settings=AppSettings(llm=LLMSettings(model='fake-source', api_key='nonworking-test-key',
        proxy_mode='direct', max_attempts=2), reading=ReadingSettings(reader_level='a2'),
        storage=StorageSettings(data_root=str(path))), llm_client=fake, content_plan_profile=profile,
        quality_routes={('synthetic-only', 'fake-review'): route()},
        epubcheck=EpubCheckValidator(settings=EpubCheckSettings(java_executable='missing-quality-test-java')))


async def successful(app):
    tasks = await drain_tasks(app)
    assert tasks and all(t.status == TaskStatus.SUCCEEDED for t in tasks), [(t.status, t.last_error) for t in tasks]
    return tasks


async def prepare(app, fake):
    document, section, blocks, source_sha = await asyncio.to_thread(import_story, app, STORY)
    fake.blocks = blocks
    await app.analysis.start(BOOK)
    await successful(app)
    conditions = app.book_conditions.require(BOOK)
    assert conditions.payload['reader_level'] == 'a2'
    with app.unit_of_work.transaction() as tx:
        analysis = tx.analyses.get_for_section(BOOK, section.id)
    fake.plan = analysis.content_plan
    item = app.approve_glossary.review(BOOK).item_for('Mara')
    app.approve_glossary(GlossaryDecisionCommand(book_id=BOOK, source_text=item.source_text, decision=GlossaryDecision.CORRECT,
        translation='Мара', note=STORY['glossary']['note'], concept_id=item.concept_id))
    run = app.start_processing()(StartProcessingCommand(BOOK))
    assert run.glossary_snapshot.entries[0].translation == 'Мара'
    tasks = await app.processing.start(BOOK)
    return document, section, run, tasks[0], source_sha


def review_command(app, key, budget=1000000):
    manifest = app.quality_review.preview(BOOK)
    return StartQualityReview(BOOK, key, 'fake-review', 'synthetic-only', budget, 'all', manifest['manifest_sha'])


async def test_full_book_quality_pipeline(tmp_path, monkeypatch):
    from linguary.application.block_processing import TranslationCall
    fake = SceneFake(reject=True)
    app = make_app(tmp_path / 'data', fake)
    document, section, run, task, source_sha = await prepare(app, fake)
    original = TranslationCall.save
    crashed = False
    def after_child(self, **kwargs):
        nonlocal crashed
        saved = original(self, **kwargs)
        if saved and len(kwargs['results']) == 1 and not crashed:
            crashed = True
            raise RuntimeError('synthetic crash after accepted child commit')
        return saved
    monkeypatch.setattr(TranslationCall, 'save', after_child)
    tasks = await drain_tasks(app)
    assert tasks[0].status == TaskStatus.FAILED
    with app.unit_of_work.transaction() as tx:
        accepted, = tx.results.list_for_run(run.id)
        binding = tx.work.get_binding('result', f'{run.id}:{accepted.block_id}')
        root, = tx.work.list_roots(BOOK, run_id=run.id)
        previous_calls = root.calls_used
        logs_before = tx.llm_calls.list_for_book(BOOK)
    assert previous_calls > 1 and any(not c.success for c in logs_before)
    with pytest.raises(ExportBookError):
        app.export_book.ensure_ready(BOOK)
    await app.aclose()
    monkeypatch.setattr(TranslationCall, 'save', original)
    later = SceneFake(reject=True)
    later.blocks, later.plan = fake.blocks, fake.plan
    restarted = make_app(tmp_path / 'data', later)
    try:
        await restarted.runner.recover_interrupted()
        with restarted.unit_of_work.transaction() as tx:
            tx.tasks.save(tx.tasks.get(task.id).retry())
        await successful(restarted)
        assert all(accepted.block_id not in r.ordered_block_ids for r in later.requests)
        with restarted.unit_of_work.transaction() as tx:
            results = tx.results.list_for_run(run.id)
            assert tx.work.get_binding('result', f'{run.id}:{accepted.block_id}') == binding
            assert tx.work.get_root(root.id).calls_used > previous_calls
            assert tx.work.root_coverage_complete(root.id)
            assert len(results) == len(tuple(document.blocks()))
            assert tx.documents.get(BOOK) == document
            calls = [c for c in tx.llm_calls.list_for_book(BOOK) if c.stage == LLMCallStage.TRANSLATION_UNIT]
            assert tx.work.get_root(root.id).calls_used == len(calls)
            assert calls[0].success is False
            assert tx.work.get_attempt(calls[0].attempt_id).state == 'rejected'
        assert {f.kind for f in later.plan.context_facts} == {'speaker', 'gender', 'pronoun_referent'}
        assert all(r.content_evidence.semantic_correctness == 'not_evaluated' for r in results)
        assert all('[b17]' not in n.note for r in results for n in r.language_notes)
        assert all('A2' in r.instructions and 'Мара' in r.input_text for r in later.requests)
        assert restarted.quality_review.get(BOOK)['state'] == 'not-requested'
        count = len(later.requests)
        limited = await restarted.quality_review.start(review_command(restarted, 'limit', budget=1))
        await successful(restarted)
        assert len(later.requests) == count
        assert restarted.quality_review.get(BOOK, limited['review_id'])['state'] == 'limit'
        later.review_mode = 'finding'
        reviewed = await restarted.quality_review.start(review_command(restarted, 'explicit-review'))
        await successful(restarted)
        state = restarted.quality_review.get(BOOK, reviewed['review_id'])
        assert state['state'] == 'completed-with-findings' and state['spent'] > 0 and state['reserved'] == 0
        assert state['semantic_correctness'] == 'not_evaluated'
        await restarted.start_export(BOOK)
        await successful(restarted)
        nodes = [n for r in exported_documents(restarted.paths.export_epub(BOOK)).values() for n in r.iter()]
        assert [(''.join(n.itertext())) for n in nodes if n.get('id', '').startswith('source-')] == STORY['blocks']
        visible = '\n'.join(''.join(n.itertext()) for n in nodes)
        for result in results:
            assert result.translated_text in visible
            for note in result.language_notes:
                assert note.note in visible
        assert hashlib.sha256(restarted.paths.source_epub(BOOK).read_bytes()).hexdigest() == source_sha
        # A changed current artifact invalidates the independent conclusion without rewriting it.
        with restarted.unit_of_work.transaction() as tx:
            tx.results.replace_results(run.id, (replace(results[0], translated_text='Мара сказала: «Я уже готова».'),))
        assert restarted.quality_review.get(BOOK, reviewed['review_id'])['state'] == 'stale'
    finally:
        await restarted.aclose()


@pytest.mark.parametrize('defect', ['erased_help', 'ambiguous_error', 'indivisible'])
async def test_partial_book_and_review_do_not_export(tmp_path, defect):
    fake = SceneFake(defect=defect, reject=defect == 'indivisible', indivisible=defect == 'indivisible')
    app = make_app(tmp_path / 'data', fake)
    try:
        _, _, run, _, _ = await prepare(app, fake)
        tasks = await drain_tasks(app)
        assert tasks[0].status == TaskStatus.FAILED
        with app.unit_of_work.transaction() as tx:
            root, = tx.work.list_roots(BOOK, run_id=run.id)
            assert not tx.work.root_coverage_complete(root.id)
            assert tx.results.list_for_run(run.id) == ()
            if defect in ('indivisible', 'ambiguous_error'):
                assert len(tx.work.list_scopes(root.id)) == 1
        reviewed = await app.quality_review.start(review_command(app, 'partial'))
        await successful(app)
        assert app.quality_review.get(BOOK, reviewed['review_id'])['state'] == 'partial'
        with pytest.raises(ExportBookError):
            await app.start_export(BOOK)
        assert not app.paths.export_epub(BOOK).exists()
    finally:
        await app.aclose()


@pytest.mark.parametrize('profile', ['legacy-v1', 'content-plan-v2'])
async def test_legacy_v1_v2_fixture_roundtrip(tmp_path, profile):
    fake = SceneFake(profile)
    app = make_app(tmp_path / 'data', fake, profile)
    try:
        document, _, run, _, source_sha = await prepare(app, fake)
        await successful(app)
        await app.start_export(BOOK)
        await successful(app)
        with app.unit_of_work.transaction() as tx:
            results = tx.results.list_for_run(run.id)
            assert len(results) == len(tuple(document.blocks()))
        assert all(r.contract_version == (1 if profile == 'legacy-v1' else 2) for r in fake.requests)
        visible = '\n'.join(''.join(r.itertext()) for r in exported_documents(app.paths.export_epub(BOOK)).values())
        assert all(text in visible for text in STORY['blocks'])
        assert hashlib.sha256(app.paths.source_epub(BOOK).read_bytes()).hexdigest() == source_sha
    finally:
        await app.aclose()


def test_synthetic_reader_cases_retain_semantic_uncertainty():
    cases = DATA['edge_cases']
    policy = ReaderTextPolicy()
    for name in ('service_label', 'quoted_label', 'unsupported_showcase'):
        case = cases[name]
        assert policy.prepare('language_note.note', case['body'], (case['quote'],)) == case.get('expected', case['body'])
    for name in ('erased_help', 'ambiguous_label'):
        case = cases[name]
        with pytest.raises(ReaderTextError):
            policy.prepare('language_note.note', case['body'], (case['quote'],))
    assert policy.semantic_correctness == 'not_evaluated'
    plan = ContentPlan(difficulties=(Difficulty('connected', (SourceAnchor('one', 0, 1, 'a'), SourceAnchor('four', 0, 1, 'd')),
        'one', 'grammar', 'critical', 'explanation', (RequiredPoint('meaning', 'Explain connection'),)),))
    assert dependency_intervals(('one', 'two', 'three', 'four'), plan) == (('one', 'two', 'three', 'four'),)
    assert local_failure_signature(Violation('quotes', 'translation_unit', 'unknown'), ('one',)) is None


class SimulatedCrash(BaseException):
    """A process exit bypasses the runner's normal task-error handling."""


@pytest.mark.parametrize('boundary', ['before_reserve', 'reserved', 'sent', 'response_received',
    'diagnostic_saved', 'settled', 'validated', 'committed'])
async def test_all_crash_boundaries(tmp_path, monkeypatch, boundary):
    from linguary.adapters.llm.logging import LoggingLLMClient
    fake = SceneFake()
    app = make_app(tmp_path / 'data', fake)
    await prepare(app, fake)
    await successful(app)
    created = await app.quality_review.start(review_command(app, 'crash-boundary'))

    def crash():
        raise SimulatedCrash(boundary)

    def interrupt_after(original, *, before=False, only_state=None):
        def interrupted(*args, **kwargs):
            if before:
                crash()
            value = original(*args, **kwargs)
            if only_state is None or args[-1] == only_state:
                crash()
            return value
        return interrupted

    with monkeypatch.context() as patch:
        budget = app.quality_review.budget
        if boundary in ('before_reserve', 'reserved'):
            patch.setattr(budget, 'reserve', interrupt_after(budget.reserve, before=boundary == 'before_reserve'))
        elif boundary == 'sent':
            patch.setattr(budget, 'mark_sent', interrupt_after(budget.mark_sent))
        elif boundary == 'response_received':
            fake.after_response = crash
        elif boundary == 'diagnostic_saved':
            original = LoggingLLMClient.generate
            async def after_diagnostic(self, request):
                response = await original(self, request)
                if request.kind == LLMResponseKind.QUALITY_REVIEW:
                    crash()
                return response
            patch.setattr(LoggingLLMClient, 'generate', after_diagnostic)
        elif boundary == 'settled':
            patch.setattr(budget, 'settle', interrupt_after(budget.settle))
        elif boundary == 'validated':
            patch.setattr(app.quality_review, '_finish_attempt',
                interrupt_after(app.quality_review._finish_attempt, only_state='validated'))
        else:
            original = app.task_queue.save
            async def before_task_ack(task):
                if task.status == TaskStatus.SUCCEEDED:
                    crash()
                await original(task)
            patch.setattr(app.task_queue, 'save', before_task_ack)
        with pytest.raises(SimulatedCrash):
            await app.runner.run_next()
    with app.unit_of_work.transaction() as tx:
        before = tx.work.get_review(created['review_id'])
        task, = [t for t in tx.tasks.list_for_book(BOOK) if t.scope_id == created['review_id']]
    await app.aclose()
    later = SceneFake()
    later.blocks, later.plan = fake.blocks, fake.plan
    restarted = make_app(tmp_path / 'data', later)
    try:
        await restarted.runner.recover_interrupted()
        await restarted.runner.recover_interrupted()
        assert await drain_tasks(restarted) == ()
        assert later.requests == []
        with restarted.unit_of_work.transaction() as tx:
            after = tx.work.get_review(created['review_id'])
            assert after.spent == before.spent
            assert tx.tasks.get(task.id).status == TaskStatus.FAILED
            if boundary in ('sent', 'response_received', 'diagnostic_saved'):
                assert after.state == 'unknown' and after.reserved == before.reserved > 0
            elif boundary in ('before_reserve', 'reserved'):
                assert after.spent == after.reserved == 0
            else:
                assert after.spent > 0 and after.reserved == 0
            if boundary == 'committed':
                assert after.state == 'completed'
                tx.tasks.save(tx.tasks.get(task.id).retry())
        if boundary == 'committed':
            await successful(restarted)
            assert later.requests == []
    finally:
        await restarted.aclose()


def test_comparison_manifest_separates_plan_actual_first_retry():
    manifest = DATA['comparison_manifest_template']
    assert manifest['status'] == 'awaiting_explicit_authorization'
    assert manifest['reader_level'] == 'a2' and manifest['semantic_correctness'] == 'not_evaluated'
    assert {'planned', 'delivered', 'first_response', 'retry', 'rejected', 'undelivered', 'spent', 'reserved', 'unknown_cost'} <= set(manifest['report_columns'])
    assert all(manifest[k] is None for k in ('source_sha256', 'permitted_fragment', 'glossary_sha256', 'routes', 'parameters',
        'before_versions', 'after_versions', 'changed_factor', 'budget_microusd', 'independent_reviewer'))
    protocol = (FIXTURES / 'PAID-COMPARISON-PROTOCOL.md').read_text()
    assert 'AC4–AC6' in protocol and 'независимого рецензента' in protocol and 'не разрешают' in protocol
