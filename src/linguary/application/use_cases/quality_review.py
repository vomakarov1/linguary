"""Explicit optional review; findings never write learning artifacts."""
from dataclasses import dataclass, replace
from hashlib import sha256
from uuid import uuid4

from linguary.application.ports.llm import LLMRequest, LLMResponseKind, LLMResponseFormatError
from linguary.application.quality_budget import QualityBudget
from linguary.application.quality_context import INSTRUCTIONS, snapshot, request_text, validate_response
from linguary.application.llm_rejections import finalize_response, record_rejection
from linguary.domain import TaskIdentity, TaskType, LLMCallStage
from linguary.domain.book_conditions import WorkStateError, thaw
from linguary.domain.llm_log import CallContext
from linguary.domain.quality_review import QualityReview


@dataclass(frozen=True)
class StartQualityReview:
    book_id: str
    request_key: str
    model: str
    route_id: str
    budget_microusd: int
    scope: tuple[str, ...] | str
    expected_manifest_sha: str


class QualityReviewService:
    def __init__(self, unit_of_work, queue, client_for, routes):
        self.uow, self.queue, self.client_for, self.routes = unit_of_work, queue, client_for, dict(routes)
        self.budget = QualityBudget(unit_of_work)

    def preview(self, book_id, scope='all'):
        with self.uow.transaction() as tx:
            return snapshot(tx, book_id, scope)

    async def start(self, command, *, require_available=False):
        if (not command.request_key.strip() or not command.model.strip() or not command.route_id.strip()
                or type(command.budget_microusd) is not int or command.budget_microusd <= 0):
            raise WorkStateError('требуются модель, маршрут и положительный бюджет USD')
        settings = dict(schema_version=1, model=command.model, route_id=command.route_id,
            budget_microusd=command.budget_microusd, scope=command.scope,
            expected_manifest_sha=command.expected_manifest_sha)
        with self.uow.transaction() as tx:
            existing = tx.work.find_review(command.book_id, command.request_key)
            if existing:
                if {k: v for k, v in thaw(existing.settings).items() if k != 'route_snapshot'} != thaw(settings):
                    raise WorkStateError('request_key_conflict')
                review = existing
            else:
                manifest = snapshot(tx, command.book_id, command.scope)
                if manifest['manifest_sha'] != command.expected_manifest_sha:
                    raise WorkStateError('manifest_changed')
                route = self.routes.get((command.route_id, command.model))
                if require_available:
                    if route is None:
                        raise WorkStateError('route_unavailable')
                    request = LLMRequest(kind=LLMResponseKind.QUALITY_REVIEW, instructions=INSTRUCTIONS,
                        input_text=request_text(manifest), max_completion_tokens=route.max_completion_tokens)
                    try:
                        estimate = route.estimate(request)
                    except Exception:
                        raise WorkStateError('price_unavailable') from None
                    if estimate > command.budget_microusd:
                        raise WorkStateError('budget')
                settings['route_snapshot'] = None if route is None else route.snapshot()
                review = QualityReview(str(uuid4()), command.book_id, command.request_key,
                    manifest, settings, command.budget_microusd, manifest['conditions_id'])
                tx.work.add_review(review)
        await self.queue.enqueue(TaskIdentity(command.book_id, TaskType.QUALITY_REVIEW, review.id))
        return self.get(command.book_id, review.id)

    def get(self, book_id, review_id=None):
        with self.uow.transaction() as tx:
            return quality_review_state(tx, book_id, review_id)

    def _update(self, review_id, state, **kwargs):
        with self.uow.transaction() as tx:
            tx.work.update_review(review_id, state=state, **kwargs)

    async def run_task(self, task):
        with self.uow.transaction() as tx:
            review = tx.work.get_review(task.scope_id)
        if review is None or review.book_id != task.book_id:
            raise WorkStateError('вычитка не найдена')
        if review.state not in ('queued', 'running'):
            return
        if self.get(task.book_id, review.id)['state'] == 'stale':
            self._update(review.id, 'stale', reason='manifest_changed')
            return
        route = self.routes.get((review.settings['route_id'], review.settings['model']))
        if route is None or thaw(review.settings.get('route_snapshot')) != route.snapshot():
            self._update(review.id, 'failed', reason='price_unavailable')
            return
        request = LLMRequest(kind=LLMResponseKind.QUALITY_REVIEW, instructions=INSTRUCTIONS,
            input_text=request_text(review.manifest), book_id=review.book_id, stage=LLMCallStage.QUALITY_REVIEW,
            task_id=task.id, scope_id=review.id, ordered_block_ids=tuple(review.manifest['scope']),
            max_completion_tokens=route.max_completion_tokens)
        try:
            estimate = route.estimate(request)
            client = self.client_for(route)
        except Exception:
            self._update(review.id, 'failed', reason='price_or_route_unavailable')
            return
        self._update(review.id, 'running')
        for ordinal in range(1, 4):
            with self.uow.transaction() as tx:
                current = tx.work.get_review(review.id)
            try:
                attempt = self.budget.reserve(review.id, estimate, price=route.snapshot(),
                    expected_revision=current.revision, task_id=task.id,
                    expected_manifest_sha=review.manifest['manifest_sha'])
            except WorkStateError:
                if self.get(task.book_id, review.id)['state'] == 'stale':
                    self._update(review.id, 'stale', reason='manifest_changed')
                else:
                    self._update(review.id, 'limit', reason='budget')
                return
            context = CallContext(review.conditions_id, 'quality_review', review.id,
                tuple(review.manifest['scope']), attempt.id, current.revision, route.model,
                sha256(INSTRUCTIONS.encode()).hexdigest(), task_id=task.id)
            outgoing = replace(request, call_context=context, attempt=ordinal)
            self.budget.mark_sent(attempt.id)
            response = None
            try:
                response = await client.generate(outgoing)
                actual = route.actual(response)
            except LLMResponseFormatError as error:
                actual = route.actual(error)
                self.budget.settle(attempt.id, actual)
                self._finish_attempt(attempt.id, 'rejected')
                if actual is None:
                    return
                if self._stopped(review.id):
                    return
                continue
            except Exception:
                self.budget.settle(attempt.id, None)
                self._finish_attempt(attempt.id, 'unknown')
                return
            self.budget.settle(attempt.id, actual)
            if actual is None:
                self._finish_attempt(attempt.id, 'unknown')
                return
            if self.get(task.book_id, review.id)['state'] == 'stale':
                self._finish_attempt(attempt.id, 'rejected')
                finalize_response(self.uow, response, 'stale')
                self._update(review.id, 'stale', reason='manifest_changed')
                return
            try:
                if response.truncated:
                    raise WorkStateError('неполный ответ вычитки')
                validate_response(review.manifest, response.payload)
            except Exception as error:
                record_rejection(self.uow, response, error)
                self._finish_attempt(attempt.id, 'rejected')
                if self._stopped(review.id):
                    return
                continue
            self._finish_attempt(attempt.id, 'validated')
            with self.uow.transaction() as tx:
                if snapshot(tx, review.book_id, tuple(review.manifest['scope']))['manifest_sha'] != review.manifest['manifest_sha']:
                    tx.work.transition_attempt(attempt.id, 'rejected', expected_state='validated')
                    finalize_response(self.uow, response, 'stale', transaction=tx)
                    tx.work.update_review(review.id, state='stale', reason='manifest_changed')
                else:
                    tx.work.transition_attempt(attempt.id, 'accepted', expected_state='validated')
                    finalize_response(self.uow, response, 'accepted', transaction=tx)
                    current = tx.work.get_review(review.id)
                    tx.work.update_review(review.id, state='limit' if current.state == 'limit' else 'completed',
                        findings=response.payload.findings, checked=response.payload.checked_artifacts,
                        missing=response.payload.missing_artifacts,
                        partial=bool(response.payload.missing_artifacts or review.manifest['missing_results'] or not review.manifest['book_complete']))
            return
        self._update(review.id, 'failed', reason='invalid_response')

    def _finish_attempt(self, attempt_id, state):
        with self.uow.transaction() as tx:
            attempt = tx.work.get_attempt(attempt_id)
            if attempt is None:
                raise WorkStateError('попытка не найдена')
            if attempt.state == state:
                return
            tx.work.transition_attempt(attempt_id, state, expected_state='sent')

    def _stopped(self, review_id):
        with self.uow.transaction() as tx:
            return tx.work.get_review(review_id).state in ('limit', 'unknown', 'stale')


def quality_review_state(tx, book_id, review_id=None):
    reviews = tx.work.list_reviews(book_id) if review_id is None else (tx.work.get_review(review_id),)
    defaults = dict(review_id=None, state='not-requested', manifest_sha=None,
        coverage=dict(checked=[], missing=[]), budget=None, spent=0, reserved=0,
        currency='USD', uncertainty='probabilistic', findings=[], reason=None,
        semantic_correctness='not_evaluated', available_routes=[], current_manifest_sha=None)
    if not reviews:
        return defaults
    review = reviews[-1]
    if review is None or review.book_id != book_id:
        raise WorkStateError('вычитка не найдена')
    progress = tx.work.review_progress(review.id)
    try:
        stale = snapshot(tx, book_id, tuple(review.manifest['scope']))['manifest_sha'] != review.manifest['manifest_sha']
    except WorkStateError:
        stale = True
    state = review.state
    if stale or state == 'stale':
        state = 'stale'
    elif state in ('queued', 'running'):
        state = 'running'
    elif state == 'completed':
        state = ('partial' if progress.get('partial') else
            'completed-with-findings' if review.findings else 'completed-without-findings')
    elif state in ('failed', 'cancelled'):
        state = 'partial'
    return dict(defaults, review_id=review.id, state=state, manifest_sha=review.manifest['manifest_sha'],
        budget=review.budget_limit, spent=review.spent, reserved=review.reserved,
        findings=thaw(review.findings), coverage=dict(checked=progress.get('checked') or [],
            missing=progress.get('missing') or []), reason=progress.get('reason'),
        uncertainty='probabilistic', semantic_correctness='not_evaluated')
