"""Explicit optional review; findings never write learning artifacts."""
from dataclasses import dataclass, replace
from hashlib import sha256
from uuid import uuid4

from linguary.application.ports.llm import LLMRequest, LLMResponseKind, LLMResponseFormatError
from linguary.application.quality_budget import QualityBudget
from linguary.application.quality_context import (
    INSTRUCTIONS,
    MAX_QUALITY_CHUNK_BLOCKS,
    partition_manifest,
    request_text,
    snapshot,
    submanifest_for_scope,
    validate_response,
)
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


def _read_transaction(uow):
    return getattr(uow, 'read_transaction', uow.transaction)()


def _chunk_request(chunk, route, *, book_id=None, task_id=None, scope_id=None):
    if book_id is None:
        return LLMRequest(
            kind=LLMResponseKind.QUALITY_REVIEW,
            instructions=INSTRUCTIONS,
            input_text=request_text(chunk),
            max_completion_tokens=route.max_completion_tokens,
        )
    return LLMRequest(
        kind=LLMResponseKind.QUALITY_REVIEW,
        instructions=INSTRUCTIONS,
        input_text=request_text(chunk),
        book_id=book_id,
        stage=LLMCallStage.QUALITY_REVIEW,
        task_id=task_id,
        scope_id=scope_id,
        ordered_block_ids=tuple(chunk['scope']),
        max_completion_tokens=route.max_completion_tokens,
    )


def _split_chunk_for_route(chunk, route):
    req = _chunk_request(chunk, route)
    try:
        route.estimate(req)
        return [chunk]
    except WorkStateError:
        scope = list(chunk.get('scope') or ())
        if len(scope) <= 1:
            raise
        mid = len(scope) // 2
        left = submanifest_for_scope(chunk, scope[:mid])
        right = submanifest_for_scope(chunk, scope[mid:])
        return _split_chunk_for_route(left, route) + _split_chunk_for_route(right, route)


def _chunks_for_route(manifest, route, *, max_blocks_per_chunk=MAX_QUALITY_CHUNK_BLOCKS):
    result = []
    for chunk in partition_manifest(manifest, max_blocks_per_chunk=max_blocks_per_chunk):
        result.extend(_split_chunk_for_route(chunk, route))
    return tuple(result)


class QualityReviewService:
    def __init__(self, unit_of_work, queue, client_for, routes):
        self.uow, self.queue, self.client_for, self.routes = unit_of_work, queue, client_for, dict(routes)
        self.budget = QualityBudget(unit_of_work)

    def _chunks_for_route(self, manifest, route):
        return _chunks_for_route(manifest, route)

    def estimate(self, manifest, route):
        chunks = self._chunks_for_route(manifest, route)
        return sum(route.estimate(_chunk_request(chunk, route)) for chunk in chunks)

    def preview(self, book_id, scope='all'):
        with _read_transaction(self.uow) as tx:
            manifest = snapshot(tx, book_id, scope)
        if self.routes:
            route_estimates = {}
            for _, route in sorted(self.routes.items()):
                key = f'{route.route_id}:{route.model}'
                try:
                    route_estimates[key] = self.estimate(manifest, route)
                except Exception:
                    route_estimates[key] = None
            manifest = {**manifest, 'route_estimates': route_estimates}
        return manifest

    async def start(self, command, *, require_available=False):
        if (not command.request_key.strip() or not command.model.strip() or not command.route_id.strip()
                or type(command.budget_microusd) is not int or command.budget_microusd <= 0):
            raise WorkStateError('требуются модель, маршрут и положительный бюджет USD')
        settings = dict(schema_version=1, model=command.model, route_id=command.route_id,
            budget_microusd=command.budget_microusd, scope=command.scope,
            expected_manifest_sha=command.expected_manifest_sha)
        manifest = None
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
                    try:
                        estimate = self.estimate(manifest, route)
                    except Exception:
                        raise WorkStateError('price_unavailable') from None
                    if estimate > command.budget_microusd:
                        raise WorkStateError('budget')
                settings['route_snapshot'] = None if route is None else route.snapshot()
                review = QualityReview(str(uuid4()), command.book_id, command.request_key,
                    manifest, settings, command.budget_microusd, manifest['conditions_id'])
                tx.work.add_review(review)
        await self.queue.enqueue(TaskIdentity(command.book_id, TaskType.QUALITY_REVIEW, review.id))
        with _read_transaction(self.uow) as tx:
            return quality_review_state(tx, command.book_id, review.id, current_snapshot=manifest)

    def get(self, book_id, review_id=None):
        with _read_transaction(self.uow) as tx:
            return quality_review_state(tx, book_id, review_id)

    def _update(self, review_id, state, **kwargs):
        with self.uow.transaction() as tx:
            tx.work.update_review(review_id, state=state, **kwargs)

    async def run_task(self, task):
        with _read_transaction(self.uow) as tx:
            review = tx.work.get_review(task.scope_id)
            if review is None or review.book_id != task.book_id:
                raise WorkStateError('вычитка не найдена')
            if review.state not in ('queued', 'running'):
                return
            try:
                manifest = snapshot(tx, task.book_id, tuple(review.manifest['scope']))
            except WorkStateError:
                manifest = None
            initial_state = (
                'stale'
                if manifest is None
                else quality_review_state(tx, task.book_id, review.id, current_snapshot=manifest)['state']
            )
        if initial_state == 'stale':
            self._update(review.id, 'stale', reason='manifest_changed')
            return
        route = self.routes.get((review.settings['route_id'], review.settings['model']))
        if route is None or thaw(review.settings.get('route_snapshot')) != route.snapshot():
            self._update(review.id, 'failed', reason='price_unavailable')
            return
        try:
            chunks = self._chunks_for_route(manifest, route)
            chunk_plans = []
            for chunk in chunks:
                req = _chunk_request(
                    chunk,
                    route,
                    book_id=review.book_id,
                    task_id=task.id,
                    scope_id=review.id,
                )
                chunk_plans.append((chunk, req, route.estimate(req)))
            client = self.client_for(route)
        except Exception:
            self._update(review.id, 'failed', reason='price_or_route_unavailable')
            return
        self._update(review.id, 'running')
        prompt_hash = sha256(INSTRUCTIONS.encode()).hexdigest()
        all_findings: list[dict] = []
        all_checked: list[str] = []
        all_missing: list[str] = []
        total_chunks = len(chunk_plans)

        def _dedup_coverage():
            dedup_missing = tuple(dict.fromkeys(all_missing))
            missing_set = set(dedup_missing)
            dedup_checked = tuple(
                sha for sha in dict.fromkeys(all_checked) if sha not in missing_set
            )
            return dedup_checked, dedup_missing

        for chunk_idx, (chunk, request, estimate) in enumerate(chunk_plans):
            is_last_chunk = chunk_idx == total_chunks - 1
            previous_error: str | None = None
            previous_raw: str | None = None
            for ordinal in range(1, 4):
                with _read_transaction(self.uow) as tx:
                    current = tx.work.get_review(review.id)
                try:
                    attempt = self.budget.reserve(
                        review.id,
                        estimate,
                        price=route.snapshot(),
                        expected_revision=current.revision,
                        task_id=task.id,
                        expected_manifest_sha=None,
                    )
                except WorkStateError as error:
                    if str(error) == 'manifest_changed' or self.get(task.book_id, review.id)['state'] == 'stale':
                        self._update(review.id, 'stale', reason='manifest_changed')
                    else:
                        dedup_checked, dedup_missing = _dedup_coverage()
                        self._update(
                            review.id,
                            'limit',
                            findings=tuple(all_findings),
                            checked=dedup_checked,
                            missing=dedup_missing,
                            reason='budget',
                            partial=bool(dedup_checked or dedup_missing),
                        )
                    return
                repair_turns = ((previous_raw, previous_error),) if ordinal > 1 and previous_error else ()
                repair_instructions = (previous_error,) if ordinal > 1 and previous_error else ()
                context = CallContext(
                    review.conditions_id,
                    'quality_review',
                    review.id,
                    tuple(chunk['scope']),
                    attempt.id,
                    current.revision,
                    route.model,
                    prompt_hash,
                    task_id=task.id,
                )
                outgoing = replace(
                    request,
                    instructions=INSTRUCTIONS,
                    call_context=context,
                    attempt=ordinal,
                    repair_turns=repair_turns,
                    repair_instructions=repair_instructions,
                )
                if not self.budget.mark_sent(attempt.id):
                    state = self.get(task.book_id, review.id)['state']
                    if state == 'stale':
                        self._update(review.id, 'stale', reason='manifest_changed')
                    elif state not in ('limit', 'unknown'):
                        raise WorkStateError('отправка вычитки отклонена')
                    return
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
                    previous_error = str(error)
                    previous_raw = getattr(error, 'raw_content', None)
                    continue
                except Exception:
                    self.budget.settle(attempt.id, None)
                    self._finish_attempt(attempt.id, 'unknown')
                    return
                self.budget.settle(attempt.id, actual)
                if actual is None:
                    self._finish_attempt(attempt.id, 'unknown')
                    return
                with _read_transaction(self.uow) as tx:
                    current_review = tx.work.get_review(review.id)
                    try:
                        after = (
                            None
                            if current_review.state == 'stale'
                            else snapshot(tx, review.book_id, tuple(review.manifest['scope']))
                        )
                    except WorkStateError:
                        after = None
                    state_after = (
                        'stale'
                        if current_review.state == 'stale' or after is None
                        else quality_review_state(tx, task.book_id, review.id, current_snapshot=after)['state']
                    )
                if state_after == 'stale':
                    self._finish_attempt(attempt.id, 'rejected')
                    finalize_response(self.uow, response, 'stale')
                    self._update(review.id, 'stale', reason='manifest_changed')
                    return
                try:
                    if response.truncated:
                        raise WorkStateError('неполный ответ вычитки')
                    payload = validate_response(chunk, response.payload)
                except Exception as error:
                    record_rejection(self.uow, response, error)
                    self._finish_attempt(attempt.id, 'rejected')
                    if self._stopped(review.id):
                        return
                    previous_error = str(error)
                    previous_raw = getattr(response, 'raw_content', None)
                    continue
                self._finish_attempt(attempt.id, 'validated')
                stop_after_chunk = False
                with self.uow.transaction() as tx:
                    current = tx.work.get_review(review.id)
                    if current.state == 'stale' or after['manifest_sha'] != review.manifest['manifest_sha']:
                        tx.work.transition_attempt(attempt.id, 'rejected', expected_state='validated')
                        finalize_response(self.uow, response, 'stale', transaction=tx)
                        tx.work.update_review(review.id, state='stale', reason='manifest_changed')
                        return
                    tx.work.transition_attempt(attempt.id, 'accepted', expected_state='validated')
                    finalize_response(self.uow, response, 'accepted', transaction=tx)
                    all_findings.extend(payload.findings)
                    all_checked.extend(payload.checked_artifacts)
                    all_missing.extend(payload.missing_artifacts)
                    dedup_checked, dedup_missing = _dedup_coverage()
                    target_state = (
                        ('limit' if current.state == 'limit' else 'completed')
                        if is_last_chunk
                        else ('limit' if current.state == 'limit' else 'running')
                    )
                    tx.work.update_review(
                        review.id,
                        state=target_state,
                        findings=tuple(all_findings),
                        checked=dedup_checked,
                        missing=dedup_missing,
                        partial=bool(
                            dedup_missing
                            or review.manifest['missing_results']
                            or not review.manifest['book_complete']
                            or not is_last_chunk
                        ),
                    )
                    stop_after_chunk = current.state in ('limit', 'unknown', 'stale')
                if stop_after_chunk:
                    return
                break
            else:
                dedup_checked, dedup_missing = _dedup_coverage()
                self._update(
                    review.id,
                    'failed',
                    findings=tuple(all_findings),
                    checked=dedup_checked,
                    missing=dedup_missing,
                    reason='invalid_response',
                    partial=bool(dedup_checked or dedup_missing),
                )
                return

    def _finish_attempt(self, attempt_id, state):
        with self.uow.transaction() as tx:
            attempt = tx.work.get_attempt(attempt_id)
            if attempt is None:
                raise WorkStateError('попытка не найдена')
            if attempt.state == state:
                return
            tx.work.transition_attempt(attempt_id, state, expected_state='sent')

    def _stopped(self, review_id):
        with _read_transaction(self.uow) as tx:
            return tx.work.get_review(review_id).state in ('limit', 'unknown', 'stale')


def quality_review_state(tx, book_id, review_id=None, *, current_snapshot=None):
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
    state = review.state
    current_manifest_sha = None if current_snapshot is None else current_snapshot.get('manifest_sha')
    if state == 'stale':
        stale = True
    elif current_snapshot is not None:
        stale = current_manifest_sha != review.manifest['manifest_sha']
    else:
        try:
            snap = snapshot(tx, book_id, tuple(review.manifest['scope']))
            current_manifest_sha = snap['manifest_sha']
            stale = current_manifest_sha != review.manifest['manifest_sha']
        except WorkStateError:
            stale = True
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
        uncertainty='probabilistic', semantic_correctness='not_evaluated',
        current_manifest_sha=current_manifest_sha)
