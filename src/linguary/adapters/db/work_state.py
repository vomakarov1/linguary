"""Durable work store; all operations use the caller's short transaction."""
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
import json
from uuid import uuid4

from sqlalchemy import select, update
from linguary.adapters.db.models import (
    BookConditionsRow, WorkBindingRow, ArtifactRevisionRow, RecoveryRootRow,
    RecoveryScopeRow, LLMAttemptRow, QualityReviewRow, QualityReservationRow,
    LLMCallLogRow, ProcessingRunRow,
)
from linguary.adapters.db.mapping import conditions_from_row, conditions_to_row
from linguary.domain.book_conditions import WorkStateError, identifier, nonnegative, thaw, version
from linguary.domain.recovery import CallAttempt, RecoveryRoot, RecoveryScope
from linguary.domain.quality_review import QualityReview, QualityReservation


def _json(value):
    value = thaw(value)
    version(value.get('schema_version'))
    return value


class SqlAlchemyWorkRepository:
    def __init__(self, session):
        self._session = session

    def _add(self, row):
        self._session.add(row)
        self._session.flush()

    def _owned(self, model, entity_id, book_id):
        row = self._session.get(model, entity_id)
        if row is None or row.book_id != book_id:
            raise WorkStateError('ссылка на отсутствующую или чужую работу')
        return row

    def add_conditions(self, conditions):
        self._add(conditions_to_row(conditions))

    def get_conditions(self, conditions_id):
        return conditions_from_row(self._session.get(BookConditionsRow, conditions_id))

    def latest_conditions(self, book_id):
        row = self._session.scalar(select(BookConditionsRow).where(BookConditionsRow.book_id == book_id).order_by(BookConditionsRow.revision.desc()).limit(1))
        return conditions_from_row(row)

    def bind(self, entity_kind, entity_id, *, book_id, conditions_id,
             attempt_id=None, artifact_revision_id=None, payload=None):
        self._owned(BookConditionsRow, conditions_id, book_id)
        for model, value in ((LLMAttemptRow, attempt_id), (ArtifactRevisionRow, artifact_revision_id)):
            if value is not None:
                row = self._owned(model, value, book_id)
                if row.conditions_id != conditions_id:
                    raise WorkStateError('условия происхождения не совпадают')
        values = dict(entity_kind=entity_kind, entity_id=entity_id, book_id=book_id,
                      conditions_id=conditions_id, attempt_id=attempt_id,
                      artifact_revision_id=artifact_revision_id,
                      payload_json=_json(payload or {'schema_version': 1}))
        previous = self._session.get(WorkBindingRow, (entity_kind, entity_id))
        if previous is not None:
            if any(getattr(previous, key) != value for key, value in values.items()):
                raise WorkStateError('связь происхождения уже существует')
            return
        self._add(WorkBindingRow(**values))

    def get_binding(self, entity_kind, entity_id):
        row = self._session.get(WorkBindingRow, (entity_kind, entity_id))
        if row is None:
            return None
        version(row.payload_json.get('schema_version'))
        return {column.name: getattr(row, column.name) for column in row.__table__.columns}

    def add_root(self, root):
        self._owned(BookConditionsRow, root.conditions_id, root.book_id)
        self._add(RecoveryRootRow(id=root.id, book_id=root.book_id, conditions_id=root.conditions_id,
                  root_key=root.root_key, manifest_json=_json(root.manifest), generation=root.generation,
                  calls_used=root.calls_used, extra_calls=root.extra_calls, status=root.status, reason=root.reason))

    def get_root(self, root_id):
        row = self._session.get(RecoveryRootRow, root_id)
        if row is None:
            return None
        return RecoveryRoot(row.id, row.book_id, row.conditions_id, row.root_key,
                            row.manifest_json, row.generation, row.calls_used,
                            row.extra_calls, row.status, row.reason)

    def add_scope(self, scope):
        root = self._session.get(RecoveryRootRow, scope.root_id)
        if root is None:
            raise WorkStateError('исходная область не существует')
        if scope.parent_id:
            parent = self._session.get(RecoveryScopeRow, scope.parent_id)
            if parent is None or parent.root_id != scope.root_id:
                raise WorkStateError('родитель области не принадлежит восстановлению')
        self._add(RecoveryScopeRow(id=scope.id, root_id=scope.root_id, parent_id=scope.parent_id,
                  ordered_blocks_json={'schema_version': 1, 'blocks': list(scope.ordered_blocks)},
                  generation=scope.generation, status=scope.status, accepted_attempt_id=scope.accepted_attempt_id))

    def get_scope(self, scope_id):
        row = self._session.get(RecoveryScopeRow, scope_id)
        if row is None:
            return None
        version(row.ordered_blocks_json.get('schema_version'))
        return RecoveryScope(row.id, row.root_id, tuple(row.ordered_blocks_json['blocks']),
                             row.generation, row.status, row.parent_id, row.accepted_attempt_id)

    def claim_scope(self, scope_id, *, expected_generation):
        nonnegative(expected_generation)
        changed = self._session.execute(update(RecoveryScopeRow).where(
            RecoveryScopeRow.id == scope_id,
            RecoveryScopeRow.generation == expected_generation,
            RecoveryScopeRow.status == 'pending',
        ).values(status='claimed', generation=RecoveryScopeRow.generation + 1))
        if changed.rowcount != 1:
            raise WorkStateError('область уже занята или изменилась')
        return self.get_scope(scope_id)

    def add_attempt(self, attempt):
        if attempt.conditions_id:
            self._owned(BookConditionsRow, attempt.conditions_id, attempt.book_id)
        if attempt.root_id:
            root = self._owned(RecoveryRootRow, attempt.root_id, attempt.book_id)
            if root.conditions_id != attempt.conditions_id:
                raise WorkStateError('условия попытки не совпадают')
        if attempt.scope_id:
            scope = self._session.get(RecoveryScopeRow, attempt.scope_id)
            if scope is None or scope.root_id != attempt.root_id or scope.generation != attempt.generation:
                raise WorkStateError('попытка не принадлежит текущей области')
        if attempt.review_id:
            review = self._owned(QualityReviewRow, attempt.review_id, attempt.book_id)
            if attempt.conditions_id != review.conditions_id:
                raise WorkStateError('условия вычитки не совпадают')
        values = attempt.to_dict()
        values.pop('schema_version')
        values['context_json'] = values.pop('context')
        values['outcome_json'] = values.pop('outcome')
        self._add(LLMAttemptRow(**values))

    def get_attempt(self, attempt_id):
        row = self._session.get(LLMAttemptRow, attempt_id)
        if row is None:
            return None
        values = {col.name: getattr(row, col.name) for col in row.__table__.columns}
        values['context'] = values.pop('context_json')
        values['outcome'] = values.pop('outcome_json')
        return CallAttempt(**values)

    def transition_attempt(self, attempt_id, state, *, expected_state, outcome=None):
        attempt = self.get_attempt(attempt_id)
        if attempt is None or attempt.state != expected_state:
            raise WorkStateError('состояние попытки изменилось')
        next_attempt = attempt.transition(state, outcome=outcome)
        changed = self._session.execute(update(LLMAttemptRow).where(
            LLMAttemptRow.id == attempt_id, LLMAttemptRow.state == expected_state,
        ).values(state=state, outcome_json=thaw(next_attempt.outcome)))
        if changed.rowcount != 1:
            raise WorkStateError('состояние попытки изменилось')
        return next_attempt

    def accept_scope(self, scope_id, *, attempt_id, expected_generation, results):
        from linguary.adapters.db.repositories import SqlAlchemyBlockResultRepository
        attempt = self.get_attempt(attempt_id)
        scope = self.get_scope(scope_id)
        if attempt is None or scope is None or attempt.scope_id != scope_id or attempt.generation != expected_generation or attempt.state != 'validated':
            raise WorkStateError('попытка не имеет права принять область')
        root = self.get_root(scope.root_id)
        latest = self.latest_conditions(root.book_id)
        run = self._session.get(ProcessingRunRow, attempt.run_id) if attempt.run_id else None
        if latest is None or latest.id != root.conditions_id or run is None or run.book_id != root.book_id or run.status != 'active':
            raise WorkStateError('версия обработки больше не активна')
        if tuple(item.block_id for item in results) != scope.ordered_blocks:
            raise WorkStateError('результаты не покрывают область в исходном порядке')
        with self._session.begin_nested():
            changed = self._session.execute(update(RecoveryScopeRow).where(
                RecoveryScopeRow.id == scope_id, RecoveryScopeRow.generation == expected_generation,
                RecoveryScopeRow.status == 'claimed', RecoveryScopeRow.accepted_attempt_id.is_(None),
            ).values(status='accepted', accepted_attempt_id=attempt_id))
            if changed.rowcount != 1:
                raise WorkStateError('область изменилась до принятия')
            SqlAlchemyBlockResultRepository(self._session).replace_results(attempt.run_id, results)
            for result in results:
                body = {'schema_version': 1, 'result': asdict(result)}
                artifact_id = str(uuid4())
                self.archive_artifacts([dict(id=artifact_id, book_id=root.book_id,
                    conditions_id=root.conditions_id, run_id=attempt.run_id, attempt_id=attempt_id,
                    kind='block_result', scope_id=result.block_id, body=body)])
                self.bind('result', f'{attempt.run_id}:{result.block_id}', book_id=root.book_id,
                          conditions_id=root.conditions_id, attempt_id=attempt_id, artifact_revision_id=artifact_id)
            self.transition_attempt(attempt_id, 'accepted', expected_state='validated')

    def archive_artifacts(self, artifacts):
        ids = []
        for artifact in artifacts:
            artifact_id = artifact.get('id') or str(uuid4())
            identifier(artifact_id)
            if self._session.get(ArtifactRevisionRow, artifact_id) is not None:
                raise WorkStateError('архивная версия неизменяема')
            body = _json(artifact['body'])
            encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
            digest = sha256(encoded.encode()).hexdigest()
            if artifact.get('sha256', digest) != digest:
                raise WorkStateError('контрольная сумма артефакта не совпадает')
            for model, key in ((BookConditionsRow, 'conditions_id'), (LLMAttemptRow, 'attempt_id')):
                if artifact.get(key):
                    self._owned(model, artifact[key], artifact['book_id'])
            self._add(ArtifactRevisionRow(id=artifact_id, book_id=artifact['book_id'],
                conditions_id=artifact.get('conditions_id'), run_id=artifact.get('run_id'),
                kind=artifact['kind'], scope_id=artifact['scope_id'], body_json=body,
                sha256=digest, attempt_id=artifact.get('attempt_id'), created_at=datetime.now(UTC)))
            ids.append(artifact_id)
        return tuple(ids)

    def get_artifact(self, artifact_id):
        row = self._session.get(ArtifactRevisionRow, artifact_id)
        if row is None:
            return None
        version(row.body_json.get('schema_version'))
        return {'id': row.id, 'book_id': row.book_id, 'body': row.body_json,
                'sha256': row.sha256, 'attempt_id': row.attempt_id}

    def add_review(self, review):
        if review.conditions_id:
            self._owned(BookConditionsRow, review.conditions_id, review.book_id)
        self._add(QualityReviewRow(id=review.id, book_id=review.book_id, conditions_id=review.conditions_id,
            request_key=review.request_key, manifest_json=_json(review.manifest), settings_json=_json(review.settings),
            state=review.state, revision=review.revision, findings_json={'schema_version': 1, 'items': thaw(review.findings)},
            budget_limit=review.budget_limit, spent=review.spent, reserved=review.reserved))

    def get_review(self, review_id):
        row = self._session.get(QualityReviewRow, review_id)
        if row is None:
            return None
        version(row.findings_json.get('schema_version'))
        return QualityReview(row.id, row.book_id, row.request_key, row.manifest_json,
            row.settings_json, row.budget_limit, row.conditions_id, row.state, row.revision,
            tuple(row.findings_json['items']), row.spent, row.reserved)

    def reserve(self, reservation, *, expected_revision):
        attempt = self.get_attempt(reservation.attempt_id)
        if attempt is None or attempt.review_id != reservation.review_id or attempt.state != 'created' or reservation.state != 'reserved':
            raise WorkStateError('попытка не допускает резервирование')
        with self._session.begin_nested():
            changed = self._session.execute(update(QualityReviewRow).where(
                QualityReviewRow.id == reservation.review_id, QualityReviewRow.revision == expected_revision,
                QualityReviewRow.state.in_(('queued', 'running')),
                QualityReviewRow.spent + QualityReviewRow.reserved + reservation.estimate <= QualityReviewRow.budget_limit,
            ).values(reserved=QualityReviewRow.reserved + reservation.estimate, revision=QualityReviewRow.revision + 1))
            if changed.rowcount != 1:
                raise WorkStateError('бюджет исчерпан или состояние вычитки изменилось')
            self._add(QualityReservationRow(attempt_id=reservation.attempt_id, review_id=reservation.review_id,
                estimate=reservation.estimate, actual=None, state='reserved', price_json=_json(reservation.price)))
            self.transition_attempt(attempt.id, 'reserved', expected_state='created')

    def get_reservation(self, attempt_id):
        row = self._session.get(QualityReservationRow, attempt_id)
        if row is None:
            return None
        return QualityReservation(row.attempt_id, row.review_id, row.estimate, row.actual, row.state, row.price_json)

    def settle(self, attempt_id, *, actual):
        current = self.get_reservation(attempt_id)
        if current is None:
            raise WorkStateError('резерв не найден')
        if current.state == 'settled' and actual == current.actual:
            return current
        if current.state == 'unknown' and actual is None:
            return current
        updated = current.transition('unknown' if actual is None else 'settled', actual=actual)
        with self._session.begin_nested():
            changed = self._session.execute(update(QualityReservationRow).where(
                QualityReservationRow.attempt_id == attempt_id, QualityReservationRow.state == current.state,
            ).values(state=updated.state, actual=actual))
            if changed.rowcount != 1:
                raise WorkStateError('резерв уже изменился')
            review = self.get_review(current.review_id)
            values = {'revision': QualityReviewRow.revision + 1}
            if actual is None:
                values['state'] = 'unknown'
            else:
                values.update(reserved=QualityReviewRow.reserved - current.estimate,
                              spent=QualityReviewRow.spent + actual)
            changed = self._session.execute(update(QualityReviewRow).where(
                QualityReviewRow.id == current.review_id, QualityReviewRow.revision == review.revision,
                QualityReviewRow.reserved >= current.estimate,
            ).values(**values))
            if changed.rowcount != 1:
                raise WorkStateError('состояние вычитки изменилось')
        return updated

    def purge_diagnostics(self, book_id, *, call_id=None, before=None):
        query = update(LLMCallLogRow).where(LLMCallLogRow.book_id == book_id, LLMCallLogRow.response_text.is_not(None))
        if call_id is not None:
            query = query.where(LLMCallLogRow.id == call_id)
        if before is not None:
            query = query.where(LLMCallLogRow.started_at < before)
        rows = self._session.scalars(select(LLMCallLogRow).where(*query._where_criteria)).all()
        for row in rows:
            meta = dict(row.diagnostic_meta_json or {'schema_version': 1})
            meta['body_status'] = 'expired' if before is not None else 'deleted'
            row.diagnostic_meta_json = meta
            row.response_text = None
        self._session.flush()
        return len(rows)

    def replace_result_binding(self, entity_id, *, book_id, conditions_id, attempt_id,
                               artifact_revision_id, expected_attempt_id):
        """Move a current projection pointer while preserving immutable archived versions."""
        conditions = self.latest_conditions(book_id)
        if conditions is None or conditions.id != conditions_id:
            raise WorkStateError('версия результата больше не активна')
        attempt = self._owned(LLMAttemptRow, attempt_id, book_id)
        artifact = self._owned(ArtifactRevisionRow, artifact_revision_id, book_id)
        if attempt.conditions_id != conditions_id or artifact.conditions_id != conditions_id or artifact.attempt_id != attempt_id:
            raise WorkStateError('происхождение новой версии не совпадает')
        current = self._session.get(WorkBindingRow, ('result', entity_id))
        if current is None:
            if expected_attempt_id is not None:
                raise WorkStateError('указатель результата изменился')
            self.bind('result', entity_id, book_id=book_id, conditions_id=conditions_id,
                      attempt_id=attempt_id, artifact_revision_id=artifact_revision_id)
            return
        if current.artifact_revision_id is None or self._session.get(ArtifactRevisionRow, current.artifact_revision_id) is None:
            raise WorkStateError('предыдущая версия результата не заархивирована')
        changed = self._session.execute(update(WorkBindingRow).where(
            WorkBindingRow.entity_kind == 'result', WorkBindingRow.entity_id == entity_id,
            WorkBindingRow.book_id == book_id, WorkBindingRow.conditions_id == conditions_id,
            WorkBindingRow.attempt_id.is_(None) if expected_attempt_id is None else WorkBindingRow.attempt_id == expected_attempt_id,
        ).values(attempt_id=attempt_id, artifact_revision_id=artifact_revision_id))
        if changed.rowcount != 1:
            raise WorkStateError('указатель результата изменился')

    def find_root_by_key(self, root_key):
        row = self._session.scalar(select(RecoveryRootRow).where(RecoveryRootRow.root_key == root_key))
        return None if row is None else self.get_root(row.id)

    def list_roots(self, book_id, *, run_id=None):
        rows = self._session.scalars(select(RecoveryRootRow).where(RecoveryRootRow.book_id == book_id)).all()
        return tuple(self.get_root(row.id) for row in rows
                     if run_id is None or row.manifest_json.get('active_run_id') == run_id)

    def list_scopes(self, root_id):
        rows = self._session.scalars(select(RecoveryScopeRow).where(RecoveryScopeRow.root_id == root_id)).all()
        return tuple(self.get_scope(row.id) for row in rows)

    def rebind_root(self, root_id, *, run_id, historical_coverage=()):
        row = self._session.get(RecoveryRootRow, root_id)
        if row is None:
            raise WorkStateError('исходная область отсутствует')
        manifest = dict(row.manifest_json)
        manifest.update(active_run_id=run_id, historical_coverage=list(historical_coverage))
        self._session.execute(update(RecoveryRootRow).where(RecoveryRootRow.id == root_id,
            RecoveryRootRow.generation == row.generation).values(manifest_json=manifest,
                generation=RecoveryRootRow.generation + 1,
                status='stopped' if row.status == 'stopped' else 'pending'))
        self._session.execute(update(RecoveryScopeRow).where(RecoveryScopeRow.root_id == root_id,
            RecoveryScopeRow.status != 'split').values(status='pending', accepted_attempt_id=None,
                generation=RecoveryScopeRow.generation + 1))
        self._session.flush()

    def stop_root(self, root_id, reason):
        changed = self._session.execute(update(RecoveryRootRow).where(RecoveryRootRow.id == root_id).values(status='stopped', reason=reason))
        if changed.rowcount != 1:
            raise WorkStateError('исходная область отсутствует')
        self._session.flush()

    def begin_recovery_attempt(self, root_id, scope_id, *, task_id, run_id, stage, assigned_model, prompt_hash):
        from linguary.application.recovery import RecoveryError
        from linguary.domain.llm_log import CallContext
        root = self.get_root(root_id)
        scope = self.get_scope(scope_id)
        if root is None or scope is None or scope.root_id != root_id:
            raise RecoveryError('unsafe_partition', root_id)
        if root.status == 'stopped':
            raise RecoveryError(root.reason or 'interrupted', root_id)
        if scope.status != 'pending':
            raise RecoveryError('interrupted', root_id)
        conditions = self.latest_conditions(root.book_id)
        run = self._session.get(ProcessingRunRow, run_id)
        if (conditions is None or conditions.id != root.conditions_id or run is None
                or run.book_id != root.book_id or run.status != 'active'
                or root.manifest.get('active_run_id') != run_id):
            raise RecoveryError('conditions_unavailable', root_id)
        count = self._session.scalar(select(__import__('sqlalchemy').func.count()).select_from(LLMAttemptRow).where(LLMAttemptRow.scope_id == scope_id))
        policy = root.manifest['policy']
        if count >= policy['max_scope_calls'] or root.calls_used >= policy['max_calls'] + root.extra_calls:
            self.stop_root(root_id, 'limit')
            return None
        claimed = self.claim_scope(scope_id, expected_generation=scope.generation)
        changed = self._session.execute(update(RecoveryRootRow).where(RecoveryRootRow.id == root_id,
            RecoveryRootRow.generation == root.generation,
            RecoveryRootRow.calls_used == root.calls_used,
            RecoveryRootRow.status.in_(('pending', 'running')),
        ).values(calls_used=RecoveryRootRow.calls_used + 1, status='running'))
        if changed.rowcount != 1:
            raise RecoveryError('interrupted', root_id)
        context = CallContext(root.conditions_id, stage, scope_id, scope.ordered_blocks,
            str(uuid4()), claimed.generation, assigned_model, prompt_hash,
            run_id=run_id, task_id=task_id, root_id=root_id)
        attempt = CallAttempt(context.attempt_id, root.book_id, state='created',
            generation=claimed.generation, ordinal=root.calls_used + 1, conditions_id=root.conditions_id,
            task_id=task_id, run_id=run_id, root_id=root_id, scope_id=scope_id,
            context={**context.to_dict(), 'root_generation': root.generation})
        self.add_attempt(attempt)
        return context

    def record_recovery_failure(self, attempt_id, violation):
        from linguary.application.recovery import local_failure_signature
        attempt = self.get_attempt(attempt_id)
        if attempt is None or attempt.scope_id is None:
            return False
        root = self.get_root(attempt.root_id)
        scope = self.get_scope(attempt.scope_id)
        if root.generation != attempt.context['root_generation'] or scope.generation != attempt.generation:
            return False
        signature = local_failure_signature(violation, scope.ordered_blocks)
        outcome = {**thaw(attempt.outcome), 'failure_signature': signature,
                   'violation': violation.to_dict() if violation is not None else None}
        state = 'failed' if attempt.state in {'sent', 'unknown', 'created'} else attempt.state
        self._session.execute(update(LLMAttemptRow).where(LLMAttemptRow.id == attempt_id,
            LLMAttemptRow.state == attempt.state).values(state=state, outcome_json=outcome))
        self._session.execute(update(RecoveryScopeRow).where(RecoveryScopeRow.id == scope.id,
            RecoveryScopeRow.generation == attempt.generation, RecoveryScopeRow.status == 'claimed').values(status='pending'))
        previous = self._session.scalars(select(LLMAttemptRow).where(LLMAttemptRow.scope_id == scope.id)
            .order_by(LLMAttemptRow.ordinal.desc()).limit(2)).all()
        self._session.flush()
        return bool(signature and len(previous) == 2 and all(row.state == 'rejected' and row.outcome_json.get('failure_signature') == signature for row in previous))

    def split_scope(self, scope_id, *, expected_generation, children):
        scope = self.get_scope(scope_id)
        if scope is None or tuple(block for group in children for block in group) != scope.ordered_blocks:
            raise WorkStateError('разбиение не покрывает область точно один раз')
        changed = self._session.execute(update(RecoveryScopeRow).where(RecoveryScopeRow.id == scope_id,
            RecoveryScopeRow.generation == expected_generation, RecoveryScopeRow.status == 'pending').values(status='split'))
        if changed.rowcount != 1:
            raise WorkStateError('область изменилась до разбиения')
        for blocks in children:
            self.add_scope(RecoveryScope(str(uuid4()), scope.root_id, tuple(blocks), parent_id=scope.id))

    def owns_recovery_attempt(self, attempt_id):
        attempt = self.get_attempt(attempt_id)
        if attempt is None or attempt.root_id is None:
            return False
        root = self.get_root(attempt.root_id)
        scope = self.get_scope(attempt.scope_id)
        if root is None or scope is None:
            return False
        return (root.generation == attempt.context['root_generation'] and root.status in {'pending', 'running'}
                and root.manifest['active_run_id'] == attempt.run_id
                and scope.generation == attempt.generation and scope.status == 'claimed'
                and attempt.state == 'validated')

    def mark_recovery_accepted(self, attempt_id):
        attempt = self.get_attempt(attempt_id)
        if not self.owns_recovery_attempt(attempt_id):
            raise WorkStateError('попытка больше не владеет областью')
        changed = self._session.execute(update(RecoveryScopeRow).where(RecoveryScopeRow.id == attempt.scope_id,
            RecoveryScopeRow.generation == attempt.generation, RecoveryScopeRow.status == 'claimed',
            RecoveryScopeRow.accepted_attempt_id.is_(None)).values(status='accepted', accepted_attempt_id=attempt_id))
        if changed.rowcount != 1:
            raise WorkStateError('область изменилась до принятия')

    def root_coverage_complete(self, root_id):
        root = self.get_root(root_id)
        if root is None or root.status == 'stopped':
            return False
        manifest = root.manifest
        scopes = self.list_scopes(root_id)
        leaves = [scope for scope in scopes if scope.status != 'split']
        blocks = list(manifest.get('historical_coverage', ()))
        if any(scope.status != 'accepted' for scope in leaves):
            return False
        for scope in leaves:
            blocks.extend(scope.ordered_blocks)
        if len(blocks) != len(set(blocks)) or set(blocks) != set(manifest['ordered_blocks']):
            return False
        run_id = manifest['active_run_id']
        run = self._session.get(ProcessingRunRow, run_id)
        conditions = self.latest_conditions(root.book_id)
        if run is None or run.status != 'active' or conditions is None or conditions.id != root.conditions_id:
            return False
        from linguary.adapters.db.repositories import SqlAlchemyBlockResultRepository
        results = SqlAlchemyBlockResultRepository(self._session)
        for scope in leaves:
            for block in scope.ordered_blocks:
                binding = self.get_binding('result', f'{run_id}:{block}')
                if binding is None:
                    return False
                if binding['attempt_id'] != scope.accepted_attempt_id:
                    replacement = self.get_attempt(binding['attempt_id'])
                    if (replacement is None or replacement.state != 'accepted' or replacement.run_id != run_id
                            or replacement.conditions_id != root.conditions_id
                            or replacement.context.get('stage') != 'block_regeneration'):
                        return False
        if any(results.get(run_id, block) is None for block in blocks):
            return False
        self._session.execute(update(RecoveryRootRow).where(RecoveryRootRow.id == root.id,
            RecoveryRootRow.generation == root.generation).values(status='complete', reason=None))
        return True

    def invalidate_run_coverage(self, book_id, run_id):
        for root in self.list_roots(book_id, run_id=run_id):
            manifest = thaw(root.manifest)
            manifest['historical_coverage'] = []
            self._session.execute(update(RecoveryRootRow).where(RecoveryRootRow.id == root.id,
                RecoveryRootRow.generation == root.generation).values(generation=RecoveryRootRow.generation + 1,
                    manifest_json=manifest, status='stopped' if root.status == 'stopped' else 'pending'))
            self._session.execute(update(RecoveryScopeRow).where(RecoveryScopeRow.root_id == root.id,
                RecoveryScopeRow.status != 'split').values(status='pending', accepted_attempt_id=None,
                    generation=RecoveryScopeRow.generation + 1))
        self._session.flush()

    def reconcile_recovery_attempts(self):
        rows = self._session.scalars(select(LLMAttemptRow).where(LLMAttemptRow.root_id.is_not(None),
            LLMAttemptRow.state.in_(('created', 'sent', 'unknown', 'validated')))).all()
        for row in rows:
            scope = self._session.get(RecoveryScopeRow, row.scope_id)
            root = self._session.get(RecoveryRootRow, row.root_id)
            if scope is None or root is None:
                continue
            if (scope.status == 'accepted' or scope.generation != row.generation
                    or root.generation != row.context_json.get('root_generation')):
                if row.state == 'created':
                    row.state = 'cancelled'
                elif row.state == 'validated':
                    row.state = 'rejected'
                else:
                    row.state = 'unknown'
                row.outcome_json = {**row.outcome_json, 'decision': 'interrupted'}
                continue
            if row.state == 'created':
                row.state = 'cancelled'
                scope.status = 'pending'
            else:
                row.state = 'unknown' if row.state in {'sent', 'unknown'} else 'rejected'
                root.status, root.reason = 'stopped', 'interrupted'
                scope.status = 'stopped'
            scope.generation += 1
            root.generation += 1
            row.outcome_json = {**row.outcome_json, 'decision': 'interrupted'}
        self._session.flush()

    def scope_attempt_count(self, scope_id):
        from sqlalchemy import func
        return self._session.scalar(select(func.count()).select_from(LLMAttemptRow).where(LLMAttemptRow.scope_id == scope_id))

    def stale_reviews(self, book_id):
        self._session.execute(update(QualityReviewRow).where(QualityReviewRow.book_id == book_id,
            QualityReviewRow.state != 'stale').values(state='stale', revision=QualityReviewRow.revision + 1))
        self._session.flush()

    def audit_stale_attempt(self, attempt_id, *, call_id=None):
        row = self._session.get(LLMAttemptRow, attempt_id)
        if row is None:
            raise WorkStateError('попытка отсутствует')
        row.outcome_json = {**row.outcome_json, 'decision': 'stale',
                            **({'call_id': call_id} if call_id is not None else {})}
        self._session.flush()
