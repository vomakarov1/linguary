"""Durable work store; all operations use the caller's short transaction."""
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
import json
from uuid import uuid4

from sqlalchemy import delete, func, or_, select, update
from linguary.adapters.db.models import (
    BlockResultRow, BookConditionsRow, WorkBindingRow, ArtifactRevisionRow, RecoveryRootRow,
    RecoveryScopeRow, LLMAttemptRow, QualityReviewRow, QualityReservationRow,
    LLMCallLogRow, ProcessingRunRow, ProcessingTaskRow, CorrectionCheckRow, CorrectionCandidateRow, CorrectionInputRow, DocumentBlockRow,
)
from linguary.adapters.db.mapping import conditions_from_row, conditions_to_row
from linguary.domain.book_conditions import WorkStateError, identifier, nonnegative, thaw, version
from linguary.domain.recovery import CallAttempt, RecoveryRoot, RecoveryScope
from linguary.domain.quality_review import QualityReview, QualityReservation
from linguary.domain.correction import (
    CorrectionCheck, CorrectionCandidate, CorrectionDecision, CORRECTION_POLICY_VERSION_V2,
)
from linguary.domain.content_plan import canonical_sha


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
        return self._binding_from_row(row)

    @staticmethod
    def _binding_from_row(row):
        if row is None:
            return None
        version(row.payload_json.get('schema_version'))
        return {column.name: getattr(row, column.name) for column in row.__table__.columns}

    def result_bindings(self, run_id):
        prefix = f'{run_id}:'
        rows = self._session.scalars(select(WorkBindingRow).where(
            WorkBindingRow.entity_kind == 'result', WorkBindingRow.entity_id >= prefix,
            WorkBindingRow.entity_id < f'{run_id};')).all()
        return {row.entity_id[len(prefix):]: self._binding_from_row(row) for row in rows}

    def set_book_paused(
        self, book_id: str, conditions_id: str | None, paused: bool
    ) -> None:
        if type(paused) is not bool:
            raise WorkStateError('некорректный признак паузы книги')
        if not paused:
            if conditions_id is not None:
                self._owned(BookConditionsRow, conditions_id, book_id)
            self._session.execute(
                delete(WorkBindingRow).where(
                    WorkBindingRow.entity_kind == 'book_queue',
                    WorkBindingRow.entity_id == book_id,
                    WorkBindingRow.book_id == book_id,
                )
            )
            self._session.flush()
            return
        if not conditions_id:
            raise WorkStateError('ссылка на отсутствующую или чужую работу')
        self._owned(BookConditionsRow, conditions_id, book_id)
        payload = {'schema_version': 1, 'paused': True}
        existing = self._session.get(WorkBindingRow, ('book_queue', book_id))
        if existing is None:
            self._add(
                WorkBindingRow(
                    entity_kind='book_queue',
                    entity_id=book_id,
                    book_id=book_id,
                    conditions_id=conditions_id,
                    attempt_id=None,
                    artifact_revision_id=None,
                    payload_json=payload,
                )
            )
            return
        if existing.book_id != book_id:
            raise WorkStateError('ссылка на отсутствующую или чужую работу')
        existing.conditions_id = conditions_id
        existing.payload_json = payload
        self._session.flush()

    def is_book_paused(self, book_id: str) -> bool:
        row = self._session.get(WorkBindingRow, ('book_queue', book_id))
        if row is None or row.book_id != book_id:
            return False
        payload = row.payload_json or {}
        version(payload.get('schema_version'))
        return bool(payload.get('paused', True))

    def correction_checks(self, run_id, section_id):
        rows = self._session.scalars(select(CorrectionCheckRow).join(
            DocumentBlockRow, DocumentBlockRow.id == CorrectionCheckRow.block_id,
        ).where(CorrectionCheckRow.run_id == run_id,
                CorrectionCheckRow.section_id == section_id).order_by(DocumentBlockRow.order)).all()
        return tuple(CorrectionCheck(row.run_id, row.block_id, row.section_id,
                                     row.input_sha256, row.attempt_id, row.basis, row.candidate_id) for row in rows)

    def replace_correction_checks(self, run_id, section_id, checks):
        checks = tuple(checks)
        run = self._session.get(ProcessingRunRow, run_id)
        if run is None or run.status != 'active':
            raise WorkStateError('запуск исправления больше не активен')
        conditions = self.latest_conditions(run.book_id)
        binding = self.get_binding('run', run_id)
        if (conditions is None or binding is None or binding['book_id'] != run.book_id
                or binding['conditions_id'] != conditions.id
                or len({check.block_id for check in checks}) != len(checks)):
            raise WorkStateError('условия отметок исправления не совпадают')
        blocks_by_id = (
            {
                row.id: row
                for row in self._session.scalars(
                    select(DocumentBlockRow).where(
                        DocumentBlockRow.book_id == run.book_id,
                        DocumentBlockRow.section_id == section_id,
                    )
                ).all()
            }
            if checks
            else {}
        )
        attempt_ids = {check.attempt_id for check in checks if check.attempt_id}
        attempts_by_id = (
            {
                row.id: self._attempt_from_row(row)
                for row in self._session.scalars(
                    select(LLMAttemptRow).where(LLMAttemptRow.id.in_(attempt_ids))
                ).all()
            }
            if attempt_ids
            else {}
        )
        is_v2 = binding['payload_json'].get('correction', {}).get('policy_version') == CORRECTION_POLICY_VERSION_V2
        groups_by_attempt = {}
        candidates_by_id = {}
        if is_v2 and checks:
            if attempt_ids:
                groups_by_attempt = {
                    row.entity_id: self._binding_from_row(row)
                    for row in self._session.scalars(
                        select(WorkBindingRow).where(
                            WorkBindingRow.entity_kind == 'correction_group',
                            WorkBindingRow.entity_id.in_(attempt_ids),
                        )
                    ).all()
                }
            candidate_ids = {check.candidate_id for check in checks if check.candidate_id}
            if candidate_ids:
                candidates_by_id = {
                    row.candidate_id: self._correction_candidate_from_row(row)
                    for row in self._session.scalars(
                        select(CorrectionCandidateRow).where(
                            CorrectionCandidateRow.candidate_id.in_(candidate_ids)
                        )
                    ).all()
                }
        for check in checks:
            block = blocks_by_id.get(check.block_id)
            attempt = attempts_by_id.get(check.attempt_id)
            if (check.run_id != run_id or check.section_id != section_id or block is None
                    or block.book_id != run.book_id or block.section_id != section_id
                    or attempt is None or attempt.state != 'accepted'
                    or attempt.book_id != run.book_id or attempt.run_id != run_id
                    or attempt.conditions_id != conditions.id
                    or attempt.context.get('stage') != 'chapter_correction'
                    or check.block_id not in attempt.context.get('ordered_block_ids', ())):
                raise WorkStateError('отметка не принадлежит принятому исправлению главы')
            if is_v2:
                if check.basis == 'unchanged':
                    group = groups_by_attempt.get(check.attempt_id)
                    if (group is None or group['payload_json'].get('run_id') != run_id
                            or check.block_id not in group['payload_json'].get('unchanged_block_ids', ())):
                        raise WorkStateError('отсутствует квитанция неизменённого блока')
                else:
                    candidate = candidates_by_id.get(check.candidate_id)
                    if (candidate is None or candidate.run_id != run_id or candidate.block_id != check.block_id
                            or candidate.proposal_attempt_id != check.attempt_id
                            or (check.basis == 'confirmed' and (candidate.state != 'auto_applied'
                                or not self._confirmed_candidate(candidate)))
                            or (check.basis == 'human' and (candidate.decision is None
                                or candidate.state not in {'accepted', 'kept'}))
                            or check.basis not in {'confirmed', 'human'}):
                        raise WorkStateError('отсутствует принятое решение исправления')
        for check in checks:
            self._session.merge(CorrectionCheckRow(**asdict(check)))
        self._session.flush()

    def get_correction_candidate(self, candidate_id):
        row = self._session.get(CorrectionCandidateRow, candidate_id) if candidate_id else None
        return self._correction_candidate_from_row(row)

    @staticmethod
    def _correction_candidate_from_row(row):
        if row is None:
            return None
        return CorrectionCandidate(row.candidate_id, row.book_id, row.run_id, row.conditions_id,
            row.section_id, row.block_id, row.input_sha256, row.input_generation, row.proposal_sha256,
            row.payload_json, row.group_id, row.proposal_attempt_id, row.state,
            row.review_attempt_id, row.review_json, row.decision_json)

    def correction_input_generations(self, run_id):
        return {row.block_id: row.generation for row in self._session.scalars(
            select(CorrectionInputRow).where(CorrectionInputRow.run_id == run_id)).all()}

    def observe_correction_input(self, run_id, block_id, fingerprint):
        run = self._session.get(ProcessingRunRow, run_id)
        block = self._session.get(DocumentBlockRow, block_id)
        if run is None or run.status != 'active' or block is None or block.book_id != run.book_id:
            raise WorkStateError('вход исправления не принадлежит активному запуску')
        row = self._session.get(CorrectionInputRow, (run_id, block_id))
        if row is None:
            self._add(CorrectionInputRow(run_id=run_id, block_id=block_id, generation=1, fingerprint=fingerprint))
            return 1
        if row.fingerprint == fingerprint:
            return row.generation
        generation = row.generation + 1
        changed = self._session.execute(update(CorrectionInputRow).where(
            CorrectionInputRow.run_id == run_id, CorrectionInputRow.block_id == block_id,
            CorrectionInputRow.generation == row.generation, CorrectionInputRow.fingerprint == row.fingerprint,
        ).values(generation=generation, fingerprint=fingerprint))
        if changed.rowcount != 1:
            raise WorkStateError('поколение входа изменилось одновременно')
        self._session.flush()
        return generation

    def advance_correction_input(self, run_id, block_id, *, expected_generation, fingerprint):
        changed = self._session.execute(update(CorrectionInputRow).where(
            CorrectionInputRow.run_id == run_id, CorrectionInputRow.block_id == block_id,
            CorrectionInputRow.generation == expected_generation,
        ).values(fingerprint=fingerprint))
        if changed.rowcount != 1:
            raise WorkStateError('поколение входа изменилось до принятия исправления')
        self._session.flush()

    def list_correction_candidates(self, run_id, *, section_id=None, states=None):
        query = select(CorrectionCandidateRow).join(
            DocumentBlockRow, DocumentBlockRow.id == CorrectionCandidateRow.block_id,
        ).where(CorrectionCandidateRow.run_id == run_id)
        if section_id is not None:
            query = query.where(CorrectionCandidateRow.section_id == section_id)
        if states is not None:
            query = query.where(CorrectionCandidateRow.state.in_(tuple(states)))
        return tuple(self._correction_candidate_from_row(row) for row in self._session.scalars(
            query.order_by(DocumentBlockRow.order, CorrectionCandidateRow.candidate_id)).all())

    def add_correction_candidates(self, candidates):
        for candidate in candidates:
            run = self._owned(ProcessingRunRow, candidate.run_id, candidate.book_id)
            block = self._owned(DocumentBlockRow, candidate.block_id, candidate.book_id)
            attempt = self.get_attempt(candidate.proposal_attempt_id)
            inputs = self._session.get(CorrectionInputRow, (candidate.run_id, candidate.block_id))
            invalid_evidence_human = (
                isinstance(candidate.payload, Mapping)
                and candidate.payload.get('evidence_valid') is False
                and candidate.state == 'awaiting_human'
                and candidate.review is not None
            )
            valid_initial_state = (
                (candidate.state == 'awaiting_review' and candidate.review is None)
                or invalid_evidence_human
            )
            if (run.status != 'active' or block.section_id != candidate.section_id
                    or inputs is None or inputs.generation != candidate.input_generation
                    or inputs.fingerprint != canonical_sha(dict(input_sha256=candidate.input_sha256,
                        source_result_binding=thaw(candidate.payload['source_binding'])))
                    or not valid_initial_state or candidate.decision is not None
                    or attempt is None or attempt.state not in {'validated', 'accepted'}
                    or attempt.book_id != candidate.book_id or attempt.run_id != candidate.run_id
                    or attempt.conditions_id != candidate.conditions_id
                    or attempt.context.get('stage') != 'chapter_correction'
                    or attempt.context.get('policy_version') != CORRECTION_POLICY_VERSION_V2
                    or candidate.block_id not in attempt.context.get('ordered_block_ids', ())):
                raise WorkStateError('кандидат не принадлежит предложению исправления')
            previous = self._session.scalar(select(CorrectionCandidateRow).where(
                CorrectionCandidateRow.run_id == candidate.run_id,
                CorrectionCandidateRow.block_id == candidate.block_id,
                CorrectionCandidateRow.input_generation == candidate.input_generation))
            if previous is not None:
                if previous.candidate_id != candidate.candidate_id or previous.proposal_sha256 != candidate.proposal_sha256:
                    raise WorkStateError('предложение для этих входов уже сохранено')
                continue
            self._add(CorrectionCandidateRow(candidate_id=candidate.candidate_id,
                book_id=candidate.book_id, run_id=candidate.run_id, conditions_id=candidate.conditions_id,
                section_id=candidate.section_id, block_id=candidate.block_id,
                input_sha256=candidate.input_sha256, input_generation=candidate.input_generation,
                proposal_sha256=candidate.proposal_sha256, payload_json=thaw(candidate.payload),
                group_id=candidate.group_id, proposal_attempt_id=candidate.proposal_attempt_id,
                review_attempt_id=candidate.review_attempt_id,
                review_json=thaw(candidate.review) if candidate.review is not None else None,
                state=candidate.state))

    def record_correction_verdict(self, candidate_id, *, proposal_sha256, input_sha256,
                                  review_attempt_id, review, state):
        candidate = self.get_correction_candidate(candidate_id)
        attempt = self.get_attempt(review_attempt_id)
        if (candidate is None or state not in {'auto_applied', 'awaiting_human'}
                or candidate.state != 'awaiting_review' or candidate.review is not None
                or candidate.proposal_sha256 != proposal_sha256 or candidate.input_sha256 != input_sha256
                or review.get('candidate_id') != candidate_id or review.get('proposal_sha256') != proposal_sha256
                or attempt is None or attempt.state not in {'validated', 'accepted'}
                or attempt.run_id != candidate.run_id or attempt.book_id != candidate.book_id
                or attempt.conditions_id != candidate.conditions_id
                or attempt.context.get('stage') != 'chapter_correction_review'
                or attempt.context.get('policy_version') != CORRECTION_POLICY_VERSION_V2
                or candidate.block_id not in attempt.context.get('ordered_block_ids', ())):
            raise WorkStateError('вердикт проверки относится к изменившемуся кандидату')
        if state == 'auto_applied' and (candidate.payload.get('evidence_valid') is not True
                or review.get('verdict') != 'confirmed'
                or any(review.get(k) is not True for k in ('error_supported', 'evidence_valid', 'minimal', 'no_semantic_loss'))):
            raise WorkStateError('кандидат не подтверждён всеми проверками')
        result = self._session.execute(update(CorrectionCandidateRow).where(
            CorrectionCandidateRow.candidate_id == candidate_id,
            CorrectionCandidateRow.state == 'awaiting_review',
            CorrectionCandidateRow.review_json.is_(None),
            CorrectionCandidateRow.proposal_sha256 == proposal_sha256,
        ).values(state=state, review_attempt_id=review_attempt_id, review_json=thaw(review)))
        if result.rowcount != 1:
            raise WorkStateError('кандидат изменился до сохранения вердикта')
        self._session.flush()

    def resolve_correction_candidate(self, candidate_id, *, proposal_sha256, input_sha256, decision):
        decision = CorrectionDecision(decision['action'], decision['reason'])
        result = self._session.execute(update(CorrectionCandidateRow).where(
            CorrectionCandidateRow.candidate_id == candidate_id,
            CorrectionCandidateRow.state == 'awaiting_human',
            CorrectionCandidateRow.proposal_sha256 == proposal_sha256,
            CorrectionCandidateRow.input_sha256 == input_sha256,
            CorrectionCandidateRow.decision_json.is_(None),
        ).values(state='accepted' if decision.action == 'accept' else 'kept',
                 decision_json={'action': decision.action, 'reason': decision.reason}))
        if result.rowcount != 1:
            raise WorkStateError('кандидат изменился до принятия решения')
        self._session.flush()

    def stale_correction_candidates(self, run_id, *, block_ids=None, candidate_ids=None):
        query = update(CorrectionCandidateRow).where(CorrectionCandidateRow.run_id == run_id,
                                                    CorrectionCandidateRow.state != 'stale')
        if block_ids is not None:
            query = query.where(CorrectionCandidateRow.block_id.in_(tuple(block_ids)))
        if candidate_ids is not None:
            query = query.where(CorrectionCandidateRow.candidate_id.in_(tuple(candidate_ids)))
        self._session.execute(query.values(state='stale'))
        self._session.flush()

    def _confirmed_candidate(self, candidate):
        review = candidate.review
        attempt = self.get_attempt(candidate.review_attempt_id) if candidate.review_attempt_id else None
        artifact = None if not review else self.get_artifact(review.get('artifact_revision_id'))
        if artifact is None:
            return False
        verdict = {k: thaw(v) for k, v in review.items() if k != 'artifact_revision_id'}
        encoded = json.dumps(artifact['body'], ensure_ascii=False, sort_keys=True,
                             separators=(',', ':'), allow_nan=False)
        return bool(candidate.payload.get('evidence_valid') is True
            and review.get('verdict') == 'confirmed'
            and all(review.get(k) is True for k in ('error_supported', 'evidence_valid', 'minimal', 'no_semantic_loss'))
            and attempt is not None and attempt.state == 'accepted'
            and attempt.context.get('stage') == 'chapter_correction_review'
            and attempt.run_id == candidate.run_id and attempt.conditions_id == candidate.conditions_id
            and artifact['kind'] == 'correction_review' and artifact['attempt_id'] == attempt.id
            and artifact['run_id'] == candidate.run_id and artifact['conditions_id'] == candidate.conditions_id
            and artifact['book_id'] == candidate.book_id
            and sha256(encoded.encode()).hexdigest() == artifact['sha256']
            and verdict in artifact['body'].get('response', {}).get('candidates', ()))

    def _candidate_successor(self, body, attempt, *, require_current=False):
        candidate = self.get_correction_candidate(body.get('correction_candidate_id'))
        if (candidate is None or candidate.proposal_attempt_id != attempt.id
                or candidate.proposal_sha256 != body.get('proposal_sha256')
                or candidate.block_id != body.get('result', {}).get('block_id')
                or candidate.book_id != attempt.book_id
                or candidate.run_id != attempt.run_id or candidate.conditions_id != attempt.conditions_id
                or thaw(candidate.payload.get('result')) != body.get('result')
                or candidate.payload.get('source_binding', {}).get('attempt_id') != body.get('source_attempt')
                or candidate.payload.get('source_binding', {}).get('artifact_revision_id') != body.get('source_revision')):
            return False
        if require_current and candidate.state not in {'auto_applied', 'accepted'}:
            return False
        return bool((candidate.state in {'accepted', 'stale'} and candidate.decision
                     and candidate.decision.get('action') == 'accept' and candidate.decision.get('reason'))
                    or (candidate.state in {'auto_applied', 'stale'} and self._confirmed_candidate(candidate)))

    def delete_correction_checks(self, run_id, *, section_id=None, block_ids=None):
        query = delete(CorrectionCheckRow).where(CorrectionCheckRow.run_id == run_id)
        if section_id is not None:
            query = query.where(CorrectionCheckRow.section_id == section_id)
        if block_ids is not None:
            query = query.where(CorrectionCheckRow.block_id.in_(tuple(block_ids)))
        self._session.execute(query)
        self._session.flush()

    def bind_operation_refs(self, task_id, *, book_id, conditions_id, operation_refs):
        refs = [dict(ref) for ref in operation_refs]
        if len({ref['root_id'] for ref in refs}) != len(refs):
            raise WorkStateError('conditions_unavailable')
        task = self._owned(ProcessingTaskRow, task_id, book_id)
        if task.status == 'superseded':
            raise WorkStateError('conditions_unavailable')
        for ref in refs:
            root = self.get_root(ref['root_id'])
            if (root is None or root.book_id != book_id or root.conditions_id != conditions_id
                    or root.policy_version != ref['policy_version'] or root.superseded_by is not None):
                raise WorkStateError('conditions_unavailable')
        binding = self._session.get(WorkBindingRow, ('task', task_id))
        if binding is None:
            self.bind('task', task_id, book_id=book_id, conditions_id=conditions_id,
                payload={'schema_version': 1, 'operation_refs': refs})
        else:
            if binding.book_id != book_id or binding.conditions_id != conditions_id:
                raise WorkStateError('conditions_unavailable')
            payload = dict(binding.payload_json)
            if 'operation_refs' in payload and payload['operation_refs'] != refs:
                raise WorkStateError('conditions_unavailable')
            binding.payload_json = {**payload, 'operation_refs': refs}
            self._session.flush()

    def attach_legacy_attempts(self, root_id, scope_id, *, task_id, attempt_ids):
        root = self.get_root(root_id)
        scope = self.get_scope(scope_id)
        if root is None or scope is None or scope.root_id != root_id or root.policy_version is not None:
            raise WorkStateError('conditions_unavailable')
        for attempt_id in attempt_ids:
            attempt = self.get_attempt(attempt_id)
            if (attempt is None or attempt.book_id != root.book_id or attempt.conditions_id != root.conditions_id
                    or attempt.run_id != root.manifest.get('active_run_id') or attempt.task_id != task_id
                    or attempt.root_id is not None or attempt.scope_id is not None
                    or attempt.state not in {'failed', 'rejected', 'cancelled'}
                    or tuple(attempt.context.get('ordered_block_ids', ())) != scope.ordered_blocks):
                raise WorkStateError('conditions_unavailable')
            self._session.execute(update(LLMAttemptRow).where(
                LLMAttemptRow.id == attempt_id, LLMAttemptRow.root_id.is_(None),
                LLMAttemptRow.scope_id.is_(None),
            ).values(root_id=root_id, scope_id=scope_id))
        self._session.flush()

    def list_task_attempts(self, task_id):
        rows = self._session.scalars(select(LLMAttemptRow).where(LLMAttemptRow.task_id == task_id)).all()
        return tuple(self._attempt_from_row(row) for row in rows)

    def add_root(self, root):
        self._owned(BookConditionsRow, root.conditions_id, root.book_id)
        self._add(RecoveryRootRow(id=root.id, book_id=root.book_id, conditions_id=root.conditions_id,
                  root_key=root.root_key, manifest_json=_json(root.manifest), generation=root.generation,
                  calls_used=root.calls_used, extra_calls=root.extra_calls, status=root.status, reason=root.reason,
                  policy_version=root.policy_version, operation_kind=root.operation_kind,
                  superseded_by=root.superseded_by, compatibility_json=None if root.compatibility is None else _json(root.compatibility)))

    def get_root(self, root_id):
        row = self._session.get(RecoveryRootRow, root_id)
        if row is None:
            return None
        return RecoveryRoot(row.id, row.book_id, row.conditions_id, row.root_key,
                            row.manifest_json, row.generation, row.calls_used,
                            row.extra_calls, row.status, row.reason, policy_version=row.policy_version,
                            operation_kind=row.operation_kind, superseded_by=row.superseded_by,
                            compatibility=row.compatibility_json)

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
        return self._scope_from_row(row)

    @staticmethod
    def _scope_from_row(row):
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
        return self._attempt_from_row(row)

    @staticmethod
    def _attempt_from_row(row):
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
        if root.policy_version is not None and not self.owns_recovery_attempt(attempt_id):
            raise WorkStateError('попытка больше не владеет областью')
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
            provenance = None if root.policy_version is None else {
                'schema_version': 1, 'root_id': root.id, 'policy_version': root.policy_version,
                'prompt_hash': attempt.context['prompt_hash'], 'attempt_id': attempt.id}
            artifacts = []
            bindings_to_create = []
            for result in results:
                body = {'schema_version': 1, 'result': asdict(result)}
                if provenance is not None:
                    body['provenance'] = provenance
                artifact_id = str(uuid4())
                artifacts.append(dict(id=artifact_id, book_id=root.book_id,
                    conditions_id=root.conditions_id, run_id=attempt.run_id, attempt_id=attempt_id,
                    kind='block_result', scope_id=result.block_id, body=body))
                bindings_to_create.append((result.block_id, artifact_id))
            if artifacts:
                self.archive_artifacts(artifacts)
            for block_id, artifact_id in bindings_to_create:
                self.bind('result', f'{attempt.run_id}:{block_id}', book_id=root.book_id,
                          conditions_id=root.conditions_id, attempt_id=attempt_id, artifact_revision_id=artifact_id,
                          payload=provenance)
            self.transition_attempt(attempt_id, 'accepted', expected_state='validated')

    def archive_artifacts(self, artifacts):
        artifacts = tuple(artifacts)
        if not artifacts:
            return ()
        ids = []
        seen_ids = set()
        rows_to_add = []
        checked_owned = set()
        now = datetime.now(UTC)
        for artifact in artifacts:
            artifact_id = artifact.get('id') or str(uuid4())
            identifier(artifact_id)
            if artifact_id in seen_ids or self._session.get(ArtifactRevisionRow, artifact_id) is not None:
                raise WorkStateError('архивная версия неизменяема')
            seen_ids.add(artifact_id)
            body = _json(artifact['body'])
            encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
            digest = sha256(encoded.encode()).hexdigest()
            if artifact.get('sha256', digest) != digest:
                raise WorkStateError('контрольная сумма артефакта не совпадает')
            for model, key in ((BookConditionsRow, 'conditions_id'), (LLMAttemptRow, 'attempt_id')):
                if artifact.get(key):
                    own_key = (model, artifact[key], artifact['book_id'])
                    if own_key not in checked_owned:
                        self._owned(model, artifact[key], artifact['book_id'])
                        checked_owned.add(own_key)
            rows_to_add.append(ArtifactRevisionRow(id=artifact_id, book_id=artifact['book_id'],
                conditions_id=artifact.get('conditions_id'), run_id=artifact.get('run_id'),
                kind=artifact['kind'], scope_id=artifact['scope_id'], body_json=body,
                sha256=digest, attempt_id=artifact.get('attempt_id'), created_at=now))
            ids.append(artifact_id)
        self._session.add_all(rows_to_add)
        self._session.flush()
        return tuple(ids)

    def get_artifact(self, artifact_id):
        row = self._session.get(ArtifactRevisionRow, artifact_id)
        if row is None:
            return None
        version(row.body_json.get('schema_version'))
        return {'id': row.id, 'book_id': row.book_id, 'body': row.body_json,
                'sha256': row.sha256, 'attempt_id': row.attempt_id,
                'conditions_id': row.conditions_id, 'run_id': row.run_id,
                'kind': row.kind, 'scope_id': row.scope_id}

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
                if actual > current.estimate or review.spent + actual > review.budget_limit:
                    values['state'] = 'limit'
            if review.state == 'stale':
                values['state'] = 'stale'
            changed = self._session.execute(update(QualityReviewRow).where(
                QualityReviewRow.id == current.review_id, QualityReviewRow.revision == review.revision,
                QualityReviewRow.reserved >= current.estimate,
            ).values(**values))
            if changed.rowcount != 1:
                raise WorkStateError('состояние вычитки изменилось')
        return updated

    def review_attempt_count(self, review_id):
        return len(self._session.scalars(select(LLMAttemptRow.id).where(LLMAttemptRow.review_id == review_id)).all())

    def find_review(self, book_id, request_key):
        row = self._session.scalar(select(QualityReviewRow).where(QualityReviewRow.book_id == book_id,
            QualityReviewRow.request_key == request_key))
        return None if row is None else self.get_review(row.id)

    def list_reviews(self, book_id):
        return tuple(self.get_review(row.id) for row in self._session.scalars(
            select(QualityReviewRow).where(QualityReviewRow.book_id == book_id)).all())

    def review_progress(self, review_id):
        row = self._session.get(QualityReviewRow, review_id)
        return {} if row is None else {key: row.findings_json.get(key) for key in ('checked', 'missing', 'reason', 'partial')}

    def update_review(self, review_id, *, state, findings=(), checked=(), missing=(), reason=None, partial=False):
        row = self._session.get(QualityReviewRow, review_id)
        if row is None:
            raise WorkStateError('вычитка не найдена')
        if state == 'running' and row.state not in ('queued', 'running'):
            raise WorkStateError('вычитка остановлена')
        row.state = row.state if row.state in ('stale', 'unknown', 'limit') else state
        row.findings_json = dict(schema_version=1, items=thaw(findings), checked=list(checked),
            missing=list(missing), reason=reason, partial=partial)
        row.revision += 1
        self._session.flush()

    def mark_review_sent(self, attempt_id):
        attempt = self.get_attempt(attempt_id)
        review = self.get_review(attempt.review_id)
        if review.state not in ('queued', 'running'):
            raise WorkStateError('вычитка остановлена')
        self.transition_attempt(attempt_id, 'sent', expected_state='reserved')

    def cancel_review_unsent(self, attempt_id):
        attempt = self.get_attempt(attempt_id)
        reservation = self.get_reservation(attempt_id)
        if (attempt is None or reservation is None or attempt.review_id != reservation.review_id
                or attempt.state != 'reserved' or reservation.state != 'reserved'):
            raise WorkStateError('попытка уже отправлена или изменена')
        reservation.transition('released')
        review = self.get_review(attempt.review_id)
        with self._session.begin_nested():
            self.transition_attempt(attempt_id, 'cancelled', expected_state='reserved')
            changed = self._session.execute(update(QualityReservationRow).where(
                QualityReservationRow.attempt_id == attempt_id,
                QualityReservationRow.review_id == attempt.review_id,
                QualityReservationRow.state == 'reserved',
            ).values(state='released'))
            if changed.rowcount != 1:
                raise WorkStateError('резерв уже изменился')
            changed = self._session.execute(update(QualityReviewRow).where(
                QualityReviewRow.id == review.id, QualityReviewRow.revision == review.revision,
                QualityReviewRow.reserved >= reservation.estimate,
            ).values(reserved=QualityReviewRow.reserved - reservation.estimate,
                     revision=QualityReviewRow.revision + 1))
            if changed.rowcount != 1:
                raise WorkStateError('состояние вычитки изменилось')

    def reconcile_quality_attempts(self):
        rows = self._session.scalars(select(LLMAttemptRow).join(QualityReservationRow,
            QualityReservationRow.attempt_id == LLMAttemptRow.id).where(
            LLMAttemptRow.review_id.is_not(None), QualityReservationRow.state == 'reserved')).all()
        for row in rows:
            if row.state != 'reserved':
                self.settle(row.id, actual=None)
                if row.state == 'sent':
                    self.transition_attempt(row.id, 'unknown', expected_state='sent')
            else:
                reservation = self.get_reservation(row.id)
                changed = self._session.execute(update(QualityReservationRow).where(
                    QualityReservationRow.attempt_id == row.id, QualityReservationRow.state == 'reserved').values(state='released'))
                if changed.rowcount == 1:
                    self._session.execute(update(QualityReviewRow).where(QualityReviewRow.id == row.review_id).values(
                        reserved=QualityReviewRow.reserved - reservation.estimate, revision=QualityReviewRow.revision + 1))
                    self.transition_attempt(row.id, 'cancelled', expected_state='reserved')
        self._session.flush()

    def purge_diagnostics(self, book_id, *, call_id=None, before=None):
        has_body = or_(
            LLMCallLogRow.response_text.is_not(None),
            func.json_type(LLMCallLogRow.diagnostic_meta_json, '$.input_text').is_not(None),
        )
        query = select(LLMCallLogRow).where(LLMCallLogRow.book_id == book_id, has_body)
        if call_id is not None:
            query = query.where(LLMCallLogRow.id == call_id)
        if before is not None:
            query = query.where(LLMCallLogRow.started_at < before)
        rows = self._session.scalars(query).all()
        status = 'expired' if before is not None else 'deleted'
        for row in rows:
            meta = dict(row.diagnostic_meta_json or {'schema_version': 1})
            meta.pop('input_text', None)
            meta['body_status'] = status
            if isinstance(meta.get('body_meta'), dict):
                body_meta = dict(meta['body_meta'])
                body_meta.update(body_status=status, stored_bytes=0)
                meta['body_meta'] = body_meta
            row.diagnostic_meta_json = meta
            row.response_text = None
        self._session.flush()
        return len(rows)

    def _translation_source(self, *, book_id, run_id, conditions_id, block_id,
                            artifact_revision_id, attempt_id):
        """Resolve accepted correction archives to their verified translation source."""
        seen = set()
        seen_attempts = set()
        while artifact_revision_id not in seen:
            if attempt_id in seen_attempts:
                return None
            seen.add(artifact_revision_id)
            seen_attempts.add(attempt_id)
            artifact = self._session.get(ArtifactRevisionRow, artifact_revision_id)
            attempt = self.get_attempt(attempt_id) if attempt_id is not None else None
            if (artifact is None or attempt is None or attempt.state != 'accepted'
                    or artifact.book_id != book_id or attempt.book_id != book_id
                    or artifact.run_id != run_id or attempt.run_id != run_id
                    or artifact.conditions_id != conditions_id or attempt.conditions_id != conditions_id
                    or artifact.attempt_id != attempt_id or artifact.scope_id != block_id
                    or artifact.kind != 'block_result'
                    or block_id not in attempt.context.get('ordered_block_ids', ())):
                return None
            body = artifact.body_json
            if (not isinstance(body, dict) or body.get('schema_version') != 1
                    or not isinstance(body.get('result'), dict)
                    or body['result'].get('block_id') != block_id):
                return None
            try:
                encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
            except (TypeError, ValueError):
                return None
            if sha256(encoded.encode()).hexdigest() != artifact.sha256:
                return None
            stage = attempt.context.get('stage')
            if stage == 'chapter_correction':
                if (attempt.context.get('policy_version') == CORRECTION_POLICY_VERSION_V2
                        and not self._candidate_successor(body, attempt)):
                    return None
                artifact_revision_id = body.get('source_revision')
                attempt_id = body.get('source_attempt')
                if not isinstance(artifact_revision_id, str) or not isinstance(attempt_id, str):
                    return None
                continue
            if stage not in {'translation_unit', 'block_regeneration'}:
                return None
            if attempt.root_id is not None:
                root = self.get_root(attempt.root_id)
                scope = self.get_scope(attempt.scope_id)
                if (root is None or scope is None or root.book_id != book_id
                        or root.conditions_id != conditions_id or root.manifest.get('active_run_id') != run_id
                        or scope.root_id != root.id or scope.status != 'accepted'
                        or scope.accepted_attempt_id != attempt.id
                        or scope.generation != attempt.generation or block_id not in scope.ordered_blocks):
                    return None
                if root.policy_version is not None:
                    expected = {'schema_version': 1, 'root_id': root.id,
                                'policy_version': root.policy_version,
                                'prompt_hash': attempt.context.get('prompt_hash'), 'attempt_id': attempt.id}
                    if (attempt.context.get('policy_version') != root.policy_version
                            or body.get('provenance') != expected):
                        return None
            return attempt
        return None

    def replace_result_binding(self, entity_id, *, book_id, conditions_id, attempt_id,
                               artifact_revision_id, expected_attempt_id, provenance,
                               expected_artifact_revision_id=None):
        """Move a current projection pointer while preserving immutable archived versions."""
        conditions = self.latest_conditions(book_id)
        if conditions is None or conditions.id != conditions_id:
            raise WorkStateError('версия результата больше не активна')
        attempt = self._owned(LLMAttemptRow, attempt_id, book_id)
        artifact = self._owned(ArtifactRevisionRow, artifact_revision_id, book_id)
        current = self._session.get(WorkBindingRow, ('result', entity_id))
        correction = attempt.context_json.get('stage') == 'chapter_correction'
        if correction:
            run = self._session.get(ProcessingRunRow, attempt.run_id)
            block = self._session.get(DocumentBlockRow, artifact.scope_id)
            body = artifact.body_json
            if (current is None or run is None or run.status != 'active' or run.book_id != book_id
                    or block is None or block.book_id != book_id
                    or entity_id != f'{run.id}:{block.id}' or artifact.run_id != run.id
                    or artifact.kind != 'block_result' or attempt.state not in {'validated', 'accepted'}
                    or current.book_id != book_id or current.conditions_id != conditions_id
                    or current.attempt_id != expected_attempt_id
                    or (expected_artifact_revision_id is not None
                        and current.artifact_revision_id != expected_artifact_revision_id)
                    or block.id not in attempt.context_json.get('ordered_block_ids', ())
                    or not isinstance(body, dict) or not isinstance(body.get('result'), dict)
                    or body['result'].get('block_id') != block.id
                    or body.get('source_revision') != current.artifact_revision_id
                    or body.get('source_attempt') != current.attempt_id
                    or artifact.id == current.artifact_revision_id
                    or self._translation_source(book_id=book_id, run_id=run.id,
                        conditions_id=conditions_id, block_id=block.id,
                        artifact_revision_id=current.artifact_revision_id,
                        attempt_id=current.attempt_id) is None):
                raise WorkStateError('исправление не является доказуемым преемником перевода')
            if (attempt.context_json.get('policy_version') == CORRECTION_POLICY_VERSION_V2
                    and not self._candidate_successor(body, self.get_attempt(attempt_id), require_current=True)):
                raise WorkStateError('результат не подтверждён решением по кандидату')
        if attempt.root_id is not None:
            root = self.get_root(attempt.root_id)
            if root is not None and root.policy_version is not None:
                expected = {'schema_version': 1, 'root_id': root.id, 'policy_version': root.policy_version,
                    'prompt_hash': attempt.context_json.get('prompt_hash'), 'attempt_id': attempt_id}
                if (provenance != expected or not self.owns_recovery_attempt(attempt_id)
                        or artifact.body_json.get('provenance') != expected):
                    raise WorkStateError('происхождение новой версии не совпадает')
        if attempt.conditions_id != conditions_id or artifact.conditions_id != conditions_id or artifact.attempt_id != attempt_id:
            raise WorkStateError('происхождение новой версии не совпадает')
        if current is None:
            if expected_attempt_id is not None:
                raise WorkStateError('указатель результата изменился')
            self.bind('result', entity_id, book_id=book_id, conditions_id=conditions_id,
                      attempt_id=attempt_id, artifact_revision_id=artifact_revision_id, payload=provenance)
            return
        if current.artifact_revision_id is None or self._session.get(ArtifactRevisionRow, current.artifact_revision_id) is None:
            raise WorkStateError('предыдущая версия результата не заархивирована')
        previous_revision = current.artifact_revision_id
        changed = self._session.execute(update(WorkBindingRow).where(
            WorkBindingRow.entity_kind == 'result', WorkBindingRow.entity_id == entity_id,
            WorkBindingRow.book_id == book_id, WorkBindingRow.conditions_id == conditions_id,
            WorkBindingRow.attempt_id.is_(None) if expected_attempt_id is None else WorkBindingRow.attempt_id == expected_attempt_id,
            WorkBindingRow.artifact_revision_id == previous_revision,
        ).values(attempt_id=attempt_id, artifact_revision_id=artifact_revision_id,
                 payload_json=_json(provenance or {'schema_version': 1})))
        if changed.rowcount != 1:
            raise WorkStateError('указатель результата изменился')

    def find_root_by_key(self, root_key):
        row = self._session.scalar(select(RecoveryRootRow).where(RecoveryRootRow.root_key == root_key))
        return None if row is None else self.get_root(row.id)

    def list_roots(self, book_id, *, run_id=None):
        rows = self._session.scalars(select(RecoveryRootRow).where(RecoveryRootRow.book_id == book_id)).all()
        return tuple(self.get_root(row.id) for row in rows
                     if run_id is None or row.manifest_json.get('active_run_id') == run_id)

    def list_active_roots(self, book_id, *, run_id):
        return tuple(root for root in self.list_roots(book_id, run_id=run_id)
                     if root.superseded_by is None and root.status != 'complete')

    def pin_legacy_policy(self, root_id, *, expected_generation, proof):
        from linguary.application.translation_policy import LEGACY_TRANSLATION_POLICY
        proof = _json(proof)
        root = self.get_root(root_id)
        if (root is None or root.generation != expected_generation or root.superseded_by is not None
                or proof.get('root_id') != root_id or proof.get('conditions_id') != root.conditions_id
                or proof.get('policy_version') != LEGACY_TRANSLATION_POLICY
                or not proof.get('snapshot_sha256') or not proof.get('prompt_sha256')):
            raise WorkStateError('conditions_unavailable')
        if root.policy_version is not None:
            if root.policy_version != LEGACY_TRANSLATION_POLICY or thaw(root.compatibility) != proof:
                raise WorkStateError('conditions_unavailable')
            return root
        conditions = self.get_conditions(root.conditions_id)
        if (conditions is None or proof['snapshot_sha256'] != sha256(json.dumps(
                thaw(conditions.payload), ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                allow_nan=False).encode()).hexdigest()
                or proof['prompt_sha256'] != conditions.payload.get('prompt_sha256')
                or proof.get('run_id') != root.manifest.get('active_run_id')):
            raise WorkStateError('conditions_unavailable')
        changed = self._session.execute(update(RecoveryRootRow).where(
            RecoveryRootRow.id == root_id, RecoveryRootRow.generation == expected_generation,
            RecoveryRootRow.policy_version.is_(None), RecoveryRootRow.superseded_by.is_(None),
        ).values(policy_version=LEGACY_TRANSLATION_POLICY, compatibility_json=proof))
        if changed.rowcount != 1:
            raise WorkStateError('generation_conflict')
        self._session.flush()
        return self.get_root(root_id)

    def restart_root(self, root_id, *, expected_generation, successor, scopes, task):
        from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
        from linguary.adapters.db.repositories import SqlAlchemyTaskRepository
        root = self.get_root(root_id)
        scopes = tuple(scopes)
        if root is None:
            raise WorkStateError('root_not_found')
        if root.generation != expected_generation or root.superseded_by is not None:
            raise WorkStateError('generation_conflict')
        latest = self.latest_conditions(root.book_id)
        run_id = root.manifest.get('active_run_id')
        run = self._session.get(ProcessingRunRow, run_id)
        if (root.status == 'complete' or latest is None or latest.id != root.conditions_id
                or run is None or run.status != 'active' or run.book_id != root.book_id
                or successor.id == root.id or successor.book_id != root.book_id
                or successor.conditions_id != root.conditions_id
                or successor.manifest.get('active_run_id') != run_id
                or successor.policy_version != CURRENT_TRANSLATION_POLICY
                or task.book_id != root.book_id or task.status.value != 'pending'
                or not ((task.task_type.value == 'regenerate_block' and task.operation_id == successor.id
                         and task.policy_version == successor.policy_version)
                        or (task.task_type.value == 'process_translation_unit'
                            and task.operation_id.startswith('dispatch:') and task.policy_version == ''))
                or successor.operation_kind != root.operation_kind
                or successor.superseded_by is not None or successor.status != 'pending'
                or successor.generation != 0 or successor.calls_used != 0 or successor.extra_calls != 0):
            raise WorkStateError('conditions_unavailable')
        ordered = tuple(root.manifest['ordered_blocks'])
        historical = set(root.manifest.get('historical_coverage', ()))
        for scope in self.list_scopes(root_id):
            if scope.status == 'accepted':
                historical.update(scope.ordered_blocks)
        if not historical <= set(ordered):
            raise WorkStateError('unsafe_partition')
        for interval in root.manifest['atomic_intervals']:
            overlap = historical.intersection(interval)
            if overlap and overlap != set(interval):
                raise WorkStateError('unsafe_partition')
        pending = tuple(block for block in ordered if block not in historical)
        actual = tuple(block for scope in scopes for block in scope.ordered_blocks)
        if (not pending or actual != pending or len(set(actual)) != len(actual)
                or any(scope.root_id != successor.id or scope.status != 'pending' or scope.generation != 0 for scope in scopes)
                or tuple(successor.manifest['ordered_blocks']) != ordered
                or successor.manifest.get('atomic_intervals') != root.manifest.get('atomic_intervals')
                or set(successor.manifest.get('historical_coverage', ())) != historical):
            raise WorkStateError('unsafe_partition')
        for interval in root.manifest['atomic_intervals']:
            if not historical.intersection(interval) and not any(set(interval) <= set(scope.ordered_blocks) for scope in scopes):
                raise WorkStateError('unsafe_partition')
        # The caller owns BEGIN IMMEDIATE; the savepoint also protects callers that catch errors.
        with self._session.begin_nested():
            self.add_root(successor)
            changed = self._session.execute(update(RecoveryRootRow).where(
                RecoveryRootRow.id == root_id, RecoveryRootRow.generation == expected_generation,
                RecoveryRootRow.superseded_by.is_(None),
            ).values(superseded_by=successor.id, generation=RecoveryRootRow.generation + 1,
                     status='stopped', reason='superseded'))
            if changed.rowcount != 1:
                raise WorkStateError('generation_conflict')
            self._session.execute(update(RecoveryScopeRow).where(
                RecoveryScopeRow.root_id == root_id, RecoveryScopeRow.status.in_(('pending', 'claimed', 'stopped')),
            ).values(status='stopped', generation=RecoveryScopeRow.generation + 1))
            self._session.execute(update(LLMAttemptRow).where(
                LLMAttemptRow.root_id == root_id, LLMAttemptRow.state.in_(('created', 'reserved')),
            ).values(state='cancelled'))
            self._session.execute(update(ProcessingTaskRow).where(
                ProcessingTaskRow.book_id == root.book_id, ProcessingTaskRow.operation_id == root_id,
                ProcessingTaskRow.status.in_(('pending', 'running', 'failed')),
            ).values(status='superseded', error_type=None, error_message=None, error_detail=None))
            bindings = self._session.scalars(select(WorkBindingRow).where(
                WorkBindingRow.book_id == root.book_id, WorkBindingRow.entity_kind == 'task')).all()
            for binding in bindings:
                payload = dict(binding.payload_json)
                refs = payload.get('operation_refs', [])
                if not any(ref.get('root_id') == root_id for ref in refs):
                    continue
                payload['operation_refs'] = [ref for ref in refs if ref.get('root_id') != root_id]
                binding.payload_json = payload
                if not payload['operation_refs']:
                    self._session.execute(update(ProcessingTaskRow).where(
                        ProcessingTaskRow.id == binding.entity_id,
                        ProcessingTaskRow.status.in_(('pending', 'running', 'failed')),
                    ).values(status='superseded', error_type=None, error_message=None, error_detail=None))
            for scope in scopes:
                self.add_scope(scope)
            SqlAlchemyTaskRepository(self._session).add(task)
            self.bind('task', task.id, book_id=root.book_id, conditions_id=successor.conditions_id,
                payload={'schema_version': 1, 'operation_refs': [
                    {'root_id': successor.id, 'policy_version': successor.policy_version}]})
        return self.get_root(successor.id)

    def extend_recovery_limit(self, root_id, *, additional_calls, expected_generation):
        root = self.get_root(root_id)
        if root is None:
            raise WorkStateError('root_not_found')
        if root.generation != expected_generation or root.superseded_by is not None:
            raise WorkStateError('generation_conflict')
        if (type(additional_calls) is not int or additional_calls <= 0
                or root.manifest['policy']['max_calls'] + root.extra_calls + additional_calls > 9223372036854775807):
            raise WorkStateError('invalid_limit')
        unknown = self._session.scalar(select(LLMAttemptRow.id).where(
            LLMAttemptRow.root_id == root_id, LLMAttemptRow.state.in_(('sent', 'unknown'))).limit(1))
        if unknown is not None:
            raise WorkStateError('unknown')
        if root.status != 'stopped' or root.reason != 'limit':
            raise WorkStateError('state_conflict')
        latest = self.latest_conditions(root.book_id)
        run = self._session.get(ProcessingRunRow, root.manifest.get('active_run_id'))
        if (latest is None or latest.id != root.conditions_id or run is None
                or run.book_id != root.book_id or run.status != 'active'):
            raise WorkStateError('conditions_unavailable')
        changed = self._session.execute(update(RecoveryRootRow).where(
            RecoveryRootRow.id == root_id, RecoveryRootRow.generation == expected_generation,
            RecoveryRootRow.status == 'stopped', RecoveryRootRow.reason == 'limit',
        ).values(extra_calls=RecoveryRootRow.extra_calls + additional_calls,
            generation=RecoveryRootRow.generation + 1, status='pending', reason=None))
        if changed.rowcount != 1:
            raise WorkStateError('generation_conflict')
        self._session.flush()
        return self.get_root(root_id)

    def list_root_attempts(self, root_id):
        rows = self._session.scalars(select(LLMAttemptRow).where(LLMAttemptRow.root_id == root_id)).all()
        return tuple(self._attempt_from_row(row) for row in rows)

    def list_attempts_for_roots(self, root_ids):
        root_ids = tuple(root_ids)
        if not root_ids:
            return {}
        grouped = {root_id: [] for root_id in root_ids}
        rows = self._session.scalars(
            select(LLMAttemptRow).where(LLMAttemptRow.root_id.in_(root_ids))
        ).all()
        for row in rows:
            grouped.setdefault(row.root_id, []).append(self._attempt_from_row(row))
        return {root_id: tuple(items) for root_id, items in grouped.items()}

    def list_scopes(self, root_id):
        rows = self._session.scalars(select(RecoveryScopeRow).where(RecoveryScopeRow.root_id == root_id)).all()
        return tuple(self._scope_from_row(row) for row in rows)

    def list_scopes_for_roots(self, root_ids):
        root_ids = tuple(root_ids)
        if not root_ids:
            return {}
        grouped = {root_id: [] for root_id in root_ids}
        rows = self._session.scalars(
            select(RecoveryScopeRow).where(RecoveryScopeRow.root_id.in_(root_ids))
        ).all()
        for row in rows:
            grouped.setdefault(row.root_id, []).append(self._scope_from_row(row))
        return {root_id: tuple(items) for root_id, items in grouped.items()}

    def rebind_root(self, root_id, *, run_id, historical_coverage=()):
        row = self._session.get(RecoveryRootRow, root_id)
        if row is None or row.superseded_by is not None:
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

    def begin_recovery_attempt(self, root_id, scope_id, *, task_id, run_id, stage, assigned_model, prompt_hash, policy_version):
        from linguary.application.recovery import RecoveryError
        from linguary.domain.llm_log import CallContext
        root = self.get_root(root_id)
        scope = self.get_scope(scope_id)
        if root is None or scope is None or scope.root_id != root_id:
            raise RecoveryError('unsafe_partition', root_id)
        from linguary.application.translation_policy import require_policy
        require_policy(policy_version)
        if root.policy_version != policy_version:
            raise RecoveryError('conditions_unavailable', root_id)
        if root.superseded_by is not None or root.status not in {'pending', 'running'}:
            raise RecoveryError(root.reason or 'interrupted', root_id)
        if scope.status != 'pending':
            raise RecoveryError('interrupted', root_id)
        conditions = self.latest_conditions(root.book_id)
        run = self._session.get(ProcessingRunRow, run_id)
        if (conditions is None or conditions.id != root.conditions_id or run is None
                or run.book_id != root.book_id or run.status != 'active'
                or root.manifest.get('active_run_id') != run_id):
            raise RecoveryError('conditions_unavailable', root_id)
        if task_id is not None:
            task = self._session.get(ProcessingTaskRow, task_id)
            binding = self.get_binding('task', task_id)
            refs = [] if binding is None else binding['payload_json'].get('operation_refs', [])
            direct = task is not None and task.operation_id == root_id and task.policy_version == policy_version
            dispatch = task is not None and task.operation_id.startswith('dispatch:') and task.policy_version == '' and {
                'root_id': root_id, 'policy_version': policy_version} in refs
            if (task is None or task.book_id != root.book_id or task.status not in {'pending', 'running'}
                    or not (direct or dispatch)):
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
            run_id=run_id, task_id=task_id, root_id=root_id, policy_version=policy_version)
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
        is_truncated = (
            violation is not None and violation.rule_code == "provider_truncated"
        )
        if not is_truncated and not (
            signature
            and len(previous) == 2
            and all(
                row.state == "rejected"
                and row.outcome_json.get("failure_signature") == signature
                for row in previous
            )
        ):
            return False
        policy = root.manifest['policy']
        scopes = self.list_scopes(root.id)
        by_id = {item.id: item for item in scopes}
        depth, parent = 0, scope
        while parent.parent_id is not None:
            depth += 1
            parent = by_id[parent.parent_id]
        leaves = sum(item.status != "split" for item in scopes)
        if "atomic_intervals" in root.manifest:
            scope_set = set(scope.ordered_blocks)
            atoms = [tuple(g) for g in root.manifest["atomic_intervals"] if set(g) <= scope_set]
            can_partition = len(atoms) >= 2
        else:
            can_partition = len(scope.ordered_blocks) > 1
        return bool(
            can_partition
            and depth < policy["max_depth"]
            and leaves + 1 <= policy["max_leaves"]
        )

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
        from linguary.application.translation_policy import TRANSLATION_POLICIES
        run = self._session.get(ProcessingRunRow, attempt.run_id)
        conditions = self.latest_conditions(root.book_id)
        return (root.policy_version in TRANSLATION_POLICIES
                and attempt.context.get('policy_version') == root.policy_version
                and root.superseded_by is None and scope.root_id == root.id
                and run is not None and run.status == 'active' and run.book_id == root.book_id
                and conditions is not None and conditions.id == root.conditions_id
                and attempt.conditions_id == root.conditions_id and attempt.book_id == root.book_id
                and root.generation == attempt.context.get('root_generation') and root.status in {'pending', 'running'}
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
        if attempt.state == 'validated':
            self.transition_attempt(attempt_id, 'accepted', expected_state='validated',
                                    outcome={**(thaw(attempt.outcome) or {}), 'decision': 'accepted'})

    def root_coverage_complete(
        self,
        root_id,
        *,
        update_state=True,
        root=None,
        scopes=None,
        bindings=None,
        attempts_by_id=None,
        present_blocks=None,
        conditions=None,
        run=None,
    ):
        root = self.get_root(root_id) if root is None else root
        if root is None or root.status == 'stopped' or root.superseded_by is not None:
            return False
        manifest = root.manifest
        scopes = self.list_scopes(root_id) if scopes is None else scopes
        leaves = [scope for scope in scopes if scope.status != 'split']
        blocks = list(manifest.get('historical_coverage', ()))
        if any(scope.status != 'accepted' for scope in leaves):
            return False
        for scope in leaves:
            blocks.extend(scope.ordered_blocks)
        if len(blocks) != len(set(blocks)) or set(blocks) != set(manifest['ordered_blocks']):
            return False
        run_id = manifest['active_run_id']
        if run is None or getattr(run, 'id', None) != run_id:
            run = self._session.get(ProcessingRunRow, run_id)
        if conditions is None:
            conditions = self.latest_conditions(root.book_id)
        if (run is None or run.status != 'active' or run.book_id != root.book_id
                or conditions is None or conditions.id != root.conditions_id):
            return False
        if bindings is None:
            bindings = self.result_bindings(run_id)
        attempts_by_id = (
            {}
            if attempts_by_id is None
            else (attempts_by_id if isinstance(attempts_by_id, dict) else dict(attempts_by_id))
        )

        def cached_attempt(attempt_id):
            if attempt_id is None:
                return None
            if attempt_id not in attempts_by_id:
                attempts_by_id[attempt_id] = self.get_attempt(attempt_id)
            return attempts_by_id[attempt_id]

        correction_sources = {}
        for block in blocks:
            binding = bindings.get(block)
            attempt = None if binding is None else cached_attempt(binding['attempt_id'])
            if attempt is not None and attempt.context.get('stage') == 'chapter_correction':
                if binding['book_id'] != root.book_id or binding['conditions_id'] != root.conditions_id:
                    return False
                source = self._translation_source(book_id=root.book_id, run_id=run_id,
                    conditions_id=root.conditions_id, block_id=block,
                    artifact_revision_id=binding['artifact_revision_id'], attempt_id=binding['attempt_id'])
                if source is None:
                    return False
                correction_sources[block] = source
        for scope in leaves:
            if root.policy_version is not None:
                accepted = cached_attempt(scope.accepted_attempt_id)
                if (accepted is None or accepted.state != 'accepted' or accepted.root_id != root.id
                        or accepted.context.get('policy_version') != root.policy_version):
                    return False
            for block in scope.ordered_blocks:
                binding = bindings.get(block)
                if binding is None:
                    return False
                source = correction_sources.get(block)
                current_attempt_id = source.id if source is not None else binding['attempt_id']
                if current_attempt_id != scope.accepted_attempt_id:
                    replacement = source if source is not None else cached_attempt(current_attempt_id)
                    if (replacement is None or replacement.state != 'accepted' or replacement.run_id != run_id
                            or replacement.book_id != root.book_id
                            or replacement.conditions_id != root.conditions_id
                            or replacement.context.get('stage') != 'block_regeneration'):
                        return False
        if present_blocks is not None:
            if any(block not in present_blocks for block in blocks):
                return False
        elif hasattr(self._session, 'scalars'):
            if blocks:
                present_blocks = set(
                    self._session.scalars(
                        select(BlockResultRow.block_id).where(
                            BlockResultRow.run_id == run_id,
                            BlockResultRow.block_id.in_(blocks),
                        )
                    ).all()
                )
                if any(block not in present_blocks for block in blocks):
                    return False
        else:
            from linguary.adapters.db.repositories import SqlAlchemyBlockResultRepository
            results = SqlAlchemyBlockResultRepository(self._session)
            if any(results.get(run_id, block) is None for block in blocks):
                return False
        if update_state:
            self._session.execute(update(RecoveryRootRow).where(RecoveryRootRow.id == root.id,
                RecoveryRootRow.generation == root.generation).values(status='complete', reason=None))
        return True

    def invalidate_run_coverage(self, book_id, run_id):
        for root in self.list_roots(book_id, run_id=run_id):
            if root.superseded_by is not None:
                continue
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
