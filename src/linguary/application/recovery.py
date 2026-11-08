"""Finite durable recovery of disjoint, source-bound processing scopes."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from uuid import uuid4

from linguary.application.ports.llm import LLMConfigurationError
from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY, require_policy
from linguary.domain.book_conditions import thaw
from linguary.domain.recovery import RecoveryRoot, RecoveryScope


NOTES_POLICY = "grounded-notes-v1"


class RecoveryError(LLMConfigurationError):
    def __init__(self, reason, root_id=None, *, original_error=None):
        self.reason, self.root_id = reason, root_id
        self.original_error = original_error
        super().__init__('recovery:' + reason)


class RecoverySplit(RecoveryError):
    def __init__(self, root_id, scope_id):
        self.scope_id = scope_id
        super().__init__('split', root_id)


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    max_calls: int = 12
    max_scope_calls: int = 3
    max_depth: int = 3
    max_leaves: int = 8

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError('предел восстановления должен быть конечным положительным целым')


def _sha(value):
    return sha256(json.dumps(thaw(value), ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def dependency_intervals(ordered_blocks, plan=None, *, indivisible=(), max_difficulty_span: int = 3):
    ordered = tuple(ordered_blocks)
    if len(set(ordered)) != len(ordered):
        raise RecoveryError('unsafe_partition')
    positions = {block: index for index, block in enumerate(ordered)}
    spans = [(index, index) for index in range(len(ordered))]
    for edge in indivisible:
        if not set(edge) <= positions.keys():
            raise RecoveryError('unsafe_partition')
        if edge:
            indices = [positions[block] for block in edge]
            spans.append((min(indices), max(indices)))
    if plan is not None:
        for difficulty in plan.difficulties:
            edge = (difficulty.owner_block_id, *(a.block_id for a in difficulty.anchors))
            if not set(edge) <= positions.keys():
                raise RecoveryError('unsafe_partition')
            if edge:
                indices = [positions[block] for block in edge]
                if max(indices) - min(indices) <= max_difficulty_span:
                    spans.append((min(indices), max(indices)))
    merged = []
    for left, right in sorted(spans):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    return tuple(ordered[left:right + 1] for left, right in merged)


def merge_unit_boundaries(boundaries, intervals):
    groups = tuple(tuple(group) for group in boundaries)
    owner = {block: index for index, group in enumerate(groups) for block in group}
    merged = []
    spans = [(index, index) for index in range(len(groups))]
    for interval in intervals:
        if not set(interval) <= owner.keys():
            raise RecoveryError('unsafe_partition')
        spans.append((min(owner[b] for b in interval), max(owner[b] for b in interval)))
    for left, right in sorted(spans):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    return tuple(tuple(block for group in groups[left:right + 1] for block in group) for left, right in merged)


def local_failure_signature(violation, ordered_blocks):
    if (violation is None or violation.scope != 'local' or not violation.block_ids
            or not set(violation.block_ids) <= set(ordered_blocks)):
        return None
    return _sha({'rule': violation.rule_code, 'stage': violation.stage,
                 'blocks': sorted(violation.block_ids), 'artifacts': sorted(violation.artifact_kinds)})


def root_manifest(*, book_id, conditions_id, glossary_sha, analysis_sha, plan_sha,
                  ordered_blocks, treatments, boundaries, atomic_intervals, max_source_chars,
                  policy=None, block_chars=None, note_owners=(), chapter_chars=None, notes_per_chapter=None):
    value = dict(schema_version=1, book_id=book_id, conditions_id=conditions_id,
        glossary_sha=glossary_sha, analysis_sha=analysis_sha, plan_sha=plan_sha,
        ordered_blocks=list(ordered_blocks), treatments=list(treatments),
        original_boundaries=[list(group) for group in boundaries],
        atomic_intervals=[list(group) for group in atomic_intervals], max_source_chars=max_source_chars,
        policy=asdict(policy or RecoveryPolicy()), block_chars=block_chars or {},
        note_owners=list(note_owners), chapter_chars=chapter_chars, notes_per_chapter=notes_per_chapter)
    return {**value, 'root_key': _sha(value)}


class Recovery:
    def __init__(self, unit_of_work, *, policy=None):
        self.uow = unit_of_work
        self.policy = policy or RecoveryPolicy()

    def prepare_roots(self, work, units, *, max_source_chars, validator, operation_refs=None, operation_kind="translation", transaction=None):
        transaction_context = lambda: nullcontext(transaction) if transaction is not None else self.uow.transaction()
        with transaction_context() as tx:
            conditions = tx.work.latest_conditions(work.document.id)
        if conditions is None:
            raise RecoveryError('conditions_unavailable')
        ordered = tuple(block for unit in units for block in unit.block_ids)
        blocks = {item.block.id: item.block for unit in units for item in unit.items}
        task_id = getattr(work, 'task_id', None)
        with transaction_context() as tx:
            if operation_refs is None and task_id is not None:
                binding = tx.work.get_binding('task', task_id)
                if binding is not None:
                    if binding['conditions_id'] != conditions.id:
                        raise RecoveryError('conditions_unavailable')
                    operation_refs = binding['payload_json'].get('operation_refs')
            if operation_refs is not None:
                return self._resume_roots(tx, work, ordered, conditions, operation_refs, operation_kind)
        if operation_kind == 'regeneration' and len(ordered) != 1:
            raise RecoveryError('unsafe_partition')
        indivisible = []
        structural = {}
        for block in blocks.values():
            if block.parent_id is not None:
                structural.setdefault(('parent', block.parent_id), []).append(block.id)
        indivisible.extend(tuple(group) for group in structural.values())
        for left, right in zip(ordered, ordered[1:]):
            first, second = blocks[left], blocks[right]
            if ((first.parent_id is not None and first.parent_id == second.parent_id)
                    or first.type == second.type and first.type.value in {'list_item', 'poem', 'quote', 'table'}):
                indivisible.append((left, right))
        intervals = dependency_intervals(ordered, None if operation_kind == 'regeneration' else work.analysis.content_plan, indivisible=indivisible)
        boundaries = tuple(unit.block_ids for unit in units)
        merged = merge_unit_boundaries(boundaries, intervals)
        note_owners = set()
        plan = work.analysis.content_plan
        if plan is not None:
            covered = {(item.difficulty_id, item.point_id) for item in plan.before_you_read_evidence}
            for difficulty in plan.difficulties:
                if (work.analysis.treatment_plan.treatment_for(difficulty.owner_block_id).value == 'READ_AS_IS'
                        and any((difficulty.id, point.id) not in covered for point in difficulty.required_points)):
                    note_owners.add(difficulty.owner_block_id)
        result = []
        with transaction_context() as tx:
            latest = tx.work.latest_conditions(work.document.id)
            if latest is None or latest.id != conditions.id:
                raise RecoveryError('conditions_unavailable')
            for group in merged:
                atomic = tuple(interval for interval in intervals if set(interval) <= set(group))
                manifest = root_manifest(book_id=work.document.id, conditions_id=conditions.id,
                    glossary_sha=_sha(asdict(work.run.glossary_snapshot)), analysis_sha=_sha(asdict(work.analysis)),
                    plan_sha=None if work.analysis.content_plan is None else work.analysis.content_plan.sha256,
                    ordered_blocks=group, treatments=tuple(work.analysis.treatment_plan.treatment_for(block).value for block in group),
                    boundaries=boundaries, atomic_intervals=atomic, max_source_chars=max_source_chars, policy=self.policy,
                    block_chars={block: len(blocks[block].text) for block in group},
                    note_owners=tuple(block for block in group if block in note_owners),
                    chapter_chars=sum(len(block.text) for block in blocks.values()),
                    notes_per_chapter=conditions.payload['checks']['language_notes_per_chapter'])
                historical = () if operation_kind == 'regeneration' else tuple(block for block in group if block in work.processed_block_ids)
                for interval in atomic:
                    overlap = set(interval).intersection(historical)
                    if overlap and overlap != set(interval):
                        raise RecoveryError('unsafe_partition')
                if historical:
                    self._validate_historical(tx, work, historical, blocks, validator, conditions)
                payload = {**manifest, 'notes_policy': NOTES_POLICY, 'active_run_id': work.run.id,
                    'historical_coverage': list(historical)}
                operation_id = str(uuid4())
                policy_version = require_policy(CURRENT_TRANSLATION_POLICY).version
                root = RecoveryRoot(operation_id, work.document.id, conditions.id,
                    f"{operation_id}:{policy_version}:{manifest['root_key']}", payload,
                    policy_version=policy_version, operation_kind=operation_kind)
                tx.work.add_root(root)
                scopes = tx.work.list_scopes(root.id)
                covered = set(root.manifest.get('historical_coverage', ()))
                for scope in scopes:
                    if scope.status != 'split':
                        covered.update(scope.ordered_blocks)
                missing = [block for block in group if block not in covered]
                if missing:
                    # Initial unresolved intervals never straddle an accepted historical block.
                    pending = []
                    current = []
                    for interval in atomic:
                        unresolved = [block for block in interval if block in missing]
                        if unresolved and len(unresolved) != len(interval):
                            raise RecoveryError('unsafe_partition', root.id)
                        if unresolved:
                            current.extend(unresolved)
                        elif current:
                            pending.append(tuple(current)); current = []
                    if current:
                        pending.append(tuple(current))
                    if sum(scope.status != 'split' for scope in scopes) + len(pending) > root.manifest['policy']['max_leaves']:
                        tx.work.stop_root(root.id, 'unsafe_partition')
                    else:
                        for scope_blocks in pending:
                            tx.work.add_scope(RecoveryScope(str(uuid4()), root.id, scope_blocks))
                result.append(root.id)
            if task_id is not None:
                tx.work.bind_operation_refs(task_id, book_id=work.document.id, conditions_id=conditions.id,
                    operation_refs=tuple({'root_id': root_id, 'policy_version': tx.work.get_root(root_id).policy_version}
                                         for root_id in result))
        return tuple(result)

    def _resume_roots(self, transaction, work, ordered, conditions, operation_refs, operation_kind):
        ids, covered = [], set()
        for ref in operation_refs:
            if not isinstance(ref, dict) or set(ref) != {'root_id', 'policy_version'}:
                raise RecoveryError('conditions_unavailable')
            require_policy(ref['policy_version'])
            root = transaction.work.get_root(ref['root_id'])
            if (root is None or root.book_id != work.document.id or root.conditions_id != conditions.id
                    or root.policy_version != ref['policy_version'] or root.superseded_by is not None
                    or root.operation_kind != operation_kind
                    or root.manifest.get('active_run_id') != work.run.id
                    or root.manifest.get('notes_policy') != NOTES_POLICY):
                raise RecoveryError('conditions_unavailable', ref['root_id'])
            targets = tuple(root.manifest.get('ordered_blocks', ()))
            if (not targets or not set(targets) <= set(ordered) or covered.intersection(targets)
                    or root.manifest.get('glossary_sha') != _sha(asdict(work.run.glossary_snapshot))
                    or root.manifest.get('analysis_sha') != _sha(asdict(work.analysis))):
                raise RecoveryError('conditions_unavailable', root.id)
            covered.update(targets)
            ids.append(root.id)
        return tuple(ids)

    def _validate_historical(self, transaction, work, block_ids, blocks, validator, conditions):
        from linguary.domain import TranslationUnit
        from linguary.application.llm_dto import TranslationResultResponse, BlockResultResponse, GlossaryHintResponse, LanguageNoteResponse
        from linguary.application.validators.content_plan import validate_content_coverage, validate_saved_content_coverage
        from linguary.domain.content_plan import ContentEvidence
        unit = TranslationUnit.of_blocks(id='historical-coverage', blocks=tuple(blocks[b] for b in block_ids), plan=work.analysis.treatment_plan)
        stored = tuple(transaction.results.get(work.run.id, block) for block in block_ids)
        if any(item is None for item in stored):
            raise RecoveryError('unsafe_partition')
        unit.validate_results(stored)
        payload = TranslationResultResponse(blocks=tuple(BlockResultResponse(result.block_id, result.treatment, result.translated_text,
            tuple(GlossaryHintResponse(h.original, h.translation, h.note) for h in result.glossary_hints)) for result in stored),
            language_notes=tuple(LanguageNoteResponse(note.block_id, note.original_excerpt, note.note, note.translated_excerpt)
                                 for result in stored for note in result.language_notes))
        validator.validate(payload, unit=unit, target_language=work.target_language,
                           note_id_factory=lambda: str(uuid4()))
        if work.analysis.content_plan is not None:
            plan = work.analysis.content_plan
            validators_version = conditions.payload['versions']['validators']
            if validators_version == 4:
                if conditions.payload.get('reader_level') != 'a2':
                    raise RecoveryError('conditions_unavailable')
                validate_saved_content_coverage(plan, stored, work.analysis.before_you_read,
                    owner_block_ids=block_ids, source_texts={block.id: block.text for block in work.document.blocks()},
                    validators_version=validators_version, english_help=conditions.payload.get('english_help', False))
            else:
                evidence = tuple(item for result in stored for item in getattr(result.content_evidence, 'evidence', ()))
                validate_content_coverage(plan, stored, work.analysis.before_you_read,
                    ContentEvidence(plan.sha256, evidence, version=plan.version), owner_block_ids=block_ids,
                    validators_version=validators_version)

    def plan_unresolved(self, root_id):
        with self.uow.transaction() as transaction:
            root = transaction.work.get_root(root_id)
            if root is None or root.superseded_by is not None:
                raise RecoveryError('conditions_unavailable', root_id)
            require_policy(root.policy_version)
            if root.status == 'stopped':
                raise RecoveryError(root.reason or 'interrupted', root_id)
            scopes = transaction.work.list_scopes(root_id)
            if any(scope.status in {'claimed', 'stopped'} for scope in scopes):
                raise RecoveryError('interrupted', root_id)
            order = {block: index for index, block in enumerate(root.manifest['ordered_blocks'])}
            return tuple(sorted((scope for scope in scopes if scope.status == 'pending'), key=lambda s: order[s.ordered_blocks[0]]))

    def split(self, root_id, scope_id):
        with self.uow.transaction() as transaction:
            root = transaction.work.get_root(root_id)
            scope = transaction.work.get_scope(scope_id)
            if root is None or scope is None or scope.root_id != root_id or root.superseded_by is not None:
                raise RecoveryError('unsafe_partition', root_id)
            atoms = [tuple(group) for group in root.manifest['atomic_intervals'] if set(group) <= set(scope.ordered_blocks)]
            scopes = transaction.work.list_scopes(root_id)
            depth, parent = 0, scope
            by_id = {item.id: item for item in scopes}
            while parent.parent_id is not None:
                depth += 1; parent = by_id[parent.parent_id]
            policy = root.manifest['policy']
            leaves = sum(item.status != 'split' for item in scopes)
            if len(atoms) < 2 or depth >= policy['max_depth'] or leaves + 1 > policy['max_leaves']:
                transaction.work.stop_root(root_id, 'unsafe_partition')
                reason = 'unsafe_partition'
            else:
                middle = len(atoms) // 2
                children = tuple(tuple(block for atom in part for block in atom)
                    for part in (atoms[:middle], atoms[middle:]))
                transaction.work.split_scope(scope_id, expected_generation=scope.generation, children=children)
                reason = None
        if reason:
            raise RecoveryError(reason, root_id)

    def assert_complete(self, root_id):
        with self.uow.transaction() as transaction:
            if not transaction.work.root_coverage_complete(root_id):
                raise RecoveryError('unsafe_partition', root_id)

