"""Classify historical operations only from verifiable persisted execution evidence."""
from dataclasses import asdict
from uuid import uuid4

from linguary.application.book_conditions import (
    A2_CONTENT_PLAN_VERSIONS, A2_CONTENT_PLAN_V5_VERSIONS, payload_digest,
)
from linguary.application.recovery import NOTES_POLICY, RecoveryError, RecoveryPolicy, root_manifest
from linguary.application.translation_policy import LEGACY_TRANSLATION_POLICY, require_policy
from linguary.domain import TaskStatus, TaskType
from linguary.domain.book_conditions import thaw
from linguary.domain.errors import UnknownBlockError
from linguary.domain.recovery import RecoveryRoot, RecoveryScope


def _unavailable(root_id=None):
    raise RecoveryError('conditions_unavailable', root_id)


def _verify_conditions(transaction, task, conditions):
    stored = transaction.work.get_conditions(conditions.id)
    binding = transaction.work.get_binding('task', task.id)
    latest = transaction.work.latest_conditions(task.book_id)
    if (stored != conditions or conditions.book_id != task.book_id or latest is None
            or latest.id != conditions.id or conditions.reproducibility == 'unknown'
            or transaction.tasks.get(task.id) != task or binding is None
            or binding['book_id'] != task.book_id or binding['conditions_id'] != conditions.id
            or task.status in {TaskStatus.SUCCEEDED, TaskStatus.SUPERSEDED}):
        _unavailable()
    payload = conditions.payload
    schemas = {'legacy-v1': 1, 'content-plan-v2': 2, 'content-plan-v3': 3}
    profile = payload.get('profile')
    implementation = 3 if profile == 'content-plan-v3' else 1
    expected = dict(runtime=1, context_builder=implementation, validators=implementation,
        schemas=schemas.get(profile), prompts=implementation)
    if payload.get('english_help', False):
        if profile != 'content-plan-v3':
            _unavailable()
        expected.update(schemas=4, prompts=4)
    templates = payload.get('prompt_templates')
    template_values = thaw(templates) if templates is not None else None
    marker = (template_values.get('analysis_contract_version')
              if isinstance(template_values, dict) else None)
    if marker is not None and (type(marker) is not int or marker != 5):
        _unavailable()
    if marker == 5:
        supported_versions = []
        if profile == 'content-plan-v3':
            quoted = dict(runtime=1, context_builder=3, validators=3, schemas=5, prompts=5)
            if payload.get('reader_level') == 'a2':
                quoted['validators'] = 4
            supported_versions.append(quoted)
        elif (profile == 'legacy-v1' and payload.get('reader_level') == 'a2'
                and not payload.get('english_help', False)):
            supported_versions.append(dict(A2_CONTENT_PLAN_V5_VERSIONS))
    else:
        supported_versions = [expected]
        if payload.get('reader_level') == 'a2':
            supported_versions.append(dict(A2_CONTENT_PLAN_VERSIONS)
                if profile == 'legacy-v1' and not payload.get('english_help', False)
                else {**expected, 'validators': 4})
    if (profile not in schemas or thaw(payload.get('versions')) not in supported_versions
            or not templates or payload_digest(templates) != payload.get('prompt_sha256')):
        _unavailable()
    return binding


def ensure_legacy_operation(transaction, *, task, conditions):
    """Return exact root references, pinning only previously unclassified operations.

    The caller's transaction includes all evidence checks and writes. Missing chapter
    roots or ambiguous historical calls fail closed instead of creating a current root.
    """
    binding = _verify_conditions(transaction, task, conditions)
    run = transaction.runs.active_for_book(task.book_id)
    document = transaction.documents.get(task.book_id)
    if run is None or document is None:
        _unavailable()
    refs = binding['payload_json'].get('operation_refs')
    new_root = None
    adopted_attempts = ()
    if task.task_type == TaskType.REGENERATE_BLOCK:
        if task.policy_version != LEGACY_TRANSLATION_POLICY or not task.operation_id:
            _unavailable()
        try:
            block = document.block(task.scope_id)
        except (KeyError, ValueError, UnknownBlockError):
            _unavailable()
        roots = (transaction.work.get_root(task.operation_id),)
        if roots[0] is None:
            attempts = transaction.work.list_task_attempts(task.id)
            completed_starts = task.attempts - (task.status == TaskStatus.RUNNING)
            if completed_starts and not attempts:
                _unavailable()
            for attempt in attempts:
                context = attempt.context
                if (attempt.root_id is not None or attempt.scope_id is not None
                        or attempt.book_id != task.book_id or attempt.conditions_id != conditions.id
                        or attempt.run_id != run.id or attempt.task_id != task.id
                        or attempt.state not in {'failed', 'rejected', 'cancelled'}
                        or context.get('conditions_id') != conditions.id
                        or context.get('stage') != 'block_regeneration'
                        or context.get('run_id') != run.id or context.get('task_id') != task.id
                        or tuple(context.get('ordered_block_ids', ())) != (block.id,)
                        or context.get('root_id') is not None or context.get('policy_version') not in (None, '')
                        or not isinstance(context.get('prompt_hash'), str) or len(context['prompt_hash']) != 64):
                    _unavailable()
            adopted_attempts = tuple(attempts)
            analysis = transaction.analyses.get_for_section(task.book_id, block.section_id)
            if analysis is None:
                _unavailable()
            treatment = analysis.treatment_plan.treatment_for(block.id).value
            saved_policy = conditions.payload.get('recovery_policy')
            policy = RecoveryPolicy(**thaw(saved_policy)) if saved_policy is not None else RecoveryPolicy()
            manifest = root_manifest(book_id=task.book_id, conditions_id=conditions.id,
                glossary_sha=payload_digest(asdict(run.glossary_snapshot)), analysis_sha=payload_digest(asdict(analysis)),
                plan_sha=None if analysis.content_plan is None else analysis.content_plan.sha256,
                ordered_blocks=(block.id,), treatments=(treatment,), boundaries=((block.id,),),
                atomic_intervals=((block.id,),), max_source_chars=len(block.text), policy=policy,
                block_chars={block.id: len(block.text)}, chapter_chars=sum(len(b.text) for b in document.section(block.section_id).ordered_blocks),
                notes_per_chapter=conditions.payload['checks']['language_notes_per_chapter'])
            new_root = RecoveryRoot(task.operation_id, task.book_id, conditions.id, manifest['root_key'],
                {**manifest, 'active_run_id': run.id, 'notes_policy': NOTES_POLICY, 'historical_coverage': []},
                operation_kind='regeneration', calls_used=len(adopted_attempts))
            roots = (new_root,)
        targets = {block.id}
    elif task.task_type == TaskType.PROCESS_TRANSLATION_UNIT:
        try:
            targets = {block.id for block in document.section(task.scope_id).ordered_blocks}
        except (KeyError, ValueError, UnknownBlockError):
            _unavailable()
        if refs is not None:
            roots = tuple(transaction.work.get_root(ref['root_id']) for ref in refs)
        else:
            roots = tuple(root for root in transaction.work.list_roots(task.book_id, run_id=run.id)
                if root.superseded_by is None and root.status != 'complete' and root.operation_kind == 'translation'
                and set(root.manifest.get('ordered_blocks', ())) <= targets)
        if not roots:
            _unavailable()
    else:
        _unavailable()
    proofs = []
    occupied = set()
    for root in roots:
        if (root is None or root.book_id != task.book_id or root.conditions_id != conditions.id
                or root.superseded_by is not None or root.manifest.get('active_run_id') != run.id
                or root.manifest.get('notes_policy') != NOTES_POLICY):
            _unavailable()
        ordered = tuple(root.manifest.get('ordered_blocks', ()))
        if not ordered or not set(ordered) <= targets or occupied.intersection(ordered):
            _unavailable(root.id)
        occupied.update(ordered)
        if root.policy_version is not None:
            require_policy(root.policy_version)
            if root.policy_version != LEGACY_TRANSLATION_POLICY or root.compatibility is None:
                _unavailable(root.id)
        manifest = thaw(root.manifest)
        original = {key: value for key, value in manifest.items()
            if key not in {'root_key', 'notes_policy', 'active_run_id', 'historical_coverage'}}
        if manifest.get('root_key') != root.root_key or payload_digest(original) != root.root_key:
            _unavailable(root.id)
        if (manifest.get('book_id') != task.book_id or manifest.get('conditions_id') != conditions.id
                or manifest.get('glossary_sha') != payload_digest(asdict(run.glossary_snapshot))):
            _unavailable(root.id)
        if root.operation_kind == 'translation':
            section_ids = {document.block(block_id).section_id for block_id in ordered}
            if len(section_ids) != 1:
                _unavailable(root.id)
            analysis = transaction.analyses.get_for_section(task.book_id, next(iter(section_ids)))
            if (analysis is None or manifest.get('analysis_sha') != payload_digest(asdict(analysis))
                    or manifest.get('plan_sha') != (None if analysis.content_plan is None else analysis.content_plan.sha256)):
                _unavailable(root.id)
        scopes = transaction.work.list_scopes(root.id) if root is not new_root else ()
        by_scope = {scope.id: scope for scope in scopes}
        for attempt in transaction.work.list_root_attempts(root.id):
            scope = by_scope.get(attempt.scope_id)
            if (attempt.book_id != task.book_id or attempt.conditions_id != conditions.id
                    or attempt.run_id != run.id or scope is None
                    or tuple(attempt.context.get('ordered_block_ids', ())) != scope.ordered_blocks
                    or not set(scope.ordered_blocks) <= set(ordered)
                    or (attempt.context.get('root_id') != root.id
                        and attempt.id not in (root.compatibility or {}).get('adopted_attempt_ids', ()))
                    or attempt.context.get('policy_version') not in (None, '', LEGACY_TRANSLATION_POLICY)):
                _unavailable(root.id)
        proof = {'schema_version': 1, 'root_id': root.id, 'conditions_id': conditions.id,
            'policy_version': LEGACY_TRANSLATION_POLICY, 'snapshot_sha256': payload_digest(conditions.payload),
            'prompt_sha256': conditions.payload['prompt_sha256'], 'run_id': run.id,
            'ordered_blocks': list(ordered), 'manifest_sha256': payload_digest(root.manifest),
            'adopted_attempt_ids': list((root.compatibility or {}).get('adopted_attempt_ids',
                tuple(attempt.id for attempt in adopted_attempts)))}
        proofs.append((root, proof))
    if refs is not None and refs != [{'root_id': root.id, 'policy_version': LEGACY_TRANSLATION_POLICY} for root in roots]:
        _unavailable()
    if new_root is not None:
        transaction.work.add_root(new_root)
        scope = RecoveryScope(str(uuid4()), new_root.id, tuple(new_root.manifest['ordered_blocks']))
        transaction.work.add_scope(scope)
        transaction.work.attach_legacy_attempts(new_root.id, scope.id,
            task_id=task.id, attempt_ids=tuple(attempt.id for attempt in adopted_attempts))
    for root, proof in proofs:
        transaction.work.pin_legacy_policy(root.id, expected_generation=root.generation, proof=proof)
    result = tuple({'root_id': root.id, 'policy_version': LEGACY_TRANSLATION_POLICY} for root in roots)
    transaction.work.bind_operation_refs(task.id, book_id=task.book_id, conditions_id=conditions.id, operation_refs=result)
    return result
