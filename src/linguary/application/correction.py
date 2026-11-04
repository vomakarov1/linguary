"""Current correction coverage and transactional recovery of required chapter work."""

from dataclasses import asdict, dataclass, fields, replace
from uuid import uuid4

from linguary.application.context_builder import textual_blocks
from linguary.application.correction_dto import CorrectionSnapshot
from linguary.domain import (
    BlockProcessingResult, BookStatus, CanonicalDocument, ChapterAnalysis, DocumentBlock,
    DocumentSection, DomainError, ProcessingRun, ProcessingTask, TaskStatus, TaskType,
)
from linguary.domain.book_conditions import BookConditions, WorkStateError, thaw
from linguary.domain.content_plan import canonical_sha
from linguary.domain.correction import correction_policy, correction_task_identity


def original_block_payload(block):
    payload = {field.name: getattr(block, field.name) for field in fields(block)}
    payload['source_reference'] = asdict(block.source_reference)
    payload['inline_annotations'] = [asdict(annotation) for annotation in block.inline_annotations]
    payload['attributes'] = dict(block.attributes)
    return payload


@dataclass(frozen=True, slots=True)
class CorrectionInputs:
    document: CanonicalDocument
    section: DocumentSection
    analysis: ChapterAnalysis | None
    run: ProcessingRun
    conditions: BookConditions
    snapshot: CorrectionSnapshot
    blocks: tuple[DocumentBlock, ...]
    results: dict[str, BlockProcessingResult]
    bindings: dict[str, dict | None]

    @property
    def context(self):
        return dict(schema_version=1, book_id=self.document.id,
            source_language=self.document.source_language,
            target_language=self.document.target_language,
            section_id=self.section.id,
            chapter_originals=[original_block_payload(block) for block in self.section.ordered_blocks],
            chapter_analysis=None if self.analysis is None else asdict(self.analysis),
            book_conditions=thaw(self.conditions.payload),
            run_glossary=asdict(self.run.glossary_snapshot),
            correction_snapshot=self.snapshot.to_payload())

    @property
    def checksums(self):
        context = self.context
        return {block.id: canonical_sha(dict(context=context, original=original_block_payload(block),
                    current_persisted_result=None if block.id not in self.results
                    else asdict(self.results[block.id]))) for block in self.blocks}

    @property
    def manifest(self):
        return dict(checksums=self.checksums, bindings=self.bindings)

    @property
    def translated(self):
        if self.blocks and self.analysis is None:
            return False
        if not {block.id for block in self.blocks} <= self.results.keys():
            return False
        if any(self.bindings[block.id] is None
               or self.bindings[block.id]['book_id'] != self.document.id
               or self.bindings[block.id]['conditions_id'] != self.conditions.id
               or not self.bindings[block.id]['attempt_id']
               or not self.bindings[block.id]['artifact_revision_id'] for block in self.blocks):
            return False
        try:
            if self.analysis is not None:
                self.analysis.ensure_ready_for_processing()
            return self.analysis is None or all(
                self.results[block.id].treatment == self.analysis.treatment_plan.treatment_for(block.id)
                for block in self.blocks)
        except DomainError:
            return False


def correction_inputs(tx, book_id, section_id):
    """Read all significant inputs through the caller's existing transaction."""
    run = tx.runs.active_for_book(book_id)
    if run is None:
        return None
    binding = tx.work.get_binding('run', run.id)
    policy = correction_policy({} if binding is None else binding['payload_json'])
    if policy is None:
        return None
    conditions = tx.work.latest_conditions(book_id)
    if (binding['book_id'] != book_id or conditions is None
            or binding['conditions_id'] != conditions.id):
        raise WorkStateError('условия исправления больше не активны')
    snapshot = CorrectionSnapshot.from_payload(policy)
    document = tx.documents.get(book_id)
    if document is None:
        raise WorkStateError('книга исправления отсутствует')
    section = next((s for s in document.ordered_sections if s.id == section_id), None)
    if section is None:
        raise WorkStateError('глава исправления отсутствует')
    blocks = textual_blocks(section)
    results = {r.block_id: r for r in tx.results.list_for_run(run.id)
               if r.block_id in {b.id for b in blocks}}
    bindings = {b.id: tx.work.get_binding('result', f'{run.id}:{b.id}') for b in blocks}
    return CorrectionInputs(document, section, tx.analyses.get_for_section(book_id, section_id),
                            run, conditions, snapshot, blocks, results, bindings)


def unchecked_blocks(tx, inputs):
    current = inputs.checksums
    checked = {}
    for check in tx.work.correction_checks(inputs.run.id, inputs.section.id):
        attempt = tx.work.get_attempt(check.attempt_id)
        if (attempt is not None and attempt.state == 'accepted'
                and attempt.run_id == inputs.run.id and attempt.book_id == inputs.document.id
                and attempt.conditions_id == inputs.conditions.id
                and attempt.context.get('stage') == 'chapter_correction'
                and check.block_id in attempt.context.get('ordered_block_ids', ())):
            checked[check.block_id] = check.input_sha256
    return tuple(b.id for b in inputs.blocks if checked.get(b.id) != current[b.id])


def task_matches_inputs(tx, task, payload, inputs):
    """Own accepted groups advance results without changing operation generation."""
    baseline = payload.get('manifest', {})
    checks = {c.block_id: c for c in tx.work.correction_checks(inputs.run.id, inputs.section.id)}
    current = inputs.manifest
    if set(baseline.get('checksums', {})) != set(current['checksums']):
        return False
    for block_id, checksum in current['checksums'].items():
        if (baseline['checksums'][block_id] == checksum
                and baseline.get('bindings', {}).get(block_id) == inputs.bindings[block_id]):
            continue
        check = checks.get(block_id)
        attempt = None if check is None else tx.work.get_attempt(check.attempt_id)
        if (check is None or check.input_sha256 != checksum or attempt is None
                or attempt.state != 'accepted' or attempt.task_id != task.id
                or attempt.run_id != inputs.run.id or attempt.conditions_id != inputs.conditions.id):
            return False
    return True


def correction_coverage_complete(tx, book_id):
    """Legacy runs retain their prior export semantics; required runs need current checks."""
    run = tx.runs.active_for_book(book_id)
    binding = None if run is None else tx.work.get_binding('run', run.id)
    if correction_policy({} if binding is None else binding['payload_json']) is None:
        return True
    document = tx.documents.get(book_id)
    if document is None:
        return False
    for section in document.ordered_sections:
        inputs = correction_inputs(tx, book_id, section.id)
        if not inputs.translated or unchecked_blocks(tx, inputs):
            return False
    return True


def _supersede(tx, task):
    if task.status is TaskStatus.SUPERSEDED:
        return
    tx.tasks.save(replace(task, status=TaskStatus.SUPERSEDED, last_error=None))
    for attempt in tx.work.list_task_attempts(task.id):
        target = {'created': 'cancelled', 'reserved': 'cancelled', 'sent': 'unknown',
                  'validated': 'rejected'}.get(attempt.state)
        if target:
            tx.work.transition_attempt(attempt.id, target, expected_state=attempt.state)


def reconcile_corrections(tx, book_id):
    """ALG-03: restore tasks from run policy, preserving explicit retries of FAILED work."""
    run = tx.runs.active_for_book(book_id)
    binding = None if run is None else tx.work.get_binding('run', run.id)
    if correction_policy({} if binding is None else binding['payload_json']) is None:
        return ()
    document = tx.documents.get(book_id)
    if document is None:
        return ()
    tasks = tx.tasks.list_for_book(book_id)
    pending = []
    complete = True
    for section in document.ordered_sections:
        inputs = correction_inputs(tx, book_id, section.id)
        missing = unchecked_blocks(tx, inputs)
        covered = inputs.translated and not missing
        complete = complete and covered
        matching = None
        for task in tasks:
            if task.task_type is not TaskType.CHAPTER_CORRECTION or task.scope_id != section.id:
                continue
            task_binding = tx.work.get_binding('task', task.id)
            payload = {} if task_binding is None else task_binding['payload_json']
            current = (task_binding is not None and task_binding['book_id'] == book_id
                and task_binding['conditions_id'] == inputs.conditions.id
                and payload.get('run_id') == run.id and payload.get('section_id') == section.id
                and bool(payload.get('input_generation')) and task_matches_inputs(tx, task, payload, inputs)
                and task.identity == correction_task_identity(book_id=book_id, run_id=run.id,
                    section_id=section.id, input_generation=payload['input_generation']))
            if current and task.status is not TaskStatus.SUPERSEDED:
                matching = task
            elif task.status is not TaskStatus.SUPERSEDED:
                _supersede(tx, task)
        if covered or not inputs.translated:
            continue
        if matching is not None and matching.status is not TaskStatus.SUCCEEDED:
            pending.append(matching)
            continue
        # A disappeared check after success is new work even if the text is unchanged.
        generation = canonical_sha(dict(manifest=inputs.manifest,
            predecessor=None if matching is None else matching.id))
        if matching is not None:
            _supersede(tx, matching)
        identity = correction_task_identity(book_id=book_id, run_id=run.id,
            section_id=section.id, input_generation=generation)
        task = tx.tasks.find_by_identity(identity)
        if task is None or task.status is TaskStatus.SUPERSEDED:
            if task is not None:
                # Returning to old input still represents a fresh operation.
                generation = canonical_sha(dict(generation=generation, nonce=str(uuid4())))
                identity = correction_task_identity(book_id=book_id, run_id=run.id,
                    section_id=section.id, input_generation=generation)
            task = ProcessingTask(id=str(uuid4()), book_id=book_id,
                task_type=identity.task_type, scope_id=identity.scope_id,
                operation_id=identity.operation_id, policy_version=identity.policy_version)
            tx.tasks.add(task)
            tx.work.bind('task', task.id, book_id=book_id, conditions_id=inputs.conditions.id,
                payload=dict(schema_version=1, run_id=run.id, section_id=section.id,
                    input_generation=generation, manifest=inputs.manifest))
        pending.append(task)
    if not complete:
        book = tx.books.get(book_id)
        if book.status in {BookStatus.READY_TO_EXPORT, BookStatus.COMPLETED} or book.export_validation_status is not None:
            tx.books.save(replace(book, status=BookStatus.PROCESSING, export_validation_status=None))
    return tuple(pending)


def invalidate_correction(tx, book_id, run_id, block_ids):
    """Revoke checks and readiness in the transaction that changes source results."""
    tx.work.delete_correction_checks(run_id, block_ids=tuple(block_ids))
    reconcile_corrections(tx, book_id)
