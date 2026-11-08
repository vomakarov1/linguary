"""Current correction coverage and transactional recovery of required chapter work."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, fields, replace
from hashlib import sha256
import json
from uuid import uuid4

_SCOPED_RECONCILE_SECTION_ID: ContextVar[str | None] = ContextVar('_SCOPED_RECONCILE_SECTION_ID', default=None)


@contextmanager
def _scoped_reconcile_section(section_id: str | None):
    token = _SCOPED_RECONCILE_SECTION_ID.set(section_id)
    try:
        yield
    finally:
        _SCOPED_RECONCILE_SECTION_ID.reset(token)

from linguary.application.context_builder import (
    _extract_word_stems,
    _matches_chapter_text,
    textual_blocks,
)
from linguary.application.correction_dto import CorrectionSnapshot, CorrectionSnapshotV2, parse_correction_snapshot
from linguary.domain import (
    BlockProcessingResult, BlockTreatment, BookStatus, CanonicalDocument, ChapterAnalysis, DocumentBlock,
    DocumentSection, DomainError, ProcessingRun, ProcessingTask, TaskStatus, TaskType,
)
from linguary.domain.book_conditions import BookConditions, WorkStateError, thaw
from linguary.domain.content_plan import canonical_sha
from linguary.domain.correction import CorrectionCheck, correction_policy, correction_task_identity, CORRECTION_POLICY_VERSION_V2


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
    snapshot: CorrectionSnapshot | CorrectionSnapshotV2
    blocks: tuple[DocumentBlock, ...]
    results: dict[str, BlockProcessingResult]
    bindings: dict[str, dict | None]
    input_generations: dict[str, int] = field(default_factory=dict)
    _context: dict = field(init=False, repr=False, compare=False)
    _context_sha256: str = field(init=False, repr=False, compare=False)
    _checksums: dict[str, str] = field(init=False, repr=False, compare=False)
    _block_aliases: dict[str, str] = field(init=False, repr=False, compare=False)
    _alias_to_block_id: dict[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        block_aliases = {
            block.id: f"b{index}"
            for index, block in enumerate(self.section.ordered_blocks, start=1)
        }
        alias_to_block_id = {alias: block_id for block_id, alias in block_aliases.items()}
        chapter_blocks = [
            {"block_id": block_aliases[block.id], "text": block.text}
            for block in self.section.ordered_blocks
        ]
        chapter_text = " ".join(block.text for block in self.section.ordered_blocks)
        word_stems = _extract_word_stems(chapter_text)
        filtered_entries = tuple(
            entry for entry in self.run.glossary_snapshot.entries
            if _matches_chapter_text(entry.source_text, chapter_text, word_stems)
        )
        filtered_glossary = replace(self.run.glossary_snapshot, entries=filtered_entries)

        context = dict(
            schema_version=1,
            book_id=self.document.id,
            source_language=self.document.source_language,
            target_language=self.document.target_language,
            section_id=self.section.id,
            chapter_blocks=chapter_blocks,
            chapter_analysis=None if self.analysis is None else asdict(self.analysis),
            run_glossary=asdict(filtered_glossary),
        )

        def encoded(value):
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()

        context_bytes = encoded(context)
        # Preserve the existing canonical JSON checksum while hashing the shared context once.
        prefix = sha256(b'{"context":' + context_bytes + b',"current_persisted_result":')
        checksums = {}
        for block in self.blocks:
            digest = prefix.copy()
            result = None if block.id not in self.results else asdict(self.results[block.id])
            digest.update(encoded(result))
            digest.update(b',"original":')
            digest.update(encoded(original_block_payload(block)))
            digest.update(b'}')
            checksums[block.id] = digest.hexdigest()
        object.__setattr__(self, '_context', context)
        object.__setattr__(self, '_context_sha256', sha256(context_bytes).hexdigest())
        object.__setattr__(self, '_checksums', checksums)
        object.__setattr__(self, '_block_aliases', block_aliases)
        object.__setattr__(self, '_alias_to_block_id', alias_to_block_id)

    @property
    def block_aliases(self) -> dict[str, str]:
        return self._block_aliases

    @property
    def alias_to_block_id(self) -> dict[str, str]:
        return self._alias_to_block_id

    @property
    def context(self):
        return self._context

    @property
    def context_sha256(self):
        return self._context_sha256

    @property
    def checksums(self):
        return self._checksums

    @property
    def manifest(self):
        result = dict(checksums=self.checksums, bindings=self.bindings)
        if isinstance(self.snapshot, CorrectionSnapshotV2):
            result['input_generations'] = self.input_generations
        return result

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


def _section_correction_inputs(
    tx,
    *,
    document,
    section,
    run,
    conditions,
    snapshot,
    analyses_by_section=None,
    result_bindings=None,
    generations=None,
    run_results=None,
):
    blocks = textual_blocks(section)
    block_ids = {b.id for b in blocks}
    list_for_section = getattr(tx.results, 'list_for_section', None)
    stored_results = (
        list_for_section(run.id, section.id)
        if callable(list_for_section)
        else (run_results if run_results is not None else tx.results.list_for_run(run.id))
    )
    results = {r.block_id: r for r in stored_results if r.block_id in block_ids}
    if result_bindings is not None:
        bindings = {b.id: result_bindings.get(b.id) for b in blocks}
    else:
        bindings = {b.id: tx.work.get_binding('result', f'{run.id}:{b.id}') for b in blocks}
    if generations is None:
        generations = (
            tx.work.correction_input_generations(run.id)
            if isinstance(snapshot, CorrectionSnapshotV2)
            else {}
        )
    analysis = (
        analyses_by_section.get(section.id)
        if analyses_by_section is not None
        else tx.analyses.get_for_section(document.id, section.id)
    )
    return CorrectionInputs(
        document,
        section,
        analysis,
        run,
        conditions,
        snapshot,
        blocks,
        results,
        bindings,
        {b.id: generations.get(b.id, 0) for b in blocks},
    )


def _load_book_correction_inputs_state(
    tx, book_id, *, run, binding, policy, section_id: str | None = None
):
    conditions = tx.work.latest_conditions(book_id)
    if (binding['book_id'] != book_id or conditions is None
            or binding['conditions_id'] != conditions.id):
        raise WorkStateError('условия исправления больше не активны')
    snapshot = parse_correction_snapshot(policy)
    analyses_by_section = None
    if section_id is not None:
        get_for_section = getattr(tx.analyses, 'get_for_section', None)
        if callable(get_for_section):
            try:
                single_analysis = get_for_section(book_id, section_id)
                analyses_by_section = {section_id: single_analysis} if single_analysis is not None else {}
            except Exception:
                analyses_by_section = None
    if analyses_by_section is None:
        list_analyses = getattr(tx.analyses, 'list_for_book', None)
        analyses_by_section = (
            {analysis.section_id: analysis for analysis in list_analyses(book_id)}
            if callable(list_analyses)
            else None
        )
    list_result_bindings = getattr(tx.work, 'result_bindings', None)
    result_bindings = (
        list_result_bindings(run.id)
        if callable(list_result_bindings)
        else None
    )
    generations = (
        tx.work.correction_input_generations(run.id)
        if isinstance(snapshot, CorrectionSnapshotV2)
        else {}
    )
    list_for_section = getattr(tx.results, 'list_for_section', None)
    run_results = (
        None
        if callable(list_for_section)
        else tx.results.list_for_run(run.id)
    )
    return dict(
        run=run,
        conditions=conditions,
        snapshot=snapshot,
        analyses_by_section=analyses_by_section,
        result_bindings=result_bindings,
        generations=generations,
        run_results=run_results,
    )


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
    snapshot = parse_correction_snapshot(policy)
    document = tx.documents.get(book_id)
    if document is None:
        raise WorkStateError('книга исправления отсутствует')
    section = next((s for s in document.ordered_sections if s.id == section_id), None)
    if section is None:
        raise WorkStateError('глава исправления отсутствует')
    return _section_correction_inputs(
        tx,
        document=document,
        section=section,
        run=run,
        conditions=conditions,
        snapshot=snapshot,
    )


def _is_pure_read_as_is(block: DocumentBlock, inputs: CorrectionInputs) -> bool:
    if inputs.section.is_notes_only:
        return False
    treatment = None
    if inputs.analysis is not None and inputs.analysis.treatment_plan is not None:
        try:
            treatment = inputs.analysis.treatment_plan.treatment_for(block.id)
        except Exception:
            pass
    if treatment is None and block.id in inputs.results:
        treatment = inputs.results[block.id].treatment
    if treatment != BlockTreatment.READ_AS_IS:
        return False
    result = inputs.results.get(block.id)
    if result is not None:
        if result.language_notes:
            return False
        if getattr(result, "fragment_assistances", ()):
            return False
    if inputs.analysis is not None and inputs.analysis.content_plan is not None:
        plan = inputs.analysis.content_plan
        for d in plan.difficulties:
            if d.owner_block_id == block.id or any(a.block_id == block.id for a in d.anchors):
                return False
    return True


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
            if isinstance(inputs.snapshot, CorrectionSnapshotV2):
                if check.basis == 'unchanged':
                    group = tx.work.get_binding('correction_group', check.attempt_id)
                    payload = {} if group is None else group['payload_json']
                    if (check.block_id not in payload.get('unchanged_block_ids', ())
                            or payload.get('manifest', {}).get('bindings', {}).get(check.block_id) != inputs.bindings[check.block_id]
                            or payload.get('manifest', {}).get('input_generations', {}).get(check.block_id) != inputs.input_generations[check.block_id]):
                        continue
                elif check.basis in {'confirmed', 'human'}:
                    candidate = tx.work.get_correction_candidate(check.candidate_id)
                    if (candidate is None or not candidate_is_current(tx, inputs, candidate)
                            or candidate.state not in {'auto_applied', 'accepted', 'kept'}
                            or candidate.proposal_attempt_id != check.attempt_id):
                        continue
                else:
                    continue
            checked[check.block_id] = check.input_sha256
    return tuple(
        b.id for b in inputs.blocks
        if checked.get(b.id) != current[b.id] and not _is_pure_read_as_is(b, inputs)
    )


def candidate_is_current(tx, inputs, candidate):
    if (candidate.state == 'stale' or candidate.run_id != inputs.run.id
            or candidate.conditions_id != inputs.conditions.id or candidate.book_id != inputs.document.id
            or candidate.section_id != inputs.section.id or candidate.block_id not in inputs.results
            or candidate.input_generation != inputs.input_generations.get(candidate.block_id)
            or candidate.payload.get('context_sha256') != inputs.context_sha256):
        return False
    binding = inputs.bindings[candidate.block_id]
    if candidate.state in {'awaiting_review', 'awaiting_human', 'kept'}:
        return (candidate.input_sha256 == inputs.checksums[candidate.block_id]
                and thaw(candidate.payload['source_binding']) == binding)
    if candidate.state not in {'auto_applied', 'accepted'} or binding is None:
        return False
    artifact = tx.work.get_artifact(binding['artifact_revision_id'])
    return (artifact is not None and artifact['body'].get('correction_candidate_id') == candidate.candidate_id
            and artifact['body'].get('proposal_sha256') == candidate.proposal_sha256
            and binding['attempt_id'] == candidate.proposal_attempt_id
            and canonical_sha(asdict(inputs.results[candidate.block_id])) == canonical_sha(thaw(candidate.payload['result'])))


def current_correction_candidates(tx, inputs, *, states=None):
    if not isinstance(inputs.snapshot, CorrectionSnapshotV2):
        return ()
    return tuple(c for c in tx.work.list_correction_candidates(inputs.run.id, section_id=inputs.section.id,
        states=states if states is not None else {'awaiting_review', 'awaiting_human', 'auto_applied', 'accepted', 'kept'})
        if candidate_is_current(tx, inputs, c))


def model_correction_blocks(tx, inputs):
    waiting = {c.block_id for c in current_correction_candidates(tx, inputs, states={'awaiting_human'})}
    return tuple(b for b in unchecked_blocks(tx, inputs) if b not in waiting)


def correction_input_fingerprint(inputs, block_id):
    return canonical_sha(dict(input_sha256=inputs.checksums[block_id],
                              source_result_binding=inputs.bindings[block_id]))


def _observe_correction_inputs(tx, inputs):
    if not isinstance(inputs.snapshot, CorrectionSnapshotV2):
        return inputs
    updated_generations = dict(inputs.input_generations)
    for block in inputs.blocks:
        generation = tx.work.observe_correction_input(inputs.run.id, block.id,
            correction_input_fingerprint(inputs, block.id))
        if inputs.input_generations.get(block.id) and generation != inputs.input_generations.get(block.id):
            tx.work.stale_correction_candidates(inputs.run.id, block_ids=(block.id,))
        updated_generations[block.id] = generation
    return replace(inputs, input_generations=updated_generations)


def _restore_correction_checks(tx, inputs, tasks):
    if not isinstance(inputs.snapshot, CorrectionSnapshotV2):
        return
    candidates = tx.work.list_correction_candidates(inputs.run.id, section_id=inputs.section.id,
        states={'awaiting_review', 'awaiting_human', 'auto_applied', 'accepted', 'kept'})
    checks = []
    for candidate in candidates:
        if not candidate_is_current(tx, inputs, candidate):
            if candidate.state != 'stale':
                tx.work.stale_correction_candidates(inputs.run.id, candidate_ids=(candidate.candidate_id,))
            continue
        if candidate.state in {'auto_applied', 'accepted', 'kept'}:
            checks.append(CorrectionCheck(inputs.run.id, candidate.block_id, inputs.section.id,
                inputs.checksums[candidate.block_id], candidate.proposal_attempt_id,
                'confirmed' if candidate.state == 'auto_applied' else 'human', candidate.candidate_id))
    for task in tasks:
        if task.task_type != TaskType.CHAPTER_CORRECTION or task.scope_id != inputs.section.id:
            continue
        for attempt in tx.work.list_task_attempts(task.id):
            if attempt.state != 'accepted' or attempt.run_id != inputs.run.id:
                continue
            group = tx.work.get_binding('correction_group', attempt.id)
            if group is None:
                continue
            payload = group['payload_json']
            members = [c for c in candidates if c.group_id == attempt.id]
            if any(c.review is None or c.state == 'stale' for c in members):
                continue
            for block_id in payload.get('unchanged_block_ids', ()):
                if (payload['manifest']['checksums'].get(block_id) == inputs.checksums.get(block_id)
                        and payload['manifest']['bindings'].get(block_id) == inputs.bindings.get(block_id)
                        and payload['manifest'].get('input_generations', {}).get(block_id) == inputs.input_generations.get(block_id)):
                    checks.append(CorrectionCheck(inputs.run.id, block_id, inputs.section.id,
                        inputs.checksums[block_id], attempt.id, 'unchanged'))
    if checks:
        unique = {c.block_id: c for c in checks}
        tx.work.replace_correction_checks(inputs.run.id, inputs.section.id, tuple(unique.values()))


def task_matches_inputs(tx, task, payload, inputs):
    """Accept proven successors of the task's exact inputs, including adopted candidates."""
    baseline = payload.get('manifest', {})
    checks = {c.block_id: c for c in tx.work.correction_checks(inputs.run.id, inputs.section.id)}
    current = inputs.manifest
    if set(baseline.get('checksums', {})) != set(current['checksums']):
        return False
    for block_id, checksum in current['checksums'].items():
        if (isinstance(inputs.snapshot, CorrectionSnapshotV2)
                and baseline.get('input_generations', {}).get(block_id) != inputs.input_generations[block_id]):
            return False
        if (baseline['checksums'][block_id] == checksum
                and baseline.get('bindings', {}).get(block_id) == inputs.bindings[block_id]):
            continue
        check = checks.get(block_id)
        attempt = None if check is None else tx.work.get_attempt(check.attempt_id)
        if (check is None or check.input_sha256 != checksum or attempt is None
                or attempt.state != 'accepted'
                or attempt.book_id != inputs.document.id or attempt.run_id != inputs.run.id
                or attempt.conditions_id != inputs.conditions.id
                or attempt.context.get('stage') != 'chapter_correction'):
            return False
        if not isinstance(inputs.snapshot, CorrectionSnapshotV2):
            if attempt.task_id != task.id:
                return False
            continue
        candidate = tx.work.get_correction_candidate(check.candidate_id)
        if (candidate is None or candidate.proposal_attempt_id != attempt.id
                or candidate.block_id != block_id
                or candidate.input_generation != baseline.get('input_generations', {}).get(block_id)
                or candidate.input_sha256 != baseline['checksums'][block_id]
                or thaw(candidate.payload['source_binding']) != baseline.get('bindings', {}).get(block_id)
                or not candidate_is_current(tx, inputs, candidate)):
            return False
        if check.basis == 'confirmed' and candidate.state == 'auto_applied':
            review_attempt = tx.work.get_attempt(candidate.review_attempt_id)
            receipt = None if review_attempt is None else tx.work.get_binding('correction_review_request', review_attempt.id)
            refs = () if receipt is None else receipt['payload_json'].get('candidate_refs', ())
            if (review_attempt is None or review_attempt.state != 'accepted' or review_attempt.task_id != task.id
                    or review_attempt.book_id != inputs.document.id or review_attempt.run_id != inputs.run.id
                    or review_attempt.conditions_id != inputs.conditions.id
                    or review_attempt.context.get('stage') != 'chapter_correction_review'
                    or (candidate.candidate_id, candidate.proposal_sha256) not in tuple(map(tuple, refs))):
                return False
        elif not (check.basis == 'human' and candidate.state == 'accepted'
                  and candidate.decision and candidate.decision.get('action') == 'accept'
                  and candidate.decision.get('reason')):
            return False
    return True


def correction_coverage_complete(tx, book_id):
    """Legacy runs retain their prior export semantics; required runs need current checks."""
    run = tx.runs.active_for_book(book_id)
    binding = None if run is None else tx.work.get_binding('run', run.id)
    policy = correction_policy({} if binding is None else binding['payload_json'])
    if policy is None:
        return True
    document = tx.documents.get(book_id)
    if document is None:
        return False
    sections = tuple(
        section for section in document.ordered_sections
        if not document.is_front_matter(section) and not section.is_notes_only
    )
    if not sections:
        return True
    shared = _load_book_correction_inputs_state(
        tx, book_id, run=run, binding=binding, policy=policy,
    )
    result_bindings = shared.get('result_bindings')
    if result_bindings is not None:
        expected_blocks = {
            block.id
            for section in document.sections
            for block in textual_blocks(section)
        }
        if not all(result_bindings.get(block_id) is not None for block_id in expected_blocks):
            return False
    elif shared.get('run_results') is not None:
        expected_blocks = {
            block.id
            for section in document.sections
            for block in textual_blocks(section)
        }
        actual_blocks = {r.block_id for r in shared['run_results']}
        if not expected_blocks <= actual_blocks:
            return False
    for section in sections:
        inputs = _section_correction_inputs(tx, document=document, section=section, **shared)
        if (not inputs.translated or model_correction_blocks(tx, inputs)
                or current_correction_candidates(tx, inputs, states={'awaiting_review'})):
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


def _reset_book_readiness(tx, book_id):
    books = getattr(tx, 'books', None)
    if books is None:
        return
    book = books.get(book_id)
    if book is not None and (
        book.status in {BookStatus.READY_TO_EXPORT, BookStatus.COMPLETED}
        or getattr(book, 'export_validation_status', None) is not None
    ):
        books.save(replace(book, status=BookStatus.PROCESSING, export_validation_status=None))


def reconcile_corrections(tx, book_id, *, section_id: str | None = None):
    """ALG-03: restore tasks from run policy, preserving explicit retries of FAILED work."""
    effective_section_id = (
        section_id if section_id is not None else _SCOPED_RECONCILE_SECTION_ID.get()
    )
    run = tx.runs.active_for_book(book_id)
    binding = None if run is None else tx.work.get_binding('run', run.id)
    policy = correction_policy({} if binding is None else binding['payload_json'])
    if policy is None:
        return ()
    document = tx.documents.get(book_id)
    if document is None:
        return ()
    sections = tuple(
        section for section in document.ordered_sections
        if not document.is_front_matter(section) and not section.is_notes_only
    )
    if effective_section_id is not None:
        matched = tuple(section for section in sections if section.id == effective_section_id)
        if not matched:
            matched = tuple(
                section
                for section in sections
                if any(block.id == effective_section_id for block in section.blocks)
            )
            if matched:
                effective_section_id = matched[0].id
        sections = matched
    if not sections:
        return ()
    shared = _load_book_correction_inputs_state(
        tx,
        book_id,
        run=run,
        binding=binding,
        policy=policy,
        section_id=effective_section_id,
    )
    tasks = tx.tasks.list_for_book(book_id)
    pending = []
    complete = True
    for section in sections:
        inputs = _section_correction_inputs(tx, document=document, section=section, **shared)
        inputs = _observe_correction_inputs(tx, inputs)
        _restore_correction_checks(tx, inputs, tasks)
        model_missing = model_correction_blocks(tx, inputs)
        awaiting_review = current_correction_candidates(tx, inputs, states={'awaiting_review'})
        covered = inputs.translated and not model_missing and not awaiting_review
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
                    section_id=section.id, input_generation=payload['input_generation'],
                    policy_version=inputs.snapshot.policy_version))
            if current and task.status is not TaskStatus.SUPERSEDED:
                matching = task
            elif task.status is not TaskStatus.SUPERSEDED:
                _supersede(tx, task)
        if covered and isinstance(inputs.snapshot, CorrectionSnapshotV2) and matching is not None:
            if matching.status not in {TaskStatus.SUCCEEDED, TaskStatus.SUPERSEDED}:
                tx.tasks.save(replace(matching, status=TaskStatus.SUCCEEDED, last_error=None))
        if covered or not inputs.translated or not model_missing:
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
            section_id=section.id, input_generation=generation, policy_version=inputs.snapshot.policy_version)
        task = tx.tasks.find_by_identity(identity)
        if task is None or task.status is TaskStatus.SUPERSEDED:
            if task is not None:
                # Returning to old input still represents a fresh operation.
                generation = canonical_sha(dict(generation=generation, nonce=str(uuid4())))
                identity = correction_task_identity(book_id=book_id, run_id=run.id,
                    section_id=section.id, input_generation=generation, policy_version=inputs.snapshot.policy_version)
            task = ProcessingTask(id=str(uuid4()), book_id=book_id,
                task_type=identity.task_type, scope_id=identity.scope_id,
                operation_id=identity.operation_id, policy_version=identity.policy_version)
            tx.tasks.add(task)
            tx.work.bind('task', task.id, book_id=book_id, conditions_id=inputs.conditions.id,
                payload=dict(schema_version=1, run_id=run.id, section_id=section.id,
                    input_generation=generation, manifest=inputs.manifest))
        pending.append(task)
    if not complete:
        _reset_book_readiness(tx, book_id)
    return tuple(pending)


def invalidate_correction(tx, book_id, run_id, block_ids, *, reconcile: bool = True) -> None:
    """Revoke checks and readiness in the transaction that changes source results."""
    block_ids = tuple(block_ids)
    tx.work.delete_correction_checks(run_id, block_ids=block_ids)
    binding = tx.work.get_binding('run', run_id)
    policy = {} if binding is None else binding['payload_json'].get('correction', {})
    if policy.get('policy_version') == CORRECTION_POLICY_VERSION_V2:
        tx.work.stale_correction_candidates(run_id, block_ids=block_ids)
    if reconcile:
        reconcile_corrections(tx, book_id)
    else:
        _reset_book_readiness(tx, book_id)
