"""Apply a human decision to an unchanged, immutable correction proposal."""

from dataclasses import dataclass

from linguary.application.block_processing import complete_processing_in_transaction
from linguary.application.correction import candidate_is_current, correction_inputs, correction_input_fingerprint
from linguary.application.correction_dto import CorrectionSnapshotV2
from linguary.application.errors import ApplicationError
from linguary.application.use_cases.correct_chapter import apply_correction_candidate, prepare_correction_candidates
from linguary.domain import TaskType
from linguary.domain.correction import CorrectionCheck, CorrectionDecision


class CorrectionCandidateNotFound(ApplicationError):
    pass


class CorrectionCandidateConflict(ApplicationError):
    pass


@dataclass(frozen=True, slots=True)
class ResolveCorrectionCommand:
    book_id: str
    candidate_id: str
    proposal_sha256: str
    input_sha256: str
    action: str
    reason: str


class ResolveCorrectionCandidate:
    def __init__(self, *, unit_of_work, validator_factory):
        self.uow, self.validator_factory = unit_of_work, validator_factory

    def __call__(self, command):
        decision = CorrectionDecision(command.action, command.reason.strip())
        with self.uow.transaction() as tx:
            candidate = tx.work.get_correction_candidate(command.candidate_id)
            if candidate is None or candidate.book_id != command.book_id or tx.books.get(command.book_id) is None:
                raise CorrectionCandidateNotFound('предложение исправления не найдено')
            inputs = correction_inputs(tx, command.book_id, candidate.section_id)
            if (inputs is None or not isinstance(inputs.snapshot, CorrectionSnapshotV2)
                    or candidate.proposal_sha256 != command.proposal_sha256
                    or candidate.input_sha256 != command.input_sha256
                    or not candidate_is_current(tx, inputs, candidate)):
                raise CorrectionCandidateConflict('данные изменились: обновите страницу результата')
            if candidate.decision is not None:
                if candidate.decision['action'] == command.action:
                    return candidate.block_id
                raise CorrectionCandidateConflict('для предложения уже сохранено другое решение')
            if candidate.state != 'awaiting_human':
                raise CorrectionCandidateConflict('предложение сейчас не ожидает решения человека')
            results = ()
            if decision.action == 'accept':
                results = prepare_correction_candidates(tx, inputs, (candidate,),
                    self.validator_factory(inputs.conditions))
            tx.work.resolve_correction_candidate(candidate.candidate_id,
                proposal_sha256=candidate.proposal_sha256, input_sha256=candidate.input_sha256,
                decision={'action': decision.action, 'reason': decision.reason})
            updated_bindings = {}
            if results:
                updated_bindings[candidate.block_id] = apply_correction_candidate(
                    tx, inputs, candidate, results[0], resolution='human'
                )
                tx.results.replace_results(inputs.run.id, results)
                tx.work.stale_reviews(command.book_id)
            after = (
                inputs.with_updated_state(
                    updated_results=results,
                    updated_bindings=updated_bindings if updated_bindings else None,
                )
                if hasattr(inputs, 'with_updated_state')
                else correction_inputs(tx, command.book_id, candidate.section_id)
            )
            tx.work.advance_correction_input(inputs.run.id, candidate.block_id,
                expected_generation=candidate.input_generation,
                fingerprint=correction_input_fingerprint(after, candidate.block_id))
            tx.work.replace_correction_checks(inputs.run.id, candidate.section_id, (
                CorrectionCheck(inputs.run.id, candidate.block_id, candidate.section_id,
                    after.checksums[candidate.block_id], candidate.proposal_attempt_id,
                    'human', candidate.candidate_id),))
            complete_processing_in_transaction(tx, book_id=command.book_id, finished_task_id=None,
                task_types={TaskType.PROCESS_TRANSLATION_UNIT, TaskType.REGENERATE_BLOCK,
                            TaskType.CHAPTER_CORRECTION})
            return candidate.block_id
