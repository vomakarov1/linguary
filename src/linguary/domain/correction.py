"""Durable chapter correction policy, coverage, and operation identity."""

from collections.abc import Mapping
from dataclasses import dataclass

from linguary.domain.book_conditions import WorkStateError, freeze, thaw
from linguary.domain.content_plan import canonical_sha
from linguary.domain.tasks import TaskIdentity, TaskType


CORRECTION_POLICY_VERSION = "chapter-correction-v1"
CORRECTION_POLICY_VERSION_V2 = "chapter-correction-v2"


def correction_policy(run_payload: Mapping) -> Mapping | None:
    """An absent run policy retains legacy processing semantics."""
    if "correction" not in run_payload:
        return None
    policy = run_payload["correction"]
    if (not isinstance(policy, Mapping) or policy.get("required") is not True
            or policy.get("policy_version") not in {CORRECTION_POLICY_VERSION, CORRECTION_POLICY_VERSION_V2}):
        raise WorkStateError("неподдерживаемая политика исправления главы")
    return policy


@dataclass(frozen=True, slots=True)
class CorrectionCheck:
    run_id: str
    block_id: str
    section_id: str
    input_sha256: str
    attempt_id: str
    basis: str = 'legacy'
    candidate_id: str | None = None

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.run_id, self.block_id, self.section_id, self.attempt_id)):
            raise WorkStateError("отметка исправления требует запуск, блок, главу и попытку")
        if (not isinstance(self.input_sha256, str) or len(self.input_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.input_sha256)):
            raise WorkStateError("неверная контрольная сумма исправления")
        if self.basis not in {'legacy', 'unchanged', 'confirmed', 'human'}:
            raise WorkStateError('неверное основание исправления')
        if (self.basis in {'confirmed', 'human'}) != (self.candidate_id is not None):
            raise WorkStateError('отметка исправления не связана с кандидатом')


@dataclass(frozen=True, slots=True)
class CorrectionCandidate:
    candidate_id: str
    book_id: str
    run_id: str
    conditions_id: str
    section_id: str
    block_id: str
    input_sha256: str
    input_generation: int
    proposal_sha256: str
    payload: Mapping
    group_id: str
    proposal_attempt_id: str
    state: str = 'awaiting_review'
    review_attempt_id: str | None = None
    review: Mapping | None = None
    decision: Mapping | None = None

    def __post_init__(self):
        if self.state not in {'awaiting_review', 'awaiting_human', 'auto_applied', 'accepted', 'kept', 'stale'}:
            raise WorkStateError('неверное состояние кандидата исправления')
        if type(self.input_generation) is not int or self.input_generation < 1:
            raise WorkStateError('неверное поколение входов исправления')
        if any(not isinstance(getattr(self, name), str) or not getattr(self, name).strip()
               for name in ('candidate_id', 'book_id', 'run_id', 'conditions_id', 'section_id',
                            'block_id', 'group_id', 'proposal_attempt_id')):
            raise WorkStateError('неполная принадлежность кандидата исправления')
        if canonical_sha(thaw(self.payload)) != self.proposal_sha256:
            raise WorkStateError('контрольная сумма предложения не совпадает')
        if any(self.payload.get(name) != getattr(self, name) for name in
               ('book_id', 'run_id', 'conditions_id', 'section_id', 'block_id',
                'input_sha256', 'input_generation', 'group_id', 'proposal_attempt_id')):
            raise WorkStateError('принадлежность предложения не совпадает с его содержимым')
        for name in ('payload', 'review', 'decision'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, freeze(value))


@dataclass(frozen=True, slots=True)
class CorrectionDecision:
    action: str
    reason: str

    def __post_init__(self):
        if self.action not in {'accept', 'keep'} or not isinstance(self.reason, str) or not self.reason.strip():
            raise WorkStateError('решение исправления требует действие и причину')


def correction_task_identity(*, book_id: str, run_id: str, section_id: str,
                             input_generation: str, policy_version: str = CORRECTION_POLICY_VERSION) -> TaskIdentity:
    """A changed input generation creates a new operation, even after success."""
    if any(not isinstance(value, str) or not value.strip()
           for value in (run_id, section_id, input_generation)):
        raise WorkStateError("неполная идентичность исправления главы")
    return TaskIdentity(
        book_id=book_id, task_type=TaskType.CHAPTER_CORRECTION, scope_id=section_id,
        operation_id=canonical_sha({"run_id": run_id, "section_id": section_id,
                                    "input_generation": input_generation}),
        policy_version=policy_version,
    )
