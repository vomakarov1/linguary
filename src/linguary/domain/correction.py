"""Durable chapter correction policy, coverage, and operation identity."""

from collections.abc import Mapping
from dataclasses import dataclass

from linguary.domain.book_conditions import WorkStateError
from linguary.domain.content_plan import canonical_sha
from linguary.domain.tasks import TaskIdentity, TaskType


CORRECTION_POLICY_VERSION = "chapter-correction-v1"


def correction_policy(run_payload: Mapping) -> Mapping | None:
    """An absent run policy retains legacy processing semantics."""
    if "correction" not in run_payload:
        return None
    policy = run_payload["correction"]
    if (not isinstance(policy, Mapping) or policy.get("required") is not True
            or policy.get("policy_version") != CORRECTION_POLICY_VERSION):
        raise WorkStateError("неподдерживаемая политика исправления главы")
    return policy


@dataclass(frozen=True, slots=True)
class CorrectionCheck:
    run_id: str
    block_id: str
    section_id: str
    input_sha256: str
    attempt_id: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.run_id, self.block_id, self.section_id, self.attempt_id)):
            raise WorkStateError("отметка исправления требует запуск, блок, главу и попытку")
        if (not isinstance(self.input_sha256, str) or len(self.input_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.input_sha256)):
            raise WorkStateError("неверная контрольная сумма исправления")


def correction_task_identity(*, book_id: str, run_id: str, section_id: str,
                             input_generation: str) -> TaskIdentity:
    """A changed input generation creates a new operation, even after success."""
    if any(not isinstance(value, str) or not value.strip()
           for value in (run_id, section_id, input_generation)):
        raise WorkStateError("неполная идентичность исправления главы")
    return TaskIdentity(
        book_id=book_id, task_type=TaskType.CHAPTER_CORRECTION, scope_id=section_id,
        operation_id=canonical_sha({"run_id": run_id, "section_id": section_id,
                                    "input_generation": input_generation}),
        policy_version=CORRECTION_POLICY_VERSION,
    )
