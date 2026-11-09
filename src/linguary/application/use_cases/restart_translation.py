"""Start a new translation operation for the unfinished part of a saved root."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from uuid import uuid4

from linguary.application.errors import ApplicationError
from linguary.application.book_conditions import payload_digest
from linguary.application.ports.persistence import UnitOfWork
from linguary.application.recovery import NOTES_POLICY, RecoveryPolicy, root_manifest
from linguary.application.task_runner import TaskQueue
from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY, require_policy
from linguary.domain import ProcessingTask, TaskType
from linguary.domain.book_conditions import WorkStateError
from linguary.domain.errors import UnknownBlockError
from linguary.domain.recovery import RecoveryRoot, RecoveryScope


class RestartTranslationError(ApplicationError):
    """The selected translation root cannot be restarted safely."""


class RestartTranslationConflict(RestartTranslationError):
    """The root changed after the caller last read it."""


@dataclass(frozen=True, slots=True)
class RestartTranslationCommand:
    """Restart one saved chapter root at the exact generation shown to the caller."""

    book_id: str
    root_id: str
    expected_generation: int


@dataclass(frozen=True, slots=True)
class RestartTranslationResult:
    """Identity of the successor operation and its durable queue task."""

    successor_root_id: str
    task_id: str


class RestartTranslation:
    """Transfer only unfinished, safe scopes to a new current-policy operation."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        queue: TaskQueue,
        recovery_policy: Callable[[], RecoveryPolicy],
    ) -> None:
        self._unit_of_work = unit_of_work
        self._queue = queue
        self._recovery_policy = recovery_policy

    def __call__(self, command: RestartTranslationCommand) -> RestartTranslationResult:
        with self._unit_of_work.transaction() as transaction:
            root = transaction.work.get_root(command.root_id)
            if root is None or root.book_id != command.book_id:
                raise RestartTranslationError("корень перевода не найден")
            if root.generation != command.expected_generation or root.superseded_by is not None:
                raise RestartTranslationConflict("операция уже изменилась; обновите состояние и повторите")
            if root.operation_kind != "translation":
                raise RestartTranslationError("перезапустить можно только обработку главы")
            try:
                require_policy(root.policy_version)
            except Exception:
                raise RestartTranslationError("версия операции больше не поддерживается") from None

            conditions = transaction.work.latest_conditions(command.book_id)
            run_id = root.manifest.get("active_run_id")
            run = transaction.runs.get(run_id) if run_id else None
            document = transaction.documents.get(command.book_id)
            if (conditions is None or conditions.id != root.conditions_id or run is None
                    or run.book_id != command.book_id or document is None
                    or run.status.value != "active"):
                raise RestartTranslationError("условия перевода или запуск книги недоступны")

            ordered = tuple(root.manifest.get("ordered_blocks", ()))
            if not ordered or len(set(ordered)) != len(ordered):
                raise RestartTranslationError("область операции повреждена")
            try:
                blocks = tuple(document.block(block_id) for block_id in ordered)
            except (KeyError, ValueError, UnknownBlockError):
                raise RestartTranslationError("исходные блоки операции недоступны") from None
            section_ids = {block.section_id for block in blocks}
            if len(section_ids) != 1:
                raise RestartTranslationError("область операции пересекает главы")
            section_id = next(iter(section_ids))
            analysis = transaction.analyses.get_for_section(command.book_id, section_id)
            if (analysis is None or root.manifest.get("analysis_sha") != payload_digest(asdict(analysis))
                    or root.manifest.get("glossary_sha") != payload_digest(asdict(run.glossary_snapshot))
                    or root.manifest.get("notes_policy") != NOTES_POLICY):
                raise RestartTranslationError("исходные условия области изменились")

            historical = set(root.manifest.get("historical_coverage", ()))
            pending_scopes = []
            leaf_blocks = []
            for scope in transaction.work.list_scopes(root.id):
                if scope.root_id != root.id or not set(scope.ordered_blocks) <= set(ordered):
                    raise RestartTranslationError("область операции повреждена")
                if scope.status == "split":
                    continue
                if scope.status == "accepted":
                    self._verify_accepted_scope(transaction, root, scope, run_id)
                    historical.update(scope.ordered_blocks)
                else:
                    pending_scopes.append(scope)
                    leaf_blocks.extend(scope.ordered_blocks)
            positions = {block_id: index for index, block_id in enumerate(ordered)}
            historical_ordered = tuple(block_id for block_id in ordered if block_id in historical)
            pending_ordered = tuple(block_id for block_id in ordered if block_id not in historical)
            if root.status == "complete" or not pending_ordered:
                raise RestartTranslationError("все блоки уже приняты; используйте повторную генерацию")
            if root.status == "stopped" and root.reason != "limit":
                raise RestartTranslationError("для этой области нельзя безопасно запустить перевод заново")
            if (not historical <= set(ordered) or len(leaf_blocks) != len(set(leaf_blocks))
                    or historical.intersection(leaf_blocks)
                    or historical | set(leaf_blocks) != set(ordered)):
                raise RestartTranslationError("неполная область перевода")
            pending_scopes.sort(key=lambda scope: positions[scope.ordered_blocks[0]])

            policy = self._recovery_policy()
            manifest = root_manifest(
                book_id=root.book_id,
                conditions_id=root.conditions_id,
                glossary_sha=root.manifest["glossary_sha"],
                analysis_sha=root.manifest["analysis_sha"],
                plan_sha=root.manifest.get("plan_sha"),
                ordered_blocks=ordered,
                treatments=tuple(root.manifest["treatments"]),
                boundaries=tuple(tuple(group) for group in root.manifest["original_boundaries"]),
                atomic_intervals=tuple(tuple(group) for group in root.manifest["atomic_intervals"]),
                max_source_chars=root.manifest["max_source_chars"],
                policy=policy,
                block_chars=root.manifest.get("block_chars", {}),
                note_owners=tuple(root.manifest.get("note_owners", ())),
                chapter_chars=root.manifest.get("chapter_chars"),
                notes_per_chapter=root.manifest.get("notes_per_chapter"),
            )
            manifest.update(
                notes_policy=root.manifest["notes_policy"],
                active_run_id=run_id,
                historical_coverage=list(historical_ordered),
            )
            successor_id = str(uuid4())
            successor = RecoveryRoot(
                successor_id,
                root.book_id,
                root.conditions_id,
                f"{successor_id}:{CURRENT_TRANSLATION_POLICY}:{manifest['root_key']}",
                manifest,
                policy_version=CURRENT_TRANSLATION_POLICY,
                operation_kind="translation",
            )
            scopes = tuple(
                RecoveryScope(str(uuid4()), successor_id, tuple(scope.ordered_blocks))
                for scope in pending_scopes
            )
            task = ProcessingTask(
                id=_new_task_id(self._queue),
                book_id=command.book_id,
                task_type=TaskType.PROCESS_TRANSLATION_UNIT,
                scope_id=blocks[0].section_id,
                operation_id="dispatch:" + str(uuid4()),
            )
            try:
                transaction.work.restart_root(
                    root.id,
                    expected_generation=command.expected_generation,
                    successor=successor,
                    scopes=scopes,
                    task=task,
                )
            except WorkStateError as error:
                if str(error) == "generation_conflict":
                    raise RestartTranslationConflict("операция уже изменилась; обновите состояние и повторите") from None
                raise RestartTranslationError("операцию нельзя безопасно запустить заново") from None

        return RestartTranslationResult(successor_root_id=successor_id, task_id=task.id)

    @staticmethod
    def _verify_accepted_scope(transaction, root, scope, run_id):
        attempt = transaction.work.get_attempt(scope.accepted_attempt_id)
        if (attempt is None or attempt.state != "accepted" or attempt.root_id != root.id
                or attempt.scope_id != scope.id or attempt.conditions_id != root.conditions_id
                or attempt.run_id != run_id or attempt.context.get("policy_version") != root.policy_version):
            raise RestartTranslationError("принятый результат не имеет проверяемого происхождения")
        for block_id in scope.ordered_blocks:
            result = transaction.results.get(run_id, block_id)
            binding = transaction.work.get_binding("result", f"{run_id}:{block_id}")
            if result is None or binding is None:
                raise RestartTranslationError("принятый результат больше недоступен")
            if binding["attempt_id"] == attempt.id:
                continue
            replacement = transaction.work.get_attempt(binding["attempt_id"])
            if (
                replacement is None
                or replacement.state != "accepted"
                or replacement.run_id != run_id
                or replacement.conditions_id != root.conditions_id
            ):
                raise RestartTranslationError(
                    "принятый результат изменился без проверяемого происхождения"
                )
            if replacement.id != attempt.id:
                stage = replacement.context.get("stage")
                if stage == "chapter_correction":
                    translation_source = getattr(transaction.work, "_translation_source", None)
                    source_attempt = (
                        translation_source(
                            book_id=root.book_id,
                            run_id=run_id,
                            conditions_id=root.conditions_id,
                            block_id=block_id,
                            artifact_revision_id=binding["artifact_revision_id"],
                            attempt_id=binding["attempt_id"],
                        )
                        if callable(translation_source)
                        else None
                    )
                    if source_attempt is None:
                        raise RestartTranslationError(
                            "принятый результат изменился без проверяемого происхождения"
                        )
                    if source_attempt.id != attempt.id and (
                        source_attempt.state != "accepted"
                        or source_attempt.run_id != run_id
                        or source_attempt.conditions_id != root.conditions_id
                        or source_attempt.context.get("stage") != "block_regeneration"
                        or tuple(source_attempt.context.get("ordered_block_ids", ())) != (block_id,)
                    ):
                        raise RestartTranslationError(
                            "принятый результат изменился без проверяемого происхождения"
                        )
                elif (
                    stage != "block_regeneration"
                    or tuple(replacement.context.get("ordered_block_ids", ())) != (block_id,)
                ):
                    raise RestartTranslationError(
                        "принятый результат изменился без проверяемого происхождения"
                    )


def _new_task_id(queue: TaskQueue) -> str:
    factory = getattr(queue, "new_task_id", None)
    return factory() if callable(factory) else str(uuid4())


__all__ = [
    "RestartTranslation",
    "RestartTranslationCommand",
    "RestartTranslationConflict",
    "RestartTranslationError",
    "RestartTranslationResult",
]
