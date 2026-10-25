"""Durable recovery and physical call identities."""
from dataclasses import dataclass, replace
from collections.abc import Mapping
from linguary.domain.book_conditions import WorkSerialization, WorkStateError, freeze, identifier, nonnegative, thaw, version


@dataclass(frozen=True, slots=True)
class CallAttempt:
    id: str
    book_id: str
    state: str = 'created'
    generation: int = 0
    ordinal: int = 0
    conditions_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    root_id: str | None = None
    scope_id: str | None = None
    review_id: str | None = None
    context: Mapping = None
    outcome: Mapping = None
    schema_version: int = 1

    def __post_init__(self):
        identifier(self.id)
        version(self.schema_version)
        for value in (self.generation, self.ordinal):
            nonnegative(value)
        for name in ('conditions_id', 'root_id', 'scope_id', 'review_id'):
            if getattr(self, name) is not None:
                identifier(getattr(self, name))
        if self.state not in {'created', 'reserved', 'sent', 'validated', 'accepted', 'rejected', 'failed', 'unknown', 'cancelled'}:
            raise WorkStateError('неизвестное состояние попытки')
        for name in ('context', 'outcome'):
            value = getattr(self, name) or {'schema_version': 1}
            version(value.get('schema_version'))
            object.__setattr__(self, name, freeze(value))

    def transition(self, state, *, outcome=None):
        allowed = {'created': {'reserved', 'sent', 'cancelled'}, 'reserved': {'sent', 'cancelled'},
                   'sent': {'validated', 'rejected', 'failed', 'unknown'},
                   'validated': {'accepted', 'rejected'}, 'unknown': {'validated', 'failed'}}
        if state not in allowed.get(self.state, set()):
            raise WorkStateError('недопустимый переход попытки')
        return replace(self, state=state, outcome=outcome or self.outcome)

    def to_dict(self):
        return {name: thaw(getattr(self, name)) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value):
        version(value.get('schema_version'))
        return cls(**value)


@dataclass(frozen=True, slots=True)
class RecoveryScope(WorkSerialization):
    id: str
    root_id: str
    ordered_blocks: tuple[str, ...]
    generation: int = 0
    status: str = 'pending'
    parent_id: str | None = None
    accepted_attempt_id: str | None = None
    schema_version: int = 1

    def __post_init__(self):
        identifier(self.id)
        identifier(self.root_id)
        version(self.schema_version)
        nonnegative(self.generation)
        object.__setattr__(self, 'ordered_blocks', tuple(self.ordered_blocks))
        if not self.ordered_blocks or len(set(self.ordered_blocks)) != len(self.ordered_blocks):
            raise WorkStateError('область должна содержать уникальные блоки')
        for value in (self.parent_id, self.accepted_attempt_id):
            if value is not None:
                identifier(value)
        if self.status not in {'pending', 'claimed', 'accepted', 'split', 'stopped'}:
            raise WorkStateError('неизвестное состояние области')
        if (self.status == 'accepted') != (self.accepted_attempt_id is not None):
            raise WorkStateError('принятая область требует попытку')

    @property
    def semantic_correctness(self):
        return 'not_evaluated'

    def transition(self, status, *, attempt_id=None):
        if status not in {'pending': {'claimed', 'stopped'}, 'claimed': {'accepted', 'pending', 'split', 'stopped'}}.get(self.status, set()):
            raise WorkStateError('недопустимый переход области')
        return replace(self, status=status, accepted_attempt_id=attempt_id)


@dataclass(frozen=True, slots=True)
class RecoveryRoot(WorkSerialization):
    id: str
    book_id: str
    conditions_id: str
    root_key: str
    manifest: Mapping
    generation: int = 0
    calls_used: int = 0
    extra_calls: int = 0
    status: str = 'pending'
    reason: str | None = None
    schema_version: int = 1

    def __post_init__(self):
        for value in (self.id, self.conditions_id):
            identifier(value)
        version(self.schema_version)
        for value in (self.generation, self.calls_used, self.extra_calls):
            nonnegative(value)
        version(self.manifest.get('schema_version'))
        object.__setattr__(self, 'manifest', freeze(self.manifest))
        if self.status not in {'pending', 'running', 'complete', 'stopped'}:
            raise WorkStateError('неизвестное состояние восстановления')


    def transition(self, status, *, reason=None):
        allowed = {'pending': {'running', 'stopped'}, 'running': {'complete', 'stopped'},
                   'stopped': {'running'}}
        if status not in allowed.get(self.status, set()):
            raise WorkStateError('недопустимый переход восстановления')
        return replace(self, status=status, reason=reason, generation=self.generation + 1)
