"""Optional review and monetary reservation state, in integer micro-USD."""
from collections.abc import Mapping
from dataclasses import dataclass, replace
from linguary.domain.book_conditions import WorkSerialization, WorkStateError, freeze, identifier, nonnegative, version


@dataclass(frozen=True, slots=True)
class QualityReservation(WorkSerialization):
    attempt_id: str
    review_id: str
    estimate: int
    actual: int | None = None
    state: str = 'reserved'
    price: Mapping = None
    schema_version: int = 1

    def __post_init__(self):
        identifier(self.attempt_id)
        identifier(self.review_id)
        version(self.schema_version)
        nonnegative(self.estimate)
        if self.actual is not None:
            nonnegative(self.actual)
        if self.state not in {'reserved', 'unknown', 'settled', 'released'}:
            raise WorkStateError('неизвестное состояние резерва')
        if (self.state == 'settled') != (self.actual is not None):
            raise WorkStateError('только завершённый резерв имеет известный расход')
        price = self.price or {'schema_version': 1}
        version(price.get('schema_version'))
        object.__setattr__(self, 'price', freeze(price))

    @property
    def held(self):
        return self.estimate if self.state in {'reserved', 'unknown'} else 0

    def transition(self, state, *, actual=None):
        if state not in {'reserved': {'unknown', 'settled', 'released'}, 'unknown': {'settled'}}.get(self.state, set()):
            raise WorkStateError('недопустимый переход резерва')
        return replace(self, state=state, actual=actual)


@dataclass(frozen=True, slots=True)
class QualityReview(WorkSerialization):
    id: str
    book_id: str
    request_key: str
    manifest: Mapping
    settings: Mapping
    budget_limit: int
    conditions_id: str | None = None
    state: str = 'queued'
    revision: int = 0
    findings: tuple = ()
    spent: int = 0
    reserved: int = 0
    schema_version: int = 1

    def __post_init__(self):
        identifier(self.id)
        version(self.schema_version)
        if self.conditions_id is not None:
            identifier(self.conditions_id)
        if not self.request_key or not self.book_id:
            raise WorkStateError('не задана идентичность вычитки')
        for value in (self.budget_limit, self.revision, self.spent, self.reserved):
            nonnegative(value)
        if self.budget_limit == 0:
            raise WorkStateError('требуется положительный бюджет')
        for name in ('manifest', 'settings'):
            value = getattr(self, name)
            version(value.get('schema_version'))
            object.__setattr__(self, name, freeze(value))
        object.__setattr__(self, 'findings', freeze(self.findings))
        if self.state not in {'queued', 'running', 'completed', 'failed', 'limit', 'unknown', 'stale', 'cancelled'}:
            raise WorkStateError('неизвестное состояние вычитки')

    def transition(self, state):
        allowed = {'queued': {'running', 'cancelled', 'stale'}, 'running': {'completed', 'failed', 'limit', 'unknown', 'stale'}, 'limit': {'running', 'cancelled'}, 'unknown': {'running', 'failed'}}
        if state not in allowed.get(self.state, set()):
            raise WorkStateError('недопустимый переход вычитки')
        return replace(self, state=state, revision=self.revision + 1)
