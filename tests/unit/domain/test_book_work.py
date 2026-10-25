from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from linguary.domain.book_conditions import BookConditions, WorkStateError, UnsupportedWorkVersion
from linguary.domain.recovery import CallAttempt, RecoveryScope
from linguary.domain.quality_review import QualityReservation


def test_immutable_conditions():
    payload = {'schema_version': 1, 'models': {'analysis': 'fake'}}
    conditions = BookConditions(str(uuid4()), 'book', 1, payload, datetime.now(UTC))
    payload['models']['analysis'] = 'changed'
    assert conditions.payload['models']['analysis'] == 'fake'
    with pytest.raises(TypeError):
        conditions.payload['models']['analysis'] = 'changed'
    with pytest.raises(FrozenInstanceError):
        conditions.revision = 2
    assert BookConditions.from_dict(conditions.to_dict()) == conditions
    with pytest.raises(UnsupportedWorkVersion):
        BookConditions.from_dict({**conditions.to_dict(), 'schema_version': 2})


def test_illegal_state_transition():
    attempt = CallAttempt(str(uuid4()), 'book')
    with pytest.raises(WorkStateError):
        attempt.transition('accepted')
    assert attempt.transition('sent').transition('validated').transition('accepted').state == 'accepted'


def test_unknown_is_not_zero():
    reservation = QualityReservation(str(uuid4()), str(uuid4()), 10)
    unknown = reservation.transition('unknown')
    assert unknown.actual is None
    assert unknown.held == 10
    assert reservation.transition('settled', actual=0).held == 0


def test_no_semantic_success_from_coverage():
    scope = RecoveryScope(str(uuid4()), str(uuid4()), ('b1',))
    assert scope.semantic_correctness == 'not_evaluated'


def test_all_work_entities_roundtrip_and_reject_unknown_versions():
    from linguary.domain.recovery import RecoveryRoot
    from linguary.domain.quality_review import QualityReview
    entities = [RecoveryRoot(str(uuid4()), 'book', str(uuid4()), 'key', {'schema_version': 1}),
                RecoveryScope(str(uuid4()), str(uuid4()), ('block',)),
                CallAttempt(str(uuid4()), 'book'),
                QualityReview(str(uuid4()), 'book', 'request', {'schema_version': 1}, {'schema_version': 1}, 1),
                QualityReservation(str(uuid4()), str(uuid4()), 1)]
    for entity in entities:
        assert type(entity).from_dict(entity.to_dict()) == entity
        with pytest.raises(UnsupportedWorkVersion):
            type(entity).from_dict({**entity.to_dict(), 'schema_version': 99})


@pytest.mark.parametrize('amount', [-1, 1.5, True])
def test_money_requires_nonnegative_integers(amount):
    with pytest.raises(WorkStateError):
        QualityReservation(str(uuid4()), str(uuid4()), amount)
