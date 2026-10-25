from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4
import pytest
from linguary.application.quality_budget import QualityBudget
from linguary.domain.quality_review import QualityReview
from linguary.domain.book_conditions import WorkStateError
from test_book_work_store import prepare


def setup_budget(uow, limit=10):
    conditions, _, _ = prepare(uow)
    review = QualityReview(str(uuid4()), 'book-1', 'review', {'schema_version': 1}, {'schema_version': 1}, limit, conditions.id)
    with uow.transaction() as tx:
        tx.work.add_review(review)
    return QualityBudget(uow), review


def test_two_workers_share_one_limit(unit_of_work):
    budget, review = setup_budget(unit_of_work)
    barrier = Barrier(2)
    def reserve(_):
        barrier.wait(timeout=5)
        try:
            return budget.reserve(review.id, 8, price={'schema_version': 1}, expected_revision=0)
        except WorkStateError:
            return None
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sum(result is not None for result in workers.map(reserve, range(2))) == 1


def test_crash_reserved_unsent_releases_once(unit_of_work):
    budget, review = setup_budget(unit_of_work)
    attempt = budget.reserve(review.id, 8, price={'schema_version': 1}, expected_revision=0)
    budget.reconcile()
    budget.reconcile()
    with unit_of_work.transaction() as tx:
        assert tx.work.get_review(review.id).reserved == 0
        assert tx.work.get_reservation(attempt.id).state == 'released'
        assert tx.work.get_attempt(attempt.id).state == 'cancelled'


def test_sent_unknown_keeps_reserve(unit_of_work):
    budget, review = setup_budget(unit_of_work)
    attempt = budget.reserve(review.id, 8, price={'schema_version': 1}, expected_revision=0)
    budget.mark_sent(attempt.id)
    budget.reconcile()
    with unit_of_work.transaction() as tx:
        current = tx.work.get_review(review.id)
        assert current.reserved == 8 and current.state == 'unknown'
    with pytest.raises(WorkStateError):
        budget.reserve(review.id, 1, price={'schema_version': 1}, expected_revision=current.revision)


@pytest.mark.parametrize('unknown', [False, True])
def test_sent_or_unknown_attempt_cannot_release_unsent_reservation(unit_of_work, unknown):
    budget, review = setup_budget(unit_of_work)
    attempt = budget.reserve(review.id, 8, price={'schema_version': 1}, expected_revision=0)
    budget.mark_sent(attempt.id)
    if unknown:
        budget.settle(attempt.id, None)
    with pytest.raises(WorkStateError):
        with unit_of_work.transaction() as tx:
            tx.work.cancel_review_unsent(attempt.id)
    with unit_of_work.transaction() as tx:
        assert tx.work.get_attempt(attempt.id).state == 'sent'
        assert tx.work.get_reservation(attempt.id).state == ('unknown' if unknown else 'reserved')
        assert tx.work.get_review(review.id).reserved == 8


def test_actual_over_estimate_stops(unit_of_work):
    budget, review = setup_budget(unit_of_work)
    attempt = budget.reserve(review.id, 3, price={'schema_version': 1}, expected_revision=0)
    budget.mark_sent(attempt.id)
    budget.settle(attempt.id, 5)
    budget.settle(attempt.id, 5)
    with unit_of_work.transaction() as tx:
        current = tx.work.get_review(review.id)
        assert current.spent == 5 and current.reserved == 0 and current.state == 'limit'


def test_crash_after_diagnostic_rejection_keeps_unknown_reserve(unit_of_work):
    budget, review = setup_budget(unit_of_work)
    attempt = budget.reserve(review.id, 8, price={'schema_version': 1}, expected_revision=0)
    budget.mark_sent(attempt.id)
    with unit_of_work.transaction() as tx:
        tx.work.transition_attempt(attempt.id, 'rejected', expected_state='sent')
    budget.reconcile()
    budget.reconcile()
    with unit_of_work.transaction() as tx:
        current = tx.work.get_review(review.id)
        assert current.state == 'unknown' and current.reserved == 8
        assert tx.work.get_reservation(attempt.id).state == 'unknown'
        assert tx.work.get_attempt(attempt.id).state == 'rejected'
    # Later independently confirmed usage settles exactly once.
    budget.settle(attempt.id, 4)
    budget.settle(attempt.id, 4)
    with unit_of_work.transaction() as tx:
        assert tx.work.get_review(review.id).spent == 4
        assert tx.work.get_review(review.id).reserved == 0


@pytest.mark.parametrize('actual,state', [(None, 'unknown'), (9, 'limit')])
def test_terminal_budget_state_cannot_restart_from_delayed_handler(unit_of_work, actual, state):
    budget, review = setup_budget(unit_of_work)
    attempt = budget.reserve(review.id, 8, price={'schema_version': 1}, expected_revision=0)
    budget.mark_sent(attempt.id)
    budget.settle(attempt.id, actual)
    with pytest.raises(WorkStateError):
        with unit_of_work.transaction() as tx:
            tx.work.update_review(review.id, state='running')
    with unit_of_work.transaction() as tx:
        assert tx.work.get_review(review.id).state == state
