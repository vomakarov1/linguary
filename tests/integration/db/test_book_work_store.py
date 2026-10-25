from datetime import UTC, datetime
from uuid import uuid4
import pytest
from sqlalchemy import text
from linguary.domain.book_conditions import BookConditions, WorkStateError
from linguary.domain.recovery import RecoveryRoot, RecoveryScope, CallAttempt
from linguary.domain.quality_review import QualityReview, QualityReservation
from persistence_factories import make_book, make_document, make_run, make_translated_result, make_task


def uid():
    return str(uuid4())


def prepare(uow):
    with uow.transaction():
        uow.books.add(make_book())
        uow.documents.save(make_document())
        uow.runs.add(make_run())
        conditions = BookConditions(uid(), 'book-1', 1, {'schema_version': 1}, datetime.now(UTC))
        uow.work.add_conditions(conditions)
        root = RecoveryRoot(uid(), 'book-1', conditions.id, uid(), {'schema_version': 1})
        scope = RecoveryScope(uid(), root.id, ('block-1',))
        uow.work.add_root(root)
        uow.work.add_scope(scope)
    return conditions, root, scope


def test_concurrent_claim_has_one_winner(unit_of_work):
    _, _, scope = prepare(unit_of_work)
    with unit_of_work.transaction():
        claimed = unit_of_work.work.claim_scope(scope.id, expected_generation=0)
        assert claimed.generation == 1
    with unit_of_work.transaction(), pytest.raises(WorkStateError):
        unit_of_work.work.claim_scope(scope.id, expected_generation=0)


def test_accept_scope_rolls_back_with_results(unit_of_work):
    conditions, root, scope = prepare(unit_of_work)
    attempt = CallAttempt(uid(), 'book-1', state='validated', conditions_id=conditions.id, root_id=root.id, scope_id=scope.id, run_id='run-1', generation=1)
    with unit_of_work.transaction():
        unit_of_work.work.claim_scope(scope.id, expected_generation=0)
        unit_of_work.work.add_attempt(attempt)
    with pytest.raises(RuntimeError), unit_of_work.transaction():
        unit_of_work.work.accept_scope(scope.id, attempt_id=attempt.id, expected_generation=1, results=[make_translated_result()])
        raise RuntimeError('abort')
    with unit_of_work.transaction():
        assert unit_of_work.results.list_for_run('run-1') == ()
        assert unit_of_work.work.get_scope(scope.id).status == 'claimed'
        unit_of_work.work.accept_scope(scope.id, attempt_id=attempt.id, expected_generation=1, results=[make_translated_result()])
        assert unit_of_work.work.get_binding('result', 'run-1:block-1')['attempt_id'] == attempt.id


def test_binding_survives_task_deletion(unit_of_work):
    conditions, _, _ = prepare(unit_of_work)
    with unit_of_work.transaction():
        unit_of_work.tasks.add(make_task())
        unit_of_work.work.bind('task', 'task-1', book_id='book-1', conditions_id=conditions.id)
        unit_of_work.tasks.delete('task-1')
        assert unit_of_work.work.get_binding('task', 'task-1')['conditions_id'] == conditions.id


def test_archives_are_append_only(unit_of_work):
    conditions, _, _ = prepare(unit_of_work)
    artifact = {'id': uid(), 'book_id': 'book-1', 'conditions_id': conditions.id, 'kind': 'analysis', 'scope_id': 'chapter', 'body': {'schema_version': 1, 'text': 'first'}}
    with unit_of_work.transaction():
        unit_of_work.work.archive_artifacts([artifact])
    with pytest.raises(WorkStateError), unit_of_work.transaction():
        unit_of_work.work.archive_artifacts([{**artifact, 'body': {'schema_version': 1, 'text': 'second'}}])
    with unit_of_work.transaction():
        assert unit_of_work.work.get_artifact(artifact['id'])['body']['text'] == 'first'


def test_old_nullable_mapping(unit_of_work):
    prepare(unit_of_work)
    with unit_of_work.transaction():
        assert unit_of_work.work.get_conditions('missing') is None
        assert unit_of_work.work.get_binding('task', 'missing') is None


def test_reserve_and_unknown_settlement(unit_of_work):
    conditions, _, _ = prepare(unit_of_work)
    review = QualityReview(uid(), 'book-1', uid(), {'schema_version': 1}, {'schema_version': 1}, 10, conditions_id=conditions.id)
    attempt = CallAttempt(uid(), 'book-1', review_id=review.id, conditions_id=conditions.id)
    with unit_of_work.transaction():
        unit_of_work.work.add_review(review)
        unit_of_work.work.add_attempt(attempt)
        unit_of_work.work.reserve(QualityReservation(attempt.id, review.id, 8), expected_revision=0)
    with pytest.raises(WorkStateError), unit_of_work.transaction():
        other = CallAttempt(uid(), 'book-1', review_id=review.id, ordinal=1, conditions_id=conditions.id)
        unit_of_work.work.add_attempt(other)
        unit_of_work.work.reserve(QualityReservation(other.id, review.id, 8), expected_revision=1)
    with unit_of_work.transaction():
        unit_of_work.work.settle(attempt.id, actual=None)
        assert unit_of_work.work.get_review(review.id).reserved == 8
        assert unit_of_work.work.get_review(review.id).state == 'unknown'
    with unit_of_work.transaction():
        unit_of_work.work.settle(attempt.id, actual=3)
        assert unit_of_work.work.get_review(review.id).reserved == 0
        assert unit_of_work.work.get_review(review.id).spent == 3


def test_parallel_scope_claim_has_one_winner(unit_of_work):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    _, _, scope = prepare(unit_of_work)
    barrier = Barrier(2)

    def claim():
        barrier.wait(timeout=5)
        try:
            with unit_of_work.transaction():
                unit_of_work.work.claim_scope(scope.id, expected_generation=0)
            return True
        except WorkStateError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: claim(), range(2))) == [False, True]


def test_stale_attempt_and_missing_coverage_cannot_write(unit_of_work):
    conditions, root, scope = prepare(unit_of_work)
    with unit_of_work.transaction():
        unit_of_work.work.claim_scope(scope.id, expected_generation=0)
        attempt = CallAttempt(uid(), 'book-1', state='validated', conditions_id=conditions.id,
                              root_id=root.id, scope_id=scope.id, run_id='run-1', generation=1)
        unit_of_work.work.add_attempt(attempt)
    with pytest.raises(WorkStateError), unit_of_work.transaction():
        unit_of_work.work.accept_scope(scope.id, attempt_id=attempt.id, expected_generation=1, results=[])
    with unit_of_work.transaction():
        unit_of_work.work.add_conditions(BookConditions(uid(), 'book-1', 2, {'schema_version': 1}, datetime.now(UTC)))
    with pytest.raises(WorkStateError), unit_of_work.transaction():
        unit_of_work.work.accept_scope(scope.id, attempt_id=attempt.id, expected_generation=1, results=[make_translated_result()])
    with unit_of_work.transaction():
        assert unit_of_work.results.list_for_run('run-1') == ()


def test_purge_diagnostics_keeps_metadata(unit_of_work):
    from persistence_factories import make_llm_call
    prepare(unit_of_work)
    with unit_of_work.transaction():
        unit_of_work.llm_calls.add(make_llm_call(), response_text='sensitive')
        assert unit_of_work.work.purge_diagnostics('other') == 0
        assert unit_of_work.work.purge_diagnostics('book-1', call_id='call-1') == 1
        assert unit_of_work.work.purge_diagnostics('book-1', call_id='call-1') == 0
        assert unit_of_work.llm_calls.response_text_for('call-1') is None
        assert len(unit_of_work.llm_calls.list_for_book('book-1')) == 1
