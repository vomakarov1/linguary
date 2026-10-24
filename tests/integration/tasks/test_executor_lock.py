"""Tracking queue executors: identifying other workers operating on the same database (ALG-02).

Recovery of interrupted tasks relies on a single query to the lock: whether the current
executor is alone operating on the database. Tests verify this behavior for executors starting
sequentially, and ensure that an executor that left is no longer counted as active.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from linguary.adapters.runtime import ExecutorLock


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    """Lock file for tracking test executors."""
    return tmp_path / "linguary.db.executors.lock"


def test_first_executor_is_alone(lock_path: Path) -> None:
    """An executor with no peers detects that it is alone."""
    lock = ExecutorLock(lock_path)

    try:
        assert lock.join() is True
    finally:
        lock.leave()


def test_executor_started_next_to_a_working_one_is_not_alone(lock_path: Path) -> None:
    """A second executor starting alongside an active one detects it is not alone, and both remain tracked."""
    first = ExecutorLock(lock_path)
    second = ExecutorLock(lock_path)
    try:
        assert first.join() is True
        first.share()

        assert second.join() is False
    finally:
        second.leave()
        first.leave()


def test_executor_that_left_is_no_longer_counted(lock_path: Path) -> None:
    """After an executor leaves, the subsequent executor starts alone again."""
    first = ExecutorLock(lock_path)
    first.join()
    first.share()
    first.leave()

    following = ExecutorLock(lock_path)
    try:
        assert following.join() is True
    finally:
        following.leave()


def test_joining_twice_is_rejected(lock_path: Path) -> None:
    """A single lock instance cannot join twice: the second entry would hang."""
    lock = ExecutorLock(lock_path)
    lock.join()
    try:
        with pytest.raises(RuntimeError, match="уже отмечен"):
            lock.join()
    finally:
        lock.leave()
