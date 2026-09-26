"""Отметка исполнителей очереди: кто ещё выполняет задачи той же базы (ALG-02).

Восстановление прерванных задач опирается на единственный ответ отметки — один ли
исполнитель работает с базой. Тесты закрепляют этот ответ для исполнителей, стартующих
друг за другом, и то, что ушедший исполнитель перестаёт считаться живым.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from linguary.adapters.runtime import ExecutorLock


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    """Файл отметки исполнителей теста."""
    return tmp_path / "linguary.db.executors.lock"


def test_first_executor_is_alone(lock_path: Path) -> None:
    """Исполнитель, рядом с которым никого нет, узнаёт, что он один."""
    lock = ExecutorLock(lock_path)

    try:
        assert lock.join() is True
    finally:
        lock.leave()


def test_executor_started_next_to_a_working_one_is_not_alone(lock_path: Path) -> None:
    """Второй исполнитель видит первого, и оба остаются отмеченными."""
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
    """После ухода исполнителя следующий снова стартует в одиночестве."""
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
    """Один объект отметки нельзя отметить повторно: вторая отметка повисла бы."""
    lock = ExecutorLock(lock_path)
    lock.join()
    try:
        with pytest.raises(RuntimeError, match="уже отмечен"):
            lock.join()
    finally:
        lock.leave()
