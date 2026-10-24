"""Tracking active queue executors: identifying other workers operating on the same database.

The queue allows multiple executors: background Web UI workers and `linguary worker`
atomically acquire tasks without conflict. The only point of contention is crash
recovery (ALG-02): at startup, `running` tasks are declared interrupted, which is valid only
when no one is actively executing them. A running task belonging to another live executor is actively
in progress, and marking it `failed` would let the user retry it while still running,
submitting the same `TranslationUnit` to the model twice.

Therefore, every executor holds a shared lock on a file next to the database for its lifetime,
and recovery is executed only by the one that obtained an exclusive lock at startup:
only an executor running with no peers acquires it. The lock is managed by the OS, and a crashed
process releases it automatically, so markers never become stale or require heartbeats.
Tasks belonging to a process that crashed alongside a live executor remain `running`
until the next isolated startup — delayed recovery is preferable to killing someone else's work.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path


class ExecutorLock:
    """Tracks a single queue executor's participation in the database executor lock."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._descriptor: int | None = None

    def join(self) -> bool:
        """Registers the executor and reports whether it is operating alone.

        `True` indicates no other executors are active and `running` tasks are abandoned.
        In this case the executor retains an exclusive lock until calling `share`:
        recovery is performed under this lock, preventing a subsequently started executor
        from grabbing a task that recovery would then cancel.
        `False` indicates peer executors are active, immediately downgrading to a shared lock.
        """
        if self._descriptor is not None:
            raise RuntimeError("исполнитель уже отмечен")
        descriptor = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            alone = _try_exclusive(descriptor)
            if not alone:
                fcntl.flock(descriptor, fcntl.LOCK_SH)
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return alone

    def share(self) -> None:
        """Allows peer executors to run alongside while preserving own registration.

        A shared lock only blocks on an exclusive lock held during startup recovery,
        keeping wait times negligible.
        """
        if self._descriptor is None:
            raise RuntimeError("исполнитель не отмечен")
        fcntl.flock(self._descriptor, fcntl.LOCK_SH)

    def leave(self) -> None:
        """Releases the executor registration; idempotent."""
        if self._descriptor is None:
            return
        descriptor, self._descriptor = self._descriptor, None
        os.close(descriptor)


def _try_exclusive(descriptor: int) -> bool:
    """Acquires an exclusive lock without blocking if no other executors are present."""
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


__all__ = ["ExecutorLock"]
