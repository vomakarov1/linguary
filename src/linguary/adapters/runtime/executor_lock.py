"""Отметка работающих исполнителей очереди: кто ещё выполняет задачи той же базы.

Очередь допускает несколько исполнителей: фоновый воркер Web UI и `linguary worker`
атомарно захватывают задачи и друг другу не мешают. Мешает им только восстановление после
аварии (ALG-02): при старте задачи `running` объявляются прерванными, а это верно лишь
тогда, когда их никто не исполняет. Задача другого живого исполнителя в `running` честно
выполняется, и пометить её `failed` — значит дать пользователю повторить её, пока она ещё
идёт, то есть отправить тот же `TranslationUnit` в модель дважды.

Поэтому каждый исполнитель на всё время работы держит разделяемую блокировку файла рядом
с базой, а восстановление выполняет только тот, кому при старте досталась исключительная:
её получает лишь исполнитель, рядом с которым других нет. Блокировку держит ОС, и процесс,
завершившийся аварийно, освобождает её сам, поэтому отметка не устаревает и её не нужно
продлевать. Задачи процесса, упавшего рядом с живым исполнителем, остаются `running`
до ближайшего старта в одиночестве — лучше позже, чем погасить чужую работу.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path


class ExecutorLock:
    """Участие одного исполнителя очереди в общей отметке исполнителей базы."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._descriptor: int | None = None

    def join(self) -> bool:
        """Отмечает исполнителя и сообщает, один ли он работает с базой.

        `True` означает, что других исполнителей нет и задачи `running` никто не
        выполняет. Тогда исполнитель держит исключительную блокировку, пока не вызовет
        `share`: восстановление выполняется под ней, иначе исполнитель, стартовавший
        следом, успел бы захватить задачу, и восстановление погасило бы уже её.
        `False` — рядом есть другие исполнители, и отметка сразу разделяемая.
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
        """Пускает рядом других исполнителей, сохраняя собственную отметку.

        Разделяемая блокировка ждёт только чужой исключительной, а её держат лишь на
        время восстановления при старте, поэтому ожидание короткое.
        """
        if self._descriptor is None:
            raise RuntimeError("исполнитель не отмечен")
        fcntl.flock(self._descriptor, fcntl.LOCK_SH)

    def leave(self) -> None:
        """Снимает отметку исполнителя; повторный вызов ничего не делает."""
        if self._descriptor is None:
            return
        descriptor, self._descriptor = self._descriptor, None
        os.close(descriptor)


def _try_exclusive(descriptor: int) -> bool:
    """Берёт исключительную блокировку без ожидания, если других исполнителей нет."""
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


__all__ = ["ExecutorLock"]
