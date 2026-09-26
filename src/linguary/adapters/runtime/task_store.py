"""Очередь локальных задач в SQLite: постановка, захват и фиксация итога (ALG-02).

Здесь живут блокирующие операции с базой. Захват задачи — единственное место приложения,
где транзакция открывается явным `BEGIN IMMEDIATE`: писатель берёт блокировку записи сразу,
до чтения кандидата, поэтому два воркера не читают одну и ту же строку как свободную.
Условное обновление с проверкой числа изменённых строк остаётся вторым рубежом: даже если
транзакцию когда-нибудь откроют отложенной, захваченная кем-то задача не будет захвачена
повторно.

Переходы состояния задачи выполняет домен: адаптер отвечает за атомарность, а не за правила.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from sqlalchemy import Engine, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from linguary.adapters.db import mapping
from linguary.adapters.db.engine import create_session_factory
from linguary.adapters.db.models import ProcessingTaskRow
from linguary.adapters.db.repositories import SqlAlchemyTaskRepository
from linguary.domain import ProcessingTask, TaskIdentity, TaskStatus

#: Явное начало транзакции записи. Отложенная транзакция SQLite берёт блокировку только
#: на первой записи, и повышение блокировки после чтения кандидата обрывается ошибкой
#: вместо ожидания, поэтому захват начинается сразу с режима записи.
_BEGIN_IMMEDIATE = "BEGIN IMMEDIATE"


class SqliteTaskStore:
    """Блокирующие операции очереди задач поверх SQLite."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._id_factory = id_factory or _new_task_id

    @classmethod
    def for_engine(
        cls, engine: Engine, *, id_factory: Callable[[], str] | None = None
    ) -> SqliteTaskStore:
        """Очередь с фабрикой сессий по умолчанию для готового движка."""
        return cls(create_session_factory(engine), id_factory=id_factory)

    def enqueue(self, identity: TaskIdentity) -> ProcessingTask:
        """Ставит задачу в очередь; повторная постановка того же ключа дубликата не создаёт."""
        existing = self._find(identity)
        if existing is not None:
            return existing
        task = ProcessingTask(
            id=self._id_factory(),
            book_id=identity.book_id,
            task_type=identity.task_type,
            scope_id=identity.scope_id,
        )
        try:
            with self._session_factory.begin() as session:
                SqlAlchemyTaskRepository(session).add(task)
        except IntegrityError:
            # Ключ занят другим писателем между проверкой и вставкой: очередь идемпотентна
            # по ключу, поэтому возвращается уже поставленная задача.
            stored = self._find(identity)
            if stored is None:
                raise
            return stored
        return task

    def claim_next_pending(self) -> ProcessingTask | None:
        """Захватывает очередную задачу `pending`, переводя её в `running` (ALG-02)."""
        session = self._session_factory()
        try:
            session.connection().exec_driver_sql(_BEGIN_IMMEDIATE)
            candidate = self._next_pending(session)
            if candidate is None:
                session.commit()
                return None
            claimed = candidate.start()
            updated = session.execute(
                update(ProcessingTaskRow)
                .where(
                    ProcessingTaskRow.id == claimed.id,
                    ProcessingTaskRow.status == TaskStatus.PENDING.value,
                )
                .values(status=claimed.status.value, attempts=claimed.attempts)
            )
            session.commit()
            return claimed if updated.rowcount == 1 else None
        finally:
            session.close()

    def save(self, task: ProcessingTask) -> None:
        """Сохраняет новое состояние задачи; отменённую задачу не воскрешает.

        Возврат к проверке глоссария удаляет задачи отменённого запуска, в том числе ту,
        которая выполняется прямо сейчас (PRD § 21.5). Её итог фиксировать уже некуда:
        отсутствие строки здесь — нормальный конец отменённой работы, а не сбой очереди,
        и воркер из-за него останавливаться не должен.
        """
        with self._session_factory.begin() as session:
            repository = SqlAlchemyTaskRepository(session)
            if repository.get(task.id) is None:
                return
            repository.save(task)

    def list_running(self) -> tuple[ProcessingTask, ...]:
        """Задачи в состоянии `running` в порядке их создания."""
        with self._session_factory() as session:
            return self._list_by_status(session, TaskStatus.RUNNING)

    def _find(self, identity: TaskIdentity) -> ProcessingTask | None:
        """Задача по идемпотентному ключу."""
        with self._session_factory() as session:
            return SqlAlchemyTaskRepository(session).find_by_identity(identity)

    def _next_pending(self, session: Session) -> ProcessingTask | None:
        """Кандидат на захват: самая ранняя задача `pending`.

        Идентификатор в сортировке — устойчивый разрыв ничьей: отметки создания задач,
        поставленных одной операцией, совпадают с точностью до записи в базу.
        """
        row = session.scalars(
            select(ProcessingTaskRow)
            .where(ProcessingTaskRow.status == TaskStatus.PENDING.value)
            .order_by(ProcessingTaskRow.created_at, ProcessingTaskRow.id)
            .limit(1)
        ).first()
        return None if row is None else mapping.to_task(row)

    def _list_by_status(
        self, session: Session, status: TaskStatus
    ) -> tuple[ProcessingTask, ...]:
        """Задачи одного состояния в порядке их создания."""
        rows = session.scalars(
            select(ProcessingTaskRow)
            .where(ProcessingTaskRow.status == status.value)
            .order_by(ProcessingTaskRow.created_at, ProcessingTaskRow.id)
        ).all()
        return tuple(mapping.to_task(row) for row in rows)


def _new_task_id() -> str:
    """Идентификатор новой задачи."""
    return str(uuid4())


__all__ = ["SqliteTaskStore"]
