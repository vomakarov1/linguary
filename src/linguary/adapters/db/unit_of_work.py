"""Unit of work поверх сессии SQLAlchemy.

Транзакционную границу задаёт прикладной сценарий: он открывает `transaction()`, выполняет
изменения через репозитории и выходит из блока. Выход без исключения фиксирует изменения,
исключение — откатывает их целиком, поэтому частично сохранённого результата одного
`TranslationUnit` не возникает (PRD § 17).

Репозитории существуют только внутри транзакции: обращение к ним снаружи — ошибка
контракта, а не молчаливое открытие собственной транзакции.

Один экземпляр обслуживает всё приложение, поэтому активная сессия хранится не в самом
объекте, а в контексте исполнения (`ContextVar`): веб-запрос в рабочем потоке FastAPI,
фоновая задача воркера и корутина сценария работают одновременно, и каждая ведёт свою
транзакцию. Общее поле сделало бы их одной транзакцией на всех, а вторая одновременная
работа получала бы отказ «транзакция уже открыта».
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

#: Счётчик экземпляров: имя `ContextVar` служит только диагностике.
_INSTANCES = count(1)

from linguary.adapters.db.engine import create_session_factory
from linguary.adapters.db.repositories import (
    SqlAlchemyBlockResultRepository,
    SqlAlchemyBookRepository,
    SqlAlchemyChapterAnalysisRepository,
    SqlAlchemyDocumentRepository,
    SqlAlchemyGlossaryRepository,
    SqlAlchemyLLMCallLogRepository,
    SqlAlchemyProcessingRunRepository,
    SqlAlchemyTaskRepository,
)
from linguary.application.ports.persistence import PersistenceError


class SqlAlchemyUnitOfWork:
    """Транзакционная граница прикладного сценария поверх SQLite."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self._current: ContextVar[Session | None] = ContextVar(
            f"linguary_unit_of_work_{next(_INSTANCES)}", default=None
        )

    @classmethod
    def for_engine(cls, engine: Engine) -> SqlAlchemyUnitOfWork:
        """Unit of work с фабрикой сессий по умолчанию для готового движка."""
        return cls(create_session_factory(engine))

    @contextmanager
    def transaction(self) -> Iterator[SqlAlchemyUnitOfWork]:
        """Открывает транзакцию и делает репозитории доступными.

        Вложенная транзакция того же исполнения по-прежнему запрещена: она означала бы,
        что сценарий не выбрал транзакционную границу явно. Одновременная транзакция
        другого запроса или фоновой задачи вложенной не считается — у неё свой контекст.
        """
        if self._current.get() is not None:
            raise PersistenceError(
                "транзакция уже открыта: вложенные транзакции не поддерживаются"
            )
        session = self._session_factory()
        token = self._current.set(session)
        try:
            with session.begin():
                yield self
        finally:
            self._current.reset(token)
            session.close()

    @property
    def books(self) -> SqlAlchemyBookRepository:
        """Репозиторий книг."""
        return SqlAlchemyBookRepository(self._active_session())

    @property
    def documents(self) -> SqlAlchemyDocumentRepository:
        """Репозиторий канонической модели документа."""
        return SqlAlchemyDocumentRepository(self._active_session())

    @property
    def glossary(self) -> SqlAlchemyGlossaryRepository:
        """Репозиторий глоссария."""
        return SqlAlchemyGlossaryRepository(self._active_session())

    @property
    def analyses(self) -> SqlAlchemyChapterAnalysisRepository:
        """Репозиторий анализа глав."""
        return SqlAlchemyChapterAnalysisRepository(self._active_session())

    @property
    def runs(self) -> SqlAlchemyProcessingRunRepository:
        """Репозиторий запусков обработки."""
        return SqlAlchemyProcessingRunRepository(self._active_session())

    @property
    def results(self) -> SqlAlchemyBlockResultRepository:
        """Репозиторий результатов обработки блоков."""
        return SqlAlchemyBlockResultRepository(self._active_session())

    @property
    def tasks(self) -> SqlAlchemyTaskRepository:
        """Репозиторий локальных задач."""
        return SqlAlchemyTaskRepository(self._active_session())

    @property
    def llm_calls(self) -> SqlAlchemyLLMCallLogRepository:
        """Репозиторий журнала вызовов LLM."""
        return SqlAlchemyLLMCallLogRepository(self._active_session())

    def _active_session(self) -> Session:
        """Сессия транзакции, открытой в текущем исполнении."""
        session = self._current.get()
        if session is None:
            raise PersistenceError(
                "репозиторий доступен только внутри transaction(): "
                "прикладной сценарий обязан задать транзакционную границу явно"
            )
        return session


__all__ = ["SqlAlchemyUnitOfWork"]
