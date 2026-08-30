"""Повторная генерация обработки одного блока (PRD § 13, § 19.3, § 21.5, ALG-04).

Пользователь запрашивает её, увидев неудачный результат конкретного блока: точечная
ошибка исправляется здесь, а неудачное решение, влияющее на всю книгу, — возвратом
к проверке глоссария (`reset_processing`).

Исходная группа блоков не пересчитывается. Модель получает отдельный запрос: сам блок
с его режимом из сохранённого плана, выжимку анализа главы, снимок глоссария запуска
и несколько соседних абзацев как контекст сцены. Соседние абзацы модель видит, но не
обрабатывает: псевдонимов у них нет, и результат по ним отвергается как чужой блок.
Заменяется только результат выбранного блока — его перевод, словарные подсказки
и `Language Notes`; соседние блоки остаются нетронутыми (PRD § 19.3).

Порядок и проверки те же, что у обработки главы (ALG-04): контекст, вызов модели
с повторами по политике, блокирующие проверки, короткая транзакция с проверкой активного
запуска. Общая часть живёт в `application/block_processing.py`.

Область задачи очереди — блок, а не глава: повторная генерация разных блоков идёт
независимо, и ключ по главе склеил бы такие запросы в одну задачу. Повторный запрос по
тому же блоку снимает свою прежнюю завершённую задачу и ставит новую — иначе идемпотентный
ключ навсегда запомнил бы первый результат (PRD § 18.1).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from uuid import uuid4

from linguary.application.block_processing import (
    BlockProcessingError,
    Sleep,
    TranslationCall,
    book_record,
    chapter_number,
    complete_processing,
    note_id_factory,
)
from linguary.application.context_builder import ContextBuilder, is_textual
from linguary.application.ports.llm import LLMClient
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.application.retry_policy import RetryPolicy
from linguary.application.task_runner import TaskQueue
from linguary.application.validators import TranslationResultValidator
from linguary.domain import (
    BookEvent,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    DocumentBlock,
    DocumentSection,
    LLMCallStage,
    ProcessingRun,
    ProcessingTask,
    TaskIdentity,
    TaskStatus,
    TaskType,
    TokenUsage,
    TranslationUnit,
    apply_book_event,
)

#: Задачи, после которых книга возвращается в `READY_TO_EXPORT`: обработка глав и повторная
#: генерация блоков одинаково держат книгу в `PROCESSING` (ALG-06).
_COMPLETING_TASK_TYPES = (TaskType.PROCESS_TRANSLATION_UNIT, TaskType.REGENERATE_BLOCK)

#: Состояния задачи, в которых повторный запрос новой задачи не создаёт: работа уже идёт.
_ACTIVE_TASK_STATUSES = (TaskStatus.PENDING, TaskStatus.RUNNING)


class RegenerateBlockError(BlockProcessingError):
    """Блок нельзя сгенерировать заново в текущем состоянии книги."""


class RegenerateBlockNotFoundError(RegenerateBlockError):
    """Книга или выбранный блок не найдены."""


@dataclass(frozen=True, slots=True)
class RegenerateBlockCommand:
    """Команда повторной генерации: какой блок какой книги обработать заново."""

    book_id: str
    block_id: str
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class RegenerateBlockResult:
    """Итог повторной генерации вместе с ценой вызовов.

    `saved` равен `False`, если результат отброшен: пока вызов был в пути, пользователь
    вернулся к проверке глоссария и запуска, с которым работа начиналась, больше нет.
    """

    block_id: str
    run_id: str
    saved: bool = False
    attempts: int = 0
    usage: TokenUsage | None = None


@dataclass(frozen=True, slots=True)
class _BlockWork:
    """Всё, что нужно для повторной генерации блока, прочитанное одной транзакцией."""

    document: CanonicalDocument
    section: DocumentSection
    block: DocumentBlock
    analysis: ChapterAnalysis
    run: ProcessingRun
    used_note_ids: frozenset[str]
    task_id: str | None = None

    @property
    def target_language(self) -> str:
        """Целевой язык книги."""
        return self.document.target_language

    @property
    def chapter_number(self) -> int:
        """Номер главы блока в порядке чтения (PRD § 19.3)."""
        return chapter_number(self.document, self.section)


class RegenerateBlock:
    """Обрабатывает выбранный блок заново и заменяет только его результат (PRD § 21.5)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        llm_client: LLMClient,
        retry_policy: RetryPolicy,
        validator: TranslationResultValidator,
        context_builder: ContextBuilder | None = None,
        id_factory: Callable[[], str] | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._call = TranslationCall(
            unit_of_work=unit_of_work,
            llm_client=llm_client,
            retry_policy=retry_policy,
            validator=validator,
            sleep=sleep or asyncio.sleep,
        )
        self._context_builder = context_builder or ContextBuilder()
        self._id_factory = id_factory or _new_id

    async def __call__(self, command: RegenerateBlockCommand) -> RegenerateBlockResult:
        """Один цикл ALG-04 по единственному блоку (PRD § 13, § 19.3)."""
        work = self._load(command)
        unit = TranslationUnit.of_blocks(
            id=self._id_factory(),
            blocks=(work.block,),
            plan=work.analysis.treatment_plan,
        )
        context = self._context_builder.for_block_regeneration(
            document=work.document,
            section=work.section,
            unit=unit,
            analysis=work.analysis,
            glossary=work.run.glossary_snapshot,
        )
        context = replace(
            context,
            request=context.request.with_logging(
                book_id=work.document.id,
                stage=LLMCallStage.BLOCK_REGENERATION,
                task_id=work.task_id,
            ),
        )
        call = await self._call.request(
            context,
            unit=unit,
            target_language=work.target_language,
            note_ids=note_id_factory(
                chapter_number=work.chapter_number, used=work.used_note_ids
            ),
        )
        saved = self._call.save(
            book_id=work.document.id, run_id=work.run.id, results=call.results
        )
        return RegenerateBlockResult(
            block_id=work.block.id,
            run_id=work.run.id,
            saved=saved,
            attempts=call.attempts,
            usage=call.usage,
        )

    def ensure_regeneratable(self, command: RegenerateBlockCommand) -> None:
        """Проверяет запрос до постановки фоновой задачи."""
        self._load(command)

    def _load(self, command: RegenerateBlockCommand) -> _BlockWork:
        """Читает книгу, блок, анализ его главы и активный запуск одной транзакцией."""
        with self._unit_of_work.transaction() as transaction:
            document = transaction.documents.get(command.book_id)
            if document is None:
                raise RegenerateBlockNotFoundError(
                    f"книги {command.book_id} нет в хранилище"
                )
            block = _block_of(document, command)
            section = document.section(block.section_id)
            analysis = transaction.analyses.get_for_section(
                command.book_id, section.id
            )
            if analysis is None:
                raise RegenerateBlockError(
                    f"глава {section.id} не проанализирована: обрабатывать её блоки нельзя"
                )
            analysis.ensure_ready_for_processing()
            _ensure_planned_as_text(analysis, block)
            run = transaction.runs.active_for_book(command.book_id)
            if run is None:
                raise RegenerateBlockError(
                    f"у книги {command.book_id} нет активного запуска обработки"
                )
            stored = transaction.results.list_for_run(run.id)
        section_block_ids = {item.id for item in section.ordered_blocks}
        return _BlockWork(
            document=document,
            section=section,
            block=block,
            analysis=analysis,
            run=run,
            used_note_ids=frozenset(
                note.id
                for result in stored
                for note in result.language_notes
                if result.block_id in section_block_ids
            ),
            task_id=command.task_id,
        )


class BlockRegenerationSequence:
    """Ставит задачи повторной генерации и возвращает книгу к экспорту (PRD § 18, § 21.5)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        queue: TaskQueue,
        regenerate_block: RegenerateBlock,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._queue = queue
        self._regenerate_block = regenerate_block

    async def start(self, book_id: str, block_id: str) -> ProcessingTask:
        """Запрос пользователя на повторную генерацию блока (ALG-02, ALG-06).

        Пока задача этого блока запланирована или выполняется, повторный запрос новой
        задачи не создаёт: пользователь нажал кнопку дважды. Завершённая задача снимается,
        иначе её идемпотентный ключ навсегда запомнил бы первый результат и второй раз
        блок сгенерировать было бы нельзя.
        """
        self._regenerate_block.ensure_regeneratable(
            RegenerateBlockCommand(book_id=book_id, block_id=block_id)
        )
        identity = TaskIdentity(
            book_id=book_id, task_type=TaskType.REGENERATE_BLOCK, scope_id=block_id
        )
        with self._unit_of_work.transaction() as transaction:
            book = book_record(transaction, book_id)
            existing = transaction.tasks.find_by_identity(identity)
            if existing is not None and existing.status in _ACTIVE_TASK_STATUSES:
                return existing
            status = _status_after_request(transaction, book)
            if existing is not None:
                transaction.tasks.delete(existing.id)
            if status is not book.status:
                transaction.books.save(
                    replace(book, status=status, export_validation_status=None)
                )
        return await self._queue.enqueue(identity)

    async def run_task(self, task: ProcessingTask) -> None:
        """Обработчик задачи `REGENERATE_BLOCK` для локального runner-а (ALG-02)."""
        if task.task_type is not TaskType.REGENERATE_BLOCK:
            raise RegenerateBlockError(
                f"задача {task.id} типа {task.task_type} не является повторной генерацией"
            )
        await self._regenerate_block(
            RegenerateBlockCommand(
                book_id=task.book_id,
                block_id=task.scope_id,
                task_id=task.id,
            )
        )
        complete_processing(
            self._unit_of_work,
            book_id=task.book_id,
            finished_task_id=task.id,
            task_types=_COMPLETING_TASK_TYPES,
        )


def _status_after_request(transaction: UnitOfWork, book: BookRecord) -> BookStatus:
    """Состояние книги после запроса повторной генерации (ALG-06).

    Книга уже в `PROCESSING` только тогда, когда идёт другая повторная генерация: сама
    обработка к этому моменту закончена, иначе перехода в `PROCESSING` не было бы. Такой
    запрос состояние не меняет — менять его не на что. Повторная генерация посреди
    обработки книги отклоняется: у блока ещё нет результата, который стоило бы заменять.
    """
    if book.status is not BookStatus.PROCESSING:
        return apply_book_event(book.status, BookEvent.BLOCK_REGENERATION_REQUESTED)
    unfinished = tuple(
        task
        for task in transaction.tasks.list_for_book(book.id)
        if task.task_type is TaskType.PROCESS_TRANSLATION_UNIT
        and task.status is not TaskStatus.SUCCEEDED
    )
    if unfinished:
        raise RegenerateBlockError(
            f"книга {book.id} ещё обрабатывается: повторная генерация блока доступна "
            f"после завершения обработки"
        )
    return book.status


def _block_of(
    document: CanonicalDocument, command: RegenerateBlockCommand
) -> DocumentBlock:
    """Блок книги, который требует команда.

    Блок, который модели не передаётся — перенесённая как есть конструкция, иллюстрация,
    пустой блок, — результата обработки не имеет, и генерировать заново нечего (PRD § 10.1).
    """
    block = next(
        (item for item in document.blocks() if item.id == command.block_id), None
    )
    if block is None:
        raise RegenerateBlockNotFoundError(
            f"блока {command.block_id} нет в книге {command.book_id}"
        )
    if not is_textual(block):
        raise RegenerateBlockError(
            f"блок {block.id} не является текстом для чтения: "
            f"обрабатывать его модель не должна"
        )
    return block


def _ensure_planned_as_text(analysis: ChapterAnalysis, block: DocumentBlock) -> None:
    """Проверяет, что план обработки главы отдаёт этот блок модели (PRD § 14)."""
    if block.id not in analysis.treatment_plan.textual_block_ids:
        raise RegenerateBlockError(
            f"блок {block.id} не входит в план обработки главы {analysis.section_id} "
            f"как текст для модели"
        )


def _new_id() -> str:
    """Идентификатор группы из одного блока: он живёт ровно один вызов."""
    return str(uuid4())


__all__ = [
    "BlockRegenerationSequence",
    "RegenerateBlock",
    "RegenerateBlockCommand",
    "RegenerateBlockError",
    "RegenerateBlockNotFoundError",
    "RegenerateBlockResult",
]
