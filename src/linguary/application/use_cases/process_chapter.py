"""Сценарий обработки главы и последовательность задач обработки (ALG-04, PRD § 14-17).

`ProcessChapter` делит главу на группы блоков (`translation_units`), обрабатывает каждую
группу отдельным вызовом модели и сохраняет её результат атомарно. Единица вычисления —
группа, единица хранения — блок: проверки отбраковывают всю группу целиком, поэтому
частично сохранённой группы не существует (PRD § 17).

Порядок одной группы задан ALG-04: собрать контекст, вызвать модель, вернуть ответ
с псевдонимов на идентификаторы блоков, прогнать блокирующие проверки и предупреждения
(`validators`), затем в короткой транзакции убедиться, что запуск книги всё ещё активен
и это тот же запуск, и заменить результаты блоков группы. Проверка запуска в транзакции
и отбрасывает поздний результат: пользователь мог вернуться к проверке глоссария, пока
вызов модели был в пути (P3.T7).

Задача очереди — глава целиком, а не отдельная группа: `TranslationUnit` не хранится,
её идентификатор живёт один вызов, и идемпотентный ключ задачи по нему не построить.
Повтор задачи главы безопасен: уже сохранённые группы пропускаются, а сохранение
оставшихся заменяет прежние результаты тех же блоков (PRD § 18.1).

`ChapterProcessingSequence` ставит задачи глав и переводит книгу в `READY_TO_EXPORT`,
когда обработанными оказываются все главы. Запуск обработки при этом остаётся активным:
полный экспорт ищет результаты через `ProcessingRunRepository.active_for_book`, и закрытый
запуск оставил бы книгу без переводов. Закрывает запуск только возврат к проверке
глоссария, удаляя его вместе с результатами (P3.T7).
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
    chapter_number,
    complete_processing,
    note_id_factory,
    total_usage,
)
from linguary.application.context_builder import ContextBuilder, textual_blocks
from linguary.application.ports.llm import LLMClient
from linguary.application.ports.persistence import UnitOfWork
from linguary.application.retry_policy import RetryPolicy
from linguary.application.task_runner import TaskQueue
from linguary.application.translation_units import (
    DEFAULT_MAX_UNIT_SOURCE_CHARS,
    plan_translation_units,
)
from linguary.application.validators import TranslationResultValidator
from linguary.domain import (
    CanonicalDocument,
    ChapterAnalysis,
    DocumentSection,
    GlossaryCandidate,
    LLMCallStage,
    ProcessingRun,
    ProcessingTask,
    TaskIdentity,
    TaskType,
    TokenUsage,
    TranslationUnit,
)

#: Фабрика идентификаторов групп блоков.
type IdFactory = Callable[[], str]


class ProcessChapterError(BlockProcessingError):
    """Главу нельзя обработать в её текущем состоянии."""


@dataclass(frozen=True, slots=True)
class ProcessChapterCommand:
    """Команда обработки: какую главу какой книги обрабатывать."""

    book_id: str
    section_id: str
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessChapterResult:
    """Итог обработки главы вместе с техническими данными вызовов.

    `units` — на сколько групп разделилась глава, `processed_units` — сколько из них
    обработано этим запуском задачи: повтор пропускает группы, результат которых уже
    сохранён. `attempts` и `usage` суммируют обращения к модели по всем группам главы:
    одна задача обработки стоит нескольких вызовов (PRD § 24).
    """

    section_id: str
    run_id: str
    units: int = 0
    processed_units: int = 0
    saved_blocks: int = 0
    attempts: int = 0
    usage: TokenUsage | None = None
    discarded: bool = False


@dataclass(frozen=True, slots=True)
class _ChapterWork:
    """Всё, что нужно для обработки главы, прочитанное одной транзакцией."""

    document: CanonicalDocument
    section: DocumentSection
    analysis: ChapterAnalysis
    run: ProcessingRun
    book_candidates: tuple[GlossaryCandidate, ...]
    processed_block_ids: frozenset[str]
    used_note_ids: frozenset[str]
    task_id: str | None = None

    @property
    def target_language(self) -> str:
        """Целевой язык книги."""
        return self.document.target_language

    @property
    def chapter_number(self) -> int:
        """Номер главы в порядке чтения: он входит в идентификаторы заметок (PRD § 19.3)."""
        return chapter_number(self.document, self.section)


class ProcessChapter:
    """Обрабатывает главу группами блоков и сохраняет результат каждой группы (ALG-04)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        llm_client: LLMClient,
        retry_policy: RetryPolicy,
        validator: TranslationResultValidator,
        max_unit_source_chars: int = DEFAULT_MAX_UNIT_SOURCE_CHARS,
        context_builder: ContextBuilder | None = None,
        id_factory: IdFactory | None = None,
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
        self._max_unit_source_chars = max_unit_source_chars
        self._context_builder = context_builder or ContextBuilder()
        self._id_factory = id_factory or _new_id

    async def __call__(self, command: ProcessChapterCommand) -> ProcessChapterResult:
        """Обрабатывает все группы блоков главы (PRD § 10, § 14).

        Глава без текстовых блоков — обложка, страница с иллюстрацией — обрабатывается
        без обращения к модели: групп у неё нет, сохранять нечего.
        """
        work = self._load(command)
        units = plan_translation_units(
            work.section.ordered_blocks,
            work.analysis.treatment_plan,
            id_factory=self._id_factory,
            max_source_chars=self._max_unit_source_chars,
        )
        note_ids = note_id_factory(
            chapter_number=work.chapter_number, used=work.used_note_ids
        )
        result = ProcessChapterResult(
            section_id=work.section.id, run_id=work.run.id, units=len(units)
        )
        for unit in units:
            if self._is_processed(unit, work=work):
                continue
            result = await self._process_unit(unit, work=work, note_ids=note_ids, result=result)
            if result.discarded:
                break
        return result

    async def _process_unit(
        self,
        unit: TranslationUnit,
        *,
        work: _ChapterWork,
        note_ids: Callable[[], str],
        result: ProcessChapterResult,
    ) -> ProcessChapterResult:
        """Один цикл ALG-04: контекст, вызов модели, проверки, атомарное сохранение."""
        context = self._context_builder.for_translation_unit(
            document=work.document,
            section=work.section,
            unit=unit,
            analysis=work.analysis,
            glossary=work.run.glossary_snapshot,
            book_candidates=work.book_candidates,
        )
        context = replace(
            context,
            request=context.request.with_logging(
                book_id=work.document.id,
                stage=LLMCallStage.TRANSLATION_UNIT,
                task_id=work.task_id,
            ),
        )
        call = await self._call.request(
            context, unit=unit, target_language=work.target_language, note_ids=note_ids
        )
        saved = self._call.save(
            book_id=work.document.id, run_id=work.run.id, results=call.results
        )
        return replace(
            result,
            processed_units=result.processed_units + 1,
            saved_blocks=result.saved_blocks + (len(call.results) if saved else 0),
            attempts=result.attempts + call.attempts,
            usage=total_usage((result.usage, call.usage)),
            discarded=not saved,
        )

    def _load(self, command: ProcessChapterCommand) -> _ChapterWork:
        """Читает книгу, анализ главы и активный запуск одной транзакцией."""
        with self._unit_of_work.transaction() as transaction:
            document = transaction.documents.get(command.book_id)
            if document is None:
                raise ProcessChapterError(f"книги {command.book_id} нет в хранилище")
            section = _section_of(document, command)
            analysis = transaction.analyses.get_for_section(
                command.book_id, command.section_id
            )
            if analysis is None:
                raise ProcessChapterError(
                    f"глава {command.section_id} не проанализирована: "
                    f"обрабатывать её нельзя"
                )
            analysis.ensure_ready_for_processing()
            run = transaction.runs.active_for_book(command.book_id)
            if run is None:
                raise ProcessChapterError(
                    f"у книги {command.book_id} нет активного запуска обработки"
                )
            stored = transaction.results.list_for_run(run.id)
            book_candidates = tuple(
                candidate
                for chapter in transaction.analyses.list_for_book(command.book_id)
                for candidate in chapter.glossary_candidates
            )
        section_block_ids = {block.id for block in section.ordered_blocks}
        return _ChapterWork(
            document=document,
            section=section,
            analysis=analysis,
            run=run,
            book_candidates=book_candidates,
            processed_block_ids=frozenset(
                result.block_id
                for result in stored
                if result.block_id in section_block_ids
            ),
            used_note_ids=frozenset(
                note.id
                for result in stored
                for note in result.language_notes
                if result.block_id in section_block_ids
            ),
            task_id=command.task_id,
        )

    def _is_processed(self, unit: TranslationUnit, *, work: _ChapterWork) -> bool:
        """Сохранён ли результат этой группы целиком (PRD § 18.1).

        Повтор задачи не платит за уже выполненную работу. Группа считается выполненной
        только целиком: частично сохранённой группы не бывает, а совпадение по части
        блоков означало бы другую разбивку главы — её нужно обработать заново.
        """
        return set(unit.block_ids) <= work.processed_block_ids


class ChapterProcessingSequence:
    """Ставит задачи обработки глав и завершает обработку книги (PRD § 14, § 18)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        queue: TaskQueue,
        process_chapter: ProcessChapter,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._queue = queue
        self._process_chapter = process_chapter

    async def start(self, book_id: str) -> tuple[ProcessingTask, ...]:
        """Ставит задачи обработки всех глав книги с текстом.

        Главы обрабатываются независимо, поэтому задачи ставятся сразу все: порядок
        обработки на результат не влияет, а параллелизм воркера остаётся полезным
        (PRD § 25). Глава без текстовых блоков задачи не получает — обрабатывать нечего.
        Повторный запуск задач не удваивает: постановка идемпотентна по ключу
        `book_id + task_type + scope_id` (ALG-02).
        """
        section_ids = self._section_ids_with_text(book_id)
        tasks = [
            await self._queue.enqueue(
                TaskIdentity(
                    book_id=book_id,
                    task_type=TaskType.PROCESS_TRANSLATION_UNIT,
                    scope_id=section_id,
                )
            )
            for section_id in section_ids
        ]
        if not tasks:
            self._complete_processing(book_id, finished_task_id=None)
        return tuple(tasks)

    async def run_task(self, task: ProcessingTask) -> None:
        """Обработчик задачи `PROCESS_TRANSLATION_UNIT` для локального runner-а (ALG-02)."""
        if task.task_type is not TaskType.PROCESS_TRANSLATION_UNIT:
            raise ProcessChapterError(
                f"задача {task.id} типа {task.task_type} не является обработкой главы"
            )
        await self._process_chapter(
            ProcessChapterCommand(
                book_id=task.book_id,
                section_id=task.scope_id,
                task_id=task.id,
            )
        )
        self._complete_processing(task.book_id, finished_task_id=task.id)

    def _section_ids_with_text(self, book_id: str) -> tuple[str, ...]:
        """Главы книги, у которых есть текстовые блоки, в порядке чтения."""
        with self._unit_of_work.transaction() as transaction:
            document = transaction.documents.get(book_id)
            if document is None:
                raise ProcessChapterError(f"книги {book_id} нет в хранилище")
            if transaction.runs.active_for_book(book_id) is None:
                raise ProcessChapterError(
                    f"у книги {book_id} нет активного запуска обработки: "
                    f"сначала завершите проверку глоссария"
                )
        return tuple(
            section.id
            for section in document.ordered_sections
            if textual_blocks(section)
        )

    def _complete_processing(self, book_id: str, *, finished_task_id: str | None) -> None:
        """Переводит книгу в `READY_TO_EXPORT`, когда обработаны все главы (ALG-06)."""
        complete_processing(
            self._unit_of_work,
            book_id=book_id,
            finished_task_id=finished_task_id,
            task_types=(TaskType.PROCESS_TRANSLATION_UNIT,),
        )


def _section_of(
    document: CanonicalDocument, command: ProcessChapterCommand
) -> DocumentSection:
    """Глава книги, которую требует команда."""
    for section in document.sections:
        if section.id == command.section_id:
            return section
    raise ProcessChapterError(
        f"главы {command.section_id} нет в книге {command.book_id}"
    )


def _new_id() -> str:
    """Идентификатор новой группы блоков."""
    return str(uuid4())


__all__ = [
    "ChapterProcessingSequence",
    "IdFactory",
    "ProcessChapter",
    "ProcessChapterCommand",
    "ProcessChapterError",
    "ProcessChapterResult",
    "Sleep",
]
