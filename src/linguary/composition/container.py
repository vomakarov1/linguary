"""Состав зависимостей приложения: единственное место, где известны конкретные адаптеры.

Домен и прикладные сценарии знают только порты, поэтому кто-то должен один раз связать
их с SQLite, EbookLib, EPUBCheck и провайдером LLM, а задачи очереди — с командами
сценариев (ADR § 3). Это делает здесь `build_application`, и её результатом пользуются
Web UI, CLI и сквозной тест: второго места сборки в приложении нет.

Что собирается:

- хранилище — движок SQLite, unit of work и очередь задач поверх одной базы;
- воркер — параллелизм фоновых задач из секции `[worker]` (PRD § 25);
- клиент модели — журнал вызовов и, если задан предел частоты, общий для всех воркеров
  ограничитель запросов;
- импорт — читатель EPUB вместе с хранилищем ресурсов книги;
- экспорт — писатель EPUB и внешняя проверка EPUBCheck;
- LLM-сценарии — анализ главы, обработка главы и повторная генерация блока с порогами
  и пределами из локальной конфигурации;
- обработчики всех пяти типов задач очереди: только здесь известно, как из идемпотентного
  ключа задачи получить команду сценария и путь целевого файла.

Синхронные сценарии подключаются к асинхронному runner-у через `to_async_handler`:
разбор EPUB и работа с SQLite блокируют цикл событий, и выполнять их прямо в корутине
нельзя (PRD § 25).
"""

from __future__ import annotations

import asyncio

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from decimal import Decimal

from linguary.adapters.config import AppSettings, load_settings
from linguary.adapters.config.settings import SettingsError
from linguary.adapters.config.system_settings import overlay_settings
from linguary.adapters.db.system_settings import SystemSettingsStore
from linguary.adapters.llm.rate_limit import LLMRateLimiter
from linguary.composition.runtime_settings import RuntimeGeneration, RuntimeSettings, RuntimeSequence
from linguary.composition.system_settings import SystemSettingsService
from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.epub import (
    EbookLibEpubReader,
    EpubExportWriter,
    FilesystemEpubResourceStore,
)
from linguary.adapters.llm import LoggingLLMClient, RateLimitedLLMClient
from linguary.adapters.llm.logging import LoggedObserver
from linguary.domain.llm_log import TokenPricing
from linguary.adapters.runtime import (
    ExecutorLock,
    SqliteTaskQueue,
    TaskWorkerPool,
    WorkerSettings,
    to_async_handler,
)
from linguary.adapters.validation import EpubCheckSettings, EpubCheckValidator
from linguary.application.context_builder import ContextBuilder
from linguary.application.ports.llm import LLMClient
from linguary.application.retry_policy import RetryPolicy
from linguary.application.task_runner import TaskHandler, TaskRunner
from linguary.application.use_cases.analyze_chapter import (
    AnalyzeChapter,
    ChapterAnalysisSequence,
)
from linguary.application.use_cases.approve_glossary import (
    ApproveGlossary,
    StartProcessing,
)
from linguary.application.use_cases.export_book import (
    ExportBook,
    ExportBookCommand,
    ExportBookSequence,
    ExportMode,
)
from linguary.application.use_cases.import_book import (
    ImportBook,
    ImportBookCommand,
    ImportBookError,
)
from linguary.application.use_cases.process_chapter import (
    ChapterProcessingSequence,
    ProcessChapter,
)
from linguary.application.use_cases.regenerate_block import (
    BlockRegenerationSequence,
    RegenerateBlock,
)
from linguary.application.use_cases.register_book import RegisterBook
from linguary.application.use_cases.reset_processing import ResetProcessing
from linguary.application.use_cases.retry_task import RetryTask
from linguary.application.validators import TranslationResultValidator
from linguary.composition.llm import ConfiguredLLMClient
from linguary.composition.migrations import apply_migrations
from linguary.composition.paths import DataPaths
from linguary.domain import (
    LLMSettingsSnapshot,
    ProcessingTask,
    TaskIdentity,
    TaskType,
)

#: Сколько соседних блоков видит модель при повторной генерации (PRD § 13). Значение
#: обосновано в P3.T7 и в файл настроек намеренно не выведено.
DEFAULT_REGENERATION_CONTEXT_BLOCKS = 2


@dataclass(frozen=True, slots=True)
class Application:
    """Собранное приложение: сценарии, очередь задач и всё, что о них нужно знать.

    Экземпляр создаётся один раз на процесс. Web UI и CLI берут отсюда готовые сценарии
    и ничего не конструируют сами.
    """

    runtime_settings: RuntimeSettings
    system_settings: SystemSettingsService
    settings_store: SystemSettingsStore
    rate_limiter: LLMRateLimiter
    paths: DataPaths
    unit_of_work: SqlAlchemyUnitOfWork
    task_queue: SqliteTaskQueue
    runner: TaskRunner
    worker_pool: TaskWorkerPool
    epubcheck: EpubCheckValidator
    register_book: RegisterBook
    import_book: ImportBook
    export_book: ExportBook
    export_sequence: ExportBookSequence
    approve_glossary: ApproveGlossary
    reset_processing: ResetProcessing
    retry_task: RetryTask
    analysis: RuntimeSequence
    processing: RuntimeSequence
    regeneration: RuntimeSequence
    _dispose: Callable[[], None]
    _close_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def settings(self) -> AppSettings:
        return self.runtime_settings.current.settings

    def start_processing(self) -> StartProcessing:
        """Сценарий запуска обработки книги.

        Собирается по требованию, а не при старте приложения: он замораживает снимок
        настроек модели, а ненастроенная LLM снимка не даёт. Отказ приходит тем же
        `LLMConfigurationError`, что и попытка обратиться к модели (PRD § 23).
        """
        settings = self.settings
        settings.llm.ensure_ready()
        return StartProcessing(
            unit_of_work=self.unit_of_work,
            glossary_confidence_threshold=(
                settings.checks.glossary_confidence_threshold
            ),
            glossary_review_min_mentions=settings.checks.glossary_review_min_mentions,
            glossary_review_min_chapters=settings.checks.glossary_review_min_chapters,
            llm_settings_snapshot=_llm_snapshot(settings),
        )

    async def start_import(self, book_id: str) -> ProcessingTask:
        """Ставит задачу импорта загруженной книги (PRD § 18)."""
        return await self.task_queue.enqueue(
            TaskIdentity(book_id=book_id, task_type=TaskType.IMPORT_BOOK)
        )

    async def start_export(self, book_id: str) -> ProcessingTask:
        """Ставит задачу сборки итогового EPUB (PRD § 20)."""
        return await self.export_sequence.start(book_id)

    def begin_close(self) -> None:
        """Запрещает новые изменения настроек и новые захваты задач."""
        self.system_settings.begin_close()
        self.worker_pool.request_stop()

    async def aclose(self) -> None:
        """Дожидается принятых изменений до закрытия поколений и базы."""
        self.begin_close()
        if self._close_task is None:
            task = asyncio.create_task(self._close())
            object.__setattr__(self, "_close_task", task)
            task.add_done_callback(self._collect_close)
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        errors = []
        for close in (self.system_settings.aclose, self.runtime_settings.aclose):
            try:
                await close()
            except Exception:
                errors.append(True)
        try:
            self.dispose()
        except Exception:
            errors.append(True)
        if errors:
            raise SettingsError("Не удалось закрыть все ресурсы приложения") from None

    @staticmethod
    def _collect_close(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    def dispose(self) -> None:
        """Закрывает соединения с базой.

        Этот метод закрывает только базу. Собственные HTTP-клиенты поколений
        освобождаются асинхронным `aclose`.
        """
        self._dispose()


def build_application(
    *,
    settings: AppSettings | None = None,
    llm_client: LLMClient | None = None,
    worker_settings: WorkerSettings | None = None,
    task_id_factory: Callable[[], str] | None = None,
    epubcheck: EpubCheckValidator | None = None,
    migrate: bool = True,
    on_logged: LoggedObserver | None = None,
    content_plan_profile: str = "legacy-v1",
) -> Application:
    """Собирает приложение по локальной конфигурации.

    `llm_client` подменяется в тестах и в сквозном сценарии (`FakeLLMClient`); без него
    берётся собственный клиент провайдера, подготовленный без сетевого обращения.
    `worker_settings` перекрывает параллелизм фоновых задач, заданный секцией `[worker]`
    (PRD § 25). `migrate` отключает применение схемы там, где базу готовит вызывающий код.
    """
    if content_plan_profile == "legacy-v1":
        import os
        content_plan_profile = os.environ.get("LINGUARY_CONTENT_PLAN_PROFILE", "legacy-v1")
    ContextBuilder(content_plan_profile=content_plan_profile)
    try:
        initial_settings = settings if settings is not None else load_settings()
    except SettingsError:
        raise SettingsError("Исходная конфигурация приложения недопустима") from None
    if worker_settings is not None:
        initial_settings = replace(
            initial_settings,
            worker=replace(initial_settings.worker, concurrency=worker_settings.concurrency),
        )
    app_settings = initial_settings
    paths = DataPaths.of_settings(app_settings.storage)
    paths.prepare()
    if migrate:
        apply_migrations(paths.database_path)
    engine = create_database_engine(paths.database_path)
    unit_of_work = SqlAlchemyUnitOfWork.for_engine(engine)
    task_queue = SqliteTaskQueue.for_engine(engine, id_factory=task_id_factory)
    settings_store = SystemSettingsStore(engine)
    try:
        stored = settings_store.load()
        app_settings = overlay_settings(initial_settings, stored)
    except Exception:
        engine.dispose()
        raise SettingsError("Сохранённые настройки приложения недопустимы") from None
    if worker_settings is not None:
        app_settings = replace(
            app_settings,
            worker=replace(app_settings.worker, concurrency=worker_settings.concurrency),
        )
    rate_limiter = LLMRateLimiter(app_settings.llm.requests_per_minute)

    def generation_factory(candidate: AppSettings) -> RuntimeGeneration:
        configured_client = ConfiguredLLMClient(candidate.llm)
        model = _model_client(
            candidate,
            inner=llm_client if llm_client is not None else configured_client,
            unit_of_work=unit_of_work,
            limiter=rate_limiter,
            on_logged=on_logged,
        )
        generation = RuntimeGeneration(
            settings=candidate,
            configured_client=configured_client,
            analysis=_analysis_sequence(
                settings=candidate, unit_of_work=unit_of_work,
                task_queue=task_queue, model=model, content_plan_profile=content_plan_profile,
            ),
            processing=_processing_sequence(
                settings=candidate, unit_of_work=unit_of_work,
                task_queue=task_queue, model=model, content_plan_profile=content_plan_profile,
            ),
            regeneration=_regeneration_sequence(
                settings=candidate, unit_of_work=unit_of_work,
                task_queue=task_queue, model=model, content_plan_profile=content_plan_profile,
            ),
        )
        if llm_client is None:
            configured_client.prepare()
        return generation

    try:
        generation = generation_factory(app_settings)
    except Exception:
        engine.dispose()
        raise SettingsError("Не удалось подготовить настройки приложения") from None
    runtime_settings = RuntimeSettings(
        initial_settings, generation,
        revision=0 if stored is None else stored.revision,
        generation_factory=generation_factory,
    )
    epubcheck = epubcheck or EpubCheckValidator(settings=EpubCheckSettings())

    import_book = ImportBook(
        unit_of_work=unit_of_work,
        reader=EbookLibEpubReader(),
        resource_store=FilesystemEpubResourceStore(root=paths.books_root),
    )
    export_book = ExportBook(
        unit_of_work=unit_of_work,
        writer=EpubExportWriter(),
        validator=epubcheck,
    )
    analysis = RuntimeSequence(runtime_settings, "analysis")
    processing = RuntimeSequence(runtime_settings, "processing")
    regeneration = RuntimeSequence(runtime_settings, "regeneration")
    runner = TaskRunner(
        queue=task_queue,
        handlers=_handlers(
            unit_of_work=unit_of_work,
            paths=paths,
            import_book=import_book,
            export_book=export_book,
            analysis=analysis,
            processing=processing,
            regeneration=regeneration,
        ),
    )
    worker_pool = TaskWorkerPool(
        runner,
        worker_settings or WorkerSettings(concurrency=app_settings.worker.concurrency),
    )
    system_settings = SystemSettingsService(
        runtime=runtime_settings, store=settings_store,
        limiter=rate_limiter, worker_pool=worker_pool,
    )
    return Application(
        runtime_settings=runtime_settings,
        system_settings=system_settings,
        settings_store=settings_store,
        rate_limiter=rate_limiter,
        paths=paths,
        unit_of_work=unit_of_work,
        task_queue=task_queue,
        runner=runner,
        worker_pool=worker_pool,
        epubcheck=epubcheck,
        register_book=RegisterBook(unit_of_work=unit_of_work),
        import_book=import_book,
        export_book=export_book,
        export_sequence=ExportBookSequence(
            unit_of_work=unit_of_work,
            queue=task_queue,
            export_book=export_book,
        ),
        approve_glossary=ApproveGlossary(
            unit_of_work=unit_of_work,
            glossary_confidence_threshold=(
                app_settings.checks.glossary_confidence_threshold
            ),
            glossary_review_min_mentions=app_settings.checks.glossary_review_min_mentions,
            glossary_review_min_chapters=app_settings.checks.glossary_review_min_chapters,
        ),
        reset_processing=ResetProcessing(unit_of_work=unit_of_work),
        retry_task=RetryTask(unit_of_work=unit_of_work),
        analysis=analysis,
        processing=processing,
        regeneration=regeneration,
        _dispose=engine.dispose,
    )


def _model_client(
    settings: AppSettings, *, inner: LLMClient, unit_of_work: SqlAlchemyUnitOfWork,
    limiter: LLMRateLimiter,
    on_logged: LoggedObserver | None = None,
) -> LLMClient:
    """Клиент модели, которым пользуются все LLM-сценарии.

    Ограничитель частоты стоит снаружи журнала: в журнал попадает длительность самого
    вызова провайдера, а не ожидание своей очереди, — по ней оценивается оставшееся
    время обработки. Ограничитель один на приложение, поэтому предел общий для всех
    воркеров.
    """
    model: LLMClient = LoggingLLMClient(
        inner=inner,
        unit_of_work=unit_of_work,
        provider=settings.llm.provider,
        model=settings.llm.model or "not_configured",
        debug_payloads=settings.llm.debug_llm_payloads,
        on_logged=on_logged,
        pricing=(
            TokenPricing(
                input_per_mtok=Decimal(str(settings.llm.input_price_per_mtok)),
                output_per_mtok=Decimal(str(settings.llm.output_price_per_mtok)),
            )
            if settings.llm.input_price_per_mtok is not None
            and settings.llm.output_price_per_mtok is not None
            else None
        ),
    )
    return RateLimitedLLMClient(inner=model, limiter=limiter)


def _analysis_sequence(
    *,
    settings: AppSettings,
    unit_of_work: SqlAlchemyUnitOfWork,
    task_queue: SqliteTaskQueue,
    model: LLMClient,
    content_plan_profile: str = "legacy-v1",
) -> ChapterAnalysisSequence:
    """Анализ глав с порогом глоссария и пределом фрагмента из настроек (PRD § 12, § 14)."""
    return ChapterAnalysisSequence(
        unit_of_work=unit_of_work,
        queue=task_queue,
        analyze_chapter=AnalyzeChapter(
            unit_of_work=unit_of_work,
            llm_client=model,
            retry_policy=RetryPolicy(max_attempts=settings.llm.max_attempts),
            glossary_confidence_threshold=settings.checks.glossary_confidence_threshold,
            max_fragment_blocks=settings.llm.effective_max_fragment_blocks,
            context_builder=_context_builder(settings, content_plan_profile=content_plan_profile),
        ),
    )


def _processing_sequence(
    *,
    settings: AppSettings,
    unit_of_work: SqlAlchemyUnitOfWork,
    task_queue: SqliteTaskQueue,
    model: LLMClient,
    content_plan_profile: str = "legacy-v1",
) -> ChapterProcessingSequence:
    """Обработка глав с пределом группы блоков из настроек (PRD § 10)."""
    return ChapterProcessingSequence(
        unit_of_work=unit_of_work,
        queue=task_queue,
        process_chapter=ProcessChapter(
            unit_of_work=unit_of_work,
            llm_client=model,
            retry_policy=RetryPolicy(max_attempts=settings.llm.max_attempts),
            validator=_validator(settings),
            max_unit_source_chars=settings.llm.max_unit_source_chars,
            context_builder=_context_builder(settings, content_plan_profile=content_plan_profile),
        ),
    )


def _regeneration_sequence(
    *,
    settings: AppSettings,
    unit_of_work: SqlAlchemyUnitOfWork,
    task_queue: SqliteTaskQueue,
    model: LLMClient,
    content_plan_profile: str = "legacy-v1",
) -> BlockRegenerationSequence:
    """Повторная генерация блока с соседним контекстом (PRD § 13, § 21.5)."""
    return BlockRegenerationSequence(
        unit_of_work=unit_of_work,
        queue=task_queue,
        regenerate_block=RegenerateBlock(
            unit_of_work=unit_of_work,
            llm_client=model,
            retry_policy=RetryPolicy(max_attempts=settings.llm.max_attempts),
            validator=_validator(settings),
            context_builder=_context_builder(settings, content_plan_profile=content_plan_profile),
        ),
    )


def _llm_snapshot(settings: AppSettings) -> LLMSettingsSnapshot:
    """Настройки модели, которыми запускается обработка книги (PRD § 22).

    Снимок замораживается вместе с глоссарием: пользователь должен видеть, какой моделью
    собран результат, даже если конфигурация приложения потом изменилась. Ключ провайдера
    в снимок не идёт — это секрет (PRD § 23).
    """
    return LLMSettingsSnapshot(
        provider=settings.llm.provider,
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        max_output_tokens=settings.llm.max_output_tokens,
    )


def _context_builder(settings: AppSettings, *, content_plan_profile: str = "legacy-v1") -> ContextBuilder:
    """Политика контекста запросов к модели (PRD § 13)."""
    return ContextBuilder(
        content_plan_profile=content_plan_profile,
        max_unit_source_chars=settings.llm.max_unit_source_chars,
        regeneration_context_blocks=DEFAULT_REGENERATION_CONTEXT_BLOCKS,
        language_notes_per_chapter=settings.checks.language_notes_per_chapter,
    )


def _validator(settings: AppSettings) -> TranslationResultValidator:
    """Проверки результата обработки с порогами настроек (PRD § 15, ALG-04)."""
    checks = settings.checks
    return TranslationResultValidator(
        target_language_min_share=checks.target_language_min_share,
        language_check_min_length=checks.language_check_min_length,
        repetition_ngram_size=checks.repetition_ngram_size,
        repetition_min_repeats=checks.repetition_min_repeats,
    )


def _handlers(
    *,
    unit_of_work: SqlAlchemyUnitOfWork,
    paths: DataPaths,
    import_book: ImportBook,
    export_book: ExportBook,
    analysis: ChapterAnalysisSequence,
    processing: ChapterProcessingSequence,
    regeneration: BlockRegenerationSequence,
) -> dict[TaskType, TaskHandler]:
    """Обработчики всех типов задач очереди (ALG-02).

    Задачи уровня книги команду сценария не несут: runner знает только идемпотентный
    ключ, а путь целевого файла экспорта — вообще настройка приложения. Собрать команду
    может только состав зависимостей, поэтому связывание живёт здесь.
    """
    return {
        TaskType.IMPORT_BOOK: to_async_handler(
            _import_handler(unit_of_work=unit_of_work, import_book=import_book)
        ),
        TaskType.EXPORT_BOOK: to_async_handler(
            _export_handler(export_book=export_book, paths=paths)
        ),
        TaskType.ANALYZE_CHAPTER: analysis.run_task,
        TaskType.PROCESS_TRANSLATION_UNIT: processing.run_task,
        TaskType.REGENERATE_BLOCK: regeneration.run_task,
    }


def _import_handler(
    *, unit_of_work: SqlAlchemyUnitOfWork, import_book: ImportBook
) -> Callable[[ProcessingTask], None]:
    """Обработчик `IMPORT_BOOK`: команда собирается из записи загруженной книги."""

    def handle(task: ProcessingTask) -> None:
        with unit_of_work.transaction() as transaction:
            book = transaction.books.get(task.book_id)
        if book is None:
            raise ImportBookError(f"книги {task.book_id} нет в хранилище")
        import_book(
            ImportBookCommand(
                book_id=book.id,
                source_epub_path=book.source_epub_path,
                target_language=book.target_language,
                source_language=book.source_language,
            )
        )

    return handle


def _export_handler(
    *, export_book: ExportBook, paths: DataPaths
) -> Callable[[ProcessingTask], None]:
    """Обработчик `EXPORT_BOOK`: целевой файл выбирает раскладка каталога данных."""

    def handle(task: ProcessingTask) -> None:
        target = paths.export_epub(task.book_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        export_book(
            ExportBookCommand(
                book_id=task.book_id,
                target_path=str(target),
                mode=ExportMode.FULL,
            )
        )

    return handle


async def run_worker(application: Application) -> None:
    """Выполняет фоновые задачи, пока приложение работает (PRD § 18.1).

    Перед первой задачей задачи, оставшиеся `running` после аварийного завершения,
    помечаются прерванными: их уже никто не исполняет, и пользователь запускает их
    повторно (ALG-02, PRD § 17). Это верно, только если рядом нет другого исполнителя
    той же базы — Web UI и `linguary worker` могут работать одновременно, — поэтому
    восстановление выполняет лишь исполнитель, стартовавший в одиночестве.

    Отметка берётся прямо в цикле событий, а не в потоке: ждать она может только чужого
    восстановления, а отменённое ожидание в потоке оставило бы отметку висеть до конца
    процесса.
    """
    lock = ExecutorLock(application.paths.executor_lock)
    alone = lock.join()
    try:
        if alone:
            await application.runner.recover_interrupted()
            lock.share()
        await application.worker_pool.run_forever()
    finally:
        lock.leave()


async def drain_tasks(application: Application) -> tuple[ProcessingTask, ...]:
    """Выполняет накопленные задачи и возвращается, когда очередь пуста.

    Режим разовой прогонки: им пользуются CLI и тесты, где ждать новых задач незачем.
    Прерванные задачи он не восстанавливает, но отмечается исполнителем: Web UI,
    запущенный во время прогонки, не должен принять её задачи за брошенные.
    """
    lock = ExecutorLock(application.paths.executor_lock)
    lock.join()
    try:
        lock.share()
        return await application.worker_pool.drain()
    finally:
        lock.leave()


__all__ = [
    "DEFAULT_REGENERATION_CONTEXT_BLOCKS",
    "Application",
    "build_application",
    "drain_tasks",
    "run_worker",
]
