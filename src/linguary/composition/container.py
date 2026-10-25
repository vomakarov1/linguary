"""Application dependency composition: the single place where concrete adapters are known.

Domain and application use cases know only ports, so something must connect
them once with SQLite, EbookLib, EPUBCheck, and the LLM provider, and map queue tasks
to use case commands (ADR § 3). This is done here by `build_application`, and its result
is used by Web UI, CLI, and end-to-end tests: there is no second composition site in the application.

What is assembled:

- storage — SQLite engine, unit of work, and task queue over a single database;
- worker — background task concurrency from the `[worker]` section (PRD § 25);
- model client — call log and, if a rate limit is specified, a rate limiter shared across all workers;
- import — EPUB reader along with the book resource store;
- export — EPUB writer and external EPUBCheck validator;
- LLM use cases — chapter analysis, chapter processing, and block regeneration with thresholds
  and limits from local configuration;
- handlers for all five queue task types: only here is it known how to obtain a use case command
  and target file path from an idempotent task key.

Synchronous use cases are connected to the async runner via `to_async_handler`:
EPUB parsing and SQLite operations block the event loop and cannot run directly in a coroutine
(PRD § 25).
"""

from __future__ import annotations

import asyncio

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from hashlib import sha256

from linguary.adapters.config import AppSettings, load_settings
from linguary.adapters.config.settings import SettingsError
from linguary.adapters.config.system_settings import overlay_settings
from linguary.adapters.db.system_settings import SystemSettingsStore
from linguary.adapters.llm.rate_limit import LLMRateLimiter
from linguary.application.diagnostics import Diagnostics, diagnostics_for
from linguary.application.book_conditions import BookConditionsService
from linguary.composition.book_runtime import BookRuntimeResolver, BoundTaskQueue, ProvenanceClient, conditions_payload
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

#: How many neighboring blocks the model sees during regeneration (PRD § 13). The value
#: was justified in P3.T7 and is intentionally not exposed to the settings file.
DEFAULT_REGENERATION_CONTEXT_BLOCKS = 2


@dataclass(frozen=True, slots=True)
class Application:
    """Assembled application: use cases, task queue, and everything needed about them.

    An instance is created once per process. Web UI and CLI obtain ready use cases from here
    and do not construct anything themselves.
    """

    runtime_settings: RuntimeSettings
    book_conditions: BookConditionsService
    book_runtime: BookRuntimeResolver
    diagnostics: Diagnostics
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
        """Book processing start use case.

        Assembled on demand rather than at application startup: it freezes a snapshot
        of model settings, and an unconfigured LLM provides no snapshot. Rejection arrives as the same
        `LLMConfigurationError` as an attempt to contact the model (PRD § 23).
        """
        settings = self.settings
        settings.llm.ensure_ready()
        return StartProcessing(
            unit_of_work=self.unit_of_work,
            book_conditions=self.book_conditions,
            glossary_confidence_threshold=(
                settings.checks.glossary_confidence_threshold
            ),
            glossary_review_min_mentions=settings.checks.glossary_review_min_mentions,
            glossary_review_min_chapters=settings.checks.glossary_review_min_chapters,
            llm_settings_snapshot=_llm_snapshot(settings),
        )

    async def start_import(self, book_id: str) -> ProcessingTask:
        """Enqueues book import task for an uploaded book (PRD § 18)."""
        return await self.task_queue.enqueue(
            TaskIdentity(book_id=book_id, task_type=TaskType.IMPORT_BOOK)
        )

    async def start_export(self, book_id: str) -> ProcessingTask:
        """Enqueues final EPUB assembly task (PRD § 20)."""
        return await self.export_sequence.start(book_id)

    def begin_close(self) -> None:
        """Disallows new settings changes and new task acquisitions."""
        self.system_settings.begin_close()
        self.worker_pool.request_stop()

    async def aclose(self) -> None:
        """Waits for accepted changes before closing generations and database."""
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
        """Closes database connections.

        This method closes only the database. HTTP clients owned by generations
        are released via asynchronous `aclose`.
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
    """Assembles application from local configuration.

    `llm_client` is substituted in tests and in the end-to-end scenario (`FakeLLMClient`); without it,
    the provider's own client prepared without network access is used.
    `worker_settings` overrides background task concurrency specified in the `[worker]` section
    (PRD § 25). `migrate` disables schema application when the database is prepared by the caller.
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

    def generation_factory(candidate: AppSettings, *, conditions=None) -> RuntimeGeneration:
        profile = content_plan_profile if conditions is None else conditions.payload["profile"]
        templates = None if conditions is None else conditions.payload["prompt_templates"]
        queue = task_queue if conditions is None else BoundTaskQueue(task_queue, book_conditions, conditions.id, task_id_factory)
        configured_client = ConfiguredLLMClient(candidate.llm)
        analysis_model = candidate.llm.effective_analysis_model
        processing_model = candidate.llm.effective_processing_model

        analysis_inner = (
            llm_client
            if llm_client is not None
            else configured_client.client_for(analysis_model)
        )
        processing_inner = (
            llm_client
            if llm_client is not None
            else configured_client.client_for(processing_model)
        )

        analysis_client = _model_client(
            candidate,
            inner=analysis_inner,
            unit_of_work=unit_of_work,
            limiter=rate_limiter,
            on_logged=on_logged,
            model_name=analysis_model or "not_configured",
        )
        processing_client = _model_client(
            candidate,
            inner=processing_inner,
            unit_of_work=unit_of_work,
            limiter=rate_limiter,
            on_logged=on_logged,
            model_name=processing_model or "not_configured",
        )
        if conditions is not None:
            analysis_client = ProvenanceClient(analysis_client, book_conditions, conditions.id)
            processing_client = ProvenanceClient(processing_client, book_conditions, conditions.id)
        generation = RuntimeGeneration(
            settings=candidate,
            configured_client=configured_client,
            analysis=_analysis_sequence(
                settings=candidate, unit_of_work=unit_of_work,
                task_queue=queue, model=analysis_client, content_plan_profile=profile, prompt_templates=templates,
            ),
            processing=_processing_sequence(
                settings=candidate, unit_of_work=unit_of_work,
                task_queue=queue, model=processing_client, content_plan_profile=profile, prompt_templates=templates,
            ),
            regeneration=_regeneration_sequence(
                settings=candidate, unit_of_work=unit_of_work,
                task_queue=queue, model=processing_client, content_plan_profile=profile, prompt_templates=templates,
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
    book_conditions = BookConditionsService(unit_of_work,
        lambda book, revision: conditions_payload(book, runtime_settings.current.settings, revision, content_plan_profile),
        lambda: runtime_settings.revision,
        lambda path: sha256(Path(path).read_bytes()).hexdigest())
    book_runtime = BookRuntimeResolver(book_conditions, runtime_settings, generation_factory)
    diagnostics = diagnostics_for(unit_of_work)
    diagnostics.ensure_available()
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
    analysis = RuntimeSequence(runtime_settings, "analysis", book_runtime=book_runtime)
    processing = RuntimeSequence(runtime_settings, "processing", book_runtime=book_runtime)
    regeneration = RuntimeSequence(runtime_settings, "regeneration", book_runtime=book_runtime)
    runner = TaskRunner(
        queue=task_queue,
        maintenance=diagnostics.ensure_available,
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
        book_conditions=book_conditions,
        book_runtime=book_runtime,
        diagnostics=diagnostics,
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
    model_name: str | None = None,
) -> LLMClient:
    """Model client used by all LLM use cases.

    The rate limiter wraps the logger: the log records the duration of the provider call itself
    rather than queue waiting time, which is used to estimate remaining processing time.
    The rate limiter is single per application, so the limit is shared across all workers.
    """
    model: LLMClient = LoggingLLMClient(
        inner=inner,
        unit_of_work=unit_of_work,
        provider=settings.llm.provider,
        model=model_name or settings.llm.model or "not_configured",
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
    prompt_templates=None,
) -> ChapterAnalysisSequence:
    """Chapter analysis with glossary threshold and fragment limit from settings (PRD § 12, § 14)."""
    return ChapterAnalysisSequence(
        unit_of_work=unit_of_work,
        queue=task_queue,
        analyze_chapter=AnalyzeChapter(
            unit_of_work=unit_of_work,
            llm_client=model,
            retry_policy=RetryPolicy(max_attempts=settings.llm.max_attempts),
            glossary_confidence_threshold=settings.checks.glossary_confidence_threshold,
            max_fragment_blocks=settings.llm.effective_max_fragment_blocks,
            context_builder=_context_builder(settings, content_plan_profile=content_plan_profile, prompt_templates=prompt_templates),
        ),
    )


def _processing_sequence(
    *,
    settings: AppSettings,
    unit_of_work: SqlAlchemyUnitOfWork,
    task_queue: SqliteTaskQueue,
    model: LLMClient,
    content_plan_profile: str = "legacy-v1",
    prompt_templates=None,
) -> ChapterProcessingSequence:
    """Chapter processing with block group limit from settings (PRD § 10)."""
    return ChapterProcessingSequence(
        unit_of_work=unit_of_work,
        queue=task_queue,
        process_chapter=ProcessChapter(
            unit_of_work=unit_of_work,
            llm_client=model,
            retry_policy=RetryPolicy(max_attempts=settings.llm.max_attempts),
            validator=_validator(settings),
            max_unit_source_chars=settings.llm.max_unit_source_chars,
            context_builder=_context_builder(settings, content_plan_profile=content_plan_profile, prompt_templates=prompt_templates),
        ),
    )


def _regeneration_sequence(
    *,
    settings: AppSettings,
    unit_of_work: SqlAlchemyUnitOfWork,
    task_queue: SqliteTaskQueue,
    model: LLMClient,
    content_plan_profile: str = "legacy-v1",
    prompt_templates=None,
) -> BlockRegenerationSequence:
    """Block regeneration with neighboring context (PRD § 13, § 21.5)."""
    return BlockRegenerationSequence(
        unit_of_work=unit_of_work,
        queue=task_queue,
        regenerate_block=RegenerateBlock(
            unit_of_work=unit_of_work,
            llm_client=model,
            retry_policy=RetryPolicy(max_attempts=settings.llm.max_attempts),
            validator=_validator(settings),
            context_builder=_context_builder(settings, content_plan_profile=content_plan_profile, prompt_templates=prompt_templates),
        ),
    )


def _llm_snapshot(settings: AppSettings) -> LLMSettingsSnapshot:
    """Model settings with which book processing is launched (PRD § 22).

    The snapshot is frozen together with the glossary: the user must see which model
    produced the result even if the application configuration changes later. The provider key
    is not included in the snapshot — it is a secret (PRD § 23).
    """
    return LLMSettingsSnapshot(
        provider=settings.llm.provider,
        model=settings.llm.effective_processing_model,
        temperature=settings.llm.temperature,
        max_output_tokens=settings.llm.max_output_tokens,
    )


def _context_builder(settings: AppSettings, *, content_plan_profile: str = "legacy-v1", prompt_templates=None) -> ContextBuilder:
    """Model request context policy (PRD § 13)."""
    return ContextBuilder(
        content_plan_profile=content_plan_profile,
        prompt_templates=prompt_templates,
        max_unit_source_chars=settings.llm.max_unit_source_chars,
        regeneration_context_blocks=DEFAULT_REGENERATION_CONTEXT_BLOCKS,
        language_notes_per_chapter=settings.checks.language_notes_per_chapter,
        reader_level=settings.reading.normalized_level,
    )


def _validator(settings: AppSettings) -> TranslationResultValidator:
    """Processing result checks with settings thresholds (PRD § 15, ALG-04)."""
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
    """Handlers for all queue task types (ALG-02).

    Book-level tasks carry no use case command: the runner knows only the idempotent
    key, and the export target file path is an application setting. Only dependency composition
    can assemble the command, so binding lives here.
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
    """`IMPORT_BOOK` handler: command is assembled from the uploaded book record."""

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
    """`EXPORT_BOOK` handler: target file is chosen by the data directory layout."""

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
    """Executes background tasks while the application runs (PRD § 18.1).

    Before the first task, tasks remaining `running` after a crash are marked interrupted:
    no one is executing them anymore, and the user reruns them (ALG-02, PRD § 17).
    This holds only if no other executor shares the same database — Web UI and `linguary worker`
    can run concurrently, so recovery is performed only by an executor starting alone.

    The lock is acquired directly in the event loop rather than in a thread: it can only wait
    for another recovery, and a cancelled wait in a thread would leave the lock dangling until process exit.
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
    """Executes accumulated tasks and returns when the queue is empty.

    One-off execution mode: used by CLI and tests where waiting for new tasks is unnecessary.
    It does not recover interrupted tasks, but registers as an executor: Web UI launched
    during draining must not mistake its tasks for abandoned ones.
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
