"""Принятые изменения настроек завершаются независимо от HTTP-наблюдателя."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine

from linguary.adapters.config.settings import AppSettings, SettingsError
from linguary.adapters.config.system_settings import SystemSettingsUpdate, candidate_settings
from linguary.adapters.db.system_settings import SettingsRevisionConflict, SystemSettingsStore
from linguary.adapters.llm.rate_limit import LLMRateLimiter
from linguary.adapters.runtime import TaskWorkerPool
from linguary.composition.runtime_settings import RuntimeGeneration, RuntimeSettings


class SettingsClosingError(SettingsError):
    """Новая операция не принята после начала завершения приложения."""


class SettingsNotAppliedError(SettingsError):
    """Подготовка или запись настроек не завершилась; изменения не применены."""


class SystemSettingsService:
    def __init__(
        self,
        *,
        runtime: RuntimeSettings,
        store: SystemSettingsStore,
        limiter: LLMRateLimiter,
        worker_pool: TaskWorkerPool,
    ) -> None:
        self._runtime = runtime
        self._store = store
        self._limiter = limiter
        self._worker_pool = worker_pool
        self._lock = asyncio.Lock()
        self._closing = False
        self._operations: set[asyncio.Task[None]] = set()
        self._errors: list[SettingsError] = []
        self._close_task: asyncio.Task[None] | None = None

    async def save(self, update: SystemSettingsUpdate, *, expected_revision: int) -> None:
        operation = self._accept(
            lambda: self._apply(update, expected_revision=expected_revision)
        )
        await asyncio.shield(operation)

    async def reset(self, *, expected_revision: int) -> None:
        operation = self._accept(
            lambda: self._apply(None, expected_revision=expected_revision)
        )
        await asyncio.shield(operation)

    def _accept(self, make_operation: Callable[[], Coroutine[None, None, None]]) -> asyncio.Task[None]:
        if self._closing:
            raise SettingsClosingError('Приложение завершает работу. Запрос не принят')
        operation = asyncio.create_task(make_operation())
        self._operations.add(operation)
        operation.add_done_callback(self._collect)
        return operation

    def _collect(self, operation: asyncio.Task[None]) -> None:
        try:
            error = operation.exception()
            if error is not None:
                self._errors.append(
                    error if isinstance(error, SettingsError)
                    else SettingsNotAppliedError('Изменения не применены')
                )
        except asyncio.CancelledError:
            self._errors.append(SettingsNotAppliedError('Изменение настроек прервано'))
        finally:
            self._operations.discard(operation)

    async def _apply(self, update: SystemSettingsUpdate | None, *, expected_revision: int) -> None:
        async with self._lock:
            if type(expected_revision) is not int or expected_revision < 0:
                raise SettingsRevisionConflict('Редакция настроек недопустима')
            if self._runtime.revision != expected_revision:
                raise SettingsRevisionConflict('Настройки изменены другой операцией')
            generation: RuntimeGeneration | None = None
            try:
                stored = await asyncio.to_thread(self._store.load)
                if (0 if stored is None else stored.revision) != expected_revision:
                    raise SettingsRevisionConflict('Настройки изменены другой операцией')
                current = self._runtime.current.settings
                candidate = (
                    self._runtime.initial_settings if update is None
                    else candidate_settings(current, update)
                )
                self._validate_targets(candidate)
                generation = self._runtime.prepare(candidate)
                if update is None:
                    result = await asyncio.to_thread(
                        self._store.reset, expected_revision=expected_revision
                    )
                else:
                    overrides = dict(
                        provider=candidate.llm.provider,
                        model=candidate.llm.model,
                        base_url=candidate.llm.base_url,
                        requests_per_minute=candidate.llm.requests_per_minute,
                        concurrency=candidate.worker.concurrency,
                    )
                    result = await asyncio.to_thread(
                        self._store.save, overrides, candidate.llm.api_key,
                        expected_revision=expected_revision,
                    )
            except Exception as error:
                if generation is not None:
                    await self._discard(generation)
                if isinstance(error, SettingsRevisionConflict):
                    raise
                raise SettingsNotAppliedError('Изменения не применены') from None
            self._runtime.publish(generation, result.revision)
            self._limiter.set_limit(candidate.llm.requests_per_minute)
            self._worker_pool.set_concurrency(candidate.worker.concurrency)

    @staticmethod
    def _validate_targets(settings: AppSettings) -> None:
        rate = settings.llm.requests_per_minute
        concurrency = settings.worker.concurrency
        if type(concurrency) is not int or concurrency < 1:
            raise SettingsNotAppliedError('Изменения не применены')
        if rate is not None and (type(rate) is not int or rate < 1):
            raise SettingsNotAppliedError('Изменения не применены')

    async def _discard(self, generation: RuntimeGeneration) -> None:
        try:
            await generation.configured_client.aclose()
        except Exception:
            self._errors.append(SettingsError('Не удалось закрыть неподключённый клиент модели'))

    def begin_close(self) -> None:
        self._closing = True

    async def drain(self) -> tuple[SettingsError, ...]:
        while self._operations:
            await asyncio.gather(*tuple(self._operations), return_exceptions=True)
        return tuple(self._errors)

    async def aclose(self) -> None:
        self.begin_close()
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
            self._close_task.add_done_callback(self._collect_close)
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        await self.drain()

    @staticmethod
    def _collect_close(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()
