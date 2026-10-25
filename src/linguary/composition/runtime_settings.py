"""Ownership of use case generations and their model clients."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from linguary.adapters.config import AppSettings
from linguary.adapters.config.settings import SettingsError
from linguary.composition.llm import ConfiguredLLMClient


@dataclass(frozen=True, slots=True)
class RuntimeGeneration:
    settings: AppSettings = field(repr=False)
    configured_client: ConfiguredLLMClient = field(repr=False)
    analysis: Any = field(repr=False)
    processing: Any = field(repr=False)
    regeneration: Any = field(repr=False)


@dataclass(slots=True)
class _GenerationState:
    generation: RuntimeGeneration
    users: int = 0
    retired: bool = False
    closing: bool = False


class RuntimeSettings:
    """Publishing is synchronous; busy generations are released after their last use case."""

    def __init__(
        self,
        initial_settings: AppSettings,
        generation: RuntimeGeneration,
        *,
        revision: int = 0,
        generation_factory: Callable[[AppSettings], RuntimeGeneration] | None = None,
    ) -> None:
        self.initial_settings = initial_settings
        self.current = generation
        self.revision = revision
        self._generation_factory = generation_factory
        self._states = {id(generation): _GenerationState(generation)}
        self._close_tasks: set[asyncio.Task[None]] = set()
        self._close_errors: list[SettingsError] = []
        self._changed = asyncio.Event()
        self._closing = False
        self._shutdown_task: asyncio.Task[None] | None = None

    def prepare(self, settings: AppSettings) -> RuntimeGeneration:
        if self._closing or self._generation_factory is None:
            raise SettingsError('Подготовка настроек недоступна')
        try:
            return self._generation_factory(settings)
        except Exception:
            raise SettingsError('Не удалось подготовить настройки') from None

    @contextmanager
    def lease(self) -> Iterator[RuntimeGeneration]:
        if self._closing:
            raise SettingsError('Приложение завершает работу')
        state = self._states[id(self.current)]
        state.users += 1
        try:
            yield state.generation
        finally:
            state.users -= 1
            self._schedule_close(state)
            self._changed.set()

    def publish(self, generation: RuntimeGeneration, revision: int) -> None:
        if self._closing:
            raise SettingsError('Приложение завершает работу')
        if generation is self.current or id(generation) in self._states:
            raise SettingsError('Поколение уже опубликовано')
        previous = self._states[id(self.current)]
        self._states[id(generation)] = _GenerationState(generation)
        self.current = generation
        self.revision = revision
        previous.retired = True
        self._schedule_close(previous)

    def _schedule_close(self, state: _GenerationState) -> None:
        if state.retired and not state.users and not state.closing:
            state.closing = True
            task = asyncio.create_task(self._close_generation(state))
            self._close_tasks.add(task)
            task.add_done_callback(self._collect_close)

    async def _close_generation(self, state: _GenerationState) -> None:
        try:
            await state.generation.configured_client.aclose()
        except Exception:
            self._close_errors.append(SettingsError('Не удалось закрыть клиент модели'))
        finally:
            del self._states[id(state.generation)]
            self._changed.set()

    def _collect_close(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            self._close_errors.append(SettingsError('Закрытие клиента модели прервано'))
        except Exception:
            self._close_errors.append(SettingsError('Не удалось закрыть клиент модели'))
        finally:
            self._close_tasks.discard(task)
            self._changed.set()

    async def aclose(self) -> None:
        if self._shutdown_task is None:
            self._closing = True
            self._shutdown_task = asyncio.create_task(self._shutdown())
            self._shutdown_task.add_done_callback(self._collect_shutdown)
        await asyncio.shield(self._shutdown_task)

    def _collect_shutdown(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    async def _shutdown(self) -> None:
        for state in self._states.values():
            state.retired = True
            self._schedule_close(state)
        while self._states or self._close_tasks:
            self._changed.clear()
            if self._close_tasks:
                await asyncio.gather(*tuple(self._close_tasks), return_exceptions=True)
            else:
                await self._changed.wait()
        if self._close_errors:
            raise SettingsError('Не удалось закрыть все клиенты модели') from None


class RuntimeSequence:
    """Stable access point selects a generation before the first await."""

    def __init__(self, runtime: RuntimeSettings, name: str, *, book_runtime=None) -> None:
        self._runtime = runtime
        self._name = name
        self._book_runtime = book_runtime

    async def start(self, *args: Any, **kwargs: Any) -> Any:
        if self._book_runtime is not None:
            book_id = args[0] if args else kwargs['book_id']
            service = self._book_runtime.service
            if self._name == 'analysis':
                conditions_id = service.freeze_before_analysis(book_id, self._runtime.revision)
            else:
                conditions_id = service.require(book_id).id
            async with self._book_runtime.lease(conditions_id, self._name) as generation:
                return await getattr(generation, self._name).start(*args, **kwargs)
        with self._runtime.lease() as generation:
            return await getattr(generation, self._name).start(*args, **kwargs)

    async def run_task(self, task: Any) -> None:
        if self._book_runtime is not None:
            from linguary.application.book_conditions import ConditionsError
            with self._book_runtime.service.uow.transaction() as transaction:
                binding = transaction.work.get_binding('task', task.id)
                if binding is None:
                    raise ConditionsError('legacy_choice_required')
            async with self._book_runtime.lease(binding['conditions_id'], self._name) as generation:
                await getattr(generation, self._name).run_task(task)
            return
        with self._runtime.lease() as generation:
            await getattr(generation, self._name).run_task(task)
