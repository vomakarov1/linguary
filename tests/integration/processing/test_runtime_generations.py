"""Поколения сохраняют занятые сценарии и собирают все закрытия."""
from dataclasses import replace
import asyncio

import pytest

from linguary.adapters.config import AppSettings, LLMSettings
from linguary.adapters.config.settings import SettingsError
from linguary.composition.runtime_settings import RuntimeGeneration, RuntimeSettings, RuntimeSequence


class Client:
    def __init__(self, *, fail=False, release=None):
        self.closed = 0
        self.fail = fail
        self.release = release
        self.started = asyncio.Event()

    async def aclose(self):
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        self.closed += 1
        if self.fail:
            raise RuntimeError('private-test-key')


class Sequence:
    def __init__(self, name, *, release=None):
        self.name = name
        self.calls = []
        self.started = asyncio.Event()
        self.release = release

    async def start(self, *args):
        return self.name

    async def run_task(self, task):
        self.calls.append(self.name)
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        self.calls.append(self.name)


def generation(name, *, client=None, sequence=None):
    settings = AppSettings(llm=LLMSettings(model=name))
    sequence = sequence or Sequence(name)
    return RuntimeGeneration(settings, client or Client(), sequence, sequence, sequence)


async def test_task_retains_generation_across_multiple_calls():
    release = asyncio.Event()
    old_sequence = Sequence('old', release=release)
    old = generation('old', sequence=old_sequence)
    new = generation('new')
    runtime = RuntimeSettings(old.settings, old)
    proxy = RuntimeSequence(runtime, 'processing')
    running = asyncio.create_task(proxy.run_task(None))
    await old_sequence.started.wait()
    runtime.publish(new, 1)
    assert old.configured_client.closed == 0
    release.set()
    await running
    assert old_sequence.calls == ['old', 'old']
    await old.configured_client.started.wait()
    await runtime.aclose()
    assert old.configured_client.closed == 1


async def test_new_task_uses_published_generation():
    old, new = generation('old'), generation('new')
    runtime = RuntimeSettings(old.settings, old)
    proxy = RuntimeSequence(runtime, 'analysis')
    runtime.publish(new, 1)
    await proxy.run_task(None)
    assert new.analysis.calls == ['new', 'new']
    assert await proxy.start('book') == 'new'
    assert runtime.revision == 1
    await runtime.aclose()


async def test_retired_client_closes_after_last_lease():
    old, new = generation('old'), generation('new')
    runtime = RuntimeSettings(old.settings, old)
    with runtime.lease() as leased:
        runtime.publish(new, 1)
        assert leased is old and old.configured_client.closed == 0
    await old.configured_client.started.wait()
    await runtime.aclose()
    assert old.configured_client.closed == 1


async def test_retired_close_failure_keeps_current_generation():
    old = generation('old', client=Client(fail=True))
    new = generation('new')
    runtime = RuntimeSettings(old.settings, old)
    runtime.publish(new, 1)
    await old.configured_client.started.wait()
    assert runtime.current is new
    with pytest.raises(SettingsError) as caught:
        await runtime.aclose()
    assert 'private-test-key' not in str(caught.value)
    assert new.configured_client.closed == 1


async def test_prepare_failure_does_not_publish():
    old = generation('old')
    def fail(settings):
        raise RuntimeError('private-test-key')
    runtime = RuntimeSettings(old.settings, old, generation_factory=fail)
    with pytest.raises(SettingsError) as caught:
        runtime.prepare(replace(old.settings, llm=LLMSettings(model='new')))
    assert 'private-test-key' not in str(caught.value)
    assert runtime.current is old and runtime.revision == 0
    await runtime.aclose()


async def test_shutdown_closes_all_owned_clients():
    old, new = generation('old'), generation('new')
    runtime = RuntimeSettings(old.settings, old)
    runtime.publish(new, 1)
    await runtime.aclose()
    await runtime.aclose()
    assert old.configured_client.closed == new.configured_client.closed == 1
    with pytest.raises(SettingsError):
        with runtime.lease():
            pass


@pytest.mark.parametrize('cancel', [False, True])
async def test_generation_release_on_error_and_cancel(cancel):
    release = asyncio.Event()
    sequence = Sequence('old', release=release)
    old, new = generation('old', sequence=sequence), generation('new')
    runtime = RuntimeSettings(old.settings, old)
    async def use():
        with runtime.lease():
            sequence.started.set()
            await release.wait()
            raise RuntimeError('task error')
    task = asyncio.create_task(use())
    await sequence.started.wait()
    runtime.publish(new, 1)
    if cancel:
        task.cancel()
    else:
        release.set()
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await task
    await runtime.aclose()
    assert old.configured_client.closed == 1


async def test_close_collects_retirement_tasks_and_errors():
    release = asyncio.Event()
    old = generation('old', client=Client(fail=True, release=release))
    new = generation('new')
    runtime = RuntimeSettings(old.settings, old)
    runtime.publish(new, 1)
    await old.configured_client.started.wait()
    closing = asyncio.create_task(runtime.aclose())
    await new.configured_client.started.wait()
    assert not closing.done()
    release.set()
    with pytest.raises(SettingsError):
        await closing
    assert old.configured_client.closed == new.configured_client.closed == 1
