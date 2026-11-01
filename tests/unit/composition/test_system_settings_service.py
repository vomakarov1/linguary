"""Shutdown must allow accepted settings operations to finish their callbacks."""
from __future__ import annotations

import subprocess
import sys
import textwrap


def test_drain_yields_to_completed_operation_callbacks():
    # A blocked event loop cannot run its own timeout, so isolate this regression.
    script = textwrap.dedent("""
        import asyncio

        from linguary.composition.system_settings import (
            SettingsNotAppliedError, SystemSettingsService,
        )

        async def complete():
            pass

        async def fail():
            raise RuntimeError("Settings operation failed")

        async def main():
            service = SystemSettingsService(
                runtime=None, store=None, limiter=None, worker_pool=None,
            )
            service._accept(complete)
            service._accept(fail)
            errors = await asyncio.create_task(service.drain())
            assert not service._operations
            assert len(errors) == 1
            assert isinstance(errors[0], SettingsNotAppliedError)
            await asyncio.sleep(0)
            assert await service.drain() == errors

        asyncio.run(main())
    """)
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True,
        text=True, timeout=5,
    )
