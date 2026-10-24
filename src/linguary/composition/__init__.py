"""Application composition root: dependency wiring for Web UI, CLI, and end-to-end tests.

The domain and application use cases know only ports; concrete adapters — SQLite, EbookLib,
EPUBCheck, LLM provider — are known only to a single place, which is this one (ADR § 3). Externally,
the package exposes the assembled application (`build_application`), data directory layout
(`DataPaths`), and two background task execution modes: continuous (`run_worker`) and draining
(`drain_tasks`).
"""

from linguary.composition.container import (
    Application,
    build_application,
    drain_tasks,
    run_worker,
)
from linguary.composition.migrations import apply_migrations
from linguary.composition.paths import DataPaths

__all__ = [
    "Application",
    "DataPaths",
    "apply_migrations",
    "build_application",
    "drain_tasks",
    "run_worker",
]
