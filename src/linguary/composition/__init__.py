"""Composition root приложения: сборка зависимостей для Web UI, CLI и сквозного теста.

Домен и прикладные сценарии знают только порты; конкретные адаптеры — SQLite, EbookLib,
EPUBCheck, провайдер LLM — известны единственному месту, и это оно (ADR § 3). Наружу
пакет отдаёт собранное приложение (`build_application`), раскладку каталога данных
(`DataPaths`) и два режима исполнения фоновых задач: постоянный (`run_worker`) и разовый
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
