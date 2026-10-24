"""Shared fixtures for Web UI integration tests.

The application is wired using the same composition root as production:
temporary data directory, database schema from Alembic migrations, and `FakeLLMClient`
instead of real provider. This tests both HTTP endpoints and dependencies — in particular,
verifying that import receives resource storage and task handlers are registered.

Background worker is disabled in tests: tasks are drained explicitly via `drain`,
otherwise outcomes would depend on timing races.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from linguary.adapters.config import (
    AppSettings,
    CheckSettings,
    LLMSettings,
    ServerSettings,
    StorageSettings,
)
from linguary.adapters.llm import FakeLLMClient
from linguary.composition import Application, build_application
from linguary.web import create_app

#: Test EPUB fixture suite (PRD § 28.1).
FIXTURE_DIRECTORY = Path(__file__).resolve().parents[2] / "fixtures" / "epub"


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """Application data directory for the test."""
    return tmp_path / "linguary-data"


@pytest.fixture
def settings(data_root: Path) -> AppSettings:
    """Local test settings: unconfigured LLM simulates first-launch state."""
    return AppSettings(
        llm=LLMSettings(),
        checks=CheckSettings(),
        storage=StorageSettings(data_root=str(data_root)),
        server=ServerSettings(),
    )


@pytest.fixture
def llm_client() -> FakeLLMClient:
    """Model client without prepared replies: tests in this package do not invoke LLM."""
    return FakeLLMClient()


@pytest.fixture
def web_task_ids() -> Callable[[], str]:
    """Predictable task identifiers: `task-1`, `task-2`, ..."""
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"task-{counter}"

    return next_id


@pytest.fixture
def application(
    settings: AppSettings, llm_client: FakeLLMClient, web_task_ids: Callable[[], str]
) -> Iterator[Application]:
    """Assembled application over a temporary data directory."""
    built = build_application(
        settings=settings, llm_client=llm_client, task_id_factory=web_task_ids
    )
    try:
        yield built
    finally:
        built.dispose()


@pytest.fixture
def configured_application(
    settings: AppSettings, llm_client: FakeLLMClient, web_task_ids: Callable[[], str]
) -> Iterator[Application]:
    """Application with fake configured model to enable starting processing."""
    configured = replace(
        settings,
        llm=LLMSettings(model="fake-model", api_key="test-key"),
    )
    built = build_application(
        settings=configured, llm_client=llm_client, task_id_factory=web_task_ids
    )
    try:
        yield built
    finally:
        built.dispose()


@pytest.fixture
def client(application: Application) -> Iterator[TestClient]:
    """Web UI test client without background worker.

    Client connects using the application server base_url: server responds only
    to configured hosts, matching production (PRD § 4.1).
    """
    app = create_app(application=application, run_background_worker=False)
    with TestClient(app, base_url=application.settings.server.base_url) as test_client:
        yield test_client


@pytest.fixture
def configured_client(
    configured_application: Application,
) -> Iterator[TestClient]:
    """Test client with configured model ready to initiate a processing run."""
    app = create_app(
        application=configured_application, run_background_worker=False
    )
    with TestClient(
        app, base_url=configured_application.settings.server.base_url
    ) as test_client:
        yield test_client
