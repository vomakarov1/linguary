"""Общие фикстуры интеграционных тестов Web UI.

Приложение собирается тем же composition root, которым его собирает настоящий запуск:
временный каталог данных, база со схемой из миграций Alembic и `FakeLLMClient` вместо
провайдера. Так тест проверяет не только endpoints, но и состав зависимостей — в
частности, что импорт получил хранилище ресурсов, а обработчики задач зарегистрированы.

Фоновый воркер в тестах выключен: задачи выполняются явным `drain`, иначе результат
зависел бы от того, кто успел первым.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
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

#: Набор тестовых EPUB (PRD § 28.1).
FIXTURE_DIRECTORY = Path(__file__).resolve().parents[2] / "fixtures" / "epub"


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """Каталог данных приложения в тесте."""
    return tmp_path / "linguary-data"


@pytest.fixture
def settings(data_root: Path) -> AppSettings:
    """Локальная конфигурация теста: ненастроенная LLM — состояние первого запуска."""
    return AppSettings(
        llm=LLMSettings(),
        checks=CheckSettings(),
        storage=StorageSettings(data_root=str(data_root)),
        server=ServerSettings(),
    )


@pytest.fixture
def llm_client() -> FakeLLMClient:
    """Клиент модели без заготовленных ответов: тесты этого пакета модель не вызывают."""
    return FakeLLMClient()


@pytest.fixture
def web_task_ids() -> Callable[[], str]:
    """Предсказуемые идентификаторы задач: `task-1`, `task-2`, ..."""
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
    """Собранное приложение поверх временного каталога данных."""
    built = build_application(
        settings=settings, llm_client=llm_client, task_id_factory=web_task_ids
    )
    try:
        yield built
    finally:
        built.dispose()


@pytest.fixture
def client(application: Application) -> Iterator[TestClient]:
    """Тестовый клиент Web UI без фонового воркера."""
    app = create_app(application=application, run_background_worker=False)
    with TestClient(app) as test_client:
        yield test_client
