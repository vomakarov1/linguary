"""Локальная конфигурация приложения: настройки адаптеров и их чтение.

Секции настроек описаны датаклассами (`LLMSettings`, `CheckSettings`, `StorageSettings`,
`ServerSettings`, `WorkerPoolSettings`), а источники — файл `~/.linguary/config.toml` и переменные окружения —
знает только загрузчик. Так адаптеры и прикладные сценарии получают готовые значения
и не читают ни файлов, ни окружения сами.
"""

from linguary.adapters.config.loader import (
    CONFIG_PATH_VARIABLE,
    DEFAULT_CONFIG_PATH,
    ENVIRONMENT_PREFIX,
    load_settings,
)
from linguary.adapters.config.settings import (
    DEFAULT_BASE_URL,
    DEFAULT_DATA_ROOT,
    DEFAULT_DATABASE_NAME,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_WORKER_CONCURRENCY,
    OPENAI_COMPATIBLE_PROVIDER,
    AppSettings,
    CheckSettings,
    LLMSettings,
    ServerSettings,
    SettingsError,
    StorageSettings,
    WorkerPoolSettings,
)

__all__ = [
    "CONFIG_PATH_VARIABLE",
    "DEFAULT_BASE_URL",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DATABASE_NAME",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_SERVER_HOST",
    "DEFAULT_SERVER_PORT",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_WORKER_CONCURRENCY",
    "ENVIRONMENT_PREFIX",
    "OPENAI_COMPATIBLE_PROVIDER",
    "AppSettings",
    "CheckSettings",
    "LLMSettings",
    "ServerSettings",
    "SettingsError",
    "StorageSettings",
    "WorkerPoolSettings",
    "load_settings",
]
