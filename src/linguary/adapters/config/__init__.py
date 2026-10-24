"""Local application configuration: adapter settings and their retrieval.

Configuration sections are defined as dataclasses (`LLMSettings`, `CheckSettings`,
`StorageSettings`, `ServerSettings`, `WorkerPoolSettings`), while sources — the
`~/.linguary/config.toml` file and environment variables — are known only to the loader.
This enables adapters and application use cases to receive resolved values without reading files
or environment variables directly.
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
    DEFAULT_READER_LEVEL,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_WORKER_CONCURRENCY,
    OPENAI_COMPATIBLE_PROVIDER,
    SUPPORTED_READER_LEVELS,
    AppSettings,
    CheckSettings,
    LLMSettings,
    ReadingSettings,
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
    "DEFAULT_READER_LEVEL",
    "DEFAULT_SERVER_HOST",
    "DEFAULT_SERVER_PORT",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_WORKER_CONCURRENCY",
    "ENVIRONMENT_PREFIX",
    "OPENAI_COMPATIBLE_PROVIDER",
    "SUPPORTED_READER_LEVELS",
    "AppSettings",
    "CheckSettings",
    "LLMSettings",
    "ReadingSettings",
    "ServerSettings",
    "SettingsError",
    "StorageSettings",
    "WorkerPoolSettings",
    "load_settings",
]
