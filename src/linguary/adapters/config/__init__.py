"""Локальная конфигурация приложения: настройки адаптеров и их чтение.

Секции настроек описаны датаклассами (`LLMSettings`, `CheckSettings`), а источники —
файл `~/.linguary/config.toml` и переменные окружения — знает только загрузчик. Так
адаптеры и прикладные сценарии получают готовые значения и не читают ни файлов,
ни окружения сами.
"""

from linguary.adapters.config.loader import (
    CONFIG_PATH_VARIABLE,
    DEFAULT_CONFIG_PATH,
    ENVIRONMENT_PREFIX,
    load_settings,
)
from linguary.adapters.config.settings import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT_SECONDS,
    OPENAI_COMPATIBLE_PROVIDER,
    AppSettings,
    CheckSettings,
    LLMSettings,
    SettingsError,
)

__all__ = [
    "CONFIG_PATH_VARIABLE",
    "DEFAULT_BASE_URL",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_TIMEOUT_SECONDS",
    "ENVIRONMENT_PREFIX",
    "OPENAI_COMPATIBLE_PROVIDER",
    "AppSettings",
    "CheckSettings",
    "LLMSettings",
    "SettingsError",
    "load_settings",
]
