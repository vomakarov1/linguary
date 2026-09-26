"""Чтение локальной конфигурации: файл настроек и переменные окружения.

Источники читаются по возрастанию приоритета: значения по умолчанию, затем файл
`~/.linguary/config.toml`, затем окружение. Такой порядок позволяет держать общие
настройки в файле, а ключ провайдера — только в окружении (PRD § 23).

Файла может не быть — это обычное состояние первого запуска. Но путь, заданный явно
переменной `LINGUARY_CONFIG`, обязан существовать: подставлять вместо указанного файла
другой нельзя, иначе приложение молча заработает не на той конфигурации.

Опечатка в имени настройки — ошибка, а не молчаливо проигнорированная строка: иначе
пользователь ищет причину в приложении, а не в своём файле. Значения настроек в тексты
ошибок не попадают: одно из них — ключ провайдера.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from linguary.adapters.config.settings import (
    AppSettings,
    CheckSettings,
    LLMSettings,
    ServerSettings,
    SettingsError,
    StorageSettings,
    WorkerPoolSettings,
)

#: Переменная окружения с путём к файлу настроек.
CONFIG_PATH_VARIABLE = "LINGUARY_CONFIG"

#: Файл настроек по умолчанию. Лежит вне репозитория: в нём может быть ключ провайдера.
DEFAULT_CONFIG_PATH = "~/.linguary/config.toml"

#: Общий префикс переменных окружения приложения.
ENVIRONMENT_PREFIX = "LINGUARY"


def _as_str(value: object, name: str) -> str:
    """Строковое значение настройки."""
    if not isinstance(value, str):
        raise SettingsError(f"настройка {name} должна быть строкой")
    return value


def _as_float(value: object, name: str) -> float:
    """Дробное значение настройки; строка из окружения приводится к числу."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise SettingsError(f"настройка {name} должна быть числом")
    try:
        return float(value)
    except ValueError as error:
        raise SettingsError(f"настройка {name} должна быть числом") from error


def _as_int(value: object, name: str) -> int:
    """Целое значение настройки; строка из окружения приводится к числу."""
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise SettingsError(f"настройка {name} должна быть целым числом")
    try:
        return int(value)
    except ValueError as error:
        raise SettingsError(f"настройка {name} должна быть целым числом") from error


#: Разбор значений секции `[llm]`: имя настройки и способ привести её значение к типу.
_LLM_FIELDS: Mapping[str, Callable[[object, str], object]] = {
    "provider": _as_str,
    "base_url": _as_str,
    "model": _as_str,
    "api_key": _as_str,
    "timeout_seconds": _as_float,
    "max_output_tokens": _as_int,
    "temperature": _as_float,
    "max_attempts": _as_int,
    "max_fragment_blocks": _as_int,
    "max_unit_source_chars": _as_int,
    "requests_per_minute": _as_int,
}

#: Разбор значений секции `[checks]`.
_CHECK_FIELDS: Mapping[str, Callable[[object, str], object]] = {
    "glossary_confidence_threshold": _as_float,
    "target_language_min_share": _as_float,
    "language_check_min_length": _as_int,
    "repetition_ngram_size": _as_int,
    "repetition_min_repeats": _as_int,
}

#: Разбор значений секции `[storage]`.
_STORAGE_FIELDS: Mapping[str, Callable[[object, str], object]] = {
    "data_root": _as_str,
    "database_path": _as_str,
}

#: Разбор значений секции `[server]`.
_SERVER_FIELDS: Mapping[str, Callable[[object, str], object]] = {
    "host": _as_str,
    "port": _as_int,
}

#: Разбор значений секции `[worker]`.
_WORKER_FIELDS: Mapping[str, Callable[[object, str], object]] = {
    "concurrency": _as_int,
}


def load_settings(
    *,
    environ: Mapping[str, str] | None = None,
    config_path: str | Path | None = None,
) -> AppSettings:
    """Собирает локальную конфигурацию из файла настроек и окружения.

    `environ` и `config_path` задаются явно в тестах и при сборке приложения; без них
    берутся окружение процесса и путь по умолчанию.
    """
    variables = os.environ if environ is None else environ
    path, is_explicit = _config_path(variables, config_path)
    document = _read_document(path, is_explicit=is_explicit)
    return AppSettings(
        llm=LLMSettings(
            **_section(
                document,
                name="llm",
                fields=_LLM_FIELDS,
                variables=variables,
            )
        ),
        checks=CheckSettings(
            **_section(
                document,
                name="checks",
                fields=_CHECK_FIELDS,
                variables=variables,
            )
        ),
        storage=StorageSettings(
            **_section(
                document,
                name="storage",
                fields=_STORAGE_FIELDS,
                variables=variables,
            )
        ),
        server=ServerSettings(
            **_section(
                document,
                name="server",
                fields=_SERVER_FIELDS,
                variables=variables,
            )
        ),
        worker=WorkerPoolSettings(
            **_section(
                document,
                name="worker",
                fields=_WORKER_FIELDS,
                variables=variables,
            )
        ),
    )


def _config_path(
    variables: Mapping[str, str], config_path: str | Path | None
) -> tuple[Path, bool]:
    """Путь к файлу настроек и признак того, что его выбрал пользователь."""
    if config_path is not None:
        return Path(config_path).expanduser(), True
    configured = variables.get(CONFIG_PATH_VARIABLE, "").strip()
    if configured:
        return Path(configured).expanduser(), True
    return Path(DEFAULT_CONFIG_PATH).expanduser(), False


def _read_document(path: Path, *, is_explicit: bool) -> Mapping[str, object]:
    """Разбирает файл настроек; отсутствие файла по умолчанию — не ошибка."""
    try:
        content = path.read_bytes()
    except FileNotFoundError as error:
        if is_explicit:
            raise SettingsError(f"файл настроек {path} не найден") from error
        return {}
    except OSError as error:
        raise SettingsError(f"файл настроек {path} нельзя прочитать: {error}") from error
    try:
        return tomllib.loads(content.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise SettingsError(f"файл настроек {path} не разбирается: {error}") from error


def _section(
    document: Mapping[str, object],
    *,
    name: str,
    fields: Mapping[str, Callable[[object, str], object]],
    variables: Mapping[str, str],
) -> dict[str, Any]:
    """Значения одной секции: сначала из файла, затем перекрытые окружением."""
    values = {**_from_document(document, name=name, fields=fields)}
    values.update(_from_environment(name=name, fields=fields, variables=variables))
    return values


def _from_document(
    document: Mapping[str, object],
    *,
    name: str,
    fields: Mapping[str, Callable[[object, str], object]],
) -> dict[str, Any]:
    """Значения секции файла настроек."""
    section = document.get(name, {})
    if not isinstance(section, Mapping):
        raise SettingsError(f"секция [{name}] файла настроек не является таблицей")
    unknown = sorted(set(section) - set(fields))
    if unknown:
        raise SettingsError(
            f"неизвестные настройки в секции [{name}]: {', '.join(unknown)}"
        )
    return {
        key: fields[key](value, f"[{name}] {key}") for key, value in section.items()
    }


def _from_environment(
    *,
    name: str,
    fields: Mapping[str, Callable[[object, str], object]],
    variables: Mapping[str, str],
) -> dict[str, Any]:
    """Значения секции, заданные переменными окружения.

    Пустая переменная считается незаданной: так пользователь снимает значение из окружения,
    не удаляя саму переменную, и остаётся то, что задано файлом.
    """
    values: dict[str, Any] = {}
    for key, parse in fields.items():
        variable = f"{ENVIRONMENT_PREFIX}_{name.upper()}_{key.upper()}"
        raw = variables.get(variable, "").strip()
        if raw:
            values[key] = parse(raw, variable)
    return values


__all__ = [
    "CONFIG_PATH_VARIABLE",
    "DEFAULT_CONFIG_PATH",
    "ENVIRONMENT_PREFIX",
    "load_settings",
]
