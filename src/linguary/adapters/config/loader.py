"""Loading local configuration: config file and environment variables.

Sources are processed in order of increasing precedence: defaults, then the
`~/.linguary/config.toml` file, then environment variables. This hierarchy allows keeping
shared settings in a file while storing the provider API key solely in environment variables (PRD § 23).

The config file may not exist — this is the expected state on first launch. However, an explicitly
specified path via `LINGUARY_CONFIG` must exist: substituting another file in its place is forbidden,
as it would lead to silent execution with incorrect configuration.

A typo in a setting name is treated as an error rather than silently ignored: otherwise, users might
look for issues in the application rather than in their configuration file. Setting values are not
included in error messages because one of them is the provider API key.
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
    DiagnosticsSettings,
    LLMSettings,
    ReadingSettings,
    RecoverySettings,
    ServerSettings,
    SettingsError,
    StorageSettings,
    WorkerPoolSettings,
)

#: Environment variable specifying the config file path.
CONFIG_PATH_VARIABLE = "LINGUARY_CONFIG"

#: Default configuration file path. Located outside the repository as it may contain provider API keys.
DEFAULT_CONFIG_PATH = "~/.linguary/config.toml"

#: Common prefix for application environment variables.
ENVIRONMENT_PREFIX = "LINGUARY"


def _as_str(value: object, name: str) -> str:
    """String setting value."""
    if not isinstance(value, str):
        raise SettingsError(f"настройка {name} должна быть строкой")
    return value


def _as_float(value: object, name: str) -> float:
    """Float setting value; string from environment is converted to a float."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise SettingsError(f"настройка {name} должна быть числом")
    try:
        return float(value)
    except ValueError as error:
        raise SettingsError(f"настройка {name} должна быть числом") from error


def _as_int(value: object, name: str) -> int:
    """Integer setting value; string from environment is converted to an integer."""
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise SettingsError(f"настройка {name} должна быть целым числом")
    try:
        return int(value)
    except ValueError as error:
        raise SettingsError(f"настройка {name} должна быть целым числом") from error


def _as_bool(value: object, name: str) -> bool:
    """Boolean setting value from TOML or environment."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise SettingsError(f"настройка {name} должна быть логическим значением")


#: Parsers for the [llm] section: setting name mapped to value conversion function.
_LLM_FIELDS: Mapping[str, Callable[[object, str], object]] = {
    "provider": _as_str,
    "base_url": _as_str,
    "model": _as_str,
    "analysis_model": _as_str,
    "processing_model": _as_str,
    "api_key": _as_str,
    "proxy_mode": _as_str,
    "proxy_url": _as_str,
    "timeout_seconds": _as_float,
    "max_output_tokens": _as_int,
    "temperature": _as_float,
    "max_attempts": _as_int,
    "max_fragment_blocks": _as_int,
    "max_unit_source_chars": _as_int,
    "requests_per_minute": _as_int,
    "debug_llm_payloads": _as_bool,
    "input_price_per_mtok": _as_float,
    "output_price_per_mtok": _as_float,
}

#: Parsers for the [checks] section.
_CHECK_FIELDS: Mapping[str, Callable[[object, str], object]] = {
    "glossary_confidence_threshold": _as_float,
    "glossary_review_min_mentions": _as_int,
    "glossary_review_min_chapters": _as_int,
    "target_language_min_share": _as_float,
    "language_check_min_length": _as_int,
    "repetition_ngram_size": _as_int,
    "repetition_min_repeats": _as_int,
    "language_notes_per_chapter": _as_int,
}

#: Parsers for the [storage] section.
_STORAGE_FIELDS: Mapping[str, Callable[[object, str], object]] = {
    "data_root": _as_str,
    "database_path": _as_str,
}

#: Parsers for the [server] section.
_SERVER_FIELDS: Mapping[str, Callable[[object, str], object]] = {
    "host": _as_str,
    "port": _as_int,
}

#: Parsers for the [worker] section.
_WORKER_FIELDS: Mapping[str, Callable[[object, str], object]] = {
    "concurrency": _as_int,
}

#: Parsers for the [reading] section.
_READING_FIELDS: Mapping[str, Callable[[object, str], object]] = {
    "reader_level": _as_str,
    "name_translation_strategy": _as_str,
}


#: Parsers for the [diagnostics] section.
_DIAGNOSTICS_FIELDS: Mapping[str, Callable[[object, str], object]] = {
    "bodies_enabled": _as_bool,
    "max_body_bytes": _as_int,
    "max_book_bytes": _as_int,
    "retention_seconds": _as_int,
}

#: Parsers for the [recovery] section.
_RECOVERY_FIELDS: Mapping[str, Callable[[object, str], object]] = {
    "max_calls": _as_int,
    "max_scope_calls": _as_int,
    "max_depth": _as_int,
    "max_leaves": _as_int,
}


def load_settings(
    *,
    environ: Mapping[str, str] | None = None,
    config_path: str | Path | None = None,
) -> AppSettings:
    """Assemble local configuration from config file and environment.

    `environ` and `config_path` are specified explicitly in tests and application composition;
    otherwise process environment and default path are used.
    """
    variables = os.environ if environ is None else environ
    path, is_explicit = _config_path(variables, config_path)
    document = _read_document(path, is_explicit=is_explicit)
    return AppSettings(
        diagnostics=DiagnosticsSettings(
            **_section(
                document,
                name="diagnostics",
                fields=_DIAGNOSTICS_FIELDS,
                variables=variables,
            )
        ),
        recovery=RecoverySettings(
            **_section(
                document,
                name="recovery",
                fields=_RECOVERY_FIELDS,
                variables=variables,
            )
        ),
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
        reading=ReadingSettings(
            **_section(
                document,
                name="reading",
                fields=_READING_FIELDS,
                variables=variables,
            )
        ),
    )


def _config_path(
    variables: Mapping[str, str], config_path: str | Path | None
) -> tuple[Path, bool]:
    """Path to the config file and flag indicating whether it was explicitly specified."""
    if config_path is not None:
        return Path(config_path).expanduser(), True
    configured = variables.get(CONFIG_PATH_VARIABLE, "").strip()
    if configured:
        return Path(configured).expanduser(), True
    return Path(DEFAULT_CONFIG_PATH).expanduser(), False


def _read_document(path: Path, *, is_explicit: bool) -> Mapping[str, object]:
    """Parse configuration file; absence of the default file is not an error."""
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
    """Values for a single section: loaded from document first, then overridden by environment."""
    values = {**_from_document(document, name=name, fields=fields)}
    values.update(_from_environment(name=name, fields=fields, variables=variables))
    return values


def _from_document(
    document: Mapping[str, object],
    *,
    name: str,
    fields: Mapping[str, Callable[[object, str], object]],
) -> dict[str, Any]:
    """Section values loaded from the configuration file."""
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
    """Section values defined by environment variables.

    An empty environment variable is treated as unset: this allows users to clear an environment
    override without unsetting the variable, falling back to what is defined in the file.
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
