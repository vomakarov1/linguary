"""Локальная конфигурация: чтение файла и окружения, готовность LLM, защита ключа."""

from __future__ import annotations

from pathlib import Path

import pytest

from linguary.adapters.config import (
    CONFIG_PATH_VARIABLE,
    CheckSettings,
    LLMSettings,
    SettingsError,
    load_settings,
)
from linguary.adapters.config.settings import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_ATTEMPTS,
    OPENAI_COMPATIBLE_PROVIDER,
)
from linguary.application.ports.llm import LLMConfigurationError

CONFIG_FILE = """
[llm]
base_url = "https://gateway.test/v1"
model = "gateway/model"
timeout_seconds = 30.0
max_output_tokens = 4096
max_attempts = 5
max_fragment_blocks = 25
max_unit_source_chars = 6000

[checks]
glossary_confidence_threshold = 0.5
language_check_min_length = 120
"""


def write_config(tmp_path: Path, content: str) -> Path:
    """Кладёт файл настроек во временный каталог."""
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_defaults_are_used_without_file_and_environment(tmp_path: Path) -> None:
    """Без заданных значений и переменных окружения берутся значения по умолчанию."""
    settings = load_settings(environ={}, config_path=write_config(tmp_path, ""))

    assert settings.llm.provider == OPENAI_COMPATIBLE_PROVIDER
    assert settings.llm.base_url == DEFAULT_BASE_URL
    assert settings.llm.max_attempts == DEFAULT_MAX_ATTEMPTS
    assert settings.llm.model == ""
    assert settings.llm.api_key is None
    assert settings.checks == CheckSettings()


def test_missing_default_file_is_not_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Отсутствие файла настроек по умолчанию — обычное состояние первого запуска."""
    monkeypatch.setenv("HOME", str(tmp_path))

    settings = load_settings(environ={})

    assert settings.llm.is_ready is False


def test_explicitly_configured_file_must_exist(tmp_path: Path) -> None:
    """Явно указанный файл настроек обязан существовать: подменять его нельзя."""
    with pytest.raises(SettingsError):
        load_settings(
            environ={CONFIG_PATH_VARIABLE: str(tmp_path / "нет-такого.toml")}
        )


def test_settings_are_read_from_file(tmp_path: Path) -> None:
    """Значения секций файла попадают в настройки."""
    settings = load_settings(
        environ={}, config_path=write_config(tmp_path, CONFIG_FILE)
    )

    assert settings.llm.base_url == "https://gateway.test/v1"
    assert settings.llm.model == "gateway/model"
    assert settings.llm.timeout_seconds == 30.0
    assert settings.llm.max_output_tokens == 4096
    assert settings.llm.max_attempts == 5
    assert settings.llm.max_fragment_blocks == 25
    assert settings.llm.max_unit_source_chars == 6000
    assert settings.checks.glossary_confidence_threshold == 0.5
    assert settings.checks.language_check_min_length == 120


def test_environment_overrides_file(tmp_path: Path) -> None:
    """Переменные окружения имеют приоритет над файлом настроек."""
    settings = load_settings(
        environ={
            "LINGUARY_LLM_MODEL": "local/model",
            "LINGUARY_LLM_API_KEY": "key-from-environment",
            "LINGUARY_LLM_MAX_ATTEMPTS": "2",
            "LINGUARY_CHECKS_TARGET_LANGUAGE_MIN_SHARE": "0.75",
        },
        config_path=write_config(tmp_path, CONFIG_FILE),
    )

    assert settings.llm.model == "local/model"
    assert settings.llm.api_key == "key-from-environment"
    assert settings.llm.max_attempts == 2
    assert settings.checks.target_language_min_share == 0.75
    assert settings.llm.base_url == "https://gateway.test/v1"


def test_blank_environment_variable_keeps_file_value(tmp_path: Path) -> None:
    """Пустая переменная окружения не затирает значение из файла."""
    settings = load_settings(
        environ={"LINGUARY_LLM_MODEL": "   "},
        config_path=write_config(tmp_path, CONFIG_FILE),
    )

    assert settings.llm.model == "gateway/model"


def test_api_key_is_read_from_environment(tmp_path: Path) -> None:
    """Ключ провайдера можно задать только окружением, не храня его в файле."""
    settings = load_settings(
        environ={"LINGUARY_LLM_API_KEY": "secret"},
        config_path=write_config(tmp_path, CONFIG_FILE),
    )

    assert settings.llm.is_ready is True


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("[llm]\nmodell = 'x'\n", id="unknown-key"),
        pytest.param("[llm]\ntimeout_seconds = 'долго'\n", id="wrong-type"),
        pytest.param("[llm]\nmax_attempts = 0\n", id="invalid-value"),
        pytest.param("[llm]\nmax_fragment_blocks = 0\n", id="empty-fragment"),
        pytest.param("[llm]\nmax_unit_source_chars = 0\n", id="empty-unit"),
        pytest.param("[llm]\n= сломано\n", id="broken-toml"),
        pytest.param("llm = 'строка'\n", id="section-is-not-a-table"),
    ],
)
def test_invalid_configuration_is_reported(tmp_path: Path, content: str) -> None:
    """Непригодная конфигурация сообщается, а не игнорируется молча."""
    with pytest.raises(SettingsError):
        load_settings(environ={}, config_path=write_config(tmp_path, content))


def test_configuration_error_does_not_expose_api_key(tmp_path: Path) -> None:
    """Ошибка значения не выносит наружу заданный ключ провайдера."""
    with pytest.raises(SettingsError) as error:
        load_settings(
            environ={"LINGUARY_LLM_MAX_ATTEMPTS": "секрет-в-переменной"},
            config_path=write_config(tmp_path, CONFIG_FILE),
        )

    assert "секрет-в-переменной" not in str(error.value)


def test_api_key_is_not_shown_in_repr() -> None:
    """Ключ провайдера не попадает в отладочное представление настроек (PRD § 24)."""
    settings = LLMSettings(model="test-model", api_key="secret-key")

    assert "secret-key" not in repr(settings)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"model": ""}, id="model-not-set"),
        pytest.param({"api_key": None}, id="key-not-set"),
    ],
)
def test_unready_llm_blocks_analysis_and_processing(overrides: dict[str, object]) -> None:
    """Ненастроенная LLM не даёт запустить анализ и обработку (PRD § 23)."""
    values: dict[str, object] = {"model": "test-model", "api_key": "secret"}
    values.update(overrides)
    settings = LLMSettings(**values)  # type: ignore[arg-type]

    assert settings.is_ready is False
    with pytest.raises(LLMConfigurationError) as error:
        settings.ensure_ready()
    assert "secret" not in str(error.value)


def test_ready_llm_passes_the_check() -> None:
    """Настроенные модель и ключ проверку проходят."""
    settings = LLMSettings(model="test-model", api_key="secret")

    settings.ensure_ready()

    assert settings.is_ready is True


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"glossary_confidence_threshold": 1.5}, id="threshold-out-of-range"),
        pytest.param({"target_language_min_share": -0.1}, id="share-out-of-range"),
        pytest.param({"repetition_ngram_size": 0}, id="empty-ngram"),
        pytest.param({"repetition_min_repeats": 1}, id="single-occurrence-is-not-a-repeat"),
    ],
)
def test_invalid_check_thresholds_are_rejected(overrides: dict[str, object]) -> None:
    """Пороги проверок вне допустимых диапазонов не принимаются (PRD § 15)."""
    with pytest.raises(SettingsError):
        CheckSettings(**overrides)  # type: ignore[arg-type]
