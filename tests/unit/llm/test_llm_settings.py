"""Local configuration: file and environment parsing, LLM readiness, key protection."""

from __future__ import annotations

from pathlib import Path

import pytest

from linguary.adapters.config import (
    CONFIG_PATH_VARIABLE,
    CheckSettings,
    DEFAULT_READER_LEVEL,
    LLMSettings,
    ReadingSettings,
    SUPPORTED_READER_LEVELS,
    SettingsError,
    load_settings,
)
from linguary.adapters.config.settings import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT_SECONDS,
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
    """Writes a settings file to a temporary directory."""
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_defaults_are_used_without_file_and_environment(tmp_path: Path) -> None:
    """Default values are used when file and environment variables are absent."""
    settings = load_settings(environ={}, config_path=write_config(tmp_path, ""))

    assert settings.llm.provider == OPENAI_COMPATIBLE_PROVIDER
    assert settings.llm.base_url == DEFAULT_BASE_URL
    assert settings.llm.max_attempts == DEFAULT_MAX_ATTEMPTS
    assert settings.llm.model == ""
    assert settings.llm.api_key is None
    assert settings.llm.debug_llm_payloads is False
    assert settings.checks == CheckSettings()


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("1", True), ("yes", True), ("false", False), ("0", False), ("no", False)],
)
def test_debug_llm_payloads_parsed_from_env(
    tmp_path: Path, value: str, expected: bool
) -> None:
    settings = load_settings(
        environ={"LINGUARY_LLM_DEBUG_LLM_PAYLOADS": value},
        config_path=write_config(tmp_path, ""),
    )
    assert settings.llm.debug_llm_payloads is expected


def test_prices_must_be_set_together() -> None:
    with pytest.raises(SettingsError):
        LLMSettings(input_price_per_mtok=3)
    with pytest.raises(SettingsError):
        LLMSettings(output_price_per_mtok=15)


def test_explicit_max_fragment_blocks_wins() -> None:
    assert LLMSettings(max_output_tokens=32000, max_fragment_blocks=25).effective_max_fragment_blocks == 25
    assert LLMSettings(max_output_tokens=32000).effective_max_fragment_blocks == 160


def test_default_timeout_is_600() -> None:
    assert DEFAULT_TIMEOUT_SECONDS == 600.0
    assert LLMSettings().timeout_seconds == 600.0


def test_prices_are_read_from_environment(tmp_path: Path) -> None:
    settings = load_settings(
        environ={
            "LINGUARY_LLM_INPUT_PRICE_PER_MTOK": "3",
            "LINGUARY_LLM_OUTPUT_PRICE_PER_MTOK": "15",
        },
        config_path=write_config(tmp_path, ""),
    )
    assert settings.llm.input_price_per_mtok == 3
    assert settings.llm.output_price_per_mtok == 15


def test_missing_default_file_is_not_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Absence of default settings file is the normal initial launch state."""
    monkeypatch.setenv("HOME", str(tmp_path))

    settings = load_settings(environ={})

    assert settings.llm.is_ready is False


def test_explicitly_configured_file_must_exist(tmp_path: Path) -> None:
    """An explicitly specified settings file must exist: it cannot be substituted."""
    with pytest.raises(SettingsError):
        load_settings(
            environ={CONFIG_PATH_VARIABLE: str(tmp_path / "нет-такого.toml")}
        )


def test_settings_are_read_from_file(tmp_path: Path) -> None:
    """Values from file sections are loaded into settings."""
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
    """Environment variables take precedence over configuration files."""
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
    """An empty environment variable does not overwrite a file value."""
    settings = load_settings(
        environ={"LINGUARY_LLM_MODEL": "   "},
        config_path=write_config(tmp_path, CONFIG_FILE),
    )

    assert settings.llm.model == "gateway/model"


def test_api_key_is_read_from_environment(tmp_path: Path) -> None:
    """Provider API key can be set via environment without storing in a file."""
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
    """Invalid configuration is reported rather than silently ignored."""
    with pytest.raises(SettingsError):
        load_settings(environ={}, config_path=write_config(tmp_path, content))


def test_configuration_error_does_not_expose_api_key(tmp_path: Path) -> None:
    """A value error does not leak the configured provider key."""
    with pytest.raises(SettingsError) as error:
        load_settings(
            environ={"LINGUARY_LLM_MAX_ATTEMPTS": "секрет-в-переменной"},
            config_path=write_config(tmp_path, CONFIG_FILE),
        )

    assert "секрет-в-переменной" not in str(error.value)


def test_api_key_is_not_shown_in_repr() -> None:
    """Provider key is not exposed in settings debug representation (PRD § 24)."""
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
    """An unconfigured LLM blocks analysis and processing (PRD § 23)."""
    values: dict[str, object] = {"model": "test-model", "api_key": "secret"}
    values.update(overrides)
    settings = LLMSettings(**values)  # type: ignore[arg-type]

    assert settings.is_ready is False
    with pytest.raises(LLMConfigurationError) as error:
        settings.ensure_ready()
    assert "secret" not in str(error.value)


def test_ready_llm_passes_the_check() -> None:
    """Properly configured model and key pass the check."""
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
    """Check thresholds outside valid ranges are rejected (PRD § 15)."""
    with pytest.raises(SettingsError):
        CheckSettings(**overrides)  # type: ignore[arg-type]


def test_worker_and_rate_limit_are_off_by_default(tmp_path: Path) -> None:
    """By default processing runs as a single worker and request rate is unlimited (PRD § 25)."""
    settings = load_settings(environ={}, config_path=write_config(tmp_path, ""))

    assert settings.worker.concurrency == 1
    assert settings.llm.requests_per_minute is None


def test_worker_concurrency_and_rate_limit_are_read_from_file(tmp_path: Path) -> None:
    """Processing concurrency is controlled by one setting, rate limit by another."""
    settings = load_settings(
        environ={},
        config_path=write_config(
            tmp_path, "[worker]\nconcurrency = 4\n\n[llm]\nrequests_per_minute = 30\n"
        ),
    )

    assert settings.worker.concurrency == 4
    assert settings.llm.requests_per_minute == 30


def test_worker_concurrency_is_read_from_environment(tmp_path: Path) -> None:
    """Worker concurrency can be set via environment variable, overriding the file."""
    settings = load_settings(
        environ={
            "LINGUARY_WORKER_CONCURRENCY": "3",
            "LINGUARY_LLM_REQUESTS_PER_MINUTE": "12",
        },
        config_path=write_config(tmp_path, "[worker]\nconcurrency = 2\n"),
    )

    assert settings.worker.concurrency == 3
    assert settings.llm.requests_per_minute == 12


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("[worker]\nconcurrency = 0\n", id="zero-concurrency"),
        pytest.param("[worker]\nconcurrency = 'много'\n", id="concurrency-not-a-number"),
        pytest.param("[worker]\nparallel = 2\n", id="unknown-worker-key"),
        pytest.param("[llm]\nrequests_per_minute = 0\n", id="zero-rate"),
        pytest.param("[llm]\nrequests_per_minute = -5\n", id="negative-rate"),
    ],
)
def test_invalid_worker_and_rate_settings_are_reported(tmp_path: Path, content: str) -> None:
    """Invalid concurrency or rate limit values cause a configuration error."""
    with pytest.raises(SettingsError):
        load_settings(environ={}, config_path=write_config(tmp_path, content))


def test_analysis_and_processing_models_default_to_empty() -> None:
    """By default, separate models are not configured."""
    settings = LLMSettings()
    assert settings.analysis_model == ""
    assert settings.processing_model == ""


def test_effective_models_fallback_to_general_model() -> None:
    """Without explicit settings, effective models fall back to the general model."""
    settings = LLMSettings(model="base/model")
    assert settings.effective_analysis_model == "base/model"
    assert settings.effective_processing_model == "base/model"


def test_effective_models_use_explicit_values() -> None:
    """Explicit settings override the general model for analysis and processing."""
    settings = LLMSettings(
        model="base/model",
        analysis_model="deep/analysis",
        processing_model="fast/translation",
    )
    assert settings.effective_analysis_model == "deep/analysis"
    assert settings.effective_processing_model == "fast/translation"


def test_analysis_and_processing_models_are_read_from_toml(tmp_path: Path) -> None:
    """Separate models are read from the [llm] section of TOML configuration."""
    content = """
    [llm]
    model = "default-model"
    analysis_model = "analysis-model"
    processing_model = "processing-model"
    """
    settings = load_settings(environ={}, config_path=write_config(tmp_path, content))
    assert settings.llm.model == "default-model"
    assert settings.llm.analysis_model == "analysis-model"
    assert settings.llm.processing_model == "processing-model"
    assert settings.llm.effective_analysis_model == "analysis-model"
    assert settings.llm.effective_processing_model == "processing-model"


def test_analysis_and_processing_models_are_read_from_environment(tmp_path: Path) -> None:
    """Separate models are read from environment variables."""
    settings = load_settings(
        environ={
            "LINGUARY_LLM_MODEL": "default-model",
            "LINGUARY_LLM_ANALYSIS_MODEL": "env-analysis-model",
            "LINGUARY_LLM_PROCESSING_MODEL": "env-processing-model",
        },
        config_path=write_config(tmp_path, ""),
    )
    assert settings.llm.model == "default-model"
    assert settings.llm.analysis_model == "env-analysis-model"
    assert settings.llm.processing_model == "env-processing-model"
    assert settings.llm.effective_analysis_model == "env-analysis-model"
    assert settings.llm.effective_processing_model == "env-processing-model"


def test_loader_parses_proxy_mode_and_proxy_url(tmp_path: Path) -> None:
    """Parameters proxy_mode and proxy_url are read from the [llm] TOML section."""
    content = """
    [llm]
    proxy_mode = "custom"
    proxy_url = "http://127.0.0.1:8080"
    """
    settings = load_settings(environ={}, config_path=write_config(tmp_path, content))
    assert settings.llm.proxy_mode == "custom"
    assert settings.llm.proxy_url == "http://127.0.0.1:8080"


@pytest.mark.parametrize("mode", ["auto", "direct"])
def test_system_settings_update_ignores_proxy_url_when_mode_not_custom(mode: str) -> None:
    """For modes other than 'custom', the provided proxy_url is cleared to None."""
    from linguary.adapters.config.system_settings import SystemSettingsUpdate

    update = SystemSettingsUpdate(
        provider="openai_compatible",
        model="test-model",
        base_url="http://localhost:8080/v1",
        requests_per_minute=30,
        concurrency=2,
        proxy_mode=mode,
        proxy_url="http://stale-proxy.local:8080",
    )
    assert update.proxy_mode == mode
    assert update.proxy_url is None


def test_reading_settings_default() -> None:
    """Default reader level is B2."""
    settings = ReadingSettings()
    assert settings.reader_level == DEFAULT_READER_LEVEL
    assert settings.normalized_level == "b2"


@pytest.mark.parametrize("level", ["a2", "b1", "b2", "c1", " A2 ", "B1", "B2", "C1"])
def test_reading_settings_valid_levels(level: str) -> None:
    """Supported CEFR levels are accepted with case and whitespace normalization."""
    settings = ReadingSettings(reader_level=level)
    assert settings.normalized_level == level.strip().lower()


@pytest.mark.parametrize("invalid", ["", "a1", "c2", "beginner", "advanced", "xyz"])
def test_reading_settings_invalid_level_raises(invalid: str) -> None:
    """An invalid reader level raises SettingsError."""
    with pytest.raises(SettingsError, match="недопустимый уровень читателя"):
        ReadingSettings(reader_level=invalid)


def test_reading_settings_loaded_from_toml(tmp_path: Path) -> None:
    """The [reading] section in TOML is loaded into AppSettings.reading."""
    content = """
    [reading]
    reader_level = "b1"
    """
    settings = load_settings(environ={}, config_path=write_config(tmp_path, content))
    assert settings.reading.reader_level == "b1"
    assert settings.reading.normalized_level == "b1"


def test_reading_settings_loaded_from_environment(tmp_path: Path) -> None:
    """Environment variable LINGUARY_READING_READER_LEVEL overrides reader level."""
    settings = load_settings(
        environ={"LINGUARY_READING_READER_LEVEL": "a2"},
        config_path=write_config(tmp_path, ""),
    )
    assert settings.reading.reader_level == "a2"
    assert settings.reading.normalized_level == "a2"



