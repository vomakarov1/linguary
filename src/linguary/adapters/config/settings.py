"""Local application settings: LLM, worker, and automated check thresholds.

Settings are organized into sections and aggregated in `AppSettings`. Each adapter receives its
dedicated section, so introducing a new adapter adds a section rather than altering the common
contract: settings that currently live adjacent to their adapters (`ExtractionLimits` for import,
`EpubCheckSettings` for external validation) can later be migrated here following the same pattern.

The provider API key is a secret: it is excluded from `repr`, error text, and invocation logs.
The initial key is loaded from the environment or a local configuration file outside the
repository; overrides made via the web interface are persisted in SQLite within the data directory.
"""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import quote, unquote, urlsplit

from linguary.application.analysis_merge import fragment_blocks_for
from linguary.application.errors import ApplicationError
from linguary.application.ports.llm import LLMConfigurationError
from linguary.application.translation_units import DEFAULT_MAX_UNIT_SOURCE_CHARS
from linguary.domain.name_translation import NameTranslationStrategy

#: Provider of the initial implementation: HTTP API compatible with OpenAI chat completions
#: and structured outputs (ADR § 3). The same adapter serves compatible gateways and local
#: servers, so they do not have separate provider names.
OPENAI_COMPATIBLE_PROVIDER = "openai_compatible"

#: Default base URL. Local server or gateway is configured via settings.
DEFAULT_BASE_URL = "https://api.openai.com/v1"

#: How long to wait for a model response per call. Literary chapter translation is lengthy,
#: so the limit is generous: it protects against stalled connections rather than optimizing for time.
# 10,000 output tokens at 53 tokens/s take about three minutes;
# the reserve allows time for provider latency and response transmission.
DEFAULT_TIMEOUT_SECONDS = 600.0

#: How many times a task is retried automatically before transitioning to `failed` (PRD § 16).
DEFAULT_MAX_ATTEMPTS = 3

#: Directory where the application keeps the database, source books, resources, and built EPUBs
#: (PRD § 22). Located adjacent to the config file, outside the repository.
DEFAULT_DATA_ROOT = "~/.linguary"

#: Database file name inside the data directory.
DEFAULT_DATABASE_NAME = "linguary.db"

#: Host where the local Web UI runs. The application stores the provider API key and full book text,
#: so by default it is accessible only from the same machine; exposing it externally is possible only
#: by explicitly configuring a different address in the local configuration.
DEFAULT_SERVER_HOST = "127.0.0.1"

#: Loopback interface hostnames: the Web UI can be opened on this machine using any of them.
_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")

#: Addresses where the server listens on all interfaces simultaneously. The application does not
#: know the machine's network hostnames, so it cannot restrict host headers for them.
_WILDCARD_HOSTS = ("0.0.0.0", "::", "*")

#: Local Web UI port.
DEFAULT_SERVER_PORT = 8000

#: How many tasks the worker executes concurrently (PRD § 25). A single task is predictable
#: and avoids hitting provider rate limits; increasing concurrency speeds up processing.
DEFAULT_WORKER_CONCURRENCY = 1

#: Default reader level (PRD § 30).
DEFAULT_READER_LEVEL = "b2"

#: Supported CEFR reader language proficiency levels.
SUPPORTED_READER_LEVELS = frozenset({"a2", "b1", "b2", "c1"})


class SettingsError(ApplicationError):
    """Local configuration is unusable: unparseable file or invalid value."""


def validate_proxy(mode: object, url: object) -> None:
    """Validate proxy URL without exposing credentials in exceptions."""
    if mode not in ('auto', 'direct', 'custom'):
        raise SettingsError('Недопустимый режим соединения')
    if mode != 'custom':
        if url is not None:
            raise SettingsError('Адрес прокси допустим только для режима «Через прокси»')
        return
    try:
        parsed = urlsplit(url) if isinstance(url, str) else None
        valid = (parsed is not None and parsed.scheme in ('http', 'https')
                 and bool(parsed.hostname) and parsed.path in ('', '/')
                 and '?' not in url and '#' not in url
                 and not any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url))
        if parsed is not None:
            parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise SettingsError('Ожидается HTTP(S)-адрес прокси без пути, параметров и фрагмента')


def proxy_secrets(url: object) -> tuple[str, ...]:
    """Full URL and credential variants for secure diagnostics."""
    if not isinstance(url, str) or not url.strip():
        return ()
    values = {url, unquote(url)}
    try:
        parsed = urlsplit(url)
        if '@' in parsed.netloc:
            values.add(parsed.netloc.rsplit('@', 1)[0])
        values.update(value for value in (parsed.username, parsed.password) if value)
        if parsed.username is not None or parsed.password is not None:
            credentials = f"{unquote(parsed.username or '')}:{unquote(parsed.password or '')}"
            values.add(b64encode(credentials.encode('utf-8')).decode('ascii'))
    except ValueError:
        pass
    values.update(unquote(value) for value in tuple(values))
    values.update(quote(value, safe='') for value in tuple(values))
    return tuple(sorted((value for value in values if value), key=len, reverse=True))


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """LLM provider configuration (PRD § 23).

    Empty `model` and `api_key` indicate unconfigured LLM — the expected state on first
    launch rather than an error: the application displays configuration status and does not
    send requests. It becomes an error only when attempting to instantiate a client.

    `max_fragment_blocks` — how many chapter text blocks are sent in one analysis request.
    This setting is placed alongside `max_output_tokens` because it is determined by the same limit:
    a chapter whose response exceeds model output limits is analyzed in sequential fragments (PRD § 14).

    `max_unit_source_chars` — how many original characters are sent in a single processing request
    (PRD § 10). Analysis and processing are measured in different units intentionally: analysis
    response costs roughly the same for a block of any length, whereas processing response is a full
    translation whose length is proportional to the original.

    `requests_per_minute` — rate limit for model requests sent across all workers combined.
    When unset, request frequency is not throttled: providers have different limits and the application
    does not guess them. The setting resides in the LLM section because it is a property of the provider
    and pricing tier rather than the worker: the same limit applies under any concurrency.
    """

    provider: str = OPENAI_COMPATIBLE_PROVIDER
    base_url: str = DEFAULT_BASE_URL
    model: str = ""
    analysis_model: str = ""
    processing_model: str = ""
    api_key: str | None = field(default=None, repr=False)
    proxy_mode: Literal["auto", "direct", "custom"] = "auto"
    proxy_url: str | None = field(default=None, repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_output_tokens: int | None = None
    temperature: float | None = None
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    max_fragment_blocks: int | None = None
    max_unit_source_chars: int = DEFAULT_MAX_UNIT_SOURCE_CHARS
    requests_per_minute: int | None = None
    debug_llm_payloads: bool = False
    input_price_per_mtok: float | None = None
    output_price_per_mtok: float | None = None

    def __post_init__(self) -> None:
        validate_proxy(self.proxy_mode, self.proxy_url)
        if not self.provider.strip():
            raise SettingsError("провайдер LLM не задан")
        if not self.base_url.strip():
            raise SettingsError("базовый адрес провайдера LLM не задан")
        if self.timeout_seconds <= 0:
            raise SettingsError("предел ожидания ответа модели должен быть положительным")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise SettingsError("лимит выходных токенов должен быть положительным")
        if self.max_attempts < 1:
            raise SettingsError("число попыток вызова модели должно быть положительным")
        if self.max_fragment_blocks is not None and self.max_fragment_blocks < 1:
            raise SettingsError("размер фрагмента главы должен быть положительным")

        if self.max_unit_source_chars < 1:
            raise SettingsError("размер группы блоков должен быть положительным")
        if self.requests_per_minute is not None and self.requests_per_minute < 1:
            raise SettingsError("предел частоты запросов к модели должен быть положительным")
        if (self.input_price_per_mtok is None) != (self.output_price_per_mtok is None):
            raise SettingsError("цены входных и выходных токенов задаются вместе")
        if self.input_price_per_mtok is not None and self.input_price_per_mtok < 0:
            raise SettingsError("цена входных токенов не может быть отрицательной")
        if self.output_price_per_mtok is not None and self.output_price_per_mtok < 0:
            raise SettingsError("цена выходных токенов не может быть отрицательной")

    @property
    def effective_max_fragment_blocks(self) -> int:
        """Explicit limit or size calculated from available output tokens."""
        return self.max_fragment_blocks or fragment_blocks_for(self.max_output_tokens)

    @property
    def effective_analysis_model(self) -> str:
        """Model for analysis: explicit analysis_model or fallback to model."""
        return self.analysis_model.strip() or self.model

    @property
    def effective_processing_model(self) -> str:
        """Model for processing and regeneration: explicit processing_model or fallback to model."""
        return self.processing_model.strip() or self.model

    @property
    def is_ready(self) -> bool:
        """Whether model and API key are configured to allow provider calls."""
        return bool(self.model.strip()) and bool(self.api_key and self.api_key.strip())

    def ensure_ready(self) -> None:
        """Verify that analysis and processing can be started (PRD § 23).

        The message lists what is missing without exposing the API key itself: it is displayed
        in the UI and CLI diagnostics.
        """
        missing = [
            name
            for name, present in (
                ("модель", bool(self.model.strip())),
                ("ключ провайдера", bool(self.api_key and self.api_key.strip())),
            )
            if not present
        ]
        if missing:
            raise LLMConfigurationError(
                f"LLM не настроена: не задано — {', '.join(missing)}; "
                f"укажите значения в локальной конфигурации или в окружении"
            )


@dataclass(frozen=True, slots=True)
class CheckSettings:
    """Automated check thresholds (PRD § 12, § 15).

    Defaults are selected so that checks detect technical defects rather than literary decisions:
    target alphabet share is checked only on long full translations, while n-gram repetition remains
    a warning.
    """

    glossary_confidence_threshold: float = 0.8
    glossary_review_min_mentions: int = 5
    glossary_review_min_chapters: int = 3
    target_language_min_share: float = 0.6
    language_check_min_length: int = 80
    repetition_ngram_size: int = 6
    repetition_min_repeats: int = 3
    language_notes_per_chapter: int = 8

    def __post_init__(self) -> None:
        for name in ("glossary_confidence_threshold", "target_language_min_share"):
            value: float = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise SettingsError(f"порог {name} вне диапазона 0..1: {value}")
        for name in ("glossary_review_min_mentions", "glossary_review_min_chapters"):
            if getattr(self, name) < 1:
                raise SettingsError(f"порог {name} должен быть положительным")
        if self.language_check_min_length < 0:
            raise SettingsError("минимальная длина проверки языка не может быть отрицательной")
        if self.repetition_ngram_size < 1:
            raise SettingsError("размер n-граммы проверки повторов должен быть положительным")
        if self.repetition_min_repeats < 2:
            raise SettingsError("повтором считается не менее двух вхождений n-граммы")
        if self.language_notes_per_chapter < 0:
            raise SettingsError("число языковых заметок на главу не может быть отрицательным")


@dataclass(frozen=True, slots=True)
class StorageSettings:
    """Where the application stores its data (PRD § 22).

    An empty `database_path` implies storing the database inside the data directory: users can
    relocate the entire application by moving a single directory. A separate path is needed when
    the database is stored away from books — for example on a different drive.
    """

    data_root: str = DEFAULT_DATA_ROOT
    database_path: str = ""

    def __post_init__(self) -> None:
        if not self.data_root.strip():
            raise SettingsError("каталог данных приложения не задан")


@dataclass(frozen=True, slots=True)
class ServerSettings:
    """Local Web UI server address (PRD § 21, § 4.1)."""

    host: str = DEFAULT_SERVER_HOST
    port: int = DEFAULT_SERVER_PORT

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise SettingsError("адрес локального сервера не задан")
        if not 1 <= self.port <= 65535:
            raise SettingsError(f"порт локального сервера вне диапазона: {self.port}")

    @property
    def is_local_only(self) -> bool:
        """Whether the server is accessible only from this machine.

        The user must be aware that an application holding the provider API key and full book text
        is exposed to the network if they explicitly configured such an address.
        """
        return self.host.strip() in _LOOPBACK_HOSTS

    @property
    def allowed_hosts(self) -> tuple[str, ...]:
        """Host header names accepted by the Web UI (PRD § 4.1).

        This list mitigates DNS rebinding: attacker pages reach the application under their own
        domain, and the application does not answer foreign host headers. Local execution is accessed
        via IP or `localhost`, so all loopback interface names are accepted equally. Running on wildcard
        interfaces does not restrict names — `*` allows any host header: the owner configures such an
        address deliberately and sees a warning on the configuration screen.
        """
        host = self.host.strip()
        if self.is_local_only:
            return _LOOPBACK_HOSTS
        if host in _WILDCARD_HOSTS:
            return ("*",)
        return (host,)

    @property
    def base_url(self) -> str:
        """Address where the Web UI can be accessed."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"


@dataclass(frozen=True, slots=True)
class WorkerPoolSettings:
    """Background worker: how many queue tasks are executed concurrently (PRD § 25).

    Concurrency speeds up chapter processing, which has independent data. Chapter analysis
    does not speed up: the next chapter is queued only after persisting the previous one (PRD § 13).
    Higher values may trigger provider rate limits — protected by `LLMSettings.requests_per_minute`.
    """

    concurrency: int = DEFAULT_WORKER_CONCURRENCY

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise SettingsError("параллелизм воркера должен быть положительным")


@dataclass(frozen=True, slots=True)
class ReadingSettings:
    """Reader adaptation and assistance settings (PRD § 30).

    `reader_level` defines the target CEFR language proficiency level:
    - `a2`: elementary (high support: 70–85% TRANSLATE blocks, dense hints);
    - `b1`: intermediate (increased support: 50–70% TRANSLATE blocks, assistance on doubt);
    - `b2`: upper intermediate (default: balanced assistance, up to 50–60% TRANSLATE);
    - `c1`: advanced (minimal assistance: predominantly READ_AS_IS, 10–25% TRANSLATE).
    """

    reader_level: str = DEFAULT_READER_LEVEL
    name_translation_strategy: str = NameTranslationStrategy.SOUND.value

    def __post_init__(self) -> None:
        if type(self.name_translation_strategy) is not str or self.name_translation_strategy not in NameTranslationStrategy._value2member_map_:
            raise SettingsError("недопустимый способ передачи имён")
        normalized = self.reader_level.strip().lower()
        if normalized not in SUPPORTED_READER_LEVELS:
            raise SettingsError(
                f"недопустимый уровень читателя: {self.reader_level!r}. "
                f"Поддерживаются: {', '.join(sorted(SUPPORTED_READER_LEVELS))}"
            )

    @property
    def normalized_level(self) -> str:
        return self.reader_level.strip().lower()


@dataclass(frozen=True, slots=True)
class DiagnosticsSettings:
    bodies_enabled: bool = True
    max_body_bytes: int = 64 * 1024
    max_book_bytes: int = 10 * 1024 * 1024
    retention_seconds: int = 7 * 24 * 60 * 60

    def __post_init__(self) -> None:
        if type(self.bodies_enabled) is not bool:
            raise SettingsError('diagnostics.bodies_enabled должна быть логической настройкой')
        for name in ('max_body_bytes', 'max_book_bytes', 'retention_seconds'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise SettingsError(f'diagnostics.{name} должна быть положительным целым числом')


@dataclass(frozen=True, slots=True)
class RecoverySettings:
    max_calls: int = 12
    max_scope_calls: int = 3
    max_depth: int = 3
    max_leaves: int = 8

    def __post_init__(self) -> None:
        for name in ('max_calls', 'max_scope_calls', 'max_depth', 'max_leaves'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise SettingsError(f'recovery.{name} должна быть положительным целым числом')


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Complete local application configuration."""

    llm: LLMSettings = field(default_factory=LLMSettings)
    checks: CheckSettings = field(default_factory=CheckSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    worker: WorkerPoolSettings = field(default_factory=WorkerPoolSettings)
    reading: ReadingSettings = field(default_factory=ReadingSettings)
    diagnostics: DiagnosticsSettings = field(default_factory=DiagnosticsSettings)
    recovery: RecoverySettings = field(default_factory=RecoverySettings)


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_DATABASE_NAME",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_READER_LEVEL",
    "DEFAULT_SERVER_HOST",
    "DEFAULT_SERVER_PORT",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_WORKER_CONCURRENCY",
    "OPENAI_COMPATIBLE_PROVIDER",
    "SUPPORTED_READER_LEVELS",
    "AppSettings",
    "DiagnosticsSettings",
    "RecoverySettings",
    "CheckSettings",
    "LLMSettings",
    "ReadingSettings",
    "ServerSettings",
    "SettingsError",
    "StorageSettings",
    "WorkerPoolSettings",
]
