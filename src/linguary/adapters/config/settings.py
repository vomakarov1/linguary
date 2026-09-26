"""Локальные настройки приложения: LLM, воркер и пороги автоматических проверок.

Настройки описаны секциями и собраны в `AppSettings`. Каждый адаптер получает свою секцию,
поэтому новый адаптер добавляет секцию, а не меняет общий контракт: настройки, которые
пока живут рядом со своими адаптерами (`ExtractionLimits` импорта, `EpubCheckSettings`
внешней проверки), позже переезжают сюда тем же способом.

Ключ провайдера — секрет: он не попадает ни в `repr`, ни в текст ошибок, ни в журнал
вызовов, и хранится только там, откуда прочитан, — в окружении или локальном файле
настроек вне репозитория (PRD § 23).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from linguary.application.analysis_merge import DEFAULT_MAX_FRAGMENT_BLOCKS
from linguary.application.errors import ApplicationError
from linguary.application.ports.llm import LLMConfigurationError
from linguary.application.translation_units import DEFAULT_MAX_UNIT_SOURCE_CHARS

#: Провайдер первой реализации: HTTP API, совместимый с OpenAI chat completions
#: и структурированными ответами (ADR § 3). Тот же адаптер обслуживает совместимые
#: шлюзы и локальные серверы, поэтому отдельного имени провайдера у них нет.
OPENAI_COMPATIBLE_PROVIDER = "openai_compatible"

#: Базовый адрес по умолчанию. Локальный сервер или шлюз задаётся настройкой.
DEFAULT_BASE_URL = "https://api.openai.com/v1"

#: Сколько ждать ответа модели на один вызов. Литературный перевод главы длинный,
#: поэтому предел щедрый: он защищает от зависшего соединения, а не экономит время.
DEFAULT_TIMEOUT_SECONDS = 180.0

#: Сколько раз задача повторяется автоматически до перехода в `failed` (PRD § 16).
DEFAULT_MAX_ATTEMPTS = 3

#: Каталог, в котором приложение держит базу, исходные книги, их ресурсы и собранные EPUB
#: (PRD § 22). Лежит рядом с файлом настроек, вне репозитория.
DEFAULT_DATA_ROOT = "~/.linguary"

#: Имя файла базы внутри каталога данных.
DEFAULT_DATABASE_NAME = "linguary.db"

#: Адрес, на котором работает локальный Web UI. Приложение хранит ключ провайдера и полный
#: текст книги, поэтому по умолчанию оно доступно только с этой же машины; выставить его
#: наружу можно единственным способом — явно задав другой адрес в локальной конфигурации.
DEFAULT_SERVER_HOST = "127.0.0.1"

#: Имена петлевого интерфейса: по любому из них Web UI открывается на этой же машине.
_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")

#: Адреса, на которых сервер слушает сразу все интерфейсы. Как машину зовут в сети,
#: приложение не знает, поэтому имена обращений к нему оно ограничить не может.
_WILDCARD_HOSTS = ("0.0.0.0", "::", "*")

#: Порт локального Web UI.
DEFAULT_SERVER_PORT = 8000

#: Сколько задач воркер выполняет одновременно (PRD § 25). Одна задача предсказуема
#: и не упирается в ограничения частоты провайдера; ускорение — изменение этой настройки.
DEFAULT_WORKER_CONCURRENCY = 1


class SettingsError(ApplicationError):
    """Локальная конфигурация непригодна: неразбираемый файл или недопустимое значение."""


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Настройка провайдера LLM (PRD § 23).

    Пустые `model` и `api_key` означают ненастроенную LLM — обычное состояние первого
    запуска, а не ошибка: приложение показывает состояние конфигурации и не отправляет
    запросов. Ошибкой оно становится в момент попытки создать клиент.

    `max_fragment_blocks` — сколько текстовых блоков главы уходит в один запрос анализа.
    Настройка стоит рядом с `max_output_tokens`, потому что определяется тем же пределом:
    глава, ответ на которую не помещается в лимит вывода модели, анализируется
    последовательными фрагментами (PRD § 14).

    `max_unit_source_chars` — сколько символов оригинала уходит в один запрос обработки
    (PRD § 10). Анализ и обработка меряются разными величинами не случайно: ответ анализа
    стоит примерно одинаково для блока любой длины, а ответ обработки — это полный
    перевод, объём которого пропорционален объёму оригинала.

    `requests_per_minute` — сколько запросов к модели приложение отправляет за минуту
    всеми воркерами вместе. Не задано — частота не ограничивается: пределы у провайдеров
    разные, и угадывать их приложение не берётся. Настройка живёт в секции LLM, потому
    что это свойство провайдера и тарифа, а не воркера: тот же предел действует при любом
    параллелизме.
    """

    provider: str = OPENAI_COMPATIBLE_PROVIDER
    base_url: str = DEFAULT_BASE_URL
    model: str = ""
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_output_tokens: int | None = None
    temperature: float | None = None
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    max_fragment_blocks: int = DEFAULT_MAX_FRAGMENT_BLOCKS
    max_unit_source_chars: int = DEFAULT_MAX_UNIT_SOURCE_CHARS
    requests_per_minute: int | None = None
    debug_llm_payloads: bool = False

    def __post_init__(self) -> None:
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
        if self.max_fragment_blocks < 1:
            raise SettingsError("размер фрагмента главы должен быть положительным")
        if self.max_unit_source_chars < 1:
            raise SettingsError("размер группы блоков должен быть положительным")
        if self.requests_per_minute is not None and self.requests_per_minute < 1:
            raise SettingsError("предел частоты запросов к модели должен быть положительным")

    @property
    def is_ready(self) -> bool:
        """Настроены ли модель и ключ, без которых вызывать провайдера нечем."""
        return bool(self.model.strip()) and bool(self.api_key and self.api_key.strip())

    def ensure_ready(self) -> None:
        """Проверяет, что анализ и обработку можно запускать (PRD § 23).

        Сообщение перечисляет недостающее, но самого ключа не содержит: оно попадает
        в интерфейс и в диагностику CLI.
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
    """Пороговые значения автоматических проверок (PRD § 12, § 15).

    Значения по умолчанию выбраны так, чтобы проверки ловили технический брак, а не
    литературные решения: доля целевого алфавита сравнивается только на длинных полных
    переводах, а повтор n-грамм остаётся предупреждением.
    """

    glossary_confidence_threshold: float = 0.8
    target_language_min_share: float = 0.6
    language_check_min_length: int = 80
    repetition_ngram_size: int = 6
    repetition_min_repeats: int = 3

    def __post_init__(self) -> None:
        for name in ("glossary_confidence_threshold", "target_language_min_share"):
            value: float = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise SettingsError(f"порог {name} вне диапазона 0..1: {value}")
        if self.language_check_min_length < 0:
            raise SettingsError("минимальная длина проверки языка не может быть отрицательной")
        if self.repetition_ngram_size < 1:
            raise SettingsError("размер n-граммы проверки повторов должен быть положительным")
        if self.repetition_min_repeats < 2:
            raise SettingsError("повтором считается не менее двух вхождений n-граммы")


@dataclass(frozen=True, slots=True)
class StorageSettings:
    """Где приложение держит свои данные (PRD § 22).

    Пустой `database_path` означает базу внутри каталога данных: пользователю достаточно
    перенести один каталог, чтобы перенести всё приложение. Отдельный путь нужен там, где
    база лежит не с книгами, — например на другом диске.
    """

    data_root: str = DEFAULT_DATA_ROOT
    database_path: str = ""

    def __post_init__(self) -> None:
        if not self.data_root.strip():
            raise SettingsError("каталог данных приложения не задан")


@dataclass(frozen=True, slots=True)
class ServerSettings:
    """Адрес локального Web UI (PRD § 21, § 4.1)."""

    host: str = DEFAULT_SERVER_HOST
    port: int = DEFAULT_SERVER_PORT

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise SettingsError("адрес локального сервера не задан")
        if not 1 <= self.port <= 65535:
            raise SettingsError(f"порт локального сервера вне диапазона: {self.port}")

    @property
    def is_local_only(self) -> bool:
        """Доступен ли сервер только с этой машины.

        Пользователь должен видеть, что приложение с ключом провайдера и полным текстом
        книги открыто в сеть, если он сам задал такой адрес.
        """
        return self.host.strip() in _LOOPBACK_HOSTS

    @property
    def allowed_hosts(self) -> tuple[str, ...]:
        """Имена в заголовке `Host`, с которыми Web UI принимает запросы (PRD § 4.1).

        Список закрывает перепривязку DNS: страница атакующего попадает в приложение под
        своим именем, а по чужому имени приложение не отвечает. Локальный запуск открывают
        и по адресу, и по имени `localhost`, поэтому равноправны все имена петлевого
        интерфейса. Запуск на всех интерфейсах именами не ограничивается — `*` означает
        любое имя: такой адрес владелец машины задаёт сам и видит предупреждение о нём
        на экране конфигурации.
        """
        host = self.host.strip()
        if self.is_local_only:
            return _LOOPBACK_HOSTS
        if host in _WILDCARD_HOSTS:
            return ("*",)
        return (host,)

    @property
    def base_url(self) -> str:
        """Адрес, по которому открывается Web UI."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"


@dataclass(frozen=True, slots=True)
class WorkerPoolSettings:
    """Фоновый воркер: сколько задач очереди выполняется одновременно (PRD § 25).

    Параллелизм ускоряет обработку глав, которые независимы по данным. Анализ глав
    от него не ускоряется: следующая глава ставится в очередь только после сохранения
    предыдущей (PRD § 13). Чем больше значение, тем скорее провайдер ответит отказом
    по частоте запросов, — от этого защищает `LLMSettings.requests_per_minute`.
    """

    concurrency: int = DEFAULT_WORKER_CONCURRENCY

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise SettingsError("параллелизм воркера должен быть положительным")


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Локальная конфигурация приложения целиком."""

    llm: LLMSettings = field(default_factory=LLMSettings)
    checks: CheckSettings = field(default_factory=CheckSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    worker: WorkerPoolSettings = field(default_factory=WorkerPoolSettings)


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_DATABASE_NAME",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_SERVER_HOST",
    "DEFAULT_SERVER_PORT",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_WORKER_CONCURRENCY",
    "OPENAI_COMPATIBLE_PROVIDER",
    "AppSettings",
    "CheckSettings",
    "LLMSettings",
    "ServerSettings",
    "SettingsError",
    "StorageSettings",
    "WorkerPoolSettings",
]
