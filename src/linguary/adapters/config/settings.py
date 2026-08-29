"""Локальные настройки приложения: LLM и пороги автоматических проверок.

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

from linguary.application.errors import ApplicationError
from linguary.application.ports.llm import LLMConfigurationError

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


class SettingsError(ApplicationError):
    """Локальная конфигурация непригодна: неразбираемый файл или недопустимое значение."""


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Настройка провайдера LLM (PRD § 23).

    Пустые `model` и `api_key` означают ненастроенную LLM — обычное состояние первого
    запуска, а не ошибка: приложение показывает состояние конфигурации и не отправляет
    запросов. Ошибкой оно становится в момент попытки создать клиент.
    """

    provider: str = OPENAI_COMPATIBLE_PROVIDER
    base_url: str = DEFAULT_BASE_URL
    model: str = ""
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_output_tokens: int | None = None
    temperature: float | None = None
    max_attempts: int = DEFAULT_MAX_ATTEMPTS

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
class AppSettings:
    """Локальная конфигурация приложения целиком."""

    llm: LLMSettings = field(default_factory=LLMSettings)
    checks: CheckSettings = field(default_factory=CheckSettings)


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_TIMEOUT_SECONDS",
    "OPENAI_COMPATIBLE_PROVIDER",
    "AppSettings",
    "CheckSettings",
    "LLMSettings",
    "SettingsError",
]
