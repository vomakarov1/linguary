"""Детерминированный клиент LLM для тестов и локальной отладки.

Заменяет провайдера заранее заданными ответами и не выполняет сетевых вызовов. Нужен
всем уровням проверки — unit, integration, сквозному сценарию (ADR § 3) — и режиму
отладки CLI, поэтому живёт в пакете рядом с настоящим адаптером, а не в тестах.

Ответы задаются в том же виде, в каком их отдаёт провайдер: текстом структурированного
ответа. Разбирает их тот же `parse_payload`, что и HTTP-клиент, поэтому сценарии
«невалидный JSON», «обрыв по лимиту токенов» и «результат для чужих блоков»
воспроизводятся точно так же, как с настоящей моделью, а не имитируются исключениями.

Сделанные запросы сохраняются: по ним проверяется состав контекста, порядок вызовов
и то, что уточняющая инструкция повтора действительно дописана к исходному запросу.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from linguary.adapters.llm.payloads import parse_payload
from linguary.application.ports.llm import (
    LLMClient,
    LLMPayload,
    LLMRequest,
    LLMResponse,
)
from linguary.domain import TokenUsage

#: Провайдер и модель в ответах клиента: они попадают в журнал вызовов, и по ним видно,
#: что запуск выполнен без обращения к настоящей модели.
FAKE_PROVIDER = "fake"
FAKE_MODEL = "fake-model"


class FakeScenarioError(Exception):
    """Сценарий клиента задан неверно: заготовленные ответы кончились.

    Ошибка сборки теста, а не поведение модели, поэтому она не наследует `LLMError`
    и не должна выглядеть как неудачный вызов, который можно повторить.
    """


@dataclass(frozen=True, slots=True)
class FakeReply:
    """Один заготовленный ответ провайдера."""

    content: str
    truncated: bool = False
    usage: TokenUsage | None = None
    model: str = FAKE_MODEL

    @classmethod
    def of_payload(
        cls,
        payload: Mapping[str, object],
        *,
        truncated: bool = False,
        usage: TokenUsage | None = None,
    ) -> FakeReply:
        """Ответ из готовой структуры: она сериализуется так же, как это делает модель."""
        return cls(
            content=json.dumps(payload, ensure_ascii=False),
            truncated=truncated,
            usage=usage,
        )


#: Что клиент делает на очередной вызов: отвечает или поднимает ошибку. Исключением
#: задаются неудачи вызова (`LLMTransportError` и другие наследники `LLMError`).
type FakeScenarioStep = FakeReply | Exception


@dataclass(slots=True)
class FakeLLMClient(LLMClient):
    """Клиент LLM с заранее заданными ответами.

    Ответы берутся по порядку вызовов. Когда заготовленные ответы кончились, работает
    `default`; без него сценарий считается исчерпанным, и клиент это сообщает, а не
    отвечает случайным образом.
    """

    replies: Sequence[FakeScenarioStep] = ()
    default: FakeScenarioStep | None = None
    provider: str = FAKE_PROVIDER
    requests: list[LLMRequest[LLMPayload]] = field(default_factory=list)

    @classmethod
    def always(cls, step: FakeScenarioStep, *, provider: str = FAKE_PROVIDER) -> FakeLLMClient:
        """Клиент, отвечающий одинаково на любой запрос."""
        return cls(default=step, provider=provider)

    @classmethod
    def replying(
        cls, steps: Iterable[FakeScenarioStep], *, provider: str = FAKE_PROVIDER
    ) -> FakeLLMClient:
        """Клиент, отвечающий заданной последовательностью."""
        return cls(replies=tuple(steps), provider=provider)

    @property
    def call_count(self) -> int:
        """Сколько запросов клиент получил."""
        return len(self.requests)

    @property
    def last_request(self) -> LLMRequest[LLMPayload]:
        """Последний полученный запрос."""
        if not self.requests:
            raise FakeScenarioError("клиент ещё не получал запросов")
        return self.requests[-1]

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        """Отдаёт очередной заготовленный ответ, разобрав его как ответ провайдера."""
        step = self._next_step()
        self.requests.append(request)
        if isinstance(step, Exception):
            raise step
        payload = parse_payload(request.kind, step.content, truncated=step.truncated)
        return LLMResponse(
            payload=payload,  # type: ignore[arg-type]  # схему выбирает request.kind
            provider=self.provider,
            model=step.model,
            truncated=step.truncated,
            usage=step.usage,
        )

    def _next_step(self) -> FakeScenarioStep:
        """Шаг сценария для текущего вызова."""
        index = len(self.requests)
        if index < len(self.replies):
            return self.replies[index]
        if self.default is not None:
            return self.default
        raise FakeScenarioError(
            f"сценарий клиента исчерпан: заготовлено ответов — {len(self.replies)}, "
            f"вызов — {index + 1}"
        )


__all__ = [
    "FAKE_MODEL",
    "FAKE_PROVIDER",
    "FakeLLMClient",
    "FakeReply",
    "FakeScenarioError",
    "FakeScenarioStep",
]
