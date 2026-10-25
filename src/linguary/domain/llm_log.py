"""Technical LLM call log (PRD § 24).

The log serves for diagnostics, cost estimation, and error troubleshooting. Full requests
and responses are not stored: they contain book text. Debug payload persistence remains
an adapter configuration and does not enter the domain record.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from linguary.domain.errors import LLMCallLogError
from linguary.domain.tasks import TaskError


class LLMCallStage(StrEnum):
    """Processing stage at which the call was made."""

    QUALITY_REVIEW = "quality_review"
    CHAPTER_ANALYSIS = "chapter_analysis"
    TRANSLATION_UNIT = "translation_unit"
    BLOCK_REGENERATION = "block_regeneration"


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Call token volume if reported by the provider."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "completion_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise LLMCallLogError(f"количество токенов не может быть отрицательным: {name}")

    @property
    def total_tokens(self) -> int | None:
        """Total token count or `None` if not provided by the provider."""
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


@dataclass(frozen=True, slots=True)
class TokenPricing:
    """Price per million prompt and completion tokens for call cost estimation."""

    input_per_mtok: Decimal
    output_per_mtok: Decimal

    def __post_init__(self) -> None:
        if self.input_per_mtok < 0 or self.output_per_mtok < 0:
            raise LLMCallLogError("стоимость токенов не может быть отрицательной")

    def cost_of(self, usage: TokenUsage | None) -> Decimal | None:
        """Cost of known tokens; absence of all token data does not imply zero."""
        if usage is None or usage.total_tokens is None:
            return None
        return (
            self.input_per_mtok * (usage.prompt_tokens or 0)
            + self.output_per_mtok * (usage.completion_tokens or 0)
        ) / Decimal(1_000_000)


@dataclass(frozen=True, slots=True)
class LLMCallLog:
    """Log entry for a single model call."""

    id: str
    book_id: str
    stage: LLMCallStage
    provider: str
    model: str
    started_at: datetime
    duration_ms: int
    attempt: int
    success: bool
    task_id: str | None = None
    error: TaskError | None = None
    usage: TokenUsage | None = None
    cost: Decimal | None = None
    attempt_id: str | None = None
    diagnostic_meta: dict | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise LLMCallLogError("идентификатор записи журнала пуст")
        if not self.book_id.strip():
            raise LLMCallLogError("запись журнала не привязана к книге")
        if not self.provider.strip() or not self.model.strip():
            raise LLMCallLogError("провайдер и модель в записи журнала обязательны")
        if self.duration_ms < 0:
            raise LLMCallLogError("длительность вызова не может быть отрицательной")
        if self.attempt < 1:
            raise LLMCallLogError("номер попытки вызова начинается с единицы")
        if self.success and self.error is not None:
            raise LLMCallLogError(f"успешный вызов {self.id} не может содержать ошибку")
        if not self.success and self.error is None:
            raise LLMCallLogError(f"у неуспешного вызова {self.id} не указана ошибка")
        if self.cost is not None and self.cost < 0:
            raise LLMCallLogError("стоимость вызова не может быть отрицательной")


@dataclass(frozen=True, slots=True)
class CallContext:
    """Designated context of exactly one physical model request."""
    conditions_id: str
    stage: str
    scope_id: str
    ordered_block_ids: tuple[str, ...]
    attempt_id: str
    generation: int
    assigned_model: str
    prompt_hash: str
    run_id: str | None = None
    task_id: str | None = None
    root_id: str | None = None

    def __post_init__(self):
        from linguary.domain.book_conditions import identifier, nonnegative
        identifier(self.conditions_id)
        identifier(self.attempt_id)
        nonnegative(self.generation)
        if self.root_id is not None:
            identifier(self.root_id)
        if not self.stage or not self.scope_id or not self.assigned_model:
            raise ValueError('неполный контекст вызова')
        if len(self.prompt_hash) != 64 or any(c not in '0123456789abcdef' for c in self.prompt_hash):
            raise ValueError('неверная контрольная сумма инструкции')
        object.__setattr__(self, 'ordered_block_ids', tuple(self.ordered_block_ids))
        if len(set(self.ordered_block_ids)) != len(self.ordered_block_ids):
            raise ValueError('блоки контекста повторяются')

    def to_dict(self):
        from dataclasses import asdict
        return {'schema_version': 1, **asdict(self)}


@dataclass(frozen=True, slots=True)
class Violation:
    """An explicit check result; exception text never determines its scope."""
    rule_code: str
    stage: str
    scope: str = 'unknown'
    block_ids: tuple[str, ...] = ()
    artifact_kinds: tuple[str, ...] = ()
    safe_message: str = 'Ответ требует проверки'

    def __post_init__(self):
        if self.scope not in {'local', 'unit', 'global', 'unknown'}:
            raise ValueError('неизвестная область нарушения')
        if self.scope == 'local' and not self.block_ids:
            raise ValueError('локальное нарушение требует проверенных блоков')
        object.__setattr__(self, 'block_ids', tuple(self.block_ids))
        object.__setattr__(self, 'artifact_kinds', tuple(self.artifact_kinds))
        object.__setattr__(self, 'safe_message', ' '.join(self.safe_message.split())[:300])

    def verified(self, context):
        if self.scope == 'local' and (context is None or not set(self.block_ids) <= set(context.ordered_block_ids)):
            return Violation(self.rule_code, self.stage, safe_message=self.safe_message)
        return self

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)
