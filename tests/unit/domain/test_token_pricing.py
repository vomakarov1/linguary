"""Расчёт стоимости вызова модели по тарифу за миллион токенов."""

from __future__ import annotations

from decimal import Decimal

import pytest

from linguary.domain.errors import LLMCallLogError
from linguary.domain.llm_log import TokenPricing, TokenUsage


def test_cost_of_prompt_and_completion() -> None:
    pricing = TokenPricing(input_per_mtok=Decimal("3"), output_per_mtok=Decimal("15"))
    assert pricing.cost_of(TokenUsage(prompt_tokens=1000, completion_tokens=2000)) == Decimal("0.033")
    assert pricing.cost_of(TokenUsage(prompt_tokens=None, completion_tokens=2000)) == Decimal("0.030")


def test_cost_of_without_usage_is_none() -> None:
    pricing = TokenPricing(input_per_mtok=Decimal("3"), output_per_mtok=Decimal("15"))
    assert pricing.cost_of(None) is None
    assert pricing.cost_of(TokenUsage()) is None


def test_negative_price_rejected() -> None:
    with pytest.raises(LLMCallLogError):
        TokenPricing(input_per_mtok=Decimal("-1"), output_per_mtok=Decimal("15"))
    with pytest.raises(LLMCallLogError):
        TokenPricing(input_per_mtok=Decimal("3"), output_per_mtok=Decimal("-1"))
