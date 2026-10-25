"""Conservative per-call review accounting, in integer micro-USD."""
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Callable
from uuid import uuid4

from linguary.domain.book_conditions import WorkStateError
from linguary.domain.quality_review import QualityReservation
from linguary.domain.recovery import CallAttempt


def microusd(value):
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise WorkStateError('неизвестный расход USD')
    amount = Fraction(value) * 1000000
    return -(-amount.numerator // amount.denominator)


@dataclass(frozen=True)
class QualityRoute:
    """Operator-verified route capability and maximum USD tariff, never inferred."""
    model: str
    route_id: str
    price_version: str
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    fee: Decimal
    max_completion_tokens: int
    input_token_bound: Callable
    proof: str
    completion_includes_reasoning: bool
    base_url: str | None = None
    bound_spec: tuple = ()

    def estimate(self, request):
        if (not self.proof or not self.price_version or not self.completion_includes_reasoning
                or type(self.max_completion_tokens) is not int or self.max_completion_tokens <= 0):
            raise WorkStateError('нет доказанной верхней оценки маршрута')
        for value in (self.input_per_mtok, self.output_per_mtok, self.fee):
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise WorkStateError('неизвестная цена USD')
        count = self.input_token_bound(request)
        if type(count) is not int or count <= 0:
            raise WorkStateError('нет верхней границы полного входа')
        return self.charge(count, self.max_completion_tokens)

    def charge(self, prompt_tokens, completion_tokens):
        if any(type(n) is not int or n < 0 for n in (prompt_tokens, completion_tokens)):
            raise WorkStateError('неизвестный расход токенов')
        amount = (prompt_tokens * Fraction(self.input_per_mtok)
            + completion_tokens * Fraction(self.output_per_mtok) + 1000000 * Fraction(self.fee))
        return -(-amount.numerator // amount.denominator)

    def cost_of(self, usage):
        if usage is None or usage.prompt_tokens is None or usage.completion_tokens is None:
            return None
        amount = self.charge(usage.prompt_tokens, usage.completion_tokens)
        with localcontext() as context:
            context.prec = max(28, len(str(amount)) + 7)
            return Decimal(amount) / Decimal(1000000)

    def actual(self, response):
        if response.cost is not None:
            return microusd(response.cost)
        usage = response.usage
        if usage is None or usage.prompt_tokens is None or usage.completion_tokens is None:
            return None
        return self.charge(usage.prompt_tokens, usage.completion_tokens)

    def snapshot(self):
        return dict(schema_version=1, model=self.model, route_id=self.route_id,
            price_version=self.price_version, currency='USD', input_per_mtok=str(self.input_per_mtok),
            output_per_mtok=str(self.output_per_mtok), fee=str(self.fee),
            max_completion_tokens=self.max_completion_tokens, proof=self.proof,
            base_url=self.base_url, bound_spec=list(self.bound_spec))


class QualityBudget:
    def __init__(self, unit_of_work):
        self.uow = unit_of_work

    def reserve(self, review_id, estimate, *, price, expected_revision, task_id=None, context=None, expected_manifest_sha=None):
        with self.uow.transaction() as tx:
            review = tx.work.get_review(review_id)
            if review is None:
                raise WorkStateError('вычитка не найдена')
            if (review.state not in ('queued', 'running') or review.revision != expected_revision
                    or type(estimate) is not int or estimate <= 0):
                raise WorkStateError('состояние вычитки изменилось')
            if expected_manifest_sha is not None:
                from linguary.application.quality_context import snapshot
                if snapshot(tx, review.book_id, tuple(review.manifest['scope']))['manifest_sha'] != expected_manifest_sha:
                    raise WorkStateError('manifest_changed')
            attempt_id = str(uuid4())
            attempt = CallAttempt(attempt_id, review.book_id, review_id=review.id,
                conditions_id=review.conditions_id, task_id=task_id,
                context=context or dict(schema_version=1, attempt_id=attempt_id, review_id=review_id,
                    manifest_sha=review.manifest.get('manifest_sha'), route=price),
                ordinal=tx.work.review_attempt_count(review.id))
            tx.work.add_attempt(attempt)
            tx.work.reserve(QualityReservation(attempt.id, review.id, estimate, price=price), expected_revision=expected_revision)
        return attempt

    def mark_sent(self, attempt_id):
        with self.uow.transaction() as tx:
            try:
                attempt = tx.work.get_attempt(attempt_id)
                if attempt is None:
                    raise WorkStateError('попытка не найдена')
                review = tx.work.get_review(attempt.review_id)
                if review.manifest.get('manifest_sha') is not None:
                    from linguary.application.quality_context import snapshot
                    if snapshot(tx, review.book_id, tuple(review.manifest['scope']))['manifest_sha'] != review.manifest['manifest_sha']:
                        raise WorkStateError('manifest_changed')
                tx.work.mark_review_sent(attempt_id)
            except WorkStateError:
                tx.work.cancel_review_unsent(attempt_id)
                return False
        return True

    def settle(self, attempt_id, actual):
        with self.uow.transaction() as tx:
            tx.work.settle(attempt_id, actual=actual)

    def reconcile(self):
        with self.uow.transaction() as tx:
            tx.work.reconcile_quality_attempts()
