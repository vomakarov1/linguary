"""Наблюдения на границе провайдера и ограничения локального сравнения."""
from __future__ import annotations

import hashlib
import json
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import httpx

from linguary.adapters.config import LLMSettings
from linguary.adapters.llm.openai_compatible import OpenAICompatibleLLMClient, safe_detail
from linguary.application.ports.llm import LLMClient, LLMConfigurationError, LLMRequest, LLMResponse, LLMResponseFormatError, LLMResponseKind
from linguary.domain import TokenUsage
from linguary.domain.llm_log import TokenPricing


class ComparisonStopped(LLMConfigurationError):
    """Условия сравнения больше не позволяют новые вызовы."""


@dataclass
class CallObservation:
    call_uid: str
    operation_id: str
    assigned_model: str
    assigned_route: str
    sequence: int
    variant_id: str = ''
    repeat: int = 1
    stage: str = ''
    chapter_id: str | None = None
    group_id: str | None = None
    call_id: str | None = None
    received_response: bool = False
    first_response: bool = False
    truncated: bool = False
    accepted: bool | None = None
    actual_model: str | None = None
    actual_provider: str | None = None
    generation_id: str | None = None
    full_response_cache: bool | None = None
    context_cache: bool | None = None
    usage: TokenUsage | None = None
    cost: Decimal | None = None
    cost_source: str | None = None
    duration_ms: int | None = None
    failure_category: str | None = None
    _raw: str | None = field(default=None, repr=False)

    def public(self):
        result = asdict(self)
        result.pop('_raw')
        if self.cost is not None:
            result['cost'] = str(self.cost)
        return result


class BudgetGuard:
    """Общий резерв учитывает уже потраченные и ещё выполняемые запросы."""
    def __init__(self, budget: Decimal):
        self.budget = Decimal(budget)
        if not self.budget.is_finite() or self.budget <= 0:
            raise ValueError('Бюджет должен быть положительным')
        self.spent = Decimal(0)
        self.reservations: dict[str, Decimal] = {}
        self.uncertain: dict[str, Decimal] = {}
        self.stopped = False

    def reserve(self, max_cost: Decimal) -> str:
        maximum = Decimal(max_cost)
        if self.stopped or not maximum.is_finite() or maximum <= 0 or self.spent + sum(self.reservations.values()) + sum(self.uncertain.values()) + maximum > self.budget:
            raise ComparisonStopped('Недостаточно согласованного резерва бюджета')
        token = uuid4().hex
        self.reservations[token] = maximum
        return token

    def settle(self, reservation: str, receipt: Decimal | None):
        maximum = self.reservations.pop(reservation)
        if receipt is None:
            self.uncertain[reservation] = maximum
        else:
            cost = Decimal(receipt)
            if not cost.is_finite() or cost < 0:
                self.uncertain[reservation] = maximum
                self.stopped = True
                raise ComparisonStopped('Недопустимая стоимость квитанции')
            self.spent += cost
            if cost > maximum or self.spent + sum(self.reservations.values()) + sum(self.uncertain.values()) > self.budget:
                self.stopped = True
                raise ComparisonStopped('Квитанция превысила согласованный резерв')


class ObservationLedger:
    def __init__(self, *, variant_id='', repeat=1, diagnostic_root: Path | None = None, retain_rejected=False, retention_seconds=0, secrets=(), clock=time.time, receipt_root: Path | None = None):
        self.observations: list[CallObservation] = []
        self.violations: list[str] = []
        self.variant_id, self.repeat = variant_id, repeat
        self.diagnostic_root = diagnostic_root
        self.retain_rejected, self.retention_seconds = retain_rejected, retention_seconds
        self.secrets, self.clock = tuple(secrets), clock
        self._requests: dict[int, CallObservation] = {}
        self._files: dict[Path, float] = {}
        self._first: set[str] = set()
        self.receipt_root = receipt_root
        self.task_scopes: dict[str, str] = {}
        self._contexts: dict[tuple[str, str, str], tuple[str, str]] = {}

    @property
    def first_responses(self):
        return len(self._first)

    def stop(self, reason):
        if reason not in self.violations:
            self.violations.append(reason)

    def ensure_open(self):
        if self.violations:
            raise ComparisonStopped('Нарушены условия сравнения; новые вызовы остановлены')

    def register_context(self, request, *, book_id, chapter_id, block_ids):
        content_hash = hashlib.sha256(request.input_text.encode()).hexdigest()
        group = hashlib.sha256(json.dumps([book_id, request.kind.value, chapter_id, list(block_ids)], separators=(",", ":")).encode()).hexdigest()
        key = book_id, request.kind.value, content_hash
        value = chapter_id, group
        if key in self._contexts and self._contexts[key] != value:
            self.stop("ambiguous_context_identity")
            raise ComparisonStopped("Неоднозначная связь контекста с группой блоков")
        self._contexts[key] = value

    def begin(self, request, *, model, route):
        self.ensure_open()
        content_hash = hashlib.sha256(request.input_text.encode()).hexdigest()
        chapter, group = self._contexts.get(
            (request.book_id or '', request.kind.value, content_hash),
            (self.task_scopes.get(request.task_id), content_hash),
        )
        # Идентичность группы задаётся исходными block ids, а не случайным unit id
        # или ответами предыдущих частей. Ремонтная инструкция в ключ не входит.
        key = '\0'.join((self.variant_id, str(self.repeat), request.book_id or '', request.kind.value, group))
        operation = hashlib.sha256(key.encode()).hexdigest()
        item = CallObservation(
            uuid4().hex, operation, model, route, len(self.observations) + 1,
            self.variant_id, self.repeat, request.kind.value,
            chapter_id=chapter, group_id=group,
        )
        self.observations.append(item)
        self._requests[id(request)] = item
        return item

    def persist(self, item):
        if self.receipt_root is None:
            return
        try:
            self.receipt_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.receipt_root.chmod(0o700)
            path = self.receipt_root / f"{item.call_uid}.json"
            with path.open("w", encoding="utf-8") as stream:
                path.chmod(0o600)
                json.dump(item.public(), stream, ensure_ascii=False, allow_nan=False)
        except (OSError, ValueError):
            self.stop("receipt_write_failed")
            raise ComparisonStopped("Не удалось записать квитанцию; новые вызовы остановлены") from None

    def received(self, item):
        item.received_response = True
        item.first_response = item.operation_id not in self._first
        self._first.add(item.operation_id)

    def finish(self, item, outcome):
        if isinstance(outcome, (LLMResponse, LLMResponseFormatError)):
            if not item.received_response:
                self.received(item)
            item.truncated = outcome.truncated
            item._raw = getattr(outcome, 'raw_content', None)
            # HTTP-boundary metadata is authoritative; never substitute configured names.
            if item.usage is None:
                item.usage = outcome.usage
        if isinstance(outcome, Exception):
            item.failure_category = 'model_output' if isinstance(outcome, LLMResponseFormatError) else 'no_response'
        self.persist(item)

    def on_logged(self, entry, request, outcome):
        item = self._requests.pop(id(request), None)
        if item is None:
            self.stop('unlinked_log')
            raise ComparisonStopped('Не удалось связать запись журнала')
        item.call_id = entry.id
        self.finish(item, outcome)

    def finalize_operation(self, operation_id, *, accepted: bool | None, call_statuses: dict[str, bool] | None = None):
        for item in self.observations:
            if item.operation_id != operation_id:
                continue
            status = call_statuses.get(item.call_id) if call_statuses is not None else accepted
            item.accepted = status
            raw, item._raw = item._raw, None
            self.persist(item)
            if status is None:
                self.stop('unknown_final_status')
                continue
            if status or raw is None or not self.retain_rejected:
                continue
            if self.diagnostic_root is None or self.retention_seconds <= 0:
                self.stop('invalid_diagnostic_policy')
                continue
            self.diagnostic_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.diagnostic_root.chmod(0o700)
            path = self.diagnostic_root / f'{item.call_uid}.json'
            safe = safe_detail(raw, api_key=None, secrets=self.secrets)
            expires = self.clock() + self.retention_seconds
            try:
                with path.open('x', encoding='utf-8') as stream:
                    path.chmod(0o600)
                    json.dump({'call_uid': item.call_uid, 'expires_at': expires, 'response': safe}, stream)
                self._files[path] = expires
            except OSError:
                self.stop('diagnostic_write_failed')
                raise ComparisonStopped('Не удалось сохранить диагностику') from None

    def cleanup(self, *, now=None):
        current = self.clock() if now is None else now
        if self.diagnostic_root is not None and self.diagnostic_root.is_dir():
            for path in self.diagnostic_root.glob("*.json"):
                if path.is_symlink():
                    continue
                try:
                    document = json.loads(path.read_text())
                    deadline = document["expires_at"]
                    if path.stem == document["call_uid"] and isinstance(deadline, (int, float)):
                        self._files[path] = deadline
                except (OSError, ValueError, KeyError, TypeError):
                    self.stop("invalid_diagnostic_file")
        for path, deadline in list(self._files.items()):
            if deadline <= current:
                path.unlink(missing_ok=True)
                del self._files[path]


_current: ContextVar[CallObservation | None] = ContextVar('comparison_call', default=None)


class ReceiptHTTPClient:
    """Получает техническую квитанцию до настоящего разбора ответа адаптером."""
    def __init__(self, http, *, ledger, model, route, pricing=None):
        self.http, self.ledger = http, ledger
        self.model, self.route, self.pricing = model, route, pricing

    async def post(self, url, **kwargs):
        item = _current.get()
        if item is None:
            raise ComparisonStopped('Нет наблюдения для HTTP-вызова')
        body = kwargs['json']
        if body.get('model') != self.model or body.get('response_format', {}).get('json_schema', {}).get('strict') is not True:
            self.ledger.stop('request_contract_mismatch')
            raise ComparisonStopped('Параметры запроса отличаются от согласованных')
        body['provider'] = {'order': [self.route], 'allow_fallbacks': False, 'require_parameters': True}
        response = await self.http.post(url, **kwargs)
        if response.status_code >= 400:
            if response.status_code < 500 and response.status_code != 429:
                self.ledger.stop("provider_rejected_contract")
            # Не раскрываем тело HTTP-ошибки через стандартные detail исключений.
            return httpx.Response(response.status_code, json={'error': 'provider request failed'}, headers=response.headers, request=response.request)
        self.ledger.received(item)
        try:
            document = response.json()
        except ValueError:
            document = {}
        if not isinstance(document, dict):
            document = {}
        item.actual_model = document.get('model') if isinstance(document.get('model'), str) else None
        item.actual_provider = document.get('provider') if isinstance(document.get('provider'), str) else None
        item.generation_id = document.get('id') if isinstance(document.get('id'), str) else None
        usage = document.get('usage') or {}
        if not isinstance(usage, dict):
            usage = {}
        prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
        if all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (prompt, completion)):
            item.usage = TokenUsage(prompt, completion)
        cost = usage.get('cost')
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and Decimal(str(cost)).is_finite() and cost >= 0:
            item.cost, item.cost_source = Decimal(str(cost)), 'provider_receipt'
        elif self.pricing is not None and item.usage is not None:
            item.cost, item.cost_source = self.pricing.cost_of(item.usage), 'fixed_route_tariff'
        cache = document.get('full_response_cache')
        item.full_response_cache = cache if isinstance(cache, bool) else None
        context = document.get('context_cache')
        item.context_cache = context if isinstance(context, bool) else None
        if item.actual_model != self.model or item.actual_provider != self.route:
            self.ledger.stop('unknown_or_changed_route')
        if item.cost is None:
            self.ledger.stop('unknown_price')
        if item.full_response_cache is not False:
            self.ledger.stop('unknown_or_full_response_cache')
        self.ledger.persist(item)
        return response


class ObservedLLMClient:
    """Клиент одного принятого маршрута; HTTP-клиент всегда внедряется явно."""
    def __init__(self, *, settings: LLMSettings, route: str, http_client, ledger: ObservationLedger, budget: BudgetGuard, max_call_cost: Decimal, pricing: TokenPricing | None = None):
        self.ledger, self.budget, self.maximum = ledger, budget, max_call_cost
        self.generation_settings = (settings.temperature, settings.max_output_tokens)
        self.model, self.route = settings.model, route
        ledger.secrets = tuple(dict.fromkeys((*ledger.secrets, *((settings.api_key,) if settings.api_key else ()))))
        self._http = http_client
        self.inner = OpenAICompatibleLLMClient(settings=settings, http_client=ReceiptHTTPClient(http_client, ledger=ledger, model=self.model, route=route, pricing=pricing))

    async def generate(self, request):
        self.ledger.ensure_open()
        try:
            reservation = self.budget.reserve(self.maximum)
        except ComparisonStopped:
            self.ledger.stop("budget_exhausted")
            raise
        item = self.ledger.begin(request, model=self.model, route=self.route)
        token = _current.set(item)
        started = time.monotonic()
        try:
            result = await self.inner.generate(request)
            self.ledger.finish(item, result)
            return result
        except Exception as error:
            self.ledger.finish(item, error)
            raise
        finally:
            item.duration_ms = int((time.monotonic() - started) * 1000)
            _current.reset(token)
            try:
                self.budget.settle(reservation, item.cost)
            except ComparisonStopped:
                self.ledger.stop('budget_exceeded')
                raise

    async def aclose(self):
        await self._http.aclose()


class StageLLMClient:
    """Никаких запасных моделей: вид запроса выбирает заранее заданный клиент."""
    def __init__(self, *, analysis: LLMClient, processing: LLMClient):
        self.clients = {LLMResponseKind.CHAPTER_ANALYSIS: analysis, LLMResponseKind.TRANSLATION_RESULT: processing}

    async def generate(self, request):
        return await self.clients[request.kind].generate(request)
