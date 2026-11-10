"""Bounded local diagnostics, independent of provider envelopes and storage technology."""
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import re

from linguary.application.ports.llm import LLMConfigurationError
from linguary.application.errors import ApplicationError


class DiagnosticNotFoundError(ApplicationError):
    """Absent or foreign diagnostic call; contains no information about its owner."""


from linguary.application.retry_policy import violation_of


@dataclass(frozen=True, slots=True)
class DiagnosticsPolicy:
    bodies_enabled: bool = True
    max_body_bytes: int = 64 * 1024
    max_book_bytes: int = 10 * 1024 * 1024
    retention_seconds: int = 7 * 24 * 60 * 60

    def __post_init__(self):
        for value in (self.max_body_bytes, self.max_book_bytes, self.retention_seconds):
            if type(value) is not int or value <= 0:
                raise ValueError('предел диагностики должен быть конечным положительным целым')


def bounded_body(body, *, max_bytes=64 * 1024, secrets=()):
    if body is None:
        return None, {'body_status': 'unavailable', 'redacted': False, 'truncated': False,
                      'original_bytes': None, 'stored_bytes': 0}
    original_bytes = len(body.encode('utf-8'))
    cleaned = body
    redacted_count = 0
    for secret in sorted((value for value in secrets if value), key=len, reverse=True):
        count = cleaned.count(secret)
        if count:
            redacted_count += count
            cleaned = cleaned.replace(secret, '[REDACTED]')
    cleaned, count_url = re.subn(r'(?i)(https?://)[^/\s@]+@', r'\1[REDACTED]@', cleaned)
    cleaned, count_auth = re.subn(r'''(?i)(authorization["']?\s*[:=]\s*["']?\s*(?:bearer|basic)\s+)[^\s"',}]+''', r'\1[REDACTED]', cleaned)
    redacted_count += count_url + count_auth
    encoded = cleaned.encode('utf-8')
    result = encoded[:max_bytes].decode('utf-8', errors='ignore')
    return result, {'body_status': 'available', 'redacted': cleaned != body,
                    'redacted_count': redacted_count,
                    'truncated': len(encoded) > max_bytes, 'original_bytes': original_bytes,
                    'stored_bytes': len(result.encode('utf-8'))}


class Diagnostics:
    """Metadata is mandatory; optional body failures never invent evidence."""
    def __init__(self, unit_of_work, *, policy=None, clock=None):
        self.uow = unit_of_work
        self.policy = policy or DiagnosticsPolicy()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.uow._diagnostics_service = self
        self.uow.diagnostics_policy = self.policy
        self.uow.diagnostics_clock = self.clock
        self._failed = False
        self._last_cleanup = None

    def ensure_available(self):
        if self._failed:
            raise LLMConfigurationError('Запись наблюдения недоступна; новые вызовы остановлены')
        self.periodic_purge()

    def needs_periodic_purge(self, now=None, *, min_interval_seconds: float = 3600.0) -> bool:
        if self._failed or self._last_cleanup is None:
            return True
        current = now if now is not None else self.clock()
        return (current - self._last_cleanup).total_seconds() >= min_interval_seconds

    def periodic_purge(self, now=None, *, min_interval_seconds: float = 3600.0) -> bool:
        current = now if now is not None else self.clock()
        if not self.needs_periodic_purge(current, min_interval_seconds=min_interval_seconds):
            return False
        self.purge_expired(current)
        self._last_cleanup = current
        return True

    def fail(self):
        self._failed = True
        raise LLMConfigurationError('Запись наблюдения недоступна; новые вызовы остановлены') from None

    def finalize_attempt(self, *args, **kwargs):
        try:
            return self._finalize_attempt(*args, **kwargs)
        except Exception:
            self.fail()

    def _finalize_attempt(self, attempt_id, decision, violation=None, body=None, *, call_id=None,
                         body_meta=None, retain_accepted=False, transaction=None):
        if decision not in {'accepted', 'rejected', 'unknown', 'stale'}:
            raise ValueError('неизвестное решение по ответу')
        if transaction is None:
            with self.uow.transaction() as active:
                return self.finalize_attempt(attempt_id, decision, violation, body, call_id=call_id,
                    body_meta=body_meta, retain_accepted=retain_accepted, transaction=active)
        record = transaction.llm_calls.diagnostic_record(call_id=call_id, attempt_id=attempt_id)
        if record is None:
            raise LLMConfigurationError('метаданные вызова отсутствуют')
        metadata = dict(record['metadata'] or {'schema_version': 1})
        previous = metadata.get('decision')
        if previous in {'accepted', 'rejected', 'stale'}:
            if previous != decision:
                raise LLMConfigurationError('решение по вызову уже зафиксировано')
            return metadata
        metadata['decision'] = decision
        if violation is not None:
            metadata['violation'] = violation.to_dict()
        if body_meta:
            metadata['body_meta'] = dict(body_meta)
        current_body_meta = dict(metadata.get('body_meta') or {})
        status = current_body_meta.get('body_status')
        keep = decision == 'rejected' or retain_accepted
        if not self.policy.bodies_enabled:
            current_body_meta.update(body_status='disabled', stored_bytes=0)
            metadata.pop('input_text', None)
            body = None
        elif not keep:
            current_body_meta.update(body_status='unavailable', stored_bytes=0)
            metadata.pop('input_text', None)
            body = None
        elif status in {'expired', 'deleted', 'disabled'}:
            metadata.pop('input_text', None)
            body = None
        elif body is not None:
            encoded_len = len(body.encode('utf-8'))
            if (
                'redacted_count' in current_body_meta
                and encoded_len <= self.policy.max_body_bytes
            ):
                current_body_meta = {
                    **current_body_meta,
                    'body_status': 'available',
                    'stored_bytes': encoded_len,
                }
            else:
                body, bounded_meta = bounded_body(body, max_bytes=self.policy.max_body_bytes)
                # Preserve adapter-reported redaction/truncation and original size.
                current_body_meta = {
                    **bounded_meta,
                    **current_body_meta,
                    'redacted': bool(current_body_meta.get('redacted')) or bounded_meta['redacted'],
                    'truncated': bool(current_body_meta.get('truncated')) or bounded_meta['truncated'],
                    'original_bytes': current_body_meta.get('original_bytes') or bounded_meta['original_bytes'],
                    'body_status': 'available',
                    'stored_bytes': len(body.encode('utf-8')),
                }
        else:
            body = record['body']
            if body is None:
                current_body_meta.update(body_status='unavailable', stored_bytes=0)
        if 'input_text' in metadata and isinstance(metadata['input_text'], str):
            if len(metadata['input_text'].encode('utf-8')) > self.policy.max_body_bytes:
                metadata['input_text'], _ = bounded_body(
                    metadata['input_text'], max_bytes=self.policy.max_body_bytes
                )
        current_body_meta['expires_at'] = (record['started_at'] + timedelta(seconds=self.policy.retention_seconds)).isoformat()
        metadata['body_meta'] = current_body_meta
        transaction.llm_calls.update_diagnostics(record['call_id'], metadata=metadata)
        try:
            transaction.llm_calls.store_diagnostic_body(record['call_id'], body)
        except Exception:
            metadata['body_meta'].update(body_status='unavailable', stored_bytes=0)
            transaction.llm_calls.update_diagnostics(record['call_id'], metadata=metadata)
        if attempt_id is not None and decision != 'unknown':
            from linguary.domain.book_conditions import thaw
            attempt = transaction.work.get_attempt(attempt_id)
            if attempt is None:
                raise LLMConfigurationError('происхождение вызова отсутствует')
            if decision == 'stale':
                transaction.work.audit_stale_attempt(attempt_id, call_id=record['call_id'])
                target = attempt.state
            else:
                target = 'accepted' if decision == 'accepted' else 'rejected'
            if attempt.state != target:
                transaction.work.transition_attempt(attempt_id, target, expected_state=attempt.state,
                    outcome={**thaw(attempt.outcome), 'decision': decision})
        transaction.llm_calls.enforce_diagnostics_policy(self.policy, self.clock(), book_id=record['book_id'])
        return metadata

    def purge_expired(self, now=None):
        try:
            with self.uow.transaction() as transaction:
                transaction.llm_calls.enforce_diagnostics_policy(self.policy, now or self.clock())
        except Exception:
            self.fail()

    def _require_book(self, transaction, book_id):
        from linguary.application.book_overview import BookNotFoundError
        if transaction.books.get(book_id) is None:
            raise BookNotFoundError('книга не найдена')

    @staticmethod
    def _metadata(entry):
        from dataclasses import asdict
        return dict(call_id=entry.id, book_id=entry.book_id, attempt_id=entry.attempt_id,
            stage=entry.stage.value, provider=entry.provider, model=entry.model,
            started_at=entry.started_at.isoformat(), success=entry.success,
            usage=None if entry.usage is None else asdict(entry.usage),
            cost=None if entry.cost is None else str(entry.cost),
            diagnostic_meta=entry.diagnostic_meta)

    def list(self, book_id):
        """Local metadata listing; diagnostic bodies never enter the list."""
        with self.uow.transaction() as transaction:
            self._require_book(transaction, book_id)
            transaction.llm_calls.enforce_diagnostics_policy(self.policy, self.clock(), book_id=book_id)
            return tuple(self._metadata(entry) for entry in transaction.llm_calls.list_for_book(book_id))

    def read(self, book_id, call_id):
        with self.uow.transaction() as transaction:
            self._require_book(transaction, book_id)
            repo = transaction.llm_calls
            record = repo.diagnostic_record(call_id=call_id)
            if record is None or record['book_id'] != book_id:
                raise DiagnosticNotFoundError('диагностическая запись не найдена')
            repo.enforce_diagnostics_policy(self.policy, self.clock(), book_id=book_id)
            record = repo.diagnostic_record(call_id=call_id)
            get_for_book = getattr(repo, 'get_for_book', None)
            if callable(get_for_book):
                entry = get_for_book(book_id, call_id)
            else:
                entry = next(
                    (item for item in repo.list_for_book(book_id) if item.id == call_id),
                    None,
                )
            if entry is None:
                raise DiagnosticNotFoundError('диагностическая запись не найдена')
            body_status = ((record['metadata'] or {}).get('body_meta') or {}).get('body_status')
            return dict(metadata=self._metadata(entry), body=record['body'],
                body_status=body_status or ('available' if record['body'] is not None else 'unavailable'))

    def purge(self, book_id, *, call_id=None):
        with self.uow.transaction() as transaction:
            self._require_book(transaction, book_id)
            if call_id is not None:
                record = transaction.llm_calls.diagnostic_record(call_id=call_id)
                if record is None or record['book_id'] != book_id:
                    raise DiagnosticNotFoundError('диагностическая запись не найдена')
            return transaction.llm_calls.purge_diagnostic_bodies(book_id, call_id=call_id)


def diagnostics_for(unit_of_work):
    existing = getattr(unit_of_work, '_diagnostics_service', None)
    return existing if existing is not None else Diagnostics(unit_of_work)
