"""Bounded local diagnostics, independent of provider envelopes and storage technology."""
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import re

from linguary.application.ports.llm import LLMConfigurationError
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
    for secret in sorted((value for value in secrets if value), key=len, reverse=True):
        cleaned = cleaned.replace(secret, '[REDACTED]')
    cleaned = re.sub(r'(?i)(https?://)[^/\s@]+@', r'\1[REDACTED]@', cleaned)
    cleaned = re.sub(r'''(?i)(authorization["']?\s*[:=]\s*["']?\s*(?:bearer|basic)\s+)[^\s"',}]+''', r'\1[REDACTED]', cleaned)
    encoded = cleaned.encode('utf-8')
    result = encoded[:max_bytes].decode('utf-8', errors='ignore')
    return result, {'body_status': 'available', 'redacted': cleaned != body,
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
        now = self.clock()
        if self._last_cleanup is None or now - self._last_cleanup >= timedelta(hours=1):
            self.purge_expired(now)
            self._last_cleanup = now

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
            body = None
        elif not keep:
            current_body_meta.update(body_status='unavailable', stored_bytes=0)
            body = None
        elif status in {'expired', 'deleted', 'disabled'}:
            body = None
        elif body is not None:
            body, bounded_meta = bounded_body(body, max_bytes=self.policy.max_body_bytes)
            # Preserve adapter-reported redaction/truncation and original size.
            current_body_meta = {**bounded_meta, **current_body_meta, 'body_status': 'available',
                                 'stored_bytes': len(body.encode('utf-8'))}
        else:
            body = record['body']
            if body is None:
                current_body_meta.update(body_status='unavailable', stored_bytes=0)
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

    def purge(self, book_id, *, call_id=None):
        with self.uow.transaction() as transaction:
            return transaction.llm_calls.purge_diagnostic_bodies(book_id, call_id=call_id)



def diagnostics_for(unit_of_work):
    existing = getattr(unit_of_work, '_diagnostics_service', None)
    return existing if existing is not None else Diagnostics(unit_of_work)
