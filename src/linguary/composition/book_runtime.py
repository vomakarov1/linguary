"""Resolve pinned work using saved parameters and locally available credentials."""
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, replace
from hashlib import sha256
import os
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from linguary.adapters.config.settings import LLMSettings, ReadingSettings, CheckSettings
from linguary.application.book_conditions import ConditionsError, payload_digest
from linguary.application.context_builder import snapshot_prompt_templates
from linguary.domain import ProcessingTask
from linguary.domain.book_conditions import thaw
from linguary.domain.recovery import CallAttempt
from linguary.domain.llm_log import CallContext
from linguary.application.diagnostics import diagnostics_for


def public_url(value):
    parsed = urlsplit(value)
    hostname = parsed.hostname or ''
    if ':' in hostname:
        hostname = f'[{hostname}]'
    if parsed.port is not None:
        hostname += f':{parsed.port}'
    return urlunsplit((parsed.scheme, hostname, parsed.path, '', ''))


def route_snapshot(llm):
    proxy = {'mode': llm.proxy_mode}
    if llm.proxy_mode == 'custom':
        proxy['url'] = public_url(llm.proxy_url)
    elif llm.proxy_mode == 'auto':
        # An allowlisted environment fingerprint excludes proxy authentication.
        proxy['environment'] = {key: public_url(os.environ[key]) if key.lower() != 'no_proxy' else os.environ[key]
            for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy') if os.environ.get(key)}
    route = {'provider': llm.provider, 'base_url': public_url(llm.base_url), 'proxy': proxy}
    return {**route, 'route_id': payload_digest(route)}


def conditions_payload(book, settings, revision, profile):
    templates = snapshot_prompt_templates(source_language=book.source_language,
        target_language=book.target_language, reader_level=settings.reading.normalized_level)
    llm = asdict(settings.llm)
    for key in ('api_key', 'proxy_url', 'base_url'):
        llm.pop(key)
    return {'schema_version': 1, 'source_language': book.source_language, 'target_language': book.target_language,
        'reader_level': settings.reading.normalized_level, 'models': {'analysis': settings.llm.effective_analysis_model,
        'processing': settings.llm.effective_processing_model}, 'route': route_snapshot(settings.llm),
        'llm_parameters': llm, 'checks': asdict(settings.checks), 'profile': profile,
        'versions': {'runtime': 1, 'context_builder': 1, 'validators': 1, 'schemas': 2 if profile == 'content-plan-v2' else 1,
                     'prompts': 1}, 'prompt_templates': templates,
        'prompt_sha256': payload_digest(templates), 'settings_revision': revision}


class BoundTaskQueue:
    """Create a task and its provenance in one transaction before workers can claim it."""
    def __init__(self, queue, service, conditions_id, id_factory=None):
        self.queue, self.service, self.conditions_id = queue, service, conditions_id
        self.id_factory = id_factory or (lambda: str(uuid4()))

    async def enqueue(self, identity):
        with self.service.uow.transaction() as transaction:
            conditions = self.service.require(identity.book_id, transaction=transaction)
            if conditions.id != self.conditions_id:
                raise ConditionsError('conditions_revision_conflict')
            task = transaction.tasks.find_by_identity(identity)
            if task is None:
                task = ProcessingTask(self.id_factory(), identity.book_id, identity.task_type, identity.scope_id)
                transaction.tasks.add(task)
            binding = transaction.work.get_binding('task', task.id)
            if binding is None:
                transaction.work.bind('task', task.id, book_id=task.book_id, conditions_id=conditions.id)
            elif binding['conditions_id'] != conditions.id:
                raise ConditionsError('conditions_revision_conflict')
        return task

    async def save(self, task):
        await self.queue.save(task)


class ProvenanceClient:
    """Persist the designated conditions and actual physical response separately."""
    def __init__(self, inner, service, conditions_id):
        self.inner, self.service, self.conditions_id = inner, service, conditions_id
        self._attempts = ContextVar(f'book_attempts_{id(self)}', default=())
        self._responses = ContextVar(f'book_responses_{id(self)}', default=())

    @property
    def attempts(self):
        return self._attempts.get()

    @property
    def responses(self):
        return self._responses.get()

    def reject_last(self):
        if self.attempts:
            with self.service.uow.transaction() as transaction:
                if transaction.work.get_attempt(self.attempts[-1]).state == 'validated':
                    transaction.work.transition_attempt(self.attempts[-1], 'rejected', expected_state='validated')

    async def generate(self, request):
        with self.service.uow.transaction() as transaction:
            conditions = self.service.require(request.book_id, transaction=transaction)
            if conditions.id != self.conditions_id:
                raise ConditionsError('conditions_revision_conflict')
            context = CallContext(conditions_id=conditions.id, task_id=request.task_id, run_id=request.run_id,
                stage=request.stage.value, scope_id=request.scope_id or request.stage.value,
                ordered_block_ids=request.ordered_block_ids, attempt_id=str(uuid4()), generation=0,
                assigned_model=conditions.payload['models']['analysis' if request.stage.value == 'chapter_analysis' else 'processing'],
                prompt_hash=sha256(request.instructions.encode()).hexdigest())
            request = replace(request, call_context=context)
            attempt = CallAttempt(context.attempt_id, request.book_id, conditions_id=conditions.id,
                task_id=request.task_id, run_id=request.run_id, state='sent', ordinal=request.attempt,
                context={**context.to_dict(), 'designated_models': thaw(conditions.payload['models']),
                         'route_id': conditions.payload['route']['route_id']})
            transaction.work.add_attempt(attempt)
        try:
            response = await self.inner.generate(request)
        except BaseException:
            with self.service.uow.transaction() as transaction:
                if transaction.work.get_attempt(attempt.id).state == 'sent':
                    transaction.work.transition_attempt(attempt.id, 'unknown', expected_state='sent')
            raise
        with self.service.uow.transaction() as transaction:
            transaction.work.transition_attempt(attempt.id, 'validated', expected_state='sent',
                outcome={'schema_version': 1, 'provider': response.provider, 'model': response.model,
                         'call_id': response.call_id})
        self._attempts.set((*self.attempts, attempt.id))
        self._responses.set((*self.responses, response))
        return response


class BookRuntimeResolver:
    def __init__(self, service, runtime, factory):
        self.service, self.runtime, self.factory = service, runtime, factory
        self.supported = {'legacy-v1': 1, 'content-plan-v2': 2}

    def settings_for(self, conditions):
        payload = conditions.payload
        expected_versions = {'runtime': 1, 'context_builder': 1, 'validators': 1, 'schemas': self.supported.get(payload.get('profile')), 'prompts': 1}
        if payload.get('profile') not in self.supported or thaw(payload.get('versions', {})) != expected_versions:
            raise ConditionsError('conditions_unavailable')
        if payload_digest(payload['prompt_templates']) != payload['prompt_sha256']:
            raise ConditionsError('conditions_unavailable')
        current = self.runtime.current.settings
        if route_snapshot(current.llm) != thaw(payload['route']):
            raise ConditionsError('route_unavailable')
        if not current.llm.api_key:
            raise ConditionsError('route_unavailable')
        llm = LLMSettings(**{**thaw(payload['llm_parameters']), 'base_url': current.llm.base_url,
                            'proxy_url': current.llm.proxy_url, 'api_key': current.llm.api_key})
        return replace(current, llm=llm, reading=ReadingSettings(reader_level=payload['reader_level']),
                       checks=CheckSettings(**thaw(payload['checks'])))

    @asynccontextmanager
    async def lease(self, conditions_id, role):
        if role not in {'analysis', 'processing', 'regeneration'}:
            raise ConditionsError('conditions_unavailable')
        with self.service.uow.transaction() as transaction:
            conditions = transaction.work.get_conditions(conditions_id)
            if conditions is None:
                raise ConditionsError('conditions_unavailable')
        settings = self.settings_for(conditions)
        generation = self.factory(settings, conditions=conditions)
        try:
            yield generation
        finally:
            await generation.configured_client.aclose()
