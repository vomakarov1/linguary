"""Resolve pinned work using saved parameters and locally available credentials."""
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import os
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from linguary.adapters.config.settings import LLMSettings, ReadingSettings, CheckSettings, RecoverySettings
from linguary.adapters.pronunciation.source import DisabledPronunciationSource, IpaDictPronunciationSource, SOURCE_VERSION
from linguary.application.book_conditions import (
    A2_CONTENT_PLAN_VERSIONS, A2_CONTENT_PLAN_V5_VERSIONS, ConditionsError, payload_digest,
)
from linguary.application.context_builder import snapshot_prompt_templates
from linguary.domain import ProcessingTask
from linguary.domain.book_conditions import thaw
from linguary.domain.recovery import CallAttempt
from linguary.domain.llm_log import CallContext
from linguary.domain.name_translation import NameTranslationPolicy
from linguary.application.diagnostics import diagnostics_for
from linguary.application.recovery import RecoveryError
from linguary.application.translation_policy import require_policy, TranslationPolicyUnavailable


def pronunciation_source_for(conditions, *, injected_source=None):
    if injected_source is not None:
        return injected_source
    if conditions is not None and conditions.payload.get('pronunciation_help_enabled'):
        versions = thaw(conditions.payload.get('pronunciation_versions', {}))
        if versions == {'selection_policy': 2, 'form_rules': 1, 'source': SOURCE_VERSION}:
            return IpaDictPronunciationSource()
    return DisabledPronunciationSource()


def effective_content_plan_profile(conditions):
    """Select the pinned internal plan without changing the reader's saved choice."""
    payload = conditions.payload
    profile = payload['profile']
    versions = thaw(payload.get('versions', {}))
    if profile != 'legacy-v1' or versions.get('validators') != 4:
        return profile
    prompt_templates = thaw(payload.get('prompt_templates', {}))
    marker = (prompt_templates.get('analysis_contract_version')
              if isinstance(prompt_templates, dict) else None)
    expected_a2_versions = (A2_CONTENT_PLAN_VERSIONS if marker is None
                            else A2_CONTENT_PLAN_V5_VERSIONS if type(marker) is int and marker == 5
                            else None)
    if (payload.get('reader_level') != 'a2' or payload.get('english_help', False)
            or versions != expected_a2_versions
            or any(type(value) is not int for value in versions.values())):
        raise ConditionsError('conditions_unavailable')
    return 'content-plan-v3'


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
        target_language=book.target_language, reader_level=settings.reading.normalized_level, profile=profile,
        validators_version=3 if profile == 'content-plan-v3' else 1,
        name_translation_policy={'version': 1, 'strategy': settings.reading.name_translation_strategy},
        analysis_contract_version=5 if profile == 'content-plan-v3' else None)
    llm = asdict(settings.llm)
    for key in ('api_key', 'proxy_url', 'base_url'):
        llm.pop(key)
    return {'schema_version': 1, 'name_translation_policy': {'version': 1, 'strategy': settings.reading.name_translation_strategy},
        'pronunciation_help_enabled': False, 'pronunciation_variant': None,
        'source_language': book.source_language, 'target_language': book.target_language,
        'reader_level': settings.reading.normalized_level, 'models': {'analysis': settings.llm.effective_analysis_model,
        'processing': settings.llm.effective_processing_model}, 'route': route_snapshot(settings.llm),
        'llm_parameters': llm, 'checks': asdict(settings.checks), 'profile': profile,
        'recovery_policy': asdict(settings.recovery),
        'versions': {'runtime': 1, 'context_builder': 3 if profile == 'content-plan-v3' else 1,
                     'validators': 3 if profile == 'content-plan-v3' else 1,
                     'schemas': {'legacy-v1': 1, 'content-plan-v2': 2, 'content-plan-v3': 5}[profile],
                     'prompts': 5 if profile == 'content-plan-v3' else 1}, 'prompt_templates': templates,
        'prompt_sha256': payload_digest(templates), 'settings_revision': revision}


class BoundTaskQueue:
    """Create a task and its provenance in one transaction before workers can claim it."""
    def __init__(self, queue, service, conditions_id, id_factory=None):
        self.queue, self.service, self.conditions_id = queue, service, conditions_id
        self.id_factory = id_factory or (lambda: str(uuid4()))

    def new_task_id(self):
        """Use the identifier factory shared with the backing queue."""
        return self.id_factory()

    async def enqueue(self, identity):
        with self.service.uow.transaction() as transaction:
            conditions = self.service.require(identity.book_id, transaction=transaction)
            if conditions.id != self.conditions_id:
                raise ConditionsError('conditions_revision_conflict')
            task = transaction.tasks.find_by_identity(identity)
            if task is None:
                task = ProcessingTask(self.id_factory(), identity.book_id, identity.task_type, identity.scope_id,
                    operation_id=identity.operation_id, policy_version=identity.policy_version)
                transaction.tasks.add(task)
            binding = transaction.work.get_binding('task', task.id)
            if binding is None:
                transaction.work.bind('task', task.id, book_id=task.book_id, conditions_id=conditions.id)
            elif binding['conditions_id'] != conditions.id:
                raise ConditionsError('conditions_revision_conflict')
        return task

    async def save(self, task):
        return await self.queue.save(task)


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

    def _require_operation_context(self, transaction, request, context):
        try:
            require_policy(context.policy_version)
        except TranslationPolicyUnavailable:
            raise ConditionsError('conditions_unavailable') from None
        attempt = transaction.work.get_attempt(context.attempt_id)
        root = transaction.work.get_root(context.root_id)
        scope = transaction.work.get_scope(context.scope_id)
        expected = thaw(context.to_dict())
        if (attempt is None or root is None or scope is None
                or root.policy_version != context.policy_version
                or root.book_id != request.book_id or attempt.book_id != root.book_id
                or root.conditions_id != self.conditions_id or context.conditions_id != root.conditions_id
                or attempt.conditions_id != root.conditions_id
                or attempt.root_id != root.id or attempt.scope_id != scope.id or scope.root_id != root.id
                or context.ordered_block_ids != scope.ordered_blocks
                or request.scope_id != scope.id or request.ordered_block_ids != scope.ordered_blocks
                or context.task_id != request.task_id or context.run_id != request.run_id
                or context.stage != request.stage.value
                or root.manifest.get('active_run_id') != request.run_id
                or context.prompt_hash != sha256(request.instructions.encode()).hexdigest()
                or any(thaw(attempt.context).get(key) != value for key, value in expected.items())):
            raise ConditionsError('conditions_unavailable')
        return attempt

    def _has_operation_ownership(self, transaction, context, attempt):
        root = transaction.work.get_root(context.root_id)
        scope = transaction.work.get_scope(context.scope_id)
        return (root.superseded_by is None and root.status in {'pending', 'running'}
            and root.generation == attempt.context.get('root_generation')
            and scope.generation == context.generation and scope.status == 'claimed')

    async def generate(self, request):
        with self.service.uow.transaction() as transaction:
            conditions = self.service.require(request.book_id, transaction=transaction)
            if conditions.id != self.conditions_id:
                raise ConditionsError('conditions_revision_conflict')
            if request.call_context is not None and request.call_context.root_id is not None:
                context = request.call_context
                attempt = self._require_operation_context(transaction, request, context)
                if attempt.state != 'created' or not self._has_operation_ownership(transaction, context, attempt):
                    raise RecoveryError('interrupted', context.root_id)
                transaction.work.transition_attempt(attempt.id, 'sent', expected_state='created')
            else:
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
            current_attempt = transaction.work.get_attempt(attempt.id)
            valid = current_attempt is not None and current_attempt.state == 'sent'
            if context.root_id is not None:
                current_attempt = self._require_operation_context(transaction, request, context)
                if response.call_context != context:
                    raise ConditionsError('conditions_unavailable')
                valid = valid and self._has_operation_ownership(transaction, context, current_attempt)
            if valid:
                transaction.work.transition_attempt(attempt.id, 'validated', expected_state='sent',
                    outcome={'schema_version': 1, 'provider': response.provider, 'model': response.model,
                             'call_id': response.call_id})
        self._attempts.set((*self.attempts, attempt.id))
        self._responses.set((*self.responses, response))
        return response


@dataclass(slots=True)
class _CachedBookGeneration:
    fingerprint: tuple
    generation: object
    active_leases: int = 0
    retired: bool = False
    closed: bool = False


class BookRuntimeResolver:
    def __init__(self, service, runtime, factory):
        self.service, self.runtime, self.factory = service, runtime, factory
        self.supported = {'legacy-v1': 1, 'content-plan-v2': 2, 'content-plan-v3': 3}
        self._cache: dict[str, _CachedBookGeneration] = {}

    async def _close_entry(self, entry: _CachedBookGeneration) -> None:
        if not entry.closed:
            entry.closed = True
            await entry.generation.configured_client.aclose()

    def settings_for(self, conditions):
        payload = conditions.payload
        marker = payload['prompt_templates'].get('pronunciation_candidates_version')
        if marker is not None and (type(marker) is not int or marker != 1
                or not payload.get('pronunciation_help_enabled') or payload.get('source_language') != 'en'
                or not payload['prompt_templates'].get('pronunciation_processing')):
            raise ConditionsError('conditions_unavailable')
        if 'name_translation_policy' in payload:
            try:
                policy = NameTranslationPolicy(**thaw(payload['name_translation_policy']))
            except (TypeError, ValueError):
                raise ConditionsError('conditions_unavailable') from None
        else:
            policy = None
        implementation = 3 if payload.get('profile') == 'content-plan-v3' else 1
        expected_versions = {'runtime': 1, 'context_builder': implementation, 'validators': implementation, 'schemas': self.supported.get(payload.get('profile')), 'prompts': implementation}
        if payload.get('english_help', False):
            if payload.get('profile') != 'content-plan-v3':
                raise ConditionsError('conditions_unavailable')
            expected_versions['schemas'] = 4
            expected_versions['prompts'] = 4
        templates = payload.get('prompt_templates')
        template_values = thaw(templates) if templates is not None else None
        marker = (template_values.get('analysis_contract_version')
                  if isinstance(template_values, dict) else None)
        if marker is not None and (type(marker) is not int or marker != 5):
            raise ConditionsError('conditions_unavailable')
        if marker == 5:
            supported_versions = []
            if payload.get('profile') == 'content-plan-v3':
                quoted_versions = {'runtime': 1, 'context_builder': 3,
                    'validators': 3, 'schemas': 5, 'prompts': 5}
                if payload.get('reader_level') == 'a2':
                    quoted_versions['validators'] = 4
                supported_versions.append(quoted_versions)
            elif (payload.get('profile') == 'legacy-v1' and payload.get('reader_level') == 'a2'
                    and not payload.get('english_help', False)):
                supported_versions.append(dict(A2_CONTENT_PLAN_V5_VERSIONS))
        else:
            supported_versions = [expected_versions]
            if payload.get('reader_level') == 'a2':
                supported_versions.append(dict(A2_CONTENT_PLAN_VERSIONS)
                    if payload.get('profile') == 'legacy-v1' and not payload.get('english_help', False)
                    else {**expected_versions, 'validators': 4})
        versions = thaw(payload.get('versions', {}))
        if (payload.get('profile') not in self.supported or not isinstance(versions, dict)
                or any(type(value) is not int for value in versions.values())
                or versions not in supported_versions):
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
        return replace(current, llm=llm, reading=ReadingSettings(reader_level=payload['reader_level'],
                       name_translation_strategy=policy.strategy.value if policy is not None else 'sound'),
                       checks=CheckSettings(**thaw(payload['checks'])),
                       recovery=RecoverySettings(**thaw(payload.get('recovery_policy', asdict(RecoverySettings())))))

    async def _evict_stale_entries(self, current_creds: tuple) -> None:
        stale = [
            (cache_key, cached)
            for cache_key, cached in tuple(self._cache.items())
            if cached.fingerprint[1:] != current_creds
        ]
        for cache_key, cached in stale:
            cached.retired = True
            if self._cache.get(cache_key) is cached:
                self._cache.pop(cache_key, None)
            if cached.active_leases == 0:
                await self._close_entry(cached)

    @asynccontextmanager
    async def lease(self, conditions_id, role):
        if role not in {'analysis', 'processing', 'regeneration'}:
            raise ConditionsError('conditions_unavailable')
        with self.service.uow.transaction() as transaction:
            conditions = transaction.work.get_conditions(conditions_id)
            if conditions is None:
                raise ConditionsError('conditions_unavailable')
        settings = self.settings_for(conditions)
        key = getattr(conditions, 'id', conditions_id)
        current_creds = (
            settings.llm.base_url,
            settings.llm.proxy_url,
            settings.llm.api_key,
        )
        await self._evict_stale_entries(current_creds)
        fingerprint = (
            getattr(conditions, 'revision', None),
            *current_creds,
        )
        entry = self._cache.get(key)
        if entry is not None and not entry.retired and entry.fingerprint == fingerprint:
            entry.active_leases += 1
        else:
            if entry is not None:
                entry.retired = True
                self._cache.pop(key, None)
                if entry.active_leases == 0:
                    await self._close_entry(entry)
            generation = self.factory(settings, conditions=conditions)
            entry = _CachedBookGeneration(
                fingerprint=fingerprint,
                generation=generation,
                active_leases=1,
            )
            self._cache[key] = entry
        try:
            yield entry.generation
        finally:
            entry.active_leases -= 1
            if not entry.retired:
                current_llm = self.runtime.current.settings.llm
                latest_fingerprint = (
                    getattr(conditions, 'revision', None),
                    current_llm.base_url,
                    current_llm.proxy_url,
                    current_llm.api_key,
                )
                if latest_fingerprint != entry.fingerprint:
                    entry.retired = True
                    if self._cache.get(key) is entry:
                        self._cache.pop(key, None)
            if entry.retired and entry.active_leases == 0:
                await self._close_entry(entry)

    async def aclose(self) -> None:
        entries = tuple(self._cache.values())
        self._cache.clear()
        for entry in entries:
            entry.retired = True
        for entry in entries:
            if entry.active_leases == 0:
                await self._close_entry(entry)
