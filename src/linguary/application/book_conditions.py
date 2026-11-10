"""Freeze execution conditions before work and preserve historical provenance."""
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
import json
from uuid import uuid4

from linguary.application.book_overview import BookNotFoundError
from linguary.application.ports.llm import LLMConfigurationError
from linguary.domain.book_conditions import BookConditions, thaw
from linguary.domain import BookStatus


PRONUNCIATION_VERSIONS = {'selection_policy': 2, 'form_rules': 1,
    'source': 'ipa-dict@972cd67dd5f96bc5660e28070cde26e3928abcd7:parser-1'}
A2_CONTENT_PLAN_VERSIONS = {'runtime': 1, 'context_builder': 3, 'validators': 4, 'schemas': 3, 'prompts': 3}
A2_CONTENT_PLAN_V5_VERSIONS = {'runtime': 1, 'context_builder': 3, 'validators': 4, 'schemas': 5, 'prompts': 5}


class ConditionsError(LLMConfigurationError):
    def __init__(self, reason_code):
        self.reason_code = reason_code
        super().__init__(reason_code)


class BookConditionsService:
    def __init__(self, unit_of_work, snapshot_factory, settings_revision, source_digest):
        self.uow = unit_of_work
        self._snapshot_factory = snapshot_factory
        self._settings_revision = settings_revision
        self._source_digest = source_digest

    def freeze_before_analysis(self, book_id, expected_settings_revision):
        read_tx = getattr(self.uow, 'read_transaction', None)
        if read_tx is not None:
            with read_tx() as transaction:
                book = transaction.books.get(book_id)
                if book is None:
                    raise ConditionsError('book_not_found')
                existing = transaction.work.latest_conditions(book_id)
                if existing is None and (
                    book.status != BookStatus.IMPORTED
                    or transaction.analyses.list_for_book(book_id)
                    or transaction.runs.active_for_book(book_id)
                ):
                    raise ConditionsError('legacy_choice_required')
            digest = self.source_digest(book)
            if existing is not None:
                if digest != existing.payload.get('source_sha256'):
                    raise ConditionsError('source_digest_mismatch')
                return existing.id
            with self.uow.transaction() as transaction:
                current_book = transaction.books.get(book_id)
                if current_book is None:
                    raise ConditionsError('book_not_found')
                effective_digest = (
                    digest
                    if current_book.source_epub_path == book.source_epub_path
                    else self.source_digest(current_book)
                )
                existing = transaction.work.latest_conditions(book_id)
                if existing is not None:
                    if effective_digest != existing.payload.get('source_sha256'):
                        raise ConditionsError('source_digest_mismatch')
                    return existing.id
                if (
                    current_book.status != BookStatus.IMPORTED
                    or transaction.analyses.list_for_book(book_id)
                    or transaction.runs.active_for_book(book_id)
                ):
                    raise ConditionsError('legacy_choice_required')
                return self.create(
                    transaction,
                    current_book,
                    expected_settings_revision,
                    source_sha256=effective_digest,
                ).id
        with self.uow.transaction() as transaction:
            book = transaction.books.get(book_id)
            if book is None:
                raise ConditionsError('book_not_found')
            existing = transaction.work.latest_conditions(book_id)
            if existing is not None:
                self.verify_source(book, existing)
                return existing.id
            if book.status != BookStatus.IMPORTED or transaction.analyses.list_for_book(book_id) or transaction.runs.active_for_book(book_id):
                raise ConditionsError('legacy_choice_required')
            return self.create(transaction, book, expected_settings_revision).id

    def create(
        self,
        transaction,
        book,
        settings_revision,
        *,
        reproducibility='complete',
        english_help=False,
        reader_level=None,
        continue_legacy=False,
        pronunciation_help_enabled=False,
        pronunciation_variant=None,
        source_sha256: str | None = None,
        seed_first_chapter: bool | None = None,
    ):
        if settings_revision != self._settings_revision():
            raise ConditionsError('settings_revision_conflict')
        previous = transaction.work.latest_conditions(book.id)
        if previous is not None:
            if source_sha256 is None:
                self.verify_source(book, previous)
            elif source_sha256 != previous.payload.get('source_sha256'):
                raise ConditionsError('source_digest_mismatch')
        payload = self._snapshot_factory(book, settings_revision)
        payload.setdefault('checks', {}).setdefault('auto_approve_glossary', False)
        if seed_first_chapter is not None:
            if type(seed_first_chapter) is not bool:
                raise ConditionsError('invalid_conditions_choice')
            payload.setdefault('checks', {})['seed_first_chapter'] = seed_first_chapter
        from linguary.application.context_builder import snapshot_prompt_templates
        if continue_legacy:
            payload.pop('name_translation_policy', None)
            payload['prompt_templates'] = snapshot_prompt_templates(
                source_language=book.source_language, target_language=book.target_language,
                reader_level=payload.get('reader_level', 'b1'), profile=payload.get('profile', 'legacy-v1'),
                validators_version=payload.get('versions', {}).get('validators', 1),
            )
            payload['prompt_sha256'] = payload_digest(payload['prompt_templates'])
        if reader_level is not None and reader_level not in {'a2', 'b1', 'b2', 'c1'}:
            raise ConditionsError('invalid_reader_level')
        payload['reader_level'] = reader_level or payload['reader_level']
        if english_help and payload['reader_level'] not in {'a2', 'b1', 'b2', 'c1'}:
            raise ConditionsError('invalid_reader_level')
        if type(pronunciation_help_enabled) is not bool:
            raise ConditionsError('invalid_pronunciation_choice')
        if pronunciation_variant not in {None, 'uk', 'us'} or (pronunciation_help_enabled and pronunciation_variant is None):
            raise ConditionsError('invalid_pronunciation_variant')
        if pronunciation_help_enabled and book.source_language != 'en':
            raise ConditionsError('pronunciation_requires_english')
        payload['pronunciation_help_enabled'] = pronunciation_help_enabled
        payload['pronunciation_variant'] = pronunciation_variant if pronunciation_help_enabled else None
        payload['pronunciation_versions'] = dict(PRONUNCIATION_VERSIONS)
        payload['english_help'] = english_help
        if english_help:
            payload['profile'] = 'content-plan-v3'
            payload['versions'] = {'runtime': 1, 'context_builder': 3, 'validators': 3,
                'schemas': 4 if continue_legacy else 5, 'prompts': 4 if continue_legacy else 5}
        profile = payload.get('profile', 'legacy-v1')
        if payload['reader_level'] == 'a2' and not continue_legacy:
            if profile == 'legacy-v1' and not english_help:
                payload['versions'] = dict(A2_CONTENT_PLAN_V5_VERSIONS)
            else:
                payload['versions'] = {**payload['versions'], 'validators': 4}
        elif continue_legacy and not english_help:
            if profile == 'legacy-v1' and payload['reader_level'] == 'a2':
                payload['versions'] = dict(A2_CONTENT_PLAN_VERSIONS)
            elif profile == 'content-plan-v3':
                contract_v5 = payload.get('prompt_templates', {}).get('analysis_contract_version') == 5
                payload['versions'] = {'runtime': 1, 'context_builder': 3,
                    'validators': 4 if payload['reader_level'] == 'a2' else 3,
                    'schemas': 5 if contract_v5 else 3, 'prompts': 5 if contract_v5 else 3}
        effective_v3 = (profile == 'content-plan-v3'
                        or profile == 'legacy-v1' and payload['reader_level'] == 'a2'
                        and not english_help)
        analysis_contract_version = 5 if effective_v3 and not continue_legacy else None
        validators_version = payload.get('versions', {}).get('validators', 1)
        if english_help or reader_level is not None or pronunciation_help_enabled or validators_version == 4:
            payload['prompt_templates'] = snapshot_prompt_templates(source_language=book.source_language,
                target_language=book.target_language, reader_level=payload['reader_level'],
                profile=payload.get('profile', 'legacy-v1'), english_help=english_help,
                pronunciation_help_enabled=pronunciation_help_enabled,
                validators_version=validators_version,
                name_translation_policy=payload.get('name_translation_policy'),
                analysis_contract_version=analysis_contract_version)
            payload['prompt_sha256'] = payload_digest(payload['prompt_templates'])
        payload['source_sha256'] = self.source_digest(book) if source_sha256 is None else source_sha256
        conditions = BookConditions(str(uuid4()), book.id, 1 if previous is None else previous.revision + 1,
            payload, datetime.now(UTC), reproducibility=reproducibility)
        transaction.work.add_conditions(conditions)
        return conditions

    def require(self, book_id, *, transaction=None):
        if transaction is None:
            open_tx = getattr(self.uow, 'read_transaction', self.uow.transaction)
            with open_tx() as current:
                return self.require(book_id, transaction=current)
        book = transaction.books.get(book_id)
        if book is None:
            raise ConditionsError('book_not_found')
        conditions = transaction.work.latest_conditions(book_id)
        if conditions is None:
            raise ConditionsError('legacy_choice_required')
        self.verify_source(book, conditions)
        return conditions

    def source_digest(self, book):
        try:
            return self._source_digest(book.source_epub_path)
        except OSError:
            raise ConditionsError('source_unavailable') from None

    def verify_source(self, book, conditions):
        if self.source_digest(book) != conditions.payload.get('source_sha256'):
            raise ConditionsError('source_digest_mismatch')

    @staticmethod
    def _resolve_initial_options(
        existing: BookConditions | None,
        *,
        reader_level: str | None,
        english_help: bool | None,
        pronunciation_help_enabled: bool | None,
        pronunciation_variant: str | None,
        seed_first_chapter: bool | None,
        default_seed_first_chapter: bool,
    ) -> tuple[bool, dict]:
        if existing is None:
            pron_enabled = bool(pronunciation_help_enabled)
            return True, dict(
                reader_level=reader_level,
                english_help=bool(english_help),
                pronunciation_help_enabled=pron_enabled,
                pronunciation_variant=(
                    pronunciation_variant
                    if (pron_enabled or pronunciation_help_enabled is None)
                    else None
                ),
                seed_first_chapter=seed_first_chapter,
            )
        existing_seed = bool(
            (existing.payload.get('checks') or {}).get(
                'seed_first_chapter',
                default_seed_first_chapter,
            )
        )
        eff_reader_level = (
            reader_level
            if reader_level is not None
            else existing.payload.get('reader_level')
        )
        eff_english_help = (
            english_help
            if english_help is not None
            else bool(existing.payload.get('english_help', False))
        )
        eff_pron_enabled = (
            pronunciation_help_enabled
            if pronunciation_help_enabled is not None
            else bool(existing.payload.get('pronunciation_help_enabled', False))
        )
        if not eff_pron_enabled:
            eff_pron_variant = (
                pronunciation_variant
                if pronunciation_help_enabled is None
                else None
            )
        else:
            eff_pron_variant = (
                pronunciation_variant
                if pronunciation_variant is not None
                else existing.payload.get('pronunciation_variant')
            )
        eff_seed = (
            seed_first_chapter
            if seed_first_chapter is not None
            else existing_seed
        )
        diverges = (
            eff_reader_level != existing.payload.get('reader_level')
            or eff_english_help != bool(existing.payload.get('english_help', False))
            or eff_pron_enabled != bool(existing.payload.get('pronunciation_help_enabled', False))
            or eff_pron_variant != existing.payload.get('pronunciation_variant')
            or eff_seed != existing_seed
        )
        return diverges, dict(
            reader_level=eff_reader_level,
            english_help=eff_english_help,
            pronunciation_help_enabled=eff_pron_enabled,
            pronunciation_variant=eff_pron_variant,
            seed_first_chapter=eff_seed,
        )

    def ensure_initial_for_analysis(
        self,
        book_id: str,
        *,
        settings_revision: int | None = None,
        reader_level: str | None = None,
        english_help: bool | None = None,
        english_help_enabled: bool | None = None,
        pronunciation_help_enabled: bool | None = None,
        pronunciation_variant: str | None = None,
        seed_first_chapter: bool | None = None,
        default_seed_first_chapter: bool = False,
        validate_conditions: Callable[[BookConditions], object] | None = None,
    ) -> BookConditions:
        if (
            english_help is not None
            and english_help_enabled is not None
            and english_help != english_help_enabled
        ):
            raise ConditionsError('invalid_conditions_choice')
        effective_english_help = (
            english_help if english_help is not None else english_help_enabled
        )
        expected_revision = (
            self._settings_revision()
            if settings_revision is None
            else settings_revision
        )
        with self.uow.transaction() as tx:
            book = tx.books.get(book_id)
            if book is None:
                raise BookNotFoundError(f"книги {book_id} нет в хранилище")
            if reader_level is not None and reader_level not in {'a2', 'b1', 'b2', 'c1'}:
                raise ConditionsError('invalid_reader_level')
            if effective_english_help is not None and type(effective_english_help) is not bool:
                raise ConditionsError('invalid_conditions_choice')
            if (
                pronunciation_help_enabled is not None
                and type(pronunciation_help_enabled) is not bool
            ):
                raise ConditionsError('invalid_pronunciation_choice')
            if pronunciation_variant not in {None, 'uk', 'us'}:
                raise ConditionsError('invalid_pronunciation_variant')
            if seed_first_chapter is not None and type(seed_first_chapter) is not bool:
                raise ConditionsError('invalid_conditions_choice')
            if type(default_seed_first_chapter) is not bool:
                raise ConditionsError('invalid_conditions_choice')
            existing = tx.work.latest_conditions(book_id)
            is_clean_imported = (
                book.status is BookStatus.IMPORTED
                and not tx.analyses.list_for_book(book_id)
                and tx.runs.active_for_book(book_id) is None
            )
            diverges, create_kwargs = self._resolve_initial_options(
                existing,
                reader_level=reader_level,
                english_help=effective_english_help,
                pronunciation_help_enabled=pronunciation_help_enabled,
                pronunciation_variant=pronunciation_variant,
                seed_first_chapter=seed_first_chapter,
                default_seed_first_chapter=default_seed_first_chapter,
            )
            if existing is None:
                if not is_clean_imported:
                    raise ConditionsError('legacy_choice_required')
                conditions = self.create(
                    tx,
                    book,
                    expected_revision,
                    **create_kwargs,
                )
                if validate_conditions is not None:
                    validate_conditions(conditions)
                return conditions
            if not diverges:
                self.verify_source(book, existing)
                if validate_conditions is not None:
                    validate_conditions(existing)
                return existing
            if not is_clean_imported:
                raise ConditionsError('conditions_already_selected')
            conditions = self.create(
                tx,
                book,
                expected_revision,
                **create_kwargs,
            )
            if validate_conditions is not None:
                validate_conditions(conditions)
            return conditions

    @staticmethod
    def archive_analysis(
        transaction,
        book_id,
        analysis,
        conditions_id,
        attempt_id=None,
        *,
        replace_binding: bool = False,
    ):
        body = {'schema_version': 1, 'analysis': asdict(analysis)}
        artifact_id, = transaction.work.archive_artifacts([dict(book_id=book_id, conditions_id=conditions_id,
            kind='analysis', scope_id=analysis.section_id, attempt_id=attempt_id, body=body)])
        if conditions_id is None:
            return
        if transaction.work.get_binding('analysis', analysis.id) is not None:
            if replace_binding:
                transaction.work.replace_analysis_binding(
                    analysis.id,
                    book_id=book_id,
                    conditions_id=conditions_id,
                    attempt_id=attempt_id,
                    artifact_revision_id=artifact_id,
                )
            return
        transaction.work.bind('analysis', analysis.id, book_id=book_id, conditions_id=conditions_id,
            attempt_id=attempt_id, artifact_revision_id=artifact_id)


def payload_digest(value):
    return sha256(json.dumps(thaw(value), sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
