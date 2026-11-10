from dataclasses import replace
from datetime import UTC, datetime, timedelta
from linguary.application.diagnostics import Diagnostics, DiagnosticsPolicy
from persistence_factories import make_book, make_llm_call

NOW = datetime(2026, 10, 25, 12, tzinfo=UTC)


def seed(unit_of_work, *, count=1):
    with unit_of_work.transaction() as transaction:
        transaction.books.add(make_book())
        for index in range(count):
            entry = replace(make_llm_call(call_id=f'call-{index}'), started_at=NOW + timedelta(seconds=index))
            transaction.llm_calls.add(entry, response_text='0123456789')


def test_disabled_body_keeps_metadata(unit_of_work):
    seed(unit_of_work)
    diagnostics = Diagnostics(unit_of_work, policy=DiagnosticsPolicy(bodies_enabled=False), clock=lambda: NOW)
    diagnostics.purge_expired()
    with unit_of_work.transaction() as transaction:
        record = transaction.llm_calls.diagnostic_record(call_id='call-0')
        assert record['body'] is None
        assert record['metadata']['body_meta']['body_status'] == 'disabled'
        assert transaction.llm_calls.list_for_book('book-1')[0].cost is not None


def test_ttl_on_read_and_startup(unit_of_work):
    seed(unit_of_work)
    diagnostics = Diagnostics(unit_of_work, policy=DiagnosticsPolicy(retention_seconds=10), clock=lambda: NOW + timedelta(seconds=11))
    diagnostics.purge_expired()
    with unit_of_work.transaction() as transaction:
        assert transaction.llm_calls.response_text_for('call-0') is None
        assert transaction.llm_calls.diagnostic_record(call_id='call-0')['metadata']['body_meta']['body_status'] == 'expired'


def test_per_book_quota(unit_of_work):
    seed(unit_of_work, count=3)
    diagnostics = Diagnostics(unit_of_work, policy=DiagnosticsPolicy(max_book_bytes=20), clock=lambda: NOW)
    diagnostics.purge_expired()
    with unit_of_work.transaction() as transaction:
        assert transaction.llm_calls.diagnostic_record(call_id='call-0')['body'] is None
        assert transaction.llm_calls.diagnostic_record(call_id='call-1')['body'] == '0123456789'
        assert transaction.llm_calls.diagnostic_record(call_id='call-2')['body'] == '0123456789'


def test_delete_book_and_explicit_purge(unit_of_work):
    seed(unit_of_work)
    diagnostics = Diagnostics(unit_of_work, clock=lambda: NOW)
    assert diagnostics.purge('book-1', call_id='call-0') == 1
    assert diagnostics.purge('book-1', call_id='call-0') == 0
    with unit_of_work.transaction() as transaction:
        assert transaction.llm_calls.diagnostic_record(call_id='call-0')['metadata']['body_meta']['body_status'] == 'deleted'


def test_accept_reject_updates_one_call(unit_of_work):
    seed(unit_of_work)
    diagnostics = Diagnostics(unit_of_work, clock=lambda: NOW)
    diagnostics.finalize_attempt(None, 'rejected', body='bounded rejected', call_id='call-0')
    diagnostics.finalize_attempt(None, 'rejected', body='must not replace', call_id='call-0')
    with unit_of_work.transaction() as transaction:
        assert len(transaction.llm_calls.list_for_book('book-1')) == 1
        assert transaction.llm_calls.diagnostic_record(call_id='call-0')['body'] == 'bounded rejected'


def test_periodic_purge_throttles_repeated_calls_while_purge_expired_runs_immediately(
    unit_of_work, monkeypatch
):
    seed(unit_of_work)
    current_time = NOW
    diagnostics = Diagnostics(
        unit_of_work,
        policy=DiagnosticsPolicy(retention_seconds=10),
        clock=lambda: current_time,
    )
    purge_calls: list[datetime | None] = []
    original_purge = diagnostics.purge_expired

    def tracked_purge(now=None):
        purge_calls.append(now)
        return original_purge(now)

    monkeypatch.setattr(diagnostics, 'purge_expired', tracked_purge)

    assert diagnostics.periodic_purge() is True
    assert len(purge_calls) == 1

    current_time = NOW + timedelta(minutes=15)
    assert diagnostics.periodic_purge() is False
    diagnostics.ensure_available()
    assert len(purge_calls) == 1

    diagnostics.purge_expired()
    assert len(purge_calls) == 2
    with unit_of_work.transaction() as transaction:
        assert transaction.llm_calls.response_text_for('call-0') is None

    current_time = NOW + timedelta(hours=1, seconds=1)
    assert diagnostics.periodic_purge() is True
    assert len(purge_calls) == 3


def test_enforce_diagnostics_policy_evicts_by_byte_length_without_loading_untouched_bodies(
    unit_of_work,
):
    from linguary.adapters.db.models import LLMCallLogRow
    from linguary.domain.book_conditions import BookConditions
    from persistence_factories import (
        BLOCK_ID,
        SECOND_BLOCK_ID,
        make_document,
        make_glossary_only_result,
        make_run,
        make_translated_result,
    )

    policy = DiagnosticsPolicy(
        bodies_enabled=True,
        max_body_bytes=64,
        max_book_bytes=40,
        retention_seconds=3600,
    )
    text_input_only = 'привет мир'  # 10 chars, 19 UTF-8 bytes
    text_one = 'абвгдеёжзи'  # 10 chars, 20 UTF-8 bytes
    text_two = 'клмнопрсту'  # 10 chars, 20 UTF-8 bytes

    with unit_of_work.transaction() as transaction:
        transaction.books.add(make_book())
        entry0 = replace(
            make_llm_call(call_id='call-0'),
            started_at=NOW,
            diagnostic_meta={
                'schema_version': 1,
                'decision': 'unknown',
                'input_text': text_input_only,
                'body_meta': {
                    'body_status': 'unavailable',
                    'stored_bytes': 0,
                    'expires_at': (NOW + timedelta(seconds=3600)).isoformat(),
                },
            },
        )
        entry1 = replace(
            make_llm_call(call_id='call-1'),
            started_at=NOW + timedelta(seconds=1),
            diagnostic_meta={
                'schema_version': 1,
                'decision': 'rejected',
                'body_meta': {
                    'body_status': 'available',
                    'stored_bytes': len(text_one.encode('utf-8')),
                    'expires_at': (NOW + timedelta(seconds=3601)).isoformat(),
                },
            },
        )
        entry2 = replace(
            make_llm_call(call_id='call-2'),
            started_at=NOW + timedelta(seconds=2),
            diagnostic_meta={
                'schema_version': 1,
                'decision': 'rejected',
                'body_meta': {
                    'body_status': 'available',
                    'stored_bytes': len(text_two.encode('utf-8')),
                    'expires_at': (NOW + timedelta(seconds=3602)).isoformat(),
                },
            },
        )
        transaction.llm_calls.add(entry0, response_text=None)
        transaction.llm_calls.add(entry1, response_text=text_one)
        transaction.llm_calls.add(entry2, response_text=text_two)

    from sqlalchemy import event

    executed_sql: list[tuple[str, object]] = []
    with unit_of_work.transaction() as transaction:
        engine = transaction._active_session().get_bind()

        def _capture_sql(conn, cursor, statement, parameters, context, executemany):
            executed_sql.append((statement, parameters))

        event.listen(engine, 'before_cursor_execute', _capture_sql)
        try:
            transaction.llm_calls.enforce_diagnostics_policy(policy, NOW, book_id='book-1')
        finally:
            event.remove(engine, 'before_cursor_execute', _capture_sql)

    select_statements = [
        (sql, params)
        for sql, params in executed_sql
        if sql.lstrip().upper().startswith('SELECT')
    ]
    assert len(select_statements) == 2
    projection_sql, _ = select_statements[0]
    assert 'CAST(' in projection_sql.upper()
    row_load_sql, row_load_params = select_statements[1]
    assert 'call-0' in repr(row_load_params)
    assert 'call-1' not in repr(row_load_params)
    assert 'call-2' not in repr(row_load_params)

    with unit_of_work.transaction() as transaction:
        rec0 = transaction.llm_calls.diagnostic_record(call_id='call-0')
        rec1 = transaction.llm_calls.diagnostic_record(call_id='call-1')
        rec2 = transaction.llm_calls.diagnostic_record(call_id='call-2')
        assert rec0['body'] is None
        assert 'input_text' not in rec0['metadata']
        assert rec0['metadata']['body_meta']['body_status'] == 'deleted'
        assert rec1['body'] == text_one
        assert rec2['body'] == text_two

        entry_ttl = replace(
            make_llm_call(call_id='call-ttl'),
            started_at=NOW + timedelta(seconds=3),
            diagnostic_meta={
                'schema_version': 1,
                'decision': 'unknown',
                'input_text': 'истекающий запрос',
                'body_meta': {
                    'body_status': 'unavailable',
                    'stored_bytes': 0,
                    'expires_at': (NOW + timedelta(seconds=3603)).isoformat(),
                },
            },
        )
        transaction.llm_calls.add(entry_ttl, response_text=None)

    with unit_of_work.transaction() as transaction:
        transaction.llm_calls.enforce_diagnostics_policy(
            policy, NOW + timedelta(seconds=3700), book_id='book-1'
        )
        rec_ttl = transaction.llm_calls.diagnostic_record(call_id='call-ttl')
        assert 'input_text' not in rec_ttl['metadata']
        assert rec_ttl['metadata']['body_meta']['body_status'] == 'expired'

        from uuid import uuid4

        conditions = BookConditions(
            id=str(uuid4()),
            book_id='book-1',
            revision=1,
            created_at=NOW,
            reproducibility='complete',
            payload={
                'schema_version': 1,
                'source_epub_sha256': 'a' * 64,
                'settings_revision': 1,
                'reader_level': 'b2',
                'english_help': False,
                'models': {'analysis': 'm', 'processing': 'm', 'correction': 'm'},
                'checks': {'correction_passes': 1},
                'versions': {
                    'prompts': 1,
                    'schemas': 1,
                    'validators': 1,
                    'translation_policy': 1,
                    'correction_policy': 1,
                    'quality_policy': 1,
                },
            },
        )
        transaction.work.add_conditions(conditions)
        assert transaction.work.is_book_paused('book-1') is False
        transaction.work.set_book_paused('book-1', conditions.id, True)
        assert transaction.work.is_book_paused('book-1') is True
        transaction.work.set_book_paused('book-1', conditions.id, False)
        assert transaction.work.is_book_paused('book-1') is False

        transaction.documents.save(make_document())
        run = make_run(run_id='run-many-1', book_id='book-1')
        transaction.runs.add(run)
        r1 = make_translated_result(block_id=BLOCK_ID)
        r2 = make_glossary_only_result(block_id=SECOND_BLOCK_ID)
        transaction.results.replace_results(run.id, (r1, r2))
        fetched = transaction.results.get_many(
            run.id, (SECOND_BLOCK_ID, 'sec-1:missing', BLOCK_ID)
        )
        assert fetched == {BLOCK_ID: r1, SECOND_BLOCK_ID: r2}
        assert transaction.results.get_many(run.id, ()) == {}


async def test_result_bindings_filters_by_block_ids_pk_and_translation_call_scopes_binding_reads(
    unit_of_work,
) -> None:
    from sqlalchemy import event
    from uuid import uuid4
    from linguary.adapters.llm import FakeLLMClient, FakeReply
    from linguary.application.block_processing import TranslationCall, note_id_factory
    from linguary.application.context_builder import ContextBuilder
    from linguary.application.retry_policy import RetryPolicy
    from linguary.application.validators import TranslationResultValidator
    from linguary.domain import (
        BlockTreatment,
        GlossarySnapshot,
        TranslationUnit,
        TranslationUnitItem,
    )
    from linguary.domain.book_conditions import BookConditions
    from linguary.domain.llm_log import CallContext
    from linguary.domain.recovery import CallAttempt
    from persistence_factories import (
        BLOCK_ID,
        SECOND_BLOCK_ID,
        make_analysis,
        make_document,
        make_glossary_only_result,
        make_run,
        make_translated_result,
    )

    cond_id = str(uuid4())
    att_id = str(uuid4())
    with unit_of_work.transaction() as tx:
        tx.books.add(make_book())
        doc = make_document()
        tx.documents.save(doc)
        conditions = BookConditions(
            id=cond_id,
            book_id='book-1',
            revision=1,
            created_at=NOW,
            reproducibility='complete',
            payload={
                'schema_version': 1,
                'source_epub_sha256': 'a' * 64,
                'settings_revision': 1,
                'reader_level': 'b2',
                'english_help': False,
                'models': {'analysis': 'm', 'processing': 'm', 'correction': 'm'},
                'checks': {'correction_passes': 1},
                'versions': {
                    'prompts': 1,
                    'schemas': 1,
                    'validators': 1,
                    'translation_policy': 1,
                    'correction_policy': 1,
                    'quality_policy': 1,
                },
            },
        )
        tx.work.add_conditions(conditions)
        run = make_run(run_id='run-scoped-bindings', book_id='book-1')
        tx.runs.add(run)
        r1 = make_translated_result(block_id=BLOCK_ID)
        r2 = make_glossary_only_result(block_id=SECOND_BLOCK_ID)
        tx.results.replace_results(run.id, (r1, r2))
        tx.work.add_attempt(
            CallAttempt(
                att_id,
                'book-1',
                state='accepted',
                conditions_id=cond_id,
                run_id=run.id,
                context={'schema_version': 1, 'stage': 'translation_unit', 'ordered_block_ids': [BLOCK_ID, SECOND_BLOCK_ID]},
            )
        )
        art_ids = tx.work.archive_artifacts(
            [
                dict(book_id='book-1', conditions_id=cond_id, attempt_id=att_id, run_id=run.id, kind='block_result', scope_id=BLOCK_ID, body={'schema_version': 1, 'result': {'block_id': BLOCK_ID}}),
                dict(book_id='book-1', conditions_id=cond_id, attempt_id=att_id, run_id=run.id, kind='block_result', scope_id=SECOND_BLOCK_ID, body={'schema_version': 1, 'result': {'block_id': SECOND_BLOCK_ID}}),
            ]
        )
        tx.work.bind('result', f'{run.id}:{BLOCK_ID}', book_id='book-1', conditions_id=cond_id, attempt_id=att_id, artifact_revision_id=art_ids[0])
        tx.work.bind('result', f'{run.id}:{SECOND_BLOCK_ID}', book_id='book-1', conditions_id=cond_id, attempt_id=att_id, artifact_revision_id=art_ids[1])

    executed_sql: list[tuple[str, object]] = []
    with unit_of_work.transaction() as tx:
        engine = tx._active_session().get_bind()

        def _capture(conn, cursor, statement, parameters, context, executemany):
            executed_sql.append((statement, parameters))

        event.listen(engine, 'before_cursor_execute', _capture)
        try:
            assert tx.work.result_bindings('run-scoped-bindings', block_ids=()) == {}
            assert [s for s, _ in executed_sql if s.lstrip().upper().startswith('SELECT')] == []
            scoped = tx.work.result_bindings('run-scoped-bindings', block_ids=(BLOCK_ID,))
        finally:
            event.remove(engine, 'before_cursor_execute', _capture)

    assert set(scoped) == {BLOCK_ID}
    selects = [(s, p) for s, p in executed_sql if s.lstrip().upper().startswith('SELECT')]
    assert len(selects) == 1
    stmt, params = selects[0]
    assert ' IN ' in stmt.upper()
    assert f'run-scoped-bindings:{BLOCK_ID}' in repr(params)
    assert f'run-scoped-bindings:{SECOND_BLOCK_ID}' not in repr(params)

    with unit_of_work.read_transaction() as tx:
        full = tx.work.result_bindings('run-scoped-bindings')
        assert set(full) == {BLOCK_ID, SECOND_BLOCK_ID}

    # Verify TranslationCall.request and TranslationCall.save pass block_ids to result_bindings
    binding_calls: list[tuple[str, object]] = []
    from linguary.adapters.db.work_state import SqlAlchemyWorkRepository

    orig_rb = SqlAlchemyWorkRepository.result_bindings

    def _spy_rb(self_repo, r_id, block_ids=None):
        binding_calls.append((r_id, None if block_ids is None else tuple(block_ids)))
        return orig_rb(self_repo, r_id, block_ids=block_ids)

    SqlAlchemyWorkRepository.result_bindings = _spy_rb  # type: ignore[method-assign]
    try:
        with unit_of_work.read_transaction() as tx:
            doc = tx.documents.get('book-1')
        sec = doc.ordered_sections[0]
        b0 = sec.ordered_blocks[0]
        unit = TranslationUnit(
            id='unit-single',
            section_id=sec.id,
            items=(TranslationUnitItem(block=b0, treatment=BlockTreatment.TRANSLATE),),
        )
        analysis = make_analysis()
        ctx = ContextBuilder(content_plan_profile='legacy-v1').for_translation_unit(
            document=doc,
            section=sec,
            unit=unit,
            analysis=analysis,
            glossary=GlossarySnapshot(),
        )
        new_att_id = str(uuid4())
        with unit_of_work.transaction() as tx:
            tx.work.add_attempt(
                CallAttempt(
                    new_att_id,
                    'book-1',
                    state='sent',
                    conditions_id=cond_id,
                    run_id='run-scoped-bindings',
                    context={'schema_version': 1, 'stage': 'translation_unit', 'ordered_block_ids': [b0.id]},
                )
            )
        call_ctx = CallContext(
            conditions_id=cond_id,
            stage='translation_unit',
            scope_id='unit-single',
            ordered_block_ids=(b0.id,),
            attempt_id=new_att_id,
            generation=0,
            assigned_model='m',
            prompt_hash='0' * 64,
            run_id='run-scoped-bindings',
        )
        from linguary.domain import LLMCallStage
        ctx = replace(
            ctx,
            request=replace(
                ctx.request,
                call_context=call_ctx,
                book_id='book-1',
                stage=LLMCallStage.TRANSLATION_UNIT,
            ),
        )
        fake = FakeLLMClient(
            [
                FakeReply.of_payload(
                    {
                        'blocks': [
                            {
                                'source_block_id': 'b1',
                                'treatment': 'TRANSLATE',
                                'translated_text': 'Обновлённый перевод достаточной длины на русском языке.',
                                'glossary_hints': [],
                            }
                        ],
                        'language_notes': [],
                    }
                )
            ]
        )

        async def _no_sleep(_s: float) -> None:
            return None

        t_call = TranslationCall(
            unit_of_work=unit_of_work,
            llm_client=fake,
            retry_policy=RetryPolicy(max_attempts=1),
            validator=TranslationResultValidator(
                target_language_min_share=0.5,
                language_check_min_length=20,
                repetition_ngram_size=4,
                repetition_min_repeats=3,
            ),
            sleep=_no_sleep,
        )
        res = await t_call.request(
            ctx,
            unit=unit,
            target_language='ru',
            note_ids=note_id_factory(chapter_number=1, used=()),
        )
        with unit_of_work.transaction() as tx:
            tx.work.transition_attempt(new_att_id, 'validated', expected_state='sent')
        t_call._validated_response.set(
            replace(t_call._validated_response.get(), call_context=call_ctx)
        )
        assert t_call.save(book_id='book-1', run_id='run-scoped-bindings', results=res.results, reconcile=False) is True
    finally:
        SqlAlchemyWorkRepository.result_bindings = orig_rb  # type: ignore[method-assign]

    assert len(binding_calls) >= 2
    assert all(b_ids == (BLOCK_ID,) for _, b_ids in binding_calls)


async def test_llm_call_log_persists_cached_and_reasoning_tokens_in_diagnostic_meta(
    unit_of_work,
) -> None:
    import pytest
    from linguary.adapters.llm import FakeLLMClient, FakeReply, LoggingLLMClient
    from linguary.application.ports.llm import LLMRequest, LLMResponseFormatError
    from linguary.domain import LLMCallStage, TokenUsage

    with unit_of_work.transaction() as tx:
        tx.books.add(make_book())

    valid_payload = {
        'blocks': [
            {
                'source_block_id': 'b1',
                'treatment': 'TRANSLATE',
                'translated_text': 'Обновлённый перевод достаточной длины на русском языке.',
                'glossary_hints': [],
            }
        ],
        'language_notes': [],
    }
    fake = FakeLLMClient.replying(
        [
            FakeReply.of_payload(
                valid_payload,
                usage=TokenUsage(
                    prompt_tokens=120,
                    completion_tokens=45,
                    cached_prompt_tokens=80,
                    reasoning_tokens=15,
                ),
            ),
            LLMResponseFormatError(
                'невалидный JSON ответа',
                raw_content='{broken',
                usage=TokenUsage(
                    prompt_tokens=90,
                    completion_tokens=30,
                    cached_prompt_tokens=0,
                    reasoning_tokens=12,
                ),
            ),
            FakeReply.of_payload(
                valid_payload,
                usage=TokenUsage(prompt_tokens=50, completion_tokens=10),
            ),
        ]
    )
    tick = 0

    def _clock() -> datetime:
        nonlocal tick
        current = NOW + timedelta(seconds=tick)
        tick += 1
        return current

    client = LoggingLLMClient(
        inner=fake,
        unit_of_work=unit_of_work,
        provider='fake',
        model='fake-model',
        clock=_clock,
        debug_payloads=True,
    )
    req = LLMRequest.for_translation(
        instructions='Системная инструкция.',
        input_text='Исходный блок для перевода.',
    ).with_logging(book_id='book-1', stage=LLMCallStage.TRANSLATION_UNIT)

    resp1 = await client.generate(req)
    with pytest.raises(LLMResponseFormatError) as exc_info:
        await client.generate(req)
    resp3 = await client.generate(req)

    with unit_of_work.transaction() as tx:
        rec1 = tx.llm_calls.diagnostic_record(call_id=resp1.call_id)
        rec2 = tx.llm_calls.diagnostic_record(call_id=exc_info.value.call_id)
        rec3 = tx.llm_calls.diagnostic_record(call_id=resp3.call_id)
        assert rec1['metadata']['usage_details'] == {
            'cached_prompt_tokens': 80,
            'reasoning_tokens': 15,
        }
        assert rec2['metadata']['usage_details'] == {
            'cached_prompt_tokens': 0,
            'reasoning_tokens': 12,
        }
        assert 'usage_details' not in rec3['metadata']

        entries = tx.llm_calls.list_for_book('book-1')
        assert len(entries) == 3
        assert entries[0].usage == TokenUsage(
            prompt_tokens=120,
            completion_tokens=45,
            cached_prompt_tokens=80,
            reasoning_tokens=15,
        )
        assert entries[1].usage == TokenUsage(
            prompt_tokens=90,
            completion_tokens=30,
            cached_prompt_tokens=0,
            reasoning_tokens=12,
        )
        assert entries[2].usage == TokenUsage(
            prompt_tokens=50,
            completion_tokens=10,
            cached_prompt_tokens=None,
            reasoning_tokens=None,
        )

        single = tx.llm_calls.get_for_book('book-1', resp1.call_id)
        assert single is not None
        assert single.usage == entries[0].usage

    diagnostics = Diagnostics(unit_of_work, clock=lambda: NOW + timedelta(seconds=10))
    diagnostics.purge('book-1')
    with unit_of_work.transaction() as tx:
        after_purge = tx.llm_calls.list_for_book('book-1')
        assert after_purge[0].usage == TokenUsage(
            prompt_tokens=120,
            completion_tokens=45,
            cached_prompt_tokens=80,
            reasoning_tokens=15,
        )



