from dataclasses import asdict, replace
import json
import pytest
from fastapi.testclient import TestClient
from web_scenario import store_book, start_run, translated_result
from test_quality_api import setup_review, recovery_fixture
from linguary.adapters.config import load_settings
from linguary.adapters.config.settings import RecoverySettings
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.composition import build_application, drain_tasks
from linguary.domain import BookStatus, TaskStatus, LLMCallStage
from linguary.web import create_app

HTML = {'accept': 'text/html'}


def test_unknown_legacy_level_label(client, application):
    scenario = store_book(application)
    response = client.get(f'/books/{scenario.book_id}', headers=HTML)
    assert 'Уровень читателя неизвестен' in response.text
    assert 'Сохранённые условия отсутствуют' in response.text
    assert 'Продолжить со старыми материалами' in response.text
    assert 'Новый анализ заменит текущий анализ' in response.text
    assert 'Подтверждаю новый анализ' in response.text
    with application.unit_of_work.transaction() as tx:
        assert tx.work.latest_conditions(scenario.book_id) is None and tx.tasks.list_for_book(scenario.book_id) == ()


def test_review_form_off_and_requires_budget(configured_client, configured_application, llm_client):
    scenario, _ = setup_review(configured_client, configured_application)
    response = configured_client.get(f'/books/{scenario.book_id}', headers=HTML)
    assert 'Добровольная вычитка выключена' in response.text
    assert 'name="budget_usd"' in response.text and 'name="budget_microusd"' not in response.text
    assert 'Предел расходов, USD' in response.text
    assert 'route_selection' in response.text and 'review' in response.text
    assert 'Оценка одного обращения' in response.text
    assert 'не гарантирует' in response.text
    assert 'Исправление оплачивается отдельно' in response.text
    assert 'name="scope"' in response.text
    assert llm_client.requests == []


@pytest.mark.parametrize('budget', ['0', '-1', 'NaN', 'Infinity', '0.0000001', '1e999999999'])
def test_invalid_html_usd_never_enqueues(configured_client, configured_application, budget):
    scenario, command = setup_review(configured_client, configured_application)
    command.pop('budget_microusd')
    response = configured_client.post(f'/books/{scenario.book_id}/quality-review',
        data={**command, 'budget_usd': budget}, headers=HTML)
    assert response.status_code == 422
    assert 'Введите положительную сумму USD' in response.text
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.list_reviews(scenario.book_id) == ()


def test_html_usd_exactly_preserves_user_limit(configured_client, configured_application):
    scenario, command = setup_review(configured_client, configured_application)
    command.pop('budget_microusd')
    response = configured_client.post(f'/books/{scenario.book_id}/quality-review',
        data={**command, 'budget_usd': '0.001201'}, headers=HTML, follow_redirects=False)
    assert response.status_code == 303
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.list_reviews(scenario.book_id)[0].budget_limit == 1201


def test_recovery_limit_explained(configured_client, configured_application):
    scenario, root = recovery_fixture(configured_client, configured_application)
    response = configured_client.get(f'/books/{scenario.book_id}', headers=HTML)
    assert 'Покрытие: 0 из 2' in response.text
    assert 'Обращения: 12 из 12' in response.text
    assert 'Достигнут предел обращений' in response.text
    assert f'/recovery/{root.id}/limit' in response.text
    assert 'Подтверждаю увеличение предела' in response.text
    assert 'не запускает повтор автоматически' in response.text


def test_production_v3_env_is_explicit_before_http_analysis(settings, monkeypatch):
    monkeypatch.setenv('LINGUARY_CONTENT_PLAN_PROFILE', 'content-plan-v3')
    app = build_application(settings=replace(settings, llm=replace(settings.llm, model='fake', api_key='synthetic')),
        llm_client=FakeLLMClient())
    try:
        scenario = store_book(app, status=BookStatus.IMPORTED, analysed_chapters=())
        with TestClient(create_app(application=app, run_background_worker=False), base_url=app.settings.server.base_url) as client:
            response = client.post(f'/books/{scenario.book_id}/analyze')
            assert response.status_code == 202, response.text
        with app.unit_of_work.transaction() as tx:
            conditions = tx.work.latest_conditions(scenario.book_id)
            assert conditions.payload['profile'] == 'content-plan-v3'
            assert conditions.payload['versions']['schemas'] == 5
            assert conditions.payload['recovery_policy'] == asdict(app.settings.recovery)
            assert tx.work.get_binding('task', response.json()['id'])['conditions_id'] == conditions.id
    finally:
        app.dispose()


async def test_recovery_env_is_pinned_across_restart_and_actual_calls(tmp_path, monkeypatch):
    from linguary.application.use_cases.approve_glossary import StartProcessingCommand
    from linguary.application.use_cases.retry_task import RetryTaskCommand
    config = tmp_path / 'config.toml'
    config.write_text(f'[storage]\ndata_root={json.dumps(str(tmp_path / "data"))}\n[llm]\nmodel="fake"\napi_key="synthetic"\n')
    initial = load_settings(config_path=config, environ={'LINGUARY_RECOVERY_MAX_CALLS': '1', 'LINGUARY_RECOVERY_MAX_SCOPE_CALLS': '1'})
    fake = FakeLLMClient()
    app = build_application(settings=initial, llm_client=fake)
    scenario = store_book(app, status=BookStatus.IMPORTED, analysed_chapters=())
    fake.replies = [FakeReply.of_payload(dict(translation_instructions='Сохраняй смысл.', chapter_summary='A scene.',
        glossary_candidates=[], before_you_read=[], treatment_plan=[dict(source_block_id=f'b{index}', treatment='TRANSLATE') for index, _ in enumerate(scenario.block_ids(), 1)])), FakeReply(content='{')]
    await app.analysis.start(scenario.book_id)
    analysed = await drain_tasks(app)
    assert all(task.status == TaskStatus.SUCCEEDED for task in analysed), [task.last_error for task in analysed]
    with app.unit_of_work.transaction() as tx:
        conditions = tx.work.latest_conditions(scenario.book_id)
        assert conditions.payload['recovery_policy']['max_calls'] == 1
    app.start_processing()(StartProcessingCommand(scenario.book_id))
    task, = await app.processing.start(scenario.book_id)
    await drain_tasks(app)
    with app.unit_of_work.transaction() as tx:
        root, = tx.work.list_roots(scenario.book_id)
        assert root.calls_used == 1 and root.reason == 'limit'
    assert sum(request.stage == LLMCallStage.TRANSLATION_UNIT for request in fake.requests) == 1
    await app.aclose()
    changed = load_settings(config_path=config, environ={'LINGUARY_RECOVERY_MAX_CALLS': '99', 'LINGUARY_RECOVERY_MAX_SCOPE_CALLS': '99'})
    restarted_fake = FakeLLMClient()
    restarted = build_application(settings=changed, llm_client=restarted_fake)
    try:
        restarted.retry_task(RetryTaskCommand(task.id))
        await drain_tasks(restarted)
        with restarted.unit_of_work.transaction() as tx:
            assert tx.work.get_root(root.id) == root
            current = tx.work.latest_conditions(scenario.book_id)
        assert restarted.book_runtime.settings_for(current).recovery.max_calls == 1
        legacy = replace(current, payload={k: v for k, v in current.payload.items() if k != 'recovery_policy'})
        assert restarted.book_runtime.settings_for(legacy).recovery == RecoverySettings()
        assert restarted_fake.requests == []
    finally:
        await restarted.aclose()


@pytest.mark.parametrize('change,message', [({'route_id': 'missing'}, 'Выбранная модель или зарегистрированный маршрут недоступны'),
    ({'expected_manifest_sha': '0' * 64}, 'Материалы книги изменились'), ({'budget_microusd': 1}, 'Указанного предела расходов недостаточно')])
def test_html_review_conflicts_are_russian_and_do_not_start(configured_client, configured_application, change, message):
    scenario, command = setup_review(configured_client, configured_application)
    response = configured_client.post(f'/books/{scenario.book_id}/quality-review', data={**command, **change}, headers=HTML)
    assert response.status_code == 409
    assert message in response.text and 'reason_code' not in response.text
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.list_reviews(scenario.book_id) == () and tx.tasks.list_for_book(scenario.book_id) == ()


def test_html_unproven_price_is_russian(configured_client, configured_application):
    scenario, command = setup_review(configured_client, configured_application)
    route = configured_application.quality_review.routes['test', 'review']
    configured_application.quality_review.routes['test', 'review'] = replace(route, proof='')
    response = configured_client.post(f'/books/{scenario.book_id}/quality-review', data=command, headers=HTML)
    assert response.status_code == 409 and 'Нет подтверждённой цены' in response.text and 'reason_code' not in response.text


def test_reanalyze_confirmation_and_html_stale_revision(configured_client, configured_application, llm_client):
    from lxml import html
    scenario, _ = setup_review(configured_client, configured_application)
    tree = html.fromstring(configured_client.get(f'/books/{scenario.book_id}', headers=HTML).text)
    form = tree.xpath('//form[input[@name="action" and @value="reanalyze"]]')[0]
    confirmation = form.xpath('.//input[@name="confirmed"]')[0]
    assert confirmation.get('required') is not None and confirmation.get('checked') is None
    values = {field.get('name'): field.get('value') for field in form.xpath('.//input[@type="hidden"]')}
    response = configured_client.post(form.get('action'), data={**values, 'confirmed': 'false'}, headers=HTML)
    assert response.status_code == 422 and 'reason_code' not in response.text
    response = configured_client.post(form.get('action'), data={**values, 'expected_revision': '0', 'confirmed': 'true'}, headers=HTML)
    assert response.status_code == 409 and 'Условия книги изменились' in response.text and 'reason_code' not in response.text
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.latest_conditions(scenario.book_id).revision == 1
        assert tx.analyses.list_for_book(scenario.book_id) and tx.tasks.list_for_book(scenario.book_id) == ()
    assert llm_client.requests == []


def test_no_internal_ids_in_learning_text(client, application):
    from lxml import html
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    start_run(application, scenario, results=[translated_result(b) for b in scenario.block_ids()])
    tree = html.fromstring(client.get(f'/books/{scenario.book_id}/result', headers=HTML).text)
    layers = tree.xpath('//p[@class="source" or @class="translation"]')
    visible = ' '.join(layer.text_content() for layer in layers)
    assert 'It is a truth universally acknowledged.' in visible and 'Общепризнанная истина' in visible
    assert all(block_id not in visible for block_id in scenario.block_ids())
    assert all(marker not in visible for marker in ('source_block_id', 'context_facts', 'required_points', 'artifact_sha'))


def test_scope_selection_get_is_read_only(configured_client, configured_application, llm_client):
    from lxml import html
    scenario, _ = setup_review(configured_client, configured_application)
    expected = configured_application.quality_review.preview(scenario.book_id, scenario.block_ids())
    response = configured_client.get(f'/books/{scenario.book_id}/result', params={'review_scope': scenario.section().id}, headers=HTML)
    assert response.status_code == 200
    tree = html.fromstring(response.text)
    form = tree.xpath('//form[contains(@action,"/quality-review")]')[0]
    assert json.loads(form.xpath('.//input[@name="scope"]')[0].get('value')) == list(scenario.block_ids())
    assert form.xpath('.//input[@name="expected_manifest_sha"]')[0].get('value') == expected['manifest_sha']
    assert not form.xpath('ancestor::*[@data-result-header]')
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.latest_conditions(scenario.book_id).revision == 1
        assert tx.tasks.list_for_book(scenario.book_id) == () and tx.work.list_reviews(scenario.book_id) == ()
    assert llm_client.requests == []


@pytest.mark.parametrize('status', [BookStatus.READY_TO_EXPORT, BookStatus.COMPLETED])
def test_incomplete_html_does_not_claim_current_epub_is_ready(client, application, status):
    scenario = store_book(application, status=status)
    start_run(application, scenario, results=[translated_result(scenario.block_ids()[0])])
    response = client.get(f'/books/{scenario.book_id}', headers=HTML)
    assert 'Книга готова к сборке EPUB' not in response.text
    assert 'Итоговый EPUB собран.' not in response.text
    assert 'полный EPUB пока недоступен' in response.text


def test_saved_conditions_visible_on_detail_result_and_poll(configured_client, configured_application):
    scenario, _ = setup_review(configured_client, configured_application)
    for endpoint, headers in [('', HTML), ('/result', HTML),
            ('/result', {**HTML, 'x-linguary-fragment': 'result-header'})]:
        response = configured_client.get(f'/books/{scenario.book_id}{endpoint}', headers=headers)
        assert response.status_code == 200
        assert 'Редакция условий: 1' in response.text
        assert 'Профиль обработки: legacy-v1' in response.text
        assert 'Модель анализа: fake-model' in response.text
        assert 'Модель обработки: fake-model' in response.text
        assert 'Сохранённые условия без способа передачи имён' in response.text


def test_html_uploaded_budget_is_rejected_without_queue(configured_client, configured_application):
    scenario, command = setup_review(configured_client, configured_application)
    command.pop('budget_microusd')
    response = configured_client.post(f'/books/{scenario.book_id}/quality-review', data=command,
        files={'budget_usd': ('budget.txt', b'1.00', 'text/plain')}, headers=HTML)
    assert response.status_code == 422
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.list_reviews(scenario.book_id) == ()
        assert tx.tasks.list_for_book(scenario.book_id) == ()


def test_partial_book_result_preserves_primary_pronunciation_hint_id_across_chapters(client, application):
    from linguary.application.book_overview import book_result
    from linguary.domain import BlockProcessingResult, BlockTreatment
    from linguary.domain.content_plan import SourceAnchor
    from linguary.domain.pronunciation import PronunciationAssistance, PronunciationEvidence

    ch1_text = "The colonel arrived early."
    ch2_text = "Later the colonel spoke softly."
    ch3_text = "Everyone listened quietly."
    scenario = store_book(
        application,
        status=BookStatus.READY_TO_EXPORT,
        chapters=((ch1_text,), (ch2_text,), (ch3_text,)),
    )
    b_ch1 = scenario.block_ids(0)[0]
    b_ch2 = scenario.block_ids(1)[0]
    b_ch3 = scenario.block_ids(2)[0]

    evidence = PronunciationEvidence(
        source_id="synthetic-dict",
        source_version="v1",
        entry_id="entry-colonel",
        sense_id="sense-1",
        part_of_speech="noun",
        dictionary_form="colonel",
        dictionary_ipa="ˈkɜː.nəl",
        attribution="Synthetic Dictionary",
    )
    occ_ch1 = PronunciationAssistance(
        anchor=SourceAnchor(b_ch1, ch1_text.index("colonel"), ch1_text.index("colonel") + len("colonel"), "colonel"),
        form="colonel",
        ipa="ˈkɜː.nəl",
        variant="uk",
        hint_id="ph-primary-chapter-1",
        evidence=evidence,
    )
    occ_ch2 = PronunciationAssistance(
        anchor=SourceAnchor(b_ch2, ch2_text.index("colonel"), ch2_text.index("colonel") + len("colonel"), "colonel"),
        form="colonel",
        ipa="ˈkɜː.nəl",
        variant="uk",
        hint_id="ph-secondary-chapter-2",
        evidence=evidence,
    )

    start_run(
        application,
        scenario,
        results=[
            BlockProcessingResult(
                block_id=b_ch1,
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Полковник прибыл рано.",
                pronunciation_assistances=(occ_ch1,),
            ),
            BlockProcessingResult(
                block_id=b_ch2,
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Позже полковник заговорил тихо.",
                pronunciation_assistances=(occ_ch2,),
            ),
            BlockProcessingResult(
                block_id=b_ch3,
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Все слушали молча.",
            ),
        ],
    )

    partial = book_result(
        application.unit_of_work,
        scenario.book_id,
        include_quality=False,
        block_ids=(b_ch2,),
    )
    assert [chapter.section_id for chapter in partial.chapters] == [scenario.section(1).id]
    assert partial.prior_pronunciation_hints[("colonel", "uk", "ˈkɜː.nəl")] == "ph-primary-chapter-1"

    single = client.get(f"/books/{scenario.book_id}/blocks/{b_ch2}")
    assert single.status_code == 200
    payload = single.json()
    assert len(payload["pronunciation_assistances"]) == 1
    assert payload["pronunciation_assistances"][0]["hint_id"] == "ph-primary-chapter-1"
    assert payload["pronunciation_assistances"][0]["is_primary"] is False

    fragment = client.get(
        f"/books/{scenario.book_id}/blocks/{b_ch2}",
        headers={**HTML, "X-Linguary-Fragment": "result-block"},
    )
    assert fragment.status_code == 200
    assert 'href="#ph-primary-chapter-1"' in fragment.text
    assert 'id="ph-secondary-chapter-2"' not in fragment.text


def test_partial_book_result_and_book_overview_avoid_loading_unaffected_chapter_results(configured_application):
    application = configured_application
    from uuid import uuid4
    from web_scenario import continue_legacy
    from linguary.adapters.db.repositories import SqlAlchemyBlockResultRepository
    from linguary.application.book_overview import book_overview, book_result
    from linguary.application.correction import CorrectionInputs, correction_input_fingerprint
    from linguary.application.correction_dto import (
        CORRECTION_VERSIONS_V2,
        CorrectionSnapshotV2,
        parse_correction_snapshot,
    )
    from linguary.domain import BlockProcessingResult, BlockTreatment
    from linguary.domain.book_conditions import thaw
    from linguary.domain.content_plan import SourceAnchor, canonical_sha
    from linguary.domain.correction import CORRECTION_POLICY_VERSION_V2, CorrectionCandidate
    from linguary.domain.pronunciation import PronunciationAssistance, PronunciationEvidence
    from linguary.domain.recovery import CallAttempt

    ch1_p1 = "The colonel arrived early."
    ch1_p2 = "Plain paragraph without pronunciation hints."
    ch2_p1 = "Later the colonel spoke softly."
    ch3_p1 = "Translated block in chapter three."
    ch3_p2 = "Untranslated block keeping book in PROCESSING."
    scenario = store_book(
        application,
        book_id="book-perf-queries",
        status=BookStatus.PROCESSING,
        chapters=((ch1_p1, ch1_p2), (ch2_p1,), (ch3_p1, ch3_p2)),
    )
    src = application.paths.source_epub(scenario.book_id)
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"synthetic perf epub")
    with application.unit_of_work.transaction() as tx:
        rec = tx.books.get(scenario.book_id)
        tx.books.save(replace(rec, source_epub_path=str(src)))
    continue_legacy(application, scenario.book_id)

    b_ch1_1, b_ch1_2 = scenario.block_ids(0)
    b_ch2_1, = scenario.block_ids(1)
    b_ch3_1, _ = scenario.block_ids(2)

    evidence = PronunciationEvidence(
        source_id="synthetic-dict",
        source_version="v1",
        entry_id="entry-colonel",
        sense_id="sense-1",
        part_of_speech="noun",
        dictionary_form="colonel",
        dictionary_ipa="ˈkɜː.nəl",
        attribution="Synthetic Dictionary",
    )
    occ_ch1 = PronunciationAssistance(
        anchor=SourceAnchor(b_ch1_1, ch1_p1.index("colonel"), ch1_p1.index("colonel") + len("colonel"), "colonel"),
        form="colonel",
        ipa="ˈkɜː.nəl",
        variant="uk",
        hint_id="ph-ch1",
        evidence=evidence,
    )
    occ_ch2 = PronunciationAssistance(
        anchor=SourceAnchor(b_ch2_1, ch2_p1.index("colonel"), ch2_p1.index("colonel") + len("colonel"), "colonel"),
        form="colonel",
        ipa="ˈkɜː.nəl",
        variant="uk",
        hint_id="ph-ch2",
        evidence=evidence,
    )
    start_run(
        application,
        scenario,
        results=[
            BlockProcessingResult(b_ch1_1, BlockTreatment.TRANSLATE, "Полковник прибыл рано.", pronunciation_assistances=(occ_ch1,)),
            BlockProcessingResult(b_ch1_2, BlockTreatment.TRANSLATE, "Обычный абзац без транскрипций."),
            BlockProcessingResult(b_ch2_1, BlockTreatment.TRANSLATE, "Позже полковник заговорил тихо.", pronunciation_assistances=(occ_ch2,)),
            BlockProcessingResult(b_ch3_1, BlockTreatment.TRANSLATE, "Переведённый блок третьей главы."),
        ],
    )

    get_many_calls: list[tuple[str, ...]] = []
    list_for_run_calls = 0
    orig_get_many = SqlAlchemyBlockResultRepository.get_many
    orig_list_for_run = SqlAlchemyBlockResultRepository.list_for_run

    def _spy_get_many(self_repo, run_id, block_ids):
        get_many_calls.append(tuple(block_ids))
        return orig_get_many(self_repo, run_id, block_ids)

    def _spy_list_for_run(self_repo, run_id):
        nonlocal list_for_run_calls
        list_for_run_calls += 1
        return orig_list_for_run(self_repo, run_id)

    SqlAlchemyBlockResultRepository.get_many = _spy_get_many  # type: ignore[method-assign]
    SqlAlchemyBlockResultRepository.list_for_run = _spy_list_for_run  # type: ignore[method-assign]
    try:
        # 1. Partial `book_result` loads full results only for target_block_ids (`b_ch2_1`), not Chapter 1
        partial = book_result(
            application.unit_of_work,
            scenario.book_id,
            include_quality=False,
            block_ids=(b_ch2_1,),
        )
        assert list_for_run_calls == 0
        assert get_many_calls == [(b_ch2_1,)]
        assert partial.prior_pronunciation_hints[("colonel", "uk", "ˈkɜː.nəl")] == "ph-ch1"

        # 2. Seed v2 correction policy + an `awaiting_review` candidate in Chapter 1 and `awaiting_human` in Chapter 2
        with application.unit_of_work.transaction() as tx:
            cond = tx.work.latest_conditions(scenario.book_id)
            run = tx.runs.active_for_book(scenario.book_id)
            doc = tx.documents.get(scenario.book_id)
            corr_policy = CorrectionSnapshotV2(
                model="fake",
                route={"provider": "fake", "base_url": "http://fake", "proxy": None, "route_id": "fake"},
                parameters={
                    "temperature": 0.0,
                    "max_output_tokens": 1000,
                    "timeout_seconds": 30,
                    "max_attempts": 1,
                    "max_unit_source_chars": 4000,
                    "input_price_per_mtok": 0.0,
                    "output_price_per_mtok": 0.0,
                },
                prompt="Correct chapter.",
                review_prompt="Review candidate.",
                versions=CORRECTION_VERSIONS_V2,
            ).to_payload()
            tx.work.bind(
                "run",
                run.id,
                book_id=scenario.book_id,
                conditions_id=cond.id,
                payload={"schema_version": 1, "correction": corr_policy},
            )
            for bid in (b_ch1_1, b_ch1_2, b_ch2_1):
                tx.work.bind(
                    "result",
                    f"{run.id}:{bid}",
                    book_id=scenario.book_id,
                    conditions_id=cond.id,
                    payload={"schema_version": 1},
                )
            bindings = tx.work.result_bindings(run.id)
            snapshot = parse_correction_snapshot(corr_policy)

            def _make_candidate(cur_tx, sec_idx: int, bid: str, *, human: bool) -> CorrectionCandidate:
                sec = scenario.section(sec_idx)
                analysis = cur_tx.analyses.get_for_section(scenario.book_id, sec.id)
                sec_res = {b.id: orig_get_many(cur_tx.results, run.id, [b.id])[b.id] for b in sec.ordered_blocks}
                inputs = CorrectionInputs(
                    doc,
                    sec,
                    analysis,
                    run,
                    cond,
                    snapshot,
                    tuple(sec.ordered_blocks),
                    sec_res,
                    {b.id: bindings.get(b.id) for b in sec.ordered_blocks},
                    {},
                )
                gen = cur_tx.work.observe_correction_input(
                    run.id, bid, correction_input_fingerprint(inputs, bid)
                )
                inputs = inputs.with_updated_state(input_generations={bid: gen})
                att_id = str(uuid4())
                cur_tx.work.add_attempt(
                    CallAttempt(
                        att_id,
                        scenario.book_id,
                        state="accepted",
                        conditions_id=cond.id,
                        run_id=run.id,
                        context={
                            "schema_version": 1,
                            "stage": "chapter_correction",
                            "policy_version": CORRECTION_POLICY_VERSION_V2,
                            "ordered_block_ids": [bid],
                        },
                    )
                )
                cur = sec_res[bid]
                prop = replace(cur, translated_text="Уточнённый перевод.")
                payload = {
                    "schema_version": 1,
                    "book_id": scenario.book_id,
                    "run_id": run.id,
                    "conditions_id": cond.id,
                    "section_id": sec.id,
                    "block_id": bid,
                    "input_sha256": inputs.checksums[bid],
                    "input_generation": gen,
                    "group_id": att_id,
                    "proposal_attempt_id": att_id,
                    "context_sha256": inputs.context_sha256,
                    "source_binding": thaw(bindings.get(bid)),
                    "original": {"text": sec.ordered_blocks[0].text},
                    "current_result": json.loads(json.dumps(asdict(cur))),
                    "result": json.loads(json.dumps(asdict(prop))),
                    "proposal": {
                        "issues": [
                            {
                                "source_quote": "colonel",
                                "current_evidence": {"kind": "quote", "quote": "полковник"},
                                "problem": "Смысл.",
                                "minimal_fix_reason": "Уточнение.",
                            }
                        ]
                    },
                    "evidence_valid": not human,
                }
                return CorrectionCandidate(
                    candidate_id=str(uuid4()),
                    run_id=run.id,
                    book_id=scenario.book_id,
                    section_id=sec.id,
                    block_id=bid,
                    conditions_id=cond.id,
                    group_id=att_id,
                    proposal_attempt_id=att_id,
                    input_generation=gen,
                    input_sha256=inputs.checksums[bid],
                    proposal_sha256=canonical_sha(payload),
                    state="awaiting_human" if human else "awaiting_review",
                    payload=payload,
                    review={"verdict": "uncertain", "reason": "Ручная проверка"} if human else None,
                )

            cand_review_only = _make_candidate(tx, 0, b_ch1_1, human=False)
            tx.work.add_correction_candidates((cand_review_only,))

        # Only `awaiting_review` exists: `book_overview` must not load any block results!
        get_many_calls.clear()
        list_for_run_calls = 0
        ov_no_human = book_overview(application.unit_of_work, scenario.book_id)
        assert ov_no_human.pending_corrections == 0
        assert list_for_run_calls == 0
        assert get_many_calls == []

        # Add `awaiting_human` in Chapter 2: `book_overview` loads results ONLY for Chapter 2 (`b_ch2_1`), never `list_for_run`!
        with application.unit_of_work.transaction() as tx2:
            cand_human = _make_candidate(tx2, 1, b_ch2_1, human=True)
            tx2.work.add_correction_candidates((cand_human,))

        get_many_calls.clear()
        list_for_run_calls = 0
        ov_with_human = book_overview(application.unit_of_work, scenario.book_id)
        assert ov_with_human.pending_corrections == 1
        assert list_for_run_calls == 0
        assert get_many_calls == [(b_ch2_1,)]
    finally:
        SqlAlchemyBlockResultRepository.get_many = orig_get_many  # type: ignore[method-assign]
        SqlAlchemyBlockResultRepository.list_for_run = orig_list_for_run  # type: ignore[method-assign]


