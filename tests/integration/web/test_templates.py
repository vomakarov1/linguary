"""HTML screens of server Web UI (P4.T4)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from web_scenario import (
    glossary_candidate,
    hinted_result,
    start_run,
    store_book,
    translated_result,
)

from linguary.adapters.config import AppSettings
from linguary.adapters.llm import FakeLLMClient
from linguary.adapters.validation import EpubCheckSettings, EpubCheckValidator
from linguary.application.ports.persistence import ExportValidationStatus
from linguary.application.use_cases.approve_glossary import (
    GlossaryDecision,
    GlossaryDecisionCommand,
)
from linguary.composition import Application, build_application
from linguary.domain import BlockTreatment, BookStatus, ConceptType
from linguary.web import create_app

HTML_HEADERS = {"accept": "text/html"}

#: Name guaranteed not to exist in `PATH`: simulates a machine without Java.
MISSING_JAVA = "linguary-java-that-does-not-exist"


@pytest.fixture
def client_without_epubcheck(
    settings: AppSettings, llm_client: FakeLLMClient
) -> Iterator[TestClient]:
    """Application client on a machine where external EPUB check is unavailable."""
    application = build_application(
        settings=settings,
        llm_client=llm_client,
        epubcheck=EpubCheckValidator(
            settings=EpubCheckSettings(java_executable=MISSING_JAVA)
        ),
    )
    try:
        with TestClient(
            create_app(application=application, run_background_worker=False),
            base_url=application.settings.server.base_url,
        ) as test_client:
            yield test_client
    finally:
        application.dispose()


def test_root_redirects_to_books_screen(client: TestClient) -> None:
    """Root of local server opens the first UI screen."""
    redirect = client.get("/", headers=HTML_HEADERS, follow_redirects=False)
    response = client.get("/", headers=HTML_HEADERS)

    assert redirect.status_code == 303
    assert redirect.headers["location"] == "/books"
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Загрузить" in response.text


def test_books_screen_renders_upload_and_existing_books(
    client: TestClient, application: Application
) -> None:
    """Book list returns HTML with upload form and book row."""
    store_book(application, status=BookStatus.GLOSSARY_REVIEW)

    response = client.get("/books", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Pride and Prejudice" in response.text
    assert "Загрузить" in response.text


def test_books_screen_warns_about_sending_the_book_to_an_external_provider(
    client: TestClient,
) -> None:
    """Upload form warns that book text is sent to an external service (PRD § 4.1, § 21.2)."""
    response = client.get("/books", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "Текст книги целиком отправляется внешнему провайдеру LLM" in response.text


def test_book_detail_screen_renders_progress_and_failed_actions(
    client: TestClient, application: Application
) -> None:
    """Card displays progress, navigation, and book actions."""
    scenario = store_book(
        application,
        status=BookStatus.PROCESSING,
        treatments={0: (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY)},
    )
    start_run(
        application,
        scenario,
        results=(translated_result(scenario.block_ids()[0]),),
    )

    response = client.get(f"/books/{scenario.book_id}", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "50%" in response.text
    assert "Глоссарий" in response.text
    assert "Результат" in response.text


def test_imported_book_screen_explains_missing_llm_configuration(
    client: TestClient, application: Application
) -> None:
    """After import without LLM, screen explicitly explains why analysis cannot proceed."""
    scenario = store_book(
        application,
        status=BookStatus.IMPORTED,
        analysed_chapters=(),
    )

    response = client.get(f"/books/{scenario.book_id}", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "Импорт завершён" in response.text
    assert "LLM не настроена" in response.text
    assert "Начать анализ" not in response.text


def test_start_analysis_form_redirects_back_to_the_book(
    configured_client: TestClient, configured_application: Application
) -> None:
    """HTML analysis command schedules task and redirects browser to book card."""
    scenario = store_book(
        configured_application,
        status=BookStatus.IMPORTED,
        analysed_chapters=(),
    )

    response = configured_client.post(
        f"/books/{scenario.book_id}/analyze",
        headers=HTML_HEADERS,
        follow_redirects=False,
    )

    with configured_application.unit_of_work.transaction() as uow:
        tasks = uow.tasks.list_for_book(scenario.book_id)
    assert response.status_code == 303
    assert response.headers["location"] == f"/books/{scenario.book_id}"
    assert len(tasks) == 1


def test_glossary_screen_renders_read_only_state_after_processing_started(
    client: TestClient, application: Application
) -> None:
    """After processing starts, HTML glossary displays read-only mode."""
    scenario = store_book(
        application,
        status=BookStatus.PROCESSING,
        glossary_candidates={0: (glossary_candidate(),)},
    )

    response = client.get(f"/books/{scenario.book_id}/glossary", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "Только чтение" in response.text
    assert "экране результата" in response.text
    assert "Принять" not in response.text


def test_glossary_screen_explains_analysis_read_only_state(
    client: TestClient, application: Application
) -> None:
    """During analysis, HTML glossary explains temporary read-only state."""
    scenario = store_book(
        application,
        status=BookStatus.ANALYZING,
        glossary_candidates={0: (glossary_candidate(),)},
    )

    response = client.get(f"/books/{scenario.book_id}/glossary", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "Анализ ещё идёт" in response.text
    assert "редактирование откроется после анализа" in response.text
    assert "Принять" not in response.text


@pytest.mark.parametrize(
    "status",
    [
        pytest.param(BookStatus.GLOSSARY_REVIEW, id="after-reset-to-review"),
        pytest.param(BookStatus.READY_TO_PROCESS, id="before-start"),
    ],
)
def test_glossary_screen_folds_decision_actions_of_decided_items(
    client: TestClient, application: Application, status: BookStatus
) -> None:
    """Decision on an item can be modified while glossary is editable (QA-07).

    Decision actions for a decided item are collapsed under "Change decision" to prevent
    screen clutter, while note editing remains accessible separately: character gender
    and other book-level notes are visible after name translation is accepted (PRD § 11, § 21.3).
    """
    candidate = glossary_candidate()
    scenario = store_book(
        application, status=status, glossary_candidates={0: (candidate,)}
    )
    application.approve_glossary(
        GlossaryDecisionCommand(
            book_id=scenario.book_id,
            source_text=candidate.source_text,
            concept_id=candidate.id,
            decision=GlossaryDecision.ACCEPT,
        )
    )

    response = client.get(f"/books/{scenario.book_id}/glossary", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "Требуют решения: 0" in response.text
    assert "Решение: принято." in response.text
    assert "Принять все требующие решения" not in response.text
    folded = response.text[response.text.index("<details"):response.text.index("</details>")]
    assert "<summary>Изменить решение</summary>" in folded
    for action in ("accept", "correct", "lock", "skip"):
        assert f'name="action" value="{action}"' in folded
    assert 'name="text"' in folded
    assert response.text.count('name="action"') == 4
    assert "Сохранить указание" in response.text


def test_glossary_screen_offers_to_change_an_optional_hint(
    client: TestClient, application: Application
) -> None:
    """An optional hint that does not require review can also be corrected (QA-07)."""
    candidate = glossary_candidate(
        source_text="entailed estate",
        suggested_translation="заповедное имение",
        concept_type=ConceptType.TERM,
    )
    scenario = store_book(application, glossary_candidates={0: (candidate,)})

    response = client.get(f"/books/{scenario.book_id}/glossary", headers=HTML_HEADERS)

    assert "Не требует решения." in response.text
    assert "<summary>Изменить решение</summary>" in response.text
    assert 'name="action" value="correct"' in response.text


def test_glossary_screen_names_return_to_the_model_variant_for_corrected_item(
    client: TestClient, application: Application
) -> None:
    """For a corrected item, "Accept" is labeled as returning to the model variant."""
    candidate = glossary_candidate()
    scenario = store_book(application, glossary_candidates={0: (candidate,)})
    application.approve_glossary(
        GlossaryDecisionCommand(
            book_id=scenario.book_id,
            source_text=candidate.source_text,
            concept_id=candidate.id,
            decision=GlossaryDecision.CORRECT,
            translation="господин Беннет",
        )
    )

    response = client.get(f"/books/{scenario.book_id}/glossary", headers=HTML_HEADERS)

    assert "Вернуть вариант модели: мистер Беннет" in response.text
    assert 'value="господин Беннет"' in response.text


def test_glossary_screen_offers_to_change_the_concept_type(
    client: TestClient, application: Application
) -> None:
    """Concept type is visible in Russian and editable via correction form (QA-11, T2)."""
    candidate = glossary_candidate()
    scenario = store_book(application, glossary_candidates={0: (candidate,)})

    response = client.get(f"/books/{scenario.book_id}/glossary", headers=HTML_HEADERS)

    assert "Тип: действующее лицо." in response.text
    assert '<select name="type"' in response.text
    assert '<option value="person" selected>действующее лицо</option>' in response.text
    assert '<option value="term">термин</option>' in response.text


def test_glossary_screen_shows_other_spellings_of_the_item(
    client: TestClient, application: Application
) -> None:
    """Merged spellings of the same name are visible in its item (QA-11)."""
    first = glossary_candidate(candidate_id="candidate-first", source_text="Mr. Bennet")
    second = glossary_candidate(
        candidate_id="candidate-second", source_text="MR BENNET", confidence=0.9
    )
    scenario = store_book(
        application, glossary_candidates={0: (first, second)},
        chapters=(("Mr. Bennet " * 5,),),
    )

    response = client.get(f"/books/{scenario.book_id}/glossary", headers=HTML_HEADERS)

    assert "<h2>Mr. Bennet</h2>" in response.text
    assert "Другие написания: MR BENNET." in response.text
    assert "Требуют решения: 1" in response.text


def test_glossary_page_marks_model_doubts(
    client: TestClient, application: Application
) -> None:
    candidate = glossary_candidate(model_flagged_review=True)
    scenario = store_book(application, glossary_candidates={0: (candidate,)})

    response = client.get(f"/books/{scenario.book_id}/glossary", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "Модель сомневается." in response.text
    assert "Требуют решения: 0" in response.text


def test_book_detail_screen_starts_processing_after_glossary_is_complete(
    client: TestClient, application: Application
) -> None:
    """After closing glossary, book card offers initiating processing."""
    candidate = glossary_candidate()
    scenario = store_book(application, glossary_candidates={0: (candidate,)})
    application.approve_glossary(
        GlossaryDecisionCommand(
            book_id=scenario.book_id,
            source_text=candidate.source_text,
            concept_id=candidate.id,
            decision=GlossaryDecision.ACCEPT,
        )
    )

    response = client.get(f"/books/{scenario.book_id}", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "Глоссарий готов" in response.text
    assert "Начать обработку" in response.text
    assert "Открыть глоссарий" not in response.text


def test_plan_and_warnings_screens_render_book_diagnostics(
    client: TestClient, application: Application
) -> None:
    """HTML plan and warning screens use the same data as JSON API."""
    scenario = store_book(
        application,
        chapters=(("Первый абзац.", "Второй абзац."),),
        treatments={0: (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY)},
    )

    plan = client.get(f"/books/{scenario.book_id}/plan", headers=HTML_HEADERS)
    warnings = client.get(f"/books/{scenario.book_id}/warnings", headers=HTML_HEADERS)

    assert plan.status_code == 200
    assert "План обработки" in plan.text
    assert "Chapter 1" in plan.text
    assert "Первый абзац." in plan.text
    assert "Словарные подсказки" in plan.text
    assert warnings.status_code == 200
    assert "Предупреждений нет" in warnings.text


def test_plan_screen_renders_empty_state_before_analysis(
    client: TestClient, application: Application
) -> None:
    """Prior to analysis, HTML plan tab explains why data is not yet available."""
    scenario = store_book(
        application,
        status=BookStatus.IMPORTED,
        analysed_chapters=(),
    )

    response = client.get(f"/books/{scenario.book_id}/plan", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "План появится после анализа книги" in response.text
    assert "Пока нет данных анализа" in response.text


def test_result_screen_renders_blocks_and_export_status(
    client: TestClient, application: Application
) -> None:
    """Result displays original, translation, hints, and EPUBCheck status."""
    scenario = store_book(
        application,
        status=BookStatus.COMPLETED,
        treatments={0: (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY)},
    )
    block_ids = scenario.block_ids()
    start_run(
        application,
        scenario,
        results=(translated_result(block_ids[0]), hinted_result(block_ids[1])),
    )
    with application.unit_of_work.transaction() as uow:
        book = uow.books.get(scenario.book_id)
        assert book is not None
        uow.books.save(
            replace(book, export_validation_status=ExportValidationStatus.NOT_VALIDATED)
        )

    response = client.get(f"/books/{scenario.book_id}/result", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "Общепризнанная истина" in response.text
    assert "truth" in response.text
    assert "EPUBCheck не выполнялся" in response.text
    assert "Скачать EPUB" in response.text
    assert f"/books/{scenario.book_id}/download" in response.text
    assert "/static/linguary.js" in response.text
    assert "data-regenerate-form" in response.text


def test_result_screen_hides_export_action_before_processing_finishes(
    client: TestClient, application: Application
) -> None:
    """Before processing completes, HTML result does not offer building EPUB."""
    scenario = store_book(
        application,
        status=BookStatus.READY_TO_PROCESS,
        treatments={0: (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY)},
    )

    response = client.get(f"/books/{scenario.book_id}/result", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "EPUB можно собрать после завершения обработки книги" in response.text
    assert "Собрать EPUB" not in response.text


def test_result_screen_explains_that_epub_is_stale_after_changes(
    client: TestClient, application: Application
) -> None:
    """Completed result without up-to-date EPUB explicitly prompts rebuilding the book."""
    scenario = store_book(
        application,
        status=BookStatus.READY_TO_EXPORT,
        treatments={0: (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY)},
    )
    block_ids = scenario.block_ids()
    start_run(
        application,
        scenario,
        results=(translated_result(block_ids[0]), hinted_result(block_ids[1])),
    )

    response = client.get(f"/books/{scenario.book_id}/result", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "EPUB нужно собрать после последних изменений" in response.text
    assert "Собрать EPUB" in response.text
    assert "Скачать EPUB" not in response.text


def test_result_screen_offers_return_to_glossary_review(
    client: TestClient, application: Application
) -> None:
    """Result screen provides confirmed return to glossary review (PRD § 21.5)."""
    scenario = store_book(
        application,
        status=BookStatus.PROCESSING,
        treatments={0: (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY)},
    )
    start_run(
        application,
        scenario,
        results=(translated_result(scenario.block_ids()[0]),),
    )

    response = client.get(f"/books/{scenario.book_id}/result", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert (
        "Возврат к проверке глоссария удалит текущие результаты обработки"
        in response.text
    )
    assert f'action="/books/{scenario.book_id}/reset"' in response.text
    assert 'name="confirmed"' in response.text


def test_result_screen_hides_return_to_glossary_before_processing_starts(
    client: TestClient, application: Application
) -> None:
    """Prior to processing start, there is nothing to reset, so action is hidden."""
    scenario = store_book(application, status=BookStatus.READY_TO_PROCESS)

    response = client.get(f"/books/{scenario.book_id}/result", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "Вернуться к проверке глоссария" not in response.text


def test_reset_form_returns_the_browser_to_glossary_review(
    client: TestClient, application: Application
) -> None:
    """Confirmed return from browser cancels run and opens glossary."""
    scenario = store_book(application, status=BookStatus.PROCESSING)
    start_run(
        application,
        scenario,
        results=(translated_result(scenario.block_ids()[0]),),
    )

    response = client.post(
        f"/books/{scenario.book_id}/reset",
        data={"confirmed": "true"},
        headers=HTML_HEADERS,
        follow_redirects=False,
    )

    with application.unit_of_work.transaction() as uow:
        assert uow.runs.active_for_book(scenario.book_id) is None
        assert uow.books.get(scenario.book_id).status is BookStatus.GLOSSARY_REVIEW
    assert response.status_code == 303
    assert response.headers["location"] == f"/books/{scenario.book_id}/glossary"


def test_system_config_screen_renders_first_run_state(client: TestClient) -> None:
    """Configuration screen displays unconfigured LLM and does not expose key."""
    response = client.get("/system/config", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "LLM не настроена" in response.text
    assert "test-key" not in response.text


def test_system_config_screen_explains_how_to_configure_the_llm(
    client: TestClient,
) -> None:
    """Alongside "LLM not configured" status, screen explains what to configure (PRD § 23)."""
    response = client.get("/system/config", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "config.toml" in response.text
    assert "LINGUARY_LLM_MODEL" in response.text


def test_system_config_screen_warns_about_the_external_provider(
    client: TestClient,
) -> None:
    """Configuration screen warns about sending book text to external party (PRD § 21.6)."""
    response = client.get("/system/config", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "Текст книги целиком отправляется внешнему провайдеру LLM" in response.text


def test_system_config_screen_explains_how_to_enable_epubcheck(
    client_without_epubcheck: TestClient,
) -> None:
    """Screen explains how to enable unavailable EPUB check (PRD § 21.6)."""
    response = client_without_epubcheck.get("/system/config", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "Внешняя проверка EPUBCheck необязательна" in response.text
    assert "~/.linguary/epubcheck" in response.text


def test_system_config_screen_renders_reading_settings(client: TestClient) -> None:
    """Configuration screen displays reader level settings block (ReadingSettings)."""
    response = client.get("/system/config", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "Уровень читателя (ReadingSettings)" in response.text
    assert "Поддерживаемые уровни владения языком (CEFR)" in response.text
    assert "[reading]" in response.text
    assert 'reader_level = "b2"' in response.text
    assert "Уровень читателя" in response.text
    assert 'name="reader_level"' in response.text
    assert 'id="settings-reader_level"' in response.text



@pytest.mark.parametrize(
    ("decision", "translation", "label"),
    [
        pytest.param(GlossaryDecision.ACCEPT, None, "принято", id="accept"),
        pytest.param(
            GlossaryDecision.CORRECT, "господин Беннет", "исправлено", id="correct"
        ),
        pytest.param(GlossaryDecision.LOCK, None, "зафиксировано", id="lock"),
        pytest.param(GlossaryDecision.SKIP, None, "пропущено", id="skip"),
    ],
)
def test_glossary_screen_names_the_decision_in_russian(
    client: TestClient,
    application: Application,
    decision: GlossaryDecision,
    translation: str | None,
    label: str,
) -> None:
    """Item decision is labeled in Russian rather than `approved` status (QA-06)."""
    candidate = glossary_candidate()
    scenario = store_book(application, glossary_candidates={0: (candidate,)})
    application.approve_glossary(
        GlossaryDecisionCommand(
            book_id=scenario.book_id,
            source_text=candidate.source_text,
            concept_id=candidate.id,
            decision=decision,
            translation=translation,
        )
    )

    response = client.get(f"/books/{scenario.book_id}/glossary", headers=HTML_HEADERS)

    assert f"Решение: {label}." in response.text
    for status in ("approved", "locked", "suggested"):
        assert status not in response.text


def test_glossary_screen_shows_model_confidence(
    client: TestClient, application: Application
) -> None:
    """Model confidence is displayed for each glossary item (QA-14)."""
    candidate = glossary_candidate(confidence=0.62, model_flagged_review=True)
    scenario = store_book(application, glossary_candidates={0: (candidate,)})

    response = client.get(f"/books/{scenario.book_id}/glossary", headers=HTML_HEADERS)

    assert "Уверенность модели: 62%" in response.text
    assert "модель просит проверить вариант" in response.text


def test_plan_screen_explains_what_a_complete_plan_means(
    client: TestClient, application: Application
) -> None:
    """Plan completeness is labeled clearly with explanation rather than a single word (QA-14)."""
    scenario = store_book(application)

    response = client.get(f"/books/{scenario.book_id}/plan", headers=HTML_HEADERS)

    assert "План полный" in response.text
    assert "Все главы проанализированы" in response.text
    assert ">Полный<" not in response.text


def test_plan_screen_explains_what_an_incomplete_plan_means(
    client: TestClient, application: Application
) -> None:
    """Incomplete plan explains what is still missing (QA-14)."""
    scenario = store_book(
        application, chapters=(("First.",), ("Second.",)), analysed_chapters=(0,)
    )

    response = client.get(f"/books/{scenario.book_id}/plan", headers=HTML_HEADERS)

    assert "План неполный" in response.text
    assert "ещё не проанализирована" in response.text


def test_notes_page_is_not_shown_as_an_ordinary_chapter(
    client: TestClient, application: Application
) -> None:
    """A notes-only page is not labeled as a regular chapter on plan and result screens (QA-04)."""
    scenario = store_book(
        application,
        status=BookStatus.COMPLETED,
        chapters=(("Первый абзац.",), ("Текст сноски.",)),
        notes_chapters=(1,),
    )
    start_run(application, scenario)

    plan = client.get(f"/books/{scenario.book_id}/plan", headers=HTML_HEADERS)
    result = client.get(f"/books/{scenario.book_id}/result", headers=HTML_HEADERS)

    for page in (plan, result):
        assert page.status_code == 200
        assert "Глава 1 - Chapter 1" in page.text
        assert "Сноски - Chapter 2" in page.text
        assert "Глава 2" not in page.text
