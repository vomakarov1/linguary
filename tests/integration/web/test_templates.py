"""HTML screens of server Web UI (P4.T4)."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import pytest
from fastapi.testclient import TestClient
from lxml import html
from web_scenario import (
    WebScenario,
    glossary_candidate,
    hinted_result,
    start_run,
    store_book,
    store_task,
    translated_result,
)

from linguary.application.ports.persistence import ExportValidationStatus
from linguary.application.use_cases.approve_glossary import (
    GlossaryDecision,
    GlossaryDecisionCommand,
)
from linguary.composition import Application
from linguary.domain import (
    BlockProcessingResult,
    BlockSegment,
    BlockTreatment,
    BlockType,
    BookStatus,
    ConceptType,
    ErrorType,
    GlossaryHint,
    LanguageNote,
    ProcessingTask,
    TaskError,
    TaskStatus,
    TaskType,
)
from linguary.domain.content_plan import (
    ContentEvidence,
    ContentPlan,
    ContextFact,
    Difficulty,
    HelpEvidence,
    RequiredPoint,
    SourceAnchor,
    artifact_key,
)
from linguary.domain.pronunciation import PronunciationAssistance, PronunciationEvidence

HTML_HEADERS = {"accept": "text/html"}
RESULT_FRAGMENT_HEADERS = {**HTML_HEADERS, "x-linguary-fragment": "result-block"}


def _plain_source(node) -> str:
    """Reading-unit source with marker links removed and line breaks kept."""
    parts: list[str] = []
    if node.text:
        parts.append(node.text)
    for child in node:
        if child.tag == "br":
            parts.append("\n")
        elif child.tag != "a" or "lg-ref" not in (child.get("class") or "").split():
            parts.append(_plain_source(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _pronunciation_book(
    application: Application,
    *,
    texts: tuple[str, ...] = ("zorp", "zorp"),
    variant: Literal["uk", "us"] = "uk",
    meaning: bool = False,
    source_label: str = "Invented test data",
) -> WebScenario:
    """Persist invented pronunciations without contacting a dictionary."""
    treatment = BlockTreatment.GLOSSARY_ONLY if meaning else BlockTreatment.READ_AS_IS
    scenario = store_book(
        application,
        status=BookStatus.COMPLETED,
        chapters=tuple((text,) for text in texts),
        treatments={index: (treatment,) for index in range(len(texts))},
    )
    results = []
    for section in scenario.document.ordered_sections:
        block, = section.ordered_blocks
        positions = [
            index for index in range(len(block.text))
            if block.text.startswith("zorp", index)
        ]
        pronunciations = tuple(
            PronunciationAssistance(
                anchor=SourceAnchor(block.id, index, index + 4, "zorp"),
                form="zorp", ipa="ˈzɔːp", variant=variant, hint_id="ph-invented",
                evidence=PronunciationEvidence(
                    "private-source", "private-version", "entry-invented", "private-sense",
                    "noun", "zorp", "ˈzɔːp", source_label,
                ),
            )
            for index in positions
        )
        results.append(BlockProcessingResult(
            block_id=block.id, treatment=treatment,
            glossary_hints=(GlossaryHint("zorp", "выдуманное значение", "тестовая помета"),) if meaning else (),
            pronunciation_assistances=pronunciations,
        ))
    start_run(application, scenario, results=tuple(reversed(results)))
    return scenario


@pytest.mark.parametrize("variant", ["uk", "us"])
def test_pronunciation_template_primary_and_repeat(
    client: TestClient, application: Application, variant: Literal["uk", "us"]
) -> None:
    scenario = _pronunciation_book(application, variant=variant)
    response = client.get(f"/books/{scenario.book_id}/result", headers=HTML_HEADERS)
    tree = html.fromstring(response.text)
    notes = tree.xpath('//ol[@class="lg-notes"]/li')

    assert len(notes) == 2
    assert not tree.xpath('//a[@class="pronunciation-repeat"]')
    for note in notes:
        term = note.xpath('./span[@class="lg-term"]')[0]
        ipa = note.xpath('./span[@class="lg-ipa"]')[0]
        assert term.text == "zorp"
        assert ipa.text == "/ˈzɔːp/"
        assert (ipa.tail or "").strip() == f"({variant.upper()})"
    assert response.text.count("/ˈzɔːp/") == 2
    assert "Invented test data" not in response.text
    assert "entry-invented" not in response.text
    assert "private-" not in response.text


def test_pronunciation_template_without_translation(
    client: TestClient, application: Application
) -> None:
    scenario = _pronunciation_book(application, texts=("zorp",))
    tree = html.fromstring(client.get(f"/books/{scenario.book_id}/result", headers=HTML_HEADERS).text)

    assert tree.xpath('//span[@class="lg-ipa"]')
    assert not tree.xpath('//p[@class="lg-tr"] | //*[@class="translation"]')
    assert "В оригинале" in tree.text_content()


def test_pronunciation_template_combines_dictionary_meaning(
    client: TestClient, application: Application
) -> None:
    scenario = _pronunciation_book(application, texts=("A zorp.",), meaning=True)
    tree = html.fromstring(client.get(f"/books/{scenario.book_id}/result", headers=HTML_HEADERS).text)
    ipa, = tree.xpath('//span[@class="lg-ipa"]')
    item = ipa.getparent()

    assert item.tag == "li"
    assert "выдуманное значение" in item.text_content()
    assert "тестовая помета" in item.text_content()
    assert len(tree.xpath('//article//li')) == 1
    assert tree.text_content().count("выдуманное значение") == 1


def test_pronunciation_template_ambiguous_dictionary_occurrences_stay_separate(
    client: TestClient, application: Application
) -> None:
    scenario = _pronunciation_book(application, texts=("zorp and zorp",), meaning=True)
    tree = html.fromstring(client.get(f"/books/{scenario.book_id}/result", headers=HTML_HEADERS).text)
    note, = tree.xpath('//ol[@class="lg-notes"]/li')

    assert "выдуманное значение" in note.text_content()
    assert note.xpath('./span[@class="lg-ipa"]')
    assert not tree.xpath('//a[@class="pronunciation-repeat"]')
    assert len(tree.xpath('//span[@class="lg-ipa"]')) == 1
    assert tree.text_content().count("выдуманное значение") == 1


def test_pronunciation_template_original_unchanged(
    client: TestClient, application: Application
) -> None:
    original = '“zorp” <em> & zorp…'
    scenario = _pronunciation_book(application, texts=(original,))
    tree = html.fromstring(client.get(f"/books/{scenario.book_id}/result", headers=HTML_HEADERS).text)
    source, = tree.xpath('//p[@class="lg-src"]')

    assert _plain_source(source) == original
    assert not source.xpath('.//*[not(self::a[@class="lg-ref"] or self::sup)]')


def test_pronunciation_template_fragment_refresh(
    client: TestClient, application: Application
) -> None:
    scenario = _pronunciation_book(application)
    primary_response = client.get(
        f"/books/{scenario.book_id}/blocks/{scenario.block_ids()[0]}", headers=RESULT_FRAGMENT_HEADERS,
    )
    primary = html.fromstring(primary_response.text)
    assert primary.xpath('//span[@class="lg-ipa"]')[0].text == "/ˈzɔːp/"
    assert 'class="lg-key"' not in primary_response.text
    for _ in range(2):
        response = client.get(
            f"/books/{scenario.book_id}/blocks/{scenario.block_ids(1)[0]}", headers=RESULT_FRAGMENT_HEADERS,
        )
        tree = html.fromstring(response.text)
        assert tree.xpath('//span[@class="lg-ipa"]')[0].text == "/ˈzɔːp/"
        assert not tree.xpath('//a[@class="pronunciation-repeat"] | //details[@class="ipa-guide"] | //table[@class="lg-key"]')
        assert _plain_source(tree.xpath('//p[@class="lg-src"]')[0]) == "zorp"

    identifier = scenario.block_ids(1)[0]
    with application.unit_of_work.transaction() as uow:
        result = next(item for item in uow.results.list_for_run(scenario.run_id) if item.block_id == identifier)
        previous, = result.pronunciation_assistances
        updated = replace(
            previous, ipa="ˈzɛp", hint_id="ph-refreshed",
            evidence=replace(previous.evidence, dictionary_ipa="ˈzɛp"),
        )
        uow.results.replace_results(scenario.run_id, [replace(result, pronunciation_assistances=(updated,))])
    response = client.get(
        f"/books/{scenario.book_id}/blocks/{identifier}", headers=RESULT_FRAGMENT_HEADERS,
    )
    tree = html.fromstring(response.text)
    assert tree.xpath('//span[@class="lg-ipa"]')[0].text == "/ˈzɛp/"
    assert not tree.xpath('//a[@class="pronunciation-repeat"] | //details[@class="ipa-guide"] | //table[@class="lg-key"]')
    assert _plain_source(tree.xpath('//p[@class="lg-src"]')[0]) == "zorp"


def test_pronunciation_ipa_guide_visible_and_keyboard_accessible(
    client: TestClient, application: Application
) -> None:
    scenario = _pronunciation_book(application, texts=("zorp",))
    response = client.get(f"/books/{scenario.book_id}/result", headers=HTML_HEADERS)
    fragment = client.get(
        f"/books/{scenario.book_id}/blocks/{scenario.block_ids()[0]}", headers=RESULT_FRAGMENT_HEADERS,
    )
    tree = html.fromstring(response.text)
    table, = tree.xpath('//table[@class="lg-key"]')

    assert not tree.xpath('//details[@class="ipa-guide"]')
    assert 'class="lg-key"' not in fragment.text
    assert table.get("hidden") is None
    assert not table.xpath('.//script | .//iframe | .//*[@hidden] | .//*[@aria-hidden="true"]')
    assert "ˈ" in table.text_content() and "ː" in table.text_content()
    assert "ударение на следующем слоге" in table.text_content()
    assert "звук долгий" in table.text_content()
    assert "Международный фонетический алфавит" not in response.text
    assert "вторичное" not in response.text


def test_pronunciation_template_escapes_attribution(
    client: TestClient, application: Application
) -> None:
    label = '<script>alert("source")</script>'
    scenario = _pronunciation_book(application, texts=("zorp",), source_label=label)
    response = client.get(f"/books/{scenario.book_id}/result", headers=HTML_HEADERS)
    tree = html.fromstring(response.text)

    assert tree.xpath('//span[@class="lg-ipa"]')
    assert not tree.xpath('//script[not(@src)]')
    assert "alert(" not in response.text
    assert label not in response.text


def test_pronunciation_template_without_confirmed_assistance(
    client: TestClient, application: Application
) -> None:
    scenario = store_book(application, status=BookStatus.COMPLETED)
    start_run(application, scenario)
    tree = html.fromstring(client.get(f"/books/{scenario.book_id}/result", headers=HTML_HEADERS).text)

    assert not tree.xpath('//*[@class="pronunciation"] | //a[@class="pronunciation-repeat"] | //details[@class="ipa-guide"]')


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


def test_system_config_screen_exposes_editable_llm_settings(
    client: TestClient,
) -> None:
    """The configuration screen provides editable connection and model fields."""
    response = client.get("/system/config", headers=HTML_HEADERS)
    tree = html.fromstring(response.text)

    assert response.status_code == 200
    assert tree.xpath('//input[@name="base_url"]')
    assert tree.xpath('//input[@name="api_key" and @type="password"]')
    assert tree.xpath('//input[@name="model"]')


def test_system_config_screen_warns_about_the_external_provider(
    client: TestClient,
) -> None:
    """Configuration screen warns about sending book text to external party (PRD § 21.6)."""
    response = client.get("/system/config", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "Текст книги целиком отправляется внешнему провайдеру LLM" in response.text


def test_result_screen_explains_unavailable_epubcheck(
    client: TestClient, application: Application,
) -> None:
    """The result page explains how to enable an unavailable EPUB check."""
    scenario = store_book(application, status=BookStatus.COMPLETED)
    start_run(application, scenario)
    with application.unit_of_work.transaction() as uow:
        book = uow.books.get(scenario.book_id)
        assert book is not None
        uow.books.save(
            replace(book, export_validation_status=ExportValidationStatus.NOT_VALIDATED)
        )
    response = client.get(f"/books/{scenario.book_id}/result", headers=HTML_HEADERS)

    assert response.status_code == 200
    assert "Внешняя проверка EPUBCheck необязательна" in response.text
    assert "~/.linguary/epubcheck" in response.text


def test_system_config_screen_renders_reading_level_control(client: TestClient) -> None:
    """Configuration screen exposes the supported reader levels as a form control."""
    response = client.get("/system/config", headers=HTML_HEADERS)
    tree = html.fromstring(response.text)

    assert response.status_code == 200
    select, = tree.xpath('//select[@name="reader_level"]')
    assert select.get("id") == "settings-reader_level"
    assert [option.get("value") for option in select.xpath("./option")] == [
        "a2", "b1", "b2", "c1"
    ]
    assert select.xpath('./option[@value="b2"][@selected]')



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


def test_result_groups_wrapper_and_refresh_url(client, application):
    scenario = _pronunciation_book(application)
    endpoint = f'/books/{scenario.book_id}/result'
    for headers in (HTML_HEADERS, {**HTML_HEADERS, 'x-linguary-fragment': 'result-groups'}):
        tree = html.fromstring(client.get(endpoint, headers=headers).text)
        root, = tree.xpath('//*[@data-result-page]')
        assert root.get('data-result-url') == endpoint
        assert len(root.xpath('.//*[@data-result-block]')) == 2
        assert not root.xpath('.//*[@data-result-header] | .//form[contains(@action, "/reset")]')


def test_result_groups_template_keeps_pronunciation_guide_and_original(client, application):
    original = '  A zorp.\n'
    scenario = _pronunciation_book(application, texts=(original, 'zorp'))
    tree = html.fromstring(client.get(f'/books/{scenario.book_id}/result', headers={
        **HTML_HEADERS, 'x-linguary-fragment': 'result-groups'}).text)
    root, = tree.xpath('//*[@data-result-page]')
    assert [_plain_source(source) for source in root.xpath('.//p[@class="lg-src"]')] == [original, 'zorp']
    assert len(root.xpath('.//table[@class="lg-key"]')) == 1
    assert not root.xpath('.//details[@class="ipa-guide"] | .//a[@class="pronunciation-repeat"]')
    assert len(root.xpath('.//span[@class="lg-ipa"]')) == 2


def test_result_page_versions_updated_script(client, application):
    scenario = _pronunciation_book(application)
    tree = html.fromstring(client.get(f'/books/{scenario.book_id}/result', headers=HTML_HEADERS).text)
    assert tree.xpath('//script/@src')[0].endswith('/static/linguary.js?v=4')
    assert tree.xpath('//link[@rel="stylesheet"]/@href')[0].endswith('/static/linguary.css?v=4')


def _units_book(application: Application) -> WebScenario:
    """Two-part paragraph with a marked note and a speaker, plus a poem note without a marker."""
    paragraph = "Alice heard a tale. Then she left."
    poem = "How doth\nImprove,\n\nAnd pour"
    scenario = store_book(
        application,
        book_id="book-layout-units",
        status=BookStatus.COMPLETED,
        chapters=((paragraph, poem),),
    )
    paragraph_id, poem_id = scenario.block_ids()
    cut = paragraph.index(" Then")
    tale_end = paragraph.index("tale") + len("tale")
    then_at = paragraph.index("Then")
    tale = LanguageNote("note-tale", paragraph_id, "tale", "Сказка здесь — история.")
    tale_key = artifact_key(
        "language_note.note",
        paragraph_id,
        {
            "original_excerpt": tale.original_excerpt,
            "note": tale.note,
            "translated_excerpt": tale.translated_excerpt,
        },
    )
    tale_anchor = SourceAnchor(paragraph_id, tale_end - len("tale"), tale_end, "tale")
    point = RequiredPoint("form", "Explain the form")
    difficulties = (
        Difficulty("grammar", (tale_anchor,), paragraph_id, "grammar", "useful", "explanation", (point,)),
        Difficulty(
            "idiom",
            (tale_anchor,),
            paragraph_id,
            "idiom",
            "useful",
            "explanation",
            (RequiredPoint("sense", "Explain the sense"),),
        ),
    )
    evidence = (
        HelpEvidence("grammar", "form", "language_note.note", paragraph_id, tale_key, tale.note),
        HelpEvidence("idiom", "sense", "language_note.note", paragraph_id, tale_key, tale.note),
    )
    speaker = SourceAnchor(paragraph_id, then_at, then_at + len("Then"), "Then")
    plan = ContentPlan(
        3,
        "content-plan-v3",
        difficulties,
        evidence,
        context_facts=(
            ContextFact("speaker-alice", "speaker", (speaker,), speaker, "Alice", "explicit"),
        ),
    )
    with application.unit_of_work.transaction() as tx:
        document = tx.documents.get(scenario.book_id)
        analysis = tx.analyses.get_for_section(scenario.book_id, scenario.section().id)
        section = document.ordered_sections[0]
        first, second = section.ordered_blocks
        blocks = (
            replace(
                first,
                segments=(BlockSegment(0, cut, 0), BlockSegment(cut, len(paragraph), 1)),
            ),
            replace(second, type=BlockType.POEM),
        )
        tx.documents.save(replace(document, sections=(replace(section, blocks=blocks),)))
        tx.analyses.save(scenario.book_id, replace(analysis, content_plan=plan))
    poem_note = LanguageNote("note-poem", poem_id, "absent-excerpt", "Без маркера.")
    start_run(
        application,
        scenario,
        results=(
            BlockProcessingResult(
                paragraph_id,
                BlockTreatment.TRANSLATE,
                "Потом она ушла.",
                language_notes=(tale,),
                content_evidence=ContentEvidence(plan.sha256, evidence, version=3),
                segment_treatments=(BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE),
                segment_translations=(None, "Потом она ушла."),
            ),
            BlockProcessingResult(
                poem_id,
                BlockTreatment.TRANSLATE,
                "Как крокодил\nСвой хвост,\n\nИ льёт",
                language_notes=(poem_note,),
            ),
        ),
    )
    return scenario


def _two_part_page(application: Application, book_id: str) -> WebScenario:
    """One paragraph split into a source part and a translated part."""
    text = "Plain words. Clear words."
    cut = text.index(" Clear")
    scenario = store_book(
        application,
        book_id=book_id,
        status=BookStatus.COMPLETED,
        chapters=((text,),),
    )
    block_id = scenario.block_ids()[0]
    with application.unit_of_work.transaction() as tx:
        document = tx.documents.get(scenario.book_id)
        analysis = tx.analyses.get_for_section(scenario.book_id, scenario.section().id)
        section = document.ordered_sections[0]
        block = section.ordered_blocks[0]
        saved = replace(
            block,
            segments=(BlockSegment(0, cut, 0), BlockSegment(cut, len(text), 1)),
        )
        tx.documents.save(replace(document, sections=(replace(section, blocks=(saved,)),)))
        tx.analyses.save(scenario.book_id, analysis)
    start_run(
        application,
        scenario,
        results=(
            BlockProcessingResult(
                block_id,
                BlockTreatment.TRANSLATE,
                "Ясные слова.",
                segment_treatments=(BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE),
                segment_translations=(None, "Ясные слова."),
            ),
        ),
    )
    return scenario


def test_result_page_renders_units_like_book(client: TestClient, application: Application) -> None:
    """Reading units use the book classes, markers, speaker span, and one legend."""
    scenario = _units_book(application)
    endpoint = f"/books/{scenario.book_id}/result"
    response = client.get(endpoint, headers=HTML_HEADERS)
    tree = html.fromstring(response.text)
    paragraph, poem = tree.xpath("//article")
    units = paragraph.xpath('./div[contains(concat(" ", @class, " "), " lg-unit ")]')
    assert [unit.get("class") for unit in units] == ["lg-unit", "lg-unit lg-cont lg-end"]
    original, translated = units
    assert not original.xpath('./p[@class="lg-tr"]')
    source = original.xpath('./p[@class="lg-src"]')[0]
    link = source.xpath('./a[@class="lg-ref"]')[0]
    assert source.text == "Alice heard a tale"
    assert link.tail == "."
    assert link.get("href") == f'#{original.xpath("./ol[@class=\'lg-notes\']/li")[0].get("id")}'
    assert link.xpath("./sup")[0].text == "1"
    kinds = original.xpath('.//span[@class="lg-kind"]')
    assert [kind.text for kind in kinds] == ["Грамматика.", "Идиома."]
    assert all("," not in (kind.text or "") for kind in kinds)
    speaker = translated.xpath('./p[@class="lg-tr"]/span[@class="lg-speaker"]')[0]
    assert speaker.text == "Alice"
    assert "Потом она ушла." not in speaker.text_content()
    assert (speaker.tail or "").lstrip() == "Потом она ушла."
    poem_source = poem.xpath('.//p[@class="lg-src"]')[0]
    poem_text = "How doth\nImprove,\n\nAnd pour"
    assert len(poem_source.xpath("./br")) == poem_text.count("\n")
    assert b"<br" in html.tostring(poem_source).split(b"Improve,")[1].split(b"And pour")[0]
    assert poem_source.xpath("./br")[1].tail is None or poem_source.xpath("./br")[1].getnext().tag == "br"
    bare = poem.xpath('.//ol[@class="lg-notes"]/li')
    assert len(bare) == 1 and "Без маркера." in bare[0].text_content()
    assert not bare[0].xpath('.//span[@class="lg-num"]')
    assert not poem.xpath('.//a[@class="lg-ref"]')
    legend = client.get(endpoint).json()["legend"]
    shown = tree.xpath('//section[@class="lg-sec"]')
    assert len(shown) == 1
    assert [item.text_content() for item in shown[0].xpath("./p")] == list(legend)
    assert response.text.count("Как устроена эта книга") == 1
    assert not tree.xpath('//table[@class="lg-key"]')
    assert "Абзац оригинала всегда идёт первым." in legend


def test_no_part_labels_in_web(client: TestClient, application: Application) -> None:
    """The page has continuing units and no part captions, frames, or group root."""
    scenario = _two_part_page(application, "book-no-parts")
    page = client.get(f"/books/{scenario.book_id}/result", headers=HTML_HEADERS)
    tree = html.fromstring(page.text)

    assert "Часть " not in page.text
    assert "Part " not in page.text
    assert "result-fragment" not in page.text
    assert "linguary-paragraph-fragment" not in page.text
    assert tree.xpath('//*[contains(concat(" ", @class, " "), " lg-cont ")]')
    stylesheet = client.get("/static/linguary.css").text
    script = client.get("/static/linguary.js").text
    assert ".result-fragment" not in stylesheet
    assert ".lg-unit" in stylesheet
    assert "data-result-groups" not in script
    assert "data-result-page" in script


def test_block_actions_kept(
    configured_client: TestClient, configured_application: Application
) -> None:
    """Refresh, the task line, and a correction stay inside the unit article."""
    from test_correction_candidates_ui import _seed_book_with_awaiting_human_candidate

    book_id, block_id, _candidate = _seed_book_with_awaiting_human_candidate(
        configured_application, book_id="book-actions-kept"
    )
    store_task(
        configured_application,
        task=ProcessingTask(
            id="regen-kept",
            book_id=book_id,
            task_type=TaskType.REGENERATE_BLOCK,
            scope_id=block_id,
            status=TaskStatus.FAILED,
            last_error=TaskError(type=ErrorType.TRANSIENT, message="сбой", detail="x"),
        ),
    )
    tree = html.fromstring(
        configured_client.get(f"/books/{book_id}/result", headers=HTML_HEADERS).text
    )
    form = tree.xpath(f'//form[@data-regenerate-form][@data-block-id="{block_id}"]')[0]
    article = form.xpath("ancestor::article[1]")[0]
    button = form.xpath("./button")[0]

    assert button.text == "Обновить"
    assert button.get("disabled") is None
    assert "Задача: Повторная генерация блока /" in article.text_content()
    assert article.xpath('.//details[@class="correction-candidate"]')
    assert article.xpath('./div[contains(concat(" ", @class, " "), " lg-unit ")]')


def test_ipa_table_rendered_in_web(client: TestClient, application: Application) -> None:
    """The IPA key is one table for the book. An empty key renders no block."""
    scenario = _pronunciation_book(application)
    endpoint = f"/books/{scenario.book_id}/result"
    tree = html.fromstring(client.get(endpoint, headers=HTML_HEADERS).text)
    payload = client.get(endpoint).json()
    rows = payload.get("ipa_key")
    assert rows, sorted(payload)
    table, = tree.xpath('//table[@class="lg-key"]')

    assert len(tree.xpath('//section[contains(concat(" ", @class, " "), " chapter ")]')) == 2
    assert not table.xpath('ancestor::section[contains(concat(" ", @class, " "), " chapter ")]')
    assert [cell.text_content() for cell in table.xpath("./thead//th")] == ["Знак", "Пример", "Как звучит"]
    body = table.xpath("./tbody/tr")
    assert len(body) == len(rows) > 0
    for row, item in zip(body, rows, strict=True):
        cells = [cell.text_content() for cell in row.xpath("./td")]
        assert cells == [item["symbol"], item["example"], item["description"]]

    quiet = store_book(
        application,
        book_id="book-no-ipa",
        status=BookStatus.COMPLETED,
        chapters=(("Plain sentence.",),),
    )
    start_run(
        application,
        quiet,
        results=(BlockProcessingResult(quiet.block_ids()[0], BlockTreatment.READ_AS_IS),),
    )
    quiet_endpoint = f"/books/{quiet.book_id}/result"
    page = client.get(quiet_endpoint, headers=HTML_HEADERS)
    payload = client.get(quiet_endpoint).json()
    assert payload["legend"] == []
    assert payload["ipa_key"] == []
    assert "Как устроена эта книга" not in page.text
    assert "Ключ к транскрипции" not in page.text
    assert 'class="lg-key"' not in page.text
