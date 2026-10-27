"""Book processing plan and warnings (API-01, PRD § 15, § 21.6)."""

from __future__ import annotations

from fastapi.testclient import TestClient
from web_scenario import (
    dropped_block_warning,
    repetition_warning,
    section_id_of,
    start_run,
    store_book,
    translated_result,
)

from linguary.composition import Application
from linguary.domain import BlockTreatment, BookStatus


def test_plan_summarises_chapters_and_treatments(
    client: TestClient, application: Application
) -> None:
    """Plan summary displays chapters and treatment breakdown (UX notes)."""
    scenario = store_book(
        application,
        chapters=(("Первый абзац.", "Второй абзац."), ("Третий абзац.",)),
        treatments={
            0: (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY),
            1: (BlockTreatment.READ_AS_IS,),
        },
    )

    plan = client.get(f"/books/{scenario.book_id}/plan").json()

    assert plan["analyzed_sections"] == 2
    assert plan["total_sections"] == 2
    assert plan["is_complete"] is True
    assert [chapter["chapter_number"] for chapter in plan["chapters"]] == [1, 2]
    assert plan["chapters"][0]["textual_blocks"] == 2
    assert [
        (block["text"], block["treatment_label"])
        for block in plan["chapters"][0]["blocks"]
    ] == [
        ("Первый абзац.", "Полный перевод"),
        ("Второй абзац.", "Словарные подсказки"),
    ]
    assert plan["treatment_counts"] == [
        {"treatment": BlockTreatment.TRANSLATE.value, "label": "Полный перевод", "count": 1},
        {
            "treatment": BlockTreatment.GLOSSARY_ONLY.value,
            "label": "Словарные подсказки",
            "count": 1,
        },
        {"treatment": BlockTreatment.READ_AS_IS.value, "label": "В оригинале", "count": 1},
    ]


def test_plan_reports_partial_analysis(
    client: TestClient, application: Application
) -> None:
    """A book analysed only partially is shown as incomplete."""
    scenario = store_book(
        application,
        chapters=(("Первый абзац.",), ("Второй абзац.",)),
        analysed_chapters=(0,),
    )

    plan = client.get(f"/books/{scenario.book_id}/plan").json()

    assert plan["analyzed_sections"] == 1
    assert plan["total_sections"] == 2
    assert plan["is_complete"] is False
    assert [chapter["chapter_number"] for chapter in plan["chapters"]] == [1]


def test_plan_before_analysis_gives_409(
    client: TestClient, application: Application
) -> None:
    """Before analysis, no plan exists: summary responds with `409` rather than empty."""
    scenario = store_book(
        application, status=BookStatus.IMPORTED, analysed_chapters=()
    )

    response = client.get(f"/books/{scenario.book_id}/plan")

    assert response.status_code == 409
    assert "не готов" in response.json()["detail"]


def test_plan_of_unknown_book_gives_404(client: TestClient) -> None:
    """Plan of a nonexistent book returns `404`."""
    assert client.get("/books/unknown/plan").status_code == 404


def test_warnings_include_import_and_result_warnings(
    client: TestClient, application: Application
) -> None:
    """Import warnings and processing issues are displayed together (PRD § 21.6)."""
    section_id = section_id_of("book-web-1", 0)
    scenario = store_book(
        application,
        status=BookStatus.PROCESSING,
        import_warnings=(dropped_block_warning(section_id),),
    )
    start_run(
        application,
        scenario,
        results=(
            translated_result(
                scenario.block_ids()[0], warnings=(repetition_warning(),)
            ),
        ),
    )

    warnings = client.get(f"/books/{scenario.book_id}/warnings").json()

    assert warnings["total"] == 2
    assert warnings["import_warnings"][0]["kind_label"] == "Фрагмент книги пропущен"
    assert warnings["import_warnings"][0]["section_id"] == section_id
    assert warnings["result_warnings"][0]["kind_label"] == "Повторы в переводе"
    assert warnings["result_warnings"][0]["chapter_number"] == 1
    assert warnings["result_warnings"][0]["block_id"] == scenario.block_ids()[0]


def test_warnings_are_empty_for_a_clean_book(
    client: TestClient, application: Application
) -> None:
    """A book with no warnings returns empty lists instead of an error."""
    scenario = store_book(application)

    warnings = client.get(f"/books/{scenario.book_id}/warnings").json()

    assert warnings == {"import_warnings": [], "result_warnings": [], "total": 0}


def test_warnings_of_unknown_book_give_404(client: TestClient) -> None:
    """Warnings for a nonexistent book return `404`."""
    assert client.get("/books/unknown/warnings").status_code == 404
    assert client.get("/books/unknown/warnings").status_code == 404
