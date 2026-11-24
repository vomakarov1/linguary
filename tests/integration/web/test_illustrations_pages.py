"""HTML page tests for optional illustration viewing, registry pagination, labels and XSS safety (`P6.T1`)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from lxml import html

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.application.analysis_pass_cache import encode_pass
from linguary.application.llm_dto import NarrativeAnalysisResponse, QuotedSceneResponse
from linguary.application.ports.persistence import BookRecord
from linguary.composition import build_application
from linguary.domain import (
    AnalysisPassKind,
    AnalysisPassResult,
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    SourceReference,
    SupportStatus,
)
from linguary.domain.content_plan import canonical_sha
from linguary.domain.illustrations import (
    IllustrationGenerationState,
    IllustrationReasonCode,
    IllustrationRegistryKind,
    IllustrationScopeSpec,
    IllustrationScopeStage,
    IllustrationScopeState,
    IllustrationSettings,
    RegistryItem,
)
from linguary.web.app import create_app
from linguary.web.labels import (
    ILLUSTRATION_REASON_LABELS,
    ILLUSTRATION_REGISTRY_KIND_LABELS,
    ILLUSTRATION_STATE_LABELS,
    ILLUSTRATION_STYLE_LABELS,
)
from linguary.web.views import templates

HTML_HEADERS = {"Accept": "text/html"}


class _StagedDummyLLM:
    supports_staged_analysis = True

    async def generate(self, request: Any) -> Any:
        raise AssertionError("HTML read tests must not call LLM")


def _make_v3_app(tmp_path: Path) -> Any:
    return build_application(
        settings=AppSettings(
            storage=StorageSettings(data_root=str(tmp_path / "data")),
            llm=LLMSettings(api_key="test-key", model="gpt-5-mini", staged_analysis=True),
        ),
        llm_client=_StagedDummyLLM(),
        content_plan_profile="content-plan-v3",
    )


def _seed_v3_book(
    app: Any,
    tmp_path: Path,
    book_id: str,
    *,
    chapter_count: int = 2,
    empty_first: bool = False,
    status: BookStatus = BookStatus.GLOSSARY_REVIEW,
) -> tuple[list[str], str]:
    epub_path = tmp_path / f"{book_id}.epub"
    epub_path.write_bytes(b"PK\x03\x04v3-web-test-epub")
    sections: list[DocumentSection] = []
    sec_ids: list[str] = []
    for idx in range(chapter_count):
        sid = f"{book_id}-sec-{idx:02d}"
        sec_ids.append(sid)
        blocks = (
            ()
            if (empty_first and idx == 0)
            else (
                DocumentBlock(
                    id=f"{book_id}:b{idx}",
                    section_id=sid,
                    type=BlockType.PARAGRAPH,
                    order=0,
                    text=f"Alice opened door {idx} beneath the lantern.",
                    source_reference=SourceReference(f"Text/ch{idx}.xhtml", "p[0]"),
                    support_status=SupportStatus.SUPPORTED,
                ),
            )
        )
        sections.append(
            DocumentSection(
                id=sid,
                document_id=book_id,
                title=f"Chapter {idx + 1}",
                order=idx,
                source_resource_path=f"Text/ch{idx}.xhtml",
                blocks=blocks,
            )
        )
    doc = CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title=f"Web Book {book_id}"),
        source_language="en",
        target_language="ru",
        sections=tuple(sections),
    )
    book = BookRecord(
        id=book_id,
        metadata=doc.metadata,
        source_language="en",
        target_language="ru",
        status=status,
        source_epub_path=str(epub_path),
        created_at=datetime(2026, 11, 12, tzinfo=UTC),
    )
    with app.unit_of_work.transaction() as tx:
        tx.books.add(book)
        tx.documents.save(doc)
        cond = app.book_conditions.create(tx, book, app.runtime_settings.revision)
    return sec_ids, cond.id


def test_illustrations_tab_present_and_aria_current(tmp_path: Path) -> None:
    app = _make_v3_app(tmp_path)
    try:
        _seed_v3_book(app, tmp_path, "book-tab")
        web = create_app(application=app, run_background_worker=False)
        with TestClient(web, base_url="http://127.0.0.1:8000") as client:
            resp = client.get("/books/book-tab/illustrations", headers=HTML_HEADERS)
            assert resp.status_code == 200
            tree = html.fromstring(resp.text)
            tabs = tree.xpath('//nav[@class="tabs"]/a[@href="/books/book-tab/illustrations"]')
            assert len(tabs) == 1
            assert tabs[0].text_content().strip() == "Иллюстрации"
            assert tabs[0].get("aria-current") == "page"
    finally:
        asyncio.run(app.aclose())


def test_illustrations_disabled_and_not_started(tmp_path: Path) -> None:
    app = _make_v3_app(tmp_path)
    try:
        _seed_v3_book(app, tmp_path, "book-dis")
        web = create_app(application=app, run_background_worker=False)
        with TestClient(web, base_url="http://127.0.0.1:8000") as client:
            # Default: disabled
            r_dis = client.get("/books/book-dis/illustrations", headers=HTML_HEADERS)
            assert r_dis.status_code == 200
            assert ILLUSTRATION_STATE_LABELS["disabled"] in r_dis.text
            assert ILLUSTRATION_REASON_LABELS["channel_disabled"] in r_dis.text

            # Enable -> not_started (compatible v3 conditions)
            app.illustrations.save_settings(
                "book-dis",
                expected_revision=0,
                settings=IllustrationSettings(
                    enabled=True,
                    max_per_chapter=2,
                    model_reference="gpt-5-mini",
                    model_prompts="gpt-5-mini",
                ),
            )
            r_ns = client.get("/books/book-dis/illustrations", headers=HTML_HEADERS)
            assert r_ns.status_code == 200
            assert ILLUSTRATION_STATE_LABELS["not_started"] in r_ns.text
    finally:
        asyncio.run(app.aclose())


def test_illustrations_all_12_states_including_revoked_28_reasons_and_4_registry_kinds_have_russian_labels() -> None:
    expected_states = {
        "disabled",
        "not_started",
        "waiting_source",
        "needs_analysis",
        "queued",
        "running",
        "ready",
        "failed",
        "stale",
        "skipped",
        "paused",
        "revoked",
    }
    assert set(ILLUSTRATION_STATE_LABELS.keys()) == expected_states
    assert len(ILLUSTRATION_STATE_LABELS) == 12
    assert len(ILLUSTRATION_REASON_LABELS) == 28
    assert set(ILLUSTRATION_REASON_LABELS.keys()) == {rc.value for rc in IllustrationReasonCode}
    assert len(ILLUSTRATION_REGISTRY_KIND_LABELS) == 4
    assert set(ILLUSTRATION_REGISTRY_KIND_LABELS.keys()) == {
        k.value for k in IllustrationRegistryKind
    }
    assert "monochrome_ink" in ILLUSTRATION_STYLE_LABELS

    # Verify all 4 globals are registered in Jinja2 environment and non-empty Cyrillic text
    for name in (
        "illustration_state_labels",
        "illustration_reason_labels",
        "illustration_registry_kind_labels",
        "illustration_style_labels",
    ):
        mapping = templates.env.globals[name]
        for key, label in mapping.items():
            assert isinstance(label, str) and label.strip() and label != key
            assert any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in label)


def test_illustrations_source_progress_counts_full_narratives(tmp_path: Path) -> None:
    app = _make_v3_app(tmp_path)
    try:
        sec_ids, cond_id = _seed_v3_book(app, tmp_path, "book-prog", chapter_count=3, empty_first=True)
        # Save a fragment pass (scope_key != "") and one full pass for sec_ids[1]
        nar = NarrativeAnalysisResponse(
            translation_instructions="Сохраняй тон.",
            chapter_summary="Alice opened the door.",
            scenes=(
                QuotedSceneResponse(
                    start_block_id="b1",
                    end_block_id="b1",
                    summary="Алиса открыла дверь.",
                    location="Холл",
                    participants=("Alice",),
                ),
            ),
        )
        encoded = encode_pass(nar)
        encoded["input_suggestions"] = []
        with app.unit_of_work.transaction() as tx:
            tx.analyses.save_pass(
                AnalysisPassResult(
                    id="frag-only",
                    book_id="book-prog",
                    section_id=sec_ids[2],
                    pass_kind=AnalysisPassKind.NARRATIVE,
                    scope_key="frag:0",
                    input_sha256="a" * 64,
                    output_sha256=canonical_sha(encoded),
                    conditions_id=cond_id,
                    attempt_id=None,
                    payload=encoded,
                )
            )
        web = create_app(application=app, run_background_worker=False)
        with TestClient(web, base_url="http://127.0.0.1:8000") as client:
            resp = client.get("/books/book-prog/illustrations", headers=HTML_HEADERS)
            assert resp.status_code == 200
            # Expected content chapters = 2 (sec-00 empty is skipped), ready = 0 (fragment does NOT count!)
            assert "0 / 2 глав" in resp.text
            assert "Пропущено служебных разделов: 1" in resp.text
    finally:
        asyncio.run(app.aclose())


def test_illustrations_sequential_analysis_waits_for_source(tmp_path: Path) -> None:
    app = _make_v3_app(tmp_path)
    try:
        sec_ids, _ = _seed_v3_book(
            app, tmp_path, "book-seq", chapter_count=2, status=BookStatus.ANALYZING
        )
        app.illustrations.save_settings(
            "book-seq",
            expected_revision=0,
            settings=IllustrationSettings(
                enabled=True,
                max_per_chapter=2,
                model_reference="gpt-5-mini",
                model_prompts="gpt-5-mini",
            ),
        )
        asyncio.run(app.illustrations.prepare("book-seq", expected_revision=1))
        web = create_app(application=app, run_background_worker=False)
        with TestClient(web, base_url="http://127.0.0.1:8000") as client:
            resp = client.get("/books/book-seq/illustrations", headers=HTML_HEADERS)
            assert resp.status_code == 200
            assert ILLUSTRATION_STATE_LABELS["waiting_source"] in resp.text
            assert sec_ids[0] in resp.text
    finally:
        asyncio.run(app.aclose())


def test_illustrations_selected_chapter_outside_pagination_window(tmp_path: Path) -> None:
    app = _make_v3_app(tmp_path)
    try:
        sec_ids, _ = _seed_v3_book(app, tmp_path, "book-win", chapter_count=5)
        web = create_app(application=app, run_background_worker=False)
        with TestClient(web, base_url="http://127.0.0.1:8000") as client:
            # Window shows chapters 0..1 (offset=0, limit=2), but selected section_id is chapter 4!
            resp = client.get(
                f"/books/book-win/illustrations?offset=0&limit=2&section_id={sec_ids[4]}",
                headers=HTML_HEADERS,
            )
            assert resp.status_code == 200
            assert "Глава: Chapter 5" in resp.text
            assert "Следующие главы" in resp.text
    finally:
        asyncio.run(app.aclose())


def test_illustrations_zero_and_skipped_reason(tmp_path: Path) -> None:
    app = _make_v3_app(tmp_path)
    try:
        sec_ids, _ = _seed_v3_book(app, tmp_path, "book-zero", chapter_count=2, empty_first=True)
        rev = app.illustrations.save_settings(
            "book-zero",
            expected_revision=0,
            settings=IllustrationSettings(
                enabled=True,
                max_per_chapter=2,
                model_reference="gpt-5-mini",
                model_prompts="gpt-5-mini",
            ),
        )
        with app.unit_of_work.transaction() as tx:
            gen, _ = tx.illustrations.create_or_get_generation(
                "book-zero",
                settings_id=rev.id,
                request_key="req-zero",
                source_sha256="a" * 64,
                source_authority={},
                requested_sections=(sec_ids[1],),
            )
            sc_skip, sc_zero = tx.illustrations.upsert_scopes(
                gen.id,
                [
                    IllustrationScopeSpec(
                        stage=IllustrationScopeStage.CHAPTER,
                        scope_key=sec_ids[0],
                        section_id=sec_ids[0],
                        section_order=0,
                        section_title="Chapter 1",
                        input_sha256="b" * 64,
                        state=IllustrationScopeState.SKIPPED,
                        reason_code=IllustrationReasonCode.NO_CONTENT.value,
                        summary={},
                    ),
                    IllustrationScopeSpec(
                        stage=IllustrationScopeStage.CHAPTER,
                        scope_key=sec_ids[1],
                        section_id=sec_ids[1],
                        section_order=1,
                        section_title="Chapter 2",
                        input_sha256="c" * 64,
                        summary={"moment_count": 0},
                    ),
                ],
            )
            tx.illustrations.accept_scope(
                sc_zero.id,
                attempt_id=None,
                expected_input_sha256=sc_zero.input_sha256,
                payload={
                    "schema_version": 1,
                    "section_id": sec_ids[1],
                    "generation_id": gen.id,
                    "style": "monochrome_ink",
                    "moments": [],
                    "zero_reason_ru": "В этой главе только спокойный диалог без выразительного действия.",
                },
                output_sha256="d" * 64,
                summary={"moment_count": 0},
            )
            del sc_skip
        web = create_app(application=app, run_background_worker=False)
        with TestClient(web, base_url="http://127.0.0.1:8000") as client:
            r1 = client.get(
                f"/books/book-zero/illustrations?section_id={sec_ids[1]}",
                headers=HTML_HEADERS,
            )
            assert r1.status_code == 200
            assert "В этой главе только спокойный диалог без выразительного действия." in r1.text
            assert ILLUSTRATION_REASON_LABELS["no_content"] in r1.text
    finally:
        asyncio.run(app.aclose())


def test_illustrations_read_only_registry_panel_paginated_via_api01_and_json_link(
    tmp_path: Path,
) -> None:
    app = _make_v3_app(tmp_path)
    try:
        sec_ids, _ = _seed_v3_book(app, tmp_path, "book-reg", chapter_count=1)
        rev = app.illustrations.save_settings(
            "book-reg",
            expected_revision=0,
            settings=IllustrationSettings(
                enabled=True,
                max_per_chapter=2,
                model_reference="gpt-5-mini",
                model_prompts="gpt-5-mini",
            ),
        )
        with app.unit_of_work.transaction() as tx:
            gen, _ = tx.illustrations.create_or_get_generation(
                "book-reg",
                settings_id=rev.id,
                request_key="req-reg",
                source_sha256="a" * 64,
                source_authority={},
                requested_sections=tuple(sec_ids),
            )
            items = [
                RegistryItem(
                    generation_id=gen.id,
                    kind=IllustrationRegistryKind.ENTITY if idx % 2 == 0 else IllustrationRegistryKind.FACT,
                    item_key=f"item-{idx:02d}",
                    entity_id=f"ent-{idx:02d}",
                    position_key=f"0000:{idx:04d}",
                    is_back_applicable=True,
                    payload_sha256="e" * 64,
                    payload={"summary_ru": f"Запись справочника №{idx}"},
                )
                for idx in range(5)
            ]
            tx.illustrations.upsert_registry_items(gen.id, items)

        web = create_app(application=app, run_background_worker=False)
        with TestClient(web, base_url="http://127.0.0.1:8000") as client:
            resp = client.get(
                "/books/book-reg/illustrations?registry_limit=2&registry_offset=0",
                headers=HTML_HEADERS,
            )
            assert resp.status_code == 200
            tree = html.fromstring(resp.text)
            assert "Запись справочника №0" in resp.text
            assert "Запись справочника №1" in resp.text
            assert "Запись справочника №2" not in resp.text
            # Next page link targets API-01 with registry_offset=2
            next_links = tree.xpath('//a[contains(@href, "registry_offset=2")]')
            assert next_links
            assert "/books/book-reg/illustrations?" in next_links[0].get("href", "")
            # JSON link explicitly labeled «JSON справочника» targeting API-04
            json_links = tree.xpath('//a[normalize-space(.)="JSON справочника"]')
            assert len(json_links) == 1
            assert "/books/book-reg/illustrations/registry" in json_links[0].get("href", "")
            # Kind filter link works via API-01
            r_fact = client.get(
                "/books/book-reg/illustrations?registry_kind=fact&registry_limit=10",
                headers=HTML_HEADERS,
            )
            assert r_fact.status_code == 200
            assert "Запись справочника №1" in r_fact.text
            assert "Запись справочника №0" not in r_fact.text
    finally:
        asyncio.run(app.aclose())


def test_illustrations_history_read_only_uses_is_history(tmp_path: Path) -> None:
    app = _make_v3_app(tmp_path)
    try:
        sec_ids, _ = _seed_v3_book(app, tmp_path, "book-hist", chapter_count=1)
        rev1 = app.illustrations.save_settings(
            "book-hist",
            expected_revision=0,
            settings=IllustrationSettings(
                enabled=True,
                max_per_chapter=2,
                model_reference="gpt-5-mini",
                model_prompts="gpt-5-mini",
            ),
        )
        with app.unit_of_work.transaction() as tx:
            gen1, _ = tx.illustrations.create_or_get_generation(
                "book-hist",
                settings_id=rev1.id,
                request_key="req-h1",
                source_sha256="a" * 64,
                source_authority={},
                requested_sections=tuple(sec_ids),
            )
            tx.illustrations.revoke_generation(
                "book-hist", gen1.id, reason_code=IllustrationReasonCode.SETTINGS_CHANGED.value
            )
        web = create_app(application=app, run_background_worker=False)
        with TestClient(web, base_url="http://127.0.0.1:8000") as client:
            # Viewing current book (no active generation, is_history=False) -> NO history banner!
            r_curr = client.get("/books/book-hist/illustrations", headers=HTML_HEADERS)
            assert r_curr.status_code == 200
            assert "Историческое поколение (только чтение)" not in r_curr.text
            # History table shows revoked Russian label
            assert ILLUSTRATION_STATE_LABELS["revoked"] in r_curr.text

            # Explicitly opening revoked gen1 -> is_history=True -> banner shown!
            r_hist = client.get(
                f"/books/book-hist/illustrations?generation_id={gen1.id}",
                headers=HTML_HEADERS,
            )
            assert r_hist.status_code == 200
            assert "Историческое поколение (только чтение)" in r_hist.text
    finally:
        asyncio.run(app.aclose())


def test_illustrations_textarea_breakout_and_xss_escaped_in_all_fields(tmp_path: Path) -> None:
    app = _make_v3_app(tmp_path)
    try:
        sec_ids, _ = _seed_v3_book(app, tmp_path, "book-xss", chapter_count=1)
        rev = app.illustrations.save_settings(
            "book-xss",
            expected_revision=0,
            settings=IllustrationSettings(
                enabled=True,
                max_per_chapter=2,
                model_reference="gpt-5-mini",
                model_prompts="gpt-5-mini",
            ),
        )
        xss = '</textarea><script>alert("xss")</script>'
        prompt_exact = f'Line 1 "quoted"\n{xss}\nLine 3'
        with app.unit_of_work.transaction() as tx:
            gen, _ = tx.illustrations.create_or_get_generation(
                "book-xss",
                settings_id=rev.id,
                request_key="req-xss",
                source_sha256="a" * 64,
                source_authority={},
                requested_sections=tuple(sec_ids),
            )
            sc = tx.illustrations.upsert_scopes(
                gen.id,
                [
                    IllustrationScopeSpec(
                        stage=IllustrationScopeStage.CHAPTER,
                        scope_key=sec_ids[0],
                        section_id=sec_ids[0],
                        section_order=0,
                        section_title="Chapter 1",
                        input_sha256="b" * 64,
                        summary={"moment_count": 1},
                    )
                ],
            )[0]
            tx.illustrations.accept_scope(
                sc.id,
                attempt_id=None,
                expected_input_sha256=sc.input_sha256,
                payload={
                    "schema_version": 1,
                    "section_id": sec_ids[0],
                    "generation_id": gen.id,
                    "style": "monochrome_ink",
                    "moments": [
                        {
                            "id": "m1",
                            "scene_id": "s0",
                            "title_ru": f"Заголовок {xss}",
                            "reason_ru": f"Причина {xss}",
                            "event_ru": f"Событие {xss}",
                            "prompt_en": prompt_exact,
                            "insert_after_block_id": "book-xss:b0",
                            "anchors": [
                                {
                                    "block_id": "book-xss:b0",
                                    "start": 0,
                                    "end": 5,
                                    "quote": f"Цитата {xss}",
                                }
                            ],
                            "character_descriptions": [
                                {
                                    "entity_id": "e1",
                                    "name_ru": f"Герой {xss}",
                                    "description_ru": f"Описание {xss}",
                                    "description_en": f"Desc {xss}",
                                    "fact_ids": [],
                                    "assumptions_ru": [f"Допущение {xss}"],
                                }
                            ],
                            "location_descriptions": [
                                {
                                    "entity_id": "l1",
                                    "name_ru": f"Место {xss}",
                                    "description_ru": f"Локация {xss}",
                                    "description_en": f"Loc {xss}",
                                    "fact_ids": [],
                                    "assumptions_ru": [],
                                }
                            ],
                            "assumptions_ru": [f"Решение {xss}"],
                        }
                    ],
                },
                output_sha256="c" * 64,
                summary={"moment_count": 1},
            )
            tx.illustrations.upsert_registry_items(
                gen.id,
                [
                    RegistryItem(
                        generation_id=gen.id,
                        kind=IllustrationRegistryKind.ENTITY,
                        item_key="e1",
                        entity_id="e1",
                        position_key="0000:0000",
                        is_back_applicable=True,
                        payload_sha256="d" * 64,
                        payload={"summary_ru": f"Реестр {xss}"},
                    )
                ],
            )

        web = create_app(application=app, run_background_worker=False)
        with TestClient(web, base_url="http://127.0.0.1:8000") as client:
            resp = client.get(
                f"/books/book-xss/illustrations?section_id={sec_ids[0]}",
                headers=HTML_HEADERS,
            )
            assert resp.status_code == 200
            # Raw <script> MUST NOT appear anywhere in HTML!
            assert "<script>alert(" not in resp.text
            tree = html.fromstring(resp.text)
            textareas = tree.xpath('//textarea[@id="prompt-m1"]')
            assert len(textareas) == 1
            assert textareas[0].get("readonly") is not None
            # Exact verbatim prompt_en without leading/trailing indentation!
            assert textareas[0].text == prompt_exact
    finally:
        asyncio.run(app.aclose())


def test_illustrations_no_meta_refresh_or_polling_selectors(tmp_path: Path) -> None:
    app = _make_v3_app(tmp_path)
    try:
        sec_ids, _ = _seed_v3_book(app, tmp_path, "book-nopoll", chapter_count=1)
        rev = app.illustrations.save_settings(
            "book-nopoll",
            expected_revision=0,
            settings=IllustrationSettings(
                enabled=True,
                max_per_chapter=2,
                model_reference="gpt-5-mini",
                model_prompts="gpt-5-mini",
            ),
        )
        with app.unit_of_work.transaction() as tx:
            gen, _ = tx.illustrations.create_or_get_generation(
                "book-nopoll",
                settings_id=rev.id,
                request_key="req-np",
                source_sha256="a" * 64,
                source_authority={},
                requested_sections=tuple(sec_ids),
            )
            tx.illustrations.update_generation_state(
                "book-nopoll", gen.id, state=IllustrationGenerationState.RUNNING
            )
        web = create_app(application=app, run_background_worker=False)
        with TestClient(web, base_url="http://127.0.0.1:8000") as client:
            resp = client.get("/books/book-nopoll/illustrations", headers=HTML_HEADERS)
            assert resp.status_code == 200
            assert 'http-equiv="refresh"' not in resp.text.lower()
            assert "data-result-page" not in resp.text
            assert "data-active-task" not in resp.text
    finally:
        asyncio.run(app.aclose())
