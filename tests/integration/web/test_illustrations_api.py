"""Web API integration tests for optional illustrations (`API-01`, `API-02`, `API-04`, P1.T2)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient

from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, ServerSettings, StorageSettings
from linguary.application.ports.persistence import BookRecord
from linguary.composition import Application, build_application
from linguary.domain import (
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    ProcessingTask,
    SourceReference,
    SupportStatus,
    TaskStatus,
    TaskType,
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
    format_position_key,
)
from linguary.web import create_app
from linguary.web.labels import (
    ILLUSTRATION_REASON_LABELS,
    ILLUSTRATION_REGISTRY_KIND_LABELS,
    ILLUSTRATION_STATE_LABELS,
    ILLUSTRATION_STYLE_LABELS,
)
from linguary.web.routes import illustrations as illustrations_route
from linguary.web.routes.result import block_anchor
from linguary.web.views import templates


def _seed_book(
    app: Application,
    book_id: str = "book-1",
    *,
    num_sections: int = 3,
    text_prefix: str = "Alice walked under the silver moon.",
) -> CanonicalDocument:
    sections = []
    for idx in range(num_sections):
        sec_id = f"{book_id}-sec-{idx}"
        blk_id = f"{book_id}:blk:{idx}"
        sections.append(
            DocumentSection(
                id=sec_id,
                document_id=book_id,
                title=f"Chapter {idx + 1}",
                order=idx,
                source_resource_path=f"Text/ch{idx + 1}.xhtml",
                blocks=(
                    DocumentBlock(
                        id=blk_id,
                        section_id=sec_id,
                        type=BlockType.PARAGRAPH,
                        order=0,
                        text=f"{text_prefix} Section {idx + 1}.",
                        source_reference=SourceReference(f"Text/ch{idx + 1}.xhtml", "p[0]"),
                        support_status=SupportStatus.SUPPORTED,
                    ),
                ),
            )
        )
    doc = CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title=f"Book {book_id}"),
        source_language="en",
        target_language="ru",
        sections=tuple(sections),
    )
    record = BookRecord(
        id=book_id,
        metadata=BookMetadata(title=f"Book {book_id}"),
        source_language="en",
        target_language="ru",
        status=BookStatus.IMPORTED,
        source_epub_path=f"/tmp/{book_id}.epub",
        created_at=datetime(2026, 11, 1, tzinfo=UTC),
    )
    with app.unit_of_work.transaction() as tx:
        tx.books.add(record)
        tx.documents.save(doc)
    return doc


def _seed_generation_with_chapters(
    app: Application,
    book_id: str,
    doc: CanonicalDocument,
    *,
    with_moments: bool = True,
) -> tuple[str, str]:
    with app.unit_of_work.transaction() as tx:
        head = tx.illustrations.get_head(book_id)
        rev = tx.illustrations.save_settings(
            book_id,
            expected_revision=head.revision_number,
            settings=IllustrationSettings(
                enabled=True,
                max_per_chapter=3,
                model_reference="openai/gpt-5-mini",
                model_prompts="openai/gpt-5",
            ),
        )
        gen, _ = tx.illustrations.create_or_get_generation(
            book_id,
            settings_id=rev.id,
            request_key=f"req-{uuid4()}",
            source_sha256="a" * 64,
            source_authority={
                "conditions_id": str(uuid4()),
                "conditions_revision": 1,
                "source_digest": "a" * 64,
                "manifest_sha256": None,
            },
            requested_sections=[s.id for s in doc.ordered_sections],
        )
        specs = []
        for sec in doc.ordered_sections:
            blk = sec.ordered_blocks[0]
            quote = blk.text[:12]
            payload = (
                {
                    "zero_reason_ru": None,
                    "moments": [
                        {
                            "id": f"m-{sec.order}",
                            "scene_id": f"sc-{sec.order}",
                            "title_ru": f"Сцена {sec.order + 1}",
                            "reason_ru": "Ключевой момент главы",
                            "event_ru": "Алиса идёт под луной",
                            "prompt_en": "Monochrome ink illustration of Alice under the silver moon.",
                            "insert_after_block_id": blk.id,
                            "anchors": [
                                {"block_id": blk.id, "start": 0, "end": len(quote), "quote": quote}
                            ],
                            "character_descriptions": [
                                {
                                    "entity_id": "ent-alice",
                                    "name_ru": "Алиса",
                                    "description_ru": "Девушка в плаще",
                                    "description_en": "Young woman in a cloak",
                                    "fact_ids": ["f-1"],
                                    "assumptions_ru": [],
                                }
                            ],
                            "location_descriptions": [],
                            "assumptions_ru": [],
                        }
                    ],
                }
                if with_moments
                else {"zero_reason_ru": "Нет визуально выразительных сцен", "moments": []}
            )
            specs.append(
                IllustrationScopeSpec(
                    stage=IllustrationScopeStage.CHAPTER,
                    scope_key=f"chapter:{sec.id}",
                    input_sha256=canonical_sha({"sec": sec.id}),
                    section_id=sec.id,
                    section_order=sec.order,
                    section_title=sec.title,
                    state=IllustrationScopeState.ACCEPTED,
                    output_sha256=canonical_sha(payload),
                    payload=payload,
                    summary={"moment_count": len(payload["moments"])},
                )
            )
        tx.illustrations.upsert_scopes(gen.id, specs)
        tx.illustrations.update_generation_state(
            book_id, gen.id, state=IllustrationGenerationState.READY
        )
        return rev.id, gen.id


def test_settings_never_enqueues_or_calls_llm(
    application: Application, client: TestClient, llm_client
) -> None:
    _seed_book(application)
    resp = client.post(
        "/books/book-1/illustrations/settings",
        json={
            "expected_revision": 0,
            "enabled": True,
            "max_per_chapter": 2,
            "style": "monochrome_ink",
            "model_reference": "openai/gpt-5-mini",
            "model_prompts": "openai/gpt-5",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["settings_revision"] == 1
    assert body["settings"]["max_per_chapter"] == 2
    assert llm_client.requests == []
    with application.unit_of_work.read_transaction() as tx:
        assert tx.tasks.list_for_book("book-1") == ()


def test_get_never_mutates(application: Application, client: TestClient, llm_client) -> None:
    _seed_book(application)
    resp = client.get("/books/book-1/illustrations")
    assert resp.status_code == 200
    assert resp.json()["state"] == "disabled"
    assert resp.json()["settings_revision"] == 0
    assert llm_client.requests == []
    with application.unit_of_work.read_transaction() as tx:
        head = tx.illustrations.get_head("book-1")
        assert head.settings_id is None
        assert head.active_generation_id is None
        assert tx.tasks.list_for_book("book-1") == ()


def test_get_succeeds_without_provider_api_key(tmp_path: Path) -> None:
    app_settings = AppSettings(
        llm=LLMSettings(api_key=None, model=""),
        checks=CheckSettings(),
        storage=StorageSettings(data_root=str(tmp_path / "nokey")),
        server=ServerSettings(),
    )
    built = build_application(settings=app_settings, llm_client=None)
    try:
        _seed_book(built)
        web_app = create_app(application=built, run_background_worker=False)
        with TestClient(web_app, base_url=app_settings.server.base_url) as tc:
            resp = tc.get("/books/book-1/illustrations")
            assert resp.status_code == 200
            data = resp.json()
            assert data["defaults_hint"]["provider_ready"] is False
            assert data["state"] == "disabled"
    finally:
        built.dispose()


def test_paginated_headers_only_one_payload(application: Application, client: TestClient) -> None:
    doc = _seed_book(application, num_sections=5)
    _seed_generation_with_chapters(application, "book-1", doc)

    page1 = client.get("/books/book-1/illustrations?offset=0&limit=2")
    assert page1.status_code == 200
    body1 = page1.json()
    assert len(body1["chapter_headers"]) == 2
    assert body1["pagination"] == {
        "offset": 0,
        "limit": 2,
        "total": 5,
        "next_offset": 2,
        "previous_offset": None,
    }
    assert body1["selected_chapter"] is None

    # Request a section outside the current header page (sec-4 while page shows 0..1).
    selected = client.get("/books/book-1/illustrations?offset=0&limit=2&section_id=book-1-sec-4")
    assert selected.status_code == 200
    body_sel = selected.json()
    assert len(body_sel["chapter_headers"]) == 2
    assert body_sel["selected_chapter"]["section_id"] == "book-1-sec-4"
    assert len(body_sel["selected_chapter"]["moments"]) == 1

    page2 = client.get("/books/book-1/illustrations?offset=4&limit=2")
    assert page2.status_code == 200
    assert page2.json()["pagination"]["previous_offset"] == 2
    assert page2.json()["pagination"]["next_offset"] is None


def test_api01_inline_registry_pagination_and_kind_filter(
    application: Application, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = _seed_book(application, num_sections=1)
    _, gen_id = _seed_generation_with_chapters(application, "book-1", doc)
    with application.unit_of_work.transaction() as tx:
        tx.illustrations.upsert_registry_items(
            gen_id,
            [
                RegistryItem(
                    generation_id=gen_id,
                    kind=IllustrationRegistryKind.ENTITY,
                    item_key="e-1",
                    entity_id="ent-alice",
                    position_key=format_position_key(0, 0, 0),
                    is_back_applicable=False,
                    payload_sha256="1" * 64,
                    payload={"name_ru": "Алиса"},
                ),
                RegistryItem(
                    generation_id=gen_id,
                    kind=IllustrationRegistryKind.FACT,
                    item_key="f-1",
                    entity_id="ent-alice",
                    position_key=format_position_key(0, 0, 1),
                    is_back_applicable=True,
                    payload_sha256="2" * 64,
                    payload={"description_ru": "Зелёные глаза"},
                ),
                RegistryItem(
                    generation_id=gen_id,
                    kind=IllustrationRegistryKind.FACT,
                    item_key="f-2",
                    entity_id="ent-alice",
                    position_key=format_position_key(0, 0, 2),
                    is_back_applicable=False,
                    payload_sha256="3" * 64,
                    payload={"description_ru": "Синий плащ"},
                ),
            ],
        )

    captured: dict[str, object] = {}

    def _fake_render(request, template, **context):
        captured["template"] = template
        captured.update(context)
        return HTMLResponse("<html>ok</html>")

    monkeypatch.setattr(illustrations_route, "render", _fake_render)
    resp = client.get(
        "/books/book-1/illustrations?registry_kind=fact&registry_offset=1&registry_limit=1",
        headers={"Accept": "text/html"},
    )
    assert resp.status_code == 200
    assert captured["template"] == "books/illustrations.html"
    assert captured["requested_generation_id"] is None
    assert captured["registry_kind"] == "fact"
    assert captured["active_tab"] == "illustrations"
    assert captured["page"] == "illustrations"
    reg = captured["registry"]
    assert reg is not None
    assert reg.pagination.total == 2
    assert reg.pagination.offset == 1
    assert reg.pagination.limit == 1
    assert len(reg.items) == 1
    assert reg.items[0].item_key == "f-2"


def test_settings_idempotent_replay_and_bool_int_rejection_for_revision_and_max_per_chapter(
    application: Application, client: TestClient
) -> None:
    _seed_book(application)
    payload = {
        "expected_revision": 0,
        "enabled": True,
        "max_per_chapter": 2,
        "style": "monochrome_ink",
        "model_reference": "openai/gpt-5-mini",
        "model_prompts": "openai/gpt-5",
    }
    first = client.post("/books/book-1/illustrations/settings", json=payload)
    assert first.status_code == 200
    assert first.json()["settings_revision"] == 1

    # Replay with expected_revision=0 (current - 1) and identical payload succeeds idempotently.
    replay = client.post("/books/book-1/illustrations/settings", json=payload)
    assert replay.status_code == 200
    assert replay.json()["settings_revision"] == 1

    # Bool rejected for expected_revision and max_per_chapter.
    bad_rev = client.post(
        "/books/book-1/illustrations/settings",
        json={**payload, "expected_revision": True},
    )
    assert bad_rev.status_code == 422
    assert bad_rev.json() == {"detail": {"reason_code": "invalid_request"}}

    bad_max = client.post(
        "/books/book-1/illustrations/settings",
        json={**payload, "expected_revision": 1, "max_per_chapter": True},
    )
    assert bad_max.status_code == 422
    assert bad_max.json() == {"detail": {"reason_code": "invalid_request"}}

    # Missing enabled in JSON is rejected.
    missing_enabled = client.post(
        "/books/book-1/illustrations/settings",
        json={"expected_revision": 1},
    )
    assert missing_enabled.status_code == 422
    assert missing_enabled.json() == {"detail": {"reason_code": "invalid_request"}}


def test_settings_change_revokes_and_supersedes_while_noop_preserves_generation(
    application: Application, client: TestClient
) -> None:
    doc = _seed_book(application, num_sections=1)
    _, gen_id = _seed_generation_with_chapters(application, "book-1", doc)
    with application.unit_of_work.transaction() as tx:
        tx.tasks.add(
            ProcessingTask(
                id="ill-task-1",
                book_id="book-1",
                task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
                scope_id="scope-1",
                status=TaskStatus.PENDING,
                operation_id=f"{gen_id}:a1",
                policy_version="illustration-v1",
            )
        )

    # Identical payload at current revision=1 is a no-op and preserves active generation & task.
    noop = client.post(
        "/books/book-1/illustrations/settings",
        json={
            "expected_revision": 1,
            "enabled": True,
            "max_per_chapter": 3,
            "style": "monochrome_ink",
            "model_reference": "openai/gpt-5-mini",
            "model_prompts": "openai/gpt-5",
        },
    )
    assert noop.status_code == 200
    assert noop.json()["settings_revision"] == 1
    assert noop.json()["active_generation_id"] == gen_id

    # Changing max_per_chapter revokes active generation and supersedes its task.
    changed = client.post(
        "/books/book-1/illustrations/settings",
        json={
            "expected_revision": 1,
            "enabled": True,
            "max_per_chapter": 2,
            "style": "monochrome_ink",
            "model_reference": "openai/gpt-5-mini",
            "model_prompts": "openai/gpt-5",
        },
    )
    assert changed.status_code == 200
    assert changed.json()["settings_revision"] == 2
    assert changed.json()["active_generation_id"] is None
    with application.unit_of_work.read_transaction() as tx:
        revoked = tx.illustrations.get_generation("book-1", gen_id)
        assert revoked is not None
        assert revoked.state is IllustrationGenerationState.REVOKED
        assert revoked.reason_code == IllustrationReasonCode.SETTINGS_CHANGED.value
        task = tx.tasks.get("ill-task-1")
        assert task is not None
        assert task.status is TaskStatus.SUPERSEDED
        assert task.last_error is None


def test_settings_form_unchecked_enabled_validation_limits_and_control_char_rejection(
    application: Application, client: TestClient
) -> None:
    _seed_book(application)
    # Unchecked checkbox (no 'enabled' field) -> disabled; whitespace models -> null.
    resp = client.post(
        "/books/book-1/illustrations/settings",
        data={
            "expected_revision": "0",
            "max_per_chapter": "2",
            "model_reference": "   ",
            "model_prompts": "",
        },
        headers={"Accept": "application/json"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is False
    assert data["settings"]["model_reference"] is None
    assert data["settings"]["model_prompts"] is None

    # Model > 200 chars rejected.
    too_long = client.post(
        "/books/book-1/illustrations/settings",
        json={
            "expected_revision": 1,
            "enabled": False,
            "max_per_chapter": 2,
            "model_reference": "m" * 201,
            "model_prompts": None,
        },
    )
    assert too_long.status_code == 422
    assert too_long.json() == {"detail": {"reason_code": "invalid_request"}}

    # Control char rejected (no silent stripping).
    ctrl = client.post(
        "/books/book-1/illustrations/settings",
        json={
            "expected_revision": 1,
            "enabled": False,
            "max_per_chapter": 2,
            "model_reference": "gpt-5\nmini",
            "model_prompts": None,
        },
    )
    assert ctrl.status_code == 422
    assert ctrl.json() == {"detail": {"reason_code": "invalid_request"}}

    # Enabled without both models rejected.
    missing_model = client.post(
        "/books/book-1/illustrations/settings",
        json={
            "expected_revision": 1,
            "enabled": True,
            "max_per_chapter": 2,
            "model_reference": "gpt-5",
            "model_prompts": None,
        },
    )
    assert missing_model.status_code == 422


def test_settings_return_params_validated_before_mutation_html_303_and_forbidden_in_json(
    application: Application, client: TestClient
) -> None:
    _seed_book(application)
    # return_* forbidden in JSON.
    json_with_return = client.post(
        "/books/book-1/illustrations/settings",
        json={
            "expected_revision": 0,
            "enabled": False,
            "max_per_chapter": 3,
            "return_offset": 0,
        },
    )
    assert json_with_return.status_code == 422
    assert json_with_return.json() == {"detail": {"reason_code": "invalid_request"}}

    # Invalid return_limit in HTML form must fail BEFORE DB mutation.
    bad_return = client.post(
        "/books/book-1/illustrations/settings",
        data={
            "expected_revision": "0",
            "max_per_chapter": "2",
            "return_limit": "999",
        },
        headers={"Accept": "text/html"},
        follow_redirects=False,
    )
    assert bad_return.status_code == 422
    assert "text/html" in bad_return.headers["content-type"]
    with application.unit_of_work.read_transaction() as tx:
        assert tx.illustrations.get_head("book-1").revision_number == 0

    # Valid HTML form -> 303 redirect preserving encoded return_*.
    ok_form = client.post(
        "/books/book-1/illustrations/settings",
        data={
            "expected_revision": "0",
            "max_per_chapter": "2",
            "return_offset": "10",
            "return_limit": "20",
            "return_section_id": "book-1-sec-0",
        },
        headers={"Accept": "text/html"},
        follow_redirects=False,
    )
    assert ok_form.status_code == 303
    assert (
        ok_form.headers["location"]
        == "/books/book-1/illustrations?offset=10&limit=20&section_id=book-1-sec-0"
    )


def test_invalid_query_params_return_reason_code_422_json_and_safe_html(
    application: Application, client: TestClient
) -> None:
    _seed_book(application)
    for url in (
        "/books/book-1/illustrations?unknown=1",
        "/books/book-1/illustrations?offset=-1",
        "/books/book-1/illustrations?limit=99",
        "/books/book-1/illustrations?limit=1&limit=2",
        "/books/book-1/illustrations?registry_kind=bogus",
    ):
        resp_json = client.get(url)
        assert resp_json.status_code == 422
        assert resp_json.json() == {"detail": {"reason_code": "invalid_request"}}

        resp_html = client.get(url, headers={"Accept": "text/html"})
        assert resp_html.status_code == 422
        assert "text/html" in resp_html.headers["content-type"]
        assert "Запрос заполнен неверно" in resp_html.text


def test_foreign_book_section_generation_404_json_and_safe_html(
    application: Application, client: TestClient
) -> None:
    doc1 = _seed_book(application, "book-1", num_sections=1)
    _seed_book(application, "book-2", num_sections=1)
    _, gen1_id = _seed_generation_with_chapters(application, "book-1", doc1)

    cases = [
        ("/books/missing/illustrations", "book_not_found"),
        ("/books/book-1/illustrations?section_id=book-2-sec-0", "section_not_found"),
        (f"/books/book-2/illustrations?generation_id={gen1_id}", "generation_not_found"),
    ]
    for url, expected_code in cases:
        resp_json = client.get(url)
        assert resp_json.status_code == 404
        assert resp_json.json() == {"detail": {"reason_code": expected_code}}

        resp_html = client.get(url, headers={"Accept": "text/html"})
        assert resp_html.status_code == 404
        assert "text/html" in resp_html.headers["content-type"]
        assert ILLUSTRATION_REASON_LABELS[expected_code] in resp_html.text


def test_foreign_origin_403_and_host_400_preserved(
    application: Application, client: TestClient
) -> None:
    _seed_book(application)
    bad_host = client.get("/books/book-1/illustrations", headers={"Host": "evil.example"})
    assert bad_host.status_code == 400

    bad_origin = client.post(
        "/books/book-1/illustrations/settings",
        json={"expected_revision": 0, "enabled": False},
        headers={"Origin": "http://evil.example"},
    )
    assert bad_origin.status_code == 403


def test_source_url_works_before_translation_and_null_after_source_change(
    application: Application, client: TestClient
) -> None:
    doc = _seed_book(application, num_sections=1)
    _seed_generation_with_chapters(application, "book-1", doc)

    # Book has NO translation run yet; source_url must still work!
    resp = client.get("/books/book-1/illustrations?section_id=book-1-sec-0")
    assert resp.status_code == 200
    body = resp.json()
    expected_url = f"/books/book-1/result#{block_anchor('book-1:blk:0')}"
    assert body["chapter_headers"][0]["source_url"] == expected_url
    assert body["selected_chapter"]["moments"][0]["source_url"] == expected_url

    # Modify source block text so anchor quote no longer matches.
    changed_doc = CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(title="Book book-1"),
        source_language="en",
        target_language="ru",
        sections=(
            DocumentSection(
                id="book-1-sec-0",
                document_id="book-1",
                title="Chapter 1",
                order=0,
                source_resource_path="Text/ch1.xhtml",
                blocks=(
                    DocumentBlock(
                        id="book-1:blk:0",
                        section_id="book-1-sec-0",
                        type=BlockType.PARAGRAPH,
                        order=0,
                        text="Completely different text.",
                        source_reference=SourceReference("Text/ch1.xhtml", "p[0]"),
                        support_status=SupportStatus.SUPPORTED,
                    ),
                ),
            ),
        ),
    )
    with application.unit_of_work.transaction() as tx:
        tx.documents.save(changed_doc)

    after = client.get("/books/book-1/illustrations?section_id=book-1-sec-0")
    assert after.status_code == 200
    assert after.json()["selected_chapter"]["moments"][0]["source_url"] is None


def test_historical_generation_after_reimport_readable_with_null_source_url(
    application: Application, client: TestClient
) -> None:
    doc = _seed_book(application, num_sections=2)
    _, gen_id = _seed_generation_with_chapters(application, "book-1", doc)

    # Revoke generation by changing settings so it becomes historical.
    client.post(
        "/books/book-1/illustrations/settings",
        json={
            "expected_revision": 1,
            "enabled": False,
            "max_per_chapter": 3,
            "style": "monochrome_ink",
            "model_reference": None,
            "model_prompts": None,
        },
    )

    # Reimport document with new section IDs.
    reimported = CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(title="Reimported"),
        source_language="en",
        target_language="ru",
        sections=(
            DocumentSection(
                id="reimp-sec-0",
                document_id="book-1",
                title="Reimported Ch 1",
                order=0,
                source_resource_path="Text/new.xhtml",
                blocks=(
                    DocumentBlock(
                        id="reimp-blk-0",
                        section_id="reimp-sec-0",
                        type=BlockType.PARAGRAPH,
                        order=0,
                        text="Brand new book content.",
                        source_reference=SourceReference("Text/new.xhtml", "p[0]"),
                        support_status=SupportStatus.SUPPORTED,
                    ),
                ),
            ),
        ),
    )
    with application.unit_of_work.transaction() as tx:
        tx.documents.save(reimported)

    resp = client.get(
        f"/books/book-1/illustrations?generation_id={gen_id}&section_id=book-1-sec-1"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_history"] is True
    assert body["state"] == "stale"
    assert [h["section_id"] for h in body["chapter_headers"]] == ["book-1-sec-0", "book-1-sec-1"]
    assert all(h["source_url"] is None for h in body["chapter_headers"])
    assert body["selected_chapter"]["section_id"] == "book-1-sec-1"
    assert body["selected_chapter"]["moments"][0]["source_url"] is None


def test_registry_read_only_json_endpoint_paginated_and_scoped(
    application: Application, client: TestClient
) -> None:
    doc = _seed_book(application, num_sections=1)
    _seed_book(application, "book-2", num_sections=1)
    _, gen_id = _seed_generation_with_chapters(application, "book-1", doc)
    with application.unit_of_work.transaction() as tx:
        tx.illustrations.upsert_registry_items(
            gen_id,
            [
                RegistryItem(
                    generation_id=gen_id,
                    kind=IllustrationRegistryKind.ENTITY,
                    item_key="e-1",
                    entity_id="ent-alice",
                    position_key=format_position_key(0, 0, 0),
                    is_back_applicable=False,
                    payload_sha256="1" * 64,
                    payload={"name_ru": "Алиса", "api_key": "must-not-leak"},
                ),
                RegistryItem(
                    generation_id=gen_id,
                    kind=IllustrationRegistryKind.FACT,
                    item_key="f-1",
                    entity_id="ent-alice",
                    position_key=format_position_key(0, 0, 1),
                    is_back_applicable=True,
                    payload_sha256="2" * 64,
                    payload={"description_ru": "Светлые волосы"},
                ),
            ],
        )

    # Even with Accept: text/html, API-04 returns JSON only.
    resp = client.get(
        "/books/book-1/illustrations/registry?kind=entity&offset=0&limit=10",
        headers={"Accept": "text/html"},
    )
    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]
    body = resp.json()
    assert body["generation_id"] == gen_id
    assert len(body["items"]) == 1
    assert "api_key" not in body["items"][0]["details"]

    # Foreign generation_id -> JSON 404 even with Accept: text/html.
    foreign = client.get(
        f"/books/book-2/illustrations/registry?generation_id={gen_id}",
        headers={"Accept": "text/html"},
    )
    assert foreign.status_code == 404
    assert "application/json" in foreign.headers["content-type"]
    assert foreign.json() == {"detail": {"reason_code": "generation_not_found"}}

    # Book without active generation -> empty registry.
    empty = client.get("/books/book-2/illustrations/registry")
    assert empty.status_code == 200
    assert empty.json()["generation_id"] is None
    assert empty.json()["items"] == []


def test_all_12_states_28_reasons_and_4_registry_kinds_have_russian_labels() -> None:
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
    assert all(val.strip() for val in ILLUSTRATION_STATE_LABELS.values())

    expected_reasons = {member.value for member in IllustrationReasonCode}
    assert set(ILLUSTRATION_REASON_LABELS.keys()) == expected_reasons
    assert len(ILLUSTRATION_REASON_LABELS) == 28
    assert all(val.strip() for val in ILLUSTRATION_REASON_LABELS.values())

    expected_kinds = {member.value for member in IllustrationRegistryKind}
    assert set(ILLUSTRATION_REGISTRY_KIND_LABELS.keys()) == expected_kinds
    assert len(ILLUSTRATION_REGISTRY_KIND_LABELS) == 4
    assert all(val.strip() for val in ILLUSTRATION_REGISTRY_KIND_LABELS.values())

    assert "monochrome_ink" in ILLUSTRATION_STYLE_LABELS
    assert templates.env.globals["illustration_state_labels"] is ILLUSTRATION_STATE_LABELS
    assert templates.env.globals["illustration_reason_labels"] is ILLUSTRATION_REASON_LABELS
    assert templates.env.globals["illustration_registry_kind_labels"] is ILLUSTRATION_REGISTRY_KIND_LABELS
    assert templates.env.globals["illustration_style_labels"] is ILLUSTRATION_STYLE_LABELS
