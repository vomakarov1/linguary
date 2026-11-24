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


# ── P4.T1 API-03 tests ────────────────────────────────────────────────────────


def _seed_staged_ready_book(
    app: Application,
    tmp_path: Path,
    *,
    book_id: str = "book-p4",
    num_sections: int = 2,
    include_front_matter: bool = False,
) -> tuple[CanonicalDocument, str, str]:
    from hashlib import sha256
    from linguary.application.analysis_pass_cache import encode_pass
    from linguary.application.context_builder import ContextBuilder
    from linguary.application.illustration_source import _pass_model_dependencies
    from linguary.application.llm_dto import NarrativeAnalysisResponse, QuotedSceneResponse
    from linguary.application.token_budget import narrative_max_fragment_blocks_for
    from linguary.composition.book_runtime import conditions_payload
    from linguary.domain import AnalysisPassKind, AnalysisPassResult
    from linguary.domain.book_conditions import BookConditions
    from linguary.domain.name_translation import NameTranslationPolicy, NameTranslationStrategy

    epub_file = tmp_path / f"{book_id}.epub"
    epub_bytes = f"epub-bytes-{book_id}".encode()
    epub_file.write_bytes(epub_bytes)
    source_sha = sha256(epub_bytes).hexdigest()
    conditions_id = str(uuid4())

    sections: list[DocumentSection] = []
    order = 0
    if include_front_matter:
        sections.append(
            DocumentSection(
                id=f"{book_id}-fm",
                document_id=book_id,
                title="Title Page",
                order=order,
                source_resource_path="Text/title.xhtml",
                blocks=(
                    DocumentBlock(
                        id=f"{book_id}:fm:0",
                        section_id=f"{book_id}-fm",
                        type=BlockType.PARAGRAPH,
                        order=0,
                        text="Published 2026.",
                        source_reference=SourceReference("Text/title.xhtml", "p[0]"),
                        support_status=SupportStatus.SUPPORTED,
                    ),
                ),
            )
        )
        order += 1

    content_sections: list[DocumentSection] = []
    for idx in range(num_sections):
        sec_id = f"{book_id}-sec-{idx}"
        blk_id = f"{book_id}:blk:{idx}"
        sec = DocumentSection(
            id=sec_id,
            document_id=book_id,
            title=f"Chapter {idx + 1}",
            order=order,
            source_resource_path=f"Text/ch{idx + 1}.xhtml",
            blocks=(
                DocumentBlock(
                    id=blk_id,
                    section_id=sec_id,
                    type=BlockType.PARAGRAPH,
                    order=0,
                    text=f"Alice walked through hall {idx + 1} with a silver lantern.",
                    source_reference=SourceReference(f"Text/ch{idx + 1}.xhtml", "p[0]"),
                    support_status=SupportStatus.SUPPORTED,
                ),
            ),
        )
        sections.append(sec)
        content_sections.append(sec)
        order += 1

    doc = CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title=f"Book {book_id}"),
        source_language="en",
        target_language="ru",
        sections=tuple(sections),
    )
    book = BookRecord(
        id=book_id,
        metadata=doc.metadata,
        source_language="en",
        target_language="ru",
        status=BookStatus.GLOSSARY_REVIEW,
        source_epub_path=str(epub_file),
        created_at=datetime(2026, 11, 1, tzinfo=UTC),
    )
    cond_payload = conditions_payload(book, app.settings, revision=1, profile="content-plan-v3")
    cond_payload["source_sha256"] = source_sha
    cond_payload["staged_analysis"] = True
    cond_payload["llm_parameters"]["staged_analysis"] = True
    cond_payload["checks"]["sequential_analysis"] = False
    prompts_version = int(cond_payload["versions"]["prompts"])
    strategy = app.settings.reading.name_translation_strategy
    ctx_builder = ContextBuilder(
        content_plan_profile="content-plan-v3",
        name_translation_policy=NameTranslationPolicy(
            version=1, strategy=NameTranslationStrategy(strategy)
        ),
        reader_level=app.settings.reading.normalized_level,
    )
    model_deps = _pass_model_dependencies(cond_payload)
    max_frag = narrative_max_fragment_blocks_for(cond_payload["llm_parameters"])

    with app.unit_of_work.transaction() as tx:
        tx.books.add(book)
        tx.documents.save(doc)
        tx.work.add_conditions(
            BookConditions(
                id=conditions_id,
                book_id=book_id,
                revision=1,
                payload=cond_payload,
                created_at=datetime(2026, 11, 1, tzinfo=UTC),
            )
        )
        for sec in content_sections:
            narrative = NarrativeAnalysisResponse(
                translation_instructions="Сохранять спокойный ритм повествования.",
                chapter_summary=f"Alice walks quietly through {sec.title} holding a lantern.",
                glossary_candidates=(),
                context_facts=(),
                scenes=(
                    QuotedSceneResponse(
                        start_block_id=sec.blocks[0].id,
                        end_block_id=sec.blocks[-1].id,
                        summary=f"Сцена в {sec.title}.",
                        location="Зал",
                        participants=(),
                        visual_facts=(),
                        visual_assumptions=(),
                        continues_previous_scene=False,
                    ),
                ),
            )
            encoded = encode_pass(narrative)
            encoded["input_suggestions"] = []
            encoded["input_context"] = {
                "schema_version": 1,
                "sequential_analysis": False,
                "parallel_batch": False,
                "seed_active": False,
                "user_decisions": [],
                "nonbinding_suggestions": [],
                "prompts_version": prompts_version,
            }
            input_sha = ctx_builder.narrative_input_sha256(
                doc,
                sec,
                (),
                (),
                (),
                model_dependencies=model_deps,
                max_fragment_blocks=max_frag,
            )
            tx.analyses.save_pass(
                AnalysisPassResult(
                    id=f"pass-{sec.id}",
                    book_id=book_id,
                    section_id=sec.id,
                    pass_kind=AnalysisPassKind.NARRATIVE,
                    scope_key="",
                    input_sha256=input_sha,
                    output_sha256=canonical_sha(encoded),
                    conditions_id=conditions_id,
                    attempt_id=None,
                    payload=encoded,
                )
            )
        tx.illustrations.save_settings(
            book_id,
            expected_revision=0,
            settings=IllustrationSettings(
                enabled=True,
                max_per_chapter=3,
                model_reference="gpt-5-mini",
                model_prompts="gpt-5-mini",
            ),
        )
    return doc, conditions_id, source_sha


def _build_ready_app(tmp_path: Path) -> tuple[Application, TestClient]:
    settings = AppSettings(
        storage=StorageSettings(data_root=str(tmp_path / "data")),
        llm=LLMSettings(
            provider="openai_compatible",
            base_url="https://api.openai.com/v1",
            model="gpt-5-mini",
            api_key="test-key",
            staged_analysis=True,
        ),
        checks=CheckSettings(sequential_analysis=False),
        server=ServerSettings(host="127.0.0.1", port=8000),
    )
    from linguary.web.dependencies import APPLICATION_STATE_ATTRIBUTE

    app = build_application(settings=settings)
    web_app = create_app(application=app, run_background_worker=False)
    setattr(web_app.state, APPLICATION_STATE_ATTRIBUTE, app)
    client = TestClient(web_app, base_url="http://127.0.0.1:8000")
    return app, client


def test_prepare_replay_one_generation_one_task(tmp_path: Path) -> None:
    app, client = _build_ready_app(tmp_path)
    try:
        _seed_staged_ready_book(app, tmp_path, book_id="book-replay", num_sections=1)
        r1 = client.post("/books/book-replay/illustrations/prepare", json={"expected_revision": 1})
        assert r1.status_code == 202
        gen_id = r1.json()["active_generation_id"]
        assert gen_id is not None

        r2 = client.post("/books/book-replay/illustrations/prepare", json={"expected_revision": 1})
        assert r2.status_code == 202
        assert r2.json()["active_generation_id"] == gen_id

        with app.unit_of_work.read_transaction() as tx:
            tasks = [
                t
                for t in tx.tasks.list_for_book("book-replay")
                if t.task_type is TaskType.PREPARE_ILLUSTRATION_REFERENCE
            ]
            assert len(tasks) == 1
            assert tasks[0].operation_id == f"{gen_id}:a1"
            assert tasks[0].policy_version == "illustration-v1"
    finally:
        import asyncio

        asyncio.run(app.aclose())


def test_prepare_after_revocation_under_same_request_key_creates_new_generation(tmp_path: Path) -> None:
    app, client = _build_ready_app(tmp_path)
    try:
        _seed_staged_ready_book(app, tmp_path, book_id="book-rev", num_sections=1)
        r1 = client.post("/books/book-rev/illustrations/prepare", json={"expected_revision": 1})
        assert r1.status_code == 202
        gen1 = r1.json()["active_generation_id"]
        assert gen1 is not None

        with app.unit_of_work.transaction() as tx:
            tx.illustrations.revoke_generation("book-rev", gen1, reason_code="source_changed")

        r2 = client.post("/books/book-rev/illustrations/prepare", json={"expected_revision": 1})
        assert r2.status_code == 202
        gen2 = r2.json()["active_generation_id"]
        assert gen2 is not None and gen2 != gen1
    finally:
        import asyncio

        asyncio.run(app.aclose())


def test_section_prepare_upgrades_to_full_book_preserving_ready_chapter(tmp_path: Path) -> None:
    app, client = _build_ready_app(tmp_path)
    try:
        _seed_staged_ready_book(app, tmp_path, book_id="book-upg", num_sections=2)
        r1 = client.post(
            "/books/book-upg/illustrations/prepare",
            json={"expected_revision": 1, "section_id": "book-upg-sec-0"},
        )
        assert r1.status_code == 202
        gen_id = r1.json()["active_generation_id"]

        # Simulate registry sealed and chapter 0 already ACCEPTED.
        with app.unit_of_work.transaction() as tx:
            tx.illustrations.seal_registry("book-upg", gen_id, registry_sha256="e" * 64)
            snap = app.illustrations._source_snapshot_fn(tx, "book-upg")  # type: ignore[misc]
            gen = tx.illustrations.get_generation("book-upg", gen_id)
            assert gen is not None
            ch0_specs = tx.illustrations.upsert_scopes(
                gen_id,
                [
                    IllustrationScopeSpec(
                        stage=IllustrationScopeStage.CHAPTER,
                        scope_key="chapter:book-upg-sec-0",
                        input_sha256="f" * 64,
                        section_id="book-upg-sec-0",
                        section_order=0,
                        section_title="Chapter 1",
                        state=IllustrationScopeState.ACCEPTED,
                        payload={"schema_version": 1, "section_id": "book-upg-sec-0", "generation_id": gen_id, "style": "monochrome_ink", "moments": [], "zero_reason_ru": "Тихая глава."},
                        output_sha256="1" * 64,
                        summary={"moment_count": 0},
                    )
                ],
            )
            del snap, ch0_specs

        # Expand to full book -> same generation, requested_sections expanded, chapter 0 stays ACCEPTED!
        r2 = client.post(
            "/books/book-upg/illustrations/prepare",
            json={"expected_revision": 1, "expected_generation_id": gen_id},
        )
        assert r2.status_code == 202
        assert r2.json()["active_generation_id"] == gen_id
        with app.unit_of_work.read_transaction() as tx:
            gen = tx.illustrations.get_generation("book-upg", gen_id)
            assert gen is not None
            assert set(gen.requested_sections) == {"book-upg-sec-0", "book-upg-sec-1"}
            ch0 = tx.illustrations.list_scopes(
                gen_id, stage=IllustrationScopeStage.CHAPTER, section_id="book-upg-sec-0"
            )[0]
            assert ch0.state is IllustrationScopeState.ACCEPTED
    finally:
        import asyncio

        asyncio.run(app.aclose())


def test_prepare_before_initial_conditions_returns_409_source_incompatible(tmp_path: Path) -> None:
    app, client = _build_ready_app(tmp_path)
    try:
        _seed_book(app, "book-nocond", num_sections=1)
        with app.unit_of_work.transaction() as tx:
            tx.illustrations.save_settings(
                "book-nocond",
                expected_revision=0,
                settings=IllustrationSettings(
                    enabled=True,
                    max_per_chapter=3,
                    model_reference="gpt-5-mini",
                    model_prompts="gpt-5-mini",
                ),
            )
        resp = client.post(
            "/books/book-nocond/illustrations/prepare", json={"expected_revision": 1}
        )
        assert resp.status_code == 409
        assert resp.json() == {"detail": {"reason_code": "source_incompatible"}}
    finally:
        import asyncio

        asyncio.run(app.aclose())


def test_prepare_on_non_content_section_returns_409(tmp_path: Path) -> None:
    app, client = _build_ready_app(tmp_path)
    try:
        _seed_staged_ready_book(
            app, tmp_path, book_id="book-fm", num_sections=1, include_front_matter=True
        )
        resp = client.post(
            "/books/book-fm/illustrations/prepare",
            json={"expected_revision": 1, "section_id": "book-fm-fm"},
        )
        assert resp.status_code == 409
        assert resp.json() == {"detail": {"reason_code": "non_content_section"}}
    finally:
        import asyncio

        asyncio.run(app.aclose())


def test_prepare_supersedes_stale_generation_and_rejects_force_new_outside_limit_409(
    tmp_path: Path,
) -> None:
    app, client = _build_ready_app(tmp_path)
    try:
        _seed_staged_ready_book(app, tmp_path, book_id="book-stale", num_sections=1)
        r1 = client.post("/books/book-stale/illustrations/prepare", json={"expected_revision": 1})
        assert r1.status_code == 202
        gen1 = r1.json()["active_generation_id"]

        # force_new_generation on a non-limit-exhausted generation -> 409 generation_conflict!
        bad_force = client.post(
            "/books/book-stale/illustrations/prepare",
            json={
                "expected_revision": 1,
                "expected_generation_id": gen1,
                "force_new_generation": True,
            },
        )
        assert bad_force.status_code == 409
        assert bad_force.json() == {"detail": {"reason_code": "generation_conflict"}}

        # Mark gen1 failed with call_limit_exhausted -> force_new_generation SUCCEEDS (202) and creates gen2!
        with app.unit_of_work.transaction() as tx:
            tx.illustrations.update_generation_state(
                "book-stale",
                gen1,
                state=IllustrationGenerationState.FAILED,
                reason_code=IllustrationReasonCode.CALL_LIMIT_EXHAUSTED.value,
            )
        ok_force = client.post(
            "/books/book-stale/illustrations/prepare",
            json={
                "expected_revision": 1,
                "expected_generation_id": gen1,
                "force_new_generation": True,
            },
        )
        assert ok_force.status_code == 202
        gen2 = ok_force.json()["active_generation_id"]
        assert gen2 != gen1

        # Simulate stale generation (source_sha mismatch against stored manifest) -> prepare revokes stale & creates gen3!
        with app.unit_of_work.transaction() as tx:
            from linguary.adapters.db.illustration_models import IllustrationGenerationRow

            row = tx.illustrations._session.get(IllustrationGenerationRow, gen2)
            assert row is not None
            row.source_sha256 = "0" * 64
        r3 = client.post(
            "/books/book-stale/illustrations/prepare",
            json={"expected_revision": 1, "expected_generation_id": gen2},
        )
        assert r3.status_code == 202
        gen3 = r3.json()["active_generation_id"]
        assert gen3 != gen2
        with app.unit_of_work.read_transaction() as tx:
            old = tx.illustrations.get_generation("book-stale", gen2)
            assert old is not None and old.state is IllustrationGenerationState.REVOKED
    finally:
        import asyncio

        asyncio.run(app.aclose())


def test_prepare_and_retry_while_book_paused_return_202_paused(tmp_path: Path) -> None:
    app, client = _build_ready_app(tmp_path)
    try:
        _, cond_id, _ = _seed_staged_ready_book(app, tmp_path, book_id="book-pause", num_sections=1)
        with app.unit_of_work.transaction() as tx:
            tx.work.set_book_paused("book-pause", cond_id, True)

        r_prep = client.post(
            "/books/book-pause/illustrations/prepare", json={"expected_revision": 1}
        )
        assert r_prep.status_code == 202
        assert r_prep.json()["state"] == "paused"
        assert r_prep.json()["reason_code"] == "book_paused"
        gen_id = r_prep.json()["active_generation_id"]

        # Fail the extract scope with retryable provider_error, then retry while paused -> 202 paused!
        with app.unit_of_work.transaction() as tx:
            scopes = tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.EXTRACT)
            tx.illustrations.fail_scope(
                scopes[0].id,
                attempt_id=None,
                reason_code=IllustrationReasonCode.PROVIDER_ERROR.value,
            )
            tx.illustrations.update_generation_state(
                "book-pause",
                gen_id,
                state=IllustrationGenerationState.FAILED,
                reason_code=IllustrationReasonCode.PROVIDER_ERROR.value,
            )

        r_retry = client.post(
            "/books/book-pause/illustrations/retry",
            json={"expected_revision": 1, "expected_generation_id": gen_id},
        )
        assert r_retry.status_code == 202
        assert r_retry.json()["state"] == "paused"
        assert r_retry.json()["reason_code"] == "book_paused"
    finally:
        import asyncio

        asyncio.run(app.aclose())


def test_retry_only_failed_scopes_and_rejects_nothing_to_retry_409(tmp_path: Path) -> None:
    app, client = _build_ready_app(tmp_path)
    try:
        _seed_staged_ready_book(app, tmp_path, book_id="book-retry", num_sections=1)
        r_prep = client.post(
            "/books/book-retry/illustrations/prepare", json={"expected_revision": 1}
        )
        gen_id = r_prep.json()["active_generation_id"]

        # Nothing failed yet -> 409 nothing_to_retry!
        r_none = client.post(
            "/books/book-retry/illustrations/retry",
            json={"expected_revision": 1, "expected_generation_id": gen_id},
        )
        assert r_none.status_code == 409
        assert r_none.json() == {"detail": {"reason_code": "nothing_to_retry"}}

        # Deterministic failure atomic_input_too_large -> not retryable -> 409 nothing_to_retry!
        with app.unit_of_work.transaction() as tx:
            sc = tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.EXTRACT)[0]
            tx.illustrations.fail_scope(
                sc.id,
                attempt_id=None,
                reason_code=IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value,
            )
        r_det = client.post(
            "/books/book-retry/illustrations/retry",
            json={"expected_revision": 1, "expected_generation_id": gen_id},
        )
        assert r_det.status_code == 409
        assert r_det.json() == {"detail": {"reason_code": "nothing_to_retry"}}
    finally:
        import asyncio

        asyncio.run(app.aclose())


def test_retry_after_interrupted_vs_call_limit_exhausted_409(tmp_path: Path) -> None:
    app, client = _build_ready_app(tmp_path)
    try:
        _seed_staged_ready_book(app, tmp_path, book_id="book-rcall", num_sections=1)
        r_prep = client.post(
            "/books/book-rcall/illustrations/prepare", json={"expected_revision": 1}
        )
        gen_id = r_prep.json()["active_generation_id"]

        # Interrupted scope IS retryable (202)!
        with app.unit_of_work.transaction() as tx:
            sc = tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.EXTRACT)[0]
            tx.illustrations.fail_scope(
                sc.id, attempt_id=None, reason_code=IllustrationReasonCode.INTERRUPTED.value
            )
        r_int = client.post(
            "/books/book-rcall/illustrations/retry",
            json={"expected_revision": 1, "expected_generation_id": gen_id},
        )
        assert r_int.status_code == 202

        # call_limit_exhausted -> 409 call_limit_exhausted!
        with app.unit_of_work.transaction() as tx:
            sc = tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.EXTRACT)[0]
            tx.illustrations.fail_scope(
                sc.id,
                attempt_id=None,
                reason_code=IllustrationReasonCode.CALL_LIMIT_EXHAUSTED.value,
            )
        r_lim = client.post(
            "/books/book-rcall/illustrations/retry",
            json={"expected_revision": 1, "expected_generation_id": gen_id},
        )
        assert r_lim.status_code == 409
        assert r_lim.json() == {"detail": {"reason_code": "call_limit_exhausted"}}
    finally:
        import asyncio

        asyncio.run(app.aclose())


def test_http_json_202_html_303_preserving_return_params_and_forbidding_return_in_json(
    tmp_path: Path,
) -> None:
    app, client = _build_ready_app(tmp_path)
    try:
        _seed_staged_ready_book(app, tmp_path, book_id="book-http", num_sections=1)
        # return_* forbidden in JSON -> 422 invalid_request!
        bad_json = client.post(
            "/books/book-http/illustrations/prepare",
            json={"expected_revision": 1, "return_offset": 0},
        )
        assert bad_json.status_code == 422
        assert bad_json.json() == {"detail": {"reason_code": "invalid_request"}}

        # HTML form -> 303 preserving return_*!
        html_ok = client.post(
            "/books/book-http/illustrations/prepare",
            data={
                "expected_revision": "1",
                "section_id": "",
                "expected_generation_id": "",
                "return_offset": "10",
                "return_limit": "20",
                "return_section_id": "book-http-sec-0",
            },
            headers={"Accept": "text/html"},
            follow_redirects=False,
        )
        assert html_ok.status_code == 303
        assert (
            html_ok.headers["location"]
            == "/books/book-http/illustrations?offset=10&limit=20&section_id=book-http-sec-0"
        )
    finally:
        import asyncio

        asyncio.run(app.aclose())
