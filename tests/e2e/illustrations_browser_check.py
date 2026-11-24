"""Playwright browser checks for illustration prompt copying, accessibility and layout (`P7.T1`).

Run explicitly via:
    uv run --with pytest-playwright python -m playwright install chromium
    uv run --with pytest-playwright pytest tests/e2e/illustrations_browser_check.py -v
"""

from __future__ import annotations

import asyncio
import hashlib
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from playwright.sync_api import sync_playwright

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.application.ports.persistence import BookRecord
from linguary.composition import build_application
from linguary.domain import (
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    SourceReference,
    SupportStatus,
)
from linguary.domain.illustrations import (
    IllustrationGenerationState,
    IllustrationScopeSpec,
    IllustrationScopeStage,
    IllustrationSettings,
)
from linguary.web.app import create_app


class _LocalStagedFakeLLM:
    supports_staged_analysis = True

    async def generate(self, request: Any) -> Any:
        raise AssertionError(f"Browser check must not invoke LLM: {request}")


@contextmanager
def _live_illustrations_server(
    tmp_path: Path,
    *,
    book_id: str,
    moments: list[dict[str, Any]],
    generation_state: IllustrationGenerationState = IllustrationGenerationState.READY,
) -> Iterator[tuple[str, str]]:
    """Seed DB strictly before starting uvicorn on a pre-bound socket and Playwright."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    epub_path = tmp_path / f"{book_id}.epub"
    epub_path.write_bytes(b"PK\x03\x04browser-e2e-epub")
    src_sha = hashlib.sha256(epub_path.read_bytes()).hexdigest()

    app = build_application(
        settings=AppSettings(
            storage=StorageSettings(data_root=str(tmp_path / "data")),
            llm=LLMSettings(api_key="browser-key", model="gpt-5-mini", staged_analysis=True),
        ),
        llm_client=_LocalStagedFakeLLM(),
        content_plan_profile="content-plan-v3",
    )
    sec_id = f"{book_id}-sec-0"
    blk_id = f"{book_id}:b0"
    doc = CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title="Browser Illustration Novel"),
        source_language="en",
        target_language="ru",
        sections=(
            DocumentSection(
                id=sec_id,
                document_id=book_id,
                title="Chapter 1: Storm",
                order=0,
                source_resource_path="Text/ch1.xhtml",
                blocks=(
                    DocumentBlock(
                        id=blk_id,
                        section_id=sec_id,
                        type=BlockType.PARAGRAPH,
                        order=0,
                        text="Alice opened the heavy iron gate beneath the storm lantern.",
                        source_reference=SourceReference("Text/ch1.xhtml", "p[0]"),
                        support_status=SupportStatus.SUPPORTED,
                    ),
                ),
            ),
        ),
    )
    book = BookRecord(
        id=book_id,
        metadata=doc.metadata,
        source_language="en",
        target_language="ru",
        status=BookStatus.COMPLETED,
        source_epub_path=str(epub_path),
        created_at=datetime(2026, 11, 12, tzinfo=UTC),
    )
    with app.unit_of_work.transaction() as tx:
        tx.books.add(book)
        tx.documents.save(doc)
        app.book_conditions.create(tx, book, app.runtime_settings.revision)

    rev = app.illustrations.save_settings(
        book_id,
        expected_revision=0,
        settings=IllustrationSettings(
            enabled=True,
            max_per_chapter=3,
            model_reference="gpt-5-mini",
            model_prompts="gpt-5-mini",
        ),
    )
    with app.unit_of_work.transaction() as tx:
        gen, _ = tx.illustrations.create_or_get_generation(
            book_id,
            settings_id=rev.id,
            request_key=f"req-{book_id}",
            source_sha256=src_sha,
            source_authority={},
            requested_sections=(sec_id,),
        )
        tx.illustrations.update_generation_state(book_id, gen.id, state=generation_state)
        sc = tx.illustrations.upsert_scopes(
            gen.id,
            [
                IllustrationScopeSpec(
                    stage=IllustrationScopeStage.CHAPTER,
                    scope_key=sec_id,
                    section_id=sec_id,
                    section_order=0,
                    section_title="Chapter 1: Storm",
                    input_sha256="b" * 64,
                    summary={"moment_count": len(moments)},
                )
            ],
        )[0]
        if moments:
            tx.illustrations.accept_scope(
                sc.id,
                attempt_id=None,
                expected_input_sha256=sc.input_sha256,
                payload={
                    "schema_version": 1,
                    "section_id": sec_id,
                    "generation_id": gen.id,
                    "style": "monochrome_ink",
                    "moments": moments,
                },
                output_sha256="c" * 64,
                summary={"moment_count": len(moments)},
            )

    web_app = create_app(application=app, run_background_worker=False)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    port = sock.getsockname()[1]

    config = uvicorn.Config(web_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    def _run_server() -> None:
        asyncio.run(server.serve(sockets=[sock]))

    thread = threading.Thread(target=_run_server, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)

    try:
        yield f"http://127.0.0.1:{port}", sec_id
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
        sock.close()
        asyncio.run(app.aclose())


def _sample_moments(book_id: str) -> list[dict[str, Any]]:
    blk_id = f"{book_id}:b0"
    return [
        {
            "id": "m1",
            "scene_id": "s0",
            "title_ru": "Первый момент",
            "reason_ru": "Завязка главы.",
            "event_ru": "Алиса открывает железные ворота.",
            "prompt_en": 'Moment 1: "iron gate"\nLine 2 </textarea><script>alert(1)</script>\nMonochrome ink.',
            "insert_after_block_id": blk_id,
            "anchors": [{"block_id": blk_id, "start": 0, "end": 12, "quote": "Alice opened"}],
            "character_descriptions": [],
            "location_descriptions": [],
            "assumptions_ru": [],
        },
        {
            "id": "m2",
            "scene_id": "s0",
            "title_ru": "Второй момент",
            "reason_ru": "Кульминация.",
            "event_ru": "Фонарь раскачивается на ветру.",
            "prompt_en": "Moment 2: swaying storm lantern, crisp black ink crosshatching.\nSecond line.",
            "insert_after_block_id": blk_id,
            "anchors": [{"block_id": blk_id, "start": 13, "end": 32, "quote": "the heavy iron gate"}],
            "character_descriptions": [],
            "location_descriptions": [],
            "assumptions_ru": [],
        },
    ]


def test_copy_prompt_exact_text_with_quotes_multiline_and_textarea_breakout_among_multiple_moments(
    tmp_path: Path,
) -> None:
    moments = _sample_moments("book-copy")
    with _live_illustrations_server(tmp_path, book_id="book-copy", moments=moments) as (base_url, sec_id):
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context()
            context.grant_permissions(["clipboard-read", "clipboard-write"])
            page = context.new_page()
            page.goto(f"{base_url}/books/book-copy/illustrations?section_id={sec_id}")

            btn1 = page.locator('button[data-prompt-id="prompt-m1"]')
            btn1.click()
            page.wait_for_function(
                "() => document.getElementById('copy-status-m1').textContent.includes('скопирован')"
            )
            copied = page.evaluate("() => navigator.clipboard.readText()")
            assert copied == moments[0]["prompt_en"]
            assert page.locator("#copy-status-m2").text_content() == ""

            # Copy second moment and confirm exact isolation
            page.locator('button[data-prompt-id="prompt-m2"]').click()
            page.wait_for_function(
                "() => document.getElementById('copy-status-m2').textContent.includes('скопирован')"
            )
            copied2 = page.evaluate("() => navigator.clipboard.readText()")
            assert copied2 == moments[1]["prompt_en"]
            browser.close()


def test_copy_failure_and_missing_clipboard_selects_readonly_text(tmp_path: Path) -> None:
    moments = _sample_moments("book-fb")
    with _live_illustrations_server(tmp_path, book_id="book-fb", moments=moments) as (base_url, sec_id):
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for init_script in (
                "Object.defineProperty(navigator, 'clipboard', { value: undefined, configurable: true });",
                "Object.defineProperty(navigator, 'clipboard', { value: { writeText: () => Promise.reject(new Error('denied')) }, configurable: true });",
            ):
                context = browser.new_context()
                page = context.new_page()
                page.add_init_script(init_script)
                page.goto(f"{base_url}/books/book-fb/illustrations?section_id={sec_id}")
                page.locator('button[data-prompt-id="prompt-m1"]').click()
                page.wait_for_function(
                    "() => document.getElementById('copy-status-m1').textContent.includes('выделен')"
                )
                active_id = page.evaluate("() => document.activeElement && document.activeElement.id")
                assert active_id == "prompt-m1"
                selected_len = page.evaluate(
                    "() => { const el = document.getElementById('prompt-m1'); return el.selectionEnd - el.selectionStart; }"
                )
                assert selected_len == len(moments[0]["prompt_en"])
                context.close()
            browser.close()


def test_illustrations_keyboard_and_no_javascript(tmp_path: Path) -> None:
    moments = _sample_moments("book-nojs")
    with _live_illustrations_server(tmp_path, book_id="book-nojs", moments=moments) as (base_url, sec_id):
        with sync_playwright() as p:
            browser = p.chromium.launch()
            # 1) No-JS context: prompt text is verbatim in readonly textarea
            ctx_nojs = browser.new_context(java_script_enabled=False)
            page_nojs = ctx_nojs.new_page()
            page_nojs.goto(f"{base_url}/books/book-nojs/illustrations?section_id={sec_id}")
            assert page_nojs.locator("#prompt-m1").input_value() == moments[0]["prompt_en"]
            assert page_nojs.locator("#prompt-m1").get_attribute("readonly") is not None
            ctx_nojs.close()

            # 2) Keyboard activation (focus + Enter) triggers copy handler
            ctx_kb = browser.new_context()
            ctx_kb.grant_permissions(["clipboard-read", "clipboard-write"])
            page_kb = ctx_kb.new_page()
            page_kb.goto(f"{base_url}/books/book-nojs/illustrations?section_id={sec_id}")
            btn = page_kb.locator('button[data-prompt-id="prompt-m1"]')
            btn.focus()
            page_kb.keyboard.press("Enter")
            page_kb.wait_for_function(
                "() => document.getElementById('copy-status-m1').textContent.includes('скопирован')"
            )
            ctx_kb.close()
            browser.close()


def test_illustrations_no_background_refresh_while_active_and_ready(tmp_path: Path) -> None:
    for state, book_id in (
        (IllustrationGenerationState.RUNNING, "book-poll-run"),
        (IllustrationGenerationState.READY, "book-poll-ready"),
    ):
        moments = _sample_moments(book_id) if state is IllustrationGenerationState.READY else []
        with _live_illustrations_server(
            tmp_path / book_id, book_id=book_id, moments=moments, generation_state=state
        ) as (base_url, sec_id):
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(f"{base_url}/books/{book_id}/illustrations?section_id={sec_id}")
                page.wait_for_load_state("networkidle")
                extra_requests: list[str] = []
                page.on("request", lambda req, bag=extra_requests: bag.append(req.url))
                page.wait_for_timeout(3200)
                assert extra_requests == [], f"Unexpected background requests: {extra_requests}"
                browser.close()


def test_illustrations_long_prompt_small_viewport_no_overflow(tmp_path: Path) -> None:
    long_token = "MonochromeInkIllustrationCrosshatching" * 15
    moments = _sample_moments("book-vp")
    moments[0]["prompt_en"] = f"{long_token} {'wide scene details ' * 40}"
    moments[0]["character_descriptions"] = [
        {
            "entity_id": "ent-alice",
            "name_ru": "Алиса",
            "description_ru": "Длинное описание персонажа " + ("без переносов " * 10),
            "description_en": long_token,
            "fact_ids": ["f1"],
            "assumptions_ru": ["Допущение по эпохе " + long_token[:60]],
        }
    ]
    moments[0]["location_descriptions"] = [
        {
            "entity_id": "loc-gate",
            "name_ru": "Железные ворота",
            "description_ru": "Старые кованые ворота в ливень.",
            "description_en": "Wrought-iron gate under heavy rain " + long_token[:80],
            "fact_ids": [],
            "assumptions_ru": [],
        }
    ]
    with _live_illustrations_server(tmp_path, book_id="book-vp", moments=moments) as (base_url, sec_id):
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for width in (360, 820):
                context = browser.new_context(viewport={"width": width, "height": 800})
                page = context.new_page()
                page.goto(f"{base_url}/books/book-vp/illustrations?section_id={sec_id}")
                overflow = page.evaluate(
                    "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
                )
                assert overflow <= 0, f"Horizontal overflow detected at {width}px: {overflow}px"
                context.close()
            browser.close()
