"""CLI integration tests for `linguary illustrations {settings|prepare|retry|status|show|registry}` (`P4.T1`, `CLI-01`)."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, StorageSettings
from linguary.application.analysis_pass_cache import encode_pass
from linguary.application.context_builder import ContextBuilder
from linguary.application.illustration_source import _pass_model_dependencies
from linguary.application.llm_dto import NarrativeAnalysisResponse, QuotedSceneResponse
from linguary.application.ports.persistence import BookRecord
from linguary.application.token_budget import narrative_max_fragment_blocks_for
from linguary.cli.app import main
from linguary.composition import build_application
from linguary.composition.book_runtime import conditions_payload
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
from linguary.domain.book_conditions import BookConditions
from linguary.domain.content_plan import canonical_sha
from linguary.domain.illustrations import (
    TIMELESS_POSITION_KEY,
    IllustrationReasonCode,
    IllustrationRegistryKind,
    IllustrationScopeStage,
    RegistryItem,
)
from linguary.domain.name_translation import NameTranslationPolicy, NameTranslationStrategy
from linguary.web.routes.result import block_anchor


def _write_config(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"""
[storage]
data_root = "{data_root.as_posix()}"

[llm]
provider = "openai_compatible"
base_url = "https://api.openai.com/v1"
model = "gpt-5-mini"
api_key = "sk-test-secret-key"
staged_analysis = true

[checks]
sequential_analysis = false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return cfg, data_root


def _seed_cli_book(tmp_path: Path, data_root: Path, book_id: str = "book-cli") -> str:
    settings = AppSettings(
        storage=StorageSettings(data_root=str(data_root)),
        llm=LLMSettings(
            provider="openai_compatible",
            base_url="https://api.openai.com/v1",
            model="gpt-5-mini",
            api_key="sk-test-secret-key",
            staged_analysis=True,
        ),
        checks=CheckSettings(sequential_analysis=False),
    )
    app = build_application(settings=settings)
    try:
        epub_path = tmp_path / f"{book_id}.epub"
        raw = b"cli-epub-content"
        epub_path.write_bytes(raw)
        digest = sha256(raw).hexdigest()
        cond_id = str(uuid4())
        sec_id = f"{book_id}-sec-0"
        blk_id = f"{book_id}:blk:0"
        sec = DocumentSection(
            id=sec_id,
            document_id=book_id,
            title="Chapter 1",
            order=0,
            source_resource_path="Text/ch1.xhtml",
            blocks=(
                DocumentBlock(
                    id=blk_id,
                    section_id=sec_id,
                    type=BlockType.PARAGRAPH,
                    order=0,
                    text="Alice stepped into the candlelit hall.",
                    source_reference=SourceReference("Text/ch1.xhtml", "p[0]"),
                    support_status=SupportStatus.SUPPORTED,
                ),
            ),
        )
        doc = CanonicalDocument(
            id=book_id,
            metadata=BookMetadata(title="CLI Novel"),
            source_language="en",
            target_language="ru",
            sections=(sec,),
        )
        book = BookRecord(
            id=book_id,
            metadata=doc.metadata,
            source_language="en",
            target_language="ru",
            status=BookStatus.GLOSSARY_REVIEW,
            source_epub_path=str(epub_path),
            created_at=datetime(2026, 11, 1, tzinfo=UTC),
        )
        cond_payload = conditions_payload(book, app.settings, revision=1, profile="content-plan-v3")
        cond_payload["source_sha256"] = digest
        prompts_version = int(cond_payload["versions"]["prompts"])
        ctx_builder = ContextBuilder(
            content_plan_profile="content-plan-v3",
            name_translation_policy=NameTranslationPolicy(
                version=1, strategy=NameTranslationStrategy(app.settings.reading.name_translation_strategy)
            ),
            reader_level=app.settings.reading.normalized_level,
        )
        with app.unit_of_work.transaction() as tx:
            tx.books.add(book)
            tx.documents.save(doc)
            tx.work.add_conditions(
                BookConditions(
                    id=cond_id,
                    book_id=book_id,
                    revision=1,
                    payload=cond_payload,
                    created_at=datetime(2026, 11, 1, tzinfo=UTC),
                )
            )
            narrative = NarrativeAnalysisResponse(
                translation_instructions="Сохранять спокойный ритм.",
                chapter_summary="Alice enters the candlelit stone hall alone.",
                glossary_candidates=(),
                context_facts=(),
                scenes=(
                    QuotedSceneResponse(
                        start_block_id=blk_id,
                        end_block_id=blk_id,
                        summary="Сцена в зале.",
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
                model_dependencies=_pass_model_dependencies(cond_payload),
                max_fragment_blocks=narrative_max_fragment_blocks_for(
                    cond_payload["llm_parameters"]
                ),
            )
            tx.analyses.save_pass(
                AnalysisPassResult(
                    id=f"pass-{sec_id}",
                    book_id=book_id,
                    section_id=sec_id,
                    pass_kind=AnalysisPassKind.NARRATIVE,
                    scope_key="",
                    input_sha256=input_sha,
                    output_sha256=canonical_sha(encoded),
                    conditions_id=cond_id,
                    attempt_id=None,
                    payload=encoded,
                )
            )
        return blk_id
    finally:
        asyncio.run(app.aclose())


def test_cli_illustrations_settings_partial_flags_prepare_retry_status_show_registry_json_dto_source_url_and_safe_errors(
    tmp_path: Path, capsys: Any
) -> None:
    cfg, data_root = _write_config(tmp_path)
    blk_id = _seed_cli_book(tmp_path, data_root, "book-cli")

    # 1. Partial settings updates: enable + set models
    rc = main(
        [
            "--config",
            str(cfg),
            "illustrations",
            "settings",
            "book-cli",
            "--expected-revision",
            "0",
            "--enable",
            "--max-per-chapter",
            "2",
            "--model-reference",
            "gpt-5-mini",
            "--model-prompts",
            "gpt-5-mini",
            "--json",
        ]
    )
    assert rc == 0
    out1 = json.loads(capsys.readouterr().out)
    assert out1["settings_revision"] == 1
    assert out1["settings"]["enabled"] is True
    assert out1["settings"]["max_per_chapter"] == 2
    assert out1["settings"]["model_reference"] == "gpt-5-mini"

    # 2. Partial update changing ONLY max-per-chapter preserves enabled and models!
    rc = main(
        [
            "--config",
            str(cfg),
            "illustrations",
            "settings",
            "book-cli",
            "--expected-revision",
            "1",
            "--max-per-chapter",
            "3",
            "--json",
        ]
    )
    assert rc == 0
    out2 = json.loads(capsys.readouterr().out)
    assert out2["settings_revision"] == 2
    assert out2["settings"]["enabled"] is True
    assert out2["settings"]["max_per_chapter"] == 3
    assert out2["settings"]["model_reference"] == "gpt-5-mini"
    assert out2["settings"]["model_prompts"] == "gpt-5-mini"

    # 3. Prepare (auto-reading current revision when --expected-revision omitted)
    rc = main(["--config", str(cfg), "illustrations", "prepare", "book-cli", "--json"])
    assert rc == 0
    out_prep = json.loads(capsys.readouterr().out)
    gen_id = out_prep["active_generation_id"]
    assert gen_id is not None
    assert out_prep["chapter_headers"][0]["source_url"] == f"/books/book-cli/result#{block_anchor(blk_id)}"

    # 4. Status + Show
    rc = main(["--config", str(cfg), "illustrations", "status", "book-cli", "--json"])
    assert rc == 0
    out_status = json.loads(capsys.readouterr().out)
    assert out_status["active_generation_id"] == gen_id

    rc = main(
        ["--config", str(cfg), "illustrations", "show", "book-cli", "book-cli-sec-0", "--json"]
    )
    assert rc == 0
    out_show = json.loads(capsys.readouterr().out)
    assert out_show["selected_chapter"]["section_id"] == "book-cli-sec-0"

    # 5. Registry (seed 1 entity and read via CLI --json)
    app = build_application(
        settings=AppSettings(
            storage=StorageSettings(data_root=str(data_root)),
            llm=LLMSettings(api_key="sk-test-secret-key"),
        )
    )
    try:
        with app.unit_of_work.transaction() as tx:
            ent_payload = {
                "entity_id": "ent-alice",
                "entity_kind": "person",
                "canonical_name": "Alice",
                "aliases": ["Alice"],
            }
            tx.illustrations.upsert_registry_items(
                gen_id,
                [
                    RegistryItem(
                        generation_id=gen_id,
                        kind=IllustrationRegistryKind.ENTITY,
                        item_key="ent-alice",
                        entity_id="ent-alice",
                        position_key=TIMELESS_POSITION_KEY,
                        is_back_applicable=False,
                        payload_sha256=canonical_sha(ent_payload),
                        payload=ent_payload,
                    )
                ],
            )
            # Also fail extract scope so we can test retry!
            sc = tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.EXTRACT)[0]
            tx.illustrations.fail_scope(
                sc.id, attempt_id=None, reason_code=IllustrationReasonCode.PROVIDER_ERROR.value
            )
    finally:
        asyncio.run(app.aclose())

    rc = main(
        [
            "--config",
            str(cfg),
            "illustrations",
            "registry",
            "book-cli",
            "--kind",
            "entity",
            "--json",
        ]
    )
    assert rc == 0
    out_reg = json.loads(capsys.readouterr().out)
    assert out_reg["generation_id"] == gen_id
    assert len(out_reg["items"]) == 1
    assert out_reg["items"][0]["entity_id"] == "ent-alice"

    # 6. Retry via CLI
    rc = main(
        [
            "--config",
            str(cfg),
            "illustrations",
            "retry",
            "book-cli",
            "--expected-generation",
            gen_id,
            "--json",
        ]
    )
    assert rc == 0
    out_retry = json.loads(capsys.readouterr().out)
    assert out_retry["active_generation_id"] == gen_id

    # 7. Safe error handling on DomainError (no Python traceback, no secret key leak, JSON stderr)
    rc = main(
        [
            "--config",
            str(cfg),
            "illustrations",
            "status",
            "missing-book",
            "--json",
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"detail": {"reason_code": "book_not_found"}}
    assert "Traceback" not in captured.err
    assert "sk-test-secret-key" not in captured.err

    # 8. Clearing model to null when disabling via empty string --model-reference ""
    rc = main(
        [
            "--config",
            str(cfg),
            "illustrations",
            "settings",
            "book-cli",
            "--expected-revision",
            "2",
            "--disable",
            "--model-reference",
            "",
            "--model-prompts",
            "",
            "--json",
        ]
    )
    assert rc == 0
    out_cleared = json.loads(capsys.readouterr().out)
    assert out_cleared["settings"]["enabled"] is False
    assert out_cleared["settings"]["model_reference"] is None
    assert out_cleared["settings"]["model_prompts"] is None


def test_cli_illustrations_validation_errors_paused_warning_and_text_output(
    tmp_path: Path, capsys: Any
) -> None:
    cfg, data_root = _write_config(tmp_path)
    _seed_cli_book(tmp_path, data_root, "book-cli-v")

    # 1. Invalid --offset / --limit rejected with exit code 1 + invalid_request
    for argv in (
        ["--config", str(cfg), "illustrations", "status", "book-cli-v", "--offset", "-1", "--json"],
        ["--config", str(cfg), "illustrations", "status", "book-cli-v", "--limit", "51", "--json"],
        ["--config", str(cfg), "illustrations", "registry", "book-cli-v", "--limit", "101", "--json"],
        [
            "--config",
            str(cfg),
            "illustrations",
            "settings",
            "book-cli-v",
            "--expected-revision",
            "0",
            "--model-reference",
            "gpt-5\nmini",
            "--json",
        ],
    ):
        rc = main(argv)
        assert rc == 1
        cap = capsys.readouterr()
        assert cap.out == ""
        assert json.loads(cap.err) == {"detail": {"reason_code": "invalid_request"}}

    # 2. Human-readable (non-JSON) text output works for settings + status
    rc = main(
        [
            "--config",
            str(cfg),
            "illustrations",
            "settings",
            "book-cli-v",
            "--expected-revision",
            "0",
            "--enable",
            "--model-reference",
            "gpt-5-mini",
            "--model-prompts",
            "gpt-5-mini",
        ]
    )
    assert rc == 0
    text_out = capsys.readouterr().out
    assert "Book: book-cli-v" in text_out
    assert "Enabled: True" in text_out

    # 3. Paused book warning goes to stderr while --json stdout stays pure JSON
    app = build_application(
        settings=AppSettings(
            storage=StorageSettings(data_root=str(data_root)),
            llm=LLMSettings(api_key="sk-test-secret-key"),
        )
    )
    try:
        with app.unit_of_work.transaction() as tx:
            cond = tx.work.latest_conditions("book-cli-v")
            assert cond is not None
            tx.work.set_book_paused("book-cli-v", cond.id, True)
    finally:
        asyncio.run(app.aclose())

    rc = main(["--config", str(cfg), "illustrations", "prepare", "book-cli-v", "--json"])
    assert rc == 0
    cap = capsys.readouterr()
    parsed = json.loads(cap.out)
    assert parsed["state"] == "paused"
    assert parsed["reason_code"] == "book_paused"
    assert "паузе" in cap.err.lower() or "paused" in cap.err.lower()
