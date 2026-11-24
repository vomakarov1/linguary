"""Unit & regression tests for synthetic illustration evaluation and adversarial validation (`P5.T1`)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.llm.openai_compatible import OpenAICompatibleLLMClient
from linguary.application.illustration_dto import (
    IllustrationExtractResponse,
)
from linguary.application.ports.llm import LLMRequest, LLMResponse, LLMResponseKind
from linguary.application.ports.persistence import BookRecord
from linguary.composition import build_application, drain_tasks
from linguary.domain import (
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    SourceReference,
    SupportStatus,
    TokenUsage,
)
from linguary.domain.illustrations import (
    IllustrationGenerationState,
    IllustrationReasonCode,
    IllustrationScopeStage,
    IllustrationScopeState,
    IllustrationSettings,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_PATH = REPO_ROOT / "tests" / "fixtures" / "illustrations" / "synthetic_cases.json"
EVALUATE_PATH = REPO_ROOT / "experiments" / "illustrations" / "evaluate.py"


def _load_evaluate_module() -> Any:
    spec = importlib.util.spec_from_file_location("linguary_eval_illustrations", EVALUATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_regression_suite_with_adversarial_replies(tmp_path: Path) -> None:
    eval_mod = _load_evaluate_module()
    out_dir = tmp_path / "eval_fake"
    report = eval_mod.run_fake_evaluation(FIXTURES_PATH, out_dir)

    assert report["invariants"]["generation_ready"] is True
    assert report["invariants"]["empty_chapter_skipped"] is True
    assert report["invariants"]["ordinary_chapter_zero_moments"] is True
    assert report["invariants"]["no_ch10_scar_in_ch2"] is True
    assert report["invariants"]["no_prompt_injection_leak"] is True
    assert report["invariants"]["homonymous_entities_separated"] is True

    adv = report["adversarial_rejections"]
    assert adv["adv_foreign_block_id"] is True
    assert adv["adv_broken_quote"] is True
    assert adv["adv_future_spoiler_in_ch2"] is True
    assert adv["adv_duplicate_scene_moment"] is True
    assert adv["adv_id_only_prompt_without_description_en"] is True

    # Also test adversarial extract replies through real IllustrationReference:
    # (a) complete=false on atomic single-block unit -> fails with atomic_input_too_large / model_output_invalid
    # (b) truncated=True with valid JSON -> fails with provider_truncated
    for case_name, mode in (
        ("adv_incomplete_on_atomic", "incomplete"),
        ("adv_truncated_valid_json", "truncated"),
    ):
        case_root = tmp_path / case_name
        fake_epub = case_root / "src.epub"
        case_root.mkdir(parents=True, exist_ok=True)
        fake_epub.write_bytes(b"PK\x03\x04adv-epub")

        class _AdversarialExtractLLM:
            supports_staged_analysis = True

            def __init__(self, reply_mode: str) -> None:
                self._reply_mode = reply_mode

            async def generate(self, request: LLMRequest[Any]) -> LLMResponse[Any]:
                if request.kind is LLMResponseKind.ILLUSTRATION_EXTRACT:
                    data = eval_mod._extract_json_block(request.input_text)
                    unit_ids = tuple(
                        str(u["source_unit_id"])
                        for u in data.get("units", ())
                        if isinstance(u, dict)
                    )
                    if self._reply_mode == "incomplete":
                        return LLMResponse(
                            payload=IllustrationExtractResponse(
                                mentions=(),
                                facts=(),
                                covered_source_unit_ids=unit_ids,
                                complete=False,
                            ),
                            provider="adv",
                            model="adv",
                            usage=TokenUsage(prompt_tokens=10, completion_tokens=10),
                        )
                    return LLMResponse(
                        payload=IllustrationExtractResponse(
                            mentions=(),
                            facts=(),
                            covered_source_unit_ids=unit_ids,
                            complete=True,
                        ),
                        provider="adv",
                        model="adv",
                        truncated=True,
                        usage=TokenUsage(prompt_tokens=10, completion_tokens=10),
                    )
                base = eval_mod.SyntheticEvalLLMClient(
                    json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
                )
                return await base.generate(request)

        app = build_application(
            settings=AppSettings(
                storage=StorageSettings(data_root=str(case_root / "data")),
                llm=LLMSettings(
                    api_key="test-key",
                    model="adv",
                    staged_analysis=True,
                    max_attempts=1,
                ),
            ),
            llm_client=_AdversarialExtractLLM(mode),
            content_plan_profile="content-plan-v3",
        )
        try:
            book_id = f"book-{case_name}"
            sec_id = f"{book_id}-sec-0"
            doc = CanonicalDocument(
                id=book_id,
                metadata=BookMetadata(title="Adversarial Book"),
                source_language="en",
                target_language="ru",
                sections=(
                    DocumentSection(
                        id=sec_id,
                        document_id=book_id,
                        title="Ch 1",
                        order=0,
                        source_resource_path="Text/ch1.xhtml",
                        blocks=(
                            DocumentBlock(
                                id=f"{book_id}:b0",
                                section_id=sec_id,
                                type=BlockType.PARAGRAPH,
                                order=0,
                                text="John stood in the doorway.",
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
                status=BookStatus.IMPORTED,
                source_epub_path=str(fake_epub),
                created_at=datetime(2026, 11, 12, tzinfo=UTC),
            )
            with app.unit_of_work.transaction() as tx:
                tx.books.add(book)
                tx.documents.save(doc)
                app.book_conditions.create(tx, book, app.runtime_settings.revision)
            asyncio.run(app.analysis.start(book_id))
            asyncio.run(drain_tasks(app))
            app.illustrations.save_settings(
                book_id,
                expected_revision=0,
                settings=IllustrationSettings(
                    enabled=True, max_per_chapter=1, model_reference="adv", model_prompts="adv"
                ),
            )
            view_start = asyncio.run(app.illustrations.prepare(book_id, expected_revision=1))
            asyncio.run(drain_tasks(app))

            with app.unit_of_work.read_transaction() as tx:
                assert view_start.active_generation_id is not None
                scopes = tx.illustrations.list_scopes(
                    view_start.active_generation_id, stage=IllustrationScopeStage.EXTRACT
                )
                assert scopes and scopes[0].state is IllustrationScopeState.FAILED
                if mode == "truncated":
                    assert scopes[0].reason_code == IllustrationReasonCode.PROVIDER_TRUNCATED.value
                else:
                    assert scopes[0].reason_code in {
                        IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value,
                        IllustrationReasonCode.MODEL_OUTPUT_INVALID.value,
                    }
                gen = tx.illustrations.get_generation(book_id, view_start.active_generation_id)
                assert gen is not None and gen.state is not IllustrationGenerationState.READY
        finally:
            asyncio.run(app.aclose())


def test_evaluate_live_refuses_without_allow_paid_calls_before_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    eval_mod = _load_evaluate_module()
    http_attempts: list[str] = []

    def _forbid_httpx(*_args: Any, **_kwargs: Any) -> None:
        http_attempts.append("httpx")
        raise AssertionError("HTTP client must not be constructed without --allow-paid-calls")

    def _forbid_llm(*_args: Any, **_kwargs: Any) -> None:
        http_attempts.append("openai_client")
        raise AssertionError("LLM client must not be constructed without --allow-paid-calls")

    monkeypatch.setattr(httpx, "AsyncClient", _forbid_httpx)
    monkeypatch.setattr(httpx, "Client", _forbid_httpx)
    monkeypatch.setattr(OpenAICompatibleLLMClient, "__init__", _forbid_llm)

    rc = eval_mod.main(
        [
            "--fixtures",
            str(FIXTURES_PATH),
            "--mode",
            "live",
            "--output",
            str(tmp_path / "live_out"),
        ]
    )
    err = capsys.readouterr().err
    assert rc != 0
    assert http_attempts == []
    assert "--allow-paid-calls" in err


def test_evaluate_fake_artifacts_contain_hashes_bytes_calls_and_rubric(tmp_path: Path) -> None:
    eval_mod = _load_evaluate_module()
    out_dir = tmp_path / "artifacts"
    rc = eval_mod.main(
        [
            "--fixtures",
            str(FIXTURES_PATH),
            "--mode",
            "fake",
            "--output",
            str(out_dir),
        ]
    )
    assert rc == 0
    report_path = out_dir / "report.json"
    summary_path = out_dir / "summary.md"
    assert report_path.exists()
    assert summary_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "fake"
    assert len(report["hashes"]["source_sha256"]) == 64
    assert len(report["hashes"]["manifest_sha256"]) == 64
    assert len(report["hashes"]["registry_sha256"]) == 64
    assert 0 < report["bounds"]["max_input_bytes_observed"] <= report["bounds"]["max_request_bytes_limit"]
    assert 0 < report["bounds"]["calls_sent"] <= report["bounds"]["max_calls_per_generation"]
    assert len(report["rubric"]) >= 4
    assert all(report["adversarial_rejections"].values())
