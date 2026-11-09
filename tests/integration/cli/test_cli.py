"""CLI integration tests against real application build."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import uvicorn

from linguary.adapters.config import CONFIG_PATH_VARIABLE
from linguary.cli import main

FIXTURE_DIRECTORY = Path(__file__).resolve().parents[2] / "fixtures" / "epub"
SYNTHETIC_EPUB = FIXTURE_DIRECTORY / "synthetic_story.epub"


def test_doctor_reports_configuration_without_secret(
    tmp_path: Path, capsys
) -> None:
    """`doctor` reports environment state without printing provider key."""
    config = _config(
        tmp_path,
        llm_model="fake-model",
        api_key="secret-value-that-must-stay-hidden",
    )

    exit_code = main(["--config", str(config), "doctor"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "LLM: ready" in output
    assert "EPUBCheck:" in output
    assert "Reader level: b2" in output
    assert "secret-value-that-must-stay-hidden" not in output


def test_doctor_reports_reader_level_override(tmp_path: Path, capsys) -> None:
    """`doctor` with --reader-level flag reports the overridden level."""
    config = _config(tmp_path)
    exit_code = main(["--config", str(config), "doctor", "--reader-level", "a2"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Reader level: a2" in output


def test_import_and_worker_once_import_the_book(tmp_path: Path, capsys) -> None:
    """CLI imports EPUB via queue task and a single worker run."""
    config = _config(tmp_path)

    imported = main(
        [
            "--config",
            str(config),
            "import",
            str(SYNTHETIC_EPUB),
            "--book-id",
            "book-cli",
            "--source-language",
            "en",
            "--target-language",
            "ru",
        ]
    )
    drained = main(["--config", str(config), "worker", "--once"])

    output = capsys.readouterr().out
    assert imported == 0
    assert drained == 0
    assert "Book: book-cli" in output
    assert "Tasks finished: 1" in output
    assert (tmp_path / "data" / "books" / "book-cli" / "source.epub").is_file()


def test_export_structure_only_writes_epub(tmp_path: Path, capsys) -> None:
    """Technical CLI export builds EPUB without LLM results."""
    config = _config(tmp_path)
    output_path = tmp_path / "smoke.epub"
    main(
        [
            "--config",
            str(config),
            "import",
            str(SYNTHETIC_EPUB),
            "--book-id",
            "book-export",
        ]
    )
    main(["--config", str(config), "worker", "--once"])

    exit_code = main(
        [
            "--config",
            str(config),
            "export",
            "book-export",
            "--output",
            str(output_path),
            "--mode",
            "structure_only",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output_path.is_file()
    assert "Export:" in output


def test_export_defaults_to_full_mode(tmp_path: Path, capsys) -> None:
    """CLI export defaults to full mode and uses full export with --mode full."""
    config = _config(tmp_path)
    output_path = tmp_path / "smoke.epub"
    main(
        [
            "--config",
            str(config),
            "import",
            str(SYNTHETIC_EPUB),
            "--book-id",
            "book-export-full",
        ]
    )
    main(["--config", str(config), "worker", "--once"])

    # By default --mode full fails on an unready book
    exit_code_default = main(
        [
            "--config",
            str(config),
            "export",
            "book-export-full",
            "--output",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code_default == 1
    assert "экспортировать целиком нельзя" in captured.err

    # Explicit --mode full behaves identically
    exit_code_full = main(
        [
            "--config",
            str(config),
            "export",
            "book-export-full",
            "--output",
            str(output_path),
            "--mode",
            "full",
        ]
    )
    captured_full = capsys.readouterr()
    assert exit_code_full == 1
    assert "экспортировать целиком нельзя" in captured_full.err



def test_analyze_without_llm_configuration_fails_before_queueing(
    tmp_path: Path, capsys
) -> None:
    """Analyze command without LLM configuration fails before scheduling and model call."""
    config = _config(tmp_path)

    exit_code = main(["--config", str(config), "analyze", "unknown-book"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "LLM не настроена" in captured.err


def test_serve_starts_the_web_ui_on_the_configured_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`serve` listens on address from `[server]` section rather than hardcoded address.

    The configuration file is passed via environment: the factory instantiates the application
    inside the server and does not see command-line arguments.
    """
    config = _config(tmp_path, server_host="0.0.0.0", server_port=9100)
    started: dict[str, object] = {}
    monkeypatch.setenv(CONFIG_PATH_VARIABLE, "")
    monkeypatch.setattr(
        uvicorn, "run", lambda target, **options: started.update(options, target=target)
    )

    exit_code = main(["--config", str(config), "serve"])

    assert exit_code == 0
    assert started["target"] == "linguary.web.app:create_app"
    assert started["factory"] is True
    assert started["host"] == "0.0.0.0"
    assert started["port"] == 9100
    assert os.environ[CONFIG_PATH_VARIABLE] == str(config)


def _config(
    tmp_path: Path,
    *,
    llm_model: str = "",
    api_key: str | None = None,
    server_host: str | None = None,
    server_port: int | None = None,
) -> Path:
    """Writes isolated configuration for CLI test."""
    lines = [
        "[storage]",
        f"data_root = {str(tmp_path / 'data')!r}",
        "",
        "[llm]",
        f"model = {llm_model!r}",
    ]
    if api_key is not None:
        lines.append(f"api_key = {api_key!r}")
    if server_host is not None or server_port is not None:
        lines.extend(("", "[server]"))
    if server_host is not None:
        lines.append(f"host = {server_host!r}")
    if server_port is not None:
        lines.append(f"port = {server_port}")
    path = tmp_path / "config.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.mark.parametrize('fails', [False, True])
def test_doctor_closes_owned_http_client(tmp_path, monkeypatch, capsys, fails):
    import httpx
    from linguary.cli import app as cli

    clients = []
    original = httpx.AsyncClient
    def create_client(**options):
        client = original(**options, transport=httpx.MockTransport(lambda request: httpx.Response(200)))
        clients.append(client)
        return client
    monkeypatch.setattr(httpx, 'AsyncClient', create_client)
    if fails:
        def fail(settings):
            raise OSError('diagnostic unavailable')
        monkeypatch.setattr(cli.LLMConfigResponse, 'of_settings', fail)
    config = _config(tmp_path, llm_model='model', api_key='private-test-key')
    assert main(['--config', str(config), 'doctor']) == (1 if fails else 0)
    assert len(clients) == 1 and all(client.is_closed for client in clients)
    captured = capsys.readouterr()
    assert 'private-test-key' not in captured.out + captured.err


@pytest.mark.parametrize('fails', [False, True])
@pytest.mark.parametrize('once', [False, True])
def test_worker_closes_owned_http_client_in_work_loop(tmp_path, monkeypatch, capsys, fails, once):
    import asyncio
    import httpx
    from linguary.application.errors import ApplicationError
    from linguary.cli import app as cli

    events = []
    clients = []
    original = httpx.AsyncClient
    class TrackedClient(original):
        async def aclose(self):
            events.append(('close', asyncio.get_running_loop()))
            await super().aclose()
    def respond(request):
        events.append(('request', asyncio.get_running_loop()))
        return httpx.Response(200)
    def create_client(**options):
        client = TrackedClient(**options, transport=httpx.MockTransport(respond))
        clients.append(client)
        return client
    async def work(application):
        await clients[-1].get('https://mock.invalid/check')
        if fails:
            raise ApplicationError('task failed')
        return []
    monkeypatch.setattr(httpx, 'AsyncClient', create_client)
    monkeypatch.setattr(cli, 'drain_tasks' if once else 'run_worker', work)
    config = _config(tmp_path, llm_model='model', api_key='private-test-key')
    args = ['--config', str(config), 'worker'] + (['--once'] if once else [])
    assert main(args) == (1 if fails else 0)
    assert len(clients) == 1 and all(client.is_closed for client in clients)
    assert [event for event, loop in events] == ['request', 'close']
    assert events[0][1] is events[1][1]
    captured = capsys.readouterr()
    assert 'private-test-key' not in captured.out + captured.err


def test_cli_supports_pause_resume_restart_translation_and_warns_on_pending_corrections_at_export(
    tmp_path: Path, capsys
) -> None:
    import asyncio
    from dataclasses import asdict, replace
    import json
    from uuid import uuid4
    from linguary.adapters.config import load_settings
    from linguary.application.book_conditions import payload_digest
    from linguary.domain.book_conditions import thaw
    from linguary.application.correction import CorrectionInputs, correction_input_fingerprint, original_block_payload
    from linguary.application.correction_dto import (
        CORRECTION_VERSIONS_V2,
        CorrectionSnapshotV2,
        parse_correction_snapshot,
    )
    from linguary.application.recovery import NOTES_POLICY, RecoveryPolicy, root_manifest
    from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
    from linguary.composition import build_application
    from linguary.domain import (
        BlockProcessingResult,
        BlockTreatment,
        BookStatus,
        ChapterAnalysis,
        LLMSettingsSnapshot,
        ProcessingRun,
        ProcessingRunStatus,
        TreatmentPlan,
        TreatmentPlanItem,
    )
    from linguary.domain.content_plan import canonical_sha
    from linguary.domain.correction import CORRECTION_POLICY_VERSION_V2, CorrectionCandidate, CorrectionCheck
    from linguary.domain.processing import GlossarySnapshot
    from linguary.domain.recovery import CallAttempt, RecoveryRoot, RecoveryScope

    config = _config(tmp_path, llm_model="fake-model", api_key="secret-cli-key")
    assert (
        main(
            [
                "--config",
                str(config),
                "import",
                str(SYNTHETIC_EPUB),
                "--book-id",
                "book-cli-ops",
            ]
        )
        == 0
    )
    assert main(["--config", str(config), "worker", "--once"]) == 0
    capsys.readouterr()

    # 1. `analyze` with initial conditions flags
    assert (
        main(
            [
                "--config",
                str(config),
                "analyze",
                "book-cli-ops",
                "--reader-level",
                "b1",
                "--english-help",
                "--pronunciation-help",
                "--pronunciation-variant",
                "us",
                "--seed-first-chapter",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Book: book-cli-ops" in out

    # 2. `pause` and `resume`
    assert main(["--config", str(config), "pause", "book-cli-ops"]) == 0
    pause_out = capsys.readouterr().out
    assert "Paused: True" in pause_out or "pause" in pause_out.lower()

    assert main(["--config", str(config), "resume", "book-cli-ops"]) == 0
    resume_out = capsys.readouterr().out
    assert "Paused: False" in resume_out or "resume" in resume_out.lower()

    # Seed analysis, active run, stopped recovery root, and awaiting_human candidate for restart-translation + export warning
    app = build_application(settings=load_settings(config_path=config))
    try:
        with app.unit_of_work.transaction() as tx:
            cond = tx.work.latest_conditions("book-cli-ops")
            assert cond is not None
            assert cond.payload["reader_level"] == "b1"
            assert cond.payload["english_help"] is True
            assert cond.pronunciation_help_enabled is True
            assert cond.pronunciation_variant == "us"
            assert cond.payload["checks"]["seed_first_chapter"] is True

            doc = tx.documents.get("book-cli-ops")
            assert doc is not None
            for task in tx.tasks.list_for_book("book-cli-ops"):
                tx.tasks.delete(task.id)

            all_results = []
            first_section = doc.ordered_sections[0]
            for section in doc.ordered_sections:
                t_blocks = tuple(b for b in section.ordered_blocks if b.text.strip())
                analysis = ChapterAnalysis.create(
                    id=f"analysis-{section.id}",
                    section_id=section.id,
                    treatment_plan=TreatmentPlan.of_treatments(
                        section.id,
                        [(b.id, BlockTreatment.TRANSLATE) for b in t_blocks],
                    ),
                    glossary_candidates=(),
                    translation_instructions="Сохраняй смысл.",
                    chapter_summary="Summary.",
                    expected_block_ids=[b.id for b in t_blocks],
                )
                tx.analyses.save("book-cli-ops", analysis)
                for b in t_blocks:
                    all_results.append(
                        BlockProcessingResult(
                            block_id=b.id,
                            treatment=BlockTreatment.TRANSLATE,
                            translated_text="Синтетический перевод абзаца.",
                        )
                    )

            run_id = "run-cli-ops"
            run = ProcessingRun(
                id=run_id,
                book_id="book-cli-ops",
                status=ProcessingRunStatus.ACTIVE,
                glossary_snapshot=GlossarySnapshot(),
                llm_settings_snapshot=LLMSettingsSnapshot(
                    provider="openai_compatible", model="fake-model"
                ),
            )
            tx.runs.add(run)
            tx.results.replace_results(run_id, all_results)

            # First test restart-translation on a stopped partial root for first_section
            sec0_blocks = tuple(b.id for b in first_section.ordered_blocks if b.text.strip())
            sec0_analysis = tx.analyses.get_for_section("book-cli-ops", first_section.id)
            policy = RecoveryPolicy(**asdict(app.settings.recovery))
            manifest = root_manifest(
                book_id="book-cli-ops",
                conditions_id=cond.id,
                glossary_sha=payload_digest(asdict(run.glossary_snapshot)),
                analysis_sha=payload_digest(asdict(sec0_analysis)),
                plan_sha=None,
                ordered_blocks=sec0_blocks,
                treatments=tuple("TRANSLATE" for _ in sec0_blocks),
                boundaries=tuple((bid,) for bid in sec0_blocks),
                atomic_intervals=tuple((bid,) for bid in sec0_blocks),
                max_source_chars=4000,
                policy=policy,
                block_chars={bid: 20 for bid in sec0_blocks},
            )
            manifest.update(
                notes_policy=NOTES_POLICY,
                active_run_id=run_id,
                historical_coverage=[],
            )
            root_id = str(uuid4())
            root = RecoveryRoot(
                root_id,
                "book-cli-ops",
                cond.id,
                str(uuid4()),
                manifest,
                calls_used=policy.max_calls,
                status="stopped",
                reason="limit",
                operation_kind="translation",
                policy_version=CURRENT_TRANSLATION_POLICY,
            )
            tx.work.add_root(root)
            for bid in sec0_blocks:
                tx.work.add_scope(RecoveryScope(str(uuid4()), root_id, (bid,)))
            book_rec = tx.books.get("book-cli-ops")
            tx.books.save(replace(book_rec, status=BookStatus.PROCESSING))
    finally:
        asyncio.run(app.aclose())

    # 3. `restart-translation`
    assert (
        main(
            [
                "--config",
                str(config),
                "restart-translation",
                "book-cli-ops",
                root_id,
                "--expected-generation",
                "0",
            ]
        )
        == 0
    )
    restart_out = capsys.readouterr().out
    assert "Successor root:" in restart_out or "root" in restart_out.lower()

    # Now make the book export-ready with 1 current `awaiting_human` correction candidate
    app2 = build_application(settings=load_settings(config_path=config))
    try:
        with app2.unit_of_work.transaction() as tx:
            cond = tx.work.latest_conditions("book-cli-ops")
            doc = tx.documents.get("book-cli-ops")
            run = tx.runs.active_for_book("book-cli-ops")
            for task in tx.tasks.list_for_book("book-cli-ops"):
                tx.tasks.delete(task.id)
            corr_policy = CorrectionSnapshotV2(
                model="fake-model",
                route={
                    "provider": "openai_compatible",
                    "base_url": "http://fake",
                    "proxy": None,
                    "route_id": "fake",
                },
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
                book_id="book-cli-ops",
                conditions_id=cond.id,
                payload={"schema_version": 1, "correction": corr_policy},
            )
            snapshot = parse_correction_snapshot(corr_policy)
            results_by_id = {r.block_id: r for r in tx.results.list_for_run(run.id)}
            for bid in results_by_id:
                if tx.work.get_binding("result", f"{run.id}:{bid}") is None:
                    tx.work.bind(
                        "result",
                        f"{run.id}:{bid}",
                        book_id="book-cli-ops",
                        conditions_id=cond.id,
                        payload={"schema_version": 1},
                    )
            bindings = tx.work.result_bindings(run.id)

            target_section = next(
                s for s in doc.ordered_sections if not doc.is_front_matter(s) and not s.is_notes_only
            )
            t_blocks = tuple(b for b in target_section.ordered_blocks if b.text.strip())
            target_block = t_blocks[0]
            sec_analysis = tx.analyses.get_for_section("book-cli-ops", target_section.id)
            sec_inputs = CorrectionInputs(
                doc,
                target_section,
                sec_analysis,
                run,
                cond,
                snapshot,
                t_blocks,
                {b.id: results_by_id[b.id] for b in t_blocks},
                {b.id: bindings.get(b.id) for b in t_blocks},
                {},
            )
            gens = {
                b.id: tx.work.observe_correction_input(
                    run.id, b.id, correction_input_fingerprint(sec_inputs, b.id)
                )
                for b in t_blocks
            }
            sec_inputs = sec_inputs.with_updated_state(input_generations=gens)
            corr_att_id = str(uuid4())
            tx.work.add_attempt(
                CallAttempt(
                    corr_att_id,
                    "book-cli-ops",
                    state="accepted",
                    conditions_id=cond.id,
                    run_id=run.id,
                    context={
                        "schema_version": 1,
                        "stage": "chapter_correction",
                        "policy_version": CORRECTION_POLICY_VERSION_V2,
                        "ordered_block_ids": [b.id for b in t_blocks],
                    },
                )
            )
            res_target = results_by_id[target_block.id]
            proposed = replace(res_target, translated_text="Альтернативный перевод.")
            cand_id = str(uuid4())
            cand_payload = {
                "schema_version": 1,
                "book_id": "book-cli-ops",
                "run_id": run.id,
                "conditions_id": cond.id,
                "section_id": target_section.id,
                "block_id": target_block.id,
                "input_sha256": sec_inputs.checksums[target_block.id],
                "input_generation": gens[target_block.id],
                "group_id": corr_att_id,
                "proposal_attempt_id": corr_att_id,
                "context_sha256": sec_inputs.context_sha256,
                "source_binding": thaw(bindings.get(target_block.id)),
                "original": json.loads(json.dumps(original_block_payload(target_block))),
                "current_result": json.loads(json.dumps(asdict(res_target))),
                "result": json.loads(json.dumps(asdict(proposed))),
                "proposal": {
                    "issues": [
                        {
                            "source_quote": target_block.text[:20],
                            "current_evidence": {"kind": "quote", "quote": "Синтетический"},
                            "problem": "Стиль.",
                            "minimal_fix_reason": "Уточнение.",
                        }
                    ]
                },
                "evidence_valid": False,
            }
            tx.work.add_correction_candidates(
                (
                    CorrectionCandidate(
                        candidate_id=cand_id,
                        run_id=run.id,
                        book_id="book-cli-ops",
                        section_id=target_section.id,
                        block_id=target_block.id,
                        conditions_id=cond.id,
                        group_id=corr_att_id,
                        proposal_attempt_id=corr_att_id,
                        input_generation=gens[target_block.id],
                        input_sha256=sec_inputs.checksums[target_block.id],
                        proposal_sha256=canonical_sha(cand_payload),
                        state="awaiting_human",
                        payload=cand_payload,
                        review={"verdict": "uncertain", "reason": "Ручная проверка"},
                    ),
                )
            )
            book_rec = tx.books.get("book-cli-ops")
            tx.books.save(replace(book_rec, status=BookStatus.READY_TO_EXPORT))
    finally:
        asyncio.run(app2.aclose())

    # 4. `export` warns on pending corrections while still exporting
    exported_epub = tmp_path / "exported_with_warning.epub"
    exit_export = main(
        [
            "--config",
            str(config),
            "export",
            "book-cli-ops",
            "--output",
            str(exported_epub),
            "--mode",
            "structure_only",
        ]
    )
    captured_export = capsys.readouterr()
    assert exit_export == 0
    assert exported_epub.is_file()
    combined_export_output = (captured_export.out + captured_export.err).lower()
    assert (
        "исправлен" in combined_export_output
        or "pending" in combined_export_output
        or "correction" in combined_export_output
    )

