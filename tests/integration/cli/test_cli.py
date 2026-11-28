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
    )
    from linguary.domain.content_plan import canonical_sha
    from linguary.domain.correction import CORRECTION_POLICY_VERSION_V2, CorrectionCandidate
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


def test_cli_analyze_updates_clean_imported_conditions_after_reanalyze_and_rejects_divergent_active_book(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio
    from linguary.adapters.config import load_settings
    from linguary.application.use_cases.reset_processing import ChooseBookConditions
    from linguary.composition import build_application

    config = _config(tmp_path, llm_model="fake-model", api_key="secret-cli-key")
    assert (
        main(
            [
                "--config",
                str(config),
                "import",
                str(SYNTHETIC_EPUB),
                "--book-id",
                "book-cli-reanalyze",
            ]
        )
        == 0
    )
    assert main(["--config", str(config), "worker", "--once"]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "--config",
                str(config),
                "analyze",
                "book-cli-reanalyze",
                "--reader-level",
                "a2",
                "--english-help",
                "--seed-first-chapter",
            ]
        )
        == 0
    )
    capsys.readouterr()

    # Divergent options while ANALYZING must return non-zero exit code
    assert (
        main(
            [
                "--config",
                str(config),
                "analyze",
                "book-cli-reanalyze",
                "--reader-level",
                "c1",
            ]
        )
        == 1
    )
    capsys.readouterr()

    app = build_application(settings=load_settings(config_path=config))
    try:
        ChooseBookConditions(app.book_conditions, app.book_runtime)(
            "book-cli-reanalyze",
            action="reanalyze",
            expected_revision=1,
            settings_revision=app.runtime_settings.revision,
            confirmed=True,
            reader_level="a2",
            english_help=True,
            seed_first_chapter=True,
        )
    finally:
        asyncio.run(app.aclose())

    # On clean IMPORTED after reanalyze, modified flags create a new conditions revision
    assert (
        main(
            [
                "--config",
                str(config),
                "analyze",
                "book-cli-reanalyze",
                "--reader-level",
                "c1",
                "--no-english-help",
                "--no-seed-first-chapter",
            ]
        )
        == 0
    )
    capsys.readouterr()

    app_check = build_application(settings=load_settings(config_path=config))
    try:
        with app_check.unit_of_work.read_transaction() as tx:
            cond = tx.work.latest_conditions("book-cli-reanalyze")
            assert cond is not None
            assert cond.revision == 3
            assert cond.payload["reader_level"] == "c1"
            assert cond.payload["english_help"] is False
            assert cond.payload["checks"]["seed_first_chapter"] is False
    finally:
        asyncio.run(app_check.aclose())


def test_cli_retry_and_restart_translation_warn_when_book_is_paused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio
    from dataclasses import asdict, replace
    from uuid import uuid4
    from linguary.adapters.config import load_settings
    from linguary.application.book_conditions import payload_digest
    from linguary.application.recovery import NOTES_POLICY, RecoveryPolicy, root_manifest
    from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
    from linguary.composition import build_application
    from linguary.domain import (
        BlockTreatment,
        BookStatus,
        ChapterAnalysis,
        ErrorType,
        LLMSettingsSnapshot,
        ProcessingRun,
        ProcessingRunStatus,
        TaskError,
        TaskStatus,
        TaskType,
        TreatmentPlan,
    )
    from linguary.domain.processing import GlossarySnapshot
    from linguary.domain.recovery import RecoveryRoot, RecoveryScope
    from linguary.domain.tasks import ProcessingTask

    config = _config(tmp_path, llm_model="fake-model", api_key="secret-cli-key")
    book_id = "book-cli-paused-warn"
    assert (
        main(
            [
                "--config",
                str(config),
                "import",
                str(SYNTHETIC_EPUB),
                "--book-id",
                book_id,
            ]
        )
        == 0
    )
    assert main(["--config", str(config), "worker", "--once"]) == 0
    assert main(["--config", str(config), "analyze", book_id, "--reader-level", "b1"]) == 0
    assert main(["--config", str(config), "pause", book_id]) == 0
    capsys.readouterr()

    failed_task_id = "task-failed-paused"
    root_id = str(uuid4())
    app = build_application(settings=load_settings(config_path=config))
    try:
        with app.unit_of_work.transaction() as tx:
            cond = tx.work.latest_conditions(book_id)
            doc = tx.documents.get(book_id)
            for existing_task in tx.tasks.list_for_book(book_id):
                tx.tasks.delete(existing_task.id)
            first_section = doc.ordered_sections[0]
            blocks = tuple(b.id for b in first_section.ordered_blocks if b.text.strip())
            analysis = ChapterAnalysis.create(
                id=f"analysis-{first_section.id}",
                section_id=first_section.id,
                treatment_plan=TreatmentPlan.of_treatments(
                    first_section.id,
                    [(bid, BlockTreatment.TRANSLATE) for bid in blocks],
                ),
                glossary_candidates=(),
                translation_instructions="Сохраняй смысл.",
                chapter_summary="Summary.",
                expected_block_ids=list(blocks),
            )
            tx.analyses.save(book_id, analysis)
            run = ProcessingRun(
                id="run-paused-cli",
                book_id=book_id,
                status=ProcessingRunStatus.ACTIVE,
                glossary_snapshot=GlossarySnapshot(),
                llm_settings_snapshot=LLMSettingsSnapshot(
                    provider="openai_compatible", model="fake-model"
                ),
            )
            tx.runs.add(run)
            policy = RecoveryPolicy(**asdict(app.settings.recovery))
            manifest = root_manifest(
                book_id=book_id,
                conditions_id=cond.id,
                glossary_sha=payload_digest(asdict(run.glossary_snapshot)),
                analysis_sha=payload_digest(asdict(analysis)),
                plan_sha=None,
                ordered_blocks=blocks,
                treatments=tuple("TRANSLATE" for _ in blocks),
                boundaries=tuple((bid,) for bid in blocks),
                atomic_intervals=tuple((bid,) for bid in blocks),
                max_source_chars=4000,
                policy=policy,
                block_chars={bid: 20 for bid in blocks},
            )
            manifest.update(
                notes_policy=NOTES_POLICY,
                active_run_id=run.id,
                historical_coverage=[],
            )
            tx.work.add_root(
                RecoveryRoot(
                    root_id,
                    book_id,
                    cond.id,
                    str(uuid4()),
                    manifest,
                    calls_used=policy.max_calls,
                    status="stopped",
                    reason="limit",
                    operation_kind="translation",
                    policy_version=CURRENT_TRANSLATION_POLICY,
                )
            )
            for bid in blocks:
                tx.work.add_scope(RecoveryScope(str(uuid4()), root_id, (bid,)))
            failed_task = ProcessingTask(
                id=failed_task_id,
                book_id=book_id,
                task_type=TaskType.ANALYZE_CHAPTER,
                scope_id=first_section.id,
                status=TaskStatus.FAILED,
                last_error=TaskError(
                    type=ErrorType.TRANSIENT, message="temporary failure"
                ),
            )
            tx.tasks.add(failed_task)
            tx.work.bind("task", failed_task_id, book_id=book_id, conditions_id=cond.id)
            book_rec = tx.books.get(book_id)
            tx.books.save(replace(book_rec, status=BookStatus.PROCESSING))
    finally:
        asyncio.run(app.aclose())

    assert main(["--config", str(config), "retry", failed_task_id]) == 0
    retry_captured = capsys.readouterr()
    assert f"Task: {failed_task_id}" in retry_captured.out
    assert "Status: pending" in retry_captured.out
    assert "Warning:" in retry_captured.err
    assert book_id in retry_captured.err
    assert "pause" in retry_captured.err.lower() or "пауз" in retry_captured.err.lower()
    assert "resume" in retry_captured.err.lower()

    assert (
        main(
            [
                "--config",
                str(config),
                "restart-translation",
                book_id,
                root_id,
                "--expected-generation",
                "0",
            ]
        )
        == 0
    )
    restart_captured = capsys.readouterr()
    assert f"Book: {book_id}" in restart_captured.out
    assert "Successor root:" in restart_captured.out
    assert "Warning:" in restart_captured.err
    assert book_id in restart_captured.err
    assert "pause" in restart_captured.err.lower() or "пауз" in restart_captured.err.lower()
    assert "resume" in restart_captured.err.lower()

    assert main(["--config", str(config), "resume", book_id]) == 0
    capsys.readouterr()

    app_unpaused = build_application(settings=load_settings(config_path=config))
    try:
        with app_unpaused.unit_of_work.transaction() as tx:
            task = tx.tasks.get(failed_task_id)
            assert task is not None
            tx.tasks.save(
                task.start().fail(
                    TaskError(
                        type=ErrorType.TRANSIENT,
                        message="temporary failure after resume",
                    )
                )
            )
    finally:
        asyncio.run(app_unpaused.aclose())

    assert main(["--config", str(config), "retry", failed_task_id]) == 0
    unpaused_captured = capsys.readouterr()
    assert f"Task: {failed_task_id}" in unpaused_captured.out
    assert unpaused_captured.err == ""


def test_cli_restart_translation_requires_acknowledge_flag_for_unknown_outcome(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unknown-outcome restart is rejected until --acknowledge-unknown-outcome is passed."""
    import asyncio
    from dataclasses import asdict, replace
    from uuid import uuid4

    from linguary.adapters.config import load_settings
    from linguary.application.book_conditions import payload_digest
    from linguary.application.recovery import NOTES_POLICY, RecoveryPolicy, root_manifest
    from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
    from linguary.composition import build_application
    from linguary.domain import (
        BlockTreatment,
        BookStatus,
        ChapterAnalysis,
        LLMSettingsSnapshot,
        ProcessingRun,
        ProcessingRunStatus,
        TreatmentPlan,
    )
    from linguary.domain.processing import GlossarySnapshot
    from linguary.domain.recovery import RecoveryRoot, RecoveryScope

    confirmation = (
        "требуется явное подтверждение перезапуска после неизвестного исхода "
        "(возможна повторная оплата)"
    )
    config = _config(tmp_path, llm_model="fake-model", api_key="secret-cli-key")
    book_id = "book-cli-unknown-outcome"
    assert (
        main(
            [
                "--config",
                str(config),
                "import",
                str(SYNTHETIC_EPUB),
                "--book-id",
                book_id,
            ]
        )
        == 0
    )
    assert main(["--config", str(config), "worker", "--once"]) == 0
    assert main(["--config", str(config), "analyze", book_id, "--reader-level", "b1"]) == 0
    capsys.readouterr()

    root_id = str(uuid4())
    app = build_application(settings=load_settings(config_path=config))
    try:
        with app.unit_of_work.transaction() as tx:
            cond = tx.work.latest_conditions(book_id)
            doc = tx.documents.get(book_id)
            assert cond is not None and doc is not None
            for existing_task in tx.tasks.list_for_book(book_id):
                tx.tasks.delete(existing_task.id)
            first_section = doc.ordered_sections[0]
            blocks = tuple(block.id for block in first_section.ordered_blocks if block.text.strip())
            analysis = ChapterAnalysis.create(
                id=f"analysis-{first_section.id}",
                section_id=first_section.id,
                treatment_plan=TreatmentPlan.of_treatments(
                    first_section.id,
                    [(block_id, BlockTreatment.TRANSLATE) for block_id in blocks],
                ),
                glossary_candidates=(),
                translation_instructions="Сохраняй смысл.",
                chapter_summary="Summary.",
                expected_block_ids=list(blocks),
            )
            tx.analyses.save(book_id, analysis)
            run = ProcessingRun(
                id="run-unknown-cli",
                book_id=book_id,
                status=ProcessingRunStatus.ACTIVE,
                glossary_snapshot=GlossarySnapshot(),
                llm_settings_snapshot=LLMSettingsSnapshot(
                    provider="openai_compatible", model="fake-model"
                ),
            )
            tx.runs.add(run)
            policy = RecoveryPolicy(**asdict(app.settings.recovery))
            manifest = root_manifest(
                book_id=book_id,
                conditions_id=cond.id,
                glossary_sha=payload_digest(asdict(run.glossary_snapshot)),
                analysis_sha=payload_digest(asdict(analysis)),
                plan_sha=None,
                ordered_blocks=blocks,
                treatments=tuple("TRANSLATE" for _ in blocks),
                boundaries=tuple((block_id,) for block_id in blocks),
                atomic_intervals=tuple((block_id,) for block_id in blocks),
                max_source_chars=4000,
                policy=policy,
                block_chars={block_id: 20 for block_id in blocks},
            )
            manifest.update(
                notes_policy=NOTES_POLICY,
                active_run_id=run.id,
                historical_coverage=[],
            )
            tx.work.add_root(
                RecoveryRoot(
                    root_id,
                    book_id,
                    cond.id,
                    str(uuid4()),
                    manifest,
                    calls_used=policy.max_calls,
                    status="stopped",
                    reason="unknown_outcome",
                    operation_kind="translation",
                    policy_version=CURRENT_TRANSLATION_POLICY,
                )
            )
            for block_id in blocks:
                tx.work.add_scope(RecoveryScope(str(uuid4()), root_id, (block_id,)))
            book_rec = tx.books.get(book_id)
            tx.books.save(replace(book_rec, status=BookStatus.PROCESSING))
    finally:
        asyncio.run(app.aclose())

    assert (
        main(
            [
                "--config",
                str(config),
                "restart-translation",
                book_id,
                root_id,
                "--expected-generation",
                "0",
            ]
        )
        == 1
    )
    rejected = capsys.readouterr()
    assert confirmation in rejected.err
    assert "Successor root:" not in rejected.out

    checked = build_application(settings=load_settings(config_path=config))
    try:
        with checked.unit_of_work.transaction() as tx:
            stored = tx.work.get_root(root_id)
            assert stored is not None
            assert stored.superseded_by is None
            assert stored.reason == "unknown_outcome"
    finally:
        asyncio.run(checked.aclose())

    assert (
        main(
            [
                "--config",
                str(config),
                "restart-translation",
                book_id,
                root_id,
                "--expected-generation",
                "0",
                "--acknowledge-unknown-outcome",
            ]
        )
        == 0
    )
    acknowledged = capsys.readouterr()
    assert f"Book: {book_id}" in acknowledged.out
    successor_line = next(
        line for line in acknowledged.out.splitlines() if line.startswith("Successor root:")
    )
    successor_id = successor_line.split(":", 1)[1].strip()
    assert successor_id

    verified = build_application(settings=load_settings(config_path=config))
    try:
        with verified.unit_of_work.transaction() as tx:
            predecessor = tx.work.get_root(root_id)
            successor = tx.work.get_root(successor_id)
            assert predecessor is not None and successor is not None
            assert predecessor.superseded_by == successor_id
            audit = successor.compatibility
            assert audit["had_unknown_outcome"] is True
            assert audit["acknowledged_duplicate_billing_risk"] is True
            assert audit["predecessor_root_id"] == root_id
            assert audit["previous_reason"] == "unknown_outcome"
    finally:
        asyncio.run(verified.aclose())


async def test_web_lifespan_waits_gracefully_before_cancelling_worker(monkeypatch) -> None:
    """Shutdown lets the worker finish inside the grace window and cancels it only after that."""
    import asyncio

    from fastapi import status

    from linguary.application.use_cases.extend_recovery_limit import (
        ExtendRecoveryLimitConflict,
        ExtendRecoveryLimitError,
        ExtendRecoveryLimitNotFoundError,
    )
    from linguary.application.use_cases.restart_translation import (
        RestartTranslationConflict,
        RestartTranslationError,
    )
    from linguary.web import app as web_app
    from linguary.web.app import _status_for, create_app

    assert web_app.DEFAULT_SHUTDOWN_GRACE_SECONDS == 5.0
    assert _status_for(RestartTranslationConflict("conflict")) == status.HTTP_409_CONFLICT
    assert _status_for(RestartTranslationError("rejected")) == status.HTTP_409_CONFLICT
    assert (
        _status_for(ExtendRecoveryLimitConflict("conflict", "generation"))
        == status.HTTP_409_CONFLICT
    )
    assert _status_for(ExtendRecoveryLimitError("rejected", "limit")) == status.HTTP_409_CONFLICT
    assert _status_for(ExtendRecoveryLimitNotFoundError()) == status.HTTP_404_NOT_FOUND

    grace = 0.25
    monkeypatch.setattr(web_app, "DEFAULT_SHUTDOWN_GRACE_SECONDS", grace)
    loop = asyncio.get_running_loop()

    class _Probe:
        def __init__(self) -> None:
            self.release = asyncio.Event()
            self.shutdown_at: float | None = None
            self.closed_at: float | None = None
            self.system_settings = self

        def begin_close(self) -> None:
            self.shutdown_at = loop.time()
            self.release.set()

        async def aclose(self) -> None:
            self.closed_at = loop.time()

    async def _shutdown(worker, probe: _Probe) -> None:
        monkeypatch.setattr(web_app, "run_worker", worker)
        web = create_app(application=probe, run_background_worker=True)
        lifespan = web.router.lifespan_context(web)
        await lifespan.__aenter__()
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(lifespan.__aexit__(None, None, None), 2)

    stubborn_started = asyncio.Event()
    cancelled_after: list[float] = []
    stubborn = _Probe()

    async def stubborn_worker(_application: object) -> None:
        stubborn_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            assert stubborn.shutdown_at is not None
            cancelled_after.append(loop.time() - stubborn.shutdown_at)
            raise

    started = stubborn_started
    await _shutdown(stubborn_worker, stubborn)
    assert cancelled_after and cancelled_after[0] >= grace * 0.8
    assert stubborn.shutdown_at is not None and stubborn.closed_at is not None
    assert stubborn.closed_at - stubborn.shutdown_at >= cancelled_after[0]

    cooperative_started = asyncio.Event()
    cooperative_finished = asyncio.Event()
    cooperative_cancelled = False
    cooperative = _Probe()

    async def cooperative_worker(application: _Probe) -> None:
        nonlocal cooperative_cancelled
        cooperative_started.set()
        try:
            await application.release.wait()
            await asyncio.sleep(grace / 5)
            cooperative_finished.set()
        except asyncio.CancelledError:
            cooperative_cancelled = True
            raise

    started = cooperative_started
    await _shutdown(cooperative_worker, cooperative)
    assert cooperative_finished.is_set()
    assert cooperative_cancelled is False
    assert cooperative.shutdown_at is not None and cooperative.closed_at is not None
    assert cooperative.closed_at - cooperative.shutdown_at < grace

    # 3. Cancelling lifespan itself during grace window cancels and awaits worker before aclose()
    cancel_started = asyncio.Event()
    events: list[str] = []

    class _CancelProbe(_Probe):
        async def aclose(self) -> None:
            events.append("aclose")
            await super().aclose()

    cancel_probe = _CancelProbe()

    async def blocking_worker(_application: object) -> None:
        cancel_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("worker_cancelled")
            raise

    monkeypatch.setattr(web_app, "run_worker", blocking_worker)
    web = create_app(application=cancel_probe, run_background_worker=True)
    lifespan = web.router.lifespan_context(web)
    await lifespan.__aenter__()
    await asyncio.wait_for(cancel_started.wait(), 1)
    exit_task = asyncio.create_task(lifespan.__aexit__(None, None, None))
    await asyncio.sleep(0.02)
    exit_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await exit_task
    assert events == ["worker_cancelled", "aclose"]


async def test_web_lifespan_with_real_worker_pool_shuts_down_cleanly(tmp_path) -> None:
    """Real lifespan -> begin_close() -> TaskWorkerPool.run_forever() exits cleanly without hanging."""
    import asyncio
    from linguary.adapters.config import AppSettings, StorageSettings
    from linguary.composition import build_application
    from linguary.web.app import create_app

    app = build_application(
        settings=AppSettings(storage=StorageSettings(data_root=str(tmp_path / "data"))),
    )
    try:
        web = create_app(application=app, run_background_worker=True)
        lifespan = web.router.lifespan_context(web)
        await lifespan.__aenter__()
        await asyncio.sleep(0.05)
        await asyncio.wait_for(lifespan.__aexit__(None, None, None), timeout=3.0)
        assert app.worker_pool.stop_requested is True
    finally:
        await app.aclose()


def test_cli_glossary_list_and_accept_all_unblock_flagged_terms(tmp_path: Path, capsys) -> None:
    """WP3: `glossary list --pending-only` and `glossary accept-all` resolve model-flagged items via CLI."""
    import json
    from linguary.adapters.config import load_settings
    from linguary.composition import build_application
    from linguary.domain import (
        BlockTreatment,
        BookStatus,
        ChapterAnalysis,
        ConceptType,
        CoverageStatus,
        GlossaryCandidate,
        TreatmentPlan,
        TreatmentPlanItem,
    )
    from dataclasses import replace

    config = _config(tmp_path, llm_model="gpt-test", api_key="test-key")
    assert main(["--config", str(config), "import", str(SYNTHETIC_EPUB), "--book-id", "book-gl"]) == 0
    assert main(["--config", str(config), "worker", "--once"]) == 0
    capsys.readouterr()

    app = build_application(settings=load_settings(config_path=config))
    try:
        with app.unit_of_work.transaction() as tx:
            doc = tx.documents.get("book-gl")
            assert doc is not None
            section = doc.ordered_sections[0]
            tx.analyses.save(
                "book-gl",
                ChapterAnalysis(
                    id="an-gl-1",
                    section_id=section.id,
                    coverage_status=CoverageStatus.COMPLETE,
                    treatment_plan=TreatmentPlan(
                        section_id=section.id,
                        items=tuple(
                            TreatmentPlanItem(
                                block_id=b.id,
                                treatment=BlockTreatment.TRANSLATE,
                            )
                            for b in section.blocks
                            if b.text.strip()
                        ),
                    ),
                    translation_instructions="",
                    chapter_summary="Summary.",
                    glossary_candidates=(
                        GlossaryCandidate(
                            id="cand-1",
                            analysis_id="an-gl-1",
                            source_text="Privet Drive",
                            type=ConceptType.PLACE,
                            suggested_translation="Привит-драйв",
                            confidence=0.95,
                            model_flagged_review=True,
                            requires_review=True,
                        ),
                    ),
                )
            )
            book = tx.books.get("book-gl")
            assert book is not None
            tx.books.save(replace(book, status=BookStatus.GLOSSARY_REVIEW))
    finally:
        import asyncio
        asyncio.run(app.aclose())

    assert main(["--config", str(config), "glossary", "list", "book-gl", "--pending-only", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["pending"] == 1
    assert listed["items"][0]["source_text"] == "Privet Drive"

    assert main(["--config", str(config), "glossary", "accept-all", "book-gl", "--json"]) == 0
    accepted = json.loads(capsys.readouterr().out)
    assert accepted["accepted_count"] == 1
    assert accepted["remaining_pending"] == 0


def test_cli_glossary_list_reports_auto_mandatory_and_summary(tmp_path: Path, capsys) -> None:
    """WP4: `glossary list --json` and text output report `auto_mandatory` and category summary."""
    import asyncio
    import json
    from dataclasses import replace
    from linguary.adapters.config import load_settings
    from linguary.composition import build_application
    from linguary.domain import (
        BlockTreatment,
        BookStatus,
        ChapterAnalysis,
        ConceptType,
        CoverageStatus,
        GlossaryCandidate,
        TermConcept,
        TermStatus,
        TermVariant,
        TreatmentPlan,
        TreatmentPlanItem,
    )
    from linguary.domain.name_translation import (
        NameTranslationAssessment,
        NameTranslationEvidence,
        NameTranslationObservation,
        NameTranslationStrategy,
    )

    config = _config(tmp_path, llm_model="gpt-test", api_key="test-key")
    assert main(["--config", str(config), "import", str(SYNTHETIC_EPUB), "--book-id", "book-gl-wp4"]) == 0
    assert main(["--config", str(config), "worker", "--once"]) == 0
    capsys.readouterr()

    app = build_application(settings=load_settings(config_path=config))
    try:
        conditions = app.book_conditions.ensure_initial_for_analysis(
            "book-gl-wp4",
            validate_conditions=app.book_runtime.settings_for,
        )
        strategy = NameTranslationStrategy(
            conditions.payload["name_translation_policy"]["strategy"]
        )
        with app.unit_of_work.transaction() as tx:
            doc = tx.documents.get("book-gl-wp4")
            assert doc is not None
            section = doc.ordered_sections[0]
            block_text = section.ordered_blocks[0].text
            # Ensure enough occurrences in the document for auto_mandatory key entity.
            updated_section = replace(
                section,
                blocks=(
                    replace(
                        section.ordered_blocks[0],
                        text=(
                            f"{block_text} Alice Alice Alice Alice Alice "
                            "Privet Drive Privet Drive Privet Drive Privet Drive Privet Drive "
                            "Hogwarts Hogwarts Hogwarts Hogwarts Hogwarts"
                        ),
                    ),
                    *section.ordered_blocks[1:],
                ),
            )
            tx.documents.save(
                replace(doc, sections=(updated_section, *doc.ordered_sections[1:]))
            )
            applied_method = "sound" if strategy is NameTranslationStrategy.SOUND else "meaning"
            exempt_evidence = NameTranslationEvidence(
                1,
                conditions.id,
                1,
                strategy,
                (
                    NameTranslationObservation(
                        "Элис",
                        0.95,
                        False,
                        NameTranslationAssessment(
                            applied_method, True, (), "Однозначное имя.", opaque_name=True
                        ),
                    ),
                ),
            )
            tx.analyses.save(
                "book-gl-wp4",
                ChapterAnalysis(
                    id="an-wp4-1",
                    section_id=section.id,
                    coverage_status=CoverageStatus.COMPLETE,
                    treatment_plan=TreatmentPlan(
                        section_id=section.id,
                        items=tuple(
                            TreatmentPlanItem(
                                block_id=b.id,
                                treatment=BlockTreatment.TRANSLATE,
                            )
                            for b in updated_section.blocks
                            if b.text.strip()
                        ),
                    ),
                    translation_instructions="",
                    chapter_summary="Summary.",
                    glossary_candidates=(
                        GlossaryCandidate(
                            id="cand-alice",
                            analysis_id="an-wp4-1",
                            source_text="Alice",
                            type=ConceptType.PERSON,
                            suggested_translation="Элис",
                            confidence=0.95,
                            model_flagged_review=False,
                            requires_review=False,
                            name_translation_evidence=exempt_evidence,
                        ),
                        GlossaryCandidate(
                            id="cand-privet",
                            analysis_id="an-wp4-1",
                            source_text="Privet Drive",
                            type=ConceptType.PLACE,
                            suggested_translation="Привит-драйв",
                            confidence=0.95,
                            model_flagged_review=True,
                            requires_review=True,
                        ),
                        GlossaryCandidate(
                            id="cand-hogwarts",
                            analysis_id="an-wp4-1",
                            source_text="Hogwarts",
                            type=ConceptType.PLACE,
                            suggested_translation="Хогвартс",
                            confidence=0.95,
                            model_flagged_review=False,
                            requires_review=True,
                        ),
                        GlossaryCandidate(
                            id="cand-term",
                            analysis_id="an-wp4-1",
                            source_text="cauldron",
                            type=ConceptType.TERM,
                            suggested_translation="котёл",
                            confidence=0.95,
                            model_flagged_review=False,
                            requires_review=False,
                        ),
                    ),
                ),
            )
            tx.glossary.save(
                TermConcept(
                    id="dec-hogwarts",
                    book_id="book-gl-wp4",
                    type=ConceptType.PLACE,
                    canonical_source_text="Hogwarts",
                    status=TermStatus.APPROVED,
                    variants=(
                        TermVariant(
                            concept_id="dec-hogwarts",
                            language="ru",
                            text="Хогвартс",
                            status=TermStatus.APPROVED,
                        ),
                    ),
                )
            )
            book = tx.books.get("book-gl-wp4")
            assert book is not None
            tx.books.save(replace(book, status=BookStatus.GLOSSARY_REVIEW))
    finally:
        asyncio.run(app.aclose())

    assert main(["--config", str(config), "glossary", "list", "book-gl-wp4", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["summary"] == {
        "decided": 1,
        "auto_mandatory": 1,
        "awaiting_review": 1,
        "suggested": 1,
    }
    by_source = {item["source_text"]: item for item in listed["items"]}
    assert by_source["Alice"]["auto_mandatory"] is True
    assert by_source["Privet Drive"]["auto_mandatory"] is False
    assert by_source["Hogwarts"]["auto_mandatory"] is False
    assert by_source["cauldron"]["auto_mandatory"] is False

    assert main(["--config", str(config), "glossary", "list", "book-gl-wp4"]) == 0
    text_out = capsys.readouterr().out
    assert "auto_mandatory: 1" in text_out
    assert "Alice -> Элис (auto_mandatory)" in text_out


def test_cli_corrections_list_resolve_and_export_strict_corrections(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """WP3: `corrections list|resolve` and `export --strict-corrections` enforce awaiting_human gate before writing."""
    import asyncio
    import json
    from dataclasses import replace
    from linguary.adapters.config import load_settings
    from linguary.application.use_cases.resolve_correction_candidate import (
        ResolveCorrectionCommand,
    )
    from linguary.cli import app as cli_app
    from linguary.composition import build_application
    from linguary.domain import (
        BookStatus,
        GlossarySnapshot,
        LLMSettingsSnapshot,
        ProcessingRun,
        ProcessingRunStatus,
    )
    from linguary.domain.content_plan import canonical_sha

    config = _config(tmp_path, llm_model="gpt-test", api_key="test-key")
    assert main(["--config", str(config), "import", str(SYNTHETIC_EPUB), "--book-id", "book-corr"]) == 0
    assert main(["--config", str(config), "worker", "--once"]) == 0
    capsys.readouterr()

    app = build_application(settings=load_settings(config_path=config))
    try:
        cond_id = app.book_conditions.freeze_before_analysis("book-corr", expected_settings_revision=0)
        with app.unit_of_work.transaction() as tx:
            doc = tx.documents.get("book-corr")
            assert doc is not None
            section = doc.ordered_sections[0]
            block = section.blocks[0]
            run = ProcessingRun(
                id="run-corr-1",
                book_id="book-corr",
                status=ProcessingRunStatus.ACTIVE,
                glossary_snapshot=GlossarySnapshot(),
                llm_settings_snapshot=LLMSettingsSnapshot(provider="openai_compatible", model="gpt-test"),
            )
            tx.runs.add(run)
            payload = {
                "schema_version": 1,
                "book_id": "book-corr",
                "run_id": run.id,
                "conditions_id": cond_id,
                "section_id": section.id,
                "block_id": block.id,
                "input_sha256": "in-sha-1",
                "input_generation": 1,
                "group_id": "grp-1",
                "proposal_attempt_id": "att-1",
                "current_result": {
                    "block_id": block.id,
                    "treatment": "translate",
                    "translated_text": "Исходный перевод.",
                },
                "result": {
                    "block_id": block.id,
                    "treatment": "translate",
                    "translated_text": "Исправленный перевод.",
                },
                "proposal": {
                    "block_id": block.id,
                    "status": "proposed",
                    "translated_text": "Исправленный перевод.",
                    "issues": [],
                },
            }
            from linguary.adapters.db.models import CorrectionCandidateRow
            expected_prop_sha = canonical_sha(payload)
            tx.work._session.add(
                CorrectionCandidateRow(
                    candidate_id="cand-corr-1",
                    book_id="book-corr",
                    run_id=run.id,
                    conditions_id=cond_id,
                    section_id=section.id,
                    block_id=block.id,
                    input_sha256="in-sha-1",
                    input_generation=1,
                    proposal_sha256=expected_prop_sha,
                    payload_json=payload,
                    group_id="grp-1",
                    proposal_attempt_id="att-1",
                    state="awaiting_human",
                    review_json={"reason": "Проверьте пунктуацию"},
                )
            )
            book = tx.books.get("book-corr")
            assert book is not None
            tx.books.save(replace(book, status=BookStatus.READY_TO_EXPORT))
    finally:
        asyncio.run(app.aclose())

    assert main(["--config", str(config), "corrections", "list", "book-corr", "--awaiting-only", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert len(listed["items"]) == 1
    assert listed["items"][0]["candidate_id"] == "cand-corr-1"
    assert listed["items"][0]["section_id"] == section.id
    assert listed["items"][0]["proposal_sha256"] == expected_prop_sha
    assert listed["items"][0]["review_reason"] == "Проверьте пунктуацию"
    assert listed["items"][0]["proposed_text"] == "Исправленный перевод."

    out_epub = tmp_path / "strict.epub"
    strict_code = main(
        [
            "--config",
            str(config),
            "export",
            "book-corr",
            "--output",
            str(out_epub),
            "--mode",
            "structure_only",
            "--strict-corrections",
        ]
    )
    err_text = capsys.readouterr().err
    assert strict_code == 1
    assert "strict-corrections" in err_text
    assert not out_epub.exists()

    captured_commands: list[ResolveCorrectionCommand] = []
    orig_build = cli_app.build_application

    def _patched_build(*args, **kwargs):
        built = orig_build(*args, **kwargs)
        return replace(
            built,
            resolve_correction_candidate=lambda cmd: (
                captured_commands.append(cmd) or block.id
            ),
        )

    monkeypatch.setattr(cli_app, "build_application", _patched_build)
    resolve_code = main(
        [
            "--config",
            str(config),
            "corrections",
            "resolve",
            "book-corr",
            "cand-corr-1",
            "--action",
            "keep",
            "--reason",
            "Оригинал точнее",
            "--json",
        ]
    )
    resolved = json.loads(capsys.readouterr().out)
    assert resolve_code == 0
    assert resolved["action"] == "keep"
    assert len(captured_commands) == 1
    assert captured_commands[0].proposal_sha256 == expected_prop_sha
    assert captured_commands[0].input_sha256 == "in-sha-1"
    assert captured_commands[0].reason == "Оригинал точнее"

    blank_reason_code = main(
        [
            "--config",
            str(config),
            "corrections",
            "resolve",
            "book-corr",
            "cand-corr-1",
            "--action",
            "keep",
            "--reason",
            "   ",
            "--json",
        ]
    )
    assert blank_reason_code == 1
    assert "error" in capsys.readouterr().err


def test_candidate_snippet_extracts_v2_result_segments_glossary_only_and_fallbacks() -> None:
    from types import SimpleNamespace
    from linguary.cli.app import _candidate_snippet

    # 1. Real CorrectChapterV2 payload with result.translated_text
    cand_text = SimpleNamespace(
        payload={
            "result": {"treatment": "translate", "translated_text": "Исправленный перевод."},
            "proposal": {"translated_text": "Другой текст."},
        },
        review={"reason": "Уточнить"},
    )
    assert _candidate_snippet(cand_text) == ("Уточнить", "Исправленный перевод.")

    # 2. Multi-segment block using result.segment_translations (matching rebuild_translation)
    cand_segments = SimpleNamespace(
        payload={
            "result": {
                "treatment": "translate",
                "translated_text": None,
                "segment_translations": ["Первый фрагмент.", None, "   ", "Третий фрагмент."],
            },
        },
        review={"reason": "Сегменты"},
    )
    assert _candidate_snippet(cand_segments) == ("Сегменты", "Первый фрагмент. Третий фрагмент.")

    # 3. GLOSSARY_ONLY and READ_AS_IS return None for proposed_text
    cand_glossary_only = SimpleNamespace(
        payload={
            "result": {
                "treatment": "glossary_only",
                "translated_text": None,
                "segment_translations": [],
                "glossary_hints": [{"original": "owl", "translation": "сова"}],
            },
            "proposal": {"translated_text": None, "segments": []},
        },
        review={"reason": "Подсказка"},
    )
    assert _candidate_snippet(cand_glossary_only) == ("Подсказка", None)

    cand_read_as_is = SimpleNamespace(
        payload={
            "result": {"treatment": "read_as_is", "translated_text": None},
        },
        review=None,
    )
    assert _candidate_snippet(cand_read_as_is) == (None, None)

    # 4. Legacy proposed_result fallback
    cand_legacy = SimpleNamespace(
        payload={"proposed_result": {"translated_text": "Старый формат."}},
        review=None,
    )
    assert _candidate_snippet(cand_legacy) == (None, "Старый формат.")

    # 5. Proposal fallback with segments list
    cand_proposal_segs = SimpleNamespace(
        payload={
            "proposal": {
                "translated_text": None,
                "segments": [
                    {"index": 1, "translated_text": "Сегмент 1."},
                    {"index": 2, "translated_text": "Сегмент 2."},
                ],
            }
        },
        review=None,
    )
    assert _candidate_snippet(cand_proposal_segs) == (None, "Сегмент 1. Сегмент 2.")


def test_cli_retry_preserves_fragment_cache_by_default_and_clears_with_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """WP3.3: `cli retry` preserves analysis fragment cache by default and clears it with `--clear-cache`."""
    import asyncio
    from uuid import uuid4

    from linguary.adapters.config import load_settings
    from linguary.composition import build_application
    from linguary.domain import (
        AnalysisPassKind,
        AnalysisPassResult,
        ErrorType,
        ProcessingTask,
        TaskError,
        TaskStatus,
        TaskType,
    )

    config = _config(tmp_path, llm_model="fake-model", api_key="secret")
    book_id = "retry-cache-book"
    assert main(["--config", str(config), "import", str(SYNTHETIC_EPUB), "--book-id", book_id]) == 0
    assert main(["--config", str(config), "worker", "--once"]) == 0
    capsys.readouterr()

    task_id = str(uuid4())
    app = build_application(settings=load_settings(config_path=config))
    try:
        with app.unit_of_work.transaction() as tx:
            doc = tx.documents.get(book_id)
            assert doc is not None
            section_id = doc.ordered_sections[0].id
            for kind, scope_key, sha_char in (
                (AnalysisPassKind.NARRATIVE, "", "1"),
                (AnalysisPassKind.NARRATIVE, "frag:0", "2"),
                (AnalysisPassKind.PEDAGOGICAL, "frag:0", "3"),
            ):
                tx.analyses.save_pass(
                    AnalysisPassResult(
                        id=str(uuid4()),
                        book_id=book_id,
                        section_id=section_id,
                        pass_kind=kind,
                        scope_key=scope_key,
                        input_sha256=sha_char * 64,
                        output_sha256=sha_char * 64,
                        conditions_id=None,
                        attempt_id=None,
                        payload={"schema_version": 1, "pass_kind": kind.value},
                    )
                )
            tx.tasks.add(
                ProcessingTask(
                    id=task_id,
                    book_id=book_id,
                    task_type=TaskType.ANALYZE_CHAPTER,
                    scope_id=section_id,
                    status=TaskStatus.FAILED,
                    attempts=1,
                    last_error=TaskError(ErrorType.VALIDATION, "ошибка анализа", "detail"),
                )
            )
    finally:
        asyncio.run(app.aclose())

    # 1. Default `cli retry` preserves fragment passes.
    assert main(["--config", str(config), "retry", task_id]) == 0
    out_default = capsys.readouterr().out
    assert f"Task: {task_id}" in out_default
    assert "Status: pending" in out_default

    app_check = build_application(settings=load_settings(config_path=config))
    try:
        with app_check.unit_of_work.transaction() as tx:
            passes = tx.analyses.list_passes_for_section(book_id, section_id)
            assert {(p.pass_kind, p.scope_key) for p in passes} == {
                (AnalysisPassKind.NARRATIVE, ""),
                (AnalysisPassKind.NARRATIVE, "frag:0"),
                (AnalysisPassKind.PEDAGOGICAL, "frag:0"),
            }
            task = tx.tasks.get(task_id)
            assert task is not None
            tx.tasks.save(
                task.start().fail(TaskError(ErrorType.VALIDATION, "повторная ошибка", "detail"))
            )
    finally:
        asyncio.run(app_check.aclose())

    # 2. `cli retry --clear-cache` clears fragment passes while preserving whole-chapter NARRATIVE.
    assert main(["--config", str(config), "retry", task_id, "--clear-cache"]) == 0
    out_cleared = capsys.readouterr().out
    assert f"Task: {task_id}" in out_cleared
    assert "Status: pending" in out_cleared

    app_after = build_application(settings=load_settings(config_path=config))
    try:
        with app_after.unit_of_work.transaction() as tx:
            passes_after = tx.analyses.list_passes_for_section(book_id, section_id)
            assert [(p.pass_kind, p.scope_key) for p in passes_after] == [
                (AnalysisPassKind.NARRATIVE, ""),
            ]
    finally:
        asyncio.run(app_after.aclose())


def test_wp5_cli_analyze_pronunciation_variant_auto_detects_uk_and_falls_back_to_us(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """WP5.1: `cli analyze --pronunciation-help --pronunciation-variant auto` resolves en-GB -> uk and plain en -> us."""
    import asyncio

    from epub_builder import build_epub, xhtml_document
    from linguary.adapters.config import load_settings
    from linguary.adapters.epub.inline import read_source_language_tag
    from linguary.composition import build_application

    uk_epub = build_epub(
        tmp_path / "uk-cli.epub",
        language="en-GB",
        documents={"chapter1.xhtml": xhtml_document("<h1>Ch 1</h1><p>Hello world.</p>")},
        toc=[("chapter1.xhtml", "Ch 1")],
    )
    en_epub = build_epub(
        tmp_path / "en-cli.epub",
        language="en",
        documents={"chapter1.xhtml": xhtml_document("<h1>Ch 1</h1><p>Hello world.</p>")},
        toc=[("chapter1.xhtml", "Ch 1")],
    )
    assert read_source_language_tag(str(uk_epub)) == "en-GB"
    assert read_source_language_tag(str(en_epub)) == "en"
    assert read_source_language_tag(str(tmp_path / "missing.epub")) is None

    config = _config(tmp_path, llm_model="gpt-test", api_key="test-key")
    assert main(["--config", str(config), "import", str(uk_epub), "--book-id", "book-uk"]) == 0
    assert main(["--config", str(config), "import", str(en_epub), "--book-id", "book-en"]) == 0
    assert main(["--config", str(config), "worker", "--once"]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "--config",
                str(config),
                "analyze",
                "book-uk",
                "--reader-level",
                "a2",
                "--pronunciation-help",
                "--pronunciation-variant",
                "auto",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--config",
                str(config),
                "analyze",
                "book-en",
                "--reader-level",
                "a2",
                "--pronunciation-help",
                "--pronunciation-variant",
                "auto",
            ]
        )
        == 0
    )

    app = build_application(settings=load_settings(config_path=config))
    try:
        with app.unit_of_work.read_transaction() as tx:
            uk_cond = tx.work.latest_conditions("book-uk")
            en_cond = tx.work.latest_conditions("book-en")
            assert uk_cond is not None
            assert uk_cond.pronunciation_help_enabled is True
            assert uk_cond.pronunciation_variant == "uk"
            assert en_cond is not None
            assert en_cond.pronunciation_help_enabled is True
            assert en_cond.pronunciation_variant == "us"
    finally:
        asyncio.run(app.aclose())





