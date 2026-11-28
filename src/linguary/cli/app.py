"""Command-line interface.

The CLI serves the local Web UI (`serve`) and provides diagnostics and local execution
of the same use cases that the Web UI itself uses: book import, task queue processing,
retry of failed tasks, running analysis and processing, and technical export in
`structure_only` mode.
"""

from __future__ import annotations

import argparse
import json
import asyncio
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import uvicorn

from linguary.adapters.config import (
    CONFIG_PATH_VARIABLE,
    SUPPORTED_READER_LEVELS,
    SettingsError,
    load_settings,
)
from linguary.adapters.epub.inline import read_source_language_tag
from linguary.application.book_conditions import ConditionsError
from linguary.application.book_overview import book_overview
from linguary.application.errors import ApplicationError
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.application.use_cases.export_book import ExportBookCommand, ExportMode
from linguary.application.use_cases.pause_book import PauseBookCommand, ResumeBookCommand
from linguary.application.use_cases.register_book import RegisterBookCommand
from linguary.application.use_cases.resolve_correction_candidate import (
    CorrectionCandidateNotFound,
    ResolveCorrectionCommand,
)
from linguary.application.use_cases.restart_translation import RestartTranslationCommand
from linguary.application.use_cases.retry_task import RetryTaskCommand
from linguary.composition import build_application, drain_tasks, run_worker
from linguary.domain import BookStatus, DomainError, TaskStatus, TaskType, detect_english_variant
from linguary.domain.book_conditions import WorkStateError
from linguary.domain.illustrations import (
    IllustrationError,
    IllustrationReasonCode,
    IllustrationRegistryKind,
    IllustrationSettings,
    IllustrationValidationError,
    normalize_model_name,
)
from linguary.web.illustration_schemas import (
    IllustrationRegistryResponse,
    IllustrationViewResponse,
)
from linguary.web.schemas import EpubCheckConfigResponse, LLMConfigResponse


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except IllustrationError as error:
        if getattr(args, "json_output", False):
            print(
                json.dumps({"detail": {"reason_code": error.reason_code}}, ensure_ascii=False),
                file=sys.stderr,
            )
        else:
            print(f"{error.reason_code}: {error}", file=sys.stderr)
        return 1
    except (ApplicationError, DomainError, SettingsError, OSError) as error:
        if getattr(args, "json_output", False):
            print(
                json.dumps(
                    {"detail": {"reason_code": "error", "message": str(error)}},
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
        else:
            print(str(error), file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    """Command-line argument parser."""
    reader_parent = argparse.ArgumentParser(add_help=False)
    reader_parent.add_argument(
        "--reader-level",
        choices=sorted(SUPPORTED_READER_LEVELS),
        help="reader level (a2, b1, b2, c1)",
    )

    parser = argparse.ArgumentParser(prog="linguary", parents=[reader_parent])
    parser.add_argument("--config", help="path to TOML configuration file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser(
        "serve", parents=[reader_parent], help="run local Web UI"
    )
    serve.add_argument(
        "--reload",
        action="store_true",
        help="restart the server when source files change (development only)",
    )
    serve.set_defaults(handler=_serve)

    doctor = subparsers.add_parser(
        "doctor", parents=[reader_parent], help="check local configuration"
    )
    doctor.set_defaults(handler=_doctor)

    import_book = subparsers.add_parser(
        "import", parents=[reader_parent], help="import EPUB and enqueue import task"
    )
    import_book.add_argument("epub", help="path to EPUB file")
    import_book.add_argument("--book-id", help="book identifier")
    import_book.add_argument("--source-language", default="en", help="source language")
    import_book.add_argument("--target-language", default="ru", help="target language")
    import_book.set_defaults(handler=_import)

    analyze = subparsers.add_parser(
        "analyze", parents=[reader_parent], help="enqueue book analysis tasks"
    )
    analyze.add_argument("book_id")
    analyze.add_argument(
        "--english-help",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable simple English explanations",
    )
    analyze.add_argument(
        "--pronunciation-help",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable pronunciation hints",
    )
    analyze.add_argument(
        "--pronunciation-variant",
        choices=["uk", "us", "auto"],
        default=None,
        help=(
            "pronunciation variant when pronunciation help is enabled "
            "(uk, us, or auto: detect from EPUB dc:language, fallback to us)"
        ),
    )
    analyze.add_argument(
        "--seed-first-chapter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="analyze the first substantive chapter before enqueuing remaining chapters",
    )
    analyze.set_defaults(handler=_analyze)

    process = subparsers.add_parser(
        "process", parents=[reader_parent], help="enqueue book processing tasks"
    )
    process.add_argument("book_id")
    process.set_defaults(handler=_process)

    pause = subparsers.add_parser(
        "pause", parents=[reader_parent], help="pause claiming queued tasks for a book"
    )
    pause.add_argument("book_id")
    pause.set_defaults(handler=_pause)

    resume = subparsers.add_parser(
        "resume", parents=[reader_parent], help="resume claiming queued tasks for a book"
    )
    resume.add_argument("book_id")
    resume.set_defaults(handler=_resume)

    restart_translation = subparsers.add_parser(
        "restart-translation",
        parents=[reader_parent],
        help="restart unfinished scopes of a translation recovery root",
    )
    restart_translation.add_argument("book_id")
    restart_translation.add_argument("root_id")
    restart_translation.add_argument(
        "--expected-generation",
        type=int,
        required=True,
        help="expected generation of the recovery root",
    )
    restart_translation.add_argument(
        "--acknowledge-unknown-outcome",
        action="store_true",
        help="confirm restart after an unknown provider outcome (possible duplicate charge)",
    )
    restart_translation.set_defaults(handler=_restart_translation)

    worker = subparsers.add_parser(
        "worker", parents=[reader_parent], help="execute local task queue"
    )
    worker.add_argument("--once", action="store_true", help="drain queue and exit")
    worker.set_defaults(handler=_worker)

    retry = subparsers.add_parser(
        "retry", parents=[reader_parent], help="retry failed task"
    )
    retry.add_argument("task_id")
    retry.add_argument(
        "--acknowledge-unknown-outcome",
        action="store_true",
        help="confirm another chapter-correction send after an unknown provider outcome",
    )
    retry.add_argument(
        "--clear-cache",
        action="store_true",
        help="clear cached intermediate fragment passes before retrying an analysis task",
    )
    retry.set_defaults(handler=_retry)

    export = subparsers.add_parser(
        "export", parents=[reader_parent], help="assemble book EPUB"
    )
    export.add_argument("book_id")
    export.add_argument("--output", required=True, help="output EPUB file path")
    export.add_argument(
        "--mode",
        choices=[mode.value for mode in ExportMode],
        default=ExportMode.FULL.value,
        help="EPUB assembly mode: full (default) or structure_only",
    )
    export.add_argument(
        "--strict-corrections",
        action="store_true",
        help="fail with exit code 1 if awaiting_human correction candidates exist",
    )
    export.set_defaults(handler=_export)

    glossary = subparsers.add_parser(
        "glossary", parents=[reader_parent], help="inspect and approve book glossary items"
    )
    glossary_actions = glossary.add_subparsers(dest="glossary_action", required=True)
    glossary_list = glossary_actions.add_parser(
        "list", parents=[reader_parent], help="list glossary review items"
    )
    glossary_list.add_argument("book_id")
    glossary_list.add_argument(
        "--pending-only",
        action="store_true",
        help="show only unresolved items requiring review",
    )
    glossary_list.add_argument("--json", dest="json_output", action="store_true")
    glossary_list.set_defaults(handler=_glossary_list)

    glossary_accept_all = glossary_actions.add_parser(
        "accept-all",
        parents=[reader_parent],
        help="accept all unresolved glossary items requiring review",
    )
    glossary_accept_all.add_argument("book_id")
    glossary_accept_all.add_argument("--json", dest="json_output", action="store_true")
    glossary_accept_all.set_defaults(handler=_glossary_accept_all)

    corrections = subparsers.add_parser(
        "corrections", parents=[reader_parent], help="inspect and resolve chapter correction candidates"
    )
    corrections_actions = corrections.add_subparsers(dest="corrections_action", required=True)
    corrections_list = corrections_actions.add_parser(
        "list", parents=[reader_parent], help="list correction candidates for active run"
    )
    corrections_list.add_argument("book_id")
    corrections_list.add_argument(
        "--awaiting-only",
        action="store_true",
        help="show only candidates in awaiting_human state",
    )
    corrections_list.add_argument("--json", dest="json_output", action="store_true")
    corrections_list.set_defaults(handler=_corrections_list)

    corrections_resolve = corrections_actions.add_parser(
        "resolve",
        parents=[reader_parent],
        help="accept or keep a correction candidate awaiting human review",
    )
    corrections_resolve.add_argument("book_id")
    corrections_resolve.add_argument("candidate_id")
    corrections_resolve.add_argument(
        "--action",
        choices=["accept", "keep"],
        required=True,
        help="human decision: accept proposal or keep current translation",
    )
    corrections_resolve.add_argument("--reason", default=None, help="decision rationale")
    corrections_resolve.add_argument("--proposal-sha256", default=None)
    corrections_resolve.add_argument("--input-sha256", default=None)
    corrections_resolve.add_argument("--json", dest="json_output", action="store_true")
    corrections_resolve.set_defaults(handler=_corrections_resolve)
    diagnostics = subparsers.add_parser('diagnostics', help='local diagnostic metadata and bodies')
    actions = diagnostics.add_subparsers(dest='diagnostics_action', required=True)
    for action in ('list', 'show', 'purge'):
        command = actions.add_parser(action)
        command.add_argument('book_id')
        if action == 'show':
            command.add_argument('call_id')
        if action == 'purge':
            command.add_argument('--call-id')
        command.set_defaults(handler=_diagnostics)

    illustrations = subparsers.add_parser(
        "illustrations",
        parents=[reader_parent],
        help="optional book illustration prompt preparation",
    )
    ill_actions = illustrations.add_subparsers(dest="illustrations_action", required=True)

    ill_settings = ill_actions.add_parser(
        "settings",
        parents=[reader_parent],
        help="update book illustration settings without starting preparation",
    )
    ill_settings.add_argument("book_id")
    ill_settings.add_argument("--expected-revision", type=int, required=True)
    enable_group = ill_settings.add_mutually_exclusive_group()
    enable_group.add_argument("--enable", dest="enabled", action="store_true", default=None)
    enable_group.add_argument("--disable", dest="enabled", action="store_false")
    ill_settings.add_argument("--max-per-chapter", type=int, default=None)
    ill_settings.add_argument("--model-reference", default=None)
    ill_settings.add_argument("--model-prompts", default=None)
    ill_settings.add_argument("--json", dest="json_output", action="store_true")
    ill_settings.set_defaults(handler=_illustrations_settings)

    ill_prepare = ill_actions.add_parser(
        "prepare",
        parents=[reader_parent],
        help="start or expand illustration preparation for a book or chapter",
    )
    ill_prepare.add_argument("book_id")
    ill_prepare.add_argument("--expected-revision", type=int, default=None)
    ill_prepare.add_argument("--expected-generation", dest="expected_generation_id", default=None)
    ill_prepare.add_argument("--section-id", default=None)
    ill_prepare.add_argument("--force-new-generation", action="store_true")
    ill_prepare.add_argument("--json", dest="json_output", action="store_true")
    ill_prepare.set_defaults(handler=_illustrations_prepare)

    ill_retry = ill_actions.add_parser(
        "retry",
        parents=[reader_parent],
        help="retry failed retryable scopes of the active illustration generation",
    )
    ill_retry.add_argument("book_id")
    ill_retry.add_argument("--expected-generation", dest="expected_generation_id", required=True)
    ill_retry.add_argument("--expected-revision", type=int, default=None)
    ill_retry.add_argument("--section-id", default=None)
    ill_retry.add_argument("--json", dest="json_output", action="store_true")
    ill_retry.set_defaults(handler=_illustrations_retry)

    ill_status = ill_actions.add_parser(
        "status",
        parents=[reader_parent],
        help="show book illustration preparation status",
    )
    ill_status.add_argument("book_id")
    ill_status.add_argument("--generation-id", default=None)
    ill_status.add_argument("--offset", type=int, default=0)
    ill_status.add_argument("--limit", type=int, default=50)
    ill_status.add_argument("--json", dest="json_output", action="store_true")
    ill_status.set_defaults(handler=_illustrations_status)

    ill_show = ill_actions.add_parser(
        "show",
        parents=[reader_parent],
        help="show illustration prompts for a single chapter",
    )
    ill_show.add_argument("book_id")
    ill_show.add_argument("section_id")
    ill_show.add_argument("--generation-id", default=None)
    ill_show.add_argument("--json", dest="json_output", action="store_true")
    ill_show.set_defaults(handler=_illustrations_show)

    ill_registry = ill_actions.add_parser(
        "registry",
        parents=[reader_parent],
        help="show read-only book illustration registry entries",
    )
    ill_registry.add_argument("book_id")
    ill_registry.add_argument("--generation-id", default=None)
    ill_registry.add_argument("--entity-id", default=None)
    ill_registry.add_argument(
        "--kind",
        choices=[member.value for member in IllustrationRegistryKind],
        default=None,
    )
    ill_registry.add_argument("--offset", type=int, default=0)
    ill_registry.add_argument("--limit", type=int, default=50)
    ill_registry.add_argument("--json", dest="json_output", action="store_true")
    ill_registry.set_defaults(handler=_illustrations_registry)
    return parser


def _serve(args: argparse.Namespace) -> int:
    """Run local Web UI at the address specified in settings (PRD § 4.1, § 21.6).

    The address is retrieved from the `[server]` section rather than CLI flags:
    the application displays the same address on the configuration screen and uses
    it to enforce allowed hostnames in the `Host` header, so they must not diverge.

    The application is assembled by the factory inside the server process and cannot
    see CLI flags: an explicitly specified config file is passed to it via an
    environment variable — just as it is provided to any other run.
    """
    settings = load_settings(config_path=args.config)
    if args.config:
        os.environ[CONFIG_PATH_VARIABLE] = args.config
    if getattr(args, "reader_level", None):
        os.environ["LINGUARY_READING_READER_LEVEL"] = args.reader_level
    uvicorn.run(
        "linguary.web.app:create_app",
        factory=True,
        reload=args.reload,
        reload_dirs=(
            [str(Path(__file__).resolve().parents[1])] if args.reload else None
        ),
        proxy_headers=False,
        host=settings.server.host,
        port=settings.server.port,
    )
    return 0


def _doctor(args: argparse.Namespace) -> int:
    """Print local environment status without secrets."""
    application = _application(args)
    try:
        llm = LLMConfigResponse.of_settings(application.settings)
        epubcheck = EpubCheckConfigResponse.of_availability(
            application.epubcheck.availability()
        )
        print(f"LLM: {'ready' if llm.is_ready else 'not configured'}")
        print(f"LLM provider: {llm.provider}")
        print(f"LLM model: {llm.model or 'not set'}")
        print(f"LLM api key: {'set' if llm.has_api_key else 'not set'}")
        print(f"EPUBCheck: {'available' if epubcheck.available else 'unavailable'}")
        if epubcheck.reason:
            print(f"EPUBCheck reason: {epubcheck.reason}")
        print(f"Data root: {application.paths.data_root}")
        print(f"Database: {application.paths.database_path}")
        print(f"Server: {application.settings.server.base_url}")
        print(f"Reader level: {application.settings.reading.normalized_level}")
        return 0
    finally:
        asyncio.run(application.aclose())


def _import(args: argparse.Namespace) -> int:
    """Copy EPUB to the data directory and enqueue import task."""
    source = Path(args.epub).expanduser()
    if source.suffix.lower() != ".epub":
        raise ApplicationError("импорт принимает только EPUB-файлы")
    book_id = args.book_id or uuid4().hex
    application = _application(args)
    try:
        target = application.paths.source_epub(book_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        book = application.register_book(
            RegisterBookCommand(
                book_id=book_id,
                source_epub_path=str(target),
                title=source.stem,
                source_language=args.source_language,
                target_language=args.target_language,
            )
        )
        task = asyncio.run(application.start_import(book.id))
        print(f"Book: {book.id}")
        print(f"Status: {book.status.value}")
        print(f"Task: {task.id}")
        return 0
    finally:
        asyncio.run(application.aclose())


def _analyze(args: argparse.Namespace) -> int:
    """Verify LLM settings, optionally pin initial conditions, and enqueue analysis."""
    application = _application(args)
    try:
        application.settings.llm.ensure_ready()
        reader_level = getattr(args, "reader_level", None)
        english_help = getattr(args, "english_help", None)
        pronunciation_help = getattr(args, "pronunciation_help", None)
        pronunciation_variant = getattr(args, "pronunciation_variant", None)
        seed_first_chapter = getattr(args, "seed_first_chapter", None)
        if pronunciation_variant == "auto":
            with application.unit_of_work.read_transaction() as tx:
                book = tx.books.get(args.book_id)
            raw_lang = (
                read_source_language_tag(book.source_epub_path)
                if book is not None
                else None
            )
            pronunciation_variant = detect_english_variant(raw_lang) or "us"
        if any(
            value is not None
            for value in (
                reader_level,
                english_help,
                pronunciation_help,
                pronunciation_variant,
                seed_first_chapter,
            )
        ):
            try:
                application.book_conditions.ensure_initial_for_analysis(
                    args.book_id,
                    reader_level=reader_level,
                    english_help=english_help,
                    pronunciation_help_enabled=pronunciation_help,
                    pronunciation_variant=pronunciation_variant,
                    seed_first_chapter=seed_first_chapter,
                    default_seed_first_chapter=application.settings.checks.seed_first_chapter,
                    validate_conditions=application.book_runtime.settings_for,
                )
            except (ConditionsError, WorkStateError) as error:
                raise ApplicationError(str(error)) from error
        task = asyncio.run(application.analysis.start(args.book_id))
        print(f"Book: {args.book_id}")
        print(f"Task: {task.id if task is not None else 'none'}")
        return 0
    finally:
        asyncio.run(application.aclose())


def _pause(args: argparse.Namespace) -> int:
    """Pause claiming pending tasks for a book."""
    application = _application(args)
    try:
        if application.pause_book is None:
            raise ApplicationError("пауза очереди недоступна")
        result = application.pause_book(PauseBookCommand(book_id=args.book_id))
        print(f"Book: {result.book_id}")
        print(f"Pause requested: {result.pause_requested}")
        print(f"Paused: {result.is_paused}")
        return 0
    finally:
        asyncio.run(application.aclose())


def _resume(args: argparse.Namespace) -> int:
    """Resume claiming pending tasks for a book."""
    application = _application(args)
    try:
        if application.resume_book is None:
            raise ApplicationError("возобновление очереди недоступно")
        result = application.resume_book(ResumeBookCommand(book_id=args.book_id))
        if application.settings.llm.is_ready:
            with application.unit_of_work.read_transaction() as tx:
                book = tx.books.get(args.book_id)
                tasks = tx.tasks.list_for_book(args.book_id)
                run = tx.runs.active_for_book(args.book_id)
                stopped_roots = (
                    any(
                        getattr(root, "superseded_by", None) is None
                        and root.status == "stopped"
                        for root in tx.work.list_roots(args.book_id, run_id=run.id)
                    )
                    if run is not None
                    else False
                )
            try:
                if book is not None and book.status is BookStatus.ANALYZING:
                    if not any(
                        task.task_type is TaskType.ANALYZE_CHAPTER
                        and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.FAILED)
                        for task in tasks
                    ):
                        asyncio.run(application.analysis.start(args.book_id))
                elif (
                    book is not None
                    and book.status is BookStatus.PROCESSING
                    and run is not None
                    and not stopped_roots
                ):
                    if not any(
                        task.task_type
                        in (
                            TaskType.PROCESS_TRANSLATION_UNIT,
                            TaskType.CHAPTER_CORRECTION,
                            TaskType.REGENERATE_BLOCK,
                        )
                        and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.FAILED)
                        for task in tasks
                    ):
                        asyncio.run(application.processing.start(args.book_id))
            except (ApplicationError, WorkStateError):
                pass
        print(f"Book: {result.book_id}")
        print(f"Paused: {result.is_paused}")
        return 0
    finally:
        asyncio.run(application.aclose())


def _warn_if_book_paused(unit_of_work: Any, book_id: str) -> None:
    open_tx = getattr(unit_of_work, "read_transaction", unit_of_work.transaction)
    with open_tx() as tx:
        is_book_paused = getattr(tx.work, "is_book_paused", None)
        paused = bool(is_book_paused(book_id)) if callable(is_book_paused) else False
    if paused:
        print(
            f"Warning: очередь задач книги {book_id} на паузе (pause_requested); "
            f"задачи ожидают до выполнения 'linguary resume {book_id}'.",
            file=sys.stderr,
        )


def _restart_translation(args: argparse.Namespace) -> int:
    """Restart unfinished scopes of a translation recovery root."""
    application = _application(args)
    try:
        result = application.restart_translation(
            RestartTranslationCommand(
                book_id=args.book_id,
                root_id=args.root_id,
                expected_generation=args.expected_generation,
                acknowledge_unknown_outcome=args.acknowledge_unknown_outcome,
            )
        )
        _warn_if_book_paused(application.unit_of_work, args.book_id)
        print(f"Book: {args.book_id}")
        print(f"Successor root: {result.successor_root_id}")
        print(f"Task: {result.task_id}")
        return 0
    finally:
        asyncio.run(application.aclose())


def _process(args: argparse.Namespace) -> int:
    """Freeze glossary and enqueue processing tasks."""
    application = _application(args)
    try:
        run = application.start_processing()(StartProcessingCommand(book_id=args.book_id))
        with application.unit_of_work.transaction() as tx:
            set_book_paused = getattr(tx.work, "set_book_paused", None)
            if callable(set_book_paused):
                set_book_paused(args.book_id, None, False)
        tasks = asyncio.run(application.processing.start(args.book_id))
        print(f"Book: {args.book_id}")
        print(f"Run: {run.id}")
        print(f"Tasks: {len(tasks)}")
        return 0
    finally:
        asyncio.run(application.aclose())


def _worker(args: argparse.Namespace) -> int:
    """Run local worker."""
    application = _application(args)
    async def execute() -> int:
        try:
            if args.once:
                tasks = await drain_tasks(application)
                ill_types = {
                    TaskType.PREPARE_ILLUSTRATION_REFERENCE,
                    TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
                }
                primary_failed = [
                    task
                    for task in tasks
                    if task.last_error is not None and task.task_type not in ill_types
                ]
                illustration_failed = [
                    task
                    for task in tasks
                    if task.last_error is not None and task.task_type in ill_types
                ]
                print(f"Tasks finished: {len(tasks)}")
                print(f"Tasks failed: {len(primary_failed)}")
                print(f"Optional illustration tasks failed: {len(illustration_failed)}")
                return 1 if primary_failed else 0
            await run_worker(application)
            return 0
        finally:
            await application.aclose()

    return asyncio.run(execute())


def _retry(args: argparse.Namespace) -> int:
    """Re-enqueue a failed task."""
    application = _application(args)
    try:
        task = application.retry_task(RetryTaskCommand(
            task_id=args.task_id,
            acknowledge_unknown_outcome=args.acknowledge_unknown_outcome,
            clear_cache=args.clear_cache,
        ))
        _warn_if_book_paused(application.unit_of_work, task.book_id)
        print(f"Task: {task.id}")
        print(f"Status: {task.status.value}")
        return 0
    finally:
        asyncio.run(application.aclose())


def _export(args: argparse.Namespace) -> int:
    """Assemble EPUB in selected mode (full by default)."""
    application = _application(args)
    try:
        if getattr(args, "strict_corrections", False):
            with application.unit_of_work.read_transaction() as tx:
                run = tx.runs.active_for_book(args.book_id)
                pending = (
                    len(tx.work.list_correction_candidates(run.id, states=("awaiting_human",)))
                    if run is not None
                    else 0
                )
            if pending > 0:
                raise ApplicationError(
                    f"экспорт отклонён (--strict-corrections): книга {args.book_id} содержит "
                    f"непросмотренные предложения исправлений ({pending})"
                )
        result = application.export_book(
            ExportBookCommand(
                book_id=args.book_id,
                target_path=args.output,
                mode=ExportMode(args.mode),
            )
        )
        with application.unit_of_work.transaction() as tx:
            from linguary.application.illustrations import has_unfinished_illustration_work

            set_book_paused = getattr(tx.work, "set_book_paused", None)
            if callable(set_book_paused) and not has_unfinished_illustration_work(tx, args.book_id):
                set_book_paused(args.book_id, None, False)
        overview = book_overview(application.unit_of_work, args.book_id)
        if overview.pending_corrections > 0:
            print(
                f"Warning: книга содержит непросмотренные предложения исправлений ({overview.pending_corrections}).",
                file=sys.stderr,
            )
        print(f"Book: {result.book_id}")
        print(f"Export: {result.path}")
        print(
            "EPUBCheck: "
            f"{result.validation_status.value if result.validation_status else 'none'}"
        )
        return 0
    finally:
        asyncio.run(application.aclose())


def _glossary_list(args: argparse.Namespace) -> int:
    """List glossary review items for a book."""
    application = _application(args)
    try:
        review = application.approve_glossary.review(args.book_id)
        items = [
            item
            for item in review.items
            if not args.pending_only or (item.requires_review and not item.is_decided)
        ]
        pending_count = sum(
            1 for item in review.items if item.requires_review and not item.is_decided
        )
        decided_count = sum(1 for item in review.items if item.is_decided)
        auto_mandatory_count = sum(1 for item in review.items if item.is_auto_mandatory)
        awaiting_review_count = sum(1 for item in review.items if item.awaits_decision)
        suggested_count = sum(
            1
            for item in review.items
            if not item.is_decided
            and not item.is_auto_mandatory
            and not item.awaits_decision
        )
        if args.json_output:
            payload = {
                "book_id": args.book_id,
                "total": len(review.items),
                "pending": pending_count,
                "summary": {
                    "decided": decided_count,
                    "auto_mandatory": auto_mandatory_count,
                    "awaiting_review": awaiting_review_count,
                    "suggested": suggested_count,
                },
                "items": [
                    {
                        "concept_id": item.concept_id,
                        "source_text": item.source_text,
                        "type": item.concept_type.value,
                        "suggested_translation": item.suggested_translation,
                        "requires_review": item.requires_review,
                        "is_decided": item.is_decided,
                        "auto_mandatory": item.is_auto_mandatory,
                        "decision": (
                            item.decision.status.value if item.decision is not None else None
                        ),
                    }
                    for item in items
                ],
            }
            print(json.dumps(payload, ensure_ascii=False))
            return 0
        print(f"Book: {args.book_id}")
        print(
            f"Items: {len(items)}/{len(review.items)} "
            f"(pending: {pending_count}, decided: {decided_count}, "
            f"auto_mandatory: {auto_mandatory_count}, awaiting_review: {awaiting_review_count}, "
            f"suggested: {suggested_count})"
        )
        for item in items:
            if item.decision is not None:
                state = item.decision.status.value
            elif item.is_auto_mandatory:
                state = "auto_mandatory"
            else:
                state = "pending"
            flag = " [review]" if item.requires_review and not item.is_decided else ""
            print(
                f"- {item.concept_id}: {item.source_text} -> {item.suggested_translation} "
                f"({state}){flag}"
            )
        return 0
    finally:
        asyncio.run(application.aclose())


def _glossary_accept_all(args: argparse.Namespace) -> int:
    """Accept all unresolved glossary items requiring review."""
    application = _application(args)
    try:
        accepted = application.approve_glossary.accept_all_awaiting(args.book_id)
        review = application.approve_glossary.review(args.book_id)
        remaining_pending = sum(
            1 for item in review.items if item.requires_review and not item.is_decided
        )
        if args.json_output:
            print(
                json.dumps(
                    {
                        "book_id": args.book_id,
                        "accepted_count": len(accepted),
                        "remaining_pending": remaining_pending,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        print(f"Book: {args.book_id}")
        print(f"Accepted: {len(accepted)}")
        print(f"Remaining pending: {remaining_pending}")
        return 0
    finally:
        asyncio.run(application.aclose())


def _extract_candidate_text(mapping: Mapping[str, Any]) -> str | None:
    raw_text = mapping.get("translated_text")
    if isinstance(raw_text, str) and raw_text.strip():
        return raw_text
    seg_translations = mapping.get("segment_translations")
    if isinstance(seg_translations, Sequence) and not isinstance(
        seg_translations, (str, bytes)
    ):
        parts = [
            item
            for item in seg_translations
            if isinstance(item, str) and item.strip()
        ]
        if parts:
            return " ".join(parts)
    segments = mapping.get("segments")
    if isinstance(segments, Sequence) and not isinstance(segments, (str, bytes)):
        parts = [
            seg["translated_text"]
            for seg in segments
            if isinstance(seg, Mapping)
            and isinstance(seg.get("translated_text"), str)
            and seg["translated_text"].strip()
        ]
        if parts:
            return " ".join(parts)
    return None


def _candidate_snippet(candidate: Any) -> tuple[str | None, str | None]:
    prop_text: str | None = None
    if isinstance(candidate.payload, Mapping):
        for key in ("result", "proposed_result", "proposal"):
            source = candidate.payload.get(key)
            if isinstance(source, Mapping):
                extracted = _extract_candidate_text(source)
                if extracted is not None:
                    prop_text = extracted
                    break
    review = candidate.review if isinstance(candidate.review, Mapping) else None
    reason = review.get("reason") if isinstance(review, Mapping) else None
    return reason, prop_text


def _corrections_list(args: argparse.Namespace) -> int:
    """List chapter correction candidates for the active processing run."""
    application = _application(args)
    try:
        with application.unit_of_work.read_transaction() as tx:
            if tx.books.get(args.book_id) is None:
                raise ApplicationError(f"книга {args.book_id} не найдена")
            run = tx.runs.active_for_book(args.book_id)
            states = ("awaiting_human",) if args.awaiting_only else None
            candidates = (
                tx.work.list_correction_candidates(run.id, states=states)
                if run is not None
                else ()
            )
        serialized = []
        for cand in candidates:
            review_reason, proposed_text = _candidate_snippet(cand)
            serialized.append(
                {
                    "candidate_id": cand.candidate_id,
                    "section_id": cand.section_id,
                    "block_id": cand.block_id,
                    "state": cand.state,
                    "proposal_sha256": cand.proposal_sha256,
                    "input_sha256": cand.input_sha256,
                    "review_reason": review_reason,
                    "proposed_text": proposed_text,
                }
            )
        if args.json_output:
            print(
                json.dumps(
                    {
                        "book_id": args.book_id,
                        "run_id": run.id if run is not None else None,
                        "items": serialized,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        print(f"Book: {args.book_id}")
        print(f"Candidates: {len(serialized)}")
        for item in serialized:
            reason_suffix = f" — {item['review_reason']}" if item["review_reason"] else ""
            print(
                f"- {item['candidate_id']} [{item['state']}] "
                f"{item['section_id']}/{item['block_id']}{reason_suffix}"
            )
        return 0
    finally:
        asyncio.run(application.aclose())


def _corrections_resolve(args: argparse.Namespace) -> int:
    """Resolve an awaiting_human correction candidate using ResolveCorrectionCandidate."""
    application = _application(args)
    try:
        resolver = application.resolve_correction_candidate
        if resolver is None:
            raise ApplicationError("разрешение предложений исправлений недоступно")
        reason = (args.reason if args.reason is not None else "Решено через CLI").strip()
        if not reason:
            raise ApplicationError("причина решения не должна быть пустой")
        proposal_sha256 = args.proposal_sha256
        input_sha256 = args.input_sha256
        if proposal_sha256 is None or input_sha256 is None:
            with application.unit_of_work.read_transaction() as tx:
                candidate = tx.work.get_correction_candidate(args.candidate_id)
                if candidate is None or candidate.book_id != args.book_id:
                    raise CorrectionCandidateNotFound("предложение исправления не найдено")
                if proposal_sha256 is None:
                    proposal_sha256 = candidate.proposal_sha256
                if input_sha256 is None:
                    input_sha256 = candidate.input_sha256
        block_id = resolver(
            ResolveCorrectionCommand(
                book_id=args.book_id,
                candidate_id=args.candidate_id,
                proposal_sha256=proposal_sha256,
                input_sha256=input_sha256,
                action=args.action,
                reason=reason,
            )
        )
        if args.json_output:
            print(
                json.dumps(
                    {
                        "book_id": args.book_id,
                        "candidate_id": args.candidate_id,
                        "block_id": block_id,
                        "action": args.action,
                        "reason": reason,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        print(f"Book: {args.book_id}")
        print(f"Candidate: {args.candidate_id}")
        print(f"Block: {block_id}")
        print(f"Action: {args.action}")
        return 0
    finally:
        asyncio.run(application.aclose())


def _diagnostics(args: argparse.Namespace) -> int:
    """Local/container alternative to loopback-only HTTP; never invokes a model."""
    application = _application(args)
    try:
        service = application.diagnostics
        if args.diagnostics_action == 'list':
            value = service.list(args.book_id)
        elif args.diagnostics_action == 'show':
            value = service.read(args.book_id, args.call_id)
        else:
            value = dict(purged=service.purge(args.book_id, call_id=args.call_id))
        print(json.dumps(value, ensure_ascii=False))
        return 0
    finally:
        asyncio.run(application.aclose())


def _print_illustration_view(view: Any, *, as_json: bool) -> None:
    dto = IllustrationViewResponse.of_view(view)
    if as_json:
        print(json.dumps(dto.model_dump(mode="json"), ensure_ascii=False))
        return
    print(f"Book: {dto.book_id}")
    print(f"State: {dto.state}")
    if dto.reason_code:
        print(f"Reason: {dto.reason_code}")
    print(f"Revision: {dto.settings_revision}")
    print(f"Enabled: {dto.enabled}")
    if dto.active_generation_id:
        print(f"Active generation: {dto.active_generation_id}")
    print(
        f"Source chapters: {dto.source_progress.ready_chapters}/{dto.source_progress.expected_chapters}"
    )
    if dto.selected_chapter is not None:
        ch = dto.selected_chapter
        print(f"Chapter: {ch.section_id} ({ch.state})")
        if ch.zero_reason_ru:
            print(f"Zero reason: {ch.zero_reason_ru}")
        for moment in ch.moments:
            print(f"- {moment.title_ru}: {moment.prompt_en}")


def _illustrations_settings(args: argparse.Namespace) -> int:
    application = _application(args)

    async def execute() -> int:
        try:
            current = application.illustrations.get(args.book_id)
            with application.unit_of_work.read_transaction() as tx:
                head = tx.illustrations.get_head(args.book_id)
                existing_policy = head.effective_settings.policy
            enabled = current.settings.enabled if args.enabled is None else bool(args.enabled)
            max_per = (
                current.settings.max_per_chapter
                if args.max_per_chapter is None
                else args.max_per_chapter
            )
            model_ref = (
                current.settings.model_reference
                if args.model_reference is None
                else (None if args.model_reference == "" else normalize_model_name(args.model_reference))
            )
            model_prompts = (
                current.settings.model_prompts
                if args.model_prompts is None
                else (None if args.model_prompts == "" else normalize_model_name(args.model_prompts))
            )
            new_settings = IllustrationSettings(
                enabled=enabled,
                max_per_chapter=max_per,
                style=current.settings.style,
                model_reference=model_ref,
                model_prompts=model_prompts,
                policy=existing_policy,
            )
            application.illustrations.save_settings(
                args.book_id,
                expected_revision=args.expected_revision,
                settings=new_settings,
            )
            view = application.illustrations.get(args.book_id)
            _print_illustration_view(view, as_json=args.json_output)
            return 0
        finally:
            await application.aclose()

    return asyncio.run(execute())


def _illustrations_prepare(args: argparse.Namespace) -> int:
    application = _application(args)

    async def execute() -> int:
        try:
            expected_rev = args.expected_revision
            if expected_rev is None:
                expected_rev = application.illustrations.get(args.book_id).settings_revision
            view = await application.illustrations.prepare(
                args.book_id,
                expected_revision=expected_rev,
                expected_generation_id=args.expected_generation_id,
                section_id=args.section_id,
                force_new_generation=bool(args.force_new_generation),
            )
            _warn_if_book_paused(application.unit_of_work, args.book_id)
            _print_illustration_view(view, as_json=args.json_output)
            return 0
        finally:
            await application.aclose()

    return asyncio.run(execute())


def _illustrations_retry(args: argparse.Namespace) -> int:
    application = _application(args)

    async def execute() -> int:
        try:
            expected_rev = args.expected_revision
            if expected_rev is None:
                expected_rev = application.illustrations.get(args.book_id).settings_revision
            view = await application.illustrations.retry(
                args.book_id,
                expected_revision=expected_rev,
                expected_generation_id=args.expected_generation_id,
                section_id=args.section_id,
            )
            _warn_if_book_paused(application.unit_of_work, args.book_id)
            _print_illustration_view(view, as_json=args.json_output)
            return 0
        finally:
            await application.aclose()

    return asyncio.run(execute())


def _illustrations_status(args: argparse.Namespace) -> int:
    if args.offset < 0 or args.limit < 1 or args.limit > 50:
        raise IllustrationValidationError(
            IllustrationReasonCode.INVALID_REQUEST,
            "допустимы offset >= 0 и 1 <= limit <= 50",
        )
    application = _application(args)

    async def execute() -> int:
        try:
            view = application.illustrations.get(
                args.book_id,
                offset=args.offset,
                limit=args.limit,
                generation_id=args.generation_id,
            )
            _print_illustration_view(view, as_json=args.json_output)
            return 0
        finally:
            await application.aclose()

    return asyncio.run(execute())


def _illustrations_show(args: argparse.Namespace) -> int:
    application = _application(args)

    async def execute() -> int:
        try:
            view = application.illustrations.get(
                args.book_id,
                section_id=args.section_id,
                generation_id=args.generation_id,
            )
            _print_illustration_view(view, as_json=args.json_output)
            return 0
        finally:
            await application.aclose()

    return asyncio.run(execute())


def _illustrations_registry(args: argparse.Namespace) -> int:
    if args.offset < 0 or args.limit < 1 or args.limit > 100:
        raise IllustrationValidationError(
            IllustrationReasonCode.INVALID_REQUEST,
            "допустимы offset >= 0 и 1 <= limit <= 100",
        )
    application = _application(args)

    async def execute() -> int:
        try:
            kind = IllustrationRegistryKind(args.kind) if args.kind is not None else None
            reg = application.illustrations.get_registry(
                args.book_id,
                generation_id=args.generation_id,
                entity_id=args.entity_id,
                kind=kind,
                offset=args.offset,
                limit=args.limit,
            )
            dto = IllustrationRegistryResponse.of_view(reg)
            if args.json_output:
                print(json.dumps(dto.model_dump(mode="json"), ensure_ascii=False))
            else:
                print(f"Book: {dto.book_id}")
                print(f"Items: {len(dto.items)}/{dto.pagination.total}")
                for item in dto.items:
                    print(f"- [{item.kind}] {item.item_key}: {item.summary_ru}")
            return 0
        finally:
            await application.aclose()

    return asyncio.run(execute())


def _application(args: argparse.Namespace):
    """Assemble application based on CLI settings."""
    settings = load_settings(config_path=args.config)
    reader_level = getattr(args, "reader_level", None)
    if reader_level is not None:
        settings = replace(
            settings,
            reading=replace(settings.reading, reader_level=reader_level),
        )
    return build_application(settings=settings)


__all__ = ["main"]
