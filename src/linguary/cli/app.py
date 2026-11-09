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
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import uvicorn

from linguary.adapters.config import (
    CONFIG_PATH_VARIABLE,
    SUPPORTED_READER_LEVELS,
    SettingsError,
    load_settings,
)
from linguary.application.book_conditions import ConditionsError
from linguary.application.book_overview import BookNotFoundError, book_overview
from linguary.application.errors import ApplicationError
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.application.use_cases.export_book import ExportBookCommand, ExportMode
from linguary.application.use_cases.pause_book import PauseBookCommand, ResumeBookCommand
from linguary.application.use_cases.register_book import RegisterBookCommand
from linguary.application.use_cases.restart_translation import RestartTranslationCommand
from linguary.application.use_cases.retry_task import RetryTaskCommand
from linguary.composition import build_application, drain_tasks, run_worker
from linguary.domain import BookStatus, TaskStatus, TaskType
from linguary.domain.book_conditions import WorkStateError
from linguary.web.schemas import EpubCheckConfigResponse, LLMConfigResponse


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ApplicationError, SettingsError, OSError) as error:
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
        choices=["uk", "us"],
        default=None,
        help="pronunciation variant when pronunciation help is enabled (uk or us)",
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
    export.set_defaults(handler=_export)
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
                with application.unit_of_work.transaction() as tx:
                    book = tx.books.get(args.book_id)
                    if book is None:
                        raise BookNotFoundError(f"книги {args.book_id} нет в хранилище")
                    existing = tx.work.latest_conditions(args.book_id)
                    is_clean_imported = (
                        book.status is BookStatus.IMPORTED
                        and not tx.analyses.list_for_book(args.book_id)
                        and tx.runs.active_for_book(args.book_id) is None
                    )
                    if existing is None:
                        if not is_clean_imported:
                            raise ConditionsError("legacy_choice_required")
                        pron_enabled = bool(pronunciation_help)
                        conditions = application.book_conditions.create(
                            tx,
                            book,
                            application.runtime_settings.revision,
                            reader_level=reader_level,
                            english_help=bool(english_help),
                            pronunciation_help_enabled=pron_enabled,
                            pronunciation_variant=(
                                pronunciation_variant
                                if (pron_enabled or pronunciation_help is None)
                                else None
                            ),
                            seed_first_chapter=seed_first_chapter,
                        )
                        application.book_runtime.settings_for(conditions)
                    else:
                        existing_seed = bool(
                            (existing.payload.get("checks") or {}).get(
                                "seed_first_chapter",
                                application.settings.checks.seed_first_chapter,
                            )
                        )
                        eff_reader_level = (
                            reader_level
                            if reader_level is not None
                            else existing.payload.get("reader_level")
                        )
                        eff_english_help = (
                            english_help
                            if english_help is not None
                            else bool(existing.payload.get("english_help", False))
                        )
                        eff_pron_enabled = (
                            pronunciation_help
                            if pronunciation_help is not None
                            else bool(existing.payload.get("pronunciation_help_enabled", False))
                        )
                        if not eff_pron_enabled:
                            eff_pron_variant = (
                                pronunciation_variant
                                if pronunciation_help is None
                                else None
                            )
                        else:
                            eff_pron_variant = (
                                pronunciation_variant
                                if pronunciation_variant is not None
                                else existing.payload.get("pronunciation_variant")
                            )
                        eff_seed = (
                            seed_first_chapter
                            if seed_first_chapter is not None
                            else existing_seed
                        )
                        diverges = (
                            eff_reader_level != existing.payload.get("reader_level")
                            or eff_english_help != bool(existing.payload.get("english_help", False))
                            or eff_pron_enabled != bool(existing.payload.get("pronunciation_help_enabled", False))
                            or eff_pron_variant != existing.payload.get("pronunciation_variant")
                            or eff_seed != existing_seed
                        )
                        if diverges:
                            if not is_clean_imported:
                                raise ConditionsError("conditions_already_selected")
                            conditions = application.book_conditions.create(
                                tx,
                                book,
                                application.runtime_settings.revision,
                                reader_level=eff_reader_level,
                                english_help=eff_english_help,
                                pronunciation_help_enabled=eff_pron_enabled,
                                pronunciation_variant=eff_pron_variant,
                                seed_first_chapter=eff_seed,
                            )
                            application.book_runtime.settings_for(conditions)
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


def _restart_translation(args: argparse.Namespace) -> int:
    """Restart unfinished scopes of a translation recovery root."""
    application = _application(args)
    try:
        result = application.restart_translation(
            RestartTranslationCommand(
                book_id=args.book_id,
                root_id=args.root_id,
                expected_generation=args.expected_generation,
            )
        )
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
                failed = [task for task in tasks if task.last_error is not None]
                print(f"Tasks finished: {len(tasks)}")
                print(f"Tasks failed: {len(failed)}")
                return 1 if failed else 0
            await run_worker(application)
            return 0
        finally:
            await application.aclose()

    return asyncio.run(execute())


def _retry(args: argparse.Namespace) -> int:
    """Re-enqueue a failed task."""
    application = _application(args)
    try:
        task = application.retry_task(RetryTaskCommand(task_id=args.task_id))
        print(f"Task: {task.id}")
        print(f"Status: {task.status.value}")
        return 0
    finally:
        asyncio.run(application.aclose())


def _export(args: argparse.Namespace) -> int:
    """Assemble EPUB in selected mode (full by default)."""
    application = _application(args)
    try:
        result = application.export_book(
            ExportBookCommand(
                book_id=args.book_id,
                target_path=args.output,
                mode=ExportMode(args.mode),
            )
        )
        with application.unit_of_work.transaction() as tx:
            set_book_paused = getattr(tx.work, "set_book_paused", None)
            if callable(set_book_paused):
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
