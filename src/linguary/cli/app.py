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
from linguary.application.errors import ApplicationError
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.application.use_cases.export_book import ExportBookCommand, ExportMode
from linguary.application.use_cases.register_book import RegisterBookCommand
from linguary.application.use_cases.retry_task import RetryTaskCommand
from linguary.composition import build_application, drain_tasks, run_worker
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
    analyze.set_defaults(handler=_analyze)

    process = subparsers.add_parser(
        "process", parents=[reader_parent], help="enqueue book processing tasks"
    )
    process.add_argument("book_id")
    process.set_defaults(handler=_process)

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
    """Verify LLM settings and enqueue the initial analysis task."""
    application = _application(args)
    try:
        application.settings.llm.ensure_ready()
        task = asyncio.run(application.analysis.start(args.book_id))
        print(f"Book: {args.book_id}")
        print(f"Task: {task.id if task is not None else 'none'}")
        return 0
    finally:
        asyncio.run(application.aclose())


def _process(args: argparse.Namespace) -> int:
    """Freeze glossary and enqueue processing tasks."""
    application = _application(args)
    try:
        run = application.start_processing()(StartProcessingCommand(book_id=args.book_id))
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
