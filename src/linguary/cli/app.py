"""Командная строка приложения.

CLI поднимает локальный Web UI (`serve`) и нужен для диагностики и локального выполнения
тех же сценариев, что использует сам Web UI: импорт книги, запуск очереди задач, повтор
неуспешной задачи, запуск анализа и обработки, а также технический экспорт в режиме
`structure_only`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import uvicorn

from linguary.adapters.config import CONFIG_PATH_VARIABLE, SettingsError, load_settings
from linguary.application.errors import ApplicationError
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.application.use_cases.export_book import ExportBookCommand, ExportMode
from linguary.application.use_cases.register_book import RegisterBookCommand
from linguary.application.use_cases.retry_task import RetryTaskCommand
from linguary.composition import build_application, drain_tasks, run_worker
from linguary.web.schemas import EpubCheckConfigResponse, LLMConfigResponse


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ApplicationError, SettingsError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    """Парсер командной строки."""
    parser = argparse.ArgumentParser(prog="linguary")
    parser.add_argument("--config", help="путь к TOML-файлу настроек")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="запустить локальный Web UI")
    serve.set_defaults(handler=_serve)

    doctor = subparsers.add_parser("doctor", help="проверить локальную конфигурацию")
    doctor.set_defaults(handler=_doctor)

    import_book = subparsers.add_parser("import", help="загрузить EPUB и поставить импорт")
    import_book.add_argument("epub", help="путь к EPUB-файлу")
    import_book.add_argument("--book-id", help="идентификатор книги")
    import_book.add_argument("--source-language", default="en", help="исходный язык")
    import_book.add_argument("--target-language", default="ru", help="целевой язык")
    import_book.set_defaults(handler=_import)

    analyze = subparsers.add_parser("analyze", help="поставить задачи анализа книги")
    analyze.add_argument("book_id")
    analyze.set_defaults(handler=_analyze)

    process = subparsers.add_parser("process", help="поставить задачи обработки книги")
    process.add_argument("book_id")
    process.set_defaults(handler=_process)

    worker = subparsers.add_parser("worker", help="выполнить локальную очередь задач")
    worker.add_argument("--once", action="store_true", help="выполнить очередь и выйти")
    worker.set_defaults(handler=_worker)

    retry = subparsers.add_parser("retry", help="повторить неуспешную задачу")
    retry.add_argument("task_id")
    retry.set_defaults(handler=_retry)

    export = subparsers.add_parser("export", help="собрать EPUB без LLM-результатов")
    export.add_argument("book_id")
    export.add_argument("--output", required=True, help="куда записать EPUB")
    export.set_defaults(handler=_export)
    return parser


def _serve(args: argparse.Namespace) -> int:
    """Поднимает локальный Web UI на адресе из настроек (PRD § 4.1, § 21.6).

    Адрес берётся из секции `[server]`, а не из аргументов запуска: тот же адрес
    приложение показывает на экране конфигурации и по нему же ограничивает имена
    в заголовке `Host`, поэтому расходиться им нельзя.

    Приложение собирает фабрика уже внутри сервера, и аргументов командной строки она
    не видит: явно указанный файл настроек доходит до неё переменной окружения — тем же
    способом, каким его задают любому другому запуску.
    """
    settings = load_settings(config_path=args.config)
    if args.config:
        os.environ[CONFIG_PATH_VARIABLE] = args.config
    uvicorn.run(
        "linguary.web.app:create_app",
        factory=True,
        host=settings.server.host,
        port=settings.server.port,
    )
    return 0


def _doctor(args: argparse.Namespace) -> int:
    """Печатает состояние локального окружения без секретов."""
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
        return 0
    finally:
        asyncio.run(application.aclose())


def _import(args: argparse.Namespace) -> int:
    """Копирует EPUB в каталог данных и ставит задачу импорта."""
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
    """Проверяет LLM-настройку и ставит первую задачу анализа."""
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
    """Замораживает глоссарий и ставит задачи обработки."""
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
    """Запускает локальный worker."""
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
    """Возвращает неуспешную задачу в очередь."""
    application = _application(args)
    try:
        task = application.retry_task(RetryTaskCommand(task_id=args.task_id))
        print(f"Task: {task.id}")
        print(f"Status: {task.status.value}")
        return 0
    finally:
        asyncio.run(application.aclose())


def _export(args: argparse.Namespace) -> int:
    """Собирает EPUB в режиме `structure_only` для технической проверки."""
    application = _application(args)
    try:
        result = application.export_book(
            ExportBookCommand(
                book_id=args.book_id,
                target_path=args.output,
                mode=ExportMode.STRUCTURE_ONLY,
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


def _application(args: argparse.Namespace):
    """Собирает приложение по настройкам CLI."""
    settings = load_settings(config_path=args.config)
    return build_application(settings=settings)


__all__ = ["main"]
