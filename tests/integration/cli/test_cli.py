"""Интеграционные проверки CLI поверх настоящей сборки приложения."""

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
    """`doctor` показывает состояние окружения, но не печатает ключ провайдера."""
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
    assert "secret-value-that-must-stay-hidden" not in output


def test_import_and_worker_once_import_the_book(tmp_path: Path, capsys) -> None:
    """CLI импортирует EPUB через задачу очереди и разовую прогонку worker-а."""
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
    """Технический экспорт CLI собирает EPUB без LLM-результатов."""
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
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output_path.is_file()
    assert "Export:" in output


def test_analyze_without_llm_configuration_fails_before_queueing(
    tmp_path: Path, capsys
) -> None:
    """Команда анализа без LLM-настройки не ставит задачу и не вызывает модель."""
    config = _config(tmp_path)

    exit_code = main(["--config", str(config), "analyze", "unknown-book"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "LLM не настроена" in captured.err


def test_serve_starts_the_web_ui_on_the_configured_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`serve` слушает адрес из секции `[server]`, а не зашитый в команду запуска.

    Файл настроек уходит в окружение: фабрика создаёт приложение уже внутри сервера
    и аргументов командной строки не видит.
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
    """Пишет изолированный конфиг CLI-теста."""
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
