"""Интеграционные проверки CLI поверх настоящей сборки приложения."""

from __future__ import annotations

from pathlib import Path

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


def _config(
    tmp_path: Path, *, llm_model: str = "", api_key: str | None = None
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
    path = tmp_path / "config.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
