"""Общая обвязка сквозных сценариев: ответы модели, чтение книги и заглушка EPUBCheck.

Оба сквозных сценария — прикладной (`test_mvp_pipeline.py`) и через Web UI
(`test_web_pipeline.py`) — работают на одной синтетической книге и одних и тех же
`fixture`-ответах, поэтому сборка ответов живёт здесь, а не дублируется в каждом файле.

Ответы готовятся только после импорта: их число и состав зависят от того, сколько глав
и текстовых блоков дал разбор настоящего EPUB, а не от заранее записанной константы.

Модуль намеренно без префикса `test_`: иначе pytest соберёт его как тестовый.
"""

from __future__ import annotations

import json
import stat
import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from linguary.adapters.llm import FakeReply
from linguary.adapters.validation import EpubCheckSettings
from linguary.application.context_builder import textual_blocks
from linguary.domain import CanonicalDocument, DocumentSection, TokenUsage

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"

#: Маленькая синтетическая книга сквозного теста (PRD § 28.1).
SYNTHETIC_EPUB = FIXTURE_ROOT / "epub" / "synthetic_story.epub"

#: Сохранённые ответы модели, из которых собираются реплики под размеры глав.
LLM_FIXTURE_ROOT = FIXTURE_ROOT / "llm"

#: Отчёт EPUBCheck без единого замечания: книга проверена и претензий нет.
CLEAN_EPUBCHECK_REPORT: dict[str, object] = {
    "checker": {"checkerVersion": "5.1.0"},
    "messages": [],
}


def llm_replies(
    document: CanonicalDocument, *, broken_reply_before_processing: bool = False
) -> tuple[FakeReply | Exception, ...]:
    """Ответы анализа и обработки под реальные размеры глав импортированной книги.

    Сначала по одному ответу анализа на главу, затем по одному ответу обработки. Флаг
    `broken_reply_before_processing` вставляет между ними ответ с оборванным JSON: так
    сценарий заодно проходит повтор неуспешного вызова и запись отказа в журнал модели
    (PRD § 16, § 24).
    """
    analysis = read_fixture("analysis-template.json")
    translation = read_fixture("translation-template.json")
    replies: list[FakeReply | Exception] = [
        _analysis_reply(section, analysis) for section in document.ordered_sections
    ]
    if broken_reply_before_processing:
        replies.append(FakeReply(content="{"))
    replies.extend(
        _translation_reply(_block_aliases(section), translation)
        for section in document.ordered_sections
    )
    return tuple(replies)


def read_fixture(name: str) -> dict[str, object]:
    """JSON-fixture ответа модели."""
    return json.loads((LLM_FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def sequential_ids(prefix: str) -> Callable[[], str]:
    """Предсказуемые идентификаторы задач: `<prefix>-1`, `<prefix>-2`, ..."""
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}-{counter}"

    return next_id


def exported_text(path: Path) -> str:
    """Весь видимый текст собранного EPUB одной строкой."""
    return "\n".join(
        "".join(root.itertext()) for root in exported_documents(path).values()
    )


def exported_documents(path: Path) -> dict[str, etree._Element]:
    """XHTML-документы собранного EPUB."""
    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
    with zipfile.ZipFile(path) as archive:
        return {
            name: etree.fromstring(archive.read(name), parser=parser)
            for name in archive.namelist()
            if name.endswith((".xhtml", ".html", ".htm"))
        }


@dataclass(frozen=True, slots=True)
class EpubCheckStub:
    """Подменённый EPUBCheck: настройки адаптера и запись его командной строки."""

    settings: EpubCheckSettings
    argv_path: Path

    @property
    def was_called(self) -> bool:
        """Запускал ли адаптер инструмент хотя бы раз."""
        return self.argv_path.is_file()

    def argv(self) -> list[str]:
        """Аргументы, с которыми адаптер запустил инструмент."""
        return json.loads(self.argv_path.read_text(encoding="utf-8"))


def install_epubcheck_stub(directory: Path) -> EpubCheckStub:
    """Кладёт в директорию заглушку Java и файл инструмента, отдающую чистый отчёт.

    Настоящий EPUBCheck в наборе не запускается: Java на машине может отсутствовать, а
    качество самого инструмента — не предмет этих тестов. Заглушка ведёт себя как
    настоящий вызов — разбирает `--json`, пишет туда отчёт и завершается нулём, — поэтому
    шаг `Validate` сквозного сценария проходит целиком, вплоть до разбора отчёта.
    """
    directory.mkdir(parents=True, exist_ok=True)
    argv_path = directory / "argv.json"
    report = json.dumps(CLEAN_EPUBCHECK_REPORT)
    script = directory / "java"
    script.write_text(
        "\n".join(
            (
                f"#!{sys.executable}",
                "import json, pathlib, sys",
                f"argv_path = pathlib.Path({str(argv_path)!r})",
                "argv = sys.argv[1:]",
                "argv_path.write_text(json.dumps(argv), encoding='utf-8')",
                f"report = {report!r}",
                "if '--json' in argv:",
                "    target = pathlib.Path(argv[argv.index('--json') + 1])",
                "    target.parent.mkdir(parents=True, exist_ok=True)",
                "    target.write_text(report, encoding='utf-8')",
                "sys.exit(0)",
                "",
            )
        ),
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    jar = directory / "epubcheck.jar"
    jar.write_bytes(b"")
    return EpubCheckStub(
        settings=EpubCheckSettings(
            java_executable=str(script),
            jar_path=str(jar),
            search_directories=(str(directory),),
        ),
        argv_path=argv_path,
    )


def _analysis_reply(
    section: DocumentSection, fixture: dict[str, object]
) -> FakeReply:
    """Ответ анализа одной главы: план обработки покрывает все её текстовые блоки."""
    return FakeReply.of_payload(
        {
            "translation_instructions": fixture["translation_instructions"],
            "chapter_summary": (
                f"{fixture['chapter_summary']} Chapter: {section.title}."
            ),
            "glossary_candidates": fixture["glossary_candidates"],
            "before_you_read": fixture["before_you_read"],
            "treatment_plan": [
                {
                    "source_block_id": alias,
                    "treatment": "TRANSLATE",
                }
                for alias in _block_aliases(section)
            ],
        }
    )


def _translation_reply(aliases: list[str], fixture: dict[str, object]) -> FakeReply:
    """Ответ обработки одной главы с переводом, подсказками и заметкой."""
    hint = fixture["glossary_hint"]
    note = fixture["language_note"]
    assert isinstance(hint, dict)
    assert isinstance(note, dict)
    return FakeReply.of_payload(
        {
            "blocks": [
                {
                    "source_block_id": alias,
                    "treatment": "TRANSLATE",
                    "translated_text": f"{fixture['translated_text']} ({alias}).",
                    "glossary_hints": [hint],
                }
                for alias in aliases
            ],
            "language_notes": [
                {
                    "source_block_id": aliases[0],
                    "original_excerpt": note["original_excerpt"],
                    "translated_excerpt": note["translated_excerpt"],
                    "note": note["note"],
                }
            ],
        },
        usage=TokenUsage(prompt_tokens=100, completion_tokens=200),
    )


def _block_aliases(section: DocumentSection) -> list[str]:
    """Псевдонимы текстовых блоков главы в том виде, в каком их видит модель."""
    return [f"b{index}" for index, _ in enumerate(textual_blocks(section), start=1)]
