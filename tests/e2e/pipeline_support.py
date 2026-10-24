"""Common test harness for end-to-end scenarios: model replies, book reading, and EPUBCheck stub.

Both end-to-end scenarios — application-level (`test_mvp_pipeline.py`) and Web UI
(`test_web_pipeline.py`) — run against the same synthetic book and fixture replies,
so reply construction lives here rather than being duplicated across test files.

Replies are prepared only after import: their count and composition depend on how many
chapters and text blocks were parsed from the actual EPUB, rather than a hardcoded constant.

This module intentionally lacks the `test_` prefix so pytest will not collect it as a test file.
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

#: Small synthetic book for end-to-end testing (PRD § 28.1).
SYNTHETIC_EPUB = FIXTURE_ROOT / "epub" / "synthetic_story.epub"

#: Stored model responses used to assemble replies matching chapter sizes.
LLM_FIXTURE_ROOT = FIXTURE_ROOT / "llm"

#: Clean EPUBCheck report with zero messages: book verified without warnings.
CLEAN_EPUBCHECK_REPORT: dict[str, object] = {
    "checker": {"checkerVersion": "5.1.0"},
    "messages": [],
}


def llm_replies(
    document: CanonicalDocument, *, broken_reply_before_processing: bool = False
) -> tuple[FakeReply | Exception, ...]:
    """Analysis and translation replies sized to actual chapter dimensions of imported book.

    First one analysis reply per chapter, then one translation reply per chapter. The
    `broken_reply_before_processing` flag inserts a malformed JSON reply in between to test
    retry of failed calls and logging failure in the model call log (PRD § 16, § 24).
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
    """JSON fixture of model response."""
    return json.loads((LLM_FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def sequential_ids(prefix: str) -> Callable[[], str]:
    """Predictable task identifiers: `<prefix>-1`, `<prefix>-2`, ..."""
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}-{counter}"

    return next_id


def exported_text(path: Path) -> str:
    """All visible text of assembled EPUB as a single string."""
    return "\n".join(
        "".join(root.itertext()) for root in exported_documents(path).values()
    )


def exported_documents(path: Path) -> dict[str, etree._Element]:
    """XHTML documents of assembled EPUB."""
    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
    with zipfile.ZipFile(path) as archive:
        return {
            name: etree.fromstring(archive.read(name), parser=parser)
            for name in archive.namelist()
            if name.endswith((".xhtml", ".html", ".htm"))
        }


@dataclass(frozen=True, slots=True)
class EpubCheckStub:
    """Mock EPUBCheck: adapter settings and recorded command line."""

    settings: EpubCheckSettings
    argv_path: Path

    @property
    def was_called(self) -> bool:
        """Whether adapter invoked the tool at least once."""
        return self.argv_path.is_file()

    def argv(self) -> list[str]:
        """Arguments with which adapter invoked the tool."""
        return json.loads(self.argv_path.read_text(encoding="utf-8"))


def install_epubcheck_stub(directory: Path) -> EpubCheckStub:
    """Installs a Java wrapper stub returning a clean report into directory.

    Real EPUBCheck is not invoked: Java may not be installed on the machine, and the quality
    of the external tool is not under test. The stub mimics real execution — parses `--json`,
    writes report there, and exits with 0 — so the pipeline's Validate step runs to completion,
    including report parsing.
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
    """Analysis reply for one chapter: treatment plan covers all its textual blocks."""
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
    """Translation reply for one chapter with translation, hints, and language note."""
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
    """Text block aliases for the chapter as seen by the model."""
    return [f"b{index}" for index, _ in enumerate(textual_blocks(section), start=1)]
