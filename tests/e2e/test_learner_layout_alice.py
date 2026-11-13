"""Control book: Alice chapters 1–2 through import, analysis, processing, and export."""

from __future__ import annotations

import asyncio
import json
import posixpath
import re
import shutil
import socket
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from pipeline_support import (
    exported_documents,
    install_epubcheck_stub,
    plain_text,
    sequential_ids,
)

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings, WorkerPoolSettings
from linguary.adapters.llm.fake import FAKE_MODEL, FakeLLMClient
from linguary.adapters.llm.payloads import parse_payload
from linguary.adapters.validation import EpubCheckValidator
from linguary.application.context_builder import source_text_for_model, textual_blocks
from linguary.application.layout.book import build_book_layout, build_layout_inputs
from linguary.application.layout.units import LayoutUnit
from linguary.application.ports.llm import LLMRequest, LLMResponse, LLMResponseKind
from linguary.application.ports.persistence import ExportValidationStatus
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.application.use_cases.register_book import RegisterBookCommand
from linguary.composition import build_application, drain_tasks
from linguary.domain import (
    BlockTreatment,
    BlockType,
    DocumentBlock,
    SupportStatus,
    TaskStatus,
)

BOOK = "alice-layout"
DESCRIPTION = "Девочка, которая увидела кролика с часами."
SPEAKER = "Кролик"
POEM_MARK = "How doth"
POEM_QUOTE = "How doth the little crocodile"
SPEECH_PREFERENCE = "I shall be late"
FALSE_FRIEND_LABEL = "Ложный друг."
SPEECH_LABEL = "Речь персонажа."
PARODY_LABEL = "Пародия."
SUMMARY = "Alice follows a rabbit into a strange world."
INSTRUCTIONS = "Сохраняй имена и смысл."
PARODY_NOTE = "Это пародия на нравоучительный стих."
FRIEND_NOTE = "Слово похоже на русское, но значит другое."
SPEECH_NOTE = "Так говорит персонаж, а не рассказчик."
LATER_NAMES = (
    "Cheshire",
    "Hatter",
    "Queen of Hearts",
    "March Hare",
    "Mock Turtle",
    "Gryphon",
    "Caterpillar",
    "Duchess",
    "Dormouse",
    "Knave",
)
_WORD = re.compile(r"\b[A-Za-z]{6,}\b")
_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "epub" / "alice_in_wonderland_ch1_ch2.epub"
)


@dataclass(frozen=True, slots=True)
class Targets:
    """Blocks chosen from the imported book. Assertions rediscover them by these ids."""

    mixed_id: str
    poem_id: str
    poem_quote: str
    friend_block_id: str
    friend_quote: str
    speech_quote: str
    alice_block_id: str


@dataclass(frozen=True, slots=True)
class AliceRun:
    app: object
    stub: object
    export_path: Path
    document: object
    blocks: dict[str, DocumentBlock]
    targets: Targets


def _forbid_network(*args, **kwargs):
    raise AssertionError("End-to-end check must not make network calls")


@pytest.fixture(autouse=True)
def network_forbidden(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", _forbid_network)
    monkeypatch.setattr(socket.socket, "connect", _forbid_network)
    monkeypatch.setattr(httpx.Client, "send", _forbid_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", _forbid_network)


def _run(awaitable):
    return asyncio.run(awaitable)


def _finish(app):
    tasks = _run(drain_tasks(app))
    failed = [
        (task.task_type, task.status, task.last_error)
        for task in tasks
        if task.status is not TaskStatus.SUCCEEDED
    ]
    assert tasks and not failed, failed
    return tasks


def _content_sections(document):
    return tuple(
        section
        for section in document.ordered_sections
        if not document.is_front_matter(section) and not section.is_notes_only
    )


def _visible(block: DocumentBlock) -> str:
    return source_text_for_model(block)


def _once(text: str, quote: str) -> bool:
    return bool(quote) and text.count(quote) == 1


def _mixed_modes(block: DocumentBlock) -> tuple[str, ...] | None:
    """First sentence stays source-only. A later sentence is translated when every span has text."""
    spans = block.segments
    if len(spans) < 2:
        return None
    grouped: dict[int, list] = {}
    for span in spans:
        grouped.setdefault(span.sentence_index, []).append(span)
    if len(grouped) < 2:
        return None
    first = spans[0].sentence_index
    modes: dict[int, str] = {}
    for index, group in grouped.items():
        if index == first or any(not block.text[span.start : span.end].strip() for span in group):
            modes[index] = "READ_AS_IS"
        else:
            modes[index] = "TRANSLATE"
    chosen = tuple(modes[span.sentence_index] for span in spans)
    if "TRANSLATE" not in chosen or "READ_AS_IS" not in chosen:
        return None
    return chosen


def _discover(document) -> Targets:
    content = _content_sections(document)
    mixed = None
    stored = 0
    for section in content:
        for block in section.ordered_blocks:
            if block.type is not BlockType.PARAGRAPH or len(block.segments) < 2:
                continue
            stored += 1
            if mixed is None and _mixed_modes(block) is not None:
                mixed = block
    if stored == 0:
        raise AssertionError("importer stored no paragraph with two or more segments")
    if mixed is None:
        raise AssertionError("no stored paragraph has two sentence indexes that can take different modes")
    poems = [
        block
        for section in document.ordered_sections
        for block in section.ordered_blocks
        if block.type is BlockType.POEM and POEM_MARK in block.text
    ]
    if len(poems) != 1:
        raise AssertionError(f"expected one poem containing {POEM_MARK!r}, found {len(poems)}")
    poem = poems[0]
    if not _once(poem.text, POEM_QUOTE) or not _once(_visible(poem), POEM_QUOTE):
        raise AssertionError("parody quote is not a single span of the poem")
    friend = _false_friend(content, mixed.id)
    speech = _speech_quote(mixed)
    alice = _alice_block(content)
    return Targets(
        mixed_id=mixed.id,
        poem_id=poem.id,
        poem_quote=POEM_QUOTE,
        friend_block_id=friend[0].id,
        friend_quote=friend[1],
        speech_quote=speech,
        alice_block_id=alice.id,
    )


def _false_friend(sections, mixed_id: str) -> tuple[DocumentBlock, str]:
    for section in sections:
        for block in section.ordered_blocks:
            if block.type is not BlockType.PARAGRAPH or block.id == mixed_id:
                continue
            if len(block.segments) >= 2:
                continue
            visible = _visible(block)
            for match in _WORD.finditer(block.text):
                word = match.group(0)
                pattern = re.compile(rf"\b{re.escape(word)}\b")
                if len(pattern.findall(block.text)) == 1 and len(pattern.findall(visible)) == 1:
                    return block, word
    raise AssertionError("no single-span paragraph has a unique false-friend word")


def _speech_quote(block: DocumentBlock) -> str:
    modes = _mixed_modes(block)
    assert modes is not None
    visible = _visible(block)
    translated = [
        block.text[span.start : span.end]
        for span, mode in zip(block.segments, modes, strict=True)
        if mode == "TRANSLATE"
    ]
    if (
        _once(block.text, SPEECH_PREFERENCE)
        and _once(visible, SPEECH_PREFERENCE)
        and any(SPEECH_PREFERENCE in piece for piece in translated)
    ):
        return SPEECH_PREFERENCE
    for piece in translated:
        quote = piece.strip()
        if len(quote) < 4 or not re.search(r"[A-Za-z]", quote):
            continue
        if _once(block.text, quote) and _once(visible, quote):
            return quote
    raise AssertionError("no translated segment has a unique speech quote")


def _alice_block(sections) -> DocumentBlock:
    pattern = re.compile(r"\bAlice\b")
    for section in sections:
        for block in textual_blocks(section):
            if pattern.search(block.text) and pattern.search(_visible(block)):
                return block
    raise AssertionError("no content block contains Alice")


def _poem_translation(text: str) -> str:
    lines = ["" if not line.strip() else f"Строка {number}." for number, line in enumerate(text.split("\n"), start=1)]
    return "\n".join(lines)


def _quote_for(block_id: str, targets: Targets) -> tuple[str, str] | None:
    if block_id == targets.poem_id:
        return targets.poem_quote, PARODY_NOTE
    if block_id == targets.friend_block_id:
        return targets.friend_quote, FRIEND_NOTE
    if block_id == targets.mixed_id:
        return targets.speech_quote, SPEECH_NOTE
    return None


class AliceModel(FakeLLMClient):
    """One reply per request, scripted from the imported blocks."""

    __slots__ = ("document", "blocks", "front_ids", "notes_ids", "targets")

    def __init__(self) -> None:
        super().__init__()
        self.document = None
        self.blocks: dict[str, DocumentBlock] = {}
        self.front_ids: frozenset[str] = frozenset()
        self.notes_ids: frozenset[str] = frozenset()
        self.targets: Targets | None = None

    async def generate[PayloadT](self, request: LLMRequest[PayloadT]) -> LLMResponse[PayloadT]:
        if not request.ordered_block_ids:
            raise AssertionError("ordered_block_ids is empty")
        targets = self.targets
        if targets is None:
            raise AssertionError("targets are unset")
        if request.kind == LLMResponseKind.CHAPTER_ANALYSIS:
            if request.contract_version != 5:
                raise AssertionError(request.contract_version)
            payload = _analysis_payload(request, self.blocks, targets)
        elif request.kind == LLMResponseKind.TRANSLATION_RESULT:
            if request.contract_version != 3:
                raise AssertionError(request.contract_version)
            payload = _processing_payload(request, self.blocks, self.front_ids, self.notes_ids, targets)
        elif request.kind == LLMResponseKind.CHAPTER_CORRECTION:
            payload = {
                "blocks": [
                    {"block_id": block_id, "status": "unchanged"}
                    for block_id in request.ordered_block_ids
                ]
            }
        else:
            raise AssertionError(request.kind)
        if request.pronunciation_candidates_version == 1:
            payload["pronunciation_candidates"] = []
        elif request.pronunciation_candidates_version is not None:
            raise AssertionError(request.pronunciation_candidates_version)
        content = json.dumps(payload, ensure_ascii=False)
        self.requests.append(request)
        parsed = parse_payload(
            request.kind,
            content,
            contract_version=request.contract_version,
            name_translation_policy_version=request.name_translation_policy_version,
            pronunciation_candidates_version=request.pronunciation_candidates_version,
        )
        return LLMResponse(
            payload=parsed,  # type: ignore[arg-type]
            provider=self.provider,
            model=FAKE_MODEL,
            truncated=False,
            usage=None,
        )


def _analysis_payload(request, blocks: dict[str, DocumentBlock], targets: Targets) -> dict:
    # Replies follow the blocks the importer stored. Segment boundaries stay as imported.
    ids = tuple(request.ordered_block_ids)
    present = set(ids)
    plan = []
    for number, block_id in enumerate(ids, start=1):
        block = blocks[block_id]
        item: dict = {"source_block_id": f"b{number}", "treatment": "TRANSLATE"}
        if block_id == targets.mixed_id:
            modes = _mixed_modes(block)
            assert modes is not None
            item["segments"] = [
                {"index": index, "treatment": mode}
                for index, mode in enumerate(modes, start=1)
            ]
            item["treatment"] = "TRANSLATE"
        plan.append(item)
    difficulties = []
    facts = []
    if targets.poem_id in present:
        difficulties.append(_difficulty(ids, targets.poem_id, "parody", targets.poem_quote, "Name the verse this poem parodies."))
    if targets.friend_block_id in present:
        difficulties.append(_difficulty(ids, targets.friend_block_id, "false_friend", targets.friend_quote, "Show that the word is a false friend."))
    if targets.mixed_id in present:
        difficulties.append(_difficulty(ids, targets.mixed_id, "speech_style", targets.speech_quote, "Mark this as a character's speech."))
        alias = f"b{ids.index(targets.mixed_id) + 1}"
        anchor = {"block_id": alias, "quote": targets.speech_quote}
        facts.append({
            "id": "speaker-rabbit",
            "kind": "speaker",
            "anchors": [anchor],
            "subject_anchor": anchor,
            "value": SPEAKER,
            "certainty": "inferred",
        })
    glossary = []
    if targets.alice_block_id in present:
        candidate = {
            "source_text": "Alice",
            "type": "person",
            "suggested_translation": "Алиса",
            "confidence": 0.95,
            "model_flagged_review": False,
            "note": "Имя девочки.",
            "description": DESCRIPTION,
        }
        if request.name_translation_policy_version == 1:
            candidate["name_translation_assessment"] = {
                "applied_method": "sound",
                "eligible_without_review": True,
                "review_reasons": [],
                "rationale": "Имя передано по звучанию.",
                "opaque_name": True,
            }
        elif request.name_translation_policy_version is not None:
            raise AssertionError(request.name_translation_policy_version)
        glossary.append(candidate)
    return {
        "translation_instructions": INSTRUCTIONS,
        "chapter_summary": SUMMARY,
        "glossary_candidates": glossary,
        "treatment_plan": plan,
        "before_you_read": [],
        "content_plan": {
            "version": 3,
            "profile": "content-plan-v3",
            "difficulties": difficulties,
            "before_you_read_evidence": [],
            "context_facts": facts,
        },
    }


def _difficulty(ids: tuple[str, ...], block_id: str, kind: str, quote: str, description: str) -> dict:
    alias = f"b{ids.index(block_id) + 1}"
    return {
        "id": kind,
        "owner_block_id": alias,
        "anchors": [{"block_id": alias, "quote": quote}],
        "kind": kind,
        "priority": "useful",
        "help_kind": "explanation",
        "required_points": [{"id": "point", "description": description}],
    }


def _processing_payload(request, blocks, front_ids, notes_ids, targets: Targets) -> dict:
    # Front matter and notes pages stay READ_AS_IS, matching the synthetic analysis.
    ids = tuple(request.ordered_block_ids)
    rows = []
    notes = []
    for number, block_id in enumerate(ids, start=1):
        alias = f"b{number}"
        block = blocks[block_id]
        kept = block_id in front_ids or block_id in notes_ids
        if kept:
            rows.append({
                "source_block_id": alias,
                "treatment": "READ_AS_IS",
                "translated_text": None,
                "glossary_hints": [],
            })
            continue
        modes = _mixed_modes(block) if block_id == targets.mixed_id else None
        if len(block.segments) >= 2:
            indexes = range(1, len(block.segments) + 1)
            if modes is not None:
                indexes = [index for index, mode in enumerate(modes, start=1) if mode == "TRANSLATE"]
            rows.append({
                "source_block_id": alias,
                "treatment": "TRANSLATE",
                "translated_text": None,
                "glossary_hints": [],
                "segments": [{"index": index, "translated_text": "Фраза."} for index in indexes],
            })
        else:
            text = _poem_translation(block.text) if block.type is BlockType.POEM else "Фраза."
            rows.append({
                "source_block_id": alias,
                "treatment": "TRANSLATE",
                "translated_text": text,
                "glossary_hints": [],
            })
        quoted = _quote_for(block_id, targets)
        if quoted is not None:
            notes.append({
                "source_block_id": alias,
                "original_excerpt": quoted[0],
                "translated_excerpt": None,
                "note": quoted[1],
            })
    payload = {"blocks": rows, "language_notes": notes, "content_evidence": []}
    return payload


def _pipeline(root: Path) -> AliceRun:
    client = AliceModel()
    stub = install_epubcheck_stub(root / "epubcheck")
    app = build_application(
        settings=AppSettings(
            llm=LLMSettings(model="fake-model", api_key="nonworking-test-key", max_attempts=1),
            storage=StorageSettings(data_root=str(root / "data")),
            worker=WorkerPoolSettings(concurrency=1),
        ),
        llm_client=client,
        task_id_factory=sequential_ids("task"),
        epubcheck=EpubCheckValidator(settings=stub.settings),
        content_plan_profile="content-plan-v3",
    )
    try:
        source = app.paths.source_epub(BOOK)
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_FIXTURE, source)
        app.register_book(RegisterBookCommand(
            book_id=BOOK,
            source_epub_path=str(source),
            title="Alice's Adventures in Wonderland",
            source_language="en",
            target_language="ru",
        ))
        _run(app.start_import(BOOK))
        _finish(app)
        with app.unit_of_work.transaction() as transaction:
            document = transaction.documents.get(BOOK)
        assert document is not None
        blocks = {
            block.id: block
            for section in document.ordered_sections
            for block in section.ordered_blocks
        }
        targets = _discover(document)
        client.document = document
        client.blocks = blocks
        client.front_ids = frozenset(
            block.id
            for section in document.ordered_sections
            if document.is_front_matter(section)
            for block in section.ordered_blocks
        )
        client.notes_ids = frozenset(
            block.id
            for section in document.ordered_sections
            if section.is_notes_only
            for block in section.ordered_blocks
        )
        client.targets = targets
        _run(app.analysis.start(BOOK))
        _finish(app)
        app.approve_glossary.accept_all_awaiting(BOOK)
        review = app.approve_glossary.review(BOOK)
        assert review.is_complete, [item.source_text for item in review.awaiting_decision]
        app.start_processing()(StartProcessingCommand(book_id=BOOK))
        _run(app.processing.start(BOOK))
        _finish(app)
        _run(app.start_export(BOOK))
        _finish(app)
        return AliceRun(
            app=app,
            stub=stub,
            export_path=app.paths.export_epub(BOOK),
            document=document,
            blocks=blocks,
            targets=targets,
        )
    except Exception:
        app.dispose()
        raise


@pytest.fixture(scope="module")
def alice_book(tmp_path_factory):
    root = tmp_path_factory.mktemp("alice-layout")
    with ExitStack() as stack:
        for target in (
            "socket.create_connection",
            "socket.socket.connect",
            "httpx.Client.send",
            "httpx.AsyncClient.send",
        ):
            stack.enter_context(patch(target, _forbid_network))
        run = _pipeline(root)
        try:
            yield run
        finally:
            run.app.dispose()


def _layout(run: AliceRun):
    return build_book_layout(build_layout_inputs(run.app.unit_of_work, BOOK))


def _local(element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _classes(element) -> list[str]:
    return (element.get("class") or "").split()


def _chapter_root(documents: dict, section):
    name = posixpath.normpath(section.source_resource_path.replace("\\", "/").strip())
    found = [root for path, root in documents.items() if path.replace("\\", "/").endswith(name)]
    assert len(found) == 1, name
    return found[0]


def _unit_divs(root):
    return [
        element
        for element in root.iter()
        if _local(element) == "div" and "lg-unit" in _classes(element)
    ]


def _writer_emits_units(block: DocumentBlock, *, nested: bool) -> bool:
    # An empty footnote is a shell. An opaque leaf is copied markup, not a reading unit.
    if block.support_status is SupportStatus.PRESERVE_OPAQUE and not nested:
        return False
    if block.type is BlockType.FOOTNOTE and not block.text:
        return False
    return True


def _reading_units(section, layout) -> list[LayoutUnit]:
    children: dict[str, list[DocumentBlock]] = {}
    for block in section.ordered_blocks:
        if block.parent_id is not None:
            children.setdefault(block.parent_id, []).append(block)
    chapter = layout.chapters[section.id]
    found: list[LayoutUnit] = []

    def walk(blocks) -> None:
        for block in blocks:
            nested = tuple(children.get(block.id, ()))
            if _writer_emits_units(block, nested=bool(nested)):
                found.extend(chapter.units[block.id])
            walk(nested)

    walk(tuple(block for block in section.ordered_blocks if block.parent_id is None))
    return found


def _source_element(unit_div):
    for child in unit_div:
        if "lg-src" in _classes(child) or _local(child) in _HEADINGS:
            return child
    raise AssertionError("reading unit has no source")


def _translation_plain(paragraph) -> str:
    # The writer keeps the translation in the speaker span's tail, after one space.
    chunks: list[str] = []
    if paragraph.text:
        chunks.append(paragraph.text)
    for child in paragraph:
        if "lg-speaker" in _classes(child):
            tail = child.tail or ""
            chunks.append(tail[1:] if tail.startswith(" ") else tail)
            continue
        if _local(child) == "sup":
            continue
        chunks.append(plain_text(child))
        if child.tail:
            chunks.append(child.tail)
    return "".join(chunks)


def _labeled(root, class_name: str) -> list[str]:
    return [
        element.text
        for element in root.iter()
        if _local(element) == "span" and class_name in _classes(element)
    ]


def _assert_unit(unit_div, unit: LayoutUnit, block: DocumentBlock) -> None:
    classes = _classes(unit_div)
    assert "lg-unit" in classes
    assert ("lg-cont" in classes) is unit.continues
    assert ("lg-end" in classes) is unit.ends_paragraph
    source = _source_element(unit_div)
    assert plain_text(source).replace("\n", "") == block.text[unit.start : unit.end].replace("\n", "")
    translated = unit.translation.strip() if unit.translation else ""
    rows = [child for child in unit_div if "lg-tr" in _classes(child)]
    if translated:
        assert len(rows) == 1
        assert _translation_plain(rows[0]).replace("\n", "") == unit.translation.replace("\n", "")
    else:
        assert rows == []
    shown = unit.speaker if translated else None
    assert _labeled(unit_div, "lg-speaker") == ([shown] if shown else [])
    assert _labeled(unit_div, "lg-kind") == [
        label for numbered in unit.notes for label in numbered.note.labels
    ]


def test_alice_export_matches_layout_structure(alice_book: AliceRun):
    layout = _layout(alice_book)
    documents = exported_documents(alice_book.export_path)
    mixed = alice_book.blocks[alice_book.targets.mixed_id]
    with alice_book.app.unit_of_work.transaction() as transaction:
        analyses = transaction.analyses.list_for_book(BOOK)
    item = next(
        plan_item
        for analysis in analyses
        for plan_item in analysis.treatment_plan.items
        if plan_item.block_id == mixed.id
    )
    assert BlockTreatment.READ_AS_IS in item.segment_treatments
    assert BlockTreatment.TRANSLATE in item.segment_treatments
    assert len(mixed.segments) >= 2
    assert len({span.sentence_index for span in mixed.segments}) >= 2
    for section in alice_book.document.ordered_sections:
        divs = _unit_divs(_chapter_root(documents, section))
        expected = _reading_units(section, layout)
        assert len(divs) == len(expected), (section.title, len(divs), len(expected))
        for unit_div, unit in zip(divs, expected, strict=True):
            _assert_unit(unit_div, unit, alice_book.blocks[unit.block_id])


def _breaks(element) -> list:
    return [child for child in element.iter() if _local(child) == "br"]


def _adjacent_break(element) -> bool:
    for node in element.iter():
        children = list(node)
        for index, child in enumerate(children[:-1]):
            if _local(child) != "br" or _local(children[index + 1]) != "br":
                continue
            if not (child.tail or "").strip():
                return True
    return False


def test_alice_poem_line_subscript_and_parody(alice_book: AliceRun):
    poem = alice_book.blocks[alice_book.targets.poem_id]
    assert POEM_QUOTE in poem.text
    layout = _layout(alice_book)
    units = next(
        chapter.units[poem.id]
        for chapter in layout.chapters.values()
        if poem.id in chapter.units
    )
    assert len(units) == 1
    translation = units[0].translation
    assert translation is not None
    source_lines = poem.text.split("\n")
    assert [not line.strip() for line in translation.split("\n")] == [not line.strip() for line in source_lines]
    assert any(PARODY_LABEL in numbered.note.labels for numbered in units[0].notes)
    documents = exported_documents(alice_book.export_path)
    section = next(section for section in alice_book.document.ordered_sections if section.id == poem.section_id)
    rendered = [
        unit_div
        for unit_div in _unit_divs(_chapter_root(documents, section))
        if POEM_QUOTE in plain_text(_source_element(unit_div))
    ]
    assert len(rendered) == 1
    source = _source_element(rendered[0])
    rows = [child for child in rendered[0] if "lg-tr" in _classes(child)]
    assert len(rows) == 1
    assert len(_breaks(source)) == poem.text.count("\n")
    assert len(_breaks(rows[0])) == poem.text.count("\n")
    if any(not line.strip() for line in source_lines):
        assert _adjacent_break(rows[0])
    assert PARODY_LABEL in _labeled(rendered[0], "lg-kind")
    with alice_book.app.unit_of_work.transaction() as transaction:
        analyses = transaction.analyses.list_for_book(BOOK)
    assert any(
        difficulty.kind == "parody" and difficulty.owner_block_id == poem.id
        for analysis in analyses
        if analysis.content_plan is not None
        for difficulty in analysis.content_plan.difficulties
    )


def test_alice_false_friend_and_character_speech_labels(alice_book: AliceRun):
    documents = exported_documents(alice_book.export_path)
    labels: list[str] = []
    speakers: list[str] = []
    for section in alice_book.document.ordered_sections:
        root = _chapter_root(documents, section)
        labels.extend(_labeled(root, "lg-kind"))
        speakers.extend(_labeled(root, "lg-speaker"))
    assert FALSE_FRIEND_LABEL in labels
    assert SPEECH_LABEL in labels
    with alice_book.app.unit_of_work.transaction() as transaction:
        analyses = transaction.analyses.list_for_book(BOOK)
    inferred = [
        fact
        for analysis in analyses
        if analysis.content_plan is not None
        for fact in analysis.content_plan.context_facts
        if fact.kind == "speaker" and fact.certainty == "inferred"
    ]
    assert len(inferred) == 1
    assert inferred[0].value == SPEAKER
    assert SPEAKER in speakers


def test_alice_who_is_who_no_later_chapters(alice_book: AliceRun):
    layout = _layout(alice_book)
    assert layout.who_is_who
    assert any(entry.description.strip() == DESCRIPTION for entry in layout.who_is_who)
    assert all(entry.first_chapter in {1, 2} for entry in layout.who_is_who)
    documents = exported_documents(alice_book.export_path)
    found = [
        root
        for path, root in documents.items()
        if path.replace("\\", "/").endswith("linguary/who-is-who.xhtml")
    ]
    assert len(found) == 1
    page = plain_text(found[0])
    assert DESCRIPTION in page
    refs = re.findall(r"Впервые в гл\. [^.]*\.", page)
    assert refs
    assert set(refs) <= {"Впервые в гл. I.", "Впервые в гл. II."}
    for name in LATER_NAMES:
        assert name not in page
        assert all(name not in entry.source for entry in layout.who_is_who)


def test_alice_epubcheck_clean(alice_book: AliceRun):
    assert alice_book.stub.was_called
    with alice_book.app.unit_of_work.transaction() as transaction:
        book = transaction.books.get(BOOK)
    assert book is not None
    assert book.export_validation_status == ExportValidationStatus.PASSED
    report = json.loads(alice_book.export_path.with_suffix(".epubcheck.json").read_text(encoding="utf-8"))
    assert report["messages"] == []
