"""A2 Harry Potter regressions, checked on real fragments without a model call.

Fragments are the Pottermore insert, glossary descriptions, and dialogue lines
from ``harry-potter-ch1-ch3-a2-deepseek-main``. Russian text of the six
replies is the staged rendering, not a new model call.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import pytest
from lxml import etree

from linguary.adapters.epub import EpubExportWriter
from linguary.application.analysis_merge import normalize_a2_dialogue_response
from linguary.application.layout.book import BookLayout, LayoutInputs, build_book_layout
from linguary.application.llm_dto import (
    PedagogicalAnalysisResponse,
    QuotedDifficultyResponse,
    QuotedSourceAnchorResponse,
    TreatmentPlanItemResponse,
)
from linguary.application.ports.epub import EpubWriteRequest
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    BlockType,
    BookMetadata,
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    DocumentBlock,
    DocumentSection,
    GlossaryCandidate,
    GlossaryHint,
    SourceReference,
    TreatmentPlan,
)
from linguary.domain.content_plan import ContentPlan, ContextFact, RequiredPoint, SourceAnchor

_BOOK_ID = "hp-a2-regression"
_OPEN = "\u2018"
_CLOSE = "\u2019"
_LATIN = re.compile(r"[A-Za-z]")

_PROMO_CREDIT = "by J.K. Rowling"
_PROMO_BLURB = (
    "The unique online experience built around the Harry Potter books. "
    "Share and participate in the stories, showcase your own Potter-related "
    "creativity and discover even more about the world of Harry Potter from "
    "the author herself."
)
_PROMO_VISIT = "Visit pottermore.com"
_PROMO_HARRY_DESCRIPTION = (
    "Name used in the phrase \u00abHarry Potter books\u00bb; here it denotes "
    "the book series, not a character action in the page."
)
_HARRY_DESCRIPTION = (
    "Младенец, сын Лили и Джеймса Поттеров; в этой главе его оставляют у Дёрсли."
)
_HARRY_LATER_DESCRIPTION = (
    "Мальчик, живущий у Дёрсли; худой, черноволосый, носит круглые очки."
)
_DURSLEY_DESCRIPTION = (
    "The surname of the conventional family living at number four, Privet Drive; "
    "Mr Dursley is the husband and father, and Mrs Dursley is his wife."
)
_DUDLEY_DESCRIPTION = "The young son of Mr and Mrs Dursley."
_DUDLEY_LATER_DESCRIPTION = (
    "Сын Вернона и Петунии, двоюродный брат Гарри; крупный и избалованный."
)
_MRS_DURSLEY_DESCRIPTION = (
    "Тётя Гарри, жена мистера Дёрсли; в финале главы она обнаружит Гарри."
)
_HAGRID_DESCRIPTION = "Огромный волшебник с густой бородой; привёз Гарри на мотоцикле."
_DUMBLEDORE_DESCRIPTION = "Пожилой волшебник; появляется на Прайвет-драйв ночью."
_VERNON_DESCRIPTION = (
    "Муж Петунии, дядя Гарри; крупный, вспыльчивый, работает в фирме по производству дрелей."
)
_PETUNIA_DESCRIPTION = (
    "Тётя Гарри, сестра Лили и жена Вернона Дёрсли; худая женщина с длинной шеей."
)
_PIERS_DESCRIPTION = "Один из приятелей Дадли, крупный и глуповатый мальчик из его компании."
_ENGLISH_DESCRIPTIONS = (
    _PROMO_HARRY_DESCRIPTION,
    _DURSLEY_DESCRIPTION,
    _DUDLEY_DESCRIPTION,
)

_NASTY = f"{_OPEN}Harry. Nasty, common name, if you ask me.{_CLOSE}"
_HAGRID_LINE = (
    f"{_OPEN}Hagrid,{_CLOSE} said Dumbledore, sounding relieved. "
    f"{_OPEN}At last. And where did you get that motorbike?{_CLOSE}"
)
_GOOD_LUCK = (
    f"{_OPEN}Good luck, Harry,{_CLOSE} he murmured. He turned on his heel and "
    "with a swish of his cloak he was gone."
)
_DEMANDED = f"{_OPEN}Are you up yet?{_CLOSE} she demanded."
_SNAPPED = f"{_OPEN}What did you say?{_CLOSE} his aunt snapped through the door."
_CAR = f"{_OPEN}That car{_CLOSE}s new, he{_CLOSE}s not sitting in it alone \u2026{_CLOSE}"
_SHOO = f"{_OPEN}Shoo!{_CLOSE} said Mr Dursley loudly."
_UP = f"{_OPEN}Up! Get up! Now!{_CLOSE}"
_WINKED = "It winked."
_BOA = "Boa Constrictor, Brazil."
_A_WHAT = f"{_OPEN}A what?{_CLOSE}"
_NO_WHY = f"{_OPEN}No,{_CLOSE} she said sharply. {_OPEN}Why?{_CLOSE}"
_NEARLY = f"{_OPEN}Nearly,{_CLOSE} said Harry."
_OH_YES = (
    f"{_OPEN}Oh, yes,{_CLOSE} said Mr Dursley, his heart sinking horribly. "
    f"{_OPEN}Yes, I quite agree.{_CLOSE}"
)
_GIANT = (
    f"{_OPEN}Borrowed it, Professor Dumbledore, sir,{_CLOSE} said the giant, "
    "climbing carefully off the motorbike as he spoke. "
    f"{_OPEN}Young Sirius Black lent it me. I{_CLOSE}ve got him, sir.{_CLOSE}"
)
_MOTORBIKES = (
    "Uncle Vernon nearly crashed into the car in front. He turned right around "
    "in his seat and yelled at Harry, his face like a gigantic beetroot with a "
    f"moustache, {_OPEN}MOTORBIKES DON{_CLOSE}T FLY!{_CLOSE}"
)
_PIERS = (
    "Harry was glad school was over, but there was no escaping Dudley"
    f"{_CLOSE}s gang, who visited the house every single day. Piers, Dennis, "
    "Malcolm and Gordon were all big and stupid, but as Dudley was the biggest "
    "and stupidest of the lot, he was the leader. The rest of them were all "
    "quite happy to join in Dudley"
    f"{_CLOSE}s favourite sport: Harry-hunting."
)

_REGRESSED = ("nasty", "hagrid-line", "good-luck", "demanded", "snapped", "car")
_EXCEPTIONS = ("shoo", "up", "winked", "boa", "a-what", "no-why", "nearly")
_SHOWN_SPEAKERS = {
    "nasty": "миссис Дёрсли",
    "oh-yes": "мистер Дёрсли",
    "giant": "Хагрид",
    "demanded": "тётя Петуния",
    "car": "дядя Вернон",
}
_HIDDEN_SPEAKERS = ("good-luck", "hagrid-line", "snapped", "motorbikes", *_EXCEPTIONS)


@dataclass(frozen=True, slots=True)
class _Line:
    """One real block and the mode ``main`` stored before normalization."""

    key: str
    section: str
    text: str
    original: BlockTreatment
    russian: str | None = None
    hint: tuple[str, str, str | None] | None = None


_LINES: tuple[_Line, ...] = (
    _Line("promo-credit", "promo", _PROMO_CREDIT, BlockTreatment.READ_AS_IS),
    _Line("promo-blurb", "promo", _PROMO_BLURB, BlockTreatment.READ_AS_IS),
    _Line("promo-visit", "promo", _PROMO_VISIT, BlockTreatment.READ_AS_IS),
    _Line(
        "nasty",
        "ch1",
        _NASTY,
        BlockTreatment.GLOSSARY_ONLY,
        "\u2014 Гарри. Противное, пошлое имя, если меня спросить.",
    ),
    _Line(
        "shoo",
        "ch1",
        _SHOO,
        BlockTreatment.GLOSSARY_ONLY,
        hint=("Shoo!", "Брысь!", "Возглас, которым прогоняют кошку."),
    ),
    _Line(
        "oh-yes",
        "ch1",
        _OH_YES,
        BlockTreatment.TRANSLATE,
        "\u2014 О да, \u2014 сказал мистер Дёрсли, и сердце у него ужасно упало. "
        "\u2014 Да, я совершенно согласен.",
    ),
    _Line(
        "hagrid-line",
        "ch1",
        _HAGRID_LINE,
        BlockTreatment.GLOSSARY_ONLY,
        "\u00abХагрид, \u2014 сказал Дамблдор с облегчением. \u2014 Наконец-то. "
        "И где ты взял этот мотоцикл?\u00bb",
    ),
    _Line(
        "good-luck",
        "ch1",
        _GOOD_LUCK,
        BlockTreatment.GLOSSARY_ONLY,
        "\u2014 Удачи, Гарри, \u2014 пробормотал он. Он повернулся на каблуках и, "
        "взметнув плащом, исчез.",
    ),
    _Line(
        "giant",
        "ch1",
        _GIANT,
        BlockTreatment.TRANSLATE,
        "\u00abОдолжил, профессор Дамблдор, сэр, \u2014 сказал великан, осторожно "
        "слезая с мотоцикла. \u2014 Молодой Сириус Блэк дал мне его на время. "
        "Он у меня, сэр.\u00bb",
    ),
    _Line("a-what", "ch1", _A_WHAT, BlockTreatment.READ_AS_IS),
    _Line(
        "no-why",
        "ch1",
        _NO_WHY,
        BlockTreatment.GLOSSARY_ONLY,
        hint=("said sharply", "сказала резко", "sharply \u2014 резко, отрывисто."),
    ),
    _Line("up", "ch2", _UP, BlockTreatment.READ_AS_IS),
    _Line(
        "demanded",
        "ch2",
        _DEMANDED,
        BlockTreatment.GLOSSARY_ONLY,
        "\u2014 Ты уже встал? \u2014 потребовала она.",
    ),
    _Line(
        "snapped",
        "ch2",
        _SNAPPED,
        BlockTreatment.GLOSSARY_ONLY,
        "\u2014 Что ты сказал? \u2014 огрызнулась тётя через дверь.",
    ),
    _Line("nearly", "ch2", _NEARLY, BlockTreatment.READ_AS_IS),
    _Line(
        "car",
        "ch2",
        _CAR,
        BlockTreatment.GLOSSARY_ONLY,
        "\u00abЭта машина новая, он не будет сидеть в ней один\u2026\u00bb",
    ),
    _Line(
        "motorbikes",
        "ch2",
        _MOTORBIKES,
        BlockTreatment.TRANSLATE,
        "Дядя Вернон чуть не врезался в машину впереди. Он полностью обернулся "
        "на своём сиденье и заорал на Гарри, и его лицо было похоже на огромную "
        "свёклу с усами: \u00abМОТОЦИКЛЫ НЕ ЛЕТАЮТ!\u00bb",
    ),
    _Line("winked", "ch2", _WINKED, BlockTreatment.READ_AS_IS),
    _Line("boa", "ch2", _BOA, BlockTreatment.READ_AS_IS),
    _Line("piers", "ch3", _PIERS, BlockTreatment.READ_AS_IS),
)

_BY_KEY = {line.key: line for line in _LINES}
_SECTIONS = (
    ("promo", "Pottermore.com", "hp01_experience_en-gb.html", None),
    ("ch1", "Chapter 1 - The Boy Who Lived", "hp01_ch001_en-gb.html", "ONE"),
    ("ch2", "Chapter 2 - The Vanishing Glass", "hp01_ch002_en-gb.html", "TWO"),
    ("ch3", "Chapter 3 - The Letters from No One", "hp01_ch003_en-gb.html", "THREE"),
)


@dataclass(frozen=True, slots=True)
class _Case:
    document: CanonicalDocument
    blocks: Mapping[str, DocumentBlock]
    section_ids: Mapping[str, str]
    normalized: PedagogicalAnalysisResponse
    layout: BookLayout
    pages: tuple[etree._Element, ...]


def _paragraph(
    section_id: str,
    resource: str,
    order: int,
    index: int,
    text: str,
) -> DocumentBlock:
    return DocumentBlock.create(
        document_id=_BOOK_ID,
        section_id=section_id,
        type=BlockType.PARAGRAPH,
        order=order,
        text=text,
        source_reference=SourceReference(resource, f"/body/p[{index}]"),
    )


def _heading(section_id: str, resource: str, text: str) -> DocumentBlock:
    return DocumentBlock.create(
        document_id=_BOOK_ID,
        section_id=section_id,
        type=BlockType.HEADING,
        order=0,
        text=text,
        source_reference=SourceReference(resource, "/body/h1[1]"),
    )


def _document() -> tuple[CanonicalDocument, dict[str, DocumentBlock], dict[str, str]]:
    """Pottermore, then three story chapters, each limited to the case lines."""
    sections: list[DocumentSection] = []
    blocks: dict[str, DocumentBlock] = {}
    section_ids: dict[str, str] = {}
    for order, (key, title, resource, chapter_word) in enumerate(_SECTIONS):
        section_id = f"sec-{key}"
        section_ids[key] = section_id
        built: list[DocumentBlock] = []
        if chapter_word is not None:
            built.append(_heading(section_id, resource, f"\u2014 CHAPTER {chapter_word} \u2014"))
        lines = [line for line in _LINES if line.section == key]
        for index, line in enumerate(lines, start=1):
            block = _paragraph(section_id, resource, len(built), index, line.text)
            built.append(block)
            blocks[line.key] = block
        sections.append(
            DocumentSection(
                id=section_id,
                document_id=_BOOK_ID,
                title=title,
                order=order,
                source_resource_path=resource,
                blocks=tuple(built),
            )
        )
    document = CanonicalDocument(
        id=_BOOK_ID,
        metadata=BookMetadata(title="Harry Potter and the Philosopher's Stone"),
        source_language="en",
        target_language="ru",
        sections=tuple(sections),
    )
    return document, blocks, section_ids


def _anchor(block: DocumentBlock, quote: str) -> SourceAnchor:
    start = block.text.index(quote)
    return SourceAnchor(block.id, start, start + len(quote), quote)


def _candidate(
    analysis_id: str,
    source: str,
    translation: str,
    description: str,
) -> GlossaryCandidate:
    slug = re.sub(r"[^a-z0-9]+", "-", source.casefold()).strip("-")
    return GlossaryCandidate(
        id=f"{analysis_id}-{slug}",
        analysis_id=analysis_id,
        source_text=source,
        type=ConceptType.PERSON,
        suggested_translation=translation,
        confidence=0.9,
        model_flagged_review=False,
        requires_review=False,
        description=description,
    )


def _fact(
    fact_id: str,
    kind_value: str,
    certainty: str,
    subject: SourceAnchor,
    anchors: Sequence[SourceAnchor],
) -> ContextFact:
    return ContextFact(
        id=fact_id,
        kind="speaker",
        anchors=tuple(anchors),
        subject_anchor=subject,
        value=kind_value,
        certainty=certainty,
    )


def _facts(blocks: Mapping[str, DocumentBlock]) -> dict[str, tuple[ContextFact, ...]]:
    """Real anchor shapes. Compound values are the forms the resolver must accept.

    ``Oh, yes`` was explicit in the run. It is inferred here so ``Mr. Dursley``
    has to be shown on a translated opening line; the stored inferred ``Mr Dursley``
    lines were left untranslated.
    """
    nasty = _anchor(blocks["nasty"], blocks["nasty"].text)
    oh_yes = _anchor(blocks["oh-yes"], blocks["oh-yes"].text)
    good_luck = _anchor(blocks["good-luck"], blocks["good-luck"].text)
    giant_subject = _anchor(blocks["giant"], "said the giant")
    hagrid_other = _anchor(blocks["hagrid-line"], f"{_OPEN}Hagrid,{_CLOSE}")
    demanded = _anchor(blocks["demanded"], "she demanded")
    car = _anchor(blocks["car"], blocks["car"].text)
    turned = _anchor(blocks["motorbikes"], "He turned right around in his seat")
    return {
        "ch1": (
            _fact("speaker-petunia", "Petunia Dursley (Mrs Dursley)", "inferred", nasty, (nasty,)),
            _fact("speaker-mr-dursley", "Mr. Dursley", "inferred", oh_yes, (oh_yes,)),
            _fact("speaker-dumbledore", "Dumbledore", "explicit", good_luck, (good_luck,)),
            _fact(
                "speaker-hagrid",
                "Hagrid (the giant on the motorbike)",
                "inferred",
                giant_subject,
                (giant_subject, hagrid_other),
            ),
        ),
        "ch2": (
            _fact(
                "speaker-aunt-petunia",
                "Aunt Petunia",
                "inferred",
                demanded,
                (demanded,),
            ),
            _fact(
                "speaker-vernon",
                "Vernon Dursley (Uncle Vernon)",
                "inferred",
                car,
                (car,),
            ),
            _fact("speaker-uncle", "Uncle Vernon", "inferred", turned, (turned,)),
        ),
    }


def _people(analysis_id: str, rows: Sequence[tuple[str, str, str]]) -> tuple[GlossaryCandidate, ...]:
    return tuple(
        _candidate(analysis_id, source, translation, description)
        for source, translation, description in rows
    )


def _analyses(
    section_ids: Mapping[str, str],
    blocks: Mapping[str, DocumentBlock],
    modes: Mapping[str, BlockTreatment],
    facts: Mapping[str, tuple[ContextFact, ...]],
) -> tuple[ChapterAnalysis, ...]:
    people = {
        "promo": (("Harry Potter", "Гарри Поттер", _PROMO_HARRY_DESCRIPTION),),
        "ch1": (
            ("Harry Potter", "Гарри Поттер", _HARRY_DESCRIPTION),
            ("Dursley", "Дёрсли", _DURSLEY_DESCRIPTION),
            ("Dudley", "Дадли", _DUDLEY_DESCRIPTION),
            ("Mrs Dursley", "миссис Дёрсли", _MRS_DURSLEY_DESCRIPTION),
            ("Hagrid", "Хагрид", _HAGRID_DESCRIPTION),
            ("Dumbledore", "Дамблдор", _DUMBLEDORE_DESCRIPTION),
        ),
        "ch2": (
            ("Dudley", "Дадли", _DUDLEY_LATER_DESCRIPTION),
            ("Harry Potter", "Гарри Поттер", _HARRY_LATER_DESCRIPTION),
            ("Petunia", "Петуния", _PETUNIA_DESCRIPTION),
            ("Vernon", "Вернон", _VERNON_DESCRIPTION),
        ),
        "ch3": (("Piers", "Пирз", _PIERS_DESCRIPTION),),
    }
    analyses: list[ChapterAnalysis] = []
    for key, title, _resource, _chapter_word in _SECTIONS:
        section_id = section_ids[key]
        analysis_id = f"analysis-{key}"
        lines = [line for line in _LINES if line.section == key]
        plan = TreatmentPlan.of_treatments(
            section_id,
            ((blocks[line.key].id, modes[blocks[line.key].id]) for line in lines),
        )
        section_facts = facts.get(key, ())
        analyses.append(
            ChapterAnalysis.create(
                id=analysis_id,
                section_id=section_id,
                treatment_plan=plan,
                translation_instructions=f"Keep glossary names while rendering {title}.",
                chapter_summary=f"Fragment of {title} used for the A2 regression check.",
                expected_block_ids=tuple(blocks[line.key].id for line in lines),
                glossary_candidates=_people(analysis_id, people[key]),
                content_plan=(
                    ContentPlan(
                        version=3,
                        profile="content-plan-v3",
                        context_facts=section_facts,
                    )
                    if section_facts
                    else None
                ),
            )
        )
    return tuple(analyses)


def _result(block: DocumentBlock, line: _Line, treatment: BlockTreatment) -> BlockProcessingResult:
    if treatment is BlockTreatment.TRANSLATE:
        if line.russian is None:
            raise AssertionError(f"{line.key} became TRANSLATE without a staged Russian text")
        return BlockProcessingResult(
            block_id=block.id,
            treatment=treatment,
            translated_text=line.russian,
        )
    if treatment is BlockTreatment.GLOSSARY_ONLY:
        if line.hint is None:
            raise AssertionError(f"{line.key} stayed GLOSSARY_ONLY without a glossary hint")
        original, translation, note = line.hint
        return BlockProcessingResult(
            block_id=block.id,
            treatment=treatment,
            glossary_hints=(GlossaryHint(original, translation, note),),
        )
    return BlockProcessingResult(block_id=block.id, treatment=BlockTreatment.READ_AS_IS)


def _read_pages(path: Path) -> tuple[etree._Element, ...]:
    """Chapter files keep their ``.html`` names; book pages are ``.xhtml``."""
    pages: list[etree._Element] = []
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith((".xhtml", ".html")):
                pages.append(etree.fromstring(archive.read(name)))
    return tuple(pages)


def _assemble(tmp_path: Path) -> _Case:
    document, blocks, section_ids = _document()
    ordered = tuple(blocks[line.key] for line in _LINES)
    point = (RequiredPoint("p1", "Объяснить значение глагола речи."),)
    normalized = normalize_a2_dialogue_response(
        PedagogicalAnalysisResponse(
            treatment_plan=tuple(
                TreatmentPlanItemResponse(block.id, _BY_KEY[key].original)
                for key, block in blocks.items()
            ),
            difficulties=(
                QuotedDifficultyResponse(
                    "d-demanded",
                    (QuotedSourceAnchorResponse(blocks["demanded"].id, "demanded"),),
                    blocks["demanded"].id,
                    "vocabulary",
                    "useful",
                    "lexical_hint",
                    point,
                ),
                QuotedDifficultyResponse(
                    "d-snapped",
                    (QuotedSourceAnchorResponse(blocks["snapped"].id, "snapped"),),
                    blocks["snapped"].id,
                    "vocabulary",
                    "useful",
                    "lexical_hint",
                    point,
                ),
            ),
        ),
        ordered,
        {block.id: block.id for block in ordered},
        reader_level="a2",
        english_help=False,
    )
    modes = {item.source_block_id: item.treatment for item in normalized.treatment_plan}
    analyses = _analyses(section_ids, blocks, modes, _facts(blocks))
    results = {
        blocks[line.key].id: _result(blocks[line.key], line, modes[blocks[line.key].id])
        for line in _LINES
    }
    layout = build_book_layout(
        LayoutInputs(
            document=document,
            results=results,
            analyses=analyses,
            concepts=(),
            reader_level="a2",
            english_help=False,
        )
    )
    target = tmp_path / "harry-a2-regressions.epub"
    EpubExportWriter().write(
        EpubWriteRequest(
            document=document,
            target_path=str(target),
            block_results=results,
            layout=layout,
        )
    )
    return _Case(document, blocks, section_ids, normalized, layout, _read_pages(target))


@pytest.fixture(scope="module")
def harry_regression(tmp_path_factory: pytest.TempPathFactory) -> _Case:
    """One export of the fragment book. The four criteria share it."""
    return _assemble(tmp_path_factory.mktemp("harry-a2"))


def _local(element: etree._Element) -> str:
    return etree.QName(element).localname


def _has_class(element: etree._Element, name: str) -> bool:
    return name in (element.get("class") or "").split()


def _child(parent: etree._Element, class_name: str) -> etree._Element | None:
    for element in parent:
        if _has_class(element, class_name):
            return element
    return None


def _text(element: etree._Element) -> str:
    return "".join(element.itertext())


def _source_text(element: etree._Element) -> str:
    """Source characters without glossary reference markers inserted into the line."""
    parts: list[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        if _has_class(child, "lg-ref"):
            if child.tail:
                parts.append(child.tail)
            continue
        parts.append(_source_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _unit(pages: Sequence[etree._Element], snippet: str) -> etree._Element:
    matches: list[etree._Element] = []
    for page in pages:
        for element in page.iter():
            if _local(element) != "div" or not _has_class(element, "lg-unit"):
                continue
            source = _child(element, "lg-src")
            if source is not None and snippet in _source_text(source):
                matches.append(element)
    assert len(matches) == 1, snippet
    return matches[0]


def _translation(unit: etree._Element) -> str | None:
    """Russian paragraph, without the speaker span."""
    paragraph = _child(unit, "lg-tr")
    if paragraph is None:
        return None
    parts: list[str] = []
    if paragraph.text:
        parts.append(paragraph.text)
    for child in paragraph:
        if _has_class(child, "lg-speaker"):
            if child.tail:
                parts.append(child.tail)
            continue
        parts.append(_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def _speaker(unit: etree._Element) -> str | None:
    paragraph = _child(unit, "lg-tr")
    if paragraph is None:
        return None
    speaker = _child(paragraph, "lg-speaker")
    if speaker is None or not speaker.text:
        return None
    return speaker.text.strip()


def _mode(case: _Case, key: str) -> BlockTreatment:
    block_id = case.blocks[key].id
    for item in case.normalized.treatment_plan:
        if item.source_block_id == block_id:
            return item.treatment
    raise AssertionError(key)


def _who_page(pages: Sequence[etree._Element]) -> etree._Element:
    for page in pages:
        for element in page.iter():
            if _local(element) == "section" and element.get("id") == "who-is-who":
                return element
    raise AssertionError("book who-is-who page is missing")


def _who_cards(page: etree._Element) -> dict[str, tuple[str, str]]:
    cards: dict[str, tuple[str, str]] = {}
    for element in page.iter():
        if _local(element) != "li":
            continue
        head = _child(element, "lg-head")
        if head is None or not head.text:
            continue
        reference = ""
        for child in element:
            if _has_class(child, "lg-src-ref") and child.text:
                reference = child.text.strip()
        cards[head.text.strip()] = ((head.tail or "").strip(), reference)
    return cards


def test_six_regressed_a2_dialogues_promoted_while_short_exceptions_stay_untouched(
    harry_regression: _Case,
) -> None:
    """Six light main replies become TRANSLATE; seven staged exceptions do not."""
    for key in _REGRESSED:
        assert _mode(harry_regression, key) is BlockTreatment.TRANSLATE
        unit = _unit(harry_regression.pages, _BY_KEY[key].text)
        assert _translation(unit) == _BY_KEY[key].russian
    assert {
        difficulty.id: difficulty.help_kind
        for difficulty in harry_regression.normalized.difficulties
    } == {
        "d-demanded": "explanation",
        "d-snapped": "explanation",
    }
    for key in _EXCEPTIONS:
        assert _mode(harry_regression, key) is _BY_KEY[key].original
        assert _mode(harry_regression, key) is not BlockTreatment.TRANSLATE
        unit = _unit(harry_regression.pages, _BY_KEY[key].text)
        assert _translation(unit) is None


def test_harry_potter_epub_has_russian_inferred_speakers_only_on_opening_dialogues(
    harry_regression: _Case,
) -> None:
    """Inferred opening lines carry a Russian name; other anchors do not."""
    seen: list[str] = []
    for page in harry_regression.pages:
        for element in page.iter():
            if _local(element) == "span" and _has_class(element, "lg-speaker"):
                assert element.text is not None
                assert _LATIN.search(element.text) is None
                seen.append(element.text.strip())
    assert seen == [
        "миссис Дёрсли",
        "мистер Дёрсли",
        "Хагрид",
        "тётя Петуния",
        "дядя Вернон",
    ]
    for key, name in _SHOWN_SPEAKERS.items():
        assert _speaker(_unit(harry_regression.pages, _BY_KEY[key].text)) == name
    for key in _HIDDEN_SPEAKERS:
        assert _speaker(_unit(harry_regression.pages, _BY_KEY[key].text)) is None


def test_pottermore_is_front_matter_and_chapter_one_is_numbered_first_with_harry_entry(
    harry_regression: _Case,
) -> None:
    """The three-paragraph Pottermore insert does not take chapter I or Harry's card."""
    document = harry_regression.document
    promo = document.section(harry_regression.section_ids["promo"])
    paragraphs = [block for block in promo.blocks if block.type is BlockType.PARAGRAPH]
    assert [block.text for block in paragraphs] == [_PROMO_CREDIT, _PROMO_BLURB, _PROMO_VISIT]
    assert not any(block.type is BlockType.HEADING for block in promo.blocks)
    assert document.is_front_matter(promo) is True
    numbers = {
        key: harry_regression.layout.chapters[section_id].number
        for key, section_id in harry_regression.section_ids.items()
    }
    assert numbers == {"promo": None, "ch1": 1, "ch2": 2, "ch3": 3}
    cards = _who_cards(_who_page(harry_regression.pages))
    tail, reference = cards["Harry Potter"]
    assert tail == f"\u2014 Гарри Поттер. {_HARRY_DESCRIPTION}"
    assert reference == "Впервые в гл. I."
    assert cards["Piers"][1] == "Впервые в гл. III."
    assert _PROMO_HARRY_DESCRIPTION not in _text(_who_page(harry_regression.pages))


def test_who_is_who_cards_contain_no_english_descriptions(harry_regression: _Case) -> None:
    """A card description is Russian, or absent when the first one was English."""
    cards = _who_cards(_who_page(harry_regression.pages))
    page_text = _text(_who_page(harry_regression.pages))
    for description in _ENGLISH_DESCRIPTIONS:
        assert description not in page_text
    assert _DUDLEY_LATER_DESCRIPTION not in page_text
    assert _HARRY_LATER_DESCRIPTION not in page_text
    assert cards["Dudley"][0] == "\u2014 Дадли."
    assert cards["Dursley"][0] == "\u2014 Дёрсли."
    assert cards["Mrs Dursley"][0] == f"\u2014 миссис Дёрсли. {_MRS_DURSLEY_DESCRIPTION}"
    assert cards["Vernon"][0] == f"\u2014 Вернон. {_VERNON_DESCRIPTION}"
    assert cards["Piers"][0] == f"\u2014 Пирз. {_PIERS_DESCRIPTION}"
    for tail, _reference in cards.values():
        description = tail.split(". ", 1)[1] if ". " in tail else ""
        assert description == "" or _LATIN.search(description) is None
    harry = next(entry for entry in harry_regression.layout.who_is_who if entry.source == "Harry Potter")
    dudley = next(entry for entry in harry_regression.layout.who_is_who if entry.source == "Dudley")
    assert harry.first_chapter == 1
    assert harry.description == _HARRY_DESCRIPTION
    assert dudley.description == ""
