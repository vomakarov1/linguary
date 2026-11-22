"""A2 Harry Potter regressions, checked on real fragments without a model call.

Fragments are the Pottermore insert, glossary descriptions, and dialogue lines
from ``harry-potter-ch1-ch3-a2-deepseek-main``. Russian text of the six
replies is the staged rendering, not a new model call.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from zipfile import ZipFile

import pytest
from lxml import etree

from linguary.adapters.epub import EpubExportWriter
from linguary.adapters.epub.splitting import segment_text
from linguary.application.analysis_merge import normalize_a2_dialogue_response
from linguary.application.layout.book import BookLayout, LayoutInputs, build_book_layout
from linguary.application.layout.units import LayoutUnit, NumberedNote
from linguary.application.layout.who_is_who import WhoEntry, book_who_is_who
from linguary.application.llm_dto import (
    BeforeYouReadEntryResponse,
    PedagogicalAnalysisResponse,
    QuotedDifficultyResponse,
    QuotedSourceAnchorResponse,
    SegmentTreatmentResponse,
    TreatmentPlanItemResponse,
)
from linguary.application.ports.epub import EpubWriteRequest
from linguary.domain import (
    BeforeYouReadEntry,
    BlockProcessingResult,
    BlockSegment,
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
    LearningMaterial,
    SourceReference,
    TreatmentPlan,
)
from linguary.domain.content_plan import ContentPlan, ContextFact, RequiredPoint, SourceAnchor
from linguary.domain.name_translation import (
    NameTranslationAssessment,
    NameTranslationEvidence,
    NameTranslationMethod,
    NameTranslationObservation,
    NameTranslationStrategy,
)

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
_BUNDLE_AND_SCAR = (
    "He laid Harry gently on the doorstep, took a letter out of his cloak, "
    f"tucked it inside Harry{_CLOSE}s blankets, and then came back to the other "
    "two. For a full minute the three of them stood and looked at the little "
    "bundle; Hagrid"
    f"{_CLOSE}s shoulders shook, Professor McGonagall blinked furiously, and "
    "the twinkling light that usually shone from Dumbledore"
    f"{_CLOSE}s eyes seemed to have gone out."
)
_NO_PROBLEMS = f"{_OPEN}No problems, were there?{_CLOSE}"
_COULDNT_YOU = f"{_OPEN}Couldn{_CLOSE}t you do something about it, Dumbledore?{_CLOSE}"
_YVONNE = f"{_OPEN}What about what{_CLOSE}s-her-name, your friend \u2013 Yvonne?{_CLOSE}"
_PIERS = (
    "Harry was glad school was over, but there was no escaping Dudley"
    f"{_CLOSE}s gang, who visited the house every single day. Piers, Dennis, "
    "Malcolm and Gordon were all big and stupid, but as Dudley was the biggest "
    "and stupidest of the lot, he was the leader. The rest of them were all "
    "quite happy to join in Dudley"
    f"{_CLOSE}s favourite sport: Harry-hunting."
)

_REGRESSED = (
    "nasty",
    "good-luck",
    "bundle-and-scar",
    "no-problems",
    "couldnt-you",
    "yvonne",
)
_GLOSSARY_KEPT = (
    "hagrid-line",
    "demanded",
    "snapped",
    "car",
)
_EXCEPTIONS = ("shoo", "up", "boa", "a-what", "no-why", "nearly")
_SHOWN_SPEAKERS = {
    "nasty": "миссис Дёрсли",
    "oh-yes": "мистер Дёрсли",
    "giant": "Хагрид",
}
_UNTRANSLATED_SPEAKERS = {
    "shoo": "мистер Дёрсли",
    "demanded": "тётя Петуния",
    "car": "дядя Вернон",
}
_HIDDEN_SPEAKERS = (
    "good-luck",
    "hagrid-line",
    "bundle-and-scar",
    "no-problems",
    "couldnt-you",
    "snapped",
    "yvonne",
    "motorbikes",
    "winked",
    "up",
    "boa",
    "a-what",
    "no-why",
    "nearly",
)


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
        hint=("sounding relieved", "с облегчением", None),
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
    _Line(
        "bundle-and-scar",
        "ch1",
        _BUNDLE_AND_SCAR,
        BlockTreatment.GLOSSARY_ONLY,
        "Он осторожно положил Гарри на порог, достал из плаща письмо, сунул его в "
        "одеяльца Гарри и вернулся к остальным двоим. Целую минуту все трое стояли "
        "и смотрели на маленький свёрток; плечи Хагрида тряслись, профессор "
        "Макгонагалл яростно моргала, а мерцающий свет, который обычно сиял в "
        "глазах Дамблдора, казалось, погас.",
        hint=("tucked it inside", "сунул его внутрь", "tuck \u2014 аккуратно подсовывать."),
    ),
    _Line(
        "no-problems",
        "ch1",
        _NO_PROBLEMS,
        BlockTreatment.GLOSSARY_ONLY,
        "\u2014 Проблем ведь не было, правда?",
        hint=("were there?", "не так ли?", "Разделительный вопрос."),
    ),
    _Line(
        "couldnt-you",
        "ch1",
        _COULDNT_YOU,
        BlockTreatment.GLOSSARY_ONLY,
        "\u2014 Разве ты не мог что-нибудь с ним сделать, Дамблдор?",
        hint=("do something about it", "что-то с этим сделать", "Повлиять на ситуацию."),
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
        hint=("demanded", "потребовала", None),
    ),
    _Line(
        "snapped",
        "ch2",
        _SNAPPED,
        BlockTreatment.GLOSSARY_ONLY,
        "\u2014 Что ты сказал? \u2014 огрызнулась тётя через дверь.",
        hint=("snapped", "огрызнулась", None),
    ),
    _Line(
        "yvonne",
        "ch2",
        _YVONNE,
        BlockTreatment.GLOSSARY_ONLY,
        "\u2014 А как насчёт этой, как её там, твоей подруги \u2014 Ивонн?",
        hint=("what\u2019s-her-name", "как её там", "Разговорная замена забытого имени."),
    ),
    _Line("nearly", "ch2", _NEARLY, BlockTreatment.READ_AS_IS),
    _Line(
        "car",
        "ch2",
        _CAR,
        BlockTreatment.GLOSSARY_ONLY,
        "\u00abЭта машина новая, он не будет сидеть в ней один\u2026\u00bb",
        hint=("alone", "один", None),
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
    _Line(
        "winked",
        "ch2",
        _WINKED,
        BlockTreatment.GLOSSARY_ONLY,
        hint=("winked", "подмигнула", None),
    ),
    _Line("boa", "ch2", _BOA, BlockTreatment.READ_AS_IS),
    _Line(
        "piers",
        "ch3",
        _PIERS,
        BlockTreatment.TRANSLATE,
        "Гарри был рад, что школа закончилась, но спасения от шайки Дадли, которая "
        "заявлялась в дом каждый день, не было. Пирз, Деннис, Малкольм и Гордон все "
        "были крупными и глупыми, но поскольку Дадли был самым крупным и глупым из всех, "
        "он был их вожаком. Остальные с радостью участвовали в любимой забаве Дадли — "
        "охоте на Гарри.",
    ),
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
    analyses: tuple[ChapterAnalysis, ...]
    epub_path: Path


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
    lines were left untranslated. ``Shoo!`` stays glossary-only, so its inferred
    name is a «Говорит …» note rather than a speaker span.
    """
    nasty = _anchor(blocks["nasty"], blocks["nasty"].text)
    shoo = _anchor(blocks["shoo"], blocks["shoo"].text)
    oh_yes = _anchor(blocks["oh-yes"], blocks["oh-yes"].text)
    good_luck = _anchor(blocks["good-luck"], blocks["good-luck"].text)
    good_luck_remark = _anchor(blocks["good-luck"], "he murmured")
    giant_subject = _anchor(blocks["giant"], "said the giant")
    hagrid_other = _anchor(blocks["hagrid-line"], f"{_OPEN}Hagrid,{_CLOSE}")
    demanded = _anchor(blocks["demanded"], "she demanded")
    car = _anchor(blocks["car"], blocks["car"].text)
    turned = _anchor(blocks["motorbikes"], "He turned right around in his seat")
    return {
        "ch1": (
            _fact("speaker-petunia", "миссис Дёрсли", "inferred", nasty, (nasty,)),
            _fact("speaker-shoo", "мистер Дёрсли", "inferred", shoo, (shoo,)),
            _fact("speaker-mr-dursley", "мистер Дёрсли", "inferred", oh_yes, (oh_yes,)),
            _fact("speaker-dumbledore", "Дамблдор", "explicit", good_luck, (good_luck,)),
            _fact("speaker-dumbledore-dup", "Дамблдор", "inferred", good_luck_remark, (good_luck_remark,)),
            _fact(
                "speaker-hagrid",
                "Хагрид (великан на мотоцикле)",
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
                "Вернон Дёрсли (дядя Вернон)",
                "inferred",
                car,
                (car,),
            ),
            _fact("speaker-uncle", "дядя Вернон", "inferred", turned, (turned,)),
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
    story_lines = tuple(line for line in _LINES if line.section != "promo")
    ordered = tuple(blocks[line.key] for line in story_lines)
    point = (RequiredPoint("p1", "Объяснить значение глагола речи."),)
    normalized = normalize_a2_dialogue_response(
        PedagogicalAnalysisResponse(
            treatment_plan=tuple(
                TreatmentPlanItemResponse(blocks[line.key].id, line.original)
                for line in story_lines
            ),
            difficulties=(
                QuotedDifficultyResponse(
                    "d-bundle",
                    (QuotedSourceAnchorResponse(blocks["bundle-and-scar"].id, "tucked it inside"),),
                    blocks["bundle-and-scar"].id,
                    "vocabulary",
                    "useful",
                    "lexical_hint",
                    point,
                ),
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
        known_names=(
            "Harry Potter",
            "Dursley",
            "Mr Dursley",
            "Mrs Dursley",
            "Dudley",
            "Hagrid",
            "Dumbledore",
            "Petunia",
            "Vernon",
            "Piers",
            "Sirius Black",
            "McGonagall",
            "Yvonne",
            "Brazil",
            "Boa Constrictor",
        ),
    )
    modes = {
        **{blocks[line.key].id: line.original for line in _LINES if line.section == "promo"},
        **{item.source_block_id: item.treatment for item in normalized.treatment_plan},
    }
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
    return _Case(
        document,
        blocks,
        section_ids,
        normalized,
        layout,
        _read_pages(target),
        analyses,
        target,
    )


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


def _mode(case: _Case, key: str) -> BlockTreatment:
    block_id = case.blocks[key].id
    for item in case.normalized.treatment_plan:
        if item.source_block_id == block_id:
            return item.treatment
    raise AssertionError(key)


def _layout_units(case: _Case, key: str) -> tuple[LayoutUnit, ...]:
    block = case.blocks[key]
    return case.layout.chapters[block.section_id].units[block.id]


def _speaker_notes(units: Sequence[LayoutUnit]) -> tuple[str, ...]:
    """Bodies of unnumbered «Говорит …» notes. Translated units have none."""
    return tuple(
        item.note.body[0]
        for unit in units
        for item in unit.notes
        if item.note.origin == "speaker"
    )


def _entry_tail(entry: WhoEntry) -> str:
    """The card tail the who-is-who page used to print after the source name."""
    tail = f"\u2014 {entry.translation}."
    if entry.description.strip():
        tail += f" {entry.description.strip()}"
    return tail


def _who_entries(
    document: CanonicalDocument,
    analyses: Sequence[ChapterAnalysis],
) -> dict[str, WhoEntry]:
    entries = book_who_is_who(document, analyses, ())
    return {entry.source: entry for entry in entries}


def _assert_who_page_absent(path: Path, pages: Sequence[etree._Element]) -> None:
    assert not any(name.endswith("who-is-who.xhtml") for name in _archive_names(path))
    assert not any(
        _local(element) == "section" and element.get("id") == "who-is-who"
        for page in pages
        for element in page.iter()
    )


def _assert_no_speaker_span(pages: Sequence[etree._Element]) -> None:
    for page in pages:
        for element in page.iter():
            if _local(element) != "p" or not _has_class(element, "lg-tr"):
                continue
            assert not any(
                _local(child) == "span" and _has_class(child, "lg-speaker")
                for child in element.iter()
            )


def test_six_regressed_a2_dialogues_promoted_while_short_exceptions_stay_untouched(
    harry_regression: _Case,
) -> None:
    """Six complex light blocks become TRANSLATE; four within A2 15-word/single-verb thresholds stay GLOSSARY_ONLY.

    ``It winked.`` keeps the glossary hint and is not an unaided READ_AS_IS line.
    """
    for key in _REGRESSED:
        assert _mode(harry_regression, key) is BlockTreatment.TRANSLATE
        unit = _unit(harry_regression.pages, _BY_KEY[key].text)
        assert _translation(unit) == _BY_KEY[key].russian
    for key in _GLOSSARY_KEPT:
        assert _mode(harry_regression, key) is BlockTreatment.GLOSSARY_ONLY
        unit = _unit(harry_regression.pages, _BY_KEY[key].text)
        assert _translation(unit) is None
    assert {
        difficulty.id: difficulty.help_kind
        for difficulty in harry_regression.normalized.difficulties
        if not difficulty.id.startswith("auto-vocab-")
    } == {
        "d-bundle": "explanation",
        "d-demanded": "lexical_hint",
        "d-snapped": "lexical_hint",
    }
    auto_diffs = [
        difficulty
        for difficulty in harry_regression.normalized.difficulties
        if difficulty.id.startswith("auto-vocab-")
    ]
    assert auto_diffs
    assert all(d.kind == "vocabulary" and d.help_kind == "lexical_hint" for d in auto_diffs)
    for key in _EXCEPTIONS:
        assert _mode(harry_regression, key) is _BY_KEY[key].original
        assert _mode(harry_regression, key) is not BlockTreatment.TRANSLATE
        unit = _unit(harry_regression.pages, _BY_KEY[key].text)
        assert _translation(unit) is None
    assert "winked" not in _EXCEPTIONS
    assert _mode(harry_regression, "winked") is BlockTreatment.GLOSSARY_ONLY
    assert _mode(harry_regression, "winked") is not BlockTreatment.READ_AS_IS
    winked = _unit(harry_regression.pages, _WINKED)
    assert _translation(winked) is None
    assert _note_terms(winked) == ["to wink"]


def test_a2_review_fixes_prompt_and_speaker_rules(harry_regression: _Case) -> None:
    """Russian/English speaker lookup and A2/SOUND/narrative prompt rules hold."""
    from linguary.application.context_builder import ContextBuilder, snapshot_prompt_templates
    from linguary.application.layout.book import _speaker_name
    from linguary.domain import TermConcept, TermStatus, TermVariant
    from linguary.domain.name_translation import NameTranslationPolicy, NameTranslationStrategy

    analyses = tuple(
        ChapterAnalysis.create(
            id="analysis-check",
            section_id=harry_regression.section_ids["ch1"],
            treatment_plan=TreatmentPlan.of_treatments(harry_regression.section_ids["ch1"], ()),
            translation_instructions="Keep names.",
            chapter_summary="Summary.",
            expected_block_ids=(),
            glossary_candidates=(
                _candidate("analysis-check", "Mrs Dursley", "миссис Дёрсли", _MRS_DURSLEY_DESCRIPTION),
                _candidate("analysis-check", "Dursley", "Дёрсли", _DURSLEY_DESCRIPTION),
                _candidate("analysis-check", "Hagrid", "Хагрид", _HAGRID_DESCRIPTION),
                _candidate("analysis-check", "Petunia", "Петуния", _PETUNIA_DESCRIPTION),
                _candidate("analysis-check", "Vernon", "Вернон", _VERNON_DESCRIPTION),
                GlossaryCandidate(
                    id="analysis-check-put-outer",
                    analysis_id="analysis-check",
                    source_text="Put-Outer",
                    type=ConceptType.FICTIONAL_CONCEPT,
                    suggested_translation="Гаситель",
                    confidence=0.9,
                    model_flagged_review=False,
                    requires_review=False,
                ),
            ),
        )
        for _ in (1,)
    )
    concepts = (
        TermConcept(
            id="concept-dumbledore",
            book_id=_BOOK_ID,
            type=ConceptType.PERSON,
            canonical_source_text="Albus Dumbledore",
            status=TermStatus.APPROVED,
            variants=(TermVariant("concept-dumbledore", "ru", "Альбус Дамблдор", TermStatus.APPROVED),),
        ),
    )
    for raw, expected in (
        ("Petunia Dursley (Mrs Dursley)", "миссис Дёрсли"),
        ("миссис Дёрсли", "миссис Дёрсли"),
        ("Mr. Dursley", "мистер Дёрсли"),
        ("мистер Дёрсли", "мистер Дёрсли"),
        ("Hagrid (the giant on the motorbike)", "Хагрид"),
        ("Хагрид", "Хагрид"),
        ("Aunt Petunia", "тётя Петуния"),
        ("тётя Петуния", "тётя Петуния"),
        ("Vernon Dursley (Uncle Vernon)", "дядя Вернон"),
        ("дядя Вернон", "дядя Вернон"),
        ("Альбус Дамблдор", "Альбус Дамблдор"),
        ("Гаситель", None),
    ):
        assert _speaker_name(raw, concepts, analyses, "ru", document=harry_regression.document) == expected

    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
    )
    narrative = templates["narrative_analysis"]
    pedagogical = templates["pedagogical_analysis"]
    assert "source_text" in narrative
    assert "inferred" in narrative and "explicit" in narrative
    assert "glossary_candidates" in narrative
    assert "It winked." in pedagogical
    assert "READ_AS_IS" in pedagogical

    section = harry_regression.document.section(harry_regression.section_ids["ch1"])
    sound_context = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
        name_translation_policy=NameTranslationPolicy(
            version=1,
            strategy=NameTranslationStrategy.SOUND,
        ),
    ).for_chapter_narrative_analysis(harry_regression.document, section)
    assert "маглы" in sound_context.request.instructions
    assert "Поз" in sound_context.request.instructions


def test_harry_potter_epub_has_russian_inferred_speakers_only_on_opening_dialogues(
    harry_regression: _Case,
) -> None:
    """Inferred opening lines keep a Russian name on the reading unit.

    The translation paragraph has no speaker span. An untranslated line keeps
    «Говорит …» in its notes; a translated line does not repeat the name there.
    """
    _assert_no_speaker_span(harry_regression.pages)
    found = [
        (line.key, unit.speaker)
        for line in _LINES
        for unit in _layout_units(harry_regression, line.key)
        if unit.speaker is not None
    ]
    assert found == [
        ("nasty", "миссис Дёрсли"),
        ("shoo", "мистер Дёрсли"),
        ("oh-yes", "мистер Дёрсли"),
        ("giant", "Хагрид"),
        ("demanded", "тётя Петуния"),
        ("car", "дядя Вернон"),
    ]
    for key, name in _SHOWN_SPEAKERS.items():
        units = _layout_units(harry_regression, key)
        assert [unit.speaker for unit in units] == [name]
        assert all(unit.translation for unit in units)
        assert _speaker_notes(units) == ()
        paragraph = _child(_unit(harry_regression.pages, _BY_KEY[key].text), "lg-tr")
        assert paragraph is not None
        assert _child(paragraph, "lg-speaker") is None
        assert _LATIN.search(name) is None
    for key, name in _UNTRANSLATED_SPEAKERS.items():
        units = _layout_units(harry_regression, key)
        assert [unit.speaker for unit in units] == [name]
        assert all(unit.translation is None for unit in units)
        assert _speaker_notes(units) == (f"Говорит {name}",)
        rendered = _unit(harry_regression.pages, _BY_KEY[key].text)
        assert _child(rendered, "lg-tr") is None
        notes = _child(rendered, "lg-notes")
        assert notes is not None
        assert f"Говорит {name}" in _text(notes)
        assert _LATIN.search(name) is None
    for key in _HIDDEN_SPEAKERS:
        units = _layout_units(harry_regression, key)
        assert all(unit.speaker is None for unit in units)
        assert _speaker_notes(units) == ()


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
    assert harry_regression.layout.who_is_who == ()
    _assert_who_page_absent(harry_regression.epub_path, harry_regression.pages)
    cards = _who_entries(document, harry_regression.analyses)
    harry = cards["Harry Potter"]
    assert _entry_tail(harry) == f"\u2014 Гарри Поттер. {_HARRY_DESCRIPTION}"
    assert harry.first_chapter == 1
    assert cards["Piers"].first_chapter == 3
    assert all(_PROMO_HARRY_DESCRIPTION not in entry.description for entry in cards.values())


def test_who_is_who_cards_contain_no_english_descriptions(harry_regression: _Case) -> None:
    """A card description is Russian, or absent when the first one was English.

    Bare `Dursley` is a family surname, not a character card. The book layout
    leaves the list empty and does not write `who-is-who.xhtml`.
    """
    assert harry_regression.layout.who_is_who == ()
    _assert_who_page_absent(harry_regression.epub_path, harry_regression.pages)
    cards = _who_entries(harry_regression.document, harry_regression.analyses)
    descriptions = "\n".join(entry.description for entry in cards.values())
    for description in _ENGLISH_DESCRIPTIONS:
        assert description not in descriptions
    assert _DUDLEY_LATER_DESCRIPTION not in descriptions
    assert _HARRY_LATER_DESCRIPTION not in descriptions
    assert _entry_tail(cards["Dudley"]) == "\u2014 Дадли."
    assert "Dursley" not in cards
    assert _entry_tail(cards["Mrs Dursley"]) == f"\u2014 миссис Дёрсли. {_MRS_DURSLEY_DESCRIPTION}"
    assert _entry_tail(cards["Vernon"]) == f"\u2014 Вернон. {_VERNON_DESCRIPTION}"
    assert _entry_tail(cards["Piers"]) == f"\u2014 Пирз. {_PIERS_DESCRIPTION}"
    for entry in cards.values():
        assert entry.description == "" or _LATIN.search(entry.description) is None
        assert entry.name_method is None
    assert cards["Harry Potter"].first_chapter == 1
    assert cards["Harry Potter"].description == _HARRY_DESCRIPTION
    assert cards["Dudley"].description == ""


_CLOAK_FIRST = (
    "Nothing like this man had ever been seen on Privet Drive: he was tall, thin and very old, "
    "judging by the silver of his hair and beard, which were both long enough to tuck into his belt, "
    "and his lined face was half hidden by the dusk that hung over the quiet street tonight."
)
_CLOAK_SECOND = (
    "He was wearing long robes, a purple cloak that swept the ground, and high-heeled buckled boots, "
    "and the old man in the violet cloak walked past the neat gardens of Privet Drive without a word "
    "to anyone waiting there under the street lamps."
)
_CLOAK = f"{_CLOAK_FIRST} {_CLOAK_SECOND}"
_AGREEMENT_QUOTE = "that's right, that's what I heard"
_AGREEMENT_LINE = f"He muttered {_AGREEMENT_QUOTE} about the violet cloak on the step."
_CLOAK_RU_FIRST = "На Тисовой улице такого человека ещё не видели."
_CLOAK_RU_SECOND = "Старик в фиолетовом плаще прошёл мимо садов."
_HOWARD_DESCRIPTION = (
    "Распространённое мужское имя; так мистер Дёрсли неуверенно называет сына Поттеров."
)
_PETUNIA_WIFE = "Тётя Гарри, сестра Лили и жена мистера Дёрсли."
_MRS_POTTER_MOTHER = "Мать Гарри, жена Джеймса."
_LILY_MOTHER = "Мать Гарри, волшебница."
_EDITORIAL_BODY = "Плащ старого волшебника"
_AGREEMENT_NOTE = "Разговорное подтверждение чужих слов, без нового смысла."


def _archive_names(path: Path) -> set[str]:
    with ZipFile(path) as archive:
        return set(archive.namelist())


def _spine_hrefs(path: Path) -> tuple[str, ...]:
    with ZipFile(path) as archive:
        package = etree.fromstring(archive.read("OEBPS/content.opf"))
    href_by_id = {
        item.get("id"): item.get("href")
        for item in package.iter()
        if _local(item) == "item" and item.get("id") and item.get("href")
    }
    return tuple(
        href_by_id[item.get("idref")]
        for item in package.iter()
        if _local(item) == "itemref"
    )


def _nav_hrefs(path: Path) -> tuple[str, ...]:
    with ZipFile(path) as archive:
        navigation = etree.fromstring(archive.read("OEBPS/nav.xhtml"))
    return tuple(
        item.get("href")
        for item in navigation.iter()
        if _local(item) == "a" and item.get("href")
    )


def _stylesheet(path: Path) -> str:
    with ZipFile(path) as archive:
        return archive.read("OEBPS/linguary.css").decode()


def _page_with(pages: Sequence[etree._Element], snippet: str) -> etree._Element:
    found = [page for page in pages if snippet in _text(page)]
    assert len(found) == 1, snippet
    return found[0]


def _note_terms(unit: etree._Element) -> list[str]:
    terms: list[str] = []
    for notes in unit.iter():
        if _local(notes) != "ol" or not _has_class(notes, "lg-notes"):
            continue
        for item in notes:
            if _local(item) != "li":
                continue
            term = _child(item, "lg-term")
            terms.append("" if term is None or term.text is None else term.text)
    return terms


def _method_evidence(method: NameTranslationMethod) -> NameTranslationEvidence:
    """Provenance that would show a method label if any method were shown.

    `_name_method` returns none when evidence is missing, before it looks at
    `_SHOWN_METHODS`. A candidate without evidence cannot lock that set.
    """
    return NameTranslationEvidence(
        version=1,
        conditions_id="hp-a2-regression",
        policy_version=1,
        strategy=NameTranslationStrategy.SOUND,
        observations=(
            NameTranslationObservation(
                suggested_translation="проверка способа",
                confidence=0.9,
                model_flagged_review=False,
                assessment=NameTranslationAssessment(
                    applied_method=method,
                    eligible_without_review=True,
                    review_reasons=(),
                    rationale="The label stays hidden while shown methods are empty.",
                ),
            ),
        ),
    )


def _feedback_block(
    section_id: str,
    resource: str,
    order: int,
    index: int,
    text: str,
    *,
    segments: tuple[BlockSegment, ...] = (),
) -> DocumentBlock:
    return DocumentBlock.create(
        document_id=_BOOK_ID,
        section_id=section_id,
        type=BlockType.PARAGRAPH,
        order=order,
        text=text,
        source_reference=SourceReference(resource, f"/body/p[{index}]"),
        segments=segments,
    )


def test_sixteen_epub_feedback_fixes_end_to_end(tmp_path: Path) -> None:
    """One learner export covers the sixteen Harry Potter reader remarks.

    The shared fixture stays untouched: Shoo and the sharply line there already
    start as glossary-only, and that book has no cover, cloak paragraph, or
    Howard. This book feeds those replies in as model TRANSLATE.
    """
    from linguary.application.context_builder import snapshot_prompt_templates

    assert len(_CLOAK) == 513
    cloak_segments = segment_text(_CLOAK)
    assert len(cloak_segments) == 2
    assert all(segment.end - segment.start <= 350 for segment in cloak_segments)

    cover_id, title_id, promo_id, chapter_id = "sec-cover", "sec-title", "sec-promo", "sec-cloak"
    cover_blocks = (
        _feedback_block(cover_id, "cover.xhtml", 0, 1, "Harry Potter and the Philosopher's Stone"),
    )
    title_blocks = (_feedback_block(title_id, "title.xhtml", 0, 1, "Harry Potter"),)
    promo_blocks = tuple(
        _feedback_block(promo_id, "hp01_experience_en-gb.html", index, index + 1, text)
        for index, text in enumerate((_PROMO_CREDIT, _PROMO_BLURB, _PROMO_VISIT))
    )
    cloak = _feedback_block(
        chapter_id, "hp01_ch001_en-gb.html", 1, 1, _CLOAK, segments=cloak_segments
    )
    shoo = _feedback_block(chapter_id, "hp01_ch001_en-gb.html", 2, 2, _SHOO)
    sharply = _feedback_block(chapter_id, "hp01_ch001_en-gb.html", 3, 3, _NO_WHY)
    agreement = _feedback_block(chapter_id, "hp01_ch001_en-gb.html", 4, 4, _AGREEMENT_LINE)
    heading = _heading(chapter_id, "hp01_ch001_en-gb.html", "— CHAPTER ONE —")
    sections = (
        DocumentSection(
            id=cover_id,
            document_id=_BOOK_ID,
            title="Cover",
            order=0,
            source_resource_path="cover.xhtml",
            blocks=cover_blocks,
        ),
        DocumentSection(
            id=title_id,
            document_id=_BOOK_ID,
            title="Title Page",
            order=1,
            source_resource_path="title.xhtml",
            blocks=title_blocks,
        ),
        DocumentSection(
            id=promo_id,
            document_id=_BOOK_ID,
            title="Pottermore.com",
            order=2,
            source_resource_path="hp01_experience_en-gb.html",
            blocks=promo_blocks,
        ),
        DocumentSection(
            id=chapter_id,
            document_id=_BOOK_ID,
            title="Chapter 1 - The Boy Who Lived",
            order=3,
            source_resource_path="hp01_ch001_en-gb.html",
            blocks=(heading, cloak, shoo, sharply, agreement),
        ),
    )
    document = CanonicalDocument(
        id=_BOOK_ID,
        metadata=BookMetadata(title="Harry Potter and the Philosopher's Stone"),
        source_language="en",
        target_language="ru",
        sections=sections,
    )
    assert document.is_cover_section(sections[0]) is True
    assert document.is_front_matter(sections[1]) is True
    assert document.is_cover_section(sections[1]) is False
    assert document.is_front_matter(sections[2]) is True
    assert document.is_cover_section(sections[2]) is False
    assert document.is_front_matter(sections[3]) is False

    point = (RequiredPoint("hint", "Короткая словарная подсказка."),)
    normalized = normalize_a2_dialogue_response(
        PedagogicalAnalysisResponse(
            treatment_plan=(
                TreatmentPlanItemResponse(shoo.id, BlockTreatment.TRANSLATE),
                TreatmentPlanItemResponse(
                    sharply.id,
                    BlockTreatment.TRANSLATE,
                    (
                        SegmentTreatmentResponse(1, BlockTreatment.TRANSLATE),
                        SegmentTreatmentResponse(2, BlockTreatment.TRANSLATE),
                    ),
                ),
            ),
            difficulties=(
                QuotedDifficultyResponse(
                    "d-shoo",
                    (QuotedSourceAnchorResponse(shoo.id, "Shoo!"),),
                    shoo.id,
                    "vocabulary",
                    "useful",
                    "explanation",
                    point,
                ),
                QuotedDifficultyResponse(
                    "d-sharply",
                    (QuotedSourceAnchorResponse(sharply.id, "sharply"),),
                    sharply.id,
                    "vocabulary",
                    "useful",
                    "explanation",
                    point,
                ),
            ),
            before_you_read=(BeforeYouReadEntryResponse("Bonfire Night", "Ночь костров, 5 ноября."),),
        ),
        (shoo, sharply),
        {shoo.id: shoo.id, sharply.id: sharply.id},
        reader_level="a2",
        english_help=False,
    )
    assert normalized.before_you_read == ()
    modes = {item.source_block_id: item for item in normalized.treatment_plan}
    assert modes[shoo.id].treatment is BlockTreatment.GLOSSARY_ONLY
    assert modes[sharply.id].treatment is BlockTreatment.GLOSSARY_ONLY
    assert [segment.treatment for segment in modes[sharply.id].segments] == [
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.GLOSSARY_ONLY,
    ]
    hints_by_owner: dict[str, list[QuotedDifficultyResponse]] = {}
    for difficulty in normalized.difficulties:
        hints_by_owner.setdefault(difficulty.owner_block_id, []).append(difficulty)
    assert [item.help_kind for item in hints_by_owner[shoo.id]] == ["lexical_hint"]
    assert [item.help_kind for item in hints_by_owner[sharply.id]] == ["lexical_hint"]
    assert hints_by_owner[shoo.id][0].anchors[0].quote == "Shoo!"
    assert hints_by_owner[sharply.id][0].anchors[0].quote == "sharply"

    cloak_translation = f"{_CLOAK_RU_FIRST} {_CLOAK_RU_SECOND}"
    results = {
        cloak.id: BlockProcessingResult(
            block_id=cloak.id,
            treatment=BlockTreatment.TRANSLATE,
            translated_text=cloak_translation,
            segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
            segment_translations=(_CLOAK_RU_FIRST, _CLOAK_RU_SECOND),
        ),
        shoo.id: BlockProcessingResult(
            block_id=shoo.id,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(GlossaryHint("Shoo!", "Брысь!", "Возглас, которым прогоняют кошку."),),
        ),
        sharply.id: BlockProcessingResult(
            block_id=sharply.id,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(
                GlossaryHint("sharply", "резко", "sharply — резко, отрывисто."),
            ),
        ),
        agreement.id: BlockProcessingResult(
            block_id=agreement.id,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(
                GlossaryHint(
                    _AGREEMENT_QUOTE,
                    "именно так, я это и слышал",
                    _AGREEMENT_NOTE,
                ),
                GlossaryHint(
                    "violet cloak",
                    "фиолетовый плащ",
                    f"{_EDITORIAL_BODY}; вариант стоит проверить",
                ),
            ),
        ),
    }
    analysis_id = "analysis-sixteen"
    people = (
        ("Mrs Dursley", "миссис Дёрсли", "Тётя Гарри, жена мистера Дёрсли; в финале главы она обнаружит Гарри."),
        ("Petunia", "Петуния", _PETUNIA_WIFE),
        ("Lily Potter", "Лили Поттер", _LILY_MOTHER),
        ("Mrs Potter", "миссис Поттер", _MRS_POTTER_MOTHER),
        ("the Potters", "Поттеры", "Семья родителей Гарри."),
        ("Dursley", "Дёрсли", _DURSLEY_DESCRIPTION),
        ("Howard", "Говард", _HOWARD_DESCRIPTION),
    )
    byr = (BeforeYouReadEntry("Bonfire Night", "Ночь костров, 5 ноября."),)
    story_blocks = (cloak, shoo, sharply, agreement)
    evidenced = {
        "Mrs Dursley": NameTranslationMethod.CREATIVE,
        "Lily Potter": NameTranslationMethod.MIXED,
    }
    candidates = []
    for source, translation, description in people:
        candidate = _candidate(analysis_id, source, translation, description)
        method = evidenced.get(source)
        if method is not None:
            candidate = replace(candidate, name_translation_evidence=_method_evidence(method))
        candidates.append(candidate)
    analysis = ChapterAnalysis.create(
        id=analysis_id,
        section_id=chapter_id,
        treatment_plan=TreatmentPlan.of_treatments(
            chapter_id,
            (
                (cloak.id, BlockTreatment.TRANSLATE),
                (shoo.id, BlockTreatment.GLOSSARY_ONLY),
                (sharply.id, BlockTreatment.GLOSSARY_ONLY),
                (agreement.id, BlockTreatment.GLOSSARY_ONLY),
            ),
        ),
        translation_instructions="Keep the glossary names.",
        chapter_summary="An old man in a violet cloak leaves Harry on the step.",
        expected_block_ids=tuple(block.id for block in story_blocks),
        glossary_candidates=tuple(candidates),
        before_you_read=byr,
    )
    layout = build_book_layout(
        LayoutInputs(
            document=document,
            results=results,
            analyses=(analysis,),
            concepts=(),
            reader_level="a2",
            english_help=False,
        )
    )
    assert layout.chapters[chapter_id].who_is_who == ()
    assert layout.who_is_who == ()
    assert all(chapter.review == () for chapter in layout.chapters.values())
    assert "before_you_read" not in layout.features.sections
    assert "who_is_who" not in layout.features.sections
    assert "review" not in layout.features.sections
    entries = book_who_is_who(document, (analysis,), ())
    who_sources = [entry.source for entry in entries]
    assert who_sources.count("Mrs Dursley") == 1
    assert "Petunia" not in who_sources
    assert who_sources.count("Lily Potter") == 1
    assert "Mrs Potter" not in who_sources
    assert "the Potters" not in who_sources
    assert "Dursley" not in who_sources
    assert "Howard" not in who_sources
    assert all(entry.name_method is None for entry in entries)

    target = tmp_path / "sixteen-remarks.epub"
    EpubExportWriter().write(
        EpubWriteRequest(
            document=document,
            target_path=str(target),
            block_results=results,
            learning_materials={
                chapter_id: LearningMaterial.for_chapter(
                    section_id=chapter_id,
                    before_you_read=byr,
                    chapter_summary=analysis.chapter_summary,
                    language_notes=(),
                )
            },
            layout=layout,
        )
    )
    names = _archive_names(target)
    assert "OEBPS/cover.xhtml" in names
    assert "OEBPS/title.xhtml" not in names
    assert "OEBPS/hp01_experience_en-gb.html" not in names
    spine = _spine_hrefs(target)
    navigation = _nav_hrefs(target)
    assert "cover.xhtml" in spine
    assert "cover.xhtml" in navigation
    for dropped in ("title.xhtml", "hp01_experience_en-gb.html"):
        assert dropped not in spine
        assert dropped not in navigation

    pages = _read_pages(target)
    _assert_who_page_absent(target, pages)
    assert not any(
        _has_class(element, "lg-who")
        or _has_class(element, "lg-method")
        or _has_class(element, "lg-review")
        for page in pages
        for element in page.iter()
    )
    rendered = "\n".join(_text(page) for page in pages)
    assert "Повторим" not in rendered
    assert "Кто есть кто" not in rendered
    assert "звучание и смысл" not in rendered
    assert "вольный перевод" not in rendered
    assert "Распространённое мужское имя" not in rendered
    assert "неуверенно называет" not in rendered
    chapter_page = _page_with(pages, "quiet street tonight")
    assert not any(_has_class(element, "linguary-before-you-read") for element in chapter_page.iter())
    cards = set(who_sources)
    assert "Mrs Dursley" in cards
    assert "Petunia" not in cards
    assert "Lily Potter" in cards
    assert "Mrs Potter" not in cards
    for absent in ("the Potters", "Dursley", "Howard"):
        assert absent not in cards

    assert _unit(pages, "quiet street tonight") is not _unit(pages, "under the street lamps")
    shoo_unit = _unit(pages, _SHOO)
    sharply_unit = _unit(pages, _NO_WHY)
    assert _translation(shoo_unit) is None
    assert _translation(sharply_unit) is None
    assert _note_terms(shoo_unit) == ["Shoo!"]
    assert _note_terms(sharply_unit) == ["sharply"]

    notes_page = _page_with(pages, _AGREEMENT_QUOTE)
    notes_text = _text(notes_page)
    assert "вариант стоит проверить" not in notes_text
    assert "Разговорное подтверждение чужих слов" not in notes_text
    assert _EDITORIAL_BODY in notes_text
    assert "именно так, я это и слышал" in notes_text
    assert ".lg-notes p {\n  display: inline;\n  margin: 0;\n}" in _stylesheet(target)

    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
    )
    prompts = "\n".join(value for value in templates.values() if isinstance(value, str))
    for phrase in (
        "Bonfire Night",
        "Ночь костров",
        "cultural_reference",
        "хотя зато",
        "буквальные кальки",
        "окказиональный",
    ):
        assert phrase in prompts
    assert "(3–5 предложений)" in templates["narrative_analysis"]
    assert "3–5" in templates["analysis"]
    assert "3–5" in templates["fragment"]


_LETTERS = "The Letters from No One"
_LETTERS_RU = "Письма ниоткуда"
_FANCY = f"{_OPEN}Fancy seeing you here, Professor McGonagall.{_CLOSE}"
_FANCY_RU = "\u2014 Вот так встреча, профессор Макгонагал."
_GROANED = "Harry groaned."
_CHUCKLED = "Uncle Vernon chuckled."
_RUMBLING = "A low rumbling sound had broken the silence around them."
_NOT_KNOWING = (
    "not knowing he was special, not knowing he was famous, not knowing he would be "
    "woken in a few hours by Mrs Dursley's scream as she opened the front door to put "
    "out the milk bottles"
)
_THANKS = "thank you very much"
_FIVE_WORDS = "silence around them at night"
_SIX_WORDS = "silence around them at the night"
_NOTE_PARAGRAPH = " ".join((_RUMBLING, _NOT_KNOWING, _THANKS, _FIVE_WORDS, _SIX_WORDS))
_VERBS = "They sniggered then nodded vigorously and spoke stiffly."
_CHAPTER_NUMBER = "\u2014 CHAPTER THREE \u2014"
_DROPPED_LABELS = ("Грамматика.", "Идиома.", "Реалия.", "Смысл.")
_SUPERSCRIPTS = (
    "a.lg-ref sup,\n"
    ".lg-num {\n"
    "  font-size: 0.7em;\n"
    "  vertical-align: super;\n"
    "  line-height: 0;\n"
    "}"
)
_TERM_WEIGHT = ".lg-term {\n  font-weight: bold;\n}"


def _rendered_heading(pages: Sequence[etree._Element], snippet: str) -> etree._Element:
    """A heading unit. Headings are not marked ``lg-src``."""
    matches: list[etree._Element] = []
    for page in pages:
        for element in page.iter():
            if _local(element) != "div" or not _has_class(element, "lg-unit"):
                continue
            for child in element:
                if _local(child) in {"h1", "h2", "h3", "h4"} and snippet in _source_text(child):
                    matches.append(element)
                    break
    assert len(matches) == 1, snippet
    return matches[0]


def _term_tails(unit: etree._Element) -> list[tuple[str, str | None]]:
    """``lg-term`` text and the tail that separates it from the note paragraph."""
    found: list[tuple[str, str | None]] = []
    for notes in unit.iter():
        if _local(notes) != "ol" or not _has_class(notes, "lg-notes"):
            continue
        for item in notes:
            if _local(item) != "li":
                continue
            term = _child(item, "lg-term")
            if term is None or not any(_local(child) == "p" for child in item):
                continue
            found.append((term.text or "", term.tail))
    return found


def test_fifteen_ux_polish_and_a2_reader_fixes_end_to_end(tmp_path: Path) -> None:
    """One export covers the fifteen UX remarks and the A2 reader findings.

    No model call. Elementary lines stay READ_AS_IS. Short B1+ lines, including
    ``It winked.``, leave that mode. The book list is empty; card filtering is
    the direct ``book_who_is_who`` result.
    """
    from linguary.application.context_builder import ContextBuilder
    from linguary.application.explained_expressions import _same_word
    from linguary.domain.content_plan import (
        ContentEvidence,
        ContentPlan,
        Difficulty,
        HelpEvidence,
        artifact_key,
    )
    from linguary.domain.name_translation import NameTranslationPolicy

    assert _same_word("shoo", "shook") is False

    chapter_id = "sec-fifteen"
    resource = "hp01_ch003_en-gb.html"
    numbered = _heading(chapter_id, resource, _CHAPTER_NUMBER)
    title = DocumentBlock.create(
        document_id=_BOOK_ID,
        section_id=chapter_id,
        type=BlockType.HEADING,
        order=1,
        text=_LETTERS,
        source_reference=SourceReference(resource, "/body/h1[2]"),
    )
    paragraph_texts = (
        _WINKED,
        _FANCY,
        _GROANED,
        _CHUCKLED,
        _UP,
        _A_WHAT,
        _NEARLY,
        _SHOO,
        _NO_WHY,
        _HAGRID_LINE,
        _NOTE_PARAGRAPH,
        _VERBS,
    )
    paragraphs = tuple(
        _feedback_block(chapter_id, resource, index + 2, index + 1, text)
        for index, text in enumerate(paragraph_texts)
    )
    (
        winked,
        fancy,
        groaned,
        chuckled,
        up,
        a_what,
        nearly,
        shoo,
        no_why,
        hagrid,
        notes_block,
        verbs,
    ) = paragraphs
    story = (
        numbered,
        title,
        winked,
        fancy,
        groaned,
        chuckled,
        up,
        a_what,
        nearly,
        shoo,
        no_why,
        hagrid,
        notes_block,
        verbs,
    )
    initial = {
        hagrid.id: BlockTreatment.TRANSLATE,
        notes_block.id: BlockTreatment.TRANSLATE,
        verbs.id: BlockTreatment.GLOSSARY_ONLY,
    }
    point = (RequiredPoint("p-fifteen", "Объяснить эту конструкцию."),)
    normalized = normalize_a2_dialogue_response(
        PedagogicalAnalysisResponse(
            treatment_plan=tuple(
                TreatmentPlanItemResponse(
                    block.id,
                    initial.get(block.id, BlockTreatment.READ_AS_IS),
                )
                for block in story
            ),
            difficulties=(
                QuotedDifficultyResponse(
                    "d-from",
                    (QuotedSourceAnchorResponse(title.id, "from no one"),),
                    title.id,
                    "grammar",
                    "useful",
                    "explanation",
                    point,
                ),
                QuotedDifficultyResponse(
                    "d-relieved",
                    (QuotedSourceAnchorResponse(hagrid.id, "sounding relieved"),),
                    hagrid.id,
                    "grammar",
                    "useful",
                    "explanation",
                    point,
                ),
            ),
        ),
        story,
        {block.id: block.id for block in story},
        reader_level="a2",
        english_help=False,
    )
    modes = {item.source_block_id: item.treatment for item in normalized.treatment_plan}
    assert modes == {
        numbered.id: BlockTreatment.READ_AS_IS,
        title.id: BlockTreatment.TRANSLATE,
        winked.id: BlockTreatment.GLOSSARY_ONLY,
        fancy.id: BlockTreatment.TRANSLATE,
        groaned.id: BlockTreatment.GLOSSARY_ONLY,
        chuckled.id: BlockTreatment.GLOSSARY_ONLY,
        up.id: BlockTreatment.READ_AS_IS,
        a_what.id: BlockTreatment.READ_AS_IS,
        nearly.id: BlockTreatment.READ_AS_IS,
        shoo.id: BlockTreatment.READ_AS_IS,
        no_why.id: BlockTreatment.READ_AS_IS,
        hagrid.id: BlockTreatment.TRANSLATE,
        notes_block.id: BlockTreatment.TRANSLATE,
        verbs.id: BlockTreatment.GLOSSARY_ONLY,
    }
    assert [d for d in normalized.difficulties if not d.id.startswith("auto-vocab-")] == []
    assert normalized.difficulties
    assert all(
        d.kind == "vocabulary" and d.help_kind == "lexical_hint"
        for d in normalized.difficulties
    )

    sniggered = GlossaryHint(
        "sniggered",
        "хихикнули, засмеялись с ехидством",
        "Snigger — хихикать, смеяться с ехидством",
    )
    nodded = GlossaryHint("nodded", "кивнула", "Nod — кивнуть")
    vigorously = GlossaryHint("vigorously", "энергично", "Vigorously — энергично, сильно")
    stiffly = GlossaryHint("stiffly", "сухо", "Помета к глаголу речи: сказала сдержанно")
    stiffly_start = verbs.text.index("stiffly")
    stiffly_anchor = SourceAnchor(
        verbs.id, stiffly_start, stiffly_start + len("stiffly"), "stiffly"
    )
    label_kinds = (
        ("d-grammar", "grammar"),
        ("d-idiom", "idiom"),
        ("d-realia", "cultural_reference"),
        ("d-sense", "semantic_relation"),
        ("d-wordplay", "wordplay"),
    )
    difficulties = tuple(
        Difficulty(
            difficulty_id,
            (stiffly_anchor,),
            verbs.id,
            kind,
            "useful",
            "explanation",
            (RequiredPoint(f"p-{difficulty_id}", "Подсказка к слову stiffly."),),
        )
        for difficulty_id, kind in label_kinds
    )
    plan = ContentPlan(difficulties=difficulties)
    hint_content = {
        "original": stiffly.original,
        "translation": stiffly.translation,
        "note": stiffly.note,
    }
    hint_key = artifact_key("glossary_hint.note", verbs.id, hint_content)
    evidence = tuple(
        HelpEvidence(
            difficulty.id,
            difficulty.required_points[0].id,
            "glossary_hint.note",
            verbs.id,
            hint_key,
            stiffly.note or "",
        )
        for difficulty in difficulties
    )
    rumbling_hint = GlossaryHint(
        _RUMBLING, "Низкий рокот нарушил тишину.", "Звук шёл из-за двери."
    )
    passage_hint = GlossaryHint(
        _NOT_KNOWING, "не зная, что он особенный", "Гарри ещё не знает о себе."
    )
    thanks_hint = GlossaryHint(
        _THANKS,
        "уж поверьте",
        "Ироничная присказка со значением \u00abУж Поверьте\u00bb, а не благодарность.",
    )
    five_hint = GlossaryHint(_FIVE_WORDS, "коротко", "Пояснение к короткой фразе.")
    six_hint = GlossaryHint(_SIX_WORDS, "чуть длиннее", "Пояснение к фразе из шести слов.")
    russians = {
        title.id: _LETTERS_RU,
        fancy.id: _FANCY_RU,
        hagrid.id: _BY_KEY["hagrid-line"].russian,
        notes_block.id: "Полный перевод абзаца.",
    }
    results = {
        block.id: BlockProcessingResult(block.id, BlockTreatment.READ_AS_IS)
        for block in (numbered, up, a_what, nearly, shoo, no_why)
    }
    results[winked.id] = BlockProcessingResult(
        winked.id,
        BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(GlossaryHint("winked", "подмигнула"),),
    )
    results[groaned.id] = BlockProcessingResult(
        groaned.id,
        BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(GlossaryHint("groaned", "охнул"),),
    )
    results[chuckled.id] = BlockProcessingResult(
        chuckled.id,
        BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(GlossaryHint("chuckled", "усмехнулся"),),
    )
    results[verbs.id] = BlockProcessingResult(
        verbs.id,
        BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(sniggered, nodded, vigorously, stiffly),
        content_evidence=ContentEvidence(plan.sha256, evidence),
    )
    for block_id, russian in russians.items():
        hints = ()
        if block_id == notes_block.id:
            hints = (rumbling_hint, passage_hint, thanks_hint, five_hint, six_hint)
        assert russian is not None
        results[block_id] = BlockProcessingResult(
            block_id,
            BlockTreatment.TRANSLATE,
            translated_text=russian,
            glossary_hints=hints,
        )
    analysis_id = "analysis-fifteen"
    people = (
        ("cine-camera", "кинокамера", "камера для съёмки"),
        ("crutches", "костыли", "опоры"),
        ("cupboard", "шкаф", "шкаф под лестницей"),
        ("racing bike", "гоночный велосипед", "велосипед, на котором ездит Гарри"),
        ("Sirius Black", "Сириус Блэк", "Крёстный Гарри; в этой главе упоминается впервые"),
    )
    analysis = ChapterAnalysis.create(
        id=analysis_id,
        section_id=chapter_id,
        treatment_plan=TreatmentPlan.of_treatments(
            chapter_id,
            tuple((block.id, modes[block.id]) for block in story),
        ),
        translation_instructions="Keep the glossary names.",
        chapter_summary="Short lines, a chapter title, and footnotes.",
        expected_block_ids=tuple(block.id for block in story),
        glossary_candidates=tuple(
            _candidate(analysis_id, source, translation, description)
            for source, translation, description in people
        ),
        content_plan=plan,
    )
    document = CanonicalDocument(
        id=_BOOK_ID,
        metadata=BookMetadata(title="Harry Potter and the Philosopher's Stone"),
        source_language="en",
        target_language="ru",
        sections=(
            DocumentSection(
                id=chapter_id,
                document_id=_BOOK_ID,
                title="Chapter 3 - The Letters from No One",
                order=0,
                source_resource_path=resource,
                blocks=story,
            ),
        ),
    )
    kept = book_who_is_who(document, (analysis,), ())
    assert [entry.source for entry in kept] == ["Sirius Black"]
    assert kept[0].translation == "Сириус Блэк"
    assert kept[0].description == "Крёстный Гарри"
    assert kept[0].first_chapter == 1
    assert "в этой главе упоминается впервые" not in kept[0].description
    for source, _translation_text, _description in people[:4]:
        assert source not in {entry.source for entry in kept}

    layout = build_book_layout(
        LayoutInputs(
            document=document,
            results=results,
            analyses=(analysis,),
            concepts=(),
            reader_level="a2",
            english_help=False,
        )
    )
    assert layout.who_is_who == ()
    assert layout.chapters[chapter_id].who_is_who == ()
    assert layout.chapters[chapter_id].review == ()
    assert "who_is_who" not in layout.features.sections
    assert "review" not in layout.features.sections

    def one(block_id: str) -> LayoutUnit:
        units = layout.chapters[chapter_id].units[block_id]
        assert len(units) == 1
        return units[0]

    footnote = [item.note for item in one(notes_block.id).notes]
    assert [note.term for note in footnote] == [
        None,
        None,
        _THANKS,
        _FIVE_WORDS,
        None,
    ]
    assert footnote[0].body == ("Звук шёл из-за двери.",)
    assert "Низкий рокот" not in "".join(footnote[0].body)
    assert footnote[1].body == ("Гарри ещё не знает о себе.",)
    assert "не зная, что он особенный" not in "".join(footnote[1].body)
    assert footnote[2].body == (
        "Ироничная присказка со значением \u00abУж Поверьте\u00bb, а не благодарность.",
    )
    assert "".join(footnote[2].body).casefold().count("уж поверьте") == 1
    assert footnote[3].body == ("коротко", "Пояснение к короткой фразе.")
    assert footnote[4].body == ("Пояснение к фразе из шести слов.",)
    assert "чуть длиннее" not in "".join(footnote[4].body)
    assert _RUMBLING not in "".join(footnote[0].body)
    assert _NOT_KNOWING not in "".join(footnote[1].body)

    verb_notes = [item.note for item in one(verbs.id).notes]
    assert [note.term for note in verb_notes] == ["to snigger", "to nod", "vigorously", "stiffly"]
    assert verb_notes[0].body == ("хихикать, засмеяться с ехидством",)
    assert "Snigger" not in "".join(verb_notes[0].body)
    assert verb_notes[1].body == ("кивнуть",)
    assert verb_notes[2].body == ("энергично",)
    assert "Vigorously" not in "".join(verb_notes[2].body)
    assert verb_notes[3].labels == ("Игра слов.",)
    assert verb_notes[3].difficulty_kinds == (
        "grammar",
        "idiom",
        "cultural_reference",
        "semantic_relation",
        "wordplay",
    )
    assert verb_notes[3].body == ("сухо", "сказала сдержанно")
    assert "Помета" not in "".join(verb_notes[3].body)
    assert one(title.id).translation == _LETTERS_RU
    assert one(title.id).notes == ()
    assert one(hagrid.id).notes == ()
    for block in (up, a_what, nearly, shoo, no_why, numbered):
        assert one(block.id).translation is None

    target = tmp_path / "fifteen-remarks.epub"
    EpubExportWriter().write(
        EpubWriteRequest(
            document=document,
            target_path=str(target),
            block_results=results,
            layout=layout,
        )
    )
    pages = _read_pages(target)
    _assert_who_page_absent(target, pages)
    _assert_no_speaker_span(pages)
    stylesheet = _stylesheet(target)
    assert _SUPERSCRIPTS in stylesheet
    assert _TERM_WEIGHT in stylesheet
    rendered = "\n".join(_text(page) for page in pages)
    assert "Повторим" not in rendered
    assert "Кто есть кто" not in rendered
    for label in _DROPPED_LABELS:
        assert label not in rendered
    assert "Помета к глаголу речи:" not in rendered
    assert "Игра слов." in rendered
    assert "сказала сдержанно" in rendered

    gaps = [
        item
        for page in pages
        for element in page.iter()
        if _local(element) == "div" and _has_class(element, "lg-unit")
        for item in _term_tails(element)
    ]
    assert gaps
    assert all(tail == " \u2014 " for _term, tail in gaps)
    assert ("to snigger", " \u2014 ") in gaps
    assert ("to nod", " \u2014 ") in gaps
    assert ("vigorously", " \u2014 ") in gaps
    assert (_THANKS, " \u2014 ") in gaps

    notes_unit = _unit(pages, _RUMBLING)
    notes_text = _text(_child(notes_unit, "lg-notes"))
    assert _RUMBLING not in notes_text
    assert _NOT_KNOWING not in notes_text
    assert "Низкий рокот" not in notes_text
    assert "не зная, что он особенный" not in notes_text
    assert notes_text.casefold().count("уж поверьте") == 1
    assert _note_terms(notes_unit) == ["", "", _THANKS, _FIVE_WORDS, ""]
    verb_unit = _unit(pages, "sniggered")
    assert _note_terms(verb_unit) == ["to snigger", "to nod", "vigorously", "stiffly"]
    assert "Snigger" not in _text(_child(verb_unit, "lg-notes"))
    assert "Vigorously" not in _text(_child(verb_unit, "lg-notes"))

    letters = _rendered_heading(pages, _LETTERS)
    assert _translation(letters) == _LETTERS_RU
    assert _child(letters, "lg-notes") is None
    chapter_heading = _rendered_heading(pages, "CHAPTER THREE")
    assert _translation(chapter_heading) is None
    hagrid_unit = _unit(pages, "sounding relieved")
    assert _translation(hagrid_unit) == _BY_KEY["hagrid-line"].russian
    assert _child(hagrid_unit, "lg-notes") is None
    assert _translation(_unit(pages, _WINKED)) is None
    assert _note_terms(_unit(pages, _WINKED)) == ["to wink"]
    assert _translation(_unit(pages, _FANCY)) == _FANCY_RU
    assert _translation(_unit(pages, _GROANED)) is None
    assert _note_terms(_unit(pages, _GROANED)) == ["to groan"]
    assert _translation(_unit(pages, _CHUCKLED)) is None
    assert _note_terms(_unit(pages, _CHUCKLED)) == ["to chuckle"]
    for text in (_UP, _A_WHAT, _NEARLY, _SHOO, _NO_WHY):
        assert _translation(_unit(pages, text)) is None

    section = document.section(chapter_id)
    sound = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
        name_translation_policy=NameTranslationPolicy(
            version=1,
            strategy=NameTranslationStrategy.SOUND,
        ),
    ).for_chapter_narrative_analysis(document, section)
    assert "«McGonagall» → «Макгонагалл»" in sound.request.instructions


def _span_end(text: str, surface: str) -> int:
    """End offset of `surface` in `text`."""
    return text.index(surface) + len(surface)


def _ref_superscripts(unit: etree._Element) -> list[str]:
    """Digits of ``a.lg-ref`` markers in this unit, in source order."""
    found: list[str] = []
    for element in unit.iter():
        if _local(element) != "a" or not _has_class(element, "lg-ref"):
            continue
        digits = [child.text for child in element if _local(child) == "sup" and child.text]
        assert digits
        found.extend(digits)
    return found


def _glossary_notes(layout: BookLayout, section_id: str, block_id: str) -> tuple[NumberedNote, ...]:
    units = layout.chapters[section_id].units[block_id]
    return tuple(
        item
        for unit in units
        for item in unit.notes
        if item.note.origin == "glossary_hint"
    )


def test_a2_student_linguist_and_teacher_review_fixes_end_to_end(tmp_path: Path) -> None:
    """One Harry Potter pass covers the student, linguist, and teacher fixes.

    No model call. Infinitive hints and ``To his surprise`` get a marker and a
    superscript. The tube note drops its unquoted duplicate, Diagon Alley drops
    the disclaimer, and the owl note says Are there. Short B1+ replies stay
    TRANSLATE. ``— CHAPTER THREE —`` does not. Chapter 5 drops the captain
    notes and keeps ``could tell``. Chapter 4 block 12 loses the later
    speech-style spoiler. ``said Harry`` is explicit. ``Петунья`` becomes
    ``Петуния``. Dialogue dashes join. The A2 prompt snapshot keeps the new rules.
    """
    from linguary.application.analysis_merge import (
        combine_staged_analysis_responses,
        content_plan_from_response,
    )
    from linguary.application.context_builder import _owned_content_plan, snapshot_prompt_templates
    from linguary.application.llm_dto import (
        ChapterAnalysisResponse,
        NarrativeAnalysisResponse,
        QuotedContentPlanResponse,
        QuotedContextFactResponse,
    )
    from linguary.application.reader_text import ReaderTextPolicy
    from linguary.domain import GlossarySnapshotEntry, TermStatus, TranslationUnit
    from linguary.domain.content_plan import Difficulty

    flutter = "None of them noticed a large tawny owl flutter past the window."
    hugged = "And the old man hugged Mr Dursley around the middle and walked off."
    bowed = "Dumbledore bowed his head. Professor McGonagall gasped."
    grab = "Harry made a grab for the letter but Uncle Vernon knocked his hand out of the way."
    upright = "The whole shack shivered and Harry sat bolt upright, staring at the door."
    thrown_to = "Hagrid threw it to Harry."
    write_down = "Could you write it down?"
    surprise = "To his surprise, Hagrid chuckled."
    tube = "Dumbledore mentioned the London Underground."
    alley = "Diagon Alley filled with showers of owls."
    cannon_text = f"{_OPEN}Where{_CLOSE}s the cannon?{_CLOSE} he said stupidly."
    budge_text = f"{_OPEN}Budge up, yeh great lump,{_CLOSE} said the stranger."
    kept_text = f"{_OPEN}Kept what from me?{_CLOSE} said Harry eagerly."
    cannon_ru = "\u2014 Где пушка? \u2014 глупо сказал он."
    budge_ru = "\u2014 Подвинься, здоровяк, \u2014 сказал незнакомец."
    kept_ru = "\u2014 Что от меня скрыли? \u2014 с жадностью спросил Гарри."
    marked = (
        (flutter, ("to flutter",), ("flutter",), "owl flutter"),
        (hugged, ("to hug", "to walk off"), ("hugged", "walked off"), "hugged"),
        (bowed, ("to bow one's head",), ("bowed his head",), "bowed his head"),
        (
            grab,
            ("to make a grab for", "to knock out of the way"),
            ("made a grab for", "knocked his hand out of the way"),
            "made a grab",
        ),
        (upright, ("to sit bolt upright",), ("sat bolt upright",), "bolt upright"),
        (thrown_to, ("to throw to",), ("threw it to",), "threw it to"),
        (write_down, ("to write down",), ("write it down",), "write it down"),
        (surprise, ("to his surprise",), ("To his surprise",), "To his surprise"),
    )
    hint_rows = {
        flutter: (GlossaryHint("to flutter", "пролететь", "flutter — промелькнуть"),),
        hugged: (GlossaryHint("to hug", "обнять"), GlossaryHint("to walk off", "уйти")),
        bowed: (GlossaryHint("to bow one's head", "склонить голову"),),
        grab: (
            GlossaryHint("to make a grab for", "схватить"),
            GlossaryHint("to knock out of the way", "отвести в сторону"),
        ),
        upright: (GlossaryHint("to sit bolt upright", "сесть прямо"),),
        thrown_to: (GlossaryHint("to throw to", "бросить"),),
        write_down: (GlossaryHint("to write down", "записать"),),
        surprise: (GlossaryHint("to his surprise", "к его удивлению"),),
        tube: (
            GlossaryHint(
                "the London Underground",
                "Лондонское метро",
                "Лондонское метро. Дамблдор шутит, что шрам у него на колене "
                "похож на схему линий метро.",
            ),
        ),
        alley: (
            GlossaryHint(
                "Diagon Alley",
                "Косой переулок",
                "Название похоже на diagonally. "
                "Это можно заметить как возможную игру слов, но текст книги не подтверждает её прямо. "
                "Читается как diagonally, но текст книги не подтверждает её прямо.",
            ),
            GlossaryHint(
                "showers of owls",
                "потоки сов",
                "Вопрос построен без is it: Going to be any more showers of owls tonight, Harry?",
            ),
        ),
    }

    chapter_id = "sec-review"
    resource = "hp01_review_en-gb.html"
    numbered = _heading(chapter_id, resource, _CHAPTER_NUMBER)
    reply_rows = (
        (cannon_text, cannon_ru),
        (budge_text, budge_ru),
        (kept_text, kept_ru),
    )
    reply_blocks = tuple(
        _feedback_block(chapter_id, resource, index + 1, index + 1, text)
        for index, (text, _russian) in enumerate(reply_rows)
    )
    cannon, budge, kept = reply_blocks
    hint_texts = tuple(hint_rows)
    hint_blocks = tuple(
        _feedback_block(chapter_id, resource, index + 4, index + 4, text)
        for index, text in enumerate(hint_texts)
    )
    hints_by_text = dict(zip(hint_texts, hint_blocks, strict=True))
    story = (numbered, *reply_blocks, *hint_blocks)
    chapter_five_text = (
        "Harry kept his eyes shut tight. "
        f"{_OPEN}I was going to a school for wizards,{_CLOSE} he thought, in my cupboard. "
        "A giant called Hagrid. When I open my eyes I could tell."
    )
    chapter_five = _feedback_block(
        "sec-ch5", "hp01_ch005_en-gb.html", 0, 1, chapter_five_text,
    )
    read_as_is = _feedback_block(
        "sec-ch5", "hp01_ch005_en-gb.html", 1, 2, "The small cupboard was dark.",
    )
    mode_blocks = (numbered, cannon, budge, kept, chapter_five, read_as_is)
    point = (RequiredPoint("p-review", "Объяснить эту конструкцию."),)

    def grammar(difficulty_id: str, block: DocumentBlock, quote: str, kind: str) -> QuotedDifficultyResponse:
        return QuotedDifficultyResponse(
            difficulty_id,
            (QuotedSourceAnchorResponse(block.id, quote),),
            block.id,
            kind,
            "useful",
            "explanation",
            point,
        )

    mode_payload = PedagogicalAnalysisResponse(
        treatment_plan=tuple(
            TreatmentPlanItemResponse(
                block.id,
                BlockTreatment.READ_AS_IS if block is read_as_is else BlockTreatment.TRANSLATE,
            )
            for block in mode_blocks
        ),
        difficulties=(
            grammar("d-in-cupboard", chapter_five, "in my cupboard", "grammar"),
            grammar("d-eyes", chapter_five, "kept his eyes shut tight", "grammar"),
            grammar("d-giant", chapter_five, "a giant called Hagrid", "vocabulary"),
            grammar("d-wizards", chapter_five, "I was going to a school for wizards", "grammar"),
            grammar("d-open", chapter_five, "when I open my eyes", "grammar"),
            grammar("d-could", chapter_five, "could tell", "vocabulary"),
            grammar("d-read", read_as_is, "in my cupboard", "grammar"),
        ),
    )
    assert mode_payload.treatment_plan[0].treatment is BlockTreatment.TRANSLATE
    assert numbered.text == "\u2014 CHAPTER THREE \u2014"
    normalized = normalize_a2_dialogue_response(
        mode_payload,
        mode_blocks,
        {block.id: block.id for block in mode_blocks},
        reader_level="a2",
        english_help=False,
    )
    modes = {item.source_block_id: item.treatment for item in normalized.treatment_plan}
    assert modes[numbered.id] is BlockTreatment.READ_AS_IS
    assert modes[cannon.id] is BlockTreatment.TRANSLATE
    assert modes[budge.id] is BlockTreatment.TRANSLATE
    assert modes[kept.id] is BlockTreatment.TRANSLATE
    assert modes[chapter_five.id] is BlockTreatment.TRANSLATE
    assert modes[read_as_is.id] is BlockTreatment.READ_AS_IS
    assert {item.id for item in normalized.difficulties} == {"d-could", "d-read"}

    ch4_id = "sec-ch4"
    ch4_resource = "hp01_ch004_en-gb.html"
    ch4_speech = (
        f"{_OPEN}Budge up, yeh great lump, and sit down here,{_CLOSE} "
        "said the stranger to the boy."
    )
    ch4_spoiler = (
        f"{_OPEN}I mighta sat on it,{_CLOSE} he said. Got summat fer meself las' night."
    )
    ch4 = tuple(
        _feedback_block(
            ch4_id,
            ch4_resource,
            index,
            index + 1,
            ch4_speech if index == 11 else ch4_spoiler if index == 22 else "The hut stood by the sea.",
        )
        for index in range(23)
    )
    assert len(ch4) == 23
    assert ch4[11].text == ch4_speech
    assert "I mighta sat on it" in ch4[22].text
    speech = QuotedDifficultyResponse(
        "d-style",
        (QuotedSourceAnchorResponse(ch4[11].id, "yeh"),),
        ch4[11].id,
        "speech_style",
        "useful",
        "explanation",
        (
            RequiredPoint("p-yeh", "В этой реплике yeh значит you."),
            RequiredPoint("p-quote", 'Позже он скажет "I mighta sat on it" про торт.'),
            RequiredPoint(
                "p-lemmas",
                "summat = something, fer = for, las' = last, meself = myself",
            ),
        ),
    )
    speech_normalized = normalize_a2_dialogue_response(
        PedagogicalAnalysisResponse(
            treatment_plan=tuple(
                TreatmentPlanItemResponse(block.id, BlockTreatment.TRANSLATE) for block in ch4
            ),
            difficulties=(speech,),
        ),
        ch4,
        {block.id: block.id for block in ch4},
        reader_level="a2",
        english_help=False,
    )
    kept_style = next(item for item in speech_normalized.difficulties if item.id == "d-style")
    assert kept_style.owner_block_id == ch4[11].id
    assert [item.id for item in kept_style.required_points] == ["p-yeh"]
    assert all("I mighta sat on it" not in item.description for item in kept_style.required_points)
    assert all("summat = something" not in item.description for item in kept_style.required_points)

    harry_text = f"{_OPEN}Why?{_CLOSE} said Harry."
    stranger_text = f"{_OPEN}Why?{_CLOSE} said the stranger."
    speaker_resource = "hp01_ch005_speakers.html"
    harry = _feedback_block("sec-speakers", speaker_resource, 0, 1, harry_text)
    potter = _feedback_block("sec-speakers", speaker_resource, 1, 2, harry_text)
    stranger = _feedback_block("sec-speakers", speaker_resource, 2, 3, stranger_text)
    speaker_blocks = (harry, potter, stranger)

    def speaker_fact(block: DocumentBlock, fact_id: str, value: str) -> QuotedContextFactResponse:
        anchor = QuotedSourceAnchorResponse(block.id, "Why?")
        return QuotedContextFactResponse(
            id=fact_id,
            kind="speaker",
            anchors=(anchor,),
            subject_anchor=anchor,
            value=value,
            certainty="inferred",
        )

    combined = combine_staged_analysis_responses(
        NarrativeAnalysisResponse(
            "Keep the voice.",
            "Harry asks why.",
            context_facts=(
                speaker_fact(harry, "f-harry", "Harry"),
                speaker_fact(potter, "f-potter", "Harry Potter"),
                speaker_fact(stranger, "f-stranger", "Harry"),
            ),
        ),
        PedagogicalAnalysisResponse(
            treatment_plan=tuple(
                TreatmentPlanItemResponse(block.id, BlockTreatment.READ_AS_IS)
                for block in speaker_blocks
            ),
        ),
        resolved_blocks=speaker_blocks,
    )
    assert [(fact.id, fact.certainty) for fact in combined.content_plan.context_facts] == [
        ("f-harry", "explicit"),
        ("f-potter", "explicit"),
        ("f-stranger", "inferred"),
    ]
    speaker_aliases = {
        f"b{index + 1}": block.id for index, block in enumerate(speaker_blocks)
    }

    def quoted_speaker(alias: str, fact_id: str, value: str) -> QuotedContextFactResponse:
        anchor = QuotedSourceAnchorResponse(alias, "Why?")
        return QuotedContextFactResponse(
            id=fact_id,
            kind="speaker",
            anchors=(anchor,),
            subject_anchor=anchor,
            value=value,
            certainty="inferred",
        )

    speaker_plan = content_plan_from_response(
        ChapterAnalysisResponse(
            translation_instructions="Keep the voice.",
            chapter_summary="Harry asks why.",
            treatment_plan=tuple(
                TreatmentPlanItemResponse(alias, BlockTreatment.READ_AS_IS)
                for alias in speaker_aliases
            ),
            content_plan=QuotedContentPlanResponse(
                context_facts=(
                    quoted_speaker("b1", "f-harry", "Harry"),
                    quoted_speaker("b2", "f-potter", "Harry Potter"),
                    quoted_speaker("b3", "f-stranger", "Harry"),
                ),
            ),
        ),
        speaker_aliases,
        {block.id: block.text for block in speaker_blocks},
        visible_texts_by_alias={
            alias: block.text
            for alias, block in zip(speaker_aliases, speaker_blocks, strict=True)
        },
        blocks=speaker_blocks,
    )
    assert speaker_plan is not None
    grouped: dict[str, set[tuple[str, str | None, str]]] = {}
    for fact in speaker_plan.context_facts:
        grouped.setdefault(fact.subject_anchor.block_id, set()).add(
            (fact.kind, fact.value, fact.certainty),
        )
    assert grouped == {
        harry.id: {("speaker", "Harry", "explicit")},
        potter.id: {("speaker", "Harry Potter", "explicit")},
        stranger.id: {("speaker", "Harry", "inferred")},
    }

    results = {
        numbered.id: BlockProcessingResult(numbered.id, BlockTreatment.READ_AS_IS),
    }
    for block, russian in zip(reply_blocks, (cannon_ru, budge_ru, kept_ru), strict=True):
        results[block.id] = BlockProcessingResult(
            block.id,
            BlockTreatment.TRANSLATE,
            translated_text=russian,
        )
    for block in hint_blocks:
        results[block.id] = BlockProcessingResult(
            block.id,
            BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=hint_rows[block.text],
        )
    document = CanonicalDocument(
        id=_BOOK_ID,
        metadata=BookMetadata(title="Harry Potter and the Philosopher's Stone"),
        source_language="en",
        target_language="ru",
        sections=(
            DocumentSection(
                id=chapter_id,
                document_id=_BOOK_ID,
                title="Chapter 4 - The Keeper of the Keys",
                order=0,
                source_resource_path=resource,
                blocks=story,
            ),
        ),
    )
    layout = build_book_layout(
        LayoutInputs(
            document=document,
            results=results,
            analyses=(),
            concepts=(),
            reader_level="a2",
            english_help=False,
        )
    )
    for text, terms, surfaces, _snippet in marked:
        block = hints_by_text[text]
        notes = _glossary_notes(layout, chapter_id, block.id)
        assert [item.note.term for item in notes] == list(terms)
        assert len(layout.chapters[chapter_id].units[block.id]) == 1
        for item, surface in zip(notes, surfaces, strict=True):
            assert item.note.marker_end is not None
            assert item.note.marker_end > 0
            assert item.note.marker_end == _span_end(text, surface)
            assert item.number is not None
    tube_notes = _glossary_notes(layout, chapter_id, hints_by_text[tube].id)
    assert tube_notes[0].note.marker_end == _span_end(tube, "the London Underground")
    assert tube_notes[0].note.body == (
        "Лондонское метро",
        "Дамблдор шутит, что шрам у него на колене похож на схему линий метро.",
    )
    assert "Лондонское метро — Лондонское метро" not in " — ".join(tube_notes[0].note.body)
    alley_notes = _glossary_notes(layout, chapter_id, hints_by_text[alley].id)
    assert [item.note.term for item in alley_notes] == ["Diagon Alley", "showers of owls"]
    assert alley_notes[0].note.marker_end == _span_end(alley, "Diagon Alley")
    assert alley_notes[1].note.marker_end == _span_end(alley, "showers of owls")
    assert [item.note.body for item in alley_notes] == [
        ("Косой переулок", "Название похоже на diagonally. Читается как diagonally."),
        (
            "потоки сов",
            "Вопрос построен без Are there: Going to be any more showers of owls tonight, Harry?",
        ),
    ]

    target = tmp_path / "review-fixes.epub"
    EpubExportWriter().write(
        EpubWriteRequest(
            document=document,
            target_path=str(target),
            block_results=results,
            layout=layout,
        )
    )
    pages = _read_pages(target)
    raw_pages = []
    with ZipFile(target) as archive:
        for name in archive.namelist():
            if name.endswith((".xhtml", ".html")):
                raw_pages.append(archive.read(name).decode())
    exported = "\n".join(raw_pages)
    for text, _terms, surfaces, snippet in marked:
        block = hints_by_text[text]
        notes = _glossary_notes(layout, chapter_id, block.id)
        unit = _unit(pages, snippet)
        assert _ref_superscripts(unit) == [str(item.number) for item in notes]
        for item, surface in zip(notes, surfaces, strict=True):
            assert f"<sup>{item.number}</sup>" in exported
            assert surface in _source_text(_child(unit, "lg-src"))
    tube_text = _text(_child(_unit(pages, "London Underground"), "lg-notes"))
    assert tube_text.count("Лондонское метро") == 1
    assert "Лондонское метро — Лондонское метро" not in tube_text
    assert "Дамблдор шутит, что шрам у него на колене похож на схему линий метро." in tube_text
    alley_text = _text(_child(_unit(pages, "Diagon Alley"), "lg-notes"))
    assert "не подтверждает её прямо" not in alley_text
    assert "без is it" not in alley_text
    assert "Вопрос построен без Are there" in alley_text
    assert "Название похоже на diagonally." in alley_text
    assert _translation(_unit(pages, "the cannon")) == cannon_ru
    assert _translation(_unit(pages, "Budge up")) == budge_ru
    assert _translation(_unit(pages, "Kept what")) == kept_ru
    assert _translation(_rendered_heading(pages, "CHAPTER THREE")) is None

    petunia_id = "analysis-petunia"
    petunia = _feedback_block(
        "sec-petunia", "hp01_ch002_en-gb.html", 0, 1, "Aunt Petunia looked up.",
    )
    petunia_quote = "Aunt Petunia"
    petunia_plan = TreatmentPlan.of_treatments(
        petunia.section_id, ((petunia.id, BlockTreatment.TRANSLATE),),
    )
    petunia_description = "Петунья отказала. У Петуньи нет ключа. Петунье сказали нет."
    petunia_candidate = _candidate(
        petunia_id, "Aunt Petunia", "тётя Петунья", "Тётя Гарри, сестра Лили.",
    )
    petunia_analysis = ChapterAnalysis.create(
        id=petunia_id,
        section_id=petunia.section_id,
        treatment_plan=petunia_plan,
        translation_instructions="Keep the glossary names.",
        chapter_summary="Aunt Petunia is named.",
        expected_block_ids=(petunia.id,),
        glossary_candidates=(petunia_candidate,),
        content_plan=ContentPlan(
            version=3,
            profile="content-plan-v3",
            difficulties=(
                Difficulty(
                    "diff-petunia",
                    (SourceAnchor(petunia.id, 0, len(petunia_quote), petunia_quote),),
                    petunia.id,
                    "vocabulary",
                    "useful",
                    "explanation",
                    (RequiredPoint("p-petunia", petunia_description),),
                ),
            ),
        ),
    )
    owned = _owned_content_plan(
        petunia_analysis,
        TranslationUnit.of_blocks(id="unit-petunia", blocks=(petunia,), plan=petunia_plan),
        decisions=(
            GlossarySnapshotEntry(
                source_text="Aunt Petunia",
                translation="тётя Петуния",
                type=ConceptType.PERSON,
                status=TermStatus.APPROVED,
            ),
        ),
        chapter_candidates=(petunia_candidate,),
    )
    assert "Петуния отказала. У Петунии нет ключа. Петунии сказали нет." in owned
    assert "Петунья" not in owned
    assert "Петуньи" not in owned
    assert "Петунье" not in owned

    policy = ReaderTextPolicy()
    assert policy.prepare("translation", "\u2013 Где пушка?", ()) == "\u2014 Где пушка?"
    assert (
        policy.prepare("translation", "К его удивлению.\n\u2014 Хагрид усмехнулся.", ())
        == "К его удивлению.\n\u2014 Хагрид усмехнулся."
    )

    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
    )
    narrative_rules = (
        "грамматический род топонимов и животных",
        "действительно несклоняемых",
        "мужские фамилии на согласный в русском склоняются всегда",
        "не копируй в заметки читателю",
        "said Harry",
        "certainty=explicit",
        "анонимных обозначений",
    )
    pedagogical_rules = (
        "speech_style — диалект, акцент, особенности речи; только первое появление особенности персонажа.",
        "«Речь персонажа.»",
        "we was",
        "you was",
        "don' need",
        "them as",
        "b(K+1)..bN",
        "не больше N",
        "не больше 1–2 заметок на абзац",
        "reader_chapter_summary",
        "Previously in Chapter",
        "Present Simple",
        "стандартные английские кавычки",
        "«ёлочек»",
        "без перевода на русский",
    )
    calques = (
        "длинным тире «— »",
        "кавычками-ёлочками",
        "род адресата во 2-м лице",
        "местоимений 3-го лица",
        "регистр обращения «ты/вы»",
        "разговорность в русском переводе",
        "без буквальных калек",
    )
    for phrase in narrative_rules:
        assert phrase in templates["narrative_analysis"]
    for phrase in pedagogical_rules:
        assert phrase in templates["pedagogical_analysis"]
    for key in ("content_processing", "learner_quality"):
        for phrase in calques:
            assert phrase in templates[key]


def test_a2_ch1_ch7_reading_quality_fixes_end_to_end(tmp_path: Path) -> None:
    """End-to-end regression across all 17 A2 Harry Potter ch1-ch7 quality fixes (no LLM calls)."""
    import asyncio
    from types import SimpleNamespace

    from linguary.adapters.db.engine import create_database_engine
    from linguary.adapters.db.models import Base
    from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
    from linguary.adapters.epub.splitting import (
        SEGMENT_MAX_CHARS,
        SEGMENT_MIN_BLOCK_CHARS,
    )
    from linguary.adapters.llm import FakeLLMClient, FakeReply
    from linguary.adapters.validation.english_spelling import IpaDictSpellingChecker
    from linguary.application.analysis_merge import (
        content_plan_from_response,
    )
    from linguary.application.context_builder import (
        ContextBuilder,
        _matches_chapter_text,
        scaled_chapter_note_limit,
        snapshot_prompt_templates,
    )
    from linguary.application.glossary_review import build_glossary_review
    from linguary.application.layout.parts import PART_MAX_CHARS
    from linguary.application.llm_dto import (
        BlockResultResponse,
        ChapterAnalysisResponse,
        GlossaryHintResponse,
        LanguageNoteResponse,
        QuotedContentPlanResponse,
        QuotedContextFactResponse,
        QuotedSceneResponse,
        TranslationResultResponse,
    )
    from linguary.application.ports.persistence import BookRecord
    from linguary.application.reader_text import ReaderTextPolicy
    from linguary.application.retry_policy import ResultValidationError, RetryPolicy
    from linguary.application.translation_units import plan_translation_units
    from linguary.application.use_cases.analyze_chapter import (
        AnalyzeChapter,
        AnalyzeChapterCommand,
    )
    from linguary.application.validators.dialogue import dialogue_violation
    from linguary.application.validators.mixed_script import note_language_leak_violation
    from linguary.application.validators.result import TranslationResultValidator
    from linguary.composition.correction_runtime import CORRECTION_PROMPT_V2
    from linguary.domain import (
        BookStatus,
        GlossarySnapshotEntry,
        LanguageNote,
        ResultWarningKind,
        TermStatus,
        TranslationUnit,
    )
    from linguary.domain.content_plan import (
        ContentEvidence,
        Difficulty,
        HelpEvidence,
        Scene,
        artifact_key,
    )
    from linguary.domain.name_translation import NameTranslationPolicy
    from linguary.domain.pronunciation import PronunciationAssistance, PronunciationEvidence
    from linguary.web.labels import RESULT_WARNING_LABELS

    # --- P3.T1: Separate segmentation thresholds (450/480 vs PART_MAX_CHARS=350) ---
    assert SEGMENT_MAX_CHARS == 450
    assert SEGMENT_MIN_BLOCK_CHARS == 480
    assert PART_MAX_CHARS == 350
    sentence_400 = (
        "Harry stood beside the enormous oak front doors of Hogwarts castle, "
        "watching the tall torches flicker against the dark stone walls while "
        "Professor McGonagall adjusted her spectacles and waited for the nervous "
        "first-year students to fall completely silent before she led them into "
        "the Great Hall where hundreds of floating candles shone high above the "
        "four long house tables tonight."
    )
    assert 360 <= len(sentence_400) <= 440
    assert len(segment_text(sentence_400)) == 1
    long_block_text = (
        "Professor Quirrell trembled near the leaky cauldron doorway while holding "
        "a heavy stack of parchment scrolls and whispering to the nervous boy beside "
        "the firelight in the narrow hall. "
        "Across the cobbled lane, Hagrid tightened his moleskin coat and pointed "
        "towards the crooked brick wall behind the pub where an old brass lantern "
        "swung gently in the cold evening wind. "
        "Harry clutched his letter tightly, staring at the emerald ink as the "
        "bricks began to shift and open into Diagon Alley."
    )
    assert len(long_block_text) > SEGMENT_MIN_BLOCK_CHARS
    long_segments = segment_text(long_block_text)
    assert len(long_segments) >= 2

    # --- P1.T1 + P1.T2 + P1.T3: Multi-chapter EPUB layout (lg-cont numbering, typography,
    #     previously/legend, 3-stage vocab dedup, 11 grammar caps, speech_style, .lg-tr CSS) ---
    book_id = "hp-ch1-ch7-e2e"
    sec_ids = tuple(f"sec-ch{idx}" for idx in range(1, 9))
    chapters_blocks: dict[str, tuple[DocumentBlock, ...]] = {}
    results: dict[str, BlockProcessingResult] = {}
    analyses: list[ChapterAnalysis] = []

    def _blk(sec_id: str, order: int, text: str, *, segments: tuple[BlockSegment, ...] = ()) -> DocumentBlock:
        return DocumentBlock.create(
            document_id=book_id,
            section_id=sec_id,
            type=BlockType.PARAGRAPH,
            order=order,
            text=text,
            source_reference=SourceReference(f"{sec_id}.html", f"/body/p[{order + 1}]"),
            segments=segments,
        )

    # Chapter 1: split block (lg-cont local numbering), display typography block,
    # local 20-block vocab dedup, 2 speech_style notes for Hagrid, used_to + must_have grammar notes.
    ch1_long = _blk(sec_ids[0], 0, long_block_text, segments=long_segments)
    ch1_typo = _blk(sec_ids[0], 1, "P-P-Potter stuttered near the old castle gate.")
    ch1_snig_1 = _blk(sec_ids[0], 2, "Dudley sniggered loudly at the doorway.")
    ch1_snig_tr_near = _blk(sec_ids[0], 3, "Piers sniggered again behind him.")
    ch1_snig_gl_near = _blk(sec_ids[0], 4, "Malcolm sniggered softly.")
    ch1_hagrid_1 = _blk(sec_ids[0], 5, f"{_OPEN}Yeh don' know what yeh are, Harry.{_CLOSE}")
    ch1_hagrid_2 = _blk(sec_ids[0], 6, f"{_OPEN}I've got summat fer yeh here.{_CLOSE}")
    ch1_quill_1 = _blk(sec_ids[0], 7, "Harry dipped his quill into the inkpot.")
    ch1_used_1 = _blk(sec_ids[0], 8, "He used to sleep in the cupboard under the stairs.")
    ch1_must_1 = _blk(sec_ids[0], 9, "Someone must have left the letter on the mat.")
    ch1_fillers = tuple(
        _blk(sec_ids[0], 10 + i, f"The quiet corridor stretched ahead, step {i + 1}.")
        for i in range(22)
    )
    chapters_blocks[sec_ids[0]] = (
        ch1_long,
        ch1_typo,
        ch1_snig_1,
        ch1_snig_tr_near,
        ch1_snig_gl_near,
        ch1_hagrid_1,
        ch1_hagrid_2,
        ch1_quill_1,
        ch1_used_1,
        ch1_must_1,
        *ch1_fillers,
    )

    # Chapter 2: quill 2nd occurrence (GLOSSARY_ONLY -> stripped note), Hagrid repeated "yeh"
    # (suppressed) + new "codswallop" (preserved), must_have gap<2 (suppressed).
    ch2_quill_gl = _blk(sec_ids[1], 0, "He bought a silver quill at the shop.")
    ch2_hagrid_rep = _blk(sec_ids[1], 1, f"{_OPEN}Yeh never told him!{_CLOSE}")
    ch2_hagrid_new = _blk(sec_ids[1], 2, f"{_OPEN}Codswallop, in my opinion.{_CLOSE}")
    ch2_must_gap1 = _blk(sec_ids[1], 3, "Hagrid must have forgotten the umbrella.")
    ch2_fillers = tuple(
        _blk(sec_ids[1], 4 + i, f"They walked past shop number {i + 1}.")
        for i in range(22)
    )
    chapters_blocks[sec_ids[1]] = (
        ch2_quill_gl,
        ch2_hagrid_rep,
        ch2_hagrid_new,
        ch2_must_gap1,
        *ch2_fillers,
    )

    # Chapter 3: quill 3rd occurrence on TRANSLATE (suppressed), used_to 2nd (kept), must_have 2nd (kept).
    ch3_quill_tr = _blk(sec_ids[2], 0, "The quill scratched across the parchment.")
    ch3_used_2 = _blk(sec_ids[2], 1, "Uncle Vernon used to shout every morning.")
    ch3_must_2 = _blk(sec_ids[2], 2, "The owl must have flown all night.")
    ch3_fillers = tuple(
        _blk(sec_ids[2], 3 + i, f"Train compartment note {i + 1}.")
        for i in range(22)
    )
    chapters_blocks[sec_ids[2]] = (ch3_quill_tr, ch3_used_2, ch3_must_2, *ch3_fillers)

    # Chapter 4: quill 4th occurrence (suppressed: count>=3 and gap<=3).
    ch4_quill_4 = _blk(sec_ids[3], 0, "Hermione sharpened her quill carefully.")
    ch4_fillers = tuple(
        _blk(sec_ids[3], 1 + i, f"Boat lantern reflection {i + 1}.")
        for i in range(22)
    )
    chapters_blocks[sec_ids[3]] = (ch4_quill_4, *ch4_fillers)

    # Chapter 5: used_to 3rd (suppressed: cap=2), must_have 3rd (kept: cap=3).
    ch5_used_3 = _blk(sec_ids[4], 0, "Dumbledore used to teach Transfiguration.")
    ch5_must_3 = _blk(sec_ids[4], 1, "Neville must have dropped the Remembrall.")
    chapters_blocks[sec_ids[4]] = (ch5_used_3, ch5_must_3)

    # Chapter 6: filler chapter for gap > 3.
    ch6_plain = _blk(sec_ids[5], 0, "The Sorting Hat sang its quiet song.")
    chapters_blocks[sec_ids[5]] = (ch6_plain,)

    # Chapter 7: must_have 4th (suppressed: cap=3).
    ch7_must_4 = _blk(sec_ids[6], 0, "Snape must have noticed the potion fumes.")
    chapters_blocks[sec_ids[6]] = (ch7_must_4,)

    # Chapter 8: quill reintroduced after gap > 3 chapters since last shown (ch2 -> ch8, gap 6).
    ch8_quill_re = _blk(sec_ids[7], 0, "Harry picked up his quill once more.")
    chapters_blocks[sec_ids[7]] = (ch8_quill_re,)

    # Populate results
    seg_count = len(long_segments)
    seg1_text = long_block_text[long_segments[0].start : long_segments[0].end]
    seg2_text = long_block_text[long_segments[1].start : long_segments[1].end]
    seg1_word = "trembled"
    seg2_word = "lantern" if "lantern" in seg2_text else "clutched"
    assert seg1_word in seg1_text
    assert seg2_word in seg2_text
    seg_tr = tuple(f"Перевод части {i + 1}." for i in range(seg_count))
    results[ch1_long.id] = BlockProcessingResult(
        block_id=ch1_long.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text=" ".join(seg_tr),
        segment_treatments=tuple(BlockTreatment.TRANSLATE for _ in range(seg_count)),
        segment_translations=seg_tr,
        glossary_hints=(
            GlossaryHint(seg1_word, "дрожал", "tremble — дрожать"),
            GlossaryHint(seg2_word, "фонарь" if seg2_word == "lantern" else "сжимал"),
        ),
    )
    stutter_anchor = SourceAnchor(
        ch1_typo.id,
        ch1_typo.text.index("stuttered"),
        ch1_typo.text.index("stuttered") + len("stuttered"),
        "stuttered",
    )
    results[ch1_typo.id] = BlockProcessingResult(
        block_id=ch1_typo.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text=(
            "— ... П-П-Поттер заикался у за\u0301мка (10–15 шагов)... Что?.. Неужели!.. Он сказал–"
        ),
        glossary_hints=(
            GlossaryHint(
                "stuttered",
                "заикался",
                "У старого за\u0301мка ... 10–15 минут — ... П-П-Поттер замолчал–",
            ),
        ),
        pronunciation_assistances=(
            PronunciationAssistance(
                anchor=stutter_anchor,
                form="stuttered",
                ipa="ˈstʌtəd",
                variant="uk",
                hint_id="ph-stuttered",
                evidence=PronunciationEvidence(
                    "ipa-dict", "1", "stuttered", "v", "verb", "stuttered", "ˈstʌtəd", "ipa-dict"
                ),
            ),
        ),
    )
    results[ch1_snig_1.id] = BlockProcessingResult(
        block_id=ch1_snig_1.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Дадли громко хихикнул в дверях.",
        glossary_hints=(GlossaryHint("sniggered", "хихикнул", "ехидно засмеялся"),),
    )
    results[ch1_snig_tr_near.id] = BlockProcessingResult(
        block_id=ch1_snig_tr_near.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Пирз снова хихикнул позади него.",
        glossary_hints=(GlossaryHint("sniggered", "хихикнул", "ехидно засмеялся"),),
    )
    results[ch1_snig_gl_near.id] = BlockProcessingResult(
        block_id=ch1_snig_gl_near.id,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(GlossaryHint("sniggered", "хихикнул", "ехидно засмеялся"),),
    )

    def _speech_parts(
        block: DocumentBlock, tr: str, diff_id: str, pt_id: str, excerpt: str, note_text: str
    ) -> tuple[DocumentBlock, str, Difficulty, ContextFact, LanguageNote, HelpEvidence]:
        anchor = _anchor(block, excerpt)
        diff = Difficulty(
            diff_id,
            (anchor,),
            block.id,
            "speech_style",
            "useful",
            "explanation",
            (RequiredPoint(pt_id, note_text),),
        )
        note_id = f"ln-{diff_id}"
        ln = LanguageNote(note_id, block.id, excerpt, note_text)
        key = artifact_key(
            "language_note.note",
            block.id,
            {"original_excerpt": excerpt, "note": note_text, "translated_excerpt": None},
        )
        ev = HelpEvidence(diff_id, pt_id, "language_note.note", block.id, key, note_text)
        fact = ContextFact(
            f"sp-{diff_id}", "speaker", (_anchor(block, block.text),), _anchor(block, block.text), "Hagrid", "explicit"
        )
        return block, tr, diff, fact, ln, ev

    sp_h1 = _speech_parts(
        ch1_hagrid_1, "— Ты не знаешь, кто ты, Гарри.", "d-hag-1", "p-h1", "Yeh", "Диалектное yeh вместо you."
    )
    sp_h2 = _speech_parts(
        ch1_hagrid_2, "— У меня тут кое-что для тебя.", "d-hag-2", "p-h2", "summat", "Диалектное summat вместо something."
    )
    sp_h3 = _speech_parts(
        ch2_hagrid_rep, "— Вы ему так и не сказали!", "d-hag-3", "p-h3", "Yeh", "Снова yeh вместо you."
    )
    sp_h4 = _speech_parts(
        ch2_hagrid_new, "— Чушь собачья, по-моему.", "d-hag-4", "p-h4", "Codswallop", "Просторечное codswallop — ерунда."
    )
    cp_ch1 = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(sp_h1[2], sp_h2[2]),
        context_facts=(sp_h1[3], sp_h2[3]),
        scenes=(Scene("sc-1", 0, ch1_hagrid_1.id, ch1_hagrid_2.id, "Hut", "Hut", ("Hagrid", "Harry")),),
    )
    cp_ch2 = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(sp_h3[2], sp_h4[2]),
        context_facts=(sp_h3[3], sp_h4[3]),
        scenes=(Scene("sc-2", 0, ch2_hagrid_rep.id, ch2_hagrid_new.id, "Shop", "Shop", ("Hagrid", "Harry")),),
    )
    for (blk, tr, _diff, _fact, ln, ev), cp_sha in (
        (sp_h1, cp_ch1.sha256),
        (sp_h2, cp_ch1.sha256),
        (sp_h3, cp_ch2.sha256),
        (sp_h4, cp_ch2.sha256),
    ):
        results[blk.id] = BlockProcessingResult(
            block_id=blk.id,
            treatment=BlockTreatment.TRANSLATE,
            translated_text=tr,
            language_notes=(ln,),
            content_evidence=ContentEvidence(cp_sha, (ev,)),
        )

    quill_hint = GlossaryHint("quill", "перо для письма", "Птичье перо, которым пишут чернилами.")
    results[ch1_quill_1.id] = BlockProcessingResult(
        ch1_quill_1.id, BlockTreatment.TRANSLATE, translated_text="Гарри обмакнул перо в чернильницу.", glossary_hints=(quill_hint,)
    )
    results[ch2_quill_gl.id] = BlockProcessingResult(
        ch2_quill_gl.id, BlockTreatment.GLOSSARY_ONLY, glossary_hints=(quill_hint,)
    )
    results[ch3_quill_tr.id] = BlockProcessingResult(
        ch3_quill_tr.id, BlockTreatment.TRANSLATE, translated_text="Перо заскрипело по пергаменту.", glossary_hints=(quill_hint,)
    )
    results[ch4_quill_4.id] = BlockProcessingResult(
        ch4_quill_4.id, BlockTreatment.TRANSLATE, translated_text="Гермиона аккуратно очинила перо.", glossary_hints=(quill_hint,)
    )
    results[ch8_quill_re.id] = BlockProcessingResult(
        ch8_quill_re.id, BlockTreatment.TRANSLATE, translated_text="Гарри снова взял перо.", glossary_hints=(quill_hint,)
    )

    for idx, blk in enumerate((ch1_used_1, ch3_used_2, ch5_used_3), start=1):
        results[blk.id] = BlockProcessingResult(
            blk.id,
            BlockTreatment.TRANSLATE,
            translated_text=f"Раньше он делал это регулярно ({idx}).",
            language_notes=(LanguageNote(f"ln-used-{idx}", blk.id, "used to", "Конструкция used to обозначает привычку в прошлом."),),
        )
    for idx, blk in enumerate((ch1_must_1, ch2_must_gap1, ch3_must_2, ch5_must_3, ch7_must_4), start=1):
        results[blk.id] = BlockProcessingResult(
            blk.id,
            BlockTreatment.TRANSLATE,
            translated_text=f"Должно быть, это уже случилось ({idx}).",
            language_notes=(LanguageNote(f"ln-must-{idx}", blk.id, "must have", "Конструкция must have + V3 выражает уверенное предположение о прошлом."),),
        )
    for blk in (*ch1_fillers, *ch2_fillers, *ch3_fillers, *ch4_fillers, ch6_plain):
        results[blk.id] = BlockProcessingResult(blk.id, BlockTreatment.READ_AS_IS)

    sections = tuple(
        DocumentSection(
            id=sec_id,
            document_id=book_id,
            title=f"Chapter {idx}",
            order=idx - 1,
            source_resource_path=f"{sec_id}.html",
            blocks=chapters_blocks[sec_id],
        )
        for idx, sec_id in enumerate(sec_ids, start=1)
    )
    document = CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title="Harry Potter ch1-ch8 E2E"),
        source_language="en",
        target_language="ru",
        sections=sections,
    )
    for idx, sec_id in enumerate(sec_ids, start=1):
        blks = chapters_blocks[sec_id]
        plan = TreatmentPlan.of_treatments(
            sec_id, tuple((b.id, results[b.id].treatment) for b in blks)
        )
        cp: ContentPlan | None = None
        if idx == 1:
            cp = cp_ch1
        elif idx == 2:
            cp = cp_ch2
        analyses.append(
            ChapterAnalysis.create(
                id=f"an-{idx}",
                section_id=sec_id,
                treatment_plan=plan,
                translation_instructions="Keep names consistent.",
                chapter_summary=f"Summary of chapter {idx}.",
                expected_block_ids=tuple(b.id for b in blks),
                content_plan=cp,
            )
        )

    # 1-chapter layout omits "previously"
    single_doc = CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title="One Chapter"),
        source_language="en",
        target_language="ru",
        sections=(sections[0],),
    )
    single_layout = build_book_layout(
        LayoutInputs(
            document=single_doc,
            results={b.id: results[b.id] for b in sections[0].blocks},
            analyses=(analyses[0],),
            concepts=(),
            reader_level="a2",
            english_help=False,
        )
    )
    assert "previously" not in single_layout.features.sections

    # Multi-chapter layout + EPUB export
    layout = build_book_layout(
        LayoutInputs(
            document=document,
            results=results,
            analyses=tuple(analyses),
            concepts=(),
            reader_level="a2",
            english_help=False,
        )
    )
    assert "previously" in layout.features.sections

    # P1.T1: Local numbering restarts at 1 inside lg-cont
    long_units = layout.chapters[sec_ids[0]].units[ch1_long.id]
    assert len(long_units) >= 2
    assert long_units[0].continues is False
    assert long_units[1].continues is True
    assert [n.number for n in long_units[0].notes if n.number is not None] == [1]
    assert [n.number for n in long_units[1].notes if n.number is not None] == [1]

    # P1.T1: Display typography on translation and note.body without touching term/ipa
    typo_unit = layout.chapters[sec_ids[0]].units[ch1_typo.id][0]
    assert typo_unit.translation is not None
    assert "…" in typo_unit.translation
    assert "..." not in typo_unit.translation
    assert "?.." in typo_unit.translation and "!.." in typo_unit.translation
    assert "П-П-Поттер" in typo_unit.translation
    assert "10–15" in typo_unit.translation
    assert "— …" in typo_unit.translation
    assert "сказал…" in typo_unit.translation
    assert "\u0301" not in typo_unit.translation
    typo_hint_note = next(item.note for item in typo_unit.notes if item.note.origin == "glossary_hint")
    typo_body = " ".join(typo_hint_note.body)
    assert "\u0301" not in typo_body
    assert "..." not in typo_body
    assert "П-П-Поттер" in typo_body and "10–15" in typo_body and "замолчал…" in typo_body
    assert typo_hint_note.term == "to stutter"
    assert typo_hint_note.ipa is not None and "ˈstʌtəd" in typo_hint_note.ipa

    # P1.T2: 3-stage vocab dedup
    assert len(layout.chapters[sec_ids[0]].units[ch1_snig_1.id][0].notes) == 1
    assert layout.chapters[sec_ids[0]].units[ch1_snig_tr_near.id][0].notes == ()
    assert len(layout.chapters[sec_ids[0]].units[ch1_snig_gl_near.id][0].notes) == 1
    ch1_q_notes = layout.chapters[sec_ids[0]].units[ch1_quill_1.id][0].notes
    ch2_q_notes = layout.chapters[sec_ids[1]].units[ch2_quill_gl.id][0].notes
    assert len(ch1_q_notes) == 1 and "Птичье перо" in " ".join(ch1_q_notes[0].note.body)
    assert len(ch2_q_notes) == 1 and "Птичье перо" not in " ".join(ch2_q_notes[0].note.body)
    assert layout.chapters[sec_ids[2]].units[ch3_quill_tr.id][0].notes == ()
    assert layout.chapters[sec_ids[3]].units[ch4_quill_4.id][0].notes == ()
    assert len(layout.chapters[sec_ids[7]].units[ch8_quill_re.id][0].notes) == 1

    # P1.T2: 11-pattern grammar caps (used_to cap=2; modal_must_have cap=3, gap>=2)
    assert len(layout.chapters[sec_ids[0]].units[ch1_used_1.id][0].notes) == 1
    assert len(layout.chapters[sec_ids[2]].units[ch3_used_2.id][0].notes) == 1
    assert layout.chapters[sec_ids[4]].units[ch5_used_3.id][0].notes == ()
    assert len(layout.chapters[sec_ids[0]].units[ch1_must_1.id][0].notes) == 1
    assert layout.chapters[sec_ids[1]].units[ch2_must_gap1.id][0].notes == ()
    assert len(layout.chapters[sec_ids[2]].units[ch3_must_2.id][0].notes) == 1
    assert len(layout.chapters[sec_ids[4]].units[ch5_must_3.id][0].notes) == 1
    assert layout.chapters[sec_ids[6]].units[ch7_must_4.id][0].notes == ()

    # P1.T2: speech_style keeps first 2 in first scene + unique later dialect anchor, suppresses repeated "yeh"
    assert len(layout.chapters[sec_ids[0]].units[ch1_hagrid_1.id][0].notes) == 1
    assert len(layout.chapters[sec_ids[0]].units[ch1_hagrid_2.id][0].notes) == 1
    assert layout.chapters[sec_ids[1]].units[ch2_hagrid_rep.id][0].notes == ()
    assert len(layout.chapters[sec_ids[1]].units[ch2_hagrid_new.id][0].notes) == 1

    # Export EPUB and verify legend.xhtml, lg-cont superscripts, CSS geometry, README sync
    epub_path = tmp_path / "hp-ch1-ch7-e2e.epub"
    EpubExportWriter().write(
        EpubWriteRequest(
            document=document,
            target_path=str(epub_path),
            block_results=results,
            layout=layout,
        )
    )
    pages = _read_pages(epub_path)
    legend_page = _page_with(pages, "Previously in…")
    assert "По мере чтения переводов становится меньше" in _text(legend_page)
    cont_units = [
        el
        for page in pages
        for el in page.iter()
        if _local(el) == "div" and _has_class(el, "lg-cont")
    ]
    assert cont_units
    assert _ref_superscripts(cont_units[0]) == ["1"]

    epub_css = _stylesheet(epub_path)
    web_css = (Path(__file__).resolve().parents[3] / "src/linguary/web/static/linguary.css").read_text()
    for css_text in (epub_css, web_css):
        tr_match = re.search(r"\.lg-tr\s*\{([^}]*)\}", css_text)
        assert tr_match is not None
        tr_rule = tr_match.group(1)
        assert "margin: 0.55em 0 0 1.3em;" in tr_rule
        assert "font-size: 0.88em;" in tr_rule
        assert "padding-left: 0.7em;" in tr_rule
        assert "border-left: 0.12em solid;" in tr_rule
        assert "opacity" not in tr_rule
    readme_text = (Path(__file__).resolve().parents[3] / "README.md").read_text()
    assert "correction_model" in readme_text

    # --- P2.T1 + P2.T2 + P2.T3: Dialogue, NOTE leak, EN-GB spelling, quality warnings, historical=True ---
    prose_block = _blk("sec-v", 0, '"Hello. How are you?"')
    poem_block = DocumentBlock.create(
        document_id=book_id,
        section_id="sec-v",
        type=BlockType.POEM,
        order=1,
        text="Oh, you may not think I'm pretty,",
        source_reference=SourceReference("sec-v.html", "/body/p[2]"),
    )
    prose_unit = TranslationUnit.of_blocks(
        id="u-prose",
        blocks=(prose_block,),
        plan=TreatmentPlan.of_treatments("sec-v", ((prose_block.id, BlockTreatment.TRANSLATE),)),
    )
    poem_unit = TranslationUnit.of_blocks(
        id="u-poem",
        blocks=(poem_block,),
        plan=TreatmentPlan.of_treatments("sec-v", ((poem_block.id, BlockTreatment.TRANSLATE),)),
    )

    def _tr_payload(block_id: str, text: str, *, hints=(), notes=()) -> TranslationResultResponse:
        return TranslationResultResponse(
            blocks=(BlockResultResponse(block_id, BlockTreatment.TRANSLATE, text, glossary_hints=hints),),
            language_notes=notes,
        )

    assert dialogue_violation(_tr_payload(prose_block.id, "— «Привет», — спросил он."), target_language="ru", unit=prose_unit) is not None
    assert dialogue_violation(_tr_payload(prose_block.id, "«Привет, — спросил он. — Как дела?»"), target_language="ru", unit=prose_unit) is not None
    assert dialogue_violation(_tr_payload(prose_block.id, "— Привет. — Как дела?"), target_language="ru", unit=prose_unit) is not None
    assert dialogue_violation(_tr_payload(prose_block.id, "«Привет», — спросил он."), target_language="ru", unit=prose_unit) is not None
    assert dialogue_violation(_tr_payload(prose_block.id, "«Что происходит?» — подумал Гарри."), target_language="ru", unit=prose_unit) is None
    assert dialogue_violation(_tr_payload(poem_block.id, "— «Пусть не кажусь я красивой»."), target_language="ru", unit=poem_unit) is None

    assert note_language_leak_violation(
        _tr_payload(
            prose_block.id,
            "— Привет, — спросил он.",
            notes=(LanguageNoteResponse(prose_block.id, "reminded", "Фраза 'was strongly reminded of Дадли' описывает память."),),
        ),
        target_language="ru",
    ) is not None
    assert note_language_leak_violation(
        _tr_payload(
            prose_block.id,
            "— Привет, — спросил он.",
            notes=(LanguageNoteResponse(prose_block.id, "hello", "Слово really усиливает вопрос."),),
        ),
        target_language="ru",
    ) is not None
    assert note_language_leak_violation(
        _tr_payload(
            prose_block.id,
            "— Привет, — спросил он.",
            notes=(LanguageNoteResponse(prose_block.id, "hello", "Время Past Perfect и британский вариант UK в IPA."),),
        ),
        target_language="ru",
    ) is None

    checker = IpaDictSpellingChecker()
    assert checker.is_known_word("colour") is True
    assert checker.is_known_word("well-known") is True
    assert checker.is_known_word("expell") is False
    validator = TranslationResultValidator(
        target_language_min_share=0.5,
        language_check_min_length=20,
        repetition_ngram_size=4,
        repetition_min_repeats=3,
        spelling_checker=checker,
    )
    assert ResultWarningKind.QUALITY in RESULT_WARNING_LABELS
    expel_block = _blk("sec-v", 2, "Hermione feared Hogwarts might expel them for codswallop.")
    expel_unit = TranslationUnit.of_blocks(
        id="u-expel",
        blocks=(expel_block,),
        plan=TreatmentPlan.of_treatments("sec-v", ((expel_block.id, BlockTreatment.TRANSLATE),)),
    )
    with pytest.raises(ResultValidationError):
        validator.validate(
            _tr_payload(
                expel_block.id,
                "Гермиона боялась, что их исключат за ерунду.",
                hints=(GlossaryHintResponse("to expell", "исключить"),),
            ),
            unit=expel_unit,
            target_language="ru",
            note_id_factory=lambda: "ln-1",
        )
    verbatim_res = validator.validate(
        _tr_payload(
            expel_block.id,
            "Гермиона боялась, что их исключат за ерунду.",
            hints=(GlossaryHintResponse("codswallop", "ерунда"),),
        ),
        unit=expel_unit,
        target_language="ru",
        note_id_factory=lambda: "ln-1",
    )
    assert verbatim_res[0].glossary_hints[0].original == "codswallop"

    warn_block = _blk(
        "sec-v",
        3,
        "Mr H. Potter bought two dozen owls and talked about You-Know-Who before Quidditch.",
    )
    warn_unit = TranslationUnit.of_blocks(
        id="u-warn",
        blocks=(warn_block,),
        plan=TreatmentPlan.of_treatments("sec-v", ((warn_block.id, BlockTreatment.TRANSLATE),)),
    )
    mandatory = (
        GlossarySnapshotEntry("Harry Potter", "Гарри Поттер", ConceptType.PERSON, TermStatus.APPROVED),
        GlossarySnapshotEntry("Quidditch", "квиддич", ConceptType.FICTIONAL_CONCEPT, TermStatus.APPROVED),
    )
    warn_res = validator.validate(
        _tr_payload(
            warn_block.id,
            "Мистер Х. Поттер купил два десятка сов, что-бы скрыться от Сам-Знаешь-Кто со залом перед игрой.",
        ),
        unit=warn_unit,
        target_language="ru",
        note_id_factory=lambda: "ln-w",
        mandatory_glossary=mandatory,
    )
    quality_msgs = " | ".join(
        w.message for w in warn_res[0].warnings if w.kind is ResultWarningKind.QUALITY
    )
    assert "dozen" in quality_msgs
    assert "что-бы" in quality_msgs
    assert "со залом" in quality_msgs
    assert "Х. Поттер" in quality_msgs
    assert "Сам-Знаешь-Кто" in quality_msgs
    assert "Quidditch" in quality_msgs

    legacy_payload = _tr_payload(
        prose_block.id,
        "— «Привет», — спросил он.",
        notes=(LanguageNoteResponse(prose_block.id, "hello", "Слово really здесь важно."),),
    )
    with pytest.raises(ResultValidationError):
        validator.validate(legacy_payload, unit=prose_unit, target_language="ru", note_id_factory=lambda: "ln-h", historical=False)
    recovered = validator.validate(legacy_payload, unit=prose_unit, target_language="ru", note_id_factory=lambda: "ln-h", historical=True)
    assert recovered[0].translated_text == "— «Привет», — спросил он."

    # --- P3.T1: Cohesive list runs & adaptive build_glossary_review ---
    list_intro = _blk("sec-list", 0, "A " + "long introductory paragraph " * 8)
    list_items = (
        _blk("sec-list", 1, "1 telescope"),
        _blk("sec-list", 2, "1 set of brass scales"),
        _blk("sec-list", 3, "1 cauldron (pewter, standard size 2)"),
        _blk("sec-list", 4, "1 set glass or crystal phials"),
    )
    list_blocks = (list_intro, *list_items)
    list_plan = TreatmentPlan.of_treatments(
        "sec-list", tuple((b.id, BlockTreatment.TRANSLATE) for b in list_blocks)
    )
    unit_counter = itertools.count(1)
    planned_units = plan_translation_units(
        list_blocks,
        list_plan,
        id_factory=lambda: f"u-{next(unit_counter)}",
        max_source_chars=260,
    )
    list_unit_ids = {
        unit.id for unit in planned_units for item in unit.items if item.block in list_items
    }
    assert len(list_unit_ids) == 1

    cand_hagrid = _candidate("an-1", "Hagrid", "Хагрид", "Великан.")
    doc_2ch = CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title="Two Chapters"),
        source_language="en",
        target_language="ru",
        sections=sections[:2],
    )
    review_2ch = build_glossary_review(
        book_id=book_id,
        analyses=(replace(analyses[0], glossary_candidates=(cand_hagrid,)), analyses[1]),
        confidence_threshold=0.8,
        count_mentions=lambda _spellings: 1,
        min_mentions=5,
        min_chapters=3,
        document=doc_2ch,
    )
    assert review_2ch.item_for("Hagrid") is not None
    assert review_2ch.item_for("Hagrid").requires_review is True
    doc_6ch = CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title="Six Chapters"),
        source_language="en",
        target_language="ru",
        sections=sections[:6],
    )
    cand_piers = _candidate("an-1", "Piers", "Пирз", "Приятель Дадли.")
    review_6ch = build_glossary_review(
        book_id=book_id,
        analyses=(replace(analyses[0], glossary_candidates=(cand_piers,)), *analyses[1:6]),
        confidence_threshold=0.8,
        count_mentions=lambda _spellings: 1,
        min_mentions=5,
        min_chapters=3,
        document=doc_6ch,
    )
    assert review_6ch.item_for("Piers") is not None
    assert review_6ch.item_for("Piers").requires_review is False

    # --- P4.T1 + P4.T2: ReaderTextPolicy, CORRECTION_PROMPT_V2, ContextBuilder progression/Ermolovich/initials/budget ---
    assert ReaderTextPolicy().prepare("translation", "— Привет.\n— Пока.", ()) == "— Привет.\n— Пока."
    for expected_check in ("two dozen", "ложных друзей", "согласование", "род животных", "ты/вы", "стыки соседних блоков"):
        assert expected_check in CORRECTION_PROMPT_V2
    a2_templates = snapshot_prompt_templates(
        source_language="en", target_language="ru", reader_level="a2", profile="content-plan-v3"
    )
    assert "букву «ё»" in a2_templates["content_processing"]
    cb = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
        name_translation_policy=NameTranslationPolicy(version=1, strategy=NameTranslationStrategy.SOUND),
    )
    ped_early = cb.for_chapter_pedagogical_analysis(
        document,
        sections[0],
        narrative_summary="Summary.",
        cumulative_words_before=0,
        prior_explained_topics=("used to",),
    )
    ped_late = cb.for_chapter_pedagogical_analysis(
        document,
        sections[0],
        narrative_summary="Summary.",
        cumulative_words_before=50_000,
    )
    assert "70–82%" in ped_early.request.input_text
    assert "62–72%" in ped_late.request.input_text
    assert "used to" in ped_early.request.input_text
    sha_early = cb.pedagogical_input_sha256(
        document, sections[0], narrative_output_sha256="nsha", cumulative_words_before=0
    )
    sha_late = cb.pedagogical_input_sha256(
        document, sections[0], narrative_output_sha256="nsha", cumulative_words_before=50_000
    )
    assert sha_early != sha_late
    narr_ctx = cb.for_chapter_narrative_analysis(document, sections[0])
    assert "«McGonagall» → «Макгонагалл»" in narr_ctx.request.instructions
    assert "маглы" in narr_ctx.request.instructions
    assert (
        _matches_chapter_text(
            "Harry Potter",
            "Mr H. Potter lived in the cupboard.",
            {"mr", "h", "potter", "lived", "cupboard"},
        )
        is True
    )
    assert scaled_chapter_note_limit(35, 8) == 8
    assert scaled_chapter_note_limit(120, 8) == 14
    assert scaled_chapter_note_limit(300, 6) == 18
    assert scaled_chapter_note_limit(300, 3) == 3

    # --- P5.T1: A2 thresholds 15/45, non-dialogue demotion, forward-quote pruning, pronoun speaker resolution ---
    b_dlg14 = _blk("sec-p5", 0, f"{_OPEN}Wait here for a minute,{_CLOSE} said Harry. {_OPEN}I will bring the key soon.{_CLOSE}")
    b_narr40 = _blk(
        "sec-p5",
        1,
        "Harry opened the small wooden door and stepped into the bright garden. "
        "Two brown owls sat quietly on the stone wall near the green gate. "
        "The warm morning sun shone over the quiet street while the neighbours slept.",
    )
    b_narr40_hard = _blk(
        "sec-p5",
        2,
        "Harry opened the rusty iron gate and stepped into the dark alley. "
        "Two strange owls sat silently on the crooked wall near the tavern. "
        "The cold fog drifted over the empty street while the lanterns flickered.",
    )
    b_item = _blk("sec-p5", 3, "1 telescope")
    b_future = _blk("sec-p5", 4, "Later they saw the giant brass cauldron boiling over green flames.")
    p5_blocks = (b_dlg14, b_narr40, b_narr40_hard, b_item, b_future)
    p5_norm = normalize_a2_dialogue_response(
        PedagogicalAnalysisResponse(
            treatment_plan=(
                TreatmentPlanItemResponse(b_dlg14.id, BlockTreatment.GLOSSARY_ONLY),
                TreatmentPlanItemResponse(b_narr40.id, BlockTreatment.GLOSSARY_ONLY),
                TreatmentPlanItemResponse(b_narr40_hard.id, BlockTreatment.GLOSSARY_ONLY),
                TreatmentPlanItemResponse(b_item.id, BlockTreatment.TRANSLATE),
                TreatmentPlanItemResponse(b_future.id, BlockTreatment.TRANSLATE),
            ),
            difficulties=(
                QuotedDifficultyResponse(
                    "d-hard-1", (QuotedSourceAnchorResponse(b_narr40_hard.id, "rusty"),),
                    b_narr40_hard.id, "vocabulary", "useful", "lexical_hint", (RequiredPoint("p1", "rusty — ржавый"),),
                ),
                QuotedDifficultyResponse(
                    "d-hard-2", (QuotedSourceAnchorResponse(b_narr40_hard.id, "crooked"),),
                    b_narr40_hard.id, "vocabulary", "useful", "lexical_hint", (RequiredPoint("p2", "crooked — кривой"),),
                ),
                QuotedDifficultyResponse(
                    "d-hard-3", (QuotedSourceAnchorResponse(b_narr40_hard.id, "drifted"),),
                    b_narr40_hard.id, "vocabulary", "useful", "lexical_hint", (RequiredPoint("p3", "drifted — плыл"),),
                ),
                QuotedDifficultyResponse(
                    "d-hard-4", (QuotedSourceAnchorResponse(b_narr40_hard.id, "flickered"),),
                    b_narr40_hard.id, "vocabulary", "useful", "lexical_hint", (RequiredPoint("p4", "flickered — мерцали"),),
                ),
                QuotedDifficultyResponse(
                    "d-fwd",
                    (
                        QuotedSourceAnchorResponse(b_narr40.id, "sat quietly"),
                        QuotedSourceAnchorResponse(b_future.id, "giant brass cauldron"),
                    ),
                    b_narr40.id,
                    "grammar",
                    "useful",
                    "explanation",
                    (
                        RequiredPoint("p-now", "Здесь sat quietly описывает состояние сов."),
                        RequiredPoint("p-later", 'Дальше появится "giant brass cauldron" в сцене.'),
                    ),
                ),
            ),
        ),
        p5_blocks,
        {b.id: b.id for b in p5_blocks},
        reader_level="a2",
        english_help=False,
    )
    p5_modes = {item.source_block_id: item.treatment for item in p5_norm.treatment_plan}
    assert p5_modes[b_dlg14.id] is BlockTreatment.GLOSSARY_ONLY
    assert p5_modes[b_narr40.id] is BlockTreatment.GLOSSARY_ONLY
    assert p5_modes[b_narr40_hard.id] is BlockTreatment.TRANSLATE
    assert p5_modes[b_item.id] is BlockTreatment.READ_AS_IS
    pruned_fwd = next(d for d in p5_norm.difficulties if d.id == "d-fwd")
    assert [a.block_id for a in pruned_fwd.anchors] == [b_narr40.id]
    assert [p.id for p in pruned_fwd.required_points] == ["p-now"]

    # Unambiguous pronoun remark resolution after scene resolution
    b_she = _blk("sec-sp", 0, f"{_OPEN}No,{_CLOSE} she said.")
    b_he = _blk("sec-sp", 1, f"{_OPEN}Why?{_CLOSE} he asked.")
    sp_payload = ChapterAnalysisResponse(
        translation_instructions="Keep names.",
        chapter_summary="Harry and Petunia talk.",
        treatment_plan=(
            TreatmentPlanItemResponse("b1", BlockTreatment.GLOSSARY_ONLY),
            TreatmentPlanItemResponse("b2", BlockTreatment.GLOSSARY_ONLY),
        ),
        content_plan=QuotedContentPlanResponse(
            version=3,
            profile="content-plan-v3",
            context_facts=(
                QuotedContextFactResponse(
                    "f-petunia",
                    "speaker",
                    (QuotedSourceAnchorResponse("b1", b_she.text),),
                    QuotedSourceAnchorResponse("b1", b_she.text),
                    "Petunia",
                    "inferred",
                ),
                QuotedContextFactResponse(
                    "f-harry",
                    "speaker",
                    (QuotedSourceAnchorResponse("b2", b_he.text),),
                    QuotedSourceAnchorResponse("b2", b_he.text),
                    "Harry",
                    "inferred",
                ),
            ),
            scenes=(
                QuotedSceneResponse("b1", "b1", "Kitchen", "Kitchen", ("Harry", "Petunia")),
                QuotedSceneResponse("b2", "b2", "Hall", "Hall", ("Harry", "Stranger")),
            ),
        ),
    )
    resolved_plan = content_plan_from_response(
        sp_payload,
        {"b1": b_she.id, "b2": b_he.id},
        texts={b_she.id: b_she.text, b_he.id: b_he.text},
        visible_texts_by_alias={"b1": b_she.text, "b2": b_he.text},
        blocks=(b_she, b_he),
        mandatory_facts={"Harry": {"gender": "male"}, "Petunia": {"gender": "female"}},
    )
    assert resolved_plan is not None
    facts_by_value = {f.value: f for f in resolved_plan.context_facts}
    assert facts_by_value["Petunia"].certainty == "explicit"
    assert facts_by_value["Harry"].certainty == "inferred"

    # --- P5.T2: AnalyzeChapter cumulative words + inter-fragment prior_explained_topics ---
    from datetime import UTC, datetime

    st_sec1 = DocumentSection(
        id="st-1",
        document_id="st-book",
        title="Chapter 1",
        order=0,
        source_resource_path="ch1.xhtml",
        blocks=(
            DocumentBlock.create(
                document_id="st-book",
                section_id="st-1",
                type=BlockType.PARAGRAPH,
                order=0,
                text=" ".join(["magic"] * 11_000),
                source_reference=SourceReference("ch1.xhtml", "/body/p[1]"),
            ),
        ),
    )
    st_sec2 = DocumentSection(
        id="st-2",
        document_id="st-book",
        title="Chapter 2",
        order=1,
        source_resource_path="ch2.xhtml",
        blocks=(
            DocumentBlock.create(
                document_id="st-book",
                section_id="st-2",
                type=BlockType.PARAGRAPH,
                order=0,
                text="Harry squinted at the sign because he used to live in the cupboard.",
                source_reference=SourceReference("ch2.xhtml", "/body/p[1]"),
            ),
            DocumentBlock.create(
                document_id="st-book",
                section_id="st-2",
                type=BlockType.PARAGRAPH,
                order=1,
                text="Aunt Petunia looked up and muttered quietly.",
                source_reference=SourceReference("ch2.xhtml", "/body/p[2]"),
            ),
        ),
    )
    st_doc = CanonicalDocument(
        id="st-book",
        metadata=BookMetadata(title="Staged E2E"),
        source_language="en",
        target_language="ru",
        sections=(st_sec1, st_sec2),
    )
    engine = create_database_engine(tmp_path / "staged-e2e.db")
    Base.metadata.create_all(engine)
    uow = SqlAlchemyUnitOfWork.for_engine(engine)
    try:
        with uow.transaction() as tx:
            tx.books.add(
                BookRecord(
                    id="st-book",
                    metadata=BookMetadata(title="Staged E2E", authors=("J.K. Rowling",)),
                    source_language="en",
                    target_language="ru",
                    status=BookStatus.ANALYZING,
                    source_epub_path=str(tmp_path / "src.epub"),
                    created_at=datetime(2026, 11, 21, tzinfo=UTC),
                )
            )
            tx.documents.save(st_doc)

        narr_reply_1 = FakeReply.of_payload(
            {
                "translation_instructions": "Keep names consistent.",
                "chapter_summary": "Harry reads the sign.",
                "glossary_candidates": [],
                "context_facts": [],
                "scenes": [
                    {
                        "start_block_id": "b1",
                        "end_block_id": "b1",
                        "summary": "Табличка в коридоре.",
                        "location": "Hallway",
                        "participants": ["Harry"],
                        "visual_facts": [],
                        "visual_assumptions": [],
                    }
                ],
            }
        )
        narr_reply_2 = FakeReply.of_payload(
            {
                "translation_instructions": "Keep names consistent.",
                "chapter_summary": "Harry reads the sign while Aunt Petunia mutters.",
                "glossary_candidates": [],
                "context_facts": [],
                "scenes": [
                    {
                        "start_block_id": "b1",
                        "end_block_id": "b1",
                        "summary": "Разговор в коридоре.",
                        "location": "Hallway",
                        "participants": ["Harry", "Aunt Petunia"],
                        "visual_facts": [],
                        "visual_assumptions": [],
                    }
                ],
            }
        )
        ped_frag1 = FakeReply.of_payload(
            {
                "treatment_plan": [{"source_block_id": "b1", "treatment": "TRANSLATE"}],
                "before_you_read": [],
                "difficulties": [
                    {
                        "id": "d-frag1",
                        "anchors": [{"block_id": "b1", "quote": "used to"}],
                        "owner_block_id": "b1",
                        "kind": "grammar",
                        "priority": "useful",
                        "help_kind": "explanation",
                        "required_points": [{"id": "p1", "description": "Конструкция used to для привычек в прошлом."}],
                    }
                ],
                "before_you_read_evidence": [],
                "reader_chapter_summary": "Harry looks at a sign.",
            }
        )
        ped_frag2 = FakeReply.of_payload(
            {
                "treatment_plan": [{"source_block_id": "b1", "treatment": "TRANSLATE"}],
                "before_you_read": [],
                "difficulties": [],
                "before_you_read_evidence": [],
                "reader_chapter_summary": "Aunt Petunia mutters quietly.",
            }
        )
        fake_llm = FakeLLMClient.replying((narr_reply_1, narr_reply_2, ped_frag1, ped_frag2))
        st_cb = ContextBuilder(
            content_plan_profile="content-plan-v3",
            reader_level="a2",
            prompt_templates=a2_templates,
        )
        staged_counter = itertools.count(1)
        analyze = AnalyzeChapter(
            unit_of_work=uow,
            llm_client=fake_llm,
            language_detector=SimpleNamespace(probability_for_language=lambda _t, _l: 1.0),
            retry_policy=RetryPolicy(max_attempts=2),
            glossary_confidence_threshold=0.8,
            max_fragment_blocks=1,
            context_builder=st_cb,
            staged_analysis=True,
            id_factory=lambda: f"st-{next(staged_counter)}",
        )
        asyncio.run(analyze(AnalyzeChapterCommand(book_id="st-book", section_id="st-2")))
        assert len(fake_llm.requests) == 4
        ped_req_1, ped_req_2 = fake_llm.requests[2], fake_llm.requests[3]
        assert "72–80%" in ped_req_1.input_text
        assert "Уже объяснено в предыдущих фрагментах этой главы" in ped_req_2.input_text
        assert "used to" in ped_req_2.input_text
    finally:
        engine.dispose()

