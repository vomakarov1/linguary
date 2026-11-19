"""A2 Harry Potter regressions, checked on real fragments without a model call.

Fragments are the Pottermore insert, glossary descriptions, and dialogue lines
from ``harry-potter-ch1-ch3-a2-deepseek-main``. Russian text of the six
replies is the staged rendering, not a new model call.
"""

from __future__ import annotations

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
    "hagrid-line",
    "good-luck",
    "bundle-and-scar",
    "no-problems",
    "couldnt-you",
    "demanded",
    "snapped",
    "yvonne",
    "car",
)
_EXCEPTIONS = ("shoo", "up", "boa", "a-what", "no-why", "nearly")
_SHOWN_SPEAKERS = {
    "nasty": "миссис Дёрсли",
    "oh-yes": "мистер Дёрсли",
    "giant": "Хагрид",
    "demanded": "тётя Петуния",
    "car": "дядя Вернон",
}
_UNTRANSLATED_SPEAKERS = {"shoo": "мистер Дёрсли"}
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
    ),
    _Line(
        "snapped",
        "ch2",
        _SNAPPED,
        BlockTreatment.GLOSSARY_ONLY,
        "\u2014 Что ты сказал? \u2014 огрызнулась тётя через дверь.",
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
    """Ten light main blocks become TRANSLATE; six staged exceptions do not.

    ``It winked.`` keeps the glossary hint and is not an unaided READ_AS_IS line.
    """
    for key in _REGRESSED:
        assert _mode(harry_regression, key) is BlockTreatment.TRANSLATE
        unit = _unit(harry_regression.pages, _BY_KEY[key].text)
        assert _translation(unit) == _BY_KEY[key].russian
    assert {
        difficulty.id: difficulty.help_kind
        for difficulty in harry_regression.normalized.difficulties
    } == {
        "d-bundle": "explanation",
        "d-demanded": "explanation",
        "d-snapped": "explanation",
    }
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
        "не одобрял воображения",
        "пострел",
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
_GROANED_RU = "Гарри охнул."
_CHUCKLED = "Uncle Vernon chuckled."
_CHUCKLED_RU = "Дядя Вернон усмехнулся."
_WINKED_RU = "Она подмигнула."
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
        winked.id: BlockTreatment.TRANSLATE,
        fancy.id: BlockTreatment.TRANSLATE,
        groaned.id: BlockTreatment.TRANSLATE,
        chuckled.id: BlockTreatment.TRANSLATE,
        up.id: BlockTreatment.READ_AS_IS,
        a_what.id: BlockTreatment.READ_AS_IS,
        nearly.id: BlockTreatment.READ_AS_IS,
        shoo.id: BlockTreatment.READ_AS_IS,
        no_why.id: BlockTreatment.READ_AS_IS,
        hagrid.id: BlockTreatment.TRANSLATE,
        notes_block.id: BlockTreatment.TRANSLATE,
        verbs.id: BlockTreatment.GLOSSARY_ONLY,
    }
    assert normalized.difficulties == ()

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
        winked.id: _WINKED_RU,
        fancy.id: _FANCY_RU,
        groaned.id: _GROANED_RU,
        chuckled.id: _CHUCKLED_RU,
        hagrid.id: _BY_KEY["hagrid-line"].russian,
        notes_block.id: "Полный перевод абзаца.",
    }
    results = {
        block.id: BlockProcessingResult(block.id, BlockTreatment.READ_AS_IS)
        for block in (numbered, up, a_what, nearly, shoo, no_why)
    }
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
    assert _translation(_unit(pages, _WINKED)) == _WINKED_RU
    assert _translation(_unit(pages, _FANCY)) == _FANCY_RU
    assert _translation(_unit(pages, _GROANED)) == _GROANED_RU
    assert _translation(_unit(pages, _CHUCKLED)) == _CHUCKLED_RU
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
    assert "«McGonagall» → «Макгонагал»" in sound.request.instructions


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
        == "К его удивлению. \u2014 Хагрид усмехнулся."
    )

    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
    )
    narrative_rules = (
        "Privet Drive — ж. р., улица",
        "tabby cat — ж. р., кошка",
        "boa constrictor — м. р., обращение на «ты»",
        "несклоняемость фамилий",
        "said Harry",
        "certainty=explicit",
        "said the stranger",
        "said the giant",
        "Hagrid",
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
        "reader_chapter_summary",
        "Previously in Chapter",
        "Present Simple",
        "стандартные английские кавычки",
        "«ёлочек»",
    )
    calques = (
        "длинным тире «— »",
        "You told me",
        "Ты сказала",
        "coat — пальто",
        "«оно»",
        "marmalade",
        "апельсиновый джем",
        "a great Muggle",
        "отпетый/здоровенный магл",
        "rather fond of",
        "весьма люблю",
        "any of the lights",
        "нигде не включая свет",
        "large chocolate ice-creams",
        "большие порции шоколадного мороженого",
        "go boil yer heads",
        "катитесь оба к лешему",
        "проблевался",
        "her lot",
        "компания, братия",
        "Going to be any more showers",
        "Are there",
    )
    for phrase in narrative_rules:
        assert phrase in templates["narrative_analysis"]
    for phrase in pedagogical_rules:
        assert phrase in templates["pedagogical_analysis"]
    for key in ("content_processing", "learner_quality"):
        for phrase in calques:
            assert phrase in templates[key]
