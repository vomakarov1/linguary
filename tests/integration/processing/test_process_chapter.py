"""Chapter processing: block groups, result saving, and language notes (P3.T6, ALG-04).

Tests run against real SQLite and `FakeLLMClient`: there are no network calls in the suite,
and everything related to transactions and result replacement is verified against
the actual storage delivered to users.
"""

from __future__ import annotations

import pytest
from processing_scenario import (
    BOOK_ID,
    SOURCE_TEXT,
    TRANSLATED_TEXT,
    block_reply,
    make_process_chapter,
    note_reply,
    store_book_with_rejected_variant,
    store_processing_book,
    translated_reply,
    translation_reply,
)

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm import FakeLLMClient, FakeScenarioError
from linguary.application.use_cases.process_chapter import (
    ProcessChapterCommand,
    ProcessChapterError,
)
from linguary.domain import BlockProcessingResult, BlockTreatment, ResultWarningKind

#: Paragraph with intentional phrase repetition: repetition check triggers a warning.
REPEATED_TRANSLATION = (
    "Он шёл и шёл вперёд, он шёл и шёл вперёд, он шёл и шёл вперёд, "
    "он шёл и шёл вперёд, и не видел конца."
)


def command(section_id: str = "section-1") -> ProcessChapterCommand:
    """Book chapter processing command."""
    return ProcessChapterCommand(book_id=BOOK_ID, section_id=section_id)


def stored_results(
    unit_of_work: SqlAlchemyUnitOfWork, run_id: str
) -> tuple[BlockProcessingResult, ...]:
    """Stored run results in document block order."""
    with unit_of_work.transaction() as uow:
        return uow.results.list_for_run(run_id)


async def test_chapter_results_are_saved_for_every_block(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Successful processing saves results for every block in the chapter (PRD § 14)."""
    scenario = store_processing_book(unit_of_work)
    client = FakeLLMClient.replying([translated_reply()])
    process, _ = make_process_chapter(unit_of_work, client)

    result = await process(command())

    assert (result.units, result.processed_units, result.saved_blocks) == (1, 1, 2)
    assert not result.discarded
    assert client.call_count == 1
    results = stored_results(unit_of_work, scenario.run_id)
    assert [item.block_id for item in results] == list(scenario.block_ids())
    assert all(item.translated_text == TRANSLATED_TEXT for item in results)
    assert all(item.treatment is BlockTreatment.TRANSLATE for item in results)


async def test_language_notes_are_saved_with_chapter_numbered_ids(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Language notes reside in their own identifier namespace `ln-chNN-NNN` (PRD § 19.3)."""
    scenario = store_processing_book(unit_of_work)
    client = FakeLLMClient.replying(
        [translated_reply(notes=(note_reply("b1"), note_reply("b2")))]
    )
    process, _ = make_process_chapter(unit_of_work, client)

    await process(command())

    results = stored_results(unit_of_work, scenario.run_id)
    assert [note.id for item in results for note in item.language_notes] == [
        "ln-ch01-001",
        "ln-ch01-002",
    ]
    assert results[0].language_notes[0].block_id == scenario.block_ids()[0]


async def test_glossary_only_block_gets_hints_without_translation(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Planned treatment determines block result composition (PRD § 10.1, § 15)."""
    scenario = store_processing_book(
        unit_of_work,
        treatments={
            "section-1": (BlockTreatment.GLOSSARY_ONLY, BlockTreatment.READ_AS_IS)
        },
    )
    client = FakeLLMClient.replying(
        [
            translation_reply(
                [
                    block_reply(
                        "b1",
                        treatment=BlockTreatment.GLOSSARY_ONLY,
                        hints=(("entailed", "заповедный"),),
                    ),
                    block_reply("b2", treatment=BlockTreatment.READ_AS_IS),
                ]
            )
        ]
    )
    process, _ = make_process_chapter(unit_of_work, client)

    await process(command())

    hinted, plain = stored_results(unit_of_work, scenario.run_id)
    assert hinted.translated_text is None
    assert [hint.original for hint in hinted.glossary_hints] == ["entailed"]
    assert plain.translated_text is None and plain.glossary_hints == ()


async def test_hints_and_notes_repeating_before_you_read_are_not_saved(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Expression from 'Before You Read' is not repeated as a hint or a note (QA-03)."""
    scenario = store_processing_book(
        unit_of_work,
        treatments={
            "section-1": (BlockTreatment.GLOSSARY_ONLY, BlockTreatment.TRANSLATE)
        },
        before_you_read=(("heaved and groaned", "вздымался и стонал"),),
    )
    client = FakeLLMClient.replying(
        [
            translation_reply(
                [
                    block_reply(
                        "b1",
                        treatment=BlockTreatment.GLOSSARY_ONLY,
                        hints=(
                            ("Heaved and  groaned,", "вздымалось и стонало"),
                            ("entailed", "заповедный"),
                        ),
                    ),
                    block_reply("b2"),
                ],
                notes=(
                    note_reply("b1", original_excerpt="heaved and groaned") | {"note": "вздымался и стонал"},
                    note_reply("b2", original_excerpt="the house heaved and groaned") | {"note": "вздымался и стонал"},
                    note_reply("b2"),
                ),
            )
        ]
    )
    process, _ = make_process_chapter(unit_of_work, client)

    await process(command())

    hinted, translated = stored_results(unit_of_work, scenario.run_id)
    assert [hint.original for hint in hinted.glossary_hints] == ["entailed"]
    assert hinted.language_notes == ()
    assert [note.original_excerpt for note in translated.language_notes] == [
        "a truth universally acknowledged"
    ]


async def test_new_fact_on_before_you_read_excerpt_is_saved(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    scenario = store_processing_book(
        unit_of_work,
        treatments={"section-1": (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE)},
        before_you_read=(("heaved and groaned", "вздымался и стонал"),),
    )
    client = FakeLLMClient.replying([
        translation_reply(
            [block_reply("b1"), block_reply("b2")],
            notes=(note_reply("b1", original_excerpt="heaved and groaned"),),
        )
    ])
    process, _ = make_process_chapter(unit_of_work, client)
    await process(command())
    first, _ = stored_results(unit_of_work, scenario.run_id)
    assert len(first.language_notes) == 1
    assert first.language_notes[0].note == "Ироническая формула зачина романа."
    assert first.language_notes[0].original_excerpt == "heaved and groaned"


async def test_glossary_only_block_keeps_its_only_hint_even_if_explained(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Mandatory hint of a GLOSSARY_ONLY block is not filtered out to an empty list."""
    scenario = store_processing_book(
        unit_of_work,
        treatments={
            "section-1": (BlockTreatment.GLOSSARY_ONLY, BlockTreatment.READ_AS_IS)
        },
        before_you_read=(("entailed", "заповедный"),),
    )
    client = FakeLLMClient.replying(
        [
            translation_reply(
                [
                    block_reply(
                        "b1",
                        treatment=BlockTreatment.GLOSSARY_ONLY,
                        hints=(("entailed", "заповедный"),),
                    ),
                    block_reply("b2", treatment=BlockTreatment.READ_AS_IS),
                ]
            )
        ]
    )
    process, _ = make_process_chapter(unit_of_work, client)

    await process(command())

    hinted, _ = stored_results(unit_of_work, scenario.run_id)
    assert [hint.original for hint in hinted.glossary_hints] == ["entailed"]


async def test_chapter_without_textual_blocks_does_not_call_the_model(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Illustration chapter is processed without calling the model (PRD § 10.1)."""
    scenario = store_processing_book(unit_of_work, chapters=((None,),))
    client = FakeLLMClient()
    process, _ = make_process_chapter(unit_of_work, client)

    result = await process(command())

    assert (result.units, result.processed_units, result.attempts) == (0, 0, 0)
    assert client.call_count == 0
    assert stored_results(unit_of_work, scenario.run_id) == ()


async def test_long_chapter_is_processed_by_several_units(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Chapter not fitting into a single request is processed in groups (PRD § 10)."""
    scenario = store_processing_book(
        unit_of_work, chapters=((SOURCE_TEXT, SOURCE_TEXT, SOURCE_TEXT, SOURCE_TEXT),)
    )
    client = FakeLLMClient.always(translated_reply())
    process, _ = make_process_chapter(
        unit_of_work, client, max_unit_source_chars=len(SOURCE_TEXT) * 2
    )

    result = await process(command())

    assert (result.units, result.processed_units, result.saved_blocks) == (2, 2, 4)
    assert client.call_count == 2
    assert len(stored_results(unit_of_work, scenario.run_id)) == 4


async def test_saved_units_are_not_requested_again(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Task retry does not pay for already saved groups (PRD § 18.1)."""
    scenario = store_processing_book(unit_of_work)
    first = FakeLLMClient.replying([translated_reply()])
    process, _ = make_process_chapter(unit_of_work, first)
    await process(command())

    second = FakeLLMClient()
    repeated, _ = make_process_chapter(unit_of_work, second)
    result = await repeated(command())

    assert (result.units, result.processed_units) == (1, 0)
    assert second.call_count == 0
    assert len(stored_results(unit_of_work, scenario.run_id)) == 2


async def test_partially_saved_chapter_continues_without_note_id_collision(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Retry of an incomplete chapter continues note numbering rather than restarting it."""
    scenario = store_processing_book(
        unit_of_work, chapters=((SOURCE_TEXT, SOURCE_TEXT, SOURCE_TEXT, SOURCE_TEXT),)
    )
    failing = FakeLLMClient.replying(
        [translated_reply(notes=(note_reply("b1"), note_reply("b2")))]
    )
    process, _ = make_process_chapter(
        unit_of_work, failing, max_unit_source_chars=len(SOURCE_TEXT) * 2
    )
    with pytest.raises(FakeScenarioError):
        await process(command())

    continued = FakeLLMClient.always(translated_reply(notes=(note_reply("b1"),)))
    resumed, _ = make_process_chapter(
        unit_of_work, continued, max_unit_source_chars=len(SOURCE_TEXT) * 2
    )
    result = await resumed(command())

    assert (result.units, result.processed_units) == (2, 1)
    note_ids = [
        note.id
        for item in stored_results(unit_of_work, scenario.run_id)
        for note in item.language_notes
    ]
    assert note_ids == ["ln-ch01-001", "ln-ch01-002", "ln-ch01-003"]


async def test_repetition_warning_is_stored_and_does_not_block_saving(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Repetitions in translation produce a warning and do not block saving (PRD § 15)."""
    scenario = store_processing_book(unit_of_work, chapters=((SOURCE_TEXT,),))
    client = FakeLLMClient.replying(
        [translation_reply([block_reply("b1", translated_text=REPEATED_TRANSLATION)])]
    )
    process, _ = make_process_chapter(unit_of_work, client)

    result = await process(command())

    assert result.saved_blocks == 1
    stored = stored_results(unit_of_work, scenario.run_id)[0]
    assert stored.translated_text == REPEATED_TRANSLATION
    assert [warning.kind for warning in stored.warnings] == [
        ResultWarningKind.REPETITION
    ]


async def test_unanalysed_chapter_is_not_processed(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Chapter without saved analysis cannot be processed (PRD § 14)."""
    store_processing_book(
        unit_of_work,
        chapters=((SOURCE_TEXT,), (SOURCE_TEXT,)),
        analysed_sections=("section-1",),
    )
    client = FakeLLMClient()
    process, _ = make_process_chapter(unit_of_work, client)

    with pytest.raises(ProcessChapterError, match="не проанализирована"):
        await process(command("section-2"))

    assert client.call_count == 0


async def test_rejected_variant_from_other_chapter_does_not_reach_the_request(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Variant rejected by user is recognized across candidates of the entire book (D-4).

    Chapter 2 analysis did not return candidate Duskmere: variant "Даскмир" was proposed
    only by chapter 1 analysis. Note `the marsh gate` and chapter instructions containing it
    still do not reach the chapter 2 processing request.
    """
    store_book_with_rejected_variant(unit_of_work)
    client = FakeLLMClient.replying([translated_reply(("b1",))])
    process, _ = make_process_chapter(unit_of_work, client)

    await process(command("section-2"))

    text = client.last_request.input_text
    assert "Даскмир" not in text
    assert "(Множественное число, склоняется.)" in text
    assert "1. Сохранять иронию рассказчика." in text


async def test_process_chapter_executes_independent_roots_concurrently_within_limit(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    import asyncio
    from linguary.application.ports.llm import LLMClient

    scenario = store_processing_book(
        unit_of_work,
        chapters=((SOURCE_TEXT, SOURCE_TEXT, SOURCE_TEXT, SOURCE_TEXT),),
    )
    max_chars = len(SOURCE_TEXT) + 5

    current_concurrency = 0
    max_concurrency = 0
    lock = asyncio.Lock()

    class ConcurrencyTrackingClient(FakeLLMClient):
        async def generate(self, request):
            nonlocal current_concurrency, max_concurrency
            async with lock:
                current_concurrency += 1
                if current_concurrency > max_concurrency:
                    max_concurrency = current_concurrency
            try:
                await asyncio.sleep(0.04)
                return await super().generate(request)
            finally:
                async with lock:
                    current_concurrency -= 1

    client = ConcurrencyTrackingClient(default=translated_reply(("b1",)))
    process, _ = make_process_chapter(
        unit_of_work, client, max_unit_source_chars=max_chars
    )
    process._max_concurrent_units = 2

    result = await process(command())

    assert result.units == 4
    assert result.processed_units == 4
    assert result.saved_blocks == 4
    assert not result.discarded
    assert max_concurrency == 2


async def test_parallel_roots_isolate_contextvars_and_preserve_recovery_invariants(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    import asyncio
    from linguary.application.ports.llm import LLMClient

    scenario = store_processing_book(
        unit_of_work,
        chapters=((SOURCE_TEXT, SOURCE_TEXT, SOURCE_TEXT),),
    )
    max_chars = len(SOURCE_TEXT) + 5

    observed_contexts = []
    context_lock = asyncio.Lock()

    class ContextObservingClient(FakeLLMClient):
        async def generate(self, request):
            await asyncio.sleep(0.02)
            async with context_lock:
                observed_contexts.append(request.call_context)
            return await super().generate(request)

    client = ContextObservingClient(
        default=translated_reply(
            ("b1",),
            notes=[note_reply("b1", note="Заметка к b1")],
        )
    )
    process, _ = make_process_chapter(
        unit_of_work, client, max_unit_source_chars=max_chars
    )
    process._max_concurrent_units = 3

    result = await process(command())

    assert result.units == 3
    assert result.processed_units == 3
    assert result.saved_blocks == 3
    assert not result.discarded

    attempt_ids = [c.attempt_id for c in observed_contexts]
    assert len(attempt_ids) == 3
    assert len(set(attempt_ids)) == 3
    scope_ids = [c.scope_id for c in observed_contexts]
    assert len(set(scope_ids)) == 3

    with unit_of_work.transaction() as tx:
        roots = tx.work.list_roots(scenario.document.id, run_id=scenario.run_id)
        assert len(roots) == 3
        for root in roots:
            assert tx.work.root_coverage_complete(root.id)

    for root in roots:
        assert not process._recovery.plan_unresolved(root.id)

    stored = stored_results(unit_of_work, scenario.run_id)
    assert len(stored) == 3
    all_notes = [note for res in stored for note in res.language_notes]
    assert len(all_notes) == 3
    note_ids = [note.id for note in all_notes]
    assert len(set(note_ids)) == 3
    assert all(nid.startswith("ln-ch01-") for nid in note_ids)


async def test_process_unit_skips_llm_for_byr_covered_or_foreign_anchor_read_as_is_and_attaches_evidence(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    from dataclasses import replace
    from linguary.application.context_builder import ContextBuilder
    from linguary.application.use_cases.export_book import _validate_content_plans
    from linguary.domain import BeforeYouReadEntry
    from linguary.domain.content_plan import (
        ContentPlan,
        Difficulty,
        HelpEvidence,
        RequiredPoint,
        SourceAnchor,
        artifact_key,
    )

    ch1_texts = (
        "Break a leg on the big stage tonight.",
        "Good morning to everyone in the hall.",
        "Quiet footsteps crossed the wooden floor.",
        "Sunlight warmed the narrow window ledge.",
        "Nobody spoke while the curtain rose.",
        f"{SOURCE_TEXT} They carried on quietly.",
    )
    ch2_texts = ("Another chapter paragraph that stays out of section-1 source_texts.",)
    scenario = store_processing_book(
        unit_of_work,
        chapters=(ch1_texts, ch2_texts),
        treatments={
            "section-1": (
                BlockTreatment.READ_AS_IS,
                BlockTreatment.READ_AS_IS,
                BlockTreatment.READ_AS_IS,
                BlockTreatment.READ_AS_IS,
                BlockTreatment.READ_AS_IS,
                BlockTreatment.TRANSLATE,
            ),
            "section-2": (BlockTreatment.READ_AS_IS,),
        },
    )
    section_1 = scenario.section(0)
    b1, b2, b3, b4, b5, b6 = section_1.ordered_blocks

    byr_entry = BeforeYouReadEntry(
        expression="Break a leg",
        explanation="Театральное пожелание ни пуха ни пера перед выходом на сцену.",
    )
    byr_key = artifact_key(
        "before_you_read.explanation",
        b1.id,
        {"expression": byr_entry.expression, "explanation": byr_entry.explanation},
    )
    byr_ev_1 = HelpEvidence(
        difficulty_id="d-byr",
        point_id="p1",
        artifact_kind="before_you_read.explanation",
        block_id=b1.id,
        artifact_key=byr_key,
        body_excerpt="Театральное пожелание",
    )
    byr_ev_2 = HelpEvidence(
        difficulty_id="d-byr",
        point_id="p2",
        artifact_kind="before_you_read.explanation",
        block_id=b1.id,
        artifact_key=byr_key,
        body_excerpt="перед выходом на сцену",
    )
    content_plan = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(
            Difficulty(
                id="d-byr",
                anchors=(SourceAnchor(block_id=b1.id, start=0, end=11, quote="Break a leg"),),
                owner_block_id=b1.id,
                kind="idiom",
                priority="critical",
                help_kind="explanation",
                required_points=(
                    RequiredPoint(id="p1", description="idiomatic wish"),
                    RequiredPoint(id="p2", description="stage context"),
                ),
            ),
            Difficulty(
                id="d-foreign",
                anchors=(
                    SourceAnchor(block_id=b2.id, start=0, end=12, quote="Good morning"),
                    SourceAnchor(
                        block_id=b6.id,
                        start=len(SOURCE_TEXT) + 6,
                        end=len(SOURCE_TEXT) + 16,
                        quote="carried on",
                    ),
                ),
                owner_block_id=b6.id,
                kind="semantic_relation",
                priority="critical",
                help_kind="explanation",
                required_points=(RequiredPoint(id="p1", description="phrasal continuity"),),
            ),
        ),
        before_you_read_evidence=(byr_ev_1, byr_ev_2),
    )

    with unit_of_work.transaction() as tx:
        existing_analysis = tx.analyses.get_for_section(BOOK_ID, "section-1")
        assert existing_analysis is not None
        updated_analysis = replace(
            existing_analysis,
            before_you_read=(byr_entry,),
            content_plan=content_plan,
        )
        tx.analyses.save(BOOK_ID, updated_analysis)

    # Only b6 (unit 6) should invoke the LLM; b1..b5 (including BYR-covered b1 and foreign-anchor b2) must skip LLM.
    client = FakeLLMClient.replying(
        [
            translation_reply(
                [
                    block_reply(
                        "b1",
                        treatment=BlockTreatment.TRANSLATE,
                        translated_text=f"{TRANSLATED_TEXT} Они спокойно продолжали.",
                    )
                ],
                notes=[
                    note_reply(
                        "b1",
                        original_excerpt="carried on",
                        translated_excerpt="продолжали",
                        note="Фразовый глагол означает продолжение прежнего действия.",
                    )
                ],
            )
        ]
    )
    process, _ = make_process_chapter(
        unit_of_work,
        client,
        max_unit_source_chars=45,
    )
    process._context_builder = ContextBuilder(content_plan_profile="content-plan-v3")
    with unit_of_work.transaction() as tx:
        conditions = tx.work.latest_conditions(BOOK_ID)
        assert conditions is not None

    loaded_work, roots = process.prepare(command("section-1"))
    assert len(roots) == 6
    assert set(loaded_work.source_texts) == {b.id for b in section_1.blocks}

    result = await process(command("section-1"))

    assert result.units == 6
    assert result.processed_units == 6
    assert result.saved_blocks == 6
    assert not result.discarded
    assert client.call_count == 1

    stored = {item.block_id: item for item in stored_results(unit_of_work, scenario.run_id)}
    assert set(stored) == {b.id for b in section_1.ordered_blocks}
    assert stored[b1.id].treatment is BlockTreatment.READ_AS_IS
    assert stored[b1.id].content_evidence.plan_sha256 == content_plan.sha256
    assert stored[b1.id].content_evidence.evidence == content_plan.before_you_read_evidence
    assert stored[b2.id].treatment is BlockTreatment.READ_AS_IS
    assert stored[b2.id].content_evidence.plan_sha256 == content_plan.sha256
    assert stored[b2.id].content_evidence.evidence == ()
    assert len(stored[b6.id].language_notes) == 1

    _validate_content_plans(
        scenario.document,
        (updated_analysis,),
        tuple(stored[b.id] for b in section_1.ordered_blocks),
        english_help=False,
        validators_version=3,
        reader_level="b1",
    )

    # When pronunciation help is enabled, a pure/BYR-covered READ_AS_IS unit still calls the LLM.
    section_2 = scenario.section(1)
    pron_client = FakeLLMClient.replying(
        [translation_reply([block_reply("b1", treatment=BlockTreatment.READ_AS_IS)])]
    )
    pron_process, _ = make_process_chapter(unit_of_work, pron_client)
    pron_process._call._pronunciation_enabled = True
    pron_result = await pron_process(command(section_2.id))
    assert pron_result.saved_blocks == 1
    assert pron_client.call_count == 1

