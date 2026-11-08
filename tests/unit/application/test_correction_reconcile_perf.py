"""Performance and query-count invariants for correction reconciliation (P1.T1)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from analysis_fakes import (
    BOOK_ID,
    make_analysis_document,
    make_analysis_section,
    make_book_record,
)
from linguary.application.block_processing import TranslationCall
from linguary.application.correction import (
    CorrectionInputs,
    _observe_correction_inputs,
    correction_coverage_complete,
    invalidate_correction,
    reconcile_corrections,
)
from linguary.application.correction_dto import (
    CORRECTION_VERSIONS_V2,
    CorrectionSnapshotV2,
)
from linguary.application.retry_policy import RetryPolicy
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    BookStatus,
    ChapterAnalysis,
    GlossarySnapshot,
    ProcessingRun,
    TreatmentPlan,
)
from linguary.domain.correction import CORRECTION_POLICY_VERSION_V2
from processing_fakes import make_validator


def _make_v2_snapshot() -> CorrectionSnapshotV2:
    return CorrectionSnapshotV2(
        model="gpt-test",
        route={
            "provider": "openai_compatible",
            "base_url": "https://example.test/v1",
            "proxy": "",
            "route_id": "route-1",
        },
        parameters={
            "temperature": 0.1,
            "max_output_tokens": 1000,
            "timeout_seconds": 30,
            "max_attempts": 1,
            "max_unit_source_chars": 1000,
            "input_price_per_mtok": 0,
            "output_price_per_mtok": 0,
        },
        prompt="Propose corrections.",
        review_prompt="Review corrections.",
        versions=CORRECTION_VERSIONS_V2,
    )


def test_invalidate_correction_without_reconcile_skips_book_wide_document_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linguary.application.correction as correction_module

    deleted_checks: list[tuple[str, tuple[str, ...]]] = []
    staled_candidates: list[tuple[str, tuple[str, ...]]] = []
    document_loads = 0

    book = replace(
        make_book_record(BookStatus.READY_TO_EXPORT),
        export_validation_status="valid",
    )
    saved_books: list[object] = []

    def fail_reconcile(tx: object, book_id: str) -> tuple[object, ...]:
        raise AssertionError("reconcile_corrections must not run when reconcile=False")

    monkeypatch.setattr(correction_module, "reconcile_corrections", fail_reconcile)

    class FakeDocuments:
        def get(self, book_id: str) -> object:
            nonlocal document_loads
            document_loads += 1
            raise AssertionError("documents.get must not be called when reconcile=False")

    class FakeBooks:
        def get(self, book_id: str) -> object:
            return book

        def save(self, updated: object) -> None:
            saved_books.append(updated)

    class FakeWork:
        def delete_correction_checks(
            self, run_id: str, *, block_ids: tuple[str, ...]
        ) -> None:
            deleted_checks.append((run_id, tuple(block_ids)))

        def get_binding(self, kind: str, entity_id: str) -> dict[str, object] | None:
            if kind == "run" and entity_id == "run-1":
                return {
                    "book_id": BOOK_ID,
                    "conditions_id": "cond-1",
                    "payload_json": {
                        "correction": {"policy_version": CORRECTION_POLICY_VERSION_V2}
                    },
                }
            return None

        def stale_correction_candidates(
            self, run_id: str, *, block_ids: tuple[str, ...] = ()
        ) -> None:
            staled_candidates.append((run_id, tuple(block_ids)))

        def stale_reviews(self, book_id: str) -> None:
            return None

    tx = SimpleNamespace(
        documents=FakeDocuments(),
        books=FakeBooks(),
        work=FakeWork(),
    )

    # Pass a generator to verify block_ids is materialized once and reused for both calls.
    invalidate_correction(
        tx,
        BOOK_ID,
        "run-1",
        (block_id for block_id in ("block-1", "block-2")),
        reconcile=False,
    )

    assert document_loads == 0
    assert deleted_checks == [("run-1", ("block-1", "block-2"))]
    assert staled_candidates == [("run-1", ("block-1", "block-2"))]
    assert len(saved_books) == 1
    assert saved_books[0].status is BookStatus.PROCESSING
    assert saved_books[0].export_validation_status is None

    # Also verify TranslationCall.save(..., reconcile=False) forwards reconcile=False.
    deleted_checks.clear()
    staled_candidates.clear()
    saved_books.clear()

    run = SimpleNamespace(
        id="run-1",
        book_id=BOOK_ID,
        glossary_snapshot=GlossarySnapshot(),
    )
    stored_results: list[BlockProcessingResult] = []

    class FakeUow:
        runs = SimpleNamespace(active_for_book=lambda _book_id: run)
        results = SimpleNamespace(
            get=lambda _run_id, _block_id: None,
            replace_results=lambda _run_id, items: stored_results.extend(items),
        )
        documents = FakeDocuments()
        books = FakeBooks()
        work = FakeWork()

        @contextmanager
        def transaction(self):
            yield self

    call = TranslationCall(
        unit_of_work=FakeUow(),  # type: ignore[arg-type]
        llm_client=SimpleNamespace(),  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_attempts=1),
        validator=make_validator(),
        sleep=lambda _s: None,  # type: ignore[arg-type]
    )
    result_item = BlockProcessingResult(
        block_id="block-1",
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Перевод.",
    )
    assert (
        call.save(
            book_id=BOOK_ID,
            run_id="run-1",
            results=(result_item,),
            reconcile=False,
        )
        is True
    )
    assert stored_results == [result_item]
    assert deleted_checks == [("run-1", ("block-1",))]
    assert staled_candidates == [("run-1", ("block-1",))]
    assert document_loads == 0


def test_reconcile_corrections_loads_document_once_across_all_chapters() -> None:
    sections = tuple(
        make_analysis_section(
            section_id=f"section-{idx}",
            title=f"Chapter {idx}",
            order=idx - 1,
            texts=(f"Chapter {idx} paragraph one.", f"Chapter {idx} paragraph two."),
        )
        for idx in (1, 2, 3)
    )
    document = make_analysis_document(sections)
    snapshot = _make_v2_snapshot()
    run = SimpleNamespace(
        id="run-1",
        book_id=BOOK_ID,
        glossary_snapshot=GlossarySnapshot(),
    )
    conditions = SimpleNamespace(id="cond-1", payload={"schema_version": 1})

    analyses_by_section: dict[str, ChapterAnalysis] = {}
    results_by_section: dict[str, tuple[BlockProcessingResult, ...]] = {}
    bindings_by_block: dict[str, dict[str, object]] = {}
    generations_by_block: dict[str, int] = {}

    for section in sections:
        plan = TreatmentPlan.of_treatments(
            section.id,
            tuple((b.id, BlockTreatment.TRANSLATE) for b in section.ordered_blocks),
        )
        analyses_by_section[section.id] = ChapterAnalysis.create(
            id=f"analysis-{section.id}",
            section_id=section.id,
            treatment_plan=plan,
            translation_instructions="",
            chapter_summary="",
            expected_block_ids=tuple(b.id for b in section.ordered_blocks),
        )
        results_by_section[section.id] = tuple(
            BlockProcessingResult(
                block_id=b.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text=f"Перевод {b.id}",
            )
            for b in section.ordered_blocks
        )
        for b in section.ordered_blocks:
            bindings_by_block[b.id] = {
                "book_id": BOOK_ID,
                "conditions_id": conditions.id,
                "attempt_id": f"att-{b.id}",
                "artifact_revision_id": f"art-{b.id}",
            }
            generations_by_block[b.id] = 1

    counts = {
        "documents_get": 0,
        "analyses_list_for_book": 0,
        "result_bindings": 0,
        "correction_input_generations": 0,
        "results_list_for_section": 0,
    }
    book = make_book_record(BookStatus.PROCESSING)
    added_tasks: list[object] = []

    class FakeDocuments:
        def get(self, book_id: str):
            counts["documents_get"] += 1
            return document

    class FakeAnalyses:
        def list_for_book(self, book_id: str):
            counts["analyses_list_for_book"] += 1
            return tuple(analyses_by_section.values())

        def get_for_section(self, book_id: str, section_id: str):
            raise AssertionError(
                "analyses.get_for_section must not be called when list_for_book is available"
            )

    class FakeResults:
        def list_for_section(self, run_id: str, section_id: str):
            counts["results_list_for_section"] += 1
            return results_by_section[section_id]

        def list_for_run(self, run_id: str):
            raise AssertionError(
                "results.list_for_run must not be called when list_for_section is available"
            )

    class FakeWork:
        def get_binding(self, kind: str, entity_id: str):
            if kind == "run" and entity_id == run.id:
                return {
                    "book_id": BOOK_ID,
                    "conditions_id": conditions.id,
                    "payload_json": {"correction": snapshot.to_payload()},
                }
            if kind == "result":
                raise AssertionError(
                    "work.get_binding('result', ...) must not be called when result_bindings is available"
                )
            return None

        def result_bindings(self, run_id: str):
            counts["result_bindings"] += 1
            return dict(bindings_by_block)

        def latest_conditions(self, book_id: str):
            return conditions

        def correction_input_generations(self, run_id: str):
            counts["correction_input_generations"] += 1
            return dict(generations_by_block)

        def observe_correction_input(
            self, run_id: str, block_id: str, fingerprint: str
        ) -> int:
            return generations_by_block[block_id]

        def list_correction_candidates(
            self, run_id: str, *, section_id: str | None = None, states=None
        ):
            return ()

        def correction_checks(self, run_id: str, section_id: str):
            return ()

        def bind(self, kind: str, entity_id: str, **kwargs) -> None:
            return None

    class FakeTasks:
        def list_for_book(self, book_id: str):
            return ()

        def find_by_identity(self, identity: object):
            return None

        def add(self, task: object) -> None:
            added_tasks.append(task)

    tx = SimpleNamespace(
        runs=SimpleNamespace(active_for_book=lambda _book_id: run),
        documents=FakeDocuments(),
        analyses=FakeAnalyses(),
        results=FakeResults(),
        work=FakeWork(),
        tasks=FakeTasks(),
        books=SimpleNamespace(get=lambda _book_id: book, save=lambda _b: None),
    )

    pending = reconcile_corrections(tx, BOOK_ID)
    assert len(pending) == 3
    assert counts["documents_get"] == 1
    assert counts["analyses_list_for_book"] == 1
    assert counts["result_bindings"] == 1
    assert counts["correction_input_generations"] == 1
    assert counts["results_list_for_section"] == 3

    counts = {key: 0 for key in counts}
    assert correction_coverage_complete(tx, BOOK_ID) is False
    assert counts["documents_get"] == 1
    assert counts["analyses_list_for_book"] == 1
    assert counts["result_bindings"] == 1
    assert counts["correction_input_generations"] == 1
    assert counts["results_list_for_section"] == 1


def test_observe_correction_inputs_updates_generations_without_rebuilding_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linguary.application.correction as correction_module

    section = make_analysis_section(texts=("First block.", "Second block."))
    document = make_analysis_document((section,))
    b1, b2 = section.ordered_blocks
    snapshot = _make_v2_snapshot()

    inputs = CorrectionInputs(
        document=document,
        section=section,
        analysis=None,
        run=SimpleNamespace(id="run-1", glossary_snapshot=GlossarySnapshot()),  # type: ignore[arg-type]
        conditions=SimpleNamespace(id="cond-1", payload={"schema_version": 1}),  # type: ignore[arg-type]
        snapshot=snapshot,
        blocks=section.ordered_blocks,
        results={
            b1.id: BlockProcessingResult(
                block_id=b1.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Первый блок.",
            ),
            b2.id: BlockProcessingResult(
                block_id=b2.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Второй блок.",
            ),
        },
        bindings={b1.id: None, b2.id: None},
        input_generations={b1.id: 1, b2.id: 2},
    )

    def forbidden_correction_inputs(tx: object, book_id: str, section_id: str):
        raise AssertionError(
            "correction_inputs must not be called by _observe_correction_inputs"
        )

    monkeypatch.setattr(
        correction_module, "correction_inputs", forbidden_correction_inputs
    )

    staled_blocks: list[tuple[str, tuple[str, ...]]] = []

    class FakeWork:
        def observe_correction_input(
            self, run_id: str, block_id: str, fingerprint: str
        ) -> int:
            return 1 if block_id == b1.id else 3

        def stale_correction_candidates(
            self, run_id: str, *, block_ids: tuple[str, ...] = ()
        ) -> None:
            staled_blocks.append((run_id, tuple(block_ids)))

    tx = SimpleNamespace(work=FakeWork())
    updated = _observe_correction_inputs(tx, inputs)

    assert updated.input_generations == {b1.id: 1, b2.id: 3}
    assert staled_blocks == [("run-1", (b2.id,))]
    assert updated.checksums == inputs.checksums
    assert updated.context_sha256 == inputs.context_sha256
