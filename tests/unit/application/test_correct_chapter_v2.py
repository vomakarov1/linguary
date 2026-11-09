"""Unit tests for CorrectChapterV2 review group persistence and chunking."""

from contextlib import nullcontext
from dataclasses import asdict, replace
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from analysis_fakes import make_analysis_document, make_analysis_section
from processing_fakes import make_validator

from linguary.application.correction import CorrectionInputs
from linguary.application.correction_dto import (
    CORRECTION_VERSIONS_V2,
    ChapterCorrectionReviewResponse,
    CorrectionSnapshotV2,
    CorrectionVerdict,
)
from linguary.application.ports.llm import LLMResponse, LLMResponseKind
from linguary.application.use_cases.correct_chapter import CorrectChapterCommand, CorrectChapterV2
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    ChapterAnalysis,
    GlossarySnapshot,
    TaskStatus,
    TaskType,
    TreatmentPlan,
)
from linguary.domain.content_plan import canonical_sha
from linguary.domain.correction import CorrectionCandidate, correction_task_identity


@pytest.mark.asyncio
async def test_review_group_persists_checks_for_unchanged_and_accepted_when_peer_awaits_human():
    section = make_analysis_section(
        texts=("First unchanged.", "Second auto applied.", "Third needs human."),
    )
    doc = make_analysis_document((section,))
    b1, b2, b3 = section.blocks
    plan = TreatmentPlan.of_treatments(
        section.id,
        (
            (b1.id, BlockTreatment.TRANSLATE),
            (b2.id, BlockTreatment.TRANSLATE),
            (b3.id, BlockTreatment.TRANSLATE),
        ),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(b1.id, b2.id, b3.id),
    )
    snapshot = CorrectionSnapshotV2(
        model="gpt-test",
        route={"provider": "openai_compatible", "base_url": "https://example.test/v1", "proxy": "", "route_id": "r1"},
        parameters={
            "temperature": 0.0,
            "max_output_tokens": 1000,
            "timeout_seconds": 30,
            "max_attempts": 1,
            "max_unit_source_chars": 8000,
            "input_price_per_mtok": 0,
            "output_price_per_mtok": 0,
        },
        prompt="Propose.",
        review_prompt="Review.",
        versions=CORRECTION_VERSIONS_V2,
    )
    res1 = BlockProcessingResult(block_id=b1.id, treatment=BlockTreatment.TRANSLATE, translated_text="Первый.")
    res2 = BlockProcessingResult(block_id=b2.id, treatment=BlockTreatment.TRANSLATE, translated_text="Второй старый.")
    res3 = BlockProcessingResult(block_id=b3.id, treatment=BlockTreatment.TRANSLATE, translated_text="Третий старый.")
    results = {b1.id: res1, b2.id: res2, b3.id: res3}
    cond_id = str(uuid4())
    run_id = str(uuid4())
    task_id = str(uuid4())
    group_id = str(uuid4())
    bindings = {
        b.id: {
            "book_id": doc.id,
            "conditions_id": cond_id,
            "attempt_id": str(uuid4()),
            "artifact_revision_id": str(uuid4()),
        }
        for idx, b in enumerate(section.blocks, start=1)
    }
    generations = {b1.id: 1, b2.id: 1, b3.id: 1}
    run = SimpleNamespace(id=run_id, glossary_snapshot=GlossarySnapshot())
    conditions = SimpleNamespace(id=cond_id, payload={"schema_version": 1, "versions": {"validators": 3}})

    def build_inputs():
        return CorrectionInputs(
            document=doc,
            section=section,
            analysis=analysis,
            run=run,
            conditions=conditions,
            snapshot=snapshot,
            blocks=section.blocks,
            results=dict(results),
            bindings=dict(bindings),
            input_generations=dict(generations),
        )

    initial_inputs = build_inputs()

    def make_candidate(cand_id: str, block, old_res, new_text: str):
        proposed_res = replace(old_res, translated_text=new_text)
        payload = json.loads(
            json.dumps(
                {
                    "schema_version": 1,
                    "book_id": doc.id,
                    "run_id": run.id,
                    "conditions_id": conditions.id,
                    "section_id": section.id,
                    "block_id": block.id,
                    "input_sha256": initial_inputs.checksums[block.id],
                    "input_generation": 1,
                    "group_id": group_id,
                    "proposal_attempt_id": group_id,
                    "context_sha256": initial_inputs.context_sha256,
                    "source_binding": bindings[block.id],
                    "original": {"text": block.text},
                    "current_result": asdict(old_res),
                    "result": asdict(proposed_res),
                    "proposal": {
                        "block_id": block.id,
                        "status": "corrected",
                        "translated_text": new_text,
                        "glossary_hints": [],
                        "language_notes": [],
                        "issues": [
                            {
                                "field_path": "translated_text",
                                "source_quote": block.text.split()[0],
                                "current_evidence": {
                                    "kind": "quote",
                                    "field_path": "translated_text",
                                    "quote": old_res.translated_text.split()[0],
                                },
                                "problem": "Точность",
                                "minimal_fix_reason": "Минимальная правка",
                            }
                        ],
                    },
                    "evidence_valid": True,
                }
            )
        )
        return CorrectionCandidate(
            candidate_id=cand_id,
            book_id=doc.id,
            run_id=run.id,
            conditions_id=conditions.id,
            section_id=section.id,
            block_id=block.id,
            input_sha256=initial_inputs.checksums[block.id],
            input_generation=1,
            proposal_sha256=canonical_sha(payload),
            payload=payload,
            group_id=group_id,
            proposal_attempt_id=group_id,
        )

    cand2 = make_candidate("cand-2", b2, res2, "Второй новый.")
    cand3 = make_candidate("cand-3", b3, res3, "Третий спорный.")
    candidates_store = {cand2.candidate_id: cand2, cand3.candidate_id: cand3}

    identity = correction_task_identity(
        book_id=doc.id,
        run_id=run.id,
        section_id=section.id,
        input_generation="gen-1",
        policy_version=snapshot.policy_version,
    )
    task = SimpleNamespace(
        id=task_id,
        book_id=doc.id,
        task_type=TaskType.CHAPTER_CORRECTION,
        scope_id=section.id,
        status=TaskStatus.RUNNING,
        attempts=1,
        operation_id=identity.operation_id,
        identity=identity,
    )
    work_bindings = {
        ("task", task.id): {
            "book_id": doc.id,
            "conditions_id": conditions.id,
            "payload_json": {
                "run_id": run.id,
                "section_id": section.id,
                "input_generation": "gen-1",
                "manifest": initial_inputs.manifest,
            },
        },
        ("correction_group", group_id): {
            "book_id": doc.id,
            "conditions_id": conditions.id,
            "payload_json": {
                "schema_version": 1,
                "run_id": run.id,
                "section_id": section.id,
                "manifest": initial_inputs.manifest,
                "block_ids": [b1.id, b2.id, b3.id],
                "unchanged_block_ids": [b1.id],
                "candidate_ids": [cand2.candidate_id, cand3.candidate_id],
            },
        },
    }
    attempts_store: dict[str, SimpleNamespace] = {}
    saved_checks: list = []
    verdicts_recorded: dict[str, str] = {}

    class FakeWork:
        def get_binding(self, kind, entity_id):
            return work_bindings.get((kind, entity_id))

        def bind(self, kind, entity_id, *, book_id, conditions_id, payload, **kwargs):
            work_bindings[(kind, entity_id)] = {
                "book_id": book_id,
                "conditions_id": conditions_id,
                "payload_json": payload,
            }

        def get_correction_candidate(self, cid):
            # Intentionally return candidate as stored without mutating .review in-place
            # unless explicitly updated, testing unconditional check persistence.
            return candidates_store.get(cid)

        def list_correction_candidates(self, run_id, *, section_id=None, states=None):
            return tuple(
                c for c in candidates_store.values()
                if states is None or c.state in states
            )

        def correction_checks(self, run_id, section_id):
            return tuple(saved_checks)

        def get_attempt(self, attempt_id):
            return attempts_store.get(attempt_id)

        def transition_attempt(self, attempt_id, new_state, *, expected_state):
            attempts_store[attempt_id].state = new_state

        def archive_artifacts(self, items):
            return tuple(f"art-{i}" for i in range(len(items)))

        def record_correction_verdict(self, candidate_id, *, state, **kwargs):
            verdicts_recorded[candidate_id] = state

        def replace_result_binding(self, key, *, book_id, conditions_id, attempt_id, artifact_revision_id, **kwargs):
            block_id = key.split(":", 1)[1]
            bindings[block_id] = {
                "book_id": book_id,
                "conditions_id": conditions_id,
                "attempt_id": attempt_id,
                "artifact_revision_id": artifact_revision_id,
            }

        def advance_correction_input(self, run_id, block_id, **kwargs):
            pass

        def correction_checks(self, run_id, section_id):
            return tuple(saved_checks)

        def replace_correction_checks(self, run_id, section_id, checks):
            saved_checks.extend(checks)

        def stale_reviews(self, book_id):
            pass

    class FakeResults:
        def replace_results(self, run_id, updated):
            for r in updated:
                results[r.block_id] = r

    fake_tx = SimpleNamespace(
        tasks=SimpleNamespace(get=lambda tid: task),
        work=FakeWork(),
        results=FakeResults(),
    )

    import linguary.application.use_cases.correct_chapter as module
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "correction_inputs", lambda tx, bid, sid: build_inputs())
    monkeypatch.setattr(module, " finalize_response", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(module, "finalize_response", lambda *a, **kw: None)

    class FakeClient:
        async def generate(self, request):
            ctx = request.call_context
            attempts_store[ctx.attempt_id] = SimpleNamespace(
                id=ctx.attempt_id,
                state="validated",
                task_id=task.id,
                run_id=run.id,
                conditions_id=conditions.id,
                context={
                    **ctx.to_dict(),
                    "task_attempt": task.attempts,
                    "operation_id": task.operation_id,
                },
            )
            payload = ChapterCorrectionReviewResponse(
                candidates=(
                    CorrectionVerdict(
                        candidate_id=cand2.candidate_id,
                        proposal_sha256=cand2.proposal_sha256,
                        verdict="confirmed",
                        error_supported=True,
                        evidence_valid=True,
                        minimal=True,
                        no_semantic_loss=True,
                        reason="Точно.",
                    ),
                    CorrectionVerdict(
                        candidate_id=cand3.candidate_id,
                        proposal_sha256=cand3.proposal_sha256,
                        verdict="needs_human",
                        error_supported=False,
                        evidence_valid=True,
                        minimal=True,
                        no_semantic_loss=True,
                        reason="Спорно.",
                    ),
                )
            )
            return LLMResponse(
                payload=payload,
                raw_content="{}",
                provider="test",
                model="gpt-test",
                call_context=ctx,
            )

    try:
        handler = CorrectChapterV2(
            unit_of_work=SimpleNamespace(transaction=lambda: nullcontext(fake_tx)),
            llm_client=FakeClient(),
            snapshot=snapshot,
            validator=make_validator(),
        )
        count = await handler._review_group(
            CorrectChapterCommand(doc.id, section.id, task.id),
            (cand2.candidate_id, cand3.candidate_id),
        )
        assert count == 2
        assert verdicts_recorded == {
            cand2.candidate_id: "auto_applied",
            cand3.candidate_id: "awaiting_human",
        }
        check_by_block = {c.block_id: c for c in saved_checks}
        assert set(check_by_block) == {b1.id, b2.id}
        assert check_by_block[b1.id].basis == "unchanged"
        assert check_by_block[b2.id].basis == "confirmed"
        assert check_by_block[b2.id].candidate_id == cand2.candidate_id
    finally:
        monkeypatch.undo()


def test_review_request_filters_chapter_context_for_candidate_window():
    from linguary.domain.content_plan import (
        ContentPlan,
        ContextFact,
        Difficulty,
        RequiredPoint,
        SourceAnchor,
    )

    section = make_analysis_section(
        texts=("Block one.", "Block two.", "Block three.", "Block four.", "Block five.", "Block six."),
    )
    doc = make_analysis_document((section,))
    b1, b2, b3, b4, b5, b6 = section.blocks

    plan = TreatmentPlan.of_treatments(
        section.id,
        tuple((b.id, BlockTreatment.TRANSLATE) for b in section.blocks),
    )
    content_plan = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(
            Difficulty(
                id="d1",
                anchors=(SourceAnchor(block_id=b1.id, start=0, end=5, quote="Block"),),
                owner_block_id=b1.id,
                kind="idiom",
                priority="critical",
                help_kind="explanation",
                required_points=(RequiredPoint(id="p1", description="meaning 1"),),
            ),
            Difficulty(
                id="d3",
                anchors=(SourceAnchor(block_id=b3.id, start=0, end=5, quote="Block"),),
                owner_block_id=b3.id,
                kind="idiom",
                priority="critical",
                help_kind="explanation",
                required_points=(RequiredPoint(id="p3", description="meaning 3"),),
            ),
            Difficulty(
                id="d6",
                anchors=(SourceAnchor(block_id=b6.id, start=0, end=5, quote="Block"),),
                owner_block_id=b6.id,
                kind="idiom",
                priority="critical",
                help_kind="explanation",
                required_points=(RequiredPoint(id="p6", description="meaning 6"),),
            ),
        ),
        context_facts=(
            ContextFact(
                id="cf1",
                kind="speaker",
                anchors=(SourceAnchor(block_id=b1.id, start=0, end=5, quote="Block"),),
                subject_anchor=SourceAnchor(block_id=b1.id, start=6, end=9, quote="one"),
                value="Speaker 1",
                certainty="explicit",
            ),
            ContextFact(
                id="cf3",
                kind="speaker",
                anchors=(SourceAnchor(block_id=b3.id, start=0, end=5, quote="Block"),),
                subject_anchor=SourceAnchor(block_id=b3.id, start=6, end=11, quote="three"),
                value="Speaker 3",
                certainty="explicit",
            ),
            ContextFact(
                id="cf6",
                kind="speaker",
                anchors=(SourceAnchor(block_id=b6.id, start=0, end=5, quote="Block"),),
                subject_anchor=SourceAnchor(block_id=b6.id, start=6, end=9, quote="six"),
                value="Speaker 6",
                certainty="explicit",
            ),
        ),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="Keep tone.",
        chapter_summary="Summary.",
        expected_block_ids=tuple(b.id for b in section.blocks),
        content_plan=content_plan,
    )
    results = {
        b.id: BlockProcessingResult(
            block_id=b.id,
            treatment=BlockTreatment.TRANSLATE,
            translated_text=f"Перевод {idx}.",
        )
        for idx, b in enumerate(section.blocks, start=1)
    }
    inputs = CorrectionInputs(
        document=doc,
        section=section,
        analysis=analysis,
        run=SimpleNamespace(id="run", glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(payload={"schema_version": 1}),
        snapshot=SimpleNamespace(to_payload=lambda: {"model": "correction"}),
        blocks=section.blocks,
        results=results,
        bindings={b.id: None for b in section.blocks},
    )
    checksums_before = dict(inputs.checksums)
    context_sha_before = inputs.context_sha256
    chapter_blocks_before = list(inputs.context["chapter_blocks"])

    candidate_payload = json.loads(json.dumps({
        "schema_version": 1,
        "book_id": doc.id,
        "run_id": "run",
        "conditions_id": "cond-1",
        "section_id": section.id,
        "block_id": b1.id,
        "input_sha256": inputs.checksums[b1.id],
        "input_generation": 1,
        "group_id": "group-1",
        "proposal_attempt_id": "att-1",
        "original": {"text": b1.text},
        "current_result": asdict(results[b1.id]),
        "result": asdict(replace(results[b1.id], translated_text="Перевод 1 уточнён.")),
        "proposal": {"issues": [{"kind": "note", "quote": "Block", "explanation": "Уточнить"}]},
    }))
    candidate = CorrectionCandidate(
        candidate_id="cand-1",
        book_id=doc.id,
        run_id="run",
        conditions_id="cond-1",
        section_id=section.id,
        block_id=b1.id,
        input_sha256=inputs.checksums[b1.id],
        input_generation=1,
        proposal_sha256=canonical_sha(candidate_payload),
        payload=candidate_payload,
        group_id="group-1",
        proposal_attempt_id="att-1",
    )
    snapshot = SimpleNamespace(
        prompt="Propose.",
        review_prompt="Review proposals.",
        parameters={"max_attempts": 1, "max_output_tokens": None},
    )
    correct_v2 = CorrectChapterV2(unit_of_work=None, llm_client=None, snapshot=snapshot, validator=None)
    task = SimpleNamespace(book_id=doc.id, id="task", scope_id=section.id)

    request = correct_v2._review_request(inputs, task, (candidate,))
    body = json.loads(request.input_text)

    assert "proposal_sha256" not in body["candidates"][0]
    assert request.correction_candidate_refs == ((candidate.candidate_id, candidate.proposal_sha256),)
    assert [cb["block_id"] for cb in body["chapter_blocks"]] == ["b1", "b2", "b3"]
    assert [item["block_id"] for item in body["chapter_analysis"]["treatment_plan"]["items"]] == [
        "b1", "b2", "b3"
    ]
    assert [d["id"] for d in body["chapter_analysis"]["content_plan"]["difficulties"]] == [
        "d1", "d3"
    ]
    assert [f["id"] for f in body["chapter_analysis"]["content_plan"]["context_facts"]] == [
        "cf1", "cf3"
    ]
    assert inputs.checksums == checksums_before
    assert inputs.context_sha256 == context_sha_before
    assert inputs.context["chapter_blocks"] == chapter_blocks_before


@pytest.mark.asyncio
async def test_correct_chapter_v1_splits_group_on_truncated_response():
    from linguary.application.correction_dto import (
        CORRECTION_VERSIONS,
        ChapterCorrectionResponse,
        CorrectionSnapshot,
        UnchangedBlockResponse,
    )
    from linguary.application.use_cases.correct_chapter import (
        CorrectChapter,
        _filtered_chapter_context,
    )
    from linguary.domain import ConceptType, GlossarySnapshotEntry, TermStatus

    section = make_analysis_section(texts=("Block one.", "Block two."))
    doc = make_analysis_document((section,))
    b1, b2 = section.blocks
    plan = TreatmentPlan.of_treatments(
        section.id,
        ((b1.id, BlockTreatment.TRANSLATE), (b2.id, BlockTreatment.TRANSLATE)),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-v1",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(b1.id, b2.id),
    )
    snapshot = CorrectionSnapshot(
        model="gpt-test",
        route={"provider": "openai_compatible", "base_url": "https://example.test/v1", "proxy": "", "route_id": "r1"},
        parameters={
            "temperature": 0.0,
            "max_output_tokens": 1000,
            "timeout_seconds": 30,
            "max_attempts": 1,
            "max_unit_source_chars": 8000,
            "input_price_per_mtok": 0,
            "output_price_per_mtok": 0,
        },
        prompt="Correct v1.",
        versions=CORRECTION_VERSIONS,
    )
    res1 = BlockProcessingResult(block_id=b1.id, treatment=BlockTreatment.TRANSLATE, translated_text="Первый.")
    res2 = BlockProcessingResult(block_id=b2.id, treatment=BlockTreatment.TRANSLATE, translated_text="Второй.")
    results = {b1.id: res1, b2.id: res2}
    cond_id = str(uuid4())
    run_id = str(uuid4())
    task_id = str(uuid4())
    bindings = {
        b.id: {
            "book_id": doc.id,
            "conditions_id": cond_id,
            "attempt_id": str(uuid4()),
            "artifact_revision_id": str(uuid4()),
        }
        for b in section.blocks
    }
    glossary_snapshot = GlossarySnapshot(
        entries=(
            GlossarySnapshotEntry(
                source_text="Block",
                translation="Блок",
                type=ConceptType.TERM,
                status=TermStatus.APPROVED,
                note="",
            ),
        )
    )
    run = SimpleNamespace(id=run_id, glossary_snapshot=glossary_snapshot)
    conditions = SimpleNamespace(id=cond_id, payload={"schema_version": 1, "versions": {"validators": 3}})

    def build_inputs():
        return CorrectionInputs(
            document=doc,
            section=section,
            analysis=analysis,
            run=run,
            conditions=conditions,
            snapshot=snapshot,
            blocks=section.blocks,
            results=dict(results),
            bindings=dict(bindings),
        )

    initial_inputs = build_inputs()
    checksums_before = dict(initial_inputs.checksums)
    orig_glossary = dict(initial_inputs.context["run_glossary"])
    filtered_ctx = _filtered_chapter_context(initial_inputs, (b1.id,))
    assert "note" not in filtered_ctx["run_glossary"]["entries"][0]
    assert initial_inputs.context["run_glossary"] == orig_glossary
    assert initial_inputs.checksums == checksums_before

    identity = correction_task_identity(
        book_id=doc.id,
        run_id=run.id,
        section_id=section.id,
        input_generation="gen-v1",
        policy_version=snapshot.policy_version,
    )
    task = SimpleNamespace(
        id=task_id,
        book_id=doc.id,
        task_type=TaskType.CHAPTER_CORRECTION,
        scope_id=section.id,
        status=TaskStatus.RUNNING,
        attempts=1,
        operation_id=identity.operation_id,
        identity=identity,
    )
    work_bindings = {
        ("task", task.id): {
            "book_id": doc.id,
            "conditions_id": conditions.id,
            "payload_json": {
                "run_id": run.id,
                "section_id": section.id,
                "input_generation": "gen-v1",
                "manifest": initial_inputs.manifest,
            },
        }
    }
    attempts_store: dict[str, SimpleNamespace] = {}
    saved_checks: list = []

    class FakeWork:
        def get_binding(self, kind, entity_id):
            return work_bindings.get((kind, entity_id))

        def bind(self, kind, entity_id, *, book_id, conditions_id, payload, **kwargs):
            work_bindings[(kind, entity_id)] = {
                "book_id": book_id,
                "conditions_id": conditions_id,
                "payload_json": payload,
            }

        def get_attempt(self, attempt_id):
            return attempts_store.get(attempt_id)

        def transition_attempt(self, attempt_id, new_state, *, expected_state):
            attempts_store[attempt_id].state = new_state

        def correction_checks(self, run_id, section_id):
            return tuple(saved_checks)

        def replace_correction_checks(self, run_id, section_id, checks):
            saved_checks.extend(checks)

        def stale_reviews(self, book_id):
            pass

    fake_tx = SimpleNamespace(
        tasks=SimpleNamespace(get=lambda tid: task),
        work=FakeWork(),
        results=SimpleNamespace(replace_results=lambda *a, **kw: None),
    )

    import linguary.application.use_cases.correct_chapter as module
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "correction_inputs", lambda tx, bid, sid: build_inputs())
    monkeypatch.setattr(module, "finalize_response", lambda *a, **kw: None)
    monkeypatch.setattr(module, "record_rejection", lambda *a, **kw: None)

    requested_groups: list[tuple[str, ...]] = []

    class TruncatingFirstCallClient:
        async def generate(self, request):
            ctx = request.call_context
            requested_groups.append(tuple(request.ordered_block_ids))
            attempts_store[ctx.attempt_id] = SimpleNamespace(
                id=ctx.attempt_id,
                state="validated",
                task_id=task.id,
                run_id=run.id,
                conditions_id=conditions.id,
                context={
                    **ctx.to_dict(),
                    "task_attempt": task.attempts,
                    "operation_id": task.operation_id,
                },
            )
            payload = ChapterCorrectionResponse(
                blocks=tuple(UnchangedBlockResponse(block_id=bid) for bid in request.ordered_block_ids)
            )
            is_multi = len(request.ordered_block_ids) > 1
            return LLMResponse(
                payload=payload,
                raw_content="{}",
                provider="test",
                model="gpt-test",
                truncated=is_multi,
                call_context=ctx,
            )

    try:
        handler = CorrectChapter(
            unit_of_work=SimpleNamespace(transaction=lambda: nullcontext(fake_tx)),
            llm_client=TruncatingFirstCallClient(),
            snapshot=snapshot,
            validator=make_validator(),
        )
        checked = await handler._group(
            CorrectChapterCommand(doc.id, section.id, task.id),
            (b1.id, b2.id),
        )
        assert checked == 2
        assert requested_groups == [
            (b1.id, b2.id),
            (b1.id,),
            (b2.id,),
        ]
        assert {c.block_id for c in saved_checks} == {b1.id, b2.id}
    finally:
        monkeypatch.undo()


def test_review_request_aliases_candidate_ids_and_resolves_verdicts():
    section = make_analysis_section(texts=("First block.", "Second block."))
    doc = make_analysis_document((section,))
    b1, b2 = section.blocks
    plan = TreatmentPlan.of_treatments(
        section.id,
        ((b1.id, BlockTreatment.TRANSLATE), (b2.id, BlockTreatment.TRANSLATE)),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(b1.id, b2.id),
    )
    res1 = BlockProcessingResult(block_id=b1.id, treatment=BlockTreatment.TRANSLATE, translated_text="Первый старый.")
    res2 = BlockProcessingResult(block_id=b2.id, treatment=BlockTreatment.TRANSLATE, translated_text="Второй старый.")
    inputs = CorrectionInputs(
        document=doc,
        section=section,
        analysis=analysis,
        run=SimpleNamespace(id="run", glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(id="cond-1", payload={"schema_version": 1}),
        snapshot=SimpleNamespace(to_payload=lambda: {"model": "correction"}),
        blocks=section.blocks,
        results={b1.id: res1, b2.id: res2},
        bindings={b1.id: None, b2.id: None},
        input_generations={b1.id: 1, b2.id: 1},
    )

    def make_candidate(cid: str, block, res, new_text: str):
        payload = json.loads(
            json.dumps(
                {
                    "schema_version": 1,
                    "book_id": doc.id,
                    "run_id": "run",
                    "conditions_id": "cond-1",
                    "section_id": section.id,
                    "block_id": block.id,
                    "input_sha256": inputs.checksums[block.id],
                    "input_generation": 1,
                    "group_id": "group-1",
                    "proposal_attempt_id": "att-1",
                    "original": {"text": block.text},
                    "current_result": asdict(res),
                    "result": asdict(replace(res, translated_text=new_text)),
                    "proposal": {"issues": [{"kind": "note", "quote": "First", "explanation": "Уточнить"}]},
                }
            )
        )
        return CorrectionCandidate(
            candidate_id=cid,
            book_id=doc.id,
            run_id="run",
            conditions_id="cond-1",
            section_id=section.id,
            block_id=block.id,
            input_sha256=inputs.checksums[block.id],
            input_generation=1,
            proposal_sha256=canonical_sha(payload),
            payload=payload,
            group_id="group-1",
            proposal_attempt_id="att-1",
        )

    c1 = make_candidate("cand-uuid-1", b1, res1, "Первый новый.")
    c2 = make_candidate("cand-uuid-2", b2, res2, "Второй новый.")
    snapshot = SimpleNamespace(
        prompt="Propose.",
        review_prompt="Review proposals.",
        parameters={"max_attempts": 1, "max_output_tokens": None},
    )
    correct_v2 = CorrectChapterV2(unit_of_work=None, llm_client=None, snapshot=snapshot, validator=None)
    task = SimpleNamespace(book_id=doc.id, id="task", scope_id=section.id)

    request = correct_v2._review_request(inputs, task, (c1, c2))
    body = json.loads(request.input_text)

    assert [item["candidate_id"] for item in body["candidates"]] == ["c1", "c2"]
    assert "cand-uuid-1" not in request.input_text
    assert "cand-uuid-2" not in request.input_text
    assert request.correction_candidate_refs == (
        (c1.candidate_id, c1.proposal_sha256),
        (c2.candidate_id, c2.proposal_sha256),
    )

    raw_review = ChapterCorrectionReviewResponse(
        candidates=(
            CorrectionVerdict(
                candidate_id="c1",
                verdict="confirmed",
                error_supported=True,
                evidence_valid=True,
                minimal=True,
                no_semantic_loss=True,
                reason="Подтверждено.",
            ),
            CorrectionVerdict(
                candidate_id="c2",
                verdict="needs_human",
                error_supported=False,
                evidence_valid=True,
                minimal=True,
                no_semantic_loss=True,
                reason="Спорно.",
            ),
        )
    )
    resolved = raw_review.with_resolved_proposal_shas(request.correction_candidate_refs)
    assert [(v.candidate_id, v.proposal_sha256) for v in resolved.candidates] == [
        (c1.candidate_id, c1.proposal_sha256),
        (c2.candidate_id, c2.proposal_sha256),
    ]
    resolved.validate_coverage(request.correction_candidate_refs)


@pytest.mark.asyncio
async def test_propose_group_skips_review_group_for_invalid_evidence_candidates(tmp_path):
    from linguary.application.correction_dto import (
        ChapterCorrectionV2Response,
        CorrectionIssue,
        CurrentQuoteEvidence,
        ProposedBlockResponse,
        UnchangedBlockResponse,
    )

    section = make_analysis_section(
        texts=("First unchanged.", "Second with invalid evidence."),
    )
    doc = make_analysis_document((section,))
    b1, b2 = section.blocks
    plan = TreatmentPlan.of_treatments(
        section.id,
        ((b1.id, BlockTreatment.TRANSLATE), (b2.id, BlockTreatment.TRANSLATE)),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(b1.id, b2.id),
    )
    snapshot = CorrectionSnapshotV2(
        model="gpt-test",
        route={"provider": "openai_compatible", "base_url": "https://example.test/v1", "proxy": "", "route_id": "r1"},
        parameters={
            "temperature": 0.0,
            "max_output_tokens": 1000,
            "timeout_seconds": 30,
            "max_attempts": 1,
            "max_unit_source_chars": 8000,
            "input_price_per_mtok": 0,
            "output_price_per_mtok": 0,
        },
        prompt="Propose.",
        review_prompt="Review.",
        versions=CORRECTION_VERSIONS_V2,
    )
    res1 = BlockProcessingResult(block_id=b1.id, treatment=BlockTreatment.TRANSLATE, translated_text="Первый.")
    res2 = BlockProcessingResult(block_id=b2.id, treatment=BlockTreatment.TRANSLATE, translated_text="Второй старый.")
    results = {b1.id: res1, b2.id: res2}
    cond_id = str(uuid4())
    run_id = str(uuid4())
    task_id = str(uuid4())
    bindings = {
        b.id: {
            "book_id": doc.id,
            "conditions_id": cond_id,
            "attempt_id": str(uuid4()),
            "artifact_revision_id": str(uuid4()),
        }
        for b in section.blocks
    }
    generations = {b1.id: 1, b2.id: 1}
    run = SimpleNamespace(id=run_id, glossary_snapshot=GlossarySnapshot())
    conditions = SimpleNamespace(id=cond_id, payload={"schema_version": 1, "versions": {"validators": 3}})

    def build_inputs():
        return CorrectionInputs(
            document=doc,
            section=section,
            analysis=analysis,
            run=run,
            conditions=conditions,
            snapshot=snapshot,
            blocks=section.blocks,
            results=dict(results),
            bindings=dict(bindings),
            input_generations=dict(generations),
        )

    initial_inputs = build_inputs()
    identity = correction_task_identity(
        book_id=doc.id,
        run_id=run.id,
        section_id=section.id,
        input_generation="gen-1",
        policy_version=snapshot.policy_version,
    )
    task = SimpleNamespace(
        id=task_id,
        book_id=doc.id,
        task_type=TaskType.CHAPTER_CORRECTION,
        scope_id=section.id,
        status=TaskStatus.RUNNING,
        attempts=1,
        operation_id=identity.operation_id,
        identity=identity,
    )
    work_bindings = {
        ("task", task.id): {
            "book_id": doc.id,
            "conditions_id": conditions.id,
            "payload_json": {
                "run_id": run.id,
                "section_id": section.id,
                "input_generation": "gen-1",
                "manifest": initial_inputs.manifest,
            },
        },
    }
    attempts_store: dict[str, SimpleNamespace] = {}
    candidates_store: dict[str, CorrectionCandidate] = {}
    saved_checks: list = []

    class FakeWork:
        def get_binding(self, kind, entity_id):
            return work_bindings.get((kind, entity_id))

        def bind(self, kind, entity_id, *, book_id, conditions_id, payload, **kwargs):
            work_bindings[(kind, entity_id)] = {
                "book_id": book_id,
                "conditions_id": conditions_id,
                "payload_json": payload,
            }

        def add_correction_candidates(self, candidates):
            for c in candidates:
                candidates_store[c.candidate_id] = c

        def get_correction_candidate(self, cid):
            return candidates_store.get(cid)

        def list_correction_candidates(self, run_id, *, section_id=None, states=None):
            return tuple(
                c for c in candidates_store.values()
                if states is None or c.state in states
            )

        def correction_checks(self, run_id, section_id):
            return tuple(saved_checks)

        def replace_correction_checks(self, run_id, section_id, checks):
            saved_checks.extend(checks)

        def get_attempt(self, attempt_id):
            return attempts_store.get(attempt_id)

        def transition_attempt(self, attempt_id, new_state, *, expected_state):
            attempts_store[attempt_id].state = new_state

        def archive_artifacts(self, items):
            return tuple(f"art-{i}" for i in range(len(items)))

    fake_tx = SimpleNamespace(
        document=doc,
        tasks=SimpleNamespace(get=lambda tid: task),
        work=FakeWork(),
        results=SimpleNamespace(replace_results=lambda *a, **kw: None),
    )

    import linguary.application.use_cases.correct_chapter as module
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "correction_inputs", lambda tx, bid, sid: build_inputs())
    monkeypatch.setattr(module, "finalize_response", lambda *a, **kw: None)

    class FakeClient:
        async def generate(self, request):
            if request.kind is LLMResponseKind.CHAPTER_CORRECTION_REVIEW:
                raise AssertionError("_review_group must not be called when all candidates have invalid evidence")
            ctx = request.call_context
            attempts_store[ctx.attempt_id] = SimpleNamespace(
                id=ctx.attempt_id,
                book_id=doc.id,
                state="validated",
                task_id=task.id,
                run_id=run.id,
                conditions_id=conditions.id,
                context={
                    **ctx.to_dict(),
                    "task_attempt": task.attempts,
                    "operation_id": task.operation_id,
                },
            )
            payload = ChapterCorrectionV2Response(
                blocks=(
                    UnchangedBlockResponse(block_id=b1.id),
                    ProposedBlockResponse(
                        block_id=b2.id,
                        translated_text="Второй новый.",
                        glossary_hints=(),
                        language_notes=(),
                        issues=(
                            CorrectionIssue(
                                field_path="translated_text",
                                source_quote="NonExistentSourceQuote",
                                current_evidence=CurrentQuoteEvidence(
                                    field_path="translated_text",
                                    quote="Второй",
                                ),
                                problem="Ошибка",
                                minimal_fix_reason="Правка",
                            ),
                        ),
                    ),
                )
            )
            return LLMResponse(
                payload=payload,
                raw_content="{}",
                provider="test",
                model="gpt-test",
                call_context=ctx,
            )

    try:
        handler = CorrectChapterV2(
            unit_of_work=SimpleNamespace(transaction=lambda: nullcontext(fake_tx)),
            llm_client=FakeClient(),
            snapshot=snapshot,
            validator=make_validator(),
        )
        result = await handler(CorrectChapterCommand(doc.id, section.id, task.id))
        assert result.checked_blocks == 2
        assert len(candidates_store) == 1
        saved_candidate, = candidates_store.values()
        assert saved_candidate.block_id == b2.id
        assert saved_candidate.state == "awaiting_human"
        assert saved_candidate.review is not None
        assert saved_candidate.review["verdict"] == "needs_human"
        assert saved_candidate.review["evidence_valid"] is False
        assert saved_candidate.review["reason"] == "Цитата или доказательство в предложении не подтверждены текстом блока."
        assert len(saved_checks) == 1
        assert saved_checks[0].block_id == b1.id
        assert saved_checks[0].basis == "unchanged"

        from datetime import UTC, datetime
        from linguary.adapters.db import Base, SqlAlchemyUnitOfWork, create_database_engine
        from linguary.adapters.db.models import (
            BookConditionsRow,
            BookRow,
            CorrectionInputRow,
            LLMAttemptRow,
            ProcessingRunRow,
        )
        from linguary.domain.book_conditions import WorkStateError, thaw

        engine = create_database_engine(tmp_path / "test.db")
        Base.metadata.create_all(engine)
        uow = SqlAlchemyUnitOfWork.for_engine(engine)
        now = datetime.now(UTC)
        with uow.transaction() as tx:
            session = tx.work._session
            session.add(
                BookRow(
                    id=doc.id,
                    title="Book",
                    authors_json=["Author"],
                    identifier=None,
                    publisher=None,
                    source_language="en",
                    target_language="ru",
                    status="processing",
                    source_epub_path="book.epub",
                    export_validation_status=None,
                    created_at=now,
                )
            )
            session.flush()
            tx.documents.save(doc)
            session.add(
                BookConditionsRow(
                    id=conditions.id,
                    book_id=doc.id,
                    revision=1,
                    schema_version=1,
                    payload_json=conditions.payload,
                    reproducibility="strict",
                    created_at=now,
                )
            )
            session.add(
                ProcessingRunRow(
                    id=run.id,
                    book_id=doc.id,
                    status="active",
                    glossary_snapshot_json=[],
                    llm_settings_json={},
                    created_at=now,
                )
            )
            session.flush()
            attempt_obj = attempts_store[saved_candidate.proposal_attempt_id]
            session.add(
                LLMAttemptRow(
                    id=attempt_obj.id,
                    book_id=doc.id,
                    conditions_id=conditions.id,
                    task_id=task.id,
                    run_id=run.id,
                    generation=0,
                    ordinal=1,
                    state="validated",
                    context_json=attempt_obj.context,
                    outcome_json={},
                )
            )
            session.add(
                CorrectionInputRow(
                    run_id=run.id,
                    block_id=b2.id,
                    generation=1,
                    fingerprint=canonical_sha(
                        dict(
                            input_sha256=saved_candidate.input_sha256,
                            source_result_binding=thaw(saved_candidate.payload["source_binding"]),
                        )
                    ),
                )
            )
            session.flush()
            tx.work.add_correction_candidates((saved_candidate,))
            persisted = tx.work.get_correction_candidate(saved_candidate.candidate_id)
            assert persisted is not None
            assert persisted.state == "awaiting_human"
            assert persisted.review == saved_candidate.review
            assert tx.work.list_correction_candidates(
                run.id, section_id=section.id, states=("awaiting_human",)
            ) == (persisted,)

            valid_evidence_payload = {**thaw(saved_candidate.payload), "evidence_valid": True}
            invalid_for_valid_evidence = replace(
                saved_candidate,
                candidate_id=str(uuid4()),
                payload=valid_evidence_payload,
                proposal_sha256=canonical_sha(valid_evidence_payload),
            )
            with pytest.raises(WorkStateError):
                tx.work.add_correction_candidates((invalid_for_valid_evidence,))
    finally:
        monkeypatch.undo()


def test_alias_error_for_retry_does_not_mutate_original_error():
    from linguary.application.retry_policy import (
        ExpectedBlock,
        ResultExpectation,
        ResultValidationError,
    )
    from linguary.application.use_cases.correct_chapter import _alias_error_for_retry

    err = ResultValidationError(
        "Unbalanced quote in block uuid-block-1",
        expectation=ResultExpectation(
            target_language="ru",
            blocks=(ExpectedBlock(block_id="uuid-block-1"),),
        ),
    )
    aliased = _alias_error_for_retry(err, {"uuid-block-1": "b1"})

    assert aliased is not err
    assert str(aliased) == "Unbalanced quote in block b1"
    assert [b.block_id for b in aliased.expectation.blocks] == ["b1"]
    assert str(err) == "Unbalanced quote in block uuid-block-1"
    assert [b.block_id for b in err.expectation.blocks] == ["uuid-block-1"]


@pytest.mark.asyncio
async def test_alias_error_for_retry_replaces_difficulty_and_fact_ids_in_single_pass():
    from linguary.application.retry_policy import ResultValidationError
    from linguary.application.use_cases.correct_chapter import _alias_error_for_retry

    diff_id = "df-" + "a" * 64
    fact_id = "cf-" + "b" * 64
    err = ResultValidationError(
        f"Block block-1 and block-10 failed on {diff_id}, {fact_id}, and sub-{diff_id}-tail"
    )
    aliased = _alias_error_for_retry(
        err,
        {"block-1": "b1", "block-10": "b10", "b1": "d99"},
        {diff_id: "d1"},
        {fact_id: "f1"},
    )
    assert str(aliased) == f"Block b1 and b10 failed on d1, f1, and sub-{diff_id}-tail"

    aliased_str = _alias_error_for_retry(
        f"Error in block-10 and {diff_id} and {fact_id}",
        {"block-1": "b1", "block-10": "b10"},
        {diff_id: "d1"},
        {fact_id: "f1"},
    )
    assert aliased_str == "Error in b10 and d1 and f1"


def test_correct_chapter_candidates_skips_unchanged_revalidation_and_allows_translated_blocks_with_english_help(
    monkeypatch,
):
    from linguary.application.correction_dto import (
        CORRECTION_VERSIONS,
        ChapterCorrectionResponse,
        CorrectedBlockResponse,
        CorrectionSnapshot,
        UnchangedBlockResponse,
    )
    from linguary.application.reader_text import ReaderTextPolicy
    from linguary.application.use_cases.correct_chapter import CorrectChapter
    from linguary.domain.content_plan import (
        ContentEvidence,
        ContentPlan,
        Difficulty,
        RequiredPoint,
        SourceAnchor,
    )
    from linguary.domain.learning import FragmentAssistance

    section = make_analysis_section(texts=("First unchanged.", "Second translated with help."))
    doc = make_analysis_document((section,))
    b1, b2 = section.blocks
    plan = TreatmentPlan.of_treatments(
        section.id,
        ((b1.id, BlockTreatment.TRANSLATE), (b2.id, BlockTreatment.TRANSLATE)),
    )
    anchor2 = SourceAnchor(block_id=b2.id, start=0, end=6, quote="Second")
    diff2 = Difficulty(
        id="df-" + "1" * 64,
        anchors=(anchor2,),
        owner_block_id=b2.id,
        kind="idiom",
        priority="critical",
        help_kind="explanation",
        required_points=(RequiredPoint(id="p1", description="point"),),
    )
    content_plan = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(diff2,),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(b1.id, b2.id),
        content_plan=content_plan,
    )
    assistance2 = FragmentAssistance(
        difficulty_id=diff2.id,
        point_ids=("p1",),
        kind="russian_explanation",
        text="Пояснение к фрагменту.",
        anchors=(anchor2,),
    )
    res1 = BlockProcessingResult(
        block_id=b1.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Первый перевод.",
        content_evidence=ContentEvidence(plan_sha256=content_plan.sha256, version=3),
    )
    res2 = BlockProcessingResult(
        block_id=b2.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Второй старый перевод.",
        fragment_assistances=(assistance2,),
        content_evidence=ContentEvidence(plan_sha256=content_plan.sha256, version=3),
    )
    snapshot = CorrectionSnapshot(
        model="gpt-test",
        route={"provider": "openai_compatible", "base_url": "https://example.test/v1", "proxy": "", "route_id": "r1"},
        parameters={
            "temperature": 0.0,
            "max_output_tokens": 1000,
            "timeout_seconds": 30,
            "max_attempts": 1,
            "max_unit_source_chars": 8000,
            "input_price_per_mtok": 0,
            "output_price_per_mtok": 0,
        },
        prompt="Correct.",
        versions=CORRECTION_VERSIONS,
    )
    inputs = CorrectionInputs(
        document=doc,
        section=section,
        analysis=analysis,
        run=SimpleNamespace(id="run-1", glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(
            id="cond-1",
            payload={"schema_version": 1, "english_help": True, "versions": {"validators": 3}},
        ),
        snapshot=snapshot,
        blocks=section.blocks,
        results={b1.id: res1, b2.id: res2},
        bindings={b1.id: None, b2.id: None},
    )

    prepare_calls: list[tuple[str, ...]] = []
    orig_prepare = ReaderTextPolicy.prepare_results

    def spy_prepare(self, results, *args, **kwargs):
        prepare_calls.append(tuple(r.block_id for r in results))
        return orig_prepare(self, results, *args, **kwargs)

    monkeypatch.setattr(ReaderTextPolicy, "prepare_results", spy_prepare)

    handler = CorrectChapter(
        unit_of_work=None,
        llm_client=None,
        snapshot=snapshot,
        validator=make_validator(),
    )
    response = ChapterCorrectionResponse(
        blocks=(
            UnchangedBlockResponse(block_id=b1.id),
            CorrectedBlockResponse(
                block_id=b2.id,
                translated_text="Второй обновлённый перевод.",
                glossary_hints=(),
                language_notes=(),
            ),
        )
    )
    candidates = handler._candidates(None, inputs, response)
    assert len(candidates) == 2
    assert candidates[0] == res1
    assert candidates[1].translated_text == "Второй обновлённый перевод."
    assert (b1.id,) not in prepare_calls
    assert prepare_calls[0] == (b2.id,)





