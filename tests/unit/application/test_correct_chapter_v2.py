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


@pytest.mark.asyncio
async def test_correct_chapter_v2_counts_unchanged_blocks_and_skips_reload_in_review_group(monkeypatch):
    import linguary.application.use_cases.correct_chapter as module
    from linguary.application.correction_dto import (
        ChapterCorrectionV2Response,
        CorrectionIssue,
        CurrentQuoteEvidence,
        ProposedBlockResponse,
        UnchangedBlockResponse,
    )
    from linguary.domain.content_plan import ContentEvidence, ContentPlan

    section = make_analysis_section(
        texts=("First unchanged.", "Second valid proposal.", "Third invalid evidence."),
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
    content_plan = ContentPlan(version=3, profile="content-plan-v3", difficulties=())
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(b1.id, b2.id, b3.id),
        content_plan=content_plan,
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
    ev = ContentEvidence(plan_sha256=content_plan.sha256, version=3)
    res1 = BlockProcessingResult(block_id=b1.id, treatment=BlockTreatment.TRANSLATE, translated_text="Первый.", content_evidence=ev)
    res2 = BlockProcessingResult(block_id=b2.id, treatment=BlockTreatment.TRANSLATE, translated_text="Второй старый.", content_evidence=ev)
    res3 = BlockProcessingResult(block_id=b3.id, treatment=BlockTreatment.TRANSLATE, translated_text="Третий старый.", content_evidence=ev)
    results = {b1.id: res1, b2.id: res2, b3.id: res3}
    cond_id = str(uuid4())
    run_id = str(uuid4())
    task_id = str(uuid4())
    bindings = {
        b.id: {
            "book_id": doc.id,
            "conditions_id": cond_id,
            "attempt_id": str(uuid4()),
            "artifact_revision_id": str(uuid4()),
            "payload_json": {"schema_version": 1},
        }
        for b in section.blocks
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
    saved_checks: dict[str, object] = {}
    results_replaced = False

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
            return tuple(saved_checks.values())

        def replace_correction_checks(self, run_id, section_id, checks):
            for c in checks:
                saved_checks[c.block_id] = c

        def get_attempt(self, attempt_id):
            return attempts_store.get(attempt_id)

        def transition_attempt(self, attempt_id, new_state, *, expected_state):
            attempts_store[attempt_id].state = new_state

        def archive_artifacts(self, items):
            return tuple(f"art-{len(attempts_store)}-{i}" for i in range(len(items)))

        def record_correction_verdict(self, candidate_id, *, state, review, review_attempt_id, **kwargs):
            old = candidates_store[candidate_id]
            candidates_store[candidate_id] = replace(
                old, state=state, review=review, review_attempt_id=review_attempt_id
            )

        def replace_result_binding(self, key, *, book_id, conditions_id, attempt_id, artifact_revision_id, **kwargs):
            block_id = key.split(":", 1)[1]
            bindings[block_id] = {
                **bindings[block_id],
                "book_id": book_id,
                "conditions_id": conditions_id,
                "attempt_id": attempt_id,
                "artifact_revision_id": artifact_revision_id,
                "payload_json": {"schema_version": 1},
            }

        def advance_correction_input(self, run_id, block_id, *, expected_generation, fingerprint):
            pass

        def stale_reviews(self, book_id):
            pass

    class FakeResults:
        def replace_results(self, run_id, updated):
            nonlocal results_replaced
            results_replaced = True
            for r in updated:
                results[r.block_id] = r

    fake_tx = SimpleNamespace(
        tasks=SimpleNamespace(get=lambda tid: task),
        work=FakeWork(),
        results=FakeResults(),
    )

    def guarded_correction_inputs(tx, bid, sid):
        if results_replaced:
            raise AssertionError("correction_inputs must not be called after results are replaced; use with_updated_state")
        return build_inputs()

    note_source_calls = 0
    orig_validate_note_sources = module.validate_note_sources

    def spy_validate_note_sources(results_arg, texts_arg):
        nonlocal note_source_calls
        note_source_calls += 1
        return orig_validate_note_sources(results_arg, texts_arg)

    monkeypatch.setattr(module, "correction_inputs", guarded_correction_inputs)
    monkeypatch.setattr(module, "validate_note_sources", spy_validate_note_sources)
    monkeypatch.setattr(module, "finalize_response", lambda *a, **kw: None)

    class FakeClient:
        async def generate(self, request):
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
            if request.kind is LLMResponseKind.CHAPTER_CORRECTION:
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
                                    source_quote="Second",
                                    current_evidence=CurrentQuoteEvidence(
                                        field_path="translated_text",
                                        quote="Второй",
                                    ),
                                    problem="Точность",
                                    minimal_fix_reason="Уточнить",
                                ),
                            ),
                        ),
                        ProposedBlockResponse(
                            block_id=b3.id,
                            translated_text="Третий спорный.",
                            glossary_hints=(),
                            language_notes=(),
                            issues=(
                                CorrectionIssue(
                                    field_path="translated_text",
                                    source_quote="MissingQuote",
                                    current_evidence=CurrentQuoteEvidence(
                                        field_path="translated_text",
                                        quote="Третий",
                                    ),
                                    problem="Ошибка",
                                    minimal_fix_reason="Правка",
                                ),
                            ),
                        ),
                    )
                )
            else:
                cand_id, prop_sha = request.correction_candidate_refs[0]
                payload = ChapterCorrectionReviewResponse(
                    candidates=(
                        CorrectionVerdict(
                            candidate_id=cand_id,
                            proposal_sha256=prop_sha,
                            verdict="confirmed",
                            error_supported=True,
                            evidence_valid=True,
                            minimal=True,
                            no_semantic_loss=True,
                            reason="Подтверждено.",
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

    handler = CorrectChapterV2(
        unit_of_work=SimpleNamespace(transaction=lambda: nullcontext(fake_tx)),
        llm_client=FakeClient(),
        snapshot=snapshot,
        validator=make_validator(),
    )
    propose_checked = await handler._propose_group(
        CorrectChapterCommand(doc.id, section.id, task.id),
        (b1.id, b2.id, b3.id),
    )
    # b1 is unchanged and b3 went directly to awaiting_human; only b2 is awaiting_review
    assert propose_checked == 2
    assert note_source_calls == 0

    awaiting_review_ids = tuple(
        c.candidate_id for c in candidates_store.values() if c.state == "awaiting_review"
    )
    assert len(awaiting_review_ids) == 1
    review_checked = await handler._review_group(
        CorrectChapterCommand(doc.id, section.id, task.id),
        awaiting_review_ids,
    )
    assert review_checked == 1
    assert propose_checked + review_checked == 3


def test_correct_chapter_v2_rebinds_content_evidence_per_block_and_matches_prepare_candidates_sha():
    from linguary.application.correction_dto import (
        ChapterCorrectionV2Response,
        CorrectionIssue,
        CurrentQuoteEvidence,
        ProposedBlockResponse,
        UnchangedBlockResponse,
    )
    from linguary.application.llm_dto import LanguageNoteResponse
    from linguary.application.retry_policy import ResultValidationError
    from linguary.application.use_cases.correct_chapter import (
        CorrectChapter,
        _alias_error_for_retry,
        prepare_correction_candidates,
    )
    from linguary.domain import LanguageNote
    from linguary.domain.book_conditions import thaw
    from linguary.domain.content_plan import (
        ContentEvidence,
        ContentPlan,
        Difficulty,
        HelpEvidence,
        RequiredPoint,
        SourceAnchor,
        artifact_key,
    )
    from linguary.domain.llm_log import Violation

    section = make_analysis_section(
        texts=(
            "Plain opening.",
            "Second idiom appears here.",
            "Third idiom appears here.",
        ),
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
    anchor2 = SourceAnchor(block_id=b2.id, start=0, end=12, quote="Second idiom")
    anchor3 = SourceAnchor(block_id=b3.id, start=0, end=11, quote="Third idiom")
    diff2 = Difficulty(
        id="df-" + "2" * 64,
        anchors=(anchor2,),
        owner_block_id=b2.id,
        kind="idiom",
        priority="critical",
        help_kind="explanation",
        required_points=(RequiredPoint(id="p2", description="explain second idiom"),),
    )
    diff3 = Difficulty(
        id="df-" + "3" * 64,
        anchors=(anchor3,),
        owner_block_id=b3.id,
        kind="idiom",
        priority="critical",
        help_kind="explanation",
        required_points=(RequiredPoint(id="p3", description="explain third idiom"),),
    )
    content_plan = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(diff2, diff3),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-ev",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(b1.id, b2.id, b3.id),
        content_plan=content_plan,
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

    old_note2 = LanguageNote(
        id="n-1-1",
        block_id=b2.id,
        original_excerpt="Second idiom",
        note="Старое объяснение второй идиомы.",
        translated_excerpt=None,
    )
    old_note3 = LanguageNote(
        id="n-1-2",
        block_id=b3.id,
        original_excerpt="Third idiom",
        note="Старое объяснение третьей идиомы.",
        translated_excerpt=None,
    )
    ev1 = ContentEvidence(plan_sha256=content_plan.sha256, evidence=(), version=3)
    ev2 = ContentEvidence(
        plan_sha256=content_plan.sha256,
        evidence=(
            HelpEvidence(
                difficulty_id=diff2.id,
                point_id="p2",
                artifact_kind="language_note.note",
                block_id=b2.id,
                artifact_key=artifact_key(
                    "language_note.note",
                    b2.id,
                    {"original_excerpt": old_note2.original_excerpt, "note": old_note2.note, "translated_excerpt": None},
                ),
                body_excerpt=old_note2.note,
            ),
        ),
        version=3,
    )
    ev3 = ContentEvidence(
        plan_sha256=content_plan.sha256,
        evidence=(
            HelpEvidence(
                difficulty_id=diff3.id,
                point_id="p3",
                artifact_kind="language_note.note",
                block_id=b3.id,
                artifact_key=artifact_key(
                    "language_note.note",
                    b3.id,
                    {"original_excerpt": old_note3.original_excerpt, "note": old_note3.note, "translated_excerpt": None},
                ),
                body_excerpt=old_note3.note,
            ),
        ),
        version=3,
    )
    res1 = BlockProcessingResult(block_id=b1.id, treatment=BlockTreatment.TRANSLATE, translated_text="Простое начало.", content_evidence=ev1)
    res2 = BlockProcessingResult(
        block_id=b2.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Вторая идиома здесь.",
        language_notes=(old_note2,),
        content_evidence=ev2,
    )
    res3 = BlockProcessingResult(
        block_id=b3.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Третья идиома здесь.",
        language_notes=(old_note3,),
        content_evidence=ev3,
    )
    results = {b1.id: res1, b2.id: res2, b3.id: res3}
    bindings = {
        b.id: {"book_id": doc.id, "conditions_id": "cond-1", "attempt_id": "att-0", "artifact_revision_id": f"art-{idx}"}
        for idx, b in enumerate(section.blocks, start=1)
    }
    inputs = CorrectionInputs(
        document=doc,
        section=section,
        analysis=analysis,
        run=SimpleNamespace(id="run-1", glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(id="cond-1", payload={"schema_version": 1, "versions": {"validators": 3}}),
        snapshot=snapshot,
        blocks=section.blocks,
        results=results,
        bindings=bindings,
        input_generations={b1.id: 1, b2.id: 1, b3.id: 1},
    )
    fake_tx = SimpleNamespace(
        work=SimpleNamespace(list_correction_candidates=lambda *a, **kw: ()),
    )

    prop2 = ProposedBlockResponse(
        block_id=b2.id,
        translated_text="Вторая идиома здесь.",
        glossary_hints=(),
        language_notes=(
            LanguageNoteResponse(
                source_block_id=b2.id,
                original_excerpt="Second idiom",
                note="Уточнённое объяснение второй идиомы.",
                translated_excerpt=None,
            ),
        ),
        issues=(
            CorrectionIssue(
                field_path="language_notes[0].note",
                source_quote="Second idiom",
                current_evidence=CurrentQuoteEvidence(field_path="language_notes[0].note", quote="Старое"),
                problem="Неточное пояснение",
                minimal_fix_reason="Уточнить смысл идиомы",
            ),
        ),
    )
    prop3 = ProposedBlockResponse(
        block_id=b3.id,
        translated_text="Третья идиома здесь.",
        glossary_hints=(),
        language_notes=(
            LanguageNoteResponse(
                source_block_id=b3.id,
                original_excerpt="here",
                note="Дополнительная заметка о наречии.",
                translated_excerpt=None,
            ),
            LanguageNoteResponse(
                source_block_id=b3.id,
                original_excerpt="Third idiom",
                note="Уточнённое объяснение третьей идиомы.",
                translated_excerpt=None,
            ),
        ),
        issues=(
            CorrectionIssue(
                field_path="language_notes[0].note",
                source_quote="Third idiom",
                current_evidence=CurrentQuoteEvidence(field_path="language_notes[0].note", quote="Старое"),
                problem="Неточное пояснение",
                minimal_fix_reason="Уточнить смысл третьей идиомы",
            ),
        ),
    )
    validator = make_validator()
    handler = CorrectChapter(unit_of_work=None, llm_client=None, snapshot=snapshot, validator=validator)
    response = ChapterCorrectionV2Response((UnchangedBlockResponse(b1.id), prop2, prop3))
    cand_results = handler._candidates(fake_tx, inputs, response)

    assert cand_results[1].content_evidence.evidence[0].body_excerpt == "Уточнённое объяснение второй идиомы."
    assert cand_results[2].content_evidence.evidence[0].body_excerpt == "Уточнённое объяснение третьей идиомы."

    immutable_b3 = json.loads(
        json.dumps(
            {
                "schema_version": 1,
                "book_id": doc.id,
                "run_id": "run-1",
                "conditions_id": "cond-1",
                "section_id": section.id,
                "block_id": b3.id,
                "input_sha256": inputs.checksums[b3.id],
                "input_generation": 1,
                "group_id": "grp-1",
                "proposal_attempt_id": "att-1",
                "context_sha256": inputs.context_sha256,
                "source_binding": bindings[b3.id],
                "original": {"text": b3.text},
                "current_result": asdict(res3),
                "result": asdict(cand_results[2]),
                "proposal": asdict(prop3),
                "evidence_valid": True,
            }
        )
    )
    cand_b3 = CorrectionCandidate(
        candidate_id="cand-b3",
        book_id=doc.id,
        run_id="run-1",
        conditions_id="cond-1",
        section_id=section.id,
        block_id=b3.id,
        input_sha256=inputs.checksums[b3.id],
        input_generation=1,
        proposal_sha256=canonical_sha(immutable_b3),
        payload=immutable_b3,
        group_id="grp-1",
        proposal_attempt_id="att-1",
    )
    fake_tx_with_cand = SimpleNamespace(
        work=SimpleNamespace(list_correction_candidates=lambda *a, **kw: (cand_b3,)),
    )
    reprepared, = prepare_correction_candidates(fake_tx_with_cand, inputs, (cand_b3,), validator)
    assert canonical_sha(asdict(reprepared)) == canonical_sha(thaw(cand_b3.payload["result"]))

    # Missing coverage raises ResultValidationError with canonical violation; _alias_error_for_retry aliases on a copy only.
    broken_prop2 = replace(prop2, language_notes=())
    with pytest.raises(ResultValidationError) as exc_info:
        handler._candidates(
            fake_tx,
            inputs,
            ChapterCorrectionV2Response((UnchangedBlockResponse(b1.id), broken_prop2, UnchangedBlockResponse(b3.id))),
        )
    err = exc_info.value
    assert err.violation is not None
    assert err.violation.block_ids == (b2.id,)

    violation_only_err = ResultValidationError(
        "Ошибка без идентификатора в тексте",
        violation=Violation("content_plan", "chapter_correction", scope="local", block_ids=(b2.id,), safe_message="msg"),
    )
    aliased_err = _alias_error_for_retry(violation_only_err, {b2.id: "b2"})
    assert aliased_err is not violation_only_err
    assert aliased_err.violation.block_ids == ("b2",)
    assert violation_only_err.violation.block_ids == (b2.id,)

    # A2 with validators v4 resolves uncovered difficulty into partial coverage + unresolved fragment_assistances.
    a2_inputs = CorrectionInputs(
        document=doc,
        section=section,
        analysis=analysis,
        run=SimpleNamespace(id="run-1", glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(
            id="cond-a2",
            payload={"schema_version": 1, "reader_level": "a2", "versions": {"validators": 4}},
        ),
        snapshot=snapshot,
        blocks=section.blocks,
        results=results,
        bindings=bindings,
        input_generations={b1.id: 1, b2.id: 1, b3.id: 1},
    )
    a2_results = handler._candidates(
        fake_tx,
        a2_inputs,
        ChapterCorrectionV2Response((UnchangedBlockResponse(b1.id), broken_prop2, UnchangedBlockResponse(b3.id))),
    )
    assert a2_results[1].content_evidence.structural_coverage == "partial"
    assert len(a2_results[1].fragment_assistances) == 1
    assert a2_results[1].fragment_assistances[0].difficulty_id == diff2.id
    assert a2_results[1].fragment_assistances[0].kind == "unresolved"


@pytest.mark.asyncio
async def test_correct_chapter_v2_materializes_null_lists_and_compacts_archive_preserving_confirmed_candidate(
    tmp_path,
    monkeypatch,
):
    from datetime import UTC, datetime
    from hashlib import sha256

    import linguary.application.use_cases.correct_chapter as module
    from linguary.adapters.db import Base, SqlAlchemyUnitOfWork, create_database_engine
    from linguary.adapters.db.models import (
        BookConditionsRow,
        BookRow,
        CorrectionInputRow,
        LLMAttemptRow,
        ProcessingRunRow,
    )
    from linguary.adapters.llm.correction_payloads import ChapterCorrectionV2Schema
    from linguary.application.correction_dto import (
        LLMPayloadError,
        proposed_block_from_payload,
    )
    from linguary.application.use_cases.correct_chapter import _filtered_chapter_context
    from linguary.domain import (
        ConceptType,
        GlossaryHint,
        GlossarySnapshotEntry,
        LanguageNote,
        TermStatus,
    )
    from linguary.domain.book_conditions import thaw

    section = make_analysis_section(
        texts=(
            "Lantern shines brightly.",
            "Second quiet sentence.",
            "Third quiet sentence.",
            "Fourth quiet sentence.",
            "Fifth quiet sentence.",
            "Harbor appears far away.",
        ),
    )
    doc = make_analysis_document((section,))
    b1, b2, b3, b4, b5, b6 = section.blocks
    plan = TreatmentPlan.of_treatments(
        section.id,
        (
            (b1.id, BlockTreatment.GLOSSARY_ONLY),
            *((b.id, BlockTreatment.TRANSLATE) for b in (b2, b3, b4, b5, b6)),
        ),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-null",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=tuple(b.id for b in section.blocks),
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
    glossary_snapshot = GlossarySnapshot(
        entries=(
            GlossarySnapshotEntry(
                source_text="Lantern",
                translation="Фонарь",
                type=ConceptType.TERM,
                status=TermStatus.APPROVED,
            ),
            GlossarySnapshotEntry(
                source_text="Harbor",
                translation="Гавань",
                type=ConceptType.PLACE,
                status=TermStatus.APPROVED,
            ),
        )
    )
    existing_hint = GlossaryHint(original="Lantern", translation="фонарь", note="светильник")
    existing_note = LanguageNote(
        id="n-1-1",
        block_id=b2.id,
        original_excerpt="Second quiet",
        note="Спокойная интонация.",
        translated_excerpt="Второе тихое",
    )
    results = {
        b1.id: BlockProcessingResult(
            block_id=b1.id,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            translated_text=None,
            glossary_hints=(existing_hint,),
        ),
        b2.id: BlockProcessingResult(
            block_id=b2.id,
            treatment=BlockTreatment.TRANSLATE,
            translated_text="Второе старое предложение.",
            language_notes=(existing_note,),
        ),
        **{
            b.id: BlockProcessingResult(
                block_id=b.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text=f"Перевод {idx}.",
            )
            for idx, b in enumerate((b3, b4, b5, b6), start=3)
        },
    }
    cond_id = str(uuid4())
    run_id = str(uuid4())
    task_id = str(uuid4())
    bindings = {
        b.id: {
            "book_id": doc.id,
            "conditions_id": cond_id,
            "attempt_id": str(uuid4()),
            "artifact_revision_id": str(uuid4()),
            "payload_json": {"schema_version": 1},
        }
        for b in section.blocks
    }
    generations = {b.id: 1 for b in section.blocks}
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
            input_generations=dict(generations),
        )

    initial_inputs = build_inputs()

    # Window around b1 includes b1..b3, so "Lantern" stays and "Harbor" (only in b6) is filtered out.
    filtered = _filtered_chapter_context(initial_inputs, (b1.id,))
    assert [e["source_text"] for e in filtered["run_glossary"]["entries"]] == ["Lantern"]

    # Schema -> DTO -> alias resolution -> proposed_block_from_payload preserves None before materialization.
    parsed_schema = ChapterCorrectionV2Schema.model_validate(
        {
            "blocks": [
                {"block_id": "b1", "status": "unchanged"},
                {
                    "block_id": "b2",
                    "status": "proposed",
                    "translated_text": "Второе тихое предложение.",
                    "glossary_hints": None,
                    "language_notes": None,
                    "issues": [
                        {
                            "field_path": "translated_text",
                            "source_quote": "Second quiet",
                            "current_evidence": {
                                "kind": "quote",
                                "field_path": "translated_text",
                                "quote": "старое",
                            },
                            "problem": "Неточный перевод",
                            "minimal_fix_reason": "Передать quiet точно",
                        }
                    ],
                },
                *[{"block_id": f"b{i}", "status": "unchanged"} for i in range(3, 7)],
            ]
        }
    )
    dto_with_aliases = parsed_schema.to_dto()
    resolved_dto = dto_with_aliases.with_resolved_block_ids(initial_inputs.alias_to_block_id)
    raw_b2 = resolved_dto.blocks[1]
    assert raw_b2.glossary_hints is None
    assert raw_b2.language_notes is None
    with pytest.raises(LLMPayloadError):
        raw_b2.validate_treatment(BlockTreatment.TRANSLATE)
    roundtrip_partial = proposed_block_from_payload(asdict(raw_b2))
    assert roundtrip_partial.glossary_hints is None
    assert roundtrip_partial.language_notes is None

    # Full _propose_group + _review_group and SQLite _confirmed_candidate check.
    engine = create_database_engine(tmp_path / "compact_archive.db")
    Base.metadata.create_all(engine)
    uow = SqlAlchemyUnitOfWork.for_engine(engine)
    now = datetime.now(UTC)

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
    archived_items: list[dict] = []
    saved_checks: dict[str, object] = {}
    captured_requests: list = []

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
            return tuple(saved_checks.values())

        def replace_correction_checks(self, run_id, section_id, checks):
            for c in checks:
                saved_checks[c.block_id] = c

        def get_attempt(self, attempt_id):
            return attempts_store.get(attempt_id)

        def transition_attempt(self, attempt_id, new_state, *, expected_state):
            attempts_store[attempt_id].state = new_state

        def archive_artifacts(self, items):
            ids = []
            for item in items:
                art_id = str(uuid4())
                archived_items.append({"id": art_id, **item})
                ids.append(art_id)
            return tuple(ids)

        def record_correction_verdict(self, candidate_id, *, state, review, review_attempt_id, **kwargs):
            old = candidates_store[candidate_id]
            candidates_store[candidate_id] = replace(
                old, state=state, review=review, review_attempt_id=review_attempt_id
            )

        def replace_result_binding(self, key, *, book_id, conditions_id, attempt_id, artifact_revision_id, **kwargs):
            block_id = key.split(":", 1)[1]
            bindings[block_id] = {
                **bindings[block_id],
                "book_id": book_id,
                "conditions_id": conditions_id,
                "attempt_id": attempt_id,
                "artifact_revision_id": artifact_revision_id,
                "payload_json": {"schema_version": 1},
            }

        def advance_correction_input(self, run_id, block_id, *, expected_generation, fingerprint):
            pass

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

    monkeypatch.setattr(module, "correction_inputs", lambda tx, bid, sid: build_inputs())
    monkeypatch.setattr(module, "finalize_response", lambda *a, **kw: None)

    class FakeClient:
        async def generate(self, request):
            captured_requests.append(request)
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
            if request.kind is LLMResponseKind.CHAPTER_CORRECTION:
                return LLMResponse(
                    payload=resolved_dto,
                    raw_content='{"huge_raw_response": true}',
                    provider="test",
                    model="gpt-test",
                    call_context=ctx,
                )
            cand_id, prop_sha = request.correction_candidate_refs[0]
            return LLMResponse(
                payload=ChapterCorrectionReviewResponse(
                    candidates=(
                        CorrectionVerdict(
                            candidate_id=cand_id,
                            proposal_sha256=prop_sha,
                            verdict="confirmed",
                            error_supported=True,
                            evidence_valid=True,
                            minimal=True,
                            no_semantic_loss=True,
                            reason="Подтверждено.",
                        ),
                    )
                ),
                raw_content='{"huge_review_raw": true}',
                provider="test",
                model="gpt-test",
                call_context=ctx,
            )

    handler = CorrectChapterV2(
        unit_of_work=SimpleNamespace(transaction=lambda: nullcontext(fake_tx)),
        llm_client=FakeClient(),
        snapshot=snapshot,
        validator=make_validator(),
    )
    all_block_ids = tuple(b.id for b in section.blocks)
    await handler._propose_group(CorrectChapterCommand(doc.id, section.id, task.id), all_block_ids)

    cand, = candidates_store.values()
    thawed_proposal = thaw(cand.payload["proposal"])
    assert thawed_proposal["glossary_hints"] == []
    assert len(thawed_proposal["language_notes"]) == 1
    assert thawed_proposal["language_notes"][0]["original_excerpt"] == "Second quiet"
    assert len(cand.payload["result"]["language_notes"]) == 1
    assert cand.payload["result"]["language_notes"][0]["id"] == "n-1-1"

    await handler._review_group(CorrectChapterCommand(doc.id, section.id, task.id), (cand.candidate_id,))
    reviewed_cand = candidates_store[cand.candidate_id]
    assert reviewed_cand.state == "auto_applied"

    # Review request filtered run_glossary to window around b2 (b1..b4) and preserved original.
    review_req = captured_requests[1]
    review_input = json.loads(review_req.input_text)
    assert [e["source_text"] for e in review_input["run_glossary"]["entries"]] == ["Lantern"]
    assert review_input["candidates"][0]["original"] == thaw(cand.payload["original"])

    # Verify compact archive structure for both proposal and review artifacts.
    proposal_art = next(a for a in archived_items if a["kind"] == "correction_proposal")
    review_art = next(a for a in archived_items if a["kind"] == "correction_review")
    expected_keys = {
        "schema_version",
        "response",
        "context_sha256",
        "input_checksums",
        "request_sha256",
        "ordered_block_ids",
        "candidate_refs",
    }
    for art, req in ((proposal_art, captured_requests[0]), (review_art, review_req)):
        assert set(art["body"].keys()) == expected_keys
        assert art["body"]["context_sha256"] == initial_inputs.context_sha256
        assert art["body"]["request_sha256"] == sha256(
            (req.instructions + "\n" + req.input_text).encode("utf-8")
        ).hexdigest()

    # Persist into real SQLite and verify SqlAlchemyWorkRepository._confirmed_candidate returns True.
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
                reproducibility="complete",
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
        for att in attempts_store.values():
            session.add(
                LLMAttemptRow(
                    id=att.id,
                    book_id=doc.id,
                    conditions_id=conditions.id,
                    task_id=task.id,
                    run_id=run.id,
                    generation=1,
                    ordinal=1,
                    state="accepted",
                    context_json=att.context,
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
                        input_sha256=cand.input_sha256,
                        source_result_binding=thaw(cand.payload["source_binding"]),
                    )
                ),
            )
        )
        session.flush()
        tx.work.archive_artifacts([review_art])
        tx.work.add_correction_candidates((cand,))
        tx.work.record_correction_verdict(
            cand.candidate_id,
            proposal_sha256=cand.proposal_sha256,
            input_sha256=cand.input_sha256,
            review_attempt_id=reviewed_cand.review_attempt_id,
            review=reviewed_cand.review,
            state="auto_applied",
        )
        persisted = tx.work.get_correction_candidate(cand.candidate_id)
        assert tx.work._confirmed_candidate(persisted) is True


def test_correction_issue_and_note_id_preservation_match_normalized_typography():
    from linguary.application.correction_dto import (
        CORRECTION_VERSIONS,
        ChapterCorrectionResponse,
        ChapterCorrectionV2Response,
        CorrectedBlockResponse,
        CorrectionIssue,
        CorrectionSnapshot,
        CurrentQuoteEvidence,
        ProposedBlockResponse,
        UnchangedBlockResponse,
    )
    from linguary.application.llm_dto import GlossaryHintResponse, LanguageNoteResponse
    from linguary.application.retry_policy import ResultValidationError
    from linguary.application.use_cases.correct_chapter import CorrectChapter
    from linguary.domain import GlossaryHint, LanguageNote
    from linguary.domain.content_plan import (
        ContentEvidence,
        ContentPlan,
        Difficulty,
        RequiredPoint,
        SourceAnchor,
    )
    from linguary.domain.learning import FragmentAssistance

    section = make_analysis_section(
        texts=(
            "“Don’t—wait,” she whispered.",
            "“Hold—on,” he muttered.",
        ),
    )
    doc = make_analysis_document((section,))
    b1, b2 = section.blocks

    # 1. CurrentQuoteEvidence.valid_for and CorrectionIssue.valid_for use find_quote_span
    #    and exclude metadata keys {'id', 'block_id', 'source_block_id'}.
    current_note = LanguageNote(
        id="ln-ch01-007",
        block_id=b1.id,
        original_excerpt="“Don’t—wait”",
        note="Старая разговорная заметка.",
        translated_excerpt="«Не жди — останься»",
    )
    current_res = BlockProcessingResult(
        block_id=b1.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="«Не жди — останься», — прошептала она.",
        language_notes=(current_note,),
    )
    current_dict = asdict(current_res)

    issue = CorrectionIssue(
        field_path="translated_text",
        source_quote="don't-wait",
        current_evidence=CurrentQuoteEvidence(
            field_path="translated_text",
            quote='"не жди - останься"',
        ),
        problem="Неточный перевод реплики",
        minimal_fix_reason="Уточнить смысл без лишних правок",
    )
    assert issue.valid_for(b1.text, current_dict) is True

    note_dict_with_meta = {
        "language_notes": [
            {
                "id": "ln-ch01-007",
                "block_id": b1.id,
                "source_block_id": "b1",
                "original_excerpt": "“Don’t—wait”",
                "note": "Старая разговорная заметка.",
                "translated_excerpt": None,
            }
        ]
    }
    assert CurrentQuoteEvidence("language_notes[0]", "ln-ch01-007").valid_for(note_dict_with_meta) is False
    assert CurrentQuoteEvidence("language_notes[0]", b1.id).valid_for(note_dict_with_meta) is False
    assert CurrentQuoteEvidence("language_notes[0]", "b1").valid_for(note_dict_with_meta) is False
    assert CurrentQuoteEvidence("language_notes[0]", "Don't-wait").valid_for(note_dict_with_meta) is True

    # 2. LanguageNote.id preservation in CorrectChapter._candidates matches normalized typography.
    plan = TreatmentPlan.of_treatments(
        section.id,
        ((b1.id, BlockTreatment.TRANSLATE), (b2.id, BlockTreatment.READ_AS_IS)),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-typo",
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
    res2 = BlockProcessingResult(block_id=b2.id, treatment=BlockTreatment.READ_AS_IS)
    inputs = CorrectionInputs(
        document=doc,
        section=section,
        analysis=analysis,
        run=SimpleNamespace(id="run-1", glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(id="cond-1", payload={"schema_version": 1, "versions": {"validators": 3}}),
        snapshot=snapshot,
        blocks=section.blocks,
        results={b1.id: current_res, b2.id: res2},
        bindings={b1.id: None, b2.id: None},
    )
    fake_tx = SimpleNamespace(work=SimpleNamespace(list_correction_candidates=lambda *a, **kw: ()))
    validator = make_validator()
    handler = CorrectChapter(unit_of_work=None, llm_client=None, snapshot=snapshot, validator=validator)

    proposed_b1 = ProposedBlockResponse(
        block_id=b1.id,
        translated_text="«Не жди», — прошептала она.",
        glossary_hints=(),
        language_notes=(
            LanguageNoteResponse(
                source_block_id=b1.id,
                original_excerpt="whispered",
                note="Новая заметка о глаголе.",
            ),
            LanguageNoteResponse(
                source_block_id=b1.id,
                original_excerpt="Don't-wait",
                note="Обновлённая разговорная заметка.",
            ),
        ),
        issues=(issue,),
    )
    candidates = handler._candidates(
        fake_tx,
        inputs,
        ChapterCorrectionV2Response((proposed_b1, UnchangedBlockResponse(b2.id))),
    )
    assert len(candidates[0].language_notes) == 2
    assert candidates[0].language_notes[1].id == "ln-ch01-007"
    assert candidates[0].language_notes[0].id != "ln-ch01-007"

    # 3. fragment_assistances overlap check in CorrectChapter._candidates uses find_quote_span.
    eh_b1_anchor = SourceAnchor(block_id=b1.id, start=1, end=11, quote="Don’t—wait")
    eh_b1_diff = Difficulty(
        id="df-" + "8" * 64,
        anchors=(eh_b1_anchor,),
        owner_block_id=b1.id,
        kind="idiom",
        priority="critical",
        help_kind="explanation",
        required_points=(RequiredPoint(id="p1", description="do not wait"),),
    )
    eh_anchor = SourceAnchor(block_id=b2.id, start=1, end=8, quote="Hold—on")
    eh_diff = Difficulty(
        id="df-" + "9" * 64,
        anchors=(eh_anchor,),
        owner_block_id=b2.id,
        kind="idiom",
        priority="critical",
        help_kind="explanation",
        required_points=(RequiredPoint(id="p1", description="wait briefly"),),
    )
    eh_content_plan = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(eh_b1_diff, eh_diff),
    )
    eh_plan = TreatmentPlan.of_treatments(
        section.id,
        ((b1.id, BlockTreatment.GLOSSARY_ONLY), (b2.id, BlockTreatment.READ_AS_IS)),
    )
    eh_analysis = ChapterAnalysis.create(
        id="analysis-eh",
        section_id=section.id,
        treatment_plan=eh_plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(b1.id, b2.id),
        content_plan=eh_content_plan,
    )
    eh_assistance = FragmentAssistance(
        difficulty_id=eh_diff.id,
        point_ids=("p1",),
        kind="sentence_rephrase",
        text="Wait for a short moment.",
        anchors=(eh_anchor,),
    )
    eh_b1_assistance = FragmentAssistance(
        difficulty_id=eh_b1_diff.id,
        point_ids=("p1",),
        kind="sentence_rephrase",
        text="Do not wait.",
        anchors=(eh_b1_anchor,),
    )
    eh_res1 = BlockProcessingResult(
        block_id=b1.id,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(GlossaryHint(original="whispered", translation="прошептала"),),
        fragment_assistances=(eh_b1_assistance,),
        content_evidence=ContentEvidence(plan_sha256=eh_content_plan.sha256, version=3),
    )
    eh_res2 = BlockProcessingResult(
        block_id=b2.id,
        treatment=BlockTreatment.READ_AS_IS,
        fragment_assistances=(eh_assistance,),
        content_evidence=ContentEvidence(plan_sha256=eh_content_plan.sha256, version=3),
    )
    v1_snapshot = CorrectionSnapshot(
        model="gpt-test",
        route={"provider": "openai_compatible", "base_url": "https://example.test/v1", "proxy": "", "route_id": "r1"},
        parameters=dict(snapshot.parameters),
        prompt="Correct.",
        versions=CORRECTION_VERSIONS,
    )
    eh_inputs = CorrectionInputs(
        document=doc,
        section=section,
        analysis=eh_analysis,
        run=SimpleNamespace(id="run-eh", glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(
            id="cond-eh",
            payload={"schema_version": 1, "english_help": True, "versions": {"validators": 3}},
        ),
        snapshot=v1_snapshot,
        blocks=section.blocks,
        results={b1.id: eh_res1, b2.id: eh_res2},
        bindings={b1.id: None, b2.id: None},
    )
    eh_handler = CorrectChapter(unit_of_work=None, llm_client=None, snapshot=v1_snapshot, validator=validator)

    with pytest.raises(ResultValidationError, match="дублирует защищённую помощь к фрагменту"):
        eh_handler._candidates(
            fake_tx,
            eh_inputs,
            ChapterCorrectionResponse(
                blocks=(
                    UnchangedBlockResponse(b1.id),
                    CorrectedBlockResponse(
                        block_id=b2.id,
                        translated_text=None,
                        glossary_hints=(),
                        language_notes=(
                            LanguageNoteResponse(
                                source_block_id=b2.id,
                                original_excerpt="hold-on",
                                note="Подожди.",
                            ),
                        ),
                    ),
                )
            ),
        )

    with pytest.raises(ResultValidationError, match="дублирует защищённую помощь к фрагменту"):
        eh_handler._candidates(
            fake_tx,
            eh_inputs,
            ChapterCorrectionResponse(
                blocks=(
                    CorrectedBlockResponse(
                        block_id=b1.id,
                        translated_text=None,
                        glossary_hints=(
                            GlossaryHintResponse(original="don't-wait", translation="не жди"),
                        ),
                        language_notes=(),
                    ),
                    UnchangedBlockResponse(b2.id),
                )
            ),
        )

    non_overlapping = eh_handler._candidates(
        fake_tx,
        eh_inputs,
        ChapterCorrectionResponse(
            blocks=(
                CorrectedBlockResponse(
                    block_id=b1.id,
                    translated_text=None,
                    glossary_hints=(
                        GlossaryHintResponse(original="whispered", translation="шепнула"),
                    ),
                    language_notes=(),
                ),
                UnchangedBlockResponse(b2.id),
            )
        ),
    )
    assert non_overlapping[0].glossary_hints[0].translation == "шепнула"


def test_materialize_for_rejects_null_translation_without_list_edits_on_translate_block():
    from linguary.application.correction_dto import (
        ChapterCorrectionV2Response,
        CorrectionIssue,
        CurrentQuoteEvidence,
        ProposedBlockResponse,
    )
    from linguary.application.llm_dto import (
        GlossaryHintResponse,
        LLMPayloadError,
        LanguageNoteResponse,
    )
    from linguary.domain import GlossaryHint, LanguageNote

    b1 = "block-1"
    b2 = "block-2"
    previous_translate = BlockProcessingResult(
        block_id=b1,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Старый перевод предложения.",
        glossary_hints=(GlossaryHint(original="Lantern", translation="фонарь"),),
        language_notes=(
            LanguageNote(
                id="ln-ch01-001",
                block_id=b1,
                original_excerpt="Lantern",
                note="Старая заметка.",
            ),
        ),
    )
    note_issue = CorrectionIssue(
        field_path="language_notes[0].note",
        source_quote="Lantern",
        current_evidence=CurrentQuoteEvidence(field_path="language_notes[0].note", quote="Старая"),
        problem="Неточная заметка",
        minimal_fix_reason="Уточнить заметку",
    )
    translation_issue = CorrectionIssue(
        field_path="translated_text",
        source_quote="Lantern",
        current_evidence=CurrentQuoteEvidence(field_path="translated_text", quote="Старый"),
        problem="Неточный перевод",
        minimal_fix_reason="Исправить слово",
    )

    # 1. Rejects null translated_text when both glossary_hints and language_notes are None on TRANSLATE block.
    all_null_proposal = ProposedBlockResponse(
        block_id=b1,
        translated_text=None,
        glossary_hints=None,
        language_notes=None,
        issues=(note_issue,),
    )
    with pytest.raises(LLMPayloadError):
        all_null_proposal.materialize_for(previous_translate)
    with pytest.raises(LLMPayloadError):
        ChapterCorrectionV2Response((all_null_proposal,)).materialize_for({b1: previous_translate})

    # 2. Rejects null translated_text when any issue targets translated_text, even if list edits are provided.
    null_with_translation_issue = ProposedBlockResponse(
        block_id=b1,
        translated_text=None,
        glossary_hints=(GlossaryHintResponse(original="Lantern", translation="светильник"),),
        language_notes=(),
        issues=(translation_issue,),
    )
    with pytest.raises(LLMPayloadError):
        null_with_translation_issue.materialize_for(previous_translate)

    # 3. Allows null translated_text on TRANSLATE block when editing notes/hints only and issues do not target translated_text.
    valid_list_only_proposal = ProposedBlockResponse(
        block_id=b1,
        translated_text=None,
        glossary_hints=None,
        language_notes=(
            LanguageNoteResponse(
                source_block_id=b1,
                original_excerpt="Lantern",
                note="Обновлённая заметка.",
            ),
        ),
        issues=(note_issue,),
    )
    materialized = valid_list_only_proposal.materialize_for(previous_translate)
    assert materialized.translated_text == "Старый перевод предложения."
    assert materialized.glossary_hints == (
        GlossaryHintResponse(original="Lantern", translation="фонарь", note=None),
    )
    assert len(materialized.language_notes) == 1
    assert materialized.language_notes[0].note == "Обновлённая заметка."

    # 4. Non-translate blocks keep translated_text=None during materialization.
    previous_glossary_only = BlockProcessingResult(
        block_id=b2,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        translated_text=None,
        glossary_hints=(GlossaryHint(original="Lantern", translation="фонарь"),),
    )
    glossary_only_proposal = ProposedBlockResponse(
        block_id=b2,
        translated_text=None,
        glossary_hints=(GlossaryHintResponse(original="Lantern", translation="светильник"),),
        language_notes=None,
        issues=(
            CorrectionIssue(
                field_path="glossary_hints[0].translation",
                source_quote="Lantern",
                current_evidence=CurrentQuoteEvidence(
                    field_path="glossary_hints[0].translation",
                    quote="фонарь",
                ),
                problem="Уточнение подсказки",
                minimal_fix_reason="Заменить перевод",
            ),
        ),
    )
    materialized_go = glossary_only_proposal.materialize_for(previous_glossary_only)
    assert materialized_go.translated_text is None
    assert materialized_go.language_notes == ()


