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
