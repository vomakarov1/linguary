"""Integration tests for non-blocking awaiting_human correction candidates and localized prompts."""

from __future__ import annotations

from dataclasses import asdict
import json
from uuid import uuid4

from fastapi import status
from fastapi.testclient import TestClient

from linguary.application.block_processing import complete_processing_in_transaction
from linguary.application.correction import (
    correction_coverage_complete,
    correction_input_fingerprint,
    correction_inputs,
    reconcile_corrections,
)
from linguary.application import quality_context
from linguary.composition import Application
from linguary.composition.correction_runtime import (
    CORRECTION_PROMPT,
    CORRECTION_PROMPT_V2,
    CORRECTION_REVIEW_PROMPT,
    correction_snapshot,
)
from linguary.domain import BookStatus, TaskType
from linguary.domain.content_plan import canonical_sha
from linguary.domain.correction import CorrectionCandidate, CorrectionCheck
from linguary.domain.llm_log import CallContext
from linguary.domain.recovery import CallAttempt
from web_scenario import (
    SOURCE_TEXT,
    TRANSLATED_TEXT,
    continue_legacy,
    start_run,
    store_book,
    translated_result,
)


def _seed_book_with_awaiting_human_candidate(
    application: Application,
) -> tuple[str, str, CorrectionCandidate]:
    scenario = store_book(
        application,
        status=BookStatus.PROCESSING,
        chapters=((SOURCE_TEXT, "She closed the window quietly."),),
    )
    b0, b1 = scenario.block_ids(0)
    r0 = translated_result(b0)
    r1 = translated_result(b1)
    conditions = continue_legacy(application, scenario.book_id)
    start_run(application, scenario, results=(r0, r1))
    snapshot = correction_snapshot(application.settings)
    section_id = scenario.section(0).id

    with application.unit_of_work.transaction() as tx:
        tx.work.bind(
            "run",
            scenario.run_id,
            book_id=scenario.book_id,
            conditions_id=conditions.id,
            payload={"schema_version": 1, "correction": snapshot.to_payload()},
        )
        for bid, res in ((b0, r0), (b1, r1)):
            art_id = str(uuid4())
            att_id = str(uuid4())
            trans_ctx = CallContext(
                conditions_id=conditions.id,
                stage="translation_unit",
                scope_id=section_id,
                ordered_block_ids=(bid,),
                attempt_id=att_id,
                generation=1,
                assigned_model=snapshot.model,
                prompt_hash="a" * 64,
                run_id=scenario.run_id,
                task_id="task-trans-1",
            )
            tx.work.add_attempt(
                CallAttempt(
                    att_id,
                    scenario.book_id,
                    conditions_id=conditions.id,
                    task_id="task-trans-1",
                    run_id=scenario.run_id,
                    ordinal=1,
                    context=trans_ctx.to_dict(),
                    state="accepted",
                )
            )
            tx.work.archive_artifacts(
                [
                    {
                        "id": art_id,
                        "book_id": scenario.book_id,
                        "conditions_id": conditions.id,
                        "run_id": scenario.run_id,
                        "attempt_id": att_id,
                        "kind": "block_result",
                        "scope_id": bid,
                        "body": {"schema_version": 1, "result": asdict(res)},
                    }
                ]
            )
            tx.work.bind(
                "result",
                f"{scenario.run_id}:{bid}",
                book_id=scenario.book_id,
                conditions_id=conditions.id,
                attempt_id=att_id,
                artifact_revision_id=art_id,
            )

        inputs = correction_inputs(tx, scenario.book_id, section_id)
        for bid in (b0, b1):
            tx.work.observe_correction_input(
                scenario.run_id,
                bid,
                correction_input_fingerprint(inputs, bid),
            )
        inputs = correction_inputs(tx, scenario.book_id, section_id)

        prop_attempt_id = str(uuid4())
        prop_ctx = CallContext(
            conditions_id=conditions.id,
            stage="chapter_correction",
            scope_id=section_id,
            ordered_block_ids=(b0, b1),
            attempt_id=prop_attempt_id,
            generation=1,
            assigned_model=snapshot.model,
            prompt_hash="c" * 64,
            run_id=scenario.run_id,
            task_id="task-corr-1",
            policy_version=snapshot.policy_version,
        )
        tx.work.add_attempt(
            CallAttempt(
                prop_attempt_id,
                scenario.book_id,
                conditions_id=conditions.id,
                task_id="task-corr-1",
                run_id=scenario.run_id,
                ordinal=1,
                context=prop_ctx.to_dict(),
                state="accepted",
            )
        )

        proposed_r1 = translated_result(b1)
        proposed_r1 = type(proposed_r1)(
            block_id=b1,
            treatment=proposed_r1.treatment,
            translated_text="Она тихо закрыла окно.",
            glossary_hints=(),
            language_notes=(),
            warnings=(),
        )
        proposal_payload = {
            "block_id": b1,
            "translated_text": proposed_r1.translated_text,
            "glossary_hints": [],
            "language_notes": [],
            "issues": [
                {
                    "field_path": "translated_text",
                    "source_quote": "closed the window",
                    "current_evidence": {
                        "field_path": "translated_text",
                        "quote": TRANSLATED_TEXT,
                        "kind": "quote",
                    },
                    "problem": "Неверный перевод абзаца.",
                    "minimal_fix_reason": "Исправляет перевод второго абзаца.",
                }
            ],
            "status": "proposed",
        }
        immutable = json.loads(
            json.dumps(
                {
                    "schema_version": 1,
                    "book_id": scenario.book_id,
                    "run_id": scenario.run_id,
                    "conditions_id": conditions.id,
                    "section_id": section_id,
                    "block_id": b1,
                    "input_sha256": inputs.checksums[b1],
                    "input_generation": inputs.input_generations[b1],
                    "group_id": prop_attempt_id,
                    "proposal_attempt_id": prop_attempt_id,
                    "context_sha256": inputs.context_sha256,
                    "source_binding": inputs.bindings[b1],
                    "original": {"text": "She closed the window quietly."},
                    "current_result": asdict(r1),
                    "result": asdict(proposed_r1),
                    "proposal": proposal_payload,
                    "evidence_valid": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        candidate = CorrectionCandidate(
            str(uuid4()),
            scenario.book_id,
            scenario.run_id,
            conditions.id,
            section_id,
            b1,
            inputs.checksums[b1],
            inputs.input_generations[b1],
            canonical_sha(immutable),
            immutable,
            prop_attempt_id,
            prop_attempt_id,
        )
        tx.work.add_correction_candidates([candidate])
        tx.work.bind(
            "correction_group",
            prop_attempt_id,
            book_id=scenario.book_id,
            conditions_id=conditions.id,
            attempt_id=prop_attempt_id,
            payload={
                "schema_version": 1,
                "run_id": scenario.run_id,
                "section_id": section_id,
                "manifest": inputs.manifest,
                "block_ids": [b0, b1],
                "unchanged_block_ids": [b0],
                "candidate_ids": [candidate.candidate_id],
            },
        )
        rev_attempt_id = str(uuid4())
        rev_ctx = CallContext(
            conditions_id=conditions.id,
            stage="chapter_correction_review",
            scope_id=section_id,
            ordered_block_ids=(b1,),
            attempt_id=rev_attempt_id,
            generation=1,
            assigned_model=snapshot.model,
            prompt_hash="d" * 64,
            run_id=scenario.run_id,
            task_id="task-corr-1",
            policy_version=snapshot.policy_version,
        )
        tx.work.add_attempt(
            CallAttempt(
                rev_attempt_id,
                scenario.book_id,
                conditions_id=conditions.id,
                task_id="task-corr-1",
                run_id=scenario.run_id,
                ordinal=1,
                context=rev_ctx.to_dict(),
                state="accepted",
            )
        )
        tx.work.record_correction_verdict(
            candidate.candidate_id,
            proposal_sha256=candidate.proposal_sha256,
            input_sha256=candidate.input_sha256,
            review_attempt_id=rev_attempt_id,
            review={
                "candidate_id": candidate.candidate_id,
                "proposal_sha256": candidate.proposal_sha256,
                "verdict": "needs_human",
                "error_supported": True,
                "evidence_valid": True,
                "minimal": False,
                "no_semantic_loss": True,
                "reason": "Предлагает полный пересказ абзаца.",
            },
            state="awaiting_human",
        )
        tx.work.replace_correction_checks(
            scenario.run_id,
            section_id,
            (
                CorrectionCheck(
                    scenario.run_id,
                    b0,
                    section_id,
                    inputs.checksums[b0],
                    prop_attempt_id,
                    "unchanged",
                ),
            ),
        )
        complete_processing_in_transaction(
            tx,
            book_id=scenario.book_id,
            finished_task_id=None,
            task_types={
                TaskType.PROCESS_TRANSLATION_UNIT,
                TaskType.REGENERATE_BLOCK,
                TaskType.CHAPTER_CORRECTION,
            },
        )

    return scenario.book_id, b1, candidate


def test_awaiting_human_candidate_does_not_block_ready_to_export_or_epub_build(
    configured_application: Application,
    configured_client: TestClient,
) -> None:
    book_id, block_id, candidate = _seed_book_with_awaiting_human_candidate(
        configured_application
    )

    with configured_application.unit_of_work.transaction() as tx:
        assert correction_coverage_complete(tx, book_id) is True
        assert reconcile_corrections(tx, book_id) == ()
        book = tx.books.get(book_id)
        assert book is not None
        assert book.status is BookStatus.READY_TO_EXPORT

    result_page = configured_client.get(
        f"/books/{book_id}/result",
        headers={"Accept": "text/html"},
    )
    assert result_page.status_code == status.HTTP_200_OK
    html = result_page.text
    assert f"/books/{book_id}/correction-candidates/{candidate.candidate_id}/resolve" in html
    assert "EPUB нельзя собрать" not in html
    assert '<form method="post" action="/books/book-web-1/export" data-export-form>' in html
    export_form_part = html.split('data-export-form>', 1)[1].split("</form>", 1)[0]
    assert "disabled" not in export_form_part

    export_resp = configured_client.post(f"/books/{book_id}/export")
    assert export_resp.status_code == status.HTTP_202_ACCEPTED
    assert export_resp.json()["task_type"] == "export_book"

    resolve_resp = configured_client.post(
        f"/books/{book_id}/correction-candidates/{candidate.candidate_id}/resolve",
        json={
            "proposal_sha256": candidate.proposal_sha256,
            "input_sha256": candidate.input_sha256,
            "action": "accept",
            "reason": "Перевод второго абзаца точен.",
        },
    )
    assert resolve_resp.status_code == status.HTTP_200_OK
    assert resolve_resp.json()["block_id"] == block_id


def test_russian_correction_and_quality_prompts_preserve_schema_contracts() -> None:
    prompts = (
        CORRECTION_PROMPT,
        CORRECTION_PROMPT_V2,
        CORRECTION_REVIEW_PROMPT,
        quality_context.INSTRUCTIONS,
    )
    for prompt in prompts:
        assert any("а" <= ch.lower() <= "я" for ch in prompt)
        assert not prompt.startswith(
            (
                "Review and correct",
                "Check the current results",
                "Evaluate each immutable",
                "Review the original",
            )
        )

    for token in (
        "block_id",
        "unchanged",
        "corrected",
        "translated_text",
        "glossary_hints",
        "language_notes",
        "TRANSLATE",
        "GLOSSARY_ONLY",
        "READ_AS_IS",
    ):
        assert token in CORRECTION_PROMPT

    for token in (
        "block_id",
        "unchanged",
        "proposed",
        "translated_text",
        "glossary_hints",
        "language_notes",
        "field_path",
        "source_quote",
        "current_evidence",
        "quote",
        "absence",
        "null",
        "empty_list",
        "empty_text",
        "missing_meaning",
        "meaning_missing",
        "glossary_hints.0.translation",
        "language_notes.0.note",
        "TRANSLATE",
        "GLOSSARY_ONLY",
        "READ_AS_IS",
    ):
        assert token in CORRECTION_PROMPT_V2

    for token in (
        "candidate_id",
        "proposal_sha256",
        "confirmed",
        "needs_human",
    ):
        assert token in CORRECTION_REVIEW_PROMPT
