"""End-to-end tests for the CorrectChapterV2 pipeline with non-blocking awaiting_human."""

from __future__ import annotations

import json

import pytest
from test_learner_quality_pipeline import (
    BOOK,
    STORY,
    SceneFake,
    make_app,
    prepare,
    successful,
)

from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application import quality_context
from linguary.application.book_overview import book_result
from linguary.application.ports.llm import LLMResponseKind
from linguary.application.use_cases.resolve_correction_candidate import (
    ResolveCorrectionCommand,
)
from linguary.composition.correction_runtime import (
    CORRECTION_PROMPT,
    CORRECTION_PROMPT_V2,
    CORRECTION_REVIEW_PROMPT,
)
from linguary.domain import BookStatus


class MixedVerdictCorrectionFake(SceneFake):
    """Proposes two candidates (one confirmed, one needs_human) and leaves others unchanged."""

    def __init__(self, profile: str = "legacy-v1") -> None:
        super().__init__(profile=profile)
        self.correction_calls = 0
        self.review_calls = 0

    async def generate(self, request):
        if request.kind is LLMResponseKind.CHAPTER_CORRECTION:
            self.correction_calls += 1
            body = json.loads(request.input_text)
            blocks_out = []
            for index, item in enumerate(body["blocks"]):
                alias = item["block_id"]
                if index == 1:
                    blocks_out.append(
                        {
                            "block_id": alias,
                            "status": "proposed",
                            "translated_text": "Она подняла лампу. Она была не тяжелее коробки.",
                            "glossary_hints": [],
                            "language_notes": [],
                            "issues": [
                                {
                                    "field_path": "translated_text",
                                    "source_quote": "lifted the lamp",
                                    "current_evidence": {
                                        "kind": "quote",
                                        "field_path": "translated_text",
                                        "quote": "Она подняла лампу.",
                                    },
                                    "problem": "Уточнение местоимения во втором предложении.",
                                    "minimal_fix_reason": "Заменяет одно слово без потери смысла.",
                                }
                            ],
                        }
                    )
                elif index == 2:
                    blocks_out.append(
                        {
                            "block_id": alias,
                            "status": "proposed",
                            "translated_text": "У него были добрые намерения, однако дверь он не открыл.",
                            "glossary_hints": [],
                            "language_notes": [],
                            "issues": [
                                {
                                    "field_path": "translated_text",
                                    "source_quote": "He meant well",
                                    "current_evidence": {
                                        "kind": "quote",
                                        "field_path": "translated_text",
                                        "quote": "Он хотел как лучше",
                                    },
                                    "problem": "Альтернативная передача оборота.",
                                    "minimal_fix_reason": "Меняет формулировку первой части.",
                                }
                            ],
                        }
                    )
                else:
                    blocks_out.append({"block_id": alias, "status": "unchanged"})
            reply = FakeReply.of_payload({"blocks": blocks_out})
            response = await FakeLLMClient.always(reply).generate(request)
            self.requests.append(request)
            return response

        if request.kind is LLMResponseKind.CHAPTER_CORRECTION_REVIEW:
            self.review_calls += 1
            verdicts = []
            for idx, (cid, sha) in enumerate(request.correction_candidate_refs):
                if idx == 0:
                    verdicts.append(
                        {
                            "candidate_id": cid,
                            "proposal_sha256": sha,
                            "verdict": "confirmed",
                            "error_supported": True,
                            "evidence_valid": True,
                            "minimal": True,
                            "no_semantic_loss": True,
                            "reason": "Точное и минимальное исправление.",
                        }
                    )
                else:
                    verdicts.append(
                        {
                            "candidate_id": cid,
                            "proposal_sha256": sha,
                            "verdict": "needs_human",
                            "error_supported": False,
                            "evidence_valid": True,
                            "minimal": False,
                            "no_semantic_loss": True,
                            "reason": "Стилистическое предпочтение, решение за человеком.",
                        }
                    )
            reply = FakeReply.of_payload({"candidates": verdicts})
            response = await FakeLLMClient.always(reply).generate(request)
            self.requests.append(request)
            return response

        return await super().generate(request)


@pytest.mark.asyncio
async def test_awaiting_human_candidate_does_not_block_ready_to_export_or_epub_build(
    tmp_path,
) -> None:
    fake = MixedVerdictCorrectionFake(profile="legacy-v1")
    app = make_app(tmp_path / "data", fake, profile="legacy-v1")
    try:
        document, section, run, _, _ = await prepare(app, fake)
        await successful(app)

        assert fake.correction_calls == 1
        assert fake.review_calls == 1

        with app.unit_of_work.transaction() as tx:
            book = tx.books.get(BOOK)
            assert book is not None
            assert book.status is BookStatus.READY_TO_EXPORT
            results = {r.block_id: r for r in tx.results.list_for_run(run.id)}
            ordered_blocks = section.ordered_blocks
            assert (
                results[ordered_blocks[1].id].translated_text
                == "Она подняла лампу. Она была не тяжелее коробки."
            )
            assert (
                results[ordered_blocks[2].id].translated_text
                == STORY["translations"][2]
            )

        summary = book_result(app.unit_of_work, BOOK)
        assert summary.epub_ready is True
        assert summary.pending_corrections == 1

        await app.start_export(BOOK)
        await successful(app)
        assert app.paths.export_epub(BOOK).is_file()

        candidate_block = summary.chapters[0].blocks[2]
        assert len(candidate_block.correction_candidates) == 1
        candidate = candidate_block.correction_candidates[0]
        assert candidate["state"] == "awaiting_human"

        resolved_block_id = app.resolve_correction_candidate(
            ResolveCorrectionCommand(
                book_id=BOOK,
                candidate_id=candidate["candidate_id"],
                proposal_sha256=candidate["proposal_sha256"],
                input_sha256=candidate["input_sha256"],
                action="keep",
                reason="Текущий перевод естественнее.",
            )
        )
        assert resolved_block_id == ordered_blocks[2].id
        after_summary = book_result(app.unit_of_work, BOOK)
        assert after_summary.pending_corrections == 0
    finally:
        await app.aclose()


def test_russian_correction_and_quality_prompts_preserve_schema_contracts() -> None:
    for prompt in (
        CORRECTION_PROMPT,
        CORRECTION_PROMPT_V2,
        CORRECTION_REVIEW_PROMPT,
        quality_context.INSTRUCTIONS,
    ):
        assert any("а" <= ch.lower() <= "я" for ch in prompt)
        assert "block_id" in CORRECTION_PROMPT_V2
        assert "candidate_id" in CORRECTION_REVIEW_PROMPT
