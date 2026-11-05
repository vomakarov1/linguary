"""Both translation routes receive source-grounded context without claiming quality."""

import hashlib
import json
from pathlib import Path

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.llm import FakeReply
from correction_fakes import CorrectionAwareFakeLLMClient as FakeLLMClient
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.composition import build_application, drain_tasks
from linguary.domain import LLMCallStage, TaskStatus
from test_book_conditions import store_imported


FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "learner_quality" / "translation_context_cases.json"


def load_cases():
    return json.loads(FIXTURE.read_text())


class ContextReplyClient(FakeLLMClient):
    """Return structurally valid replies for each request's exact target aliases."""

    async def generate(self, request):
        if request.stage == LLMCallStage.CHAPTER_ANALYSIS:
            payload = {
                "translation_instructions": "Сохраняй точные события и детали оригинала.",
                "chapter_summary": "A scene with a locally resolved reference.",
                "glossary_candidates": [],
                "treatment_plan": [
                    {"source_block_id": f"b{index}", "treatment": "TRANSLATE"}
                    for index, _ in enumerate(request.ordered_block_ids, 1)
                ],
                "before_you_read": [],
            }
        else:
            payload = {
                "blocks": [
                    {
                        "source_block_id": f"b{index}",
                        "treatment": "TRANSLATE",
                        "translated_text": f"Полный перевод блока {index} сохраняет детали сцены.",
                        "glossary_hints": [],
                    }
                    for index, _ in enumerate(request.ordered_block_ids, 1)
                ],
                "language_notes": [],
            }
        self.default = FakeReply.of_payload(payload)
        return await super().generate(request)


def make_context_app(tmp_path, *, client=None, max_attempts=1):
    settings = AppSettings(
        llm=LLMSettings(
            model="context-fixture-model",
            api_key="synthetic-secret",
            proxy_mode="direct",
            max_attempts=max_attempts,
            max_unit_source_chars=1,
        ),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
    )
    return build_application(settings=settings, llm_client=client or ContextReplyClient())


async def run_cases(app, cases):
    def chapter(case):
        before = next(item["original"] for item in case["context"] if item["position"] == "before")
        after = next(item["original"] for item in case["context"] if item["position"] == "after")
        return before, case["target_original"], after

    chapters = tuple(chapter(case) for case in cases)
    document = store_imported(app, chapters)
    await app.analysis.start("book-1")
    analyses = await drain_tasks(app)
    assert analyses and all(task.status is TaskStatus.SUCCEEDED for task in analyses)
    run = app.start_processing()(StartProcessingCommand("book-1"))
    await app.processing.start("book-1")
    processing = await drain_tasks(app)
    assert processing and all(task.status is TaskStatus.SUCCEEDED for task in processing)
    targets = tuple(section.blocks[1].id for section in document.sections)
    for target in targets:
        await app.regeneration.start("book-1", target)
    regeneration = await drain_tasks(app)
    assert regeneration and all(task.status is TaskStatus.SUCCEEDED for task in regeneration)
    return document, run, targets


def requests_for(client, stage):
    return [request for request in client.requests if request.stage == stage]


async def test_brief_cases_reach_both_translation_routes(tmp_path):
    cases = load_cases()["cases"]
    assert load_cases()["semantic_correctness"] == "not_evaluated"
    client = ContextReplyClient()
    app = make_context_app(tmp_path, client=client)
    try:
        document, _, targets = await run_cases(app, cases)
        standard = requests_for(client, LLMCallStage.TRANSLATION_UNIT)
        repeated = requests_for(client, LLMCallStage.BLOCK_REGENERATION)
        assert len(standard) == len(cases) * 3
        assert len(repeated) == len(cases)
        standard_by_target = {request.ordered_block_ids[0]: request for request in standard}
        repeated_by_target = {request.ordered_block_ids[0]: request for request in repeated}
        for index, (case, target) in enumerate(zip(cases, targets, strict=True)):
            chapter = document.sections[index]
            block_ids = tuple(block.id for block in chapter.blocks)
            assert standard_by_target[target].ordered_block_ids == (target,)
            assert repeated_by_target[target].ordered_block_ids == (target,)
            for request in (standard_by_target[target], repeated_by_target[target]):
                assert request.call_context.policy_version == "translation-context-v2"
                assert request.call_context.prompt_hash == hashlib.sha256(
                    request.instructions.encode()
                ).hexdigest()
                assert case["target_original"] in request.input_text
                assert all(item["original"] in request.input_text for item in case["context"])
                assert "Политика translation-context-v2" in request.instructions
            assert target == block_ids[1]
    finally:
        await app.aclose()


async def test_neighbour_output_is_rejected(tmp_path):
    class NeighbourReturningClient(ContextReplyClient):
        async def generate(self, request):
            if request.stage == LLMCallStage.TRANSLATION_UNIT:
                self.default = FakeReply.of_payload({
                    "blocks": [
                        {"source_block_id": "b1", "treatment": "TRANSLATE",
                         "translated_text": "Это полный перевод целевого блока сцены.", "glossary_hints": []},
                        {"source_block_id": "b2", "treatment": "TRANSLATE",
                         "translated_text": "Сосед не должен вернуться в ответ.", "glossary_hints": []},
                    ],
                    "language_notes": [],
                })
                return await FakeLLMClient.generate(self, request)
            return await super().generate(request)

    client = NeighbourReturningClient()
    app = make_context_app(tmp_path, client=client, max_attempts=1)
    try:
        document = store_imported(app, (("Before scene.", "The target happened.", "After scene."),))
        await app.analysis.start("book-1")
        assert all(task.status is TaskStatus.SUCCEEDED for task in await drain_tasks(app))
        app.start_processing()(StartProcessingCommand("book-1"))
        await app.processing.start("book-1")
        completed = await drain_tasks(app)
        assert completed[0].status is TaskStatus.FAILED
        with app.unit_of_work.transaction() as transaction:
            assert transaction.results.list_for_run(transaction.runs.active_for_book("book-1").id) == ()
        assert client.call_count >= 2
        assert document.sections[0].blocks
    finally:
        await app.aclose()


async def test_old_analysis_keeps_optional_facts_absent(tmp_path):
    cases = load_cases()["cases"]
    case = next(item for item in cases if item["id"] == "it")
    client = ContextReplyClient()
    app = make_context_app(tmp_path, client=client)
    try:
        await run_cases(app, (case,))
        with app.unit_of_work.transaction() as transaction:
            analysis = transaction.analyses.get_for_section("book-1", "section-1")
            assert analysis.content_plan is None
        for request in requests_for(client, LLMCallStage.TRANSLATION_UNIT) + requests_for(
            client, LLMCallStage.BLOCK_REGENERATION
        ):
            assert "context_facts" not in request.instructions
            assert '"context_facts"' not in request.input_text
    finally:
        await app.aclose()


async def test_contextual_instructions_preserve_idiom_negation_and_details(tmp_path):
    cases = load_cases()["cases"]
    case = next(item for item in cases if item["id"] == "idiom")
    client = ContextReplyClient()
    app = make_context_app(tmp_path, client=client)
    try:
        _, _, (target_id,) = await run_cases(app, (case,))
        requests = requests_for(client, LLMCallStage.TRANSLATION_UNIT) + requests_for(
            client, LLMCallStage.BLOCK_REGENERATION
        )
        target_requests = [request for request in requests
                           if len(request.ordered_block_ids) == 1
                           and request.ordered_block_ids == (target_id,)]
        assert len(target_requests) == 2
        for request in target_requests:
            assert "действия, отношения, отрицания" in request.instructions
            assert "Смысл идиом и устойчивых выражений передавай естественно" in request.instructions
            assert "Не сокращай и не добавляй" in request.instructions
            assert "He meant well, but he did not open the door." in request.input_text
    finally:
        await app.aclose()
