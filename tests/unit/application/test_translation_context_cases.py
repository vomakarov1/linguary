"""Portable source evidence and semantic expectations, not generated quality claims."""

import hashlib
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "learner_quality"


def load_cases():
    return json.loads((FIXTURES / "translation_context_cases.json").read_text())


def test_translation_cases_have_source_context_and_semantic_expectations():
    fixture = load_cases()
    assert fixture["schema_version"] == 1
    assert fixture["semantic_correctness"] == "not_evaluated"
    for case in fixture["cases"]:
        assert case["target_original"]
        assert case["context"]
        assert case["expected_facts"]
        assert case["forbidden_additions"]
        assert isinstance(case["expected_references"], dict)
        source = case["source"]
        assert source["kind"] in {"epub", "synthetic_fixture"}
        assert source["locator"]
        assert len(source["sha256"]) == 64
        assert source["target_sha256"] == hashlib.sha256(case["target_original"].encode()).hexdigest()
        for neighbour in case["context"]:
            assert neighbour["position"] in {"before", "after"}
            assert neighbour["original"]
            assert neighbour["locator"] != source["locator"]
            assert neighbour["sha256"] == hashlib.sha256(neighbour["original"].encode()).hexdigest()
        assert "actual_translation" not in case
        assert "quality_passed" not in case


def test_quality_cases_cover_all_brief_examples():
    cases = {case["id"]: case for case in load_cases()["cases"]}
    assert set(cases) == {"rattled", "owls", "long_robes", "it", "earmuffs", "idiom", "negation"}
    for case_id in ("rattled", "owls", "long_robes", "earmuffs"):
        source = cases[case_id]["source"]
        assert source["kind"] == "epub"
        assert source["file"] == "Harry_Potter_Philosophers_Stone_chapter_1_and_opening_chapter_2.epub"
        assert source["verification"] == "full_target_and_adjacent_paragraphs"
        assert source["locator"]["member"] == "OEBPS/hp01_ch001_en-gb.html"
    stories = json.loads((FIXTURES / "stories.json").read_text())
    for case_id, index in (("it", 1), ("idiom", 2), ("negation", 2)):
        case = cases[case_id]
        assert case["source"]["kind"] == "synthetic_fixture"
        assert case["source"]["file"] == "tests/fixtures/learner_quality/stories.json"
        assert case["target_original"] == stories["scene"]["blocks"][index]
    assert cases["it"]["expected_references"]["It"] == "lamp"
    assert cases["negation"]["expected_references"]["negated_action"] == "open the door"
