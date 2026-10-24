"""Exact anchors and immutable contract of the content plan."""
from dataclasses import FrozenInstanceError, replace

import pytest

from linguary.domain.content_plan import ContentPlan, Difficulty, SourceAnchor, RequiredPoint, ContentEvidence
from linguary.domain.errors import DomainError


def difficulty():
    return Difficulty("d", (SourceAnchor("a", 2, 6, "echo"), SourceAnchor("b", 0, 4, "echo")), "a", "wordplay", "critical", "explanation", (RequiredPoint("p", "Explain both meanings"),))


def test_repeated_quote_requires_exact_offset():
    SourceAnchor("a", 7, 11, "echo").validate({"a": "😀 echo echo"})
    with pytest.raises(DomainError):
        SourceAnchor("a", 3, 7, "echo").validate({"a": "😀 echo echo"})
    with pytest.raises(DomainError):
        SourceAnchor("a", 0, 4, "echo").validate({"other": "echo"})


def test_unknown_version_and_duplicate_point_rejected():
    with pytest.raises(DomainError):
        ContentPlan(version=3)
    with pytest.raises(DomainError):
        replace(difficulty(), required_points=(RequiredPoint("p", "one"), RequiredPoint("p", "two")))


def test_multiblock_difficulty_has_single_owner():
    d = difficulty()
    with pytest.raises(DomainError):
        replace(d, owner_block_id="unknown")
    with pytest.raises(FrozenInstanceError):
        d.owner_block_id = "b"
    assert d.canonical_id == replace(d, id="model-local").canonical_id
    assert d.canonical_id != replace(d, anchors=(SourceAnchor("different", 2, 6, "echo"),), owner_block_id="different").canonical_id


def test_plan_sha_changes_with_owner_and_obligations():
    plan = ContentPlan(difficulties=(difficulty(),))
    assert plan.sha256 != ContentPlan(difficulties=(replace(difficulty(), owner_block_id="b"),)).sha256
    evidence = ContentEvidence(plan.sha256)
    assert evidence.semantic_correctness == "not_evaluated"
    with pytest.raises(DomainError):
        ContentEvidence("wrong")
