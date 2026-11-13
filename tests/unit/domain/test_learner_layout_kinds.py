"""Learner difficulty kinds and the default glossary-candidate description."""

from __future__ import annotations

import pytest

from linguary.domain.content_plan import DIFFICULTY_KINDS, Difficulty, RequiredPoint, SourceAnchor
from linguary.domain.errors import DomainError
from linguary.domain.glossary import ConceptType, GlossaryCandidate

_LEARNER_KINDS = frozenset({
    "wordplay",
    "semantic_relation",
    "grammar",
    "cultural_reference",
    "idiom",
    "vocabulary",
    "false_friend",
    "speech_style",
    "parody",
})


def _difficulty(kind: str) -> Difficulty:
    return Difficulty(
        "d",
        (SourceAnchor("a", 0, 4, "echo"),),
        "a",
        kind,
        "critical",
        "explanation",
        (RequiredPoint("p", "Explain the difficulty"),),
    )


def test_difficulty_accepts_false_friend_speech_style_parody() -> None:
    """False friends, speech style, and parody are accepted difficulty kinds."""
    assert DIFFICULTY_KINDS == _LEARNER_KINDS
    for kind in ("false_friend", "speech_style", "parody"):
        assert _difficulty(kind).kind == kind


def test_unknown_kind_still_rejected() -> None:
    """A kind outside the closed set is still a domain error."""
    with pytest.raises(DomainError, match="неизвестный вид трудности"):
        _difficulty("pun")


def test_candidate_description_defaults_empty() -> None:
    """A glossary candidate starts with an empty description."""
    candidate = GlossaryCandidate(
        id="cand-1",
        analysis_id="analysis-1",
        source_text="doublethink",
        type=ConceptType.TERM,
        suggested_translation="двоемыслие",
        confidence=0.9,
        model_flagged_review=False,
        requires_review=False,
    )
    assert candidate.description == ""
