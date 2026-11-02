"""Operation policy identities preserve old serialized work without upgrading it."""

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from uuid import uuid4

import pytest

from linguary.domain.llm_log import CallContext
from linguary.domain.book_conditions import WorkStateError
from linguary.domain.recovery import RecoveryRoot

LEGACY = "translation-context-legacy-v1"
CURRENT = "translation-context-v2"


def make_root(**kwargs):
    return RecoveryRoot(str(uuid4()), "book", str(uuid4()), "original-key",
                        {"schema_version": 1, "content_plan_version": 3}, **kwargs)


def test_root_policy_is_not_content_plan_version():
    root = make_root(policy_version=CURRENT, operation_kind="regeneration")
    assert root.schema_version == 1
    assert root.manifest["content_plan_version"] == 3
    assert root.policy_version == CURRENT
    assert RecoveryRoot.from_dict(root.to_dict()) == root
    with pytest.raises(FrozenInstanceError):
        root.policy_version = LEGACY


def test_old_root_does_not_acquire_current_policy():
    root = make_root()
    historical = root.to_dict()
    for field in ("policy_version", "operation_kind", "superseded_by", "compatibility"):
        historical.pop(field)
    restored = RecoveryRoot.from_dict(historical)
    assert restored.policy_version is None
    assert restored.operation_kind == "translation"
    assert restored.superseded_by is None
    assert restored.compatibility is None
    assert restored.root_key == root.root_key
    assert restored.manifest == root.manifest


def test_root_compatibility_is_immutable_and_roundtrips():
    proof = {"schema_version": 1, "snapshot_hashes": ["original"]}
    root = make_root(policy_version=LEGACY, compatibility=proof,
                     superseded_by=str(uuid4()))
    proof["snapshot_hashes"].append("later")
    assert root.compatibility["snapshot_hashes"] == ("original",)
    assert RecoveryRoot.from_dict(root.to_dict()) == root


def test_translation_context_keeps_policy_and_actual_prompt_hash():
    sent_instruction = "Translate complete targets; preserve all details."
    context = CallContext(str(uuid4()), "translation_unit", "scope", ("b1",),
                          str(uuid4()), 0, "selected-model",
                          sha256(sent_instruction.encode()).hexdigest(),
                          root_id=str(uuid4()), policy_version=CURRENT)
    serialized = context.to_dict()
    assert serialized["policy_version"] == CURRENT
    assert serialized["prompt_hash"] == sha256(sent_instruction.encode()).hexdigest()
    assert serialized["schema_version"] == 1
    assert serialized["root_id"] == context.root_id
    repaired = replace(context, prompt_hash=sha256(b"repaired instruction").hexdigest())
    assert repaired.prompt_hash != context.prompt_hash
    assert repaired.policy_version == context.policy_version


def test_old_call_context_does_not_acquire_current_policy():
    context = CallContext(str(uuid4()), "chapter_analysis", "scope", (),
                          str(uuid4()), 0, "model", "a" * 64)
    assert context.policy_version is None
    assert context.to_dict()["policy_version"] is None


@pytest.mark.parametrize("stage", ["translation_unit", "block_regeneration"])
def test_explicit_translation_policy_requires_operation_root(stage):
    with pytest.raises(ValueError):
        CallContext(str(uuid4()), stage, "scope", ("b1",), str(uuid4()),
                    0, "model", "a" * 64, policy_version=CURRENT)


@pytest.mark.parametrize("changes", [
    {"operation_kind": "analysis"}, {"policy_version": ""},
    {"superseded_by": "invalid"}, {"compatibility": {"schema_version": 2}},
])
def test_invalid_operation_metadata_is_rejected(changes):
    with pytest.raises(WorkStateError):
        make_root(**changes)
