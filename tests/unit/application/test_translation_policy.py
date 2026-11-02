"""Policy selection never invents a version for historical work."""

import inspect

import pytest

from linguary.application.ports.persistence import WorkRepository
from linguary.application.translation_policy import (
    CURRENT_TRANSLATION_POLICY,
    TRANSLATION_POLICIES,
    TranslationPolicyUnavailable,
    require_policy,
)


@pytest.mark.parametrize("missing", [None, "", " "])
def test_missing_policy_never_means_current(missing):
    with pytest.raises(TranslationPolicyUnavailable):
        require_policy(missing)


def test_unknown_policy_fails_closed():
    with pytest.raises(TranslationPolicyUnavailable):
        require_policy("translation-context-future")
    assert set(TRANSLATION_POLICIES) == {
        "translation-context-legacy-v1", "translation-context-v2"
    }
    assert require_policy("translation-context-legacy-v1").version != CURRENT_TRANSLATION_POLICY


def test_context_limits_are_shared_by_both_translation_routes():
    chapter_policy = require_policy(CURRENT_TRANSLATION_POLICY)
    regeneration_policy = require_policy(CURRENT_TRANSLATION_POLICY)
    assert chapter_policy is regeneration_policy
    assert chapter_policy.neighbours_per_side == 2
    assert chapter_policy.max_neighbour_chars == 8000
    assert chapter_policy.capacity_retries_without_neighbours == 1
    with pytest.raises(TypeError):
        TRANSLATION_POLICIES["other"] = chapter_policy


def test_work_port_exposes_atomic_operation_contracts():
    contracts = {
        "restart_root": ("root_id", "expected_generation", "successor", "scopes", "task"),
        "list_active_roots": ("book_id", "run_id"),
        "pin_legacy_policy": ("root_id", "expected_generation", "proof"),
    }
    for method, parameters in contracts.items():
        signature = inspect.signature(getattr(WorkRepository, method))
        assert tuple(signature.parameters)[1:] == parameters
        assert signature.parameters[parameters[1]].kind == inspect.Parameter.KEYWORD_ONLY
    assert "provenance" in inspect.signature(WorkRepository.replace_result_binding).parameters
    assert "policy_version" in inspect.signature(WorkRepository.begin_recovery_attempt).parameters
    assert inspect.signature(WorkRepository.replace_result_binding).parameters["provenance"].default is inspect.Parameter.empty
    assert inspect.signature(WorkRepository.begin_recovery_attempt).parameters["policy_version"].default is inspect.Parameter.empty
