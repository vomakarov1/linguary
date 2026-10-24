"""Architectural tests: dependency direction between `linguary` layers.

Layers depend inward only: `domain` ← `application` ← `adapters` / `web` / `cli`.
`domain` and `application` additionally do not depend on third-party libraries, nor on
standard library modules providing access to filesystem, network, processes, or database,
and do not read files directly.
"""

from __future__ import annotations

import pytest
from import_rules import (
    ADAPTERS_RULE,
    APPLICATION_RULE,
    COMPOSITION_RULE,
    DOMAIN_RULE,
    PACKAGE_ROOT,
    WEB_RULE,
    LayerRule,
    SourceModule,
    check_layer,
    check_module,
    format_violations,
    iter_public_modules,
)

LAYER_PACKAGES = ("domain", "application", "adapters", "web", "cli")

INNER_RULES = (DOMAIN_RULE, APPLICATION_RULE)

#: Outer layers: third-party libraries are permitted, but dependencies on peer
#: layers or inward dependencies from more inner layers are forbidden.
OUTER_RULES = (ADAPTERS_RULE, COMPOSITION_RULE, WEB_RULE)


def source_module(source: str) -> SourceModule:
    """Synthetic `domain` module for testing the rules themselves."""
    return SourceModule(module="linguary.domain.sample", package="linguary.domain", source=source)


def test_all_layer_packages_exist() -> None:
    """Package structure contains all application layers."""
    missing = [name for name in LAYER_PACKAGES if not (PACKAGE_ROOT / name / "__init__.py").is_file()]
    assert not missing, f"нет пакетов слоёв: {', '.join(missing)}"


@pytest.mark.parametrize("rule", INNER_RULES, ids=lambda rule: rule.layer)
def test_inner_layer_has_modules_to_check(rule: LayerRule) -> None:
    """An inner layer has modules other than package `__init__.py`.

    Without this check, import rules would stay green on an empty layer and cease
    to provide guarantees.
    """
    modules = iter_public_modules(rule.directory)
    assert modules, f"слой {rule.layer} пуст, проверять импорты не на чем"


@pytest.mark.parametrize("rule", INNER_RULES, ids=lambda rule: rule.layer)
def test_inner_layer_dependencies(rule: LayerRule) -> None:
    """`domain` and `application` do not depend on outer layers or external technologies."""
    violations = check_layer(rule)
    assert not violations, (
        f"нарушены границы слоя {rule.layer}:\n{format_violations(violations)}"
    )


def test_adapters_do_not_depend_on_web_and_cli() -> None:
    """Adapters do not depend on Web layer or CLI: dependency points in the opposite direction."""
    violations = check_layer(ADAPTERS_RULE)
    assert not violations, (
        f"нарушены границы слоя adapters:\n{format_violations(violations)}"
    )


@pytest.mark.parametrize("rule", OUTER_RULES, ids=lambda rule: rule.layer)
def test_outer_layer_dependencies(rule: LayerRule) -> None:
    """Outer layers depend inward only: `composition` and `web` know nothing about CLI.

    Composition root is assembled once in `composition`, and views consume it from there:
    Web UI does not assemble adapters itself and does not depend on CLI.
    """
    violations = check_layer(rule)
    assert not violations, (
        f"нарушены границы слоя {rule.layer}:\n{format_violations(violations)}"
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("import sqlalchemy\n", id="import-third-party"),
        pytest.param("from lxml import etree\n", id="from-import-third-party"),
        pytest.param("import linguary.adapters.db\n", id="import-outer-layer"),
        pytest.param("from linguary.web import app\n", id="from-import-outer-layer"),
        pytest.param("from pathlib import Path\n", id="import-filesystem-stdlib"),
        pytest.param("import os.path\n", id="import-infrastructure-stdlib"),
        pytest.param(
            "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    import sqlalchemy\n",
            id="type-checking-import",
        ),
        pytest.param("data = open('book.epub').read()\n", id="filesystem-call"),
    ],
)
def test_rules_detect_violation(source: str) -> None:
    """Rules actually catch forbidden code rather than staying silent on any input."""
    violations = check_module(source_module(source), DOMAIN_RULE)
    assert violations, f"нарушение не обнаружено в исходнике:\n{source}"


@pytest.mark.parametrize(
    ("rule", "package", "source"),
    [
        pytest.param(
            WEB_RULE,
            "linguary.web",
            "from linguary.cli import main\n",
            id="web-imports-cli",
        ),
        pytest.param(
            COMPOSITION_RULE,
            "linguary.composition",
            "from linguary.web.app import create_app\n",
            id="composition-imports-web",
        ),
    ],
)
def test_outer_rules_detect_dependency_on_a_sibling_layer(
    rule: LayerRule, package: str, source: str
) -> None:
    """An outer layer cannot depend on a peer view of the same application."""
    module = SourceModule(module=f"{package}.sample", package=package, source=source)

    violations = check_module(module, rule)

    assert violations, f"нарушение не обнаружено в исходнике:\n{source}"


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("from dataclasses import dataclass\n", id="stdlib"),
        pytest.param("from linguary.domain.errors import DomainError\n", id="own-layer"),
        pytest.param("from .errors import DomainError\n", id="relative-own-layer"),
    ],
)
def test_rules_allow_permitted_imports(source: str) -> None:
    """Permitted imports are not treated as violations by the rules."""
    violations = check_module(source_module(source), DOMAIN_RULE)
    assert not violations, f"ложное нарушение:\n{format_violations(violations)}"
