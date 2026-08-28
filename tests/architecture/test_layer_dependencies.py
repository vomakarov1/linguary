"""Архитектурные тесты: направление зависимостей между слоями `linguary`.

Слои зависят только внутрь: `domain` ← `application` ← `adapters` / `web` / `cli`.
`domain` и `application` дополнительно не зависят от сторонних библиотек, от модулей
стандартной библиотеки, дающих доступ к файловой системе, сети, процессам и БД, и не
читают файлы напрямую.
"""

from __future__ import annotations

import pytest
from import_rules import (
    ADAPTERS_RULE,
    APPLICATION_RULE,
    DOMAIN_RULE,
    PACKAGE_ROOT,
    LayerRule,
    SourceModule,
    check_layer,
    check_module,
    format_violations,
    iter_public_modules,
)

LAYER_PACKAGES = ("domain", "application", "adapters", "web", "cli")

INNER_RULES = (DOMAIN_RULE, APPLICATION_RULE)


def source_module(source: str) -> SourceModule:
    """Синтетический модуль слоя `domain` для проверки самих правил."""
    return SourceModule(module="linguary.domain.sample", package="linguary.domain", source=source)


def test_all_layer_packages_exist() -> None:
    """Каркас пакета содержит все слои приложения."""
    missing = [name for name in LAYER_PACKAGES if not (PACKAGE_ROOT / name / "__init__.py").is_file()]
    assert not missing, f"нет пакетов слоёв: {', '.join(missing)}"


@pytest.mark.parametrize("rule", INNER_RULES, ids=lambda rule: rule.layer)
def test_inner_layer_has_modules_to_check(rule: LayerRule) -> None:
    """Во внутреннем слое есть модули, кроме пакетных `__init__.py`.

    Без этой проверки правила импортов остаются зелёными на пустом слое и перестают
    что-либо гарантировать.
    """
    modules = iter_public_modules(rule.directory)
    assert modules, f"слой {rule.layer} пуст, проверять импорты не на чем"


@pytest.mark.parametrize("rule", INNER_RULES, ids=lambda rule: rule.layer)
def test_inner_layer_dependencies(rule: LayerRule) -> None:
    """`domain` и `application` не зависят от внешних слоёв и технологий."""
    violations = check_layer(rule)
    assert not violations, (
        f"нарушены границы слоя {rule.layer}:\n{format_violations(violations)}"
    )


def test_adapters_do_not_depend_on_web_and_cli() -> None:
    """Адаптеры не зависят от Web-слоя и CLI: зависимость идёт в обратную сторону."""
    violations = check_layer(ADAPTERS_RULE)
    assert not violations, (
        f"нарушены границы слоя adapters:\n{format_violations(violations)}"
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
    """Правила действительно ловят запрещённый код, а не молчат на любом входе."""
    violations = check_module(source_module(source), DOMAIN_RULE)
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
    """Разрешённые импорты правила не считают нарушением."""
    violations = check_module(source_module(source), DOMAIN_RULE)
    assert not violations, f"ложное нарушение:\n{format_violations(violations)}"
