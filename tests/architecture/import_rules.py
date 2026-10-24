"""Import parsing for package `linguary` and inter-layer dependency rules.

The module reads sources as text and parses them via `ast`, so verification does not
depend on whether third-party libraries are installed, and works even on modules that
cannot be imported. Rules are declared declaratively (`LayerRule`); verification returns
a list of violations with exact source locations.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

#: Package root in `src/` layout.
PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "linguary"

#: Project root package name.
ROOT_PACKAGE = "linguary"

#: Standard library modules for the current interpreter.
STDLIB_MODULES = frozenset(sys.stdlib_module_names)

#: Standard library modules providing access to filesystem, network, processes, and DB.
#: For `domain` and `application`, this is external technology akin to a third-party driver.
INFRASTRUCTURE_STDLIB = frozenset(
    {
        "os",
        "io",
        "pathlib",
        "shutil",
        "tempfile",
        "glob",
        "fileinput",
        "zipfile",
        "tarfile",
        "sqlite3",
        "dbm",
        "shelve",
        "socket",
        "ssl",
        "http",
        "urllib",
        "ftplib",
        "smtplib",
        "subprocess",
        "webbrowser",
    }
)

#: Calls representing direct filesystem operations from layer code.
FILESYSTEM_CALLS = frozenset({"open"})


@dataclass(frozen=True)
class SourceModule:
    """Source of a single module along with its dotted name and package name."""

    module: str
    package: str
    source: str

    @classmethod
    def from_path(cls, path: Path) -> SourceModule:
        """Reads a module by path inside `src/`."""
        relative = path.resolve().relative_to(PACKAGE_ROOT.parent).with_suffix("")
        parts = list(relative.parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        module = ".".join(parts)
        package = module if is_package else ".".join(parts[:-1])
        return cls(module=module, package=package, source=path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class LayerRule:
    """Dependency rule for a single layer.

    :param layer: layer name for error messages;
    :param package: dotted package name of the layer;
    :param allowed_internal: project packages permitted for import by the layer;
    :param allow_third_party: whether third-party libraries and file operations are permitted;
    :param forbidden_stdlib: forbidden standard library modules.
    """

    layer: str
    package: str
    allowed_internal: frozenset[str]
    allow_third_party: bool
    forbidden_stdlib: frozenset[str] = frozenset()

    @property
    def directory(self) -> Path:
        """Layer directory inside the package."""
        relative = self.package.removeprefix(f"{ROOT_PACKAGE}.").split(".")
        return PACKAGE_ROOT.joinpath(*relative)


@dataclass(frozen=True)
class Violation:
    """Layer rule violation with exact location in source."""

    module: str
    line: int
    target: str
    reason: str

    def __str__(self) -> str:
        return f"{self.module}:{self.line}: {self.target} — {self.reason}"


DOMAIN_RULE = LayerRule(
    layer="domain",
    package=f"{ROOT_PACKAGE}.domain",
    allowed_internal=frozenset({f"{ROOT_PACKAGE}.domain"}),
    allow_third_party=False,
    forbidden_stdlib=INFRASTRUCTURE_STDLIB,
)

APPLICATION_RULE = LayerRule(
    layer="application",
    package=f"{ROOT_PACKAGE}.application",
    allowed_internal=frozenset({f"{ROOT_PACKAGE}.domain", f"{ROOT_PACKAGE}.application"}),
    allow_third_party=False,
    forbidden_stdlib=INFRASTRUCTURE_STDLIB,
)

ADAPTERS_RULE = LayerRule(
    layer="adapters",
    package=f"{ROOT_PACKAGE}.adapters",
    allowed_internal=frozenset(
        {
            f"{ROOT_PACKAGE}.domain",
            f"{ROOT_PACKAGE}.application",
            f"{ROOT_PACKAGE}.adapters",
        }
    ),
    allow_third_party=True,
)

#: Composition root: the only place aware of both ports and concrete adapters.
#: Views are unknown to it — dependency points from them to it, not vice versa.
COMPOSITION_RULE = LayerRule(
    layer="composition",
    package=f"{ROOT_PACKAGE}.composition",
    allowed_internal=frozenset(
        {
            f"{ROOT_PACKAGE}.domain",
            f"{ROOT_PACKAGE}.application",
            f"{ROOT_PACKAGE}.adapters",
            f"{ROOT_PACKAGE}.composition",
        }
    ),
    allow_third_party=True,
)

#: Web UI: takes dependencies from `composition` and does not depend on CLI —
#: another view of the same application.
WEB_RULE = LayerRule(
    layer="web",
    package=f"{ROOT_PACKAGE}.web",
    allowed_internal=frozenset(
        {
            f"{ROOT_PACKAGE}.domain",
            f"{ROOT_PACKAGE}.application",
            f"{ROOT_PACKAGE}.adapters",
            f"{ROOT_PACKAGE}.composition",
            f"{ROOT_PACKAGE}.web",
        }
    ),
    allow_third_party=True,
)


def iter_modules(directory: Path) -> list[Path]:
    """All `.py` files in the layer, sorted for deterministic output."""
    return sorted(directory.rglob("*.py"))


def iter_public_modules(directory: Path) -> list[Path]:
    """Layer modules excluding package `__init__.py`.

    Used as protection against false-positive test: if a layer consists solely of empty
    `__init__.py`, there is nothing to check.
    """
    return [path for path in iter_modules(directory) if path.name != "__init__.py"]


def imported_names(module: SourceModule) -> list[tuple[int, str]]:
    """All names imported by a module as pairs of (line, dotted name).

    Relative imports are resolved to absolute names using current module's package.
    """
    names: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(module.source)):
        if isinstance(node, ast.Import):
            names.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append((node.lineno, _absolute_import(node, module.package)))
    return names


def _absolute_import(node: ast.ImportFrom, package: str) -> str:
    """Absolute source name for `from ... import ...`."""
    if not node.level:
        return node.module or ""
    parts = package.split(".")
    prefix = ".".join(parts[: len(parts) - node.level + 1])
    return f"{prefix}.{node.module}" if node.module else prefix


def check_module(module: SourceModule, rule: LayerRule) -> list[Violation]:
    """Checks a single module against layer rule."""
    violations = [
        Violation(module=module.module, line=line, target=name, reason=reason)
        for line, name in imported_names(module)
        if (reason := _import_violation_reason(name, rule)) is not None
    ]
    violations.extend(_filesystem_violations(module, rule))
    return violations


def _import_violation_reason(name: str, rule: LayerRule) -> str | None:
    """Reason why import is forbidden for the layer, or `None` if import is allowed."""
    if not name:
        return None
    root = name.split(".")[0]
    if root == ROOT_PACKAGE:
        allowed = any(
            name == package or name.startswith(f"{package}.") for package in rule.allowed_internal
        )
        return None if allowed else f"слой {rule.layer} не зависит от внешнего слоя проекта"
    if root in rule.forbidden_stdlib:
        return (
            f"слой {rule.layer} не работает напрямую с файловой системой, сетью, "
            f"процессами и БД: это ответственность адаптеров"
        )
    if root in STDLIB_MODULES:
        return None
    if rule.allow_third_party:
        return None
    return f"слой {rule.layer} не зависит от сторонних библиотек"


def _filesystem_violations(module: SourceModule, rule: LayerRule) -> list[Violation]:
    """Direct filesystem accesses (`open` calls) within the layer."""
    if rule.allow_third_party:
        return []
    return [
        Violation(
            module=module.module,
            line=node.lineno,
            target=f"{node.func.id}()",
            reason=f"слой {rule.layer} не читает и не пишет файлы",
        )
        for node in ast.walk(ast.parse(module.source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FILESYSTEM_CALLS
    ]


def check_layer(rule: LayerRule) -> list[Violation]:
    """Checks all modules of the layer and returns violations across the whole layer."""
    violations: list[Violation] = []
    for path in iter_modules(rule.directory):
        violations.extend(check_module(SourceModule.from_path(path), rule))
    return violations


def format_violations(violations: list[Violation]) -> str:
    """Error message for test assertion."""
    return "\n".join(f"  {violation}" for violation in violations)
