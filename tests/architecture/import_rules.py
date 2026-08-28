"""Разбор импортов пакета `linguary` и правила зависимостей между слоями.

Модуль читает исходники как текст и разбирает их через `ast`, поэтому проверка не зависит
от того, установлены ли сторонние библиотеки, и срабатывает даже на модулях, которые нельзя
импортировать. Правила описываются декларативно (`LayerRule`), проверка возвращает список
нарушений с точным местом.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

#: Корень пакета в layout `src/`.
PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "linguary"

#: Имя корневого пакета проекта.
ROOT_PACKAGE = "linguary"

#: Модули стандартной библиотеки текущего интерпретатора.
STDLIB_MODULES = frozenset(sys.stdlib_module_names)

#: Модули стандартной библиотеки, дающие доступ к файловой системе, сети, процессам и БД.
#: Для `domain` и `application` это такая же внешняя технология, как сторонний драйвер.
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

#: Вызовы, означающие прямую работу с файловой системой из кода слоя.
FILESYSTEM_CALLS = frozenset({"open"})


@dataclass(frozen=True)
class SourceModule:
    """Исходник одного модуля вместе с его dotted-именем и именем пакета."""

    module: str
    package: str
    source: str

    @classmethod
    def from_path(cls, path: Path) -> SourceModule:
        """Читает модуль по пути внутри `src/`."""
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
    """Правило зависимостей одного слоя.

    :param layer: имя слоя для сообщений об ошибке;
    :param package: dotted-имя пакета слоя;
    :param allowed_internal: пакеты проекта, импорт которых слою разрешён;
    :param allow_third_party: разрешены ли сторонние библиотеки и работа с файлами;
    :param forbidden_stdlib: запрещённые модули стандартной библиотеки.
    """

    layer: str
    package: str
    allowed_internal: frozenset[str]
    allow_third_party: bool
    forbidden_stdlib: frozenset[str] = frozenset()

    @property
    def directory(self) -> Path:
        """Директория слоя внутри пакета."""
        relative = self.package.removeprefix(f"{ROOT_PACKAGE}.").split(".")
        return PACKAGE_ROOT.joinpath(*relative)


@dataclass(frozen=True)
class Violation:
    """Нарушение правила слоя с точным местом в исходнике."""

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


def iter_modules(directory: Path) -> list[Path]:
    """Все `.py`-файлы слоя, отсортированные для стабильного вывода."""
    return sorted(directory.rglob("*.py"))


def iter_public_modules(directory: Path) -> list[Path]:
    """Модули слоя без пакетных `__init__.py`.

    Используется как защита от ложно-зелёного теста: если слой состоит из одних пустых
    `__init__.py`, проверять в нём нечего.
    """
    return [path for path in iter_modules(directory) if path.name != "__init__.py"]


def imported_names(module: SourceModule) -> list[tuple[int, str]]:
    """Все импортируемые модулем имена как пары «строка, dotted-имя».

    Относительные импорты приводятся к абсолютным по имени пакета текущего модуля.
    """
    names: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(module.source)):
        if isinstance(node, ast.Import):
            names.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append((node.lineno, _absolute_import(node, module.package)))
    return names


def _absolute_import(node: ast.ImportFrom, package: str) -> str:
    """Абсолютное имя источника для `from ... import ...`."""
    if not node.level:
        return node.module or ""
    parts = package.split(".")
    prefix = ".".join(parts[: len(parts) - node.level + 1])
    return f"{prefix}.{node.module}" if node.module else prefix


def check_module(module: SourceModule, rule: LayerRule) -> list[Violation]:
    """Проверяет один модуль по правилу слоя."""
    violations = [
        Violation(module=module.module, line=line, target=name, reason=reason)
        for line, name in imported_names(module)
        if (reason := _import_violation_reason(name, rule)) is not None
    ]
    violations.extend(_filesystem_violations(module, rule))
    return violations


def _import_violation_reason(name: str, rule: LayerRule) -> str | None:
    """Причина, по которой импорт запрещён слою, либо `None`, если импорт допустим."""
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
    """Прямые обращения к файловой системе (вызов `open`) внутри слоя."""
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
    """Проверяет все модули слоя и возвращает нарушения по всему слою."""
    violations: list[Violation] = []
    for path in iter_modules(rule.directory):
        violations.extend(check_module(SourceModule.from_path(path), rule))
    return violations


def format_violations(violations: list[Violation]) -> str:
    """Сообщение об ошибке для assert-а теста."""
    return "\n".join(f"  {violation}" for violation in violations)
