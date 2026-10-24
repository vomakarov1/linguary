"""Versioned analysis obligations and precise references to educational assistance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

from linguary.domain.errors import DomainError


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceAnchor:
    block_id: str
    start: int
    end: int
    quote: str

    def __post_init__(self) -> None:
        if (type(self.start) is not int or type(self.end) is not int
                or not self.block_id.strip() or not self.quote.strip()
                or self.start < 0 or self.end <= self.start
                or self.end - self.start != len(self.quote)):
            raise DomainError("некорректная привязка к оригиналу")

    def validate(self, texts: Mapping[str, str]) -> None:
        if self.block_id not in texts or texts[self.block_id][self.start:self.end] != self.quote:
            raise DomainError("цитата и точные смещения не совпадают с оригиналом")


@dataclass(frozen=True, slots=True)
class RequiredPoint:
    id: str
    description: str

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.description.strip():
            raise DomainError("смысловой аспект должен иметь идентификатор и описание")


@dataclass(frozen=True, slots=True)
class Difficulty:
    id: str
    anchors: tuple[SourceAnchor, ...]
    owner_block_id: str
    kind: str
    priority: str
    help_kind: str
    required_points: tuple[RequiredPoint, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchors", tuple(sorted(self.anchors, key=lambda a: (a.block_id, a.start, a.end, a.quote))))
        object.__setattr__(self, "required_points", tuple(sorted(self.required_points, key=lambda p: p.id)))
        if not self.id.strip() or not self.anchors or self.owner_block_id not in {a.block_id for a in self.anchors}:
            raise DomainError("трудность должна иметь единственного владельца среди привязок")
        if self.kind not in {"wordplay", "semantic_relation", "grammar", "cultural_reference", "idiom", "vocabulary"}:
            raise DomainError("неизвестный вид трудности")
        if self.priority not in {"critical", "useful"} or self.help_kind not in {"explanation", "lexical_hint"}:
            raise DomainError("неизвестный приоритет или канал помощи")
        ids = [point.id for point in self.required_points]
        if not ids or len(ids) != len(set(ids)):
            raise DomainError("смысловые аспекты пусты или повторяются")

    @property
    def canonical_id(self) -> str:
        return canonical_sha({"version": 2, "anchors": sorted((asdict(a) for a in self.anchors), key=lambda a: (a["block_id"], a["start"], a["end"])), "kind": self.kind,
                              "required_points": sorted((asdict(p) for p in self.required_points), key=lambda p: p["id"])})


@dataclass(frozen=True, slots=True)
class HelpEvidence:
    difficulty_id: str
    point_id: str
    artifact_kind: str
    block_id: str
    artifact_key: str
    body_excerpt: str

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.difficulty_id, self.point_id, self.block_id, self.artifact_key, self.body_excerpt)):
            raise DomainError("свидетельство требует аспект, артефакт и непустое тело")
        if self.artifact_kind not in {"before_you_read.explanation", "language_note.note", "glossary_hint.note", "glossary_hint.translation"}:
            raise DomainError("неизвестный вид артефакта помощи")


def artifact_key(kind: str, block_id: str, content: object) -> str:
    return canonical_sha({"kind": kind, "block_id": block_id, "content": content})


@dataclass(frozen=True, slots=True)
class ContentPlan:
    version: int = 2
    profile: str = "content-plan-v2"
    difficulties: tuple[Difficulty, ...] = ()
    before_you_read_evidence: tuple[HelpEvidence, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "difficulties", tuple(sorted(self.difficulties, key=lambda d: d.id)))
        object.__setattr__(self, "before_you_read_evidence", tuple(sorted(self.before_you_read_evidence, key=lambda e: (e.difficulty_id, e.point_id, e.artifact_key, e.body_excerpt))))
        if self.version != 2 or self.profile != "content-plan-v2":
            raise DomainError("неизвестная версия содержательного плана")
        ids = [d.id for d in self.difficulties]
        if len(ids) != len(set(ids)):
            raise DomainError("идентификаторы трудностей повторяются")
        for evidence in self.before_you_read_evidence:
            self.validate_evidence(evidence)

    def validate_evidence(self, evidence: HelpEvidence) -> None:
        difficulty = next((d for d in self.difficulties if d.id == evidence.difficulty_id), None)
        if difficulty is None or evidence.point_id not in {p.id for p in difficulty.required_points} or evidence.block_id != difficulty.owner_block_id:
            raise DomainError("свидетельство относится к чужой трудности, аспекту или владельцу")

    @property
    def sha256(self) -> str:
        value = asdict(self)
        value["difficulties"] = sorted(value["difficulties"], key=lambda d: d["id"])
        value["before_you_read_evidence"] = sorted(value["before_you_read_evidence"], key=lambda e: (e["difficulty_id"], e["point_id"], e["artifact_key"]))
        return canonical_sha(value)


@dataclass(frozen=True, slots=True)
class ContentEvidence:
    plan_sha256: str
    evidence: tuple[HelpEvidence, ...] = ()
    version: int = 2
    structural_coverage: str = "complete"
    semantic_correctness: str = "not_evaluated"

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if self.version != 2 or len(self.plan_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.plan_sha256):
            raise DomainError("неизвестная версия свидетельств или неверный SHA плана")
        if self.structural_coverage != "complete" or self.semantic_correctness != "not_evaluated":
            raise DomainError("неподдерживаемая оценка содержательной помощи")
