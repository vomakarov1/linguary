"""Accepted pronunciation of one exact occurrence and its durable evidence."""
from dataclasses import dataclass
from typing import Literal

from linguary.domain.content_plan import SourceAnchor
from linguary.domain.errors import DomainError


@dataclass(frozen=True, slots=True)
class PronunciationEvidence:
    source_id: str
    source_version: str
    entry_id: str
    sense_id: str
    part_of_speech: str
    dictionary_form: str
    dictionary_ipa: str
    attribution: str
    rule_id: str | None = None
    rule_version: str | None = None

    def __post_init__(self) -> None:
        for name in ('source_id', 'source_version', 'entry_id', 'sense_id',
                     'part_of_speech', 'dictionary_form', 'dictionary_ipa', 'attribution'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DomainError('неполное свидетельство произношения')
        if (self.rule_id is None) != (self.rule_version is None):
            raise DomainError('правило словоформы требует идентификатора и версии')
        if self.rule_id is not None and any(not isinstance(value, str) or not value.strip()
                                            for value in (self.rule_id, self.rule_version)):
            raise DomainError('некорректное правило словоформы')


@dataclass(frozen=True, slots=True)
class PronunciationAssistance:
    anchor: SourceAnchor
    form: str
    ipa: str
    variant: Literal['uk', 'us']
    hint_id: str
    evidence: PronunciationEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.anchor, SourceAnchor) or not isinstance(self.evidence, PronunciationEvidence):
            raise DomainError('произношение требует привязки и свидетельства')
        if any(not isinstance(value, str) or not value.strip() for value in (self.form, self.ipa, self.hint_id)):
            raise DomainError('неполная помощь произношения')
        if self.variant not in {'uk', 'us'} or self.form != self.anchor.quote:
            raise DomainError('произношение не соответствует варианту или фактической форме')
        if self.evidence.rule_id is None and (
                self.form != self.evidence.dictionary_form or self.ipa != self.evidence.dictionary_ipa):
            raise DomainError('производное произношение требует проверяемого правила')
