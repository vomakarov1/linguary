"""Accepted pronunciation of one exact occurrence and its durable evidence."""
import re
from dataclasses import dataclass
from typing import Literal

from linguary.domain.content_plan import SourceAnchor
from linguary.domain.errors import DomainError


@dataclass(frozen=True, slots=True)
class PronunciationEvidence:
    source_id: str
    source_version: str
    entry_id: str
    sense_id: str | None
    part_of_speech: str | None
    dictionary_form: str
    dictionary_ipa: str
    attribution: str
    rule_id: str | None = None
    rule_version: str | None = None
    kind: Literal['dictionary_sense', 'lexicon_form'] = 'dictionary_sense'
    license_id: str | None = None
    source_uri: str | None = None

    def __post_init__(self) -> None:
        for name in ('source_id', 'source_version', 'entry_id',
                     'dictionary_form', 'dictionary_ipa', 'attribution'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DomainError('неполное свидетельство произношения')
        if self.kind not in ('dictionary_sense', 'lexicon_form'):
            raise DomainError('неизвестный вид свидетельства произношения')
        if any(value is not None and (not isinstance(value, str) or not value.strip())
               for value in (self.license_id, self.source_uri)):
            raise DomainError('некорректные сведения об источнике произношения')
        if self.kind == 'lexicon_form':
            if (self.sense_id is not None or self.part_of_speech is not None
                    or self.rule_id is not None or self.rule_version is not None
                    or not self._has_pinned_lexicon_source()):
                raise DomainError('некорректное свидетельство точной формы лексикона')
        elif any(not isinstance(value, str) or not value.strip()
                 for value in (self.sense_id, self.part_of_speech)):
            raise DomainError('неполное свидетельство произношения')
        if (self.rule_id is None) != (self.rule_version is None):
            raise DomainError('правило словоформы требует идентификатора и версии')
        if self.rule_id is not None and any(not isinstance(value, str) or not value.strip()
                                            for value in (self.rule_id, self.rule_version)):
            raise DomainError('некорректное правило словоформы')

    def _has_pinned_lexicon_source(self) -> bool:
        version = re.fullmatch(r'ipa-dict@([a-f0-9]{40}):parser-[1-9][0-9]*', self.source_version)
        if self.source_id != 'ipa-dict' or version is None:
            return False
        snapshot = version[1]
        for filename, license_id in (('en_UK.txt', 'GPL-3.0'), ('en_US.txt', 'MIT')):
            uri = f'https://raw.githubusercontent.com/open-dict-data/ipa-dict/{snapshot}/data/{filename}'
            if self.license_id == license_id and self.source_uri == uri:
                return True
        return False

    @property
    def lexicon_variant(self) -> Literal['uk', 'us'] | None:
        if self.kind != 'lexicon_form' or self.source_uri is None:
            return None
        return 'uk' if self.source_uri.endswith('/data/en_UK.txt') else 'us'


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
        if self.evidence.kind == 'lexicon_form' and self.variant != self.evidence.lexicon_variant:
            raise DomainError('вариант произношения не соответствует источнику лексикона')
        if self.evidence.rule_id is None and (
                self.form != self.evidence.dictionary_form or self.ipa != self.evidence.dictionary_ipa):
            raise DomainError('производное произношение требует проверяемого правила')
