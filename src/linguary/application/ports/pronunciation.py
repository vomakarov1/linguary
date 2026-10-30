"""Licensed pronunciation lookup without book text or model-supplied addresses."""
from dataclasses import dataclass
from typing import Literal, Protocol
import unicodedata

from linguary.domain.pronunciation import PronunciationEvidence


type PronunciationVariant = Literal['uk', 'us']


def valid_lookup_form(form: str) -> bool:
    return (isinstance(form, str) and 0 < len(form) <= 200
        and any(char.isalpha() for char in form)
        and all(char.isalpha() or unicodedata.category(char).startswith('M')
                or char in "'-’" for char in form))


def _valid_ipa(text: str) -> bool:
    """Reject malformed source text; character shape never proves a reading."""
    return isinstance(text, str) and bool(text.strip()) and all(
        char in 'ˈˌːˑ˞. /-()‿|βθχ'
        or unicodedata.category(char).startswith('M')
        or (char.isalpha() and char == char.lower() and 'LATIN' in unicodedata.name(char, ''))
        for char in text)


@dataclass(frozen=True, slots=True)
class InflectionRule:
    """A versioned suffix transformation supplied by the trusted source adapter."""
    id: str
    version: str
    grammatical_form: str
    form_suffix: str
    ipa_suffix: str
    base_form_suffix: str = ''
    base_ipa_suffix: str = ''

    def confirms(self, entry: 'SourceEntry') -> bool:
        evidence = entry.evidence
        return (bool(self.id.strip() and self.version.strip())
            and (self.id, self.version) == (evidence.rule_id, evidence.rule_version)
            and self.grammatical_form == entry.grammatical_form
            and self._apply(evidence.dictionary_form, self.base_form_suffix, self.form_suffix) == entry.form
            and self._apply(evidence.dictionary_ipa, self.base_ipa_suffix, self.ipa_suffix) == entry.ipa)

    @staticmethod
    def _apply(base, removed, added):
        if not base.endswith(removed):
            return None
        return (base[:-len(removed)] if removed else base) + added


@dataclass(frozen=True, slots=True)
class SourceEntry:
    form: str
    ipa: str
    variant: PronunciationVariant
    sense: str | None
    grammatical_form: str | None
    evidence: PronunciationEvidence
    proper_name_confirmed: bool = False
    inflection_rule: InflectionRule | None = None

    def __post_init__(self):
        if (not valid_lookup_form(self.form) or self.variant not in {'uk', 'us'}
                or not isinstance(self.evidence, PronunciationEvidence)
                or not _valid_ipa(self.ipa) or not _valid_ipa(self.evidence.dictionary_ipa)
                or type(self.proper_name_confirmed) is not bool):
            raise ValueError('неполная запись источника произношения')
        if self.evidence.kind == 'lexicon_form':
            if (self.sense is not None or self.grammatical_form is not None
                    or self.proper_name_confirmed or self.inflection_rule is not None
                    or self.variant != self.evidence.lexicon_variant
                    or self.form != self.evidence.dictionary_form
                    or self.ipa != self.evidence.dictionary_ipa):
                raise ValueError('некорректная запись точной формы лексикона')
        elif not all(isinstance(value, str) and value.strip()
                     for value in (self.sense, self.grammatical_form)):
            raise ValueError('неполная запись источника произношения')


@dataclass(frozen=True, slots=True)
class SourceLookupResult:
    status: Literal['found', 'missing', 'unavailable']
    entries: tuple[SourceEntry, ...] = ()
    reason: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'entries', tuple(self.entries))
        if (self.status not in {'found', 'missing', 'unavailable'}
                or bool(self.entries) != (self.status == 'found')
                or any(not isinstance(entry, SourceEntry) for entry in self.entries)
                or (self.status == 'unavailable' and not self.reason)):
            raise ValueError('некорректный результат поиска произношения')

    @classmethod
    def found(cls, entries):
        return cls('found', tuple(entries))

    @classmethod
    def missing(cls):
        return cls('missing')

    @classmethod
    def unavailable(cls, reason):
        return cls('unavailable', reason=reason)


class PronunciationSource(Protocol):
    @property
    def configured(self) -> bool:
        ...

    async def lookup(self, form: str, variant: PronunciationVariant) -> SourceLookupResult:
        """Only permitted normalized records; transport failures are unavailable."""
        ...
