"""Offline notices for the pinned pronunciation data shipped with the application."""

from dataclasses import dataclass
from hashlib import sha256
from importlib import resources
import json

from linguary.adapters.pronunciation.source import SNAPSHOT, SOURCE_VERSION
from linguary.application.ports.epub import EpubWriteRequest
from linguary.application.pronunciation_projection import project_pronunciations


_NOTICE_HASHES = {
    'NOTICE.md': '1d144ead48e326050355263419bf85bd05578870a66a28f001c51a0f06ce0fbf',
    'LICENSES.txt': '4a171dff6208fa073f675dc255359cad07206a7a9df25d6ee76fd7e87fe072f7',
}


IPA_KEY_LINK_LABEL = 'Ключ к знакам'


def sources_ipa_key_href(href: str | None) -> str | None:
    """Address of the IPA key page, or none when that page is not in the book.

    A blank value and an external address are not a link.
    """
    if not isinstance(href, str):
        return None
    candidate = href.strip()
    if not candidate or _external_href(candidate):
        return None
    return candidate


def _external_href(href: str) -> bool:
    scheme, separator, _ = href.partition(':')
    return bool(separator) and scheme.isalpha()


def load_pronunciation_source_notices() -> str:
    """Return only the known snapshot's complete, verified notices."""
    directory = resources.files('linguary.adapters.pronunciation').joinpath('data', SNAPSHOT)
    texts = []
    for name, digest in _NOTICE_HASHES.items():
        content = directory.joinpath(name).read_bytes()
        if sha256(content).hexdigest() != digest:
            raise ValueError('invalid pronunciation source notices')
        texts.append(content.decode('utf-8'))
    return f'Источник транскрипций: ipa-dict\nВерсия: {SOURCE_VERSION}\n\n' + '\n\n'.join(texts)


@dataclass(frozen=True, slots=True)
class PronunciationNoticeBundle:
    text: str
    entries: tuple[dict[str, str], ...]

    def provenance_json(self) -> bytes:
        return json.dumps({'schema_version': 1, 'entries': self.entries},
                          ensure_ascii=False, indent=2).encode('utf-8')


def pronunciation_notice_bundle(request: EpubWriteRequest) -> PronunciationNoticeBundle | None:
    """Include provenance only for the accepted occurrences shown in the book."""
    blocks = tuple(block for section in request.document.ordered_sections for block in section.ordered_blocks)
    visible = {(item.block_id, item.start, item.end, item.quote, item.ipa, item.variant,
                item.source_ref, item.source_label)
               for item in project_pronunciations(blocks, request.block_results)}
    entries = {}
    for block in blocks:
        result = request.block_results.get(block.id)
        if result is None or result.block_id != block.id:
            continue
        for item in result.pronunciation_assistances:
            evidence = item.evidence
            key = (item.anchor.block_id, item.anchor.start, item.anchor.end, item.anchor.quote,
                   item.ipa, item.variant, evidence.entry_id, evidence.attribution)
            if evidence.kind != 'lexicon_form' or item.anchor.block_id != block.id or key not in visible:
                continue
            if evidence.source_id != 'ipa-dict' or evidence.source_version != SOURCE_VERSION:
                raise ValueError('pronunciation source notices are unavailable')
            entry = dict(form=item.form, ipa=item.ipa, variant=item.variant,
                source_id=evidence.source_id, source_version=evidence.source_version,
                license_id=evidence.license_id, entry_id=evidence.entry_id,
                source_uri=evidence.source_uri)
            entries.setdefault((evidence.source_version, evidence.entry_id), entry)
    if not entries:
        return None
    return PronunciationNoticeBundle(load_pronunciation_source_notices(), tuple(entries.values()))
