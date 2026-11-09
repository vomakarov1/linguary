"""Verified book-wide pronunciation occurrences shared by web and EPUB."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from linguary.application.validators.pronunciation import (
    group_pronunciations,
    is_complete_pronunciation_form,
    pronunciation_identity,
)
from linguary.domain import BlockProcessingResult, DocumentBlock
from linguary.domain.errors import DomainError


@dataclass(frozen=True, slots=True)
class PronunciationOccurrence:
    hint_id: str
    block_id: str
    start: int
    end: int
    quote: str
    ipa: str
    variant: Literal['uk', 'us']
    source_label: str
    source_ref: str | None
    is_primary: bool


def project_pronunciations(
    blocks: Sequence[DocumentBlock],
    results: Mapping[str, BlockProcessingResult],
    *,
    initial_seen: Mapping[tuple[str, str, str], str] | None = None,
) -> tuple[PronunciationOccurrence, ...]:
    """Reject foreign/stale anchors and choose first occurrences in source order."""
    texts = {block.id: block.text for block in blocks}
    order = {block.id: index for index, block in enumerate(blocks)}
    accepted = []
    for block in blocks:
        result = results.get(block.id)
        if result is None or result.block_id != block.id:
            continue
        for item in result.pronunciation_assistances:
            if item.anchor.block_id != block.id:
                continue
            try:
                item.anchor.validate(texts)
            except DomainError:
                continue
            if not is_complete_pronunciation_form(block.text, item.anchor.start, item.anchor.end):
                continue
            accepted.append(item)
    projected = []
    for group in group_pronunciations(accepted, block_order=order):
        norm_form, norm_ipa, variant = pronunciation_identity(group.primary)
        prior_hint_id = None
        if initial_seen:
            prior_hint_id = initial_seen.get((norm_form, variant, norm_ipa)) or initial_seen.get(
                (norm_form, norm_ipa, variant)
            )
        hint_id = prior_hint_id if prior_hint_id is not None else group.hint_id
        for item in group.occurrences:
            anchor = item.anchor
            projected.append(PronunciationOccurrence(
                hint_id, anchor.block_id, anchor.start, anchor.end, anchor.quote,
                item.ipa, item.variant, item.evidence.attribution, item.evidence.entry_id,
                prior_hint_id is None and item is group.primary,
            ))
    return tuple(sorted(projected, key=lambda item: (order[item.block_id], item.start, item.end)))
