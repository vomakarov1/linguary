"""Validation of truncated generation by formal indicators (PRD § 15, ALG-04).

There are three indicators: the provider reported truncation due to token limit, the translation
does not end with terminal punctuation, or a pair of quotes or brackets is left unclosed. The first
is reported by the provider itself (`LLMResponse.truncated`) — checking it is mandatory in processing:
the response might be complete in terms of block structure but cut short in text, and coverage checks
(which cover this gap in chapter analysis) would detect nothing here.

The two textual indicators mirror the original text, otherwise they catch normal writing rather
than truncation. A heading, caption, or list item legitimately ends without a period, and multi-paragraph
dialogue conventionally opens quotes on each paragraph while closing them only on the last. Therefore
terminal punctuation is required in translation only where the source block ends with it, and an
unclosed pair is considered truncation only when the original lacked such an imbalance.
A block with several segments is checked per `TRANSLATE` slice, not as one string.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from linguary.application.llm_dto import BlockResultResponse, TranslationResultResponse
from linguary.application.validators.quotes import (
    british_quote_counts,
    has_unclosed_british_quote,
)
from linguary.application.validators.treatment import planned_segment_modes
from linguary.domain import (
    BlockTreatment,
    TranslationUnit,
    TranslationUnitItem,
    without_footnote_markers,
)

#: Punctuation marks that conclude a completed sentence.
_TERMINAL_PUNCTUATION = frozenset(".!?…:;")

#: Trailing characters that can appear after terminal punctuation.
_TRAILING_CLOSERS = "»\"”’'*)]}"

#: Brackets whose imbalance indicates truncated text.
_BRACKETS: tuple[tuple[str, str], ...] = (("(", ")"), ("[", "]"), ("{", "}"))

#: Symmetric quotation mark: pairing is determined by the parity of occurrence count.
_SYMMETRIC_QUOTE = '"'


def truncation_violation(
    payload: TranslationResultResponse, *, unit: TranslationUnit, truncated: bool
) -> str | None:
    """Violation description of truncated generation, or `None`.

    `truncated` is a signal from the provider: output was cut off by token limit
    rather than finished naturally by the model.
    """
    if truncated:
        return "провайдер сообщил об обрыве ответа по лимиту выходных токенов"
    items = {item.block.id: item for item in unit.items}
    for block in payload.blocks:
        item = items.get(block.source_block_id)
        if item is None:
            continue
        issue = _block_truncation_issue(block, item)
        if issue is not None:
            return issue[0]
    return None


def _checked_translations(
    block: BlockResultResponse, item: TranslationUnitItem
) -> Iterator[tuple[int | None, str, str]]:
    """TRANSLATE slices to compare. A one-span block yields the whole text once."""
    modes = planned_segment_modes(item)
    if not modes:
        if block.translated_text is None:
            return
        yield None, item.block.text, block.translated_text
        return
    by_index = {segment.index: segment.translated_text for segment in block.segments}
    for index, (span, mode) in enumerate(
        zip(item.block.segment_spans, modes, strict=True), start=1
    ):
        if mode is not BlockTreatment.TRANSLATE:
            continue
        text = by_index.get(index)
        if text is None:
            continue
        yield index, item.block.text[span.start:span.end], text


def _subject(block_id: str, segment_index: int | None) -> str:
    """Russian noun phrase naming the translation that failed the check."""
    if segment_index is None:
        return f"блока {block_id}"
    return f"сегмента {segment_index} блока {block_id}"


def _slice_truncation_issue(
    text: str,
    *,
    source: str,
    block_id: str,
    segment_index: int | None = None,
    check_quotes: bool = True,
) -> tuple[str, str, str] | None:
    """Return `(message, rule_code, safe_message)` for one slice, or `None`."""
    subject = _subject(block_id, segment_index)
    if _ends_with_punctuation(source) and not _ends_with_punctuation(text):
        return (
            f"перевод {subject} не заканчивается терминальной пунктуацией",
            "terminal_punctuation",
            "Перевод не завершён терминальной пунктуацией",
        )
    unbalanced = _unbalanced(text, source=source, check_quotes=check_quotes)
    if unbalanced is not None:
        code = "quote_balance" if unbalanced == "кавычки" else "bracket_balance"
        safe_message = (
            "Нарушена парность кавычек"
            if code == "quote_balance"
            else "Нарушена парность скобок"
        )
        return f"в переводе {subject} не закрыты {unbalanced}", code, safe_message
    return None


def _has_quote_mark(text: str) -> bool:
    """Whether `text` contains any quotation mark (excluding apostrophes in contractions)."""
    if any(ch in text for ch in '"«»„“”'):
        return True
    brit_open, brit_close = british_quote_counts(text)
    return brit_open > 0 or brit_close > 0


def _translate_runs(
    block: BlockResultResponse, item: TranslationUnitItem
) -> list[list[tuple[int | None, str, str]]]:
    """Contiguous runs of `TRANSLATE` slices split by quote-bearing non-`TRANSLATE` segments."""
    modes = planned_segment_modes(item)
    by_index = {segment.index: segment.translated_text for segment in block.segments}
    runs: list[list[tuple[int | None, str, str]]] = []
    current: list[tuple[int | None, str, str]] = []
    for index, (span, mode) in enumerate(
        zip(item.block.segment_spans, modes, strict=True), start=1
    ):
        seg_source = item.block.text[span.start:span.end]
        if mode is not BlockTreatment.TRANSLATE:
            if current and _has_quote_mark(seg_source):
                runs.append(current)
                current = []
            continue
        text = by_index.get(index)
        if text is None:
            continue
        current.append((index, seg_source, text))
    if current:
        runs.append(current)
    return runs


def _block_truncation_issue(
    block: BlockResultResponse, item: TranslationUnitItem
) -> tuple[str, str, str] | None:
    """Return `(message, rule_code, safe_message)` for the first truncation issue in `block`."""
    slices = list(_checked_translations(block, item))
    if not slices:
        return None
    if not planned_segment_modes(item):
        segment_index, source, text = slices[0]
        return _slice_truncation_issue(
            text,
            source=source,
            block_id=block.source_block_id,
            segment_index=segment_index,
            check_quotes=True,
        )
    for segment_index, source, text in slices:
        issue = _slice_truncation_issue(
            text,
            source=source,
            block_id=block.source_block_id,
            segment_index=segment_index,
            check_quotes=False,
        )
        if issue is not None:
            return issue
    for run in _translate_runs(block, item):
        bad_indices: list[int | None] = []
        nonzero_indices: list[int | None] = []
        for segment_index, source, text in run:
            tr_bal = _balances(text, allow_midline_british_open=True)["кавычки"]
            src_bal = _balances(source, allow_midline_british_open=True)["кавычки"]
            if tr_bal != 0:
                nonzero_indices.append(segment_index)
            if not _quote_balance_matches(tr_bal, source=source, source_balance=src_bal):
                bad_indices.append(segment_index)
        joined_tr = " ".join(tr for _, _, tr in run)
        joined_src = " ".join(src for _, src, _ in run)
        total_tr = _balances(joined_tr, allow_midline_british_open=True)["кавычки"]
        total_src = _balances(joined_src, allow_midline_british_open=True)["кавычки"]
        if total_tr == 0 or _quote_balance_matches(
            total_tr, source=joined_src, source_balance=total_src
        ):
            continue
        report_index = (
            bad_indices[0]
            if bad_indices
            else (nonzero_indices[0] if nonzero_indices else run[0][0])
        )
        subject = _subject(block.source_block_id, report_index)
        return (
            f"в переводе {subject} не закрыты кавычки",
            "quote_balance",
            "Нарушена парность кавычек",
        )
    return None


def _ends_with_punctuation(text: str) -> bool:
    """Whether the text ends with a sentence-terminal punctuation mark."""
    stripped = without_footnote_markers(text).rstrip().rstrip(_TRAILING_CLOSERS).rstrip()
    return bool(stripped) and stripped[-1] in _TERMINAL_PUNCTUATION


def _quote_balance_matches(
    tr_balance: int, *, source: str, source_balance: int
) -> bool:
    """Whether translation quote balance is consistent with `source`."""
    if tr_balance == 0 or tr_balance == source_balance:
        return True
    if abs(tr_balance) == 1 and source.count(_SYMMETRIC_QUOTE) % 2 == 1:
        return True
    return tr_balance == 1 and _has_british_unclosed_opening_quote(source)


def _unbalanced(text: str, *, source: str, check_quotes: bool = True) -> str | None:
    """Type of paired punctuation whose imbalance was introduced in the translation."""
    original = _balances(source)
    for name, balance in _balances(text, allow_midline_british_open=True).items():
        if name == "кавычки":
            if not check_quotes:
                continue
            if _quote_balance_matches(
                balance, source=source, source_balance=original[name]
            ):
                continue
            return name
        if balance == 0 or balance == original[name]:
            continue
        return name
    return None


def _has_british_unclosed_opening_quote(source: str) -> bool:
    """Whether `source` opens British single-quoted dialogue left unclosed at paragraph end."""
    return has_unclosed_british_quote(source)


def _balances(
    text: str, *, allow_midline_british_open: bool = False
) -> Mapping[str, int]:
    """Imbalance for each type of paired punctuation: number of open marks left unclosed."""
    balances = {
        f"скобки {opening}{closing}": text.count(opening) - text.count(closing)
        for opening, closing in _BRACKETS
    }
    # « and „ are always openings. » and ” are always closings.
    # “ is closing when paired with „ (Russian/German „...“), and opening when paired with ” (English “...”).
    n_low = text.count("„")
    n_high_open = text.count("“")
    n_high_close = text.count("”")
    paired_low = min(n_low, n_high_open)
    brit_open, brit_close = british_quote_counts(text)
    if (
        brit_open > brit_close
        and not allow_midline_british_open
        and not text.lstrip().startswith("‘")
    ):
        brit_open = brit_close
    openings = text.count("«") + n_low + (n_high_open - paired_low) + brit_open
    closings = text.count("»") + n_high_close + paired_low + brit_close
    balances["кавычки"] = (openings - closings) + text.count(_SYMMETRIC_QUOTE) % 2
    return balances


__all__ = ["truncation_violation"]


def truncation_diagnostic(payload, *, unit, truncated):
    """Structured provenance is formed where the punctuation check observes its block."""
    from linguary.domain.llm_log import Violation

    if truncated:
        return (
            truncation_violation(payload, unit=unit, truncated=True),
            Violation(
                "provider_truncated",
                "translation_unit",
                "unit",
                tuple(item.block.id for item in unit.items),
                ("translated_text",),
                "Провайдер сообщил об обрыве ответа",
            ),
        )
    items = {item.block.id: item for item in unit.items}
    for block in payload.blocks:
        item = items.get(block.source_block_id)
        if item is None:
            continue
        issue = _block_truncation_issue(block, item)
        if issue is not None:
            message, code, safe_message = issue
            return (
                message,
                Violation(
                    code,
                    "translation_unit",
                    "local",
                    (block.source_block_id,),
                    ("translated_text",),
                    safe_message,
                ),
            )
    return None
