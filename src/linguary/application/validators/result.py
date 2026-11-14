"""Validation of single translation unit processing results (ALG-04, PRD § 15, § 17).

The package's single entry point: the processing use case hands parsed model responses here
and receives either block results ready for persistence or `ResultValidationError`.
There is no intermediate state — a blocking validation rejects the entire unit,
and partially saved `TranslationUnit` records do not exist (PRD § 17).

Rejection always carries `ResultExpectation`: the retry policy derives a clarification
instruction from it containing the list of expected blocks, their treatments, and the target
language (ALG-05). The expectation can be supplied from the outside — the use case addresses
blocks to the model as it chooses, and clarification instructions must use the same block names
as the request.

Check order is significant: response coverage is checked first, because subsequent validators
access response blocks by group identifiers.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import replace

from linguary.application.llm_dto import (
    BlockResultResponse,
    LanguageNoteResponse,
    TranslationResultResponse,
)
from linguary.application.retry_policy import (
    ExpectedBlock,
    ResultExpectation,
    ResultValidationError,
)
from linguary.application.validators.coverage import coverage_violation
from linguary.application.validators.language import language_violation
from linguary.application.validators.mixed_script import mixed_script_violation
from linguary.application.validators.paragraphs import paragraph_violation
from linguary.application.validators.repetition import repetition_warning
from linguary.application.validators.service_messages import service_message_violation
from linguary.application.validators.tails import normalized_tail
from linguary.application.validators.treatment import (
    content_violation,
    joined_segment_translation,
    planned_segment_modes,
    treatment_violation,
)
from linguary.application.validators.truncation import truncation_violation, truncation_diagnostic
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    DocumentBlock,
    DomainError,
    GlossaryHint,
    LanguageNote,
    ResultWarning,
    ResultWarningKind,
    TranslationUnit,
    TranslationUnitItem,
    footnote_references,
    known_markers_only,
    without_footnote_markers,
)
from linguary.domain.llm_log import Violation

#: Footnote marker in a translation. The same shape the domain strips or keeps.
_MARKER = re.compile(r" ?\[\^(\d+)\]")


class TranslationResultValidator:
    """Blocking validations and warnings for processing results (PRD § 15).

    Thresholds are provided by `CheckSettings`, where validity is already guaranteed,
    so they are not re-checked here: validations have no duplicate defaults.
    """

    def __init__(
        self,
        *,
        target_language_min_share: float,
        language_check_min_length: int,
        repetition_ngram_size: int,
        repetition_min_repeats: int,
    ) -> None:
        self._target_language_min_share = target_language_min_share
        self._language_check_min_length = language_check_min_length
        self._repetition_ngram_size = repetition_ngram_size
        self._repetition_min_repeats = repetition_min_repeats

    def validate(
        self,
        payload: TranslationResultResponse,
        *,
        unit: TranslationUnit,
        target_language: str,
        note_id_factory: Callable[[], str],
        truncated: bool = False,
        expectation: ResultExpectation | None = None,
    ) -> tuple[BlockProcessingResult, ...]:
        """Validates the model response and builds block processing results.

        The response must be expressed in unit block IDs: if the use case addressed
        blocks using aliases, it maps them back to IDs before validation and passes
        its own `expectation` — the one listing blocks as seen by the model.

        `note_id_factory` generates learning note identifiers (`ln-ch03-001`): their
        numbering depends on the chapter and already stored notes, which is use case knowledge.
        """
        from linguary.domain.learning import validate_assistance_body
        for assistance in payload.fragment_assistances:
            try:
                validate_assistance_body(assistance.kind, assistance.text)
            except DomainError as error:
                raise ResultValidationError(str(error), expectation=expectation) from error
        expected = expectation or expectation_for_unit(
            unit, target_language=target_language
        )
        # Цельный перевод снимает сегменты только по явному признаку и только
        # у ответа без segments. Пустые режимы ожидания сегменты не снимают.
        unit = _unit_for_whole_translation(unit, expected, payload)
        self._ensure(coverage_violation(payload, unit=unit), expected)
        for block in payload.blocks:
            single = replace(payload, blocks=(block,))
            self._ensure_block(
                treatment_violation(single, unit=unit),
                expected,
                block_id=block.source_block_id,
                rule_code="treatment_mismatch",
                safe_message="Режим блока не совпадает с планом обработки",
            )
        for block in payload.blocks:
            single = replace(payload, blocks=(block,))
            self._ensure_block(
                content_violation(single, unit=unit),
                expected,
                block_id=block.source_block_id,
                rule_code="content_violation",
                safe_message="Содержимое блока не прошло структурную проверку",
            )
        for block in payload.blocks:
            single = replace(payload, blocks=(block,))
            self._ensure_block(
                paragraph_violation(single, unit=unit),
                expected,
                block_id=block.source_block_id,
                rule_code="paragraph_count_mismatch",
                safe_message="Нарушено число строк или абзацев перевода",
            )
        normalized_payload = replace(
            payload,
            blocks=tuple(_normalize_block(block) for block in payload.blocks),
        )
        items = {item.block.id: item for item in unit.items}
        for block in normalized_payload.blocks:
            item = items.get(block.source_block_id)
            if item is None:
                continue
            self._ensure_block(
                _foreign_marker_violation(block, item.block),
                expected,
                block_id=block.source_block_id,
                rule_code="foreign_footnote_marker",
                safe_message="Маркер сноски относится к другому сегменту",
            )
        truncation = truncation_diagnostic(normalized_payload, unit=unit, truncated=truncated)
        if truncation is not None:
            message, violation = truncation
            raise ResultValidationError(message, expectation=expected, violation=violation)
        for block in normalized_payload.blocks:
            single = replace(normalized_payload, blocks=(block,))
            self._ensure_block(
                language_violation(
                    single,
                    target_language=target_language,
                    min_share=self._target_language_min_share,
                    min_length=self._language_check_min_length,
                ),
                expected,
                block_id=block.source_block_id,
                rule_code="wrong_target_language",
                safe_message="Перевод выполнен не на целевом языке",
            )
        text_payload = _payload_with_joined_translation(normalized_payload)
        for block in text_payload.blocks:
            single = replace(text_payload, blocks=(block,))
            self._ensure_block(
                service_message_violation(single),
                expected,
                block_id=block.source_block_id,
                rule_code="service_message_leak",
                safe_message="В ответе обнаружено служебное сообщение модели",
            )
        for block in text_payload.blocks:
            single = replace(text_payload, blocks=(block,))
            self._ensure_block(
                mixed_script_violation(single),
                expected,
                block_id=block.source_block_id,
                rule_code="mixed_script_leak",
                safe_message="В тексте обнаружено смешение алфавитов внутри слова",
            )
        results = tuple(
            self._result(
                block,
                unit=unit,
                notes=normalized_payload.notes_for(block.source_block_id),
                note_id_factory=note_id_factory,
            )
            for block in normalized_payload.blocks
        )
        self._ensure_domain_rules(results, unit=unit, expectation=expected)
        return results

    def _result(
        self,
        block: BlockResultResponse,
        *,
        unit: TranslationUnit,
        notes: Sequence[LanguageNoteResponse],
        note_id_factory: Callable[[], str],
    ) -> BlockProcessingResult:
        """Result for a single block: treatment from plan, model texts, warnings.

        Footnote link markers (`[^1]`) are preserved only in translations and only when
        the source block contains that footnote link: hints and notes do not need links,
        and hallucinated markers would lead nowhere.
        """
        item = next(
            item for item in unit.items if item.block.id == block.source_block_id
        )
        source = item.block
        modes = planned_segment_modes(item)
        segment_translations = _segment_translations(block, source, modes)
        if modes:
            translated_text = _joined(segment_translations)
        else:
            translated_text = (
                known_markers_only(block.translated_text, source)
                if block.translated_text is not None
                else None
            )
            segment_translations = ()
        return BlockProcessingResult(
            block_id=block.source_block_id,
            treatment=item.treatment,
            translated_text=translated_text,
            glossary_hints=tuple(
                GlossaryHint(
                    original=_unmarked(hint.original),
                    translation=_unmarked(hint.translation),
                    note=_unmarked(hint.note) if hint.note is not None else None,
                )
                for hint in block.glossary_hints
            ),
            language_notes=tuple(
                LanguageNote(
                    id=note_id_factory(),
                    block_id=note.source_block_id,
                    original_excerpt=_unmarked(note.original_excerpt),
                    note=_unmarked(note.note),
                    translated_excerpt=(
                        _unmarked(note.translated_excerpt)
                        if note.translated_excerpt is not None
                        else None
                    ),
                )
                for note in notes
            ),
            warnings=self._warnings(block),
            segment_treatments=modes,
            segment_translations=segment_translations,
        )

    def _warnings(self, block: BlockResultResponse) -> tuple[ResultWarning, ...]:
        """Non-blocking warnings for a block result (PRD § 15)."""
        text = joined_segment_translation(block)
        if text is None:
            return ()
        repetition = repetition_warning(
            text,
            ngram_size=self._repetition_ngram_size,
            min_repeats=self._repetition_min_repeats,
        )
        if repetition is None:
            return ()
        return (ResultWarning(kind=ResultWarningKind.REPETITION, message=repetition),)

    def _ensure(self, violation: str | None, expectation: ResultExpectation) -> None:
        """Converts an encountered violation into rejection of the entire unit."""
        if violation is not None:
            raise ResultValidationError(violation, expectation=expectation)

    def _ensure_block(
        self,
        problem: str | Violation | None,
        expectation: ResultExpectation,
        *,
        block_id: str,
        rule_code: str,
        safe_message: str,
    ) -> None:
        """Converts a block-scoped violation into rejection with structured Violation."""
        if problem is None:
            return
        actual_rule_code = getattr(problem, "rule_code", None) or rule_code
        message = (
            problem.safe_message
            if isinstance(problem, Violation) and problem.safe_message
            else str(problem)
        )
        raise ResultValidationError(
            message,
            expectation=expectation,
            violation=Violation(
                actual_rule_code,
                "translation_unit",
                scope="local",
                block_ids=(block_id,),
                safe_message=safe_message,
            ),
        )

    def _ensure_domain_rules(
        self,
        results: Sequence[BlockProcessingResult],
        *,
        unit: TranslationUnit,
        expectation: ResultExpectation,
    ) -> None:
        """Domain re-verification of the assembled unit.

        Rules of coverage and order are owned by `TranslationUnit` itself, and it has
        the final say. Execution reaches here only if an application-level check missed
        something; domain errors are converted into the same rejection with expectation
        so retries receive the block list rather than a bare error message.
        """
        try:
            # The plan collapses a uniform multi-segment tuple. The stored result
            # keeps one mode per span, so the domain comparison uses that tuple.
            _unit_for_domain_check(unit).validate_results(results)
        except DomainError as error:
            raise ResultValidationError(str(error), expectation=expectation) from error


def _unmarked(text: str) -> str:
    """Text stripped of footnote markers; a field consisting purely of a marker is preserved."""
    unmarked = without_footnote_markers(text)
    if unmarked == text:
        return text
    return unmarked.strip() or text


def _normalize_block(block: BlockResultResponse) -> BlockResultResponse:
    """Applies the tail normalization to the block text and to each segment."""
    translated = (
        normalized_tail(block.translated_text)
        if block.translated_text is not None
        else None
    )
    segments = tuple(
        replace(segment, translated_text=normalized_tail(segment.translated_text))
        for segment in block.segments
    )
    if translated == block.translated_text and segments == block.segments:
        return block
    return replace(block, translated_text=translated, segments=segments)


def _segment_translations(
    block: BlockResultResponse,
    source: DocumentBlock,
    modes: tuple[BlockTreatment, ...],
) -> tuple[str | None, ...]:
    """One translation slot per span. Unknown footnote markers are stripped."""
    if not modes:
        return ()
    by_index = {segment.index: segment.translated_text for segment in block.segments}
    return tuple(
        None if index not in by_index else known_markers_only(by_index[index], source)
        for index, _mode in enumerate(modes, start=1)
    )


def _joined(translations: tuple[str | None, ...]) -> str | None:
    """Same join `BlockProcessingResult.rebuild_translation` requires."""
    parts = [text for text in translations if text and text.strip()]
    if not parts:
        return None
    return " ".join(parts)


def _foreign_marker_violation(
    block: BlockResultResponse, source: DocumentBlock
) -> str | None:
    """A marker whose reference starts outside this segment's span."""
    if len(source.segments) < 2 or not block.segments:
        return None
    references = footnote_references(source)
    spans = source.segment_spans
    for segment in block.segments:
        if not 1 <= segment.index <= len(spans):
            continue
        span = spans[segment.index - 1]
        for match in _MARKER.finditer(segment.translated_text):
            ordinal = int(match.group(1))
            if not 1 <= ordinal <= len(references):
                continue
            reference = references[ordinal - 1]
            if span.start <= reference.start < span.end:
                continue
            return (
                f"маркер [^{ordinal}] в переводе сегмента {segment.index} "
                f"блока {block.source_block_id} относится к другому сегменту"
            )
    return None


def _payload_with_joined_translation(
    payload: TranslationResultResponse,
) -> TranslationResultResponse:
    """Gives text checks the join while leaving segment fields for result assembly."""
    blocks = []
    for block in payload.blocks:
        if not block.segments:
            blocks.append(block)
            continue
        joined = joined_segment_translation(block)
        if joined is None and block.treatment.requires_translation:
            blocks.append(block)
            continue
        blocks.append(replace(block, segments=(), translated_text=joined))
    return replace(payload, blocks=tuple(blocks))


def _unit_for_domain_check(unit: TranslationUnit) -> TranslationUnit:
    """Unit whose segment modes are the ones stored on the result."""
    items: list[TranslationUnitItem] = []
    changed = False
    for item in unit.items:
        modes = planned_segment_modes(item)
        if modes == item.segment_treatments:
            items.append(item)
            continue
        changed = True
        items.append(replace(item, segment_treatments=modes))
    if not changed:
        return unit
    return replace(unit, items=tuple(items))


def _unit_for_whole_translation(
    unit: TranslationUnit,
    expectation: ResultExpectation,
    payload: TranslationResultResponse,
) -> TranslationUnit:
    """Один промежуток только у цельного ответа с явным разрешением.

    Пустой `segment_treatments` по-прежнему значит, что требований по сегментам
    нет, и сам сегменты не снимает. Ответ со `segments` проверяется по
    сохранённому плану, даже если цельный перевод для этого блока допустим.
    """
    if len(expectation.blocks) != len(unit.items):
        return unit
    answered = {block.source_block_id: block for block in payload.blocks}
    items: list[TranslationUnitItem] = []
    changed = False
    for item, spec in zip(unit.items, expectation.blocks, strict=True):
        answer = answered.get(item.block.id)
        if (
            answer is None
            or answer.segments
            or not spec.whole_translation_allowed
            or len(item.block.segments) < 2
        ):
            items.append(item)
            continue
        changed = True
        items.append(
            replace(
                item,
                block=replace(item.block, segments=()),
                segment_treatments=(),
            )
        )
    if not changed:
        return unit
    return replace(unit, items=tuple(items))


def expectation_for_unit(
    unit: TranslationUnit, *, target_language: str
) -> ResultExpectation:
    """What a retry call for this unit must return (ALG-05).

    Blocks are listed by their native IDs: suitable for use cases that address
    blocks to the model using their IDs. A multi-segment block lists one mode
    per span. A one-span block keeps an empty tuple.
    """
    return ResultExpectation(
        target_language=target_language,
        blocks=tuple(
            ExpectedBlock(
                block_id=item.block.id,
                treatment=item.treatment,
                segment_treatments=planned_segment_modes(item),
            )
            for item in unit.items
        ),
    )


__all__ = ["TranslationResultValidator", "expectation_for_unit"]
