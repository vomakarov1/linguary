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
    treatment_violation,
)
from linguary.application.validators.truncation import truncation_violation, truncation_diagnostic
from linguary.domain import (
    BlockProcessingResult,
    DomainError,
    GlossaryHint,
    LanguageNote,
    ResultWarning,
    ResultWarningKind,
    TranslationUnit,
    known_markers_only,
    without_footnote_markers,
)


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
        expected = expectation or expectation_for_unit(
            unit, target_language=target_language
        )
        self._ensure(coverage_violation(payload, unit=unit), expected)
        self._ensure(treatment_violation(payload, unit=unit), expected)
        self._ensure(content_violation(payload, unit=unit), expected)
        self._ensure(paragraph_violation(payload, unit=unit), expected)
        normalized_payload = replace(
            payload,
            blocks=tuple(
                replace(
                    block,
                    translated_text=normalized_tail(block.translated_text),
                )
                if block.translated_text is not None
                else block
                for block in payload.blocks
            ),
        )
        truncation = truncation_diagnostic(normalized_payload, unit=unit, truncated=truncated)
        if truncation is not None:
            message, violation = truncation
            raise ResultValidationError(message, expectation=expected, violation=violation)
        self._ensure(
            language_violation(
                payload,
                target_language=target_language,
                min_share=self._target_language_min_share,
                min_length=self._language_check_min_length,
            ),
            expected,
        )
        self._ensure(service_message_violation(payload), expected)
        self._ensure(mixed_script_violation(payload), expected)
        results = tuple(
            self._result(
                block,
                unit=unit,
                notes=payload.notes_for(block.source_block_id),
                note_id_factory=note_id_factory,
            )
            for block in payload.blocks
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
        source = next(
            item.block for item in unit.items if item.block.id == block.source_block_id
        )
        return BlockProcessingResult(
            block_id=block.source_block_id,
            treatment=unit.treatment_for(block.source_block_id),
            translated_text=(
                known_markers_only(normalized_tail(block.translated_text), source)
                if block.translated_text is not None
                else None
            ),
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
        )

    def _warnings(self, block: BlockResultResponse) -> tuple[ResultWarning, ...]:
        """Non-blocking warnings for a block result (PRD § 15)."""
        if block.translated_text is None:
            return ()
        repetition = repetition_warning(
            block.translated_text,
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
            unit.validate_results(results)
        except DomainError as error:
            raise ResultValidationError(str(error), expectation=expectation) from error


def _unmarked(text: str) -> str:
    """Text stripped of footnote markers; a field consisting purely of a marker is preserved."""
    unmarked = without_footnote_markers(text)
    if unmarked == text:
        return text
    return unmarked.strip() or text


def expectation_for_unit(
    unit: TranslationUnit, *, target_language: str
) -> ResultExpectation:
    """What a retry call for this unit must return (ALG-05).

    Blocks are listed by their native IDs: suitable for use cases that address
    blocks to the model using their IDs.
    """
    return ResultExpectation(
        target_language=target_language,
        blocks=tuple(
            ExpectedBlock(block_id=item.block.id, treatment=item.treatment)
            for item in unit.items
        ),
    )


__all__ = ["TranslationResultValidator", "expectation_for_unit"]
