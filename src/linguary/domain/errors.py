"""Base domain layer errors."""


class DomainError(Exception):
    """Domain invariant violation.

    Base class for all domain model errors: specific entity errors inherit from it,
    allowing the application layer to distinguish domain invariant violations from
    infrastructure errors.
    """


class IdentifierError(DomainError):
    """Invalid components of a stable identifier."""


class DocumentStructureError(DomainError):
    """Structural invariant violation in canonical document, section, or block."""


class UnknownBlockTypeError(DocumentStructureError):
    """Block type is not in the supported set."""


class UnknownBlockError(DomainError):
    """Access to a `block_id` not present in the known set of blocks."""


class GlossaryError(DomainError):
    """Glossary invariant violation: concept, variant, or candidate."""


class TreatmentPlanError(DomainError):
    """Treatment plan invariant violation for chapter blocks."""


class AnalysisIncompleteError(DomainError):
    """Chapter analysis is incomplete: processing this chapter cannot be started."""


class NonTextualBlockError(DomainError):
    """Non-textual block used where only textual blocks are allowed.

    Blocks with treatments `PRESERVE_ONLY` and `DROP_WITH_WARNING` do not enter
    `TranslationUnit`, are not passed to the model as context, and have no processing result.
    """


class BlockResultError(DomainError):
    """Block processing result does not conform to its treatment mode."""


class TranslationUnitError(DomainError):
    """Computational block group invariant violation."""


class MissingBlockResultError(TranslationUnitError):
    """Model response lacks a result for one of the blocks in the group."""


class BlockOrderError(TranslationUnitError):
    """Result block order does not match source block order."""


class TreatmentMismatchError(TranslationUnitError):
    """Result treatment mode does not match the persisted chapter plan."""


class ProcessingRunError(DomainError):
    """Book processing run invariant violation."""


class ActiveRunConflictError(ProcessingRunError):
    """The book has more than one active `ProcessingRun`."""


class TaskTransitionError(DomainError):
    """Local task state transition is forbidden."""


class BookStatusTransitionError(DomainError):
    """Book state transition is absent from the allowed transitions table."""


class LLMCallLogError(DomainError):
    """LLM call log entry invariant violation."""
