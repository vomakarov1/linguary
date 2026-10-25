"""LLM call port (PRD § 23, ADR § 3).

The application layer only knows `LLMClient.generate`: it constructs the request text
and receives a parsed internal DTO. Provider details, HTTP, structured response formatting,
and Pydantic remain behind the `linguary.adapters.llm` adapter.

The request is typed by response kind (`LLMRequest[ChapterAnalysisResponse]`), so the use case
receives the expected DTO without union discrimination on its side; the adapter selects
the parsing schema by `kind`.

The method is asynchronous: calling the model is network I/O, and the local task runner
executes handlers on the event loop, where configurable concurrency (PRD § 25) is specifically
designed for such waiting.

Port errors are structured so that retry policy (PRD § 16, ALG-05) can make decisions
without inspecting message text:

- `LLMConfigurationError` — nothing to call with: key or model not configured, provider rejected key.
  Retrying is pointless until the user fixes settings;
- `LLMTransportError` — transient failure: timeout, connection drop, rate limit, provider-side error;
- `LLMProviderError` — provider rejected the request itself and would repeat the rejection;
- `LLMResponseFormatError` — response cannot be parsed against schema, including generation truncation
  due to token limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

from linguary.application.errors import ApplicationError
from linguary.application.llm_dto import ChapterAnalysisResponse, TranslationResultResponse
from linguary.domain.llm_log import CallContext, Violation
from linguary.application.quality_context import QualityReviewResponse
from linguary.domain import LLMCallStage, TokenUsage

#: Structured responses that the model can return (PRD § 23).
type LLMPayload = ChapterAnalysisResponse | TranslationResultResponse | QualityReviewResponse


class LLMError(ApplicationError):
    """Calling the model did not yield a usable result."""


class LLMConfigurationError(LLMError):
    """LLM is not configured or the provider rejected configuration.

    Book analysis and processing do not start on this error: Web UI shows configuration state
    and corrective action, while CLI prints a diagnostic message (PRD § 23).
    The error text does not contain provider API keys.
    """


class LLMTransportError(LLMError):
    """Transient call failure: timeout, connection drop, rate limit, provider error.

    `retry_after_seconds` — how long the provider requested to wait before the next request
    (`Retry-After` header of `429` and `503` responses), if reported. Retrying earlier will
    receive the same rejection, so retry policy does not make pauses shorter.
    """

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LLMProviderError(LLMError):
    """Provider rejected the request and will return the same rejection on retry."""


class LLMResponseFormatError(LLMError):
    """Model response cannot be parsed against structured response schema.

    `truncated` indicates the provider reported output truncation due to token limits:
    the response was cut off by output limits rather than semantically by the model,
    and retrying makes sense only with different limits.
    """

    def __init__(
        self,
        message: str,
        *,
        truncated: bool = False,
        raw_content: str | None = None,
        usage: TokenUsage | None = None,
        cost: Decimal | None = None,
        provider: str | None = None,
        model: str | None = None,
        violation: Violation | None = None,
        body_meta: dict | None = None,
        call_context: CallContext | None = None,
    ) -> None:
        super().__init__(message)
        self.truncated = truncated
        self.raw_content = raw_content
        self.usage = usage
        self.cost = cost
        self.provider = provider
        self.model = model
        self.violation = violation
        self.body_meta = body_meta
        self.call_context = call_context
        self.call_id = None


class LLMResponseKind(StrEnum):
    """Kind of structured response requested by the use case."""

    QUALITY_REVIEW = "quality_review"
    CHAPTER_ANALYSIS = "chapter_analysis"
    TRANSLATION_RESULT = "translation_result"


@dataclass(frozen=True, slots=True)
class LLMRequest[PayloadT: LLMPayload]:
    """Request to the model: invariant instructions and assembled context.

    Request text is assembled by `ContextBuilder`: the port knows nothing of chapters
    or glossary. `kind` determines which structured response schema the adapter requires
    from the provider and which DTO it returns to the use case.
    """

    kind: LLMResponseKind
    instructions: str
    input_text: str
    book_id: str | None = None
    stage: LLMCallStage | None = None
    task_id: str | None = None
    attempt: int = 1
    contract_version: int = 1
    call_context: CallContext | None = None
    scope_id: str | None = None
    ordered_block_ids: tuple[str, ...] = ()
    run_id: str | None = None
    max_completion_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.instructions.strip():
            raise LLMError("инструкция запроса к модели пуста")
        if not self.input_text.strip():
            raise LLMError("содержимое запроса к модели пусто")
        if self.attempt < 1:
            raise LLMError("номер попытки вызова модели начинается с единицы")
        if (self.book_id is None) != (self.stage is None):
            raise LLMError("журналируемый запрос должен содержать книгу и этап")
        if self.book_id is not None and not self.book_id.strip():
            raise LLMError("идентификатор книги для журнала LLM пуст")
        if self.task_id is not None and not self.task_id.strip():
            raise LLMError("идентификатор задачи для журнала LLM пуст")

    def with_logging(
        self,
        *,
        book_id: str,
        stage: LLMCallStage,
        task_id: str | None = None,
        scope_id: str | None = None,
        ordered_block_ids: tuple[str, ...] = (),
        run_id: str | None = None,
    ) -> LLMRequest[PayloadT]:
        """Returns the same request with metadata for the call log."""
        return replace(
            self, book_id=book_id, stage=stage, task_id=task_id, attempt=1,
            scope_id=scope_id, ordered_block_ids=ordered_block_ids, run_id=run_id
        )

    @classmethod
    def for_chapter_analysis(
        cls, *, instructions: str, input_text: str, contract_version: int = 1
    ) -> LLMRequest[ChapterAnalysisResponse]:
        """Request for chapter or chapter fragment analysis (PRD § 11, § 14)."""
        return LLMRequest[ChapterAnalysisResponse](
            kind=LLMResponseKind.CHAPTER_ANALYSIS,
            instructions=instructions,
            input_text=input_text,
            contract_version=contract_version,
        )

    @classmethod
    def for_translation(
        cls, *, instructions: str, input_text: str, contract_version: int = 1
    ) -> LLMRequest[TranslationResultResponse]:
        """Request for block group or single block processing (PRD § 10, § 14)."""
        return LLMRequest[TranslationResultResponse](
            kind=LLMResponseKind.TRANSLATION_RESULT,
            instructions=instructions,
            input_text=input_text,
            contract_version=contract_version,
        )


@dataclass(frozen=True, slots=True)
class LLMResponse[PayloadT: LLMPayload]:
    """Parsed model response along with technical call metadata.

    `truncated` — indicates output cutoff by token limits if reported by provider:
    this is a blocking check on results (PRD § 15). `usage` is populated only when
    the provider provides token volumes; the call log permits it to be missing (PRD § 24).
    """

    payload: PayloadT
    provider: str
    model: str
    truncated: bool = False
    usage: TokenUsage | None = None
    call_id: str | None = None
    raw_content: str | None = field(default=None, repr=False)
    cost: Decimal | None = None
    call_context: CallContext | None = None
    body_meta: dict | None = None


class LLMClient(Protocol):
    """Client for calling model with structured response."""

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        """Executes request and returns parsed response.

        The implementation must map any failure to a subclass of `LLMError`: retry
        policy discriminates them by type rather than text (PRD § 16).
        """


__all__ = [
    "LLMClient",
    "LLMConfigurationError",
    "LLMError",
    "LLMPayload",
    "LLMProviderError",
    "LLMRequest",
    "LLMResponse",
    "LLMResponseFormatError",
    "LLMResponseKind",
    "LLMTransportError",
]
