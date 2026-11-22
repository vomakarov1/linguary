"""Chapter analysis use case and analysis task sequence (PRD § 13-14, ADR § 3).

By default, `AnalyzeChapter` executes a single model request per chapter and saves `ChapterAnalysis`:
glossary candidates, chapter translation instructions, block treatment plan, `Before You Read`,
and a brief chapter event summary. No separate preparatory requests are made (for glossary,
plan, or summary), and there is no separate pass for recounting previous chapter events:
the `Previously in Chapter ...` section is assembled by the exporter from chapter `N-1`'s summary
(PRD § 19.2). Opt-in content-plan-v3 analysis runs narrative and pedagogical requests
sequentially, persisting each validated pass before assembling the complete chapter.

A chapter that does not fit into a single request is a fallback scenario (PRD § 14): it is sent
to the model in sequential fragments, and its analysis is assembled from the fragment responses
(`analysis_merge`, ALG-03). The chunking threshold is a configuration setting, not a property
of the use case: it depends on the output limit of the specific model.

The model call is executed outside a transaction, while the result is saved in a single short
transaction (ADR § 3). Retries are managed by `RetryPolicy` (ALG-05): the loop here waits
for backoff before retrying transient errors and appends clarification instructions to the
original request without replacing the assembled context.

`ChapterAnalysisSequence` manages ordering: chapters are analyzed strictly sequentially
in reading order because chapter context includes candidates from previous chapters (PRD § 25).
Ordering is enforced via a queue rather than worker locks: a book has at most one active
`ANALYZE_CHAPTER` task at a time, and the next is enqueued only after saving the previous
analysis. The worker parallelism setting remains shared and does not affect analysis ordering:
only tasks for different books can run in parallel.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast
from uuid import uuid4

from linguary.application.a2_core_vocab import non_core_a2_words
from linguary.application.analysis_merge import (
    DEFAULT_MAX_FRAGMENT_BLOCKS,
    DEFAULT_MAX_FRAGMENT_CHARS,
    DEFAULT_MAX_FRAGMENT_WEIGHT,
    ChapterFragment,
    FragmentAnalysis,
    before_you_read_entries,
    content_plan_from_response,
    validate_content_plan,
    validate_fact_glossary,
    glossary_context_constraints,
    kept_candidate_description,
    merge_chapter_analysis,
    name_translation_evidence,
    plan_fragments,
    plan_items_from_response,
    validate_narrative_response,
    merge_narrative_fragments,
    merged_glossary_suggestion_keys,
    combine_staged_analysis_responses,
    normalize_pedagogical_response,
    unify_chapter_analysis_verse_stanzas,
    select_staged_a2_summary,
)
from linguary.application.analysis_pass_cache import decode_pass, encode_pass, pedagogical_response
from linguary.application.block_labels import without_block_labels
from linguary.application.context_builder import (
    ChapterAnalysisContext,
    ContextBuilder,
    FragmentFindings,
    chapter_note_limit,
    textual_blocks,
    source_text_for_model,
)
from linguary.application.errors import ApplicationError
from linguary.application.reader_text import (
    prepare_analysis_payload,
    record_reader_warnings,
    validate_a2_summary,
)
from linguary.application.glossary_review import candidates_with_decisions
from linguary.application.llm_dto import (
    ChapterAnalysisResponse,
    GlossaryCandidateResponse,
    NarrativeAnalysisResponse,
    PedagogicalAnalysisResponse,
)
from linguary.application.llm_rejections import record_rejection, finalize_response
from linguary.application.ports.llm import LLMClient, LLMRequest, LLMResponseFormatError
from linguary.application.ports.language_detection import LanguageDetector
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.application.retry_policy import ResultValidationError, RetryPolicy
from linguary.application.validators.mixed_script import mixed_script_word
from linguary.application.task_runner import TaskQueue
from linguary.domain.content_plan import ContentPlan, ContextFact, Scene, canonical_sha
from linguary.domain.book_conditions import thaw
from linguary.domain.document import _PART_DIVIDER_PATTERN
from linguary.domain.errors import DomainError
from linguary.domain.name_translation import NameTranslationPolicy
from linguary.domain import (
    BeforeYouReadEntry,
    BlockTreatment,
    BookEvent,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    DocumentBlock,
    DocumentSection,
    GlossaryCandidate,
    ProcessingTask,
    TaskIdentity,
    TaskStatus,
    TaskType,
    TokenUsage,
    LLMCallStage,
    TreatmentPlan,
    TermConcept,
    apply_book_event,
    glossary_key,
    AnalysisPassKind,
    AnalysisPassResult,
)

#: Sleep between model call attempts. The application layer does not own the event loop,
#: but must execute the sleep: the retry policy only calculates the duration (ALG-05).
type Sleep = Callable[[float], Awaitable[None]]

#: Factory for IDs of persisted analysis records.
type IdFactory = Callable[[], str]

_UNSET = object()

_A2_VOCAB_NAME_TYPES = frozenset({
    ConceptType.PERSON,
    ConceptType.PLACE,
    ConceptType.ORGANIZATION,
})


class AnalyzeChapterError(ApplicationError):
    """The chapter cannot be analyzed in its current state."""


@dataclass(frozen=True, slots=True)
class AnalyzeChapterCommand:
    """Analysis command: which chapter of which book to analyze."""

    book_id: str
    section_id: str
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class AnalyzeChapterResult:
    """Chapter analysis result along with technical call metadata.

    `attempts` and `usage` are used for observability metrics (PRD § 29.1): how many
    model calls the chapter took and the token usage reported by the provider.
    An empty chapter requires no analysis and has no model call: `attempts` is 0.
    """

    analysis: ChapterAnalysis
    attempts: int = 0
    usage: TokenUsage | None = None


class AnalyzeChapter:
    """Analyzes a single chapter and persists its `ChapterAnalysis`."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        llm_client: LLMClient,
        language_detector: LanguageDetector,
        retry_policy: RetryPolicy,
        glossary_confidence_threshold: float,
        max_fragment_blocks: int = DEFAULT_MAX_FRAGMENT_BLOCKS,
        context_builder: ContextBuilder | None = None,
        id_factory: IdFactory | None = None,
        sleep: Sleep | None = None,
        name_translation_policy: NameTranslationPolicy | None = None,
        conditions_id: str | None = None,
        max_concurrency: int = 1,
        seed_first_chapter: bool = False,
        sequential_analysis: bool = True,
        max_output_tokens: int | None = 32_768,
        staged_analysis: bool = False,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._llm_client = llm_client
        self._language_detector = language_detector
        self._retry_policy = retry_policy
        self._glossary_confidence_threshold = glossary_confidence_threshold
        self._max_fragment_blocks = max_fragment_blocks
        self._context_builder = context_builder or ContextBuilder()
        self._name_translation_policy = name_translation_policy or self._context_builder.name_translation_policy
        self._conditions_id = conditions_id
        self._max_concurrency = max_concurrency
        self._seed_first_chapter = seed_first_chapter
        self._sequential_analysis = bool(sequential_analysis)
        self._max_output_tokens = int(max_output_tokens) if max_output_tokens is not None else 32_768
        self._staged_analysis = staged_analysis
        self._id_factory = id_factory or _new_id
        self._sleep = sleep or asyncio.sleep
        from contextvars import ContextVar
        self._loaded_concepts: ContextVar[tuple[TermConcept, ...] | None] = ContextVar(
            f"analyze_chapter_concepts_{id(self)}", default=None
        )
        self._resolved_content_plan: ContextVar[ContentPlan | None] = ContextVar(
            f"analyze_chapter_content_plan_{id(self)}", default=None
        )
        self._staged_attempt_ids: ContextVar[tuple[str, ...]] = ContextVar(
            f"analyze_chapter_staged_attempts_{id(self)}", default=()
        )

    async def __call__(self, command: AnalyzeChapterCommand) -> AnalyzeChapterResult:
        """Analyzes a chapter and persists the result.

        A chapter fitting in a single request is analyzed directly. A longer chapter
        is sent to the model in sequential fragments, and its analysis is assembled
        from their responses (PRD § 14, ALG-03).
        """
        token_attempts = None
        token_responses = None
        token_concepts = self._loaded_concepts.set(None)
        token_plan = self._resolved_content_plan.set(None)
        token_staged_attempts = self._staged_attempt_ids.set(())
        attempts_var: Any = getattr(self._llm_client, "_attempts", None)
        responses_var: Any = getattr(self._llm_client, "_responses", None)
        if attempts_var is not None:
            token_attempts = attempts_var.set(())
        if responses_var is not None:
            token_responses = responses_var.set(())
        try:
            document, accumulated, user_decisions, nonbinding_suggestions = self._load(command)
            concepts = self._loaded_concepts.get()
            section = _section_of(document, command)
            blocks = textual_blocks(section)
            if not blocks:
                analysis = _empty_analysis(section, id=self._id_factory())
                self._save(command.book_id, analysis, task_id=command.task_id)
                return AnalyzeChapterResult(analysis=analysis)
            if document.is_front_matter(section):
                if _has_meaningful_part_title(document, section):
                    analysis, attempts, usage = await self._analyze_part_divider(
                        command=command,
                        document=document,
                        section=section,
                        blocks=blocks,
                        accumulated=accumulated,
                        user_decisions=user_decisions,
                        nonbinding_suggestions=nonbinding_suggestions,
                        concepts=concepts or (),
                    )
                    self._save(command.book_id, analysis, task_id=command.task_id)
                    return AnalyzeChapterResult(analysis=analysis, attempts=attempts, usage=usage)
                analysis = self._front_matter_analysis(section, blocks, id=self._id_factory())
                self._save(command.book_id, analysis, task_id=command.task_id)
                return AnalyzeChapterResult(analysis=analysis)
            if self._staged_analysis and self._context_builder._content_plan_profile == 'content-plan-v3':
                analysis, attempts, usage = await self._analyze_staged(
                    command, document, section, accumulated, user_decisions, nonbinding_suggestions,
                    concepts or ())
                self._save(command.book_id, analysis, task_id=command.task_id)
                return AnalyzeChapterResult(analysis=analysis, attempts=attempts, usage=usage)
            fragments = list(plan_fragments(
                blocks, max_blocks=self._max_fragment_blocks, max_chars=DEFAULT_MAX_FRAGMENT_CHARS,
            ))
            if len(fragments) == 1:
                attempts = 1
                try:
                    analysis, attempts, usage = await self._analyze_whole_chapter(
                        document=document,
                        section=section,
                        blocks=blocks,
                        accumulated=accumulated,
                        user_decisions=user_decisions,
                        nonbinding_suggestions=nonbinding_suggestions,
                        command=command,
                        concepts=concepts,
                    )
                except LLMResponseFormatError as error:
                    if error.truncated and len(blocks) > 1:
                        mid = len(blocks) // 2
                        initial_fragments = [
                            ChapterFragment(order=0, blocks=tuple(blocks[:mid])),
                            ChapterFragment(order=1, blocks=tuple(blocks[mid:])),
                        ]
                        analysis, fragment_attempts, fragment_usage = await self._analyze_by_fragments(
                            document=document,
                            section=section,
                            blocks=blocks,
                            fragments=initial_fragments,
                            accumulated=accumulated,
                            user_decisions=user_decisions,
                            nonbinding_suggestions=nonbinding_suggestions,
                            command=command,
                            concepts=concepts,
                        )
                        attempts += fragment_attempts
                        usage = _total_usage([error.usage, fragment_usage])
                    else:
                        raise
            else:
                analysis, attempts, usage = await self._analyze_by_fragments(
                    document=document,
                    section=section,
                    blocks=blocks,
                    fragments=fragments,
                    accumulated=accumulated,
                    user_decisions=user_decisions,
                    nonbinding_suggestions=nonbinding_suggestions,
                    command=command,
                    concepts=concepts,
                )
            self._save(command.book_id, analysis, task_id=command.task_id)
            return AnalyzeChapterResult(analysis=analysis, attempts=attempts, usage=usage)
        finally:
            self._staged_attempt_ids.reset(token_staged_attempts)
            self._resolved_content_plan.reset(token_plan)
            self._loaded_concepts.reset(token_concepts)
            if token_attempts is not None and attempts_var is not None:
                attempts_var.reset(token_attempts)
            if token_responses is not None and responses_var is not None:
                responses_var.reset(token_responses)

    @staticmethod
    def _to_aliases(context: ChapterAnalysisContext, message: str) -> str:
        return _to_aliases(context, message)

    def _part_divider_treatment_plan(
        self,
        section: DocumentSection,
        blocks: Sequence[DocumentBlock],
        candidates: Sequence[GlossaryCandidate],
    ) -> TreatmentPlan:
        """Treatment plan for part dividers with meaningful titles (P7.T2)."""
        if self._context_builder._english_help:
            return TreatmentPlan.of_treatments(
                section.id,
                ((block.id, BlockTreatment.READ_AS_IS) for block in blocks),
            )
        is_a2 = self._is_a2_reader()
        has_candidates = bool(candidates)
        treatments: list[tuple[str, BlockTreatment]] = []
        for block in blocks:
            collapsed = " ".join(block.text.split())
            remainder = _PART_DIVIDER_PATTERN.sub(" ", collapsed)
            if not any(char.isalpha() for char in remainder):
                treatment = BlockTreatment.READ_AS_IS
            elif is_a2 and len(_WORD_TOKEN_RE.findall(remainder)) > 3:
                treatment = BlockTreatment.TRANSLATE
            elif has_candidates or (is_a2 and bool(non_core_a2_words(block.text))):
                treatment = BlockTreatment.GLOSSARY_ONLY
            else:
                treatment = BlockTreatment.READ_AS_IS
            treatments.append((block.id, treatment))
        return TreatmentPlan.of_treatments(section.id, treatments)

    async def _analyze_part_divider(
        self,
        *,
        command: AnalyzeChapterCommand,
        document: CanonicalDocument,
        section: DocumentSection,
        blocks: Sequence[DocumentBlock],
        accumulated: Sequence[GlossaryCandidate],
        user_decisions: Sequence[TermConcept],
        nonbinding_suggestions: Sequence[TermConcept],
        concepts: Sequence[TermConcept],
    ) -> tuple[ChapterAnalysis, int, TokenUsage | None]:
        """Lightweight glossary-only analysis for part dividers with meaningful titles."""
        analysis_id = self._id_factory()
        base = self._front_matter_analysis(section, blocks, id=analysis_id)
        if self._staged_analysis and self._context_builder._content_plan_profile == 'content-plan-v3':
            narrative, _saved, _scenes, _facts, attempts, usage, _deps, _cond = (
                await self._run_staged_narrative_pass(
                    command,
                    document,
                    section,
                    accumulated,
                    user_decisions,
                    nonbinding_suggestions,
                    concepts,
                )
            )
            candidates = self._candidates(narrative, analysis_id=analysis_id)
            return (
                replace(
                    base,
                    treatment_plan=self._part_divider_treatment_plan(
                        section, blocks, candidates
                    ),
                    glossary_candidates=candidates,
                ),
                attempts,
                usage,
            )
        context = self._context_builder.for_chapter_analysis(
            document=document,
            section=section,
            accumulated_candidates=accumulated,
            user_decisions=user_decisions,
            nonbinding_suggestions=nonbinding_suggestions,
        )
        context = replace(
            context,
            before_you_read_limit=None,
            language_notes_per_chapter=None,
            note_limits_by_block=None,
            note_group_limits=(),
            request=context.request.with_logging(
                book_id=command.book_id,
                stage=LLMCallStage.CHAPTER_ANALYSIS,
                task_id=command.task_id,
                scope_id=section.id,
                ordered_block_ids=tuple(block.id for block in blocks),
            ),
        )
        payload, attempts, usage = await self._request_analysis(
            context,
            blocks=blocks,
            source_language=document.source_language,
            concepts=concepts,
            part_divider=True,
        )
        candidates = self._candidates(payload, analysis_id=analysis_id)
        return (
            replace(
                base,
                treatment_plan=self._part_divider_treatment_plan(
                    section, blocks, candidates
                ),
                glossary_candidates=candidates,
            ),
            attempts,
            usage,
        )

    def _should_enforce_v7_name_policy(
        self,
        command: AnalyzeChapterCommand,
        *,
        conditions: object = _UNSET,
    ) -> bool:
        """Whether cached narrative validation must enforce V7 name-translation policy checks."""
        resolved_conditions = conditions
        if resolved_conditions is _UNSET:
            with self._unit_of_work.transaction() as tx:
                work = getattr(tx, 'work', None)
                resolved_conditions = work.latest_conditions(command.book_id) if work else None
        conditions_payload = getattr(resolved_conditions, 'payload', None)
        if resolved_conditions is not None and isinstance(conditions_payload, Mapping):
            versions = conditions_payload.get('versions')
            if isinstance(versions, Mapping):
                prompts_version = versions.get('prompts')
                if isinstance(prompts_version, int) and prompts_version < 7:
                    return False
        return True

    def _narrative_max_fragment_blocks(self) -> int:
        """Maximum blocks per narrative fragment scaled by output token budget."""
        return min(self._max_fragment_blocks, max(20, self._max_output_tokens // 650))

    def _pedagogical_max_fragment_blocks(self) -> int:
        """Maximum blocks per pedagogical fragment scaled by output token budget."""
        return min(self._max_fragment_blocks, max(40, self._max_output_tokens // 400))

    async def _run_staged_narrative_pass(
        self, command: AnalyzeChapterCommand, document: CanonicalDocument, section: DocumentSection,
        accumulated: Sequence[GlossaryCandidate], user_decisions: Sequence[TermConcept],
        suggestions: Sequence[TermConcept], concepts: Sequence[TermConcept],
    ) -> tuple[
        NarrativeAnalysisResponse,
        AnalysisPassResult,
        tuple[Scene, ...],
        tuple[ContextFact, ...],
        int,
        TokenUsage | None,
        Mapping[str, object],
        str | None,
    ]:
        blocks = textual_blocks(section)
        aliases = {b.id: b.id for b in blocks}
        texts = {b.id: b.text for b in blocks}
        visible = {b.id: source_text_for_model(b) for b in blocks}
        is_notes_only = section.is_notes_only or document.is_front_matter(section)
        with self._unit_of_work.transaction() as tx:
            work = getattr(tx, 'work', None)
            conditions = work.latest_conditions(command.book_id) if work else None
            previous_narrative = tx.analyses.get_pass(command.book_id, section.id, AnalysisPassKind.NARRATIVE)
            list_passes_for_book = getattr(tx.analyses, 'list_passes_for_book', None)
            book_narratives = (
                list_passes_for_book(command.book_id, pass_kind=AnalysisPassKind.NARRATIVE, scope_key="")
                if callable(list_passes_for_book)
                else ((previous_narrative,) if previous_narrative is not None else ())
            )
        # A suggestion generated by this pass or sibling chapter passes is output, not a new
        # upstream dependency. Keep suggestions that were already inputs even if repeated.
        cached_narrative: NarrativeAnalysisResponse | None = None
        previous_inputs: set[tuple[str, str, str]] = set()
        own_outputs: set[tuple[str, str, str]] = set()
        if previous_narrative is not None:
            verified_narrative = self._verify_cached_pass(previous_narrative)
            if isinstance(verified_narrative, NarrativeAnalysisResponse):
                cached_narrative = verified_narrative
                prev_payload: Mapping[str, Any] = previous_narrative.payload
                previous_inputs.update(tuple(item) for item in prev_payload.get('input_suggestions', ()))
                own_outputs.update(_candidate_suggestion_key(item) for item in
                    prev_payload.get('response', {}).get('glossary_candidates', ()))
                own_outputs.update(merged_glossary_suggestion_keys(cached_narrative.glossary_candidates))
        for other_pass in book_narratives:
            if other_pass is None or (previous_narrative is not None and other_pass.id == previous_narrative.id):
                continue
            try:
                other_decoded = self._verify_cached_pass(other_pass)
            except AnalyzeChapterError:
                continue
            other_payload: Mapping[str, Any] = other_pass.payload
            own_outputs.update(_candidate_suggestion_key(item) for item in
                other_payload.get('response', {}).get('glossary_candidates', ()))
            if isinstance(other_decoded, NarrativeAnalysisResponse):
                own_outputs.update(merged_glossary_suggestion_keys(other_decoded.glossary_candidates))
        if own_outputs:
            own_outputs.update(merged_glossary_suggestion_keys(accumulated))
            suggestions = tuple(s for s in suggestions
                if _suggestion_key(s, document.target_language) not in own_outputs - previous_inputs)
        known_names = _known_names_from_glossary(
            accumulated,
            user_decisions,
            suggestions,
            target_language=document.target_language,
        )
        expected_conditions_id = (getattr(self._llm_client, 'conditions_id', None)
                                  or self._conditions_id or (conditions.id if conditions else None))
        dependencies = conditions.payload if conditions else {}
        narrative_hash = self._context_builder.narrative_input_sha256(document, section,
            accumulated, user_decisions, suggestions,
            model_dependencies=_pass_model_dependencies(dependencies, AnalysisPassKind.NARRATIVE),
            max_fragment_blocks=self._narrative_max_fragment_blocks())
        if previous_narrative is not None and previous_narrative.input_sha256 == narrative_hash:
            saved = previous_narrative
            narrative = cached_narrative
        else:
            saved = None
            narrative = None
        attempts = 0
        usages: list[TokenUsage | None] = []
        if saved is None or narrative is None:
            narrative_parts, count, usage = await self._run_staged_fragments(
                command, document, section, accumulated, user_decisions, suggestions, concepts,
                pass_kind=AnalysisPassKind.NARRATIVE,
                chapter_input_sha=narrative_hash,
                expected_conditions_id=expected_conditions_id,
                known_names=known_names)
            attempts += count
            usages.append(usage)
            narrative, scenes, facts = merge_narrative_fragments(
                narrative_parts, section_blocks=section.ordered_blocks
            )
            enforce_v7 = self._should_enforce_v7_name_policy(command, conditions=conditions)
            merged_name_policy = self._name_translation_policy if enforce_v7 else None
            scenes, facts = validate_narrative_response(narrative, aliases, texts, visible,
                ordered_block_ids=tuple(aliases), is_notes_only=is_notes_only,
                mandatory_facts=glossary_context_constraints(concepts),
                blocks=blocks, source_offsets=True,
                source_language=document.source_language,
                target_language=document.target_language,
                language_detector=self._language_detector,
                known_names=known_names,
                name_translation_policy=merged_name_policy,
                enforce_strict_gender_subject=enforce_v7)
            saved = self._save_pass(command, AnalysisPassKind.NARRATIVE, narrative_hash,
                                    narrative, expected_conditions_id,
                                    input_suggestions=tuple(_suggestion_key(s, document.target_language)
                                                            for s in suggestions))
        else:
            if saved.attempt_id is not None:
                self._staged_attempt_ids.set((*self._staged_attempt_ids.get(), saved.attempt_id))
            enforce_v7 = self._should_enforce_v7_name_policy(command, conditions=conditions)
            cached_name_policy = self._name_translation_policy if enforce_v7 else None
            scenes, facts = validate_narrative_response(narrative, aliases, texts, visible,
                ordered_block_ids=tuple(aliases), is_notes_only=is_notes_only,
                mandatory_facts=glossary_context_constraints(concepts),
                blocks=blocks, source_offsets=True,
                source_language=document.source_language,
                target_language=document.target_language,
                language_detector=self._language_detector,
                known_names=known_names,
                name_translation_policy=cached_name_policy,
                enforce_strict_gender_subject=enforce_v7)
        return narrative, saved, scenes, facts, attempts, _total_usage(usages), dependencies, expected_conditions_id

    async def _analyze_staged(
        self, command: AnalyzeChapterCommand, document: CanonicalDocument, section: DocumentSection,
        accumulated: Sequence[GlossaryCandidate], user_decisions: Sequence[TermConcept],
        suggestions: Sequence[TermConcept], concepts: Sequence[TermConcept],
    ) -> tuple[ChapterAnalysis, int, TokenUsage | None]:
        blocks = textual_blocks(section)
        aliases = {b.id: b.id for b in blocks}
        narrative, saved, scenes, facts, attempts, narrative_usage, dependencies, expected_conditions_id = (
            await self._run_staged_narrative_pass(
                command, document, section, accumulated, user_decisions, suggestions, concepts,
            )
        )
        usages: list[TokenUsage | None] = [narrative_usage]
        cumulative_words = _cumulative_words_before(document, section)
        prior_chapter_topics = self._load_prior_chapter_topics(command, document, section)
        pedagogical_known_names = _a2_vocab_known_names_with_narrative(
            _a2_vocab_known_names_from_glossary(
                accumulated,
                user_decisions,
                (*suggestions, *concepts),
            ),
            narrative,
            scenes,
        )
        pedagogical_hash = self._context_builder.pedagogical_input_sha256(document, section,
            narrative_output_sha256=saved.output_sha256, narrative_summary=narrative.chapter_summary,
            narrative_instructions=narrative.translation_instructions,
            context_facts=facts, glossary_candidates=narrative.glossary_candidates,
            accumulated_candidates=accumulated, user_decisions=user_decisions,
            model_dependencies=_pass_model_dependencies(dependencies, AnalysisPassKind.PEDAGOGICAL),
            max_fragment_blocks=self._pedagogical_max_fragment_blocks(),
            cumulative_words_before=cumulative_words,
            prior_chapter_topics=prior_chapter_topics)
        cached_pedagogical = self._cached_pass(command, AnalysisPassKind.PEDAGOGICAL, pedagogical_hash)
        if cached_pedagogical is None:
            parts, count, usage = await self._run_staged_fragments(
                command, document, section, accumulated, user_decisions, suggestions, concepts,
                pass_kind=AnalysisPassKind.PEDAGOGICAL, narrative=narrative, facts=facts,
                chapter_input_sha=pedagogical_hash,
                expected_conditions_id=expected_conditions_id,
                cumulative_words_before=cumulative_words,
                known_names=pedagogical_known_names,
                prior_chapter_topics=prior_chapter_topics)
            attempts += count
            usages.append(usage)
            merged = merge_chapter_analysis(parts, id=self._id_factory(), section_id=section.id,
                expected_block_ids=tuple(aliases),
                glossary_confidence_threshold=self._glossary_confidence_threshold,
                id_factory=self._id_factory)
            reader_level = 'a2' if self._is_a2_reader() else self._context_builder._reader_level
            pedagogical = pedagogical_response(merged, reader_level=reader_level)
            if reader_level == 'a2' and len(parts) > 1:
                pedagogical = replace(
                    pedagogical,
                    reader_chapter_summary=select_staged_a2_summary(
                        parts, narrative_summary=narrative.chapter_summary
                    ),
                )
            analysis = self._assemble_staged(narrative, pedagogical, scenes, document, section,
                                             accumulated, user_decisions,
                                             concepts=concepts,
                                             cumulative_words_before=cumulative_words,
                                             known_names=pedagogical_known_names,
                                             prior_chapter_topics=prior_chapter_topics)
            self._save_pass(command, AnalysisPassKind.PEDAGOGICAL, pedagogical_hash,
                            pedagogical, expected_conditions_id)
        else:
            saved_pedagogical, pedagogical = cached_pedagogical
            assert isinstance(pedagogical, PedagogicalAnalysisResponse)
            if saved_pedagogical.attempt_id is not None:
                self._staged_attempt_ids.set((*self._staged_attempt_ids.get(), saved_pedagogical.attempt_id))
            analysis = self._assemble_staged(narrative, pedagogical, scenes, document, section,
                                             accumulated, user_decisions,
                                             concepts=concepts,
                                             cumulative_words_before=cumulative_words,
                                             known_names=pedagogical_known_names,
                                             prior_chapter_topics=prior_chapter_topics)
        return analysis, attempts, _total_usage(usages)

    def _assemble_staged(
        self, narrative: NarrativeAnalysisResponse, pedagogical: PedagogicalAnalysisResponse,
        scenes: Sequence[Scene], document: CanonicalDocument, section: DocumentSection,
        accumulated: Sequence[GlossaryCandidate], user_decisions: Sequence[TermConcept],
        *,
        concepts: Sequence[TermConcept] = (),
        cumulative_words_before: int = 0,
        known_names: Sequence[str] = (),
        prior_chapter_topics: Sequence[str] = (),
    ) -> ChapterAnalysis:
        blocks = textual_blocks(section)
        context = self._context_builder.for_chapter_pedagogical_analysis(document, section,
            narrative_summary=narrative.chapter_summary,
            narrative_instructions=narrative.translation_instructions,
            glossary_candidates=narrative.glossary_candidates,
            accumulated_candidates=accumulated,
            user_decisions=user_decisions,
            cumulative_words_before=cumulative_words_before,
            prior_chapter_topics=prior_chapter_topics)
        # Both saved DTOs use stable references scoped to this chapter.
        context = replace(context, block_ids_by_alias={b.id: b.id for b in blocks},
            visible_texts_by_alias={b.id: source_text_for_model(b) for b in blocks},
            expectation=replace(context.expectation,
                blocks=tuple(replace(e, block_id=b.id) for e, b in
                             zip(context.expectation.blocks, blocks, strict=True))))
        resolved_known_names = (
            tuple(known_names)
            if known_names
            else _a2_vocab_known_names_with_narrative(
                _a2_vocab_known_names_from_glossary(
                    accumulated,
                    user_decisions,
                    concepts,
                ),
                narrative,
                scenes,
            )
        )
        pedagogical = self._normalize_a2_dialogue(
            pedagogical, blocks, context.block_ids_by_alias, known_names=resolved_known_names,
        )
        if self._is_a2_reader():
            validate_a2_summary(pedagogical.reader_chapter_summary or '',
                source_texts=tuple(b.text for b in blocks), source_language=document.source_language,
                language_detector=self._language_detector,
                require_parallel_russian=self._a2_summary_requires_parallel_russian(),
                narrative_summary=narrative.chapter_summary)
        mandatory_facts = glossary_context_constraints((*concepts, *user_decisions))
        payload = combine_staged_analysis_responses(
            narrative,
            pedagogical,
            resolved_scenes=scenes,
            resolved_blocks=blocks,
            mandatory_facts=mandatory_facts,
        )
        _ensure_plan_covers_chapter(payload, context)
        plan = self._validate_pedagogical(
            payload, context, blocks, source_offsets=True, mandatory_facts=mandatory_facts,
        )
        analysis = merge_chapter_analysis([FragmentAnalysis.of_response(payload, order=0,
            block_ids_by_alias=context.block_ids_by_alias, texts={b.id: b.text for b in blocks},
            visible_texts_by_alias=context.visible_texts_by_alias, resolved_content_plan=plan,
            blocks=blocks)],
            id=self._id_factory(), section_id=section.id, expected_block_ids=tuple(b.id for b in blocks),
            glossary_confidence_threshold=self._glossary_confidence_threshold,
            id_factory=self._id_factory, name_translation_policy=self._name_translation_policy,
            conditions_id=self._conditions_id)
        return replace(analysis, content_plan=plan, scenes=tuple(scenes))

    def _a2_summary_requires_parallel_russian(self) -> bool:
        """Preserve legacy two-paragraph validation for books with frozen old prompt templates."""
        templates = getattr(self._context_builder, "_prompt_templates", None)
        if not isinstance(templates, Mapping):
            return False
        return any(
            isinstance(value, str) and "ровно два абзаца" in value
            for value in templates.values()
        )

    def _validate_pedagogical(
        self, payload: ChapterAnalysisResponse, context: ChapterAnalysisContext,
        blocks: Sequence[DocumentBlock], *, source_offsets: bool = False,
        mandatory_facts: Mapping[str, Mapping[str, str]] | None = None,
    ) -> ContentPlan | None:
        return validate_content_plan(payload, context.block_ids_by_alias, {b.id: b.text for b in blocks},
            english_help=self._context_builder._english_help,
            before_you_read_limit=context.before_you_read_limit,
            language_notes_per_chapter=context.language_notes_per_chapter,
            note_limits_by_block=context.note_limits_by_block, note_group_limits=context.note_group_limits,
            visible_texts_by_alias=context.visible_texts_by_alias, blocks=blocks,
            source_offsets=source_offsets, mandatory_facts=mandatory_facts)

    def _is_a2_reader(self) -> bool:
        """Whether the configured reader level is A2 (case- and whitespace-insensitive)."""
        return (getattr(self._context_builder, '_reader_level', '') or '').strip().lower() == 'a2'

    def _normalize_a2_dialogue(
        self,
        payload,
        blocks,
        block_ids_by_alias,
        *,
        known_names: Sequence[str] = (),
    ):
        if not isinstance(payload, (ChapterAnalysisResponse, PedagogicalAnalysisResponse)):
            return payload
        return normalize_pedagogical_response(
            payload,
            blocks,
            block_ids_by_alias,
            reader_level=getattr(self._context_builder, '_reader_level', None),
            english_help=bool(self._context_builder._english_help),
            known_names=known_names,
        )

    async def _run_staged_fragments(
        self, command: AnalyzeChapterCommand, document: CanonicalDocument, section: DocumentSection,
        accumulated: Sequence[GlossaryCandidate], user_decisions: Sequence[TermConcept],
        suggestions: Sequence[TermConcept], concepts: Sequence[TermConcept], *, pass_kind: AnalysisPassKind,
        narrative: NarrativeAnalysisResponse | None = None, facts: Sequence[ContextFact] = (),
        chapter_input_sha: str | None = None, expected_conditions_id: str | None = None,
        cumulative_words_before: int = 0,
        known_names: Sequence[str] = (),
        prior_chapter_topics: Sequence[str] = (),
    ) -> tuple[list, int, TokenUsage | None]:
        is_narrative = pass_kind is AnalysisPassKind.NARRATIVE
        if is_narrative:
            fragments = plan_fragments(
                textual_blocks(section),
                max_blocks=self._narrative_max_fragment_blocks(),
                max_chars=DEFAULT_MAX_FRAGMENT_CHARS,
                block_overhead=450,
                max_weight=DEFAULT_MAX_FRAGMENT_WEIGHT,
            )
        else:
            fragments = plan_fragments(
                textual_blocks(section),
                max_blocks=self._pedagogical_max_fragment_blocks(),
                max_chars=DEFAULT_MAX_FRAGMENT_CHARS,
                block_overhead=120,
                max_weight=DEFAULT_MAX_FRAGMENT_WEIGHT,
            )
        saved_fragment_passes: tuple[AnalysisPassResult, ...] = ()
        with self._unit_of_work.transaction() as tx:
            list_passes = getattr(tx.analyses, 'list_passes_for_section', None)
            if callable(list_passes):
                saved_passes = list_passes(command.book_id, section.id)
                saved_fragment_passes = tuple(
                    saved_pass
                    for saved_pass in saved_passes
                    if saved_pass.pass_kind is pass_kind and saved_pass.scope_key
                )
        notes = chapter_note_limit(section, self._context_builder._language_notes_per_chapter)
        queue: list[tuple[ChapterFragment, int, int]] = []
        for fragment in fragments:
            byr_quota = 0 if is_narrative else 10 // len(fragments) + (fragment.order < 10 % len(fragments))
            note_quota = 0 if is_narrative else notes // len(fragments) + (fragment.order < notes % len(fragments))
            recovered = _recover_cached_fragment_split(fragment, saved_fragment_passes)
            if len(recovered) > 1:
                byr_shares = _distribute_quota(byr_quota, recovered, len(fragment.blocks))
                note_shares = _distribute_quota(note_quota, recovered, len(fragment.blocks))
                for sub_fragment, sub_byr, sub_note in zip(
                    recovered, byr_shares, note_shares, strict=True
                ):
                    queue.append((sub_fragment, sub_byr, sub_note))
            else:
                queue.append((fragment, byr_quota, note_quota))
        ped_known_names = (
            ()
            if is_narrative
            else _a2_vocab_known_names_with_narrative(known_names, narrative)
        )
        parts, usages = [], []
        attempts = 0
        while queue:
            fragment, byr_quota, note_quota = queue.pop(0)
            fragment = replace(fragment, order=len(parts))
            total = len(parts) + 1 + len(queue)
            if is_narrative:
                previous = None
                if parts:
                    raw, _, _ = parts[-1]
                    instruction_lines = [p[0].translation_instructions for p in parts if p[0].translation_instructions]
                    last_scene = next((s[-1] for _, s, _ in reversed(parts) if s), None)
                    if last_scene is not None:
                        scene_info = json.dumps(
                            dict(
                                summary=last_scene.summary,
                                location=last_scene.location,
                                participants=list(last_scene.participants),
                                visual_facts=[
                                    dict(quote=f.anchor.quote, subject=f.subject, detail=f.detail)
                                    for f in last_scene.visual_facts
                                ],
                                visual_assumptions=list(last_scene.visual_assumptions),
                            ),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        instruction_lines.append('Последняя сцена: ' + scene_info)
                    previous = FragmentFindings(
                        summary=raw.chapter_summary,
                        instructions='\n'.join(instruction_lines),
                        expressions=(),
                        candidates=_merge_fragment_candidates(parts),
                    )
                context = self._context_builder.for_chapter_narrative_analysis(document, section,
                    accumulated, user_decisions, suggestions, fragment=fragment if total > 1 else None,
                    fragment_count=total, findings=previous)
            else:
                assert narrative is not None
                prior_topics = _prior_explained_topics(parts)
                context = self._context_builder.for_chapter_pedagogical_analysis(document, section,
                    narrative_summary=narrative.chapter_summary,
                    narrative_instructions=narrative.translation_instructions,
                    context_facts=facts,
                    glossary_candidates=narrative.glossary_candidates,
                    accumulated_candidates=accumulated, user_decisions=user_decisions,
                    fragment=fragment if total > 1 else None, fragment_count=total,
                    before_you_read_limit=byr_quota, language_notes_limit=note_quota,
                    cumulative_words_before=cumulative_words_before,
                    prior_explained_topics=prior_topics,
                    prior_chapter_topics=prior_chapter_topics)
                if parts:
                    previous_summary = (
                        select_staged_a2_summary(
                            parts, narrative_summary=narrative.chapter_summary
                        )
                        if self._is_a2_reader()
                        else parts[-1].chapter_summary
                    )
                    context = replace(context, request=replace(context.request,
                        input_text=context.request.input_text + '\nУже найдено в предыдущих частях главы:\n' +
                            'Пересказ: ' + previous_summary + '\nВыражения BYR: ' +
                            ', '.join(e.expression for p in parts for e in p.before_you_read)))
            context = replace(context, request=context.request.with_logging(book_id=command.book_id,
                stage=LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE if is_narrative else LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL,
                task_id=command.task_id, scope_id=section.id, ordered_block_ids=fragment.block_ids))

            fragment_known_names = known_names
            if is_narrative and parts:
                prior_part_names = [
                    name
                    for part_payload, part_scenes, _ in parts
                    for name in (
                        *(c.source_text for c in part_payload.glossary_candidates),
                        *(c.suggested_translation for c in part_payload.glossary_candidates),
                        *(p for scene in part_scenes for p in scene.participants),
                    )
                    if isinstance(name, str) and name.strip()
                ]
                if prior_part_names:
                    fragment_known_names = tuple(dict.fromkeys((*known_names, *prior_part_names)))

            def validate(
                payload,
                *,
                from_cache: bool = False,
                _context=context,
                _fragment=fragment,
                _fragment_known_names=fragment_known_names,
            ):
                if is_narrative:
                    _ensure_candidates_single_script(payload, _context)
                    enforce_v7 = not from_cache or self._should_enforce_v7_name_policy(command)
                    name_policy = self._name_translation_policy if enforce_v7 else None
                    return validate_narrative_response(payload, _context.block_ids_by_alias,
                        {b.id: b.text for b in _fragment.blocks}, _context.visible_texts_by_alias,
                        ordered_block_ids=_fragment.block_ids,
                        is_notes_only=section.is_notes_only or document.is_front_matter(section),
                        mandatory_facts=glossary_context_constraints(concepts),
                        blocks=_fragment.blocks,
                        source_language=document.source_language,
                        target_language=document.target_language,
                        language_detector=self._language_detector,
                        known_names=_fragment_known_names,
                        name_translation_policy=name_policy,
                        enforce_strict_gender_subject=enforce_v7)
                assert narrative is not None
                combined = combine_staged_analysis_responses(
                    NarrativeAnalysisResponse(narrative.translation_instructions, narrative.chapter_summary), payload)
                if self._is_a2_reader():
                    validate_a2_summary(payload.reader_chapter_summary or '',
                        source_texts=tuple(b.text for b in _fragment.blocks), source_language=document.source_language,
                        language_detector=self._language_detector,
                        require_parallel_russian=self._a2_summary_requires_parallel_russian(),
                        narrative_summary=narrative.chapter_summary)
                _ensure_plan_covers_chapter(combined, _context)
                resolved_plan = self._validate_pedagogical(combined, _context, _fragment.blocks)
                # The first fragment has no prior parts, but segment indexes are still rejected here.
                candidate_part = FragmentAnalysis.of_response(
                    combined,
                    order=len(parts),
                    block_ids_by_alias=_context.block_ids_by_alias,
                    texts={b.id: b.text for b in _fragment.blocks},
                    visible_texts_by_alias=_context.visible_texts_by_alias,
                    resolved_content_plan=resolved_plan,
                    blocks=_fragment.blocks,
                )
                if parts:
                    expected_so_far = tuple(
                        item.block_id
                        for part in (*parts, candidate_part)
                        for item in part.treatments
                    )
                    merge_chapter_analysis(
                        [*parts, candidate_part],
                        id=self._id_factory(),
                        section_id=section.id,
                        expected_block_ids=expected_so_far,
                        glossary_confidence_threshold=self._glossary_confidence_threshold,
                        id_factory=self._id_factory,
                    )
                return resolved_plan

            frag_scope_key = f"{fragment.blocks[0].id}:{fragment.blocks[-1].id}" if total > 1 else ""
            frag_input_sha = (
                canonical_sha(
                    dict(
                        schema_version=1,
                        chapter_input_sha256=chapter_input_sha,
                        scope_key=frag_scope_key,
                        instructions=context.request.instructions,
                        input=_normalize_fragment_header_for_sha(context.request.input_text),
                        contract_version=context.request.contract_version,
                        before_you_read_limit=context.before_you_read_limit,
                        notes=context.language_notes_per_chapter,
                    )
                )
                if frag_scope_key and chapter_input_sha is not None
                else None
            )
            cached_frag = (
                self._cached_pass(command, pass_kind, frag_input_sha, scope_key=frag_scope_key)
                if frag_input_sha is not None
                else None
            )
            if cached_frag is not None:
                saved_frag, payload = cached_frag
                if saved_frag.attempt_id is not None:
                    self._staged_attempt_ids.set((*self._staged_attempt_ids.get(), saved_frag.attempt_id))
                if not is_narrative:
                    payload = self._normalize_a2_dialogue(
                        payload,
                        fragment.blocks,
                        context.block_ids_by_alias,
                        known_names=ped_known_names,
                    )
                resolved = validate(payload, from_cache=True)
            else:
                try:
                    payload, resolved, count, usage = await self._request_staged(
                        context,
                        validate,
                        blocks=fragment.blocks,
                        normalize_dialogue=not is_narrative,
                        known_names=ped_known_names,
                    )
                except LLMResponseFormatError as error:
                    if not error.truncated or len(fragment.blocks) <= 1:
                        raise
                    attempts += getattr(error, 'analysis_attempts', 1)
                    usages.append(error.usage)
                    midpoint = len(fragment.blocks) // 2
                    # Accepted siblings already own their shares. Split only this parent's
                    # remaining allocation, including on repeated truncation of its children.
                    queue[:0] = [
                        (ChapterFragment(0, fragment.blocks[:midpoint]), (byr_quota + 1) // 2, (note_quota + 1) // 2),
                        (ChapterFragment(0, fragment.blocks[midpoint:]), byr_quota // 2, note_quota // 2)]
                    continue
                attempts += count
                usages.append(usage)
                if frag_input_sha is not None:
                    self._save_pass(
                        command,
                        pass_kind,
                        frag_input_sha,
                        payload,
                        expected_conditions_id,
                        scope_key=frag_scope_key,
                        input_suggestions=tuple(_suggestion_key(s, document.target_language) for s in suggestions)
                        if is_narrative
                        else (),
                    )
            if is_narrative:
                parts.append((payload, *resolved))
            else:
                assert narrative is not None
                assert isinstance(payload, PedagogicalAnalysisResponse)
                combined = combine_staged_analysis_responses(
                    NarrativeAnalysisResponse(narrative.translation_instructions, narrative.chapter_summary), payload)
                parts.append(FragmentAnalysis.of_response(combined, order=len(parts),
                    block_ids_by_alias=context.block_ids_by_alias,
                    texts={b.id: b.text for b in fragment.blocks},
                    visible_texts_by_alias=context.visible_texts_by_alias,
                    resolved_content_plan=resolved, blocks=fragment.blocks))
        return parts, attempts, _total_usage(usages)

    async def _request_staged[ResolutionT](
        self, context: ChapterAnalysisContext,
        validate: Callable[[NarrativeAnalysisResponse | PedagogicalAnalysisResponse], ResolutionT],
        *,
        blocks: Sequence[DocumentBlock] = (),
        normalize_dialogue: bool = False,
        known_names: Sequence[str] = (),
    ) -> tuple[NarrativeAnalysisResponse | PedagogicalAnalysisResponse, ResolutionT, int, TokenUsage | None]:
        request, attempts, usages = context.request, 0, []
        while True:
            attempts += 1
            response = None
            try:
                response = await self._llm_client.generate(request)
                usages.append(response.usage)
                if response.truncated:
                    raise LLMResponseFormatError('ответ модели оборван по лимиту выходных токенов',
                        raw_content=response.raw_content, truncated=True, usage=response.usage,
                        cost=response.cost, provider=response.provider, model=response.model,
                        call_context=response.call_context)
                try:
                    response = replace(response, payload=_prepare_staged_payload(response.payload))
                    if normalize_dialogue:
                        response = replace(response, payload=self._normalize_a2_dialogue(
                            response.payload,
                            blocks,
                            context.block_ids_by_alias,
                            known_names=known_names,
                        ))
                    resolved = validate(response.payload)
                    if isinstance(response.payload, PedagogicalAnalysisResponse):
                        record_reader_warnings(self._unit_of_work, response,
                            [('summary', response.payload.reader_chapter_summary or ''),
                             *(('before_you_read.explanation', e.explanation) for e in response.payload.before_you_read)],
                            reader_level=self._context_builder._reader_level)
                except (ApplicationError, DomainError) as error:
                    raise ResultValidationError(str(error), expectation=context.expectation,
                        violation=getattr(error, 'violation', None)) from error
            except Exception as error:
                if response is not None:
                    record_rejection(self._unit_of_work, response, error)
                    reject_last = getattr(self._llm_client, 'reject_last', None)
                    if callable(reject_last):
                        reject_last()
                if isinstance(error, LLMResponseFormatError) and error.truncated:
                    cast(Any, error).analysis_attempts = attempts
                    error.usage = _total_usage([*usages, error.usage] if response is None else usages)
                    raise
                decision = self._retry_policy.decide(_worded_for_the_model(error, context),
                    attempts=attempts, raw_content=response.raw_content if response else None)
                if not decision.should_retry:
                    raise
                if decision.backoff_seconds > 0:
                    await self._sleep(decision.backoff_seconds)
                request = decision.next_request(request)
                continue
            return response.payload, resolved, attempts, _total_usage(usages)

    def _cached_pass(
        self, command: AnalyzeChapterCommand, kind: AnalysisPassKind, input_sha: str,
        *, scope_key: str = "",
    ) -> tuple[AnalysisPassResult, NarrativeAnalysisResponse | PedagogicalAnalysisResponse] | None:
        with self._unit_of_work.transaction() as tx:
            saved = tx.analyses.get_pass(command.book_id, command.section_id, kind, scope_key=scope_key)
        if saved is None or saved.input_sha256 != input_sha:
            return None
        decoded = self._verify_cached_pass(saved)
        return saved, decoded

    def _verify_cached_pass(
        self, saved: AnalysisPassResult
    ) -> NarrativeAnalysisResponse | PedagogicalAnalysisResponse:
        kind = saved.pass_kind
        try:
            decoded = decode_pass(saved.payload, narrative=kind is AnalysisPassKind.NARRATIVE)
            normalized = encode_pass(decoded)
            if kind is AnalysisPassKind.NARRATIVE:
                keys = saved.payload['input_suggestions']
                if (not isinstance(keys, (tuple, list)) or any(
                    not isinstance(item, (tuple, list)) or len(item) != 3
                    or any(not isinstance(part, str) for part in item) for item in keys)):
                    raise ValueError('неверный список входных подсказок')
                normalized['input_suggestions'] = keys
            if (canonical_sha(thaw(saved.payload)) != saved.output_sha256
                    or canonical_sha(normalized) != saved.output_sha256):
                raise ValueError('не совпадает контрольная сумма')
            return decoded
        except (ValueError, TypeError, KeyError, AttributeError, ApplicationError, DomainError) as error:
            raise AnalyzeChapterError(f'сохранённый проход {kind.value} повреждён или несовместим') from error

    def _save_pass(
        self, command: AnalyzeChapterCommand, kind: AnalysisPassKind, input_sha: str,
        payload: NarrativeAnalysisResponse | PedagogicalAnalysisResponse, conditions_id: str | None,
        *, scope_key: str = "", input_suggestions: Sequence[tuple[str, str, str]] = (),
    ) -> AnalysisPassResult:
        from linguary.application.book_conditions import ConditionsError
        try:
            return self._save_pass_current(command, kind, input_sha, payload, conditions_id,
                                           scope_key=scope_key, input_suggestions=input_suggestions)
        except ConditionsError:
            self._finalize_stale_responses()
            raise

    def _save_pass_current(
        self, command: AnalyzeChapterCommand, kind: AnalysisPassKind, input_sha: str,
        payload: NarrativeAnalysisResponse | PedagogicalAnalysisResponse, conditions_id: str | None,
        *, scope_key: str = "", input_suggestions: Sequence[tuple[str, str, str]] = (),
    ) -> AnalysisPassResult:
        from linguary.application.book_conditions import ConditionsError
        encoded = encode_pass(payload)
        if kind is AnalysisPassKind.NARRATIVE:
            encoded['input_suggestions'] = input_suggestions
        with self._unit_of_work.transaction() as tx:
            work = getattr(tx, 'work', None)
            if conditions_id is not None and work is not None:
                latest = work.latest_conditions(command.book_id)
                if latest is None or latest.id != conditions_id:
                    raise ConditionsError('conditions_revision_conflict')
                service = getattr(self._llm_client, 'service', None)
                if service is not None:
                    service.verify_source(tx.books.get(command.book_id), latest)
                if command.task_id is not None and tx.tasks.get(command.task_id) is None:
                    raise ConditionsError('task_no_longer_active')
            attempts = getattr(self._llm_client, 'attempts', ())
            saved = AnalysisPassResult(id=self._id_factory(), book_id=command.book_id,
                section_id=command.section_id, pass_kind=kind, scope_key=scope_key, input_sha256=input_sha,
                output_sha256=canonical_sha(encoded), conditions_id=conditions_id,
                attempt_id=attempts[-1] if attempts else None, payload=encoded)
            if not scope_key:
                previous = tx.analyses.get_pass(command.book_id, command.section_id, kind)
                if previous is not None and work is not None:
                    work.archive_artifacts([dict(book_id=command.book_id, kind='analysis_pass',
                        scope_id=command.section_id + ':' + kind.value, conditions_id=previous.conditions_id,
                        attempt_id=previous.attempt_id, body=dict(schema_version=1, pass_kind=kind.value,
                            input_sha256=previous.input_sha256, output_sha256=previous.output_sha256,
                            payload=thaw(previous.payload)))])
                delete_fragments = getattr(tx.analyses, 'delete_fragment_passes_for_section', None)
                if callable(delete_fragments):
                    delete_fragments(command.book_id, command.section_id, kind)
            tx.analyses.save_pass(saved)
            for response in getattr(self._llm_client, 'responses', ()):
                attempt = tx.work.get_attempt(response.call_context.attempt_id)
                if attempt.state == 'validated':
                    finalize_response(self._unit_of_work, response, 'accepted', transaction=tx)
        return saved

    def _front_matter_analysis(
        self, section: DocumentSection, blocks: Sequence[DocumentBlock], *, id: str
    ) -> ChapterAnalysis:
        profile = getattr(self._context_builder, "_content_plan_profile", "legacy-v1")
        reader_level = getattr(self._context_builder, "_reader_level", "")
        plan = None
        if profile == "content-plan-v3":
            plan = ContentPlan(version=3, profile="content-plan-v3", difficulties=(), before_you_read_evidence=(), context_facts=())
        elif profile == "content-plan-v2":
            plan = ContentPlan(version=2, profile="content-plan-v2", difficulties=(), before_you_read_evidence=(), context_facts=())
        elif reader_level == "a2":
            plan = ContentPlan(version=3, profile="content-plan-v3", difficulties=(), before_you_read_evidence=(), context_facts=())
        return ChapterAnalysis.create(
            id=id,
            section_id=section.id,
            treatment_plan=TreatmentPlan.of_treatments(
                section.id,
                ((b.id, BlockTreatment.READ_AS_IS) for b in blocks),
            ),
            translation_instructions="",
            chapter_summary="",
            expected_block_ids=[block.id for block in blocks],
            glossary_candidates=(),
            before_you_read=(),
            content_plan=plan,
        )

    async def _analyze_whole_chapter(
        self,
        *,
        document: CanonicalDocument,
        section: DocumentSection,
        blocks: Sequence[DocumentBlock],
        accumulated: Sequence[GlossaryCandidate],
        user_decisions: Sequence[TermConcept],
        nonbinding_suggestions: Sequence[TermConcept],
        command: AnalyzeChapterCommand,
        concepts: Sequence[TermConcept] | None = None,
    ) -> tuple[ChapterAnalysis, int, TokenUsage | None]:
        """Main scenario: entire chapter in a single request (PRD § 14)."""
        context = self._context_builder.for_chapter_analysis(
            document=document, section=section, accumulated_candidates=accumulated,
            user_decisions=user_decisions, nonbinding_suggestions=nonbinding_suggestions,
        )
        context = replace(
            context,
            request=context.request.with_logging(
                book_id=command.book_id,
                stage=LLMCallStage.CHAPTER_ANALYSIS,
                task_id=command.task_id,
                scope_id=section.id,
                ordered_block_ids=tuple(block.id for block in blocks),
            ),
        )
        known_names = _a2_vocab_known_names_from_glossary(
            accumulated,
            user_decisions,
            (*nonbinding_suggestions, *(concepts or ())),
        )
        payload, attempts, usage = await self._request_analysis(
            context,
            blocks=blocks,
            source_language=document.source_language,
            concepts=concepts,
            known_names=known_names,
        )
        analysis = self._to_analysis(
            payload,
            context=context,
            section=section,
            blocks=blocks,
            resolved_content_plan=self._resolved_content_plan.get(),
        )
        return analysis, attempts, usage

    async def _analyze_by_fragments(
        self,
        *,
        document: CanonicalDocument,
        section: DocumentSection,
        blocks: Sequence[DocumentBlock],
        fragments: Sequence[ChapterFragment],
        accumulated: Sequence[GlossaryCandidate],
        user_decisions: Sequence[TermConcept],
        nonbinding_suggestions: Sequence[TermConcept],
        command: AnalyzeChapterCommand,
        concepts: Sequence[TermConcept] | None = None,
    ) -> tuple[ChapterAnalysis, int, TokenUsage | None]:
        """Fallback scenario: long chapter processed by fragments (PRD § 14, ALG-03).

        Fragments are requested sequentially, each with its own blocks and retry loop.
        A fragment that fails after all attempts raises the original error: the chapter
        analysis is not saved at all rather than saved incomplete, preventing incomplete
        results from reaching the user (ALG-03).
        """
        analysed: list[FragmentAnalysis] = []
        attempts = 0
        usages: list[TokenUsage | None] = []
        queue: list[ChapterFragment] = list(fragments)
        base_known_names = _a2_vocab_known_names_from_glossary(
            accumulated,
            user_decisions,
            (*nonbinding_suggestions, *(concepts or ())),
        )
        while queue:
            fragment = queue.pop(0)
            previous = None
            if analysed:
                previous = FragmentFindings(
                    summary=next(
                        (
                            part.chapter_summary
                            for part in reversed(analysed)
                            if part.chapter_summary
                        ),
                        "",
                    ),
                    instructions="\n".join(
                        part.translation_instructions
                        for part in analysed
                        if part.translation_instructions
                    ),
                    expressions=tuple(
                        entry.expression
                        for part in analysed
                        for entry in part.before_you_read
                    ),
                    candidates=_merge_fragment_candidates(analysed),
                )
            total_fragments = len(analysed) + 1 + len(queue)
            context = self._context_builder.for_chapter_fragment(
                document=document,
                section=section,
                fragment=fragment,
                fragment_count=total_fragments,
                accumulated_candidates=accumulated,
                user_decisions=user_decisions,
                nonbinding_suggestions=nonbinding_suggestions,
                findings=previous,
            )
            context = replace(
                context,
                request=context.request.with_logging(
                    book_id=command.book_id,
                    stage=LLMCallStage.CHAPTER_ANALYSIS,
                    task_id=command.task_id,
                    scope_id=section.id,
                    ordered_block_ids=tuple(block.id for block in fragment.blocks),
                ),
            )
            fragment_known_names = tuple(
                dict.fromkeys(
                    (
                        *base_known_names,
                        *(
                            c.source_text.strip()
                            for part in analysed
                            for c in part.glossary_candidates
                            if c.type in _A2_VOCAB_NAME_TYPES and c.source_text.strip()
                        ),
                    )
                )
            )
            try:
                payload, fragment_attempts, usage = await self._request_analysis(
                    context,
                    blocks=fragment.blocks,
                    source_language=document.source_language,
                    concepts=concepts,
                    known_names=fragment_known_names,
                )
            except LLMResponseFormatError as error:
                if error.truncated and len(fragment.blocks) > 1:
                    attempts += 1
                    if error.usage is not None:
                        usages.append(error.usage)
                    mid = len(fragment.blocks) // 2
                    left = ChapterFragment(order=0, blocks=tuple(fragment.blocks[:mid]))
                    right = ChapterFragment(order=0, blocks=tuple(fragment.blocks[mid:]))
                    queue = [left, right, *queue]
                    for idx, item in enumerate(queue):
                        queue[idx] = replace(item, order=len(analysed) + idx)
                    continue
                raise

            attempts += fragment_attempts
            usages.append(usage)
            analysed.append(
                FragmentAnalysis.of_response(
                    payload,
                    order=fragment.order,
                    block_ids_by_alias=context.block_ids_by_alias,
                    texts={b.id: b.text for b in fragment.blocks},
                    visible_texts_by_alias=context.visible_texts_by_alias,
                    resolved_content_plan=self._resolved_content_plan.get(),
                    blocks=fragment.blocks,
                )
            )
        analysed = [replace(part, order=i) for i, part in enumerate(analysed)]
        analysis = merge_chapter_analysis(
            analysed,
            id=self._id_factory(),
            section_id=section.id,
            expected_block_ids=[block.id for block in blocks],
            glossary_confidence_threshold=self._glossary_confidence_threshold,
            id_factory=self._id_factory,
            name_translation_policy=self._name_translation_policy,
            conditions_id=self._conditions_id,
        )
        analysis = unify_chapter_analysis_verse_stanzas(
            analysis,
            blocks,
            english_help=bool(self._context_builder._english_help),
        )
        return analysis, attempts, _total_usage(usages)

    def _is_sequential_active(self, transaction: UnitOfWork, book_id: str) -> bool:
        work_repo = getattr(transaction, "work", None)
        conditions = (
            work_repo.latest_conditions(book_id)
            if work_repo is not None
            else None
        )
        if conditions is not None and isinstance(getattr(conditions, "payload", None), Mapping):
            return bool(
                (conditions.payload.get("checks") or {}).get(
                    "sequential_analysis", self._sequential_analysis
                )
            )
        return bool(self._sequential_analysis)

    def _parallel_batch_and_seed(
        self, transaction: UnitOfWork, command: AnalyzeChapterCommand
    ) -> tuple[bool, bool]:
        if self._is_sequential_active(transaction, command.book_id):
            return False, False
        tasks_repo = getattr(transaction, "tasks", None)
        parallel_batch = self._max_concurrency > 1 or (
            tasks_repo is not None
            and any(
                task.task_type is TaskType.ANALYZE_CHAPTER
                and task.scope_id != command.section_id
                and task.status is TaskStatus.RUNNING
                for task in tasks_repo.list_for_book(command.book_id)
            )
        )
        work_repo = getattr(transaction, "work", None)
        conditions = (
            work_repo.latest_conditions(command.book_id)
            if work_repo is not None
            else None
        )
        if conditions is not None and isinstance(getattr(conditions, "payload", None), Mapping):
            seed_active = bool(
                (conditions.payload.get("checks") or {}).get(
                    "seed_first_chapter", self._seed_first_chapter
                )
            )
        else:
            seed_active = bool(self._seed_first_chapter)
        return parallel_batch, seed_active

    def _load_prior_chapter_topics(
        self,
        command: AnalyzeChapterCommand,
        document: CanonicalDocument,
        section: DocumentSection,
    ) -> tuple[str, ...]:
        with self._unit_of_work.transaction() as transaction:
            analyses = transaction.analyses.list_for_book(command.book_id)
            parallel_batch, seed_active = self._parallel_batch_and_seed(transaction, command)
        return _prior_chapter_topics(
            document,
            section,
            analyses,
            parallel_batch=parallel_batch,
            seed_active=seed_active,
        )

    def _load(
        self, command: AnalyzeChapterCommand
    ) -> tuple[CanonicalDocument, tuple[GlossaryCandidate, ...], tuple[TermConcept, ...], tuple[TermConcept, ...]]:
        """Reads the book and accumulated glossary decisions for the book.

        The user's decision on a candidate overrides the model's suggestion: analysis repeats
        accumulated decisions verbatim and must not return a modified version.

        Candidates from previous chapters are all included, with the model suggestion where no decision exists.
        Candidates from the current and subsequent chapters are included only with user decisions:
        decisions apply book-wide, and re-analyzing a chapter after glossary review must see them (PRD § 11),
        while model suggestions for subsequent chapters provide no value to this chapter.
        `list_for_book` itself preserves reading order.
        """
        with self._unit_of_work.transaction() as transaction:
            document = transaction.documents.get(command.book_id)
            if document is None:
                raise AnalyzeChapterError(f"книги {command.book_id} нет в хранилище")
            current_order = _section_of(document, command).order
            orders = {section.id: section.order for section in document.sections}
            analyses = transaction.analyses.list_for_book(command.book_id)
            concepts = tuple(transaction.glossary.list_for_book(command.book_id))
            self._loaded_concepts.set(concepts)
            parallel_batch, seed_active = self._parallel_batch_and_seed(transaction, command)
            seed_section = _first_content_section(document) if (parallel_batch and seed_active) else None
            seed_section_ids: set[str] = set()
            if seed_section is not None:
                if current_order >= seed_section.order:
                    seed_section_ids.update(
                        section.id
                        for section in document.ordered_sections
                        if section.order < seed_section.order
                        and _is_meaningful_part_divider(document, section)
                    )
                if seed_section.order < current_order:
                    seed_section_ids.add(seed_section.id)
            seed_candidates = (
                tuple(
                    candidate
                    for analysis in analyses
                    if analysis.section_id in seed_section_ids
                    for candidate in analysis.glossary_candidates
                )
                if seed_section_ids
                else ()
            )
            if self._context_builder.name_translation_policy is not None:
                accumulated = (
                    seed_candidates
                    if parallel_batch
                    else tuple(
                        candidate
                        for analysis in analyses
                        if orders.get(analysis.section_id, current_order) < current_order
                        for candidate in analysis.glossary_candidates
                    )
                )
                decisions = tuple(concept for concept in concepts if concept.status.is_user_decision)
                suggestions = tuple(concept for concept in concepts if not concept.status.is_user_decision)
                return document, accumulated, decisions, suggestions
            if parallel_batch:
                all_candidates = tuple(
                    candidate
                    for analysis in analyses
                    for candidate in analysis.glossary_candidates
                )
                non_seed_candidates = (
                    tuple(
                        candidate
                        for analysis in analyses
                        if analysis.section_id not in seed_section_ids
                        for candidate in analysis.glossary_candidates
                    )
                    if seed_section_ids
                    else all_candidates
                )
                known_keys = {glossary_key(c.source_text) for c in all_candidates}
                concept_candidates = tuple(
                    GlossaryCandidate.create(
                        id=concept.id,
                        analysis_id="user-decision",
                        source_text=concept.canonical_source_text,
                        type=concept.type,
                        suggested_translation=variant.text,
                        confidence=1.0,
                        model_flagged_review=False,
                        confidence_threshold=self._glossary_confidence_threshold,
                        note=concept.note,
                    )
                    for concept in concepts
                    if concept.status.is_user_decision
                    and glossary_key(concept.canonical_source_text) not in known_keys
                    and (variant := concept.variant_for(document.target_language)) is not None
                )
                accumulated = candidates_with_decisions(
                    seed_candidates,
                    concepts,
                    language=document.target_language,
                    decided_only=(*non_seed_candidates, *concept_candidates),
                )
            else:
                accumulated = candidates_with_decisions(
                    (
                        candidate
                        for analysis in analyses
                        if orders.get(analysis.section_id, current_order) < current_order
                        for candidate in analysis.glossary_candidates
                    ),
                    concepts,
                    language=document.target_language,
                    decided_only=(
                        candidate
                        for analysis in analyses
                        if orders.get(analysis.section_id, current_order) >= current_order
                        for candidate in analysis.glossary_candidates
                    ),
                )
        return document, accumulated, (), ()

    async def _request_analysis(
        self,
        context: ChapterAnalysisContext,
        *,
        blocks: Sequence[DocumentBlock],
        source_language: str,
        concepts: Sequence[TermConcept] | None = None,
        part_divider: bool = False,
        known_names: Sequence[str] = (),
    ) -> tuple[ChapterAnalysisResponse, int, TokenUsage | None]:
        """Executes model call with retries per policy (ALG-05).

        When attempts are exhausted, raises the original error: typing the error and
        transitioning the task to `failed` is the responsibility of the runner, not this loop.
        """
        request: LLMRequest[ChapterAnalysisResponse] = cast(
            LLMRequest[ChapterAnalysisResponse], context.request
        )
        resolved_concepts = (
            tuple(concepts) if concepts is not None else self._loaded_concepts.get()
        )
        attempts = 0
        while True:
            attempts += 1
            response = None
            try:
                response = await self._llm_client.generate(request)
                if response.truncated:
                    raise LLMResponseFormatError(
                        "ответ модели оборван по лимиту выходных токенов",
                        raw_content=response.raw_content,
                        truncated=True,
                        usage=response.usage,
                        cost=response.cost,
                        provider=response.provider,
                        model=response.model,
                        call_context=response.call_context,
                    )
                try:
                    if request.contract_version in {3, 5}:
                        response = replace(response, payload=prepare_analysis_payload(response.payload))
                    if part_divider:
                        _ensure_candidates_single_script(response.payload, context)
                        self._resolved_content_plan.set(None)
                    else:
                        payload_scenes = (
                            *getattr(response.payload, "scenes", ()),
                            *(
                                getattr(response.payload.content_plan, "scenes", ())
                                if response.payload.content_plan is not None
                                else ()
                            ),
                        )
                        response_known_names = tuple(
                            dict.fromkeys(
                                (
                                    *known_names,
                                    *(
                                        c.source_text.strip()
                                        for c in response.payload.glossary_candidates
                                        if c.type in _A2_VOCAB_NAME_TYPES and c.source_text.strip()
                                    ),
                                    *(
                                        participant.strip()
                                        for scene in payload_scenes
                                        for participant in getattr(scene, "participants", ())
                                        if isinstance(participant, str) and participant.strip()
                                    ),
                                )
                            )
                        )
                        response = replace(response, payload=self._normalize_a2_dialogue(
                            response.payload,
                            blocks,
                            context.block_ids_by_alias,
                            known_names=response_known_names,
                        ))
                        if self._context_builder._reader_level == "a2":
                            validate_a2_summary(
                                response.payload.chapter_summary,
                                source_texts=tuple(block.text for block in blocks),
                                source_language=source_language,
                                language_detector=self._language_detector,
                                require_parallel_russian=self._a2_summary_requires_parallel_russian(),
                            )
                        if request.contract_version in {3, 5}:
                            record_reader_warnings(self._unit_of_work, response,
                                [('summary', response.payload.chapter_summary),
                                 *(( 'before_you_read.explanation', entry.explanation) for entry in response.payload.before_you_read)],
                                reader_level=self._context_builder._reader_level)
                        _ensure_plan_covers_chapter(response.payload, context)
                        plan_items_from_response(
                            response.payload,
                            block_ids_by_alias=context.block_ids_by_alias,
                            blocks=blocks,
                            require_all=True,
                        )
                        _ensure_candidates_single_script(response.payload, context)
                        plan = validate_content_plan(
                            response.payload, context.block_ids_by_alias, {b.id: b.text for b in blocks},
                            english_help=self._context_builder._english_help,
                            before_you_read_limit=context.before_you_read_limit,
                            language_notes_per_chapter=context.language_notes_per_chapter,
                            note_limits_by_block=context.note_limits_by_block,
                            note_group_limits=context.note_group_limits,
                            visible_texts_by_alias=context.visible_texts_by_alias,
                            blocks=blocks,
                        )
                        if request.contract_version in {3, 5}:
                            if resolved_concepts is None:
                                if request.book_id is not None:
                                    with self._unit_of_work.transaction() as transaction:
                                        resolved_concepts = tuple(transaction.glossary.list_for_book(request.book_id))
                                else:
                                    resolved_concepts = ()
                            validate_fact_glossary(plan, glossary_context_constraints(resolved_concepts))
                        self._resolved_content_plan.set(plan)
                except (ApplicationError, DomainError) as error:
                    if not isinstance(error, ResultValidationError) or error.expectation is None:
                        raise ResultValidationError(
                            str(error),
                            expectation=context.expectation,
                            violation=getattr(error, "violation", None),
                        ) from error
                    raise
            except Exception as error:
                if response is not None:
                    record_rejection(self._unit_of_work, response, error)
                    reject_last = getattr(self._llm_client, 'reject_last', None)
                    if callable(reject_last):
                        reject_last()
                decision = self._retry_policy.decide(
                    _worded_for_the_model(error, context),
                    attempts=attempts,
                    raw_content=response.raw_content if response is not None else None,
                )
                if not decision.should_retry:
                    raise
                if decision.backoff_seconds > 0:
                    await self._sleep(decision.backoff_seconds)
                request = decision.next_request(request)
                continue
            return response.payload, attempts, response.usage

    def _to_analysis(
        self,
        payload: ChapterAnalysisResponse,
        *,
        context: ChapterAnalysisContext,
        section: DocumentSection,
        blocks: Sequence[DocumentBlock],
        resolved_content_plan: ContentPlan | None = None,
    ) -> ChapterAnalysis:
        """Converts model response to a domain chapter analysis.

        Request aliases are mapped back to stable block IDs, and the plan
        is assembled in original document order: the model response does not decide
        chapter block ordering.
        """
        analysis_id = self._id_factory()
        return ChapterAnalysis.create(
            id=analysis_id,
            section_id=section.id,
            treatment_plan=TreatmentPlan(
                section_id=section.id,
                items=plan_items_from_response(
                    payload,
                    block_ids_by_alias=context.block_ids_by_alias,
                    blocks=blocks,
                    require_all=True,
                ),
            ),
            translation_instructions=without_block_labels(payload.translation_instructions),
            chapter_summary=payload.chapter_summary if payload.content_plan is not None and payload.content_plan.version == 3 else without_block_labels(payload.chapter_summary),
            expected_block_ids=[block.id for block in blocks],
            glossary_candidates=self._candidates(payload, analysis_id=analysis_id),
            before_you_read=_before_you_read(payload),
            content_plan=(
                resolved_content_plan
                if resolved_content_plan is not None or payload.content_plan is None
                else content_plan_from_response(
                    payload, context.block_ids_by_alias, {b.id: b.text for b in blocks},
                    visible_texts_by_alias=context.visible_texts_by_alias,
                    blocks=blocks,
                )
            ),
        )

    def _candidates(
        self, payload: ChapterAnalysisResponse | NarrativeAnalysisResponse, *, analysis_id: str
    ) -> tuple[GlossaryCandidate, ...]:
        """Glossary candidates of the chapter with computed `requires_review` (DTO-01).

        The model's flag remains an input signal: whether a candidate requires user review
        is decided by the application under PRD § 12 rules.
        """
        return tuple(
            replace(GlossaryCandidate.create(
                id=self._id_factory(),
                analysis_id=analysis_id,
                source_text=candidate.source_text,
                type=candidate.type,
                suggested_translation=candidate.suggested_translation,
                confidence=candidate.confidence,
                model_flagged_review=candidate.model_flagged_review,
                confidence_threshold=self._glossary_confidence_threshold,
                note=candidate.note,
            ), name_translation_evidence=name_translation_evidence(
                (candidate,), policy=self._name_translation_policy,
                conditions_id=self._conditions_id,
            ), description=kept_candidate_description(candidate.type, (candidate,)))
            for candidate in payload.glossary_candidates
        )

    def _save(self, book_id: str, analysis: ChapterAnalysis, *, task_id=None) -> None:
        from linguary.application.book_conditions import ConditionsError
        try:
            self._save_current(book_id, analysis, task_id=task_id)
        except ConditionsError:
            self._finalize_stale_responses()
            raise

    def _finalize_stale_responses(self) -> None:
        responses = getattr(self._llm_client, 'responses', ())
        if not responses:
            return
        with self._unit_of_work.transaction() as transaction:
            for response in responses:
                if response.call_context is None:
                    continue
                attempt = transaction.work.get_attempt(response.call_context.attempt_id)
                if (attempt is not None and attempt.state == 'validated'
                        and attempt.outcome.get('decision') not in {'accepted', 'rejected', 'stale'}):
                    finalize_response(self._unit_of_work, response, 'stale', transaction=transaction)

    def _save_current(self, book_id: str, analysis: ChapterAnalysis, *, task_id=None) -> None:
        """Saves chapter analysis in a single short transaction."""
        with self._unit_of_work.transaction() as transaction:
            conditions_id = getattr(self._llm_client, 'conditions_id', None)
            if conditions_id is not None:
                from linguary.application.book_conditions import BookConditionsService, ConditionsError
                conditions = transaction.work.latest_conditions(book_id)
                if conditions is None or conditions.id != conditions_id:
                    raise ConditionsError('conditions_revision_conflict')
                book = transaction.books.get(book_id)
                client_service = getattr(self._llm_client, 'service', None)
                if client_service is not None:
                    client_service.verify_source(book, conditions)
                if task_id is not None and transaction.tasks.get(task_id) is None:
                    raise ConditionsError('task_no_longer_active')
                attempts = getattr(self._llm_client, 'attempts', ())
                staged_attempts = self._staged_attempt_ids.get()
                attempt_id = (
                    attempts[-1]
                    if attempts
                    else (staged_attempts[-1] if staged_attempts else None)
                )
                BookConditionsService.archive_analysis(transaction, book_id, analysis, conditions_id, attempt_id)
                for response in getattr(self._llm_client, 'responses', ()):
                    current_attempt = transaction.work.get_attempt(response.call_context.attempt_id)
                    if current_attempt.state == 'validated':
                        finalize_response(self._unit_of_work, response, 'accepted', transaction=transaction)
                for accepted in attempts:
                    current = transaction.work.get_attempt(accepted)
                    if current.state == 'validated':
                        transaction.work.transition_attempt(accepted, 'accepted', expected_state='validated')
            transaction.analyses.save(book_id, analysis)


def _record_success_in_tx(transaction: UnitOfWork, task_id: str | None) -> None:
    if task_id is None:
        return
    tasks_repo = getattr(transaction, "tasks", None)
    if tasks_repo is None:
        return
    task = tasks_repo.get(task_id)
    if task is not None and task.status is TaskStatus.RUNNING:
        tasks_repo.save(task.succeed())


class ChapterAnalysisSequence:
    """Enqueues chapter analysis tasks for all unanalyzed chapters in parallel (P1.T2)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        queue: TaskQueue,
        analyze_chapter: AnalyzeChapter,
        auto_start_processing: bool = False,
        start_processing: Callable[[str], Awaitable[object]] | None = None,
        seed_first_chapter: bool = False,
        sequential_analysis: bool | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._queue = queue
        self._analyze_chapter = analyze_chapter
        self._auto_start_processing = auto_start_processing
        self._start_processing = start_processing
        self._seed_first_chapter = bool(
            seed_first_chapter or getattr(analyze_chapter, "_seed_first_chapter", False)
        )
        if self._seed_first_chapter and hasattr(analyze_chapter, "_seed_first_chapter"):
            analyze_chapter._seed_first_chapter = True
        if sequential_analysis is None:
            self._sequential_analysis = bool(
                getattr(analyze_chapter, "_sequential_analysis", True)
            )
        else:
            self._sequential_analysis = bool(sequential_analysis)
            if hasattr(analyze_chapter, "_sequential_analysis"):
                analyze_chapter._sequential_analysis = self._sequential_analysis

    async def start(self, book_id: str) -> ProcessingTask | None:
        """Starts book analysis: transitions book to `ANALYZING` and enqueues chapters.

        Returns `None` if there is nothing left to analyze: a book without chapters or one
        already fully analyzed. Re-running after application restart resumes from unanalyzed chapters.

        If analysis of chapters previously failed with a retryable error, restarting
        returns tasks to the queue. Returns the first task to preserve caller expectations.
        """
        self._begin_analysis(book_id)
        tasks = await self._enqueue_all(book_id)
        for i, task in enumerate(tasks):
            if task is not None and task.status is TaskStatus.FAILED and task.is_retryable:
                task = task.retry()
                await self._queue.save(task)
                tasks[i] = task
        return tasks[0] if tasks else None

    async def run_task(self, task: ProcessingTask) -> None:
        """Handler for `ANALYZE_CHAPTER` task for the local runner (ALG-02)."""
        if task.task_type is not TaskType.ANALYZE_CHAPTER:
            raise AnalyzeChapterError(
                f"задача {task.id} типа {task.task_type} не является анализом главы"
            )
        await self._analyze_chapter(
            AnalyzeChapterCommand(
                book_id=task.book_id,
                section_id=task.scope_id,
                task_id=task.id,
            )
        )
        with self._unit_of_work.transaction() as transaction:
            sequential_active = self._is_sequential_active(transaction, task.book_id)
            if sequential_active:
                _record_success_in_tx(transaction, task.id)
        if sequential_active:
            enqueued = await self._enqueue_all(
                task.book_id,
                finished_task_id=task.id,
                complete_if_empty=False,
            )
            if not enqueued and self._try_complete_analysis(
                task.book_id, finished_task_id=task.id
            ):
                assert self._start_processing is not None
                await self._start_processing(task.book_id)
            return
        if self._should_enqueue_remaining_wave(task.book_id, task.scope_id):
            await self._enqueue_all(task.book_id)
        if self._try_complete_analysis(task.book_id, finished_task_id=task.id):
            assert self._start_processing is not None
            await self._start_processing(task.book_id)

    async def _enqueue_all(
        self,
        book_id: str,
        *,
        finished_task_id: str | None = None,
        complete_if_empty: bool = True,
    ) -> list[ProcessingTask]:
        """Enqueues tasks for the next chapter (sequential) or wave (parallel)."""
        section_ids = self._section_ids_for_wave(
            book_id, finished_task_id=finished_task_id
        )
        if not section_ids:
            if complete_if_empty and self._try_complete_analysis(
                book_id, finished_task_id=finished_task_id
            ):
                assert self._start_processing is not None
                await self._start_processing(book_id)
            return []
        tasks = []
        for section_id in section_ids:
            task = await self._queue.enqueue(
                TaskIdentity(
                    book_id=book_id,
                    task_type=TaskType.ANALYZE_CHAPTER,
                    scope_id=section_id,
                )
            )
            tasks.append(task)
        return tasks

    def _is_sequential_active(self, transaction: UnitOfWork, book_id: str) -> bool:
        work_repo = getattr(transaction, "work", None)
        conditions = (
            work_repo.latest_conditions(book_id)
            if work_repo is not None
            else None
        )
        if conditions is not None and isinstance(getattr(conditions, "payload", None), Mapping):
            return bool(
                (conditions.payload.get("checks") or {}).get(
                    "sequential_analysis", self._sequential_analysis
                )
            )
        return bool(self._sequential_analysis)

    def _is_seed_active(self, transaction: UnitOfWork, book_id: str) -> bool:
        work_repo = getattr(transaction, "work", None)
        conditions = (
            work_repo.latest_conditions(book_id)
            if work_repo is not None
            else None
        )
        if conditions is not None and isinstance(getattr(conditions, "payload", None), Mapping):
            return bool(
                (conditions.payload.get("checks") or {}).get(
                    "seed_first_chapter", self._seed_first_chapter
                )
            )
        return bool(self._seed_first_chapter)

    def _section_ids_for_wave(
        self, book_id: str, *, finished_task_id: str | None = None
    ) -> list[str]:
        """Chapters in reading order to enqueue in the current wave."""
        with self._unit_of_work.transaction() as transaction:
            document = transaction.documents.get(book_id)
            if document is None:
                raise AnalyzeChapterError(f"книги {book_id} нет в хранилище")
            analyzed = {
                analysis.section_id
                for analysis in transaction.analyses.list_for_book(book_id)
            }
            sequential_active = self._is_sequential_active(transaction, book_id)
            if sequential_active:
                tasks_repo = getattr(transaction, "tasks", None)
                has_active_task = (
                    tasks_repo is not None
                    and any(
                        task.task_type is TaskType.ANALYZE_CHAPTER
                        and task.id != finished_task_id
                        and task.status in {TaskStatus.PENDING, TaskStatus.RUNNING}
                        for task in tasks_repo.list_for_book(book_id)
                    )
                )
                seed_active = False
            else:
                has_active_task = False
                seed_active = self._is_seed_active(transaction, book_id)
        ordered = document.ordered_sections
        unanalyzed = [section for section in ordered if section.id not in analyzed]
        if sequential_active:
            if has_active_task or not unanalyzed:
                return []
            return [unanalyzed[0].id]
        if not seed_active or len(unanalyzed) <= 1:
            return [section.id for section in unanalyzed]
        has_analyzed_content = any(
            section.id in analyzed and _is_content_section(document, section)
            for section in ordered
        )
        if has_analyzed_content:
            return [section.id for section in unanalyzed]
        first_content = next(
            (section for section in unanalyzed if _is_content_section(document, section)),
            None,
        )
        if first_content is None:
            return [section.id for section in unanalyzed]
        prewave_dividers = [
            section.id
            for section in unanalyzed
            if section.order < first_content.order
            and _is_meaningful_part_divider(document, section)
        ]
        if prewave_dividers:
            return prewave_dividers
        return [
            section.id
            for section in unanalyzed
            if section.id == first_content.id
            or (
                section.order < first_content.order
                and document.is_front_matter(section)
            )
        ]

    def _should_enqueue_remaining_wave(
        self, book_id: str, finished_section_id: str
    ) -> bool:
        with self._unit_of_work.transaction() as transaction:
            if not self._is_seed_active(transaction, book_id):
                return False
            document = transaction.documents.get(book_id)
            if document is None:
                return False
            seed_section = _first_content_section(document)
            if seed_section is None:
                return False
            analyzed = {
                analysis.section_id
                for analysis in transaction.analyses.list_for_book(book_id)
            }
            if finished_section_id == seed_section.id:
                if seed_section.id not in analyzed:
                    return False
                return any(
                    section.id not in analyzed
                    for section in document.ordered_sections
                )
            prewave_dividers = [
                section
                for section in document.ordered_sections
                if section.order < seed_section.order
                and _is_meaningful_part_divider(document, section)
            ]
            if any(section.id == finished_section_id for section in prewave_dividers):
                if seed_section.id in analyzed:
                    return False
                return all(section.id in analyzed for section in prewave_dividers)
            return False

    def _unanalyzed_section_ids(self, book_id: str) -> list[str]:
        """Chapters in reading order for which no analysis exists yet."""
        with self._unit_of_work.transaction() as transaction:
            document = transaction.documents.get(book_id)
            if document is None:
                raise AnalyzeChapterError(f"книги {book_id} нет в хранилище")
            analyzed = {
                analysis.section_id
                for analysis in transaction.analyses.list_for_book(book_id)
            }
        return [
            section.id
            for section in document.ordered_sections
            if section.id not in analyzed
        ]

    def _next_section_id(self, book_id: str) -> str | None:
        """First chapter in reading order for which no analysis exists yet."""
        ids = self._unanalyzed_section_ids(book_id)
        return ids[0] if ids else None

    async def _enqueue_next(self, book_id: str) -> ProcessingTask | None:
        """Enqueues the next unanalyzed chapter task or completes analysis (legacy alias)."""
        tasks = await self._enqueue_all(book_id)
        return tasks[0] if tasks else None

    def _begin_analysis(self, book_id: str) -> None:
        """Transitions book to `ANALYZING` (ALG-06)."""
        with self._unit_of_work.transaction() as transaction:
            book = self._book(transaction, book_id)
            next_status = (
                book.status
                if book.status is BookStatus.ANALYZING
                else apply_book_event(book.status, BookEvent.ANALYSIS_STARTED)
            )
            work_repo = getattr(transaction, "work", None)
            set_book_paused = getattr(work_repo, "set_book_paused", None)
            if callable(set_book_paused):
                set_book_paused(book_id, None, False)
            if book.status is not BookStatus.ANALYZING:
                transaction.books.save(_with_status(book, next_status))

    def _complete_analysis(self, book_id: str) -> None:
        """Transitions analyzed book to glossary review (ALG-06)."""
        self._try_complete_analysis(book_id)

    def _try_complete_analysis(self, book_id: str, *, finished_task_id: str | None = None) -> bool:
        """Atomically checks if all chapters are analyzed and performs auto-approve if configured."""
        should_start_processing = False
        with self._unit_of_work.transaction() as transaction:
            _record_success_in_tx(transaction, finished_task_id)
            book = self._book(transaction, book_id)
            if book.status is not BookStatus.ANALYZING:
                return False

            tasks_repo = getattr(transaction, "tasks", None)
            if tasks_repo is not None:
                unfinished = [
                    t for t in tasks_repo.list_for_book(book_id)
                    if t.task_type is TaskType.ANALYZE_CHAPTER
                    and t.id != finished_task_id
                    and t.status not in {TaskStatus.SUCCEEDED, TaskStatus.SUPERSEDED}
                ]
                if unfinished:
                    return False

            updated_analyses: Sequence[ChapterAnalysis] | None = None
            work_repo = getattr(transaction, "work", None)
            document = transaction.documents.get(book_id)
            if document is not None:
                analyses = transaction.analyses.list_for_book(book_id)
                analyzed = {
                    analysis.section_id
                    for analysis in analyses
                }
                if any(section.id not in analyzed for section in document.ordered_sections):
                    return False

                from linguary.application.analysis_merge import deduplicate_book_before_you_read
                updated_analyses = deduplicate_book_before_you_read(document, analyses)
                original_by_id = {a.id: a for a in analyses}
                can_rearchive = (
                    work_repo is not None
                    and callable(getattr(work_repo, "get_binding", None))
                    and callable(getattr(work_repo, "archive_artifacts", None))
                    and callable(getattr(work_repo, "replace_analysis_binding", None))
                )
                for updated in updated_analyses:
                    if updated != original_by_id.get(updated.id):
                        transaction.analyses.save(book_id, updated)
                        if can_rearchive:
                            binding = work_repo.get_binding("analysis", updated.id)
                            if binding is not None and binding.get("conditions_id") is not None:
                                from linguary.application.book_conditions import BookConditionsService
                                BookConditionsService.archive_analysis(
                                    transaction,
                                    book_id,
                                    updated,
                                    binding["conditions_id"],
                                    binding.get("attempt_id"),
                                    replace_binding=True,
                                )

            book = _with_status(
                book, apply_book_event(book.status, BookEvent.ANALYSIS_COMPLETED)
            )
            transaction.books.save(book)

            conditions = work_repo.latest_conditions(book_id) if work_repo is not None else None
            auto_approve = False
            auto_start_processing = self._auto_start_processing
            confidence_threshold = 0.7
            min_mentions = 5
            min_chapters = 3
            policy = None

            if conditions is not None:
                checks = conditions.payload.get("checks", {})
                auto_approve = checks.get("auto_approve_glossary", False)
                auto_start_processing = bool(
                    checks.get("auto_start_processing", False) or self._auto_start_processing
                )
                confidence_threshold = checks.get("glossary_confidence_threshold", 0.7)
                min_mentions = checks.get("glossary_review_min_mentions", 5)
                min_chapters = checks.get("glossary_review_min_chapters", 3)
                if "name_translation_policy" in conditions.payload:
                    from linguary.domain.book_conditions import thaw
                    from linguary.domain.name_translation import NameTranslationPolicy
                    try:
                        policy = NameTranslationPolicy(**thaw(conditions.payload["name_translation_policy"]))
                    except (TypeError, ValueError):
                        policy = None

            if auto_approve:
                from linguary.application.glossary_review import build_glossary_review
                from linguary.application.mention_counter import mention_counter
                from linguary.domain import TermConcept, TermStatus, TermVariant

                analyses = (
                    updated_analyses
                    if updated_analyses is not None
                    else transaction.analyses.list_for_book(book_id)
                )
                count_mentions = mention_counter(document) if document else None
                review = build_glossary_review(
                    book_id=book_id,
                    analyses=analyses,
                    confidence_threshold=confidence_threshold,
                    decisions=transaction.glossary.list_for_book(book_id),
                    count_mentions=count_mentions,
                    min_mentions=min_mentions,
                    min_chapters=min_chapters,
                    name_translation_policy=policy,
                    conditions_id=conditions.id if conditions is not None else None,
                    document=document,
                )

                approved_concept_ids: set[str] = set()
                for item in review.items:
                    if (
                        item.awaits_decision
                        and bool(item.suggested_translation and item.suggested_translation.strip())
                        and item.confidence >= float(confidence_threshold)
                        and not item.model_flagged_review
                    ):
                        concept = TermConcept(
                            id=item.concept_id,
                            book_id=book.id,
                            type=item.type,
                            canonical_source_text=item.source_text,
                            status=TermStatus.APPROVED,
                            note=item.suggested_note,
                            variants=(
                                TermVariant(
                                    concept_id=item.concept_id,
                                    language=book.target_language,
                                    text=item.suggested_translation,
                                    status=TermStatus.APPROVED,
                                ),
                            ),
                        )
                        transaction.glossary.save(concept)
                        approved_concept_ids.add(item.concept_id)
                        for superseded in item.superseded_decisions:
                            transaction.glossary.delete(superseded.id)

                is_review_complete = all(
                    not item.requires_review
                    or item.decision is not None
                    or item.concept_id in approved_concept_ids
                    for item in review.items
                )
                if is_review_complete:
                    book = _with_status(
                        book, apply_book_event(book.status, BookEvent.GLOSSARY_REVIEW_COMPLETED)
                    )
                    transaction.books.save(book)
                    if auto_start_processing:
                        if self._start_processing is not None:
                            should_start_processing = True
                        else:
                            book = _with_status(
                                book,
                                apply_book_event(
                                    book.status, BookEvent.PROCESSING_RUN_STARTED
                                ),
                            )
                            transaction.books.save(book)
            if not should_start_processing and work_repo is not None:
                set_book_paused = getattr(work_repo, "set_book_paused", None)
                if callable(set_book_paused):
                    set_book_paused(book_id, None, False)
        return should_start_processing

    def _book(self, transaction: UnitOfWork, book_id: str) -> BookRecord:
        """Book record in open transaction."""
        book = transaction.books.get(book_id)
        if book is None:
            raise AnalyzeChapterError(f"книги {book_id} нет в хранилище")
        return book


_WORD_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_FRAGMENT_HEADER_RE = re.compile(
    r"Фрагмент главы: \d+-й из \d+\. Остальные фрагменты главы тебе не показаны\."
)


def _normalize_fragment_header_for_sha(input_text: str) -> str:
    """Normalize dynamic fragment position/total header so cache keys survive queue resplits."""
    return _FRAGMENT_HEADER_RE.sub("Фрагмент главы.", input_text)


def _recover_cached_fragment_split(
    fragment: ChapterFragment,
    saved_fragment_passes: Sequence[AnalysisPassResult],
) -> list[ChapterFragment]:
    """Recover contiguous cached sub-fragment split inside `fragment` (+ uncovered tail)."""
    if not saved_fragment_passes or len(fragment.blocks) <= 1:
        return [fragment]
    index_by_id = {block.id: idx for idx, block in enumerate(fragment.blocks)}
    by_start: dict[int, list[int]] = {}
    for saved_pass in saved_fragment_passes:
        scope_key = saved_pass.scope_key
        if not scope_key:
            continue
        matched_span: tuple[int, int] | None = None
        if scope_key.count(":") == 1:
            start_id, end_id = scope_key.split(":", 1)
            if start_id in index_by_id and end_id in index_by_id:
                matched_span = (index_by_id[start_id], index_by_id[end_id])
        else:
            for block_id, start_idx in index_by_id.items():
                prefix = f"{block_id}:"
                if scope_key.startswith(prefix):
                    end_id = scope_key[len(prefix):]
                    if end_id in index_by_id:
                        matched_span = (start_idx, index_by_id[end_id])
                        break
        if matched_span is None:
            continue
        start_idx, end_idx = matched_span
        if start_idx > end_idx:
            continue
        if start_idx == 0 and end_idx == len(fragment.blocks) - 1:
            continue
        by_start.setdefault(start_idx, []).append(end_idx)
    if 0 not in by_start:
        return [fragment]
    recovered: list[ChapterFragment] = []
    pos = 0
    while pos < len(fragment.blocks) and pos in by_start:
        end_idx = max(by_start[pos])
        recovered.append(
            ChapterFragment(order=0, blocks=tuple(fragment.blocks[pos : end_idx + 1]))
        )
        pos = end_idx + 1
    if pos < len(fragment.blocks):
        recovered.append(
            ChapterFragment(order=0, blocks=tuple(fragment.blocks[pos:]))
        )
    return recovered if len(recovered) > 1 else [fragment]


def _distribute_quota(
    total_quota: int,
    sub_fragments: Sequence[ChapterFragment],
    parent_block_count: int,
) -> list[int]:
    """Proportionally distribute parent quota across recovered sub-fragments."""
    if total_quota <= 0 or not sub_fragments:
        return [0] * len(sub_fragments)
    if (
        len(sub_fragments) == 2
        and len(sub_fragments[0].blocks) == parent_block_count // 2
    ):
        return [(total_quota + 1) // 2, total_quota // 2]
    weights = [len(sf.blocks) for sf in sub_fragments]
    total_weight = sum(weights)
    if total_weight <= 0:
        return [0] * len(sub_fragments)
    shares = [(total_quota * weight) // total_weight for weight in weights]
    remainder = total_quota - sum(shares)
    if remainder > 0:
        ranked = sorted(
            range(len(sub_fragments)),
            key=lambda idx: (-((total_quota * weights[idx]) % total_weight), idx),
        )
        for idx in ranked[:remainder]:
            shares[idx] += 1
    return shares


def _iter_part_candidates(part: Any) -> Sequence[GlossaryCandidateResponse]:
    if isinstance(part, tuple) and part and hasattr(part[0], "glossary_candidates"):
        return cast(Sequence[GlossaryCandidateResponse], part[0].glossary_candidates)
    if hasattr(part, "glossary_candidates"):
        return cast(Sequence[GlossaryCandidateResponse], part.glossary_candidates)
    return cast(Sequence[GlossaryCandidateResponse], part)


def _merge_fragment_candidates(
    parts: Sequence[object],
) -> tuple[GlossaryCandidateResponse, ...]:
    """Deterministically merge glossary candidates across earlier fragments of the chapter."""
    ordered_keys: list[str] = []
    by_key: dict[str, list[GlossaryCandidateResponse]] = {}
    for part in parts:
        for candidate in _iter_part_candidates(part):
            key = glossary_key(candidate.source_text)
            if key not in by_key:
                ordered_keys.append(key)
                by_key[key] = []
            by_key[key].append(candidate)

    merged: list[GlossaryCandidateResponse] = []
    for key in ordered_keys:
        variants = by_key[key]
        best = variants[0]
        for candidate in variants[1:]:
            if candidate.confidence >= best.confidence:
                best = candidate
        best_note = ""
        for candidate in variants:
            note = candidate.note.strip()
            if note and len(note) >= len(best_note):
                best_note = note
        best_description = ""
        for candidate in variants:
            description = candidate.description.strip()
            if description and len(description) >= len(best_description):
                best_description = description
        merged.append(
            replace(
                best,
                note=best_note,
                description=best_description,
                name_translation_assessment=best.name_translation_assessment,
            )
        )
    return tuple(merged)


def _a2_vocab_known_names_from_glossary(
    accumulated: Sequence[GlossaryCandidate],
    user_decisions: Sequence[TermConcept],
    suggestions: Sequence[TermConcept],
) -> tuple[str, ...]:
    """Collect proper-name source forms (PERSON/PLACE/ORGANIZATION) for A2 vocabulary filtering."""
    names: list[str] = []
    for candidate in accumulated:
        if candidate.type in _A2_VOCAB_NAME_TYPES and candidate.source_text.strip():
            names.append(candidate.source_text.strip())
    for concept in (*user_decisions, *suggestions):
        if concept.type not in _A2_VOCAB_NAME_TYPES:
            continue
        if concept.canonical_source_text.strip():
            names.append(concept.canonical_source_text.strip())
        for variant in concept.variants:
            source_text = getattr(variant, "source_text", None)
            if isinstance(source_text, str) and source_text.strip():
                names.append(source_text.strip())
    return tuple(dict.fromkeys(names))


def _a2_vocab_known_names_with_narrative(
    base_names: Sequence[str],
    narrative: NarrativeAnalysisResponse | None = None,
    scenes: Sequence[Scene] = (),
) -> tuple[str, ...]:
    """Combine prior proper-name source forms with current chapter proper-name candidates and participants."""
    names: list[str] = [
        name.strip() for name in base_names if isinstance(name, str) and name.strip()
    ]
    if narrative is not None:
        for candidate in narrative.glossary_candidates:
            if candidate.type in _A2_VOCAB_NAME_TYPES and candidate.source_text.strip():
                names.append(candidate.source_text.strip())
        for scene in narrative.scenes:
            for participant in scene.participants:
                if isinstance(participant, str) and participant.strip():
                    names.append(participant.strip())
    for scene in scenes:
        for participant in scene.participants:
            if isinstance(participant, str) and participant.strip():
                names.append(participant.strip())
    return tuple(dict.fromkeys(names))


def _known_names_with_narrative(
    base_names: Sequence[str],
    narrative: NarrativeAnalysisResponse | None = None,
    scenes: Sequence[Scene] = (),
) -> tuple[str, ...]:
    """Combine prior glossary names with current chapter narrative candidates and participants."""
    names: list[str] = [
        name.strip() for name in base_names if isinstance(name, str) and name.strip()
    ]
    if narrative is not None:
        for candidate in narrative.glossary_candidates:
            if candidate.source_text.strip():
                names.append(candidate.source_text.strip())
            if candidate.suggested_translation.strip():
                names.append(candidate.suggested_translation.strip())
        for scene in narrative.scenes:
            for participant in scene.participants:
                if isinstance(participant, str) and participant.strip():
                    names.append(participant.strip())
    for scene in scenes:
        for participant in scene.participants:
            if isinstance(participant, str) and participant.strip():
                names.append(participant.strip())
    return tuple(dict.fromkeys(names))


def _cumulative_words_before(
    document: CanonicalDocument, section: DocumentSection
) -> int:
    """Total words in content sections strictly preceding `section` in reading order."""
    total = 0
    for other in getattr(document, "ordered_sections", document.sections):
        if other.id == section.id:
            break
        if not _is_content_section(document, other):
            continue
        for block in textual_blocks(other):
            total += len(_WORD_TOKEN_RE.findall(block.text))
    return total


def _known_names_from_glossary(
    accumulated: Sequence[GlossaryCandidate],
    user_decisions: Sequence[TermConcept],
    suggestions: Sequence[TermConcept],
    *,
    target_language: str,
) -> tuple[str, ...]:
    """Collect known source and translated names from earlier chapters and glossary concepts."""
    names: list[str] = []
    for candidate in accumulated:
        if candidate.source_text.strip():
            names.append(candidate.source_text.strip())
        if candidate.suggested_translation.strip():
            names.append(candidate.suggested_translation.strip())
    for concept in (*user_decisions, *suggestions):
        if concept.canonical_source_text.strip():
            names.append(concept.canonical_source_text.strip())
        variant = concept.variant_for(target_language)
        if variant is not None and variant.text.strip():
            names.append(variant.text.strip())
    return tuple(dict.fromkeys(names))


def _collect_explained_topics(
    items: Sequence[FragmentAnalysis | ChapterAnalysis],
) -> tuple[str, ...]:
    """Collect deduplicated lemmas and grammar topics from fragment or chapter analyses."""
    topics: list[str] = []
    seen: set[str] = set()

    def add(topic: str) -> None:
        compact = " ".join(topic.split()).strip()
        if not compact:
            return
        key = compact.casefold()
        if key in seen:
            return
        seen.add(key)
        topics.append(compact)

    for item in items:
        for entry in item.before_you_read:
            add(entry.expression)
        plan = item.content_plan
        if plan is None:
            continue
        for difficulty in plan.difficulties:
            quotes = tuple(
                dict.fromkeys(
                    " ".join(anchor.quote.split())
                    for anchor in difficulty.anchors
                    if anchor.quote.strip()
                )
            )
            if difficulty.kind == "grammar":
                descriptions = "; ".join(
                    dict.fromkeys(
                        " ".join(point.description.split())
                        for point in difficulty.required_points
                        if point.description.strip()
                    )
                )
                if len(descriptions) > 120:
                    descriptions = descriptions[:117].rstrip() + "..."
                quote_label = ", ".join(quotes[:3])
                if quote_label and descriptions:
                    add(f"grammar: {quote_label} — {descriptions}")
                elif quote_label:
                    add(f"grammar: {quote_label}")
                elif descriptions:
                    add(f"grammar: {descriptions}")
            else:
                for quote in quotes:
                    add(quote if len(quote) <= 80 else quote[:77].rstrip() + "...")
    return tuple(topics)


def _prior_explained_topics(parts: Sequence[FragmentAnalysis]) -> tuple[str, ...]:
    """Collect deduplicated lemmas and grammar topics explained in earlier chapter fragments."""
    return _collect_explained_topics(parts)


DEFAULT_PRIOR_CHAPTER_TOPICS_WINDOW = 5
DEFAULT_MAX_PRIOR_CHAPTER_TOPICS = 60


def _prior_chapter_topics(
    document: CanonicalDocument,
    section: DocumentSection,
    analyses: Sequence[ChapterAnalysis],
    *,
    parallel_batch: bool,
    seed_active: bool,
    max_chapters: int = DEFAULT_PRIOR_CHAPTER_TOPICS_WINDOW,
    max_topics: int = DEFAULT_MAX_PRIOR_CHAPTER_TOPICS,
) -> tuple[str, ...]:
    """Collect bounded deduplicated topics explained in recent earlier content chapters."""
    ordered_sections = getattr(document, "ordered_sections", document.sections)
    if parallel_batch:
        if not seed_active:
            return ()
        seed_section = _first_content_section(document)
        if seed_section is None or seed_section.order >= section.order:
            return ()
        eligible_sections = (seed_section,)
    else:
        eligible_sections = tuple(
            other
            for other in ordered_sections
            if other.order < section.order and _is_content_section(document, other)
        )
    if not eligible_sections:
        return ()
    analyses_by_section = {analysis.section_id: analysis for analysis in analyses}
    ordered_prior = tuple(
        analyses_by_section[other.id]
        for other in eligible_sections
        if other.id in analyses_by_section
    )
    if not ordered_prior:
        return ()
    if max_chapters > 0 and len(ordered_prior) > max_chapters:
        ordered_prior = ordered_prior[-max_chapters:]
    all_topics = _collect_explained_topics(ordered_prior)
    if max_topics <= 0 or len(all_topics) <= max_topics:
        return all_topics
    kept_by_chapter: list[list[str]] = []
    seen: set[str] = set()
    remaining = max_topics
    for analysis in reversed(ordered_prior):
        if remaining <= 0:
            break
        chosen: list[str] = []
        for topic in _collect_explained_topics((analysis,)):
            if remaining <= 0:
                break
            key = topic.casefold()
            if key in seen:
                continue
            seen.add(key)
            chosen.append(topic)
            remaining -= 1
        if chosen:
            kept_by_chapter.append(chosen)
    kept_by_chapter.reverse()
    return tuple(topic for chapter_topics in kept_by_chapter for topic in chapter_topics)


def _has_meaningful_part_title(
    document: CanonicalDocument, section: DocumentSection
) -> bool:
    checker = getattr(document, "has_meaningful_part_title", None)
    return bool(checker(section)) if callable(checker) else False


def _is_meaningful_part_divider(
    document: CanonicalDocument, section: DocumentSection
) -> bool:
    is_front_matter = getattr(document, "is_front_matter", lambda _s: False)
    return (
        bool(is_front_matter(section))
        and _has_meaningful_part_title(document, section)
        and bool(textual_blocks(section))
    )


def _is_content_section(
    document: CanonicalDocument, section: DocumentSection
) -> bool:
    is_front_matter = getattr(document, "is_front_matter", lambda _s: False)
    return not is_front_matter(section) and not getattr(section, "is_notes_only", False)


def _first_content_section(document: CanonicalDocument) -> DocumentSection | None:
    sections = getattr(document, "ordered_sections", document.sections)
    return next(
        (section for section in sections if _is_content_section(document, section)),
        None,
    )


def _section_of(
    document: CanonicalDocument, command: AnalyzeChapterCommand
) -> DocumentSection:
    """Book chapter required by command."""
    for section in document.sections:
        if section.id == command.section_id:
            return section
    raise AnalyzeChapterError(
        f"главы {command.section_id} нет в книге {command.book_id}"
    )


def _ensure_plan_covers_chapter(
    payload: ChapterAnalysisResponse, context: ChapterAnalysisContext
) -> None:
    """Blocking validation of treatment plan (PRD § 15).

    The plan must cover every provided block exactly once and contain no foreign blocks.
    Response ordering is not checked: the plan is persisted in document order,
    and permuted response rows do not affect the result.
    """
    expected = context.expectation.block_ids
    returned = payload.planned_block_ids
    unknown = sorted(set(returned) - set(expected))
    if unknown:
        raise ResultValidationError(
            f"план обработки главы {context.section_id} содержит чужие блоки: {unknown}",
            expectation=context.expectation,
        )
    missing = sorted(set(expected) - set(returned))
    if missing:
        raise ResultValidationError(
            f"план обработки главы {context.section_id} не покрывает блоки: "
            f"{[context.block_id_for(alias) for alias in missing]}",
            expectation=context.expectation,
        )
    duplicated = sorted({block for block in returned if returned.count(block) > 1})
    if duplicated:
        raise ResultValidationError(
            f"блоки главы {context.section_id} получили по два режима обработки: "
            f"{[context.block_id_for(alias) for alias in duplicated]}",
            expectation=context.expectation,
        )


def _to_aliases(context: ChapterAnalysisContext, message: str) -> str:
    """Replaces internal block IDs in validation message with prompt aliases (b1..bN)."""
    to_aliases = getattr(context, "to_aliases", None)
    if callable(to_aliases):
        return to_aliases(message)
    mapping = {
        block_id: alias
        for alias, block_id in context.block_ids_by_alias.items()
        if block_id
    }
    if not mapping:
        return message
    pattern = re.compile(
        r"(?<![0-9A-Za-z_-])("
        + "|".join(
            re.escape(key) for key in sorted(mapping, key=len, reverse=True)
        )
        + r")(?![0-9A-Za-z_-])"
    )
    return pattern.sub(lambda match: mapping[match.group(0)], message)


def _worded_for_the_model(
    error: Exception, context: ChapterAnalysisContext
) -> Exception:
    """Same error rephrased using block aliases seen by the model (ALG-05)."""
    if not isinstance(error, ResultValidationError):
        return error
    violation = getattr(error, "violation", None)
    if violation is not None:
        aliases_by_block_id = {
            block_id: alias
            for alias, block_id in getattr(context, "block_ids_by_alias", {}).items()
            if block_id
        }
        block_ids = getattr(violation, "block_ids", ())
        aliased_block_ids = tuple(
            aliases_by_block_id.get(block_id, block_id)
            for block_id in block_ids
        )
        safe_message = getattr(violation, "safe_message", "")
        aliased_safe_message = (
            _to_aliases(context, safe_message) if safe_message else safe_message
        )
        if aliased_block_ids != block_ids or aliased_safe_message != safe_message:
            violation = replace(
                violation,
                block_ids=aliased_block_ids,
                safe_message=aliased_safe_message,
            )
    return ResultValidationError(
        _to_aliases(context, str(error)),
        expectation=error.expectation or context.expectation,
        violation=violation,
    )


def _ensure_candidates_single_script(
    payload: ChapterAnalysisResponse, context: ChapterAnalysisContext
) -> None:
    """Glossary candidate translation suggestion contains no mixed-script words (ALG-04).

    The glossary candidate is propagated to all book chapters, so a mixed-script word
    is rejected here before analysis is saved, rather than surfacing later in every block.
    """
    for candidate in payload.glossary_candidates:
        word = mixed_script_word(candidate.suggested_translation)
        if word is not None:
            raise ResultValidationError(
                f"вариант перевода кандидата глоссария {candidate.source_text!r} "
                f"содержит слово из смеси латиницы и кириллицы: {word!r}. Каждое слово "
                f"пиши одним алфавитом",
                expectation=context.expectation,
            )


def _empty_analysis(section: DocumentSection, *, id: str) -> ChapterAnalysis:
    """Complete analysis for a textless chapter: cover, insert, illustration page.

    There is nothing to pass to the model, but the chapter must be marked as analyzed:
    otherwise the book would never reach glossary review. The plan for such a chapter
    is empty and covers all its textual blocks (none exist).
    """
    return ChapterAnalysis.create(
        id=id,
        section_id=section.id,
        treatment_plan=TreatmentPlan(section_id=section.id),
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(),
    )


def _before_you_read(
    payload: ChapterAnalysisResponse,
) -> tuple[BeforeYouReadEntry, ...]:
    """Chapter `Before You Read` content (PRD § 19.1)."""
    return before_you_read_entries(payload.before_you_read, strict=payload.content_plan is not None and payload.content_plan.version == 3)


def _total_usage(usages: Sequence[TokenUsage | None]) -> TokenUsage | None:
    """Total token usage for chapter: provider counts each call separately.

    A provider not reporting token usage leaves `None` — it is not replaced with a sum:
    zero would mean a free call, not missing data (PRD § 24).
    """
    reported = [usage for usage in usages if usage is not None]
    if not reported:
        return None
    return TokenUsage(
        prompt_tokens=_total_tokens(usage.prompt_tokens for usage in reported),
        completion_tokens=_total_tokens(usage.completion_tokens for usage in reported),
    )


def _total_tokens(values: Iterable[int | None]) -> int | None:
    """Sum of values reported by the provider."""
    reported = [value for value in values if value is not None]
    return sum(reported) if reported else None


def _with_status(book: BookRecord, status: BookStatus) -> BookRecord:
    """Book record with new status."""
    return replace(book, status=status)


def _new_id() -> str:
    """ID for new analysis record."""
    return str(uuid4())


__all__ = [
    "AnalyzeChapter",
    "AnalyzeChapterCommand",
    "AnalyzeChapterError",
    "AnalyzeChapterResult",
    "ChapterAnalysisSequence",
    "IdFactory",
    "Sleep",
]


def _pass_model_dependencies(payload: Mapping, kind: AnalysisPassKind) -> dict:
    """Only this pass's public generation dependencies, never credentials/revisions."""
    models = payload.get('models', {})
    params = payload.get('llm_parameters', {})
    route = payload.get('route', {})
    dependencies = dict(
        model=models.get('analysis_passes', {}).get(kind.value) or models.get('analysis')
        or params.get('analysis_model') or params.get('model', ''),
        provider=route.get('provider', params.get('provider', '')),
        base_url=route.get('base_url', ''),
        temperature=params.get('temperature'),
        max_output_tokens=params.get('max_output_tokens'),
    )
    if params.get('reasoning_effort') is not None:
        dependencies['reasoning_effort'] = params['reasoning_effort']
    if params.get('reasoning_max_tokens') is not None:
        dependencies['reasoning_max_tokens'] = params['reasoning_max_tokens']
    return dependencies


def _candidate_suggestion_key(candidate: Mapping) -> tuple[str, str, str]:
    return (glossary_key(candidate['source_text']), candidate['suggested_translation'], candidate['note'])


def _suggestion_key(suggestion: TermConcept | GlossaryCandidate, language: str) -> tuple[str, str, str]:
    if isinstance(suggestion, TermConcept):
        variant = suggestion.variant_for(language)
        return (glossary_key(suggestion.canonical_source_text), variant.text if variant else '', suggestion.note)
    return (glossary_key(suggestion.source_text), suggestion.suggested_translation, suggestion.note)


def _prepare_staged_payload(
    payload: NarrativeAnalysisResponse | PedagogicalAnalysisResponse,
) -> NarrativeAnalysisResponse | PedagogicalAnalysisResponse:
    if isinstance(payload, NarrativeAnalysisResponse):
        prepared = prepare_analysis_payload(ChapterAnalysisResponse(payload.translation_instructions,
            payload.chapter_summary, glossary_candidates=payload.glossary_candidates))
        return replace(payload, chapter_summary=prepared.chapter_summary,
                       glossary_candidates=prepared.glossary_candidates)
    combined = combine_staged_analysis_responses(
        NarrativeAnalysisResponse('Accepted narrative.', 'Accepted summary.'), payload)
    prepared = prepare_analysis_payload(combined)
    return replace(payload, before_you_read=prepared.before_you_read,
        before_you_read_evidence=prepared.content_plan.before_you_read_evidence,
        reader_chapter_summary=prepared.chapter_summary if payload.reader_chapter_summary is not None else None)
