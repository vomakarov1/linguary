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
from dataclasses import asdict, dataclass, replace
from uuid import uuid4

from linguary.application.analysis_merge import (
    DEFAULT_MAX_FRAGMENT_BLOCKS,
    DEFAULT_MAX_FRAGMENT_CHARS,
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
)
from linguary.application.analysis_pass_cache import decode_pass, encode_pass, pedagogical_response
from linguary.application.block_labels import without_block_labels
from linguary.application.context_builder import (
    ChapterAnalysisContext,
    ContextBuilder,
    FragmentFindings,
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
from linguary.application.llm_dto import ChapterAnalysisResponse, NarrativeAnalysisResponse, PedagogicalAnalysisResponse
from linguary.application.llm_rejections import record_rejection, finalize_response
from linguary.application.ports.llm import LLMClient, LLMRequest, LLMResponseFormatError
from linguary.application.ports.language_detection import LanguageDetector
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.application.retry_policy import ResultValidationError, RetryPolicy
from linguary.application.validators.mixed_script import mixed_script_word
from linguary.application.task_runner import TaskQueue
from linguary.domain.content_plan import ContentPlan, ContextFact, Scene, canonical_sha
from linguary.domain.book_conditions import thaw
from linguary.domain.errors import DomainError
from linguary.domain.name_translation import NameTranslationPolicy
from linguary.domain import (
    BeforeYouReadEntry,
    BlockTreatment,
    BookEvent,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
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
        if hasattr(self._llm_client, "_attempts"):
            token_attempts = self._llm_client._attempts.set(())
        if hasattr(self._llm_client, "_responses"):
            token_responses = self._llm_client._responses.set(())
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
            if token_attempts is not None:
                self._llm_client._attempts.reset(token_attempts)
            if token_responses is not None:
                self._llm_client._responses.reset(token_responses)

    async def _analyze_staged(
        self, command: AnalyzeChapterCommand, document: CanonicalDocument, section: DocumentSection,
        accumulated: Sequence[GlossaryCandidate], user_decisions: Sequence[TermConcept],
        suggestions: Sequence[TermConcept], concepts: Sequence[TermConcept],
    ) -> tuple[ChapterAnalysis, int, TokenUsage | None]:
        blocks = textual_blocks(section)
        aliases = {b.id: b.id for b in blocks}
        texts = {b.id: b.text for b in blocks}
        visible = {b.id: source_text_for_model(b) for b in blocks}
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
            cached_narrative = self._verify_cached_pass(previous_narrative)
            previous_inputs.update(tuple(item) for item in previous_narrative.payload.get('input_suggestions', ()))
            own_outputs.update(_candidate_suggestion_key(item) for item in
                previous_narrative.payload.get('response', {}).get('glossary_candidates', ()))
            own_outputs.update(merged_glossary_suggestion_keys(cached_narrative.glossary_candidates))
        for other_pass in book_narratives:
            if other_pass is None or (previous_narrative is not None and other_pass.id == previous_narrative.id):
                continue
            try:
                other_decoded = self._verify_cached_pass(other_pass)
            except AnalyzeChapterError:
                continue
            own_outputs.update(_candidate_suggestion_key(item) for item in
                other_pass.payload.get('response', {}).get('glossary_candidates', ()))
            own_outputs.update(merged_glossary_suggestion_keys(other_decoded.glossary_candidates))
        if own_outputs:
            own_outputs.update(merged_glossary_suggestion_keys(accumulated))
            suggestions = tuple(s for s in suggestions
                if _suggestion_key(s, document.target_language) not in own_outputs - previous_inputs)
        expected_conditions_id = (getattr(self._llm_client, 'conditions_id', None)
                                  or self._conditions_id or (conditions.id if conditions else None))
        dependencies = conditions.payload if conditions else {}
        narrative_hash = self._context_builder.narrative_input_sha256(document, section,
            accumulated, user_decisions, suggestions,
            model_dependencies=_pass_model_dependencies(dependencies, AnalysisPassKind.NARRATIVE),
            max_fragment_blocks=self._max_fragment_blocks)
        if previous_narrative is not None and previous_narrative.input_sha256 == narrative_hash:
            saved = previous_narrative
            narrative = cached_narrative
        else:
            saved = None
            narrative = None
        attempts = 0
        usages = []
        if saved is None or narrative is None:
            narrative_parts, count, usage = await self._run_staged_fragments(
                command, document, section, accumulated, user_decisions, suggestions, concepts,
                pass_kind=AnalysisPassKind.NARRATIVE,
                chapter_input_sha=narrative_hash,
                expected_conditions_id=expected_conditions_id)
            attempts += count
            usages.append(usage)
            narrative, scenes, facts = merge_narrative_fragments(narrative_parts)
            validate_narrative_response(narrative, aliases, texts, visible,
                ordered_block_ids=tuple(aliases), is_notes_only=section.is_notes_only,
                mandatory_facts=glossary_context_constraints(concepts))
            saved = self._save_pass(command, AnalysisPassKind.NARRATIVE, narrative_hash,
                                    narrative, expected_conditions_id,
                                    input_suggestions=tuple(_suggestion_key(s, document.target_language)
                                                            for s in suggestions))
        else:
            if saved.attempt_id is not None:
                self._staged_attempt_ids.set((*self._staged_attempt_ids.get(), saved.attempt_id))
            scenes, facts = validate_narrative_response(narrative, aliases, texts, visible,
                ordered_block_ids=tuple(aliases), is_notes_only=section.is_notes_only,
                mandatory_facts=glossary_context_constraints(concepts))
        pedagogical_hash = self._context_builder.pedagogical_input_sha256(document, section,
            narrative_output_sha256=saved.output_sha256, narrative_summary=narrative.chapter_summary,
            context_facts=facts, glossary_candidates=narrative.glossary_candidates,
            accumulated_candidates=accumulated, user_decisions=user_decisions,
            model_dependencies=_pass_model_dependencies(dependencies, AnalysisPassKind.PEDAGOGICAL),
            max_fragment_blocks=self._max_fragment_blocks)
        cached_pedagogical = self._cached_pass(command, AnalysisPassKind.PEDAGOGICAL, pedagogical_hash)
        if cached_pedagogical is None:
            parts, count, usage = await self._run_staged_fragments(
                command, document, section, accumulated, user_decisions, suggestions, concepts,
                pass_kind=AnalysisPassKind.PEDAGOGICAL, narrative=narrative, facts=facts,
                chapter_input_sha=pedagogical_hash,
                expected_conditions_id=expected_conditions_id)
            attempts += count
            usages.append(usage)
            merged = merge_chapter_analysis(parts, id=self._id_factory(), section_id=section.id,
                expected_block_ids=tuple(aliases),
                glossary_confidence_threshold=self._glossary_confidence_threshold,
                id_factory=self._id_factory)
            pedagogical = pedagogical_response(merged, reader_level=self._context_builder._reader_level)
            analysis = self._assemble_staged(narrative, pedagogical, scenes, document, section,
                                             accumulated, user_decisions)
            self._save_pass(command, AnalysisPassKind.PEDAGOGICAL, pedagogical_hash,
                            pedagogical, expected_conditions_id)
        else:
            saved_pedagogical, pedagogical = cached_pedagogical
            if saved_pedagogical.attempt_id is not None:
                self._staged_attempt_ids.set((*self._staged_attempt_ids.get(), saved_pedagogical.attempt_id))
            analysis = self._assemble_staged(narrative, pedagogical, scenes, document, section,
                                             accumulated, user_decisions)
        return analysis, attempts, _total_usage(usages)

    def _assemble_staged(
        self, narrative: NarrativeAnalysisResponse, pedagogical: PedagogicalAnalysisResponse,
        scenes: Sequence[Scene], document: CanonicalDocument, section: DocumentSection,
        accumulated: Sequence[GlossaryCandidate], user_decisions: Sequence[TermConcept],
    ) -> ChapterAnalysis:
        blocks = textual_blocks(section)
        context = self._context_builder.for_chapter_pedagogical_analysis(document, section,
            narrative_summary=narrative.chapter_summary,
            glossary_candidates=narrative.glossary_candidates,
            accumulated_candidates=accumulated,
            user_decisions=user_decisions)
        # Both saved DTOs use stable references scoped to this chapter.
        context = replace(context, block_ids_by_alias={b.id: b.id for b in blocks},
            visible_texts_by_alias={b.id: source_text_for_model(b) for b in blocks},
            expectation=replace(context.expectation,
                blocks=tuple(replace(e, block_id=b.id) for e, b in
                             zip(context.expectation.blocks, blocks, strict=True))))
        if self._context_builder._reader_level == 'a2':
            validate_a2_summary(pedagogical.reader_chapter_summary or '',
                source_texts=tuple(b.text for b in blocks), source_language=document.source_language,
                language_detector=self._language_detector)
        payload = combine_staged_analysis_responses(narrative, pedagogical, resolved_scenes=scenes)
        _ensure_plan_covers_chapter(payload, context)
        plan = self._validate_pedagogical(payload, context, blocks)
        analysis = merge_chapter_analysis([FragmentAnalysis.of_response(payload, order=0,
            block_ids_by_alias=context.block_ids_by_alias, texts={b.id: b.text for b in blocks},
            visible_texts_by_alias=context.visible_texts_by_alias, resolved_content_plan=plan,
            blocks=blocks)],
            id=self._id_factory(), section_id=section.id, expected_block_ids=tuple(b.id for b in blocks),
            glossary_confidence_threshold=self._glossary_confidence_threshold,
            id_factory=self._id_factory, name_translation_policy=self._name_translation_policy,
            conditions_id=self._conditions_id)
        return replace(analysis, content_plan=plan, scenes=tuple(scenes))

    def _validate_pedagogical(
        self, payload: ChapterAnalysisResponse, context: ChapterAnalysisContext,
        blocks: Sequence[DocumentBlock],
    ) -> ContentPlan | None:
        return validate_content_plan(payload, context.block_ids_by_alias, {b.id: b.text for b in blocks},
            english_help=self._context_builder._english_help,
            before_you_read_limit=context.before_you_read_limit,
            language_notes_per_chapter=context.language_notes_per_chapter,
            note_limits_by_block=context.note_limits_by_block, note_group_limits=context.note_group_limits,
            visible_texts_by_alias=context.visible_texts_by_alias)

    async def _run_staged_fragments(
        self, command: AnalyzeChapterCommand, document: CanonicalDocument, section: DocumentSection,
        accumulated: Sequence[GlossaryCandidate], user_decisions: Sequence[TermConcept],
        suggestions: Sequence[TermConcept], concepts: Sequence[TermConcept], *, pass_kind: AnalysisPassKind,
        narrative: NarrativeAnalysisResponse | None = None, facts: Sequence[ContextFact] = (),
        chapter_input_sha: str | None = None, expected_conditions_id: str | None = None,
    ) -> tuple[list, int, TokenUsage | None]:
        fragments = plan_fragments(
            textual_blocks(section), max_blocks=self._max_fragment_blocks, max_chars=DEFAULT_MAX_FRAGMENT_CHARS,
        )
        is_narrative = pass_kind is AnalysisPassKind.NARRATIVE
        notes = self._context_builder._language_notes_per_chapter
        queue = [(fragment,
            0 if is_narrative else 10 // len(fragments) + (fragment.order < 10 % len(fragments)),
            0 if is_narrative else notes // len(fragments) + (fragment.order < notes % len(fragments)))
            for fragment in fragments]
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
                    )
                context = self._context_builder.for_chapter_narrative_analysis(document, section,
                    accumulated, user_decisions, suggestions, fragment=fragment if total > 1 else None,
                    fragment_count=total, findings=previous)
            else:
                context = self._context_builder.for_chapter_pedagogical_analysis(document, section,
                    narrative_summary=narrative.chapter_summary, context_facts=facts,
                    glossary_candidates=narrative.glossary_candidates,
                    accumulated_candidates=accumulated, user_decisions=user_decisions,
                    fragment=fragment if total > 1 else None, fragment_count=total,
                    before_you_read_limit=byr_quota, language_notes_limit=note_quota)
                if parts:
                    context = replace(context, request=replace(context.request,
                        input_text=context.request.input_text + '\nУже найдено в предыдущих частях главы:\n' +
                            'Пересказ: ' + parts[-1].chapter_summary + '\nВыражения BYR: ' +
                            ', '.join(e.expression for p in parts for e in p.before_you_read)))
            context = replace(context, request=context.request.with_logging(book_id=command.book_id,
                stage=LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE if is_narrative else LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL,
                task_id=command.task_id, scope_id=section.id, ordered_block_ids=fragment.block_ids))

            def validate(payload):
                if is_narrative:
                    _ensure_candidates_single_script(payload, context)
                    return validate_narrative_response(payload, context.block_ids_by_alias,
                        {b.id: b.text for b in fragment.blocks}, context.visible_texts_by_alias,
                        ordered_block_ids=fragment.block_ids,
                        is_notes_only=section.is_notes_only or document.is_front_matter(section),
                        mandatory_facts=glossary_context_constraints(concepts))
                combined = combine_staged_analysis_responses(
                    NarrativeAnalysisResponse(narrative.translation_instructions, narrative.chapter_summary), payload)
                if self._context_builder._reader_level == 'a2':
                    validate_a2_summary(payload.reader_chapter_summary or '',
                        source_texts=tuple(b.text for b in fragment.blocks), source_language=document.source_language,
                        language_detector=self._language_detector)
                _ensure_plan_covers_chapter(combined, context)
                resolved_plan = self._validate_pedagogical(combined, context, fragment.blocks)
                # The first fragment has no prior parts, but segment indexes are still rejected here.
                candidate_part = FragmentAnalysis.of_response(
                    combined,
                    order=len(parts),
                    block_ids_by_alias=context.block_ids_by_alias,
                    texts={b.id: b.text for b in fragment.blocks},
                    visible_texts_by_alias=context.visible_texts_by_alias,
                    resolved_content_plan=resolved_plan,
                    blocks=fragment.blocks,
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
                        input=context.request.input_text,
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
                resolved = validate(payload)
            else:
                try:
                    payload, resolved, count, usage = await self._request_staged(context, validate)
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
                    if hasattr(self._llm_client, 'reject_last'):
                        self._llm_client.reject_last()
                if isinstance(error, LLMResponseFormatError) and error.truncated:
                    error.analysis_attempts = attempts
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
            if conditions_id is not None:
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
        payload, attempts, usage = await self._request_analysis(
            context,
            blocks=blocks,
            source_language=document.source_language,
            concepts=concepts,
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
            try:
                payload, fragment_attempts, usage = await self._request_analysis(
                    context,
                    blocks=fragment.blocks,
                    source_language=document.source_language,
                    concepts=concepts,
                )
            except LLMResponseFormatError as error:
                if error.truncated and len(fragment.blocks) > 1:
                    attempts += 1
                    if error.usage is not None:
                        usages.append(error.usage)
                    mid = len(fragment.blocks) // 2
                    left = ChapterFragment(order=0, blocks=tuple(fragment.blocks[:mid]))
                    right = ChapterFragment(order=0, blocks=tuple(fragment.blocks[mid:]))
                    queue = [left, right] + queue
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
        return analysis, attempts, _total_usage(usages)

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
            seed_section = _first_content_section(document) if (parallel_batch and seed_active) else None
            seed_section_id = (
                seed_section.id
                if seed_section is not None and seed_section.order < current_order
                else None
            )
            seed_candidates = (
                tuple(
                    candidate
                    for analysis in analyses
                    if analysis.section_id == seed_section_id
                    for candidate in analysis.glossary_candidates
                )
                if seed_section_id is not None
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
                        if analysis.section_id != seed_section_id
                        for candidate in analysis.glossary_candidates
                    )
                    if seed_section_id is not None
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
    ) -> tuple[ChapterAnalysisResponse, int, TokenUsage | None]:
        """Executes model call with retries per policy (ALG-05).

        When attempts are exhausted, raises the original error: typing the error and
        transitioning the task to `failed` is the responsibility of the runner, not this loop.
        """
        request: LLMRequest[ChapterAnalysisResponse] = context.request
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
                    if self._context_builder._reader_level == "a2":
                        validate_a2_summary(
                            response.payload.chapter_summary,
                            source_texts=tuple(block.text for block in blocks),
                            source_language=source_language,
                            language_detector=self._language_detector,
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
                    )
                    if request.contract_version in {3, 5}:
                        if resolved_concepts is None:
                            with self._unit_of_work.transaction() as transaction:
                                resolved_concepts = tuple(transaction.glossary.list_for_book(request.book_id))
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
                    if hasattr(self._llm_client, 'reject_last'):
                        self._llm_client.reject_last()
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
                )
            ),
        )

    def _candidates(
        self, payload: ChapterAnalysisResponse, *, analysis_id: str
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
                self._llm_client.service.verify_source(book, conditions)
                if task_id is not None and transaction.tasks.get(task_id) is None:
                    raise ConditionsError('task_no_longer_active')
                attempts = self._llm_client.attempts
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

    async def start(self, book_id: str) -> ProcessingTask | None:
        """Starts book analysis: transitions book to `ANALYZING` and enqueues all chapters.

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
        if self._should_enqueue_remaining_wave(task.book_id, task.scope_id):
            await self._enqueue_all(task.book_id)
        if self._try_complete_analysis(task.book_id, finished_task_id=task.id):
            assert self._start_processing is not None
            await self._start_processing(task.book_id)

    async def _enqueue_all(self, book_id: str) -> list[ProcessingTask]:
        """Enqueues tasks for all unanalyzed chapters in parallel (or seed wave first)."""
        section_ids = self._section_ids_for_wave(book_id)
        if not section_ids:
            if self._try_complete_analysis(book_id):
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

    def _section_ids_for_wave(self, book_id: str) -> list[str]:
        """Chapters in reading order to enqueue in the current wave."""
        with self._unit_of_work.transaction() as transaction:
            document = transaction.documents.get(book_id)
            if document is None:
                raise AnalyzeChapterError(f"книги {book_id} нет в хранилище")
            analyzed = {
                analysis.section_id
                for analysis in transaction.analyses.list_for_book(book_id)
            }
            seed_active = self._is_seed_active(transaction, book_id)
        ordered = document.ordered_sections
        unanalyzed = [section for section in ordered if section.id not in analyzed]
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
            if seed_section is None or finished_section_id != seed_section.id:
                return False
            analyzed = {
                analysis.section_id
                for analysis in transaction.analyses.list_for_book(book_id)
            }
            if seed_section.id not in analyzed:
                return False
            return any(
                section.id not in analyzed
                for section in document.ordered_sections
            )

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


AnalyzeChapter._to_aliases = staticmethod(_to_aliases)  # type: ignore[attr-defined]


def _worded_for_the_model(
    error: Exception, context: ChapterAnalysisContext
) -> Exception:
    """Same error rephrased using block aliases seen by the model (ALG-05)."""
    if not isinstance(error, ResultValidationError):
        return error
    violation = getattr(error, "violation", None)
    if violation is not None and getattr(violation, "block_ids", ()):
        aliases_by_block_id = {
            block_id: alias
            for alias, block_id in getattr(context, "block_ids_by_alias", {}).items()
            if block_id
        }
        aliased_block_ids = tuple(
            aliases_by_block_id.get(block_id, block_id)
            for block_id in violation.block_ids
        )
        if aliased_block_ids != violation.block_ids:
            violation = replace(violation, block_ids=aliased_block_ids)
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
    return dict(model=models.get('analysis_passes', {}).get(kind.value) or models.get('analysis')
        or params.get('analysis_model') or params.get('model', ''),
        provider=route.get('provider', params.get('provider', '')),
        base_url=route.get('base_url', ''),
        temperature=params.get('temperature'), max_output_tokens=params.get('max_output_tokens'))


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
