"""Deterministic verification of book-wide NARRATIVE readiness and manifest assembly (`ALG-01`)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from linguary.application.analysis_merge import (
    glossary_context_constraints,
    merged_glossary_suggestion_keys,
    validate_narrative_response,
)
from linguary.application.analysis_pass_cache import (
    decode_pass,
    encode_pass,
    validate_narrative_input_context,
)
from linguary.application.analysis_pass_inputs import (
    NarrativePassInputs,
    deserialize_nonbinding_suggestions,
    deserialize_user_decisions,
    is_illustrable_content_section,
    is_meaningful_part_divider,
)
from linguary.application.book_conditions import (
    ConditionsError,
    effective_content_plan_profile,
    payload_digest,
    supported_condition_versions,
)
from linguary.application.context_builder import (
    ContextBuilder,
    source_text_for_model,
    textual_blocks,
)
from linguary.application.errors import ApplicationError
from linguary.application.llm_dto import NarrativeAnalysisResponse
from linguary.application.ports.language_detection import LanguageDetector
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.application.token_budget import narrative_max_fragment_blocks_for
from linguary.domain import (
    AnalysisPassKind,
    AnalysisPassResult,
    ConceptType,
    DomainError,
    GlossaryCandidate,
    TaskStatus,
    TaskType,
    TermConcept,
    glossary_key,
)
from linguary.domain.book_conditions import thaw
from linguary.domain.content_plan import Scene, canonical_sha
from linguary.domain.illustrations import IllustrationReasonCode
from linguary.domain.name_translation import NameTranslationPolicy

_NAME_TYPES = frozenset({ConceptType.PERSON, ConceptType.PLACE, ConceptType.ORGANIZATION})


def _pass_model_dependencies(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror `analyze_chapter._pass_model_dependencies(..., NARRATIVE)`."""
    models = payload.get("models", {})
    params = payload.get("llm_parameters", {})
    route = payload.get("route", {})
    dependencies: dict[str, Any] = dict(
        model=models.get("analysis_passes", {}).get(AnalysisPassKind.NARRATIVE.value)
        or models.get("analysis")
        or params.get("analysis_model")
        or params.get("model", ""),
        provider=route.get("provider", params.get("provider", "")),
        base_url=route.get("base_url", ""),
        temperature=params.get("temperature"),
        max_output_tokens=params.get("max_output_tokens"),
    )
    if params.get("reasoning_effort") is not None:
        dependencies["reasoning_effort"] = params["reasoning_effort"]
    if params.get("reasoning_max_tokens") is not None:
        dependencies["reasoning_max_tokens"] = params["reasoning_max_tokens"]
    return dependencies


def _candidate_suggestion_key(candidate: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        glossary_key(candidate["source_text"]),
        candidate["suggested_translation"],
        candidate["note"],
    )


def _suggestion_key(
    suggestion: TermConcept | GlossaryCandidate, language: str
) -> tuple[str, str, str]:
    if isinstance(suggestion, TermConcept):
        variant = suggestion.variant_for(language)
        return (
            glossary_key(suggestion.canonical_source_text),
            variant.text if variant else "",
            suggestion.note,
        )
    return (
        glossary_key(suggestion.source_text),
        suggestion.suggested_translation,
        suggestion.note,
    )


def _known_names_from_glossary(
    accumulated: Sequence[GlossaryCandidate],
    user_decisions: Sequence[TermConcept],
    suggestions: Sequence[TermConcept],
    *,
    target_language: str,
) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for candidate in accumulated:
        if candidate.type in _NAME_TYPES:
            key = glossary_key(candidate.source_text)
            if key not in seen:
                seen.add(key)
                names.append(candidate.source_text)
    for concept in (*user_decisions, *suggestions):
        if concept.type in _NAME_TYPES:
            key = glossary_key(concept.canonical_source_text)
            if key not in seen:
                seen.add(key)
                names.append(concept.canonical_source_text)
            variant = concept.variant_for(target_language)
            if variant is not None and variant.text.strip():
                vkey = glossary_key(variant.text)
                if vkey not in seen:
                    seen.add(vkey)
                    names.append(variant.text)
    return tuple(names)


def _verify_pass_envelope(
    saved: AnalysisPassResult,
) -> tuple[NarrativeAnalysisResponse, dict[str, Any] | None, str]:
    """Verify stored NARRATIVE pass envelope and return `(decoded, input_context, response_sha)`."""
    if saved.scope_key != "":
        raise ValueError("фрагментный проход не принимается как полный проход главы")
    decoded = decode_pass(saved.payload, pass_kind=AnalysisPassKind.NARRATIVE)
    normalized = encode_pass(decoded)
    response_sha = canonical_sha(normalized)

    keys = saved.payload["input_suggestions"]
    if not isinstance(keys, (tuple, list)) or any(
        not isinstance(item, (tuple, list))
        or len(item) != 3
        or any(not isinstance(part, str) for part in item)
        for item in keys
    ):
        raise ValueError("неверный список входных подсказок")
    normalized["input_suggestions"] = thaw(keys)

    input_context: dict[str, Any] | None = None
    if "input_context" in saved.payload:
        input_context = validate_narrative_input_context(saved.payload["input_context"])
        normalized["input_context"] = thaw(saved.payload["input_context"])

    if (
        canonical_sha(thaw(saved.payload)) != saved.output_sha256
        or canonical_sha(normalized) != saved.output_sha256
    ):
        raise ValueError("не совпадает контрольная сумма сохранённого прохода")
    return decoded, input_context, response_sha


@dataclass(frozen=True, slots=True)
class SectionManifestEntry:
    section_id: str
    order: int
    title: str
    skipped: bool
    skip_reason: str | None
    block_digests: Mapping[str, str]
    pass_id: str | None = None
    input_sha256: str | None = None
    output_sha256: str | None = None
    response_sha256: str | None = None
    conditions_id: str | None = None
    narrative: NarrativeAnalysisResponse | None = None
    scenes: tuple[Scene, ...] = ()


@dataclass(frozen=True, slots=True)
class IllustrationSourceSnapshot:
    """Complete deterministic snapshot of a book's narrative readiness (`ALG-01`)."""

    book_id: str
    compatible: bool
    ready: bool
    reason_code: str | None
    source_sha256: str | None
    conditions_id: str | None
    conditions_revision: int | None
    expected_chapters: int
    ready_chapters: int
    skipped_sections: int
    blocking_section_id: str | None
    content_section_ids: tuple[str, ...]
    manifest: Mapping[str, Any] | None
    manifest_sha256: str | None
    source_authority: Mapping[str, Any] | None
    entries: tuple[SectionManifestEntry, ...]


def compute_reference_fingerprint(
    snapshot: IllustrationSourceSnapshot,
    *,
    policy_payload: Mapping[str, Any],
    model_reference: str,
    route_id: str,
) -> str:
    """Content-addressed fingerprint for the book's illustration reference (`ALG-01`)."""
    sections_digest = [
        {
            "section_id": e.section_id,
            "order": e.order,
            "skipped": e.skipped,
            "block_digests": dict(e.block_digests),
            "input_sha256": e.input_sha256,
            "response_sha256": e.response_sha256,
        }
        for e in snapshot.entries
    ]
    return canonical_sha(
        {
            "kind": "illustration_reference_v1",
            "source_sha256": snapshot.source_sha256,
            "sections": sections_digest,
            "model_reference": model_reference,
            "route_id": route_id,
            "policy": dict(policy_payload),
        }
    )


def compute_chapter_fingerprint(
    entry: SectionManifestEntry,
    *,
    registry_sha256: str,
    policy_payload: Mapping[str, Any],
    model_prompts: str,
    route_id: str,
    style: str,
    max_per_chapter: int,
) -> str:
    """Content-addressed fingerprint for one chapter's illustration prompts (`ALG-01`)."""
    return canonical_sha(
        {
            "kind": "illustration_chapter_v2",
            "registry_sha256": registry_sha256,
            "section_id": entry.section_id,
            "block_digests": dict(entry.block_digests),
            "input_sha256": entry.input_sha256,
            "response_sha256": entry.response_sha256,
            "model_prompts": model_prompts,
            "route_id": route_id,
            "style": style,
            "max_per_chapter": max_per_chapter,
            "policy": dict(policy_payload),
        }
    )


class IllustrationSource:
    """Evaluates staged NARRATIVE readiness without API keys or `BookRuntimeResolver`."""

    def __init__(
        self,
        *,
        language_detector: LanguageDetector,
        source_digest_fn: Callable[[str], str],
    ) -> None:
        self._language_detector = language_detector
        self._source_digest_fn = source_digest_fn

    def _incompatible(
        self,
        book_id: str,
        *,
        reason_code: str,
        source_sha256: str | None = None,
        conditions_id: str | None = None,
        conditions_revision: int | None = None,
    ) -> IllustrationSourceSnapshot:
        return IllustrationSourceSnapshot(
            book_id=book_id,
            compatible=False,
            ready=False,
            reason_code=reason_code,
            source_sha256=source_sha256,
            conditions_id=conditions_id,
            conditions_revision=conditions_revision,
            expected_chapters=0,
            ready_chapters=0,
            skipped_sections=0,
            blocking_section_id=None,
            content_section_ids=(),
            manifest=None,
            manifest_sha256=None,
            source_authority=None,
            entries=(),
        )

    def _resolve_source_digest(
        self,
        book: BookRecord,
        expected_sha: str | None,
        *,
        precomputed_source_digest: str | None,
    ) -> tuple[str | None, str | None]:
        if precomputed_source_digest is not None:
            actual = precomputed_source_digest
        else:
            try:
                actual = self._source_digest_fn(book.source_epub_path)
            except ConditionsError as err:
                if err.reason_code == "source_digest_mismatch":
                    return None, IllustrationReasonCode.SOURCE_CHANGED.value
                return None, IllustrationReasonCode.STORAGE_UNAVAILABLE.value
            except OSError:
                return None, IllustrationReasonCode.STORAGE_UNAVAILABLE.value
        if expected_sha is not None and actual != expected_sha:
            return actual, IllustrationReasonCode.SOURCE_CHANGED.value
        return actual, None

    def snapshot(
        self,
        tx: UnitOfWork,
        book_id: str,
        *,
        precomputed_source_digest: str | None = None,
    ) -> IllustrationSourceSnapshot:
        book = tx.books.get(book_id)
        if book is None:
            return self._incompatible(
                book_id, reason_code=IllustrationReasonCode.BOOK_NOT_FOUND.value
            )

        conditions = tx.work.latest_conditions(book_id)
        if conditions is None:
            return self._incompatible(
                book_id, reason_code=IllustrationReasonCode.SOURCE_INCOMPATIBLE.value
            )

        payload = conditions.payload
        expected_source_sha = payload.get("source_sha256")
        if not isinstance(expected_source_sha, str) or not expected_source_sha:
            return self._incompatible(
                book_id,
                reason_code=IllustrationReasonCode.SOURCE_INCOMPATIBLE.value,
                conditions_id=conditions.id,
                conditions_revision=conditions.revision,
            )

        # Verify source file digest first / independently of provider API key.
        actual_source_sha, digest_err = self._resolve_source_digest(
            book,
            expected_source_sha,
            precomputed_source_digest=precomputed_source_digest,
        )
        if digest_err is not None:
            return self._incompatible(
                book_id,
                reason_code=digest_err,
                source_sha256=actual_source_sha,
                conditions_id=conditions.id,
                conditions_revision=conditions.revision,
            )
        assert actual_source_sha is not None

        # Validate staged content-plan-v3 conditions strictly without requiring API key.
        try:
            eff_profile = effective_content_plan_profile(conditions)
        except ConditionsError:
            return self._incompatible(
                book_id,
                reason_code=IllustrationReasonCode.SOURCE_INCOMPATIBLE.value,
                source_sha256=actual_source_sha,
                conditions_id=conditions.id,
                conditions_revision=conditions.revision,
            )
        if not bool(payload.get("staged_analysis")) or eff_profile != "content-plan-v3":
            return self._incompatible(
                book_id,
                reason_code=IllustrationReasonCode.SOURCE_INCOMPATIBLE.value,
                source_sha256=actual_source_sha,
                conditions_id=conditions.id,
                conditions_revision=conditions.revision,
            )

        templates = payload.get("prompt_templates")
        template_values = thaw(templates) if isinstance(templates, Mapping) else None
        if (
            not isinstance(template_values, dict)
            or payload_digest(template_values) != payload.get("prompt_sha256")
        ):
            return self._incompatible(
                book_id,
                reason_code=IllustrationReasonCode.SOURCE_INCOMPATIBLE.value,
                source_sha256=actual_source_sha,
                conditions_id=conditions.id,
                conditions_revision=conditions.revision,
            )

        versions = thaw(payload.get("versions", {}))
        marker = template_values.get("analysis_contract_version")
        supported = supported_condition_versions(
            payload.get("profile"),
            reader_level=payload.get("reader_level"),
            english_help=bool(payload.get("english_help", False)),
            analysis_contract_version=marker,
        )
        if (
            not isinstance(versions, dict)
            or any(type(v) is not int for v in versions.values())
            or versions not in supported
        ):
            return self._incompatible(
                book_id,
                reason_code=IllustrationReasonCode.SOURCE_INCOMPATIBLE.value,
                source_sha256=actual_source_sha,
                conditions_id=conditions.id,
                conditions_revision=conditions.revision,
            )

        raw_name_policy = payload.get("name_translation_policy")
        name_policy: NameTranslationPolicy | None = None
        if raw_name_policy is not None:
            try:
                name_policy = NameTranslationPolicy(**thaw(raw_name_policy))
            except (TypeError, ValueError):
                return self._incompatible(
                    book_id,
                    reason_code=IllustrationReasonCode.SOURCE_INCOMPATIBLE.value,
                    source_sha256=actual_source_sha,
                    conditions_id=conditions.id,
                    conditions_revision=conditions.revision,
                )

        document = tx.documents.get(book_id)
        if document is None:
            return IllustrationSourceSnapshot(
                book_id=book_id,
                compatible=True,
                ready=False,
                reason_code=IllustrationReasonCode.NARRATIVE_MISSING.value,
                source_sha256=actual_source_sha,
                conditions_id=conditions.id,
                conditions_revision=conditions.revision,
                expected_chapters=0,
                ready_chapters=0,
                skipped_sections=0,
                blocking_section_id=None,
                content_section_ids=(),
                manifest=None,
                manifest_sha256=None,
                source_authority={
                    "conditions_id": conditions.id,
                    "conditions_revision": conditions.revision,
                    "source_digest": actual_source_sha,
                    "manifest_sha256": None,
                },
                entries=(),
            )

        checks = thaw(payload.get("checks", {}))
        llm_params = thaw(payload.get("llm_parameters", {}))
        confidence_threshold = float(checks.get("glossary_confidence_threshold", 0.75))
        default_sequential = bool(checks.get("sequential_analysis", True))
        default_seed = bool(checks.get("seed_first_chapter", False))
        prompts_version = versions.get("prompts", 8)
        max_fragment_blocks = narrative_max_fragment_blocks_for(llm_params)
        model_deps = _pass_model_dependencies(payload)

        context_builder = ContextBuilder(
            content_plan_profile=eff_profile,
            prompt_templates=template_values,
            name_translation_policy=name_policy,
            max_unit_source_chars=int(llm_params.get("max_unit_source_chars", 6000)),
            language_notes_per_chapter=int(checks.get("language_notes_per_chapter", 10)),
            reader_level=str(payload.get("reader_level", "b1")),
        )

        analyses = tx.analyses.list_for_book(book_id)
        analyses_by_section = {a.section_id: a for a in analyses}
        raw_passes = tx.analyses.list_passes_for_book(
            book_id, pass_kind=AnalysisPassKind.NARRATIVE, scope_key=""
        )
        passes_by_section: dict[str, AnalysisPassResult] = {
            p.section_id: p for p in raw_passes
        }

        # Decode and envelope-verify all full NARRATIVE passes up front so we can
        # reconstruct `own_outputs` and prior `accumulated` identically to AnalyzeChapter.
        verified_envelope: dict[
            str, tuple[AnalysisPassResult, NarrativeAnalysisResponse, dict[str, Any] | None, str]
        ] = {}
        for sec in document.ordered_sections:
            saved = passes_by_section.get(sec.id)
            if saved is None:
                continue
            try:
                decoded, input_ctx, resp_sha = _verify_pass_envelope(saved)
                verified_envelope[sec.id] = (saved, decoded, input_ctx, resp_sha)
            except (ValueError, TypeError, KeyError, AttributeError, ApplicationError, DomainError):
                continue

        # Collect outputs from all verified narrative passes across the book (for suggestion filtering).
        all_pass_outputs: dict[str, set[tuple[str, str, str]]] = {}
        for sec_id, (saved, decoded, _ctx, _sha) in verified_envelope.items():
            outs: set[tuple[str, str, str]] = set()
            raw_resp = saved.payload.get("response")
            raw_cands = raw_resp.get("glossary_candidates", ()) if isinstance(raw_resp, Mapping) else ()
            if isinstance(raw_cands, (list, tuple)):
                for item in raw_cands:
                    if isinstance(item, Mapping):
                        outs.add(_candidate_suggestion_key(item))
            outs.update(merged_glossary_suggestion_keys(decoded.glossary_candidates))
            all_pass_outputs[sec_id] = outs

        verified_narratives_for_accumulated: dict[
            str, tuple[NarrativeAnalysisResponse, str | None]
        ] = {}
        entries: list[SectionManifestEntry] = []
        content_section_ids: list[str] = []
        ready_chapters = 0
        skipped_sections = 0
        blocking_section_id: str | None = None

        for section in document.ordered_sections:
            block_digests = {
                block.id: canonical_sha(block.text) for block in section.ordered_blocks
            }
            if not is_illustrable_content_section(document, section):
                skipped_sections += 1
                skip_reason = (
                    "non_content_section"
                    if (document.is_front_matter(section) or getattr(section, "is_notes_only", False))
                    else "no_content"
                )
                # Meaningful part divider NARRATIVE still contributes to prior accumulated!
                if is_meaningful_part_divider(document, section) and section.id in verified_envelope:
                    saved_pd, decoded_pd, _ctx_pd, _sha_pd = verified_envelope[section.id]
                    verified_narratives_for_accumulated[section.id] = (
                        decoded_pd,
                        saved_pd.conditions_id or conditions.id,
                    )
                entries.append(
                    SectionManifestEntry(
                        section_id=section.id,
                        order=section.order,
                        title=section.title,
                        skipped=True,
                        skip_reason=skip_reason,
                        block_digests=block_digests,
                    )
                )
                continue

            content_section_ids.append(section.id)
            existing_analysis = analyses_by_section.get(section.id)
            env = verified_envelope.get(section.id)

            # Detect legacy/mixed ChapterAnalysis without compatible staged NARRATIVE pass.
            if env is None and existing_analysis is not None:
                plan = existing_analysis.content_plan
                if plan is None or plan.profile != "content-plan-v3":
                    return self._incompatible(
                        book_id,
                        reason_code=IllustrationReasonCode.SOURCE_INCOMPATIBLE.value,
                        source_sha256=actual_source_sha,
                        conditions_id=conditions.id,
                        conditions_revision=conditions.revision,
                    )

            if env is None:
                if blocking_section_id is None:
                    blocking_section_id = section.id
                entries.append(
                    SectionManifestEntry(
                        section_id=section.id,
                        order=section.order,
                        title=section.title,
                        skipped=False,
                        skip_reason=None,
                        block_digests=block_digests,
                    )
                )
                continue

            saved, decoded, input_ctx, resp_sha = env
            # Reconstruct inputs and verify saved.input_sha256 + validate_narrative_response.
            matched_scenes = self._verify_section_pass(
                document=document,
                section=section,
                saved=saved,
                decoded=decoded,
                input_ctx=input_ctx,
                analyses=analyses,
                verified_narratives=verified_narratives_for_accumulated,
                all_pass_outputs=all_pass_outputs,
                context_builder=context_builder,
                name_policy=name_policy,
                confidence_threshold=confidence_threshold,
                default_sequential=default_sequential,
                default_seed=default_seed,
                default_prompts_version=prompts_version,
                max_fragment_blocks=max_fragment_blocks,
                model_deps=model_deps,
                fallback_conditions_id=conditions.id,
            )
            if matched_scenes is None:
                if blocking_section_id is None:
                    blocking_section_id = section.id
                entries.append(
                    SectionManifestEntry(
                        section_id=section.id,
                        order=section.order,
                        title=section.title,
                        skipped=False,
                        skip_reason=None,
                        block_digests=block_digests,
                    )
                )
                continue

            verified_narratives_for_accumulated[section.id] = (
                decoded,
                saved.conditions_id or conditions.id,
            )
            ready_chapters += 1
            entries.append(
                SectionManifestEntry(
                    section_id=section.id,
                    order=section.order,
                    title=section.title,
                    skipped=False,
                    skip_reason=None,
                    block_digests=block_digests,
                    pass_id=saved.id,
                    input_sha256=saved.input_sha256,
                    output_sha256=saved.output_sha256,
                    response_sha256=resp_sha,
                    conditions_id=saved.conditions_id or conditions.id,
                    narrative=decoded,
                    scenes=matched_scenes,
                )
            )

        expected_chapters = len(content_section_ids)
        if expected_chapters == 0:
            manifest: dict[str, Any] = {
                "schema_version": 1,
                "book_id": book_id,
                "source_sha256": actual_source_sha,
                "conditions_id": conditions.id,
                "conditions_revision": conditions.revision,
                "sections": [
                    {
                        "section_id": e.section_id,
                        "order": e.order,
                        "title": e.title,
                        "skipped": e.skipped,
                        "skip_reason": e.skip_reason,
                        "block_digests": dict(e.block_digests),
                    }
                    for e in entries
                ],
            }
            manifest_sha = canonical_sha(manifest)
            return IllustrationSourceSnapshot(
                book_id=book_id,
                compatible=True,
                ready=True,
                reason_code=IllustrationReasonCode.NO_CONTENT.value,
                source_sha256=actual_source_sha,
                conditions_id=conditions.id,
                conditions_revision=conditions.revision,
                expected_chapters=0,
                ready_chapters=0,
                skipped_sections=skipped_sections,
                blocking_section_id=None,
                content_section_ids=(),
                manifest=manifest,
                manifest_sha256=manifest_sha,
                source_authority={
                    "conditions_id": conditions.id,
                    "conditions_revision": conditions.revision,
                    "source_digest": actual_source_sha,
                    "manifest_sha256": manifest_sha,
                },
                entries=tuple(entries),
            )

        if blocking_section_id is not None:
            tasks = tx.tasks.list_for_book(book_id)
            failed = any(
                t.task_type is TaskType.ANALYZE_CHAPTER
                and t.scope_id == blocking_section_id
                and t.status is TaskStatus.FAILED
                for t in tasks
            )
            reason = (
                IllustrationReasonCode.NARRATIVE_FAILED.value
                if failed
                else IllustrationReasonCode.NARRATIVE_MISSING.value
            )
            return IllustrationSourceSnapshot(
                book_id=book_id,
                compatible=True,
                ready=False,
                reason_code=reason,
                source_sha256=actual_source_sha,
                conditions_id=conditions.id,
                conditions_revision=conditions.revision,
                expected_chapters=expected_chapters,
                ready_chapters=ready_chapters,
                skipped_sections=skipped_sections,
                blocking_section_id=blocking_section_id,
                content_section_ids=tuple(content_section_ids),
                manifest=None,
                manifest_sha256=None,
                source_authority={
                    "conditions_id": conditions.id,
                    "conditions_revision": conditions.revision,
                    "source_digest": actual_source_sha,
                    "manifest_sha256": None,
                },
                entries=tuple(entries),
            )

        manifest = {
            "schema_version": 1,
            "book_id": book_id,
            "source_sha256": actual_source_sha,
            "conditions_id": conditions.id,
            "conditions_revision": conditions.revision,
            "sections": [
                {
                    "section_id": e.section_id,
                    "order": e.order,
                    "title": e.title,
                    "skipped": e.skipped,
                    "skip_reason": e.skip_reason,
                    "block_digests": dict(e.block_digests),
                    "pass_id": e.pass_id,
                    "input_sha256": e.input_sha256,
                    "output_sha256": e.output_sha256,
                    "response_sha256": e.response_sha256,
                    "conditions_id": e.conditions_id,
                }
                for e in entries
            ],
        }
        manifest_sha = canonical_sha(manifest)
        return IllustrationSourceSnapshot(
            book_id=book_id,
            compatible=True,
            ready=True,
            reason_code=None,
            source_sha256=actual_source_sha,
            conditions_id=conditions.id,
            conditions_revision=conditions.revision,
            expected_chapters=expected_chapters,
            ready_chapters=ready_chapters,
            skipped_sections=skipped_sections,
            blocking_section_id=None,
            content_section_ids=tuple(content_section_ids),
            manifest=manifest,
            manifest_sha256=manifest_sha,
            source_authority={
                "conditions_id": conditions.id,
                "conditions_revision": conditions.revision,
                "source_digest": actual_source_sha,
                "manifest_sha256": manifest_sha,
            },
            entries=tuple(entries),
        )

    def _verify_section_pass(
        self,
        *,
        document,
        section,
        saved: AnalysisPassResult,
        decoded: NarrativeAnalysisResponse,
        input_ctx: dict[str, Any] | None,
        analyses,
        verified_narratives: Mapping[str, tuple[NarrativeAnalysisResponse, str | None]],
        all_pass_outputs: Mapping[str, set[tuple[str, str, str]]],
        context_builder: ContextBuilder,
        name_policy: NameTranslationPolicy | None,
        confidence_threshold: float,
        default_sequential: bool,
        default_seed: bool,
        default_prompts_version: int,
        max_fragment_blocks: int,
        model_deps: Mapping[str, Any],
        fallback_conditions_id: str,
    ) -> tuple[Scene, ...] | None:
        blocks = textual_blocks(section)
        aliases = {b.id: b.id for b in blocks}
        texts = {b.id: b.text for b in blocks}
        visible = {b.id: source_text_for_model(b) for b in blocks}
        is_notes_only = section.is_notes_only or document.is_front_matter(section)

        raw_prev_inputs = saved.payload.get("input_suggestions", ())
        previous_inputs: set[tuple[str, str, str]] = (
            {
                (str(item[0]), str(item[1]), str(item[2]))
                for item in raw_prev_inputs
                if isinstance(item, (list, tuple)) and len(item) == 3
            }
            if isinstance(raw_prev_inputs, (list, tuple))
            else set()
        )
        own_outputs: set[tuple[str, str, str]] = set()
        for outs in all_pass_outputs.values():
            own_outputs.update(outs)

        if input_ctx is not None:
            ctx_decisions = deserialize_user_decisions(
                input_ctx["user_decisions"],
                book_id=document.id,
                source_language=document.source_language,
                target_language=document.target_language,
            )
            ctx_suggestions = deserialize_nonbinding_suggestions(
                input_ctx["nonbinding_suggestions"],
                book_id=document.id,
                source_language=document.source_language,
                target_language=document.target_language,
            )
            concepts = (*ctx_decisions, *ctx_suggestions)
            eff = NarrativePassInputs.for_section(
                document,
                section,
                analyses=analyses,
                concepts=concepts,
                name_translation_policy=name_policy,
                glossary_confidence_threshold=confidence_threshold,
                sequential_analysis=input_ctx["sequential_analysis"],
                parallel_batch=input_ctx["parallel_batch"],
                seed_active=input_ctx["seed_active"],
                verified_narratives_by_section=verified_narratives,
                fallback_conditions_id=fallback_conditions_id,
            )
            suggestions = eff.nonbinding_suggestions
            if own_outputs and suggestions:
                filtered_out = (own_outputs | set(merged_glossary_suggestion_keys(eff.accumulated))) - previous_inputs
                suggestions = tuple(
                    s
                    for s in suggestions
                    if _suggestion_key(s, document.target_language) not in filtered_out
                )
            computed_sha = context_builder.narrative_input_sha256(
                document,
                section,
                eff.accumulated,
                eff.user_decisions,
                suggestions,
                model_dependencies=model_deps,
                max_fragment_blocks=max_fragment_blocks,
            )
            if computed_sha != saved.input_sha256:
                return None
            prompts_ver = input_ctx["prompts_version"]
            enforce_v7 = type(prompts_ver) is int and prompts_ver >= 7
            mandatory_facts = glossary_context_constraints(concepts)
            known_names = _known_names_from_glossary(
                eff.accumulated,
                eff.user_decisions,
                suggestions,
                target_language=document.target_language,
            )
        else:
            # Legacy v1 pass without input_context: cryptographic check against empty
            # pre-analysis user_decisions/suggestions (deterministic parallel_batch probe
            # when sequential_analysis=False), immune to post-analysis glossary edits.
            candidate_modes = (
                ((False, False),)
                if default_sequential
                else ((False, default_seed), (True, default_seed))
            )
            matched_eff = None
            for parallel_batch, seed_active in candidate_modes:
                eff = NarrativePassInputs.for_section(
                    document,
                    section,
                    analyses=analyses,
                    concepts=(),
                    name_translation_policy=name_policy,
                    glossary_confidence_threshold=confidence_threshold,
                    sequential_analysis=default_sequential,
                    parallel_batch=parallel_batch,
                    seed_active=seed_active,
                    verified_narratives_by_section=verified_narratives,
                    fallback_conditions_id=fallback_conditions_id,
                )
                computed_sha = context_builder.narrative_input_sha256(
                    document,
                    section,
                    eff.accumulated,
                    (),
                    (),
                    model_dependencies=model_deps,
                    max_fragment_blocks=max_fragment_blocks,
                )
                if computed_sha == saved.input_sha256:
                    matched_eff = eff
                    break
            if matched_eff is None:
                return None
            enforce_v7 = type(default_prompts_version) is int and default_prompts_version >= 7
            mandatory_facts = {}
            known_names = _known_names_from_glossary(
                matched_eff.accumulated,
                (),
                (),
                target_language=document.target_language,
            )

        effective_name_policy = name_policy if enforce_v7 else None
        try:
            scenes, _facts = validate_narrative_response(
                decoded,
                aliases,
                texts,
                visible,
                ordered_block_ids=tuple(aliases),
                is_notes_only=is_notes_only,
                mandatory_facts=mandatory_facts,
                blocks=blocks,
                source_offsets=True,
                source_language=document.source_language,
                target_language=document.target_language,
                language_detector=self._language_detector,
                known_names=known_names,
                name_translation_policy=effective_name_policy,
                enforce_strict_gender_subject=enforce_v7,
            )
            return scenes
        except (ApplicationError, DomainError, ValueError, TypeError):
            return None


__all__ = [
    "IllustrationSource",
    "IllustrationSourceSnapshot",
    "SectionManifestEntry",
    "compute_chapter_fingerprint",
    "compute_reference_fingerprint",
]
