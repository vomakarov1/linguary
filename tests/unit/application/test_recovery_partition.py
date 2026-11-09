from dataclasses import replace
import pytest
from linguary.application.recovery import dependency_intervals, merge_unit_boundaries, local_failure_signature, root_manifest
from linguary.domain.content_plan import ContentPlan, ContextFact, Difficulty, SourceAnchor, RequiredPoint
from linguary.domain.llm_log import Violation


def plan():
    return ContentPlan(difficulties=(Difficulty('d', (SourceAnchor('b1',0,1,'a'), SourceAnchor('b3',0,1,'c')), 'b1', 'grammar', 'critical', 'explanation', (RequiredPoint('p','Explain relation'),)),))


def test_indivisible_cross_block_dependency():
    intervals = dependency_intervals(('b1','b2','b3','b4'), plan())
    assert intervals == (('b1','b2','b3'), ('b4',))
    assert merge_unit_boundaries((('b1','b2'), ('b3','b4')), intervals) == (('b1','b2','b3','b4'),)


def test_global_or_unknown_error_not_split():
    assert local_failure_signature(Violation('quotes','translation_unit','global'), ('b1',)) is None
    assert local_failure_signature(Violation('quotes','translation_unit','unknown'), ('b1',)) is None
    assert local_failure_signature(Violation('quotes','translation_unit','local',('foreign',)), ('b1',)) is None
    assert local_failure_signature(Violation('quotes','translation_unit','local',('b1',)), ('b1',))


def test_original_group_boundary_frozen():
    kwargs = dict(book_id='book', conditions_id='conditions', glossary_sha='g', analysis_sha='a', plan_sha=None,
        ordered_blocks=('b1','b2'), treatments=('TRANSLATE','TRANSLATE'), boundaries=(('b1','b2'),),
        atomic_intervals=(('b1',),('b2',)), max_source_chars=100)
    first = root_manifest(**kwargs)
    assert first['root_key'] == root_manifest(**kwargs)['root_key']
    assert 'unit_id' not in first
    assert first['root_key'] != root_manifest(**{**kwargs,'max_source_chars': 101})['root_key']


@pytest.mark.parametrize('cross_block', [False, True])
def test_prepare_roots_splits_technical_group_only_between_plan_dependencies(cross_block):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from linguary.application.recovery import Recovery
    from linguary.domain import BlockTreatment, BlockType

    roots, scopes = [], []
    class WorkStore:
        def latest_conditions(self, book_id):
            return SimpleNamespace(id='00000000-0000-0000-0000-000000000001', payload={'checks': {'language_notes_per_chapter': 2}})
        def find_root_by_key(self, key):
            return None
        def add_root(self, root):
            roots.append(root)
        def list_scopes(self, root_id):
            return tuple(s for s in scopes if s.root_id == root_id)
        def add_scope(self, scope):
            scopes.append(scope)
    class Uow:
        @contextmanager
        def transaction(self):
            yield SimpleNamespace(work=store)
    store = WorkStore()
    from linguary.domain import GlossarySnapshot
    from linguary.domain.analysis import ChapterAnalysis
    from linguary.domain import TreatmentPlan, TreatmentPlanItem, CoverageStatus
    ids = ('b1', 'b2', 'b3', 'b4')
    treatment = TreatmentPlan('section', tuple(TreatmentPlanItem(b, BlockTreatment.TRANSLATE) for b in ids))
    analysis = ChapterAnalysis(id='analysis', section_id='section', coverage_status=CoverageStatus.COMPLETE,
        treatment_plan=treatment, translation_instructions='Translate.', chapter_summary='Scene.',
        content_plan=plan() if cross_block else None)
    blocks = tuple(SimpleNamespace(id=b, text='abcd', parent_id=None,
        type=BlockType.PARAGRAPH, attributes={'split_group': 'original-paragraph'}) for b in ids)
    units = (SimpleNamespace(block_ids=ids, items=tuple(SimpleNamespace(block=b) for b in blocks)),)
    work = SimpleNamespace(document=SimpleNamespace(id='book'), analysis=analysis,
        run=SimpleNamespace(id='run', glossary_snapshot=GlossarySnapshot()), processed_block_ids=set())
    Recovery(Uow()).prepare_roots(work, units, max_source_chars=100, validator=None)
    assert len(roots) == 1
    atoms = tuple(tuple(group) for group in roots[0].manifest['atomic_intervals'])
    assert atoms == ((('b1', 'b2', 'b3'), ('b4',)) if cross_block else tuple((b,) for b in ids))
    assert scopes[0].ordered_blocks == ids
    assert tuple(b.attributes['split_group'] for b in blocks) == ('original-paragraph',) * 4


def test_context_facts_do_not_merge_distant_units_into_single_recovery_root():
    from contextlib import contextmanager
    from types import SimpleNamespace
    from linguary.application.recovery import Recovery
    from linguary.domain import BlockTreatment, BlockType, CoverageStatus, GlossarySnapshot, TreatmentPlan, TreatmentPlanItem
    from linguary.domain.analysis import ChapterAnalysis

    plan_with_fact = ContentPlan(
        version=3,
        profile='content-plan-v3',
        context_facts=(
            ContextFact(
                id='f1',
                kind='pronoun_referent',
                anchors=(SourceAnchor('b1', 0, 1, 'a'),),
                subject_anchor=SourceAnchor('b4', 0, 1, 'd'),
                value='Alice',
                certainty='explicit',
            ),
        ),
    )
    ids = ('b1', 'b2', 'b3', 'b4')
    intervals = dependency_intervals(ids, plan_with_fact)
    assert intervals == (('b1',), ('b2',), ('b3',), ('b4',))
    assert merge_unit_boundaries((('b1', 'b2'), ('b3', 'b4')), intervals) == (('b1', 'b2'), ('b3', 'b4'))

    roots, scopes = [], []
    class WorkStore:
        def latest_conditions(self, book_id):
            return SimpleNamespace(id='00000000-0000-0000-0000-000000000001', payload={'checks': {'language_notes_per_chapter': 2}})
        def find_root_by_key(self, key):
            return None
        def add_root(self, root):
            roots.append(root)
        def list_scopes(self, root_id):
            return tuple(s for s in scopes if s.root_id == root_id)
        def add_scope(self, scope):
            scopes.append(scope)
    class Uow:
        @contextmanager
        def transaction(self):
            yield SimpleNamespace(work=store)
    store = WorkStore()

    treatment = TreatmentPlan('section', tuple(TreatmentPlanItem(b, BlockTreatment.TRANSLATE) for b in ids))
    analysis = ChapterAnalysis(
        id='analysis',
        section_id='section',
        coverage_status=CoverageStatus.COMPLETE,
        treatment_plan=treatment,
        translation_instructions='Translate.',
        chapter_summary='Scene.',
        content_plan=plan_with_fact,
    )
    blocks = {
        b: SimpleNamespace(id=b, text='abcd', parent_id=None, type=BlockType.PARAGRAPH, attributes={})
        for b in ids
    }
    units = (
        SimpleNamespace(block_ids=('b1', 'b2'), items=tuple(SimpleNamespace(block=blocks[b]) for b in ('b1', 'b2'))),
        SimpleNamespace(block_ids=('b3', 'b4'), items=tuple(SimpleNamespace(block=blocks[b]) for b in ('b3', 'b4'))),
    )
    work = SimpleNamespace(
        document=SimpleNamespace(id='book'),
        analysis=analysis,
        run=SimpleNamespace(id='run', glossary_snapshot=GlossarySnapshot()),
        processed_block_ids=set(),
    )
    Recovery(Uow()).prepare_roots(work, units, max_source_chars=100, validator=None)
    assert len(roots) == 2
    assert [tuple(r.manifest['ordered_blocks']) for r in roots] == [('b1', 'b2'), ('b3', 'b4')]


def test_distant_difficulty_anchors_do_not_merge_entire_chapter_into_single_recovery_root():
    from contextlib import contextmanager
    from types import SimpleNamespace
    from linguary.application.recovery import Recovery
    from linguary.domain import BlockTreatment, BlockType, CoverageStatus, GlossarySnapshot, TreatmentPlan, TreatmentPlanItem
    from linguary.domain.analysis import ChapterAnalysis

    plan_with_distant_difficulty = ContentPlan(
        version=3,
        profile='content-plan-v3',
        difficulties=(
            Difficulty(
                id='d-distant',
                anchors=(
                    SourceAnchor('b1', 0, 1, 'a'),
                    SourceAnchor('b6', 0, 1, 'f'),
                ),
                owner_block_id='b1',
                kind='idiom',
                priority='critical',
                help_kind='explanation',
                required_points=(RequiredPoint('p1', 'Explain idiom'),),
            ),
        ),
    )
    ids = ('b1', 'b2', 'b3', 'b4', 'b5', 'b6')
    intervals = dependency_intervals(ids, plan_with_distant_difficulty)
    assert intervals == (('b1',), ('b2',), ('b3',), ('b4',), ('b5',), ('b6',))
    assert merge_unit_boundaries((('b1', 'b2'), ('b3', 'b4'), ('b5', 'b6')), intervals) == (
        ('b1', 'b2'),
        ('b3', 'b4'),
        ('b5', 'b6'),
    )
    # Explicit structural indivisible edges are still merged unconditionally
    assert dependency_intervals(ids, None, indivisible=(('b1', 'b6'),)) == (ids,)
    # Custom max_difficulty_span allows wider spans when explicitly requested
    assert dependency_intervals(ids, plan_with_distant_difficulty, max_difficulty_span=5) == (ids,)

    roots, scopes = [], []

    class WorkStore:
        def latest_conditions(self, book_id):
            return SimpleNamespace(
                id='00000000-0000-0000-0000-000000000001',
                payload={'checks': {'language_notes_per_chapter': 2}},
            )

        def find_root_by_key(self, key):
            return None

        def add_root(self, root):
            roots.append(root)

        def list_scopes(self, root_id):
            return tuple(s for s in scopes if s.root_id == root_id)

        def add_scope(self, scope):
            scopes.append(scope)

    class Uow:
        @contextmanager
        def transaction(self):
            yield SimpleNamespace(work=store)

    store = WorkStore()
    treatment = TreatmentPlan('section', tuple(TreatmentPlanItem(b, BlockTreatment.TRANSLATE) for b in ids))
    analysis = ChapterAnalysis(
        id='analysis',
        section_id='section',
        coverage_status=CoverageStatus.COMPLETE,
        treatment_plan=treatment,
        translation_instructions='Translate.',
        chapter_summary='Scene.',
        content_plan=plan_with_distant_difficulty,
    )
    blocks = {
        b: SimpleNamespace(id=b, text='abcd', parent_id=None, type=BlockType.PARAGRAPH, attributes={})
        for b in ids
    }
    units = (
        SimpleNamespace(block_ids=('b1', 'b2'), items=tuple(SimpleNamespace(block=blocks[b]) for b in ('b1', 'b2'))),
        SimpleNamespace(block_ids=('b3', 'b4'), items=tuple(SimpleNamespace(block=blocks[b]) for b in ('b3', 'b4'))),
        SimpleNamespace(block_ids=('b5', 'b6'), items=tuple(SimpleNamespace(block=blocks[b]) for b in ('b5', 'b6'))),
    )
    work = SimpleNamespace(
        document=SimpleNamespace(id='book'),
        analysis=analysis,
        run=SimpleNamespace(id='run', glossary_snapshot=GlossarySnapshot()),
        processed_block_ids=set(),
    )
    Recovery(Uow()).prepare_roots(work, units, max_source_chars=100, validator=None)
    assert len(roots) == 3
    assert [tuple(r.manifest['ordered_blocks']) for r in roots] == [('b1', 'b2'), ('b3', 'b4'), ('b5', 'b6')]


@pytest.mark.asyncio
async def test_local_validator_violation_triggers_recovery_split_on_repeated_signature(
    monkeypatch: pytest.MonkeyPatch,
):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import linguary.application.block_processing as block_processing_module
    import linguary.application.diagnostics as diagnostics_module
    from processing_fakes import (
        BOOK_ID,
        SECTION_ID,
        TARGET_LANGUAGE,
        make_blocks,
        make_response,
        make_unit,
        make_validator,
        note_ids,
    )
    from linguary.application.block_processing import TranslationCall
    from linguary.application.context_builder import ContextBuilder
    from linguary.application.llm_dto import BlockResultResponse
    from linguary.application.ports.llm import LLMResponse
    from linguary.application.recovery import RecoverySplit, local_failure_signature
    from linguary.application.retry_policy import RetryPolicy
    from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
    from linguary.domain import (
        BlockTreatment,
        BookMetadata,
        CanonicalDocument,
        ChapterAnalysis,
        DocumentSection,
        GlossarySnapshot,
        TreatmentPlan,
    )

    monkeypatch.setattr(
        diagnostics_module,
        "diagnostics_for",
        lambda uow: SimpleNamespace(ensure_available=lambda: None, fail=lambda: None),
    )
    monkeypatch.setattr(
        block_processing_module,
        "record_rejection",
        lambda uow, response, error: None,
    )

    blocks = make_blocks(2)
    unit = make_unit(blocks)
    section = DocumentSection(
        id=SECTION_ID,
        document_id=BOOK_ID,
        title="Chapter 1",
        order=0,
        source_resource_path="ch1.xhtml",
        blocks=blocks,
    )
    document = CanonicalDocument(
        id=BOOK_ID,
        metadata=BookMetadata(title="Book", authors=("Author",)),
        source_language="en",
        target_language=TARGET_LANGUAGE,
        sections=(section,),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=section.id,
        treatment_plan=TreatmentPlan.of_treatments(
            section.id, ((item.block.id, item.treatment) for item in unit.items)
        ),
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=unit.block_ids,
    )
    context = ContextBuilder().for_translation_unit(
        document=document,
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=GlossarySnapshot(),
    )
    context = replace(
        context,
        request=replace(
            context.request,
            book_id=BOOK_ID,
            stage=SimpleNamespace(value="translation_unit"),
        ),
    )

    recorded_violations = []
    seen_signatures: list[tuple[str, tuple[str, ...]] | None] = []
    physical_ctx = SimpleNamespace(
        conditions_id="00000000-0000-0000-0000-000000000001",
        attempt_id="attempt-1",
        root_id="root-1",
        scope_id="scope-1",
        stage="translation_unit",
        ordered_block_ids=unit.block_ids,
        policy_version=CURRENT_TRANSLATION_POLICY,
        prompt_hash="hash",
    )

    class WorkStore:
        def get_root(self, root_id):
            return SimpleNamespace(
                id=root_id,
                policy_version=CURRENT_TRANSLATION_POLICY,
                conditions_id=physical_ctx.conditions_id,
                calls_used=1,
                extra_calls=0,
                manifest={"policy": {"max_calls": 10, "max_scope_calls": 4}},
            )

        def get_conditions(self, conditions_id):
            return SimpleNamespace(
                id=conditions_id,
                payload={"models": {"processing": "gpt-test"}, "versions": {"validators": 1}},
            )

        def begin_recovery_attempt(self, root_id, scope_id, **kwargs):
            return physical_ctx

        def owns_recovery_attempt(self, attempt_id):
            return True

        def record_recovery_failure(self, attempt_id, violation):
            recorded_violations.append(violation)
            sig = local_failure_signature(violation, unit.block_ids)
            seen_signatures.append(sig)
            return sig is not None and seen_signatures.count(sig) >= 2

        def result_bindings(self, run_id):
            return {}

    class RunsStore:
        def active_for_book(self, book_id):
            return None

    class Uow:
        @contextmanager
        def transaction(self):
            yield SimpleNamespace(work=WorkStore(), runs=RunsStore(), llm_calls=SimpleNamespace())

    generate_calls = 0

    class FakeLLMClient:
        async def generate(self, request):
            nonlocal generate_calls
            generate_calls += 1
            return LLMResponse(
                payload=make_response(
                    (
                        BlockResultResponse(
                            source_block_id="b1",
                            treatment=BlockTreatment.TRANSLATE,
                            translated_text="Первая строка.\nВторая строка.",
                        ),
                        BlockResultResponse(
                            source_block_id="b2",
                            treatment=BlockTreatment.TRANSLATE,
                            translated_text="Перевод второго блока.",
                        ),
                    )
                ),
                provider="fake",
                model="gpt-test",
                truncated=False,
                call_context=request.call_context,
            )

    call = TranslationCall(
        unit_of_work=Uow(),
        llm_client=FakeLLMClient(),
        retry_policy=RetryPolicy(max_attempts=3),
        validator=make_validator(),
        sleep=None,
    )

    with pytest.raises(RecoverySplit) as exc_info:
        await call.request(
            context,
            unit=unit,
            target_language=TARGET_LANGUAGE,
            note_ids=note_ids(),
            recovery=("root-1", "scope-1"),
        )

    assert (exc_info.value.root_id, exc_info.value.scope_id) == ("root-1", "scope-1")
    assert generate_calls == 2
    assert len(recorded_violations) == 2
    assert all(v.rule_code == "paragraph_count_mismatch" for v in recorded_violations)
    assert all(v.scope == "local" and v.block_ids == (blocks[0].id,) for v in recorded_violations)


@pytest.mark.asyncio
async def test_provider_truncated_multi_block_unit_triggers_immediate_recovery_split(
    monkeypatch: pytest.MonkeyPatch,
):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import linguary.application.block_processing as block_processing_module
    import linguary.application.diagnostics as diagnostics_module
    from processing_fakes import (
        BOOK_ID,
        SECTION_ID,
        TARGET_LANGUAGE,
        make_blocks,
        make_response,
        make_unit,
        make_validator,
        note_ids,
    )
    from linguary.application.block_processing import TranslationCall
    from linguary.application.context_builder import ContextBuilder
    from linguary.application.llm_dto import BlockResultResponse
    from linguary.application.ports.llm import LLMResponse
    from linguary.application.recovery import RecoverySplit
    from linguary.application.retry_policy import RetryPolicy
    from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
    from linguary.domain import (
        BlockTreatment,
        BookMetadata,
        CanonicalDocument,
        ChapterAnalysis,
        DocumentSection,
        GlossarySnapshot,
        TreatmentPlan,
    )

    monkeypatch.setattr(
        diagnostics_module,
        "diagnostics_for",
        lambda uow: SimpleNamespace(ensure_available=lambda: None, fail=lambda: None),
    )
    monkeypatch.setattr(
        block_processing_module,
        "record_rejection",
        lambda uow, response, error: None,
    )

    blocks = make_blocks(2)
    unit = make_unit(blocks)
    section = DocumentSection(
        id=SECTION_ID,
        document_id=BOOK_ID,
        title="Chapter 1",
        order=0,
        source_resource_path="ch1.xhtml",
        blocks=blocks,
    )
    document = CanonicalDocument(
        id=BOOK_ID,
        metadata=BookMetadata(title="Book", authors=("Author",)),
        source_language="en",
        target_language=TARGET_LANGUAGE,
        sections=(section,),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=section.id,
        treatment_plan=TreatmentPlan.of_treatments(
            section.id, ((item.block.id, item.treatment) for item in unit.items)
        ),
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=unit.block_ids,
    )
    context = ContextBuilder().for_translation_unit(
        document=document,
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=GlossarySnapshot(),
    )
    context = replace(
        context,
        request=replace(
            context.request,
            book_id=BOOK_ID,
            stage=SimpleNamespace(value="translation_unit"),
        ),
    )

    recorded_violations = []
    physical_ctx = SimpleNamespace(
        conditions_id="00000000-0000-0000-0000-000000000001",
        attempt_id="attempt-1",
        root_id="root-1",
        scope_id="scope-1",
        stage="translation_unit",
        ordered_block_ids=unit.block_ids,
        policy_version=CURRENT_TRANSLATION_POLICY,
        prompt_hash="hash",
    )

    class WorkStore:
        def get_root(self, root_id):
            return SimpleNamespace(
                id=root_id,
                policy_version=CURRENT_TRANSLATION_POLICY,
                conditions_id=physical_ctx.conditions_id,
                calls_used=1,
                extra_calls=0,
                manifest={"policy": {"max_calls": 10, "max_scope_calls": 4}},
            )

        def get_conditions(self, conditions_id):
            return SimpleNamespace(
                id=conditions_id,
                payload={"models": {"processing": "gpt-test"}, "versions": {"validators": 1}},
            )

        def begin_recovery_attempt(self, root_id, scope_id, **kwargs):
            return physical_ctx

        def owns_recovery_attempt(self, attempt_id):
            return True

        def record_recovery_failure(self, attempt_id, violation):
            recorded_violations.append(violation)
            return False

        def result_bindings(self, run_id):
            return {}

    class RunsStore:
        def active_for_book(self, book_id):
            return None

    class Uow:
        @contextmanager
        def transaction(self):
            yield SimpleNamespace(work=WorkStore(), runs=RunsStore(), llm_calls=SimpleNamespace())

    generate_calls = 0

    class FakeLLMClient:
        async def generate(self, request):
            nonlocal generate_calls
            generate_calls += 1
            return LLMResponse(
                payload=make_response(
                    (
                        BlockResultResponse(
                            source_block_id="b1",
                            treatment=BlockTreatment.TRANSLATE,
                            translated_text="Перевод первого блока.",
                        ),
                        BlockResultResponse(
                            source_block_id="b2",
                            treatment=BlockTreatment.TRANSLATE,
                            translated_text="Перевод второго блока.",
                        ),
                    )
                ),
                provider="fake",
                model="gpt-test",
                truncated=True,
                call_context=request.call_context,
            )

    call = TranslationCall(
        unit_of_work=Uow(),
        llm_client=FakeLLMClient(),
        retry_policy=RetryPolicy(max_attempts=3),
        validator=make_validator(),
        sleep=None,
    )

    with pytest.raises(RecoverySplit) as exc_info:
        await call.request(
            context,
            unit=unit,
            target_language=TARGET_LANGUAGE,
            note_ids=note_ids(),
            recovery=("root-1", "scope-1"),
        )

    assert (exc_info.value.root_id, exc_info.value.scope_id) == ("root-1", "scope-1")
    assert generate_calls == 1
    assert len(recorded_violations) == 1
    assert recorded_violations[0].rule_code == "provider_truncated"


@pytest.mark.asyncio
async def test_record_recovery_failure_does_not_split_single_block_or_single_atom_scope(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import linguary.application.block_processing as block_processing_module
    import linguary.application.diagnostics as diagnostics_module
    from processing_fakes import (
        BOOK_ID,
        SECTION_ID,
        TARGET_LANGUAGE,
        make_blocks,
        make_response,
        make_unit,
        make_validator,
        note_ids,
    )
    from linguary.adapters.db.engine import create_database_engine
    from linguary.adapters.db.models import Base
    from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
    from linguary.application.block_processing import TranslationCall
    from linguary.application.context_builder import ContextBuilder
    from linguary.application.llm_dto import BlockResultResponse
    from linguary.application.ports.llm import LLMResponse
    from linguary.application.ports.persistence import BookRecord
    from linguary.application.recovery import RecoveryError, root_manifest
    from linguary.application.retry_policy import ResultValidationError, RetryPolicy
    from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
    from linguary.domain import (
        BlockTreatment,
        BookMetadata,
        BookStatus,
        CanonicalDocument,
        ChapterAnalysis,
        DocumentSection,
        GlossarySnapshot,
        TreatmentPlan,
    )
    from linguary.domain.book_conditions import BookConditions
    from linguary.domain.recovery import CallAttempt, RecoveryRoot, RecoveryScope

    engine = create_database_engine(tmp_path / "recovery_split_test.db")
    Base.metadata.create_all(engine)
    uow = SqlAlchemyUnitOfWork.for_engine(engine)
    try:
        from datetime import UTC, datetime
        from uuid import uuid4
        cond_id = "00000000-0000-0000-0000-000000000001"
        root_single_id = "00000000-0000-0000-0000-000000000011"
        scope_single_id = "00000000-0000-0000-0000-000000000021"
        root_atom_id = "00000000-0000-0000-0000-000000000012"
        scope_atom_id = "00000000-0000-0000-0000-000000000022"
        root_div_id = "00000000-0000-0000-0000-000000000013"
        scope_div_id = "00000000-0000-0000-0000-000000000023"
        with uow.transaction() as tx:
            tx.books.add(
                BookRecord(
                    id=BOOK_ID,
                    metadata=BookMetadata(title="Book", authors=("Author",)),
                    source_language="en",
                    target_language="ru",
                    status=BookStatus.PROCESSING,
                    source_epub_path="book.epub",
                    created_at=datetime.now(UTC),
                )
            )
            tx.work.add_conditions(
                BookConditions(
                    id=cond_id,
                    book_id=BOOK_ID,
                    revision=1,
                    payload={"schema_version": 1, "models": {"processing": "gpt-test"}},
                    created_at=datetime.now(UTC),
                )
            )
            single_manifest = root_manifest(
                book_id=BOOK_ID,
                conditions_id=cond_id,
                glossary_sha="g",
                analysis_sha="a",
                plan_sha=None,
                ordered_blocks=("b1",),
                treatments=("TRANSLATE",),
                boundaries=(("b1",),),
                atomic_intervals=(("b1",),),
                max_source_chars=100,
            )
            single_root = RecoveryRoot(
                root_single_id,
                BOOK_ID,
                cond_id,
                "key-single",
                single_manifest,
                policy_version=CURRENT_TRANSLATION_POLICY,
                operation_kind="regeneration",
            )
            tx.work.add_root(single_root)
            tx.work.add_scope(RecoveryScope(scope_single_id, root_single_id, ("b1",)))

            atom_manifest = root_manifest(
                book_id=BOOK_ID,
                conditions_id=cond_id,
                glossary_sha="g",
                analysis_sha="a",
                plan_sha=None,
                ordered_blocks=("b1", "b2"),
                treatments=("TRANSLATE", "TRANSLATE"),
                boundaries=(("b1", "b2"),),
                atomic_intervals=(("b1", "b2"),),
                max_source_chars=100,
            )
            atom_root = RecoveryRoot(
                root_atom_id,
                BOOK_ID,
                cond_id,
                "key-atom",
                atom_manifest,
                policy_version=CURRENT_TRANSLATION_POLICY,
            )
            tx.work.add_root(atom_root)
            tx.work.add_scope(RecoveryScope(scope_atom_id, root_atom_id, ("b1", "b2")))

            divisible_manifest = root_manifest(
                book_id=BOOK_ID,
                conditions_id=cond_id,
                glossary_sha="g",
                analysis_sha="a",
                plan_sha=None,
                ordered_blocks=("b1", "b2"),
                treatments=("TRANSLATE", "TRANSLATE"),
                boundaries=(("b1", "b2"),),
                atomic_intervals=(("b1",), ("b2",)),
                max_source_chars=100,
            )
            divisible_root = RecoveryRoot(
                root_div_id,
                BOOK_ID,
                cond_id,
                "key-divisible",
                divisible_manifest,
                policy_version=CURRENT_TRANSLATION_POLICY,
            )
            tx.work.add_root(divisible_root)
            tx.work.add_scope(RecoveryScope(scope_div_id, root_div_id, ("b1", "b2")))

        violation = Violation("paragraph_count_mismatch", "translation_unit", "local", ("b1",))

        for ordinal in (1, 2):
            with uow.transaction() as tx:
                for root_id, scope_id in (
                    (root_single_id, scope_single_id),
                    (root_atom_id, scope_atom_id),
                    (root_div_id, scope_div_id),
                ):
                    attempt_id = str(uuid4())
                    claimed = tx.work.claim_scope(scope_id, expected_generation=ordinal - 1)
                    tx.work.add_attempt(
                        CallAttempt(
                            attempt_id,
                            BOOK_ID,
                            state="rejected",
                            generation=claimed.generation,
                            ordinal=ordinal,
                            conditions_id=cond_id,
                            root_id=root_id,
                            scope_id=scope_id,
                            context={"schema_version": 1, "root_generation": 0},
                        )
                    )
                    should_split = tx.work.record_recovery_failure(attempt_id, violation)
                    if ordinal == 1:
                        assert should_split is False
                    elif root_id in {root_single_id, root_atom_id}:
                        assert should_split is False
                    else:
                        assert should_split is True
    finally:
        engine.dispose()

    monkeypatch.setattr(
        diagnostics_module,
        "diagnostics_for",
        lambda uow_arg: SimpleNamespace(ensure_available=lambda: None, fail=lambda: None),
    )
    monkeypatch.setattr(
        block_processing_module,
        "record_rejection",
        lambda uow_arg, response, error: None,
    )

    blocks = make_blocks(1)
    unit = make_unit(blocks)
    section = DocumentSection(
        id=SECTION_ID,
        document_id=BOOK_ID,
        title="Chapter 1",
        order=0,
        source_resource_path="ch1.xhtml",
        blocks=blocks,
    )
    document = CanonicalDocument(
        id=BOOK_ID,
        metadata=BookMetadata(title="Book", authors=("Author",)),
        source_language="en",
        target_language=TARGET_LANGUAGE,
        sections=(section,),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=section.id,
        treatment_plan=TreatmentPlan.of_treatments(
            section.id, ((item.block.id, item.treatment) for item in unit.items)
        ),
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=unit.block_ids,
    )
    context = ContextBuilder().for_translation_unit(
        document=document,
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=GlossarySnapshot(),
    )
    context = replace(
        context,
        request=replace(
            context.request,
            book_id=BOOK_ID,
            stage=SimpleNamespace(value="block_regeneration"),
        ),
    )
    physical_ctx = SimpleNamespace(
        conditions_id="00000000-0000-0000-0000-000000000001",
        attempt_id="attempt-1",
        root_id="root-1",
        scope_id="scope-1",
        stage="block_regeneration",
        ordered_block_ids=unit.block_ids,
        policy_version=CURRENT_TRANSLATION_POLICY,
        prompt_hash="hash",
    )

    class SingleBlockWorkStore:
        def get_root(self, root_id):
            return SimpleNamespace(
                id=root_id,
                policy_version=CURRENT_TRANSLATION_POLICY,
                conditions_id=physical_ctx.conditions_id,
                calls_used=1,
                extra_calls=0,
                manifest={"policy": {"max_calls": 10, "max_scope_calls": 4}},
            )

        def get_conditions(self, conditions_id):
            return SimpleNamespace(
                id=conditions_id,
                payload={"models": {"processing": "gpt-test"}, "versions": {"validators": 1}},
            )

        def begin_recovery_attempt(self, root_id, scope_id, **kwargs):
            return physical_ctx

        def owns_recovery_attempt(self, attempt_id):
            return True

        def record_recovery_failure(self, attempt_id, violation):
            return True

        def scope_attempt_count(self, scope_id):
            return 3

        def stop_root(self, root_id, reason):
            pass

        def result_bindings(self, run_id):
            return {}

    class RunsStore:
        def active_for_book(self, book_id):
            return None

    class FakeUow:
        @contextmanager
        def transaction(self):
            yield SimpleNamespace(work=SingleBlockWorkStore(), runs=RunsStore(), llm_calls=SimpleNamespace())

    generate_calls = 0

    class FakeLLMClient:
        async def generate(self, request):
            nonlocal generate_calls
            generate_calls += 1
            return LLMResponse(
                payload=make_response(
                    (
                        BlockResultResponse(
                            source_block_id="b1",
                            treatment=BlockTreatment.TRANSLATE,
                            translated_text="Первая строка.\nВторая строка.",
                        ),
                    )
                ),
                provider="fake",
                model="gpt-test",
                truncated=False,
                call_context=request.call_context,
            )

    call = TranslationCall(
        unit_of_work=FakeUow(),
        llm_client=FakeLLMClient(),
        retry_policy=RetryPolicy(max_attempts=3),
        validator=make_validator(),
        sleep=None,
    )

    with pytest.raises(RecoveryError) as exc_info:
        await call.request(
            context,
            unit=unit,
            target_language=TARGET_LANGUAGE,
            note_ids=note_ids(),
            recovery=("root-1", "scope-1"),
        )

    assert not isinstance(exc_info.value, block_processing_module.RecoverySplit)
    assert isinstance(exc_info.value.original_error, ResultValidationError)
    assert generate_calls == 3


@pytest.mark.asyncio
async def test_provider_truncated_indivisible_atomic_interval_does_not_trigger_recovery_split(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    from contextlib import contextmanager
    from datetime import UTC, datetime
    from types import SimpleNamespace
    from uuid import uuid4
    import linguary.application.block_processing as block_processing_module
    import linguary.application.diagnostics as diagnostics_module
    from processing_fakes import (
        BOOK_ID,
        SECTION_ID,
        TARGET_LANGUAGE,
        make_blocks,
        make_response,
        make_unit,
        make_validator,
        note_ids,
    )
    from linguary.adapters.db.engine import create_database_engine
    from linguary.adapters.db.models import Base
    from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
    from linguary.application.block_processing import TranslationCall, _worded_for_the_model
    from linguary.application.context_builder import ContextBuilder
    from linguary.application.llm_dto import BlockResultResponse
    from linguary.application.ports.llm import LLMResponse, LLMResponseFormatError
    from linguary.application.ports.persistence import BookRecord
    from linguary.application.recovery import RecoveryError, root_manifest
    from linguary.application.retry_policy import ResultValidationError, RetryPolicy
    from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
    from linguary.domain import (
        BlockTreatment,
        BookMetadata,
        BookStatus,
        CanonicalDocument,
        ChapterAnalysis,
        DocumentSection,
        GlossarySnapshot,
        TreatmentPlan,
    )
    from linguary.domain.book_conditions import BookConditions
    from linguary.domain.recovery import CallAttempt, RecoveryRoot, RecoveryScope

    engine = create_database_engine(tmp_path / "truncated_atomic_split_test.db")
    Base.metadata.create_all(engine)
    uow = SqlAlchemyUnitOfWork.for_engine(engine)
    try:
        cond_id = "00000000-0000-0000-0000-000000000001"
        root_atom_id = "00000000-0000-0000-0000-000000000012"
        scope_atom_id = "00000000-0000-0000-0000-000000000022"
        root_div_id = "00000000-0000-0000-0000-000000000013"
        scope_div_id = "00000000-0000-0000-0000-000000000023"
        with uow.transaction() as tx:
            tx.books.add(
                BookRecord(
                    id=BOOK_ID,
                    metadata=BookMetadata(title="Book", authors=("Author",)),
                    source_language="en",
                    target_language="ru",
                    status=BookStatus.PROCESSING,
                    source_epub_path="book.epub",
                    created_at=datetime.now(UTC),
                )
            )
            tx.work.add_conditions(
                BookConditions(
                    id=cond_id,
                    book_id=BOOK_ID,
                    revision=1,
                    payload={"schema_version": 1, "models": {"processing": "gpt-test"}},
                    created_at=datetime.now(UTC),
                )
            )
            atom_manifest = root_manifest(
                book_id=BOOK_ID,
                conditions_id=cond_id,
                glossary_sha="g",
                analysis_sha="a",
                plan_sha=None,
                ordered_blocks=("b1", "b2"),
                treatments=("TRANSLATE", "TRANSLATE"),
                boundaries=(("b1", "b2"),),
                atomic_intervals=(("b1", "b2"),),
                max_source_chars=100,
            )
            tx.work.add_root(
                RecoveryRoot(
                    root_atom_id,
                    BOOK_ID,
                    cond_id,
                    "key-atom",
                    atom_manifest,
                    policy_version=CURRENT_TRANSLATION_POLICY,
                )
            )
            tx.work.add_scope(RecoveryScope(scope_atom_id, root_atom_id, ("b1", "b2")))

            divisible_manifest = root_manifest(
                book_id=BOOK_ID,
                conditions_id=cond_id,
                glossary_sha="g",
                analysis_sha="a",
                plan_sha=None,
                ordered_blocks=("b1", "b2"),
                treatments=("TRANSLATE", "TRANSLATE"),
                boundaries=(("b1", "b2"),),
                atomic_intervals=(("b1",), ("b2",)),
                max_source_chars=100,
            )
            tx.work.add_root(
                RecoveryRoot(
                    root_div_id,
                    BOOK_ID,
                    cond_id,
                    "key-divisible",
                    divisible_manifest,
                    policy_version=CURRENT_TRANSLATION_POLICY,
                )
            )
            tx.work.add_scope(RecoveryScope(scope_div_id, root_div_id, ("b1", "b2")))

        truncated_violation = Violation("provider_truncated", "translation_unit", "unknown")
        with uow.transaction() as tx:
            atom_attempt_id = str(uuid4())
            claimed_atom = tx.work.claim_scope(scope_atom_id, expected_generation=0)
            tx.work.add_attempt(
                CallAttempt(
                    atom_attempt_id,
                    BOOK_ID,
                    state="rejected",
                    generation=claimed_atom.generation,
                    ordinal=1,
                    conditions_id=cond_id,
                    root_id=root_atom_id,
                    scope_id=scope_atom_id,
                    context={"schema_version": 1, "root_generation": 0},
                )
            )
            assert tx.work.record_recovery_failure(atom_attempt_id, truncated_violation) is False

            div_attempt_id = str(uuid4())
            claimed_div = tx.work.claim_scope(scope_div_id, expected_generation=0)
            tx.work.add_attempt(
                CallAttempt(
                    div_attempt_id,
                    BOOK_ID,
                    state="rejected",
                    generation=claimed_div.generation,
                    ordinal=1,
                    conditions_id=cond_id,
                    root_id=root_div_id,
                    scope_id=scope_div_id,
                    context={"schema_version": 1, "root_generation": 0},
                )
            )
            assert tx.work.record_recovery_failure(div_attempt_id, truncated_violation) is True
    finally:
        engine.dispose()

    monkeypatch.setattr(
        diagnostics_module,
        "diagnostics_for",
        lambda uow_arg: SimpleNamespace(ensure_available=lambda: None, fail=lambda: None),
    )
    monkeypatch.setattr(
        block_processing_module,
        "record_rejection",
        lambda uow_arg, response, error: None,
    )

    blocks = make_blocks(2)
    unit = make_unit(blocks)
    section = DocumentSection(
        id=SECTION_ID,
        document_id=BOOK_ID,
        title="Chapter 1",
        order=0,
        source_resource_path="ch1.xhtml",
        blocks=blocks,
    )
    document = CanonicalDocument(
        id=BOOK_ID,
        metadata=BookMetadata(title="Book", authors=("Author",)),
        source_language="en",
        target_language=TARGET_LANGUAGE,
        sections=(section,),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=section.id,
        treatment_plan=TreatmentPlan.of_treatments(
            section.id, ((item.block.id, item.treatment) for item in unit.items)
        ),
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=unit.block_ids,
    )
    context = ContextBuilder().for_translation_unit(
        document=document,
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=GlossarySnapshot(),
    )
    context = replace(
        context,
        request=replace(
            context.request,
            book_id=BOOK_ID,
            stage=SimpleNamespace(value="translation_unit"),
        ),
    )

    rephrased = _worded_for_the_model(ResultValidationError("ошибка в b1", expectation=None), context)
    assert isinstance(rephrased, ResultValidationError)
    assert rephrased.expectation == context.expectation

    physical_ctx = SimpleNamespace(
        conditions_id="00000000-0000-0000-0000-000000000001",
        attempt_id="attempt-1",
        root_id="root-atom",
        scope_id="scope-atom",
        stage="translation_unit",
        ordered_block_ids=unit.block_ids,
        policy_version=CURRENT_TRANSLATION_POLICY,
        prompt_hash="hash",
    )

    class IndivisibleWorkStore:
        def get_root(self, root_id):
            return SimpleNamespace(
                id=root_id,
                policy_version=CURRENT_TRANSLATION_POLICY,
                conditions_id=physical_ctx.conditions_id,
                calls_used=1,
                extra_calls=0,
                manifest={
                    "policy": {"max_calls": 10, "max_scope_calls": 4, "max_depth": 4, "max_leaves": 8},
                    "atomic_intervals": [list(unit.block_ids)],
                },
            )

        def get_conditions(self, conditions_id):
            return SimpleNamespace(
                id=conditions_id,
                payload={"models": {"processing": "gpt-test"}, "versions": {"validators": 1}},
            )

        def begin_recovery_attempt(self, root_id, scope_id, **kwargs):
            return physical_ctx

        def owns_recovery_attempt(self, attempt_id):
            return True

        def record_recovery_failure(self, attempt_id, violation):
            return False

        def list_scopes(self, root_id):
            return (SimpleNamespace(id="scope-atom", parent_id=None, status="pending"),)

        def scope_attempt_count(self, scope_id):
            return 2

        def stop_root(self, root_id, reason):
            pass

        def result_bindings(self, run_id):
            return {}

    class RunsStore:
        def active_for_book(self, book_id):
            return None

    class FakeUow:
        @contextmanager
        def transaction(self):
            yield SimpleNamespace(work=IndivisibleWorkStore(), runs=RunsStore(), llm_calls=SimpleNamespace())

    generate_calls = 0

    class FakeLLMClient:
        async def generate(self, request):
            nonlocal generate_calls
            generate_calls += 1
            return LLMResponse(
                payload=make_response(
                    (
                        BlockResultResponse(
                            source_block_id="b1",
                            treatment=BlockTreatment.TRANSLATE,
                            translated_text="Перевод первого блока.",
                        ),
                        BlockResultResponse(
                            source_block_id="b2",
                            treatment=BlockTreatment.TRANSLATE,
                            translated_text="Перевод второго блока.",
                        ),
                    )
                ),
                provider="fake",
                model="gpt-test",
                truncated=True,
                call_context=request.call_context,
            )

    call = TranslationCall(
        unit_of_work=FakeUow(),
        llm_client=FakeLLMClient(),
        retry_policy=RetryPolicy(max_attempts=2),
        validator=make_validator(),
        sleep=None,
    )

    with pytest.raises(RecoveryError) as exc_info:
        await call.request(
            context,
            unit=unit,
            target_language=TARGET_LANGUAGE,
            note_ids=note_ids(),
            recovery=("root-atom", "scope-atom"),
        )

    assert not isinstance(exc_info.value, block_processing_module.RecoverySplit)
    assert exc_info.value.reason == "unsafe_partition"
    assert isinstance(exc_info.value.original_error, ResultValidationError)
    assert generate_calls == 1

    from linguary.application.use_cases.process_chapter import ProcessChapter, ProcessChapterCommand
    from linguary.application.use_cases.regenerate_block import RegenerateBlock, RegenerateBlockCommand

    cand_ch1 = SimpleNamespace(id="c1", analysis_id=analysis.id)
    cand_ch2 = SimpleNamespace(id="c2", analysis_id="analysis-2")
    analysis_ch1 = replace(analysis, glossary_candidates=(cand_ch1,))
    analysis_ch2 = replace(
        analysis,
        id="analysis-2",
        section_id="section-2",
        treatment_plan=replace(analysis.treatment_plan, section_id="section-2"),
        glossary_candidates=(cand_ch2,),
    )

    get_for_section_calls = 0

    class FakeAnalysesRepo:
        def list_for_book(self, book_id):
            return (analysis_ch1, analysis_ch2)

        def get_for_section(self, book_id, sec_id):
            nonlocal get_for_section_calls
            get_for_section_calls += 1
            return analysis_ch1

    fake_tx = SimpleNamespace(
        documents=SimpleNamespace(get=lambda book_id: document),
        analyses=FakeAnalysesRepo(),
        runs=SimpleNamespace(
            active_for_book=lambda book_id: SimpleNamespace(
                id="run-1",
                glossary_snapshot=GlossarySnapshot(),
            )
        ),
        results=SimpleNamespace(list_for_section=lambda run_id, sec_id: ()),
        work=SimpleNamespace(
            latest_conditions=lambda book_id: None,
            list_active_roots=lambda book_id, run_id=None: (),
        ),
    )

    pc = ProcessChapter.__new__(ProcessChapter)
    chapter_work = pc._load(
        ProcessChapterCommand(book_id=BOOK_ID, section_id=section.id),
        transaction=fake_tx,
    )
    assert get_for_section_calls == 0
    assert chapter_work.book_candidates == (cand_ch2,)

    rb = RegenerateBlock.__new__(RegenerateBlock)
    block_work = rb._load(
        RegenerateBlockCommand(book_id=BOOK_ID, block_id=blocks[0].id),
        transaction=fake_tx,
    )
    assert get_for_section_calls == 0
    assert block_work.book_candidates == (cand_ch2,)


