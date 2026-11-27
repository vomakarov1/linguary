from dataclasses import replace

import pytest

from linguary.application.analysis_merge import (
    AnalysisMergeError,
    combine_staged_analysis_responses,
    content_plan_from_response,
    merge_narrative_fragments,
    resolve_scenes,
    validate_narrative_response,
)
from linguary.application.llm_dto import (
    NarrativeAnalysisResponse,
    PedagogicalAnalysisResponse,
    QuotedContextFactResponse,
    QuotedSceneResponse,
    QuotedSceneVisualFactResponse,
    QuotedSourceAnchorResponse,
    TreatmentPlanItemResponse,
)
from linguary.domain import BlockTreatment, BlockType, DocumentBlock, SourceReference, SupportStatus

ALIASES = {'b1': 'id-1', 'b2': 'id-2', 'b3': 'id-3'}
TEXTS = {'id-1': 'Holmes wore a red coat.', 'id-2': 'Watson opened the door.', 'id-3': 'Lestrade arrived.'}
VISIBLE = {alias: TEXTS[block] for alias, block in ALIASES.items()}
ORDER = ('id-1', 'id-2', 'id-3')


def scene(start='b1', end='b3', **kwargs):
    return QuotedSceneResponse(start, end, 'At Baker Street.', **kwargs)


def resolve(scenes):
    return resolve_scenes(scenes, ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER)


@pytest.mark.parametrize('scenes', [
    (scene('b3', 'b1'),),
    (scene('b1', 'b2'), scene('b2', 'b3')),
    (scene('b1', 'b4'),),
])
def test_resolve_scenes_rejects_inverted_overlapping_or_foreign_bounds(scenes):
    with pytest.raises(AnalysisMergeError) as exc:
        resolve(scenes)
    assert exc.value.violation.rule_code == 'scene_bounds'


def test_resolve_scenes_checks_ordered_chapter_membership():
    with pytest.raises(AnalysisMergeError) as exc:
        resolve_scenes((scene(),), ALIASES, TEXTS, VISIBLE, ordered_block_ids=('id-1',))
    assert exc.value.violation.rule_code == 'scene_bounds'


def test_resolve_scenes_resolves_visual_fact_and_canonical_id():
    raw = scene(visual_facts=(QuotedSceneVisualFactResponse(
        QuotedSourceAnchorResponse('b1', 'red coat'), 'Holmes', 'A red coat',
    ),), visual_assumptions=('Warm lamplight',), participants=('Holmes',))
    resolved, = resolve((raw,))
    assert (resolved.start_block_id, resolved.end_block_id, resolved.order) == ('id-1', 'id-3', 0)
    assert resolved.id == resolved.canonical_id
    assert resolved.visual_facts[0].anchor.block_id == 'id-1'
    assert (resolved.visual_facts[0].anchor.start, resolved.visual_facts[0].anchor.end) == (14, 22)
    assert resolved.visual_assumptions == ('Warm lamplight',)


@pytest.mark.parametrize('block,quote,code,expected_blocks,expected_scope', [
    ('b3', 'Lestrade', 'scene_visual_anchor_out_of_bounds', ('id-3',), 'local'),
    ('b1', 'blue coat', 'unresolved_quote', ('id-1',), 'local'),
    ('b9', 'Holmes', 'unresolved_quote', (), 'global'),
])
def test_resolve_scenes_rejects_invalid_visual_fact(block, quote, code, expected_blocks, expected_scope):
    raw = scene('b1', 'b2', visual_facts=(QuotedSceneVisualFactResponse(
        QuotedSourceAnchorResponse(block, quote), 'Visitor', 'Coat',
    ),))
    with pytest.raises(AnalysisMergeError) as exc:
        resolve((raw,))
    assert exc.value.violation is not None
    assert exc.value.violation.rule_code == code
    assert exc.value.violation.scope == expected_scope
    assert exc.value.violation.block_ids == expected_blocks
    assert block in str(exc.value)
    assert quote in str(exc.value)


@pytest.mark.parametrize('scenes', [(), (scene('b2', 'b3'),), (scene('b1', 'b2'),),
                                     (scene('b1', 'b1'), scene('b3', 'b3'))])
def test_validate_narrative_response_rejects_scene_coverage_holes(scenes):
    with pytest.raises(AnalysisMergeError) as exc:
        validate_narrative_response(NarrativeAnalysisResponse('Keep voice.', 'Summary.', scenes=scenes),
                                    ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER)
    assert exc.value.violation.rule_code == 'scene_bounds'


def test_notes_only_sections_have_no_scenes():
    payload = NarrativeAnalysisResponse('Keep voice.', 'Summary.')
    assert validate_narrative_response(payload, ALIASES, TEXTS, VISIBLE,
                                       ordered_block_ids=ORDER, is_notes_only=True) == ((), ())
    with pytest.raises(AnalysisMergeError) as exc:
        validate_narrative_response(replace(payload, scenes=(scene(),)), ALIASES, TEXTS, VISIBLE,
                                    ordered_block_ids=ORDER, is_notes_only=True)
    assert exc.value.violation.rule_code == 'notes_only_scenes'


def gender_fact(value='male'):
    anchor = QuotedSourceAnchorResponse('b1', 'Holmes')
    return QuotedContextFactResponse('f', 'gender', (anchor,), anchor, value, 'explicit')


def test_validate_narrative_resolves_facts_and_rejects_glossary_conflicts():
    payload = NarrativeAnalysisResponse('Keep voice.', 'Summary.', context_facts=(gender_fact(),), scenes=(scene(),))
    scenes, facts = validate_narrative_response(payload, ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER)
    assert len(scenes) == 1
    assert facts[0].subject_anchor.block_id == 'id-1'
    assert facts[0].id == facts[0].canonical_id
    with pytest.raises(AnalysisMergeError) as exc:
        validate_narrative_response(payload, ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
                                    mandatory_facts={'Holmes': {'gender': 'female'}})
    assert exc.value.violation.rule_code == 'glossary_fact_conflict'


def test_validate_narrative_rejects_unresolved_context_quote():
    payload = NarrativeAnalysisResponse('Keep voice.', 'Summary.', context_facts=(gender_fact(),), scenes=(scene(),))
    with pytest.raises(AnalysisMergeError) as exc:
        validate_narrative_response(payload, ALIASES, TEXTS, {**VISIBLE, 'b1': 'Someone'}, ordered_block_ids=ORDER)
    assert exc.value.violation.rule_code == 'unresolved_quote'


@pytest.mark.parametrize('continues,expected', [(True, 1), (False, 2)])
def test_merge_boundary_scene_requires_explicit_continuation_and_unions_participants(continues, expected):
    raw1 = scene('b1', 'b2', location='Baker Street', participants=('Holmes', 'Watson'))
    raw2 = scene('b3', 'b3', location='Baker Street', participants=('Lestrade', 'Watson'), continues_previous_scene=continues)
    one = NarrativeAnalysisResponse('Keep voice.', 'First.', scenes=(raw1,))
    two = NarrativeAnalysisResponse('Keep voice.\n\nUse formal speech.', 'Full chapter.', scenes=(raw2,))
    narrative, merged, facts = merge_narrative_fragments([(one, resolve((raw1,)), ()), (two, resolve((raw2,)), ())])
    assert len(merged) == expected
    assert narrative.chapter_summary == 'Full chapter.'
    assert narrative.translation_instructions.count('Keep voice.') == 1
    assert facts == ()
    assert tuple(s.order for s in merged) == tuple(range(expected))
    assert all(s.id == s.canonical_id for s in merged)
    if continues:
        assert (merged[0].start_block_id, merged[0].end_block_id) == ('id-1', 'id-3')
        assert merged[0].participants == ('Holmes', 'Watson', 'Lestrade')


def test_staged_response_assembly_and_resolution_preserves_scenes_and_facts():
    raw = scene(visual_facts=(QuotedSceneVisualFactResponse(QuotedSourceAnchorResponse('b1', 'red coat'), 'Holmes', 'Red coat'),))
    n = NarrativeAnalysisResponse('Keep voice.', 'Full narrative.', context_facts=(gender_fact(),), scenes=(raw,))
    scenes, facts = validate_narrative_response(n, ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER)
    n, scenes, facts = merge_narrative_fragments([(n, scenes, facts)])
    p = PedagogicalAnalysisResponse(treatment_plan=(TreatmentPlanItemResponse('b1', BlockTreatment.TRANSLATE),),
                                     reader_chapter_summary='Simple summary.')
    combined = combine_staged_analysis_responses(n, p, resolved_scenes=scenes)
    assert combined.chapter_summary == 'Simple summary.'
    assert combined.treatment_plan == p.treatment_plan
    plan = content_plan_from_response(combined, {**ALIASES, **{b: b for b in ORDER}}, TEXTS,
                                      visible_texts_by_alias={**VISIBLE, **TEXTS})
    assert plan.scenes == scenes
    assert plan.context_facts == facts
    assert combine_staged_analysis_responses(n, PedagogicalAnalysisResponse()).chapter_summary == 'Full narrative.'


def test_merge_fragments_normalizes_repeated_local_aliases_before_assembly():
    fragments = []
    for block, name in [('id-a', 'Holmes'), ('id-b', 'Watson')]:
        anchor = QuotedSourceAnchorResponse('b1', name)
        payload = NarrativeAnalysisResponse('Keep voice.', name,
            context_facts=(QuotedContextFactResponse('f', 'gender', (anchor,), anchor, 'male', 'explicit'),),
            scenes=(scene('b1', 'b1', participants=(name,)),))
        resolved, facts = validate_narrative_response(payload, {'b1': block}, {block: name}, {'b1': name},
                                                     ordered_block_ids=(block,))
        fragments.append((payload, resolved, facts))
    narrative, scenes, facts = merge_narrative_fragments(fragments)
    combined = combine_staged_analysis_responses(narrative, PedagogicalAnalysisResponse(), resolved_scenes=scenes)
    texts = {'id-a': 'Holmes', 'id-b': 'Watson'}
    plan = content_plan_from_response(combined, {b: b for b in texts}, texts, visible_texts_by_alias=texts)
    assert {f.subject_anchor.block_id for f in plan.context_facts} == {'id-a', 'id-b'}
    assert [(s.start_block_id, s.end_block_id) for s in plan.scenes] == [('id-a', 'id-a'), ('id-b', 'id-b')]
    assert len(plan.context_facts) == 2


@pytest.mark.parametrize('source,visible_text,quote', [
    ('Holmes wore a red coat.', 'RED COAT', 'RED COAT'),
    ('RED COAT', 'Holmes wore a red coat.', 'RED COAT'),
    ('He is tall.', "He's tall.", "He's tall."),
    ('Holmes wore a red coat.', '!red coat!', '!red coat!'),
    ('Holmes wore a red coat.', 'Holmes wore a red coat.', 'red! coat'),
    ('Holmes wore a red coat.', 'Holmes wore a red coat.', 'blue coat'),
])
def test_visual_facts_reject_unaligned_views_and_non_typographic_mismatch(source, visible_text, quote):
    raw = scene('b1', 'b1', visual_facts=(QuotedSceneVisualFactResponse(
        QuotedSourceAnchorResponse('b1', quote), 'Visitor', 'A detail'),))
    with pytest.raises(AnalysisMergeError) as exc:
        resolve_scenes(
            (raw,),
            {'b1': 'id-1'},
            {'id-1': source},
            {'b1': visible_text},
            ordered_block_ids=('id-1',),
        )
    assert exc.value.violation is not None
    assert exc.value.violation.rule_code == 'unresolved_quote'


def test_context_facts_keep_legacy_matching_rules():
    anchor = QuotedSourceAnchorResponse('b1', 'HOLMES')
    fact = QuotedContextFactResponse('fact', 'gender', (anchor,), anchor, 'male', 'explicit')
    payload = NarrativeAnalysisResponse('Keep voice.', 'Summary.', context_facts=(fact,), scenes=(scene(),))
    _, facts = validate_narrative_response(payload, ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER)
    assert facts[0].subject_anchor.quote == 'Holmes'


def test_merge_boundary_scene_implicitly_joins_matching_location_and_participant_set():
    first_fact = QuotedSceneVisualFactResponse(QuotedSourceAnchorResponse('b1', 'red coat'), 'Holmes', 'Red coat')
    second_fact = QuotedSceneVisualFactResponse(QuotedSourceAnchorResponse('b3', 'Lestrade'), 'Lestrade', 'At the door')
    raw1 = scene('b1', 'b2', location='Baker Street', participants=('Holmes', 'Watson'),
                 visual_facts=(first_fact,), visual_assumptions=('Warm light',))
    raw2 = scene('b3', 'b3', location='Baker Street', participants=('Watson', 'Holmes'),
                 visual_facts=(second_fact,), visual_assumptions=('Warm light', 'Dark room'))
    raw2 = replace(raw2, summary='The visitor brings news.')
    n1 = NarrativeAnalysisResponse('Keep voice.', 'First.', scenes=(raw1,))
    n2 = NarrativeAnalysisResponse('Keep voice.', 'Full.', scenes=(raw2,))
    _, merged, _ = merge_narrative_fragments([(n1, resolve((raw1,)), ()), (n2, resolve((raw2,)), ())])
    assert len(merged) == 1
    assert (merged[0].start_block_id, merged[0].end_block_id, merged[0].order) == ('id-1', 'id-3', 0)
    assert merged[0].id == merged[0].canonical_id
    assert merged[0].participants == ('Holmes', 'Watson')
    assert len(merged[0].visual_facts) == 2
    assert merged[0].visual_assumptions == ('Warm light', 'Dark room')
    assert merged[0].summary == 'At Baker Street. The visitor brings news.'


@pytest.mark.parametrize('location,participants', [
    ('Bank', ('Holmes', 'Watson')),
    ('Baker Street', ('Holmes',)),
    (None, ('Holmes', 'Watson')),
    ('', ('Holmes', 'Watson')),
    ('Baker Street', ()),
])
def test_implicit_boundary_continuation_needs_known_matching_setting(location, participants):
    first = scene('b1', 'b2', location='Baker Street', participants=('Holmes', 'Watson'))
    second = scene('b3', 'b3', location=location, participants=participants)
    fragments = [(NarrativeAnalysisResponse('Keep voice.', 'First.', scenes=(first,)), resolve((first,)), ()),
                 (NarrativeAnalysisResponse('Keep voice.', 'Full.', scenes=(second,)), resolve((second,)), ())]
    assert len(merge_narrative_fragments(fragments)[1]) == 2


def test_implicit_boundary_continuation_does_not_skip_an_empty_fragment():
    first = scene('b1', 'b1', location='Baker Street', participants=('Holmes',))
    last = scene('b3', 'b3', location='Baker Street', participants=('Holmes',))
    fragments = [(NarrativeAnalysisResponse('Keep voice.', 'First.', scenes=(first,)), resolve((first,)), ()),
                 (NarrativeAnalysisResponse('Keep voice.', 'Middle.'), (), ()),
                 (NarrativeAnalysisResponse('Keep voice.', 'Full.', scenes=(last,)), resolve((last,)), ())]
    assert len(merge_narrative_fragments(fragments)[1]) == 2


@pytest.mark.parametrize('location,participants', [(None, ('Holmes',)), ('', ('Holmes',)),
                                                ('   ', ('Holmes',)), ('Baker Street', ())])
def test_implicit_boundary_continuation_does_not_guess_unknown_matching_setting(location, participants):
    first = scene('b1', 'b2', location=location, participants=participants)
    second = scene('b3', 'b3', location=location, participants=participants)
    fragments = [(NarrativeAnalysisResponse('Keep voice.', 'First.', scenes=(first,)), resolve((first,)), ()),
                 (NarrativeAnalysisResponse('Keep voice.', 'Full.', scenes=(second,)), resolve((second,)), ())]
    assert len(merge_narrative_fragments(fragments)[1]) == 2


def test_visual_facts_accept_typography_and_dash_spaces():
    text = (
        "Holmes wore a red coat — one’s only defense — in dead-spots. "
        "He is tall–very tall - and calm."
    )
    aliases = {'b1': 'id-1'}
    texts = {'id-1': text}
    visible = {'b1': text}
    raw = scene(
        'b1',
        'b1',
        visual_facts=(
            QuotedSceneVisualFactResponse(QuotedSourceAnchorResponse('b1', 'RED COAT'), 'Holmes', 'Red coat'),
            QuotedSceneVisualFactResponse(QuotedSourceAnchorResponse('b1', "one's only defense"), 'Holmes', 'Defense'),
            QuotedSceneVisualFactResponse(QuotedSourceAnchorResponse('b1', "He's tall"), 'Holmes', 'Height'),
            QuotedSceneVisualFactResponse(QuotedSourceAnchorResponse('b1', 'red coat—one\'s only defense—in dead-spots'), 'Holmes', 'Dash tolerance'),
            QuotedSceneVisualFactResponse(QuotedSourceAnchorResponse('b1', 'tall - very tall—and calm'), 'Holmes', 'En-dash and spaced hyphen'),
        ),
    )
    resolved, = resolve_scenes((raw,), aliases, texts, visible, ordered_block_ids=('id-1',))
    assert [fact.anchor.quote for fact in resolved.visual_facts] == [
        'red coat',
        'one’s only defense',
        'He is tall',
        'red coat — one’s only defense — in dead-spots',
        'tall–very tall - and calm',
    ]

    # Intra-word hyphen in quote must not match spaced dash in source text.
    spaced_source = 'Beware of dead - spots on the floor.'
    bad = scene(
        'b1',
        'b1',
        visual_facts=(QuotedSceneVisualFactResponse(QuotedSourceAnchorResponse('b1', 'dead-spots'), 'Floor', 'Danger'),),
    )
    with pytest.raises(AnalysisMergeError) as exc:
        resolve_scenes((bad,), {'b1': 'id-1'}, {'id-1': spaced_source}, {'b1': spaced_source}, ordered_block_ids=('id-1',))
    assert exc.value.violation is not None
    assert exc.value.violation.rule_code == 'unresolved_quote'


def test_visual_fact_error_reports_fact_block_and_quote_including_unknown_alias():
    local_raw = scene(
        'b1',
        'b3',
        visual_facts=(
            QuotedSceneVisualFactResponse(QuotedSourceAnchorResponse('b2', 'blue door'), 'Door', 'Wrong color'),
        ),
    )
    with pytest.raises(AnalysisMergeError) as local_exc:
        resolve((local_raw,))
    assert local_exc.value.violation is not None
    assert local_exc.value.violation.rule_code == 'unresolved_quote'
    assert local_exc.value.violation.stage == 'chapter_analysis'
    assert local_exc.value.violation.scope == 'local'
    assert local_exc.value.violation.block_ids == ('id-2',)
    assert 'b2' in str(local_exc.value)
    assert 'blue door' in str(local_exc.value)
    assert 'id-2' not in str(local_exc.value)

    unknown_raw = scene(
        'b1',
        'b3',
        visual_facts=(
            QuotedSceneVisualFactResponse(QuotedSourceAnchorResponse('b9', 'Holmes'), 'Visitor', 'Unknown alias'),
        ),
    )
    with pytest.raises(AnalysisMergeError) as unknown_exc:
        resolve((unknown_raw,))
    assert unknown_exc.value.violation is not None
    assert unknown_exc.value.violation.rule_code == 'unresolved_quote'
    assert unknown_exc.value.violation.stage == 'chapter_analysis'
    assert unknown_exc.value.violation.scope == 'global'
    assert unknown_exc.value.violation.block_ids == ()
    assert 'b9' in str(unknown_exc.value)
    assert 'Holmes' in str(unknown_exc.value)


def test_validate_narrative_collects_all_scene_quote_and_conflict_errors_in_one_pass():
    aliases = {'b1': 'id-1', 'b2': 'id-2', 'b3': 'id-3', 'b4': 'id-4', 'b5': 'id-5'}
    texts = {
        'id-1': 'Holmes wore a red coat.',
        'id-2': 'Watson opened the door.',
        'id-3': 'Lestrade checked the hall.',
        'id-4': 'Hudson brought tea.',
        'id-5': 'Moriarty watched outside.',
    }
    visible = {alias: texts[block_id] for alias, block_id in aliases.items()}
    order = ('id-1', 'id-2', 'id-3', 'id-4', 'id-5')

    holmes_anchor = QuotedSourceAnchorResponse('b1', 'Holmes')
    hudson_anchor = QuotedSourceAnchorResponse('b4', 'Hudson')
    lestrade_anchor = QuotedSourceAnchorResponse('b3', 'Lestrade')
    payload = NarrativeAnalysisResponse(
        'Keep voice.',
        'Summary.',
        context_facts=(
            QuotedContextFactResponse(
                'f-bad-1', 'speaker',
                (QuotedSourceAnchorResponse('b2', 'missing context quote 1'),),
                QuotedSourceAnchorResponse('b2', 'missing context quote 1'),
                'Watson', 'explicit',
            ),
            QuotedContextFactResponse(
                'f-bad-2', 'speaker',
                (QuotedSourceAnchorResponse('b4', 'missing context quote 2'),),
                QuotedSourceAnchorResponse('b4', 'missing context quote 2'),
                'Hudson', 'explicit',
            ),
            QuotedContextFactResponse('f-conflict-1', 'gender', (holmes_anchor,), holmes_anchor, 'male', 'explicit'),
            QuotedContextFactResponse('f-conflict-2', 'gender', (holmes_anchor,), holmes_anchor, 'female', 'explicit'),
            QuotedContextFactResponse('f-glossary-1', 'gender', (hudson_anchor,), hudson_anchor, 'male', 'explicit'),
            QuotedContextFactResponse('f-glossary-2', 'gender', (lestrade_anchor,), lestrade_anchor, 'female', 'explicit'),
        ),
        scenes=(
            scene(
                'b1',
                'b2',
                visual_facts=(
                    QuotedSceneVisualFactResponse(QuotedSourceAnchorResponse('b1', 'missing visual quote'), 'Holmes', 'Coat'),
                    QuotedSceneVisualFactResponse(QuotedSourceAnchorResponse('b5', 'Moriarty'), 'Moriarty', 'Outside'),
                ),
            ),
            scene(
                'b4',
                'b5',
                visual_facts=(
                    QuotedSceneVisualFactResponse(QuotedSourceAnchorResponse('b9', 'ghost'), 'Ghost', 'Missing alias'),
                ),
            ),
        ),
    )

    with pytest.raises(AnalysisMergeError) as exc:
        validate_narrative_response(
            payload,
            aliases,
            texts,
            visible,
            ordered_block_ids=order,
            mandatory_facts={'Hudson': {'gender': 'female'}, 'Lestrade': {'gender': 'male'}},
        )
    message = str(exc.value)
    assert exc.value.violation is not None
    assert exc.value.violation.rule_code == 'scene_bounds'
    assert exc.value.violation.scope == 'local'
    assert set(exc.value.violation.block_ids) == {'id-1', 'id-2', 'id-3', 'id-4', 'id-5'}
    for expected in (
        'b3',
        'missing visual quote',
        'b1',
        'Moriarty',
        'b5',
        'ghost',
        'b9',
        'missing context quote 1',
        'b2',
        'missing context quote 2',
        'b4',
        'конфликт атрибуций одного субъекта',
        'Holmes',
        'контекстная атрибуция противоречит обязательному решению глоссария',
        'b4: «Hudson»',
        'b3: «Lestrade»',
        'Hudson',
        'Lestrade',
    ):
        assert expected in message

    # Verify strict rule_code priority order across subsets:
    # notes_only_scenes > scene_bounds > unresolved_quote > scene_visual_anchor_out_of_bounds > context_fact_conflict > glossary_fact_conflict
    full_scenes = (scene('b1', 'b5'),)
    with pytest.raises(AnalysisMergeError) as notes_exc:
        validate_narrative_response(
            NarrativeAnalysisResponse(
                'Keep voice.', 'Summary.',
                context_facts=(payload.context_facts[0],),
                scenes=full_scenes,
            ),
            aliases, texts, visible, ordered_block_ids=order, is_notes_only=True,
        )
    assert notes_exc.value.violation.rule_code == 'notes_only_scenes'
    assert 'missing context quote 1' in str(notes_exc.value)

    with pytest.raises(AnalysisMergeError) as quote_exc:
        validate_narrative_response(
            NarrativeAnalysisResponse(
                'Keep voice.', 'Summary.',
                context_facts=(payload.context_facts[0], payload.context_facts[2], payload.context_facts[3], payload.context_facts[4]),
                scenes=(scene('b1', 'b2', visual_facts=(QuotedSceneVisualFactResponse(QuotedSourceAnchorResponse('b5', 'Moriarty'), 'Moriarty', 'Outside'),)), scene('b3', 'b5')),
            ),
            aliases, texts, visible, ordered_block_ids=order,
            mandatory_facts={'Hudson': {'gender': 'female'}},
        )
    assert quote_exc.value.violation.rule_code == 'unresolved_quote'
    assert 'missing context quote 1' in str(quote_exc.value)
    assert 'Moriarty' in str(quote_exc.value)

    with pytest.raises(AnalysisMergeError) as oob_exc:
        validate_narrative_response(
            NarrativeAnalysisResponse(
                'Keep voice.', 'Summary.',
                context_facts=(payload.context_facts[2], payload.context_facts[3], payload.context_facts[4]),
                scenes=(scene('b1', 'b2', visual_facts=(QuotedSceneVisualFactResponse(QuotedSourceAnchorResponse('b5', 'Moriarty'), 'Moriarty', 'Outside'),)), scene('b3', 'b5')),
            ),
            aliases, texts, visible, ordered_block_ids=order,
            mandatory_facts={'Hudson': {'gender': 'female'}},
        )
    assert oob_exc.value.violation.rule_code == 'scene_visual_anchor_out_of_bounds'
    assert 'конфликт атрибуций одного субъекта' in str(oob_exc.value)
    assert 'Hudson' in str(oob_exc.value)

    with pytest.raises(AnalysisMergeError) as conflict_exc:
        validate_narrative_response(
            NarrativeAnalysisResponse(
                'Keep voice.', 'Summary.',
                context_facts=(payload.context_facts[2], payload.context_facts[3], payload.context_facts[4]),
                scenes=full_scenes,
            ),
            aliases, texts, visible, ordered_block_ids=order,
            mandatory_facts={'Hudson': {'gender': 'female'}},
        )
    assert conflict_exc.value.violation.rule_code == 'context_fact_conflict'
    assert conflict_exc.value.violation.block_ids == ('id-1', 'id-4')
    assert 'Holmes' in str(conflict_exc.value)
    assert 'Hudson' in str(conflict_exc.value)

    # Inverted and overlapping scenes report exact blocks.
    with pytest.raises(AnalysisMergeError) as bounds_exc:
        validate_narrative_response(
            NarrativeAnalysisResponse(
                'Keep voice.', 'Summary.',
                scenes=(scene('b1', 'b2'), scene('b2', 'b3'), scene('b5', 'b4')),
            ),
            aliases, texts, visible, ordered_block_ids=order,
        )
    assert bounds_exc.value.violation.rule_code == 'scene_bounds'
    assert 'b2' in str(bounds_exc.value)
    assert 'b5' in str(bounds_exc.value)
    assert 'b4' in str(bounds_exc.value)


def _doc_block(
    block_id: str,
    order: int,
    *,
    block_type: BlockType = BlockType.PARAGRAPH,
    support_status: SupportStatus = SupportStatus.SUPPORTED,
    text: str = 'Text.',
) -> DocumentBlock:
    return DocumentBlock(
        id=block_id,
        section_id='section-1',
        type=block_type,
        order=order,
        text=text,
        source_reference=SourceReference('ch1.xhtml', f'/body/p[{order + 1}]'),
        support_status=support_status,
    )


def test_merge_narrative_fragments_preserves_location_change_without_chain_duplication():
    def frag(block_id: str, location: str | None, *, continues: bool = False):
        raw = scene('b1', 'b1', location=location, participants=('Nick', 'Allie'), continues_previous_scene=continues)
        resolved = resolve_scenes(
            (raw,),
            {'b1': block_id},
            {block_id: 'Scene text.'},
            {'b1': 'Scene text.'},
            ordered_block_ids=(block_id,),
        )
        return NarrativeAnalysisResponse('Keep voice.', 'Summary.', scenes=(raw,)), resolved, ()

    # 3+ fragment chain: Forest -> Road + Road -> Forest -> Road (never Forest -> Road -> Road) -> Inn
    _, merged_chain, _ = merge_narrative_fragments([
        frag('id-1', 'Forest'),
        frag('id-2', 'Road', continues=True),
        frag('id-3', '  road  ', continues=True),
        frag('id-4', 'Inn', continues=True),
    ])
    assert len(merged_chain) == 1
    assert merged_chain[0].location == 'Forest → Road → Inn'

    # Empty / whitespace-only locations are ignored on both sides.
    _, blank_prev, _ = merge_narrative_fragments([
        frag('id-1', '   '),
        frag('id-2', '  Road  ', continues=True),
    ])
    assert len(blank_prev) == 1
    assert blank_prev[0].location == 'Road'

    _, blank_next, _ = merge_narrative_fragments([
        frag('id-1', '  Forest  '),
        frag('id-2', '', continues=True),
    ])
    assert len(blank_next) == 1
    assert blank_next[0].location == 'Forest'

    _, both_blank, _ = merge_narrative_fragments([
        frag('id-1', '   '),
        frag('id-2', None, continues=True),
    ])
    assert len(both_blank) == 1
    assert both_blank[0].location is None


@pytest.mark.parametrize('sep_type,sep_status', [
    (BlockType.OTHER, SupportStatus.PRESERVE_OPAQUE),
    (BlockType.PAGE_BREAK, SupportStatus.SUPPORTED),
    (BlockType.IMAGE, SupportStatus.SUPPORTED),
])
def test_merge_narrative_fragments_respects_scene_separator_blocks(sep_type, sep_status):
    b1 = _doc_block('id-1', 0, text='Holmes entered.')
    sep = _doc_block('id-sep', 1, block_type=sep_type, support_status=sep_status, text='')
    b2 = _doc_block('id-2', 2, text='Watson waited.')
    section_blocks = (b1, sep, b2)

    raw1 = scene('b1', 'b1', location='Baker Street', participants=('Holmes', 'Watson'))
    raw2_explicit = scene('b1', 'b1', location='Baker Street', participants=('Holmes', 'Watson'), continues_previous_scene=True)
    raw2_implicit = scene('b1', 'b1', location='Baker Street', participants=('Watson', 'Holmes'), continues_previous_scene=False)

    res1 = resolve_scenes((raw1,), {'b1': 'id-1'}, {'id-1': b1.text}, {'b1': b1.text}, ordered_block_ids=('id-1',))
    res2 = resolve_scenes((raw2_explicit,), {'b1': 'id-2'}, {'id-2': b2.text}, {'b1': b2.text}, ordered_block_ids=('id-2',))
    res2_implicit = resolve_scenes((raw2_implicit,), {'b1': 'id-2'}, {'id-2': b2.text}, {'b1': b2.text}, ordered_block_ids=('id-2',))

    # Explicit continuation AND implicit continuation must both be blocked by a separator strictly between boundary blocks.
    _, explicit_scenes, _ = merge_narrative_fragments(
        [(NarrativeAnalysisResponse('Keep.', 'S1.', scenes=(raw1,)), res1, ()),
         (NarrativeAnalysisResponse('Keep.', 'S2.', scenes=(raw2_explicit,)), res2, ())],
        section_blocks=section_blocks,
    )
    assert len(explicit_scenes) == 2

    _, implicit_scenes, _ = merge_narrative_fragments(
        [(NarrativeAnalysisResponse('Keep.', 'S1.', scenes=(raw1,)), res1, ()),
         (NarrativeAnalysisResponse('Keep.', 'S2.', scenes=(raw2_implicit,)), res2_implicit, ())],
        section_blocks=section_blocks,
    )
    assert len(implicit_scenes) == 2

    # Without a separator between boundary blocks (normal supported paragraph or section_blocks omitted), merging still works.
    normal_mid = _doc_block('id-mid', 1, block_type=BlockType.PARAGRAPH, support_status=SupportStatus.SUPPORTED, text='Mid.')
    _, normal_merged, _ = merge_narrative_fragments(
        [(NarrativeAnalysisResponse('Keep.', 'S1.', scenes=(raw1,)), res1, ()),
         (NarrativeAnalysisResponse('Keep.', 'S2.', scenes=(raw2_explicit,)), res2, ())],
        section_blocks=(b1, normal_mid, b2),
    )
    assert len(normal_merged) == 1

    _, compat_merged, _ = merge_narrative_fragments(
        [(NarrativeAnalysisResponse('Keep.', 'S1.', scenes=(raw1,)), res1, ()),
         (NarrativeAnalysisResponse('Keep.', 'S2.', scenes=(raw2_explicit,)), res2, ())]
    )
    assert len(compat_merged) == 1


def test_validate_narrative_rejects_wrong_summary_and_allows_latin_names_in_russian_scenes():
    from types import SimpleNamespace

    def stub_detector(scores):
        def probability_for_language(text: str, language_code: str) -> float | None:
            return scores(text, language_code)

        return SimpleNamespace(probability_for_language=probability_for_language)

    # Backward compatibility: without language params, English scene summaries still pass.
    legacy = NarrativeAnalysisResponse('Keep voice.', 'Holmes investigates.', scenes=(scene('b1', 'b3'),))
    assert len(validate_narrative_response(legacy, ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER)[0]) == 1

    # 1. Valid English chapter_summary + Russian scene summaries with borrowed Latin names (short & long).
    ru_scene_short = QuotedSceneResponse('b1', 'b2', 'Holmes и Watson в комнате.', participants=('Holmes', 'Watson'))
    ru_scene_long = QuotedSceneResponse(
        'b3', 'b3',
        'Инспектор Lestrade приходит на Baker Street и разговаривает с Holmes.',
        participants=('Lestrade',),
    )
    ok_payload = NarrativeAnalysisResponse(
        'Сохранять голос автора.',
        'Holmes wears a red coat while Watson opens the door for Lestrade.',
        scenes=(ru_scene_short, ru_scene_long),
    )
    detector_ok = stub_detector(
        lambda text, lang: 0.95 if (lang == 'en' and 'Holmes wears' in text) or (lang == 'ru' and 'приходит' in text) else 0.01
    )
    scenes, _ = validate_narrative_response(
        ok_payload,
        ALIASES,
        TEXTS,
        VISIBLE,
        ordered_block_ids=ORDER,
        source_language='en',
        target_language='ru',
        language_detector=detector_ok,
    )
    assert len(scenes) == 2

    # 1b. Russian scene summary dominated by capitalized Latin names still passes even if detector is unsure on full text.
    latin_heavy_scene = QuotedSceneResponse(
        'b1', 'b3',
        'Sherlock Holmes и John Watson встречают Inspector Lestrade на Baker Street.',
        participants=('Holmes', 'Watson', 'Lestrade'),
    )
    latin_heavy_detector = stub_detector(
        lambda text, lang: (
            0.95 if lang == 'en' and 'Holmes wears' in text
            else (0.95 if lang == 'ru' and 'Sherlock' not in text else 0.01)
        )
    )
    assert len(validate_narrative_response(
        replace(ok_payload, scenes=(latin_heavy_scene,)),
        ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
        source_language='en', target_language='ru', language_detector=latin_heavy_detector,
    )[0]) == 1

    # 2. Russian chapter_summary rejected when source_language='en'.
    with pytest.raises(AnalysisMergeError) as cyr_summary_exc:
        validate_narrative_response(
            replace(ok_payload, chapter_summary='Холмс в красном пальто ждёт Лестрейда.'),
            ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
            source_language='en', target_language='ru',
        )
    assert cyr_summary_exc.value.violation.rule_code == 'narrative_summary_language'

    # 3. English chapter_summary containing stray Cyrillic rejected for Latin source.
    with pytest.raises(AnalysisMergeError) as mixed_summary_exc:
        validate_narrative_response(
            replace(ok_payload, chapter_summary='Holmes wears a red coat в Лондоне.'),
            ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
            source_language='en', target_language='ru',
        )
    assert mixed_summary_exc.value.violation.rule_code == 'narrative_summary_language'

    # 4. Wrong Latin language in chapter_summary rejected via LanguageDetector.
    spanish_detector = stub_detector(lambda _text, lang: 0.01 if lang == 'en' else 0.95)
    with pytest.raises(AnalysisMergeError) as es_summary_exc:
        validate_narrative_response(
            replace(ok_payload, chapter_summary='La historia comienza en Londres con Holmes y Watson.'),
            ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
            source_language='en', target_language='ru', language_detector=spanish_detector,
        )
    assert es_summary_exc.value.violation.rule_code == 'narrative_summary_language'

    # 5. Pure English scenes[].summary rejected when target_language='ru'.
    with pytest.raises(AnalysisMergeError) as en_scene_exc:
        validate_narrative_response(
            replace(ok_payload, scenes=(scene('b1', 'b3'),)),
            ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
            source_language='en', target_language='ru',
        )
    assert en_scene_exc.value.violation.rule_code == 'scene_summary_language'

    # 6. Long mixed-English scene summary with one stray Cyrillic word rejected via LanguageDetector.
    long_bad_scene = QuotedSceneResponse(
        'b1', 'b3',
        'Holmes and Watson walk through the dark street and wait for Lestrade (комната).',
        participants=('Holmes',),
    )
    detector_reject_ru = stub_detector(lambda _text, lang: 0.95 if lang == 'en' else 0.01)
    with pytest.raises(AnalysisMergeError) as long_scene_exc:
        validate_narrative_response(
            replace(ok_payload, scenes=(long_bad_scene,)),
            ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
            source_language='en', target_language='ru', language_detector=detector_reject_ru,
        )
    assert long_scene_exc.value.violation.rule_code == 'scene_summary_language'

    # 7. Priority: narrative_summary_language outranks scene_summary_language.
    with pytest.raises(AnalysisMergeError) as prio_exc:
        validate_narrative_response(
            NarrativeAnalysisResponse('Keep voice.', 'Русский пересказ.', scenes=(scene('b1', 'b3'),)),
            ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
            source_language='en', target_language='ru',
        )
    assert prio_exc.value.violation.rule_code == 'narrative_summary_language'


def test_validate_narrative_checks_spoilers_with_known_names_assessment_conflicts_and_visual_facts():
    from linguary.application.llm_dto import GlossaryCandidateResponse
    from linguary.domain import ConceptType
    from linguary.domain.name_translation import (
        NameTranslationAssessment,
        NameTranslationMethod,
        NameTranslationReviewReason,
    )

    # 1. Name translation assessment conflicts (all 4 rules).
    bad_assessments = (
        NameTranslationAssessment(
            NameTranslationMethod.SOUND, False,
            (NameTranslationReviewReason.CREATIVE_ADAPTATION,),
            'Транскрипция.', opaque_name=False,
        ),
        NameTranslationAssessment(
            NameTranslationMethod.SOUND, False,
            (NameTranslationReviewReason.PERSONAL_NAME_SUBSTITUTION,),
            'Транскрипция.', opaque_name=False,
        ),
        NameTranslationAssessment(
            NameTranslationMethod.SOUND, False,
            (NameTranslationReviewReason.WORDPLAY,),
            'Непрозрачное имя.', opaque_name=True,
        ),
        NameTranslationAssessment(
            NameTranslationMethod.SOUND, False,
            (NameTranslationReviewReason.ANAGRAM,),
            'Непрозрачное имя.', opaque_name=True,
        ),
        NameTranslationAssessment(
            NameTranslationMethod.MEANING, True,
            (), 'Перевод по смыслу.', opaque_name=True,
        ),
        NameTranslationAssessment(
            NameTranslationMethod.CREATIVE, True,
            (), 'Творческий вариант.', opaque_name=False,
        ),
        NameTranslationAssessment(
            NameTranslationMethod.UNCERTAIN, True,
            (), 'Неуверенный вариант.', opaque_name=False,
        ),
    )
    for bad_assessment in bad_assessments:
        candidate = GlossaryCandidateResponse(
            source_text='Everlost',
            type=ConceptType.PLACE,
            suggested_translation='Эверлост',
            confidence=0.9,
            model_flagged_review=False,
            name_translation_assessment=bad_assessment,
        )
        with pytest.raises(AnalysisMergeError) as assess_exc:
            validate_narrative_response(
                NarrativeAnalysisResponse(
                    'Keep voice.', 'Summary.',
                    glossary_candidates=(candidate,),
                    scenes=(scene('b1', 'b3'),),
                ),
                ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
            )
        assert assess_exc.value.violation.rule_code == 'name_assessment_conflict'

    valid_candidate = GlossaryCandidateResponse(
        source_text='Holmes',
        type=ConceptType.PERSON,
        suggested_translation='Холмс',
        confidence=0.95,
        model_flagged_review=False,
        name_translation_assessment=NameTranslationAssessment(
            NameTranslationMethod.SOUND, True, (), 'Непрозрачная фамилия.', opaque_name=True,
        ),
    )

    # 2. Anti-spoilers: scenes[].participants vs known_names & stem/'s matching (Harry Potter <-> Harry's).
    sp_aliases = {'b1': 'id-1', 'b2': 'id-2', 'b3': 'id-3'}
    sp_texts = {
        'id-1': '‘Stay back!’ he shouted in the dark.',
        'id-2': 'The boy opened the door.',
        'id-3': "Harry's owl landed beside Lestrade.",
    }
    sp_visible = {alias: sp_texts[bid] for alias, bid in sp_aliases.items()}
    sp_order = ('id-1', 'id-2', 'id-3')

    spoiler_scenes = (
        QuotedSceneResponse('b1', 'b2', 'Разговор в темноте.', participants=('Harry Potter',)),
        QuotedSceneResponse('b3', 'b3', 'Появляется сова.', participants=('Harry Potter', 'Lestrade')),
    )
    with pytest.raises(AnalysisMergeError) as spoiler_exc:
        validate_narrative_response(
            NarrativeAnalysisResponse('Keep voice.', 'Summary.', scenes=spoiler_scenes),
            sp_aliases, sp_texts, sp_visible, ordered_block_ids=sp_order,
        )
    assert spoiler_exc.value.violation.rule_code == 'spoiler_name_leak'

    # With known_names=('Harry',), participant 'Harry Potter' in scene 1 is allowed.
    resolved_scenes, _ = validate_narrative_response(
        NarrativeAnalysisResponse(
            'Keep voice.', 'Summary.',
            glossary_candidates=(valid_candidate,),
            scenes=spoiler_scenes,
        ),
        sp_aliases, sp_texts, sp_visible, ordered_block_ids=sp_order,
        known_names=('Harry',),
    )
    assert len(resolved_scenes) == 2

    # 3. context_facts anti-spoiler: original certainty=='explicit' before first mention fails,
    #    while original certainty=='inferred' passes and is NOT promoted to explicit before first mention
    #    (keeping validate_narrative_response idempotent across merge/cache revalidation).
    b1_quote_anchor = QuotedSourceAnchorResponse('b1', '‘Stay back!’')
    b1_remark_anchor = QuotedSourceAnchorResponse('b1', 'he shouted')
    explicit_spoiler_fact = QuotedContextFactResponse(
        'f-sp-explicit', 'speaker', (b1_remark_anchor,), b1_quote_anchor, 'Harry Potter', 'explicit',
    )
    with pytest.raises(AnalysisMergeError) as fact_spoiler_exc:
        validate_narrative_response(
            NarrativeAnalysisResponse(
                'Keep voice.', 'Summary.',
                context_facts=(explicit_spoiler_fact,),
                scenes=spoiler_scenes,
            ),
            sp_aliases, sp_texts, sp_visible, ordered_block_ids=sp_order,
            known_names=('Harry',),
        )
    assert fact_spoiler_exc.value.violation.rule_code == 'spoiler_name_leak'
    assert fact_spoiler_exc.value.violation.block_ids == ('id-1',)

    inferred_pronoun_fact = QuotedContextFactResponse(
        'f-sp-inferred', 'speaker', (b1_remark_anchor,), b1_quote_anchor, 'Harry Potter', 'inferred',
    )
    _, kept_facts = validate_narrative_response(
        NarrativeAnalysisResponse(
            'Keep voice.', 'Summary.',
            context_facts=(inferred_pronoun_fact,),
            scenes=spoiler_scenes,
            glossary_candidates=(
                GlossaryCandidateResponse(
                    source_text='Harry Potter',
                    type=ConceptType.PERSON,
                    suggested_translation='Гарри Поттер',
                    confidence=0.95,
                    model_flagged_review=False,
                    note='мужской род',
                ),
            ),
        ),
        sp_aliases, sp_texts, sp_visible, ordered_block_ids=sp_order,
        known_names=('Harry',),
    )
    # Anti-spoiler guard keeps inferred before first_pos so repeated validation stays idempotent.
    assert kept_facts[0].certainty == 'inferred'

    # 4. Visual facts: filter out obvious sound/smell/taste markers without failing the chunk,
    #    while preserving visual forms like беззвучно, музыкальные магазины, магазин сладостей.
    for bad_detail in (
        'Громкий звук шагов за дверью',
        'Резкий запах дыма в комнате',
        'Горький вкус лекарства',
        'Loud sound of footsteps',
        'Sweet smell of rain',
    ):
        bad_vf_scene = scene(
            'b1', 'b3',
            visual_facts=(QuotedSceneVisualFactResponse(
                QuotedSourceAnchorResponse('b1', 'red coat'), 'Holmes', bad_detail,
            ),),
        )
        filtered_scenes, _ = validate_narrative_response(
            NarrativeAnalysisResponse('Keep voice.', 'Summary.', scenes=(bad_vf_scene,)),
            ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
        )
        assert len(filtered_scenes) == 1
        assert filtered_scenes[0].visual_facts == ()
        assert filtered_scenes[0].id == filtered_scenes[0].canonical_id

    good_vf_scene = scene(
        'b1', 'b3',
        visual_facts=(
            QuotedSceneVisualFactResponse(
                QuotedSourceAnchorResponse('b1', 'red coat'), 'Holmes', 'Беззвучно шевелит губами в красном пальто',
            ),
            QuotedSceneVisualFactResponse(
                QuotedSourceAnchorResponse('b2', 'opened the door'), 'Watson', 'Бесшумно открывает дверь и гуляет с револьвером',
            ),
        ),
    )
    kept_scenes, _ = validate_narrative_response(
        NarrativeAnalysisResponse('Keep voice.', 'Summary.', scenes=(good_vf_scene,)),
        ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
    )
    assert len(kept_scenes) == 1
    assert len(kept_scenes[0].visual_facts) == 2

    # 5. Strict priority: name_assessment_conflict > spoiler_name_leak > context_fact_conflict.
    bad_vf_spoiler_scene = (
        QuotedSceneResponse(
            'b1', 'b2', 'Разговор.', participants=('Harry Potter',),
            visual_facts=(QuotedSceneVisualFactResponse(
                QuotedSourceAnchorResponse('b1', 'Stay back!'), 'Boy', 'Громкий звук крика',
            ),),
        ),
        QuotedSceneResponse('b3', 'b3', 'Сова.', participants=('Harry Potter',)),
    )
    with pytest.raises(AnalysisMergeError) as prio_assess_exc:
        validate_narrative_response(
            NarrativeAnalysisResponse(
                'Keep voice.', 'Summary.',
                glossary_candidates=(GlossaryCandidateResponse(
                    source_text='Everlost', type=ConceptType.PLACE,
                    suggested_translation='Эверлост', confidence=0.9,
                    model_flagged_review=False,
                    name_translation_assessment=bad_assessments[0],
                ),),
                scenes=bad_vf_spoiler_scene,
            ),
            sp_aliases, sp_texts, sp_visible, ordered_block_ids=sp_order,
        )
    assert prio_assess_exc.value.violation.rule_code == 'name_assessment_conflict'

    with pytest.raises(AnalysisMergeError) as prio_spoiler_exc:
        validate_narrative_response(
            NarrativeAnalysisResponse('Keep voice.', 'Summary.', scenes=bad_vf_spoiler_scene),
            sp_aliases, sp_texts, sp_visible, ordered_block_ids=sp_order,
        )
    assert prio_spoiler_exc.value.violation.rule_code == 'spoiler_name_leak'


def test_validate_narrative_visual_facts_allow_visual_objects_and_colors_while_filtering_sounds():
    visual_ok_scene = scene(
        'b1', 'b3',
        visual_facts=(
            QuotedSceneVisualFactResponse(
                QuotedSourceAnchorResponse('b1', 'red coat'),
                'Holmes',
                'Держит скрипку, накинут кислотно-зелёный шарф как цветовой акцент',
            ),
            QuotedSceneVisualFactResponse(
                QuotedSourceAnchorResponse('b2', 'opened the door'),
                'Watson',
                'Стоит у дверного звонка в безвкусном наряде',
            ),
            QuotedSceneVisualFactResponse(
                QuotedSourceAnchorResponse('b1', 'red coat'),
                'Holmes',
                'На улице видны книжные и музыкальные магазины и магазин сладостей',
            ),
            QuotedSceneVisualFactResponse(
                QuotedSourceAnchorResponse('b2', 'opened the door'),
                'Watson',
                'Держит кружку сладкого чая и плитку горького шоколада',
            ),
        ),
    )
    kept_scenes, _ = validate_narrative_response(
        NarrativeAnalysisResponse('Keep voice.', 'Summary.', scenes=(visual_ok_scene,)),
        ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
    )
    assert len(kept_scenes) == 1
    assert len(kept_scenes[0].visual_facts) == 4

    for bad_detail in (
        'Скрипит деревянная половица под ногами',
        'Кислый привкус во рту',
        'Звон колокольчика у двери',
        'Сладкий привкус сиропа',
        'Горький вкус лекарства',
    ):
        # Even an unresolved quote on a filtered non-visual fact must not abort the chunk.
        bad_scene = scene(
            'b1', 'b3',
            visual_facts=(
                QuotedSceneVisualFactResponse(
                    QuotedSourceAnchorResponse('b1', 'red coat'), 'Holmes', 'Красное пальто',
                ),
                QuotedSceneVisualFactResponse(
                    QuotedSourceAnchorResponse('b1', 'nonexistent quote'), 'Holmes', bad_detail,
                ),
            ),
        )
        filtered_scenes, _ = validate_narrative_response(
            NarrativeAnalysisResponse('Keep voice.', 'Summary.', scenes=(bad_scene,)),
            ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
        )
        assert [vf.detail for vf in filtered_scenes[0].visual_facts] == ['Красное пальто']
        assert filtered_scenes[0].id == filtered_scenes[0].canonical_id


def test_validate_narrative_does_not_false_flag_lowercase_or_article_role_participants():
    role_aliases = {'b1': 'id-1', 'b2': 'id-2'}
    role_texts = {
        'id-1': 'The boy opened the door while the giant waited.',
        'id-2': '‘Boy!’ shouted Uncle Vernon. ‘Giant!’',
    }
    role_visible = {alias: role_texts[bid] for alias, bid in role_aliases.items()}
    role_order = ('id-1', 'id-2')

    for role_participant in ('the boy', 'boy', 'The boy', 'Boy', 'the giant', 'Giant'):
        role_scenes = (
            QuotedSceneResponse('b1', 'b1', 'У двери.', participants=(role_participant,)),
            QuotedSceneResponse('b2', 'b2', 'Крик в коридоре.', participants=(role_participant, 'Uncle Vernon')),
        )
        resolved, _ = validate_narrative_response(
            NarrativeAnalysisResponse('Keep voice.', 'Summary.', scenes=role_scenes),
            role_aliases, role_texts, role_visible, ordered_block_ids=role_order,
        )
        assert len(resolved) == 2

    # Unintroduced role noun appearing only in a later block is still caught as a spoiler.
    late_role_texts = {
        'id-1': 'The corridor was silent and dark.',
        'id-2': '‘Boy!’ shouted Uncle Vernon.',
    }
    late_role_visible = {alias: late_role_texts[bid] for alias, bid in role_aliases.items()}
    late_role_scenes = (
        QuotedSceneResponse('b1', 'b1', 'Тихий коридор.', participants=('Boy',)),
        QuotedSceneResponse('b2', 'b2', 'Крик в коридоре.', participants=('Boy', 'Uncle Vernon')),
    )
    with pytest.raises(AnalysisMergeError) as late_role_exc:
        validate_narrative_response(
            NarrativeAnalysisResponse('Keep voice.', 'Summary.', scenes=late_role_scenes),
            role_aliases, late_role_texts, late_role_visible, ordered_block_ids=role_order,
        )
    assert late_role_exc.value.violation.rule_code == 'spoiler_name_leak'

    # Proper capitalized surname colliding with earlier lowercase word ('white' vs 'Mary White' or 'White') still catches spoiler.
    proper_texts = {
        'id-1': 'She wore a white dress in the hall.',
        'id-2': '‘I am Mary White,’ she said.',
    }
    proper_visible = {alias: proper_texts[bid] for alias, bid in role_aliases.items()}
    for proper_participant in ('Mary White', 'White'):
        proper_scenes = (
            QuotedSceneResponse('b1', 'b1', 'В зале.', participants=(proper_participant,)),
            QuotedSceneResponse('b2', 'b2', 'Знакомство.', participants=(proper_participant,)),
        )
        with pytest.raises(AnalysisMergeError) as proper_exc:
            validate_narrative_response(
                NarrativeAnalysisResponse('Keep voice.', 'Summary.', scenes=proper_scenes),
                role_aliases, proper_texts, proper_visible, ordered_block_ids=role_order,
            )
        assert proper_exc.value.violation.rule_code == 'spoiler_name_leak'


def test_validate_narrative_ignores_anonymous_and_descriptive_role_labels_in_spoiler_check():
    from linguary.application.llm_dto import GlossaryCandidateResponse
    from linguary.domain import ConceptType

    aliases = {'b1': 'id-1', 'b2': 'id-2'}
    texts = {
        'id-1': 'Mr Dursley walked straight into someone in the dark outside the door.',
        'id-2': (
            'Surely a sensible person, an old man, two men, cloaked people, '
            'the boy, a tabby cat, whisperers, her mother, the king, a father, '
            'and a passer-by can call Voldemort, Tom Riddle, Harry Potter, '
            'and Professor McGonagall by name.'
        ),
    }
    visible = {alias: texts[bid] for alias, bid in aliases.items()}
    order = ('id-1', 'id-2')
    glossary = (
        GlossaryCandidateResponse(
            source_text='The Boy Who Lived',
            type=ConceptType.FICTIONAL_CONCEPT,
            suggested_translation='Мальчик, который выжил',
            confidence=0.95,
            model_flagged_review=False,
        ),
        GlossaryCandidateResponse(
            source_text='Old Man Marley',
            type=ConceptType.PERSON,
            suggested_translation='Старик Марли',
            confidence=0.9,
            model_flagged_review=False,
        ),
        GlossaryCandidateResponse(
            source_text='Voldemort',
            type=ConceptType.PERSON,
            suggested_translation='Волдеморт',
            confidence=0.98,
            model_flagged_review=False,
        ),
        GlossaryCandidateResponse(
            source_text='Harry',
            type=ConceptType.PERSON,
            suggested_translation='Гарри',
            confidence=0.98,
            model_flagged_review=False,
        ),
    )

    for neutral_label in (
        'unnamed person',
        'Unnamed person',
        'Unnamed Person',
        'the unnamed person',
        'The Unnamed Person',
        'Unknown Man',
        'unnamed father',
        'cloaked people',
        'Cloaked people',
        'old man',
        'Old man',
        'Tiny old man',
        'Two men',
        'Tabby cat',
        'Passer-by',
        'the old king',
        'her mother',
        'the boy',
        'The boy',
        'person',
        'Person',
        'whisperers',
        'Мальчик',
        'неназванный человек',
    ):
        scenes = (
            QuotedSceneResponse('b1', 'b1', 'Столкновение у двери.', participants=('Mr Dursley', neutral_label)),
            QuotedSceneResponse('b2', 'b2', 'Разговор ночью.', participants=('Voldemort',)),
        )
        resolved, _ = validate_narrative_response(
            NarrativeAnalysisResponse('Keep voice.', 'Summary.', glossary_candidates=glossary, scenes=scenes),
            aliases, texts, visible, ordered_block_ids=order,
        )
        assert len(resolved) == 2

    # Explicit context_facts with neutral anonymous label also pass.
    neutral_fact = QuotedContextFactResponse(
        'f-neutral', 'pronoun_referent',
        (QuotedSourceAnchorResponse('b1', 'someone'),),
        QuotedSourceAnchorResponse('b1', 'someone'),
        'unnamed person', 'explicit',
    )
    validate_narrative_response(
        NarrativeAnalysisResponse(
            'Keep voice.', 'Summary.',
            glossary_candidates=glossary,
            context_facts=(neutral_fact,),
            scenes=(
                QuotedSceneResponse('b1', 'b1', 'Столкновение у двери.', participants=('Mr Dursley', 'unnamed person')),
                QuotedSceneResponse('b2', 'b2', 'Разговор ночью.', participants=('Voldemort',)),
            ),
        ),
        aliases, texts, visible, ordered_block_ids=order,
    )

    # Real spoiler names — capitalized, lowercase, prefixed, titled, or translated — are caught.
    for spoiler_label in (
        'Voldemort',
        'voldemort',
        'Unnamed Voldemort',
        'The Dark Voldemort',
        'the dark voldemort',
        'harry potter',
        'tom riddle',
        'professor mcgonagall',
        'Волдеморт',
        'волдеморт',
        'мальчик гарри',
    ):
        scenes = (
            QuotedSceneResponse('b1', 'b1', 'Столкновение у двери.', participants=('Mr Dursley', spoiler_label)),
            QuotedSceneResponse('b2', 'b2', 'Разговор ночью.', participants=('Voldemort',)),
        )
        with pytest.raises(AnalysisMergeError) as exc:
            validate_narrative_response(
                NarrativeAnalysisResponse('Keep voice.', 'Summary.', glossary_candidates=glossary, scenes=scenes),
                aliases, texts, visible, ordered_block_ids=order,
            )
        assert exc.value.violation.rule_code == 'spoiler_name_leak', spoiler_label

    # Lowercase multi-word and titled proper names without glossary_candidates are still caught.
    for no_glossary_spoiler in ('voldemort', 'tom riddle', 'harry potter', 'professor mcgonagall'):
        scenes_no_glossary = (
            QuotedSceneResponse('b1', 'b1', 'Столкновение у двери.', participants=('Mr Dursley', no_glossary_spoiler)),
            QuotedSceneResponse('b2', 'b2', 'Разговор ночью.', participants=('Voldemort',)),
        )
        with pytest.raises(AnalysisMergeError) as no_glossary_exc:
            validate_narrative_response(
                NarrativeAnalysisResponse('Keep voice.', 'Summary.', scenes=scenes_no_glossary),
                aliases, texts, visible, ordered_block_ids=order,
            )
        assert no_glossary_exc.value.violation.rule_code == 'spoiler_name_leak', no_glossary_spoiler

    # Explicit context_facts with lowercase spoiler name are also caught.
    spoiler_fact = QuotedContextFactResponse(
        'f-spoiler', 'pronoun_referent',
        (QuotedSourceAnchorResponse('b1', 'someone'),),
        QuotedSourceAnchorResponse('b1', 'someone'),
        'tom riddle', 'explicit',
    )
    with pytest.raises(AnalysisMergeError) as fact_exc:
        validate_narrative_response(
            NarrativeAnalysisResponse(
                'Keep voice.', 'Summary.',
                context_facts=(spoiler_fact,),
                scenes=(
                    QuotedSceneResponse('b1', 'b1', 'Столкновение у двери.', participants=('Mr Dursley',)),
                    QuotedSceneResponse('b2', 'b2', 'Разговор ночью.', participants=('Voldemort',)),
                ),
            ),
            aliases, texts, visible, ordered_block_ids=order,
        )
    assert fact_exc.value.violation.rule_code == 'spoiler_name_leak'


def test_validate_narrative_allows_short_part_divider_titles_with_real_langid_and_rejects_russian():
    from linguary.adapters.validation import Py3LangidDetector

    detector = Py3LangidDetector()
    for divider_title in (
        'PART TWO: Skinjackers',
        'Part Two: The Chocolate Ogre',
        'Afterlights',
    ):
        aliases = {'b1': 'part-b1'}
        texts = {'part-b1': divider_title}
        visible = {'b1': divider_title}
        payload = NarrativeAnalysisResponse(
            translation_instructions='Сохранять тон.',
            chapter_summary=divider_title,
            scenes=(),
        )
        # Accepted both as notes-only / part-divider and as a short chapter summary.
        validate_narrative_response(
            payload,
            aliases,
            texts,
            visible,
            ordered_block_ids=('part-b1',),
            source_language='en',
            target_language='ru',
            language_detector=detector,
            is_notes_only=True,
        )
        validate_narrative_response(
            replace(
                payload,
                scenes=(QuotedSceneResponse('b1', 'b1', 'Название части.', participants=()),),
            ),
            aliases,
            texts,
            visible,
            ordered_block_ids=('part-b1',),
            source_language='en',
            target_language='ru',
            language_detector=detector,
            is_notes_only=False,
        )

    # Russian or mixed-Cyrillic chapter_summary for an English divider is still rejected.
    aliases = {'b1': 'part-b1'}
    texts = {'part-b1': 'PART TWO: Skinjackers'}
    visible = {'b1': 'PART TWO: Skinjackers'}
    for bad_summary in ('Часть вторая: Скинджекеры', 'PART TWO: Послесветы'):
        for notes_only in (True, False):
            scenes = () if notes_only else (QuotedSceneResponse('b1', 'b1', 'Название части.'),)
            with pytest.raises(AnalysisMergeError) as exc_info:
                validate_narrative_response(
                    NarrativeAnalysisResponse('Keep tone.', bad_summary, scenes=scenes),
                    aliases,
                    texts,
                    visible,
                    ordered_block_ids=('part-b1',),
                    source_language='en',
                    target_language='ru',
                    language_detector=detector,
                    is_notes_only=notes_only,
                )
            assert exc_info.value.violation.rule_code == 'narrative_summary_language'


def test_validate_narrative_name_assessment_conflicts_policy_meaning_descriptive_and_7_reasons():
    from linguary.application.llm_dto import GlossaryCandidateResponse
    from linguary.domain import ConceptType
    from linguary.domain.name_translation import (
        NameTranslationAssessment,
        NameTranslationMethod,
        NameTranslationPolicy,
        NameTranslationReviewReason,
        NameTranslationStrategy,
    )

    all_seven = (
        'ambiguous, wordplay, anagram, uncertain_etymology, '
        'creative_adaptation, personal_name_substitution, other'
    )
    meaning_policy = NameTranslationPolicy(1, NameTranslationStrategy.MEANING)
    sound_policy = NameTranslationPolicy(1, NameTranslationStrategy.SOUND)

    # 1. Existing conflict message must include all 7 allowed review_reasons.
    creative_conflict = GlossaryCandidateResponse(
        source_text='Everlost',
        type=ConceptType.PLACE,
        suggested_translation='Эверлост',
        confidence=0.9,
        model_flagged_review=False,
        name_translation_assessment=NameTranslationAssessment(
            NameTranslationMethod.CREATIVE, True, (), 'Творческий вариант.', opaque_name=False,
        ),
    )
    with pytest.raises(AnalysisMergeError) as existing_exc:
        validate_narrative_response(
            NarrativeAnalysisResponse(
                'Keep voice.', 'Summary.',
                glossary_candidates=(creative_conflict,),
                scenes=(scene('b1', 'b3'),),
            ),
            ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
        )
    assert existing_exc.value.violation.rule_code == 'name_assessment_conflict'
    assert all_seven in str(existing_exc.value)

    # 2. Under MEANING policy: SOUND + opaque_name=False + eligible_without_review=True is rejected.
    sound_transparent = GlossaryCandidateResponse(
        source_text='Everlost',
        type=ConceptType.PLACE,
        suggested_translation='Эверлост',
        confidence=0.9,
        model_flagged_review=False,
        name_translation_assessment=NameTranslationAssessment(
            NameTranslationMethod.SOUND, True, (), 'Транскрипция прозрачного названия.', opaque_name=False,
        ),
    )
    with pytest.raises(AnalysisMergeError) as meaning_exc:
        validate_narrative_response(
            NarrativeAnalysisResponse(
                'Keep voice.', 'Summary.',
                glossary_candidates=(sound_transparent,),
                scenes=(scene('b1', 'b3'),),
            ),
            ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
            name_translation_policy=meaning_policy,
        )
    assert meaning_exc.value.violation.rule_code == 'name_assessment_conflict'
    assert all_seven in str(meaning_exc.value)

    # Same candidate is accepted under SOUND policy or when policy is None.
    validate_narrative_response(
        NarrativeAnalysisResponse(
            'Keep voice.', 'Summary.',
            glossary_candidates=(sound_transparent,),
            scenes=(scene('b1', 'b3'),),
        ),
        ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
        name_translation_policy=sound_policy,
    )
    validate_narrative_response(
        NarrativeAnalysisResponse(
            'Keep voice.', 'Summary.',
            glossary_candidates=(sound_transparent,),
            scenes=(scene('b1', 'b3'),),
        ),
        ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
    )

    # Under MEANING policy: SOUND + opaque_name=False + eligible_without_review=False (with reason) is allowed,
    # and SOUND + opaque_name=True + eligible_without_review=True is allowed.
    sound_for_review = GlossaryCandidateResponse(
        source_text='Everlost',
        type=ConceptType.PLACE,
        suggested_translation='Эверлост',
        confidence=0.9,
        model_flagged_review=True,
        name_translation_assessment=NameTranslationAssessment(
            NameTranslationMethod.SOUND,
            False,
            (NameTranslationReviewReason.AMBIGUOUS,),
            'Нужно решение пользователя.',
            opaque_name=False,
        ),
    )
    sound_opaque = GlossaryCandidateResponse(
        source_text='Holmes',
        type=ConceptType.PERSON,
        suggested_translation='Холмс',
        confidence=0.95,
        model_flagged_review=False,
        name_translation_assessment=NameTranslationAssessment(
            NameTranslationMethod.SOUND, True, (), 'Непрозрачная фамилия.', opaque_name=True,
        ),
    )
    validate_narrative_response(
        NarrativeAnalysisResponse(
            'Keep voice.', 'Summary.',
            glossary_candidates=(sound_for_review, sound_opaque),
            scenes=(scene('b1', 'b3'),),
        ),
        ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
        name_translation_policy=meaning_policy,
    )

    # 3. Descriptive noun phrase with lowercase article (the/a/an + lowercase) + SOUND + opaque_name=False is rejected.
    for phrase in ('the freckled boy', 'a dark hallway', 'an old clock'):
        bad_descriptive = GlossaryCandidateResponse(
            source_text=phrase,
            type=ConceptType.PERSON,
            suggested_translation='веснушчатый мальчик',
            confidence=0.85,
            model_flagged_review=True,
            name_translation_assessment=NameTranslationAssessment(
                NameTranslationMethod.SOUND,
                False,
                (NameTranslationReviewReason.AMBIGUOUS,),
                'Прозвище.',
                opaque_name=False,
            ),
        )
        with pytest.raises(AnalysisMergeError) as desc_exc:
            validate_narrative_response(
                NarrativeAnalysisResponse(
                    'Keep voice.', 'Summary.',
                    glossary_candidates=(bad_descriptive,),
                    scenes=(scene('b1', 'b3'),),
                ),
                ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
            )
        assert desc_exc.value.violation.rule_code == 'name_assessment_conflict'
        assert all_seven in str(desc_exc.value)

    # MEANING + opaque_name=False on a capitalized descriptive nickname ('Iron Hand') passes.
    good_descriptive = GlossaryCandidateResponse(
        source_text='Iron Hand',
        type=ConceptType.PERSON,
        suggested_translation='Железная Рука',
        confidence=0.9,
        model_flagged_review=False,
        name_translation_assessment=NameTranslationAssessment(
            NameTranslationMethod.MEANING, True, (), 'Описательное прозвище.', opaque_name=False,
        ),
    )
    validate_narrative_response(
        NarrativeAnalysisResponse(
            'Keep voice.', 'Summary.',
            glossary_candidates=(good_descriptive,),
            scenes=(scene('b1', 'b3'),),
        ),
        ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
        name_translation_policy=meaning_policy,
    )

    # Lowercase generic person phrase ('the freckled boy') is rejected even with MEANING + opaque_name=False.
    lowercase_descriptive = GlossaryCandidateResponse(
        source_text='the freckled boy',
        type=ConceptType.PERSON,
        suggested_translation='веснушчатый мальчик',
        confidence=0.9,
        model_flagged_review=False,
        name_translation_assessment=NameTranslationAssessment(
            NameTranslationMethod.MEANING, True, (), 'Описательное прозвище.', opaque_name=False,
        ),
    )
    with pytest.raises(AnalysisMergeError) as generic_exc:
        validate_narrative_response(
            NarrativeAnalysisResponse(
                'Keep voice.', 'Summary.',
                glossary_candidates=(lowercase_descriptive,),
                scenes=(scene('b1', 'b3'),),
            ),
            ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
            name_translation_policy=meaning_policy,
        )
    assert generic_exc.value.violation is not None
    assert generic_exc.value.violation.rule_code == 'generic_person_candidate'


def test_validate_narrative_speaker_subject_anchor_must_intersect_quoted_dialogue():
    q_aliases = {'b1': 'id-1', 'b2': 'id-2', 'b3': 'id-3', 'b4': 'id-4'}
    q_texts = {
        'id-1': '‘Don’t move!’ Holmes shouted in the dark.',
        'id-2': '“Wait here,” Watson whispered softly.',
        'id-3': '«Стой на месте!» — крикнул Lestrade.',
        'id-4': 'Holmes muttered softly to himself.',
    }
    q_visible = {alias: q_texts[bid] for alias, bid in q_aliases.items()}
    q_order = ('id-1', 'id-2', 'id-3', 'id-4')
    q_scenes = (
        QuotedSceneResponse(
            'b1', 'b4', 'Напряжённая сцена.', participants=('Holmes', 'Watson', 'Lestrade'),
        ),
    )

    # Rejects remark-only subject_anchor in blocks with British single, curly double, and guillemet quotes.
    for alias, expected_bid, bad_subject, speaker in (
        ('b1', 'id-1', 'Holmes shouted', 'Holmes'),
        ('b2', 'id-2', 'Watson whispered', 'Watson'),
        ('b3', 'id-3', 'крикнул Lestrade', 'Lestrade'),
    ):
        anchor = QuotedSourceAnchorResponse(alias, bad_subject)
        bad_fact = QuotedContextFactResponse('f-bad', 'speaker', (anchor,), anchor, speaker, 'explicit')
        with pytest.raises(AnalysisMergeError) as exc:
            validate_narrative_response(
                NarrativeAnalysisResponse(
                    'Keep voice.', 'Summary.',
                    context_facts=(bad_fact,),
                    scenes=q_scenes,
                ),
                q_aliases, q_texts, q_visible, ordered_block_ids=q_order,
            )
        assert exc.value.violation.rule_code == 'speaker_subject_anchor'
        assert exc.value.violation.block_ids == (expected_bid,)

    # Accepts quoted speech (with or without outer quote chars), non-quoted block, and non-speaker fact on remark.
    good_facts = (
        QuotedContextFactResponse(
            'f-ok-1', 'speaker',
            (QuotedSourceAnchorResponse('b1', 'Holmes shouted'),),
            QuotedSourceAnchorResponse('b1', '‘Don’t move!’'),
            'Holmes', 'explicit',
        ),
        QuotedContextFactResponse(
            'f-ok-2', 'speaker',
            (QuotedSourceAnchorResponse('b2', 'Watson whispered'),),
            QuotedSourceAnchorResponse('b2', 'Wait here,'),
            'Watson', 'explicit',
        ),
        QuotedContextFactResponse(
            'f-ok-3', 'speaker',
            (QuotedSourceAnchorResponse('b3', 'крикнул Lestrade'),),
            QuotedSourceAnchorResponse('b3', '«Стой на месте!»'),
            'Lestrade', 'explicit',
        ),
        QuotedContextFactResponse(
            'f-ok-4', 'speaker',
            (QuotedSourceAnchorResponse('b4', 'Holmes muttered'),),
            QuotedSourceAnchorResponse('b4', 'Holmes muttered'),
            'Holmes', 'explicit',
        ),
        QuotedContextFactResponse(
            'f-ok-gender', 'gender',
            (QuotedSourceAnchorResponse('b1', 'Holmes shouted'),),
            QuotedSourceAnchorResponse('b1', 'Holmes'),
            'male', 'explicit',
        ),
    )
    _, resolved_facts = validate_narrative_response(
        NarrativeAnalysisResponse(
            'Keep voice.', 'Summary.',
            context_facts=good_facts,
            scenes=q_scenes,
        ),
        q_aliases, q_texts, q_visible, ordered_block_ids=q_order,
    )
    assert len(resolved_facts) == 5


def test_validate_narrative_speaker_subject_anchor_handles_straight_quotes_with_typographic_apostrophe():
    aliases = {'b1': 'id-1'}
    texts = {'id-1': '"Don’t move!" said Harry.'}
    visible = {'b1': texts['id-1']}
    scenes = (QuotedSceneResponse('b1', 'b1', 'Сцена с Гарри.', participants=('Harry',)),)

    bad_anchor = QuotedSourceAnchorResponse('b1', 'said Harry')
    bad_fact = QuotedContextFactResponse('f-bad', 'speaker', (bad_anchor,), bad_anchor, 'Harry', 'explicit')
    with pytest.raises(AnalysisMergeError) as exc:
        validate_narrative_response(
            NarrativeAnalysisResponse('Keep voice.', 'Summary.', context_facts=(bad_fact,), scenes=scenes),
            aliases, texts, visible, ordered_block_ids=('id-1',),
        )
    assert exc.value.violation.rule_code == 'speaker_subject_anchor'
    assert exc.value.violation.block_ids == ('id-1',)

    good_fact = QuotedContextFactResponse(
        'f-ok', 'speaker',
        (QuotedSourceAnchorResponse('b1', 'said Harry'),),
        QuotedSourceAnchorResponse('b1', '"Don’t move!"'),
        'Harry', 'explicit',
    )
    _, resolved = validate_narrative_response(
        NarrativeAnalysisResponse('Keep voice.', 'Summary.', context_facts=(good_fact,), scenes=scenes),
        aliases, texts, visible, ordered_block_ids=('id-1',),
    )
    assert len(resolved) == 1


def test_validate_narrative_rejects_gender_plural_subject_with_and():
    aliases = {'b1': 'id-1', 'b2': 'id-2'}
    texts = {
        'id-1': (
            'The brother and sister ran toward his mother and his father while '
            'Nick and Allie watched the boy and girl.'
        ),
        'id-2': 'Her brother waited quietly by the gate.',
    }
    visible = {'b1': texts['id-1'], 'b2': texts['id-2']}
    scenes = (
        QuotedSceneResponse('b1', 'b2', 'Сцена с детьми.', participants=('brother', 'sister', 'Nick', 'Allie')),
    )

    for quote, gender_value in (
        ('brother and sister', 'male'),
        ('his mother and his father', 'female'),
        ('boy and girl', 'male'),
        ('Nick and Allie', 'male'),
    ):
        anchor = QuotedSourceAnchorResponse('b1', quote)
        bad_fact = QuotedContextFactResponse('f-plural', 'gender', (anchor,), anchor, gender_value, 'explicit')
        with pytest.raises(AnalysisMergeError) as exc:
            validate_narrative_response(
                NarrativeAnalysisResponse('Keep voice.', 'Summary.', context_facts=(bad_fact,), scenes=scenes),
                aliases, texts, visible, ordered_block_ids=('id-1', 'id-2'),
            )
        assert exc.value.violation is not None
        assert exc.value.violation.rule_code == 'gender_plural_subject'
        assert exc.value.violation.block_ids == ('id-1',)

    # Legacy cache validation with enforce_strict_gender_subject=False allows cached V5/V6 facts.
    cached_anchor = QuotedSourceAnchorResponse('b1', 'brother and sister')
    cached_fact = QuotedContextFactResponse('f-legacy', 'gender', (cached_anchor,), cached_anchor, 'male', 'explicit')
    _, legacy_resolved = validate_narrative_response(
        NarrativeAnalysisResponse('Keep voice.', 'Summary.', context_facts=(cached_fact,), scenes=scenes),
        aliases, texts, visible, ordered_block_ids=('id-1', 'id-2'),
        enforce_strict_gender_subject=False,
    )
    assert len(legacy_resolved) == 1

    # Single-character gender and non-male/female value on plural group are accepted.
    single_anchor = QuotedSourceAnchorResponse('b2', 'brother')
    plural_anchor = QuotedSourceAnchorResponse('b1', 'brother and sister')
    good_facts = (
        QuotedContextFactResponse('f-ok-single', 'gender', (single_anchor,), single_anchor, 'male', 'explicit'),
        QuotedContextFactResponse('f-ok-unknown', 'gender', (plural_anchor,), plural_anchor, None, 'unknown'),
    )
    _, resolved = validate_narrative_response(
        NarrativeAnalysisResponse('Keep voice.', 'Summary.', context_facts=good_facts, scenes=scenes),
        aliases, texts, visible, ordered_block_ids=('id-1', 'id-2'),
    )
    assert len(resolved) == 2


def test_validate_narrative_checks_translation_instructions_language():
    scenes = (QuotedSceneResponse('b1', 'b3', 'Холмс и Ватсон в комнате.'),)

    # 1. Valid Russian instructions with quoted English terms and short glosses pass.
    for valid_instructions in (
        '«Gringotts» — «Гринготтс»',
        'Слово «Muggle» переводить как «магл».',
        'Передавать Leaky Cauldron как «Дырявый котёл».',
        'Сохранять живой разговорный тон.\n\n«Gringotts» — «Гринготтс».',
        '',
    ):
        resolved_scenes, _ = validate_narrative_response(
            NarrativeAnalysisResponse(valid_instructions, 'Holmes wears a red coat.', scenes=scenes),
            ALIASES,
            TEXTS,
            VISIBLE,
            ordered_block_ids=ORDER,
            source_language='en',
            target_language='ru',
            is_notes_only=False,
        )
        assert len(resolved_scenes) == 1

    # 2. English translation instructions (even with a Cyrillic quote) are rejected when target_language='ru'.
    for bad_instructions in (
        'Translate "Muggle" as «магл»',
        'Keep the voice.',
        'Сохранять иронию.\n\nTranslate "Muggle" as «магл».',
    ):
        with pytest.raises(AnalysisMergeError) as exc_info:
            validate_narrative_response(
                NarrativeAnalysisResponse(bad_instructions, 'Holmes wears a red coat.', scenes=scenes),
                ALIASES,
                TEXTS,
                VISIBLE,
                ordered_block_ids=ORDER,
                source_language='en',
                target_language='ru',
                is_notes_only=False,
            )
        assert exc_info.value.violation is not None
        assert exc_info.value.violation.rule_code == 'translation_instructions_language'
        assert exc_info.value.violation.block_ids == ()

    # 3. Notes-only sections and non-Russian target languages skip this check.
    validate_narrative_response(
        NarrativeAnalysisResponse('Translate "Muggle" as «магл»', 'Holmes wears a red coat.', scenes=()),
        ALIASES,
        TEXTS,
        VISIBLE,
        ordered_block_ids=ORDER,
        source_language='en',
        target_language='ru',
        is_notes_only=True,
    )
    validate_narrative_response(
        NarrativeAnalysisResponse('Translate "Muggle" as «магл»', 'Holmes wears a red coat.', scenes=(scene('b1', 'b3'),)),
        ALIASES,
        TEXTS,
        VISIBLE,
        ordered_block_ids=ORDER,
        source_language='en',
        target_language='en',
        is_notes_only=False,
    )

    # 4. Priority: scene_summary_language outranks translation_instructions_language.
    with pytest.raises(AnalysisMergeError) as prio_exc:
        validate_narrative_response(
            NarrativeAnalysisResponse(
                'Translate "Muggle" as «магл»',
                'Holmes wears a red coat.',
                scenes=(scene('b1', 'b3'),),
            ),
            ALIASES,
            TEXTS,
            VISIBLE,
            ordered_block_ids=ORDER,
            source_language='en',
            target_language='ru',
            is_notes_only=False,
        )
    assert prio_exc.value.violation is not None
    assert prio_exc.value.violation.rule_code == 'scene_summary_language'


def test_validate_narrative_rejects_generic_lowercase_person_candidates():
    from linguary.application.llm_dto import GlossaryCandidateResponse
    from linguary.domain import ConceptType
    from linguary.domain.name_translation import (
        NameTranslationAssessment,
        NameTranslationMethod,
        NameTranslationPolicy,
        NameTranslationReviewReason,
        NameTranslationStrategy,
    )

    meaning_policy = NameTranslationPolicy(1, NameTranslationStrategy.MEANING)

    for generic in (
        'the boy',
        'the baker',
        'the giant',
        'the goblin',
        'a stranger',
        'the owner of the hotel',
        'a member of his family',
        'the old man',
        'an old wizard',
        'younger boy',
        'the freckled boy',
        'the freckled kid',
    ):
        cand = GlossaryCandidateResponse(
            source_text=generic,
            type=ConceptType.PERSON,
            suggested_translation='персонаж',
            confidence=0.9,
            model_flagged_review=False,
        )
        with pytest.raises(AnalysisMergeError) as exc_info:
            validate_narrative_response(
                NarrativeAnalysisResponse(
                    'Keep voice.', 'Summary.',
                    glossary_candidates=(cand,),
                    scenes=(scene('b1', 'b3'),),
                ),
                ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
            )
        assert exc_info.value.violation is not None
        assert exc_info.value.violation.rule_code == 'generic_person_candidate'

    # Allowed: capitalized particles/names, non-PERSON type, and capitalized descriptive nicknames ('Iron Hand', 'Chocolate Boy').
    allowed = (
        GlossaryCandidateResponse(
            source_text='van Helsing',
            type=ConceptType.PERSON,
            suggested_translation='ван Хельсинг',
            confidence=0.95,
            model_flagged_review=False,
        ),
        GlossaryCandidateResponse(
            source_text='de la Cruz',
            type=ConceptType.PERSON,
            suggested_translation='де ла Крус',
            confidence=0.95,
            model_flagged_review=False,
        ),
        GlossaryCandidateResponse(
            source_text='The boy',
            type=ConceptType.PERSON,
            suggested_translation='Мальчик',
            confidence=0.9,
            model_flagged_review=False,
        ),
        GlossaryCandidateResponse(
            source_text='the boy',
            type=ConceptType.TERM,
            suggested_translation='мальчик',
            confidence=0.9,
            model_flagged_review=False,
        ),
        GlossaryCandidateResponse(
            source_text='Iron Hand',
            type=ConceptType.PERSON,
            suggested_translation='Железная Рука',
            confidence=0.9,
            model_flagged_review=False,
            name_translation_assessment=NameTranslationAssessment(
                NameTranslationMethod.MEANING, True, (), 'Описательное прозвище.', opaque_name=False,
            ),
        ),
        GlossaryCandidateResponse(
            source_text='Chocolate Boy',
            type=ConceptType.PERSON,
            suggested_translation='Шоколадный Мальчик',
            confidence=0.9,
            model_flagged_review=False,
            name_translation_assessment=NameTranslationAssessment(
                NameTranslationMethod.MEANING, True, (), 'Описательное прозвище.', opaque_name=False,
            ),
        ),
    )
    validate_narrative_response(
        NarrativeAnalysisResponse(
            'Keep voice.', 'Summary.',
            glossary_candidates=allowed,
            scenes=(scene('b1', 'b3'),),
        ),
        ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
        name_translation_policy=meaning_policy,
    )

    # Priority: name_assessment_conflict outranks generic_person_candidate.
    both_conflict = GlossaryCandidateResponse(
        source_text='the old man',
        type=ConceptType.PERSON,
        suggested_translation='старик',
        confidence=0.85,
        model_flagged_review=True,
        name_translation_assessment=NameTranslationAssessment(
            NameTranslationMethod.SOUND,
            False,
            (NameTranslationReviewReason.AMBIGUOUS,),
            'Описательное.',
            opaque_name=False,
        ),
    )
    with pytest.raises(AnalysisMergeError) as prio_exc:
        validate_narrative_response(
            NarrativeAnalysisResponse(
                'Keep voice.', 'Summary.',
                glossary_candidates=(both_conflict,),
                scenes=(scene('b1', 'b3'),),
            ),
            ALIASES, TEXTS, VISIBLE, ordered_block_ids=ORDER,
        )
    assert prio_exc.value.violation is not None
    assert prio_exc.value.violation.rule_code == 'name_assessment_conflict'


def test_find_quote_span_matches_ellipsis_and_three_dots_both_ways() -> None:
    """WP2.4: find_quote_span treats Unicode '…' and ASCII '...' equivalently without matching arbitrary text."""
    from linguary.application.analysis_merge import find_quote_span

    text_unicode = "He paused… then spoke softly."
    span1 = find_quote_span(text_unicode, "He paused... then spoke")
    assert span1 is not None
    assert text_unicode[span1[0] : span1[1]] == "He paused… then spoke"

    text_ascii = "He paused... then spoke softly."
    span2 = find_quote_span(text_ascii, "He paused… then spoke")
    assert span2 is not None
    assert text_ascii[span2[0] : span2[1]] == "He paused... then spoke"

    # Must not match two dots or arbitrary text.
    assert find_quote_span("He paused.. then spoke.", "He paused… then spoke") is None
    assert find_quote_span("He paused--- then spoke.", "He paused... then spoke") is None
    assert find_quote_span("He pausedabc then spoke.", "He paused… then spoke") is None


def test_generic_person_candidate_filter_broadens_lowercase_heads() -> None:
    from linguary.application.analysis_merge import _is_generic_person_candidate

    for generic in (
        "younger boy",
        "the freckled boy",
        "the freckled kid",
        "the baker",
        "the giant",
        "the red-haired younger boy",
        "a nervous passenger",
        "the silent figure",
        "the owner of the hotel",
    ):
        assert _is_generic_person_candidate(generic) is True, generic

    for allowed in (
        "Chocolate Boy",
        "Iron Hand",
        "The boy",
        "van Helsing",
        "the dark voldemort",
        "boy",
    ):
        assert _is_generic_person_candidate(allowed) is False, allowed


def test_spoiler_monotonic_first_name_block_index_with_later_capitalized_nickname() -> None:
    """Adding a later block with capitalized 'Chocolate Boy' must not shift lowercase generic match forward."""
    from linguary.application.analysis_merge import (
        _extract_block_spoiler_tokens,
        _first_name_block_index,
    )
    from linguary.application.llm_dto import GlossaryCandidateResponse
    from linguary.domain import ConceptType

    fragment_texts = {
        "b0": "Chapter 2.",
        "b1": "Fog covered the road.",
        "b2": "The younger freckled boy looked down at the cracked asphalt.",
        "b3": "He stepped carefully onto the gravel.",
    }
    full_texts = {
        **fragment_texts,
        **{f"b{i}": f"Filler line {i}." for i in range(4, 38)},
        "b38": "Everyone in the forest called him Chocolate Boy.",
    }
    frag_tokens = _extract_block_spoiler_tokens(tuple(fragment_texts), fragment_texts)
    full_tokens = _extract_block_spoiler_tokens(tuple(full_texts), full_texts)

    glossary_with_lowercase = (
        GlossaryCandidateResponse(
            source_text="the freckled boy",
            type=ConceptType.PERSON,
            suggested_translation="веснушчатый мальчик",
            confidence=0.9,
            model_flagged_review=False,
        ),
        GlossaryCandidateResponse(
            source_text="Chocolate Boy",
            type=ConceptType.PERSON,
            suggested_translation="Шоколадный Мальчик",
            confidence=0.9,
            model_flagged_review=False,
        ),
    )

    # Monotonicity: 'the freckled boy' matches at block 2 both before and after block 38 is appended.
    assert _first_name_block_index("the freckled boy", frag_tokens, glossary_with_lowercase) == 2
    assert _first_name_block_index("the freckled boy", full_tokens, glossary_with_lowercase) == 2

    # Even if a capitalized 'The Freckled Boy' candidate existed, lowercase generic phrase stays at block 2.
    glossary_with_cap = (
        GlossaryCandidateResponse(
            source_text="The Freckled Boy",
            type=ConceptType.PERSON,
            suggested_translation="Веснушчатый Мальчик",
            confidence=0.9,
            model_flagged_review=False,
        ),
    )
    assert _first_name_block_index("the freckled boy", full_tokens, glossary_with_cap) == 2

    # Capitalized 'Chocolate Boy' strips generic head 'boy' and resolves to block 38 (not block 2).
    assert _first_name_block_index("Chocolate Boy", full_tokens, glossary_with_lowercase) == 38

    # Lowercase 'younger boy' keeps 'boy' and is treated as a neutral role label (no spoiler).
    assert _first_name_block_index("younger boy", full_tokens, glossary_with_lowercase) is None

    # Regression guard: 'the dark voldemort' with capitalized glossary 'Voldemort' still prefers capitalized match.
    spoiler_texts = {
        "b0": "A dark shadow passed by the door.",
        "b1": "Then Voldemort spoke from the fog.",
    }
    spoiler_tokens = _extract_block_spoiler_tokens(("b0", "b1"), spoiler_texts)
    voldemort_glossary = (
        GlossaryCandidateResponse(
            source_text="Voldemort",
            type=ConceptType.PERSON,
            suggested_translation="Волдеморт",
            confidence=0.98,
            model_flagged_review=False,
        ),
    )
    assert _first_name_block_index("the dark voldemort", spoiler_tokens, voldemort_glossary) == 1


def test_first_name_block_index_lowercase_non_glossary_ignores_sentence_initial_cap() -> None:
    """Lowercase non-glossary label appearing lowercase before a later capitalized occurrence is not a spoiler."""
    from linguary.application.analysis_merge import (
        _extract_block_spoiler_tokens,
        _first_name_block_index,
    )

    texts = {
        **{f"b{i}": f"Filler paragraph {i}." for i in range(150)},
        "b19": "The first-years followed Professor McGonagall down the corridor.",
        "b21": "All the first-years looked up at the enchanted ceiling.",
        "b145": "First-years should follow the prefects to the dormitories.",
    }
    ordered = tuple(f"b{i}" for i in range(150))
    block_tokens = _extract_block_spoiler_tokens(ordered, texts)

    assert _first_name_block_index("first-years", block_tokens, ()) is None

    # Proper name written lowercase without glossary still resolves when there is no earlier lowercase mention.
    proper_texts = {
        "b0": "Someone knocked softly at the door.",
        "b1": "Then Voldemort stepped out of the mist.",
    }
    proper_tokens = _extract_block_spoiler_tokens(("b0", "b1"), proper_texts)
    assert _first_name_block_index("voldemort", proper_tokens, ()) == 1


def test_opening_remark_names_speaker_requires_capitalized_single_word_match() -> None:
    """Single-word speaker lookup in opening remark requires capitalization, while full phrases and proper names match."""
    from linguary.application.analysis_merge import _opening_remark_names_speaker

    assert (
        _opening_remark_names_speaker(
            "What looked like a fat little monk was saying: ‘Forgive and forget, I say, we ought to give him a second chance —’",
            "Fat Friar",
        )
        is False
    )
    assert _opening_remark_names_speaker("Hagrid said: ‘Hello’", "Rubeus Hagrid") is True
    assert _opening_remark_names_speaker("Harry said: ‘Hi’", "Harry Potter") is True
    assert (
        _opening_remark_names_speaker(
            "‘Come along,’ asked Professor McGonagall.",
            "Professor McGonagall",
        )
        is True
    )
    assert _opening_remark_names_speaker("‘Listen,’ said Ron to Harry.", "Harry Potter") is False
    assert _opening_remark_names_speaker("The boy said: ‘Look.’", "the boy") is True


def test_explicit_speaker_if_remark_names_blocks_spoiler_promotion_on_both_branches() -> None:
    """Anti-spoiler guard blocks inferred->explicit promotion in both name and pronoun branches when first_pos > fact_pos."""
    from linguary.application.analysis_merge import (
        _explicit_speaker_if_remark_names,
        _extract_block_spoiler_tokens,
    )
    from linguary.application.llm_dto import GlossaryCandidateResponse
    from linguary.domain import ConceptType
    from linguary.domain.content_plan import ContextFact, Scene, SourceAnchor

    # Branch 1 (name-match): 'Boy' is capitalized in opening remark, matching 'Chocolate Boy',
    # but distinctive name 'Chocolate Boy' only first appears in b1 (first_pos=1 > fact_pos=0).
    name_texts = {
        "b0": "A Boy shouted: ‘Stop right there!’",
        "b1": "Everyone in the hall called him Chocolate Boy.",
    }
    name_order = ("b0", "b1")
    name_tokens = _extract_block_spoiler_tokens(name_order, name_texts)
    name_block_order = {"b0": 0, "b1": 1}
    quote_span = (name_texts["b0"].index("‘"), len(name_texts["b0"]))
    anchor_b0 = SourceAnchor("b0", quote_span[0], quote_span[1], name_texts["b0"][quote_span[0]:quote_span[1]])
    name_fact = ContextFact(
        id="f-name",
        kind="speaker",
        anchors=(anchor_b0,),
        subject_anchor=anchor_b0,
        value="Chocolate Boy",
        certainty="inferred",
    )
    # Without spoiler context, branch 1 would promote; with block_tokens+block_order it must stay inferred.
    assert (
        _explicit_speaker_if_remark_names(
            name_fact, name_texts, None, ordered_block_ids=name_order
        ).certainty
        == "explicit"
    )
    kept_name = _explicit_speaker_if_remark_names(
        name_fact,
        name_texts,
        None,
        ordered_block_ids=name_order,
        block_tokens=name_tokens,
        block_order=name_block_order,
    )
    assert kept_name.certainty == "inferred"

    # Branch 2 (pronoun-match): 'he said' uniquely resolves to 'Fat Friar' in the scene,
    # but 'Fat Friar' only first appears in b1 (first_pos=1 > fact_pos=0).
    pronoun_texts = {
        "b0": "‘Forgive and forget,’ he said.",
        "b1": "The Fat Friar bowed politely. ‘Welcome,’ he said.",
    }
    pronoun_order = ("b0", "b1")
    pronoun_tokens = _extract_block_spoiler_tokens(pronoun_order, pronoun_texts)
    pronoun_block_order = {"b0": 0, "b1": 1}
    glossary = (
        GlossaryCandidateResponse(
            source_text="Fat Friar",
            type=ConceptType.PERSON,
            suggested_translation="Толстый Монах",
            confidence=0.95,
            model_flagged_review=False,
            note="Пол: мужской.",
        ),
    )
    scene = Scene(
        id="s1",
        order=0,
        start_block_id="b0",
        end_block_id="b1",
        summary="Монах приветствует учеников.",
        location="Great Hall",
        participants=("Fat Friar",),
    )
    q0_end = pronoun_texts["b0"].index("’") + 1
    p_anchor_b0 = SourceAnchor("b0", 0, q0_end, pronoun_texts["b0"][:q0_end])
    pronoun_fact_b0 = ContextFact(
        id="f-pronoun-b0",
        kind="speaker",
        anchors=(p_anchor_b0,),
        subject_anchor=p_anchor_b0,
        value="Fat Friar",
        certainty="inferred",
    )
    # Without spoiler context, branch 2 would promote; with block_tokens+block_order it must stay inferred.
    assert (
        _explicit_speaker_if_remark_names(
            pronoun_fact_b0,
            pronoun_texts,
            None,
            scenes=(scene,),
            glossary_candidates=glossary,
            ordered_block_ids=pronoun_order,
        ).certainty
        == "explicit"
    )
    kept_pronoun = _explicit_speaker_if_remark_names(
        pronoun_fact_b0,
        pronoun_texts,
        None,
        scenes=(scene,),
        glossary_candidates=glossary,
        ordered_block_ids=pronoun_order,
        block_tokens=pronoun_tokens,
        block_order=pronoun_block_order,
    )
    assert kept_pronoun.certainty == "inferred"

    # Non-spoiler position (b1, where first_pos=1 == fact_pos=1) is still promoted to explicit.
    q1_start = pronoun_texts["b1"].index("‘Welcome,’")
    q1_end = q1_start + len("‘Welcome,’")
    p_anchor_b1 = SourceAnchor("b1", q1_start, q1_end, pronoun_texts["b1"][q1_start:q1_end])
    pronoun_fact_b1 = ContextFact(
        id="f-pronoun-b1",
        kind="speaker",
        anchors=(p_anchor_b1,),
        subject_anchor=p_anchor_b1,
        value="Fat Friar",
        certainty="inferred",
    )
    promoted_pronoun = _explicit_speaker_if_remark_names(
        pronoun_fact_b1,
        pronoun_texts,
        None,
        scenes=(scene,),
        glossary_candidates=glossary,
        ordered_block_ids=pronoun_order,
        block_tokens=pronoun_tokens,
        block_order=pronoun_block_order,
    )
    assert promoted_pronoun.certainty == "explicit"


def test_validate_narrative_response_idempotent_across_fragment_merge_and_revalidation() -> None:
    """Fragment validation -> merge_narrative_fragments -> repeated validate_narrative_response is idempotent."""
    from linguary.application.analysis_merge import merge_narrative_fragments
    from linguary.application.llm_dto import GlossaryCandidateResponse
    from linguary.domain import BlockType, ConceptType, DocumentBlock, SupportStatus

    raw_texts = {
        "blk-19": "What looked like a fat little monk was saying: ‘Forgive and forget, I say.’",
        "blk-20": "‘We ought to give him a second chance,’ he said.",
        "blk-23": "The Fat Friar said: ‘Hope to see you in Hufflepuff!’",
    }
    blocks = tuple(
        DocumentBlock(
            id=bid,
            section_id="sec-1",
            order=idx,
            type=BlockType.PARAGRAPH,
            text=text,
            source_reference=f"p{idx}",
            support_status=SupportStatus.SUPPORTED,
        )
        for idx, (bid, text) in enumerate(raw_texts.items())
    )
    order = tuple(raw_texts)
    frag_aliases = {"b1": "blk-19", "b2": "blk-20", "b3": "blk-23"}
    frag_visible = {alias: raw_texts[bid] for alias, bid in frag_aliases.items()}
    glossary = (
        GlossaryCandidateResponse(
            source_text="Fat Friar",
            type=ConceptType.PERSON,
            suggested_translation="Толстый Монах",
            confidence=0.95,
            model_flagged_review=False,
            note="Пол: мужской.",
        ),
    )
    payload = NarrativeAnalysisResponse(
        translation_instructions="Сохранять голос автора.",
        chapter_summary="The ghosts greet the students.",
        glossary_candidates=glossary,
        context_facts=(
            QuotedContextFactResponse(
                "f19",
                "speaker",
                (QuotedSourceAnchorResponse("b1", "‘Forgive and forget, I say.’"),),
                QuotedSourceAnchorResponse("b1", "‘Forgive and forget, I say.’"),
                "Fat Friar",
                "inferred",
            ),
            QuotedContextFactResponse(
                "f20",
                "speaker",
                (QuotedSourceAnchorResponse("b2", "‘We ought to give him a second chance,’"),),
                QuotedSourceAnchorResponse("b2", "‘We ought to give him a second chance,’"),
                "Fat Friar",
                "inferred",
            ),
            QuotedContextFactResponse(
                "f23",
                "speaker",
                (QuotedSourceAnchorResponse("b3", "‘Hope to see you in Hufflepuff!’"),),
                QuotedSourceAnchorResponse("b3", "‘Hope to see you in Hufflepuff!’"),
                "Fat Friar",
                "inferred",
            ),
        ),
        scenes=(
            QuotedSceneResponse(
                "b1",
                "b3",
                "Призраки беседуют с первокурсниками.",
                location="Great Hall",
                participants=("Fat Friar",),
            ),
        ),
    )

    frag_scenes, frag_facts = validate_narrative_response(
        payload, frag_aliases, raw_texts, frag_visible, ordered_block_ids=order, blocks=blocks
    )
    by_block_frag = {f.subject_anchor.block_id: f.certainty for f in frag_facts}
    assert by_block_frag == {"blk-19": "inferred", "blk-20": "inferred", "blk-23": "explicit"}

    merged, _, _ = merge_narrative_fragments(
        [(payload, frag_scenes, frag_facts)], section_blocks=blocks
    )
    chapter_aliases = {bid: bid for bid in order}
    # Chapter-wide validation + second re-validation (as in illustration_source.py:930).
    ch_scenes_1, ch_facts_1 = validate_narrative_response(
        merged, chapter_aliases, raw_texts, raw_texts, ordered_block_ids=order, blocks=blocks
    )
    ch_scenes_2, ch_facts_2 = validate_narrative_response(
        merged, chapter_aliases, raw_texts, raw_texts, ordered_block_ids=order, blocks=blocks
    )
    assert ch_scenes_1 == ch_scenes_2
    assert ch_facts_1 == ch_facts_2
    assert {f.subject_anchor.block_id: f.certainty for f in ch_facts_2} == {
        "blk-19": "inferred",
        "blk-20": "inferred",
        "blk-23": "explicit",
    }

    # combine_staged_analysis_responses and content_plan_from_response also preserve anti-spoiler certainty.
    combined = combine_staged_analysis_responses(
        merged,
        PedagogicalAnalysisResponse(),
        resolved_scenes=ch_scenes_1,
        resolved_blocks=blocks,
    )
    assert combined.content_plan is not None
    assert {f.subject_anchor.block_id: f.certainty for f in combined.content_plan.context_facts} == {
        "blk-19": "inferred",
        "blk-20": "inferred",
        "blk-23": "explicit",
    }
    plan = content_plan_from_response(
        combined, chapter_aliases, raw_texts, visible_texts_by_alias=raw_texts, blocks=blocks
    )
    assert plan is not None
    assert {f.subject_anchor.block_id: f.certainty for f in plan.context_facts} == {
        "blk-19": "inferred",
        "blk-20": "inferred",
        "blk-23": "explicit",
    }


def test_opening_remark_names_speaker_handles_unicode_and_hyphens() -> None:
    """`_opening_remark_names_speaker` preserves Unicode letters and per-word capitalization across hyphens."""
    from linguary.application.analysis_merge import _opening_remark_names_speaker

    assert _opening_remark_names_speaker('Éowyn said, "Stand back!"', "Éowyn") is True
    assert _opening_remark_names_speaker('A well-fat monk said, "Hello!"', "Fat Friar") is False
    assert _opening_remark_names_speaker('Well-fat monk said, "Hello!"', "Fat Friar") is False
    assert _opening_remark_names_speaker('A non-Harry voice said, "Hello!"', "Harry Potter") is True











