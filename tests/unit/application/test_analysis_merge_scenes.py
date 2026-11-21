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
from linguary.domain import BlockTreatment

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


@pytest.mark.parametrize('source,quote', [
    ('Holmes wore a red coat.', 'RED COAT'),
    ('He is tall.', "He's tall."),
    ('Holmes wore a red coat.', '!red coat!'),
])
@pytest.mark.parametrize('imprecise_view', ['source', 'visible'])
def test_visual_facts_require_verbatim_quotes_in_both_views(source, quote, imprecise_view):
    texts = {'id-1': source if imprecise_view == 'source' else quote}
    visible = {'b1': source if imprecise_view == 'visible' else quote}
    raw = scene('b1', 'b1', visual_facts=(QuotedSceneVisualFactResponse(
        QuotedSourceAnchorResponse('b1', quote), 'Visitor', 'A detail'),))
    with pytest.raises(AnalysisMergeError) as exc:
        resolve_scenes((raw,), {'b1': 'id-1'}, texts, visible, ordered_block_ids=('id-1',))
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
            QuotedContextFactResponse('f-glossary-2', 'gender', (QuotedSourceAnchorResponse('b5', 'Moriarty'),), QuotedSourceAnchorResponse('b5', 'Moriarty'), 'female', 'explicit'),
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
            mandatory_facts={'Hudson': {'gender': 'female'}, 'Moriarty': {'gender': 'male'}},
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
        'Hudson',
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
