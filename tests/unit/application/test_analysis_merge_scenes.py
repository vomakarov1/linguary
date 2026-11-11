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


@pytest.mark.parametrize('block,quote,code', [
    ('b3', 'Lestrade', 'scene_visual_anchor_out_of_bounds'),
    ('b1', 'blue coat', 'unresolved_quote'),
    ('b9', 'Holmes', 'unresolved_quote'),
])
def test_resolve_scenes_rejects_invalid_visual_fact(block, quote, code):
    raw = scene('b1', 'b2', visual_facts=(QuotedSceneVisualFactResponse(
        QuotedSourceAnchorResponse(block, quote), 'Visitor', 'Coat',
    ),))
    with pytest.raises(AnalysisMergeError) as exc:
        resolve((raw,))
    assert exc.value.violation.rule_code == code


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
