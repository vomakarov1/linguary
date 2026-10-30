"""Reading run results: layout of notes and warnings and the count of processed blocks.

Notes and warnings are loaded for the entire run at once, so their block affiliation
is repository behavior rather than a single-row query. The tests ensure that each block
receives its records in their original order, and the count of processed blocks is computed
separately from assembling the results themselves.
"""

from __future__ import annotations

import pytest
from persistence_factories import (
    BLOCK_ID,
    BOOK_ID,
    RUN_ID,
    SECOND_BLOCK_ID,
    make_book,
    make_document,
    make_run,
)

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    LanguageNote,
    ResultWarning,
    ResultWarningKind,
)


def annotated_result(block_id: str, *, marker: str) -> BlockProcessingResult:
    """Block translation with two notes and two warnings tagged with their block."""
    return BlockProcessingResult(
        block_id=block_id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text=f"Перевод блока {marker}.",
        language_notes=tuple(
            LanguageNote(
                id=f"ln-{marker}-{index}",
                block_id=block_id,
                original_excerpt=f"excerpt {marker} {index}",
                translated_excerpt=f"фрагмент {marker} {index}",
                note=f"Заметка {marker} {index}.",
            )
            for index in (1, 2)
        ),
        warnings=tuple(
            ResultWarning(
                kind=ResultWarningKind.REPETITION,
                message=f"Замечание {marker} {index}.",
            )
            for index in (1, 2)
        ),
    )


@pytest.fixture
def stored_results(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> tuple[BlockProcessingResult, ...]:
    """Run with results for two blocks: each has its own notes and warnings."""
    results = (
        annotated_result(BLOCK_ID, marker="first"),
        annotated_result(SECOND_BLOCK_ID, marker="second"),
    )
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
        uow.documents.save(make_document())
        uow.runs.add(make_run())
        uow.results.replace_results(RUN_ID, list(results))
    return results


def test_results_of_a_run_keep_their_own_notes_and_warnings(
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_results: tuple[BlockProcessingResult, ...],
) -> None:
    """Run result listing assigns each block its own notes and warnings."""
    with unit_of_work.transaction() as uow:
        listed = uow.results.list_for_run(RUN_ID)
        single = uow.results.get(RUN_ID, SECOND_BLOCK_ID)

    assert listed == stored_results
    assert single == stored_results[1]


def test_count_for_run_matches_the_number_of_stored_results(
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_results: tuple[BlockProcessingResult, ...],
) -> None:
    """Processed block count matches the number of stored results."""
    with unit_of_work.transaction() as uow:
        assert uow.results.count_for_run(RUN_ID) == len(stored_results)


def test_count_for_run_without_results_is_zero(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Run without any results yields zero rather than an error."""
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
        uow.runs.add(make_run())

    with unit_of_work.transaction() as uow:
        assert uow.results.count_for_run(RUN_ID) == 0


def test_block_replacement_keeps_evidence_atomic(unit_of_work, stored_results):
    from dataclasses import replace
    from linguary.domain.content_plan import ContentEvidence, HelpEvidence
    evidence = ContentEvidence('a' * 64, (HelpEvidence(
        'difficulty-1', 'meaning', 'language_note.note', SECOND_BLOCK_ID,
        'key-1', stored_results[1].language_notes[0].note,
    ),))
    changed = replace(stored_results[1], translated_text='Новый перевод.', content_evidence=evidence)
    with unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, [changed])
    with unit_of_work.transaction() as uow:
        assert uow.results.list_for_run(RUN_ID) == (stored_results[0], changed)
        assert uow.results.get(RUN_ID, SECOND_BLOCK_ID) == changed
    with pytest.raises(RuntimeError), unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, [replace(changed, content_evidence=())])
        raise RuntimeError('отмена замены')
    with unit_of_work.transaction() as uow:
        assert uow.results.list_for_run(RUN_ID) == (stored_results[0], changed)


def test_group_serialization_failure_rolls_back_all_rows(unit_of_work, stored_results):
    from dataclasses import replace
    from linguary.application.ports.persistence import PersistenceError
    from linguary.domain.content_plan import ContentEvidence
    changed = replace(stored_results[0], translated_text='Новый перевод.', content_evidence=ContentEvidence('a' * 64))
    invalid = replace(stored_results[1], content_evidence={'private-book-text': 'invalid'})
    with pytest.raises(PersistenceError), unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, [changed, invalid])
    with unit_of_work.transaction() as uow:
        assert uow.results.list_for_run(RUN_ID) == stored_results


def assistance(block_id, *, kind='lexical_hint'):
    from linguary.domain.learning import FragmentAssistance
    from linguary.domain.content_plan import SourceAnchor
    return FragmentAssistance('difficulty-1', ('point-1',), kind,
                              None if kind == 'unresolved' else 'go back',
                              (SourceAnchor(block_id, 0, 2, 'He'),))


@pytest.mark.parametrize('kind', ['lexical_hint', 'sentence_rephrase', 'russian_explanation', 'unresolved'])
def test_fragment_assistance_round_trip_and_atomic_replacement(unit_of_work, stored_results, kind):
    from dataclasses import replace
    changed = replace(stored_results[0], fragment_assistances=(assistance(BLOCK_ID, kind=kind),))
    with unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, (changed,))
    with unit_of_work.transaction() as uow:
        assert uow.results.get(RUN_ID, BLOCK_ID) == changed
        assert uow.results.list_for_run(RUN_ID) == (changed, stored_results[1])
    with pytest.raises(RuntimeError), unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, (replace(changed, fragment_assistances=()),))
        raise RuntimeError('cancel replacement')
    with unit_of_work.transaction() as uow:
        assert uow.results.list_for_run(RUN_ID) == (changed, stored_results[1])


def test_invalid_assistance_rolls_back_whole_replacement(unit_of_work, stored_results):
    from dataclasses import replace
    from linguary.application.ports.persistence import PersistenceError
    invalid = replace(stored_results[1])
    object.__setattr__(invalid, 'fragment_assistances', ({'difficulty_id': 'invalid'},))
    changed = replace(stored_results[0], fragment_assistances=(assistance(BLOCK_ID),))
    with pytest.raises(PersistenceError), unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, (changed, invalid))
    with unit_of_work.transaction() as uow:
        assert uow.results.list_for_run(RUN_ID) == stored_results


def test_legacy_archive_body_and_checksum_are_not_rewritten(unit_of_work, stored_results):
    from dataclasses import asdict
    import hashlib
    import json
    body = {'schema_version': 1, 'result': asdict(stored_results[0])}
    body['result'].pop('fragment_assistances')
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    with unit_of_work.transaction() as uow:
        artifact_id, = uow.work.archive_artifacts([dict(
            book_id=BOOK_ID, kind='block_result', scope_id=BLOCK_ID, body=body,
        )])
    with unit_of_work.transaction() as uow:
        archived = uow.work.get_artifact(artifact_id)
        assert archived['body'] == json.loads(canonical)
        assert archived['sha256'] == digest
        assert archived['body']['result'].get('fragment_assistances', []) == []
        assert uow.results.get(RUN_ID, BLOCK_ID).fragment_assistances == ()


def test_failed_archive_and_result_replacement_preserve_accepted_help(unit_of_work, stored_results):
    from dataclasses import asdict, replace
    old = replace(stored_results[0], fragment_assistances=(assistance(BLOCK_ID),))
    with unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, (old,))
        accepted_id, = uow.work.archive_artifacts([dict(book_id=BOOK_ID, kind='block_result',
            scope_id=BLOCK_ID, body={'schema_version': 1, 'result': asdict(old)})])
    replacement = replace(old, fragment_assistances=(assistance(BLOCK_ID, kind='unresolved'),))
    with pytest.raises(RuntimeError), unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, (replacement,))
        rejected_id, = uow.work.archive_artifacts([dict(book_id=BOOK_ID, kind='block_result',
            scope_id=BLOCK_ID, body={'schema_version': 1, 'result': asdict(replacement)})])
        raise RuntimeError('failure after archive')
    with unit_of_work.transaction() as uow:
        assert uow.results.get(RUN_ID, BLOCK_ID) == old
        assert uow.results.get(RUN_ID, SECOND_BLOCK_ID) == stored_results[1]
        assert uow.work.get_artifact(rejected_id) is None
        assert uow.work.get_artifact(accepted_id)['body']['result']['fragment_assistances'][0]['text'] == 'go back'


def pronunciation(block_id=BLOCK_ID):
    from linguary.domain.pronunciation import PronunciationAssistance, PronunciationEvidence
    from linguary.domain.content_plan import SourceAnchor
    return PronunciationAssistance(
        anchor=SourceAnchor(block_id, 0, 2, 'It'), form='It', ipa='ɪt', variant='uk',
        hint_id='synthetic-hint-1', evidence=PronunciationEvidence(
            source_id='synthetic', source_version='invented-v1', entry_id='invented-entry',
            sense_id='invented-sense', part_of_speech='pronoun',
            dictionary_form='It', dictionary_ipa='ɪt', attribution='Invented test data'))


@pytest.mark.parametrize('derived', [False, True])
def test_pronunciation_result_roundtrip(unit_of_work, stored_results, derived):
    from dataclasses import replace
    help = pronunciation()
    if derived:
        from linguary.domain.content_plan import SourceAnchor
        help = replace(help, anchor=SourceAnchor(BLOCK_ID, 3, 6, 'was'), form='was', ipa='wɒz',
                       evidence=replace(help.evidence, dictionary_form='be', dictionary_ipa='biː',
                                        rule_id='invented-rule', rule_version='invented-v1'))
    changed = replace(stored_results[0], pronunciation_assistances=(help,))
    with unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, (changed,))
    with unit_of_work.transaction() as uow:
        assert uow.results.get(RUN_ID, BLOCK_ID) == changed
        assert uow.results.list_for_run(RUN_ID) == (changed, stored_results[1])


def test_pronunciation_legacy_null(unit_of_work, stored_results):
    with unit_of_work.transaction() as uow:
        assert uow.results.get(RUN_ID, BLOCK_ID).pronunciation_assistances == ()


def test_pronunciation_archive_preserves_evidence(unit_of_work, stored_results, engine):
    from dataclasses import asdict, replace
    from linguary.application.use_cases.reset_processing import _archive_run_results
    old = replace(stored_results[0], pronunciation_assistances=(pronunciation(),))
    with unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, (old,))
        _archive_run_results(uow, BOOK_ID, make_run())
        uow.runs.delete(RUN_ID)
    with unit_of_work.transaction() as uow:
        from sqlalchemy import select
        from linguary.adapters.db.models import ArtifactRevisionRow
        with engine.connect() as connection:
            artifact_id = connection.scalar(select(ArtifactRevisionRow.id).where(ArtifactRevisionRow.scope_id == BLOCK_ID))
        artifact = uow.work.get_artifact(artifact_id)
        assert artifact['body']['result']['pronunciation_assistances'] == [asdict(pronunciation())]


def test_pronunciation_atomic_replace(unit_of_work, stored_results):
    from dataclasses import replace
    old = replace(stored_results[0], pronunciation_assistances=(pronunciation(),))
    with unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, (old,))
    with pytest.raises(RuntimeError), unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, (replace(old, pronunciation_assistances=()),))
        raise RuntimeError('cancel pronunciation replacement')
    with unit_of_work.transaction() as uow:
        assert uow.results.list_for_run(RUN_ID) == (old, stored_results[1])


def test_pronunciation_rejects_duplicate_anchor(stored_results):
    from dataclasses import replace
    from linguary.domain.errors import BlockResultError
    with pytest.raises(BlockResultError):
        replace(stored_results[0], pronunciation_assistances=(pronunciation(), pronunciation()))


def test_pronunciation_rechecks_original_before_storage(unit_of_work, stored_results):
    from dataclasses import replace
    from linguary.domain.content_plan import SourceAnchor
    from linguary.domain.errors import DomainError
    invalid = replace(pronunciation(), anchor=SourceAnchor(BLOCK_ID, 0, 2, 'XX'), form='XX',
                      evidence=replace(pronunciation().evidence, dictionary_form='XX'))
    with pytest.raises(DomainError), unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, (replace(stored_results[0], pronunciation_assistances=(invalid,)),))
    with unit_of_work.transaction() as uow:
        assert uow.results.list_for_run(RUN_ID) == stored_results


def test_pronunciation_unavailable_preserves_accepted_help(unit_of_work, stored_results):
    from dataclasses import replace
    from linguary.application.block_processing import TranslationCall
    old = replace(stored_results[0], pronunciation_assistances=(pronunciation(),))
    with unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, (old,))
    call = TranslationCall(unit_of_work=unit_of_work, llm_client=None, retry_policy=None,
                           validator=None, sleep=None)
    changed = replace(old, translated_text='Новый перевод.', pronunciation_assistances=())
    assert call.save(book_id=BOOK_ID, run_id=RUN_ID, results=(changed,), pronunciation_unavailable=True)
    with unit_of_work.transaction() as uow:
        assert uow.results.get(RUN_ID, BLOCK_ID) == replace(changed, pronunciation_assistances=old.pronunciation_assistances)
    assert call.save(book_id=BOOK_ID, run_id=RUN_ID, results=(changed,))
    with unit_of_work.transaction() as uow:
        assert uow.results.get(RUN_ID, BLOCK_ID) == changed


def lexicon_pronunciation():
    from dataclasses import replace
    from linguary.domain.pronunciation import PronunciationEvidence
    snapshot = '972cd67dd5f96bc5660e28070cde26e3928abcd7'
    return replace(pronunciation(), evidence=PronunciationEvidence(
        source_id='ipa-dict', source_version=f'ipa-dict@{snapshot}:parser-1',
        entry_id='en_UK.txt:1:It', sense_id=None, part_of_speech=None,
        dictionary_form='It', dictionary_ipa='ɪt', attribution='IPA-Dict',
        kind='lexicon_form', license_id='GPL-3.0',
        source_uri=f'https://raw.githubusercontent.com/open-dict-data/ipa-dict/{snapshot}/data/en_UK.txt',
    ))


def test_pronunciation_legacy_evidence_json_roundtrip(unit_of_work, stored_results, engine):
    from dataclasses import asdict, replace
    from sqlalchemy import update
    from linguary.adapters.db.models import BlockResultRow
    old = pronunciation()
    changed = replace(stored_results[0], pronunciation_assistances=(old,))
    with unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, (changed,))
    legacy = asdict(old)
    for field in ('kind', 'license_id', 'source_uri'):
        legacy['evidence'].pop(field, None)
    with engine.begin() as connection:
        connection.execute(update(BlockResultRow).where(
            BlockResultRow.run_id == RUN_ID, BlockResultRow.block_id == BLOCK_ID,
        ).values(pronunciation_assistances_json=[legacy]))

    with unit_of_work.transaction() as uow:
        restored = uow.results.get(RUN_ID, BLOCK_ID)
        assert restored == changed
        assert restored.pronunciation_assistances[0].evidence.kind == 'dictionary_sense'
        uow.results.replace_results(RUN_ID, (restored,))
    with unit_of_work.transaction() as uow:
        assert uow.results.get(RUN_ID, BLOCK_ID) == changed


def test_lexicon_form_evidence_roundtrip_and_archive(unit_of_work, stored_results, engine):
    from dataclasses import asdict, replace
    from sqlalchemy import select
    from linguary.adapters.db.models import ArtifactRevisionRow
    from linguary.application.use_cases.reset_processing import _archive_run_results
    help = lexicon_pronunciation()
    changed = replace(stored_results[0], pronunciation_assistances=(help,))
    with unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, (changed,))
    with unit_of_work.transaction() as uow:
        assert uow.results.get(RUN_ID, BLOCK_ID) == changed
        _archive_run_results(uow, BOOK_ID, make_run())
        uow.runs.delete(RUN_ID)
    with engine.connect() as connection:
        artifact_id = connection.scalar(select(ArtifactRevisionRow.id).where(
            ArtifactRevisionRow.scope_id == BLOCK_ID,
        ))
    with unit_of_work.transaction() as uow:
        artifact = uow.work.get_artifact(artifact_id)
        assert artifact['body']['result']['pronunciation_assistances'] == [asdict(help)]


@pytest.mark.parametrize('change', [
    'unknown-field', 'unknown-kind', 'partial-kind', 'partial-license',
    'mixed-kind', 'wrong-license-type', 'wrong-kind-type', 'foreign-uri', 'legacy-with-kind',
])
def test_pronunciation_unknown_kind_and_fields_rejected(change):
    from dataclasses import asdict
    from linguary.adapters.db.mapping import _decode_pronunciation
    from linguary.application.ports.persistence import PersistenceError
    raw = asdict(lexicon_pronunciation())
    evidence = raw['evidence']
    if change == 'unknown-field':
        evidence['other'] = 'unexpected'
    elif change == 'unknown-kind':
        evidence['kind'] = 'other'
    elif change == 'partial-kind':
        evidence.pop('kind')
    elif change == 'partial-license':
        evidence.pop('license_id')
    elif change == 'mixed-kind':
        evidence['kind'] = 'dictionary_sense'
    elif change == 'wrong-license-type':
        evidence['license_id'] = True
    elif change == 'wrong-kind-type':
        evidence['kind'] = True
    elif change == 'foreign-uri':
        evidence['source_uri'] = 'https://example.com/other'
    else:
        raw = asdict(pronunciation())
        raw['evidence'].pop('license_id')
        raw['evidence'].pop('source_uri')
    with pytest.raises(PersistenceError):
        _decode_pronunciation([raw])
