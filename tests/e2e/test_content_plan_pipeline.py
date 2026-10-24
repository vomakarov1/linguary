"""Isolated full candidate path: fake verifies contract rather than model quality."""
from dataclasses import dataclass
from html import escape
import hashlib
import json
from pathlib import Path
import socket

from ebooklib import epub
import httpx
import pytest
from pipeline_support import exported_documents, sequential_ids

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings, WorkerPoolSettings
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.adapters.validation import EpubCheckSettings, EpubCheckValidator
from linguary.application.context_builder import textual_blocks
from linguary.application.use_cases.approve_glossary import GlossaryDecision, GlossaryDecisionCommand, StartProcessingCommand
from linguary.application.use_cases.register_book import RegisterBookCommand
from linguary.composition import build_application, drain_tasks
from linguary.domain import TaskStatus

STORIES = json.loads((Path(__file__).resolve().parents[1] / 'fixtures/content_plan/stories.json').read_text())
BOOK = 'content-plan-story'


@pytest.fixture(autouse=True)
def network_forbidden(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('End-to-end check must not make network calls')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(httpx.Client, 'send', forbidden)
    monkeypatch.setattr(httpx.AsyncClient, 'send', forbidden)


def run(awaitable):
    import asyncio
    return asyncio.run(awaitable)


def finish(app):
    tasks = run(drain_tasks(app))
    assert tasks and all(t.status is TaskStatus.SUCCEEDED for t in tasks), [(t.status, t.last_error) for t in tasks]
    return tasks


def make_app(root, client, story, *, candidate=True):
    return build_application(
        settings=AppSettings(llm=LLMSettings(model='fake-model', api_key='nonworking-test-key', max_attempts=2,
                                            max_fragment_blocks=story.get('fragment_blocks', 100), max_unit_source_chars=1_000_000),
                             storage=StorageSettings(data_root=str(root)), worker=WorkerPoolSettings(concurrency=1)),
        llm_client=client, task_id_factory=sequential_ids('task'),
        epubcheck=EpubCheckValidator(settings=EpubCheckSettings(java_executable='missing-content-plan-java')),
        **({'content_plan_profile': 'content-plan-v2'} if candidate else {}),
    )


def import_story(app, story):
    path = app.paths.source_epub(BOOK)
    path.parent.mkdir(parents=True, exist_ok=True)
    book = epub.EpubBook()
    book.set_identifier('independent-content-plan')
    book.set_title('Independent synthetic story')
    book.set_language('en')
    chapter = epub.EpubHtml(title='Story', file_name='story.xhtml', lang='en')
    chapter.content = '<html><body>' + ''.join(f'<p id="source-{i}">{escape(t)}</p>' for i, t in enumerate(story['blocks'])) + '</body></html>'
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.toc = (chapter,)
    book.spine = [chapter]
    epub.write_epub(str(path), book)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    app.register_book(RegisterBookCommand(book_id=BOOK, source_epub_path=str(path), title='Independent synthetic story', source_language='en', target_language='ru'))
    run(app.start_import(BOOK))
    finish(app)
    with app.unit_of_work.transaction() as uow:
        document = uow.documents.get(BOOK)
    sections = tuple(s for s in document.ordered_sections if textual_blocks(s))
    assert len(sections) == 1
    blocks = textual_blocks(sections[0])
    assert [b.text for b in blocks] == story['blocks']
    return document, sections[0], blocks, digest


def offset(text, anchor):
    start = -1
    for _ in range(anchor.get('occurrence', 0) + 1):
        start = text.index(anchor['quote'], start + 1)
    return start


def analysis_payload(story, indices, *, candidate=True):
    aliases = {i: f'b{n}' for n, i in enumerate(indices, 1)}
    difficulties, evidence = [], []
    for number, d in enumerate(story['difficulties']):
        if d['owner'] not in aliases:
            continue
        anchors = []
        for anchor in d['anchors']:
            start = offset(story['blocks'][anchor['block']], anchor)
            anchors.append(dict(block_id=aliases[anchor['block']], start=start, end=start + len(anchor['quote']), quote=anchor['quote']))
        difficulties.append(dict(id=f'd{number}', owner_block_id=aliases[d['owner']], anchors=anchors, kind=d['kind'], priority='critical', help_kind='explanation',
                                 required_points=[dict(id=p['id'], description=p['description']) for p in d['points']]))
        for p in d['points']:
            if p.get('channel') == 'byr':
                evidence.append(dict(difficulty_id=f'd{number}', point_id=p['id'], artifact_kind='before_you_read.explanation', source_block_id=aliases[d['owner']], artifact_index=0, body_excerpt=p['body']))
    glossary = story.get('glossary')
    payload = dict(translation_instructions='Сохранять смысл и принятые пользователем имена.', chapter_summary='Independent synthetic scene.',
                   treatment_plan=[dict(source_block_id=aliases[i], treatment=story['treatments'][i]) for i in indices],
                   glossary_candidates=([dict(source_text=glossary['source'], type='person', suggested_translation=glossary['suggested'], confidence=0.95, model_flagged_review=True, note='Имя героини.')] if glossary else []),
                   before_you_read=story['byr'])
    if candidate:
        payload['content_plan'] = dict(version=2, profile='content-plan-v2', difficulties=difficulties, before_you_read_evidence=evidence)
    return payload


def processing_payload(story, blocks, plan, *, indices=None, regenerated=False, omit_evidence=False):
    indices = list(range(len(blocks))) if indices is None else indices
    aliases = {blocks[i].id: f'b{n}' for n, i in enumerate(indices, 1)}
    response = dict(blocks=[dict(source_block_id=aliases[blocks[i].id], treatment=story['treatments'][i], translated_text=story['translations'][i], glossary_hints=[]) for i in indices], language_notes=[], content_evidence=[])
    for number, expected in enumerate(story['difficulties']):
        owner = blocks[expected['owner']].id
        if owner not in aliases:
            continue
        actual = next(d for d in plan.difficulties if d.owner_block_id == owner and {p.id for p in d.required_points} == {p['id'] for p in expected['points']})
        for point in expected['points']:
            if point.get('channel') == 'byr':
                continue
            body = story['regenerated'] if regenerated else point['body']
            index = len(response['language_notes'])
            response['language_notes'].append(dict(source_block_id=aliases[owner], original_excerpt=expected['anchors'][-1]['quote'], translated_excerpt=None, note=body))
            response['content_evidence'].append(dict(difficulty_id=actual.id, point_id=point['id'], artifact_kind='language_note.note', source_block_id=aliases[owner], artifact_index=index, body_excerpt=body))
    if story['byr'] and not regenerated:
        response['language_notes'].append(dict(source_block_id=aliases[blocks[-1].id], original_excerpt=story['byr'][0]['expression'], translated_excerpt=None, note=story['byr'][0]['explanation']))
    if omit_evidence:
        response['content_evidence'] = []
    return response


@dataclass
class Completed:
    story: dict
    plan: object
    results: tuple
    requests: tuple
    roots: dict
    logs: tuple


def pipeline(tmp_path, name, *, retry=False, regenerate=False):
    story = STORIES[name]
    client = FakeLLMClient()
    app = make_app(tmp_path / 'data', client, story)
    try:
        document, section, blocks, source_sha = import_story(app, story)
        chunk = story.get('fragment_blocks', len(blocks))
        groups = [list(range(i, min(i + chunk, len(blocks)))) for i in range(0, len(blocks), chunk)]
        client.replies = tuple(FakeReply.of_payload(analysis_payload(story, indices)) for indices in groups)
        run(app.analysis.start(BOOK))
        finish(app)
        assert len(client.requests) == len(groups)
        assert all(r.contract_version == 2 for r in client.requests)
        with app.unit_of_work.transaction() as uow:
            analysis = uow.analyses.get_for_section(BOOK, section.id)
        plan = analysis.content_plan
        assert len(plan.difficulties) == len(story['difficulties'])
        for expected in story['difficulties']:
            actual = next(d for d in plan.difficulties if d.owner_block_id == blocks[expected['owner']].id)
            assert {p.id for p in actual.required_points} == {p['id'] for p in expected['points']}
            assert {(a.block_id, a.start, a.quote) for a in actual.anchors} == {(blocks[a['block']].id, offset(story['blocks'][a['block']], a), a['quote']) for a in expected['anchors']}
        review = app.approve_glossary.review(BOOK)
        if 'glossary' in story:
            choice = story['glossary']
            item = review.item_for(choice['source'])
            assert item is not None
            app.approve_glossary(GlossaryDecisionCommand(book_id=BOOK, source_text=item.source_text, concept_id=item.concept_id,
                decision=GlossaryDecision.CORRECT, translation=choice['corrected'], note=choice['note']))
        processing_run = app.start_processing()(StartProcessingCommand(book_id=BOOK))
        if 'glossary' in story:
            entry = next(e for e in processing_run.glossary_snapshot.entries if e.source_text == story['glossary']['source'])
            assert entry.translation == story['glossary']['corrected']
            assert entry.note == story['glossary']['note']
        good = processing_payload(story, blocks, plan)
        replies = [FakeReply.of_payload(good)]
        if retry:
            replies.insert(0, FakeReply.of_payload(processing_payload(story, blocks, plan, omit_evidence=True)))
        # FakeLLMClient consumes a single cursor across stages.
        client.replies = (*client.replies, *replies)
        run(app.processing.start(BOOK))
        finish(app)
        with app.unit_of_work.transaction() as uow:
            before = uow.results.list_for_run(processing_run.id)
            logs = uow.llm_calls.list_for_book(BOOK)
        assert len(before) == len(blocks)
        for result, treatment in zip(before, story['treatments'], strict=True):
            assert result.treatment == treatment
            if treatment == 'TRANSLATE':
                assert result.translated_text
            elif treatment == 'READ_AS_IS':
                assert not result.translated_text and not result.glossary_hints
            assert result.content_evidence.plan_sha256 == plan.sha256
            assert result.content_evidence.semantic_correctness == 'not_evaluated'
        if retry:
            rejected = [log for log in logs if not log.success]
            assert len(rejected) == 1 and rejected[0].error is not None
            assert 'аспект' in rejected[0].error.message
            assert len(client.requests) == len(groups) + 2
        assert all(request.contract_version == 2 for request in client.requests)
        if regenerate:
            owner = story['difficulties'][0]['owner']
            client.replies = (*client.replies, FakeReply.of_payload(processing_payload(story, blocks, plan, indices=[owner], regenerated=True)))
            run(app.regeneration.start(BOOK, blocks[owner].id))
            finish(app)
        with app.unit_of_work.transaction() as uow:
            after = uow.results.list_for_run(processing_run.id)
            assert uow.documents.get(BOOK) == document
            assert uow.analyses.get_for_section(BOOK, section.id) == analysis
            assert uow.runs.active_for_book(BOOK).glossary_snapshot == processing_run.glossary_snapshot
        if regenerate:
            for i, (old, new) in enumerate(zip(before, after, strict=True)):
                if i == owner:
                    assert new.language_notes[0].note == story['regenerated']
                    assert new.content_evidence != old.content_evidence
                else:
                    assert new == old
        assert hashlib.sha256(app.paths.source_epub(BOOK).read_bytes()).hexdigest() == source_sha
        run(app.start_export(BOOK))
        finish(app)
        roots = exported_documents(app.paths.export_epub(BOOK))
        nodes = [node for root in roots.values() for node in root.iter()]
        originals = [node for node in nodes if node.get('id', '').startswith('source-')]
        assert [(node.get('id'), ''.join(node.itertext())) for node in originals] == [
            (f'source-{i}', text) for i, text in enumerate(story['blocks'])]
        visible = '\n'.join(''.join(root.itertext()) for root in roots.values())
        for forbidden in ('content-plan-v2', 'plan_sha256', 'structural_coverage', 'content_evidence', *(d.id for d in plan.difficulties)):
            assert forbidden not in visible
        for expected in story['difficulties']:
            for p in expected['points']:
                body = story['regenerated'] if regenerate and p.get('channel') != 'byr' else p['body']
                assert body in visible
                if p.get('channel') == 'byr':
                    containers = [n for n in nodes if n.tag.rsplit('}', 1)[-1] == 'dd'
                                  and n.getparent().getparent().get('class') == 'linguary-before-you-read']
                else:
                    containers = [n for n in nodes if n.tag.rsplit('}', 1)[-1] == 'p'
                                  and n.getparent().get('class') == 'linguary-language-note'
                                  and n.get('class') != 'linguary-excerpt']
                assert sum(''.join(n.itertext()) == body for n in containers) == 1
                assert all(not any(a.get('hidden') is not None or a.get('aria-hidden') == 'true'
                                   or 'display:none' in a.get('style', '').replace(' ', '')
                                   for a in (n, *n.iterancestors())) for n in containers)
        if regenerate:
            assert story['difficulties'][0]['points'][-1]['body'] not in visible
        if 'glossary' in story:
            assert story['glossary']['corrected'] in visible
            assert story['glossary']['suggested'] not in visible
            assert any(story['glossary']['note'] in request.input_text for request in client.requests[len(groups):])
        return Completed(story, plan, after, tuple(client.requests), roots, tuple(logs))
    finally:
        app.dispose()


def test_multiblock_wordplay_survives_dedup_and_epub(tmp_path):
    completed = pipeline(tmp_path, 'ledger', retry=True, regenerate=True)
    assert len(completed.plan.difficulties[0].anchors) == 2
    assert sum(len(r.language_notes) for r in completed.results) == 1


def test_grammar_and_cultural_help_without_lexical_flood(tmp_path):
    completed = pipeline(tmp_path, 'lantern')
    assert {d.kind for d in completed.plan.difficulties} == {'grammar', 'cultural_reference'}
    assert sum(len(r.language_notes) for r in completed.results) == 2
    assert all(not r.glossary_hints for r in completed.results)


def test_simple_story_accepts_empty_plan(tmp_path):
    completed = pipeline(tmp_path, 'window')
    assert not completed.plan.difficulties
    assert all(not r.translated_text and not r.language_notes and not r.glossary_hints for r in completed.results)


def test_long_chapter_alias_merge_and_repeated_quote(tmp_path):
    completed = pipeline(tmp_path, 'echo')
    analyses = completed.requests[:2]
    assert all('[b1]' in r.input_text and '[b2]' in r.input_text for r in analyses)
    assert STORIES['echo']['blocks'][2] not in analyses[0].input_text
    assert STORIES['echo']['blocks'][0] not in analyses[1].input_text
    assert len({d.id for d in completed.plan.difficulties}) == 2


def test_user_glossary_and_original_unchanged(tmp_path):
    pipeline(tmp_path, 'lantern')


def test_candidate_is_opt_in_and_network_free(tmp_path):
    story = STORIES['window']
    for candidate in (False, True):
        client = FakeLLMClient()
        app = make_app(tmp_path / str(candidate), client, story, candidate=candidate)
        try:
            _, section, blocks, _ = import_story(app, story)
            client.replies = (FakeReply.of_payload(analysis_payload(story, list(range(len(blocks))), candidate=candidate)),)
            run(app.analysis.start(BOOK))
            finish(app)
            assert client.requests[0].contract_version == (2 if candidate else 1)
            with app.unit_of_work.transaction() as uow:
                analysis = uow.analyses.get_for_section(BOOK, section.id)
            assert (analysis.content_plan is not None) == candidate
        finally:
            app.dispose()
