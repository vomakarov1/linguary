"""Final bodies and readiness are checked through the real EPUB/export boundary."""
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4
import zipfile

from lxml import etree
import pytest
from web_scenario import store_book, start_run, translated_result

from linguary.adapters.epub.writer import EpubExportWriter, ExportCheckFailure, ExportCheckKind
from linguary.application.ports.epub import EpubWriteRequest, EpubExportFatalError
from linguary.application.use_cases.export_book import ExportBookCommand
from linguary.domain import BeforeYouReadEntry, BookStatus, LearningMaterial
from linguary.domain.content_plan import ContentPlan, ContentEvidence, Difficulty, RequiredPoint, SourceAnchor, HelpEvidence, artifact_key
from linguary.domain.quality_review import QualityReview


BODY = 'Языковая справка: выражение truth (в переводе «истина») обозначает правду.'


def ready(app, version):
    scenario = store_book(app, status=BookStatus.READY_TO_EXPORT)
    owner = scenario.block_ids()[0]
    entry = BeforeYouReadEntry('truth', BODY)
    plan = None
    if version > 1:
        source = scenario.section().blocks[0].text
        start = source.index('truth')
        difficulty = Difficulty('truth', (SourceAnchor(owner, start, start + 5, 'truth'),), owner,
            'vocabulary', 'critical', 'explanation', (RequiredPoint('meaning', 'Explain the word'),))
        evidence = HelpEvidence('truth', 'meaning', 'before_you_read.explanation', owner,
            artifact_key('before_you_read.explanation', owner, dict(expression=entry.expression, explanation=BODY)), BODY)
        plan = ContentPlan(version, f'content-plan-v{version}', (difficulty,), (evidence,))
    with app.unit_of_work.transaction() as tx:
        analysis = tx.analyses.get_for_section(scenario.book_id, scenario.section().id)
        tx.analyses.save(scenario.book_id, replace(analysis, before_you_read=(entry,), content_plan=plan))
    results = [translated_result(b) for b in scenario.block_ids()]
    if plan:
        results = [replace(r, content_evidence=ContentEvidence(plan.sha256,
            plan.before_you_read_evidence if r.block_id == owner else (), version=version)) for r in results]
    start_run(app, scenario, results=results)
    return scenario, plan, tuple(results)


def roots(path):
    with zipfile.ZipFile(path) as archive:
        return [etree.fromstring(archive.read(name)) for name in archive.namelist() if name.endswith('.xhtml')]


@pytest.mark.parametrize('version', [1, 2])
def test_v1_v2_old_epub_export(application, tmp_path, version):
    scenario, _, _ = ready(application, version)
    path = tmp_path / 'old.epub'
    application.export_book(ExportBookCommand(scenario.book_id, str(path)))
    visible = '\n'.join(''.join(root.itertext()) for root in roots(path))
    assert 'выражение truth обозначает правду' in visible
    assert '(в переводе' not in visible


def test_v3_export_body_equals_evidence_body(application, tmp_path):
    scenario, plan, before = ready(application, 3)
    path = tmp_path / 'v3.epub'
    application.export_book(ExportBookCommand(scenario.book_id, str(path)))
    bodies = [node.text for root in roots(path) for node in root.iter() if node.tag.endswith('}dd')]
    assert bodies == [plan.before_you_read_evidence[0].body_excerpt]
    with application.unit_of_work.transaction() as tx:
        assert tx.results.list_for_run(scenario.run_id) == before


@pytest.mark.parametrize('change', ['evidence', 'body', 'source'])
def test_v3_export_rechecks_final_bodies_before_queue(client, application, change):
    scenario, _, before = ready(application, 3)
    with application.unit_of_work.transaction() as tx:
        if change == 'evidence':
            tx.results.replace_results(scenario.run_id, (replace(before[0], content_evidence=()),))
        elif change == 'body':
            tx.results.replace_results(scenario.run_id, (replace(before[0], translated_text='в b17'),))
        else:
            analysis = tx.analyses.get_for_section(scenario.book_id, scenario.section().id)
            wrong = replace(analysis.content_plan.difficulties[0].anchors[0], quote='false')
            difficulty = replace(analysis.content_plan.difficulties[0], anchors=(wrong,))
            tx.analyses.save(scenario.book_id, replace(analysis, content_plan=replace(analysis.content_plan, difficulties=(difficulty,))))
    assert client.post(f'/books/{scenario.book_id}/export').status_code == 409
    assert not application.paths.export_epub(scenario.book_id).exists()


def test_incomplete_recovery_export_409(configured_client, configured_application):
    from test_quality_api import recovery_fixture
    client, application = configured_client, configured_application
    scenario, _ = recovery_fixture(client, application)
    with application.unit_of_work.transaction() as tx:
        tx.books.save(replace(tx.books.get(scenario.book_id), status=BookStatus.READY_TO_EXPORT))
        tx.results.replace_results(scenario.run_id, [translated_result(b) for b in scenario.block_ids()])
    assert client.post(f'/books/{scenario.book_id}/export').status_code == 409
    assert not application.paths.export_epub(scenario.book_id).exists()


def test_review_findings_do_not_block_epub(application, tmp_path):
    scenario, _, _ = ready(application, 3)
    with application.unit_of_work.transaction() as tx:
        tx.work.add_review(QualityReview(str(uuid4()), scenario.book_id, 'explicit', {'schema_version': 1},
            {'schema_version': 1}, 10, state='completed', findings=({'issue': 'Check meaning'},)))
    path = tmp_path / 'findings.epub'
    application.export_book(ExportBookCommand(scenario.book_id, str(path)))
    assert path.is_file()


def test_failed_assembly_leaves_no_partial_file(application, tmp_path, monkeypatch):
    import linguary.adapters.epub.writer as writer
    scenario, _, _ = ready(application, 3)
    path = tmp_path / 'failed.epub'
    monkeypatch.setattr(writer, 'verify_exported_epub', lambda *a, **k: (ExportCheckFailure(ExportCheckKind.CONTAINER, 'synthetic failure'),))
    with pytest.raises(EpubExportFatalError):
        application.export_book(ExportBookCommand(scenario.book_id, str(path)))
    assert not path.exists() and not path.with_name(path.name + '.part').exists()


def test_source_text_quotes_footnotes_resources_unchanged(tmp_path):
    from linguary.adapters.epub import EbookLibEpubReader, FilesystemEpubResourceStore
    from linguary.adapters.epub.writer import verify_exported_epub
    from linguary.application.ports.epub import EpubReadRequest, StoreResourcesRequest
    source = Path(__file__).resolve().parents[2] / 'fixtures/epub/synthetic_story.epub'
    document = EbookLibEpubReader().read(EpubReadRequest('preserved', str(source), 'ru')).document
    stored = FilesystemEpubResourceStore(root=tmp_path / 'resources').store(
        StoreResourcesRequest('preserved', str(source), document))
    writer = EpubExportWriter()
    baseline, verified = tmp_path / 'original.epub', tmp_path / 'verified.epub'
    request = EpubWriteRequest(document, str(baseline), stored.paths, str(source))
    writer.write(request)
    writer.write(replace(request, target_path=str(verified), verified_body_sections=frozenset(s.id for s in document.sections),
        learning_materials={s.id: LearningMaterial(s.id, before_you_read=(BeforeYouReadEntry('truth', BODY),))
            for s in document.sections}))
    actual_roots = roots(verified)
    for root in actual_roots:
        for node in root.xpath('//*[@class="linguary-before-you-read"]'):
            node.getparent().remove(node)
    assert [etree.tostring(root) for root in actual_roots] == [etree.tostring(root) for root in roots(baseline)]
    assert verify_exported_epub(verified) == ()
    nodes = [n for root in actual_roots for n in root.iter()]
    assert any('“You held it wrapped in wool,”' in ''.join(n.itertext()) for n in nodes)
    assert any(n.get('id') == 'fn1' for n in nodes)
    assert any(n.get('href', '').endswith('#fn1') for n in nodes)
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(verified) as result:
        for name in ('images/marker.png', 'style.css'):
            saved = next(n for n in result.namelist() if n.endswith('/' + name))
            assert result.read(saved) == original.read('OEBPS/' + name)


def test_write_error_removes_draft_and_preserves_previous_export(application, tmp_path, monkeypatch):
    import linguary.adapters.epub.writer as writer
    scenario, _, _ = ready(application, 3)
    path = tmp_path / 'previous.epub'
    path.write_bytes(b'previous successful export')
    def failed_write(draft, **kwargs):
        draft.write_bytes(b'partial archive')
        raise OSError('synthetic disk failure')
    monkeypatch.setattr(writer, '_write_archive', failed_write)
    with pytest.raises(OSError):
        application.export_book(ExportBookCommand(scenario.book_id, str(path)))
    assert path.read_bytes() == b'previous successful export'
    assert not path.with_name(path.name + '.part').exists()


def test_archived_results_do_not_complete_current_run(client, application):
    scenario, _, results = ready(application, 3)
    with application.unit_of_work.transaction() as tx:
        old = tx.runs.active_for_book(scenario.book_id)
        archives = tx.work.archive_artifacts([dict(book_id=scenario.book_id, conditions_id=None,
            kind='result', scope_id=r.block_id, body={'schema_version': 1, 'result': asdict(r)}) for r in results])
        tx.runs.delete(old.id)
        tx.runs.add(replace(old, id='current-partial-run'))
        tx.results.replace_results('current-partial-run', results[:1])
        assert all(tx.work.get_artifact(artifact) is not None for artifact in archives)
    assert client.post(f'/books/{scenario.book_id}/export').status_code == 409
    assert client.get(f'/books/{scenario.book_id}/download').status_code == 409
    assert not application.paths.export_epub(scenario.book_id).exists()
