"""Export accepts a finished successor and still rejects a live unfinished root."""

from types import SimpleNamespace

import pytest

from linguary.application.use_cases.export_book import ExportBookError, _ensure_recovery_complete


class _Work:
    def __init__(self, roots, complete_ids):
        self._roots = roots
        self._complete = set(complete_ids)
        self.checked = []

    def list_roots(self, book_id, run_id=None):
        return self._roots

    def root_coverage_complete(self, root_id, update_state=False):
        self.checked.append(root_id)
        return root_id in self._complete


class _Tx:
    def __init__(self, work):
        self.work = work


def test_superseded_root_does_not_block_export(monkeypatch):
    import linguary.application.correction as correction

    monkeypatch.setattr(correction, "correction_coverage_complete", lambda *_args, **_kwargs: True)
    old = SimpleNamespace(id="old", superseded_by="new")
    current = SimpleNamespace(id="new", superseded_by=None)
    work = _Work((old, current), {"new"})

    _ensure_recovery_complete(_Tx(work), "book", "run")

    assert work.checked == ["new"]


def test_live_incomplete_root_still_blocks_export(monkeypatch):
    import linguary.application.correction as correction

    monkeypatch.setattr(correction, "correction_coverage_complete", lambda *_args, **_kwargs: True)
    current = SimpleNamespace(id="current", superseded_by=None)
    work = _Work((current,), set())

    with pytest.raises(ExportBookError, match="восстановление книги не завершено"):
        _ensure_recovery_complete(_Tx(work), "book", "run")
