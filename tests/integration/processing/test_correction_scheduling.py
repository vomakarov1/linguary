"""Correction recovery scans the library at startup, not while polling an empty queue."""

from linguary.adapters.db.repositories import SqlAlchemyBookRepository
from test_book_conditions import make_app, store_imported


async def test_empty_queue_polling_does_not_rescan_library(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    scans = []
    list_all = SqlAlchemyBookRepository.list_all

    def record_library_scan(repository):
        books = list_all(repository)
        scans.append(tuple(book.id for book in books))
        return books

    monkeypatch.setattr(SqlAlchemyBookRepository, 'list_all', record_library_scan)
    try:
        store_imported(app)
        await app.runner.recover_interrupted()
        assert ('book-1',) in scans
        scans.clear()

        assert await app.runner.run_next() is None
        assert await app.runner.run_next() is None
        assert scans == []
    finally:
        await app.aclose()
