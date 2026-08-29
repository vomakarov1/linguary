"""Endpoints книг: список, загрузка, карточка, план обработки и предупреждения (API-01).

Routes проверяют то, что пришло снаружи, вызывают прикладные сценарии и превращают их
результат в ответ. Ни состояний книги, ни правил прогресса здесь нет: карточку и сводку
плана считает `linguary.application.book_overview`, а действия выполняют сценарии
(ADR § 3).

Отказы сценариев в коды HTTP переводит единый набор обработчиков ошибок приложения
(`linguary.web.app`), поэтому здесь нет ни одного `try` вокруг вызова сценария.
"""

from __future__ import annotations

import asyncio
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, Query, UploadFile, status
from fastapi.responses import RedirectResponse

from linguary.application.book_overview import (
    book_overview,
    book_warnings,
    list_books,
    processing_plan,
)
from linguary.application.use_cases.register_book import RegisterBookCommand
from linguary.domain import BookStatus
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import (
    BookDetailResponse,
    BookListResponse,
    BookWarningsResponse,
    ProcessingPlanResponse,
)
from linguary.web.uploads import (
    book_title_of_filename,
    ensure_epub_filename,
    save_upload,
)

router = APIRouter(tags=["books"])

#: Код языка книги: коротая строка вроде `en` или `ru` (PRD § 21.2). Пара языков в MVP
#: одна, но интерфейс её не зашивает.
_LanguageCode = Annotated[str, Form(min_length=2, max_length=32)]


@router.get("/books", response_model=BookListResponse)
def get_books(
    application: CurrentApplication,
    book_status: Annotated[
        BookStatus | None,
        Query(alias="status", description="Показать только книги в этом состоянии"),
    ] = None,
) -> BookListResponse:
    """Список книг и их состояний (PRD § 21.1)."""
    return BookListResponse.of_books(
        list_books(application.unit_of_work, status=book_status)
    )


@router.post(
    "/books/upload",
    status_code=status.HTTP_303_SEE_OTHER,
    response_class=RedirectResponse,
)
async def upload_book(
    application: CurrentApplication,
    file: Annotated[UploadFile, File(description="Файл книги в формате EPUB")],
    source_language: _LanguageCode,
    target_language: _LanguageCode,
) -> RedirectResponse:
    """Принимает EPUB, заводит книгу и ставит задачу импорта (PRD § 18, § 21.2).

    Книга появляется в списке сразу, в состоянии «Импортируется»: разбор файла идёт
    фоновой задачей, и держать пользователя в ожидании незачем.
    """
    ensure_epub_filename(file.filename)
    book_id = str(uuid4())
    source_path = application.paths.source_epub(book_id)
    await save_upload(file, target=source_path)
    try:
        await asyncio.to_thread(
            application.register_book,
            RegisterBookCommand(
                book_id=book_id,
                source_epub_path=str(source_path),
                title=book_title_of_filename(file.filename),
                source_language=source_language.strip(),
                target_language=target_language.strip(),
            ),
        )
    except BaseException:
        source_path.unlink(missing_ok=True)
        raise
    await application.start_import(book_id)
    return RedirectResponse(
        url=f"/books/{book_id}", status_code=status.HTTP_303_SEE_OTHER
    )


@router.get("/books/{book_id}", response_model=BookDetailResponse)
def get_book(application: CurrentApplication, book_id: str) -> BookDetailResponse:
    """Карточка книги: состояние, прогресс, счётчики режимов и ошибки (PRD § 21.4)."""
    return BookDetailResponse.of_overview(
        book_overview(application.unit_of_work, book_id)
    )


@router.get("/books/{book_id}/plan", response_model=ProcessingPlanResponse)
def get_book_plan(
    application: CurrentApplication, book_id: str
) -> ProcessingPlanResponse:
    """Сводка плана обработки книги по главам (PRD § 21, UX notes)."""
    return ProcessingPlanResponse.of_plan(
        processing_plan(application.unit_of_work, book_id)
    )


@router.get("/books/{book_id}/warnings", response_model=BookWarningsResponse)
def get_book_warnings(
    application: CurrentApplication, book_id: str
) -> BookWarningsResponse:
    """Предупреждения импорта и замечания к результату обработки (PRD § 21.6)."""
    return BookWarningsResponse.of_warnings(
        book_warnings(application.unit_of_work, book_id)
    )


__all__ = ["router"]
