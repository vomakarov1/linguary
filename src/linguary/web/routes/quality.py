"""Explicit optional review and recovery budget actions shared by JSON and forms."""
import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import ValidationError
from linguary.application.book_overview import BookNotFoundError
from linguary.application.use_cases.quality_review import StartQualityReview
from linguary.domain.book_conditions import WorkStateError
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import StartQualityReviewRequest, ExtendRecoveryLimitRequest, QualityReviewStateResponse
from linguary.web.views import wants_html

router = APIRouter(tags=['quality'])
_CONFLICTS = {'request_key_conflict', 'manifest_changed', 'route_unavailable', 'price_unavailable',
    'budget', 'unknown', 'generation_conflict', 'state_conflict', 'conditions_unavailable'}


def require_book(application, book_id):
    with application.unit_of_work.transaction() as tx:
        if tx.books.get(book_id) is None:
            raise BookNotFoundError('книга не найдена')


async def read_input(request, schema):
    try:
        if request.headers.get('content-type', '').startswith('application/json'):
            raw = await request.json()
        else:
            form = await request.form()
            raw = dict(form)
            for key in ('budget_microusd', 'additional_calls', 'expected_generation'):
                if key in raw:
                    raw[key] = int(raw[key])
            if 'confirmed' in raw:
                raw['confirmed'] = raw['confirmed'] in ('true', 'on', '1')
            if 'scope' in raw and raw['scope'] != 'all':
                raw['scope'] = json.loads(raw['scope'])
        return schema.model_validate(raw)
    except (ValidationError, ValueError, TypeError):
        raise HTTPException(422, detail={'reason_code': 'invalid_request'}) from None


def operation_error(error):
    reason = str(error)
    if reason == 'invalid_limit':
        return HTTPException(422, detail={'reason_code': 'invalid_limit'})
    if reason == 'неверный охват вычитки':
        return HTTPException(422, detail={'reason_code': 'invalid_scope'})
    if reason in ('вычитка не найдена', 'root_not_found'):
        return HTTPException(404, detail={'reason_code': 'not_found'})
    return HTTPException(409, detail={'reason_code': reason if reason in _CONFLICTS else 'state_conflict'})


def review_response(application, book_id, review_id=None):
    require_book(application, book_id)
    try:
        value = application.quality_review.get(book_id, review_id)
    except WorkStateError as error:
        raise operation_error(error) from None
    value['available_routes'] = [dict(route_id=route.route_id, model=route.model)
        for _, route in sorted(application.quality_review.routes.items())]
    try:
        value['current_manifest_sha'] = application.quality_review.preview(book_id)['manifest_sha']
    except WorkStateError:
        value['current_manifest_sha'] = None
    return QualityReviewStateResponse.model_validate(value)


@router.get('/books/{book_id}/quality-review', response_model=QualityReviewStateResponse)
def get_review(request: Request, application: CurrentApplication, book_id: str, review_id: str | None = None):
    return review_response(application, book_id, review_id)


@router.post('/books/{book_id}/quality-review', response_model=QualityReviewStateResponse, status_code=202)
async def start_review(request: Request, application: CurrentApplication, book_id: str):
    require_book(application, book_id)
    value = await read_input(request, StartQualityReviewRequest)
    command = StartQualityReview(book_id=book_id, **value.model_dump(exclude={'currency'}))
    try:
        created = await application.quality_review.start(command, require_available=True)
    except WorkStateError as error:
        raise operation_error(error) from None
    if wants_html(request):
        return RedirectResponse(f'/books/{book_id}', status_code=303)
    return review_response(application, book_id, created['review_id'])


@router.post('/books/{book_id}/recovery/{root_id}/limit')
async def extend_limit(request: Request, application: CurrentApplication, book_id: str, root_id: str):
    require_book(application, book_id)
    value = await read_input(request, ExtendRecoveryLimitRequest)
    try:
        with application.unit_of_work.transaction() as tx:
            root = tx.work.get_root(root_id)
            if root is None or root.book_id != book_id:
                raise HTTPException(404, detail={'reason_code': 'not_found'})
            root = tx.work.extend_recovery_limit(root_id, **value.model_dump(exclude={'confirmed'}))
    except WorkStateError as error:
        raise operation_error(error) from None
    if wants_html(request):
        return RedirectResponse(f'/books/{book_id}', status_code=303)
    return dict(root_id=root.id, generation=root.generation, calls_used=root.calls_used,
        limit=root.manifest['policy']['max_calls'] + root.extra_calls)
