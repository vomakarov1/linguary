"""Explicit optional review and recovery budget actions shared by JSON and forms."""
import json
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from shlex import quote
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import ValidationError
from linguary.application.book_overview import BookNotFoundError
from linguary.application.use_cases.quality_review import StartQualityReview
from linguary.domain.book_conditions import WorkStateError
from linguary.web.dependencies import CurrentApplication
from linguary.web.schemas import StartQualityReviewRequest, ExtendRecoveryLimitRequest, QualityReviewStateResponse
from linguary.web.views import wants_html, render_error

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
            if schema is StartQualityReviewRequest and 'budget_usd' in raw:
                try:
                    entered = raw.pop('budget_usd')
                    if not isinstance(entered, str):
                        raise ValueError
                    entered = entered.strip()
                    if len(entered) > 32 or re.fullmatch(r'[0-9]+(?:[.,][0-9]{1,6})?', entered) is None:
                        raise ValueError
                    amount = Decimal(entered.replace(',', '.'))
                    if not amount.is_finite() or amount <= 0 or 'budget_microusd' in raw:
                        raise ValueError
                    exact = Fraction(amount) * 1000000
                    if exact.denominator != 1 or exact.numerator > 9223372036854775807:
                        raise ValueError
                    raw['budget_microusd'] = exact.numerator
                except (ValueError, InvalidOperation):
                    raise HTTPException(422, detail='Введите положительную сумму USD с точностью не более шести знаков после запятой') from None
            if schema is StartQualityReviewRequest and 'route_selection' in raw:
                if 'model' in raw or 'route_id' in raw:
                    raise ValueError
                selection = json.loads(raw.pop('route_selection'))
                if not isinstance(selection, list) or len(selection) != 2:
                    raise ValueError
                raw['route_id'], raw['model'] = selection
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
    try:
        value = await read_input(request, StartQualityReviewRequest)
        command = StartQualityReview(book_id=book_id, **value.model_dump(exclude={'currency'}))
        created = await application.quality_review.start(command, require_available=True)
    except WorkStateError as error:
        return input_error_response(request, operation_error(error))
    except HTTPException as error:
        return input_error_response(request, error)
    if wants_html(request):
        return RedirectResponse(f'/books/{book_id}', status_code=303)
    return review_response(application, book_id, created['review_id'])


@router.post('/books/{book_id}/recovery/{root_id}/limit')
async def extend_limit(request: Request, application: CurrentApplication, book_id: str, root_id: str):
    require_book(application, book_id)
    try:
        value = await read_input(request, ExtendRecoveryLimitRequest)
        with application.unit_of_work.transaction() as tx:
            root = tx.work.get_root(root_id)
            if root is None or root.book_id != book_id:
                raise HTTPException(404, detail={'reason_code': 'not_found'})
            root = tx.work.extend_recovery_limit(root_id, **value.model_dump(exclude={'confirmed'}))
    except WorkStateError as error:
        return input_error_response(request, operation_error(error))
    except HTTPException as error:
        return input_error_response(request, error)
    if wants_html(request):
        return RedirectResponse(f'/books/{book_id}', status_code=303)
    return dict(root_id=root.id, generation=root.generation, calls_used=root.calls_used,
        limit=root.manifest['policy']['max_calls'] + root.extra_calls)


def quality_page_context(request, application, book_id):
    """Read-only form inputs and local estimates; no clients, calls, tasks, or bodies."""
    from linguary.application.ports.llm import LLMRequest, LLMResponseKind
    from linguary.application.quality_context import INSTRUCTIONS, request_text
    from linguary.web.routes.diagnostics import require_local
    from linguary.web import labels
    base = dict(forms=[], scope_choices=[], settings_revision=application.runtime_settings.revision, legacy=False,
        llm_ready=application.settings.llm.is_ready,
        review_labels=labels.QUALITY_REVIEW_LABELS, reason_labels=labels.QUALITY_REASON_LABELS, usd=usd_amount,
        retry_tasks=[], diagnostics=[], diagnostic_list_command=f'linguary diagnostics list {quote(book_id)}')
    if request.headers.get('x-linguary-fragment') == 'result-header':
        return base
    with application.unit_of_work.transaction() as tx:
        document = tx.documents.get(book_id)
        calls = tx.llm_calls.list_for_book(book_id)
        tasks = tx.tasks.list_for_book(book_id)
        legacy = tx.work.latest_conditions(book_id) is None and bool(tx.analyses.list_for_book(book_id) or tx.runs.active_for_book(book_id))
    try:
        require_local(request)
        local_diagnostics = True
    except HTTPException:
        local_diagnostics = False
    scopes = [('all', 'all', 'Вся книга: все доступные исходные и учебные слои')]
    if document:
        from linguary.application.context_builder import textual_blocks
        scopes.extend((section.id, tuple(block.id for block in textual_blocks(section)), f'Глава {number}: {section.title}')
            for number, section in enumerate(document.ordered_sections, 1) if textual_blocks(section))
    selected = next((item for item in scopes if item[0] == request.query_params.get('review_scope')), scopes[0])
    _, scope, title = selected
    forms = []
    try:
        manifest = application.quality_review.preview(book_id, scope)
    except WorkStateError:
        manifest = None
    if manifest is not None:
        routes = []
        for _, route in sorted(application.quality_review.routes.items()):
            try:
                amount = route.estimate(LLMRequest(kind=LLMResponseKind.QUALITY_REVIEW, instructions=INSTRUCTIONS,
                    input_text=request_text(manifest), max_completion_tokens=route.max_completion_tokens))
                estimate = usd_amount(amount)
            except Exception:
                estimate = None
            routes.append(dict(model=route.model, route_id=route.route_id,
                selection=json.dumps([route.route_id, route.model]), estimate_usd=estimate))
        forms.append(dict(title=title, scope='all' if scope == 'all' else json.dumps(scope),
            manifest_sha=manifest['manifest_sha'], request_key=str(uuid4()), routes=routes))
    diagnostics = []
    for call in calls:
        meta = call.diagnostic_meta or {}
        if meta.get('decision') not in ('rejected', 'unknown', 'stale'):
            continue
        status = (meta.get('body_meta') or {}).get('body_status', 'unavailable')
        diagnostics.append(dict(status=labels.DIAGNOSTIC_BODY_LABELS.get(status, labels.DIAGNOSTIC_BODY_LABELS['unavailable']),
            url=f'/books/{book_id}/diagnostics/{call.id}' if local_diagnostics else None,
            command=f'linguary diagnostics show {quote(book_id)} {quote(call.id)}'))
    return dict(base, forms=forms, scope_choices=[dict(value=key, title=title, selected=key == selected[0]) for key, _, title in scopes],
        legacy=legacy, retry_tasks=[dict(id=task.id, label=labels.task_type_label(task.task_type)) for task in tasks
            if task.is_retryable and task.task_type.value in ('process_translation_unit', 'regenerate_block')],
        diagnostics=diagnostics)


def input_error_response(request, error):
    if not wants_html(request):
        raise error
    from linguary.web import labels
    if isinstance(error.detail, dict):
        message = labels.QUALITY_OPERATION_ERRORS.get(error.detail.get('reason_code'), 'Действие недоступно. Обновите страницу книги.')
    else:
        message = str(error.detail)
    return render_error(request, status_code=error.status_code, messages=(message,))


def usd_amount(microusd):
    return f'{microusd // 1000000}.{microusd % 1000000:06d}'
