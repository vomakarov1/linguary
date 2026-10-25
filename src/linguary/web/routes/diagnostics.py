"""Diagnostic bodies require a real local peer, independently of public host settings."""
from ipaddress import ip_address
from urllib.parse import urlsplit
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from linguary.web.dependencies import CurrentApplication

_HEADERS = {'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'}


def _loopback(value):
    try:
        address = ip_address(value)
        return address.is_loopback or bool(getattr(address, 'ipv4_mapped', None) and address.ipv4_mapped.is_loopback)
    except ValueError:
        return False


def require_local(request: Request):
    """Never trusts proxy-supplied peers, even when the server listens on all interfaces."""
    allowed = request.client is not None and _loopback(request.client.host)
    allowed = allowed and not any(name.lower() == 'forwarded' or name.lower().startswith('x-forwarded-')
        for name in request.headers)
    host = request.headers.get('host', '').lower()
    try:
        authority = urlsplit('//' + host)
        authority.port  # Reject malformed ports/IPv6 authorities.
        allowed = allowed and not authority.username and not authority.password and not authority.path
        allowed = allowed and not authority.query and not authority.fragment and '@' not in host
        allowed = allowed and (authority.hostname == 'localhost' or _loopback(authority.hostname or ''))
    except ValueError:
        allowed = False
    origin = request.headers.get('origin')
    if origin is not None and origin.strip().lower() != f'{request.url.scheme}://{host}':
        allowed = False
    fetch_site = request.headers.get('sec-fetch-site')
    if fetch_site is not None and fetch_site.strip().lower() not in ('same-origin', 'none'):
        allowed = False
    if not allowed:
        raise HTTPException(403, detail={'reason_code': 'local_connection_required'}, headers=_HEADERS)


router = APIRouter(tags=['diagnostics'], dependencies=[Depends(require_local)])


@router.get('/books/{book_id}/diagnostics/{call_id}')
def get_diagnostic(request: Request, application: CurrentApplication, book_id: str, call_id: str):
    value = application.diagnostics.read(book_id, call_id)
    if request.headers.get('accept', '').startswith('text/plain'):
        return PlainTextResponse(value['body'] or '', headers={**_HEADERS, 'X-Linguary-Body-Status': value['body_status']})
    return JSONResponse(value, headers=_HEADERS)


@router.delete('/books/{book_id}/diagnostics/{call_id}', status_code=204)
def delete_diagnostic(application: CurrentApplication, book_id: str, call_id: str):
    application.diagnostics.purge(book_id, call_id=call_id)
    return Response(status_code=204, headers=_HEADERS)
