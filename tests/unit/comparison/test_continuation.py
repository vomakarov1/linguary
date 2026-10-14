from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from linguary.application.ports.llm import LLMRequest
from linguary.comparison.llm import ObservationLedger
from linguary.comparison.protocol import draft_protocol, stable_hash
from linguary.comparison.runner import _finalize


@pytest.mark.parametrize('missing_log', [False, True])
def test_response_verdict_does_not_depend_on_failed_chapter(missing_log):
    ledger = ObservationLedger()
    request = LLMRequest.for_translation(instructions='translate', input_text='group')
    good = ledger.begin(request, model='model', route='route')
    good.call_id = 'good'
    bad = ledger.begin(request, model='model', route='route')
    bad.call_id = 'bad'
    logs = [SimpleNamespace(id='bad', success=False, task_id='chapter')]
    if not missing_log:
        logs.append(SimpleNamespace(id='good', success=True, task_id='chapter'))
    @contextmanager
    def transaction():
        yield SimpleNamespace(llm_calls=SimpleNamespace(list_for_book=lambda _: logs),
            tasks=SimpleNamespace(list_for_book=lambda _: [SimpleNamespace(id='chapter', status=SimpleNamespace(value='failed'))]))
    application = SimpleNamespace(unit_of_work=SimpleNamespace(transaction=transaction))
    _finalize(application, ledger, ['book'])
    assert bad.accepted is False
    assert good.accepted is (None if missing_log else True)
    assert ledger.violations == (['unknown_final_status'] if missing_log else [])


def test_legacy_protocol_hash_remains_unchanged_without_continuation():
    protocol = draft_protocol()
    old_document = protocol.model_dump(mode='json')
    old_document.pop('continuation', None)
    assert protocol.content_hash == stable_hash(old_document)
    assert protocol.content_hash == 'c71051d6ba5da5fb90d27f2a395cc3775c7bea91468e9afceaf40e52f7038f7a'
