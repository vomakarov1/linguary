import json
import pytest
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.adapters.llm.payloads import parse_payload, schema_for
from linguary.application.ports.llm import LLMRequest, LLMResponseKind, LLMResponseFormatError


async def test_fake_review_response_validation():
    payload = dict(findings=[], checked_artifacts=['a' * 64], missing_artifacts=[])
    client = FakeLLMClient([FakeReply.of_payload(payload)])
    result = await client.generate(LLMRequest(kind=LLMResponseKind.QUALITY_REVIEW, instructions='review', input_text='{}'))
    assert result.payload.checked_artifacts == ('a' * 64,)
    assert schema_for(LLMResponseKind.QUALITY_REVIEW).model_json_schema()['additionalProperties'] is False


@pytest.mark.parametrize('payload', [dict(findings=[], checked_artifacts=[]),
    dict(findings=[], checked_artifacts=[], missing_artifacts=[], extra=True),
    dict(findings=[dict(issue='unbound')], checked_artifacts=[], missing_artifacts=[])])
def test_strict_schema_rejects_incomplete_or_unbound_findings(payload):
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.QUALITY_REVIEW, json.dumps(payload))
