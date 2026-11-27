"""Unit tests for flat strict illustration LLM schemas, wire bytes, and Diagnostics isolation (`P2.T2`)."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest

from linguary.adapters.config import LLMSettings
from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm import FakeLLMClient, FakeReply, LoggingLLMClient
from linguary.adapters.llm.illustration_payloads import ILLUSTRATION_SCHEMA_CLASSES
from linguary.adapters.llm.json_schema import strict_json_schema
from linguary.adapters.llm.openai_compatible import (
    OpenAICompatibleLLMClient,
    serialize_request,
)
from linguary.adapters.llm.payloads import parse_payload, schema_for
from linguary.application.diagnostics import Diagnostics, DiagnosticsPolicy, diagnostics_for
from linguary.application.ports.llm import (
    LLMRequest,
    LLMRequestTooLargeError,
    LLMResponseFormatError,
    LLMResponseKind,
)
from linguary.application.ports.persistence import BookRecord
from linguary.application.retry_policy import classify_error
from linguary.comparison.llm import BudgetGuard, ObservationLedger, ObservedLLMClient
from linguary.comparison.protocol import CostBounds, RouteMetadata
from linguary.composition.migrations import apply_migrations
from linguary.domain import BookMetadata, BookStatus, ErrorType, LLMCallStage
from linguary.domain.llm_log import CallContext

_ALL_ILLUSTRATION_KINDS = (
    LLMResponseKind.ILLUSTRATION_EXTRACT,
    LLMResponseKind.ILLUSTRATION_ALIAS,
    LLMResponseKind.ILLUSTRATION_DESIGN,
    LLMResponseKind.ILLUSTRATION_FACT_SCOPE,
    LLMResponseKind.ILLUSTRATION_CANDIDATES,
    LLMResponseKind.ILLUSTRATION_SELECTION,
    LLMResponseKind.ILLUSTRATION_PROMPT,
)


def _valid_json_for(kind: LLMResponseKind) -> dict[str, Any]:
    if kind is LLMResponseKind.ILLUSTRATION_EXTRACT:
        return {
            "contract_version": 1,
            "mentions": [
                {
                    "mention_id": "m1",
                    "name": "Alice",
                    "entity_kind": "person",
                    "alias_of_hint": None,
                    "quote_block_id": "b1",
                    "quote": "  Alice stepped  ",
                }
            ],
            "facts": [
                {
                    "fact_id": "f1",
                    "subject_mention_id": "m1",
                    "temporal_scope": "permanent",
                    "back_applicable": True,
                    "reveals_future_spoiler": False,
                    "description_ru": "Зелёные глаза",
                    "description_en": "Green eyes",
                    "quote_block_id": "b1",
                    "quote": "  her green eyes  ",
                }
            ],
            "covered_source_unit_ids": ["u1"],
            "complete": True,
        }
    if kind is LLMResponseKind.ILLUSTRATION_ALIAS:
        return {
            "contract_version": 1,
            "decisions": [
                {"pair_id": "p1", "verdict": "same_entity", "evidence_fact_ids": ["f1"]}
            ],
        }
    if kind is LLMResponseKind.ILLUSTRATION_DESIGN:
        return {
            "contract_version": 1,
            "entity_id": "e1",
            "traits": [
                {
                    "trait": "hair",
                    "description_ru": "Тёмные волосы",
                    "description_en": "Dark hair",
                    "assumption_ru": "Не указано явно",
                }
            ],
        }
    if kind is LLMResponseKind.ILLUSTRATION_FACT_SCOPE:
        return {
            "contract_version": 1,
            "event_id": "ev1",
            "applicable_fact_ids": ["f1"],
            "excluded_fact_ids": [],
            "compact_description_ru": "Алиса в плаще",
            "compact_description_en": "Alice in a cloak",
            "assumptions_ru": [],
            "complete": True,
        }
    if kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
        return {
            "contract_version": 1,
            "scene_id": "sc1",
            "events": [
                {
                    "event_id": "ev1",
                    "title_ru": "Вход в зал",
                    "event_ru": "Алиса входит с фонарём",
                    "significance_ru": "Открытие главы",
                    "event_kind": "arrival",
                    "quote_anchors": [{"block_id": "b1", "quote": "  Alice stepped  "}],
                    "insert_after_block_id": "b1",
                    "participant_names": ["Alice"],
                    "location_names": ["Hall"],
                }
            ],
            "zero_reason_ru": None,
            "covered_source_unit_ids": ["b1"],
            "complete": True,
        }
    if kind is LLMResponseKind.ILLUSTRATION_SELECTION:
        return {
            "contract_version": 1,
            "selected_candidate_ids": ["ev1"],
            "selection_reasons_ru": [{"candidate_id": "ev1", "reason_ru": "Ключевой кадр"}],
            "rejected_reasons_ru": [],
            "zero_reason_ru": None,
        }
    return {
        "contract_version": 1,
        "moment_id": "m1",
        "scene_composition_en": "Wide ink composition of Alice entering the dark hall.",
        "depicted_entity_ids": ["e1"],
        "used_fact_ids": ["f1"],
        "used_design_keys": ["hair"],
        "assumptions_ru": [],
    }


def _walk_schema(node: Any, path: str = "$") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(node, dict):
        found.append((path, node))
        for key, value in node.items():
            if key in {"properties", "$defs"} and isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    found.extend(_walk_schema(sub_val, f"{path}.{key}.{sub_key}"))
            else:
                found.extend(_walk_schema(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            found.extend(_walk_schema(item, f"{path}[{idx}]"))
    return found


def test_strict_json_schemas_valid_openai_root_object_no_empty_nodes_no_free_dicts_no_ref_siblings() -> None:
    for schema_cls in ILLUSTRATION_SCHEMA_CLASSES:
        strict = strict_json_schema(schema_cls)
        assert strict.get("type") == "object"
        assert strict.get("additionalProperties") is False
        assert isinstance(strict.get("required"), list)
        assert set(strict["required"]) == set(strict["properties"].keys())

        for path, node in _walk_schema(strict):
            assert node != {}, f"empty schema node at {path}"
            if "$ref" in node:
                assert set(node.keys()) == {"$ref"}, f"$ref sibling keys at {path}: {node}"
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, f"free dict at {path}: {node}"
                assert set(node.get("required", [])) == set(node.get("properties", {}).keys())


def test_quote_whitespace_preserved_and_bool_rejected_on_contract_version_literal() -> None:
    raw = _valid_json_for(LLMResponseKind.ILLUSTRATION_EXTRACT)
    dto = parse_payload(LLMResponseKind.ILLUSTRATION_EXTRACT, json.dumps(raw))
    assert dto.mentions[0].quote == "  Alice stepped  "
    assert dto.facts[0].quote == "  her green eyes  "

    cand_raw = _valid_json_for(LLMResponseKind.ILLUSTRATION_CANDIDATES)
    cand_dto = parse_payload(LLMResponseKind.ILLUSTRATION_CANDIDATES, json.dumps(cand_raw))
    assert cand_dto.events[0].quote_anchors[0].quote == "  Alice stepped  "

    for kind in _ALL_ILLUSTRATION_KINDS:
        bad = {**_valid_json_for(kind), "contract_version": True}
        with pytest.raises(LLMResponseFormatError):
            parse_payload(kind, json.dumps(bad))


def test_schema_for_illustration_kinds_rejects_invalid_versions_with_format_error() -> None:
    for kind in _ALL_ILLUSTRATION_KINDS:
        assert schema_for(kind, contract_version=1) is not None
        for kwargs in (
            {"contract_version": 2},
            {"contract_version": True},
            {"contract_version": 1, "name_translation_policy_version": 1},
            {"contract_version": 1, "pronunciation_candidates_version": 1},
        ):
            with pytest.raises(LLMResponseFormatError):
                schema_for(kind, **kwargs)  # type: ignore[arg-type]


def test_actual_wire_bytes_include_schema_options_and_repairs() -> None:
    settings = LLMSettings(
        provider="openai_compatible",
        base_url="https://api.openai.com/v1",
        model="gpt-5-mini",
        api_key="test-key",
        temperature=0.2,
        max_output_tokens=4096,
    )
    base_req = LLMRequest(
        kind=LLMResponseKind.ILLUSTRATION_EXTRACT,
        instructions="Extract visual facts.",
        input_text="Alice stepped into the hall.",
    )
    repair_req = replace(
        base_req,
        repair_turns=((None, "Fix quote offset."),),
    )
    b1 = serialize_request(base_req, settings, include_openrouter_provider=False)
    b2 = serialize_request(repair_req, settings, include_openrouter_provider=False)
    assert len(b2) > len(b1)
    decoded = json.loads(b2.decode("utf-8"))
    assert decoded["response_format"]["json_schema"]["strict"] is True
    assert decoded["temperature"] == 0.2
    assert decoded["max_completion_tokens"] == 4096
    assert any(m["content"] == "Fix quote offset." for m in decoded["messages"])


def test_openrouter_provider_option_and_anthropic_explicit_cache_parity_in_budget() -> None:
    settings = LLMSettings(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4",
        api_key="test-key",
        temperature=0.2,
        max_output_tokens=4096,
    )
    req = LLMRequest(
        kind=LLMResponseKind.ILLUSTRATION_EXTRACT,
        instructions="Extract facts.",
        input_text="A" * 2048,
    )
    no_cache = serialize_request(
        req, settings, include_openrouter_provider=True, explicit_cache=False
    )
    with_cache = serialize_request(
        req, settings, include_openrouter_provider=True, explicit_cache=True
    )
    assert json.loads(no_cache)["provider"] == {"require_parameters": True}
    assert len(with_cache) > len(no_cache)


async def test_openai_client_compatible_with_comparison_receipt_client() -> None:
    reply_doc = {
        "model": "gpt-5-mini",
        "provider": "openai",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cost": 0.001,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(_valid_json_for(LLMResponseKind.ILLUSTRATION_PROMPT))},
            }
        ],
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=reply_doc)

    ledger = ObservationLedger()
    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as raw_http:
        settings = LLMSettings(
            provider="openai_compatible",
            base_url="https://api.openai.com/v1",
            model="gpt-5-mini",
            api_key="test-key",
            temperature=0.2,
            max_output_tokens=4096,
        )
        observed = ObservedLLMClient(
            settings=settings,
            route="openai",
            http_client=raw_http,
            ledger=ledger,
            budget=BudgetGuard(Decimal("1.0")),
            max_call_cost=Decimal("0.5"),
            cost_bounds=CostBounds(
                max_request_bytes=65536,
                max_input_tokens=8192,
                input_price_per_mtok=1.0,
                output_price_per_mtok=2.0,
                output_includes_reasoning=True,
                input_bound_method="synthetic",
                evidence="unit test",
            ),
            route_metadata=RouteMetadata(
                provider_names=("openai",),
                model_versions=("gpt-5-mini",),
            ),
        )
        req = LLMRequest(
            kind=LLMResponseKind.ILLUSTRATION_PROMPT,
            instructions="Compose monochrome prompt.",
            input_text="Scene details.",
            max_request_bytes=24576,
        )
        resp = await observed.generate(req)
        assert resp.payload.moment_id == "m1"
        assert len(ledger.observations) == 1


async def test_oversized_request_raises_before_http_and_skips_call_log(tmp_path: Path) -> None:
    db = tmp_path / "oversize.db"
    apply_migrations(db)
    uow = SqlAlchemyUnitOfWork.for_engine(create_database_engine(db))
    with uow.transaction() as tx:
        tx.books.add(
            BookRecord(
                id="book-1",
                metadata=BookMetadata(title="B"),
                source_language="en",
                target_language="ru",
                status=BookStatus.ANALYZING,
                source_epub_path="/tmp/b.epub",
                created_at=datetime(2026, 11, 1, tzinfo=UTC),
            )
        )

    called = False

    def _handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    settings = LLMSettings(
        provider="openai_compatible",
        base_url="https://api.openai.com/v1",
        model="gpt-5-mini",
        api_key="test-key",
        temperature=0.2,
        max_output_tokens=4096,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as http:
        inner = OpenAICompatibleLLMClient(
            settings=settings,
            http_client=http,
            allow_explicit_prompt_cache=False,
        )
        logged = LoggingLLMClient(
            inner=inner,
            unit_of_work=uow,
            provider="openai_compatible",
            model="gpt-5-mini",
        )
        attempt_id = str(uuid4())
        req = LLMRequest(
            kind=LLMResponseKind.ILLUSTRATION_EXTRACT,
            instructions="Extract facts.",
            input_text="X" * 5000,
            book_id="book-1",
            stage=LLMCallStage.ILLUSTRATION_REFERENCE,
            max_request_bytes=500,
            call_context=CallContext(
                conditions_id=str(uuid4()),
                stage=LLMCallStage.ILLUSTRATION_REFERENCE.value,
                scope_id="scope-1",
                ordered_block_ids=(),
                attempt_id=attempt_id,
                generation=1,
                assigned_model="gpt-5-mini",
                prompt_hash=sha256(b"Extract facts.").hexdigest(),
            ),
        )
        with pytest.raises(LLMRequestTooLargeError) as err:
            await logged.generate(req)
        assert not called
        assert classify_error(err.value).type is ErrorType.CONFIGURATION
        with uow.read_transaction() as tx:
            assert tx.llm_calls.list_for_book("book-1") == ()
        assert diagnostics_for(uow)._failed is False


async def test_global_reasoning_settings_do_not_break_illustration_calls() -> None:
    # Clean illustration settings strip reasoning_effort / reasoning_max_tokens so payload has neither.
    settings = LLMSettings(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-5-mini",
        api_key="test-key",
        temperature=0.2,
        max_output_tokens=4096,
        reasoning_effort=None,
        reasoning_max_tokens=None,
    )
    req = LLMRequest(
        kind=LLMResponseKind.ILLUSTRATION_ALIAS,
        instructions="Resolve aliases.",
        input_text="Pairs.",
    )
    wire = json.loads(
        serialize_request(req, settings, include_openrouter_provider=True, explicit_cache=False)
    )
    assert "reasoning" not in wire
    assert "reasoning_effort" not in wire


async def test_truncated_valid_json_rejected_in_live_and_fake_clients_all_7_kinds() -> None:
    for kind in _ALL_ILLUSTRATION_KINDS:
        valid_str = json.dumps(_valid_json_for(kind))
        # Direct parse_payload with truncated=True
        with pytest.raises(LLMResponseFormatError) as direct_err:
            parse_payload(kind, valid_str, truncated=True)
        assert direct_err.value.truncated is True
        assert direct_err.value.violation is not None
        assert direct_err.value.violation.rule_code == "provider_truncated"

        # FakeLLMClient with truncated=True
        fake = FakeLLMClient([FakeReply(content=valid_str, truncated=True)])
        req = LLMRequest(kind=kind, instructions="Do task.", input_text="Input.")
        with pytest.raises(LLMResponseFormatError) as fake_err:
            await fake.generate(req)
        assert fake_err.value.truncated is True

        # Live OpenAICompatibleLLMClient with finish_reason="length"
        def _handler(_r: httpx.Request, body: str = valid_str) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "gpt-5-mini",
                    "choices": [{"finish_reason": "length", "message": {"content": body}}],
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as http:
            client = OpenAICompatibleLLMClient(
                settings=LLMSettings(model="gpt-5-mini", api_key="k"),
                http_client=http,
                allow_explicit_prompt_cache=False,
            )
            with pytest.raises(LLMResponseFormatError) as live_err:
                await client.generate(req)
            assert live_err.value.truncated is True


def test_malformed_and_extra_field_llm_json_rejected() -> None:
    for kind in _ALL_ILLUSTRATION_KINDS:
        with pytest.raises(LLMResponseFormatError):
            parse_payload(kind, "{not valid json")
        extra = {**_valid_json_for(kind), "unexpected_extra_key": "boom"}
        with pytest.raises(LLMResponseFormatError):
            parse_payload(kind, json.dumps(extra))


async def test_illustration_format_error_does_not_poison_diagnostics_on_real_sqlite_or_fake_stubs(
    tmp_path: Path,
) -> None:
    db = tmp_path / "diag.db"
    apply_migrations(db)
    uow = SqlAlchemyUnitOfWork.for_engine(create_database_engine(db))
    diag = Diagnostics(uow, policy=DiagnosticsPolicy())
    with uow.transaction() as tx:
        tx.books.add(
            BookRecord(
                id="book-1",
                metadata=BookMetadata(title="B"),
                source_language="en",
                target_language="ru",
                status=BookStatus.ANALYZING,
                source_epub_path="/tmp/b.epub",
                created_at=datetime(2026, 11, 1, tzinfo=UTC),
            )
        )

    fake = FakeLLMClient([FakeReply(content="{broken")])
    logged = LoggingLLMClient(
        inner=fake,
        unit_of_work=uow,
        provider="fake",
        model="gpt-5-mini",
    )
    attempt_id = str(uuid4())
    req = LLMRequest(
        kind=LLMResponseKind.ILLUSTRATION_EXTRACT,
        instructions="Extract facts.",
        input_text="Input text.",
        book_id="book-1",
        stage=LLMCallStage.ILLUSTRATION_REFERENCE,
        call_context=CallContext(
            conditions_id=str(uuid4()),
            stage=LLMCallStage.ILLUSTRATION_REFERENCE.value,
            scope_id="scope-1",
            ordered_block_ids=(),
            attempt_id=attempt_id,
            generation=1,
            assigned_model="gpt-5-mini",
            prompt_hash=sha256(b"Extract facts.").hexdigest(),
        ),
    )
    with pytest.raises(LLMResponseFormatError):
        await logged.generate(req)
    # Real SQLite diagnostics must NOT be poisoned even though attempt_id is NOT in llm_attempts!
    assert diag._failed is False
    diag.ensure_available()
    with uow.read_transaction() as tx:
        rec = tx.llm_calls.diagnostic_record(attempt_id=attempt_id)
        assert rec is not None
        assert rec["stage"] == LLMCallStage.ILLUSTRATION_REFERENCE.value


async def test_unique_uuid_attempt_id_per_physical_call_under_logging_client(
    tmp_path: Path,
) -> None:
    db = tmp_path / "uniq.db"
    apply_migrations(db)
    uow = SqlAlchemyUnitOfWork.for_engine(create_database_engine(db))
    with uow.transaction() as tx:
        tx.books.add(
            BookRecord(
                id="book-1",
                metadata=BookMetadata(title="B"),
                source_language="en",
                target_language="ru",
                status=BookStatus.ANALYZING,
                source_epub_path="/tmp/b.epub",
                created_at=datetime(2026, 11, 1, tzinfo=UTC),
            )
        )
    valid = json.dumps(_valid_json_for(LLMResponseKind.ILLUSTRATION_EXTRACT))
    fake = FakeLLMClient([FakeReply(content="{broken"), FakeReply(content=valid)])
    logged = LoggingLLMClient(inner=fake, unit_of_work=uow, provider="fake", model="gpt-5-mini")
    cond_id = str(uuid4())

    req1 = LLMRequest(
        kind=LLMResponseKind.ILLUSTRATION_EXTRACT,
        instructions="Extract.",
        input_text="Input.",
        book_id="book-1",
        stage=LLMCallStage.ILLUSTRATION_REFERENCE,
        call_context=CallContext(
            conditions_id=cond_id,
            stage=LLMCallStage.ILLUSTRATION_REFERENCE.value,
            scope_id="scope-1",
            ordered_block_ids=(),
            attempt_id=str(uuid4()),
            generation=1,
            assigned_model="gpt-5-mini",
            prompt_hash=sha256(b"Extract.").hexdigest(),
        ),
    )
    with pytest.raises(LLMResponseFormatError):
        await logged.generate(req1)

    req2 = replace(
        req1,
        attempt=2,
        call_context=replace(req1.call_context, attempt_id=str(uuid4())),  # type: ignore[arg-type]
    )
    resp2 = await logged.generate(req2)
    assert resp2.payload.complete is True
    with uow.read_transaction() as tx:
        calls = tx.llm_calls.list_for_book("book-1")
        assert len(calls) == 2
        assert len({c.attempt_id for c in calls}) == 2


def test_illustration_candidates_schema_accepts_up_to_256_covered_ids_and_rejects_257() -> None:
    base = _valid_json_for(LLMResponseKind.ILLUSTRATION_CANDIDATES)
    ok_41 = {**base, "covered_source_unit_ids": [f"b-{i}" for i in range(41)]}
    dto_41 = parse_payload(LLMResponseKind.ILLUSTRATION_CANDIDATES, json.dumps(ok_41))
    assert len(dto_41.covered_source_unit_ids) == 41

    ok_256 = {**base, "covered_source_unit_ids": [f"b-{i}" for i in range(256)]}
    dto_256 = parse_payload(LLMResponseKind.ILLUSTRATION_CANDIDATES, json.dumps(ok_256))
    assert len(dto_256.covered_source_unit_ids) == 256

    too_many = {**base, "covered_source_unit_ids": [f"b-{i}" for i in range(257)]}
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.ILLUSTRATION_CANDIDATES, json.dumps(too_many))


def test_illustration_design_trait_allows_empty_assumption_ru_and_keeps_required_in_strict_schema() -> None:
    payload = _valid_json_for(LLMResponseKind.ILLUSTRATION_DESIGN)
    payload["traits"][0]["assumption_ru"] = ""
    dto = parse_payload(LLMResponseKind.ILLUSTRATION_DESIGN, json.dumps(payload))
    assert dto.traits[0].assumption_ru == ""

    strict = strict_json_schema(schema_for(LLMResponseKind.ILLUSTRATION_DESIGN, contract_version=1))
    trait_def = strict["$defs"]["IllustrationDesignTraitSchema"]
    assert "assumption_ru" in trait_def["required"]


def test_selection_candidates_and_prompt_strict_schema_descriptions_and_depicted_entity_ids() -> None:
    sel_strict = strict_json_schema(
        schema_for(LLMResponseKind.ILLUSTRATION_SELECTION, contract_version=1)
    )
    for field in (
        "selected_candidate_ids",
        "selection_reasons_ru",
        "rejected_reasons_ru",
        "zero_reason_ru",
    ):
        assert sel_strict["properties"][field].get("description")
    assert "selected_candidate_ids" in sel_strict["properties"]["selection_reasons_ru"]["description"]

    cand_strict = strict_json_schema(
        schema_for(LLMResponseKind.ILLUSTRATION_CANDIDATES, contract_version=1)
    )
    ev_props = cand_strict["$defs"]["IllustrationCandidateEventSchema"]["properties"]
    assert ev_props["quote_anchors"].get("description")
    assert "block_id" in ev_props["quote_anchors"]["description"]
    assert ev_props["participant_names"].get("description")
    assert ev_props["location_names"].get("description")

    prompt_strict = strict_json_schema(
        schema_for(LLMResponseKind.ILLUSTRATION_PROMPT, contract_version=1)
    )
    assert "depicted_entity_ids" in prompt_strict["required"]
    assert prompt_strict["properties"]["depicted_entity_ids"].get("description")

    valid_prompt = _valid_json_for(LLMResponseKind.ILLUSTRATION_PROMPT)
    dto = parse_payload(LLMResponseKind.ILLUSTRATION_PROMPT, json.dumps(valid_prompt))
    assert dto.depicted_entity_ids == ("e1",)

    missing_depicted = {k: v for k, v in valid_prompt.items() if k != "depicted_entity_ids"}
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.ILLUSTRATION_PROMPT, json.dumps(missing_depicted))


def test_illustration_prompt_used_fact_ids_cap_128_and_required_in_strict_schema() -> None:
    prompt_strict = strict_json_schema(
        schema_for(LLMResponseKind.ILLUSTRATION_PROMPT, contract_version=1)
    )
    assert "used_fact_ids" in prompt_strict["required"]
    used_fids_prop = prompt_strict["properties"]["used_fact_ids"]
    assert "minItems" not in used_fids_prop
    assert "default" not in used_fids_prop

    base = _valid_json_for(LLMResponseKind.ILLUSTRATION_PROMPT)
    dto_48 = parse_payload(
        LLMResponseKind.ILLUSTRATION_PROMPT,
        json.dumps({**base, "used_fact_ids": [f"f-{i}" for i in range(48)]}),
    )
    assert len(dto_48.used_fact_ids) == 48

    dto_128 = parse_payload(
        LLMResponseKind.ILLUSTRATION_PROMPT,
        json.dumps({**base, "used_fact_ids": [f"f-{i}" for i in range(128)]}),
    )
    assert len(dto_128.used_fact_ids) == 128

    with pytest.raises(LLMResponseFormatError):
        parse_payload(
            LLMResponseKind.ILLUSTRATION_PROMPT,
            json.dumps({**base, "used_fact_ids": [f"f-{i}" for i in range(129)]}),
        )

    missing_used = {k: v for k, v in base.items() if k != "used_fact_ids"}
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.ILLUSTRATION_PROMPT, json.dumps(missing_used))

