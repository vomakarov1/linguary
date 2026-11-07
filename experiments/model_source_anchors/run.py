#!/usr/bin/env python3
"""Run and summarize the source-anchor comparison through OpenRouter."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import statistics
import shutil
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


MODEL_ID = "deepseek/deepseek-v3.2"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
GENERATION_URL = "https://openrouter.ai/api/v1/generation"
TEMPERATURE = 0.2
TOP_P = 1.0
MAX_TOKENS = 512
REPLICATES = 2
SCHEDULE_SEED = 20261107
API_TIMEOUT_SECONDS = 120
METADATA_TIMEOUT_SECONDS = 30

ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = Path(__file__).with_name("cases.json")
FEATURE_DIR = ROOT / ".ai/develop-context.local/BRIEF-model-source-anchors-2026-11-07"
DEFAULT_RUNS_DIR = FEATURE_DIR / "runs"

COMMON_INSTRUCTIONS = """You help a reader understand an English literary excerpt.
Choose one specific detail, image, or idea from the excerpt that merits a short reading aid.
Write one concise explanation grounded only in the supplied passage. Return exactly one
source quote that supports the explanation. Copy the quote exactly, including punctuation.
Do not invent context or quote text that is absent from the passage."""

OFFSET_INSTRUCTIONS = """Return the source anchor with quote, start, and end. start and end
are zero-based Unicode code-point indexes in the exact source-block text between the delimiters;
exclude the block label and delimiters themselves. end is excluded, and quote must equal the
source-block text sliced as [start:end]."""

QUOTE_ONLY_INSTRUCTIONS = """Return the source anchor with quote only. Do not return any
position, offset, block identifier, or other anchor field."""

VARIANTS = ("with_offsets", "quote_only")
JUDGMENT_FIELDS = (
    "manual_fact_supported",
    "manual_quote_supports_aid",
    "manual_reason",
)


def anchor_schema(variant: str) -> dict[str, Any]:
    properties: dict[str, Any] = {"quote": {"type": "string", "minLength": 1}}
    required = ["quote"]
    if variant == "with_offsets":
        properties["start"] = {"type": "integer", "minimum": 0}
        properties["end"] = {"type": "integer", "minimum": 1}
        required.extend(("start", "end"))
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def response_schema(variant: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reading_aid": {"type": "string", "minLength": 1},
            "anchor": anchor_schema(variant),
        },
        "required": ["reading_aid", "anchor"],
        "additionalProperties": False,
    }


def instructions(variant: str) -> str:
    if variant == "with_offsets":
        anchor_instruction = OFFSET_INSTRUCTIONS
    elif variant == "quote_only":
        anchor_instruction = QUOTE_ONLY_INSTRUCTIONS
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return f"{COMMON_INSTRUCTIONS}\n\n{anchor_instruction}"


def user_input(case: dict[str, Any]) -> str:
    return f"Source block {case['block_id']}:\n---\n{case['text']}\n---"


def build_request(case: dict[str, Any], variant: str) -> dict[str, Any]:
    schema_name = "source_anchor_with_offsets" if variant == "with_offsets" else "source_anchor_quote_only"
    return {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": instructions(variant)},
            {"role": "user", "content": user_input(case)},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "provider": {"require_parameters": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": response_schema(variant),
            },
        },
    }


def make_schedule(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(SCHEDULE_SEED)
    schedule: list[dict[str, Any]] = []
    for case in cases:
        for replicate in range(1, REPLICATES + 1):
            pair = [
                {"case_id": case["id"], "replicate": replicate, "variant": variant}
                for variant in VARIANTS
            ]
            rng.shuffle(pair)
            schedule.extend(pair)
    return schedule


def read_cases(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if manifest.get("schema_version") != 1 or not isinstance(cases, list) or len(cases) != 10:
        raise ValueError("Expected a version 1 dataset containing exactly 10 cases.")
    ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not all(
            isinstance(case.get(key), str) and case[key]
            for key in ("id", "block_id", "text", "expected_fact")
        ):
            raise ValueError("Every case needs an id, block_id, text, and expected_fact.")
        if case["id"] in ids:
            raise ValueError(f"Duplicate case id: {case['id']}")
        ids.add(case["id"])
        if sha256(case["text"].encode("utf-8")).hexdigest() != case.get("text_sha256"):
            raise ValueError(f"Frozen source text hash does not match for {case['id']}.")
        if case.get("expected_anchor_count") != 1 or not case.get("reference_quotes"):
            raise ValueError(f"Expected one annotated anchor for {case['id']}.")
    return manifest, cases


def json_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def safe_error(error: Exception, api_key: str) -> str:
    message = str(error).replace(api_key, "[redacted]") if api_key else str(error)
    return message[:1000]


def post_completion(body: dict[str, Any], api_key: str) -> dict[str, Any]:
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(
        CHAT_URL,
        data=encoded,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "Linguary source anchor experiment",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
            raw = response.read()
            headers = response.headers
            status_code = response.status
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        document = json.loads(raw.decode("utf-8"))
        return {
            "status_code": status_code,
            "elapsed_ms": elapsed_ms,
            "document": document,
            "generation_id": headers.get("X-Generation-Id"),
        }
    except HTTPError as error:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        try:
            detail = error.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        return {
            "status_code": error.code,
            "elapsed_ms": elapsed_ms,
            "error": safe_error(URLError(detail or error.reason), api_key),
            "raw_error": detail.replace(api_key, "[redacted]")[:2000],
        }
    except Exception as error:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "status_code": None,
            "elapsed_ms": elapsed_ms,
            "error": safe_error(error, api_key),
        }


def extract_content(document: dict[str, Any]) -> str:
    choices = document.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("response has no first choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("response has no message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "".join(part for part in parts if isinstance(part, str))
    raise ValueError("response content is not text")


def parse_output(content: str, variant: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError) as error:
        return None, f"invalid JSON: {error}"
    if not isinstance(value, dict) or set(value) != {"reading_aid", "anchor"}:
        return None, "response must contain only reading_aid and anchor"
    anchor = value["anchor"]
    if not isinstance(value["reading_aid"], str) or not value["reading_aid"].strip():
        return None, "reading_aid must be a non-empty string"
    if not isinstance(anchor, dict) or not isinstance(anchor.get("quote"), str) or not anchor["quote"]:
        return None, "anchor.quote must be a non-empty string"
    expected_keys = {"quote", "start", "end"} if variant == "with_offsets" else {"quote"}
    if set(anchor) != expected_keys:
        return None, f"anchor fields must be exactly {sorted(expected_keys)}"
    if variant == "with_offsets":
        start, end = anchor.get("start"), anchor.get("end")
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            return None, "start/end must be valid non-negative half-open offsets"
    return value, None


def generation_id(result: dict[str, Any]) -> str | None:
    if result.get("generation_id"):
        return result["generation_id"]
    document = result.get("document")
    candidate = document.get("id") if isinstance(document, dict) else None
    return candidate if isinstance(candidate, str) and candidate.startswith("gen-") else None


def fetch_generation_metadata(api_key: str, generation: str) -> dict[str, Any]:
    url = f"{GENERATION_URL}?{urlencode({'id': generation})}"
    request = Request(url, headers={"Authorization": f"Bearer {api_key}"}, method="GET")
    try:
        with urlopen(request, timeout=METADATA_TIMEOUT_SECONDS) as response:
            document = json.loads(response.read().decode("utf-8"))
        data = document.get("data", {})
        if not isinstance(data, dict):
            data = {}
        return {
            key: data.get(key)
            for key in ("id", "model", "provider_name", "tokens_prompt", "tokens_completion", "total_cost", "latency")
        }
    except Exception as error:
        return {"metadata_error": safe_error(error, api_key)}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_experiment(cases_path: Path, output_root: Path, api_key: str) -> Path:
    source_manifest, cases = read_cases(cases_path)
    schedule = make_schedule(cases)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    case_file_hash = sha256(cases_path.read_bytes()).hexdigest()
    shutil.copyfile(cases_path, run_dir / "cases_snapshot.json")
    request_manifest = {
        "experiment": "model-source-anchors",
        "run_id": run_id,
        "created_at": utc_now(),
        "source": source_manifest["source"],
        "cases_file_sha256": case_file_hash,
        "model": MODEL_ID,
        "generation_parameters": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "provider_preferences": {"require_parameters": True},
        },
        "replicates_per_case_and_variant": REPLICATES,
        "scheduled_model_calls": len(schedule),
        "schedule_seed": SCHEDULE_SEED,
        "schedule": schedule,
        "automatic_retries": 0,
        "variants": {
            "with_offsets": {"instructions": instructions("with_offsets"), "schema": response_schema("with_offsets")},
            "quote_only": {"instructions": instructions("quote_only"), "schema": response_schema("quote_only")},
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(request_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    case_by_id = {case["id"]: case for case in cases}
    records: list[dict[str, Any]] = []
    for index, scheduled in enumerate(schedule, start=1):
        case = case_by_id[scheduled["case_id"]]
        body = build_request(case, scheduled["variant"])
        started_at = utc_now()
        result = post_completion(body, api_key)
        document = result.get("document")
        raw_content = None
        parsed = None
        format_error = None
        status = "http_error" if result.get("error") else "ok"
        if isinstance(document, dict):
            try:
                raw_content = extract_content(document)
                parsed, format_error = parse_output(raw_content, scheduled["variant"])
                if format_error:
                    status = "format_error"
            except Exception as error:
                format_error = safe_error(error, api_key)
                status = "format_error"
        record = {
            "call_id": f"{scheduled['case_id']}-r{scheduled['replicate']}-{scheduled['variant']}",
            **scheduled,
            "started_at": started_at,
            "status": status,
            "status_code": result.get("status_code"),
            "elapsed_ms": result.get("elapsed_ms"),
            "request_sha256": json_sha(body),
            "input_text": user_input(case),
            "input_sha256": sha256(case["text"].encode("utf-8")).hexdigest(),
            "expected_fact": case["expected_fact"],
            "request_body": body,
            "model_requested": MODEL_ID,
            "model_returned": document.get("model") if isinstance(document, dict) else None,
            "generation_id": generation_id(result),
            "usage": document.get("usage") if isinstance(document, dict) else None,
            "response_body": document,
            "raw_content": raw_content,
            "parsed_output": parsed,
            "format_error": format_error,
            "error": result.get("error"),
            "raw_error": result.get("raw_error"),
        }
        records.append(record)
        write_jsonl(run_dir / "results.jsonl", records)
        print(f"[{index}/{len(schedule)}] {record['call_id']}: {record['status']}")

    metadata_records = []
    for record in records:
        generation = record.get("generation_id")
        if generation:
            metadata_records.append(
                {
                    "call_id": record["call_id"],
                    "generation_id": generation,
                    "metadata": fetch_generation_metadata(api_key, generation),
                }
            )
    (run_dir / "generation_metadata.json").write_text(
        json.dumps(metadata_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    build_evaluation_csv(run_dir, cases, schedule, records)
    write_report(run_dir)
    print(f"Saved experiment run to {run_dir}")
    return run_dir


def occurrences(text: str, quote: str) -> list[int]:
    if not quote:
        return []
    found: list[int] = []
    position = 0
    while True:
        position = text.find(quote, position)
        if position < 0:
            return found
        found.append(position)
        position += 1


def evaluation_row(record: dict[str, Any], case: dict[str, Any]) -> dict[str, str]:
    output = record.get("parsed_output")
    anchor = output.get("anchor", {}) if isinstance(output, dict) else {}
    quote = anchor.get("quote", "") if isinstance(anchor, dict) else ""
    positions = occurrences(case["text"], quote) if isinstance(quote, str) else []
    start, end = anchor.get("start"), anchor.get("end")
    offsets_match_quote = (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and 0 <= start < end <= len(case["text"])
        and case["text"][start:end] == quote
    )
    reference_match = any(item.get("text") == quote for item in case.get("reference_quotes", []))
    if record.get("variant") == "with_offsets" and offsets_match_quote:
        resolved_start, resolved_end = start, end
    elif record.get("variant") == "quote_only" and len(positions) == 1:
        resolved_start, resolved_end = positions[0], positions[0] + len(quote)
    else:
        resolved_start = resolved_end = ""
    return {
        "call_id": str(record.get("call_id", "")),
        "case_id": str(record.get("case_id", "")),
        "variant": str(record.get("variant", "")),
        "replicate": str(record.get("replicate", "")),
        "call_status": str(record.get("status", "missing")),
        "expected_fact": case["expected_fact"],
        "reading_aid": output.get("reading_aid", "") if isinstance(output, dict) else "",
        "quote": quote,
        "quote_exact_occurrences": str(len(positions)),
        "reference_quote_match": str(reference_match).lower(),
        "start": "" if start is None else str(start),
        "end": "" if end is None else str(end),
        "offsets_match_quote": "" if record.get("variant") == "quote_only" else str(offsets_match_quote).lower(),
        "resolved_start": str(resolved_start) if resolved_start != "" else "",
        "resolved_end": str(resolved_end) if resolved_end != "" else "",
        "manual_fact_supported": "",
        "manual_quote_supports_aid": "",
        "manual_reason": "",
    }


def build_evaluation_csv(
    run_dir: Path,
    cases: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> Path:
    path = run_dir / "evaluation.csv"
    if path.exists():
        return path
    case_by_id = {case["id"]: case for case in cases}
    record_by_id = {record["call_id"]: record for record in records}
    fields = [
        "call_id", "case_id", "variant", "replicate", "call_status", "expected_fact",
        "reading_aid", "quote", "quote_exact_occurrences", "reference_quote_match", "start",
        "end", "offsets_match_quote", "resolved_start", "resolved_end", *JUDGMENT_FIELDS,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scheduled in schedule:
            call_id = f"{scheduled['case_id']}-r{scheduled['replicate']}-{scheduled['variant']}"
            record = record_by_id.get(call_id, {**scheduled, "call_id": call_id, "status": "missing"})
            writer.writerow(evaluation_row(record, case_by_id[scheduled["case_id"]]))
    return path


def write_report(run_dir: Path) -> Path:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    records_path = run_dir / "results.jsonl"
    if records_path.exists():
        with records_path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        records = []
    metadata_path = run_dir / "generation_metadata.json"
    metadata_records = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else []
    metadata_by_call = {item.get("call_id"): item.get("metadata", {}) for item in metadata_records}
    evaluations_path = run_dir / "evaluation.csv"
    if evaluations_path.exists():
        with evaluations_path.open(encoding="utf-8", newline="") as handle:
            evaluations = list(csv.DictReader(handle))
    else:
        evaluations = []
    expected = 10 * REPLICATES
    summaries: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        variant_records = [row for row in records if row.get("variant") == variant]
        variant_evaluations = [row for row in evaluations if row.get("variant") == variant]
        http_success = sum(
            isinstance(row.get("status_code"), int) and 200 <= row["status_code"] < 300
            for row in variant_records
        )
        valid_format = sum(row.get("call_status") == "ok" for row in variant_evaluations)
        exact_quote = sum(int(row.get("quote_exact_occurrences") or 0) > 0 for row in variant_evaluations)
        if variant == "with_offsets":
            resolved = sum(row.get("offsets_match_quote") == "true" for row in variant_evaluations)
        else:
            resolved = sum(int(row.get("quote_exact_occurrences") or 0) == 1 for row in variant_evaluations)
        fact_ratings = [row.get("manual_fact_supported", "").strip().lower() for row in variant_evaluations]
        quote_ratings = [row.get("manual_quote_supports_aid", "").strip().lower() for row in variant_evaluations]
        overall = 0
        for row in variant_evaluations:
            anchor_resolved = (
                row.get("offsets_match_quote") == "true"
                if variant == "with_offsets"
                else int(row.get("quote_exact_occurrences") or 0) == 1
            )
            if (
                row.get("call_status") == "ok"
                and row.get("manual_fact_supported", "").strip().lower() == "yes"
                and row.get("manual_quote_supports_aid", "").strip().lower() == "yes"
                and anchor_resolved
            ):
                overall += 1
        costs: list[float] = []
        for row in variant_records:
            metadata = metadata_by_call.get(row.get("call_id"), {})
            cost = metadata.get("total_cost")
            if cost is None:
                usage = row.get("usage") or {}
                cost = usage.get("cost") if isinstance(usage, dict) else None
            if isinstance(cost, (int, float)):
                costs.append(float(cost))
        latencies = [
            float(row["elapsed_ms"])
            for row in variant_records
            if isinstance(row.get("elapsed_ms"), (int, float))
        ]
        providers = sorted(
            {
                metadata_by_call.get(row.get("call_id"), {}).get("provider_name")
                for row in variant_records
                if metadata_by_call.get(row.get("call_id"), {}).get("provider_name")
            }
        )
        summaries[variant] = {
            "http_success": http_success,
            "valid_format": valid_format,
            "exact_quote": exact_quote,
            "resolved": resolved,
            "fact_yes": sum(value == "yes" for value in fact_ratings),
            "fact_judged": sum(value in {"yes", "no", "uncertain"} for value in fact_ratings),
            "quote_yes": sum(value == "yes" for value in quote_ratings),
            "quote_judged": sum(value in {"yes", "no", "uncertain"} for value in quote_ratings),
            "overall": overall,
            "cost_total": sum(costs),
            "cost_known": len(costs),
            "latency_mean": statistics.mean(latencies) if latencies else None,
            "latency_known": len(latencies),
            "providers": providers,
        }

    lines = [
        "# Отчёт: сравнение привязки цитат",
        "",
        f"- Модель: `{manifest['model']}`",
        f"- Запланировано вызовов модели: {manifest['scheduled_model_calls']}",
        f"- SHA-256 файла наборов: `{manifest['cases_file_sha256']}`",
        f"- Источник: [Project Gutenberg, eBook №1952]({manifest['source']['ebook_url']})",
        "- Ручная оценка смысловой пригодности заполняется в `evaluation.csv`.",
        "",
        "| Показатель | `with_offsets` | `quote_only` |",
        "| --- | ---: | ---: |",
        f"| HTTP-ответ получен | {summaries['with_offsets']['http_success']}/{expected} | {summaries['quote_only']['http_success']}/{expected} |",
        f"| Ответ прошёл проверку формата | {summaries['with_offsets']['valid_format']}/{expected} | {summaries['quote_only']['valid_format']}/{expected} |",
        f"| Цитата найдена дословно | {summaries['with_offsets']['exact_quote']}/{expected} | {summaries['quote_only']['exact_quote']}/{expected} |",
        f"| Привязка разрешима | {summaries['with_offsets']['resolved']}/{expected} | {summaries['quote_only']['resolved']}/{expected} |",
        f"| Целевой факт подтверждён вручную | {summaries['with_offsets']['fact_yes']}/{expected} (оценено {summaries['with_offsets']['fact_judged']}/{expected}) | {summaries['quote_only']['fact_yes']}/{expected} (оценено {summaries['quote_only']['fact_judged']}/{expected}) |",
        f"| Цитата подтверждает пояснение | {summaries['with_offsets']['quote_yes']}/{expected} (оценено {summaries['with_offsets']['quote_judged']}/{expected}) | {summaries['quote_only']['quote_yes']}/{expected} (оценено {summaries['quote_only']['quote_judged']}/{expected}) |",
        f"| Общий успех | {summaries['with_offsets']['overall']}/{expected} | {summaries['quote_only']['overall']}/{expected} |",
        f"| Стоимость USD (известно вызовов) | {summaries['with_offsets']['cost_total']:.8f} ({summaries['with_offsets']['cost_known']}/{expected}) | {summaries['quote_only']['cost_total']:.8f} ({summaries['quote_only']['cost_known']}/{expected}) |",
        f"| Средняя задержка, мс (измерено вызовов) | {summaries['with_offsets']['latency_mean'] if summaries['with_offsets']['latency_mean'] is not None else 'нет данных'} ({summaries['with_offsets']['latency_known']}/{expected}) | {summaries['quote_only']['latency_mean'] if summaries['quote_only']['latency_mean'] is not None else 'нет данных'} ({summaries['quote_only']['latency_known']}/{expected}) |",
        f"| Фактические провайдеры | {', '.join(summaries['with_offsets']['providers']) or 'неизвестны'} | {', '.join(summaries['quote_only']['providers']) or 'неизвестны'} |",
    ]
    lines.extend(
        [
            "",
            "## Пояснения",
            "",
            "- В каждой схеме знаменатель вызовов и ожидаемых опор — 20; отсутствующий или ошибочный ответ считается неуспехом.",
            "- Для `with_offsets` привязка разрешима, если `text[start:end]` точно равен цитате.",
            "- Для `quote_only` привязка разрешима только при ровно одном точном вхождении цитаты в исходном блоке.",
            "- Заполните `manual_fact_supported` и `manual_quote_supports_aid` значениями `yes`, `no` или `uncertain`; после этого повторно запустите отчёт.",
            "- Выводы относятся только к выбранному рассказу и этим десяти отрывкам.",
        ]
    )
    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_command(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing; no requests were sent.")
    run_experiment(args.cases, args.output_root, api_key)


def report_command(args: argparse.Namespace) -> None:
    run_manifest_path = args.run_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise SystemExit(f"Not an experiment run directory: {args.run_dir}")
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    cases_path = args.run_dir / "cases_snapshot.json"
    if sha256(cases_path.read_bytes()).hexdigest() != manifest["cases_file_sha256"]:
        raise SystemExit("The frozen case snapshot does not match the run manifest.")
    _, cases = read_cases(cases_path)
    schedule = manifest["schedule"]
    records_path = args.run_dir / "results.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if records_path.exists() else []
    build_evaluation_csv(args.run_dir, cases, schedule, records)
    report = write_report(args.run_dir)
    print(f"Updated {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="send the 40 scheduled model requests")
    run_parser.add_argument("--execute", action="store_true", help="explicitly authorize sending requests")
    run_parser.add_argument("--cases", type=Path, default=CASES_PATH)
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_RUNS_DIR)
    run_parser.set_defaults(handler=run_command)
    report_parser = commands.add_parser("report", help="rebuild the evaluation CSV and report")
    report_parser.add_argument("run_dir", type=Path)
    report_parser.set_defaults(handler=report_command)
    args = parser.parse_args()
    if args.command == "run" and not args.execute:
        raise SystemExit("No model requests sent. Add --execute to run the 40-call experiment.")
    args.handler(args)


if __name__ == "__main__":
    main()
