"""Antigravity run_workflow script: local OpenAI-compatible bridge for Linguary.

Intercepts POST /v1/chat/completions from Linguary CLI commands, validates responses
recursively against the requested OpenAI strict JSON Schema, and fulfills completions
using stateless `linguary-llm` subagents routed by model tier.

Must use Python 3 standard library only (runs under the workflow python3 interpreter;
Linguary CLI itself is invoked via `uv run python -m linguary.cli`).
"""

import asyncio
import json
import os
import re
import shutil
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(WORKSPACES[0]) if WORKSPACES else Path.cwd()

MODEL_TIER_MAP = {
    "subagent-flash-lite": "flash_lite",
    "subagent-flash": "flash",
    "subagent-pro": "pro",
    "subagent-inherit": "inherit",
}

VALID_TIERS = frozenset({"flash_lite", "flash", "pro", "inherit"})


def tier_to_model_name(tier: str) -> str:
    norm = str(tier or "pro").strip().lower()
    if norm not in VALID_TIERS:
        raise ValueError(f"Unsupported agent tier {tier!r}; expected one of {sorted(VALID_TIERS)}")
    return f"subagent-{norm.replace('_', '-')}"


def extract_json_object(raw: str):
    """Parse JSON from raw agent string, stripping optional markdown fences."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _resolve_ref(ref: str, root_schema: dict) -> dict | None:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    node = root_schema
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node if isinstance(node, dict) else None


def _matches_type(value, expected_type: str) -> bool:
    if expected_type == "null":
        return value is None
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


def validate_json_schema(
    value,
    schema: dict | None,
    root_schema: dict | None = None,
    path: str = "$",
    errors: list[str] | None = None,
    max_errors: int = 12,
) -> list[str]:
    """Pure-stdlib recursive validator for OpenAI strict JSON Schema subset."""
    if errors is None:
        errors = []
    if not isinstance(schema, dict) or len(errors) >= max_errors:
        return errors
    if root_schema is None:
        root_schema = schema

    if "$ref" in schema:
        resolved = _resolve_ref(schema["$ref"], root_schema)
        if resolved is not None:
            return validate_json_schema(value, resolved, root_schema, path, errors, max_errors)

    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        branch_errors = []
        for branch in schema["anyOf"]:
            sub = validate_json_schema(value, branch, root_schema, path, [], max_errors=4)
            if not sub:
                return errors
            branch_errors.append(sub[0])
        errors.append(f"{path}: failed all anyOf branches ({'; '.join(branch_errors[:3])})")
        return errors

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}, got {value!r}")
        return errors

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value {value!r} not in enum {schema['enum']!r}")
        return errors

    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        if not _matches_type(value, expected_type):
            errors.append(f"{path}: expected {expected_type}, got {type(value).__name__}")
            return errors
    elif isinstance(expected_type, list):
        if not any(_matches_type(value, t) for t in expected_type):
            errors.append(f"{path}: expected one of {expected_type}, got {type(value).__name__}")
            return errors

    if isinstance(value, dict):
        required = schema.get("required") or []
        for req_key in required:
            if req_key not in value:
                errors.append(f"{path}.{req_key}: missing required field")
                if len(errors) >= max_errors:
                    return errors
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    errors.append(f"{path}.{k}: unexpected additional property")
                    if len(errors) >= max_errors:
                        return errors
        for k, v in value.items():
            if k in props:
                validate_json_schema(v, props[k], root_schema, f"{path}.{k}", errors, max_errors)
                if len(errors) >= max_errors:
                    return errors

    elif isinstance(value, list):
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for idx, item in enumerate(value):
                validate_json_schema(
                    item, items_schema, root_schema, f"{path}[{idx}]", errors, max_errors
                )
                if len(errors) >= max_errors:
                    return errors

    return errors


def format_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)
    return str(content)


def build_agent_prompt(body: dict) -> tuple[str, str, dict | None]:
    messages = body.get("messages", [])
    rf = body.get("response_format") or {}
    js = rf.get("json_schema") if isinstance(rf, dict) else None
    schema_name = (js.get("name") if isinstance(js, dict) else None) or "completion"
    schema_obj = js.get("schema") if isinstance(js, dict) else None

    parts = [
        f"TASK: Execute the OpenAI-compatible structured completion `{schema_name}`.",
        "CRITICAL RULES:",
        "- Do NOT call any file, search, or codebase tools. All input data is below.",
        "- Return ONLY a single valid JSON object (no markdown fences, no commentary) as a string to `finish`.",
        "- Strictly follow every instruction, constraint, exact block ID, enum value, and EVERY required key (including nested objects and list elements) from the JSON schema.",
        "",
    ]

    for idx, msg in enumerate(messages, 1):
        role = str(msg.get("role", "user")).upper()
        content = format_message_content(msg.get("content", ""))
        parts.append(f"=== MESSAGE {idx} ({role}) ===")
        parts.append(content)
        parts.append("")

    if schema_obj is not None:
        parts.append("=== REQUIRED JSON SCHEMA ===")
        parts.append(json.dumps(schema_obj, ensure_ascii=False))
        parts.append("")

    parts.append(
        "Now produce ONLY the raw JSON object matching the schema and submit it via `finish`."
    )
    return "\n".join(parts), schema_name, schema_obj


async def read_http_request(reader: asyncio.StreamReader):
    header_bytes = await reader.readuntil(b"\r\n\r\n")
    header_text = header_bytes.decode("iso-8859-1")
    lines = header_text.split("\r\n")
    parts = lines[0].split(" ")
    method = parts[0]
    path = parts[1] if len(parts) > 1 else "/"
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    if headers.get("transfer-encoding", "").lower() == "chunked":
        chunks = []
        while True:
            size_line = await reader.readline()
            size = int(size_line.strip().split(b";", 1)[0], 16)
            if size == 0:
                await reader.readline()
                break
            chunk = await reader.readexactly(size)
            chunks.append(chunk)
            await reader.readexactly(2)
        body_bytes = b"".join(chunks)
    else:
        length = int(headers.get("content-length", "0"))
        body_bytes = await reader.readexactly(length) if length > 0 else b""
    return method, path, headers, body_bytes


async def send_http_json(writer: asyncio.StreamWriter, status_code: int, reason: str, payload: dict):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    header = (
        f"HTTP/1.1 {status_code} {reason}\r\n"
        f"Content-Type: application/json; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode("ascii")
    writer.write(header + body)
    await writer.drain()


async def run_cmd(cmd: list[str], label: str) -> str:
    log(f"Running {label}: {' '.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(REPO_ROOT),
    )
    out_b, err_b = await proc.communicate()
    out = out_b.decode("utf-8", errors="replace")
    err = err_b.decode("utf-8", errors="replace")
    log(f"[{label}] exit={proc.returncode}")
    if out.strip():
        log(f"[{label} stdout]\n{out.strip()[:3000]}")
    if err.strip():
        log(f"[{label} stderr]\n{err.strip()[:3000]}")
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}:\n{err or out}")
    return out


# --- Workflow parameters ---
smoke_test = bool(args.get("smoke_test", False))
work_dir = Path(str(args.get("work_dir") or (REPO_ROOT / ".linguary_subagent_work"))).expanduser()
work_dir.mkdir(parents=True, exist_ok=True)

data_root = work_dir / "linguary_data"
config_path = work_dir / "linguary_bridge.toml"
metrics_path = work_dir / "bridge_metrics.json"

bind_port = int(args.get("port", 18787))
if bind_port <= 0 and not smoke_test:
    raise ValueError("Use a fixed non-zero `port` (e.g. 18787) for Linguary runs so resumes match pinned BookConditions.")

reader_level = str(args.get("reader_level", "a2")).strip().lower()
concurrency = int(args.get("concurrency", 3))
timeout_seconds = float(args.get("timeout_seconds", 600))
max_attempts = int(args.get("max_attempts", 2))
sequential_analysis = bool(args.get("sequential_analysis", True))
auto_accept_glossary = bool(args.get("auto_accept_glossary", True))
run_epubcheck = bool(args.get("run_epubcheck", True))
resume_from = str(args.get("resume_from", "") or "").strip().lower()

narrative_model = tier_to_model_name(args.get("narrative_tier", "flash"))
pedagogical_model = tier_to_model_name(args.get("pedagogical_tier", "pro"))
processing_model = tier_to_model_name(args.get("processing_tier", "pro"))
correction_model = tier_to_model_name(args.get("correction_tier", "pro"))
default_model = tier_to_model_name(args.get("default_tier", "pro"))

call_counter = 0
metrics = []
if resume_from and metrics_path.exists():
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        call_counter = max((m.get("call_id", 0) for m in metrics), default=0)
    except Exception:
        metrics = []


async def handle_completion(body: dict) -> dict:
    global call_counter
    call_counter += 1
    cid = call_counter

    req_model = str(body.get("model") or default_model)
    tier = MODEL_TIER_MAP.get(req_model, "pro")
    prompt, schema_name, schema_obj = build_agent_prompt(body)
    t0 = time.monotonic()
    repairs = 0

    log(f"[bridge #{cid}] -> {schema_name} model={req_model} (tier={tier}) prompt_chars={len(prompt)}")

    worker = agent(
        role=f"LLM #{cid}: {schema_name} ({tier})",
        type_name="linguary-llm",
        model=tier,
    )

    try:
        raw = await worker(prompt, schema={"type": "string"})
        parsed = None
        last_err = None
        for attempt in range(3):
            try:
                parsed = extract_json_object(raw)
                schema_errors = (
                    validate_json_schema(parsed, schema_obj, schema_obj)
                    if isinstance(schema_obj, dict)
                    else []
                )
                if not schema_errors:
                    last_err = None
                    break
                last_err = "; ".join(schema_errors)
            except Exception as exc:
                last_err = f"JSON parse error: {exc}"

            if attempt < 2:
                repairs += 1
                log(f"[bridge #{cid}] local schema repair #{repairs}: {last_err[:300]}")
                raw = await worker(
                    f"Your previous JSON response failed schema validation:\n{last_err}\n\n"
                    "Fix EVERY listed error and return ONLY a single valid raw JSON object matching "
                    "the required JSON schema, with no markdown fences or extra text.",
                    schema={"type": "string"},
                )

        if last_err is not None:
            raise ValueError(f"Bridge failed to obtain schema-valid JSON after repairs: {last_err}")

        content_str = json.dumps(parsed, ensure_ascii=False)
        elapsed = round(time.monotonic() - t0, 2)
        metrics.append(
            {
                "call_id": cid,
                "schema_name": schema_name,
                "requested_model": req_model,
                "agent_tier": tier,
                "prompt_chars": len(prompt),
                "response_chars": len(content_str),
                "bridge_repairs": repairs,
                "elapsed_seconds": elapsed,
                "status": "ok",
            }
        )
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[bridge #{cid}] <- OK in {elapsed}s resp_chars={len(content_str)} repairs={repairs}")

        return {
            "id": f"chatcmpl-subagent-{cid}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content_str},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": max(1, len(prompt) // 4),
                "completion_tokens": max(1, len(content_str) // 4),
                "total_tokens": max(2, (len(prompt) + len(content_str)) // 4),
            },
        }
    except Exception as exc:
        elapsed = round(time.monotonic() - t0, 2)
        metrics.append(
            {
                "call_id": cid,
                "schema_name": schema_name,
                "requested_model": req_model,
                "agent_tier": tier,
                "prompt_chars": len(prompt),
                "response_chars": 0,
                "bridge_repairs": repairs,
                "elapsed_seconds": elapsed,
                "status": f"error: {exc}",
            }
        )
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[bridge #{cid}] <- ERROR in {elapsed}s: {exc}")
        raise
    finally:
        del worker


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        method, path, headers, body_bytes = await read_http_request(reader)
        if method == "GET" and path.rstrip("/").endswith("/models"):
            await send_http_json(
                writer,
                200,
                "OK",
                {"object": "list", "data": [{"id": m, "object": "model"} for m in MODEL_TIER_MAP]},
            )
            return
        if method == "POST" and path.rstrip("/").endswith("/chat/completions"):
            body = json.loads(body_bytes.decode("utf-8"))
            result = await handle_completion(body)
            await send_http_json(writer, 200, "OK", result)
            return
        await send_http_json(
            writer, 404, "Not Found", {"error": {"message": f"Unknown route: {method} {path}"}}
        )
    except Exception as exc:
        log(f"[bridge] HTTP handler error: {exc}\n{traceback.format_exc()[-600:]}")
        try:
            await send_http_json(
                writer, 500, "Internal Server Error", {"error": {"message": str(exc)}}
            )
        except Exception:
            pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


def write_config(port: int):
    data_root.mkdir(parents=True, exist_ok=True)
    toml = f"""
[llm]
provider = "subagent-bridge"
base_url = "http://127.0.0.1:{port}/v1"
model = "{default_model}"
narrative_analysis_model = "{narrative_model}"
pedagogical_analysis_model = "{pedagogical_model}"
staged_analysis = true
processing_model = "{processing_model}"
correction_model = "{correction_model}"
api_key = "local-subagent-bridge-key"
proxy_mode = "direct"
timeout_seconds = {timeout_seconds:g}
max_attempts = {max_attempts}
debug_llm_payloads = true

[checks]
auto_approve_glossary = true
auto_start_processing = false
sequential_analysis = {"true" if sequential_analysis else "false"}

[storage]
data_root = "{data_root}"

[worker]
concurrency = {concurrency}

[reading]
reader_level = "{reader_level}"
""".strip() + "\n"
    config_path.write_text(toml, encoding="utf-8")


# --- Main execution ---
phase("Start Local OpenAI Bridge")
server = await asyncio.start_server(handle_client, "127.0.0.1", bind_port)
actual_port = server.sockets[0].getsockname()[1]
log(f"Bridge listening on http://127.0.0.1:{actual_port}/v1")

try:
    if smoke_test:
        phase("Smoke Test")
        test_code = f"""
import urllib.request, json
req = urllib.request.Request(
    "http://127.0.0.1:{actual_port}/v1/chat/completions",
    data=json.dumps({{
        "model": "subagent-flash",
        "messages": [
            {{"role": "system", "content": "Return JSON with nested item."}},
            {{"role": "user", "content": "Say hello in Russian inside result.greeting."}}
        ],
        "response_format": {{
            "type": "json_schema",
            "json_schema": {{
                "name": "smoke_test",
                "strict": True,
                "schema": {{
                    "type": "object",
                    "$defs": {{
                        "Inner": {{
                            "type": "object",
                            "properties": {{"greeting": {{"type": "string"}}}},
                            "required": ["greeting"],
                            "additionalProperties": False
                        }}
                    }},
                    "properties": {{"result": {{"$ref": "#/$defs/Inner"}}}},
                    "required": ["result"],
                    "additionalProperties": False
                }}
            }}
        }}
    }}).encode("utf-8"),
    headers={{"Content-Type": "application/json", "Authorization": "Bearer test"}}
)
with urllib.request.urlopen(req, timeout=120) as resp:
    print(resp.read().decode("utf-8"))
"""
        out = await run_cmd(["uv", "run", "python", "-c", test_code], "smoke_test")
        print("SMOKE_OK:", out)
    else:
        epub_path = Path(str(args["epub_path"])).expanduser()
        if not epub_path.is_file():
            raise FileNotFoundError(f"Input EPUB not found: {epub_path}")
        book_id = str(args.get("book_id") or re.sub(r"[^a-z0-9-]+", "-", epub_path.stem.lower()).strip("-"))
        output_epub = Path(str(args.get("output_epub") or (work_dir / f"{book_id}-{reader_level}-adapted.epub"))).expanduser()

        if not resume_from:
            if data_root.exists():
                shutil.rmtree(data_root)
            metrics.clear()
            call_counter = 0

        write_config(actual_port)

        base_cmd = [
            "uv", "run", "python", "-m", "linguary.cli",
            "--config", str(config_path),
            "--reader-level", reader_level,
        ]

        if not resume_from:
            phase("Doctor & Import")
            await run_cmd(base_cmd + ["doctor"], "doctor")
            await run_cmd(base_cmd + ["import", str(epub_path), "--book-id", book_id], "import")
            await run_cmd(base_cmd + ["worker", "--once"], "worker-import")

        if resume_from in ("", "analyze"):
            phase("Analyze Chapters")
            analyze_cmd = base_cmd + ["analyze", book_id]
            if args.get("english_help") is True:
                analyze_cmd.append("--english-help")
            elif args.get("english_help") is False:
                analyze_cmd.append("--no-english-help")
            if args.get("pronunciation_help") is True:
                analyze_cmd.append("--pronunciation-help")
            elif args.get("pronunciation_help") is False:
                analyze_cmd.append("--no-pronunciation-help")
            if args.get("pronunciation_variant"):
                analyze_cmd.extend(["--pronunciation-variant", str(args["pronunciation_variant"])])
            await run_cmd(analyze_cmd, "analyze")
            await run_cmd(base_cmd + ["worker", "--once"], "worker-analyze")

        if auto_accept_glossary and resume_from in ("", "analyze", "process"):
            phase("Approve Flagged Glossary Items")
            approve_code = f"""
import asyncio
from linguary.adapters.config import load_settings
from linguary.composition import build_application

settings = load_settings(config_path={str(config_path)!r})
app = build_application(settings=settings)
try:
    review = app.approve_glossary.review({book_id!r})
    pending = [i.source_text for i in review.items if i.requires_review and i.decision is None]
    if pending:
        app.approve_glossary.accept_all_awaiting({book_id!r})
        print(f"Auto-accepted {{len(pending)}} flagged glossary item(s): {{pending}}")
    else:
        print("No pending glossary items.")
finally:
    asyncio.run(app.aclose())
"""
            await run_cmd(["uv", "run", "python", "-c", approve_code], "approve-glossary")

        if resume_from in ("", "analyze", "process", "worker", "retry_failed"):
            phase("Process, Translate & Correct")
            if resume_from == "retry_failed":
                retry_code = f"""
import sqlite3
con = sqlite3.connect({str(data_root / "linguary.db")!r})
ids = [r[0] for r in con.execute("SELECT id FROM processing_tasks WHERE book_id=? AND status='failed'", ({book_id!r},))]
print("\\n".join(ids))
"""
                failed_out = await run_cmd(["uv", "run", "python", "-c", retry_code], "list-failed")
                for tid in [line.strip() for line in failed_out.splitlines() if line.strip()]:
                    await run_cmd(base_cmd + ["retry", tid], f"retry-{tid[:8]}")
            elif resume_from == "worker":
                await run_cmd(base_cmd + ["resume", book_id], "resume")
            else:
                await run_cmd(base_cmd + ["process", book_id], "process")
            await run_cmd(base_cmd + ["worker", "--once"], "worker-process")

        phase("Export & Validate")
        output_epub.parent.mkdir(parents=True, exist_ok=True)
        await run_cmd(base_cmd + ["export", book_id, "--output", str(output_epub)], "export")
        if run_epubcheck:
            await run_cmd(["epubcheck", str(output_epub)], "epubcheck")

        print(
            json.dumps(
                {
                    "book_id": book_id,
                    "reader_level": reader_level,
                    "exported_epub": str(output_epub),
                    "total_llm_calls": len(metrics),
                    "metrics_path": str(metrics_path),
                    "config_path": str(config_path),
                    "data_root": str(data_root),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
finally:
    server.close()
    await server.wait_closed()
    log("Bridge stopped.")
