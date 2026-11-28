---
name: linguary-subagent-run
description: >-
  Run end-to-end Linguary EPUB book processing (import, analyze, glossary approval,
  translate/process, chapter correction, export, epubcheck) locally without OpenRouter
  by intercepting OpenAI-compatible HTTP requests and fulfilling them with Antigravity
  subagents via run_workflow. Use when the user asks to process, translate, adapt, or
  test a book in Linguary using local subagents instead of external OpenRouter models.
---

# Linguary Subagent Run (Local OpenAI Bridge)

Runs the full Linguary CLI pipeline locally without OpenRouter by starting an in-process
OpenAI-compatible HTTP server (`POST /v1/chat/completions`) inside `run_workflow` and
dispatching each LLM request to a dedicated stateless subagent (`linguary-llm`).

No changes to Linguary source code are required: the workflow writes an isolated TOML
config pointing `[llm].base_url` to `http://127.0.0.1:<port>/v1` with `proxy_mode = "direct"`
and an isolated `storage.data_root` so `~/.linguary` is never modified.

## Prerequisites

1. Run from the Linguary repository root (`/Users/vladislavmakarov/www/linguary`) so `uv run`
   and Alembic migrations resolve cleanly.
2. Read the `run-workflow` skill if you have not already loaded it in this session.
3. Ensure `define_subagent` has registered the stateless `linguary-llm` subagent in the
   current conversation before launching the workflow.

## Model Tier Presets

Linguary routes each pipeline stage by model name in `config.toml`; the bridge maps model
names to Antigravity `run_workflow` agent tiers (`flash_lite`, `flash`, `pro`, `inherit`):

| Preset | Narrative analysis | Pedagogical analysis | Processing / Translation | Chapter correction | When to use |
| --- | --- | --- | --- | --- | --- |
| **Balanced (default)** | `flash` | `pro` | `pro` | `pro` | Best A2–C1 literary & pedagogical quality (1–3 chapters) |
| **Fast** | `flash` | `flash` | `pro` | `pro` | ~1.5–2× faster analysis while keeping `pro` for Russian translation & QA |
| **Debug / Smoke** | `flash` | `flash` | `flash` | `flash` | Fast pipeline/bug reproduction without paying `pro` latency |

## Steps

### 1. Define the `linguary-llm` subagent

Call `define_subagent` once per conversation:

- `Name`: `"linguary-llm"`
- `EnableWriteTools`: `false`
- `EnableMCPTools`: `false`
- `EnableSubagentTools`: `false`
- `Description`: `"Stateless LLM completion backend for Linguary OpenAI-compatible bridge. Never calls codebase tools; immediately produces the exact requested structured JSON."`
- `SystemPrompt`:
  > You are acting strictly as a stateless LLM completion backend for the Linguary book-adaptation pipeline.
  > Rules:
  > 1. NEVER call any codebase, file, search, or shell tools. Every single piece of source text, context, instructions, and schema is already provided inside the user prompt.
  > 2. Follow the system instructions and user input from Linguary carefully and accurately.
  > 3. Produce ONLY the exact raw JSON object matching the required JSON schema and immediately submit it via the `finish` tool. Do not wrap in markdown fences or add commentary.
  > 4. Double-check all required fields (including nested objects/arrays), exact block IDs, enum values, natural Russian phrasing, and JSON syntax before calling `finish`.

### 2. Optional 5-second smoke test

Before a long book run, verify the bridge and subagent routing with `smoke_test: true`:

```json
{
  "smoke_test": true,
  "work_dir": "<conversation_scratch_dir>/linguary_run",
  "port": 18787
}
```

Call `run_workflow` with:
- `ScriptPath`: `/Users/vladislavmakarov/www/linguary/.agents/skills/linguary-subagent-run/scripts/subagent_bridge_workflow.py`
- `Cwd`: `/Users/vladislavmakarov/www/linguary`
- `WaitMsBeforeAsync`: `10000`

Confirm output contains `SMOKE_OK` and valid JSON.

### 3. Launch full book processing

Call `run_workflow` with `ScriptPath` pointing to
`./scripts/subagent_bridge_workflow.py` (using its absolute path) and `Args`:

```json
{
  "epub_path": "/abs/path/to/book.epub",
  "output_epub": "/abs/path/to/book-adapted.epub",
  "work_dir": "/abs/path/to/scratch/linguary_run",
  "book_id": "my-book-a2",
  "reader_level": "a2",
  "port": 18787,
  "concurrency": 3,
  "narrative_tier": "flash",
  "pedagogical_tier": "pro",
  "processing_tier": "pro",
  "correction_tier": "pro",
  "auto_accept_glossary": true,
  "sequential_analysis": true,
  "run_epubcheck": true
}
```

Key argument notes:
- **`port` (default `18787`):** ALWAYS keep a fixed deterministic port across retries/resumes!
  Linguary pins `base_url` (including port) inside `book_conditions` and `work_bindings`;
  changing the port mid-run triggers `route_unavailable`.
- **`auto_accept_glossary` (default `true`):** After `worker-analyze`, the script invokes
  `uv run python -m linguary.cli glossary accept-all <book_id> --json` before `process` so
  any model-flagged terms are approved hands-free. Set `false` if you want to inspect items
  manually via `linguary glossary list <book_id> --pending-only`.
- **`resume_from`:** Optional (`""` = fresh run wiping `work_dir/linguary_data`, `"analyze"`,
  `"process"`, `"worker"`, `"retry_failed"`, `"export"`). Preserves existing `linguary.db`
  and appends to `bridge_metrics.json`.
- **`english_help` / `pronunciation_help` / `pronunciation_variant`:** Optional flags forwarded
  to `linguary analyze` when explicitly set (`true`/`false`, `"uk"`/`"us"`).

### 4. Verify output and inspect telemetry

When the workflow completes, check:
1. `output_epub` exists and passed `epubcheck` (`0 fatals / 0 errors / 0 warnings`).
2. `work_dir/bridge_metrics.json` — call counts, tier distribution, latencies, and local
   JSON Schema repair counts.
3. `work_dir/linguary_data/linguary.db` — verify `books.status == 'COMPLETED'`, check
   `processing_tasks`, `llm_call_logs` (for any first-attempt semantic validation retries),
   `correction_candidates`, and `result_warnings`.

## Troubleshooting

- **`Address already in use` on `port`:** Pick another fixed port (e.g. `18788`) before
  starting a fresh run; when resuming an existing `work_dir`, reuse the exact port stored in
  `work_dir/linguary_bridge.toml`.
- **Long pedagogical calls timing out:** Increase `timeout_seconds` (default `600`) or switch
  `pedagogical_tier` to `"flash"`.
- **Unresolved `awaiting_human` correction candidates:** `linguary export` prints a warning
  to stderr and still writes the EPUB (pass `--strict-corrections` to fail instead). Inspect
  and resolve candidates via `linguary corrections list <book_id> --awaiting-only` and
  `linguary corrections resolve <book_id> <candidate_id> --action {accept|keep}`.
