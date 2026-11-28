# Linguary

A local Python application for adapting English EPUB fiction for language learners using OpenAI-compatible LLMs: en→ru translation, vocabulary and linguistic notes, optional simple English explanations, and optional IPA pronunciation hints. State is stored in SQLite; LLM requests go over HTTP to an external or self-hosted provider.

## Features

- **CEFR-adapted reading (`a2`, `b1`, `b2`, `c1`)**: combines original English text, selective Russian translations, and numbered inline notes per paragraph or segment.
- **Staged chapter analysis**: runs separate narrative and pedagogical passes by default with pass caching, sequential context accumulation, and per-stage model overrides.
- **Glossary & name strategies**: extracts characters, places, and recurring terms (`sound` or `meaning` strategy) with review via Web UI, CLI, or automatic approval.
- **Optional English & IPA help**: per-book simple English explanations and UK/US IPA hints backed by pinned local `ipa-dict` snapshots.
- **Resilient processing & QA**: per-book pause/resume, bounded automatic translation recovery, manual block regeneration, chapter correction review, optional quality review, and optional illustration prompt preparation.
- **Reader-ready EPUB export**: generates adapted EPUBs with a book legend, optional IPA key, and optional EPUBCheck validation.

## Requirements & Installation

- Python 3.15+
- [uv](https://docs.astral.sh/uv/)
- Optional: Java + [EPUBCheck](https://www.w3.org/publishing/epubcheck/) (auto-detected in `~/.linguary/epubcheck`, Homebrew paths, and `/usr/share/java/epubcheck`; preinstalled in Docker)

```sh
uv sync
```

This creates `.venv`, installs pinned dependencies from `uv.lock`, and installs `linguary` in editable mode. Run the application from the repository root so top-level Alembic migrations (`alembic/`) resolve properly.

## Quick Start

```sh
uv run python -m linguary.cli serve
```

Starts the local Web UI at `http://127.0.0.1:8000` (configurable in `[server]`), migrates SQLite automatically, and runs the background worker in-process. Pass `--reload` during development.

Before analyzing a book, configure an LLM model and API key via the Web UI (`/system/config`), environment variables (`LINGUARY_LLM_MODEL`, `LINGUARY_LLM_API_KEY`), or `~/.linguary/config.toml`. Use `/system/config` (or `cli doctor`) to check readiness and test provider connectivity (`POST /system/config/test`).

> **Local-only default:** Because Linguary stores API keys and full book texts, it binds to `127.0.0.1` by default. Exposing it on another interface requires setting `[server] host` explicitly, which also controls accepted `Host` headers.

## CLI Reference

All commands accept global `--config PATH` and `--reader-level {a2,b1,b2,c1}` flags:

```sh
uv run python -m linguary.cli [--config PATH] [--reader-level LEVEL] <command>
```

| Command | Purpose |
| --- | --- |
| `serve [--reload]` | Start the Web UI with the built-in background worker |
| `doctor` | Check local configuration readiness (LLM, EPUBCheck, paths, server) without secrets |
| `import <epub> [--book-id ID] [--source-language en] [--target-language ru]` | Register an EPUB and enqueue import |
| `analyze <book_id> [--[no-]english-help] [--[no-]pronunciation-help] [--pronunciation-variant uk\|us\|auto] [--[no-]seed-first-chapter]` | Pin book options and enqueue chapter analysis |
| `glossary {list\|accept-all} <book_id>` | Inspect (`--pending-only`) or approve unresolved glossary items (`--json`) |
| `process <book_id>` | Freeze the approved glossary and enqueue chapter processing |
| `pause <book_id>` / `resume <book_id>` | Pause or resume claiming queued tasks for a book |
| `worker [--once]` | Run a standalone task queue worker (`--once` drains the queue and exits) |
| `retry <task_id> [--clear-cache] [--acknowledge-unknown-outcome]` | Re-enqueue a failed task (optionally clearing cached analysis passes or acknowledging unknown correction outcomes) |
| `restart-translation <book_id> <root_id> --expected-generation N [--acknowledge-unknown-outcome]` | Restart unfinished scopes of a translation recovery root |
| `corrections {list\|resolve} <book_id>` | List (`--awaiting-only`) or resolve (`<candidate_id> --action accept\|keep`) chapter correction candidates |
| `export <book_id> --output PATH [--mode full\|structure_only] [--strict-corrections]` | Assemble an EPUB (`--strict-corrections` fails if `awaiting_human` candidates remain) |
| `diagnostics {list\|show\|purge} <book_id>` | Inspect or purge stored diagnostic calls locally (`show <call_id>`, `purge [--call-id ID]`) |
| `illustrations {settings\|prepare\|retry\|status\|show\|registry} <book_id>` | Manage optional book illustration prompt preparation |

## Running with Docker

The Docker image bundles Java and EPUBCheck 5.4.0.

**Docker Compose** (mounts `./src` and `./alembic` with `serve --reload`; defaults `LINGUARY_LLM_MODEL` to `gpt-4o-mini`):

```sh
LINGUARY_LLM_API_KEY="your_api_key" docker compose up --build
```

**Standalone container:**

```sh
docker build -t linguary .
docker run -d \
  -p 127.0.0.1:8000:8000 \
  -e LINGUARY_LLM_API_KEY="your_api_key" \
  -e LINGUARY_LLM_MODEL="gpt-4o-mini" \
  -v linguary_data:/data \
  linguary
```

Data and SQLite persist in the `linguary_data` volume (`/data`).

## Configuration

Settings resolve in order of increasing precedence:

1. Built-in defaults
2. TOML file (`~/.linguary/config.toml`, or `--config` / `LINGUARY_CONFIG`, which must exist)
3. Environment variables (`LINGUARY_<SECTION>_<SETTING>`)
4. Web UI overrides (`/system/config`) persisted in SQLite (reset via `POST /system/config/reset`)

Secrets (API keys, proxy credentials) are never committed or returned by API responses.

| Section | Key settings |
| --- | --- |
| `[llm]` | `provider`, `base_url`, `api_key`, `model`, `staged_analysis` (`true`), per-stage overrides (`analysis_model`, `narrative_analysis_model`, `pedagogical_analysis_model`, `processing_model`, `correction_model`), `proxy_mode` (`auto`/`direct`/`custom`), `proxy_url`, `timeout_seconds`, `max_output_tokens`, `reasoning_effort`, `reasoning_max_tokens`, `temperature`, `max_attempts`, `max_fragment_blocks`, `max_unit_source_chars`, `requests_per_minute`, `debug_llm_payloads`, `input_price_per_mtok`, `output_price_per_mtok` |
| `[checks]` | Quality thresholds (`glossary_confidence_threshold`, `glossary_review_min_mentions`, `glossary_review_min_chapters`, `target_language_min_share`, `language_check_min_length`, `repetition_ngram_size`, `repetition_min_repeats`, `language_notes_per_chapter`) and workflow flags (`auto_approve_glossary`, `auto_start_processing`, `sequential_analysis` — default `true`, `seed_first_chapter`) |
| `[storage]` | `data_root` (`~/.linguary`), `database_path` (defaults to `<data_root>/linguary.db`) |
| `[server]` | `host` (`127.0.0.1`), `port` (`8000`) |
| `[worker]` | `concurrency` (`1`) |
| `[reading]` | `reader_level` (`a2`/`b1`/`b2`/`c1`; default `b2`), `name_translation_strategy` (`sound`/`meaning`; default `sound`) |
| `[diagnostics]` | `bodies_enabled`, `max_body_bytes`, `max_book_bytes`, `retention_seconds` |
| `[recovery]` | `max_calls`, `max_scope_calls`, `max_depth`, `max_leaves` |

For best chapter correction quality, set `correction_model` to a stronger model than `processing_model` (or default `model`).

Extra environment variables outside TOML sections:
- `LINGUARY_CONFIG` — explicit TOML path (overridden by `--config`)
- `LINGUARY_QUALITY_ROUTES` — path to a local JSON registry of attested model routes for optional quality review

### Concurrency & Throughput

- **`[worker] concurrency`** (`1` by default) controls parallel tasks during chapter processing, and during analysis when `[checks] sequential_analysis = false`. By default, `sequential_analysis = true` runs chapters one by one with cumulative glossary and summary context; when disabled, `seed_first_chapter = true` seeds the first story chapter before running the rest in parallel.
- **`[llm] requests_per_minute`** (unset by default) caps provider requests across all workers. The client always respects `Retry-After` on `429`/`5xx` responses.

```toml
[worker]
concurrency = 4

[llm]
requests_per_minute = 60
```

Active chapters and an approximate remaining time estimate appear on the book card once the first chapter or processing unit completes.

## Testing & Checks

```sh
make check                                      # Ruff, ty, and architecture tests (runs in CI on Python 3.15)
uv run pytest -v                                # Full test suite
uv run pytest -m "not slow"                     # Skip slow tests (large EPUBs / live LLM)
uv run pytest --cov=linguary --cov-fail-under=90  # Coverage check (separate run)
```

Test layout: `tests/unit`, `tests/integration`, `tests/e2e`, `tests/architecture`, and `tests/fixtures`.

## Architecture

Clean Architecture modular monolith with strictly inward dependencies:
`domain` ← `application` ← `adapters` / `web` / `cli`.

| Package | Responsibility |
| --- | --- |
| `linguary.domain` | Entities, policies, invariants; pure Python (no third-party or infra stdlib imports) |
| `linguary.application` | Use cases, ports (`Protocol`), layout builders, transactional boundaries |
| `linguary.adapters` | Port implementations: `db`, `epub`, `llm`, `config`, `pronunciation`, `validation`, `runtime` |
| `linguary.composition` | Composition root wiring use cases and task queues to adapters |
| `linguary.web` | FastAPI routes, schemas, and Jinja2 templates |
| `linguary.cli` | CLI entry point |
| `linguary.comparison` | Model/variant comparison harness and blind-review reports |

Layer boundaries are enforced via AST import checks in `tests/architecture`.
