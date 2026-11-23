# Linguary

A local application for adapting EPUB fiction books for language learners using OpenAI-compatible LLMs: en→ru translation, vocabulary and linguistic notes, optional simple English explanations, and optional IPA pronunciation hints. It stores state in SQLite and communicates with an external or self-hosted LLM provider over HTTP.

## Features

- **CEFR-adapted reading (`a2`, `b1`, `b2`, `c1`)**: plans block and per-segment treatments inside paragraphs, combining original English text, selective Russian translations, and numbered inline notes.
- **Staged & sequential chapter analysis**: runs separate narrative (scene mapping) and pedagogical analysis passes by default with independent pass caching and optional per-stage model overrides. Chapters are analyzed sequentially by default (`sequential_analysis = true`) so each chapter sees cumulative glossary candidates and prior summaries; disabling it enables parallel chapter analysis with an optional seed-first chapter wave (`seed_first_chapter`).
- **Glossary & name translation**: extracts characters, places, and recurring terms with configurable name strategies (`sound` or `meaning`), interactive glossary review in the Web UI, and optional automatic approval and processing start.
- **Optional English & pronunciation help**: supports per-book simple English explanations and UK/US IPA pronunciation hints backed by pinned local `ipa-dict` snapshots.
- **Resilient execution & quality workflow**: supports per-book pause/resume, bounded automatic translation recovery, manual block regeneration, chapter correction candidates, and optional quality review.
- **Reader-ready EPUB export**: builds adapted EPUB files with a book legend, optional IPA transcription key, and optional validation via EPUBCheck.

## Requirements

- Python 3.15+
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- Optional: Java and [EPUBCheck](https://www.w3.org/publishing/epubcheck/) for local EPUB validation (auto-detected in `~/.linguary/epubcheck`, Homebrew's `epubcheck` formula paths, and `/usr/share/java/epubcheck`; preinstalled in the Docker image)

## Installation

```sh
uv sync
```

This command creates a `.venv` virtual environment, installs dependencies according to `uv.lock`, and installs the `linguary` package in editable mode.

## Running the Application

```sh
uv run python -m linguary.cli serve
```

This starts the Web UI at the address configured in `[server]` — by default, `http://127.0.0.1:8000`. The application automatically migrates the SQLite database to the current schema, creates the data directory, and runs background tasks in-process, so a separate worker process is not required for normal operation. Pass `--reload` during development to restart the server on source changes.

Before analyzing or processing a book, configure your LLM model and API key via the Web UI (`/system/config`), environment variables (`LINGUARY_LLM_MODEL` and `LINGUARY_LLM_API_KEY`), or `~/.linguary/config.toml`.

By default, the server is accessible only from the local machine because the application stores the provider API key and full book texts. Exposing it on another interface should only be done intentionally via `[server] host`, which also controls the `Host` headers accepted by the Web UI.

The application must be run from the repository source tree — including via the editable installation from `uv sync`. Database migrations live in the top-level `alembic/` directory alongside the package rather than inside it, so a standalone wheel does not include them.

### CLI Commands

All commands accept global `--config <path>` and `--reader-level {a2,b1,b2,c1}` flags:

```sh
uv run python -m linguary.cli [--config PATH] [--reader-level LEVEL] <command>
```

| Command | Purpose |
| --- | --- |
| `serve [--reload]` | Start the local Web UI with the built-in background worker |
| `doctor` | Print local configuration readiness (LLM, EPUBCheck, paths, server, reader level) without secrets |
| `import <epub> [--book-id ID] [--source-language en] [--target-language ru]` | Register an EPUB file and enqueue its import task |
| `analyze <book_id> [--[no-]english-help] [--[no-]pronunciation-help] [--pronunciation-variant uk\|us] [--[no-]seed-first-chapter]` | Pin optional book conditions and enqueue chapter analysis tasks |
| `process <book_id>` | Freeze the approved glossary and enqueue chapter processing tasks |
| `pause <book_id>` / `resume <book_id>` | Pause or resume claiming queued tasks for a book |
| `restart-translation <book_id> <root_id> --expected-generation N` | Restart unfinished scopes of a translation recovery root |
| `worker [--once]` | Run a standalone local task queue worker (`--once` drains the queue and exits) |
| `retry <task_id>` | Re-enqueue a failed task |
| `export <book_id> --output PATH [--mode full\|structure_only]` | Assemble an EPUB (`full` by default, or `structure_only` for technical checks) |
| `diagnostics {list\|show\|purge} <book_id>` | Inspect or purge stored diagnostic calls locally without invoking an LLM (`show` takes `<call_id>`; `purge` accepts `--call-id ID`) |

### Running with Docker

The Docker image bundles Java and EPUBCheck 5.4.0 alongside the application. You can run it with Docker Compose (defaults `LINGUARY_LLM_MODEL` to `gpt-4o-mini` if unset):

```sh
LINGUARY_LLM_API_KEY="your_api_key" docker compose up --build
```

The Web UI is bound to `http://127.0.0.1:8000` (`http://localhost:8000`). The data directory and SQLite database are persisted in the `linguary_data` volume (`/data` inside the container). Note that `docker-compose.yml` mounts `./src` and `./alembic` and runs `serve --reload` for local development.

Direct execution with Docker is also supported:

```sh
docker build -t linguary .
docker run -d \
  -p 127.0.0.1:8000:8000 \
  -e LINGUARY_LLM_API_KEY="your_api_key" \
  -e LINGUARY_LLM_MODEL="gpt-4o-mini" \
  -v linguary_data:/data \
  linguary
```

## Configuration

Settings are resolved in order of increasing precedence:

1. Built-in defaults
2. Local TOML configuration file (`~/.linguary/config.toml`, or an explicit path passed via `--config` / `LINGUARY_CONFIG`, which must exist)
3. Environment variables formatted as `LINGUARY_<SECTION>_<SETTING>`
4. Editable overrides saved through the Web UI (`/system/config`) and stored in SQLite (resettable via `POST /system/config/reset`)

Secrets such as the LLM API key and proxy credentials are never stored in the repository or returned by API responses.

Configuration sections:

- `[llm]` — LLM provider, models, transport, and limits:
  - Provider & models: `provider`, `base_url`, `api_key`, default `model`, optional per-stage overrides (`analysis_model`, `narrative_analysis_model`, `pedagogical_analysis_model`, `processing_model`, `correction_model`), and `staged_analysis` (default `true`). For best translation and note quality during chapter correction, configure `correction_model` to a stronger model than `processing_model` (or the default `model`);
  - Proxy & call limits: `proxy_mode` (`auto` / `direct` / `custom`), `proxy_url`, `timeout_seconds`, `max_output_tokens`, `temperature`, `max_attempts`, `max_fragment_blocks`, `max_unit_source_chars`, `requests_per_minute`, `debug_llm_payloads`;
  - Local cost estimates: optional `input_price_per_mtok` and `output_price_per_mtok`.
- `[checks]` — thresholds for automated checks (`glossary_confidence_threshold`, `glossary_review_min_mentions`, `glossary_review_min_chapters`, `target_language_min_share`, `language_check_min_length`, `repetition_ngram_size`, `repetition_min_repeats`, `language_notes_per_chapter`) and workflow automation flags (`auto_approve_glossary`, `auto_start_processing`, `sequential_analysis` — default `true`, `seed_first_chapter`);
- `[storage]` — `data_root` (default `~/.linguary`) and optional `database_path` (defaults to `<data_root>/linguary.db`);
- `[server]` — `host` (default `127.0.0.1`) and `port` (default `8000`) for the local Web UI;
- `[worker]` — background task `concurrency` (default `1`);
- `[reading]` — target CEFR `reader_level` (`a2`, `b1`, `b2`, `c1`; default `b2`) and `name_translation_strategy` (`sound` or `meaning`; default `sound`);
- `[diagnostics]` — retention and size limits for stored diagnostic call bodies (`bodies_enabled`, `max_body_bytes`, `max_book_bytes`, `retention_seconds`);
- `[recovery]` — automatic translation recovery bounds (`max_calls`, `max_scope_calls`, `max_depth`, `max_leaves`).

Additional environment variables outside TOML sections:

- `LINGUARY_CONFIG` — explicit path to the TOML configuration file (overridden by CLI `--config`);
- `LINGUARY_QUALITY_ROUTES` — path to a local JSON registry of attested model routes required to enable optional quality review.

### Throughput & Progress Estimation

Processing throughput is controlled primarily by two settings:

- `[worker] concurrency` (`LINGUARY_WORKER_CONCURRENCY`) — number of queue tasks executed concurrently (default: `1`). Increasing concurrency speeds up chapter processing and, when `[checks] sequential_analysis = false` (`LINGUARY_CHECKS_SEQUENTIAL_ANALYSIS=false`), chapter analysis. By default, `sequential_analysis = true` analyzes chapters one by one with cumulative glossary candidates and prior chapter summaries in context. When `sequential_analysis` is disabled, all unanalyzed chapters are queued in parallel, and enabling `[checks] seed_first_chapter = true` (`LINGUARY_CHECKS_SEED_FIRST_CHAPTER=true`) analyzes the first story chapter first and then runs the remaining chapters in parallel with its glossary candidates in context.
- `[llm] requests_per_minute` (`LINGUARY_LLM_REQUESTS_PER_MINUTE`) — request rate limit sent to the provider across all workers combined. Unset by default (unthrottled). Set this according to your provider limits if `concurrency > 1` triggers `429` rate-limit errors. The client always respects the `Retry-After` header on `429` and `5xx` responses before retrying.

```toml
[worker]
concurrency = 4

[llm]
requests_per_minute = 60
```

While analysis or processing is running, the book card displays all currently active chapters and an approximate remaining time estimate based on median call duration, extrapolated remaining work, effective concurrency, and any configured `requests_per_minute` limit (shown once the first chapter is analyzed or the first processing unit completes).

### Configuration Screen & Connection Check

The configuration screen (`GET /system/config`) — and `uv run python -m linguary.cli doctor` in the CLI — shows whether the LLM is ready, which model is active, whether external EPUBCheck validation is available, where data is stored, and the configured server address (the Web UI also warns if the server is exposed beyond loopback). In the Web UI, you can edit runtime settings (`POST /system/config`), test provider connectivity (`POST /system/config/test`), and reset saved overrides back to file/environment defaults.

## Testing

Run Ruff, ty (application code and Alembic migrations), and the architecture tests with:

```sh
make check
```

The command uses the dependencies pinned in `uv.lock`. GitHub Actions runs the same command on Python 3.15.

Run the full test suite with:

```sh
uv run pytest -v
```

Line coverage of the `linguary` package is a separate command. It is not part of the ordinary run:

```sh
uv run pytest --cov=linguary --cov-fail-under=90
```

To skip tests marked `slow` (e.g., large EPUBs or real-LLM tests):

```sh
uv run pytest -m "not slow"
```

Tests are organized by scope:

- `tests/unit` — isolated domain, application, layout, and adapter unit tests
- `tests/integration` — database, EPUB, processing and task-queue, CLI, and Web UI integration tests
- `tests/e2e` — end-to-end processing and export pipelines
- `tests/architecture` — static import-boundary checks across layers
- `tests/fixtures` — shared EPUB and test data fixtures

## Project Structure

A modular monolith following Clean Architecture, where dependencies point strictly inward:
`domain` ← `application` ← `adapters` / `web` / `cli`.

| Package | Contents |
| --- | --- |
| `linguary.domain` | Domain entities, policies, and invariants; pure Python without third-party libraries |
| `linguary.application` | Use cases, ports (`Protocol`), layout builders, and transactional boundaries |
| `linguary.adapters` | Port implementations: `db`, `epub`, `llm`, `config`, `pronunciation`, `validation`, `runtime` |
| `linguary.composition` | Dependency composition root: wires application use cases and task queues to concrete adapters |
| `linguary.web` | HTTP layer: FastAPI routes, schemas, and server-side Jinja2 templates |
| `linguary.cli` | Command-line interface entry point |
| `linguary.comparison` | Local comparison harness for controlled model/variant evaluation and anonymized blind-review reports |

Layer boundaries are enforced by architectural tests in `tests/architecture`: they inspect AST imports and file calls in `domain` and `application` to forbid third-party libraries, infrastructure stdlib modules, and outer layers, and verify that `adapters`, `composition`, and `web` do not depend on outer or sibling layers.
