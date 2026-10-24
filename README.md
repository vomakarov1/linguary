# Linguary

A local application for processing EPUB fiction books with LLMs: en→ru translation, vocabulary hints, and linguistic notes for language learners. The application runs locally on the user's machine, stores data in SQLite, and communicates with an external LLM provider over HTTP.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/) for dependency and environment management

## Installation

```sh
uv sync
```

This command creates a `.venv` virtual environment, installs dependencies according to `uv.lock`, and installs the `linguary` package in editable mode.

## Testing

```sh
uv run pytest -v
```

Testing conventions, directory layout, and instructions for running individual test suites are documented in the project's testing playbook.

## Project Structure

A modular monolith following Clean Architecture, where dependencies point strictly inward:
`domain` ← `application` ← `adapters` / `web` / `cli`.

| Package                | Contents                                                                  |
| ---------------------- | ------------------------------------------------------------------------- |
| `linguary.domain`      | Domain entities and invariants; pure Python without third-party libraries |
| `linguary.application` | Use cases and ports (Protocol/ABC), transactional boundaries              |
| `linguary.adapters`    | Port implementations: `db`, `epub`, `llm`, `config`, `validation`, `runtime` |
| `linguary.composition` | Dependency composition: the single place aware of concrete adapters       |
| `linguary.web`         | HTTP layer: FastAPI and server-side Jinja2 templates                      |
| `linguary.cli`         | Command-line interface entry point                                        |

Layer boundaries are enforced by architectural tests in `tests/architecture`: they inspect imports in `domain` and `application` modules and forbid dependencies on external libraries, standard library infrastructure modules, and outer layers.

## Running the Application

```sh
uv run python -m linguary.cli serve
```

This command starts the Web UI at the address specified in the `[server]` section of the local configuration — by default, `http://127.0.0.1:8000`. The application automatically migrates the database to the current schema, creates the data directory, and executes background tasks within its own process, so a separate worker process is not required for normal operation.

By default, the server is accessible only from the local machine: the application stores the provider API key and the full text of books. Exposing it on another network interface should only be done intentionally by changing `[server] host`: the application also uses this address to restrict the hostnames from which it accepts requests.

The application must be run from the repository source tree — including via the editable installation from `uv sync`. Database migrations reside in the `alembic/` directory alongside the package rather than inside it, so an independently built wheel does not include them.

### Running with Docker

The application can be run in a container using Docker Compose:

```sh
LINGUARY_LLM_API_KEY="your_api_key" docker compose up --build
```

The Web UI will be available at `http://localhost:8000`. The data directory and SQLite database are persisted in the `linguary_data` volume (`/data` inside the container).

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

Paths to local data, the LLM provider endpoint and API key, and processing limits are read from environment variables or a local configuration file (`~/.linguary/config.toml`) and are never stored in the repository code. Configuration sections:
- `[llm]` — provider, model, API key, and limits;
- `[checks]` — thresholds for automated checks;
- `[storage]` — data directory and database file path;
- `[server]` — address and port for the local Web UI;
- `[worker]` — background task concurrency.
Any value can be overridden with an environment variable formatted as `LINGUARY_<SECTION>_<SETTING>`.

Processing throughput is configured via two settings:

- `[worker] concurrency` (`LINGUARY_WORKER_CONCURRENCY`) — the number of tasks executed concurrently (default: `1`). This speeds up chapter processing, as chapters are independent of one another. Chapter analysis is not accelerated by this setting: chapters are analyzed strictly sequentially because each chapter needs to see glossary candidates from preceding ones;
- `[llm] requests_per_minute` (`LINGUARY_LLM_REQUESTS_PER_MINUTE`) — how many requests per minute the application sends to the provider across all workers combined. Unset by default, meaning request rate is not limited. Set this according to your provider rate limits if `concurrency` greater than 1 results in `429` rate-limit errors. The application always respects the delay specified by the provider in the `Retry-After` header for `429` and `503` responses before retrying.

```toml
[worker]
concurrency = 4

[llm]
requests_per_minute = 60
```

While analysis or processing is in progress, the book card displays all currently running chapters and an estimated remaining time. The estimate is an approximation: the median model call duration for the book multiplied by remaining work and divided by concurrency; it is not shown until the first call completes.

The application displays its current configuration status on the configuration and environment screen (`GET /system/config`): whether the LLM is configured, which model is active, whether external EPUBCheck validation is available, and where data is stored. The provider API key is never included in the response.
