.PHONY: check

check:
	uv run --frozen ruff check .
	uv run --frozen ty check
	uv run --frozen pytest tests/architecture
