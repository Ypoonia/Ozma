.PHONY: format lint test check

format:
	.venv/bin/python -m ruff format src tests
	.venv/bin/python -m ruff check --fix src tests

lint:
	.venv/bin/python -m ruff format --check src tests
	.venv/bin/python -m ruff check src tests

test:
	.venv/bin/python -m pytest

check: lint test
