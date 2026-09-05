PYTHON ?= /opt/homebrew/bin/python3.12
VENV := .venv
BIN := $(VENV)/bin

# Some PyTorch operations are not implemented in Metal: route them to the CPU
# instead of raising.
export PYTORCH_ENABLE_MPS_FALLBACK := 1

.PHONY: setup run test test-model lint fmt openapi clean

setup: ## Create the venv and install dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev]"

run: ## Start the service (downloads the weights on first launch)
	$(BIN)/uvicorn app.main:app --host $${API_HOST:-127.0.0.1} --port $${API_PORT:-8000}

dev: ## Same, with auto-reload
	$(BIN)/uvicorn app.main:app --reload --host $${API_HOST:-127.0.0.1} --port $${API_PORT:-8000}

test: ## Fast suite, no weights needed
	$(BIN)/python -m pytest

test-model: ## Integration tests against the real TimesFM weights (slow)
	$(BIN)/python -m pytest -m model -o addopts="" -v

lint: ## Check style
	$(BIN)/ruff check .

fmt: ## Fix and format
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

openapi: ## Export the OpenAPI schema (to generate a TypeScript client)
	$(BIN)/python -c "import json; from app.main import create_app; print(json.dumps(create_app().openapi(), indent=2, ensure_ascii=False))" > openapi.json
	@echo "openapi.json written"

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache openapi.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
