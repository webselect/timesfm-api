PYTHON ?= /opt/homebrew/bin/python3.12
VENV := .venv
BIN := $(VENV)/bin

# Certaines operations PyTorch ne sont pas implementees en Metal : bascule sur CPU
# au lieu de lever une exception.
export PYTORCH_ENABLE_MPS_FALLBACK := 1

.PHONY: setup run test test-model lint fmt openapi clean

setup: ## Cree le venv et installe les dependances
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev]"

run: ## Demarre le service (telecharge les poids au premier lancement)
	$(BIN)/uvicorn app.main:app --host $${API_HOST:-127.0.0.1} --port $${API_PORT:-8000}

dev: ## Idem avec rechargement automatique
	$(BIN)/uvicorn app.main:app --reload --host $${API_HOST:-127.0.0.1} --port $${API_PORT:-8000}

test: ## Suite rapide, sans les poids
	$(BIN)/python -m pytest

test-model: ## Tests d integration avec les vrais poids TimesFM (lent)
	$(BIN)/python -m pytest -m model -o addopts="" -v

lint: ## Verifie le style
	$(BIN)/ruff check .

fmt: ## Corrige et formate
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

openapi: ## Exporte le schema OpenAPI (pour generer un client TypeScript)
	$(BIN)/python -c "import json; from app.main import create_app; print(json.dumps(create_app().openapi(), indent=2, ensure_ascii=False))" > openapi.json
	@echo "openapi.json ecrit"

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache openapi.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
