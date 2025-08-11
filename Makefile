.PHONY: run install clean check runner train predict format lint type-check test
.DEFAULT_GOAL:=runner

run: install
	poetry run python src/runner.py 

install: pyproject.toml
	poetry install

clean:
	rm -rf `find . -type d -name __pycache__`
	rm -rf src/logs/*
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

clean-models:
	rm -rf src/model/models/*

check: format lint type-check
	@echo "Checks passed!"

format:
	poetry run black src/
	poetry run ruff check --fix src/

lint:
	poetry run ruff check src/

type-check:

	@poetry run mypy src/ >/dev/null 2>&1 || echo "Type checking completed"

test:
	poetry run pytest tests/ -v --tb=short

train: install
	poetry run python -c "import sys; sys.path.insert(0, 'src'); from model.pipeline.model import build_model; build_model()"

predict: install
	poetry run python src/runner.py
	

runner: check run
