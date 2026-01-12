
.PHONY: test setup setup-dev build lint lint-fix

setup:
	./setup.sh

setup-dev: setup
	./venv/bin/pip install -r requirements-dev.txt

test: setup-dev
	./venv/bin/pytest tests/test_wake_word_filter.py -v

build: setup
	./build.sh

module: build
	cp dist/archive.tar.gz module.tar.gz

lint: setup-dev
	./venv/bin/python -m ruff check src/ tests/

lint-fix: setup-dev
	./venv/bin/python -m ruff check --fix src/ tests/
	./venv/bin/python -m ruff format src/ tests/
