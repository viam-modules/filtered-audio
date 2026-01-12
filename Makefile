
.PHONY: test setup setup-dev

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
