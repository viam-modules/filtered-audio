
.PHONY: test setup setup-dev

setup:
	./setup.sh

setup-dev:
	pip install -r requirements-dev.txt

test: setup-dev
	pytest tests/test_wake_word_filter.py -v

build: setup
	./build.sh

module: build
	cp dist/archive.tar.gz module.tar.gz
