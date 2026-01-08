
.PHONY: test setup setup-dev

setup:
	pip install -r requirements.txt

setup-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/test_wake_word_filter.py -v
