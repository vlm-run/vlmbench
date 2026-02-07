default: help

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  clean       Remove build artifacts"
	@echo "  lint        Run pre-commit hooks"
	@echo "  test        Run tests"
	@echo "  dist        Build distribution"

clean: clean-build clean-pyc

clean-build:
	rm -rf build/ dist/ *.egg-info .eggs/ site/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -delete

clean-pyc:
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} +

lint:
	pre-commit run --all-files

test:
	pytest -sv tests

dist: clean
	python -m build --sdist --wheel

.PHONY: default help clean clean-build clean-pyc lint test dist
