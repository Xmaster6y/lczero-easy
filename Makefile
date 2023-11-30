# CI
.PHONY: checks
checks:
	poetry run pre-commit run --all-files

.PHONY: tests
tests:
	poetry run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s
