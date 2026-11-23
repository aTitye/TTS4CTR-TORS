.PHONY: install test clean audit

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest tests -q

audit:
	python scripts/audit_public_repo.py

clean:
	rm -rf .pytest_cache ./**/__pycache__ outputs
