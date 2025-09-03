.PHONY: setup run-min test

setup:
	python -m pip install -U pip
	python -m pip install -U -r requirements.txt

run-min:
	PYTHONPATH=. python experiments/minimal_run.py

test:
	PYTHONPATH=. pytest -q
	pytest -q
