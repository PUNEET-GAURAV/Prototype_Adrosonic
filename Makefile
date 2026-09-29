PY ?= python3
setup:
	$(PY) -m pip install -r requirements.txt
qdrant:
	docker compose up -d
index:
	$(PY) scripts/build_index.py --dataset demo --n 100000 --recreate
smoke:
	$(PY) -m pytest -q tests && $(PY) scripts/build_index.py --dataset demo --n 5000 --recreate
ui:
	PYTHONPATH=src $(PY) -m uvicorn verity.api:app --port 8000
bench:
	$(PY) scripts/bench_latency.py
test:
	$(PY) -m pytest -q tests
