.PHONY: setup data train serve monitor simulate-drift test down
setup:
	python -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dev]"
data:
	python -m pipelines.generate_data
train:
	python -m pipelines.train
serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000
monitor:
	python -m pipelines.monitor
simulate-drift:
	python -m pipelines.simulate_drift
test:
	pytest -q
down:
	docker compose down -v
