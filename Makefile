.PHONY: install train report test serve docker lint all

install:
	pip install -r requirements-dev.txt && pip install -e .

train:
	python -m churn.train

report:
	python -m churn.report

test:
	pytest -q

lint:
	ruff check src api tests

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

docker:
	docker build -t churn-api . && docker run -p 8000:8000 churn-api

all: install train report test
