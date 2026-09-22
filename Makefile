.PHONY: help install test demo-artifact train docker-build docker-run docker-smoke clean

IMAGE ?= views-model
TAG   ?= $(shell git describe --always --dirty 2>/dev/null || echo dev)
PORT  ?= 8000

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install pinned runtime deps, the package, and test extras
	pip install --no-cache-dir -r requirements.txt
	pip install --no-cache-dir -e ".[test]"

test:  ## Run the test suite (synthetic fixtures only, no dataset needed)
	pytest -q

demo-artifact:  ## Train a throwaway model on synthetic data -- no Kaggle download
	python -m views_model.synthetic --out artifacts/views_baseline.joblib

train:  ## Train on the real dataset (requires data/ from Kaggle)
	views-train \
		--postings data/postings.csv \
		--companies data/companies/company_industries.csv \
		--out artifacts/views_baseline.joblib

docker-build: demo-artifact  ## Build the serving image (tag = short git SHA)
	docker build \
		--tag "$(IMAGE):$(TAG)" \
		--label "org.opencontainers.image.revision=$$(git rev-parse HEAD)" \
		--label "com.views-model.artifact=synthetic-demo" \
		.

docker-run: docker-build  ## Build and serve on $(PORT)
	docker run --rm -p $(PORT):8000 "$(IMAGE):$(TAG)"

docker-smoke:  ## Curl /ready and /predict against a running container
	curl -fsS http://localhost:$(PORT)/ready && echo
	curl -fsS -X POST http://localhost:$(PORT)/predict \
		-H 'Content-Type: application/json' \
		-d '{"postings":[{"title":"senior data engineer","company_id":1,"description":"we are hiring","normalized_salary":120000,"formatted_work_type":"FULL_TIME","formatted_experience_level":"Mid-Senior level","pay_period":"YEARLY","application_type":"OffsiteApply","original_listed_time":1704067200000,"listed_time":1704067200000,"expiry":1706659200000}],"target":"views"}' \
		&& echo

clean:  ## Remove build artifacts and caches
	rm -rf artifacts/*.joblib .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
