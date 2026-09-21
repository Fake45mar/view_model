# views-model

Baseline model + serving layer that predicts the number of views a LinkedIn
job posting will receive. 

## Layout

```
src/views_model/
  config.py     # feature lists & constants
  features.py   # build_features: raw postings -> model input
  pipeline.py   # sklearn Pipeline (preprocessing + HGB regressor)
  train.py      # CLI: trains both models, writes artifact
  predict.py    # load_bundle + predict_views
  api.py        # FastAPI app (POST /predict, GET /health, /metrics)
tests/test_smoke.py
Dockerfile      .dockerignore   requirements.txt   pyproject.toml
artifacts/      # produced by `views-train`
```

The notebook ([assignment_prep.ipynb](assignment_prep.ipynb)) contains the
exploration that led to the current feature set and model choice; this
package is its productionised form.

## Data

The dataset is not included in the repo. Download the LinkedIn Job Postings
dataset from Kaggle:

<https://www.kaggle.com/datasets/arshkon/linkedin-job-postings/data>

Unzip the archive **into the `data/` folder** at the repo root so the layout
matches what the trainer expects:

```
data/
├── postings.csv
├── companies/
│   ├── companies.csv
│   ├── company_industries.csv
│   ├── company_specialities.csv
│   └── employee_counts.csv
├── jobs/
│   ├── benefits.csv
│   ├── job_industries.csv
│   ├── job_skills.csv
│   └── salaries.csv
└── mappings/
    ├── industries.csv
    └── skills.csv
```

Quick way (requires a Kaggle API token in `~/.kaggle/kaggle.json`):

```bash
mkdir -p data && cd data
kaggle datasets download -d arshkon/linkedin-job-postings
unzip -o linkedin-job-postings.zip && rm linkedin-job-postings.zip
cd ..
```

## Quickstart

```bash
# 1. Install (editable)
pip install -e .

# 2. Train  (writes artifacts/views_baseline.joblib)
views-train \
    --postings data/postings.csv \
    --companies data/companies/company_industries.csv \
    --out artifacts/views_baseline.joblib

# 3. Serve
uvicorn views_model.api:app --reload
```

Then:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/metrics | jq
curl -s -X POST http://localhost:8000/predict \
     -H 'content-type: application/json' \
     -d '{"target":"views","postings":[{
           "company_id": 1234, "title": "Data Engineer",
           "description": "...", "normalized_salary": 120000,
           "remote_allowed": 1, "sponsored": 0,
           "formatted_work_type": "Full-time",
           "formatted_experience_level": "Mid-Senior level",
           "pay_period": "YEARLY", "application_type": "OffsiteApply",
           "original_listed_time": 1713000000000,
           "listed_time": 1713000000000,
           "expiry": 1715592000000
         }]}'
```

## Docker

The image bakes in a pre-trained artifact (training is a separate, offline
step — keeps the serving image small and reproducible):

```bash
# Train first so artifacts/views_baseline.joblib exists.
views-train

# Then build & run.
docker build -t views-model:latest .
docker run --rm -p 8000:8000 views-model:latest
```

The artifact path can be overridden with `VIEWS_MODEL_ARTIFACT`.

## Models

Two regressors are fit on the same feature set and stored in the same bundle:

| key | target | use case |
|---|---|---|
| `views` | `log1p(views)` | predict total views to date |
| `views_per_day` | `log1p(views / age_days)` | rank postings by intrinsic attractiveness |

Both are evaluated on the same held-out test set; metrics are persisted in
the bundle and exposed via `GET /metrics`.

## Tests

```bash
pip install pytest
pytest -q
```
