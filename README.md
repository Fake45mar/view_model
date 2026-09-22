# views-model

Baseline model + serving layer that predicts how many views a LinkedIn job
posting will receive.

Architecture notes, tradeoffs and failure modes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quickstart — no dataset required

Everything below runs on synthetic data generated in-process. No Kaggle
download, no credentials, no cloud.

```bash
make install          # pinned runtime deps + package + test extras
make test             # 28 tests, ~1s, passes with data/ absent
make demo-artifact    # trains a throwaway model on synthetic rows (~2s)
make docker-run       # builds the image and serves on :8000
```

Then, in another shell:

```bash
make docker-smoke     # curls /ready and /predict
```

> The demo artifact is trained on invented rows. Its predictions are
> meaningless — it exists to exercise the packaging and serving path. See
> [Training on real data](#training-on-real-data) for a usable model.

## Endpoints

| method | path | purpose |
|---|---|---|
| `GET` | `/health` | Liveness. 200 whenever the process is up, **even if the model failed to load.** |
| `GET` | `/ready` | Readiness. 200 when startup validation passed; **503 with the failing check** otherwise. |
| `GET` | `/model-info` | Training metrics and artifact provenance (sklearn version, split boundary, feature list). |
| `POST` | `/predict` | Batch prediction, capped at `VIEWS_MODEL_MAX_BATCH` (default 500). |

`/health` and `/ready` are deliberately different. A broken artifact leaves the
service **running and unready** rather than crash-looping: the operator can
reach it, and `/ready` says what failed.

```bash
curl -s localhost:8000/ready | jq
# {"ready": true, "checks": {"artifact_loads":"ok","sklearn_version":"ok (1.7.2)",
#                            "feature_contract":"ok","smoke_prediction":"ok"}}
```

```bash
curl -s -X POST localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"target":"views","postings":[{
        "title":"Data Engineer","company_id":1234,
        "description":"...","normalized_salary":120000,
        "remote_allowed":1,"sponsored":0,
        "formatted_work_type":"FULL_TIME",
        "formatted_experience_level":"Mid-Senior level",
        "pay_period":"YEARLY","application_type":"OffsiteApply",
        "original_listed_time":1713000000000,
        "listed_time":1713000000000,
        "expiry":1715592000000}]}'
```

`/model-info` is named that, not `/metrics`, so the conventional Prometheus
scrape path stays free for real runtime metrics.

## Layout

```
src/views_model/
  config.py      # feature lists, pipeline hyperparameters, constants
  features.py    # build_features: raw postings -> model input
  pipeline.py    # sklearn Pipeline (TF-IDF + SVD, one-hot, HGB regressor)
  train.py       # CLI: time-based split, fits both models, writes the bundle
  predict.py     # load_bundle + predict_views
  startup.py     # artifact validation run at boot
  synthetic.py   # deterministic fixtures + demo-artifact CLI
  api.py         # FastAPI app
tests/
  conftest.py              # synthetic fixtures, session-scoped training
  test_smoke.py            # train/predict round trip, time-split holdout
  test_age_features.py     # regression tests for the training-serving skew
  test_startup_and_api.py  # validation failures and the readiness contract
.github/workflows/ci.yml   # tests -> build image -> run -> curl /predict
Makefile  Dockerfile  requirements.in  requirements.txt  pyproject.toml
```

## Docker


```bash
make docker-build              # tag = git describe --always --dirty
docker images views-model
```

The tag is the source commit, and a build from an uncommitted tree is tagged
`-dirty` so it can never be mistaken for something reproducible. Everything
else that identifies a build goes into OCI labels rather than the tag string:

```bash
docker inspect --format '{{json .Config.Labels}}' views-model:$(git describe --always --dirty) | jq
```

 Tag
*immutability* is a registry policy (an ECR repository setting), not something
Docker enforces — see the architecture notes.

`VIEWS_MODEL_ARTIFACT` overrides the artifact path.

## Training on real data

The dataset is not committed and not ours to redistribute.
Download the [LinkedIn Job Postings dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings/data)
and unzip it into `data/`:

```
data/
├── postings.csv
├── companies/company_industries.csv
├── jobs/…
└── mappings/…
```

With a Kaggle API token in `~/.kaggle/kaggle.json`:

```bash
mkdir -p data && cd data
kaggle datasets download -d arshkon/linkedin-job-postings
unzip -o linkedin-job-postings.zip && rm linkedin-job-postings.zip && cd ..
```

Then:

```bash
make train    # writes artifacts/views_baseline.joblib
```

## Models

Two regressors are fit on the same feature set and stored in one bundle:

| key | target | use case |
|---|---|---|
| `views` | `log1p(views)` | total views accumulated to date |
| `views_per_day` | `log1p(views / age_days)` | rank postings by intrinsic attractiveness |

Both are evaluated on a **time-based** holdout — the most recent 20% of
postings — and the metrics are persisted in the bundle and served by
`/model-info`. The initial code used a random split; on time-ordered data
that leaks the future into training and inflates the score. Numbers and
discussion are in the architecture notes.

## Tests

```bash
make test
```

28 tests, about a second. They run on synthetic fixtures generated in-process
and **pass with `data/` absent entirely** — that is what keeps CI fast,
deterministic, and independent of a full kaggle dataset download.

## Assumptions

- **Python 3.10.** Matches `requires-python`, the lockfile, and the base image.
  3.10 reaches end-of-life in October 2026, so the upgrade is scheduled work.
- **`requirements.txt` is authoritative for runtime versions.** It is compiled
  from `requirements.in` by `pip-compile`. The image installs it first and then
  runs `pip install --no-deps .`, so the package's own dependency list can
  never override the lockfile.
- **`data/` and `artifacts/` are never committed.** Datasets come from Kaggle,
  artifacts are build output. In production artifacts come from a registry.
- **The model is loaded once, at boot.** Replacing the artifact file on a
  running container does nothing; a new model means a new image and a new
  digest. This keeps model and code releases separately identifiable.
- **The container runs as a non-root user** and does not own the artifact
  files, so the serving process can read the model but not modify it.
- **Timestamps are epoch milliseconds**, matching `postings.csv`.
- **The service scores one posting at a time against the current clock.**
  Age features are measured from request time, not from a value frozen at
  training — see finding #1 in the architecture notes.
