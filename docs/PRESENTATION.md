# Presentation notes

15 minutes for MLOps engineers and data scientists, then questions.

Full detail is in [ARCHITECTURE.md](ARCHITECTURE.md). These are speaking notes,
not slides.

---

## Timing

| minutes | part |
|---|---|
| 0-2 | What I inherited and how I chose what to do |
| 2-7 | Findings, starting with the age bug |
| 7-10 | Delivery: CI, readiness, image identity |
| 10-13 | Retraining and LLM enrichment, diagrams only |
| 13-15 | What I would do next, and questions |

The age bug is the part worth the most time. If the discussion runs long,
the LLM section compresses to a minute and the detail stays in the
architecture notes.

---

## 0-2 min. What I inherited

A FastAPI service and an sklearn model that both worked. The task says the
model itself is not what is being judged, so I treated this as inherited
software that needs a path to production.

I spent the time in this order:

1. Make the tests run without the 504 MB dataset.
2. Fix the two correctness bugs I found.
3. Make the delivery pipeline real: pinned versions, CI, a container that can
   be rolled back.
4. Write down the rest.

**Why that order.** A pipeline that ships a broken model faster is worse than
no pipeline. So correctness first, then delivery. Retraining and LLM enrichment
are design only, because the brief asks for design and because building either
one badly in four hours would be worse than a clear plan.

---

## 2-7 min. Findings

### The age bug

One feature is `age_days`. It is `reference_time − posting_listed_time`.

When training, the reference is the newest posting in the data, **2024-04-20**.
The old code saved that date inside the model file and used it again when
serving.

Every request comes after 2024-04-20. So the subtraction goes negative:

```
2024-04-20 − a posting listed today = −885 days
```

During training the model only saw ages from **0 to 136 days**. The trees learn
rules like "is age below 12 days?". Every negative number is below every rule,
so they all follow the same path and give the same answer.


| posting age | prediction, old code | prediction, fixed |
|---|---|---|
| 0 days | 5.73 | 5.73 |
| 5 days | 5.73 | 46.01 |
| 30 days | 5.73 | 28.83 |
| 90 days | 5.73 | 28.83 |

The old column never changes. A 5-day-old posting and a 90-day-old posting got
the same number. The model had learned from this feature, but in production the
value was always the same, so the feature was dead.

Predictions were 5 to 33 times off, depending on age.

**Why nobody saw it.** No error. No crash. No change in latency. Output that
looks reasonable. And at age 0 both columns agree, so a smoke test written with
a fresh posting passes.

**The fix is three lines.** Measure age from the request time, not from the
saved date. The value is in finding it, not in the diff.

### The second bug

The train/test split used `train_test_split`, which shuffles. The postings are
ordered by time, so the model was training on rows newer than its own test set.

| split | MAE | R² |
|---|---|---|
| random | 9.79 | 0.052 |
| time-based | 2.82 | **0.003** |

Compare R², not MAE. The two test sets are different. The newest postings have
collected fewer views, so a smaller target gives a smaller absolute error. R²
is measured against its own test set, so it is the one that compares.

0.003 means the model explains almost none of the variance.

The random split was the only thing making this model
look like it worked. I could have left it and shown a better number. The honest
number is more useful to the team that owns this model.

### The rest, one line each

- sklearn version was not pinned and not recorded in the model file. A version
  mismatch gives a warning, not an error, and then serves different numbers.
  Now it is pinned, recorded, and checked at startup.
- `/metrics` returned training metrics. That path belongs to Prometheus.
  Renamed to `/model-info`.
- `was_relisted` is probably leakage. A posting may be relisted because it got
  few views. I documented it and did not remove it, because changing the model
  is not what this task is about.

---

## 7-10 min. Delivery

### Tests

28 tests, under a second, and they pass with the `data/` folder renamed away.

That last part was the hard bit. A small fake dataset does not work directly,
because `TruncatedSVD(n_components=50)` needs more TF-IDF features than a small
vocabulary produces. So the pipeline settings became arguments, and the test
data generator produces a vocabulary big enough to clear that limit.

For each bug fix there is a test, and **I put the bug back and checked the test
fails.** A test that has never failed is not yet a test.

### CI

Tests, then build the image, start the container, wait for `/ready`, send a
real prediction, check the answer.

It waits for `/ready` and not for the open port. An open port only means uvicorn
started. `/ready` means the model loaded and can actually produce a prediction.

The image CI builds contains a model trained on fake rows. It proves the
packaging and serving path. It is not a release image.

### Readiness

This was my operational improvement. A broken model file used to crash the
process.

Now the service starts and stays up:

- `/health` returns 200. The process is alive.
- `/ready` returns 503 and names the check that failed.
- `/predict` returns 503.

Four checks run at startup: the file loads, the sklearn version matches, the
feature list matches this build, and a real prediction runs end to end.

**Why not crash.** A crash loop gives the operator a restart counter. This gives
them a reason.

### Image identity

The tag is the git commit. If the working tree has uncommitted changes the tag
ends with `-dirty`, so a local build can never be confused with a reproducible
one. CI always builds from a clean checkout.

Everything else goes into OCI labels, not into the tag string.

**The digest is the identity, not the tag.** Build once, promote the same digest
between environments. Rollback is deploying the previous digest.

I tested the full flow locally against a `registry:2` container. Push, then run
by digest. Tag immutability is a registry setting, on ECR it is
`imageTagMutability=IMMUTABLE`. Docker itself will always let you move a tag, so
that part is documented, not implemented.

### Container

Runs as a normal user that does not own the model file, so it can read the model
and cannot change it. `exec` makes uvicorn PID 1, so it receives SIGTERM.
`docker stop` now finishes in 0 seconds with a clean shutdown. Before, Docker
waited for the timeout and killed it.

---

## 10-13 min. Design only

The two diagrams in the architecture notes cover both of these.

### Retraining

Three points:

1. **Promotion is not the same as training.** Training runs on a schedule.
   Promotion needs a human, at least until the checks have caught a real
   problem once.
2. **The checks.** Beat the current model on data after `split_time_ms`, pass a
   fixed set of golden cases, and keep the prediction distribution close to the
   current model. A model with better average error and no variation is a step
   back.
3. **A silent scheduler is the worst failure.** If retraining has not run inside
   its window, that is an alert too.

### LLM enrichment

Three points:

1. **It runs offline, never inside `/predict`.** An LLM call in the request path
   puts an external service with its own outages into the serving path.
   Enrichment writes to a cache, the model reads the cache.
2. **The cache key includes the prompt version, the model version, and the
   taxonomy version.** Change any of them and the output changes.
3. **A prompt change is a model change.** If the prompt changes, the training
   data must be enriched again before a model trained on it is promoted. A model
   trained on `prompt_v1` features must never be served `prompt_v2` features.
   That is the age bug again, one level up, and just as silent.

On privacy: job descriptions contain salary ranges, names, and contact details.
Raw text and raw LLM responses never go into logs. Log the feature values, token
counts, and a hash of the input.

---

## 13-15 min. Next week

1. JSON logging and a real Prometheus `/metrics`. Right now the service has no
   way to tell anyone it is misbehaving. That is exactly why the age bug
   survived.
2. Decide the horizon question and change the API.
3. Startup probe with a failure limit, so a pod that can never load the model
   restarts instead of sitting there unused.

---

## Questions to expect

### Would you deploy a model with R² of 0.003?

No. And that is the point of measuring it properly.

The model is not the deliverable here, the pipeline is. What I would hand back
is: on a fair test the baseline explains almost none of the variation, the
previous number came from a leaky split, and here are two specific things to
try first. Drop
`was_relisted`, which is probably leakage, and decide whether the target should
be views per day rather than total views.

Shipping it behind a flag to collect real traffic is also reasonable. Shipping
it as a product feature is not.

### Why spend the time on the baseline and CI, not on monitoring or retraining?

Because a delivery pipeline that ships a broken model faster is worse than no
pipeline.

Also, the bug I found is invisible to monitoring. No errors, no latency change.
Adding dashboards before fixing it would have made me feel covered without being
covered.

Monitoring is first on the next-week list, and the specific signal is prediction
distribution, not error rate.

### Why fake data instead of a small sample of the real data?

Three reasons. A real sample has to be stored somewhere, versioned, and checked
for licence. Generated data is identical on every run, so CI cannot become flaky
because of the data. And the fixture needs specific properties that a random
sample does not guarantee, like enough vocabulary for the SVD step.

The tradeoff is real: these tests check correctness, not model quality. Model
quality is checked separately, with real data, by whoever runs `make train`.

### Where can training and serving drift apart, or data leak?

Three found here, and a fourth I left.

- Skew: `age_days` against a saved reference time.
- Leakage: random split on time-ordered data.
- Leakage: `was_relisted`, which is probably a result and not an input.
- Left in place: `snapshot_ms` is the maximum over the whole dataset, so
  training rows are aged against a reference that includes the test period. The
  offline numbers stay a little optimistic. Fixing it properly means rebuilding
  features per split, which did not fit the timebox.

### Which failure is most likely, and how would you find it?

Another silent feature problem, like the age bug. No exception, no latency
change, normal-looking output, and a passing smoke test.

Error rate monitoring would never catch it. What catches it is watching the
prediction distribution, plus feature rules checked at serving time.

One detail worth saying: my first test for this asserted `age_days >= 0`. That
test would have **passed** on the bug, because −885 is also just a number. The
test that actually works asserts that two postings of different ages get
different predictions.

### How do you roll back code and model separately?

Today they are one artifact, so separation comes from a compatibility rule, not
from two releases. New code must serve the old model, and the old model must
still work with new code. `schema_version` in the model file checks this at
startup, so breaking the rule requires changing that number on purpose.

When models start shipping more often than code, the model file moves out of the
image and is downloaded at startup by digest. Then the two roll back separately.
The cost is that "which model is running" becomes a runtime question instead of
a property of the deployment.

### What would you do with another week?

The three items above, in order. Then immutable training snapshots with the id
recorded in the model file, and scheduled retraining with manual promotion.

I would not add a model registry, a data versioning tool, or an orchestrator
yet. Each one is another system to run and debug. The architecture notes give a
condition for each: a registry when a second person starts training models, data
versioning when a second person produces training data, an orchestrator when
retraining has steps that need their own retries.

