# Views model: what I found and what I changed

## Starting point

A working service and a working model, and about five hours.

I treated it as **inherited software that needs a path to production**, not as a
modelling task, because the brief says the model is not what is being judged.

My first impression was that it looked like a sketch. Unpinned requirements, a
pyproject and a Dockerfile together, and nobody had decided whether this is a
library or a service.

So I worked in this order: **correctness first, then delivery, then write down
the rest.** A pipeline that ships a broken model faster is worse than no
pipeline.

One thing ties the whole talk together. **Both bugs I found were silent.** No
error, no alert, output that looked fine.

---

## Finding 1: age was measured against a frozen date

What caught my eye was a date stored inside the model file. A saved model
should not carry a date with it.

At training that date is the newest posting in the data, which is correct. The
serving code read the same date back out of the file, so **every request was
measured against April 2024**.

| posting age | old code | fixed |
|---|---|---|
| 0 days | 5.73 | 5.73 |
| 5 days | 5.73 | 46.01 |
| 30 days | 5.73 | 28.83 |
| 90 days | 5.73 | 28.83 |

**The old column never changes.** `age_days` was minus 885 for every request.
During training the model only saw ages from 0 to 136 days, so every negative
number falls below every rule the trees learned. They all follow the same path
and give the same answer.

The feature was alive in training and **dead in production**. Predictions were
5 to 33 times off, depending on age.

Nobody saw it because there was no error and the output looked reasonable. And
at age zero both columns agree, so a smoke test written with a fresh posting
passes.

The fix is three lines. Measure age from the request time.

---

## Finding 2: the split leaked the future

The train and test split was random on time-ordered data, so the model trained
on rows newer than its own test set.

| split | MAE | R² |
|---|---|---|
| random | 9.79 | 0.052 |
| time-based | 2.82 | 0.003 |

Full dataset, 122,160 rows.

**Compare R², not MAE.** The two test sets are different. The time-based test
set is the newest postings, which have collected fewer views, so a smaller
target gives a smaller absolute error. R² is measured against the variance of
its own test set, so it is the one that compares.

0.003 means the model **explains almost none of the variation** in views.

The random split was the only thing making this model look like it worked. I
could have kept it and shown a better number.

---

## Delivery

**28 tests, under a second, and they pass with the data folder renamed away.**
The dataset is 504 MB and needs Kaggle credentials, so CI could never have run
without this.

For both bugs I put the bug back and checked the suite goes red. **A test that
has never failed is not a test yet.**

CI runs the tests, builds the image, starts it, waits for readiness and sends a
real prediction. **One minute forty four.**

It waits for `/ready`, not for the open port. An open port only means uvicorn
started.

A broken model file used to crash the process. Now the service starts and stays
up. `/health` returns 200, `/ready` returns **503 and names the check that
failed**, and `/predict` refuses.

A crash loop gives an operator a restart counter. This gives them a reason.

The container runs as a normal user that cannot modify the model file, and
uvicorn is PID 1, so `docker stop` finishes in **0 seconds** with a clean
shutdown instead of being killed after a timeout.

---

## What breaks first

The weakest part is that **there is no monitoring**, so the system cannot tell
anyone when it is wrong.

What breaks first is staleness. It was trained on data ending **April 2024** and
there is no retraining. Nothing detects that, and it fails the same way the age
bug did. Silently.

So the first two things I would build are **prediction distribution as a
metric**, and scheduled retraining with manual promotion.

Retraining and LLM enrichment are designed but not built. Both are in the
architecture notes.

---

## Numbers

| | |
|---|---|
| `age_days` in production | −885, the same for every request |
| the dead column | 5.73, four times |
| R², random against time-based | 0.052 against 0.003 |
| tests | 28, under a second, no dataset |
| full CI run | 1m 44s |
| shutdown after the fix | 0 seconds |
| training data ends | April 2024 |
