# Clutch

[![CI](https://github.com/Sergio3O6/Clutch-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Sergio3O6/Clutch-Predictor/actions/workflows/ci.yml)

Clutch is a service that predicts whether a League of Legends team wins, based
on the state of the match at the ten-minute mark. It takes the gold and
experience differentials, the objectives taken and conceded, and the
kill/death/assist line, and returns the call along with the probability behind
it.

**Work in progress.** The model trains, the API runs locally, and every change
is tested, linted, and built into a container that gets started and questioned
before the build passes. Nothing is deployed yet. The status list below tracks
what is actually built, and this README will change as the rest lands.

## Prediction target

Given the state of a ranked match at minute 10, predict whether the observed
team goes on to win.

Training data covers 24,912 matches, one row each, with outcomes split
49.8/50.2. The full target definition and the column exclusions that keep the
match result from leaking into the features are in
[`docs/DATA_AND_TARGET.md`](docs/DATA_AND_TARGET.md).

## Results

Logistic regression on 33 features, measured on a held-out 20% of matches:

| Metric | Value |
|---|---|
| Majority baseline | 0.5033 |
| Holdout accuracy | 0.7313 |
| Holdout ROC AUC | 0.8065 |
| Holdout Brier score | 0.1795 |
| Cross-validated accuracy | 0.7228 (+/- 0.0033) |

Gold differential carries most of the signal, at a coefficient of +0.96 against
+0.45 for experience and under 0.15 for everything else.

Accuracy is 73% across all matches and 86% on the 44% of matches where the model
is confident, meaning a probability below 0.25 or above 0.75. The rest are close
games where minute 10 genuinely is not decisive yet.

## API

Runs locally for now. `POST /predict` takes all 33 features. Nothing is optional
and unrecognised fields are rejected, so a caller cannot pass a value the model
ignores and believe it was used.

```
POST /predict
{ "gold_diff": 6000, "exp_diff": 4000, "kills": 5, ... }

200 { "will_win": true, "win_probability": 0.9799 }
```

`GET /health` reports whether the model is loaded, which is what a container
health check needs to know. A process that is running without a model cannot
serve traffic and should not pass.

```
200 { "status": "ok", "model_loaded": true, "n_features": 33 }
```

Interactive docs are generated at `/docs` when the service is running.

## Setup

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The training slice is committed, so you can train and run the service without
downloading anything:

```bash
python model/train.py              # writes model/clutch_model.joblib
uvicorn api.main:app --reload      # http://localhost:8000/docs
```

Tests and linting:

```bash
pytest
ruff check .
```

### Regenerating the training slice

Only needed if you want to rebuild `data/processed/matches.csv` from source.
The raw dataset is 37 MB and is not committed. Place `lol_ranked_games.csv` in
`data/raw/`, then:

```bash
python data/prepare_data.py
```

## Container

```bash
docker build -t clutch .
docker run -p 8000:8000 clutch
```

It is a multi-stage build. Dependencies resolve into a virtualenv in a builder
stage, and the final image copies only that virtualenv, so pip, compilers and
the wheel cache never ship. The model artifact is installed as package data
rather than mounted, so a running container needs nothing external to find its
weights. It runs as an unprivileged user, and its health check calls the same
`/health` endpoint a load balancer would.

Verification happens in CI rather than locally, for the reason described below.

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every pull
request and every push to `main`, in two jobs that run in parallel:

| Job | What it does |
|---|---|
| tests and lint | `pytest` over the feature rules, the API contract, and the committed artifact, then `ruff` |
| container build and smoke test | builds the image, starts it, and interrogates the running service |

The second job is the one worth explaining. Building an image proves very little
on its own: it will build cleanly and still fail to serve if the artifact was
not packaged, if a runtime dependency was left behind in the dev extra, or if
the unprivileged user cannot read what it needs. So the job starts the
container, waits for `/health` to report a loaded model, and then posts a fixed
reference match and asserts it comes back with the same probability the
artifact produces outside the container. That last assertion is what proves the
image is serving the weights this repository pinned, rather than merely that
something is listening on the port.

The smoke test is
[`tests/smoke_container.py`](tests/smoke_container.py), deliberately written
against the standard library alone so the container job never installs the
dependency tree a second time to send one HTTP request. Its pinned values are
restated from the artifact contract test, and a test in that suite fails if the
two ever drift apart.

## Design decisions

**The trained model is committed to the repository.** A binary in git is
usually a smell. It is here because the alternative is worse for a service this
small: fetching weights from object storage at boot means the container needs
credentials, network access, and a failure path for a bad download, all to
retrieve a file measured in kilobytes. Committing it makes CI deterministic and
offline, lets the contract tests pin the exact bytes that ship, and means a
running task needs nothing external to answer a request. It would stop being
the right call the moment the model is retrained on a schedule or grows large.

**A prepared slice of the data is committed, not the raw dataset.** The source
file is 37 MB of time series, roughly ten snapshots per match. What training
needs is one row per match at minute 10 with the leaking columns already gone.
Committing that slice keeps the repository small and lets anyone clone and
train without downloading anything, while the script that produces it stays in
the tree so the reduction is reproducible rather than magic.

**Feature selection lives in its own module, not in the training script.** Most
of the 59 raw columns will leak the match result if fed in unchecked. Those
exclusion rules are the most important logic in the project, so they are
expressed as data in `model/features.py`, covered directly by tests, and
imported by both the preparation script and the trainer. One copy means the two
cannot disagree about what the model is allowed to see.

**Logistic regression, not something stronger.** Accuracy is not the point of
this project. A linear model trains in seconds, keeps its coefficients
readable, and makes a leaked column obvious as an outsized weight instead of
hiding it inside an ensemble. The training script also warns when accuracy
lands *above* a per-frame ceiling, on the grounds that minute-10 state does not
support a score that good and the likely explanation is leakage.

**One uvicorn worker per container.** Fargate scales by running more tasks.
Adding in-container workers would make a single task's CPU and memory usage
harder to reason about for no benefit at this size.

## Planned

Terraform will provision an ECR repository, a Fargate service, the surrounding
networking, and IAM roles scoped to what each component needs, with state kept
remote. Nothing cloud-side is built yet.

## Status

- [x] Repository structure
- [x] Dataset selection and prediction target definition
- [x] Feature engineering and model training
- [x] Prediction API
- [x] Container image
- [x] Automated tests, linting, and container verification on every change
- [ ] Cloud infrastructure defined in code
- [ ] Deployment
- [ ] Separate staging and production environments
- [ ] Full documentation

## Repository layout

| Path | Contents |
|---|---|
| `data/` | Preparation script and the committed training slice. Raw data is gitignored. |
| `model/` | Feature selection rules, training script, and the serialized artifact |
| `api/` | FastAPI application and request schemas |
| `infra/` | Terraform configuration. Empty for now. |
| `tests/` | pytest suite, plus the standalone container smoke test |
| `.github/workflows/` | CI pipeline |
| `docs/` | Data and target definition, design notes |
