# Clutch

Clutch is a service that predicts whether a League of Legends team wins, based
on the state of the match at the ten-minute mark. It takes the gold and
experience differentials, the objectives taken and conceded, and the
kill/death/assist line, and returns the call along with the probability behind
it.

**Work in progress.** The model trains and the API runs locally. There is no
container, no CI, and nothing deployed. The status list below tracks what is
actually built, and this README will change as the rest lands.

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

## Planned

None of this is built yet.

The service will be packaged as a multi-stage Docker image with the model
artifact baked in, so a running container needs nothing external. GitHub Actions
will run tests, lint, and the image build on every push. Terraform will
provision an ECR repository, a Fargate service, the surrounding networking, and
IAM roles scoped to what each component needs, with state kept remote.

## Status

- [x] Repository structure
- [x] Dataset selection and prediction target definition
- [x] Feature engineering and model training
- [x] Prediction API
- [ ] Container image
- [ ] Automated tests and linting on push
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
| `tests/` | pytest suite |
| `docs/` | Data and target definition, design notes |
