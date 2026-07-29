# Clutch

Clutch is a service that predicts whether a League of Legends team wins, based
on the state of the match at the ten-minute mark. It takes the gold and
experience differentials, the objectives taken and conceded, and the
kill/death/assist line, and returns the call along with the probability behind
it.

This is early. Right now the repository holds the dataset definition and the
project structure. The model and the service are not built yet.

## Prediction target

Given the state of a ranked match at minute 10, predict whether the observed
team goes on to win.

Training data covers 24,912 matches, one row each, with outcomes split
49.8/50.2. The full target definition and the column exclusions that keep the
match result from leaking into the features are in
[`docs/DATA_AND_TARGET.md`](docs/DATA_AND_TARGET.md).

## Planned architecture

The model will be trained offline and serialized to disk, then served behind a
FastAPI application with Pydantic schemas validating requests at the boundary.
That application will be packaged as a multi-stage Docker image with the model
artifact baked in, so a running container needs nothing external to serve
traffic.

GitHub Actions will run the test suite, lint, and build the image on every push.
Terraform will provision the AWS side: an ECR repository for images, a Fargate
service to run them, the surrounding networking, and IAM roles scoped to what
each component actually needs. Terraform state will be kept remote rather than
local.

None of this is built or deployed yet.

## Status

- [x] Repository structure
- [x] Dataset selection and prediction target definition
- [ ] Feature engineering and model training
- [ ] Prediction API
- [ ] Container image
- [ ] Automated tests and linting on push
- [ ] Cloud infrastructure defined in code
- [ ] Deployment
- [ ] Separate staging and production environments
- [ ] Full documentation

## Setup

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The raw dataset is not committed to the repository. It belongs at
`data/raw/lol_ranked_games.csv`.

## Repository layout

| Path | Contents |
|---|---|
| `data/` | Raw dataset (gitignored) and the processed slice used for training |
| `model/` | Feature engineering, training, and the serialized artifact |
| `api/` | FastAPI application |
| `infra/` | Terraform configuration |
| `tests/` | pytest suite |
| `docs/` | Data and target definition, design notes |

Most of these directories are empty at this point.
