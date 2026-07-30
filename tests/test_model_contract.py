"""Contract tests for the committed model artifact.

Everything else in this repository exists to ship model/clutch_model.joblib
safely, and until now nothing verified it was the file we think it is. A
retrain with a different seed, a scikit-learn version bump changing a default,
or a reordered feature list would all pass the API tests, which only assert
that a probability is between 0 and 1.

These tests pin the artifact's actual behaviour instead. If one fails after a
deliberate retrain, update the expected values in the same commit that retrains,
so the change is visible in review rather than silent.
"""

import joblib
import pandas as pd
import pytest
from pydantic.alias_generators import to_camel
from sklearn.model_selection import train_test_split
from smoke_container import EXPECTED_PROBABILITY as SMOKE_PROBABILITY
from smoke_container import PAYLOAD as SMOKE_PAYLOAD

from api.main import MODEL_PATH
from model.features import TARGET, feature_columns
from model.train import DEFAULT_SEED, HOLDOUT_FRACTION

DATA_PATH = MODEL_PATH.parent.parent / "data" / "processed" / "matches.csv"

#: The exact 33 features, in the order the pipeline expects them.
EXPECTED_FEATURES = [
    "goldDiff", "expDiff", "champLevelDiff", "isFirstTower", "isFirstBlood",
    "killedFireDrake", "killedWaterDrake", "killedAirDrake", "killedEarthDrake",
    "lostFireDrake", "lostWaterDrake", "lostAirDrake", "lostEarthDrake",
    "killedRiftHerald", "lostRiftHerald",
    "destroyedTopInnerTurret", "destroyedMidInnerTurret", "destroyedBotInnerTurret",
    "lostTopInnerTurret", "lostMidInnerTurret", "lostBotInnerTurret",
    "destroyedTopOuterTurret", "destroyedMidOuterTurret", "destroyedBotOuterTurret",
    "lostTopOuterTurret", "lostMidOuterTurret", "lostBotOuterTurret",
    "kills", "deaths", "assists",
    "wardsPlaced", "wardsDestroyed", "wardsLost",
]

#: A fixed input and the probability the committed artifact returns for it.
#: Measured, not chosen. Any drift here means the weights changed.
REFERENCE_INPUT = {
    "goldDiff": 2400, "expDiff": 1100, "champLevelDiff": 0.6,
    "isFirstTower": False, "isFirstBlood": True,
    "killedFireDrake": 1, "killedRiftHerald": 1,
    "kills": 6, "deaths": 3, "assists": 8,
    "wardsPlaced": 18, "wardsDestroyed": 4, "wardsLost": 5,
}
REFERENCE_PROBABILITY = 0.8504427853741403

#: Bounds on holdout accuracy. The floor catches a model that silently got
#: worse; the ceiling is the leakage alarm, since minute-10 state does not
#: support scores this high without an excluded column having crept back in.
ACCURACY_FLOOR = 0.72
ACCURACY_CEILING = 0.80


@pytest.fixture(scope="module")
def pipeline():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def holdout(pipeline):
    """The same held-out split train.py measures, reproduced by seed."""
    matches = pd.read_csv(DATA_PATH)
    features = matches[feature_columns(matches.columns)]
    _, x_test, _, y_test = train_test_split(
        features,
        matches[TARGET],
        test_size=HOLDOUT_FRACTION,
        random_state=DEFAULT_SEED,
        stratify=matches[TARGET],
    )
    return x_test, y_test


def test_artifact_exists_and_loads(pipeline):
    assert pipeline is not None


def test_feature_contract_is_exact(pipeline):
    """Order matters. A reordered list would silently mis-map every input."""
    assert list(pipeline.feature_names_in_) == EXPECTED_FEATURES


def test_reference_input_returns_the_expected_probability(pipeline):
    """Pins the weights. This is the test that catches an unintended retrain."""
    row = pd.DataFrame(
        [{name: REFERENCE_INPUT.get(name, 0) for name in EXPECTED_FEATURES}],
        columns=EXPECTED_FEATURES,
    )

    probability = float(pipeline.predict_proba(row)[0, 1])

    assert probability == pytest.approx(REFERENCE_PROBABILITY, abs=1e-9)


def test_smoke_payload_agrees_with_the_artifact(pipeline):
    """Keep the container smoke test's pinned values honest.

    tests/smoke_container.py restates this reference match in wire format and
    hardcodes the expected probability, because it runs on the CI runner with
    nothing but the standard library available. That duplication is the price of
    not installing scikit-learn twice, and this test is what stops it drifting:
    a retrain that updates the constants here but not there fails now, in the
    fast job, rather than in the container job several minutes later.
    """
    row = {to_camel(name): value for name, value in SMOKE_PAYLOAD.items()}

    assert sorted(row) == sorted(EXPECTED_FEATURES), "smoke payload is not the schema"

    frame = pd.DataFrame([row], columns=EXPECTED_FEATURES)
    probability = float(pipeline.predict_proba(frame)[0, 1])

    assert probability == pytest.approx(SMOKE_PROBABILITY, abs=1e-9)
    assert SMOKE_PROBABILITY == REFERENCE_PROBABILITY


def test_holdout_accuracy_stays_in_band(pipeline, holdout):
    x_test, y_test = holdout

    accuracy = pipeline.score(x_test, y_test)

    assert accuracy >= ACCURACY_FLOOR, "model got worse"
    assert accuracy <= ACCURACY_CEILING, "suspiciously high, check for leakage"


def test_probabilities_are_calibrated(pipeline, holdout):
    """A returned probability should mean what it says.

    Accuracy would happily reward a model that is often right but wildly
    overconfident. Since the API hands callers a confidence number, the number
    has to track reality: across matches predicted at roughly 70%, roughly 70%
    should actually win.
    """
    x_test, y_test = holdout
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    frame = pd.DataFrame({"p": probabilities, "won": y_test.to_numpy()})
    buckets = frame.groupby(pd.cut(frame["p"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0]))

    for interval, group in buckets:
        if len(group) < 100:
            continue
        predicted = group["p"].mean()
        observed = group["won"].mean()
        assert observed == pytest.approx(predicted, abs=0.05), (
            f"bucket {interval}: predicted {predicted:.3f}, "
            f"observed {observed:.3f} over {len(group)} matches"
        )
