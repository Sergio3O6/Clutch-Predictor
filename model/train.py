"""Train the minute-10 win predictor and serialize it.

Logistic regression on scaled features. The model is deliberately plain: the
coefficients stay readable, training takes seconds, and a leaked column shows up
as an obvious outlier rather than hiding inside an ensemble.

Usage:
    python model/train.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.features import PREDICTION_FRAME, TARGET, feature_columns

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "matches.csv"
DEFAULT_OUT = REPO_ROOT / "model" / "clutch_model.joblib"

DEFAULT_SEED = 42
HOLDOUT_FRACTION = 0.2
CV_FOLDS = 5

#: Accuracy ceilings above which a run is more likely to be leaking than winning,
#: keyed by prediction frame. These are calibrated per frame on purpose: the same
#: honest feature set scores 0.731 at frame 10 and 0.840 at frame 24, because
#: more of the match has happened, so a single global threshold would either miss
#: real leakage late or cry wolf constantly. A frame with no entry here gets no
#: check rather than an invented one.
LEAKAGE_ALERT_ACCURACY = {10: 0.80, 24: 0.90}


def build_pipeline(seed: int = DEFAULT_SEED) -> Pipeline:
    """Scaling plus logistic regression.

    Scaling matters here because the feature ranges are wildly different: gold
    differential runs to the thousands while the objective flags are 0 or 1.
    Keeping it inside the pipeline means the same transform is applied at
    serving time without the API having to know about it.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


def train(data_path: Path, out_path: Path, seed: int = DEFAULT_SEED) -> dict:
    """Fit the model, report holdout metrics, and write the artifact."""
    matches = pd.read_csv(data_path)
    features = matches[feature_columns(matches.columns)]
    labels = matches[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=HOLDOUT_FRACTION,
        random_state=seed,
        stratify=labels,
    )

    pipeline = build_pipeline(seed)
    cv_scores = cross_val_score(pipeline, x_train, y_train, cv=CV_FOLDS)

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "majority_baseline": max(labels.mean(), 1 - labels.mean()),
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "holdout_accuracy": accuracy_score(y_test, predictions),
        "holdout_roc_auc": roc_auc_score(y_test, probabilities),
        "holdout_brier": brier_score_loss(y_test, probabilities),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "n_features": features.shape[1],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    return metrics


def top_coefficients(pipeline: Pipeline, count: int = 8) -> pd.Series:
    """Largest absolute coefficients, as a sanity check on what drives calls."""
    # Feature names live on the pipeline, not the classifier: the scaler hands
    # the classifier a bare numpy array.
    coefficients = pd.Series(
        pipeline.named_steps["classifier"].coef_[0],
        index=pipeline.feature_names_in_,
    )
    return coefficients.reindex(
        coefficients.abs().sort_values(ascending=False).index
    ).head(count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not args.data.exists():
        parser.error(
            f"training data not found at {args.data}. "
            "Run python data/prepare_data.py first."
        )

    metrics = train(args.data, args.out, args.seed)

    print(f"trained on {metrics['n_train']:,} matches, "
          f"held out {metrics['n_test']:,}, "
          f"{metrics['n_features']} features")
    print()
    print(f"  majority baseline  {metrics['majority_baseline']:.4f}")
    print(f"  cv accuracy        {metrics['cv_accuracy_mean']:.4f} "
          f"(+/- {metrics['cv_accuracy_std']:.4f})")
    print(f"  holdout accuracy   {metrics['holdout_accuracy']:.4f}")
    print(f"  holdout roc auc    {metrics['holdout_roc_auc']:.4f}")
    print(f"  holdout brier      {metrics['holdout_brier']:.4f}")
    print()
    print(f"wrote {args.out}")

    ceiling = LEAKAGE_ALERT_ACCURACY.get(PREDICTION_FRAME)
    if ceiling is None:
        print()
        print(
            f"NOTE: no leakage ceiling calibrated for frame {PREDICTION_FRAME}, "
            "so this run was not checked. Add one to LEAKAGE_ALERT_ACCURACY."
        )
    elif metrics["holdout_accuracy"] > ceiling:
        print()
        print(
            f"WARNING: accuracy {metrics['holdout_accuracy']:.4f} is above "
            f"{ceiling:.2f}, which state at minute {PREDICTION_FRAME} should "
            "not support. Check the feature list against "
            "docs/DATA_AND_TARGET.md before trusting this."
        )


if __name__ == "__main__":
    main()
