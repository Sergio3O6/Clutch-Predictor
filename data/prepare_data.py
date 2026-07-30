"""Build the committed training slice from the raw dataset.

The raw file is a time series of roughly ten snapshots per match and is too
large to keep in git. This script reduces it to one row per match at the
prediction frame, drops every column that would leak the result, and writes a
small CSV that training reads. That slice is committed, so training and CI run
offline and deterministically.

Usage:
    python data/prepare_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from model.features import (
    PREDICTION_FRAME,
    TARGET,
    excluded_columns,
    feature_columns,
    select_prediction_frame,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = REPO_ROOT / "data" / "raw" / "lol_ranked_games.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "matches.csv"

#: Kept in the slice for traceability back to the source data. Excluded from
#: the feature set by model.features, so training never sees it.
REFERENCE_COLUMN = "gameId"


def prepare(raw_path: Path, out_path: Path) -> pd.DataFrame:
    """Read the raw time series and return the committed training slice."""
    frames = pd.read_csv(raw_path)
    sliced = select_prediction_frame(frames)

    if sliced.empty:
        raise ValueError(f"no rows at frame {PREDICTION_FRAME} in {raw_path}")

    if not sliced[REFERENCE_COLUMN].is_unique:
        raise ValueError(
            "expected one row per match after slicing, found duplicates. "
            "The source data may hold both team perspectives."
        )

    keep = [REFERENCE_COLUMN, *feature_columns(frames.columns), TARGET]
    slice_out = sliced[keep]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    slice_out.to_csv(out_path, index=False)
    return slice_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.raw.exists():
        parser.error(
            f"raw dataset not found at {args.raw}. It is not committed to the "
            "repository, see the setup section of the README."
        )

    header = pd.read_csv(args.raw, nrows=0)
    # gameId and the target are excluded from the features but still written to
    # the slice, so report only the columns that leave the file entirely.
    withheld = sorted(
        excluded_columns(header.columns) - {REFERENCE_COLUMN, TARGET}
    )

    result = prepare(args.raw, args.out)

    print(f"wrote {args.out}")
    print(f"  matches:  {len(result):,}")
    print(f"  features: {len(result.columns) - 2}")
    print(f"  win rate: {result[TARGET].mean():.4f}")
    print(f"  withheld {len(withheld)} columns: {', '.join(withheld)}")


if __name__ == "__main__":
    main()
