"""Feature selection for the minute-10 win predictor.

The exclusion rules here are the reason this module exists separately from the
training script. Most of the 59 columns in the source data will leak the match
result if fed in unchecked, so the rules are expressed as data and covered by
tests rather than being buried in a training pipeline where nobody reads them.

Each rule is documented in docs/DATA_AND_TARGET.md.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

#: The point in the match the model predicts from. One row per match at this
#: frame, which is what lets an ordinary random train/test split stay honest.
PREDICTION_FRAME = 10

TARGET = "hasWon"

#: Identifiers carry no signal and would let the model memorise matches.
IDENTIFIER_COLUMNS = frozenset({"gameId", "frame"})

#: gameDuration is the *total* length of the match. At minute 10 that value has
#: not happened yet, and it tracks the outcome closely because one-sided games
#: end early.
FUTURE_INFORMATION_COLUMNS = frozenset({"gameDuration"})

#: Baron Nashor and the Elder Drake do not spawn before minute 20, so these are
#: constant at the prediction frame.
IMPOSSIBLE_OBJECTIVE_COLUMNS = frozenset(
    {
        "killedBaronNashor",
        "lostBaronNashor",
        "killedElderDrake",
        "lostElderDrake",
    }
)

#: Substrings identifying structures that only fall once a match is decided.
#: Matched by substring rather than listed explicitly so the rule survives new
#: columns, and stated in terms of what the structures mean so it still holds if
#: PREDICTION_FRAME ever moves later into the game. Outer and inner turrets are
#: deliberately absent: those fall in normal early play.
TERMINAL_STRUCTURE_MARKERS = ("NexusTurret", "Inhibitor", "BaseTurret")


def terminal_structure_columns(columns: Iterable[str]) -> set[str]:
    """Columns describing structures that only fall in a decided game."""
    return {
        column
        for column in columns
        if any(marker in column for marker in TERMINAL_STRUCTURE_MARKERS)
    }


def excluded_columns(columns: Iterable[str]) -> set[str]:
    """Every column kept out of the feature set, for any reason."""
    columns = list(columns)
    return (
        set(IDENTIFIER_COLUMNS)
        | set(FUTURE_INFORMATION_COLUMNS)
        | set(IMPOSSIBLE_OBJECTIVE_COLUMNS)
        | terminal_structure_columns(columns)
        | {TARGET}
    )


def feature_columns(columns: Iterable[str]) -> list[str]:
    """The columns the model is allowed to see, in their original order."""
    columns = list(columns)
    excluded = excluded_columns(columns)
    return [column for column in columns if column not in excluded]


def select_prediction_frame(frames: pd.DataFrame) -> pd.DataFrame:
    """Reduce the time series to the single slice the model predicts from.

    The source data holds roughly ten snapshots per match. Keeping all of them
    would put rows from the same match on both sides of a random split, and the
    model would score well by recognising individual matches.
    """
    sliced = frames.loc[frames["frame"] == PREDICTION_FRAME]
    return sliced.reset_index(drop=True)


def build_features(frames: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split raw rows into the model matrix and the labels.

    Raises:
        KeyError: if the target column is absent, which usually means the
            caller passed prediction input rather than training data.
    """
    if TARGET not in frames.columns:
        raise KeyError(f"missing target column {TARGET!r}")

    sliced = select_prediction_frame(frames)
    selected = feature_columns(frames.columns)
    return sliced[selected], sliced[TARGET]
