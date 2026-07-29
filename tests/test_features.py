"""Tests for the feature selection rules.

These exist to keep leaked columns out of the model. Each exclusion rule in
docs/DATA_AND_TARGET.md has a test here, so a column that quietly reappears in
the feature set fails the build rather than inflating the accuracy score.
"""

import pandas as pd
import pytest

from model.features import (
    PREDICTION_FRAME,
    TARGET,
    build_features,
    excluded_columns,
    feature_columns,
    select_prediction_frame,
)


@pytest.fixture
def raw_frame():
    """A miniature version of the real dataset schema.

    Two matches, each with three time slices, carrying one column from every
    exclusion category plus a handful of legitimate features.
    """
    rows = []
    for game_id in (1, 2):
        for frame in (10, 12, 14):
            rows.append(
                {
                    "gameId": game_id,
                    "frame": frame,
                    "hasWon": game_id % 2,
                    "gameDuration": 1443000,
                    "goldDiff": 100 * frame,
                    "expDiff": 50 * frame,
                    "champLevelDiff": 0.4,
                    "kills": 5,
                    "deaths": 3,
                    "assists": 7,
                    "isFirstBlood": 1,
                    "isFirstTower": 0,
                    "killedFireDrake": 1,
                    "lostFireDrake": 0,
                    "killedRiftHerald": 1,
                    "lostRiftHerald": 0,
                    "wardsPlaced": 20,
                    "wardsDestroyed": 4,
                    "wardsLost": 6,
                    "destroyedTopOuterTurret": 0,
                    "killedBaronNashor": 0,
                    "lostBaronNashor": 0,
                    "killedElderDrake": 0,
                    "lostElderDrake": 0,
                    "destroyedTopInhibitor": 0,
                    "lostMidInhibitor": 0,
                    "destroyedBotNexusTurret": 0,
                    "lostTopNexusTurret": 0,
                    "destroyedMidBaseTurret": 0,
                    "lostBotBaseTurret": 0,
                }
            )
    return pd.DataFrame(rows)


def test_game_duration_is_excluded(raw_frame):
    """gameDuration is the total match length, so it leaks the future."""
    assert "gameDuration" in excluded_columns(raw_frame.columns)
    assert "gameDuration" not in feature_columns(raw_frame.columns)


def test_terminal_structure_columns_are_excluded(raw_frame):
    """Nexus turrets, inhibitors and base turrets mean the game is already over."""
    terminal = [
        "destroyedTopInhibitor",
        "lostMidInhibitor",
        "destroyedBotNexusTurret",
        "lostTopNexusTurret",
        "destroyedMidBaseTurret",
        "lostBotBaseTurret",
    ]
    selected = feature_columns(raw_frame.columns)
    for column in terminal:
        assert column not in selected


def test_outer_turret_columns_survive(raw_frame):
    """The exclusion targets terminal structures only, not every turret column."""
    assert "destroyedTopOuterTurret" in feature_columns(raw_frame.columns)


def test_impossible_objectives_are_excluded(raw_frame):
    """Baron and Elder Drake cannot spawn before minute 20, so they are constant."""
    selected = feature_columns(raw_frame.columns)
    for column in (
        "killedBaronNashor",
        "lostBaronNashor",
        "killedElderDrake",
        "lostElderDrake",
    ):
        assert column not in selected


def test_identifiers_and_target_are_excluded(raw_frame):
    selected = feature_columns(raw_frame.columns)
    for column in ("gameId", "frame", TARGET):
        assert column not in selected


def test_legitimate_features_are_kept(raw_frame):
    selected = feature_columns(raw_frame.columns)
    for column in (
        "goldDiff",
        "expDiff",
        "champLevelDiff",
        "kills",
        "deaths",
        "assists",
        "isFirstBlood",
        "killedRiftHerald",
        "wardsPlaced",
    ):
        assert column in selected


def test_prediction_frame_gives_one_row_per_match(raw_frame):
    """A single time slice is what makes an ordinary random split sound."""
    sliced = select_prediction_frame(raw_frame)

    assert len(sliced) == raw_frame["gameId"].nunique()
    assert sliced["gameId"].is_unique
    assert (sliced["frame"] == PREDICTION_FRAME).all()


def test_build_features_splits_x_and_y(raw_frame):
    features, labels = build_features(raw_frame)

    assert len(features) == len(labels) == raw_frame["gameId"].nunique()
    assert TARGET not in features.columns
    assert list(features.columns) == feature_columns(raw_frame.columns)
    assert labels.tolist() == [1, 0]


def test_build_features_rejects_missing_target(raw_frame):
    with pytest.raises(KeyError):
        build_features(raw_frame.drop(columns=[TARGET]))
