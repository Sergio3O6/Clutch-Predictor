# Data and Prediction Target

This document fixes the prediction problem precisely before any model code is
written. Everything here was measured against the actual dataset, not assumed.

## Dataset

**Source:** League of Legends ranked match history, per-minute frame snapshots.
**File:** `data/raw/lol_ranked_games.csv` (37 MB, gitignored, see below).

| Property | Value |
|---|---|
| Rows | 242,572 |
| Columns | 59 |
| Unique matches (`gameId`) | 24,912 |
| Frames per match | ~10 median (minute 10 onward, every 2 minutes, max 24) |
| Nulls | 0 |
| Duplicate `(gameId, frame)` pairs | 0 |
| Label balance (`hasWon`) | 49.8% positive |

Each row is one **team's perspective** at one point in time. `hasWon` is constant
within a `gameId`. There is exactly one perspective per match rather than a
mirrored pair, so the label is already balanced without resampling.

### Why the raw file is not committed

At 37 MB it would sit in git history permanently and could not be removed later
without rewriting history. The raw file is gitignored instead, and a small
processed slice is committed to `data/processed/`, produced reproducibly from
the raw file. That keeps clones and CI checkouts fast while still letting
training run offline and deterministically.

## The target

> Given the state of a ranked match at **minute 10**, predict whether the
> observed team goes on to win.

Binary classification. The response returns the predicted label plus a
probability, so downstream consumers can threshold it themselves.

**Population:** rows where `frame == 10`, which is exactly 24,912 rows, one per
match.

There is a second reason to restrict to a single frame beyond simplicity.
Because each match contributes exactly one row, an ordinary random train/test
split is already sound. Training across multiple frames would put ~10 rows from
the same match on both sides of the split, inflating scores through memorization
of individual matches, and would require grouping the split by `gameId` to stay
honest.

### Rejected alternative: pre-match prediction

The original framing was "predict the winner before the match starts," which is
the more interesting product. It is **not buildable from this dataset**: there is
no team identity, champion, player, rank, patch, or region column. Every one of
the 59 columns is in-game state. Adopting the pre-match framing would have meant
sourcing a different dataset (pro-match data such as Oracle's Elixir carries team
and patch columns). The minute-10 framing was chosen instead to keep the focus on
the deployment pipeline rather than on data sourcing.

## Leakage exclusions

These columns are excluded from the feature set. The rules are recorded here and
enforced in the feature engineering module.

### 1. `gameDuration`, information from the future

Present in every row, but it is the *total* length of the match. At minute 10 you
cannot know how long the game will run, and duration correlates strongly with
outcome because one-sided games end early. Using it would leak the result.

### 2. Terminal-state structures

`*NexusTurret`, `*Inhibitor`, `*BaseTurret` (12 columns). These encode a base
already being broken, which means the game is effectively decided. At frame 10
they are under 0.15% nonzero and so carry almost no signal anyway, but the
exclusion is stated as a *rule* rather than a convenience: it continues to hold
if the frame is ever moved later.

### 3. Structurally impossible objectives

`killedBaronNashor`, `lostBaronNashor`, `killedElderDrake`, `lostElderDrake`. All
four are exactly **0.00% nonzero** at frame 10, because neither objective spawns
before minute 20. Constant columns contribute nothing.

### 4. Identifiers

`gameId` (match key) and `frame` (constant at 10 after filtering).

## Surviving features

Measured share of nonzero values at frame 10:

| Feature | Nonzero |
|---|---|
| `goldDiff` | 99.98% |
| `expDiff` | 99.97% |
| `isFirstBlood` | 99.06% |
| `kills` / `deaths` / `assists` | 99.06% / 98.89% / 96.59% |
| `wardsPlaced` / `wardsDestroyed` / `wardsLost` | 100% / 86.48% / 85.34% |
| `champLevelDiff` | 81.64% |
| `killedRiftHerald` / `lostRiftHerald` | 13.51% / 11.53% |
| elemental drake killed/lost pairs (8 cols) | ~9-11% each |
| `isFirstTower` | 2.52% |
| `destroyedTopOuterTurret` / `lostTopOuterTurret` | 1.33% / 1.08% |

Final column selection and any derived features are decided during feature
engineering. This document defines the boundary of what is *permitted*, not the
final list.

## Expected performance

Minute-10 state is informative but far from decisive. Comparable public work on
similar snapshots lands around 70-75% accuracy, and the baseline to beat is 50.2%
(majority class). A model materially above ~80% should be treated as a leakage
bug rather than a success. That is the main reason these exclusions are written
down before any training code exists.
