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

Present in every row, but it is the *total* length of the match, which has not
happened yet at minute 10.

The reason to exclude it is the API contract rather than accuracy. A caller
predicting a live match cannot supply this value. If it were a model input,
`/predict` would demand a field nobody can fill, and callers would end up
passing a zero or a guess that the model then weights as though it were real.

It is worth being precise about the accuracy claim, because the obvious
assumption turns out to be wrong. Duration was measured against the outcome:

| Measurement | Value |
|---|---|
| Correlation with `hasWon` | -0.0301 |
| Correlation with `abs(goldDiff)` | -0.3904 |
| Win rate, matches under 20 min | 0.5293 |
| Win rate, matches over 35 min | 0.4848 |

Duration tracks the *margin* of a match, not its *direction*. Short games are
one-sided, but because this dataset records a single team perspective per match,
a short game is roughly as likely to be a fast loss as a fast win. Adding the
column back to the feature set moves holdout accuracy by -0.0004, and the model
assigns it a coefficient of -0.075 against +0.962 for `goldDiff`.

So excluding it costs nothing and gains nothing measurable here. It stays
excluded because it is unavailable at prediction time, and because that weak
relationship is not guaranteed to stay weak on a dataset carrying both
perspectives or a different distribution of game lengths.

### 2. Terminal-state structures

`*NexusTurret`, `*Inhibitor`, `*BaseTurret` (12 columns). These encode a base
already being broken, which means the game is effectively decided. At frame 10
they are under 0.15% nonzero and so carry almost no signal anyway, but the
exclusion is stated as a *rule* rather than a convenience: it continues to hold
if the frame is ever moved later.

Measured, these columns are not currently doing damage. Adding all twelve back
moves holdout accuracy by -0.0006 at frame 10 and by -0.0019 at frame 24. The
`destroyed*` and `lost*` pairs partly cancel, and by the time a base is falling
the legitimate features already describe a decided game. The rule is kept
because it is stated in terms of what the structures mean, which does not depend
on those numbers staying where they are.

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

Minute-10 state is informative but far from decisive. The baseline to beat is
50.3% (majority class), and the current logistic regression reaches 73.1%
holdout accuracy with an ROC AUC of 0.807 and a Brier score of 0.180.
Cross-validation on the training split gives 72.3% (+/- 0.3%), so the holdout is
not a lucky draw.

At frame 10, accuracy materially above 80% should be treated as a leakage bug
rather than a success. That ceiling is specific to this prediction point and
does not generalise: the same honest feature set reaches 84.0% at frame 24,
simply because more of the match has happened. Any change to the prediction
frame needs its own ceiling.

One further note on splitting. Training across every frame with an ordinary
random split scores 78.8%, against 77.6% for a split grouped by `gameId`, with
19,933 of 24,912 matches appearing on both sides. The 1.2 point gap is real but
small because logistic regression has only 33 coefficients and cannot memorise
individual matches. A higher-capacity model would inflate considerably more
under the same contamination, so the single-frame design matters more, not less,
if the model is ever changed.
