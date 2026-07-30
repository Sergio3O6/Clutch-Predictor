"""Request and response models for the prediction service.

Every one of the model's 33 features is required. Nothing is optional and
nothing defaults, because a missing field would otherwise become a zero that
the model weights as though it were observed. Rejecting the request is the
honest response to input we do not have.

Field names are snake_case on the wire and converted to the dataset's camelCase
column names on the way into the model.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

#: Counts cannot be negative. No upper bounds are asserted: the plausible
#: maximum for kills or wards at minute 10 has not been measured, and inventing
#: a ceiling risks rejecting legitimate input for the sake of looking strict.
Count = Field(ge=0)


class MatchState(BaseModel):
    """The state of a match at minute 10, from one team's perspective."""

    # extra="forbid" is deliberate. Silently ignoring an unrecognised field
    # would let a caller send gameDuration, believe it was used, and get a
    # prediction that ignored it.
    model_config = ConfigDict(extra="forbid")

    gold_diff: int
    exp_diff: int
    champ_level_diff: float

    is_first_tower: bool
    is_first_blood: bool

    killed_fire_drake: int = Count
    killed_water_drake: int = Count
    killed_air_drake: int = Count
    killed_earth_drake: int = Count
    lost_fire_drake: int = Count
    lost_water_drake: int = Count
    lost_air_drake: int = Count
    lost_earth_drake: int = Count

    killed_rift_herald: int = Count
    lost_rift_herald: int = Count

    destroyed_top_inner_turret: int = Count
    destroyed_mid_inner_turret: int = Count
    destroyed_bot_inner_turret: int = Count
    lost_top_inner_turret: int = Count
    lost_mid_inner_turret: int = Count
    lost_bot_inner_turret: int = Count

    destroyed_top_outer_turret: int = Count
    destroyed_mid_outer_turret: int = Count
    destroyed_bot_outer_turret: int = Count
    lost_top_outer_turret: int = Count
    lost_mid_outer_turret: int = Count
    lost_bot_outer_turret: int = Count

    kills: int = Count
    deaths: int = Count
    assists: int = Count

    wards_placed: int = Count
    wards_destroyed: int = Count
    wards_lost: int = Count

    def to_feature_row(self, columns: list[str]) -> pd.DataFrame:
        """Build a single-row frame with columns in the order the model expects.

        Reindexing against the pipeline's own feature names rather than trusting
        declaration order means a reordered schema cannot silently feed the
        model its features shuffled.
        """
        row = {to_camel(name): value for name, value in self.model_dump().items()}
        return pd.DataFrame([row], columns=columns)


class Prediction(BaseModel):
    """The model's call, plus the probability it is based on."""

    will_win: bool
    win_probability: float = Field(ge=0.0, le=1.0)


class Health(BaseModel):
    status: str
    model_loaded: bool
    n_features: int
