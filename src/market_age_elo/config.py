from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


DEFAULT_PEAK_AGE_BY_POSITION: Dict[str, float] = {
    "GK": 29.0,
    "DEF": 27.0,
    "MID": 27.0,
    "ATT": 26.0,
}


@dataclass(frozen=True)
class MarketAgeAdjustedEloConfig:
    use_market_age_adjustment: bool = True
    elo_scale: float = 400.0
    home_advantage: float = 40.0
    home_advantage_mode: str = "symmetric"  # symmetric|home_only

    beta_mv: float = 18.0
    beta_age: float = 1.2
    peak_age_by_position: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PEAK_AGE_BY_POSITION)
    )
    fallback_peak_age: float = 27.0

    market_value_normalization_level: Tuple[str, ...] = (
        "season",
        "league",
        "position_group",
    )
    min_group_size_for_mv_norm: int = 30
    mv_winsor_limits: Tuple[float, float] = (-3.0, 3.0)

    position_filter: Tuple[str, ...] = ("ATT",)
    min_minutes_played: float = 1.0
    fixture_state_for_updates: str = "Full Time"

    initial_player_elo: float = 1500.0
    player_k_factor: float = 20.0
    minutes_scale_updates: bool = True

    performance_target: str = "scored_binary"  # scored_binary|goals_per_90_capped

    team_elo_home_pre_col: Optional[str] = None
    team_elo_away_pre_col: Optional[str] = None
    fallback_team_elo_initial: float = 1500.0
    fallback_team_elo_k: float = 20.0
    fallback_team_elo_home_advantage: float = 40.0

    train_split_frac: float = 0.7
    validation_split_frac: float = 0.15

    experiment_id: str = "market_age_adjusted_elo"

    @staticmethod
    def from_mapping(mapping: Optional[Mapping[str, Any]]) -> "MarketAgeAdjustedEloConfig":
        if mapping is None:
            return MarketAgeAdjustedEloConfig()

        defaults = MarketAgeAdjustedEloConfig()
        values: Dict[str, Any] = {}
        for field_name in defaults.__dataclass_fields__.keys():  # type: ignore[attr-defined]
            if field_name in mapping:
                values[field_name] = mapping[field_name]

        if "peak_age_by_position" in values:
            merged = dict(defaults.peak_age_by_position)
            merged.update(values["peak_age_by_position"] or {})
            values["peak_age_by_position"] = merged

        if "market_value_normalization_level" in values:
            values["market_value_normalization_level"] = tuple(
                values["market_value_normalization_level"]
            )
        if "position_filter" in values:
            values["position_filter"] = tuple(values["position_filter"])

        config = MarketAgeAdjustedEloConfig(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.elo_scale <= 0:
            raise ValueError("elo_scale must be > 0")
        if self.player_k_factor < 0:
            raise ValueError("player_k_factor must be >= 0")
        if self.min_group_size_for_mv_norm < 1:
            raise ValueError("min_group_size_for_mv_norm must be >= 1")
        if self.home_advantage_mode not in {"symmetric", "home_only"}:
            raise ValueError("home_advantage_mode must be one of {'symmetric', 'home_only'}")
        if self.train_split_frac <= 0 or self.train_split_frac >= 1:
            raise ValueError("train_split_frac must be in (0,1)")
        if self.validation_split_frac <= 0 or self.validation_split_frac >= 1:
            raise ValueError("validation_split_frac must be in (0,1)")
        if self.train_split_frac + self.validation_split_frac >= 1:
            raise ValueError("train_split_frac + validation_split_frac must be < 1")
        if len(self.mv_winsor_limits) != 2 or self.mv_winsor_limits[0] > self.mv_winsor_limits[1]:
            raise ValueError("mv_winsor_limits must be an ordered pair (low, high)")

    @property
    def market_value_levels(self) -> Tuple[Tuple[str, ...], ...]:
        base = tuple(self.market_value_normalization_level)
        fallbacks = [base]

        if "league" in base:
            fallbacks.append(tuple(c for c in base if c != "league"))
        if "season" in base:
            fallbacks.append(tuple(c for c in base if c != "season"))
        if "position_group" not in base:
            fallbacks.append(("position_group",))
        else:
            fallbacks.append(("position_group",))

        deduped = []
        for lvl in fallbacks:
            if not lvl:
                continue
            if lvl not in deduped:
                deduped.append(lvl)
        return tuple(deduped)


def as_config(config: Optional[Mapping[str, Any] | MarketAgeAdjustedEloConfig]) -> MarketAgeAdjustedEloConfig:
    if isinstance(config, MarketAgeAdjustedEloConfig):
        config.validate()
        return config
    return MarketAgeAdjustedEloConfig.from_mapping(config)
