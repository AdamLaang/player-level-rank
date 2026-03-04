from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .config import MarketAgeAdjustedEloConfig, as_config


def compute_age_peak_distance_sq(
    player_age_years: Any,
    position_group: Any,
    peak_age_by_position: Optional[Mapping[str, float]] = None,
    fallback_peak_age: float = 27.0,
) -> Any:
    peak_map = dict(peak_age_by_position or {})

    if isinstance(position_group, pd.Series):
        peak = position_group.map(lambda pos: peak_map.get(pos, fallback_peak_age)).astype(float)
        age = pd.to_numeric(player_age_years, errors="coerce")
        out = (age - peak) ** 2
        out = out.where(age.notna(), 0.0)
        return out

    pos = position_group
    peak = float(peak_map.get(pos, fallback_peak_age))
    if player_age_years is None or pd.isna(player_age_years):
        return 0.0
    age_value = float(player_age_years)
    return float((age_value - peak) ** 2)


def compute_effective_player_rating(
    player_elo_pre: Any,
    z_mv: Any,
    age_peak_distance_sq: Any,
    beta_mv: float,
    beta_age: float,
    use_market_age_adjustment: bool = True,
) -> Any:
    if not use_market_age_adjustment:
        return player_elo_pre

    z_mv_clean = 0.0 if z_mv is None or pd.isna(z_mv) else z_mv
    age_clean = 0.0 if age_peak_distance_sq is None or pd.isna(age_peak_distance_sq) else age_peak_distance_sq

    if isinstance(player_elo_pre, pd.Series):
        elo = pd.to_numeric(player_elo_pre, errors="coerce")
        z = pd.to_numeric(z_mv_clean, errors="coerce").fillna(0.0)
        age_dist = pd.to_numeric(age_clean, errors="coerce").fillna(0.0)
        return elo + (beta_mv * z) - (beta_age * age_dist)

    return float(player_elo_pre) + (beta_mv * float(z_mv_clean)) - (beta_age * float(age_clean))


def compute_expected_player_score(
    effective_player_rating: Any,
    opponent_rating_pre: Any,
    is_home: Any,
    home_advantage: float,
    elo_scale: float,
    home_advantage_mode: str = "symmetric",
) -> Any:
    if home_advantage_mode not in {"symmetric", "home_only"}:
        raise ValueError("home_advantage_mode must be one of {'symmetric', 'home_only'}")

    if isinstance(effective_player_rating, pd.Series) or isinstance(opponent_rating_pre, pd.Series):
        eff = pd.to_numeric(effective_player_rating, errors="coerce")
        opp = pd.to_numeric(opponent_rating_pre, errors="coerce")
        home_series = pd.Series(is_home, index=eff.index, dtype="boolean")

        if home_advantage_mode == "symmetric":
            h = np.where(home_series.fillna(False), home_advantage, -home_advantage)
        else:
            h = np.where(home_series.fillna(False), home_advantage, 0.0)

        delta = eff + h - opp
        expected = 1.0 / (1.0 + np.power(10.0, -delta / elo_scale))
        return pd.Series(expected, index=eff.index).clip(1e-6, 1.0 - 1e-6)

    home_flag = bool(is_home)
    if home_advantage_mode == "symmetric":
        h = home_advantage if home_flag else -home_advantage
    else:
        h = home_advantage if home_flag else 0.0

    delta = float(effective_player_rating) + h - float(opponent_rating_pre)
    expected = 1.0 / (1.0 + np.power(10.0, -delta / elo_scale))
    return float(np.clip(expected, 1e-6, 1.0 - 1e-6))


def compute_player_residual(observed_score: Any, expected_score: Any) -> Any:
    if isinstance(observed_score, pd.Series) or isinstance(expected_score, pd.Series):
        observed = pd.to_numeric(observed_score, errors="coerce")
        expected = pd.to_numeric(expected_score, errors="coerce")
        return observed - expected

    return float(observed_score) - float(expected_score)


def update_player_elo(
    player_elo_pre: float,
    observed_score: float,
    expected_score: float,
    k_factor: float,
    minutes_played: Optional[float] = None,
    minutes_scale_updates: bool = True,
) -> float:
    k_eff = float(k_factor)
    if minutes_scale_updates:
        if minutes_played is None or pd.isna(minutes_played):
            minutes_weight = 1.0
        else:
            minutes_weight = float(np.clip(float(minutes_played) / 90.0, 0.0, 1.0))
        k_eff *= minutes_weight

    return float(player_elo_pre) + k_eff * (float(observed_score) - float(expected_score))


def run_player_elo_updates(
    modeling_df: pd.DataFrame,
    config: Optional[Mapping[str, Any] | MarketAgeAdjustedEloConfig] = None,
) -> pd.DataFrame:
    cfg = as_config(config)

    df = modeling_df.copy()
    if df.empty:
        return df

    df = df.sort_values(["match_date", "match_id", "player_id"]).reset_index(drop=True)

    ratings: Dict[Any, float] = defaultdict(lambda: float(cfg.initial_player_elo))

    player_elo_pre = []
    player_elo_post = []
    effective_ratings = []
    expected_scores = []
    residuals = []
    mv_adjustments = []
    age_adjustments = []

    for row in df.itertuples(index=False):
        pid = getattr(row, "player_id")
        pre = ratings[pid]

        age_dist = getattr(row, "age_peak_distance_sq", 0.0)
        z_mv = getattr(row, "z_mv", 0.0)
        if pd.isna(age_dist):
            age_dist = 0.0
        if pd.isna(z_mv):
            z_mv = 0.0

        effective = compute_effective_player_rating(
            pre,
            z_mv,
            age_dist,
            beta_mv=cfg.beta_mv,
            beta_age=cfg.beta_age,
            use_market_age_adjustment=cfg.use_market_age_adjustment,
        )

        expected = compute_expected_player_score(
            effective_player_rating=effective,
            opponent_rating_pre=getattr(row, "opponent_rating_pre"),
            is_home=getattr(row, "is_home"),
            home_advantage=cfg.home_advantage,
            elo_scale=cfg.elo_scale,
            home_advantage_mode=cfg.home_advantage_mode,
        )

        observed = getattr(row, "observed_performance_score")
        residual = compute_player_residual(observed, expected)

        post = update_player_elo(
            player_elo_pre=pre,
            observed_score=observed,
            expected_score=expected,
            k_factor=cfg.player_k_factor,
            minutes_played=getattr(row, "minutes_played", None),
            minutes_scale_updates=cfg.minutes_scale_updates,
        )

        if bool(getattr(row, "is_update_eligible", True)):
            ratings[pid] = post
        else:
            post = pre

        player_elo_pre.append(pre)
        player_elo_post.append(post)
        effective_ratings.append(float(effective))
        expected_scores.append(float(expected))
        residuals.append(float(residual))

        if cfg.use_market_age_adjustment:
            mv_adjustments.append(cfg.beta_mv * float(z_mv))
            age_adjustments.append(-cfg.beta_age * float(age_dist))
        else:
            mv_adjustments.append(0.0)
            age_adjustments.append(0.0)

    df["player_elo_pre"] = player_elo_pre
    df["player_elo_post"] = player_elo_post
    df["effective_player_rating"] = effective_ratings
    df["expected_score"] = expected_scores
    df["performance_residual"] = residuals
    df["is_overperforming"] = df["performance_residual"] > 0
    df["market_value_adjustment"] = mv_adjustments
    df["age_adjustment"] = age_adjustments

    return df
