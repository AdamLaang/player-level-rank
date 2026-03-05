from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .config import MarketAgeAdjustedEloConfig, as_config


def _compute_age_delta(
    player_age_years: Any,
    position_group: Any,
    peak_age_by_position: Optional[Mapping[str, float]] = None,
    fallback_peak_age: float = 27.0,
) -> Any:
    peak_map = dict(peak_age_by_position or {})

    if isinstance(position_group, pd.Series):
        peak = position_group.map(lambda pos: peak_map.get(pos, fallback_peak_age)).astype(float)
        age = pd.to_numeric(player_age_years, errors="coerce")
        delta = age - peak
        return delta.where(age.notna(), 0.0)

    pos = position_group
    peak = float(peak_map.get(pos, fallback_peak_age))
    if player_age_years is None or pd.isna(player_age_years):
        return 0.0
    return float(player_age_years) - peak


def compute_age_peak_distance(
    player_age_years: Any,
    position_group: Any,
    peak_age_by_position: Optional[Mapping[str, float]] = None,
    fallback_peak_age: float = 27.0,
    distance_mode: str = "quadratic",
) -> Any:
    if distance_mode not in {"quadratic", "absolute"}:
        raise ValueError("distance_mode must be one of {'quadratic', 'absolute'}")
    delta = _compute_age_delta(
        player_age_years=player_age_years,
        position_group=position_group,
        peak_age_by_position=peak_age_by_position,
        fallback_peak_age=fallback_peak_age,
    )
    if isinstance(delta, pd.Series):
        if distance_mode == "quadratic":
            return delta**2
        return delta.abs()
    if distance_mode == "quadratic":
        return float(delta**2)
    return float(abs(delta))


def compute_age_peak_distance_sq(
    player_age_years: Any,
    position_group: Any,
    peak_age_by_position: Optional[Mapping[str, float]] = None,
    fallback_peak_age: float = 27.0,
) -> Any:
    return compute_age_peak_distance(
        player_age_years=player_age_years,
        position_group=position_group,
        peak_age_by_position=peak_age_by_position,
        fallback_peak_age=fallback_peak_age,
        distance_mode="quadratic",
    )


def compute_age_penalty_term(
    player_age_years: Any,
    position_group: Any,
    beta_age: float,
    age_penalty_mode: str = "quadratic",
    beta_age_young: Optional[float] = None,
    beta_age_old: Optional[float] = None,
    peak_age_by_position: Optional[Mapping[str, float]] = None,
    fallback_peak_age: float = 27.0,
) -> Any:
    if age_penalty_mode not in {"quadratic", "absolute", "asymmetric_quadratic"}:
        raise ValueError("age_penalty_mode must be one of {'quadratic', 'absolute', 'asymmetric_quadratic'}")

    delta = _compute_age_delta(
        player_age_years=player_age_years,
        position_group=position_group,
        peak_age_by_position=peak_age_by_position,
        fallback_peak_age=fallback_peak_age,
    )

    beta_young = float(beta_age if beta_age_young is None or pd.isna(beta_age_young) else beta_age_young)
    beta_old = float(beta_age if beta_age_old is None or pd.isna(beta_age_old) else beta_age_old)
    beta_sym = float(beta_age)

    if isinstance(delta, pd.Series):
        if age_penalty_mode == "quadratic":
            return beta_sym * (delta**2)
        if age_penalty_mode == "absolute":
            return beta_sym * delta.abs()
        out = np.where(delta < 0.0, beta_young * np.square(delta), beta_old * np.square(delta))
        return pd.Series(out, index=delta.index)

    delta_f = float(delta)
    if age_penalty_mode == "quadratic":
        return float(beta_sym * (delta_f**2))
    if age_penalty_mode == "absolute":
        return float(beta_sym * abs(delta_f))
    if delta_f < 0.0:
        return float(beta_young * (delta_f**2))
    return float(beta_old * (delta_f**2))


def compute_effective_player_rating(
    player_elo_pre: Any,
    z_mv: Any,
    z_mv_team_context: Any,
    age_peak_distance_sq: Any,
    beta_mv: float,
    beta_mv_team: float,
    beta_age: float,
    use_market_age_adjustment: bool = True,
) -> Any:
    if not use_market_age_adjustment:
        return player_elo_pre

    z_mv_clean = 0.0 if z_mv is None or pd.isna(z_mv) else z_mv
    z_mv_team_clean = (
        0.0 if z_mv_team_context is None or pd.isna(z_mv_team_context) else z_mv_team_context
    )
    age_clean = 0.0 if age_peak_distance_sq is None or pd.isna(age_peak_distance_sq) else age_peak_distance_sq

    if isinstance(player_elo_pre, pd.Series):
        elo = pd.to_numeric(player_elo_pre, errors="coerce")
        z = pd.to_numeric(z_mv_clean, errors="coerce").fillna(0.0)
        z_team = pd.to_numeric(z_mv_team_clean, errors="coerce").fillna(0.0)
        age_dist = pd.to_numeric(age_clean, errors="coerce").fillna(0.0)
        return elo + (beta_mv * z) + (beta_mv_team * z_team) - (beta_age * age_dist)

    return (
        float(player_elo_pre)
        + (beta_mv * float(z_mv_clean))
        + (beta_mv_team * float(z_mv_team_clean))
        - (beta_age * float(age_clean))
    )


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
    team_expected_scores = []
    team_observed_scores = []
    team_residuals = []
    player_vs_team_residuals = []
    mv_adjustments = []
    mv_team_adjustments = []
    age_adjustments = []

    for row in df.itertuples(index=False):
        pid = getattr(row, "player_id")
        pre = ratings[pid]

        age_penalty_term = compute_age_penalty_term(
            player_age_years=getattr(row, "player_age_years", np.nan),
            position_group=getattr(row, "position_group", None),
            beta_age=cfg.beta_age,
            age_penalty_mode=cfg.age_penalty_mode,
            beta_age_young=cfg.beta_age_young,
            beta_age_old=cfg.beta_age_old,
            peak_age_by_position=cfg.peak_age_by_position,
            fallback_peak_age=cfg.fallback_peak_age,
        )
        z_mv = getattr(row, "z_mv", 0.0)
        z_mv_team = getattr(row, "z_mv_team_context", 0.0)
        if pd.isna(age_penalty_term):
            age_penalty_term = 0.0
        if pd.isna(z_mv):
            z_mv = 0.0
        if pd.isna(z_mv_team):
            z_mv_team = 0.0
        if not cfg.use_team_market_value_context:
            z_mv_team = 0.0

        effective = compute_effective_player_rating(
            pre,
            z_mv,
            z_mv_team,
            age_penalty_term,
            beta_mv=cfg.beta_mv,
            beta_mv_team=cfg.beta_mv_team,
            beta_age=1.0,
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

        team_expected = compute_expected_player_score(
            effective_player_rating=getattr(row, "team_rating_pre"),
            opponent_rating_pre=getattr(row, "opponent_rating_pre"),
            is_home=getattr(row, "is_home"),
            home_advantage=cfg.home_advantage,
            elo_scale=cfg.elo_scale,
            home_advantage_mode=cfg.home_advantage_mode,
        )
        team_observed = getattr(row, "team_observed_score", np.nan)
        if pd.isna(team_observed):
            team_residual = 0.0
        else:
            team_residual = float(team_observed) - float(team_expected)

        observed = getattr(row, "observed_performance_score")
        residual = compute_player_residual(observed, expected)

        if cfg.use_player_vs_team_relative_update:
            adjusted_observed = float(observed) - team_residual
            post = update_player_elo(
                player_elo_pre=pre,
                observed_score=adjusted_observed,
                expected_score=expected,
                k_factor=cfg.player_k_factor,
                minutes_played=getattr(row, "minutes_played", None),
                minutes_scale_updates=cfg.minutes_scale_updates,
            )
        else:
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
        team_expected_scores.append(float(team_expected))
        team_observed_scores.append(float(team_observed) if not pd.isna(team_observed) else np.nan)
        team_residuals.append(float(team_residual))
        player_vs_team_residuals.append(float(residual) - float(team_residual))

        if cfg.use_market_age_adjustment:
            mv_adjustments.append(cfg.beta_mv * float(z_mv))
            if cfg.use_team_market_value_context:
                mv_team_adjustments.append(cfg.beta_mv_team * float(z_mv_team))
            else:
                mv_team_adjustments.append(0.0)
            age_adjustments.append(-float(age_penalty_term))
        else:
            mv_adjustments.append(0.0)
            mv_team_adjustments.append(0.0)
            age_adjustments.append(0.0)

    df["player_elo_pre"] = player_elo_pre
    df["player_elo_post"] = player_elo_post
    df["effective_player_rating"] = effective_ratings
    df["expected_score"] = expected_scores
    df["performance_residual"] = residuals
    df["team_expected_score"] = team_expected_scores
    df["team_observed_score"] = team_observed_scores
    df["team_residual"] = team_residuals
    df["player_vs_team_residual"] = player_vs_team_residuals
    df["is_overperforming"] = df["performance_residual"] > 0
    df["market_value_adjustment"] = mv_adjustments
    df["market_value_team_adjustment"] = mv_team_adjustments
    df["age_adjustment"] = age_adjustments

    return df
