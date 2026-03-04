from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import MarketAgeAdjustedEloConfig, as_config


def assign_position_group(position: Optional[str]) -> str:
    if position is None or (isinstance(position, float) and np.isnan(position)):
        return "UNK"

    normalized = str(position).strip().lower()
    if not normalized:
        return "UNK"
    if "goal" in normalized:
        return "GK"
    if "def" in normalized:
        return "DEF"
    if "mid" in normalized:
        return "MID"
    if any(token in normalized for token in ("att", "forw", "wing", "striker")):
        return "ATT"
    return "UNK"


def compute_player_age_at_match(
    match_date: Any,
    date_of_birth: Any,
    fallback_age_years: Optional[float] = None,
) -> float:
    match_ts = pd.to_datetime(match_date, utc=True, errors="coerce")
    dob_ts = pd.to_datetime(date_of_birth, utc=True, errors="coerce")

    if pd.notna(match_ts) and pd.notna(dob_ts):
        age_years = (match_ts - dob_ts).days / 365.2425
        return float(max(age_years, 0.0))

    if fallback_age_years is not None and not pd.isna(fallback_age_years):
        return float(max(float(fallback_age_years), 0.0))

    return float("nan")


def compute_log_market_value(market_value: Any) -> float:
    if market_value is None or pd.isna(market_value):
        return float("nan")
    value = max(float(market_value), 0.0)
    return float(np.log1p(value))


@dataclass
class MarketValueNormalizer:
    levels: Tuple[Tuple[str, ...], ...]
    min_group_size: int = 30
    winsor_limits: Tuple[float, float] = (-3.0, 3.0)

    def __post_init__(self) -> None:
        self.stats_by_level: Dict[Tuple[str, ...], pd.DataFrame] = {}
        self.global_mean: float = float("nan")
        self.global_std: float = float("nan")
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame, log_mv_col: str = "log_market_value") -> "MarketValueNormalizer":
        if log_mv_col not in df.columns:
            raise KeyError(f"Missing required column: {log_mv_col}")

        fit_df = df.loc[df[log_mv_col].notna()].copy()
        if fit_df.empty:
            self.global_mean = 0.0
            self.global_std = 0.0
            self.stats_by_level = {lvl: pd.DataFrame(columns=[*lvl, "mean", "std", "count"]) for lvl in self.levels}
            self.is_fitted = True
            return self

        self.global_mean = float(fit_df[log_mv_col].mean())
        self.global_std = float(fit_df[log_mv_col].std(ddof=0))

        self.stats_by_level = {}
        for level in self.levels:
            grouped = (
                fit_df.groupby(list(level), dropna=False)[log_mv_col]
                .agg(mean="mean", std=lambda s: s.std(ddof=0), count="count")
                .reset_index()
            )
            grouped["std"] = grouped["std"].fillna(0.0)
            self.stats_by_level[level] = grouped

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, log_mv_col: str = "log_market_value") -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("MarketValueNormalizer must be fit before transform")

        out = df.copy()
        if log_mv_col not in out.columns:
            raise KeyError(f"Missing required column: {log_mv_col}")

        out["z_mv"] = np.nan
        out["mv_norm_level"] = ""
        out["mv_norm_group_size"] = np.nan

        for level in self.levels:
            stats = self.stats_by_level.get(level)
            if stats is None or stats.empty:
                continue

            join_stats = stats.rename(columns={"mean": "_mv_mean", "std": "_mv_std", "count": "_mv_count"})
            merged = out[list(level)].merge(join_stats, on=list(level), how="left")

            eligible = (
                out["z_mv"].isna()
                & out[log_mv_col].notna()
                & merged["_mv_count"].ge(self.min_group_size)
                & merged["_mv_std"].gt(0)
            )
            if not eligible.any():
                continue

            out.loc[eligible, "z_mv"] = (
                out.loc[eligible, log_mv_col] - merged.loc[eligible, "_mv_mean"]
            ) / merged.loc[eligible, "_mv_std"]
            out.loc[eligible, "mv_norm_level"] = "+".join(level)
            out.loc[eligible, "mv_norm_group_size"] = merged.loc[eligible, "_mv_count"]

        fallback_mask = out["z_mv"].isna() & out[log_mv_col].notna()
        if fallback_mask.any() and self.global_std > 0:
            out.loc[fallback_mask, "z_mv"] = (
                out.loc[fallback_mask, log_mv_col] - self.global_mean
            ) / self.global_std
            out.loc[fallback_mask, "mv_norm_level"] = "global"
            out.loc[fallback_mask, "mv_norm_group_size"] = np.nan

        missing_mv = out[log_mv_col].isna()
        out.loc[missing_mv, "z_mv"] = 0.0
        out.loc[missing_mv, "mv_norm_level"] = "missing->0"

        out["z_mv"] = out["z_mv"].fillna(0.0)
        out["z_mv"] = out["z_mv"].clip(*self.winsor_limits)
        out["market_value_missing"] = missing_mv

        return out

    def metadata(self) -> Dict[str, Any]:
        return {
            "levels": [list(level) for level in self.levels],
            "min_group_size": self.min_group_size,
            "winsor_limits": list(self.winsor_limits),
            "global_mean": self.global_mean,
            "global_std": self.global_std,
            "groups_per_level": {"+".join(level): len(stats) for level, stats in self.stats_by_level.items()},
        }


def compute_market_value_zscore(
    df: pd.DataFrame,
    config: Optional[Mapping[str, Any] | MarketAgeAdjustedEloConfig] = None,
    normalizer: Optional[MarketValueNormalizer] = None,
    fit: bool = False,
    log_mv_col: str = "log_market_value",
) -> tuple[pd.DataFrame, MarketValueNormalizer]:
    cfg = as_config(config)

    if normalizer is None:
        normalizer = MarketValueNormalizer(
            levels=cfg.market_value_levels,
            min_group_size=cfg.min_group_size_for_mv_norm,
            winsor_limits=cfg.mv_winsor_limits,
        )
        fit = True

    if fit:
        normalizer.fit(df, log_mv_col=log_mv_col)

    transformed = normalizer.transform(df, log_mv_col=log_mv_col)
    return transformed, normalizer


def _autodetect_team_elo_columns(fixtures_df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    cols = {c.lower(): c for c in fixtures_df.columns}

    candidate_pairs = [
        ("home_team_elo_pre", "away_team_elo_pre"),
        ("home_elo_pre", "away_elo_pre"),
        ("home_elo", "away_elo"),
        ("elo_home_pre", "elo_away_pre"),
        ("home_rating_pre", "away_rating_pre"),
    ]
    for home_lower, away_lower in candidate_pairs:
        if home_lower in cols and away_lower in cols:
            return cols[home_lower], cols[away_lower]

    return None, None


def _expected_home_outcome(home_rating: float, away_rating: float, home_advantage: float, scale: float = 400.0) -> float:
    delta = (home_rating + home_advantage) - away_rating
    return 1.0 / (1.0 + np.power(10.0, -delta / scale))


def build_fixture_team_elo_pre_table(
    fixtures_df: pd.DataFrame,
    config: Optional[Mapping[str, Any] | MarketAgeAdjustedEloConfig] = None,
) -> pd.DataFrame:
    cfg = as_config(config)
    fx = fixtures_df.copy()

    home_col = cfg.team_elo_home_pre_col
    away_col = cfg.team_elo_away_pre_col

    if not home_col or not away_col:
        auto_home, auto_away = _autodetect_team_elo_columns(fx)
        home_col = home_col or auto_home
        away_col = away_col or auto_away

    if home_col and away_col and home_col in fx.columns and away_col in fx.columns:
        out = fx[["fixture_id", home_col, away_col]].copy()
        out = out.rename(
            columns={
                "fixture_id": "match_id",
                home_col: "home_team_elo_pre",
                away_col: "away_team_elo_pre",
            }
        )
        out["team_elo_source"] = "fixture_precomputed"
        return out

    fx["match_date"] = pd.to_datetime(fx.get("starting_at"), utc=True, errors="coerce")
    sort_keys = ["match_date"]
    if "starting_at_timestamp" in fx.columns:
        sort_keys.append("starting_at_timestamp")
    sort_keys.append("fixture_id")
    fx = fx.sort_values(sort_keys).reset_index(drop=True)

    ratings: Dict[str, float] = defaultdict(lambda: float(cfg.fallback_team_elo_initial))
    records: List[Dict[str, Any]] = []

    for row in fx.itertuples(index=False):
        fixture_id = getattr(row, "fixture_id")
        home_team = getattr(row, "home_name")
        away_team = getattr(row, "away_name")

        home_pre = ratings[home_team]
        away_pre = ratings[away_team]

        records.append(
            {
                "match_id": fixture_id,
                "home_team_elo_pre": home_pre,
                "away_team_elo_pre": away_pre,
                "team_elo_source": "derived_fallback",
            }
        )

        state = getattr(row, "state", None)
        home_goals = getattr(row, "home_goals", np.nan)
        away_goals = getattr(row, "away_goals", np.nan)
        if state != cfg.fixture_state_for_updates:
            continue
        if pd.isna(home_goals) or pd.isna(away_goals):
            continue

        expected_home = _expected_home_outcome(
            home_pre,
            away_pre,
            cfg.fallback_team_elo_home_advantage,
            scale=cfg.elo_scale,
        )
        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals < away_goals:
            actual_home = 0.0
        else:
            actual_home = 0.5

        delta = cfg.fallback_team_elo_k * (actual_home - expected_home)
        ratings[home_team] = home_pre + delta
        ratings[away_team] = away_pre - delta

    return pd.DataFrame.from_records(records)


def build_player_match_modeling_table(
    players_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    config: Optional[Mapping[str, Any] | MarketAgeAdjustedEloConfig] = None,
    normalizer: Optional[MarketValueNormalizer] = None,
    fit_normalizer: bool = True,
    include_mv_zscore: bool = True,
) -> tuple[pd.DataFrame, Dict[str, float], Optional[MarketValueNormalizer]]:
    cfg = as_config(config)

    players = players_df.copy()
    fixtures = fixtures_df.copy()

    players = players.rename(
        columns={
            "sportmonks_player_id": "player_id",
            "fixture_id": "match_id",
            "team_name": "team_id",
            "marketvalue": "market_value_raw",
        }
    )

    fixture_cols = [
        "fixture_id",
        "home_name",
        "away_name",
        "home_goals",
        "away_goals",
        "state",
        "starting_at",
        "league",
        "season",
        "country",
    ]
    available_fixture_cols = [c for c in fixture_cols if c in fixtures.columns]
    fixture_core = fixtures[available_fixture_cols].copy().rename(
        columns={
            "fixture_id": "match_id",
            "state": "fixture_state",
            "starting_at": "fixture_starting_at",
            "league": "fixture_league",
            "season": "fixture_season",
            "country": "fixture_country",
            "home_goals": "fixture_home_goals",
            "away_goals": "fixture_away_goals",
        }
    )

    df = players.merge(fixture_core, on="match_id", how="left")

    df["league"] = df.get("league").fillna(df.get("fixture_league"))
    df["season"] = df.get("season").fillna(df.get("fixture_season"))
    if "league_country" in df.columns:
        df["country"] = df.get("league_country").fillna(df.get("fixture_country"))
    else:
        df["country"] = df.get("fixture_country")

    df["match_date"] = pd.to_datetime(
        df.get("starting_at").fillna(df.get("fixture_starting_at")),
        utc=True,
        errors="coerce",
    )

    df["position_group"] = df.get("position").map(assign_position_group)

    if cfg.position_filter:
        df = df[df["position_group"].isin(cfg.position_filter)].copy()

    df["team_id"] = df["team_id"].astype(str)
    df["is_home"] = df["team_id"] == df.get("home_name")
    df["opponent_team_id"] = np.where(df["is_home"], df.get("away_name"), df.get("home_name"))

    df["minutes_played"] = pd.to_numeric(df.get("minutes_played"), errors="coerce")
    df = df[df["minutes_played"].fillna(0.0) >= cfg.min_minutes_played].copy()

    dob = pd.to_datetime(df.get("date_of_birth"), utc=True, errors="coerce")
    age_fallback = pd.to_numeric(df.get("age"), errors="coerce")
    age_from_dob = (df["match_date"] - dob).dt.days / 365.2425
    df["player_age_years"] = age_from_dob.where(dob.notna(), age_fallback)

    df["market_value_raw"] = pd.to_numeric(df.get("market_value_raw"), errors="coerce")

    df = df.sort_values(["player_id", "match_date", "match_id"]).reset_index(drop=True)
    df["market_value"] = df.groupby("player_id", dropna=False)["market_value_raw"].ffill()
    df["log_market_value"] = np.log1p(df["market_value"].clip(lower=0.0))
    df.loc[df["market_value"].isna(), "log_market_value"] = np.nan

    team_elo_pre = build_fixture_team_elo_pre_table(fixtures, config=cfg)
    df = df.merge(team_elo_pre, on="match_id", how="left")

    df["team_rating_pre"] = np.where(df["is_home"], df["home_team_elo_pre"], df["away_team_elo_pre"])
    df["opponent_rating_pre"] = np.where(
        df["is_home"],
        df["away_team_elo_pre"],
        df["home_team_elo_pre"],
    )

    home_goals = pd.to_numeric(df.get("fixture_home_goals"), errors="coerce")
    away_goals = pd.to_numeric(df.get("fixture_away_goals"), errors="coerce")
    team_observed = np.where(
        home_goals > away_goals,
        np.where(df["is_home"], 1.0, 0.0),
        np.where(home_goals < away_goals, np.where(df["is_home"], 0.0, 1.0), 0.5),
    )
    missing_score = home_goals.isna() | away_goals.isna()
    df["team_observed_score"] = np.where(missing_score, np.nan, team_observed)

    goals = pd.to_numeric(df.get("goals"), errors="coerce").fillna(0.0)
    if cfg.performance_target == "goals_per_90_capped":
        minutes_factor = (df["minutes_played"] / 90.0).replace(0, np.nan)
        df["observed_performance_score"] = (goals / minutes_factor).fillna(0.0).clip(0.0, 1.0)
    else:
        df["observed_performance_score"] = (goals > 0).astype(float)

    df["is_update_eligible"] = df.get("fixture_state").eq(cfg.fixture_state_for_updates)

    if include_mv_zscore:
        df, normalizer = compute_market_value_zscore(
            df,
            config=cfg,
            normalizer=normalizer,
            fit=fit_normalizer or normalizer is None,
            log_mv_col="log_market_value",
        )
    else:
        df["z_mv"] = 0.0

    df = df.sort_values(["match_date", "match_id", "player_id"]).reset_index(drop=True)

    quality_summary = summarize_data_quality(df)

    required_cols = [
        "player_id",
        "match_id",
        "match_date",
        "team_id",
        "opponent_team_id",
        "position_group",
        "is_home",
        "minutes_played",
        "opponent_rating_pre",
        "player_age_years",
        "market_value",
        "observed_performance_score",
        "z_mv",
        "league",
        "season",
    ]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise KeyError(f"Failed to build modeling table; missing columns: {missing_required}")

    df["player_elo_pre"] = np.nan

    return df, quality_summary, normalizer


def summarize_data_quality(df: pd.DataFrame) -> Dict[str, float]:
    total = float(len(df)) if len(df) else 1.0
    return {
        "rows": float(len(df)),
        "missing_age_pct": float(df["player_age_years"].isna().sum() / total),
        "missing_market_value_pct": float(df["market_value"].isna().sum() / total),
        "missing_position_group_pct": float(df["position_group"].eq("UNK").sum() / total),
    }
