from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import MarketAgeAdjustedEloConfig, as_config
from .features import (
    MarketValueNormalizer,
    build_player_match_modeling_table,
    summarize_data_quality,
)
from .model import compute_age_peak_distance_sq, run_player_elo_updates


def _load_default_data(
    players_path: str | Path = "data/elo_base_players.csv",
    fixtures_path: str | Path = "data/elo_base_fixtures.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    players = pd.read_csv(players_path)
    fixtures = pd.read_csv(fixtures_path)
    return players, fixtures


def _assign_time_splits(df: pd.DataFrame, cfg: MarketAgeAdjustedEloConfig) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="object")

    sorted_dates = pd.Series(sorted(df["match_date"].dropna().unique()))
    if sorted_dates.empty:
        return pd.Series(["test"] * len(df), index=df.index)

    n_dates = len(sorted_dates)
    train_idx = max(int(np.floor(n_dates * cfg.train_split_frac)) - 1, 0)
    valid_idx = max(int(np.floor(n_dates * (cfg.train_split_frac + cfg.validation_split_frac))) - 1, train_idx)

    train_cutoff = sorted_dates.iloc[train_idx]
    valid_cutoff = sorted_dates.iloc[min(valid_idx, n_dates - 1)]

    split = pd.Series(index=df.index, dtype="object")
    split[df["match_date"] <= train_cutoff] = "train"
    split[(df["match_date"] > train_cutoff) & (df["match_date"] <= valid_cutoff)] = "validation"
    split[df["match_date"] > valid_cutoff] = "test"
    split = split.fillna("test")
    return split


def _binary_log_loss(y_true: pd.Series, y_prob: pd.Series, eps: float = 1e-9) -> float:
    y = y_true.astype(float).clip(0.0, 1.0)
    p = y_prob.astype(float).clip(eps, 1.0 - eps)
    loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(loss.mean())


def _brier_score(y_true: pd.Series, y_prob: pd.Series) -> float:
    y = y_true.astype(float)
    p = y_prob.astype(float)
    return float(np.mean((y - p) ** 2))


def _calibration_table(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    eval_df = df[["expected_score", "observed_performance_score"]].dropna().copy()
    if eval_df.empty:
        return pd.DataFrame(columns=["bin", "count", "avg_pred", "avg_obs"])

    unique_scores = eval_df["expected_score"].nunique()
    q = min(bins, max(1, unique_scores))
    eval_df["bin"] = pd.qcut(eval_df["expected_score"], q=q, duplicates="drop")

    return (
        eval_df.groupby("bin", observed=False)
        .agg(
            count=("expected_score", "size"),
            avg_pred=("expected_score", "mean"),
            avg_obs=("observed_performance_score", "mean"),
        )
        .reset_index()
    )


def _residual_bucket_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_df = df[["player_age_years", "z_mv", "performance_residual"]].dropna(subset=["performance_residual"]).copy()
    if eval_df.empty:
        empty = pd.DataFrame(columns=["bucket", "count", "mean_residual"])
        return empty.copy(), empty.copy()

    age_bins = [16, 20, 23, 26, 29, 32, 36, 45]
    eval_df["age_bucket"] = pd.cut(eval_df["player_age_years"], bins=age_bins, include_lowest=True)

    age_table = (
        eval_df.groupby("age_bucket", observed=False)
        .agg(count=("performance_residual", "size"), mean_residual=("performance_residual", "mean"))
        .reset_index()
        .rename(columns={"age_bucket": "bucket"})
    )

    unique_mv = eval_df["z_mv"].nunique()
    q = min(5, max(1, unique_mv))
    eval_df["mv_bucket"] = pd.qcut(eval_df["z_mv"], q=q, duplicates="drop")

    mv_table = (
        eval_df.groupby("mv_bucket", observed=False)
        .agg(count=("performance_residual", "size"), mean_residual=("performance_residual", "mean"))
        .reset_index()
        .rename(columns={"mv_bucket": "bucket"})
    )

    return age_table, mv_table


def _metric_rows(pred_df: pd.DataFrame, variant_name: str) -> list[dict[str, Any]]:
    eval_base = pred_df[pred_df["is_update_eligible"]].copy()
    rows: list[dict[str, Any]] = []

    for split_name in ("train", "validation", "test", "all"):
        if split_name == "all":
            part = eval_base
        else:
            part = eval_base[eval_base["dataset_split"] == split_name]

        if part.empty:
            rows.append(
                {
                    "variant": variant_name,
                    "split": split_name,
                    "rows": 0,
                    "log_loss": np.nan,
                    "brier_score": np.nan,
                    "mean_residual": np.nan,
                }
            )
            continue

        rows.append(
            {
                "variant": variant_name,
                "split": split_name,
                "rows": int(len(part)),
                "log_loss": _binary_log_loss(part["observed_performance_score"], part["expected_score"]),
                "brier_score": _brier_score(part["observed_performance_score"], part["expected_score"]),
                "mean_residual": float(part["performance_residual"].mean()),
            }
        )

    return rows


def _build_player_season_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = ["player_id", "player_name", "season"]
    existing_group_cols = [c for c in group_cols if c in df.columns]
    grouped = df.groupby(existing_group_cols, dropna=False)
    out = (
        grouped.agg(
            matches=("match_id", "nunique"),
            minutes=("minutes_played", "sum"),
            average_residual=("performance_residual", "mean"),
            cumulative_residual=("performance_residual", "sum"),
        )
        .reset_index()
        .astype({"matches": int})
    )

    weighted = (
        df.assign(
            _minutes=df["minutes_played"].fillna(0.0),
            _weighted_residual=df["performance_residual"] * df["minutes_played"].fillna(0.0),
        )
        .groupby(existing_group_cols, dropna=False)
        .agg(_minutes_sum=("_minutes", "sum"), _weighted_sum=("_weighted_residual", "sum"))
        .reset_index()
    )

    out = out.merge(weighted, on=existing_group_cols, how="left")
    out["minutes_weighted_avg_residual"] = np.where(
        out["_minutes_sum"] > 0,
        out["_weighted_sum"] / out["_minutes_sum"],
        out["average_residual"],
    )
    out = out.drop(columns=["_minutes_sum", "_weighted_sum"])
    return out


def _build_opponent_strength_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["opponent_strength_bucket", "count", "mean_residual"])

    eval_df = df[["opponent_rating_pre", "performance_residual"]].dropna().copy()
    if eval_df.empty:
        return pd.DataFrame(columns=["opponent_strength_bucket", "count", "mean_residual"])

    q = min(5, max(1, eval_df["opponent_rating_pre"].nunique()))
    eval_df["opponent_strength_bucket"] = pd.qcut(eval_df["opponent_rating_pre"], q=q, duplicates="drop")
    return (
        eval_df.groupby("opponent_strength_bucket", observed=False)
        .agg(count=("performance_residual", "size"), mean_residual=("performance_residual", "mean"))
        .reset_index()
    )


def _prepare_modeling_table(
    players_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    cfg: MarketAgeAdjustedEloConfig,
) -> Dict[str, Any]:
    base_table, quality_summary, _ = build_player_match_modeling_table(
        players_df,
        fixtures_df,
        config=cfg,
        include_mv_zscore=False,
    )

    base_table["dataset_split"] = _assign_time_splits(base_table, cfg)

    base_table["age_peak_distance_sq"] = compute_age_peak_distance_sq(
        player_age_years=base_table["player_age_years"],
        position_group=base_table["position_group"],
        peak_age_by_position=cfg.peak_age_by_position,
        fallback_peak_age=cfg.fallback_peak_age,
    )

    train_mask = base_table["dataset_split"].eq("train")
    normalizer = MarketValueNormalizer(
        levels=cfg.market_value_levels,
        min_group_size=cfg.min_group_size_for_mv_norm,
        winsor_limits=cfg.mv_winsor_limits,
    )
    normalizer.fit(base_table.loc[train_mask])
    modeling_table = normalizer.transform(base_table)

    z_mv_summary = (
        modeling_table.groupby("position_group", dropna=False)["z_mv"]
        .agg(count="size", mean="mean", std="std", min="min", max="max")
        .reset_index()
    )

    return {
        "modeling_table": modeling_table,
        "quality_summary": quality_summary,
        "normalizer": normalizer,
        "z_mv_summary": z_mv_summary,
    }


def _evaluate_variants(
    modeling_table: pd.DataFrame,
    variant_configs: Mapping[str, MarketAgeAdjustedEloConfig],
) -> Dict[str, Any]:
    variant_outputs: Dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, Any]] = []
    calibration_tables: Dict[str, pd.DataFrame] = {}
    age_bucket_tables: Dict[str, pd.DataFrame] = {}
    mv_bucket_tables: Dict[str, pd.DataFrame] = {}

    for variant_name, variant_cfg in variant_configs.items():
        variant_df = run_player_elo_updates(modeling_table, config=variant_cfg)
        variant_df["variant"] = variant_name
        variant_outputs[variant_name] = variant_df
        metric_rows.extend(_metric_rows(variant_df, variant_name))

        test_df = variant_df[(variant_df["dataset_split"] == "test") & (variant_df["is_update_eligible"])]
        calibration_tables[variant_name] = _calibration_table(test_df)
        age_table, mv_table = _residual_bucket_tables(test_df)
        age_bucket_tables[variant_name] = age_table
        mv_bucket_tables[variant_name] = mv_table

    metrics_df = pd.DataFrame(metric_rows).sort_values(["split", "variant"]).reset_index(drop=True)
    return {
        "variant_outputs": variant_outputs,
        "variant_metrics": metrics_df,
        "calibration_tables_test": calibration_tables,
        "age_bucket_residuals_test": age_bucket_tables,
        "mv_bucket_residuals_test": mv_bucket_tables,
    }


def run_backtest_market_age_adjusted_elo(
    players_df: Optional[pd.DataFrame] = None,
    fixtures_df: Optional[pd.DataFrame] = None,
    config: Optional[Mapping[str, Any] | MarketAgeAdjustedEloConfig] = None,
) -> Dict[str, Any]:
    cfg = as_config(config)

    if players_df is None or fixtures_df is None:
        players_df, fixtures_df = _load_default_data()
    prepared = _prepare_modeling_table(players_df, fixtures_df, cfg)
    modeling_table = prepared["modeling_table"]
    quality_summary = prepared["quality_summary"]
    normalizer = prepared["normalizer"]

    variant_configs = {
        "baseline": replace(cfg, use_market_age_adjustment=False, beta_mv=0.0, beta_age=0.0),
        "market_only": replace(cfg, use_market_age_adjustment=True, beta_mv=cfg.beta_mv, beta_age=0.0),
        "age_only": replace(cfg, use_market_age_adjustment=True, beta_mv=0.0, beta_age=cfg.beta_age),
        "market_age": replace(cfg, use_market_age_adjustment=True),
    }

    evaluated = _evaluate_variants(modeling_table, variant_configs)

    main_output = evaluated["variant_outputs"]["market_age"]
    player_season_diagnostics = _build_player_season_diagnostics(main_output)
    opponent_strength_diagnostics = _build_opponent_strength_diagnostics(main_output)

    return {
        "config": cfg,
        "quality_summary": quality_summary,
        "modeling_table": modeling_table,
        "variant_outputs": evaluated["variant_outputs"],
        "variant_metrics": evaluated["variant_metrics"],
        "normalization_metadata": normalizer.metadata(),
        "z_mv_summary": prepared["z_mv_summary"],
        "calibration_tables_test": evaluated["calibration_tables_test"],
        "age_bucket_residuals_test": evaluated["age_bucket_residuals_test"],
        "mv_bucket_residuals_test": evaluated["mv_bucket_residuals_test"],
        "player_season_diagnostics": player_season_diagnostics,
        "opponent_strength_diagnostics": opponent_strength_diagnostics,
    }


def run_grid_search_market_age_adjusted_elo(
    players_df: Optional[pd.DataFrame] = None,
    fixtures_df: Optional[pd.DataFrame] = None,
    config: Optional[Mapping[str, Any] | MarketAgeAdjustedEloConfig] = None,
    beta_mv_grid: Optional[Sequence[float]] = None,
    beta_age_grid: Optional[Sequence[float]] = None,
    objective_split: str = "validation",
    objective_metric: str = "log_loss",
    rerun_best_backtest: bool = True,
) -> Dict[str, Any]:
    cfg = as_config(config)
    objective_split = str(objective_split).strip().lower()
    objective_metric = str(objective_metric).strip().lower()

    valid_splits = {"train", "validation", "test", "all"}
    if objective_split not in valid_splits:
        raise ValueError(f"objective_split must be one of {sorted(valid_splits)}")

    valid_metrics = {"log_loss", "brier_score", "mean_residual"}
    if objective_metric not in valid_metrics:
        raise ValueError(f"objective_metric must be one of {sorted(valid_metrics)}")

    if players_df is None or fixtures_df is None:
        players_df, fixtures_df = _load_default_data()

    mv_values = list(beta_mv_grid) if beta_mv_grid is not None else [0.0, 6.0, 12.0, 18.0, 24.0, 30.0]
    age_values = list(beta_age_grid) if beta_age_grid is not None else [0.0, 0.6, 1.2, 1.8, 2.4]
    if not mv_values or not age_values:
        raise ValueError("beta_mv_grid and beta_age_grid must each contain at least one value")

    mv_values = [float(v) for v in mv_values]
    age_values = [float(v) for v in age_values]

    prepared = _prepare_modeling_table(players_df, fixtures_df, cfg)
    modeling_table = prepared["modeling_table"]

    grid_rows: list[Dict[str, Any]] = []

    for beta_mv in mv_values:
        for beta_age in age_values:
            variant_cfg = replace(
                cfg,
                use_market_age_adjustment=True,
                beta_mv=beta_mv,
                beta_age=beta_age,
                experiment_id=f"{cfg.experiment_id}_mv{beta_mv:g}_age{beta_age:g}",
            )
            variant_df = run_player_elo_updates(modeling_table, config=variant_cfg)
            metric_table = pd.DataFrame(_metric_rows(variant_df, "grid")).set_index("split")

            row: Dict[str, Any] = {"beta_mv": beta_mv, "beta_age": beta_age}
            for split in ("train", "validation", "test", "all"):
                if split not in metric_table.index:
                    row[f"{split}_rows"] = 0
                    row[f"{split}_log_loss"] = np.nan
                    row[f"{split}_brier_score"] = np.nan
                    row[f"{split}_mean_residual"] = np.nan
                    continue
                split_metrics = metric_table.loc[split]
                row[f"{split}_rows"] = int(split_metrics["rows"])
                row[f"{split}_log_loss"] = float(split_metrics["log_loss"])
                row[f"{split}_brier_score"] = float(split_metrics["brier_score"])
                row[f"{split}_mean_residual"] = float(split_metrics["mean_residual"])

            objective_raw = row.get(f"{objective_split}_{objective_metric}", np.nan)
            row["objective_split"] = objective_split
            row["objective_metric"] = objective_metric
            row["objective_raw"] = objective_raw
            if pd.isna(objective_raw):
                row["objective_value"] = np.nan
            elif objective_metric == "mean_residual":
                row["objective_value"] = abs(float(objective_raw))
            else:
                row["objective_value"] = float(objective_raw)
            grid_rows.append(row)

    grid_results = pd.DataFrame(grid_rows).sort_values(
        ["objective_value", "test_log_loss", "beta_mv", "beta_age"],
        na_position="last",
    ).reset_index(drop=True)
    best_rows = grid_results.dropna(subset=["objective_value"])
    best_params: Optional[Dict[str, float]]
    if best_rows.empty:
        best_params = None
    else:
        best = best_rows.iloc[0]
        best_params = {"beta_mv": float(best["beta_mv"]), "beta_age": float(best["beta_age"])}

    best_backtest = None
    if rerun_best_backtest and best_params is not None:
        best_cfg = replace(
            cfg,
            use_market_age_adjustment=True,
            beta_mv=best_params["beta_mv"],
            beta_age=best_params["beta_age"],
            experiment_id=f"{cfg.experiment_id}_grid_best",
        )
        best_backtest = run_backtest_market_age_adjusted_elo(
            players_df=players_df,
            fixtures_df=fixtures_df,
            config=best_cfg,
        )

    return {
        "config": cfg,
        "quality_summary": prepared["quality_summary"],
        "normalization_metadata": prepared["normalizer"].metadata(),
        "z_mv_summary": prepared["z_mv_summary"],
        "beta_mv_grid": mv_values,
        "beta_age_grid": age_values,
        "objective_split": objective_split,
        "objective_metric": objective_metric,
        "grid_results": grid_results,
        "best_params": best_params,
        "best_model_backtest": best_backtest,
    }


def save_backtest_outputs(results: Dict[str, Any], output_dir: str | Path) -> Dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}

    results["variant_metrics"].to_csv(out_dir / "variant_metrics.csv", index=False)
    written["variant_metrics"] = out_dir / "variant_metrics.csv"

    results["modeling_table"].to_csv(out_dir / "modeling_table.csv", index=False)
    written["modeling_table"] = out_dir / "modeling_table.csv"

    results["player_season_diagnostics"].to_csv(out_dir / "player_season_diagnostics.csv", index=False)
    written["player_season_diagnostics"] = out_dir / "player_season_diagnostics.csv"

    results["opponent_strength_diagnostics"].to_csv(out_dir / "opponent_strength_diagnostics.csv", index=False)
    written["opponent_strength_diagnostics"] = out_dir / "opponent_strength_diagnostics.csv"

    for variant, table in results["calibration_tables_test"].items():
        path = out_dir / f"calibration_test_{variant}.csv"
        table.to_csv(path, index=False)
        written[f"calibration_test_{variant}"] = path

    for variant, table in results["age_bucket_residuals_test"].items():
        path = out_dir / f"age_bucket_residuals_test_{variant}.csv"
        table.to_csv(path, index=False)
        written[f"age_bucket_residuals_test_{variant}"] = path

    for variant, table in results["mv_bucket_residuals_test"].items():
        path = out_dir / f"mv_bucket_residuals_test_{variant}.csv"
        table.to_csv(path, index=False)
        written[f"mv_bucket_residuals_test_{variant}"] = path

    for variant, table in results["variant_outputs"].items():
        path = out_dir / f"player_match_outputs_{variant}.csv"
        table.to_csv(path, index=False)
        written[f"player_match_outputs_{variant}"] = path

    results["z_mv_summary"].to_csv(out_dir / "z_mv_summary.csv", index=False)
    written["z_mv_summary"] = out_dir / "z_mv_summary.csv"

    quality_df = pd.DataFrame([results["quality_summary"]])
    quality_df.to_csv(out_dir / "data_quality_summary.csv", index=False)
    written["data_quality_summary"] = out_dir / "data_quality_summary.csv"

    return written


def save_grid_search_outputs(results: Dict[str, Any], output_dir: str | Path) -> Dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}

    results["grid_results"].to_csv(out_dir / "grid_search_results.csv", index=False)
    written["grid_search_results"] = out_dir / "grid_search_results.csv"

    summary = {
        "objective_split": results["objective_split"],
        "objective_metric": results["objective_metric"],
        "beta_mv_grid": results["beta_mv_grid"],
        "beta_age_grid": results["beta_age_grid"],
        "best_params": results["best_params"],
        "quality_summary": results["quality_summary"],
    }
    summary_path = out_dir / "grid_search_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    written["grid_search_summary"] = summary_path

    results["z_mv_summary"].to_csv(out_dir / "z_mv_summary.csv", index=False)
    written["z_mv_summary"] = out_dir / "z_mv_summary.csv"

    quality_df = pd.DataFrame([results["quality_summary"]])
    quality_df.to_csv(out_dir / "data_quality_summary.csv", index=False)
    written["data_quality_summary"] = out_dir / "data_quality_summary.csv"

    best_backtest = results.get("best_model_backtest")
    if isinstance(best_backtest, dict):
        nested = save_backtest_outputs(best_backtest, out_dir / "best_model_backtest")
        for key, path in nested.items():
            written[f"best_model_backtest/{key}"] = path

    return written
