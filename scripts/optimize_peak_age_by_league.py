#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Sequence
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_age_elo.backtest import run_backtest_market_age_adjusted_elo
from market_age_elo.config import MarketAgeAdjustedEloConfig


def _load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}

    config_path = Path(path)
    text = config_path.read_text()

    if config_path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("YAML config requested but PyYAML is not installed. Use JSON or install PyYAML.") from exc

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("Config file must parse to a mapping/object")
    return loaded


def _parse_float_grid(values: str) -> Sequence[float]:
    out = []
    for part in values.split(","):
        token = part.strip()
        if not token:
            continue
        out.append(float(token))
    if not out:
        raise ValueError("Peak-age grid cannot be empty")
    return out


def _objective_value(metric: str, raw: float) -> float:
    if pd.isna(raw):
        return np.nan
    if metric == "mean_residual":
        return abs(float(raw))
    return float(raw)


def _variant_metric_row(
    variant_metrics: pd.DataFrame,
    variant: str,
    split: str,
) -> pd.Series:
    row = variant_metrics[(variant_metrics["variant"] == variant) & (variant_metrics["split"] == split)]
    if row.empty:
        raise KeyError(f"Missing variant/split metrics for variant={variant}, split={split}")
    return row.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search optimal peak age (ATT by default) per league using backtest validation/test metrics"
    )
    parser.add_argument("--players", default="data/elo_base_players.csv", help="Path to player-level CSV")
    parser.add_argument("--fixtures", default="data/elo_base_fixtures.csv", help="Path to fixture-level CSV")
    parser.add_argument("--config", default="config/market_age_adjusted_elo.yaml", help="Optional JSON/YAML config")
    parser.add_argument("--league-col", default="league", help="League column in player CSV")
    parser.add_argument(
        "--peak-age-grid",
        default="22,23,24,25,26,27,28,29,30",
        help="Comma-separated peak-age candidates",
    )
    parser.add_argument(
        "--position-group",
        default="ATT",
        help="Position group key in peak_age_by_position to optimize (default ATT)",
    )
    parser.add_argument(
        "--variant",
        default="market_age",
        choices=["baseline", "market_only", "age_only", "market_age"],
        help="Variant used for objective scoring",
    )
    parser.add_argument(
        "--objective-split",
        default="validation",
        choices=["train", "validation", "test", "all"],
        help="Split used to pick best peak age",
    )
    parser.add_argument(
        "--objective-metric",
        default="log_loss",
        choices=["log_loss", "brier_score", "mean_residual"],
        help="Metric used to pick best peak age",
    )
    parser.add_argument(
        "--min-player-rows-per-league",
        type=int,
        default=1000,
        help="Minimum player rows required to evaluate a league",
    )
    parser.add_argument(
        "--min-fixtures-per-league",
        type=int,
        default=100,
        help="Minimum unique fixtures required to evaluate a league",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/peak_age_by_league",
        help="Directory for search outputs",
    )
    args = parser.parse_args()

    players_df = pd.read_csv(args.players)
    fixtures_df = pd.read_csv(args.fixtures)
    base_cfg = MarketAgeAdjustedEloConfig.from_mapping(_load_config(args.config))
    peak_grid = [float(v) for v in _parse_float_grid(args.peak_age_grid)]

    if args.league_col not in players_df.columns:
        raise KeyError(f"Missing league column '{args.league_col}' in players CSV")
    if "fixture_id" not in players_df.columns or "fixture_id" not in fixtures_df.columns:
        raise KeyError("Both players and fixtures CSVs must contain fixture_id")

    leagues = (
        players_df[args.league_col]
        .dropna()
        .astype(str)
        .value_counts()
        .rename_axis("league")
        .reset_index(name="player_rows")
    )
    leagues = leagues[leagues["player_rows"] >= int(args.min_player_rows_per_league)].copy()

    search_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for league in leagues["league"]:
        league_players = players_df[players_df[args.league_col].astype(str) == league].copy()
        fixture_ids = league_players["fixture_id"].dropna().unique()
        league_fixtures = fixtures_df[fixtures_df["fixture_id"].isin(fixture_ids)].copy()
        n_fixtures = int(pd.Series(fixture_ids).nunique())

        if n_fixtures < int(args.min_fixtures_per_league):
            skipped_rows.append(
                {
                    "league": league,
                    "reason": "insufficient_fixtures",
                    "player_rows": int(len(league_players)),
                    "fixtures": n_fixtures,
                }
            )
            continue

        for peak_age in peak_grid:
            peak_map = dict(base_cfg.peak_age_by_position)
            peak_map[str(args.position_group)] = float(peak_age)
            cfg = replace(
                base_cfg,
                peak_age_by_position=peak_map,
                experiment_id=f"{base_cfg.experiment_id}_peak_{args.position_group}_{peak_age:g}",
            )

            results = run_backtest_market_age_adjusted_elo(
                players_df=league_players,
                fixtures_df=league_fixtures,
                config=cfg,
            )
            vm = results["variant_metrics"]

            obj_row = _variant_metric_row(vm, variant=args.variant, split=args.objective_split)
            objective_raw = float(obj_row[args.objective_metric])
            objective_val = _objective_value(args.objective_metric, objective_raw)

            val_row = _variant_metric_row(vm, variant=args.variant, split="validation")
            test_row = _variant_metric_row(vm, variant=args.variant, split="test")
            all_row = _variant_metric_row(vm, variant=args.variant, split="all")

            search_rows.append(
                {
                    "league": league,
                    "position_group": args.position_group,
                    "peak_age": float(peak_age),
                    "player_rows": int(len(league_players)),
                    "fixtures": n_fixtures,
                    "variant": args.variant,
                    "objective_split": args.objective_split,
                    "objective_metric": args.objective_metric,
                    "objective_raw": objective_raw,
                    "objective_value": objective_val,
                    "validation_log_loss": float(val_row["log_loss"]),
                    "validation_brier_score": float(val_row["brier_score"]),
                    "validation_mean_residual": float(val_row["mean_residual"]),
                    "test_log_loss": float(test_row["log_loss"]),
                    "test_brier_score": float(test_row["brier_score"]),
                    "test_mean_residual": float(test_row["mean_residual"]),
                    "all_log_loss": float(all_row["log_loss"]),
                    "all_brier_score": float(all_row["brier_score"]),
                    "all_mean_residual": float(all_row["mean_residual"]),
                }
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if search_rows:
        search_df = pd.DataFrame(search_rows).sort_values(
            ["league", "objective_value", "test_log_loss", "peak_age"]
        ).reset_index(drop=True)
    else:
        search_df = pd.DataFrame(
            columns=[
                "league",
                "position_group",
                "peak_age",
                "objective_value",
            ]
        )

    if not search_df.empty:
        best_df = (
            search_df.sort_values(["league", "objective_value", "test_log_loss", "peak_age"])
            .groupby("league", as_index=False)
            .first()
            .reset_index(drop=True)
        )
    else:
        best_df = pd.DataFrame(columns=["league", "peak_age"])

    skipped_df = pd.DataFrame(skipped_rows)

    search_path = out_dir / "peak_age_search_results.csv"
    best_path = out_dir / "peak_age_best_by_league.csv"
    skipped_path = out_dir / "peak_age_skipped_leagues.csv"

    search_df.to_csv(search_path, index=False)
    best_df.to_csv(best_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    summary = {
        "objective_split": args.objective_split,
        "objective_metric": args.objective_metric,
        "variant": args.variant,
        "position_group": args.position_group,
        "peak_age_grid": list(peak_grid),
        "leagues_evaluated": int(search_df["league"].nunique()) if not search_df.empty else 0,
        "leagues_skipped": int(len(skipped_df)),
        "best_by_league_rows": int(len(best_df)),
    }
    summary_path = out_dir / "peak_age_search_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Wrote files:")
    print(f"- search_results: {search_path}")
    print(f"- best_by_league: {best_path}")
    print(f"- skipped_leagues: {skipped_path}")
    print(f"- summary: {summary_path}")
    if not best_df.empty:
        print("\nBest peak age by league:")
        print(best_df[["league", "peak_age", "objective_value", "test_log_loss"]].to_string(index=False))


if __name__ == "__main__":
    main()
