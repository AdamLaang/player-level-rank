#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_age_elo.backtest import (
    run_backtest_market_age_adjusted_elo,
    run_grid_search_market_age_adjusted_elo,
    save_backtest_outputs,
    save_grid_search_outputs,
)
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
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. Use JSON or install PyYAML."
        ) from exc

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("Config file must parse to a mapping/object")
    return loaded


def _parse_float_grid(values: str) -> Sequence[float]:
    parsed = []
    for chunk in values.split(","):
        token = chunk.strip()
        if not token:
            continue
        parsed.append(float(token))
    if not parsed:
        raise ValueError("Grid values cannot be empty")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run attacker-only market+age adjusted player ELO backtest")
    parser.add_argument("--players", default="data/elo_base_players.csv", help="Path to player-level CSV")
    parser.add_argument("--fixtures", default="data/elo_base_fixtures.csv", help="Path to fixture-level CSV")
    parser.add_argument("--config", default=None, help="Optional JSON/YAML config override")
    parser.add_argument("--output-dir", default="outputs/market_age_adjusted_elo", help="Directory for CSV outputs")
    parser.add_argument("--grid-search", action="store_true", help="Run beta_mv/beta_age grid search")
    parser.add_argument(
        "--beta-mv-grid",
        default="0,6,12,18,24,30",
        help="Comma-separated beta_mv values for grid search",
    )
    parser.add_argument(
        "--beta-age-grid",
        default="0,0.6,1.2,1.8,2.4",
        help="Comma-separated beta_age values for grid search",
    )
    parser.add_argument(
        "--objective-split",
        default="validation",
        choices=["train", "validation", "test", "all"],
        help="Split used for selecting best grid-search parameters",
    )
    parser.add_argument(
        "--objective-metric",
        default="log_loss",
        choices=["log_loss", "brier_score", "mean_residual"],
        help="Metric used for selecting best grid-search parameters",
    )
    parser.add_argument(
        "--no-rerun-best-backtest",
        action="store_true",
        help="Do not run full baseline-vs-adjusted backtest for best grid-search parameters",
    )

    args = parser.parse_args()

    config_mapping = _load_config(args.config)
    config = MarketAgeAdjustedEloConfig.from_mapping(config_mapping)

    players_df = pd.read_csv(args.players)
    fixtures_df = pd.read_csv(args.fixtures)

    if args.grid_search:
        beta_mv_grid = _parse_float_grid(args.beta_mv_grid)
        beta_age_grid = _parse_float_grid(args.beta_age_grid)
        grid_results = run_grid_search_market_age_adjusted_elo(
            players_df=players_df,
            fixtures_df=fixtures_df,
            config=config,
            beta_mv_grid=beta_mv_grid,
            beta_age_grid=beta_age_grid,
            objective_split=args.objective_split,
            objective_metric=args.objective_metric,
            rerun_best_backtest=not args.no_rerun_best_backtest,
        )
        written = save_grid_search_outputs(grid_results, args.output_dir)

        print("Grid search top results:")
        print(grid_results["grid_results"].head(10).to_string(index=False))
        print("\nBest parameters:")
        print(grid_results["best_params"])
        if isinstance(grid_results.get("best_model_backtest"), dict):
            print("\nBest-parameter variant metrics:")
            print(grid_results["best_model_backtest"]["variant_metrics"].to_string(index=False))
        print("\nQuality summary:")
        print(grid_results["quality_summary"])
    else:
        results = run_backtest_market_age_adjusted_elo(
            players_df=players_df,
            fixtures_df=fixtures_df,
            config=config,
        )
        written = save_backtest_outputs(results, args.output_dir)
        print("Variant metrics:")
        print(results["variant_metrics"].to_string(index=False))
        print("\nQuality summary:")
        print(results["quality_summary"])

    print("\nWrote files:")
    for key, path in sorted(written.items()):
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
