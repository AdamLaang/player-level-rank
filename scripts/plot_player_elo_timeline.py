#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_age_elo.visualization import (
    load_player_timeline_data,
    plot_player_elo_timeline_interactive,
    plot_player_elo_timeline,
    select_player_timeseries,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot player ranking timeline with optional age/market-value overlays"
    )
    parser.add_argument(
        "--input-csv",
        default="outputs/market_age_adjusted_elo_grid_wide_mv_k_matched_off/best_model_backtest/player_match_outputs_market_age.csv",
        help="Path to player match output CSV",
    )
    parser.add_argument("--player-id", type=int, default=None, help="Player ID selector")
    parser.add_argument("--player-name", default=None, help="Player name selector (exact match)")
    parser.add_argument(
        "--ranking-col",
        "--elo-col",
        dest="ranking_col",
        default="player_elo_post",
        help="Player ranking column to plot (alias: --elo-col)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Rolling window size for the smoothed player ranking line",
    )
    parser.add_argument(
        "--backend",
        default="plotly",
        choices=["plotly", "matplotlib"],
        help="Plotting backend",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output path. Default: .html for plotly, .png for matplotlib",
    )
    parser.add_argument(
        "--hide-age",
        action="store_true",
        help="Do not include age line/panel",
    )
    parser.add_argument(
        "--hide-market-value",
        action="store_true",
        help="Do not include market-value line/panel",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay series on multiple y-axes instead of stacked subplots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figure interactively in addition to saving",
    )

    args = parser.parse_args()

    df = load_player_timeline_data(args.input_csv)
    player_df = select_player_timeseries(df, player_id=args.player_id, player_name=args.player_name)

    if args.output_path:
        out_path = Path(args.output_path)
    else:
        selector = str(args.player_id) if args.player_id is not None else str(args.player_name or "player")
        selector = selector.strip().replace(" ", "_").lower()
        default_suffix = ".html" if args.backend == "plotly" else ".png"
        out_path = ROOT / "outputs" / "plots" / f"player_{selector}_{args.ranking_col}{default_suffix}"

    if args.backend == "plotly":
        plot_player_elo_timeline_interactive(
            player_df=player_df,
            output_path=out_path,
            elo_col=args.ranking_col,
            smooth_window=args.smooth_window,
            include_age=not args.hide_age,
            include_market_value=not args.hide_market_value,
            use_subplots=not args.overlay,
            show=args.show,
        )
    else:
        plot_player_elo_timeline(
            player_df=player_df,
            output_path=out_path,
            elo_col=args.ranking_col,
            smooth_window=args.smooth_window,
            include_age=not args.hide_age,
            include_market_value=not args.hide_market_value,
            use_subplots=not args.overlay,
            show=args.show,
        )
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
