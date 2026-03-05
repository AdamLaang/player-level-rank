#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_age_elo.backtest import (
    _build_player_multiseason_diagnostics,
    _build_player_season_diagnostics,
)


def _resolve_csv_path(
    explicit_path: Optional[str],
    backtest_dir: Path,
    filename: str,
) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return backtest_dir / filename


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label} at: {path}\n"
            "Run the backtest first, e.g.:\n"
            "python3 scripts/run_backtest_market_age_adjusted_elo.py "
            "--config config/market_age_adjusted_elo.yaml "
            "--output-dir outputs/market_age_adjusted_elo"
        )
    return pd.read_csv(path)


def _load_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def _has_cols(df: pd.DataFrame, cols: set[str]) -> bool:
    return cols.issubset(set(df.columns))


def _primary_season_metric(df: pd.DataFrame) -> Optional[str]:
    for col in (
        "eb_prob_overperforming",
        "eb_shrunk_residual",
        "minutes_weighted_avg_residual",
        "average_residual",
        "cumulative_residual",
    ):
        if col in df.columns:
            return col
    return None


def _primary_multiseason_metric(df: pd.DataFrame) -> Optional[str]:
    for col in (
        "combined_prob_overperforming",
        "combined_shrunk_residual",
        "minutes_weighted_shrunk_residual",
        "mean_season_shrunk_residual",
        "mean_season_residual",
    ):
        if col in df.columns:
            return col
    return None


def _load_or_derive_diagnostics(
    season_path: Path,
    multiseason_path: Path,
    player_match_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_df = _load_csv_if_exists(season_path)
    multiseason_df = _load_csv_if_exists(multiseason_path)

    season_needs_rebuild = season_df is None or not _has_cols(
        season_df, {"eb_shrunk_residual", "eb_prob_overperforming", "eb_signal"}
    )
    multiseason_needs_rebuild = multiseason_df is None or not _has_cols(
        multiseason_df,
        {"combined_shrunk_residual", "combined_prob_overperforming", "combined_signal"},
    )

    if season_needs_rebuild or multiseason_needs_rebuild:
        if not player_match_path.exists():
            missing = []
            if season_df is None:
                missing.append(str(season_path))
            if multiseason_df is None:
                missing.append(str(multiseason_path))
            raise FileNotFoundError(
                "Missing/legacy diagnostics and no player match output to rebuild from.\n"
                f"Tried season diagnostics: {season_path}\n"
                f"Tried multi-season diagnostics: {multiseason_path}\n"
                f"Tried player match output: {player_match_path}\n"
                "Run the backtest again or point --player-match-csv to player_match_outputs_market_age.csv."
            )

        player_match_df = pd.read_csv(player_match_path)
        rebuilt_season = _build_player_season_diagnostics(player_match_df)
        rebuilt_multiseason = _build_player_multiseason_diagnostics(rebuilt_season)

        if season_needs_rebuild:
            season_df = rebuilt_season
            print(
                "Info: season diagnostics were missing or legacy format; "
                "rebuilt from player_match_outputs_market_age.csv."
            )
        if multiseason_needs_rebuild:
            multiseason_df = rebuilt_multiseason
            print(
                "Info: multi-season diagnostics were missing or legacy format; "
                "rebuilt from player_match_outputs_market_age.csv."
            )

    if season_df is None:
        raise FileNotFoundError(f"Missing season diagnostics at: {season_path}")
    if multiseason_df is None:
        raise FileNotFoundError(f"Missing multi-season diagnostics at: {multiseason_path}")
    return season_df, multiseason_df


def _recompute_single_season_diagnostics(
    player_match_path: Path,
    season: str,
) -> pd.DataFrame:
    if not player_match_path.exists():
        raise FileNotFoundError(
            "Single-season recompute requested but player match output is missing.\n"
            f"Tried: {player_match_path}\n"
            "Pass --player-match-csv or run backtest to generate player_match_outputs_market_age.csv."
        )
    player_match_df = pd.read_csv(player_match_path)
    if "season" not in player_match_df.columns:
        raise KeyError(
            "player_match_outputs_market_age.csv does not contain a `season` column, "
            "so single-season recompute cannot be performed."
        )
    season_df = player_match_df[player_match_df["season"].astype(str) == str(season)].copy()
    if season_df.empty:
        return pd.DataFrame()
    return _build_player_season_diagnostics(season_df)


def _filter_player(
    df: pd.DataFrame,
    player_id: Optional[int],
    player_name: Optional[str],
    player_name_match: str,
) -> pd.DataFrame:
    out = df.copy()

    if player_id is not None and "player_id" in out.columns:
        out = out[out["player_id"] == player_id]

    if player_name:
        names = out.get("player_name")
        if names is not None:
            names = names.astype(str)
            if player_name_match == "exact":
                out = out[names.str.lower() == player_name.lower()]
            else:
                out = out[names.str.lower().str.contains(player_name.lower(), na=False)]

    return out


def _print_section(title: str, df: pd.DataFrame, cols: list[str]) -> None:
    print(f"\n{title}")
    if df.empty:
        print("No rows.")
        return
    existing_cols = [c for c in cols if c in df.columns]
    print(df[existing_cols].to_string(index=False))


def _season_over_under_tables(
    season_df: pd.DataFrame,
    min_minutes: float,
    min_matches: int,
    top_n: int,
    season_filter: Optional[str],
    signal_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = season_df.copy()
    if season_filter:
        df = df[df["season"].astype(str) == str(season_filter)]

    if "minutes" in df.columns:
        df = df[df["minutes"].fillna(0.0) >= float(min_minutes)]
    if "matches" in df.columns:
        df = df[df["matches"].fillna(0) >= int(min_matches)]
    metric_col = _primary_season_metric(df)
    if metric_col is None or df.empty:
        return df.head(0).copy(), df.head(0).copy()

    over_base = df.copy()
    under_base = df.copy()
    if signal_only and "eb_signal" in df.columns:
        over_base = over_base[over_base["eb_signal"] == "likely_overperforming"]
        under_base = under_base[under_base["eb_signal"] == "likely_underperforming"]

    tie_cols = [c for c in ("minutes", "matches") if c in df.columns]
    over = (
        over_base.sort_values(
            [metric_col, *tie_cols],
            ascending=[False, *([False] * len(tie_cols))],
        )
        .head(top_n)
        .copy()
    )
    under = (
        under_base.sort_values(
            [metric_col, *tie_cols],
            ascending=[True, *([False] * len(tie_cols))],
        )
        .head(top_n)
        .copy()
    )
    return over, under


def _multiseason_over_under_tables(
    multiseason_df: pd.DataFrame,
    min_minutes: float,
    min_seasons: int,
    top_n: int,
    signal_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = multiseason_df.copy()

    if "minutes" in df.columns:
        df = df[df["minutes"].fillna(0.0) >= float(min_minutes)]
    if "seasons" in df.columns:
        df = df[df["seasons"].fillna(0) >= int(min_seasons)]
    metric_col = _primary_multiseason_metric(df)
    if metric_col is None or df.empty:
        return df.head(0).copy(), df.head(0).copy()

    over_base = df.copy()
    under_base = df.copy()
    if signal_only and "combined_signal" in df.columns:
        over_base = over_base[over_base["combined_signal"] == "likely_overperforming"]
        under_base = under_base[under_base["combined_signal"] == "likely_underperforming"]

    tie_cols = [c for c in ("minutes", "seasons", "matches") if c in df.columns]
    over = (
        over_base.sort_values(
            [metric_col, *tie_cols],
            ascending=[False, *([False] * len(tie_cols))],
        )
        .head(top_n)
        .copy()
    )
    under = (
        under_base.sort_values(
            [metric_col, *tie_cols],
            ascending=[True, *([False] * len(tie_cols))],
        )
        .head(top_n)
        .copy()
    )
    return over, under


def _save_optional(
    output_dir: Optional[str],
    tables: dict[str, pd.DataFrame],
) -> None:
    if not output_dir:
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if table is None or table.empty:
            continue
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find over/under-performing players from stabilized season and multi-season diagnostics"
    )
    parser.add_argument(
        "--backtest-dir",
        default="outputs/market_age_adjusted_elo",
        help="Directory containing diagnostics CSV outputs from backtest",
    )
    parser.add_argument(
        "--season-csv",
        default=None,
        help="Optional explicit path to player_season_diagnostics.csv",
    )
    parser.add_argument(
        "--multiseason-csv",
        default=None,
        help="Optional explicit path to player_multiseason_diagnostics.csv",
    )
    parser.add_argument(
        "--player-match-csv",
        default=None,
        help=(
            "Optional explicit path to player_match_outputs_market_age.csv. "
            "Used to rebuild missing/legacy diagnostics."
        ),
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "season", "multiseason", "player"],
        help="Report mode: all, season-only, multiseason-only, or single-player view",
    )
    parser.add_argument("--player-id", type=int, default=None, help="Player ID filter for --mode player")
    parser.add_argument("--player-name", default=None, help="Player name filter for --mode player")
    parser.add_argument(
        "--player-name-match",
        default="contains",
        choices=["contains", "exact"],
        help="Name matching strategy for --player-name",
    )
    parser.add_argument("--season", default=None, help="Optional season filter for season report (e.g. 2024/2025)")
    parser.add_argument(
        "--season-scope",
        default="within",
        choices=["within", "global_filter"],
        help=(
            "How to handle --season for season reports: "
            "`within` recomputes diagnostics inside that season only; "
            "`global_filter` filters precomputed global diagnostics."
        ),
    )
    parser.add_argument("--min-season-minutes", type=float, default=900.0, help="Minimum minutes for season ranking")
    parser.add_argument("--min-season-matches", type=int, default=10, help="Minimum matches for season ranking")
    parser.add_argument(
        "--min-multiseason-minutes",
        type=float,
        default=1800.0,
        help="Minimum total minutes for multi-season ranking",
    )
    parser.add_argument(
        "--min-multiseason-seasons",
        type=int,
        default=2,
        help="Minimum number of seasons for multi-season ranking",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Rows to show for over/under tables")
    parser.add_argument(
        "--signal-only",
        action="store_true",
        help="Keep only rows with non-uncertain signal labels",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to save produced report tables as CSV",
    )

    args = parser.parse_args()
    backtest_dir = Path(args.backtest_dir)

    season_path = _resolve_csv_path(args.season_csv, backtest_dir, "player_season_diagnostics.csv")
    multiseason_path = _resolve_csv_path(
        args.multiseason_csv, backtest_dir, "player_multiseason_diagnostics.csv"
    )
    player_match_path = _resolve_csv_path(
        args.player_match_csv, backtest_dir, "player_match_outputs_market_age.csv"
    )

    season_df, multiseason_df = _load_or_derive_diagnostics(
        season_path=season_path,
        multiseason_path=multiseason_path,
        player_match_path=player_match_path,
    )

    if args.season and args.season_scope == "within":
        season_within_df = _recompute_single_season_diagnostics(
            player_match_path=player_match_path,
            season=str(args.season),
        )
        if season_within_df.empty:
            print(f"Info: no player rows found for season={args.season} in {player_match_path}")
            season_df = season_within_df
        else:
            season_df = season_within_df
            print(
                "Info: recomputed season diagnostics within selected season "
                f"({args.season}) from player_match_outputs_market_age.csv."
            )

    saved_tables: dict[str, pd.DataFrame] = {}

    if args.mode in {"all", "season"}:
        season_over, season_under = _season_over_under_tables(
            season_df=season_df,
            min_minutes=args.min_season_minutes,
            min_matches=args.min_season_matches,
            top_n=args.top_n,
            season_filter=args.season,
            signal_only=args.signal_only,
        )
        _print_section(
            "Season Overperformers",
            season_over,
            [
                "player_id",
                "player_name",
                "season",
                "matches",
                "minutes",
                "minutes_weighted_avg_residual",
                "eb_shrunk_residual",
                "eb_prob_overperforming",
                "eb_signal",
            ],
        )
        _print_section(
            "Season Underperformers",
            season_under,
            [
                "player_id",
                "player_name",
                "season",
                "matches",
                "minutes",
                "minutes_weighted_avg_residual",
                "eb_shrunk_residual",
                "eb_prob_overperforming",
                "eb_signal",
            ],
        )
        saved_tables["season_overperformers"] = season_over
        saved_tables["season_underperformers"] = season_under

    if args.mode in {"all", "multiseason"}:
        multi_over, multi_under = _multiseason_over_under_tables(
            multiseason_df=multiseason_df,
            min_minutes=args.min_multiseason_minutes,
            min_seasons=args.min_multiseason_seasons,
            top_n=args.top_n,
            signal_only=args.signal_only,
        )
        _print_section(
            "Multi-Season Overperformers",
            multi_over,
            [
                "player_id",
                "player_name",
                "seasons",
                "matches",
                "minutes",
                "combined_shrunk_residual",
                "combined_prob_overperforming",
                "combined_signal",
            ],
        )
        _print_section(
            "Multi-Season Underperformers",
            multi_under,
            [
                "player_id",
                "player_name",
                "seasons",
                "matches",
                "minutes",
                "combined_shrunk_residual",
                "combined_prob_overperforming",
                "combined_signal",
            ],
        )
        saved_tables["multiseason_overperformers"] = multi_over
        saved_tables["multiseason_underperformers"] = multi_under

    if args.mode == "player":
        player_seasons = _filter_player(
            season_df,
            player_id=args.player_id,
            player_name=args.player_name,
            player_name_match=args.player_name_match,
        ).sort_values(["player_name", "season"])
        if args.season:
            player_seasons = player_seasons[player_seasons["season"].astype(str) == str(args.season)]

        player_multi = _filter_player(
            multiseason_df,
            player_id=args.player_id,
            player_name=args.player_name,
            player_name_match=args.player_name_match,
        ).sort_values(["player_name"])

        _print_section(
            "Player Season History",
            player_seasons,
            [
                "player_id",
                "player_name",
                "season",
                "matches",
                "minutes",
                "minutes_weighted_avg_residual",
                "eb_shrunk_residual",
                "eb_prob_overperforming",
                "eb_signal",
            ],
        )
        _print_section(
            "Player Multi-Season Summary",
            player_multi,
            [
                "player_id",
                "player_name",
                "seasons",
                "matches",
                "minutes",
                "combined_shrunk_residual",
                "combined_prob_overperforming",
                "combined_signal",
            ],
        )
        saved_tables["player_season_history"] = player_seasons
        saved_tables["player_multiseason_summary"] = player_multi

    _save_optional(args.output_dir, saved_tables)


if __name__ == "__main__":
    main()
