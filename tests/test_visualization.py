from __future__ import annotations

import sys
from pathlib import Path
import unittest

import matplotlib

matplotlib.use("Agg")
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_age_elo.visualization import (
    plot_player_elo_timeline,
    plot_player_elo_timeline_interactive,
    select_player_timeseries,
)


class TestVisualization(unittest.TestCase):
    @staticmethod
    def _sample_player_df() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "player_id": 100,
                    "player_name": "Attacker A",
                    "match_date": "2024-08-01T12:00:00+00:00",
                    "player_elo_post": 1510.0,
                    "performance_residual": 0.12,
                    "player_age_years": 24.2,
                    "market_value": 20_000_000,
                },
                {
                    "match_id": 2,
                    "player_id": 100,
                    "player_name": "Attacker A",
                    "match_date": "2024-08-08T12:00:00+00:00",
                    "player_elo_post": 1522.0,
                    "performance_residual": -0.03,
                    "player_age_years": 24.3,
                    "market_value": 22_000_000,
                },
                {
                    "match_id": 1,
                    "player_id": 200,
                    "player_name": "Attacker B",
                    "match_date": "2024-08-01T12:00:00+00:00",
                    "player_elo_post": 1495.0,
                    "performance_residual": -0.11,
                    "player_age_years": 26.2,
                    "market_value": 15_000_000,
                },
            ]
        )

    @staticmethod
    def _sample_player_df_long() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "match_id": 10,
                    "player_id": 100,
                    "player_name": "Attacker A",
                    "match_date": "2024-08-01T12:00:00+00:00",
                    "player_elo_post": 1505.0,
                    "performance_residual": 0.20,
                    "player_age_years": 24.1,
                    "market_value": 20_000_000,
                },
                {
                    "match_id": 11,
                    "player_id": 100,
                    "player_name": "Attacker A",
                    "match_date": "2024-08-08T12:00:00+00:00",
                    "player_elo_post": 1515.0,
                    "performance_residual": -0.10,
                    "player_age_years": 24.2,
                    "market_value": 21_000_000,
                },
                {
                    "match_id": 12,
                    "player_id": 100,
                    "player_name": "Attacker A",
                    "match_date": "2024-08-15T12:00:00+00:00",
                    "player_elo_post": 1528.0,
                    "performance_residual": 0.15,
                    "player_age_years": 24.3,
                    "market_value": 22_000_000,
                },
                {
                    "match_id": 13,
                    "player_id": 100,
                    "player_name": "Attacker A",
                    "match_date": "2024-08-22T12:00:00+00:00",
                    "player_elo_post": 1536.0,
                    "performance_residual": -0.08,
                    "player_age_years": 24.4,
                    "market_value": 23_000_000,
                },
                {
                    "match_id": 14,
                    "player_id": 100,
                    "player_name": "Attacker A",
                    "match_date": "2024-08-29T12:00:00+00:00",
                    "player_elo_post": 1542.0,
                    "performance_residual": 0.10,
                    "player_age_years": 24.5,
                    "market_value": 24_000_000,
                },
            ]
        )

    def test_select_player_timeseries(self) -> None:
        df = self._sample_player_df()
        df["match_date"] = pd.to_datetime(df["match_date"], utc=True)

        out_by_id = select_player_timeseries(df, player_id=100)
        self.assertEqual(len(out_by_id), 2)
        self.assertTrue((out_by_id["player_id"] == 100).all())

        out_by_name = select_player_timeseries(df, player_name="Attacker B")
        self.assertEqual(len(out_by_name), 1)
        self.assertEqual(int(out_by_name.iloc[0]["player_id"]), 200)

    def test_plot_player_elo_timeline_smoke(self) -> None:
        df = self._sample_player_df()
        df["match_date"] = pd.to_datetime(df["match_date"], utc=True)
        player_df = select_player_timeseries(df, player_id=100)

        out_dir = ROOT / "outputs" / "test_plots"
        out_path = out_dir / "attacker_a_timeline.png"
        fig = plot_player_elo_timeline(
            player_df=player_df,
            output_path=out_path,
            elo_col="player_elo_post",
            include_age=True,
            include_market_value=True,
            use_subplots=True,
            show=False,
        )

        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 0)
        fig.clf()

    def test_plot_player_elo_timeline_interactive_smoke(self) -> None:
        df = self._sample_player_df()
        df["match_date"] = pd.to_datetime(df["match_date"], utc=True)
        player_df = select_player_timeseries(df, player_id=100)

        out_dir = ROOT / "outputs" / "test_plots"
        out_path = out_dir / "attacker_a_timeline.html"
        _fig = plot_player_elo_timeline_interactive(
            player_df=player_df,
            output_path=out_path,
            elo_col="player_elo_post",
            include_age=True,
            include_market_value=True,
            use_subplots=True,
            show=False,
        )
        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 0)
        trace_names = [trace.name for trace in _fig.data]
        self.assertIn("Player Ranking (Raw)", trace_names)
        self.assertTrue(any(name and "Smoothed" in name for name in trace_names))
        self.assertIn("Performance Residual", trace_names)
        self.assertEqual(trace_names[-1], "Age")

    def test_interactive_supports_baseline_and_residual_spline(self) -> None:
        df = self._sample_player_df_long()
        df["match_date"] = pd.to_datetime(df["match_date"], utc=True)
        player_df = select_player_timeseries(df, player_id=100)
        baseline_df = player_df.copy()
        baseline_df["player_elo_post"] = baseline_df["player_elo_post"] - 18.0

        fig = plot_player_elo_timeline_interactive(
            player_df=player_df,
            output_path=None,
            elo_col="player_elo_post",
            smooth_window=3,
            residual_col="performance_residual",
            include_residual=True,
            include_age=False,
            include_market_value=False,
            baseline_player_df=baseline_df,
            baseline_elo_col="player_elo_post",
            residual_spline=True,
            use_subplots=True,
            show=False,
        )

        trace_names = [trace.name for trace in fig.data]
        self.assertIn("Performance Residual (Spline)", trace_names)
        self.assertIn("Baseline Ranking (Smoothed, window=3)", trace_names)


if __name__ == "__main__":
    unittest.main()
