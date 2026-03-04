from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_age_elo.backtest import (
    run_bayesian_optimization_market_age_adjusted_elo,
    run_backtest_market_age_adjusted_elo,
    run_grid_search_market_age_adjusted_elo,
)
from market_age_elo.config import MarketAgeAdjustedEloConfig
from market_age_elo.features import (
    assign_position_group,
    build_player_match_modeling_table,
    compute_log_market_value,
    compute_player_age_at_match,
)
from market_age_elo.model import (
    compute_age_peak_distance_sq,
    compute_effective_player_rating,
    compute_expected_player_score,
)


class TestMarketAgeAdjustedElo(unittest.TestCase):
    @staticmethod
    def _sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
        fixtures = pd.DataFrame(
            [
                {
                    "fixture_id": 1,
                    "home_name": "A",
                    "away_name": "B",
                    "state": "Full Time",
                    "starting_at": "2024-08-01 12:00:00+00",
                    "league": "Premier League",
                    "season": "2024/2025",
                    "country": "England",
                    "home_goals": 2,
                    "away_goals": 1,
                },
                {
                    "fixture_id": 2,
                    "home_name": "B",
                    "away_name": "A",
                    "state": "Full Time",
                    "starting_at": "2024-08-08 12:00:00+00",
                    "league": "Premier League",
                    "season": "2024/2025",
                    "country": "England",
                    "home_goals": 0,
                    "away_goals": 1,
                },
                {
                    "fixture_id": 3,
                    "home_name": "A",
                    "away_name": "B",
                    "state": "Full Time",
                    "starting_at": "2024-08-15 12:00:00+00",
                    "league": "Premier League",
                    "season": "2024/2025",
                    "country": "England",
                    "home_goals": 1,
                    "away_goals": 0,
                },
            ]
        )

        players = pd.DataFrame(
            [
                {
                    "fixture_id": 1,
                    "sportmonks_player_id": 100,
                    "team_name": "A",
                    "player_name": "Attacker A",
                    "position": "Attacker",
                    "minutes_played": 90,
                    "date_of_birth": "2000-01-01",
                    "age": 24,
                    "marketvalue": 20000000,
                    "goals": 1,
                    "starting_at": "2024-08-01 12:00:00+00",
                    "league": "Premier League",
                    "season": "2024/2025",
                },
                {
                    "fixture_id": 1,
                    "sportmonks_player_id": 200,
                    "team_name": "B",
                    "player_name": "Attacker B",
                    "position": "Attacker",
                    "minutes_played": 90,
                    "date_of_birth": "1998-01-01",
                    "age": 26,
                    "marketvalue": 15000000,
                    "goals": 0,
                    "starting_at": "2024-08-01 12:00:00+00",
                    "league": "Premier League",
                    "season": "2024/2025",
                },
                {
                    "fixture_id": 2,
                    "sportmonks_player_id": 100,
                    "team_name": "A",
                    "player_name": "Attacker A",
                    "position": "Attacker",
                    "minutes_played": 90,
                    "date_of_birth": "2000-01-01",
                    "age": 24,
                    "marketvalue": 22000000,
                    "goals": 1,
                    "starting_at": "2024-08-08 12:00:00+00",
                    "league": "Premier League",
                    "season": "2024/2025",
                },
                {
                    "fixture_id": 2,
                    "sportmonks_player_id": 200,
                    "team_name": "B",
                    "player_name": "Attacker B",
                    "position": "Attacker",
                    "minutes_played": 90,
                    "date_of_birth": "1998-01-01",
                    "age": 26,
                    "marketvalue": 15000000,
                    "goals": 0,
                    "starting_at": "2024-08-08 12:00:00+00",
                    "league": "Premier League",
                    "season": "2024/2025",
                },
                {
                    "fixture_id": 3,
                    "sportmonks_player_id": 100,
                    "team_name": "A",
                    "player_name": "Attacker A",
                    "position": "Attacker",
                    "minutes_played": 90,
                    "date_of_birth": "2000-01-01",
                    "age": 24,
                    "marketvalue": 22000000,
                    "goals": 1,
                    "starting_at": "2024-08-15 12:00:00+00",
                    "league": "Premier League",
                    "season": "2024/2025",
                },
                {
                    "fixture_id": 3,
                    "sportmonks_player_id": 200,
                    "team_name": "B",
                    "player_name": "Attacker B",
                    "position": "Attacker",
                    "minutes_played": 90,
                    "date_of_birth": "1998-01-01",
                    "age": 26,
                    "marketvalue": 14000000,
                    "goals": 0,
                    "starting_at": "2024-08-15 12:00:00+00",
                    "league": "Premier League",
                    "season": "2024/2025",
                },
            ]
        )
        return fixtures, players

    def test_assign_position_group(self) -> None:
        self.assertEqual(assign_position_group("Attacker"), "ATT")
        self.assertEqual(assign_position_group("Defender"), "DEF")
        self.assertEqual(assign_position_group("Goalkeeper"), "GK")
        self.assertEqual(assign_position_group("Midfielder"), "MID")
        self.assertEqual(assign_position_group(None), "UNK")

    def test_compute_player_age_at_match(self) -> None:
        age = compute_player_age_at_match(
            match_date="2025-01-01 00:00:00+00",
            date_of_birth="2000-01-01",
            fallback_age_years=99,
        )
        self.assertAlmostEqual(age, 25.0, places=1)

        fallback = compute_player_age_at_match(
            match_date="2025-01-01",
            date_of_birth=None,
            fallback_age_years=22,
        )
        self.assertEqual(fallback, 22.0)

    def test_compute_log_market_value(self) -> None:
        self.assertAlmostEqual(compute_log_market_value(0), 0.0)
        self.assertAlmostEqual(compute_log_market_value(999999), np.log1p(999999))
        self.assertTrue(np.isnan(compute_log_market_value(None)))

    def test_effective_rating_formula(self) -> None:
        eff = compute_effective_player_rating(
            player_elo_pre=1500,
            z_mv=1.5,
            age_peak_distance_sq=4,
            beta_mv=18.0,
            beta_age=1.2,
            use_market_age_adjustment=True,
        )
        self.assertAlmostEqual(eff, 1500 + (18.0 * 1.5) - (1.2 * 4))

    def test_flag_off_matches_baseline_expectation(self) -> None:
        player_elo = 1512.0
        opp_elo = 1490.0

        eff_off = compute_effective_player_rating(
            player_elo_pre=player_elo,
            z_mv=2.5,
            age_peak_distance_sq=16,
            beta_mv=18.0,
            beta_age=1.2,
            use_market_age_adjustment=False,
        )
        self.assertEqual(eff_off, player_elo)

        expected_off = compute_expected_player_score(
            effective_player_rating=eff_off,
            opponent_rating_pre=opp_elo,
            is_home=True,
            home_advantage=40,
            elo_scale=400,
            home_advantage_mode="symmetric",
        )
        expected_baseline = compute_expected_player_score(
            effective_player_rating=player_elo,
            opponent_rating_pre=opp_elo,
            is_home=True,
            home_advantage=40,
            elo_scale=400,
            home_advantage_mode="symmetric",
        )
        self.assertAlmostEqual(expected_off, expected_baseline, places=12)

    def test_build_table_and_backtest(self) -> None:
        fixtures, players = self._sample_inputs()

        cfg = MarketAgeAdjustedEloConfig(
            min_group_size_for_mv_norm=1,
            train_split_frac=0.5,
            validation_split_frac=0.25,
        )

        table, quality, _ = build_player_match_modeling_table(players, fixtures, config=cfg)
        self.assertGreater(len(table), 0)
        self.assertIn("z_mv", table.columns)
        self.assertIn("opponent_rating_pre", table.columns)
        self.assertAlmostEqual(quality["missing_market_value_pct"], 0.0)

        results = run_backtest_market_age_adjusted_elo(players_df=players, fixtures_df=fixtures, config=cfg)
        self.assertIn("variant_metrics", results)
        self.assertEqual(set(results["variant_outputs"].keys()), {"baseline", "market_only", "age_only", "market_age"})
        self.assertTrue({"expected_score", "performance_residual", "effective_player_rating"}.issubset(results["variant_outputs"]["market_age"].columns))

    def test_grid_search_beta_mv_beta_age(self) -> None:
        fixtures, players = self._sample_inputs()
        cfg = MarketAgeAdjustedEloConfig(
            min_group_size_for_mv_norm=1,
            train_split_frac=0.5,
            validation_split_frac=0.25,
        )
        search = run_grid_search_market_age_adjusted_elo(
            players_df=players,
            fixtures_df=fixtures,
            config=cfg,
            beta_mv_grid=[0.0, 8.0],
            beta_age_grid=[0.0, 1.2],
            player_k_grid=[10.0, 20.0],
            objective_split="test",
            objective_metric="log_loss",
            rerun_best_backtest=True,
        )

        self.assertIn("grid_results", search)
        self.assertEqual(len(search["grid_results"]), 8)
        self.assertIn("best_params", search)
        self.assertIsNotNone(search["best_params"])
        self.assertTrue(
            {"beta_mv", "beta_age", "player_k_factor", "objective_value"}.issubset(
                search["grid_results"].columns
            )
        )
        self.assertIn("player_k_factor", search["best_params"])
        self.assertIn("best_model_backtest", search)
        self.assertIn("variant_metrics", search["best_model_backtest"])

    def test_bayesian_optimization_beta_mv_beta_age(self) -> None:
        fixtures, players = self._sample_inputs()
        cfg = MarketAgeAdjustedEloConfig(
            min_group_size_for_mv_norm=1,
            train_split_frac=0.5,
            validation_split_frac=0.25,
        )
        search = run_bayesian_optimization_market_age_adjusted_elo(
            players_df=players,
            fixtures_df=fixtures,
            config=cfg,
            beta_mv_bounds=(0.0, 12.0),
            beta_age_bounds=(0.0, 2.0),
            player_k_bounds=(5.0, 30.0),
            n_initial_points=5,
            n_iterations=4,
            candidate_pool_size=200,
            objective_split="validation",
            objective_metric="log_loss",
            random_seed=7,
            rerun_best_backtest=True,
        )

        self.assertIn("search_results", search)
        # unique evaluations should be <= initial + iterations
        self.assertLessEqual(len(search["search_results"]), 9)
        self.assertGreaterEqual(len(search["search_results"]), 5)
        self.assertIn("best_params", search)
        self.assertIsNotNone(search["best_params"])
        self.assertIn("player_k_factor", search["best_params"])
        self.assertTrue(
            {"beta_mv", "beta_age", "player_k_factor", "objective_value", "stage", "iteration"}.issubset(
                search["search_results"].columns
            )
        )
        self.assertIn("best_model_backtest", search)
        self.assertIn("variant_metrics", search["best_model_backtest"])


if __name__ == "__main__":
    unittest.main()
