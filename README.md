# Player ELO (Market + Age Adjusted)

This repository now includes an attacker-focused, config-driven player ELO pipeline with optional age and market-value adjusted expectation.

# Documentation and model formulation

The full model formulation, training and evaluation strategy can be found in /docs

## Run backtest

```bash
python3 scripts/run_backtest_market_age_adjusted_elo.py \
  --players data/elo_base_players.csv \
  --fixtures data/elo_base_fixtures.csv \
  --config config/market_age_adjusted_elo.yaml \
  --output-dir outputs/market_age_adjusted_elo
```

## Run beta grid search

```bash
python3 scripts/run_backtest_market_age_adjusted_elo.py \
  --players data/elo_base_players.csv \
  --fixtures data/elo_base_fixtures.csv \
  --config config/market_age_adjusted_elo.yaml \
  --grid-search \
  --beta-mv-grid "0,6,12,18,24,30" \
  --beta-age-grid "0,0.6,1.2,1.8,2.4" \
  --objective-split validation \
  --objective-metric log_loss \
  --output-dir outputs/market_age_adjusted_elo_grid
```

## Core implemented functions

- `build_player_match_modeling_table(...)`
- `compute_player_age_at_match(...)`
- `assign_position_group(...)`
- `compute_log_market_value(...)`
- `compute_market_value_zscore(...)`
- `compute_age_peak_distance_sq(...)`
- `compute_effective_player_rating(...)`
- `compute_expected_player_score(...)`
- `compute_player_residual(...)`
- `update_player_elo(...)`
- `run_backtest_market_age_adjusted_elo(...)`

## Notes

- Scope is attacker-only by default (`position_filter: [ATT]`).
- Fixture-level precomputed team ELO is used when present; otherwise fixture team ELO pre-ratings are derived sequentially from results.
- Market-value normalization is fit on training split only to avoid leakage.
