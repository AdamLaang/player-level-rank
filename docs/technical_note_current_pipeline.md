# Current Player-ELO Pipeline Review (Attacker Scope)

## What exists in this repository
- No executable player-ELO code is present in the repository snapshot.
- Available inputs are:
  - `data/elo_base_players.csv` (player-match rows)
  - `data/elo_base_fixtures.csv` (fixture rows)
- Fixture data in this snapshot does **not** contain explicit `*_elo_*` columns, so the implementation supports two modes:
  1. Use fixture precomputed team ELO columns if present.
  2. Fallback: derive sequential fixture-level team ELO pre-ratings from match outcomes.

## Baseline expectation formula implemented
For each attacker-player match row:

- Baseline effective rating (feature flag off):
  - `R_eff = R_player_pre`
- Expected score:
  - `E = 1 / (1 + 10^(-(R_eff + H - R_opp_pre) / elo_scale))`
- Home term (`H`) is config-driven:
  - `symmetric`: `+home_advantage` for home, `-home_advantage` for away
  - `home_only`: `+home_advantage` for home, `0` for away

## Baseline rating update formula implemented
- `R_player_post = R_player_pre + K_eff * (S - E)`
- `K_eff`:
  - `K_eff = player_k_factor * clip(minutes_played / 90, 0, 1)` when minute scaling is enabled
  - else `K_eff = player_k_factor`

## Observed performance target used (`S`)
- Default attacker target in this implementation:
  - `S = 1.0` if player scored at least one goal in the match
  - `S = 0.0` otherwise
- Alternative config option:
  - `goals_per_90_capped` (capped to `[0,1]`)

## Insertion point for market-value + age adjustment
The adjustment is inserted between baseline `R_player_pre` and expectation `E`:

- `R_eff = R_player_pre + beta_mv * z_mv - beta_age * age_peak_distance_sq`
- Then `E` is computed from `R_eff` instead of raw `R_player_pre`.
- If `use_market_age_adjustment=false`, the path reverts to baseline expectation logic.

## Data-path and leakage controls
- Attacker-only filter (`position_group == ATT`) is applied in modeling-table build.
- Market value alignment uses forward-fill per player over time (no look-ahead).
- Market-value normalizer is fit on training dates only and applied to validation/test.
- Missing market value: `z_mv = 0`.
- Missing age: neutral age term (`age_peak_distance_sq = 0`).
- Missing position: fallback group `UNK` + fallback peak age.
