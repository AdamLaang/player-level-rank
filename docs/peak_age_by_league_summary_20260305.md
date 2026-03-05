# Age-Penalty Methods and Peak-Age Search by League (March 5, 2026)

## Scope
- Data snapshot used in this run contains one league: `Premier League`.
- Objective: find the best `peak_age` for attackers (`position_group=ATT`) per league.
- Selection criterion: lowest validation `log_loss` for `variant=market_age`.

## Age-penalty methods now supported

For player age `a` and position peak `p`:

1. Quadratic:
   - `age_penalty = beta_age * (a - p)^2`
2. Absolute:
   - `age_penalty = beta_age * |a - p|`
3. Asymmetric quadratic:
   - `age_penalty = beta_age_young * (p - a)^2` when `a < p`
   - `age_penalty = beta_age_old * (a - p)^2` when `a >= p`

Effective rating in all cases:

`R_eff = R_player_pre + beta_mv*z_mv + beta_mv_team*z_mv_team_context - age_penalty`

## Commands used

Asymmetric peak-age search:

```bash
python3 scripts/optimize_peak_age_by_league.py \
  --players data/elo_base_players.csv \
  --fixtures data/elo_base_fixtures.csv \
  --config config/market_age_adjusted_elo_asymmetric_age.yaml \
  --peak-age-grid 22,23,24,25,26,27,28,29,30 \
  --position-group ATT \
  --variant market_age \
  --objective-split validation \
  --objective-metric log_loss \
  --output-dir outputs/peak_age_by_league_asymmetric
```

Quadratic peak-age search:

```bash
python3 scripts/optimize_peak_age_by_league.py \
  --players data/elo_base_players.csv \
  --fixtures data/elo_base_fixtures.csv \
  --config config/market_age_adjusted_elo.yaml \
  --peak-age-grid 22,23,24,25,26,27,28,29,30 \
  --position-group ATT \
  --variant market_age \
  --objective-split validation \
  --objective-metric log_loss \
  --output-dir outputs/peak_age_by_league_quadratic
```

## Best peak age per model (ATT)

| Model | League | Best peak age | Validation log loss | Test log loss |
| --- | --- | ---: | ---: | ---: |
| Asymmetric quadratic | Premier League | 30.0 | 0.504229 | 0.472449 |
| Quadratic | Premier League | 30.0 | 0.529638 | 0.496286 |

## Full grid results (Premier League, ATT)

| peak_age | Asymmetric val log loss | Asymmetric test log loss | Quadratic val log loss | Quadratic test log loss |
| ---: | ---: | ---: | ---: | ---: |
| 22 | 0.544630 | 0.510086 | 0.545471 | 0.511021 |
| 23 | 0.544276 | 0.509883 | 0.546955 | 0.513029 |
| 24 | 0.542216 | 0.508032 | 0.547421 | 0.513900 |
| 25 | 0.538364 | 0.504510 | 0.546845 | 0.513609 |
| 26 | 0.532823 | 0.499509 | 0.545233 | 0.512167 |
| 27 | 0.526048 | 0.493324 | 0.542623 | 0.509619 |
| 28 | 0.518585 | 0.486337 | 0.539085 | 0.506044 |
| 29 | 0.511013 | 0.479137 | 0.534715 | 0.501553 |
| 30 | 0.504229 | 0.472449 | 0.529638 | 0.496286 |

## Notes
- Both models improved as `peak_age` increased over this tested range (`22..30`).
- Because the optimum is at the upper grid boundary (`30`), rerunning with a wider grid (for example `30..34`) is recommended before fixing a final peak age.

## Artifacts
- `outputs/peak_age_by_league_asymmetric/peak_age_search_results.csv`
- `outputs/peak_age_by_league_asymmetric/peak_age_best_by_league.csv`
- `outputs/peak_age_by_league_asymmetric/peak_age_search_summary.json`
- `outputs/peak_age_by_league_quadratic/peak_age_search_results.csv`
- `outputs/peak_age_by_league_quadratic/peak_age_best_by_league.csv`
- `outputs/peak_age_by_league_quadratic/peak_age_search_summary.json`
