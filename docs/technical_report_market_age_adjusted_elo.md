# Technical Report: Attacker-Level ELO with Age and Market-Value Adjusted Expectation

**Project:** `player_ELO`  
**Date:** March 4, 2026  
**Scope:** English Premier League attackers (`position_group = ATT`), seasons 2024/2025 and 2025/2026.

## 1. Objective
The objective is to estimate attacker over/under-performance relative to an expectation that can depend on:
1. pre-match player ELO,
2. age relative to position peak,
3. market-value percentile (normalized within comparable cohorts).

The model is evaluated as a probabilistic forecaster of player scoring events (`S_{i,t} \in \{0,1\}`), where `1` means the attacker scored at least one goal in the match.

## 2. Mathematical Specification

### 2.1 Baseline player expectation
For player \(i\) in match \(t\):

\[
E^{\text{base}}_{i,t} = \sigma\left(\frac{R_{i,t} + H_{i,t} - R^{opp}_t}{s}\right), \quad
\sigma(x)=\frac{1}{1+10^{-x}}
\]

where:
- \(R_{i,t}\): pre-match player ELO,
- \(R^{opp}_t\): opponent pre-match team ELO,
- \(H_{i,t}\): home term (`+home_advantage` for home, `-home_advantage` for away in this run),
- \(s\): ELO scale (`400`).

### 2.2 Market-age adjusted expectation

\[
R^{eff}_{i,t}=R_{i,t}+\beta_{mv} z^{mv}_{i,t}-\beta_{age}\,(age_{i,t}-peak_{pos(i)})^2
\]

\[
E_{i,t}=\sigma\left(\frac{R^{eff}_{i,t}+H_{i,t}-R^{opp}_t}{s}\right)
\]

For this attacker-only study, \(peak_{ATT}=26\) years.

### 2.3 Market-value normalization

\[
\log mv_{i,t}=\log(1+mv_{i,t})
\]

\[
z^{mv}_{i,t}=\frac{\log mv_{i,t}-\mu_g}{\sigma_g}
\]

with groups \(g\) defined by `(season, league, position_group)` when sample size is sufficient, with fallback grouping and winsorization to `[-3,3]`.

### 2.4 Update rule

\[
R_{i,t+1}=R_{i,t}+K_{i,t}(S_{i,t}-E_{i,t}), \quad
K_{i,t}=K\cdot\min(1,\tfrac{minutes_{i,t}}{90})
\]

Residual for diagnostics:

\[
\text{residual}_{i,t}=S_{i,t}-E_{i,t}
\]

## 3. Data and Experimental Design

### 3.1 Sample
- Player-match rows used: **5,747** (all `Full Time`, all attacker rows after minute filter).
- Unique players: **229**.
- Unique matches: **664**.

Missingness after feature pipeline:
- Missing age: **0.21%**
- Missing market value: **1.90%**
- Missing position group: **0.00%**

### 3.2 Time split (strict chronological)
- Train: 3,979 rows, 463 matches, 2024-08-16 to 2025-10-25 14:00 UTC
- Validation: 914 rows, 102 matches, 2025-10-25 16:30 to 2025-12-30 20:15 UTC
- Test: 854 rows, 99 matches, 2026-01-01 to 2026-03-03 20:15 UTC

### 3.3 Anti-leakage controls
- Market value aligned to each match via per-player historical carry-forward only (no future values).
- Market-value normalizer fit on train split only and applied to validation/test.
- Pre-match values used for expectations; updates applied after observed outcome.

## 4. Evaluation Framework

### 4.1 Proper scoring rules
For outcome \(y\in\{0,1\}\), predicted probability \(p\):

\[
\text{LogLoss}=-\frac{1}{N}\sum_i\left[y_i\log p_i +(1-y_i)\log(1-p_i)\right]
\]

\[
\text{Brier}=\frac{1}{N}\sum_i(y_i-p_i)^2
\]

Lower is better for both.

### 4.2 Model tests run
1. **Ablation test:** baseline vs market-only vs age-only vs market+age.
2. **Grid search test:** \(\beta_{mv}\in\{0,8,16,24\}\), \(\beta_{age}\in\{0,0.6,1.2,1.8\}\), optimized on validation log loss.
3. **Paired significance test:** per-row loss deltas between baseline and adjusted models.
4. **Bootstrap robustness test:** 4,000 bootstrap resamples for log-loss and Brier deltas.
5. **Calibration tests:** event-rate comparison, ECE/MCE, linear calibration slope/intercept.
6. **Residual bias tests:** mean residual by age bucket, market-value bucket, and opponent-strength bucket.

## 5. Results

### 5.1 Ablation at default parameters (`beta_mv=18`, `beta_age=1.2`)

| Split | Baseline LogLoss | Market-only | Age-only | Market+Age |
|---|---:|---:|---:|---:|
| Train | 0.631042 | 0.626453 | 0.609786 | **0.606294** |
| Validation | 0.574801 | 0.575690 | **0.559529** | 0.560694 |
| Test | 0.567729 | 0.565312 | 0.553724 | **0.551814** |

Interpretation:
- Age term contributes most of the gain.
- Market-value term adds small extra test improvement at default settings but hurts validation slightly.

### 5.2 Grid-search surface (optimized on validation log loss)
Selected parameters: **`beta_mv=0.0`, `beta_age=1.8`**.

Validation log-loss surface:

| beta_age \ beta_mv | 0 | 8 | 16 | 24 |
|---|---:|---:|---:|---:|
| 0.0 | 0.574801 | 0.575089 | 0.575549 | 0.576179 |
| 0.6 | 0.566821 | 0.567172 | 0.567693 | 0.568383 |
| 1.2 | 0.559529 | 0.559942 | 0.560522 | 0.561271 |
| 1.8 | **0.552907** | 0.553379 | 0.554018 | 0.554823 |

Test log-loss surface (same grid):

| beta_age \ beta_mv | 0 | 8 | 16 | 24 |
|---|---:|---:|---:|---:|
| 0.0 | 0.567729 | 0.566555 | 0.565541 | 0.564684 |
| 0.6 | 0.560386 | 0.559330 | 0.558430 | 0.557685 |
| 1.2 | 0.553724 | 0.552779 | 0.551988 | 0.551348 |
| 1.8 | 0.547716 | 0.546877 | 0.546187 | **0.545645** |

Interpretation:
- Increasing \(\beta_{age}\) improves both validation and test across the tested range.
- Higher \(\beta_{mv}\) helps test modestly but worsens validation under this objective.
- Under strict validation selection, the final model is effectively age-only.

### 5.3 Final selected model performance (`beta_mv=0`, `beta_age=1.8`)

| Split | Baseline LogLoss | Final LogLoss | Delta | Relative Improvement |
|---|---:|---:|---:|---:|
| Validation | 0.574801 | 0.552907 | 0.021894 | 3.81% |
| Test | 0.567729 | 0.547716 | 0.020013 | 3.53% |
| All | 0.612689 | 0.585363 | 0.027326 | 4.46% |

Brier improvements are similar magnitude (test: 0.190759 -> 0.181355, ~4.93% relative).

### 5.4 Statistical significance of improvement
Paired row-level deltas (baseline minus final model):

- **Validation log-loss delta:** +0.021894, 95% CI `[0.01750, 0.02629]`
- **Test log-loss delta:** +0.020013, 95% CI `[0.01537, 0.02465]`

Bootstrap (4,000 resamples):

- Validation log-loss delta 95% bootstrap CI: `[0.01764, 0.02635]`, `P(delta<=0)=0.0`
- Test log-loss delta 95% bootstrap CI: `[0.01546, 0.02448]`, `P(delta<=0)=0.0`

Conclusion: improvements are statistically strong under paired resampling.

### 5.5 Calibration diagnostics (test split)
Observed scoring prevalence is low (`~15.46%`), while raw predicted means are higher:
- Baseline mean predicted probability: `0.3773`
- Final mean predicted probability: `0.3591`

Calibration error (10-bin ECE):
- Baseline ECE: **0.2254**
- Final ECE: **0.2046** (improved)

Linear calibration fit \(y \approx a + b\hat p\):
- Baseline: `a=0.0969`, `b=0.1529`
- Final: `a=0.0879`, `b=0.1856`

Interpretation:
- Final model is better calibrated than baseline but still materially overconfident for this binary target which is expected for this small case where age and MW doesn't change much.

### 5.6 Residual diagnostics (test split)
Mean residuals improve (less negative) across age and value buckets.

By age quartile (baseline -> final):
- youngest: `-0.2910 -> -0.2482`
- Q2: `-0.2036 -> -0.2031`
- Q3: `-0.1628 -> -0.1614`
- oldest: `-0.2332 -> -0.2053`

By market-value z-score quartile (baseline -> final):
- low: `-0.2724 -> -0.2377`
- Q2: `-0.2293 -> -0.2080`
- Q3: `-0.2068 -> -0.1982`
- high: `-0.1746 -> -0.1681`

By opponent-strength quintile, residuals also move upward consistently (less underprediction gap).

## 6. Parameter Interpretation
With final \(\beta_{age}=1.8\) and `peak_age=26` (ATT):

\[
\Delta R_{age}=-1.8(age-26)^2
\]

Example penalties:
- age 24 or 28 (2 years from peak): `-7.2` Elo
- age 21 or 31 (5 years from peak): `-45` Elo
- age 19 or 33 (7 years from peak): `-88.2` Elo

This is a strong symmetric penalty; it captures age curvature but assumes equal penalty for equally distant younger vs older players.

## 7. Key Findings
1. Age adjustment materially improves predictive accuracy over baseline for attacker scoring events.
2. Market-value adjustment did not improve validation log loss in the tested grid, though test gains appeared for larger \(\beta_{mv}\).
3. Final tuned model (by validation objective) is age-only (`beta_mv=0`, `beta_age=1.8`).
4. Improvements are statistically significant under paired and bootstrap tests.
5. Absolute calibration remains imperfect (systematically high probabilities), indicating the next bottleneck is calibration/link-function design rather than feature availability.

## 8. Recommended Next Mathematical Tests
1. **Post-hoc calibration layer** (Platt/Isotonic/Beta calibration) on validation only, then evaluate on test.
2. **Asymmetric age curve** (different coefficients for pre-peak vs post-peak):
   \(\beta_{young}(peak-age)_+^2 + \beta_{old}(age-peak)_+^2\).
3. **Non-binary target sensitivity** (e.g., capped goals-per-90, xG-based signal) to reduce sparsity-induced overconfidence.
4. **Joint tuning of home advantage and ELO scale** with \(\beta\) grid to separate calibration from ranking effects.
5. **Temporal stability test**: rolling-origin backtests and parameter drift checks.

## 9. Reproducibility
Grid search command used:

```bash
python3 scripts/run_backtest_market_age_adjusted_elo.py \
  --config config/market_age_adjusted_elo.yaml \
  --grid-search \
  --beta-mv-grid "0,8,16,24" \
  --beta-age-grid "0,0.6,1.2,1.8" \
  --objective-split validation \
  --objective-metric log_loss \
  --output-dir outputs/market_age_adjusted_elo_grid
```

Primary artifacts:
- `outputs/market_age_adjusted_elo_grid/grid_search_results.csv`
- `outputs/market_age_adjusted_elo_grid/grid_search_summary.json`
- `outputs/market_age_adjusted_elo_grid/best_model_backtest/variant_metrics.csv`
- `outputs/market_age_adjusted_elo_grid/best_model_backtest/player_match_outputs_*.csv`
