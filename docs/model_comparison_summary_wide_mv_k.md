# Model Comparison Summary (Wide MV + K Search)

## Data Snapshot
- Players rows: 31,449
- Fixtures rows: 760
- Full-time fixtures: 664
- Attacker rows with minutes >=1: 5,747

## Best Parameters by Run
- `default_config`: beta_mv=18.0, beta_age=1.2, player_k_factor=20.0
- `grid_wide_mv_k_best`: beta_mv=0.0, beta_age=4.0, player_k_factor=40.0
- `bayes_wide_mv_k_best`: beta_mv=0.0, beta_age=4.0, player_k_factor=60.0

## Market+Age vs Baseline (Validation/Test/All)
| scenario | split | beta_mv | beta_age | player_k_factor | baseline_log_loss | market_age_log_loss | delta_log_loss | rel_log_loss_improvement_pct | baseline_brier | market_age_brier | delta_brier | rel_brier_improvement_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default_config | validation | 18.000000 | 1.200000 | 20.000000 | 0.574801 | 0.560694 | 0.014107 | 2.454239 | 0.193807 | 0.187169 | 0.006638 | 3.425301 |
| default_config | test | 18.000000 | 1.200000 | 20.000000 | 0.567729 | 0.551814 | 0.015915 | 2.803257 | 0.190759 | 0.183281 | 0.007477 | 3.919744 |
| default_config | all | 18.000000 | 1.200000 | 20.000000 | 0.612689 | 0.590946 | 0.021743 | 3.548843 | 0.211627 | 0.201339 | 0.010287 | 4.861020 |
| grid_wide_mv_k_best | validation | 0.000000 | 4.000000 | 40.000000 | 0.536568 | 0.503901 | 0.032667 | 6.088111 | 0.176809 | 0.162263 | 0.014546 | 8.227205 |
| grid_wide_mv_k_best | test | 0.000000 | 4.000000 | 40.000000 | 0.525788 | 0.498284 | 0.027504 | 5.231083 | 0.172158 | 0.159590 | 0.012568 | 7.300216 |
| grid_wide_mv_k_best | all | 0.000000 | 4.000000 | 40.000000 | 0.576824 | 0.536502 | 0.040321 | 6.990249 | 0.195329 | 0.176634 | 0.018695 | 9.571162 |
| bayes_wide_mv_k_best | validation | 0.000000 | 4.000000 | 60.000000 | 0.515265 | 0.487185 | 0.028080 | 5.449700 | 0.167588 | 0.155194 | 0.012394 | 7.395613 |
| bayes_wide_mv_k_best | test | 0.000000 | 4.000000 | 60.000000 | 0.505019 | 0.482355 | 0.022664 | 4.487716 | 0.163293 | 0.152998 | 0.010295 | 6.304558 |
| bayes_wide_mv_k_best | all | 0.000000 | 4.000000 | 60.000000 | 0.555885 | 0.520977 | 0.034908 | 6.279649 | 0.186067 | 0.169999 | 0.016067 | 8.635236 |

## grid_best_positive_mv_validation
| beta_mv | beta_age | player_k_factor | validation_log_loss | test_log_loss | all_log_loss | objective_value |
| --- | --- | --- | --- | --- | --- | --- |
| 10.000000 | 4.000000 | 40.000000 | 0.504220 | 0.497150 | 0.535451 | 0.504220 |

## grid_best_positive_mv_test
| beta_mv | beta_age | player_k_factor | validation_log_loss | test_log_loss | all_log_loss | objective_value |
| --- | --- | --- | --- | --- | --- | --- |
| 100.000000 | 4.000000 | 40.000000 | 0.514317 | 0.492622 | 0.538209 | 0.514317 |

## bayes_best_positive_mv_validation
| beta_mv | beta_age | player_k_factor | validation_log_loss | test_log_loss | all_log_loss | objective_value | stage | iteration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.764640 | 3.982152 | 56.430299 | 0.489721 | 0.484363 | 0.523104 | 0.489721 | ei | 14 |

## bayes_best_positive_mv_test
| beta_mv | beta_age | player_k_factor | validation_log_loss | test_log_loss | all_log_loss | objective_value | stage | iteration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 120.000000 | 4.000000 | 60.000000 | 0.497113 | 0.474101 | 0.521362 | 0.497113 | initial | 0 |

## Interpretation
- Jointly tuning `player_k_factor` materially improved model quality and also improved the baseline comparator (because baseline uses the same tuned K in each run).
- Even with wider MV ranges and tuned K, validation-optimal models still selected `beta_mv = 0` while pushing `beta_age` and `K` higher.
- Best positive-MV Bayesian candidate vs overall best (validation objective): dValLogLoss=0.002537, dTestLogLoss=0.002007.
- Full outputs include losses and absolute/relative baseline deltas for every split and variant.