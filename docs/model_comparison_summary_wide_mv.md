# Model Comparison Summary (Wide MV Search)

## Data Snapshot
- Players rows: 31,449
- Fixtures rows: 760
- Full-time fixtures: 664
- Attacker rows with minutes >=1: 5,747

## Best Parameters by Run
- `default_config`: beta_mv=18.0, beta_age=1.2
- `grid_wide_mv_best`: beta_mv=0.0, beta_age=3.0
- `bayes_wide_mv_best`: beta_mv=0.0, beta_age=4.0

## Market+Age vs Baseline (Validation/Test/All)
| scenario | split | beta_mv | beta_age | baseline_log_loss | market_age_log_loss | delta_log_loss | rel_log_loss_improvement_pct | baseline_brier | market_age_brier | delta_brier | rel_brier_improvement_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default_config | validation | 18.000000 | 1.200000 | 0.574801 | 0.560694 | 0.014107 | 2.454239 | 0.193807 | 0.187169 | 0.006638 | 3.425301 |
| default_config | test | 18.000000 | 1.200000 | 0.567729 | 0.551814 | 0.015915 | 2.803257 | 0.190759 | 0.183281 | 0.007477 | 3.919744 |
| default_config | all | 18.000000 | 1.200000 | 0.612689 | 0.590946 | 0.021743 | 3.548843 | 0.211627 | 0.201339 | 0.010287 | 4.861020 |
| grid_wide_mv_best | validation | 0.000000 | 3.000000 | 0.574801 | 0.541574 | 0.033226 | 5.780502 | 0.193807 | 0.178652 | 0.015155 | 7.819741 |
| grid_wide_mv_best | test | 0.000000 | 3.000000 | 0.567729 | 0.537544 | 0.030185 | 5.316737 | 0.190759 | 0.176721 | 0.014037 | 7.358723 |
| grid_wide_mv_best | all | 0.000000 | 3.000000 | 0.612689 | 0.571990 | 0.040699 | 6.642702 | 0.211627 | 0.192545 | 0.019081 | 9.016436 |
| bayes_wide_mv_best | validation | 0.000000 | 4.000000 | 0.574801 | 0.533954 | 0.040846 | 7.106192 | 0.193807 | 0.175360 | 0.018447 | 9.518278 |
| bayes_wide_mv_best | test | 0.000000 | 4.000000 | 0.567729 | 0.530801 | 0.036928 | 6.504450 | 0.190759 | 0.173686 | 0.017073 | 8.950004 |
| bayes_wide_mv_best | all | 0.000000 | 4.000000 | 0.612689 | 0.563476 | 0.049213 | 8.032286 | 0.211627 | 0.188526 | 0.023100 | 10.915676 |

## grid_best_positive_mv_validation
| beta_mv | beta_age | validation_log_loss | test_log_loss | all_log_loss | objective_value |
| --- | --- | --- | --- | --- | --- |
| 10.000000 | 3.000000 | 0.542333 | 0.536756 | 0.571079 | 0.542333 |

## grid_best_positive_mv_test
| beta_mv | beta_age | validation_log_loss | test_log_loss | all_log_loss | objective_value |
| --- | --- | --- | --- | --- | --- |
| 40.000000 | 3.000000 | 0.546134 | 0.535718 | 0.570654 | 0.546134 |

## bayes_best_positive_mv_validation
| beta_mv | beta_age | validation_log_loss | test_log_loss | all_log_loss | objective_value | stage | iteration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11.301282 | 3.902489 | 0.535619 | 0.530701 | 0.563519 | 0.535619 | initial | 0 |

## bayes_best_positive_mv_test
| beta_mv | beta_age | validation_log_loss | test_log_loss | all_log_loss | objective_value | stage | iteration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11.301282 | 3.902489 | 0.535619 | 0.530701 | 0.563519 | 0.535619 | initial | 0 |

## Interpretation
- Increasing the MV search range did not change the validation-optimal solution: best models still choose `beta_mv = 0` with larger `beta_age`.
- Positive `beta_mv` settings can slightly improve **test** log loss in several runs, but they remain worse on **validation** under the current objective, so they are not selected.
- CSV attachments include full per-variant losses and explicit baseline deltas for all scenarios and splits.