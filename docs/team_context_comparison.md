# Team-Context Comparison (Wide MV + Team MV + K Searches)


## Best Parameters

| run | label | beta_mv | beta_mv_team | beta_age | player_k_factor |
| --- | --- | --- | --- | --- | --- |
| grid_off | Grid (No Team Context, Matched Ranges) | 0.000000 | 0.000000 | 4.000000 | 60.000000 |
| grid_team | Grid (Team Context) | 0.000000 | 0.000000 | 4.000000 | 60.000000 |
| bayes_off | Bayes (No Team Context) | 0.000000 | 0.000000 | 4.000000 | 60.000000 |
| bayes_team | Bayes (Team Context) | 0.000000 | 0.000000 | 4.000000 | 60.000000 |


## Market+Age vs Baseline (each run)

| label | split | beta_mv | beta_mv_team | beta_age | player_k_factor | baseline_log_loss | market_age_log_loss | delta_log_loss | rel_log_loss_impr_pct | baseline_brier | market_age_brier | delta_brier | rel_brier_impr_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Grid (No Team Context, Matched Ranges) | validation | 0.000000 | 0.000000 | 4.000000 | 60.000000 | 0.515265 | 0.487185 | 0.028080 | 5.449700 | 0.167588 | 0.155194 | 0.012394 | 7.395613 |
| Grid (No Team Context, Matched Ranges) | test | 0.000000 | 0.000000 | 4.000000 | 60.000000 | 0.505019 | 0.482355 | 0.022664 | 4.487716 | 0.163293 | 0.152998 | 0.010295 | 6.304558 |
| Grid (No Team Context, Matched Ranges) | all | 0.000000 | 0.000000 | 4.000000 | 60.000000 | 0.555885 | 0.520977 | 0.034908 | 6.279649 | 0.186067 | 0.169999 | 0.016067 | 8.635236 |
| Grid (Team Context) | validation | 0.000000 | 0.000000 | 4.000000 | 60.000000 | 0.515265 | 0.487185 | 0.028080 | 5.449700 | 0.167588 | 0.155194 | 0.012394 | 7.395613 |
| Grid (Team Context) | test | 0.000000 | 0.000000 | 4.000000 | 60.000000 | 0.505019 | 0.482355 | 0.022664 | 4.487716 | 0.163293 | 0.152998 | 0.010295 | 6.304558 |
| Grid (Team Context) | all | 0.000000 | 0.000000 | 4.000000 | 60.000000 | 0.555885 | 0.520977 | 0.034908 | 6.279649 | 0.186067 | 0.169999 | 0.016067 | 8.635236 |
| Bayes (No Team Context) | validation | 0.000000 | 0.000000 | 4.000000 | 60.000000 | 0.515265 | 0.487185 | 0.028080 | 5.449700 | 0.167588 | 0.155194 | 0.012394 | 7.395613 |
| Bayes (No Team Context) | test | 0.000000 | 0.000000 | 4.000000 | 60.000000 | 0.505019 | 0.482355 | 0.022664 | 4.487716 | 0.163293 | 0.152998 | 0.010295 | 6.304558 |
| Bayes (No Team Context) | all | 0.000000 | 0.000000 | 4.000000 | 60.000000 | 0.555885 | 0.520977 | 0.034908 | 6.279649 | 0.186067 | 0.169999 | 0.016067 | 8.635236 |
| Bayes (Team Context) | validation | 0.000000 | 0.000000 | 4.000000 | 60.000000 | 0.515265 | 0.487185 | 0.028080 | 5.449700 | 0.167588 | 0.155194 | 0.012394 | 7.395613 |
| Bayes (Team Context) | test | 0.000000 | 0.000000 | 4.000000 | 60.000000 | 0.505019 | 0.482355 | 0.022664 | 4.487716 | 0.163293 | 0.152998 | 0.010295 | 6.304558 |
| Bayes (Team Context) | all | 0.000000 | 0.000000 | 4.000000 | 60.000000 | 0.555885 | 0.520977 | 0.034908 | 6.279649 | 0.186067 | 0.169999 | 0.016067 | 8.635236 |


## Team Context ON vs OFF Delta (same strategy)

| strategy | split | off_market_age_log_loss | team_market_age_log_loss | team_minus_off_market_age_log_loss | off_baseline_log_loss | team_baseline_log_loss | team_minus_off_baseline_log_loss | off_best_beta_mv_team | team_best_beta_mv_team | off_best_k | team_best_k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grid | validation | 0.487185 | 0.487185 | 0.000000 | 0.515265 | 0.515265 | 0.000000 | 0.000000 | 0.000000 | 60.000000 | 60.000000 |
| grid | test | 0.482355 | 0.482355 | 0.000000 | 0.505019 | 0.505019 | 0.000000 | 0.000000 | 0.000000 | 60.000000 | 60.000000 |
| grid | all | 0.520977 | 0.520977 | 0.000000 | 0.555885 | 0.555885 | 0.000000 | 0.000000 | 0.000000 | 60.000000 | 60.000000 |
| bayes | validation | 0.487185 | 0.487185 | 0.000000 | 0.515265 | 0.515265 | 0.000000 | 0.000000 | 0.000000 | 60.000000 | 60.000000 |
| bayes | test | 0.482355 | 0.482355 | 0.000000 | 0.505019 | 0.505019 | 0.000000 | 0.000000 | 0.000000 | 60.000000 | 60.000000 |
| bayes | all | 0.520977 | 0.520977 | 0.000000 | 0.555885 | 0.555885 | 0.000000 | 0.000000 | 0.000000 | 60.000000 | 60.000000 |


## Best Positive Team-MV Candidate per Run

| run | best_positive_beta_mv_team | best_positive_beta_mv | best_positive_beta_age | best_positive_player_k_factor | best_positive_validation_log_loss | best_positive_test_log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| grid_off |  |  |  |  |  |  |
| grid_team | 5.000000 | 0.000000 | 4.000000 | 60.000000 | 0.487905 | 0.481919 |
| bayes_off |  |  |  |  |  |  |
| bayes_team | 1.252901 | 30.018225 | 3.720597 | 57.915452 | 0.491093 | 0.481259 |
