# Monitoring report

Model `lgbm-202610042219`, policy `1.0.0`. New month: simulated acquisition channel that oversamples applicants without income.

## Summary

_Skipped: Claude was unavailable (set ANTHROPIC_API_KEY to enable it)._

## Decisions

| Metric | Baseline | New month |
|---|---|---|
| applicants | 5000 | 5000 |
| approval_rate | 0.634 | 0.485 |
| declined_by_policy | 0.2088 | 0.3994 |
| mean_probability_of_default | 0.0683 | 0.0682 |

## Feature drift (PSI; moderate >= 0.1, major >= 0.25)

| Feature | PSI | Status |
|---|---|---|
| monthly_income | 0.1789 | moderate |
| debt_ratio | 0.1472 | moderate |
| dependents | 0.0267 | stable |
| revolving_utilization | 0.0082 | stable |
| age | 0.0060 | stable |
| open_credit_lines | 0.0053 | stable |
| past_due_30_59 | 0.0011 | stable |
| real_estate_loans | 0.0008 | stable |
| past_due_60_89 | 0.0004 | stable |
| past_due_90 | 0.0000 | stable |
