# Monitoring report

Model `lgbm-202610042219`, policy `1.0.0`. Samples of 5,000 applicants. New month: simulated acquisition channel that oversamples applicants without income.

## Summary

**Headline:** Approvals fell from 63.4% to 48.5% this month, driven by more applications failing the income rule, while predicted risk stayed flat.

**Population change:** Two applicant features shifted moderately, measured by PSI (how much a feature's distribution moved compared with the baseline). These were monthly_income (0.1789) and debt_ratio (0.1472). All other features, including age, remained stable.

**Business impact:** Policy declines rose from 20.88% to 39.94% of applicants. This rise comes entirely from the income_required rule, which declines applications with zero or missing monthly income. Its failure rate matched the policy decline rate in both periods (0.2088 at baseline, 0.3994 this month). No applicants failed minimum_age in either period. Among applicants the model scored, mean probability of default was essentially unchanged (0.0683 vs 0.0682). The approval drop therefore reflects fewer applicants clearing policy, not riskier applicants.

**Recommended next step:** Check whether the jump in zero or missing monthly income reflects real applicants or a data capture problem, such as a form, channel or integration change. Prioritise this before any policy or model adjustment, because a data fault would mean creditworthy applicants are being declined unnecessarily.

## Decisions

| Metric | Baseline | New month |
|---|---|---|
| approval_rate | 0.634 | 0.485 |
| declined_by_policy | 0.2088 | 0.3994 |
| mean_probability_of_default | 0.0683 | 0.0682 |

## Policy rules (share of applicants failing each; one can fail several)

| Rule | Baseline | New month |
|---|---|---|
| minimum_age | 0.0 | 0.0 |
| income_required | 0.2088 | 0.3994 |

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
