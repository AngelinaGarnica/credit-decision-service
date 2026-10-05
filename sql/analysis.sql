-- Default rate by segment over the cleaned `applicants` table.
-- `defaulted` is 0/1, so AVG(defaulted) is the default rate.
-- Portable SQL: runs on PostgreSQL (target) and SQLite.

-- 1. Default rate by age band.
SELECT
    CASE
        WHEN age < 30 THEN '18-29'
        WHEN age < 45 THEN '30-44'
        WHEN age < 60 THEN '45-59'
        ELSE '60+'
    END AS age_band,
    COUNT(*) AS applicants,
    ROUND(100.0 * AVG(defaulted), 2) AS default_rate_pct
FROM applicants
GROUP BY age_band
ORDER BY MIN(age);

-- 2. Default rate by debt ratio band (monthly debt payments / monthly income).
-- Only rows with income: when income is missing, the source stores monthly debt in dollars.
SELECT
    CASE
        WHEN debt_ratio < 0.3 THEN '< 0.3'
        WHEN debt_ratio < 0.5 THEN '0.3-0.5'
        WHEN debt_ratio < 1 THEN '0.5-1'
        ELSE '>= 1'
    END AS debt_ratio_band,
    COUNT(*) AS applicants,
    ROUND(100.0 * AVG(defaulted), 2) AS default_rate_pct
FROM applicants
WHERE monthly_income IS NOT NULL
GROUP BY debt_ratio_band
ORDER BY MIN(debt_ratio);

-- 3. Default rate by monthly income band, with missing income as its own segment.
SELECT
    CASE
        WHEN monthly_income IS NULL THEN 'missing'
        WHEN monthly_income = 0 THEN '0'
        WHEN monthly_income < 2500 THEN '< 2,500'
        WHEN monthly_income < 5000 THEN '2,500-5,000'
        WHEN monthly_income < 10000 THEN '5,000-10,000'
        ELSE '>= 10,000'
    END AS income_band,
    COUNT(*) AS applicants,
    ROUND(100.0 * AVG(defaulted), 2) AS default_rate_pct
FROM applicants
GROUP BY income_band
ORDER BY MIN(monthly_income) NULLS FIRST;
