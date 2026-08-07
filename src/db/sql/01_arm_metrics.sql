-- Headline outcome metrics for each treatment arm.
--
-- These are descriptive point estimates only: no standard errors, no
-- significance. Inference is Feature 4's job, and deliberately does not live in
-- SQL, where it is easy to compute a difference and hard to compute the
-- uncertainty around it.
--
-- The `segment` column is a DuckDB ENUM whose order was set when the table was
-- built, so ORDER BY places the control arm first without a CASE expression.

CREATE OR REPLACE VIEW v_arm_metrics AS
SELECT
    segment                                        AS arm,
    count(*)                                       AS customers,

    -- Funnel counts
    sum(visit)                                     AS visitors,
    sum(conversion)                                AS converters,

    -- Rates across all customers assigned to the arm. Denominator is always
    -- everyone assigned, never everyone who visited, so these stay unbiased
    -- estimates of the causal effect of assignment.
    avg(visit)                                     AS visit_rate,
    avg(conversion)                                AS conversion_rate,
    avg(spend)                                     AS mean_spend,

    -- Revenue and its concentration
    sum(spend)                                     AS total_spend,
    sum(spend) / nullif(sum(conversion), 0)        AS mean_spend_per_converter,

    -- Spend is ~99% zeros; its standard deviation dwarfs its mean. Carrying
    -- that here makes the variance problem visible in the dashboard rather
    -- than a surprise in the power analysis.
    stddev_samp(spend)                             AS spend_stddev
FROM customers
GROUP BY segment
ORDER BY segment;
