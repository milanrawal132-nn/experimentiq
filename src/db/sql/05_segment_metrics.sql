-- Outcome metrics for every (dimension, level, arm) cell, plus the lift of
-- each email arm over control within that cell.
--
-- This is the view the dashboard slices and the subgroup analysis in Feature 7
-- reads. It is descriptive: the lifts here are point estimates within cells
-- that can get small, and comparing many of them will surface apparent effects
-- by chance alone. Feature 7 applies the multiple-comparison correction and
-- distinguishes the pre-registered subgroups from the exploratory ones; this
-- view deliberately does not pretend to.

CREATE OR REPLACE VIEW v_segment_metrics AS
WITH cells AS (
    SELECT
        d.dimension,
        d.level,
        c.segment                       AS arm,
        count(*)                        AS customers,
        avg(c.visit)                    AS visit_rate,
        avg(c.conversion)               AS conversion_rate,
        avg(c.spend)                    AS mean_spend
    FROM customers AS c
    JOIN v_customer_dimensions AS d USING (customer_id)
    GROUP BY d.dimension, d.level, c.segment
),

control AS (
    SELECT
        dimension,
        level,
        customers                       AS control_customers,
        visit_rate                      AS control_visit_rate,
        conversion_rate                 AS control_conversion_rate,
        mean_spend                      AS control_mean_spend
    FROM cells
    WHERE arm = 'No E-Mail'
)

SELECT
    cells.dimension,
    cells.level,
    cells.arm,
    cells.customers,
    cells.visit_rate,
    cells.conversion_rate,
    cells.mean_spend,

    control.control_customers,
    control.control_visit_rate,
    control.control_conversion_rate,
    control.control_mean_spend,

    cells.visit_rate      - control.control_visit_rate       AS visit_lift,
    cells.conversion_rate - control.control_conversion_rate  AS conversion_lift,
    cells.mean_spend      - control.control_mean_spend       AS spend_lift,

    -- The smaller of the two arms in this cell. A lift computed off a few
    -- hundred customers is mostly noise, and surfacing the number that governs
    -- that is more honest than silently filtering small cells out.
    least(cells.customers, control.control_customers)        AS min_arm_customers
FROM cells
JOIN control USING (dimension, level)
WHERE cells.arm <> 'No E-Mail'
ORDER BY cells.dimension, cells.level, cells.arm;
