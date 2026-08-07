-- Lift of each treatment arm over the shared control, in long format.
--
-- The three outcomes are unpivoted into rows so that one view serves all of
-- them; the alternative is three near-identical columns per outcome, which the
-- dashboard would then have to unpivot anyway.
--
-- Again: point estimates with no uncertainty attached. A lift here being
-- non-zero means nothing on its own.

CREATE OR REPLACE VIEW v_arm_lift AS
WITH outcomes AS (
    SELECT arm, visit_rate, conversion_rate, mean_spend
    FROM v_arm_metrics
),

-- One row per (arm, outcome) rather than one row per arm.
long AS (
    UNPIVOT outcomes
    ON visit_rate, conversion_rate, mean_spend
    INTO NAME outcome VALUE arm_value
),

control AS (
    SELECT outcome, arm_value AS control_value
    FROM long
    WHERE arm = 'No E-Mail'
)

SELECT
    long.arm,
    long.outcome,
    long.arm_value                                          AS treatment_value,
    control.control_value,
    long.arm_value - control.control_value                  AS absolute_lift,

    -- nullif guards against a zero-rate control arm. It cannot happen in this
    -- dataset, but a view that divides by a measured quantity should not
    -- assume the measurement.
    (long.arm_value - control.control_value)
        / nullif(control.control_value, 0)                  AS relative_lift
FROM long
JOIN control USING (outcome)
WHERE long.arm <> 'No E-Mail'
ORDER BY long.arm, long.outcome;
