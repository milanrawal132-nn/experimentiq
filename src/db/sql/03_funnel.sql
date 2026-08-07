-- The visit -> conversion -> spend funnel, per arm.
--
-- Feature 1 established that the outcomes nest exactly:
--     visit = 1  is implied by  conversion = 1  <=>  spend > 0
--
-- Separating the funnel into an assignment-level rate and a conditional rate
-- answers a question the headline metrics cannot: when an email lifts
-- conversion, is that because it drove more people to the site, or because it
-- made the people who arrived more likely to buy?

CREATE OR REPLACE VIEW v_funnel AS
SELECT
    segment                                                     AS arm,
    count(*)                                                    AS assigned,
    sum(visit)                                                  AS visited,
    sum(conversion)                                             AS converted,

    -- Unconditional: share of everyone assigned to the arm.
    avg(visit)                                                  AS visit_rate,
    avg(conversion)                                             AS conversion_rate,

    -- Conditional: share of visitors who converted. This is NOT a causal
    -- quantity -- conditioning on visiting, which is itself affected by the
    -- treatment, breaks the randomisation. It is reported as a diagnostic for
    -- where in the funnel the effect sits, and is labelled as such wherever it
    -- is displayed.
    sum(conversion) / nullif(sum(visit), 0)::DOUBLE             AS conversion_rate_given_visit,
    sum(spend)      / nullif(sum(conversion), 0)                AS spend_per_converter
FROM customers
GROUP BY segment
ORDER BY segment;
