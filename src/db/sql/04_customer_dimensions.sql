-- Every customer's pre-treatment attributes, reshaped from one column per
-- attribute into one row per (customer, dimension, level).
--
-- Why unpivot at all: the dashboard, the subgroup analysis in Feature 7 and
-- the balance checks in Feature 3 all want to iterate over "each way of
-- slicing the customer base" without hard-coding the list of slices. In wide
-- form that means five near-identical GROUP BY queries; in long form it is one
-- GROUP BY with an extra key.
--
-- Only pre-treatment attributes appear here. Slicing outcomes by anything
-- measured after the send would condition on a post-treatment variable and
-- break the comparison between arms.

CREATE OR REPLACE VIEW v_customer_dimensions AS
UNPIVOT (
    SELECT
        customer_id,
        history_segment::VARCHAR                    AS "Prior spend band",
        recency_bucket::VARCHAR                     AS "Recency",
        zip_code::VARCHAR                           AS "Location",
        channel::VARCHAR                            AS "Purchase channel",

        -- The three binary flags are relabelled rather than left as 0/1, so
        -- the dashboard never has to translate a raw indicator into a legend.
        CASE newbie WHEN 1 THEN 'New customer'
                    ELSE 'Existing customer' END    AS "Tenure",
        CASE mens   WHEN 1 THEN 'Bought mens'
                    ELSE 'No mens purchase' END     AS "Mens history",
        CASE womens WHEN 1 THEN 'Bought womens'
                    ELSE 'No womens purchase' END   AS "Womens history"
    FROM customers
)
ON COLUMNS(* EXCLUDE (customer_id))
INTO NAME dimension VALUE level;
