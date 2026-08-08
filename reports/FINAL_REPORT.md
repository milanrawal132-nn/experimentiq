# ExperimentIQ — final report

*Generated from `reports/results/` by `python -m src.report.build`. Every
figure below is read from a result table rather than typed in.*

---

## Executive summary

**64,000 customers were randomly assigned** to one of three groups: no email, a
mens-merchandise campaign, or a womens-merchandise campaign. Site visits,
conversions and spend were recorded over the following two weeks. Because
assignment was random and verified as such before anything else was measured,
every comparison below is causal rather than correlational.

**Both campaigns work, and one works considerably better.** The mens campaign
raised site visits by +7.66 pp (72% relative), the womens campaign by +4.52 pp
(43%). 6 of 6 campaign-outcome comparisons remain significant after correcting
for having run several tests, and on visits the mens campaign is 1.7x the
womens campaign.

**In money, only one of them is a decision.** An email costs $0.10 and returns
30% of whatever extra spend it causes, so it must generate $0.33 of incremental
spend simply to pay for itself. The mens campaign clears that comfortably at
$0.131 per email ([$0.046, $0.216]). The womens campaign returns $0.027 with an
interval of [-$0.049, $0.104] — it may be profitable, and this experiment
cannot say.

**Personalised targeting is not worth building.** A model choosing a campaign
per customer beats sending the mens campaign to everyone by +0.18 pp, which is
not distinguishable from zero (p = 0.40). Its decisions are directionally
correct; the gain is simply too small for an experiment of this size to
resolve.

### Recommendation

**Send the mens campaign to every customer the budget covers.** It is the only
option demonstrated to make money, it needs no model, no scoring pipeline and
no monitoring, and nothing tested here improves on it measurably.

![Outcomes by arm](figures/01_outcomes_by_arm.png)

---

## The decisions, and how much confidence each carries

| Decision | Answer | Confidence |
|---|---|---|
| Should we email at all? | **Yes** | High — every outcome significant after correction |
| Which campaign as a default? | **Mens E-Mail** | High — larger on all three outcomes |
| Does the mens campaign pay for itself? | **Yes**, $0.131/email | High — interval excludes zero |
| Does the womens campaign pay for itself? | **Unknown** | None — interval spans zero |
| Should we personalise per customer? | **No** | Moderate — a null result, not a proven zero |
| Should we withhold email from anyone? | **Only on cost grounds** | Low — no demonstrable revenue effect |
| How much budget should we spend? | **All of it** | Moderate — return is 1.74x at a full send |

The confidence column is doing real work. Three of these are backed by
intervals that exclude the alternative; three rest on intervals that are simply
too wide to decide, and are marked as such rather than rounded to the nearer
answer.

**The margin assumption is load-bearing for exactly one row.** The mens
campaign stays profitable down to a 20.6% gross margin even if its true effect
sits at the pessimistic end of its interval, so the assumed 30% is not what
makes that decision. The womens campaign would need 59.2% under the same
pessimism — for that arm the assumption *is* the answer, which is why it is not
treated as one.

---

## What we found

### 1. The experiment is sound, and that was established first

Nothing downstream means anything if assignment was not random, so the
randomisation was checked before any effect was estimated.

- **Arm sizes** are consistent with the intended equal split
  (chi-square 0.20, p = 0.90).
- **No customer attribute is imbalanced.** The largest standardised difference
  across every pre-treatment attribute is 0.016, against a
  conventional concern threshold of 0.1.
- **No model can predict which arm a customer landed in** from their attributes:
  pseudo-R² of 0.00022. That test was verified to have teeth by running
  it on a deliberately confounded copy of the data, where it fires immediately.

### 2. Both campaigns work

![Treatment effects](figures/04_treatment_effects.png)

| Outcome | Mens E-Mail | Womens E-Mail |
|---|---|---|
| Visit | +7.66 pp (72%) | +4.52 pp (43%) |
| Conversion | +0.68 pp (119%) | +0.31 pp (54%) |
| Spend | $0.77 (118%) | $0.42 (65%) |

Six comparisons were run, so p-values are Holm-corrected before anything is
called significant. All six survive.

### 3. Not every result is equally solid

Significance says an effect is unlikely to be zero. It does not say the
experiment was large enough to measure that effect dependably. Those come apart
here.

The smallest visit effect this design can detect at 80% power is 0.85 pp.
Judging each result by whether the *low end of its interval* clears that
threshold — rather than by its p-value, which is circular — **4 of 6 results
are robust**. The two that are not are the womens campaign on conversion and on
spend: real effects, measured at the edge of what this experiment can see.

### 4. Variance reduction cannot rescue the weak results

CUPED removes exactly the square of the correlation between the outcome and a
pre-period covariate. That is an identity, so the technique cannot be argued
into working — it can only be handed a better covariate.

The best correlation available in this dataset is 0.159, giving a 2.5% variance
reduction on visits and 0.04% on spend. The implementation was verified against
synthetic data with a known correlation, where it recovers the theoretical
reduction to three decimals, so this is a fact about the data rather than a
bug.

**The finding is about instrumentation, not analysis.** `history` is a coarse
lifetime-spend band, not a pre-period measurement of the outcome. Logging each
customer's spend over a fixed pre-experiment window would plausibly correlate
0.4–0.6 and cut variance 16–36%, at the cost of one table.

### 5. One campaign varies by customer; the other does not

![Heterogeneous effects](figures/07_heterogeneous_effects.png)

The womens campaign lifts visits by +7.31 pp among customers who previously
bought womens merchandise and +1.11 pp among those who did not — a 6.6x gap
(interaction F = 94, adjusted p = 3.0e-21). The mens campaign shows no
significant interaction with any pre-registered attribute: it works about
equally well on everyone.

Subgroups were fixed in `config.py` with a written rationale each, before any
subgroup effect was estimated. Heterogeneity is judged by an **interaction
test**, not by comparing per-subgroup p-values — two estimates can straddle a
significance threshold while being statistically indistinguishable from each
other.

**No subgroup is harmed by either campaign** at the resolution available.

### 6. Uplift models agree, and the simplest one wins

Individual-level models were fitted to rank customers by predicted effect, with
every prediction made out-of-fold and every score compared against the
distribution produced by 500 random rankings on the same data.

| | Womens E-Mail | Mens E-Mail |
|---|---|---|
| Learners beating the null | **5 of 5** | 1 of 5 |
| Best Qini | 0.064 (z = 8.4) | 0.019 |

This was predicted before any model was fitted: finding 6 said the womens
campaign varies and the mens campaign does not, so a working model should find
signal on one and not the other. It did. The best model is the **logistic
T-learner** — the simplest tested — and profiling its top-ranked customers
recovers *prior womens purchase* independently, the same mechanism finding 5
identified without being told about it.

### 7. Targeting is directionally right and financially irrelevant

![Targeting policy](figures/09_targeting_policy.png)

Policies are valued by inverse propensity weighting: a customer received one
arm, so a policy that would have assigned them differently is estimated by
reweighting the customers whose actual arm matched.

| Policy | Visit rate |
|---|---|
| Best campaign per customer | 0.1845 |
| Everyone gets Mens | 0.1828 |
| Everyone gets Womens | 0.1514 |
| Random assignment | 0.1474 |
| Email nobody | 0.1062 |

The learned policy beats sending mens to everyone by +0.18 pp, interval [-0.23
pp, +0.59 pp], p = 0.40.

**Its decisions audit as correct.** Among the customers it assigns mens, the
mens campaign really is the better one; among those it assigns womens, the
ordering genuinely flips. That audit uses observed outcomes only, no model. The
problem is the size of the second gap, not its direction — the population-level
gain lands roughly 5x below what this experiment can resolve.

The honest statement is therefore **not** "personalisation does not work" but
*this experiment cannot determine whether it does*. Those support opposite
decisions, which is why the distinction is worth the paragraph.

### 8. In money, the recommendation simplifies further

![Budget optimiser](figures/10_budget_optimiser.png)

Ranking a budget by the **spend** uplift model — which failed its null test —
loses -$3140 against simply sending mens to everyone (p = 0.034). Ranking by
the **visit** model, which passed, is indistinguishable from using no model at
all (p = 0.65).

That pairing is the most transferable result in this project. A model that
passes a null test is harmless in deployment even when it adds nothing; a model
that fails one is actively expensive, because it overrides a default that was
already known to be correct. **The cost of deploying an unvalidated model is
not zero.**

---

## What this analysis could not determine

Listing these is not hedging. Each one is a question a reader might reasonably
think has been answered here, and has not.

**Whether the womens campaign is profitable.** It lifts spend $0.42 against a
$0.33 break-even, with an interval of [$0.17, $0.68] that contains the
threshold. This is the one open question a realistically-sized follow-up could
close.

**Whether personalisation is worth building.** The gain is below the design's
resolution. Answering it as posed would need on the order of 1.4 million
customers; answering it on the ~30% of customers where the policy actually
deviates is a far smaller experiment.

**Whether spend effects vary by customer.** Spend is the noisiest outcome by a
wide margin and its uplift models fail their null test. Nothing here supports
ranking customers on predicted spend.

**Anything beyond two weeks.** Outcomes were measured over a fixed short
window. Repeat purchasing, unsubscribes, and list fatigue are outside what was
recorded, and a campaign that wins over two weeks can lose over a year.

**Anything about individual people.** The dataset carries no customer
identifier, and 6,562 rows are exact duplicates. These are retained rather than
dropped — with a coarse feature space, distinct customers can legitimately
share a row, and removing them would non-randomly delete customers with common
attribute profiles. Every result is therefore a statement about rows, and about
the population they represent, not about tracked individuals.

**Whether the two business assumptions hold.** Gross margin and cost per email
are stated in `config.py` because the dataset does not contain them. Every
profit conclusion is conditional on them; every causal conclusion is not.

---

## How the analysis defends itself

The choices below are the ones most likely to be asked about.

**The three-arm test is analysed as two two-arm tests**, each against the
shared control, with the resulting multiplicity corrected explicitly. This
keeps every method in its standard binary-treatment form rather than requiring
multi-treatment variants.

**Randomisation was verified before anything was estimated**, and the check was
itself checked — running it on a deliberately confounded copy of the data
confirms it discriminates rather than always passing.

**Subgroups were pre-registered** in version control with a rationale each,
before any subgroup effect was computed. Searching every possible split and
reporting the strongest is how subgroup analysis earns its reputation.

**Every model prediction is out-of-fold.** On data constructed with no
heterogeneity at all, in-sample scoring produces a Qini of 0.578 against an
honest 0.012 — a 47x inflation from leakage alone.

**Every model score is compared against a null.** Qini is high-variance, and a
model that has learned nothing still returns a non-zero score more often than
not. Five learners were tried, so the p-values are also corrected for having
picked the best of five.

**Policy comparisons are paired.** Two policies agreeing on most customers have
correlated errors; differencing per customer before aggregating makes the
comparison roughly twice as precise and prevents a null result being
manufactured by a loose standard error.

**Greedy budget allocation is exactly optimal, not a heuristic**, because every
email costs the same — the test suite brute-forces this against every subset
rather than asserting it in a comment.

**Assumptions are isolated.** Anything not measured lives in `config.py` and
nowhere else, so what is conditional on an assumption and what is not can be
told apart by reading one file.

**Inference does not live in SQL.** The warehouse views report point estimates
and no uncertainty, deliberately: SQL makes it easy to compute a difference and
awkward to compute the interval around it.

---

## Consistency checks

Eleven modules produced the tables above, each runnable on its own. Nothing
forces them to have been run against the same data or the same configuration,
so before quoting them this report verifies relationships that hold only if
they describe one experiment.

**12 of 12 checks passed.**

| Check | Question | Result | Detail |
|---|---|---|---|
| Arm sizes agree | Do Feature 3 and Feature 4 describe the same number of customers? | pass | randomisation table 64,000, effects table 64,000 |
| Control arm is shared | Do both comparisons measure the same control customers? | pass | largest disagreement across outcomes 0.00e+00 |
| Holm correction is conservative | Did the multiplicity correction raise every p-value? | pass | 6 tests, 0 adjusted below their raw value |
| Significance agrees with the intervals | Does every significant result have an interval excluding zero? | pass | 6 significant results, 0 straddling zero |
| Robustness implies significance | Is anything called robust that was not even significant? | pass | 4 of 6 robust, 0 without significance |
| CUPED reduction equals the squared correlation | Does the variance reduction match its theoretical value? | pass | largest departure from rho-squared across 12 rows: 3.36e-16 |
| Policy values recover the observed arm means | Does inverse propensity weighting reproduce what was measured? | pass | largest gap across the three fixed policies 8.33e-17 |
| Profit follows from the spend effect | Is profit per email exactly 30% of spend lift minus $0.10? | pass | largest gap across both campaigns 1.11e-16 |
| Break-even margin zeroes the profit | Does the quoted break-even margin actually break even? | pass | largest residual profit at break-even 1.39e-17 |
| Personalisation gain sits below the detection threshold | Is the null result a power limitation rather than a measured zero? | pass | gain +0.18 pp against an MDE of 0.85 pp (4.8x larger) |
| Budget accounting nets off the baseline | Is reported gain measured against emailing nobody? | pass | largest residual across 84 budget levels 3.64e-12 |
| Uplift significance requires beating the null | Did every significant learner clear the random-ranking null? | pass | 6 of 10 learners significant, 0 below the 95th percentile of the null |

The two worth reading closely are *Policy values recover the observed arm
means* and *Profit follows from the spend effect*. Both compare a quantity
computed two entirely different ways — an inverse-propensity estimate against a
plain average, and a profit model against a spend effect run through the
margin. They cannot agree by accident.

---

## Appendix — where everything comes from

| Result | Table | Notebook |
|---|---|---|
| Randomisation checks | `03_srm.csv`, `03_covariate_balance.csv`, `03_omnibus_balance.csv` | `notebooks/03_randomisation_diagnostics.ipynb` |
| Treatment effects | `04_ab_test_results.csv` | `notebooks/04_treatment_effects.ipynb` |
| Power and detectability | `05_power_analysis.csv` | `notebooks/05_power_analysis.ipynb` |
| Variance reduction | `06_cuped.csv` | `notebooks/06_cuped.ipynb` |
| Subgroup effects | `07_subgroup_effects.csv`, `07_interaction_tests.csv` | `notebooks/07_heterogeneous_effects.ipynb` |
| Uplift models | `08_uplift_models.csv` | `notebooks/08_uplift_models.ipynb` |
| Targeting policy | `09_policy_values.csv`, `09_policy_differences.csv` | `notebooks/09_targeting_policy.ipynb` |
| Profit and budget | `10_campaign_economics.csv`, `10_budget_curve.csv`, `10_ranking_comparison.csv` | `notebooks/10_budget_optimiser.ipynb` |

Tables live in `reports/results/`. To regenerate everything from the raw CSV:

```bash
python -m src.data.load
python -m src.db.warehouse
python -m src.analysis.diagnostics
python -m src.analysis.ab_test
python -m src.analysis.power
python -m src.analysis.cuped
python -m src.analysis.heterogeneity
python -m src.models.uplift
python -m src.models.policy
python -m src.models.budget
python -m src.report.build
```

Source: [Kevin Hillstrom's MineThatData E-Mail Analytics
Challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-
and-data.html).
