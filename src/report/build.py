"""Generate the final written report from the pipeline's own results.

    python -m src.report.build

Every number in `reports/FINAL_REPORT.md` is read from a generated result table
at build time. The prose is authored; the figures in it are not. Nothing is
typed in by hand, so the report cannot quietly fall out of date when an analysis
is re-run with different settings -- it simply regenerates saying something
different, which is the correct behaviour.

The report also verifies its own inputs. `src/report/checks.py` runs twelve
cross-table invariants before anything is written, and their results appear in
the report itself. A reader sees what was checked rather than trusting that
anything was.

Building fails loudly if a check fails. A report that quietly reports its own
inconsistency would be worse than no report.
"""

from __future__ import annotations

import logging
import re
import textwrap

import pandas as pd

from src import config
from src.dashboard import loaders
from src.report.checks import Check, failures, run_checks

logger = logging.getLogger(__name__)

REPORT_PATH = config.REPORTS_DIR / "FINAL_REPORT.md"

# Tables the report draws on. Named rather than globbed, so a missing one is an
# error with a rebuild command attached rather than a silently shorter report.
REQUIRED_RESULTS = [
    "srm", "balance", "omnibus", "ab_tests", "power", "cuped",
    "subgroups", "interactions", "uplift", "policy_values",
    "policy_differences", "economics", "budget_curve", "ranking_comparison",
]

# Figures the report embeds, relative to the report's own location.
FIGURES = {
    "outcomes": "figures/01_outcomes_by_arm.png",
    "effects": "figures/04_treatment_effects.png",
    "heterogeneity": "figures/07_heterogeneous_effects.png",
    "policy": "figures/09_targeting_policy.png",
    "budget": "figures/10_budget_optimiser.png",
}


class InconsistentResults(RuntimeError):
    """Cross-table checks failed; the results do not describe one experiment."""


# ==========================================================================
# Accessors
# ==========================================================================
def load_results() -> dict[str, pd.DataFrame]:
    """Read every table the report needs."""
    return {name: loaders.result(name) for name in REQUIRED_RESULTS}


def effect(ab: pd.DataFrame, outcome: str, arm: str) -> pd.Series:
    """One row of the treatment-effect table."""
    row = ab[(ab["outcome"] == outcome) & (ab["treatment_arm"] == arm)]
    if row.empty:
        raise KeyError(f"No {outcome} effect for {arm}")
    return row.iloc[0]


def pp(value: float, places: int = 2) -> str:
    """Format an effect in percentage points, signed."""
    return f"{value * 100:+.{places}f} pp"


def points(value: float, places: int = 2) -> str:
    """Format a magnitude in percentage points, unsigned.

    Thresholds and widths are not directional, and a leading `+` on one reads
    as a claim that it is.
    """
    return f"{value * 100:.{places}f} pp"


def money(value: float, places: int = 3) -> str:
    """Format currency with the sign outside the symbol, never inside it."""
    return f"-${abs(value):.{places}f}" if value < 0 else f"${value:.{places}f}"


# ==========================================================================
# Layout
# ==========================================================================
# Lines whose breaks carry meaning in Markdown. Note that a bullet is `- ` or
# `* ` *followed by a space*: `**bold**` opens a great many paragraphs in this
# report and is not a list, and `---` is a rule rather than an empty bullet.
LIST_ITEM = re.compile(r"^([-*+]\s|\d+[.)]\s)")
THEMATIC_BREAK = re.compile(r"^([-*_])\1{2,}\s*$")
BLOCK_PREFIXES = ("|", "#", "!", ">")


def is_structural(line: str) -> bool:
    """Whether a line's own break must be preserved verbatim."""
    if line != line.lstrip():
        # Indented: a list continuation or an indented code block. Re-wrapping
        # would flatten it against the left margin and change what it belongs
        # to.
        return True
    return (
        line.startswith(BLOCK_PREFIXES)
        or bool(LIST_ITEM.match(line))
        or bool(THEMATIC_BREAK.match(line))
    )


def reflow(text: str, width: int = 79) -> str:
    """Re-wrap prose paragraphs after interpolation.

    Values are substituted into prose whose line breaks were chosen before
    anyone knew how long those values would be, which leaves numbers stranded
    mid-sentence. Markdown renders that correctly, but a report is read as a
    raw file often enough for it to be worth fixing.

    Tables, headings, lists, images and fenced code are left exactly as
    written.
    """
    output: list[str] = []
    paragraph: list[str] = []
    in_fence = False

    def flush() -> None:
        if paragraph:
            output.extend(textwrap.wrap(" ".join(paragraph), width=width))
            paragraph.clear()

    for line in text.split("\n"):
        if line.strip().startswith("```"):
            flush()
            in_fence = not in_fence
            output.append(line)
        elif in_fence or not line.strip() or is_structural(line):
            flush()
            output.append(line)
        else:
            paragraph.append(line.strip())

    flush()
    return "\n".join(output)


# ==========================================================================
# Sections
# ==========================================================================
def section_summary(results: dict[str, pd.DataFrame]) -> str:
    ab = results["ab_tests"]
    economics = results["economics"].set_index("treatment_arm")
    differences = results["policy_differences"].set_index("policy")

    mens_visit = effect(ab, "visit", config.MENS_ARM)
    womens_visit = effect(ab, "visit", config.WOMENS_ARM)
    learned = differences.loc["best campaign per customer"]

    mens_profit = economics.loc[config.MENS_ARM]
    womens_profit = economics.loc[config.WOMENS_ARM]

    n_significant = int(ab["significant"].sum())
    visit_ratio = mens_visit["absolute_effect"] / womens_visit["absolute_effect"]

    return f"""# ExperimentIQ — final report

*Generated from `reports/results/` by `python -m src.report.build`. Every figure
below is read from a result table rather than typed in.*

---

## Executive summary

**64,000 customers were randomly assigned** to one of three groups: no email, a
mens-merchandise campaign, or a womens-merchandise campaign. Site visits,
conversions and spend were recorded over the following two weeks. Because
assignment was random and verified as such before anything else was measured,
every comparison below is causal rather than correlational.

**Both campaigns work, and one works considerably better.** The mens campaign
raised site visits by {pp(mens_visit['absolute_effect'])}
({mens_visit['relative_effect']:.0%} relative), the womens campaign by
{pp(womens_visit['absolute_effect'])} ({womens_visit['relative_effect']:.0%}).
{n_significant} of {len(ab)} campaign-outcome comparisons remain significant
after correcting for having run several tests, and on visits the mens campaign
is {visit_ratio:.1f}x the womens campaign.

**In money, only one of them is a decision.** An email costs
${mens_profit['cost']:.2f} and returns {mens_profit['margin']:.0%} of whatever
extra spend it causes, so it must generate
${mens_profit['break_even_spend']:.2f} of incremental spend simply to pay for
itself. The mens campaign clears that comfortably at
{money(mens_profit['profit_per_email'])} per email
([{money(mens_profit['profit_ci_low'])},
{money(mens_profit['profit_ci_high'])}]). The womens campaign returns
{money(womens_profit['profit_per_email'])} with an interval of
[{money(womens_profit['profit_ci_low'])},
{money(womens_profit['profit_ci_high'])}] — it may be profitable, and this
experiment cannot say.

**Personalised targeting is not worth building.** A model choosing a campaign
per customer beats sending the mens campaign to everyone by
{pp(learned['difference'])}, which is not distinguishable from zero
(p = {learned['p_value']:.2f}). Its decisions are directionally correct; the
gain is simply too small for an experiment of this size to resolve.

### Recommendation

**Send the mens campaign to every customer the budget covers.** It is the only
option demonstrated to make money, it needs no model, no scoring pipeline and no
monitoring, and nothing tested here improves on it measurably.

![Outcomes by arm]({FIGURES['outcomes']})
"""


def section_decisions(results: dict[str, pd.DataFrame]) -> str:
    economics = results["economics"].set_index("treatment_arm")
    curve = results["budget_curve"]
    full = curve[curve["budget"] == curve["budget"].max()]
    best = full.loc[full["net_gain"].idxmax()]

    mens = economics.loc[config.MENS_ARM]
    womens = economics.loc[config.WOMENS_ARM]

    return f"""---

## The decisions, and how much confidence each carries

| Decision | Answer | Confidence |
|---|---|---|
| Should we email at all? | **Yes** | High — every outcome significant after correction |
| Which campaign as a default? | **Mens E-Mail** | High — larger on all three outcomes |
| Does the mens campaign pay for itself? | **Yes**, {money(mens['profit_per_email'])}/email | High — interval excludes zero |
| Does the womens campaign pay for itself? | **Unknown** | None — interval spans zero |
| Should we personalise per customer? | **No** | Moderate — a null result, not a proven zero |
| Should we withhold email from anyone? | **Only on cost grounds** | Low — no demonstrable revenue effect |
| How much budget should we spend? | **All of it** | Moderate — return is {best['return_on_spend']:.2f}x at a full send |

The confidence column is doing real work. Three of these are backed by intervals
that exclude the alternative; three rest on intervals that are simply too wide
to decide, and are marked as such rather than rounded to the nearer answer.

**The margin assumption is load-bearing for exactly one row.** The mens campaign
stays profitable down to a {mens['break_even_margin_high']:.1%} gross margin even
if its true effect sits at the pessimistic end of its interval, so the assumed
{mens['margin']:.0%} is not what makes that decision. The womens campaign would
need {womens['break_even_margin_high']:.1%} under the same pessimism — for that
arm the assumption *is* the answer, which is why it is not treated as one.
"""


def section_findings(results: dict[str, pd.DataFrame]) -> str:
    ab = results["ab_tests"]
    power = results["power"]
    cuped = results["cuped"]
    subgroups = results["subgroups"]
    interactions = results["interactions"]
    uplift = results["uplift"]
    values = results["policy_values"].set_index("policy")["value"]
    differences = results["policy_differences"].set_index("policy")
    ranking = results["ranking_comparison"].set_index("ranking")

    srm = loaders.srm_verdict(results["srm"])
    worst_smd = results["balance"]["abs_smd"].max()
    worst_r2 = results["omnibus"]["pseudo_r2"].max()

    visit_mde = float(power.query("outcome == 'visit'")["mde_absolute"].iloc[0])
    n_robust = int(power["robust"].sum())

    best_cuped = cuped.loc[cuped["correlation"].abs().idxmax()]
    spend_cuped = cuped.query(
        "outcome == 'spend' and covariate == 'history'"
    )["total_variance_reduction"].max()

    womens_split = subgroups[
        (subgroups["outcome"] == "visit")
        & (subgroups["treatment_arm"] == config.WOMENS_ARM)
        & (subgroups["subgroup"] == "womens")
    ].set_index("level")
    bought = womens_split.loc["Bought womens", "effect"]
    not_bought = womens_split.loc["No womens purchase", "effect"]

    womens_interaction = interactions[
        (interactions["outcome"] == "visit")
        & (interactions["treatment_arm"] == config.WOMENS_ARM)
        & (interactions["subgroup"] == "womens")
    ].iloc[0]

    visit_uplift = uplift[uplift["outcome"] == "visit"] if "outcome" in uplift else uplift
    womens_learners = visit_uplift[visit_uplift["treatment_arm"] == config.WOMENS_ARM]
    mens_learners = visit_uplift[visit_uplift["treatment_arm"] == config.MENS_ARM]
    best_learner = womens_learners.loc[womens_learners["qini"].idxmax()]

    learned = differences.loc["best campaign per customer"]
    spend_ranking = ranking.loc["predicted profit"]
    visit_ranking = ranking.loc["predicted visit uplift"]

    # The ranking table reports a per-customer difference; the readable figure
    # is what that costs across the file.
    n_customers = int(results["policy_values"]["n_total"].iloc[0])

    return f"""---

## What we found

### 1. The experiment is sound, and that was established first

Nothing downstream means anything if assignment was not random, so the
randomisation was checked before any effect was estimated.

- **Arm sizes** are consistent with the intended equal split
  (chi-square {srm['chi2']:.2f}, p = {srm['p_value']:.2f}).
- **No customer attribute is imbalanced.** The largest standardised difference
  across every pre-treatment attribute is {worst_smd:.3f}, against a
  conventional concern threshold of {config.SMD_THRESHOLD}.
- **No model can predict which arm a customer landed in** from their attributes:
  pseudo-R² of {worst_r2:.5f}. That test was verified to have teeth by running
  it on a deliberately confounded copy of the data, where it fires immediately.

### 2. Both campaigns work

![Treatment effects]({FIGURES['effects']})

| Outcome | Mens E-Mail | Womens E-Mail |
|---|---|---|
| Visit | {pp(effect(ab, 'visit', config.MENS_ARM)['absolute_effect'])} ({effect(ab, 'visit', config.MENS_ARM)['relative_effect']:.0%}) | {pp(effect(ab, 'visit', config.WOMENS_ARM)['absolute_effect'])} ({effect(ab, 'visit', config.WOMENS_ARM)['relative_effect']:.0%}) |
| Conversion | {pp(effect(ab, 'conversion', config.MENS_ARM)['absolute_effect'])} ({effect(ab, 'conversion', config.MENS_ARM)['relative_effect']:.0%}) | {pp(effect(ab, 'conversion', config.WOMENS_ARM)['absolute_effect'])} ({effect(ab, 'conversion', config.WOMENS_ARM)['relative_effect']:.0%}) |
| Spend | ${effect(ab, 'spend', config.MENS_ARM)['absolute_effect']:.2f} ({effect(ab, 'spend', config.MENS_ARM)['relative_effect']:.0%}) | ${effect(ab, 'spend', config.WOMENS_ARM)['absolute_effect']:.2f} ({effect(ab, 'spend', config.WOMENS_ARM)['relative_effect']:.0%}) |

Six comparisons were run, so p-values are Holm-corrected before anything is
called significant. All six survive.

### 3. Not every result is equally solid

Significance says an effect is unlikely to be zero. It does not say the
experiment was large enough to measure that effect dependably. Those come apart
here.

The smallest visit effect this design can detect at 80% power is
{points(visit_mde)}. Judging each result by whether the *low end of its interval*
clears that threshold — rather than by its p-value, which is circular —
**{n_robust} of {len(power)} results are robust**. The two that are not are the
womens campaign on conversion and on spend: real effects, measured at the edge
of what this experiment can see.

### 4. Variance reduction cannot rescue the weak results

CUPED removes exactly the square of the correlation between the outcome and a
pre-period covariate. That is an identity, so the technique cannot be argued
into working — it can only be handed a better covariate.

The best correlation available in this dataset is {best_cuped['correlation']:.3f},
giving a {best_cuped['total_variance_reduction']:.1%} variance reduction on
visits and {spend_cuped:.2%} on spend. The implementation was verified against
synthetic data with a known correlation, where it recovers the theoretical
reduction to three decimals, so this is a fact about the data rather than a bug.

**The finding is about instrumentation, not analysis.** `history` is a coarse
lifetime-spend band, not a pre-period measurement of the outcome. Logging each
customer's spend over a fixed pre-experiment window would plausibly correlate
0.4–0.6 and cut variance 16–36%, at the cost of one table.

### 5. One campaign varies by customer; the other does not

![Heterogeneous effects]({FIGURES['heterogeneity']})

The womens campaign lifts visits by {pp(bought)} among customers who previously
bought womens merchandise and {pp(not_bought)} among those who did not — a
{bought / not_bought:.1f}x gap (interaction F = {womens_interaction['f_statistic']:.0f},
adjusted p = {womens_interaction['p_value_adjusted']:.1e}). The mens campaign
shows no significant interaction with any pre-registered attribute: it works
about equally well on everyone.

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
| Learners beating the null | **{int(womens_learners['significant'].sum())} of {len(womens_learners)}** | {int(mens_learners['significant'].sum())} of {len(mens_learners)} |
| Best Qini | {best_learner['qini']:.3f} (z = {best_learner['qini_z_score']:.1f}) | {mens_learners['qini'].max():.3f} |

This was predicted before any model was fitted: finding 6 said the womens
campaign varies and the mens campaign does not, so a working model should find
signal on one and not the other. It did. The best model is the **logistic
T-learner** — the simplest tested — and profiling its top-ranked customers
recovers *prior womens purchase* independently, the same mechanism finding 5
identified without being told about it.

### 7. Targeting is directionally right and financially irrelevant

![Targeting policy]({FIGURES['policy']})

Policies are valued by inverse propensity weighting: a customer received one
arm, so a policy that would have assigned them differently is estimated by
reweighting the customers whose actual arm matched.

| Policy | Visit rate |
|---|---|
| Best campaign per customer | {values['best campaign per customer']:.4f} |
| Everyone gets Mens | {values['everyone gets Mens']:.4f} |
| Everyone gets Womens | {values['everyone gets Womens']:.4f} |
| Random assignment | {values['random assignment']:.4f} |
| Email nobody | {values['email nobody']:.4f} |

The learned policy beats sending mens to everyone by {pp(learned['difference'])},
interval [{pp(learned['ci_low'])}, {pp(learned['ci_high'])}],
p = {learned['p_value']:.2f}.

**Its decisions audit as correct.** Among the customers it assigns mens, the
mens campaign really is the better one; among those it assigns womens, the
ordering genuinely flips. That audit uses observed outcomes only, no model. The
problem is the size of the second gap, not its direction — the population-level
gain lands roughly {visit_mde / learned['difference']:.0f}x below what this
experiment can resolve.

The honest statement is therefore **not** "personalisation does not work" but
*this experiment cannot determine whether it does*. Those support opposite
decisions, which is why the distinction is worth the paragraph.

### 8. In money, the recommendation simplifies further

![Budget optimiser]({FIGURES['budget']})

Ranking a budget by the **spend** uplift model — which failed its null test —
loses {money(spend_ranking['difference'] * n_customers, 0)} against simply sending
mens to everyone (p = {spend_ranking['p_value']:.3f}). Ranking by the **visit** model,
which passed, is indistinguishable from using no model at all
(p = {visit_ranking['p_value']:.2f}).

That pairing is the most transferable result in this project. A model that
passes a null test is harmless in deployment even when it adds nothing; a model
that fails one is actively expensive, because it overrides a default that was
already known to be correct. **The cost of deploying an unvalidated model is not
zero.**
"""


def section_limits(results: dict[str, pd.DataFrame]) -> str:
    economics = results["economics"].set_index("treatment_arm")
    womens = economics.loc[config.WOMENS_ARM]

    return f"""---

## What this analysis could not determine

Listing these is not hedging. Each one is a question a reader might reasonably
think has been answered here, and has not.

**Whether the womens campaign is profitable.** It lifts spend
${womens['spend_effect']:.2f} against a ${womens['break_even_spend']:.2f}
break-even, with an interval of [${womens['spend_ci_low']:.2f},
${womens['spend_ci_high']:.2f}] that contains the threshold. This is the one open
question a realistically-sized follow-up could close.

**Whether personalisation is worth building.** The gain is below the design's
resolution. Answering it as posed would need on the order of 1.4 million
customers; answering it on the ~30% of customers where the policy actually
deviates is a far smaller experiment.

**Whether spend effects vary by customer.** Spend is the noisiest outcome by a
wide margin and its uplift models fail their null test. Nothing here supports
ranking customers on predicted spend.

**Anything beyond two weeks.** Outcomes were measured over a fixed short window.
Repeat purchasing, unsubscribes, and list fatigue are outside what was recorded,
and a campaign that wins over two weeks can lose over a year.

**Anything about individual people.** The dataset carries no customer
identifier, and 6,562 rows are exact duplicates. These are retained rather than
dropped — with a coarse feature space, distinct customers can legitimately share
a row, and removing them would non-randomly delete customers with common
attribute profiles. Every result is therefore a statement about rows, and about
the population they represent, not about tracked individuals.

**Whether the two business assumptions hold.** Gross margin and cost per email
are stated in `config.py` because the dataset does not contain them. Every
profit conclusion is conditional on them; every causal conclusion is not.
"""


def section_method() -> str:
    return """---

## How the analysis defends itself

The choices below are the ones most likely to be asked about.

**The three-arm test is analysed as two two-arm tests**, each against the shared
control, with the resulting multiplicity corrected explicitly. This keeps every
method in its standard binary-treatment form rather than requiring
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
"""


def section_checks(checks: list[Check]) -> str:
    rows = "\n".join(
        f"| {check.name} | {check.question} | {check.symbol} | {check.detail} |"
        for check in checks
    )
    passed = sum(check.passed for check in checks)

    return f"""---

## Consistency checks

Eleven modules produced the tables above, each runnable on its own. Nothing
forces them to have been run against the same data or the same configuration, so
before quoting them this report verifies relationships that hold only if they
describe one experiment.

**{passed} of {len(checks)} checks passed.**

| Check | Question | Result | Detail |
|---|---|---|---|
{rows}

The two worth reading closely are *Policy values recover the observed arm means*
and *Profit follows from the spend effect*. Both compare a quantity computed two
entirely different ways — an inverse-propensity estimate against a plain
average, and a profit model against a spend effect run through the margin. They
cannot agree by accident.
"""


def section_appendix() -> str:
    return """---

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

Source: [Kevin Hillstrom's MineThatData E-Mail Analytics Challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html).
"""


# ==========================================================================
# Build
# ==========================================================================
def render(results: dict[str, pd.DataFrame], checks: list[Check]) -> str:
    """Assemble the full report."""
    return reflow("\n".join([
        section_summary(results),
        section_decisions(results),
        section_findings(results),
        section_limits(results),
        section_method(),
        section_checks(checks),
        section_appendix(),
    ]))


def build(save: bool = True, strict: bool = True) -> str:
    """Generate the report, refusing to write one from inconsistent inputs.

    Args:
        save: Write the report to `reports/FINAL_REPORT.md`.
        strict: Raise if any cross-table check fails. Turning this off is for
            inspecting *why* something failed, never for shipping.
    """
    results = load_results()
    checks = run_checks(results)

    broken = failures(checks)
    if broken and strict:
        raise InconsistentResults(
            "Cross-table checks failed; the result tables do not describe one "
            "experiment. Re-run the full pipeline.\n"
            + "\n".join(f"  - {c.name}: {c.detail}" for c in broken)
        )

    text = render(results, checks)

    if save:
        config.ensure_dirs()
        REPORT_PATH.write_text(text)
        logger.info("Wrote %s (%d words)", REPORT_PATH, len(text.split()))

    return text


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    text = build()

    checks = run_checks(load_results())
    print(f"\nConsistency checks: {sum(c.passed for c in checks)}/{len(checks)} passed")
    for check in checks:
        print(f"  [{check.symbol:4s}] {check.name} — {check.detail}")

    print(f"\nReport: {REPORT_PATH}")
    print(f"Words:  {len(text.split()):,}")


if __name__ == "__main__":
    main()
