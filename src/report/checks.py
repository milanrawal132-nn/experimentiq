"""Consistency checks across independently generated result tables.

Eleven modules wrote the tables this report draws on, each runnable on its own.
Nothing forces them to have been run against the same data, with the same
configuration, or even in the same week. A report assembled from tables that
disagree with one another would read exactly like a report assembled from
tables that agree.

So the report checks its own inputs before quoting them. The checks here are
not restatements of what each module already tests -- those exist and pass in
`tests/`. They are *cross-table* invariants: relationships that hold only if
two separately-generated files describe the same experiment.

The strongest are the ones that span features. Feature 9's inverse propensity
estimate of "send Mens to everyone" must equal Feature 4's observed mean in the
Mens arm, because they are the same quantity computed two entirely different
ways. Feature 10's profit per email must equal Feature 4's spend effect run
through the margin. If those hold, the tables describe one experiment.

Every check runs at build time and its result is printed into the report, so a
reader sees what was verified rather than taking it on trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src import config

# Comparisons between two float representations of the same quantity, written
# by different modules. Generous enough to survive a CSV round trip, tight
# enough that a genuinely different number fails.
TOLERANCE = 1e-6


@dataclass(frozen=True)
class Check:
    """One cross-table invariant and whether it held."""

    name: str
    question: str
    passed: bool
    detail: str

    @property
    def symbol(self) -> str:
        return "pass" if self.passed else "FAIL"


def _close(left: float, right: float, tolerance: float = TOLERANCE) -> bool:
    return bool(np.isclose(left, right, rtol=0, atol=tolerance))


# ==========================================================================
# The checks
# ==========================================================================
def check_arm_sizes(results: dict[str, pd.DataFrame]) -> Check:
    """Do the arm sizes in the effects table match the randomisation table?"""
    srm = results["srm"]
    visit = results["ab_tests"].query("outcome == 'visit'")

    from_srm = int(srm["observed"].sum())
    from_ab = int(visit["n_treated"].sum() + visit["n_control"].iloc[0])

    return Check(
        name="Arm sizes agree",
        question="Do Feature 3 and Feature 4 describe the same number of customers?",
        passed=from_srm == from_ab == 64_000,
        detail=f"randomisation table {from_srm:,}, effects table {from_ab:,}",
    )


def check_shared_control(results: dict[str, pd.DataFrame]) -> Check:
    """Both comparisons use one control arm, so its mean cannot differ."""
    ab = results["ab_tests"]
    spreads = ab.groupby("outcome")["control_mean"].agg(lambda s: s.max() - s.min())

    return Check(
        name="Control arm is shared",
        question="Do both comparisons measure the same control customers?",
        passed=bool((spreads.abs() < TOLERANCE).all()),
        detail=f"largest disagreement across outcomes {spreads.abs().max():.2e}",
    )


def check_correction_is_conservative(results: dict[str, pd.DataFrame]) -> Check:
    """A multiplicity correction can only ever raise a p-value."""
    ab = results["ab_tests"]
    violations = int((ab["p_value_adjusted"] < ab["p_value"] - TOLERANCE).sum())

    return Check(
        name="Holm correction is conservative",
        question="Did the multiplicity correction raise every p-value?",
        passed=violations == 0,
        detail=f"{len(ab)} tests, {violations} adjusted below their raw value",
    )


def check_significance_matches_intervals(results: dict[str, pd.DataFrame]) -> Check:
    """Every result called significant must have an interval clear of zero."""
    ab = results["ab_tests"]
    significant = ab[ab["significant"]]
    straddling = int(((significant["ci_low"] <= 0) & (significant["ci_high"] >= 0)).sum())

    return Check(
        name="Significance agrees with the intervals",
        question="Does every significant result have an interval excluding zero?",
        passed=straddling == 0,
        detail=f"{len(significant)} significant results, {straddling} straddling zero",
    )


def check_robust_implies_significant(results: dict[str, pd.DataFrame]) -> Check:
    """Robustness is the stricter bar, so it cannot hold where significance fails."""
    power = results["power"].merge(
        results["ab_tests"][["outcome", "treatment_arm", "significant"]],
        on=["outcome", "treatment_arm"],
    )
    violations = int((power["robust"] & ~power["significant"]).sum())

    return Check(
        name="Robustness implies significance",
        question="Is anything called robust that was not even significant?",
        passed=violations == 0,
        detail=f"{int(power['robust'].sum())} of {len(power)} robust, "
               f"{violations} without significance",
    )


def check_cuped_identity(results: dict[str, pd.DataFrame]) -> Check:
    """CUPED removes exactly the square of the correlation. An identity, not a rule."""
    cuped = results["cuped"]
    predicted = cuped["correlation"] ** 2
    worst = float((predicted - cuped["total_variance_reduction"]).abs().max())

    return Check(
        name="CUPED reduction equals the squared correlation",
        question="Does the variance reduction match its theoretical value?",
        passed=worst < 1e-4,
        detail=f"largest departure from rho-squared across "
               f"{len(cuped)} rows: {worst:.2e}",
    )


def check_policy_recovers_arm_means(results: dict[str, pd.DataFrame]) -> Check:
    """The strongest cross-feature check available.

    Feature 9 estimates "send Mens to everyone" by reweighting the customers who
    happened to receive Mens. Feature 4 simply averages that arm. They are the
    same quantity reached two entirely different ways, so they must agree --
    and if they do, the two features are describing one experiment.
    """
    values = results["policy_values"].set_index("policy")["value"]
    ab = results["ab_tests"].query("outcome == 'visit'").set_index("treatment_arm")

    pairs = {
        "everyone gets Mens": ab.loc[config.MENS_ARM, "treated_mean"],
        "everyone gets Womens": ab.loc[config.WOMENS_ARM, "treated_mean"],
        "email nobody": ab.loc[config.MENS_ARM, "control_mean"],
    }
    gaps = {name: abs(values[name] - expected) for name, expected in pairs.items()}

    return Check(
        name="Policy values recover the observed arm means",
        question="Does inverse propensity weighting reproduce what was measured?",
        passed=all(gap < TOLERANCE for gap in gaps.values()),
        detail=f"largest gap across the three fixed policies {max(gaps.values()):.2e}",
    )


def check_profit_follows_from_spend(results: dict[str, pd.DataFrame]) -> Check:
    """Feature 10's profit must be Feature 4's spend effect run through the margin."""
    economics = results["economics"].set_index("treatment_arm")
    spend = results["ab_tests"].query("outcome == 'spend'").set_index("treatment_arm")

    margin = float(economics["margin"].iloc[0])
    cost = float(economics["cost"].iloc[0])

    gaps = {
        arm: abs(
            economics.loc[arm, "profit_per_email"]
            - (margin * spend.loc[arm, "absolute_effect"] - cost)
        )
        for arm in economics.index
    }

    return Check(
        name="Profit follows from the spend effect",
        question=f"Is profit per email exactly {margin:.0%} of spend lift minus ${cost:.2f}?",
        passed=all(gap < TOLERANCE for gap in gaps.values()),
        detail=f"largest gap across both campaigns {max(gaps.values()):.2e}",
    )


def check_break_even_margin(results: dict[str, pd.DataFrame]) -> Check:
    """At the break-even margin, profit per email is zero by construction."""
    economics = results["economics"]
    residual = (
        economics["break_even_margin"] * economics["spend_effect"] - economics["cost"]
    ).abs().max()

    return Check(
        name="Break-even margin zeroes the profit",
        question="Does the quoted break-even margin actually break even?",
        passed=bool(residual < TOLERANCE),
        detail=f"largest residual profit at break-even {residual:.2e}",
    )


def check_personalisation_below_detection(results: dict[str, pd.DataFrame]) -> Check:
    """The claim the report leans on hardest, checked rather than asserted.

    Feature 9 found personalisation's gain not significant. Feature 5 fixed the
    smallest effect the design can detect. The report's reading -- that this is
    a power limitation rather than a demonstration of no value -- only holds if
    the gain really is below that threshold.
    """
    gain = float(
        results["policy_differences"]
        .set_index("policy")
        .loc["best campaign per customer", "difference"]
    )
    mde = float(
        results["power"].query("outcome == 'visit'")["mde_absolute"].iloc[0]
    )

    return Check(
        name="Personalisation gain sits below the detection threshold",
        question="Is the null result a power limitation rather than a measured zero?",
        passed=0 < gain < mde,
        detail=f"gain {gain * 100:+.2f} pp against an MDE of {mde * 100:.2f} pp "
               f"({mde / gain:.1f}x larger)",
    )


def check_budget_accounting(results: dict[str, pd.DataFrame]) -> Check:
    """Net gain is profit above the do-nothing baseline, not total profit."""
    curve = results["budget_curve"]
    residual = (
        curve["net_gain"] - (curve["total_profit"] - curve["baseline_profit"])
    ).abs().max()

    return Check(
        name="Budget accounting nets off the baseline",
        question="Is reported gain measured against emailing nobody?",
        passed=bool(residual < 1e-3),
        detail=f"largest residual across {len(curve)} budget levels {residual:.2e}",
    )


def check_uplift_significance(results: dict[str, pd.DataFrame]) -> Check:
    """Nothing can be called significant while scoring below the random null."""
    uplift = results["uplift"]
    significant = uplift[uplift["significant"]]
    violations = int((significant["qini_percentile"] < 95.0).sum())

    return Check(
        name="Uplift significance requires beating the null",
        question="Did every significant learner clear the random-ranking null?",
        passed=violations == 0,
        detail=f"{len(significant)} of {len(uplift)} learners significant, "
               f"{violations} below the 95th percentile of the null",
    )


CHECKS: list[Callable[[dict[str, pd.DataFrame]], Check]] = [
    check_arm_sizes,
    check_shared_control,
    check_correction_is_conservative,
    check_significance_matches_intervals,
    check_robust_implies_significant,
    check_cuped_identity,
    check_policy_recovers_arm_means,
    check_profit_follows_from_spend,
    check_break_even_margin,
    check_personalisation_below_detection,
    check_budget_accounting,
    check_uplift_significance,
]


def run_checks(results: dict[str, pd.DataFrame]) -> list[Check]:
    """Run every cross-table invariant."""
    return [check(results) for check in CHECKS]


def failures(checks: list[Check]) -> list[Check]:
    return [check for check in checks if not check.passed]
