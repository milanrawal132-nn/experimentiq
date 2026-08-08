"""What the dashboard reads, and the one thing it recomputes.

The dashboard displays results produced by Features 3 through 10. It does not
re-run any of them, and that is a design decision rather than a shortcut.

**Why the dashboard does not re-run inference.** Re-fitting uplift models on
every interaction would take tens of seconds and make the application unusable.
The worse problem is correctness: a dashboard that re-runs its own analysis can
quietly show numbers that disagree with the committed results, and there would
be no way for a reader to tell which is right. Reading the generated CSVs means
what the dashboard shows and what the repository reports are the same artifact
by construction.

**The one exception is the profit model.** Margin and cost per email are
assumptions written into `config.py`, not measurements, so the dashboard exposes
them as controls. That is safe to recompute live because it is not inference:
profit is a linear transform of the spend effect, so

    profit_effect = margin * spend_effect - cost
    profit_interval = margin * spend_interval - cost

exactly. Within an arm the send cost is a constant shift, so it changes no
variance; scaling by the margin scales the standard error and leaves the
degrees of freedom untouched. The test suite pins this against
`budget.campaign_economics` recomputed from the raw data, so the live numbers
are provably the same numbers rather than a fast approximation of them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.db import warehouse

# Every generated table the dashboard can show, keyed by the short name the
# application uses. Adding a page means adding an entry here rather than
# hard-coding a filename in the view layer.
RESULT_FILES = {
    "srm": "03_srm.csv",
    "balance": "03_covariate_balance.csv",
    "omnibus": "03_omnibus_balance.csv",
    "ab_tests": "04_ab_test_results.csv",
    "power": "05_power_analysis.csv",
    "cuped": "06_cuped.csv",
    "subgroups": "07_subgroup_effects.csv",
    "interactions": "07_interaction_tests.csv",
    "uplift": "08_uplift_models.csv",
    "policy_values": "09_policy_values.csv",
    "policy_differences": "09_policy_differences.csv",
    "economics": "10_campaign_economics.csv",
    "budget_curve": "10_budget_curve.csv",
    "ranking_comparison": "10_ranking_comparison.csv",
}

# The command that regenerates each group of results, shown to the user when a
# file is missing rather than making them work it out from a traceback.
REBUILD_COMMANDS = {
    "03": "python -m src.analysis.diagnostics",
    "04": "python -m src.analysis.ab_test",
    "05": "python -m src.analysis.power",
    "06": "python -m src.analysis.cuped",
    "07": "python -m src.analysis.heterogeneity",
    "08": "python -m src.models.uplift",
    "09": "python -m src.models.policy",
    "10": "python -m src.models.budget",
}


class MissingArtifact(FileNotFoundError):
    """A result file the dashboard needs has not been generated yet."""


def result_path(name: str) -> Path:
    """Resolve a short name to the file it is stored in."""
    if name not in RESULT_FILES:
        raise KeyError(f"Unknown result {name!r}. Available: {sorted(RESULT_FILES)}")
    return config.RESULTS_DIR / RESULT_FILES[name]


def rebuild_command(name: str) -> str:
    """The command that would produce this result."""
    return REBUILD_COMMANDS[RESULT_FILES[name][:2]]


def result(name: str) -> pd.DataFrame:
    """Read one generated result table.

    Raises `MissingArtifact` naming the command to run, rather than letting a
    bare `FileNotFoundError` surface in the browser.
    """
    path = result_path(name)
    if not path.exists():
        raise MissingArtifact(
            f"{path.name} has not been generated. Run: {rebuild_command(name)}"
        )
    return pd.read_csv(path)


def available() -> dict[str, bool]:
    """Which results exist on disk right now."""
    return {name: result_path(name).exists() for name in RESULT_FILES}


def missing() -> list[str]:
    """Results the dashboard would need but cannot find."""
    return [name for name, exists in available().items() if not exists]


def rebuild_plan() -> list[str]:
    """The distinct commands that would produce everything missing.

    Ordered as the pipeline runs, so following the list top to bottom works.
    """
    commands = {rebuild_command(name) for name in missing()}
    return [c for c in REBUILD_COMMANDS.values() if c in commands]


# ==========================================================================
# Warehouse
# ==========================================================================
# The warehouse names its arm column `arm` and its metrics by aggregation
# rather than by outcome, so the dashboard's outcome selector needs a mapping
# rather than string interpolation on the outcome name.
ARM_COLUMN = "arm"

ARM_METRIC_COLUMNS = {
    "visit": "visit_rate",
    "conversion": "conversion_rate",
    "spend": "mean_spend",
}


def arm_metrics() -> pd.DataFrame:
    """Per-arm outcome rates, straight from the DuckDB view."""
    return warehouse.table("v_arm_metrics")


def arm_metric_column(outcome: str) -> str:
    """The `v_arm_metrics` column holding this outcome."""
    if outcome not in ARM_METRIC_COLUMNS:
        raise KeyError(
            f"Unknown outcome {outcome!r}. Available: {sorted(ARM_METRIC_COLUMNS)}"
        )
    return ARM_METRIC_COLUMNS[outcome]


def funnel() -> pd.DataFrame:
    return warehouse.table("v_funnel")


def segment_metrics(dimension: str | None = None) -> pd.DataFrame:
    """Outcome rates by customer attribute, optionally filtered to one.

    Parameterised rather than interpolated: the dimension arrives from a
    dashboard control, and a select box today can become a text input
    tomorrow.
    """
    if dimension is None:
        return warehouse.table("v_segment_metrics")
    return warehouse.query(
        "SELECT * FROM v_segment_metrics WHERE dimension = ?", [dimension]
    )


def dimensions() -> list[str]:
    """The attributes available for slicing, in the view's own order."""
    frame = warehouse.query(
        "SELECT DISTINCT dimension FROM v_segment_metrics ORDER BY dimension"
    )
    return frame["dimension"].tolist()


def treatment_metrics(frame: pd.DataFrame) -> list[str]:
    """Metric columns describing the treated arm, excluding the control mirror.

    `v_segment_metrics` carries the control arm's own rates alongside each
    treated row so lift can be computed in SQL. Offering those as chart metrics
    would let a reader plot the control rate broken down by arm, which is the
    same number three times.
    """
    return [
        column
        for column in frame.columns
        if column.endswith(("_rate", "_lift")) and not column.startswith("control_")
    ]


# ==========================================================================
# The randomisation test, recovered from its saved counts
# ==========================================================================
def srm_verdict(
    srm_counts: pd.DataFrame, alpha: float = config.SRM_ALPHA
) -> dict[str, float | bool]:
    """Rebuild the sample ratio mismatch test from the saved arm sizes.

    Feature 3 writes `03_srm.csv` with the observed and expected counts but not
    the statistic they produced, so the test has to be reconstructed to be
    displayed. That is safe here for the same reason the profit model is: this
    is arithmetic on the saved numbers, not a re-run of the analysis. The
    chi-square goodness-of-fit statistic is a closed-form function of observed
    and expected counts, and the test suite pins the result against
    `diagnostics.srm_test` on the real data.
    """
    observed = srm_counts["observed"].to_numpy(dtype=float)
    expected = srm_counts["expected"].to_numpy(dtype=float)

    chi2 = float(((observed - expected) ** 2 / expected).sum())
    dof = len(observed) - 1
    p_value = float(stats.chi2.sf(chi2, dof))

    return {
        "chi2": chi2,
        "dof": dof,
        "p_value": p_value,
        "alpha": alpha,
        "passed": p_value >= alpha,
    }


# ==========================================================================
# The live profit model
# ==========================================================================
def live_economics(
    ab_results: pd.DataFrame,
    margin: float = config.GROSS_MARGIN,
    cost: float = config.COST_PER_EMAIL,
) -> pd.DataFrame:
    """Recompute campaign profitability at a chosen margin and send cost.

    Pure arithmetic on Feature 4's spend effect -- see the module docstring for
    why that is exact rather than an approximation. Takes the results frame as
    an argument instead of reading it, so the transform can be tested on
    constructed inputs.
    """
    if margin <= 0:
        raise ValueError(f"margin must be positive, got {margin}")

    spend = ab_results[ab_results["outcome"] == "spend"].copy()
    if spend.empty:
        raise ValueError("ab_results contains no spend rows")

    effect = spend["absolute_effect"]
    profit = margin * effect - cost

    return pd.DataFrame({
        "treatment_arm": spend["treatment_arm"].to_numpy(),
        "spend_effect": effect.to_numpy(),
        "spend_ci_low": spend["ci_low"].to_numpy(),
        "spend_ci_high": spend["ci_high"].to_numpy(),
        "profit_per_email": profit.to_numpy(),
        "profit_ci_low": (margin * spend["ci_low"] - cost).to_numpy(),
        "profit_ci_high": (margin * spend["ci_high"] - cost).to_numpy(),
        # Break-even is a property of the spend effect and the send cost. It
        # does not move when the assumed margin moves, which is exactly why it
        # is the number to quote in a discussion about the assumption.
        "break_even_spend": cost / margin,
        "break_even_margin": np.where(effect > 0, cost / effect, np.inf),
        "break_even_margin_high": np.where(
            spend["ci_low"] > 0, cost / spend["ci_low"], np.inf
        ),
        "margin": margin,
        "cost": cost,
    })


def verdict(economics_row: pd.Series) -> str:
    """A one-line reading of whether a campaign pays for itself.

    Three states, not two. "Pays for itself" and "loses money" both require the
    interval to sit clear of zero; anything else is genuinely undecided, and
    collapsing that into "not profitable" would misreport a wide interval as a
    negative finding.
    """
    if economics_row["profit_ci_low"] > 0:
        return "pays for itself"
    if economics_row["profit_ci_high"] < 0:
        return "loses money"
    return "cannot be determined"


def budget_projection(
    economics: pd.DataFrame,
    treatment_arm: str,
    n_customers: int,
    cost: float = config.COST_PER_EMAIL,
    n_points: int = 21,
) -> pd.DataFrame:
    """Expected profit from emailing a given number of customers.

    Linear in the number of sends, and exactly so: a policy that emails k
    customers chosen without reference to their attributes earns k times the
    average incremental profit per email. Feature 10's measured curve wanders
    around this line, and the gap between the two is estimation noise rather
    than a targeting effect -- worth showing side by side, because reading
    structure into that wander is the mistake the whole feature warns about.
    """
    row = economics[economics["treatment_arm"] == treatment_arm]
    if row.empty:
        raise ValueError(f"No economics row for arm {treatment_arm!r}")
    row = row.iloc[0]

    sends = np.linspace(0, n_customers, n_points).round().astype(int)
    return pd.DataFrame({
        "n_sends": sends,
        "budget": sends * cost,
        "net_gain": sends * row["profit_per_email"],
        "ci_low": sends * row["profit_ci_low"],
        "ci_high": sends * row["profit_ci_high"],
    })


# ==========================================================================
# Headline numbers
# ==========================================================================
def headline() -> dict[str, float | int | str]:
    """The handful of numbers the overview page leads with.

    Read from the generated results rather than recomputed, so the landing page
    of the dashboard and the README cannot drift apart.
    """
    ab = result("ab_tests")
    visit = ab[ab["outcome"] == "visit"].set_index("treatment_arm")
    policy = result("policy_values").set_index("policy")
    economics = result("economics").set_index("treatment_arm")

    # Each row is one arm against the shared control, so the experiment total
    # is the control arm once plus every treated arm. Adding a single row's
    # treated and control counts would report a two-arm comparison as the whole
    # experiment.
    customers = int(visit["n_treated"].sum() + visit["n_control"].iloc[0])

    return {
        "customers": customers,
        "mens_visit_lift": float(visit.loc[config.MENS_ARM, "absolute_effect"]),
        "womens_visit_lift": float(visit.loc[config.WOMENS_ARM, "absolute_effect"]),
        "significant_results": int(ab["significant"].sum()),
        "total_results": int(len(ab)),
        "best_policy": str(policy["value"].idxmax()),
        "best_policy_value": float(policy["value"].max()),
        "mens_profit_per_email": float(
            economics.loc[config.MENS_ARM, "profit_per_email"]
        ),
        "womens_profit_per_email": float(
            economics.loc[config.WOMENS_ARM, "profit_per_email"]
        ),
    }
