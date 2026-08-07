"""Heterogeneous treatment effects.

Everything so far reports averages. An average is compatible with a great deal
of disagreement underneath it: a campaign that lifts visits by 7.7 points on
average could be lifting one group by 15 points while actively driving another
away. Features 8 to 10 propose to target customers individually, which only
makes sense if that variation exists. This feature asks whether it does.

Two methodological commitments shape the analysis.

**Subgroups are pre-registered.** The list lives in `config.PREREGISTERED_SUBGROUPS`
with a stated rationale for each, fixed before any subgroup effect was
estimated. Splitting the data every possible way and reporting the strongest
split is how subgroup analysis earned its reputation; the defence is committing
in advance, in version control.

**Heterogeneity is tested by interaction, not by comparing subgroup p-values.**
"Significant among new customers, not significant among established ones" is
not evidence that the effect differs between them -- two estimates can fall on
opposite sides of a significance threshold while being statistically
indistinguishable from each other. The question requires testing the difference
directly.

Run as a script for the full analysis:

    python -m src.analysis.heterogeneity
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from src import config
from src.analysis.ab_test import estimate_effect
from src.data.load import make_comparison_frame, load_processed

logger = logging.getLogger(__name__)

# Minimum customers per arm within a subgroup level before its effect is
# reported. Below this the estimate is dominated by noise and invites exactly
# the over-reading this module is built to avoid.
MIN_CELL_SIZE = 500


@dataclass(frozen=True)
class InteractionResult:
    """Joint test of whether an effect differs across a subgroup's levels."""

    outcome: str
    treatment_arm: str
    subgroup: str
    preregistered: bool
    rationale: str
    f_statistic: float
    df_num: int
    df_denom: float
    p_value: float
    n: int

    @property
    def levels(self) -> int:
        return self.df_num + 1


def _label(frame: pd.DataFrame, subgroup: str) -> pd.Series:
    """Readable level labels, including for the 0/1 indicator columns."""
    readable = {
        "mens": {1: "Bought mens", 0: "No mens purchase"},
        "womens": {1: "Bought womens", 0: "No womens purchase"},
        "newbie": {1: "New customer", 0: "Established customer"},
    }
    if subgroup in readable:
        return frame[subgroup].map(readable[subgroup]).astype("object")
    return frame[subgroup].astype("object")


# ==========================================================================
# Effects within subgroup levels
# ==========================================================================
def subgroup_effects(
    df: pd.DataFrame,
    outcome: str,
    treatment_arm: str,
    subgroup: str,
    control_arm: str = config.CONTROL_ARM,
    alpha: float = config.ALPHA,
    min_cell_size: int = MIN_CELL_SIZE,
) -> pd.DataFrame:
    """Estimate the treatment effect separately within each subgroup level.

    These are descriptive. Each level's estimate is a valid causal effect for
    that subpopulation -- randomisation holds within any pre-treatment split --
    but comparing them across levels by eye is not a test, which is what
    `interaction_test` is for.
    """
    frame = make_comparison_frame(df, treatment_arm, control_arm)
    frame = frame.assign(_level=_label(frame, subgroup))

    records = []
    for level, subset in frame.groupby("_level", observed=True):
        n_treated = int((subset["treated"] == 1).sum())
        n_control = int((subset["treated"] == 0).sum())
        if min(n_treated, n_control) < min_cell_size:
            logger.info(
                "Skipping %s = %s: only %d/%d customers per arm",
                subgroup, level, n_treated, n_control,
            )
            continue

        estimate = estimate_effect(
            subset, outcome, treatment_arm, control_arm, alpha
        )
        records.append(
            {
                "outcome": outcome,
                "treatment_arm": treatment_arm,
                "subgroup": subgroup,
                "level": level,
                "n_treated": n_treated,
                "n_control": n_control,
                "control_mean": estimate.control_mean,
                "effect": estimate.absolute_effect,
                "ci_low": estimate.ci_low,
                "ci_high": estimate.ci_high,
                "relative_effect": estimate.relative_effect,
                "p_value": estimate.p_value,
            }
        )

    return pd.DataFrame(records)


# ==========================================================================
# The interaction test
# ==========================================================================
def interaction_test(
    df: pd.DataFrame,
    outcome: str,
    treatment_arm: str,
    subgroup: str,
    control_arm: str = config.CONTROL_ARM,
) -> InteractionResult:
    """Test whether the treatment effect differs across a subgroup's levels.

    Fits `outcome ~ treated * subgroup` and jointly tests every interaction
    coefficient against zero. That single test answers the actual question --
    does the effect vary -- rather than the question a stack of per-level
    p-values answers, which is whether each level's effect differs from zero.

    A linear model is used even for the binary outcomes. On the additive scale
    an interaction coefficient is a difference of risk differences, which is
    the quantity of interest and the one that carries over to the targeting
    decisions in Features 9 and 10. A logistic interaction is a ratio of odds
    ratios: a different quantity, non-collapsible, and capable of showing
    interaction on the multiplicative scale where none exists on the additive
    one. Heteroskedasticity-robust (HC3) errors handle the non-constant
    variance a linear probability model induces.
    """
    frame = make_comparison_frame(df, treatment_arm, control_arm)
    frame = frame.assign(_level=_label(frame, subgroup).astype(str))

    model = smf.ols(f"{outcome} ~ treated * C(_level)", data=frame).fit(
        cov_type="HC3"
    )

    interaction_terms = [
        name for name in model.params.index if "treated:" in name
    ]
    if not interaction_terms:
        raise ValueError(f"Subgroup {subgroup!r} has only one level")

    test = model.f_test(" = 0, ".join(interaction_terms) + " = 0")

    registered = subgroup in config.PREREGISTERED_SUBGROUPS
    return InteractionResult(
        outcome=outcome,
        treatment_arm=treatment_arm,
        subgroup=subgroup,
        preregistered=registered,
        rationale=(
            config.PREREGISTERED_SUBGROUPS.get(subgroup)
            or config.EXPLORATORY_SUBGROUPS.get(subgroup, "")
        ),
        f_statistic=float(np.squeeze(test.statistic)),
        df_num=int(test.df_num),
        df_denom=float(test.df_denom),
        p_value=float(np.squeeze(test.pvalue)),
        n=int(model.nobs),
    )


# ==========================================================================
# Orchestration
# ==========================================================================
def run_heterogeneity_analysis(
    df: pd.DataFrame | None = None,
    outcomes: list[str] | None = None,
    include_exploratory: bool = True,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run interaction tests and per-level effects for every subgroup.

    Interaction p-values are corrected with Benjamini-Hochberg rather than
    Holm. Subgroup analysis is a screen: its output is a shortlist of effects
    worth modelling in Feature 8, not a set of claims to act on directly.
    Controlling the false discovery rate is the right guarantee for a screen,
    and controlling family-wise error would discard genuine signal that the
    uplift models could have used.
    """
    frame = load_processed() if df is None else df
    outcomes = outcomes or config.OUTCOMES

    subgroups = dict(config.PREREGISTERED_SUBGROUPS)
    if include_exploratory:
        subgroups.update(config.EXPLORATORY_SUBGROUPS)

    interactions: list[InteractionResult] = []
    effects: list[pd.DataFrame] = []

    for treatment_arm, control_arm in config.COMPARISONS:
        for outcome in outcomes:
            for subgroup in subgroups:
                interactions.append(
                    interaction_test(
                        frame, outcome, treatment_arm, subgroup, control_arm
                    )
                )
                effects.append(
                    subgroup_effects(
                        frame, outcome, treatment_arm, subgroup, control_arm
                    )
                )

    interaction_table = pd.DataFrame([asdict(r) for r in interactions])
    interaction_table["levels"] = [r.levels for r in interactions]

    # Correct within the primary outcome's pre-registered family. Mixing the
    # exploratory and underpowered tests into the same correction would
    # penalise the pre-registered hypotheses for questions asked afterwards.
    primary = (
        interaction_table["preregistered"]
        & (interaction_table["outcome"] == config.HETEROGENEITY_PRIMARY_OUTCOME)
    )
    interaction_table["p_value_adjusted"] = np.nan
    interaction_table.loc[primary, "p_value_adjusted"] = multipletests(
        interaction_table.loc[primary, "p_value"], alpha=config.ALPHA, method="fdr_bh"
    )[1]
    interaction_table["significant"] = (
        interaction_table["p_value_adjusted"] < config.ALPHA
    )

    effect_table = pd.concat(effects, ignore_index=True)
    effect_table["harmful"] = effect_table["ci_high"] < 0

    if save:
        config.ensure_dirs()
        interaction_table.to_csv(
            config.RESULTS_DIR / "07_interaction_tests.csv", index=False
        )
        effect_table.to_csv(
            config.RESULTS_DIR / "07_subgroup_effects.csv", index=False
        )
        logger.info("Wrote heterogeneity analysis to %s", config.RESULTS_DIR)

    return {"interactions": interaction_table, "effects": effect_table}


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    results = run_heterogeneity_analysis()
    interactions, effects = results["interactions"], results["effects"]

    primary = interactions[
        (interactions["outcome"] == config.HETEROGENEITY_PRIMARY_OUTCOME)
        & interactions["preregistered"]
    ]

    print("\n" + "=" * 92)
    print(f"INTERACTION TESTS - pre-registered subgroups, "
          f"outcome = {config.HETEROGENEITY_PRIMARY_OUTCOME}")
    print("=" * 92 + "\n")
    print(primary[[
        "treatment_arm", "subgroup", "levels", "f_statistic",
        "p_value", "p_value_adjusted", "significant",
    ]].to_string(index=False, formatters={
        "f_statistic": "{:.3f}".format,
        "p_value": "{:.4f}".format,
        "p_value_adjusted": "{:.4f}".format,
    }))

    print("\n" + "-" * 92)
    print("Largest effect differences within a subgroup")
    print("-" * 92 + "\n")

    primary_effects = effects[
        effects["outcome"] == config.HETEROGENEITY_PRIMARY_OUTCOME
    ]
    spread = (
        primary_effects.groupby(["treatment_arm", "subgroup"])["effect"]
        .agg(["min", "max"])
        .assign(spread=lambda t: t["max"] - t["min"])
        .sort_values("spread", ascending=False)
    )
    print(spread.to_string(formatters={
        "min": "{:+.4f}".format,
        "max": "{:+.4f}".format,
        "spread": "{:.4f}".format,
    }))

    print("\n" + "-" * 92)
    print("Is any subgroup actively harmed?")
    print("-" * 92 + "\n")
    harmful = effects[effects["harmful"]]
    if harmful.empty:
        negative = effects[effects["effect"] < 0]
        print("  No subgroup has a confidence interval lying entirely below zero.")
        print(f"  {len(negative)} of {len(effects)} level estimates are negative at "
              "the point estimate, all with intervals spanning zero.")
    else:
        print(harmful[[
            "treatment_arm", "outcome", "subgroup", "level",
            "effect", "ci_low", "ci_high",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
