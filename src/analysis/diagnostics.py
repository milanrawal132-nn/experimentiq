"""Randomisation diagnostics.

Before any treatment effect is believed, the randomisation itself has to be
checked. Three questions, in increasing order of strength:

1. **Sample ratio mismatch.** Did each arm receive the number of customers the
   design intended? A mismatch means the assignment or logging mechanism is
   broken, and no amount of careful downstream statistics repairs that.
2. **Covariate balance.** Are the arms comparable on attributes measured
   *before* the send? Randomisation guarantees this in expectation, not in any
   single draw.
3. **Omnibus balance.** Can treatment assignment be predicted from the
   pre-treatment covariates *jointly*? This catches subtle multivariate
   imbalance that per-covariate checks miss.

Run as a script to produce the full diagnostic report:

    python -m src.analysis.diagnostics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from src import config
from src.data.load import load_processed, make_comparison_frame

logger = logging.getLogger(__name__)


# ==========================================================================
# 1. Sample ratio mismatch
# ==========================================================================
@dataclass(frozen=True)
class SRMResult:
    """Outcome of a chi-square goodness-of-fit test on the arm sizes."""

    counts: pd.Series
    expected: pd.Series
    chi2: float
    dof: int
    p_value: float
    alpha: float
    passed: bool

    @property
    def verdict(self) -> str:
        if self.passed:
            return (
                f"No sample ratio mismatch (p = {self.p_value:.3f}). The arm "
                "sizes are consistent with the intended split."
            )
        return (
            f"SAMPLE RATIO MISMATCH (p = {self.p_value:.2e}). The assignment "
            "or logging mechanism is suspect; do not interpret the results."
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "observed": self.counts,
                "expected": self.expected,
                "difference": self.counts - self.expected,
                "share": self.counts / self.counts.sum(),
            }
        )


def srm_test(
    df: pd.DataFrame,
    expected_shares: dict[str, float] | None = None,
    alpha: float = config.SRM_ALPHA,
) -> SRMResult:
    """Test arm sizes against the intended allocation.

    Args:
        df: The experiment data.
        expected_shares: Intended share per arm. Defaults to an equal split
            across all arms, which is what this experiment was designed for.
        alpha: Significance level. Defaults to the deliberately strict
            `config.SRM_ALPHA` rather than the usual 0.05 -- see the note there.

    Note:
        This is a goodness-of-fit test over all three arms at once, not three
        pairwise tests. A single omnibus test is the right shape: the question
        is whether the observed allocation as a whole departs from the design.
    """
    counts = df[config.TREATMENT_COL].value_counts().sort_index()

    if expected_shares is None:
        shares = pd.Series(1 / len(counts), index=counts.index)
    else:
        shares = pd.Series(expected_shares).reindex(counts.index)
        if shares.isna().any():
            raise ValueError(
                f"expected_shares is missing arms: "
                f"{list(shares[shares.isna()].index)}"
            )
        if not np.isclose(shares.sum(), 1.0):
            raise ValueError(f"expected_shares must sum to 1, got {shares.sum()}")

    expected = shares * counts.sum()
    chi2, p_value = stats.chisquare(f_obs=counts.to_numpy(), f_exp=expected.to_numpy())

    return SRMResult(
        counts=counts,
        expected=expected,
        chi2=float(chi2),
        dof=len(counts) - 1,
        p_value=float(p_value),
        alpha=alpha,
        passed=bool(p_value >= alpha),
    )


# ==========================================================================
# 2. Per-covariate balance
# ==========================================================================
def _balance_variables(df: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    """Flatten the covariates into comparable numeric series.

    Categorical covariates are expanded into one 0/1 indicator per level. Once
    everything is numeric, a single standardised-mean-difference formula
    applies to all of them -- for a 0/1 indicator the sample variance is just
    p(1-p), so the continuous formula reduces to the proportion formula rather
    than needing a separate case.

    `recency_bucket` is deliberately excluded: it is a deterministic function
    of `recency`, so including both would double-count the same imbalance.
    """
    variables: list[tuple[str, str, pd.Series]] = []

    for column in config.NUMERIC_COVARIATES:
        variables.append((column, "", df[column].astype(float)))

    for column in config.CATEGORICAL_COVARIATES:
        levels = (
            df[column].cat.categories
            if isinstance(df[column].dtype, pd.CategoricalDtype)
            else sorted(df[column].unique())
        )
        for level in levels:
            variables.append(
                (column, str(level), (df[column] == level).astype(float))
            )

    return variables


def standardised_mean_difference(
    treated: np.ndarray | pd.Series, control: np.ndarray | pd.Series
) -> float:
    """Cohen's d using the pooled standard deviation of the two arms.

    The point of standardising is that the result is unitless, so `history`
    measured in dollars and `newbie` measured as a 0/1 flag land on the same
    scale and can be read off one chart against one threshold.

    The zero-variance case splits in two, and conflating them would be a real
    error: if both arms are constant at the *same* value the covariate is
    perfectly balanced (0.0), but if they are constant at *different* values
    the arms are perfectly separated, which is maximal imbalance, not none.
    """
    treated = np.asarray(treated, dtype=float)
    control = np.asarray(control, dtype=float)

    var_treated = treated.var(ddof=1)
    var_control = control.var(ddof=1)
    pooled_sd = np.sqrt((var_treated + var_control) / 2)
    difference = treated.mean() - control.mean()

    if pooled_sd == 0:
        if difference == 0:
            return 0.0
        return float(np.inf if difference > 0 else -np.inf)
    return float(difference / pooled_sd)


def covariate_balance(
    df: pd.DataFrame,
    treatment_arm: str,
    control_arm: str = config.CONTROL_ARM,
) -> pd.DataFrame:
    """Compare pre-treatment covariates between one treatment arm and control.

    Returns one row per covariate (per level, for categoricals) with the two
    arm means, the standardised mean difference, and a Welch t-test p-value.

    Both columns are reported because they answer different questions, and at
    this sample size they disagree in an instructive way. The SMD measures
    *how large* an imbalance is; the p-value measures *how surely it is not
    zero*. With ~21,000 customers per arm, a trivially small imbalance can be
    statistically significant, so the SMD against a fixed threshold is the
    decision-relevant number and the p-value is context.
    """
    frame = make_comparison_frame(df, treatment_arm, control_arm)
    is_treated = frame["treated"] == 1

    records = []
    for variable, level, series in _balance_variables(frame):
        treated_values = series[is_treated]
        control_values = series[~is_treated]

        smd = standardised_mean_difference(treated_values, control_values)
        # Welch's t-test: does not assume equal variances between arms, which
        # matters for indicator variables whose variance depends on the mean.
        _, p_value = stats.ttest_ind(
            treated_values, control_values, equal_var=False
        )

        records.append(
            {
                "covariate": variable,
                "level": level,
                "treatment_arm": treatment_arm,
                "treated_mean": treated_values.mean(),
                "control_mean": control_values.mean(),
                "difference": treated_values.mean() - control_values.mean(),
                "smd": smd,
                "abs_smd": abs(smd),
                "p_value": float(p_value),
                "balanced": abs(smd) < config.SMD_THRESHOLD,
            }
        )

    return pd.DataFrame(records)


# ==========================================================================
# 3. Omnibus balance
# ==========================================================================
@dataclass(frozen=True)
class OmnibusResult:
    """Likelihood-ratio test of whether covariates jointly predict assignment."""

    treatment_arm: str
    llr_statistic: float
    dof: int
    p_value: float
    pseudo_r2: float
    n: int
    alpha: float = config.ALPHA
    separated: bool = False
    covariates: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (not self.separated) and self.p_value >= self.alpha

    @property
    def verdict(self) -> str:
        if self.separated:
            return (
                f"PERFECT SEPARATION for {self.treatment_arm}: the covariates "
                "predict assignment exactly. This is not a randomised "
                "experiment -- treatment was determined by a pre-treatment "
                "attribute."
            )
        if self.passed:
            return (
                f"Covariates do not jointly predict assignment to "
                f"{self.treatment_arm} (p = {self.p_value:.3f}, "
                f"pseudo-R2 = {self.pseudo_r2:.5f}). Consistent with valid "
                "randomisation."
            )
        return (
            f"Covariates DO jointly predict assignment to "
            f"{self.treatment_arm} (p = {self.p_value:.4f}). Investigate the "
            "assignment mechanism before interpreting effects."
        )


def omnibus_balance_test(
    df: pd.DataFrame,
    treatment_arm: str,
    control_arm: str = config.CONTROL_ARM,
) -> OmnibusResult:
    """Test whether the covariates *jointly* predict treatment assignment.

    Fits a logistic regression of the treatment indicator on every
    pre-treatment covariate and compares it against an intercept-only model by
    likelihood-ratio test.

    This is strictly stronger than checking covariates one at a time. Several
    individually-negligible imbalances can combine into a real one, and a
    per-covariate scan would miss it. Under valid randomisation the covariates
    carry no information about assignment, so the model should be no better
    than the intercept and the p-value should be uniform on [0, 1].

    The pseudo-R-squared is worth more than the p-value here: it says how much
    of the assignment is explained at all. A value near zero means the
    covariates are essentially uninformative about who was treated, which is
    what randomisation is supposed to produce.
    """
    frame = make_comparison_frame(df, treatment_arm, control_arm)

    terms = list(config.NUMERIC_COVARIATES) + [
        f"C({column})" for column in config.CATEGORICAL_COVARIATES
    ]
    formula = "treated ~ " + " + ".join(terms)

    try:
        model = smf.logit(formula, data=frame).fit(disp=0)
    except (PerfectSeparationError, np.linalg.LinAlgError):
        # The fit fails when a covariate predicts assignment exactly. That is
        # not an error to propagate -- it is the most extreme possible answer
        # to the question being asked, so it is reported as such rather than
        # surfacing as an opaque LinAlgError from deep inside the solver.
        logger.warning(
            "Perfect separation fitting assignment to %s; covariates predict "
            "treatment exactly.",
            treatment_arm,
        )
        return OmnibusResult(
            treatment_arm=treatment_arm,
            llr_statistic=float("inf"),
            dof=len(terms),
            p_value=0.0,
            pseudo_r2=1.0,
            n=len(frame),
            separated=True,
            covariates=terms,
        )

    return OmnibusResult(
        treatment_arm=treatment_arm,
        llr_statistic=float(model.llr),
        dof=int(model.df_model),
        p_value=float(model.llr_pvalue),
        pseudo_r2=float(model.prsquared),
        n=int(model.nobs),
        covariates=terms,
    )


# ==========================================================================
# Orchestration
# ==========================================================================
def run_diagnostics(
    df: pd.DataFrame | None = None, save: bool = True
) -> dict[str, object]:
    """Run every diagnostic and optionally write the tables to reports/."""
    frame = load_processed() if df is None else df

    srm = srm_test(frame)

    balance = pd.concat(
        [
            covariate_balance(frame, treatment_arm, control_arm)
            for treatment_arm, control_arm in config.COMPARISONS
        ],
        ignore_index=True,
    )

    omnibus = [
        omnibus_balance_test(frame, treatment_arm, control_arm)
        for treatment_arm, control_arm in config.COMPARISONS
    ]
    omnibus_frame = pd.DataFrame(
        [
            {
                "treatment_arm": result.treatment_arm,
                "n": result.n,
                "llr_statistic": result.llr_statistic,
                "dof": result.dof,
                "p_value": result.p_value,
                "pseudo_r2": result.pseudo_r2,
                "passed": result.passed,
            }
            for result in omnibus
        ]
    )

    if save:
        config.ensure_dirs()
        srm.to_frame().to_csv(config.RESULTS_DIR / "03_srm.csv")
        balance.to_csv(config.RESULTS_DIR / "03_covariate_balance.csv", index=False)
        omnibus_frame.to_csv(config.RESULTS_DIR / "03_omnibus_balance.csv", index=False)
        logger.info("Wrote diagnostic tables to %s", config.RESULTS_DIR)

    return {"srm": srm, "balance": balance, "omnibus": omnibus_frame}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = run_diagnostics()

    srm: SRMResult = results["srm"]
    balance: pd.DataFrame = results["balance"]
    omnibus: pd.DataFrame = results["omnibus"]

    print("\n" + "=" * 72)
    print("1. SAMPLE RATIO MISMATCH")
    print("=" * 72)
    print(srm.to_frame().to_string(formatters={
        "observed": "{:,.0f}".format,
        "expected": "{:,.1f}".format,
        "difference": "{:+,.1f}".format,
        "share": "{:.4%}".format,
    }))
    print(f"\nchi2 = {srm.chi2:.4f}, dof = {srm.dof}, p = {srm.p_value:.4f} "
          f"(alpha = {srm.alpha})")
    print(srm.verdict)

    print("\n" + "=" * 72)
    print("2. COVARIATE BALANCE")
    print("=" * 72)
    worst = balance.nlargest(8, "abs_smd")[
        ["treatment_arm", "covariate", "level", "treated_mean",
         "control_mean", "smd", "p_value"]
    ]
    print("Eight largest imbalances by |SMD|:\n")
    print(worst.to_string(index=False, formatters={
        "treated_mean": "{:.4f}".format,
        "control_mean": "{:.4f}".format,
        "smd": "{:+.4f}".format,
        "p_value": "{:.3f}".format,
    }))
    print(f"\nLargest |SMD| overall: {balance['abs_smd'].max():.4f} "
          f"(threshold {config.SMD_THRESHOLD})")
    print(f"Covariates above threshold: {(~balance['balanced']).sum()} "
          f"of {len(balance)}")

    significant = (balance["p_value"] < config.ALPHA).sum()
    print(f"\nBalance tests with p < {config.ALPHA}: {significant} of {len(balance)} "
          f"(expected by chance alone: ~{config.ALPHA * len(balance):.1f})")

    print("\n" + "=" * 72)
    print("3. OMNIBUS BALANCE")
    print("=" * 72)
    print(omnibus.to_string(index=False, formatters={
        "n": "{:,}".format,
        "llr_statistic": "{:.3f}".format,
        "p_value": "{:.4f}".format,
        "pseudo_r2": "{:.6f}".format,
    }))


if __name__ == "__main__":
    main()
