"""Treatment effect estimation for the three-arm email experiment.

Feature 3 established that the randomisation is sound, so a difference in
means between arms is an unbiased estimate of the causal effect of assignment.
This module estimates those differences, attaches uncertainty to them, and
corrects for the fact that six tests are being run at once.

Method is chosen by outcome type:

- `visit` and `conversion` are binary, so a two-proportion z-test applies.
- `spend` is continuous, heavily skewed and ~99% zeros, so Welch's t-test
  applies -- and its confidence interval is checked against a bootstrap, since
  the skew is severe enough that the normal approximation deserves verifying
  rather than assuming.

Run as a script for the full results table:

    python -m src.analysis.ab_test
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src import config
from src.data.load import load_processed, make_comparison_frame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EffectEstimate:
    """A single treatment-versus-control comparison for one outcome."""

    outcome: str
    treatment_arm: str
    control_arm: str
    n_treated: int
    n_control: int
    treated_mean: float
    control_mean: float
    absolute_effect: float
    relative_effect: float
    standard_error: float
    ci_low: float
    ci_high: float
    test_statistic: float
    p_value: float
    test_name: str
    alpha: float = config.ALPHA

    @property
    def significant(self) -> bool:
        """Uncorrected significance. Prefer the Holm-adjusted flag instead.

        Exposed only so the difference between the raw and corrected verdicts
        can be shown; nothing in the reporting path uses it to make a call.
        """
        return self.p_value < self.alpha


# ==========================================================================
# Tests
# ==========================================================================
def two_proportion_test(
    treated: np.ndarray | pd.Series,
    control: np.ndarray | pd.Series,
    alpha: float = config.ALPHA,
) -> dict[str, float | str]:
    """Two-proportion z-test with a confidence interval for the difference.

    The test statistic and the confidence interval deliberately use *different*
    standard errors, which is the part most often got wrong:

    - The **test** assumes the null hypothesis is true, meaning both arms share
      one underlying rate. The best estimate of that shared rate is the pooled
      proportion, so the null standard error is computed from it.
    - The **interval** makes no such assumption -- it describes a range of
      plausible non-zero differences, so pooling the arms' rates would be
      assuming the very thing being estimated. It uses the unpooled standard
      error instead.

    Using the pooled SE for both would produce an interval inconsistent with
    its own p-value near the significance boundary.
    """
    treated = np.asarray(treated, dtype=float)
    control = np.asarray(control, dtype=float)

    n_treated, n_control = len(treated), len(control)
    p_treated, p_control = treated.mean(), control.mean()
    difference = p_treated - p_control

    # Pooled SE, under the null of a single shared rate -- for the test.
    p_pooled = (treated.sum() + control.sum()) / (n_treated + n_control)
    se_pooled = np.sqrt(
        p_pooled * (1 - p_pooled) * (1 / n_treated + 1 / n_control)
    )
    z = difference / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * stats.norm.sf(abs(z))

    # Unpooled SE, assuming nothing -- for the interval.
    se_unpooled = np.sqrt(
        p_treated * (1 - p_treated) / n_treated
        + p_control * (1 - p_control) / n_control
    )
    critical = stats.norm.ppf(1 - alpha / 2)

    return {
        "absolute_effect": float(difference),
        "standard_error": float(se_unpooled),
        "ci_low": float(difference - critical * se_unpooled),
        "ci_high": float(difference + critical * se_unpooled),
        "test_statistic": float(z),
        "p_value": float(p_value),
        "test_name": "two-proportion z-test",
    }


def welch_t_test(
    treated: np.ndarray | pd.Series,
    control: np.ndarray | pd.Series,
    alpha: float = config.ALPHA,
) -> dict[str, float | str]:
    """Welch's t-test for a difference in means, with a confidence interval.

    Welch rather than Student because the arms need not share a variance --
    and for spend they demonstrably do not, since a treatment that induces
    more purchases raises the mean and the variance together. Welch costs
    almost nothing when variances are equal and is correct when they are not,
    so there is no reason to prefer the pooled-variance version.
    """
    treated = np.asarray(treated, dtype=float)
    control = np.asarray(control, dtype=float)

    n_treated, n_control = len(treated), len(control)
    mean_treated, mean_control = treated.mean(), control.mean()
    var_treated = treated.var(ddof=1)
    var_control = control.var(ddof=1)

    difference = mean_treated - mean_control
    se = np.sqrt(var_treated / n_treated + var_control / n_control)

    # Welch-Satterthwaite degrees of freedom.
    dof = (var_treated / n_treated + var_control / n_control) ** 2 / (
        (var_treated / n_treated) ** 2 / (n_treated - 1)
        + (var_control / n_control) ** 2 / (n_control - 1)
    )

    t_statistic = difference / se if se > 0 else 0.0
    p_value = 2 * stats.t.sf(abs(t_statistic), df=dof)
    critical = stats.t.ppf(1 - alpha / 2, df=dof)

    return {
        "absolute_effect": float(difference),
        "standard_error": float(se),
        "ci_low": float(difference - critical * se),
        "ci_high": float(difference + critical * se),
        "test_statistic": float(t_statistic),
        "p_value": float(p_value),
        "test_name": "Welch's t-test",
    }


# ==========================================================================
# Bootstrap
# ==========================================================================
def _bootstrap_means(
    values: np.ndarray, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Bootstrap distribution of the sample mean.

    For 0/1 data, resampling n observations with replacement and counting the
    successes *is* a draw from Binomial(n, p). Recognising that replaces an
    (n_boot x n) resampling matrix with n_boot scalar draws -- exact, not an
    approximation, and orders of magnitude cheaper.

    Continuous data has no such shortcut, so it is resampled directly, in
    chunks to bound peak memory.
    """
    n = len(values)
    unique = np.unique(values)
    if len(unique) <= 2 and np.isin(unique, (0.0, 1.0)).all():
        return rng.binomial(n, values.mean(), size=n_boot) / n

    means = np.empty(n_boot)
    chunk = max(1, min(n_boot, 2_000_000 // max(n, 1)))
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        indices = rng.integers(0, n, size=(size, n))
        means[start : start + size] = values[indices].mean(axis=1)
    return means


def bootstrap_effect(
    treated: np.ndarray | pd.Series,
    control: np.ndarray | pd.Series,
    n_boot: int = config.N_BOOTSTRAP,
    alpha: float = config.ALPHA,
    seed: int = config.RANDOM_SEED,
) -> dict[str, float]:
    """Percentile bootstrap intervals for the absolute and relative effect.

    Two reasons this exists alongside the analytic tests:

    1. **It checks the normal approximation.** Spend is ~99% zeros with a
       standard deviation ~14x its mean. The central limit theorem should still
       apply at n > 21,000, but "should" is worth verifying when the skew is
       this extreme -- and agreement between the two intervals is evidence, not
       decoration.
    2. **It gives the relative effect an interval.** A ratio of two random
       means has no clean closed-form standard error, so dividing the absolute
       CI by the control mean would understate the uncertainty by ignoring the
       denominator's own variability. Resampling handles it directly.
    """
    treated = np.asarray(treated, dtype=float)
    control = np.asarray(control, dtype=float)
    rng = np.random.default_rng(seed)

    treated_means = _bootstrap_means(treated, n_boot, rng)
    control_means = _bootstrap_means(control, n_boot, rng)

    absolute = treated_means - control_means
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(control_means != 0, absolute / control_means, np.nan)

    lower, upper = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "absolute_ci_low": float(np.percentile(absolute, lower)),
        "absolute_ci_high": float(np.percentile(absolute, upper)),
        "relative_ci_low": float(np.nanpercentile(relative, lower)),
        "relative_ci_high": float(np.nanpercentile(relative, upper)),
        "n_boot": n_boot,
    }


# ==========================================================================
# Orchestration
# ==========================================================================
def estimate_effect(
    df: pd.DataFrame,
    outcome: str,
    treatment_arm: str,
    control_arm: str = config.CONTROL_ARM,
    alpha: float = config.ALPHA,
) -> EffectEstimate:
    """Estimate one treatment effect, choosing the test by outcome type."""
    if outcome not in config.OUTCOMES:
        raise ValueError(f"Unknown outcome {outcome!r}; expected {config.OUTCOMES}")

    frame = make_comparison_frame(df, treatment_arm, control_arm)
    treated = frame.loc[frame["treated"] == 1, outcome]
    control = frame.loc[frame["treated"] == 0, outcome]

    test = (
        two_proportion_test(treated, control, alpha)
        if outcome in config.BINARY_OUTCOMES
        else welch_t_test(treated, control, alpha)
    )

    control_mean = float(control.mean())
    return EffectEstimate(
        outcome=outcome,
        treatment_arm=treatment_arm,
        control_arm=control_arm,
        n_treated=len(treated),
        n_control=len(control),
        treated_mean=float(treated.mean()),
        control_mean=control_mean,
        relative_effect=(
            test["absolute_effect"] / control_mean if control_mean else float("nan")
        ),
        alpha=alpha,
        **test,
    )


def run_ab_tests(
    df: pd.DataFrame | None = None,
    alpha: float = config.ALPHA,
    method: str = config.MULTIPLE_TESTING_METHOD,
    save: bool = True,
) -> pd.DataFrame:
    """Estimate every effect and correct for multiple comparisons.

    Six tests are run: three outcomes across two treatment arms. Testing each
    at alpha = 0.05 gives roughly a 26% chance of at least one false positive
    if no treatment did anything, so the family is corrected as a whole.

    Holm rather than Bonferroni: both control the family-wise error rate, but
    Holm is uniformly more powerful -- it never rejects less and sometimes
    rejects more, so choosing Bonferroni would be discarding power for nothing.
    """
    frame = load_processed() if df is None else df

    estimates = [
        estimate_effect(frame, outcome, treatment_arm, control_arm, alpha)
        for treatment_arm, control_arm in config.COMPARISONS
        for outcome in config.OUTCOMES
    ]

    results = pd.DataFrame([asdict(estimate) for estimate in estimates])

    rejected, adjusted, _, _ = multipletests(
        results["p_value"], alpha=alpha, method=method
    )
    results["p_value_adjusted"] = adjusted
    results["significant"] = rejected
    results["correction"] = method

    if save:
        config.ensure_dirs()
        results.to_csv(config.RESULTS_DIR / "04_ab_test_results.csv", index=False)
        logger.info("Wrote results to %s", config.RESULTS_DIR)

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = run_ab_tests()

    print("\n" + "=" * 88)
    print("TREATMENT EFFECTS  (Holm-corrected across 6 tests)")
    print("=" * 88 + "\n")

    display = results[[
        "treatment_arm", "outcome", "control_mean", "treated_mean",
        "absolute_effect", "ci_low", "ci_high", "relative_effect",
        "p_value", "p_value_adjusted", "significant",
    ]]
    print(display.to_string(index=False, formatters={
        "control_mean": "{:.4f}".format,
        "treated_mean": "{:.4f}".format,
        "absolute_effect": "{:+.4f}".format,
        "ci_low": "{:+.4f}".format,
        "ci_high": "{:+.4f}".format,
        "relative_effect": "{:+.1%}".format,
        "p_value": "{:.2e}".format,
        "p_value_adjusted": "{:.2e}".format,
    }))

    print("\n" + "-" * 88)
    print("Bootstrap check on spend (the skewed outcome)")
    print("-" * 88 + "\n")

    frame = load_processed()
    for treatment_arm, control_arm in config.COMPARISONS:
        comparison = make_comparison_frame(frame, treatment_arm, control_arm)
        boot = bootstrap_effect(
            comparison.loc[comparison["treated"] == 1, "spend"],
            comparison.loc[comparison["treated"] == 0, "spend"],
        )
        analytic = results[
            (results["treatment_arm"] == treatment_arm)
            & (results["outcome"] == "spend")
        ].iloc[0]

        print(f"{treatment_arm}")
        print(f"  Welch CI     : [{analytic['ci_low']:+.4f}, {analytic['ci_high']:+.4f}]")
        print(f"  Bootstrap CI : [{boot['absolute_ci_low']:+.4f}, "
              f"{boot['absolute_ci_high']:+.4f}]")
        print(f"  Relative CI  : [{boot['relative_ci_low']:+.1%}, "
              f"{boot['relative_ci_high']:+.1%}]")
        print()


if __name__ == "__main__":
    main()
