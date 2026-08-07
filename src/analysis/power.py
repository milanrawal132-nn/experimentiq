"""Power and minimum detectable effect.

Feature 4 found significant effects everywhere, which invites a lazy
conclusion: the experiment was well powered. That does not follow. Finding an
effect says the effect was large enough to find, not that the design was
capable of finding effects worth acting on.

The question worth asking retrospectively is the **minimum detectable effect**:
given the sample size and the variance actually observed, how large would an
effect have had to be for this experiment to detect it reliably? An observed
effect close to the MDE was detected, but only just, and would not replicate
dependably.

The question *not* worth asking is "what was the power to detect the effect we
observed?" -- see `observed_power` for why that number is circular.

Run as a script for the full power report:

    python -m src.analysis.power
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

from src import config
from src.data.load import load_processed, make_comparison_frame

logger = logging.getLogger(__name__)

# The largest absolute effect the solvers will search over. Proportions cannot
# exceed 1, and no plausible spend effect approaches this.
_MAX_EFFECT = 1e6


@dataclass(frozen=True)
class PowerResult:
    """Detectability of one outcome, given the design actually run."""

    outcome: str
    treatment_arm: str
    n_per_arm: int
    control_mean: float
    control_sd: float
    observed_effect: float
    observed_ci_low: float
    mde_absolute: float
    mde_relative: float
    effect_to_mde_ratio: float
    ci_low_to_mde_ratio: float
    alpha: float
    power_target: float

    @property
    def robust(self) -> bool:
        """Whether even the pessimistic end of the interval clears the MDE.

        This is the non-circular way to judge a design against its result. The
        point estimate is one draw; the confidence interval is the range of
        true effects the data supports. If the *lower* bound still exceeds the
        MDE, then the design would have found the effect even in the least
        favourable case consistent with the evidence.
        """
        return self.observed_ci_low > self.mde_absolute

    @property
    def verdict(self) -> str:
        if self.robust:
            return (
                f"Robust: even at the low end of its interval "
                f"({self.observed_ci_low:+.4f}) the effect clears the "
                f"{self.mde_absolute:.4f} detection threshold."
            )
        if self.observed_effect > self.mde_absolute:
            return (
                f"Detected, but not robustly: the point estimate clears the "
                f"threshold while the low end of its interval "
                f"({self.observed_ci_low:+.4f}) does not. A true effect at "
                "that end would often have been missed by this design."
            )
        return (
            "Below the detection threshold: an effect this size would usually "
            "be missed, so finding it here owed something to chance."
        )


# ==========================================================================
# Power, given an effect
# ==========================================================================
def _power_from_z(effect: float, standard_error: float, alpha: float) -> float:
    """Two-sided power for a normally distributed test statistic.

    Both rejection regions are included. The far tail is negligible for any
    effect worth discussing, but omitting it makes the function subtly wrong
    at effects near zero, where power should approach alpha rather than zero.
    """
    if standard_error <= 0:
        return 1.0 if effect != 0 else float(alpha)

    critical = stats.norm.ppf(1 - alpha / 2)
    z = abs(effect) / standard_error
    return float(stats.norm.sf(critical - z) + stats.norm.cdf(-critical - z))


def _se_proportions(p_control: float, effect: float, n_per_arm: int) -> float:
    """Standard error of a difference in proportions under the alternative.

    The treated arm's variance is evaluated at `p_control + effect`, not at
    `p_control`. A binomial variance depends on its own mean, so a treatment
    that moves the rate also moves the variance -- assuming otherwise
    understates the standard error for effects that raise a low base rate,
    which is exactly the situation for conversion here.
    """
    p_treated = np.clip(p_control + effect, 0.0, 1.0)
    return float(
        np.sqrt(
            p_control * (1 - p_control) / n_per_arm
            + p_treated * (1 - p_treated) / n_per_arm
        )
    )


def _se_means(sd: float, n_per_arm: int) -> float:
    """Standard error of a difference in means, assuming a common SD."""
    return float(np.sqrt(2 * sd**2 / n_per_arm))


def power_for_proportions(
    p_control: float,
    effect: float,
    n_per_arm: int,
    alpha: float = config.ALPHA,
) -> float:
    """Power to detect an absolute change of `effect` in a rate."""
    return _power_from_z(effect, _se_proportions(p_control, effect, n_per_arm), alpha)


def power_for_means(
    sd: float,
    effect: float,
    n_per_arm: int,
    alpha: float = config.ALPHA,
) -> float:
    """Power to detect a difference of `effect` in means."""
    return _power_from_z(effect, _se_means(sd, n_per_arm), alpha)


# ==========================================================================
# Minimum detectable effect
# ==========================================================================
def _solve_for_effect(power_fn, power_target: float, upper: float) -> float:
    """Invert a power function: find the effect that achieves `power_target`.

    Power is monotone increasing in effect size, so a bracketed root find is
    both safe and exact to machine precision. Closed forms exist for the
    simple cases, but not once the treated arm's variance is allowed to depend
    on the effect, and one solver covering every case is preferable to a
    closed form that quietly applies to only some of them.
    """
    if power_fn(upper) < power_target:
        return float("nan")
    return float(
        optimize.brentq(lambda e: power_fn(e) - power_target, 1e-12, upper)
    )


def mde_for_proportion(
    p_control: float,
    n_per_arm: int,
    alpha: float = config.ALPHA,
    power: float = config.POWER_TARGET,
) -> float:
    """Smallest absolute change in a rate detectable at the target power."""
    upper = min(1 - p_control, 1.0)
    return _solve_for_effect(
        lambda e: power_for_proportions(p_control, e, n_per_arm, alpha),
        power,
        upper,
    )


def mde_for_mean(
    sd: float,
    n_per_arm: int,
    alpha: float = config.ALPHA,
    power: float = config.POWER_TARGET,
) -> float:
    """Smallest difference in means detectable at the target power.

    The textbook closed form is

        MDE = (z_{1-alpha/2} + z_{power}) * sd * sqrt(2 / n)

    which is what this function returns to about seven significant figures.
    It is solved numerically anyway, because the closed form silently drops
    the far rejection tail -- the chance of rejecting in the wrong direction.
    That term is around 1e-6 and never changes a decision, but keeping it
    means `power_for_means(sd, mde_for_mean(...))` returns the target power
    exactly rather than nearly, so the two functions cannot drift apart.
    """
    return _solve_for_effect(
        lambda e: power_for_means(sd, e, n_per_arm, alpha),
        power,
        upper=max(sd * 100, 1.0),
    )


# ==========================================================================
# Required sample size
# ==========================================================================
def _solve_for_n(power_fn, power_target: float) -> float:
    """Invert a power function over sample size.

    Power is monotone increasing in n, so the same bracketed root find
    applies. n is treated as continuous during the solve and rounded by the
    caller if a whole number of customers is wanted.
    """
    lower, upper = 2.0, 1e12
    if power_fn(lower) >= power_target:
        return lower
    if power_fn(upper) < power_target:
        return float("inf")
    return float(optimize.brentq(lambda n: power_fn(n) - power_target, lower, upper))


def required_n_for_mean(
    sd: float,
    effect: float,
    alpha: float = config.ALPHA,
    power: float = config.POWER_TARGET,
) -> float:
    """Customers per arm needed to detect `effect` in a mean.

    Equivalent to the closed form

        n = 2 * sd^2 * (z_{1-alpha/2} + z_{power})^2 / effect^2

    Note the square: halving the effect you want to detect quadruples the
    sample size required. That single fact explains most of this feature's
    conclusions.
    """
    if effect == 0:
        return float("inf")
    return _solve_for_n(lambda n: power_for_means(sd, effect, n, alpha), power)


def required_n_for_proportion(
    p_control: float,
    effect: float,
    alpha: float = config.ALPHA,
    power: float = config.POWER_TARGET,
) -> float:
    """Customers per arm needed to detect an absolute change of `effect`."""
    if effect == 0:
        return float("inf")
    return _solve_for_n(
        lambda n: power_for_proportions(p_control, effect, n, alpha), power
    )


# ==========================================================================
# The observed-power fallacy
# ==========================================================================
def observed_power(
    observed_effect: float,
    standard_error: float,
    alpha: float = config.ALPHA,
) -> float:
    """Power computed from the effect the experiment happened to observe.

    Warning:
        This number is not informative and nothing in this project makes a
        decision with it. It is implemented so the notebook can demonstrate
        *why* it is uninformative.

        Observed power is a deterministic, monotone function of the p-value:
        a small p-value always yields high observed power, and a large one
        always yields low observed power. It therefore contains no information
        the p-value did not already carry. Reporting it as "the study had 12%
        power, so absence of evidence is inconclusive" is circular -- it
        restates the non-significant result as though it were independent
        corroboration of it.

        The non-circular retrospective question is the minimum detectable
        effect, which depends on the design rather than on the result.
    """
    return _power_from_z(observed_effect, standard_error, alpha)


# ==========================================================================
# Curves and orchestration
# ==========================================================================
def power_curve(
    outcome: str,
    control_mean: float,
    control_sd: float,
    n_per_arm: int,
    effects: np.ndarray,
    alpha: float = config.ALPHA,
) -> pd.DataFrame:
    """Power across a range of hypothetical effect sizes."""
    if outcome in config.BINARY_OUTCOMES:
        power = [power_for_proportions(control_mean, e, n_per_arm, alpha) for e in effects]
    else:
        power = [power_for_means(control_sd, e, n_per_arm, alpha) for e in effects]

    return pd.DataFrame(
        {
            "outcome": outcome,
            "effect": effects,
            "relative_effect": effects / control_mean,
            "power": power,
        }
    )


def run_power_analysis(
    df: pd.DataFrame | None = None,
    alpha: float = config.ALPHA,
    power_target: float = config.POWER_TARGET,
    save: bool = True,
) -> pd.DataFrame:
    """Minimum detectable effect for every outcome and treatment arm."""
    frame = load_processed() if df is None else df
    from src.analysis.ab_test import estimate_effect

    results = []
    for treatment_arm, control_arm in config.COMPARISONS:
        comparison = make_comparison_frame(frame, treatment_arm, control_arm)
        control = comparison.loc[comparison["treated"] == 0]
        n_per_arm = int(min((comparison["treated"] == 1).sum(), len(control)))

        for outcome in config.OUTCOMES:
            control_mean = float(control[outcome].mean())
            control_sd = float(control[outcome].std(ddof=1))
            estimate = estimate_effect(frame, outcome, treatment_arm, control_arm, alpha)

            if outcome in config.BINARY_OUTCOMES:
                mde = mde_for_proportion(control_mean, n_per_arm, alpha, power_target)
            else:
                mde = mde_for_mean(control_sd, n_per_arm, alpha, power_target)

            results.append(
                PowerResult(
                    outcome=outcome,
                    treatment_arm=treatment_arm,
                    n_per_arm=n_per_arm,
                    control_mean=control_mean,
                    control_sd=control_sd,
                    observed_effect=estimate.absolute_effect,
                    observed_ci_low=estimate.ci_low,
                    mde_absolute=mde,
                    mde_relative=mde / control_mean if control_mean else float("nan"),
                    effect_to_mde_ratio=(
                        estimate.absolute_effect / mde if mde else float("nan")
                    ),
                    ci_low_to_mde_ratio=(
                        estimate.ci_low / mde if mde else float("nan")
                    ),
                    alpha=alpha,
                    power_target=power_target,
                )
            )

    table = pd.DataFrame([asdict(result) for result in results])
    table["robust"] = [result.robust for result in results]
    table["verdict"] = [result.verdict for result in results]

    if save:
        config.ensure_dirs()
        table.to_csv(config.RESULTS_DIR / "05_power_analysis.csv", index=False)
        logger.info("Wrote power analysis to %s", config.RESULTS_DIR)

    return table


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    table = run_power_analysis()

    print("\n" + "=" * 96)
    print(f"MINIMUM DETECTABLE EFFECT  (alpha = {config.ALPHA}, "
          f"power = {config.POWER_TARGET:.0%})")
    print("=" * 96 + "\n")

    print(table[[
        "treatment_arm", "outcome", "mde_absolute", "mde_relative",
        "observed_effect", "observed_ci_low", "effect_to_mde_ratio",
        "ci_low_to_mde_ratio", "robust",
    ]].to_string(index=False, formatters={
        "mde_absolute": "{:.4f}".format,
        "mde_relative": "{:.1%}".format,
        "observed_effect": "{:+.4f}".format,
        "observed_ci_low": "{:+.4f}".format,
        "effect_to_mde_ratio": "{:.2f}x".format,
        "ci_low_to_mde_ratio": "{:.2f}x".format,
    }))

    print("\n" + "-" * 96)
    print("Assessment")
    print("-" * 96 + "\n")
    for _, row in table.iterrows():
        print(f"  {row['treatment_arm']} / {row['outcome']}")
        print(f"    {row['verdict']}\n")

    print("\n" + "-" * 96)
    print("Sample size required to detect a 10% relative improvement per arm")
    print("-" * 96 + "\n")

    frame = load_processed()
    control = frame[frame[config.TREATMENT_COL] == config.CONTROL_ARM]
    for outcome in config.OUTCOMES:
        mean = control[outcome].mean()
        target = 0.10 * mean
        if outcome in config.BINARY_OUTCOMES:
            n = required_n_for_proportion(mean, target)
        else:
            n = required_n_for_mean(control[outcome].std(ddof=1), target)
        print(f"  {outcome:<11} +{target:.4f} absolute  ->  {n:>12,.0f} per arm "
              f"({n / 21306:.1f}x the arm size actually used)")


if __name__ == "__main__":
    main()
