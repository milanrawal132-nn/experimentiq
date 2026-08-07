"""CUPED: variance reduction using pre-experiment covariates.

Feature 5 established the problem. Detecting a 10% improvement in spend would
need roughly 23x the customers this experiment had, and you cannot always buy
more customers. CUPED attacks the other side of the same inequality: instead of
raising the sample size, reduce the variance of the outcome.

The method subtracts off the part of the outcome that a pre-experiment
covariate already explains:

    Y_cuped = Y - theta * (X - mean(X)),   theta = Cov(Y, X) / Var(X)

Because `X` is measured before treatment, subtracting it cannot bias the
estimate -- the adjustment has mean zero within each arm, so the expected
treatment effect is unchanged. What changes is the noise around it.

The entire benefit is one number:

    variance reduction = rho^2

where rho is the correlation between the outcome and the covariate. That is not
a rule of thumb, it is an identity, and it means CUPED cannot be argued into
working. Either a well-correlated pre-period covariate exists or the technique
has nothing to offer.

Run as a script for the full comparison:

    python -m src.analysis.cuped
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from src import config
from src.analysis.ab_test import welch_t_test
from src.data.load import load_processed, make_comparison_frame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CupedResult:
    """One treatment effect, estimated with and without CUPED adjustment."""

    outcome: str
    treatment_arm: str
    covariate: str
    theta: float
    correlation: float
    predicted_variance_reduction: float
    total_variance_reduction: float
    realised_variance_reduction: float

    effect_raw: float
    ci_low_raw: float
    ci_high_raw: float
    se_raw: float
    p_value_raw: float

    effect_cuped: float
    ci_low_cuped: float
    ci_high_cuped: float
    se_cuped: float
    p_value_cuped: float

    @property
    def ci_width_raw(self) -> float:
        return self.ci_high_raw - self.ci_low_raw

    @property
    def ci_width_cuped(self) -> float:
        return self.ci_high_cuped - self.ci_low_cuped

    @property
    def ci_narrowing(self) -> float:
        """Fractional reduction in confidence interval width."""
        if self.ci_width_raw == 0:
            return 0.0
        return 1 - self.ci_width_cuped / self.ci_width_raw

    @property
    def equivalent_sample_multiplier(self) -> float:
        """Extra customers CUPED is worth, as a multiple of the arm size.

        Precision scales with sqrt(n), so cutting variance by a factor v is
        equivalent to multiplying the sample by 1 / (1 - v). A 25% variance
        reduction is worth a third more customers; a 50% reduction doubles the
        effective sample.
        """
        remaining = 1 - self.realised_variance_reduction
        return float("inf") if remaining <= 0 else 1 / remaining


# ==========================================================================
# The estimator
# ==========================================================================
def cuped_theta(outcome: np.ndarray, covariate: np.ndarray) -> float:
    """The variance-minimising coefficient, Cov(Y, X) / Var(X).

    This is the ordinary least squares slope of Y on X, which is not a
    coincidence: CUPED subtracts the linear projection of the outcome onto the
    covariate, leaving the residual.

    Estimated on the pooled sample rather than within arms. Pooling is valid
    because the covariate is pre-treatment and therefore independent of
    assignment, and it is preferable because a single theta estimated from
    twice the data is more stable -- and because estimating separate thetas per
    arm would let the adjustment itself differ by arm, which reintroduces
    exactly the bias the method is designed to avoid.
    """
    covariate_variance = np.var(covariate, ddof=1)
    if covariate_variance == 0:
        return 0.0
    return float(np.cov(outcome, covariate, ddof=1)[0, 1] / covariate_variance)


def apply_cuped(
    outcome: np.ndarray, covariate: np.ndarray, theta: float | None = None
) -> np.ndarray:
    """Return the CUPED-adjusted outcome.

    Centring the covariate is what keeps the estimator unbiased: the adjustment
    term has mean zero over the pooled sample, so it shifts individual values
    without moving the overall mean.
    """
    outcome = np.asarray(outcome, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    if theta is None:
        theta = cuped_theta(outcome, covariate)
    return outcome - theta * (covariate - covariate.mean())


def variance_reduction(raw: np.ndarray, adjusted: np.ndarray) -> float:
    """Fraction of total outcome variance removed by the adjustment.

    This is the figure usually quoted for CUPED, and it equals the squared
    correlation between outcome and covariate.
    """
    raw_variance = np.var(raw, ddof=1)
    if raw_variance == 0:
        return 0.0
    return float(1 - np.var(adjusted, ddof=1) / raw_variance)


def _pooled_within_arm_variance(values: np.ndarray, is_treated: np.ndarray) -> float:
    """Variance within arms, pooled across them."""
    treated, control = values[is_treated], values[~is_treated]
    n_t, n_c = len(treated), len(control)
    if n_t + n_c <= 2:
        return 0.0
    return float(
        ((n_t - 1) * np.var(treated, ddof=1) + (n_c - 1) * np.var(control, ddof=1))
        / (n_t + n_c - 2)
    )


def within_arm_variance_reduction(
    raw: np.ndarray, adjusted: np.ndarray, is_treated: np.ndarray
) -> float:
    """Fraction of *within-arm* variance removed by the adjustment.

    This, not the total-variance figure, is what determines precision. A
    standard error is built from variation within each arm; the between-arm
    variation created by the treatment effect belongs to the signal, and CUPED
    neither removes it nor should.

    The two definitions differ only by the treatment effect's contribution to
    total variance, which is negligible whenever the effect is small relative
    to the outcome's spread -- as it is throughout this dataset. It is
    reported separately anyway, because it is the quantity that predicts how
    much the confidence interval narrows, and quoting a number that *nearly*
    predicts the thing you care about is a habit worth not forming.
    """
    raw_variance = _pooled_within_arm_variance(raw, is_treated)
    if raw_variance == 0:
        return 0.0
    return float(
        1 - _pooled_within_arm_variance(adjusted, is_treated) / raw_variance
    )


# ==========================================================================
# Effect estimation with CUPED
# ==========================================================================
def cuped_effect(
    df: pd.DataFrame,
    outcome: str,
    treatment_arm: str,
    covariate: str = config.CUPED_COVARIATE,
    control_arm: str = config.CONTROL_ARM,
    alpha: float = config.ALPHA,
    covariate_values: pd.Series | None = None,
    covariate_name: str | None = None,
) -> CupedResult:
    """Estimate one treatment effect with and without CUPED adjustment.

    Args:
        covariate_values: Optional precomputed covariate, used by the CUPAC
            variant where the control variate is a model prediction rather
            than a column of the data.
    """
    frame = make_comparison_frame(df, treatment_arm, control_arm)

    y = frame[outcome].to_numpy(dtype=float)
    if covariate_values is not None:
        x = np.asarray(covariate_values, dtype=float)
        if len(x) != len(frame):
            raise ValueError(
                f"covariate_values has length {len(x)}, expected {len(frame)}"
            )
    else:
        x = frame[covariate].to_numpy(dtype=float)

    theta = cuped_theta(y, x)
    y_cuped = apply_cuped(y, x, theta)

    correlation = float(np.corrcoef(y, x)[0, 1]) if np.var(x, ddof=1) > 0 else 0.0
    is_treated = frame["treated"].to_numpy() == 1

    raw = welch_t_test(y[is_treated], y[~is_treated], alpha)
    adjusted = welch_t_test(y_cuped[is_treated], y_cuped[~is_treated], alpha)

    return CupedResult(
        outcome=outcome,
        treatment_arm=treatment_arm,
        covariate=covariate_name or covariate,
        theta=theta,
        correlation=correlation,
        predicted_variance_reduction=correlation**2,
        total_variance_reduction=variance_reduction(y, y_cuped),
        realised_variance_reduction=within_arm_variance_reduction(
            y, y_cuped, is_treated
        ),
        effect_raw=raw["absolute_effect"],
        ci_low_raw=raw["ci_low"],
        ci_high_raw=raw["ci_high"],
        se_raw=raw["standard_error"],
        p_value_raw=raw["p_value"],
        effect_cuped=adjusted["absolute_effect"],
        ci_low_cuped=adjusted["ci_low"],
        ci_high_cuped=adjusted["ci_high"],
        se_cuped=adjusted["standard_error"],
        p_value_cuped=adjusted["p_value"],
    )


# ==========================================================================
# ANCOVA -- the regression form of the same idea
# ==========================================================================
def ancova_effect(
    df: pd.DataFrame,
    outcome: str,
    treatment_arm: str,
    covariates: list[str] | None = None,
    control_arm: str = config.CONTROL_ARM,
    alpha: float = config.ALPHA,
) -> dict[str, float]:
    """Estimate the treatment effect by regression adjustment.

    Regressing the outcome on the treatment indicator plus pre-treatment
    covariates gives the same variance reduction as CUPED, and with a single
    covariate the two are asymptotically identical. Showing both is worthwhile
    because they are usually presented as different techniques when they are
    the same idea in two notations -- CUPED subtracts the projection onto X,
    OLS partials it out.

    The regression form generalises more naturally to several covariates,
    which is why it is used here for the multi-covariate comparison.
    """
    frame = make_comparison_frame(df, treatment_arm, control_arm)
    terms = covariates if covariates is not None else [config.CUPED_COVARIATE]

    formula = f"{outcome} ~ treated + " + " + ".join(terms)
    model = smf.ols(formula, data=frame).fit(cov_type="HC3")

    interval = model.conf_int(alpha=alpha).loc["treated"]
    return {
        "effect": float(model.params["treated"]),
        "se": float(model.bse["treated"]),
        "ci_low": float(interval[0]),
        "ci_high": float(interval[1]),
        "p_value": float(model.pvalues["treated"]),
        "r_squared": float(model.rsquared),
        "covariates": terms,
    }


# ==========================================================================
# CUPAC -- a learned control variate
# ==========================================================================
def cupac_covariate(
    df: pd.DataFrame,
    outcome: str,
    treatment_arm: str,
    control_arm: str = config.CONTROL_ARM,
    features: list[str] | None = None,
    n_splits: int = 5,
    seed: int = config.RANDOM_SEED,
) -> pd.Series:
    """Build a learned control variate from all pre-treatment covariates.

    CUPED uses one covariate. Its natural generalisation -- sometimes called
    CUPAC -- replaces that covariate with a model's prediction of the outcome
    from every pre-treatment feature, which can capture more of the outcome's
    variance than any single column.

    Two properties keep it valid:

    1. **The model never sees the treatment indicator.** It predicts the
       outcome from pre-treatment features alone, so its prediction is itself
       a pre-treatment quantity and independent of assignment.
    2. **Predictions are out-of-fold.** A model evaluated on rows it was
       trained on has already seen their outcomes, and the resulting prediction
       would correlate with the noise in those specific outcomes rather than
       with their predictable part. Cross-fitting removes that dependence, and
       without it the variance reduction is overstated.
    """
    frame = make_comparison_frame(df, treatment_arm, control_arm)
    columns = features if features is not None else config.PRE_TREATMENT_COVARIATES

    design = pd.get_dummies(frame[columns], drop_first=False).astype(float)
    y = frame[outcome].to_numpy(dtype=float)
    predictions = np.zeros(len(frame))

    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in folds.split(design):
        model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=4, learning_rate=0.08, random_state=seed
        )
        model.fit(design.iloc[train_index], y[train_index])
        predictions[test_index] = model.predict(design.iloc[test_index])

    return pd.Series(predictions, index=frame.index, name=f"cupac_{outcome}")


# ==========================================================================
# Orchestration
# ==========================================================================
def run_cuped_analysis(
    df: pd.DataFrame | None = None,
    save: bool = True,
    include_cupac: bool = True,
) -> pd.DataFrame:
    """Compare raw, CUPED and CUPAC estimates for every outcome and arm."""
    frame = load_processed() if df is None else df

    results: list[CupedResult] = []
    for treatment_arm, control_arm in config.COMPARISONS:
        for outcome in config.OUTCOMES:
            results.append(
                cuped_effect(
                    frame, outcome, treatment_arm,
                    covariate=config.CUPED_COVARIATE, control_arm=control_arm,
                )
            )
            if include_cupac:
                learned = cupac_covariate(frame, outcome, treatment_arm, control_arm)
                results.append(
                    cuped_effect(
                        frame, outcome, treatment_arm, control_arm=control_arm,
                        covariate_values=learned,
                        covariate_name="CUPAC (all covariates)",
                    )
                )

    table = pd.DataFrame([asdict(result) for result in results])
    table["ci_width_raw"] = [r.ci_width_raw for r in results]
    table["ci_width_cuped"] = [r.ci_width_cuped for r in results]
    table["ci_narrowing"] = [r.ci_narrowing for r in results]
    table["equivalent_sample_multiplier"] = [
        r.equivalent_sample_multiplier for r in results
    ]

    if save:
        config.ensure_dirs()
        table.to_csv(config.RESULTS_DIR / "06_cuped.csv", index=False)
        logger.info("Wrote CUPED analysis to %s", config.RESULTS_DIR)

    return table


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    table = run_cuped_analysis()

    print("\n" + "=" * 100)
    print("CUPED VARIANCE REDUCTION")
    print("=" * 100 + "\n")

    print(table[[
        "treatment_arm", "outcome", "covariate", "correlation",
        "realised_variance_reduction", "ci_narrowing",
        "equivalent_sample_multiplier",
    ]].to_string(index=False, formatters={
        "correlation": "{:+.4f}".format,
        "realised_variance_reduction": "{:.3%}".format,
        "ci_narrowing": "{:.3%}".format,
        "equivalent_sample_multiplier": "{:.4f}x".format,
    }))

    print("\n" + "-" * 100)
    print("Effect estimates are preserved, as they must be")
    print("-" * 100 + "\n")

    single = table[table["covariate"] == config.CUPED_COVARIATE]
    print(single[[
        "treatment_arm", "outcome", "effect_raw", "effect_cuped",
        "p_value_raw", "p_value_cuped",
    ]].to_string(index=False, formatters={
        "effect_raw": "{:+.5f}".format,
        "effect_cuped": "{:+.5f}".format,
        "p_value_raw": "{:.3e}".format,
        "p_value_cuped": "{:.3e}".format,
    }))

    print("\n" + "-" * 100)
    print("What CUPED would deliver at various covariate correlations")
    print("-" * 100 + "\n")
    print(f"{'correlation':>12}  {'variance cut':>13}  {'equivalent sample':>18}")
    for rho in [0.02, 0.10, 0.30, 0.50, 0.70, 0.90]:
        reduction = rho**2
        print(f"{rho:>12.2f}  {reduction:>12.1%}  {1 / (1 - reduction):>17.2f}x")


if __name__ == "__main__":
    main()
