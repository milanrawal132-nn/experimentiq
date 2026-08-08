"""Uplift modelling: individual-level treatment effects.

Feature 7 found the treatment effect varies by customer, but only in
pre-defined subgroups. Uplift models estimate an effect for each *individual*
from the combination of their attributes, which is what a targeting policy
needs.

The difficulty is that no individual treatment effect is ever observed. A
customer is either emailed or not; the other outcome does not exist. So an
uplift model cannot be scored like a classifier -- there is no per-row label
to compare against. Evaluation instead works on groups: rank customers by
predicted uplift, then check whether the treated-minus-control gap really is
larger among the top-ranked than the bottom-ranked. That is what the Qini
curve measures.

Two consequences shape this module:

**Everything is out-of-fold.** A model scored on rows it trained on will rank
those rows using their own outcomes, and the Qini will look excellent for a
model with no real signal.

**A Qini score means nothing without a null.** Qini is high-variance, and a
model that has learned nothing still produces a non-zero score more often than
not. Every model here is compared against the distribution of Qini scores from
randomly shuffled rankings on the same data.

Feature 7 supplies a built-in check on all of this: the Womens campaign varies
strongly by purchase history, the Mens campaign barely varies at all. A model
finding real signal on Womens and little on Mens is behaving correctly. A model
finding strong signal on *both* is overfitting.

Run as a script for the full comparison:

    python -m src.models.uplift
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklift.metrics import qini_auc_score, uplift_at_k, uplift_auc_score
from sklift.models import ClassTransformation, SoloModel, TwoModels

from src import config
from src.data.load import load_processed, make_comparison_frame

logger = logging.getLogger(__name__)

N_FOLDS = 5
N_NULL_DRAWS = 500
TOP_K_FRACTIONS = (0.1, 0.2, 0.3)


@contextlib.contextmanager
def _quiet_sklift():
    """Silence a deprecation warning raised inside scikit-uplift's metrics.

    sklift 0.5.1 calls `sklearn.utils.extmath.stable_cumsum`, removed in
    scikit-learn 1.10. It fires once per metric call and would otherwise bury
    the output. Scoped to these calls rather than filtered globally, so a
    warning from our own code still surfaces.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*stable_cumsum.*", category=FutureWarning
        )
        yield


@dataclass(frozen=True)
class UpliftResult:
    """Cross-validated performance of one uplift learner on one comparison."""

    treatment_arm: str
    outcome: str
    learner: str
    n: int

    qini: float
    qini_null_mean: float
    qini_null_std: float
    qini_z_score: float
    qini_percentile: float
    p_value: float
    p_value_normal: float

    uplift_auc: float
    uplift_at_k: dict[str, float] = field(default_factory=dict)

    @property
    def beats_random(self) -> bool:
        """Whether the Qini exceeds the 95th percentile of the null.

        A one-sided test against the distribution of Qini scores produced by
        random rankings on this exact data. It is the minimum bar: a model that
        cannot clear it has not demonstrated any ability to rank customers.
        """
        return self.qini_percentile >= 95.0

    def p_value_normal_adjusted(self, n_learners: int) -> float:
        """Sidak-adjusted parametric p. See `p_value_adjusted`."""
        return float(1 - (1 - self.p_value_normal) ** n_learners)

    def p_value_adjusted(self, n_learners: int) -> float:
        """Sidak-adjusted p, for having picked the best of several learners.

        Fitting five learners and reporting the winner is a search, and the
        winner's p-value is the maximum of five draws rather than one. Without
        the adjustment, roughly one arm in four would appear to show signal
        from noise alone.

        Sidak assumes independence, which overstates the correction here since
        the learners share data and are highly correlated -- so the adjusted
        value is conservative rather than exact.
        """
        return float(1 - (1 - self.p_value) ** n_learners)


# ==========================================================================
# Features
# ==========================================================================
def build_design_matrix(
    frame: pd.DataFrame, features: list[str] | None = None
) -> pd.DataFrame:
    """One-hot encode the pre-treatment covariates.

    Only pre-treatment attributes are used. Including anything measured after
    the send would let the model peek at the outcome it is meant to predict
    the effect on.

    Args:
        features: Columns to use. Defaults to every pre-treatment covariate;
            overridden by the tests, which build synthetic experiments with
            their own feature names.
    """
    columns = features if features is not None else config.PRE_TREATMENT_COVARIATES
    return pd.get_dummies(frame[columns], drop_first=False).astype(float)


# ==========================================================================
# Learners
# ==========================================================================
def _boosted_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=150, max_depth=4, learning_rate=0.08, random_state=seed
    )


def make_learner(name: str, seed: int = config.RANDOM_SEED):
    """Construct one uplift meta-learner by name.

    The five differ in how they turn ordinary supervised models into an effect
    estimate:

    - **S-learner** fits one model on everyone with treatment as a feature,
      then differences its prediction with treatment on and off. Simple, but
      a boosted tree can ignore the treatment feature entirely, in which case
      predicted uplift collapses toward zero.
    - **T-learner** fits separate models on the treated and control arms and
      differences them. No risk of ignoring treatment, but each model sees
      half the data and their independent errors both land in the difference.
    - **X-learner (ddr_control)** imputes the effect for control customers
      using the treated model, then fits a model to that imputed effect.
      Designed for imbalanced arms; here the arms are even, so it mostly
      serves as a check that the ranking is not an artifact of one approach.
    - **Class transformation** relabels the outcome so that ordinary
      classification recovers the uplift ordering. Elegant, and it needs
      roughly balanced arms -- which this experiment has.
    - **Logistic T-learner** is the deliberately simple baseline. With three
      of the strongest signals being binary purchase-history flags, a linear
      model may lose nothing to the boosted version.
    """
    learners = {
        "s_learner": lambda: SoloModel(
            estimator=_boosted_classifier(seed), method="dummy"
        ),
        "t_learner": lambda: TwoModels(
            estimator_trmnt=_boosted_classifier(seed),
            estimator_ctrl=_boosted_classifier(seed),
            method="vanilla",
        ),
        "x_learner": lambda: TwoModels(
            estimator_trmnt=_boosted_classifier(seed),
            estimator_ctrl=_boosted_classifier(seed),
            method="ddr_control",
        ),
        "class_transform": lambda: ClassTransformation(
            estimator=_boosted_classifier(seed)
        ),
        "logistic_t_learner": lambda: TwoModels(
            estimator_trmnt=make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
            ),
            estimator_ctrl=make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
            ),
            method="vanilla",
        ),
    }
    if name not in learners:
        raise ValueError(f"Unknown learner {name!r}; expected {sorted(learners)}")
    return learners[name]()


LEARNERS = [
    "s_learner",
    "t_learner",
    "x_learner",
    "class_transform",
    "logistic_t_learner",
]


# ==========================================================================
# Out-of-fold prediction
# ==========================================================================
def out_of_fold_uplift(
    design: pd.DataFrame,
    outcome: np.ndarray,
    treatment: np.ndarray,
    learner: str,
    n_folds: int = N_FOLDS,
    seed: int = config.RANDOM_SEED,
) -> np.ndarray:
    """Predicted uplift for every customer, from a model that never saw them.

    Folds are stratified on the treatment-by-outcome combination so every fold
    keeps both arms and both outcome classes in proportion. With a conversion
    rate under 1%, an unstratified split can leave a fold with too few positive
    treated cases to fit at all.
    """
    predictions = np.zeros(len(design))
    strata = treatment.astype(str) + "_" + outcome.astype(str)

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_index, test_index in folds.split(design, strata):
        model = make_learner(learner, seed)
        model.fit(
            design.iloc[train_index],
            outcome[train_index],
            treatment[train_index],
        )
        predictions[test_index] = model.predict(design.iloc[test_index])

    return predictions


# ==========================================================================
# The null distribution
# ==========================================================================
def qini_null_distribution(
    outcome: np.ndarray,
    treatment: np.ndarray,
    n_draws: int = N_NULL_DRAWS,
    seed: int = config.RANDOM_SEED,
) -> np.ndarray:
    """Qini scores obtained by ranking customers at random.

    This is the reference every model has to beat. Qini is a noisy statistic:
    on a finite sample a meaningless ranking still produces a spread of scores
    either side of zero, and the spread is wide when the treatment effect is
    small. Without knowing that spread, a positive Qini is uninterpretable.
    """
    rng = np.random.default_rng(seed)
    scores = np.empty(n_draws)
    with _quiet_sklift():
        for draw in range(n_draws):
            scores[draw] = qini_auc_score(
                outcome, rng.permutation(len(outcome)).astype(float), treatment
            )
    return scores


# ==========================================================================
# Evaluation
# ==========================================================================
def evaluate_uplift(
    outcome: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    treatment_arm: str,
    outcome_name: str,
    learner: str,
    null_scores: np.ndarray,
) -> UpliftResult:
    """Score an out-of-fold uplift ranking against the random-ranking null."""
    with _quiet_sklift():
        qini = float(qini_auc_score(outcome, uplift, treatment))
        uplift_auc = float(uplift_auc_score(outcome, uplift, treatment))
        at_k = {
            f"top_{int(fraction * 100)}pct": float(
                uplift_at_k(outcome, uplift, treatment, strategy="overall", k=fraction)
            )
            for fraction in TOP_K_FRACTIONS
        }

    null_mean, null_std = float(null_scores.mean()), float(null_scores.std(ddof=1))

    return UpliftResult(
        treatment_arm=treatment_arm,
        outcome=outcome_name,
        learner=learner,
        n=len(outcome),
        qini=qini,
        qini_null_mean=null_mean,
        qini_null_std=null_std,
        qini_z_score=(qini - null_mean) / null_std if null_std > 0 else 0.0,
        qini_percentile=float((null_scores < qini).mean() * 100),
        # One-sided p, floored at 1/n_draws: with a finite null the smallest
        # observable p is the resolution of the simulation, and reporting 0
        # would claim more certainty than the draws support.
        p_value=max(
            float((null_scores >= qini).mean()), 1.0 / len(null_scores)
        ),
        # The empirical p bottoms out at the simulation's resolution, which
        # makes a z of 2.4 and a z of 8.4 both report 0.002 and look alike.
        # The null Qini is a mean of many exchangeable contributions and is
        # close to normal (checked in the notebook), so the normal tail
        # separates them: 0.0075 against 2.6e-17.
        p_value_normal=float(
            stats.norm.sf((qini - null_mean) / null_std) if null_std > 0 else 0.5
        ),
        uplift_auc=uplift_auc,
        uplift_at_k=at_k,
    )


# ==========================================================================
# Continuous outcomes
# ==========================================================================
def regression_t_learner_uplift(
    design: pd.DataFrame,
    outcome: np.ndarray,
    treatment: np.ndarray,
    n_folds: int = N_FOLDS,
    seed: int = config.RANDOM_SEED,
) -> np.ndarray:
    """Out-of-fold uplift for a continuous outcome, via a two-model approach.

    scikit-uplift's learners are classifiers, so spend needs its own path:
    fit a regressor per arm and difference their predictions. Feature 10 needs
    spend uplift to convert effects into profit, but Feature 5 showed spend is
    the noisiest outcome by a wide margin -- these estimates should be expected
    to carry very little signal.
    """
    predictions = np.zeros(len(design))
    strata = treatment.astype(str) + "_" + (outcome > 0).astype(int).astype(str)

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_index, test_index in folds.split(design, strata):
        train_design = design.iloc[train_index]
        train_outcome, train_treatment = outcome[train_index], treatment[train_index]

        treated_model = HistGradientBoostingRegressor(
            max_iter=150, max_depth=4, learning_rate=0.08, random_state=seed
        ).fit(train_design[train_treatment == 1], train_outcome[train_treatment == 1])
        control_model = HistGradientBoostingRegressor(
            max_iter=150, max_depth=4, learning_rate=0.08, random_state=seed
        ).fit(train_design[train_treatment == 0], train_outcome[train_treatment == 0])

        test_design = design.iloc[test_index]
        predictions[test_index] = treated_model.predict(
            test_design
        ) - control_model.predict(test_design)

    return predictions


# ==========================================================================
# Orchestration
# ==========================================================================
def run_uplift_analysis(
    df: pd.DataFrame | None = None,
    outcome: str = config.HETEROGENEITY_PRIMARY_OUTCOME,
    learners: list[str] | None = None,
    n_folds: int = N_FOLDS,
    n_null_draws: int = N_NULL_DRAWS,
    save: bool = True,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Cross-validate every learner on both comparisons and score against null."""
    frame = load_processed() if df is None else df
    learners = learners or LEARNERS

    results: list[UpliftResult] = []
    uplift_scores: dict[tuple[str, str], np.ndarray] = {}

    for treatment_arm, control_arm in config.COMPARISONS:
        comparison = make_comparison_frame(frame, treatment_arm, control_arm)
        design = build_design_matrix(comparison)
        y = comparison[outcome].to_numpy()
        t = comparison["treated"].to_numpy()

        logger.info("Building null distribution for %s", treatment_arm)
        null_scores = qini_null_distribution(y, t, n_null_draws, seed)

        for learner in learners:
            logger.info("Fitting %s on %s", learner, treatment_arm)
            uplift = out_of_fold_uplift(design, y, t, learner, n_folds, seed)
            uplift_scores[(treatment_arm, learner)] = uplift
            results.append(
                evaluate_uplift(
                    y, uplift, t, treatment_arm, outcome, learner, null_scores
                )
            )

    table = pd.DataFrame([asdict(r) for r in results])
    for fraction in TOP_K_FRACTIONS:
        key = f"top_{int(fraction * 100)}pct"
        table[key] = [r.uplift_at_k[key] for r in results]
    table = table.drop(columns=["uplift_at_k"])
    table["beats_random"] = [r.beats_random for r in results]
    table["p_value_adjusted"] = [
        r.p_value_adjusted(len(learners)) for r in results
    ]
    table["p_value_normal_adjusted"] = [
        r.p_value_normal_adjusted(len(learners)) for r in results
    ]
    table["significant"] = table["p_value_normal_adjusted"] < config.ALPHA

    if save:
        config.ensure_dirs()
        table.to_csv(config.RESULTS_DIR / "08_uplift_models.csv", index=False)
        logger.info("Wrote uplift results to %s", config.RESULTS_DIR)

    return table


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    table = run_uplift_analysis()

    print("\n" + "=" * 104)
    print(f"UPLIFT MODELS  (outcome = {config.HETEROGENEITY_PRIMARY_OUTCOME}, "
          f"{N_FOLDS}-fold out-of-fold, null from {N_NULL_DRAWS} random rankings)")
    print("=" * 104 + "\n")

    print(table[[
        "treatment_arm", "learner", "qini", "qini_z_score",
        "p_value_normal", "p_value_normal_adjusted", "significant",
        "top_10pct", "top_30pct",
    ]].to_string(index=False, formatters={
        "qini": "{:+.5f}".format,
        "qini_z_score": "{:+.2f}".format,
        "p_value_normal": "{:.2e}".format,
        "p_value_normal_adjusted": "{:.2e}".format,
        "top_10pct": "{:+.4f}".format,
        "top_30pct": "{:+.4f}".format,
    }))

    print("\n" + "-" * 104)
    print("Does this match Feature 7's prediction?")
    print("-" * 104 + "\n")
    print("  Feature 7 found the Womens campaign varies strongly by purchase")
    print("  history and the Mens campaign barely varies at all. An uplift model")
    print("  should therefore find signal on Womens and little on Mens.\n")
    for arm in [config.WOMENS_ARM, config.MENS_ARM]:
        subset = table[table["treatment_arm"] == arm]
        best = subset.loc[subset["qini"].idxmax()]
        print(f"  {arm:<14} {int(subset['significant'].sum())}/{len(subset)} "
              f"learners significant after adjustment   "
              f"best: {best['learner']} (z = {best['qini_z_score']:+.2f})")


if __name__ == "__main__":
    main()
