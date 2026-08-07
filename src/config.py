"""Central configuration: paths, experiment design, and business assumptions.

Every module reads its constants from here so that an assumption is stated
once and can be defended once.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATABASE_DIR = DATA_DIR / "database"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

RAW_CSV = RAW_DIR / "hillstrom_email_marketing.csv"
PROCESSED_PARQUET = PROCESSED_DIR / "hillstrom_clean.parquet"
DUCKDB_PATH = DATABASE_DIR / "experimentiq.duckdb"

# --------------------------------------------------------------------------
# Experiment design
# --------------------------------------------------------------------------
TREATMENT_COL = "segment"

CONTROL_ARM = "No E-Mail"
MENS_ARM = "Mens E-Mail"
WOMENS_ARM = "Womens E-Mail"
ARMS = [CONTROL_ARM, MENS_ARM, WOMENS_ARM]

# The three-arm experiment is analysed as two independent two-arm tests, each
# against the shared control. This keeps every downstream method (t-tests,
# CUPED, uplift models) in its standard binary-treatment form.
COMPARISONS = [
    (MENS_ARM, CONTROL_ARM),
    (WOMENS_ARM, CONTROL_ARM),
]

# Expected allocation. The experiment was designed as an equal three-way split,
# which is what the sample ratio mismatch check in Feature 3 tests against.
EXPECTED_ARM_SHARE = 1 / 3

# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------
BINARY_OUTCOMES = ["visit", "conversion"]
CONTINUOUS_OUTCOMES = ["spend"]
OUTCOMES = BINARY_OUTCOMES + CONTINUOUS_OUTCOMES

# --------------------------------------------------------------------------
# Covariates
# --------------------------------------------------------------------------
# All of these describe the customer *before* the campaign was sent, so they
# are safe to use for balance checks, CUPED adjustment and uplift features
# without inducing post-treatment bias.
PRE_TREATMENT_COVARIATES = [
    "recency",
    "history",
    "history_segment",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
]

CATEGORICAL_COVARIATES = ["history_segment", "zip_code", "channel"]
NUMERIC_COVARIATES = ["recency", "history", "mens", "womens", "newbie"]

# Prior spend is the pre-period covariate used for CUPED (Feature 6). The
# dataset has no repeated pre-experiment measurement of the outcome itself,
# so `history` is the closest valid stand-in.
CUPED_COVARIATE = "history"

# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
ALPHA = 0.05
POWER_TARGET = 0.80

# Sample ratio mismatch is tested at a much stricter level than the outcome
# analysis. An SRM is a serious defect that invalidates the whole experiment,
# so the test is run to avoid crying wolf: at alpha = 0.05 one experiment in
# twenty would be flagged spuriously. 0.001 is the level used in the industry
# literature on SRM diagnosis.
SRM_ALPHA = 0.001

# Standardised mean difference above which a covariate is treated as
# meaningfully imbalanced. 0.1 is the conventional threshold from the matching
# and observational-study literature.
SMD_THRESHOLD = 0.10

# --------------------------------------------------------------------------
# Subgroup analysis
# --------------------------------------------------------------------------
# Fixed before any subgroup effect was estimated. Searching every way of
# splitting the data and reporting whichever split looks strongest is how
# subgroup analysis earns its bad reputation; committing to a short list with
# a stated rationale, in version control, is the defence. Anything not named
# here is exploratory and is labelled as such in the results.
PREREGISTERED_SUBGROUPS = {
    "mens": "Prior mens purchasers should respond more to the Mens campaign",
    "womens": "Prior womens purchasers should respond more to the Womens campaign",
    "recency_bucket": "Recent purchasers are more engaged, so likely more responsive",
    "newbie": "New and established customers may respond differently",
    "channel": "Prior purchase channel may moderate responsiveness to email",
}

EXPLORATORY_SUBGROUPS = {
    "history_segment": "Spend band may moderate response",
    "zip_code": "Location may moderate response",
}

# Splitting the sample costs power, and detecting an interaction of a given
# size needs roughly four times the customers that detecting a main effect of
# that size needs. Feature 5 showed visit is the only outcome with power to
# spare, so it carries the primary heterogeneity analysis; conversion and
# spend are reported but treated as underpowered.
HETEROGENEITY_PRIMARY_OUTCOME = "visit"

# Six tests are run in total (3 outcomes x 2 comparisons), so p-values are
# corrected for multiplicity before anything is called significant.
MULTIPLE_TESTING_METHOD = "holm"

N_BOOTSTRAP = 10_000
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Business assumptions for the profit model
# --------------------------------------------------------------------------
# Hillstrom publishes neither margin nor send cost, so these are explicit
# assumptions rather than measured values. They are exposed as inputs in the
# dashboard so the profit conclusion can be stress-tested.
GROSS_MARGIN = 0.30
COST_PER_EMAIL = 0.10


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create the output directories the pipeline writes to.

    The derived-data directories are gitignored, so they will not exist in a
    fresh clone. Every module that writes output calls this first.
    """
    for directory in (PROCESSED_DIR, DATABASE_DIR, FIGURES_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
