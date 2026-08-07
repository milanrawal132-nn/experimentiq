# ExperimentIQ

Causal experimentation and campaign-targeting platform built on the Hillstrom email
marketing experiment.

It answers four questions in order:

1. **Did the campaigns work?** — A/B analysis of a three-arm email experiment.
2. **Do I believe the result?** — randomisation diagnostics, power analysis, variance reduction.
3. **Who should receive which campaign?** — uplift modelling and a targeting policy.
4. **How should the budget be spent?** — profit-based allocation under a budget constraint.

---

## The experiment

64,000 customers were randomised into three arms:

| Arm | Customers |
|---|---:|
| Womens E-Mail | 21,387 |
| Mens E-Mail | 21,307 |
| No E-Mail | 21,306 |

Outcomes measured in the two weeks after the send: `visit`, `conversion`, `spend`.
Pre-treatment customer attributes: `recency`, `history_segment`, `history`, `mens`,
`womens`, `zip_code`, `newbie`, `channel`.

Source: [Kevin Hillstrom's MineThatData E-Mail Analytics Challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html).

### Two design decisions worth stating up front

**The three-arm test is analysed as two two-arm tests.** Each email arm is compared
against the shared `No E-Mail` control. This keeps every method — t-tests, CUPED,
uplift learners — in its standard binary-treatment form rather than requiring
multi-treatment variants, and the multiplicity it introduces is corrected for
explicitly.

**The 6,562 duplicate rows are kept, not dropped.** The dataset carries no customer
identifier, and the feature space is coarse enough (8 mostly low-cardinality
attributes) that distinct customers can legitimately share an identical row. With no
way to distinguish a genuine collision from a true duplicate, removing them would
non-randomly delete customers with common attribute profiles and bias the arm
comparison. They are retained and the decision is documented rather than hidden.

---

## Roadmap

Each feature is built as a notebook first, then promoted to a module under `src/`,
and lands as its own commit.

- [x] **0 — Project setup.** Curated dependencies, configuration, repository hygiene.
- [ ] **1 — Data layer.** Validated load to parquet with explicit data contracts.
- [ ] **2 — DuckDB warehouse.** Analytical store and SQL metric views.
- [ ] **3 — Experiment diagnostics.** Sample ratio mismatch, covariate balance.
- [ ] **4 — A/B analysis.** Lifts, confidence intervals, significance with Holm correction.
- [ ] **5 — Power and MDE.** Achieved power and minimum detectable effects per outcome.
- [ ] **6 — CUPED.** Variance reduction using prior spend as the pre-period covariate.
- [ ] **7 — Heterogeneous effects.** Pre-registered subgroup analysis.
- [ ] **8 — Uplift models.** Meta-learners evaluated by Qini and uplift@k.
- [ ] **9 — Targeting policy.** Per-customer campaign assignment, evaluated out of sample.
- [ ] **10 — Budget optimiser.** Incremental-profit allocation under a budget constraint.
- [ ] **11 — Dashboard.** Streamlit application over the DuckDB warehouse.
- [ ] **12 — Final report.** Executive summary and results write-up.

---

## Repository layout

```
experimentiq/
├── data/
│   ├── raw/            # source CSV (version-controlled)
│   ├── processed/      # derived parquet (rebuilt, ignored)
│   └── database/       # DuckDB file (rebuilt, ignored)
├── notebooks/          # exploration, one per feature
├── src/                # production modules
│   └── config.py       # paths, experiment design, business assumptions
├── reports/
│   ├── figures/        # generated charts
│   └── results/        # generated result tables
└── requirements.txt
```

---

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Verify the configuration resolves:

```bash
python -c "from src.config import RAW_CSV; print(RAW_CSV.exists())"
```

---

## Assumptions

The profit model in Feature 10 needs two numbers the dataset does not provide. They
are stated in `src/config.py` and exposed as adjustable inputs in the dashboard:

| Assumption | Value | Why it matters |
|---|---|---|
| Gross margin | 30% | Converts incremental revenue into incremental profit |
| Cost per email | $0.10 | Sets the break-even uplift for sending |

Conclusions about profit are conditional on these; conclusions about causal effect
are not.

---

## Licence

MIT — see [LICENSE](LICENSE).
