"""ExperimentIQ dashboard.

    streamlit run app/dashboard.py

Presents the results of Features 3 through 10 and lets the two business
assumptions behind the profit model be changed live.

The application reads generated result tables; it does not re-run any analysis.
`src/dashboard/loaders.py` explains why, and why the profit model is the single
deliberate exception.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.dashboard import loaders

# The project palette, kept identical to the generated figures so the dashboard
# and the README read as one piece of work.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

ARM_COLOUR = {
    config.CONTROL_ARM: BLUE,
    config.MENS_ARM: ORANGE,
    config.WOMENS_ARM: AQUA,
}

PAGES = [
    "Overview",
    "Randomisation checks",
    "Treatment effects",
    "Who responds",
    "Targeting policy",
    "Profit and budget",
    "Explore the warehouse",
]

st.set_page_config(page_title="ExperimentIQ", page_icon="📊", layout="wide")


# ==========================================================================
# Cached access
# ==========================================================================
# Caching lives here rather than in `src`, so the loaders stay importable and
# testable without Streamlit installed.
@st.cache_data(show_spinner=False)
def load_result(name: str) -> pd.DataFrame:
    return loaders.result(name)


@st.cache_data(show_spinner=False)
def load_arm_metrics() -> pd.DataFrame:
    return loaders.arm_metrics()


@st.cache_data(show_spinner=False)
def load_segment_metrics(dimension: str) -> pd.DataFrame:
    return loaders.segment_metrics(dimension)


@st.cache_data(show_spinner=False)
def load_dimensions() -> list[str]:
    return loaders.dimensions()


@st.cache_data(show_spinner=False)
def load_headline() -> dict:
    return loaders.headline()


def styled(figure: go.Figure, height: int = 380) -> go.Figure:
    """Apply the project's chart styling."""
    figure.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK, size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    figure.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    figure.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return figure


def guard(*names: str) -> bool:
    """Show a rebuild instruction instead of a traceback when results are absent.

    A fresh clone has no generated results, since they are derived from the raw
    CSV by the pipeline. Telling the reader which command produces what they
    asked for is more useful than a stack trace about a missing file.
    """
    absent = [name for name in names if not loaders.result_path(name).exists()]
    if not absent:
        return True

    st.warning("This page needs results that have not been generated yet.")
    st.code("\n".join(dict.fromkeys(loaders.rebuild_command(n) for n in absent)))
    return False


# ==========================================================================
# Pages
# ==========================================================================
def page_overview() -> None:
    st.title("ExperimentIQ")
    st.caption(
        "Causal analysis of a three-arm email experiment on 64,000 customers, "
        "from randomisation checks through to a budget decision."
    )

    if not guard("ab_tests", "policy_values", "economics"):
        return

    numbers = load_headline()
    columns = st.columns(4)
    columns[0].metric("Customers", f"{numbers['customers']:,}")
    columns[1].metric(
        "Mens E-Mail visit lift", f"{numbers['mens_visit_lift'] * 100:+.2f} pp"
    )
    columns[2].metric(
        "Womens E-Mail visit lift", f"{numbers['womens_visit_lift'] * 100:+.2f} pp"
    )
    columns[3].metric(
        "Significant after Holm",
        f"{numbers['significant_results']} of {numbers['total_results']}",
    )

    st.divider()

    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Outcomes by arm")
        metrics = load_arm_metrics()
        outcome = st.selectbox(
            "Outcome", config.OUTCOMES, key="overview_outcome"
        )
        column = loaders.arm_metric_column(outcome)
        arms = metrics[loaders.ARM_COLUMN]

        figure = go.Figure(
            go.Bar(
                x=arms,
                y=metrics[column],
                marker_color=[ARM_COLOUR.get(a, MUTED) for a in arms],
            )
        )
        figure.update_layout(yaxis_title=column.replace("_", " "))
        st.plotly_chart(styled(figure), width="stretch")

    with right:
        st.subheader("What the analysis concluded")
        st.markdown(
            f"""
- Randomisation is sound, so the arm differences are causal.
- Both campaigns work on visits; **Mens is roughly twice Womens**.
- The **Womens** campaign varies sharply by purchase history; the Mens
  campaign works about equally well on everyone.
- Personalised targeting is *directionally correct* but its gain is far below
  what this experiment can detect.
- In money, only **Mens E-Mail** demonstrably pays for itself
  (**${numbers['mens_profit_per_email']:+.3f}** per email against
  ${numbers['womens_profit_per_email']:+.3f} for Womens).
            """
        )
        st.info(
            "Every figure here is read from the generated result tables. "
            "The dashboard does not re-run the analysis — see the Profit and "
            "budget page for the single deliberate exception."
        )


def page_diagnostics() -> None:
    st.title("Randomisation checks")
    st.caption(
        "Nothing downstream is causal unless assignment was random. These run "
        "before any effect is estimated."
    )

    if not guard("srm", "balance", "omnibus"):
        return

    srm = loaders.srm_verdict(load_result("srm"))

    left, right = st.columns(2)
    left.metric(
        "Sample ratio mismatch",
        "no mismatch" if srm["passed"] else "MISMATCH",
        delta=f"chi-square {srm['chi2']:.2f}, p = {srm['p_value']:.3f}",
        delta_color="off",
    )
    left.caption(
        f"Tested at α = {srm['alpha']}, deliberately stricter than the outcome "
        "analysis: an SRM invalidates the whole experiment, so the test is run "
        "to avoid crying wolf."
    )

    balance = load_result("balance")
    worst = balance["smd"].abs().max()
    right.metric(
        "Largest standardised mean difference",
        f"{worst:.3f}",
        delta=f"threshold {config.SMD_THRESHOLD}",
        delta_color="off",
    )

    st.subheader("Covariate balance")
    st.caption(
        "Each pre-treatment attribute compared across arms. Anything beyond "
        f"±{config.SMD_THRESHOLD} would be a concern."
    )

    figure = go.Figure()
    for arm in balance["treatment_arm"].unique():
        rows = balance[balance["treatment_arm"] == arm]
        figure.add_trace(
            go.Bar(
                y=rows["covariate"],
                x=rows["smd"],
                name=arm,
                orientation="h",
                marker_color=ARM_COLOUR.get(arm, MUTED),
            )
        )
    for edge in (-config.SMD_THRESHOLD, config.SMD_THRESHOLD):
        figure.add_vline(x=edge, line_dash="dash", line_color=MUTED)
    figure.update_layout(barmode="group", xaxis_title="standardised mean difference")
    st.plotly_chart(styled(figure, height=460), width="stretch")

    st.subheader("Omnibus balance test")
    st.caption(
        "Can a model predict which arm a customer landed in from their "
        "attributes alone? Under valid randomisation it cannot."
    )
    st.dataframe(load_result("omnibus"), width="stretch", hide_index=True)


def page_effects() -> None:
    st.title("Treatment effects")
    st.caption(
        "Each email arm against the shared control, with Holm-corrected "
        "significance across all six tests."
    )

    if not guard("ab_tests", "power"):
        return

    ab = load_result("ab_tests")
    outcome = st.selectbox("Outcome", config.OUTCOMES, key="effects_outcome")
    rows = ab[ab["outcome"] == outcome]

    figure = go.Figure()
    for row in rows.itertuples():
        figure.add_trace(
            go.Scatter(
                x=[row.absolute_effect],
                y=[row.treatment_arm],
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[row.ci_high - row.absolute_effect],
                    arrayminus=[row.absolute_effect - row.ci_low],
                ),
                mode="markers",
                marker=dict(size=13, color=ARM_COLOUR.get(row.treatment_arm, MUTED)),
                name=row.treatment_arm,
            )
        )
    figure.add_vline(x=0, line_color=INK)
    figure.update_layout(xaxis_title=f"absolute effect on {outcome}")
    st.plotly_chart(styled(figure, height=280), width="stretch")

    st.dataframe(
        rows[[
            "treatment_arm", "control_mean", "treated_mean", "absolute_effect",
            "relative_effect", "ci_low", "ci_high", "p_value",
            "p_value_adjusted", "significant", "test_name",
        ]],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Can this experiment detect what it found?")
    st.caption(
        "The minimum detectable effect is fixed by the design, not by the "
        "result. A finding is called robust when the lower end of its interval "
        "clears that threshold — not when its p-value is small."
    )
    st.dataframe(
        load_result("power")[[
            "outcome", "treatment_arm", "observed_effect", "observed_ci_low",
            "mde_absolute", "mde_relative", "robust", "verdict",
        ]],
        width="stretch",
        hide_index=True,
    )


def page_heterogeneity() -> None:
    st.title("Who responds")
    st.caption(
        "Subgroup effects were pre-registered before estimation, and "
        "heterogeneity is judged by an interaction test rather than by "
        "comparing per-subgroup p-values."
    )

    if not guard("subgroups", "interactions", "uplift"):
        return

    interactions = load_result("interactions")
    st.subheader("Does the effect genuinely vary?")
    st.dataframe(interactions, width="stretch", hide_index=True)

    st.subheader("Effect by subgroup")
    subgroups = load_result("subgroups")
    subgroup = st.selectbox(
        "Subgroup", sorted(subgroups["subgroup"].unique()), key="subgroup"
    )
    rows = subgroups[subgroups["subgroup"] == subgroup]

    figure = go.Figure()
    for arm in rows["treatment_arm"].unique():
        arm_rows = rows[rows["treatment_arm"] == arm]
        figure.add_trace(
            go.Bar(
                x=arm_rows["level"],
                y=arm_rows["effect"],
                name=arm,
                marker_color=ARM_COLOUR.get(arm, MUTED),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=arm_rows["ci_high"] - arm_rows["effect"],
                    arrayminus=arm_rows["effect"] - arm_rows["ci_low"],
                ),
            )
        )
    figure.update_layout(barmode="group", yaxis_title="absolute effect")
    st.plotly_chart(styled(figure), width="stretch")

    st.subheader("Uplift models")
    st.caption(
        "Every Qini score is compared against the distribution produced by 500 "
        "random rankings on the same data. A model that cannot beat that has "
        "not demonstrated any ability to rank customers."
    )
    st.dataframe(
        load_result("uplift")[[
            "treatment_arm", "learner", "qini", "qini_z_score",
            "p_value_normal_adjusted", "significant",
        ]],
        width="stretch",
        hide_index=True,
    )


def page_policy() -> None:
    st.title("Targeting policy")
    st.caption(
        "Policies are valued by inverse propensity weighting: a customer "
        "received one arm, so a policy that would have assigned them "
        "differently is estimated by reweighting the customers whose actual "
        "arm matched."
    )

    if not guard("policy_values", "policy_differences"):
        return

    values = load_result("policy_values").sort_values("value")

    figure = go.Figure(
        go.Bar(
            y=values["policy"],
            x=values["value"],
            orientation="h",
            marker_color=BLUE,
            error_x=dict(
                type="data",
                symmetric=False,
                array=values["ci_high"] - values["value"],
                arrayminus=values["value"] - values["ci_low"],
            ),
        )
    )
    figure.update_layout(xaxis_title="visit rate under the policy")
    st.plotly_chart(styled(figure, height=360), width="stretch")

    st.subheader("Against the best fixed rule")
    st.caption(
        "Beating 'email nobody' is trivial. The honest benchmark is the "
        "strongest thing achievable without a model: send the better campaign "
        "to everyone. Comparisons are paired, so customers the two policies "
        "agree on contribute no noise."
    )
    st.dataframe(
        load_result("policy_differences")[[
            "policy", "versus", "difference", "ci_low", "ci_high",
            "p_value", "n_disagree",
        ]],
        width="stretch",
        hide_index=True,
    )
    st.info(
        "Personalisation does not significantly beat sending Mens E-Mail to "
        "everyone. Its decisions are directionally correct, but the gain is "
        "roughly 5x below the smallest effect this experiment can resolve — a "
        "limit of the design, not evidence the policy is worthless."
    )


def page_profit() -> None:
    st.title("Profit and budget")
    st.caption(
        "Margin and cost per email are assumptions, not measurements. They are "
        "controls here so the profit conclusion can be stress-tested by "
        "whoever is making the decision."
    )

    if not guard("ab_tests", "budget_curve", "ranking_comparison"):
        return

    left, right = st.columns(2)
    margin = left.slider(
        "Gross margin", min_value=0.05, max_value=0.60,
        value=float(config.GROSS_MARGIN), step=0.01, format="%.0f%%",
    )
    cost = right.slider(
        "Cost per email", min_value=0.0, max_value=0.50,
        value=float(config.COST_PER_EMAIL), step=0.01, format="$%.2f",
    )

    economics = loaders.live_economics(load_result("ab_tests"), margin, cost)

    st.metric(
        "Incremental spend an email must generate to break even",
        f"${economics['break_even_spend'].iloc[0]:.3f}",
    )

    columns = st.columns(len(economics))
    for column, row in zip(columns, economics.to_dict("records")):
        reading = loaders.verdict(pd.Series(row))
        column.metric(
            row["treatment_arm"],
            f"${row['profit_per_email']:+.3f} per email",
            delta=reading,
            delta_color="normal" if reading == "pays for itself" else "off",
        )
        column.caption(
            f"95% CI ${row['profit_ci_low']:+.3f} to ${row['profit_ci_high']:+.3f} "
            f"· break-even margin {row['break_even_margin']:.1%} "
            f"({row['break_even_margin_high']:.1%} pessimistically)"
        )

    st.divider()
    st.subheader("What a budget buys")

    arm = st.selectbox(
        "Campaign", economics["treatment_arm"].tolist(), key="profit_arm"
    )
    n_customers = int(load_headline()["customers"])
    projection = loaders.budget_projection(economics, arm, n_customers, cost)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=projection["budget"], y=projection["ci_high"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=projection["budget"], y=projection["ci_low"],
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(42,120,214,0.16)", name="95% interval",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=projection["budget"], y=projection["net_gain"],
            mode="lines", line=dict(color=ARM_COLOUR.get(arm, BLUE), width=3),
            name="expected profit",
        )
    )
    figure.add_hline(y=0, line_color=INK)
    figure.update_layout(
        xaxis_title="email budget ($)",
        yaxis_title="incremental profit above emailing nobody ($)",
    )
    st.plotly_chart(styled(figure, height=400), width="stretch")

    st.caption(
        "Exactly linear, and that is not a simplification: a policy emailing k "
        "customers chosen without reference to their attributes earns k times "
        "the average incremental profit per email. Feature 10's measured curve "
        "wanders around this line; the wander is estimation noise, not a "
        "targeting effect."
    )

    st.subheader("Does ranking the budget by a model help?")
    st.caption(
        "Measured at the configured assumptions rather than the sliders, "
        "because it depends on per-customer model scores rather than on "
        "arithmetic."
    )
    st.dataframe(
        load_result("ranking_comparison")[[
            "ranking", "versus", "difference", "ci_low", "ci_high",
            "p_value", "n_disagree",
        ]],
        width="stretch",
        hide_index=True,
    )
    st.warning(
        "Ranking by the spend model — which failed its null test in Feature 8 — "
        "is significantly worse than sending Mens to everyone. Ranking by the "
        "visit model, which passed, is harmless but adds nothing measurable. "
        "The cost of deploying an unvalidated model is not zero."
    )


def page_explorer() -> None:
    st.title("Explore the warehouse")
    st.caption(
        "The DuckDB views behind the analysis. They report point estimates and "
        "no uncertainty by design — inference lives in Python, where the "
        "standard errors are."
    )

    dimension = st.selectbox("Customer attribute", load_dimensions())
    rows = load_segment_metrics(dimension)

    outcome = st.selectbox("Metric", loaders.treatment_metrics(rows))

    figure = go.Figure()
    for arm in rows[loaders.ARM_COLUMN].unique():
        arm_rows = rows[rows[loaders.ARM_COLUMN] == arm]
        figure.add_trace(
            go.Bar(
                x=arm_rows["level"],
                y=arm_rows[outcome],
                name=str(arm),
                marker_color=ARM_COLOUR.get(arm, MUTED),
            )
        )
    figure.update_layout(barmode="group", yaxis_title=outcome.replace("_", " "))
    st.plotly_chart(styled(figure), width="stretch")

    st.dataframe(rows, width="stretch", hide_index=True)


# ==========================================================================
# Shell
# ==========================================================================
PAGE_RENDERERS = {
    "Overview": page_overview,
    "Randomisation checks": page_diagnostics,
    "Treatment effects": page_effects,
    "Who responds": page_heterogeneity,
    "Targeting policy": page_policy,
    "Profit and budget": page_profit,
    "Explore the warehouse": page_explorer,
}


def main() -> None:
    st.sidebar.title("ExperimentIQ")
    page = st.sidebar.radio("Section", PAGES, label_visibility="collapsed")

    outstanding = loaders.missing()
    if outstanding:
        st.sidebar.warning(f"{len(outstanding)} result tables not yet generated")
        with st.sidebar.expander("How to generate them"):
            st.code("\n".join(loaders.rebuild_plan()))

    st.sidebar.divider()
    st.sidebar.caption(
        "Built on the Hillstrom MineThatData e-mail experiment. "
        "Results are read from generated tables; the profit model is "
        "recomputed live from the margin and cost controls."
    )

    PAGE_RENDERERS[page]()


main()
