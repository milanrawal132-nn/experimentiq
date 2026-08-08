"""Tests for the generated final report.

The report's claim is that every number in it came from a result table and that
its inputs were verified before it quoted them. Two kinds of test back that up.

**Each consistency check is broken on purpose.** A check that passes on good
data proves nothing on its own -- `check_arm_sizes` would also pass if it
compared a number against itself. So every check is run against a deliberately
corrupted copy of the results and must fail. Same principle the data contracts
in `tests/test_data.py` are held to: a contract that cannot fail is not a
contract.

**The report is checked for the things generation makes easy to get wrong** --
stale figure links, unfilled values, non-determinism, and silently building a
report from results that disagree with one another.
"""

import pandas as pd
import pytest

from src import config
from src.report import checks as report_checks
from src.report.build import (
    FIGURES,
    REQUIRED_RESULTS,
    InconsistentResults,
    build,
    is_structural,
    load_results,
    money,
    points,
    pp,
    reflow,
    render,
)


@pytest.fixture(scope="module")
def results():
    return load_results()


@pytest.fixture(scope="module")
def report(results):
    return build(save=False)


def corrupt(results: dict[str, pd.DataFrame], name: str, **changes):
    """A copy of the results with one table altered."""
    broken = {key: frame.copy() for key, frame in results.items()}
    for column, value in changes.items():
        broken[name][column] = value
    return broken


# ==========================================================================
# Every check must be capable of failing
# ==========================================================================
class TestChecksCanFail:
    def test_all_checks_pass_on_the_real_results(self, results):
        failed = report_checks.failures(report_checks.run_checks(results))
        assert failed == [], [c.name for c in failed]

    def test_arm_sizes(self, results):
        broken = corrupt(results, "srm", observed=[1, 2, 3])
        assert not report_checks.check_arm_sizes(broken).passed

    def test_shared_control(self, results):
        broken = {k: v.copy() for k, v in results.items()}
        broken["ab_tests"].loc[0, "control_mean"] += 0.05
        assert not report_checks.check_shared_control(broken).passed

    def test_correction_is_conservative(self, results):
        broken = {k: v.copy() for k, v in results.items()}
        broken["ab_tests"]["p_value_adjusted"] = (
            broken["ab_tests"]["p_value"] / 10
        )
        assert not report_checks.check_correction_is_conservative(broken).passed

    def test_significance_matches_intervals(self, results):
        broken = corrupt(results, "ab_tests", ci_low=-1.0, ci_high=1.0)
        assert not report_checks.check_significance_matches_intervals(broken).passed

    def test_robust_implies_significant(self, results):
        broken = corrupt(results, "ab_tests", significant=False)
        assert not report_checks.check_robust_implies_significant(broken).passed

    def test_cuped_identity(self, results):
        broken = corrupt(results, "cuped", total_variance_reduction=0.5)
        assert not report_checks.check_cuped_identity(broken).passed

    def test_policy_recovers_arm_means(self, results):
        broken = {k: v.copy() for k, v in results.items()}
        broken["policy_values"]["value"] += 0.01
        assert not report_checks.check_policy_recovers_arm_means(broken).passed

    def test_profit_follows_from_spend(self, results):
        broken = corrupt(results, "economics", profit_per_email=1.0)
        assert not report_checks.check_profit_follows_from_spend(broken).passed

    def test_break_even_margin(self, results):
        broken = corrupt(results, "economics", break_even_margin=0.99)
        assert not report_checks.check_break_even_margin(broken).passed

    def test_personalisation_below_detection(self, results):
        """If the gain ever exceeded the MDE, the report's central reading of
        the null result would be wrong and must stop being asserted."""
        broken = corrupt(results, "policy_differences", difference=0.5)
        assert not report_checks.check_personalisation_below_detection(broken).passed

    def test_budget_accounting(self, results):
        broken = corrupt(results, "budget_curve", net_gain=0.0)
        assert not report_checks.check_budget_accounting(broken).passed

    def test_uplift_significance(self, results):
        broken = corrupt(results, "uplift", significant=True, qini_percentile=1.0)
        assert not report_checks.check_uplift_significance(broken).passed

    def test_every_registered_check_is_covered_here(self):
        """Adding a check without a failure test would let it rot unnoticed."""
        covered = {
            name.removeprefix("test_")
            for name in dir(TestChecksCanFail)
            if name.startswith("test_")
        }
        registered = {
            check.__name__.removeprefix("check_")
            for check in report_checks.CHECKS
        }
        assert registered <= covered


# ==========================================================================
# Building
# ==========================================================================
class TestBuild:
    def test_refuses_to_build_from_inconsistent_results(
        self, results, monkeypatch
    ):
        """The failure mode that matters. A report that quietly documents its
        own inconsistency is worse than no report."""
        broken = corrupt(results, "economics", profit_per_email=99.0)
        monkeypatch.setattr("src.report.build.load_results", lambda: broken)

        with pytest.raises(InconsistentResults, match="Profit follows"):
            build(save=False)

    def test_can_be_built_non_strict_for_diagnosis(self, results, monkeypatch):
        broken = corrupt(results, "economics", profit_per_email=99.0)
        monkeypatch.setattr("src.report.build.load_results", lambda: broken)

        text = build(save=False, strict=False)
        assert "FAIL" in text

    def test_is_deterministic(self, results):
        assert build(save=False) == build(save=False)

    def test_writes_to_the_reports_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
        monkeypatch.setattr("src.report.build.REPORT_PATH", tmp_path / "R.md")
        monkeypatch.setattr(config, "ensure_dirs", lambda: None)

        build(save=True)
        assert (tmp_path / "R.md").exists()

    def test_needs_every_declared_result(self):
        from src.dashboard import loaders

        for name in REQUIRED_RESULTS:
            assert name in loaders.RESULT_FILES


class TestContent:
    @pytest.mark.parametrize("heading", [
        "## Executive summary",
        "## The decisions, and how much confidence each carries",
        "## What we found",
        "## What this analysis could not determine",
        "## How the analysis defends itself",
        "## Consistency checks",
        "## Appendix — where everything comes from",
    ])
    def test_section_present(self, report, heading):
        assert heading in report

    def test_every_figure_referenced_actually_exists(self, report):
        """Guards against link rot when a figure is renamed."""
        for path in FIGURES.values():
            assert path in report, path
            assert (config.REPORTS_DIR / path).exists(), path

    def test_every_notebook_referenced_actually_exists(self, report):
        import re

        for match in re.findall(r"notebooks/[\w./]+\.ipynb", report):
            assert (config.PROJECT_ROOT / match).exists(), match

    def test_no_unfilled_placeholders(self, report):
        """A stray brace means a value was written literally instead of read."""
        assert "{" not in report
        assert "}" not in report

    def test_reports_the_headline_numbers_from_their_tables(self, report, results):
        ab = results["ab_tests"]
        mens = ab.query(
            "outcome == 'visit' and treatment_arm == @config.MENS_ARM"
        ).iloc[0]
        assert pp(mens["absolute_effect"]) in report

        economics = results["economics"].set_index("treatment_arm")
        assert money(economics.loc[config.MENS_ARM, "profit_per_email"]) in report

    def test_records_the_result_of_every_check(self, report, results):
        """The reader sees what was verified rather than being told it was."""
        for check in report_checks.run_checks(results):
            assert check.name in report

    def test_states_what_could_not_be_determined(self, report):
        """The section most likely to be quietly dropped, and the one that
        makes the rest trustworthy."""
        assert "womens campaign is profitable" in report
        assert "beyond two weeks" in report


# ==========================================================================
# Formatting
# ==========================================================================
class TestFormatting:
    def test_effects_are_signed(self):
        assert pp(0.0766) == "+7.66 pp"
        assert pp(-0.0766) == "-7.66 pp"

    def test_magnitudes_are_not_signed(self):
        """A detection threshold has no direction, and a leading plus reads as
        a claim that it does."""
        assert points(0.0085) == "0.85 pp"

    def test_negative_money_puts_the_sign_outside_the_symbol(self):
        assert money(-0.049) == "-$0.049"
        assert money(0.131) == "$0.131"


class TestReflow:
    def test_joins_a_paragraph_split_by_interpolation(self):
        text = "The mens campaign returned\n$0.131 per email."
        assert reflow(text) == "The mens campaign returned $0.131 per email."

    def test_a_bold_opener_is_not_mistaken_for_a_bullet(self):
        """`**` and `* ` differ by one space and most paragraphs here open
        bold, so getting this wrong strands a value on its own line."""
        text = "**In money, this is a decision.** An email costs\n$0.10 today."
        assert reflow(text) == "**In money, this is a decision.** An email costs $0.10 today."

    def test_tables_are_left_alone(self):
        table = "| a | b |\n|---|---|\n| 1 | 2 |"
        assert reflow(table) == table

    def test_headings_are_left_alone(self):
        assert reflow("## Executive summary") == "## Executive summary"

    def test_thematic_breaks_survive(self):
        assert reflow("---") == "---"

    def test_list_items_keep_their_own_lines(self):
        items = "- first item\n- second item"
        assert reflow(items) == items

    def test_indented_continuations_keep_their_indent(self):
        """Re-wrapping these would flatten them against the margin and change
        which bullet they belong to."""
        text = "- a finding\n  (with detail)"
        assert reflow(text) == text

    def test_fenced_code_is_untouched(self):
        code = "```bash\npython -m src.report.build\nls\n```"
        assert reflow(code) == code

    def test_images_keep_their_own_line(self):
        image = "![Outcomes by arm](figures/01_outcomes_by_arm.png)"
        assert reflow(image) == image

    def test_wraps_at_the_requested_width(self):
        text = " ".join(["word"] * 60)
        assert all(len(line) <= 40 for line in reflow(text, width=40).split("\n"))

    @pytest.mark.parametrize("line,structural", [
        ("| a | b |", True),
        ("## heading", True),
        ("- bullet", True),
        ("* bullet", True),
        ("1. numbered", True),
        ("---", True),
        ("> quote", True),
        ("![image](x.png)", True),
        ("  indented", True),
        ("**bold opener** and prose", False),
        ("ordinary prose", False),
        ("*emphasis* leading a line", False),
    ])
    def test_line_classification(self, line, structural):
        assert is_structural(line) is structural


def test_rendered_report_is_substantial(report):
    """A report that silently lost a section would still pass every section
    check if the section were empty."""
    assert len(report.split()) > 2_000
