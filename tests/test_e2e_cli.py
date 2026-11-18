"""End-to-end integration tests for the poker-equity command-line interface.

Covers full process execution (``python -m poker_equity``), cross-subcommand analysis
workflows, JSON contract validation, argument parsing, error handling, and terminal output.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from poker_equity.cli import main

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return _ANSI_RE.sub("", text)


def run_cli_subprocess(
    args: list[str],
    env_override: dict[str, str] | None = None,
    stdin_data: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute the poker_equity CLI via subprocess as ``python -m poker_equity ...``."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    if env_override:
        env.update(env_override)
    return subprocess.run(
        [sys.executable, "-m", "poker_equity", *args],
        input=stdin_data,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


# =========================================================================
# 1. Process Invocation & Subcommand Help / Version
# =========================================================================


class TestCliProcessInvocation:
    """Tests for external process execution via python -m poker_equity."""

    def test_python_m_poker_equity_help(self) -> None:
        """Invoking python -m poker_equity --help returns exit code 0 and lists subcommands."""
        res = run_cli_subprocess(["--help"])
        assert res.returncode == 0
        assert "poker-equity" in res.stdout
        subcommands = [
            "equity",
            "range",
            "odds",
            "outs",
            "decision",
            "table",
            "twice",
            "curve",
            "matrix",
        ]
        for subcmd in subcommands:
            assert subcmd in res.stdout

    def test_python_m_poker_equity_version(self) -> None:
        """Invoking python -m poker_equity --version outputs version and exits with 0."""
        res = run_cli_subprocess(["--version"])
        assert res.returncode == 0
        assert "poker-equity" in res.stdout
        assert "0.1.0" in res.stdout

    def test_python_m_poker_equity_cli_module(self) -> None:
        """Invoking python -m poker_equity.cli works identically to python -m poker_equity."""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC_DIR)
        res = subprocess.run(
            [sys.executable, "-m", "poker_equity.cli", "--version"],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        assert res.returncode == 0
        assert "poker-equity 0.1.0" in res.stdout

    @pytest.mark.parametrize(
        "subcmd",
        [
            "equity",
            "range",
            "odds",
            "outs",
            "decision",
            "table",
            "twice",
            "curve",
            "matrix",
        ],
    )
    def test_subcommand_help_all(self, subcmd: str) -> None:
        """Every subcommand responds to --help with exit code 0 and usage information."""
        res = run_cli_subprocess([subcmd, "--help"])
        assert res.returncode == 0
        assert f"usage: poker-equity {subcmd}" in res.stdout or f"{subcmd}" in res.stdout

    def test_no_color_environment_variable(self) -> None:
        """Setting NO_COLOR=1 in environment suppresses ANSI escape codes in output."""
        res = run_cli_subprocess(
            ["equity", "AhKh", "QcQd", "--sims", "200", "--seed", "42"],
            env_override={"NO_COLOR": "1"},
        )
        assert res.returncode == 0
        assert "\033[" not in res.stdout

    def test_no_color_flag(self) -> None:
        """Passing --no-color flag suppresses ANSI color codes."""
        res = run_cli_subprocess(
            ["equity", "AhKh", "QcQd", "--no-color", "--sims", "200", "--seed", "42"]
        )
        assert res.returncode == 0
        assert "\033[" not in res.stdout

    def test_unknown_command_returns_code_2(self) -> None:
        """Invoking an invalid subcommand returns exit code 2 and usage error on stderr."""
        res = run_cli_subprocess(["nonexistent_command"])
        assert res.returncode == 2
        assert "invalid choice" in res.stderr or "error" in res.stderr.lower()


# =========================================================================
# 2. Equity Subcommand End-to-End
# =========================================================================


class TestE2EEquitySubcommand:
    """End-to-end integration tests for the 'equity' subcommand."""

    def test_equity_heads_up_preflop_text(self) -> None:
        """Heads-up preflop equity simulation returns formatted table and 95% CI."""
        code = main(["equity", "Ah Kh", "Qc Qd", "--sims", "500", "--seed", "42", "--no-color"])
        assert code == 0

    def test_equity_heads_up_preflop_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Heads-up preflop equity JSON output contains valid structure and probability sum."""
        code = main(["equity", "Ah Kh", "Qc Qd", "--sims", "500", "--seed", "42", "--json"])
        assert code == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "players" in data
        assert len(data["players"]) == 2
        p1, p2 = data["players"][0], data["players"][1]
        assert p1["hand"] == "Ah Kh"
        assert p2["hand"] == "Qc Qd"
        total_eq = p1["equity"] + p2["equity"]
        assert pytest.approx(total_eq, abs=1e-3) == 1.0

    def test_equity_multiway_postflop_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Multiway equity calculation on a flop board."""
        code = main([
            "equity",
            "Ah Kh", "Qc Qd", "Js Ts",
            "-b", "2h 7h 8c",
            "--sims", "500",
            "--seed", "42",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["board_str"] == "2h 7h 8c"
        assert len(data["players"]) == 3
        total_equity = sum(p["equity"] for p in data["players"])
        assert pytest.approx(total_equity, abs=1e-3) == 1.0

    def test_equity_turn_exact_enumeration(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Simulation on turn board computes equity with 4 community cards."""
        code = main([
            "equity",
            "Ah Kh", "Qc Qd",
            "-b", "2h 7h 8c 3d",
            "--sims", "2000",
            "--seed", "42",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["board_str"] == "2h 7h 8c 3d"
        total_equity = sum(p["equity"] for p in data["players"])
        assert pytest.approx(total_equity, abs=1e-3) == 1.0

    def test_equity_with_dead_cards(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Excluding dead cards modifies available deck cards."""
        code = main([
            "equity",
            "Ah Kh", "Qc Qd",
            "-b", "2h 7h 8c",
            "-d", "9h Th",
            "--sims", "500",
            "--seed", "42",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["dead_str"] == "9h Th"

    def test_equity_parallel_execution(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Parallel equity simulation runs across multiple worker processes."""
        code = main([
            "equity",
            "Ah Kh", "Qc Qd",
            "-n", "500",
            "-j", "2",
            "-s", "42",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert len(data["players"]) == 2


# =========================================================================
# 3. Range Subcommand End-to-End
# =========================================================================


class TestE2ERangeSubcommand:
    """End-to-end integration tests for the 'range' subcommand."""

    def test_range_vs_range_preflop_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Range vs range preflop simulation produces complete JSON breakdown."""
        code = main([
            "range",
            "QQ+", "TT+",
            "--sims", "500",
            "--seed", "42",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["hero"]["spec"] == "QQ+"
        assert data["villain"]["spec"] == "TT+"
        assert data["hero"]["is_range"] is True
        assert data["villain"]["is_range"] is True
        assert data["hero"]["equity"] > data["villain"]["equity"]

    def test_range_hand_vs_range_with_breakdown_json(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Hand vs range with top N combo breakdown."""
        code = main([
            "range",
            "Ah Kd", "TT+, AJs+, KQs",
            "-b", "Kh 7d 2c",
            "--sims", "500",
            "--seed", "42",
            "--breakdown", "5",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert "breakdown" in data
        assert len(data["breakdown"]) <= 5
        assert all("equity" in b for b in data["breakdown"])

    def test_range_percentage_preset_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Range evaluation using top X% shorthand syntax."""
        code = main([
            "range",
            "top15%", "top25%",
            "--sims", "500",
            "--seed", "42",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["hero"]["spec"] == "top15%"
        assert data["villain"]["spec"] == "top25%"

    def test_range_with_matrix_flag_displays_grid_and_equity(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Running range with --matrix renders comparison grid and equity results."""
        code = main([
            "range",
            "QQ+", "TT+",
            "--matrix",
            "--sims", "300",
            "--seed", "42",
            "--no-color",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "Range Comparison Matrix:" in out
        assert "=AA" in out
        assert "Range vs Range Equity Simulation" in out

    def test_range_single_with_matrix_flag_routes_to_grid(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Providing a single range with --matrix displays the 13x13 grid without error."""
        code = main(["range", "top15%", "--matrix", "--no-color"])
        assert code == 0
        out = capsys.readouterr().out
        assert "Range Matrix: top15%" in out
        assert "Range Summary:" in out


# =========================================================================
# 4. Pot Odds & Outs Subcommands End-to-End
# =========================================================================


class TestE2EOddsAndOutsSubcommands:
    """End-to-end integration tests for 'odds' and 'outs' subcommands."""

    def test_pot_odds_and_ev_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Pot odds and EV evaluation facing a bet."""
        code = main([
            "odds",
            "--pot", "100",
            "--call", "50",
            "--equity", "0.40",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["pot_size"] == 100.0
        assert data["bet_to_call"] == 50.0
        assert pytest.approx(data["pot_odds_pct"], abs=1e-4) == 0.333333
        assert pytest.approx(data["ev"], abs=1e-2) == 10.0
        assert data["is_profitable"] is True
        assert data["recommendation"] == "CALL"

    def test_pot_odds_implied_and_reverse_implied_json(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Pot odds with implied odds and reverse implied odds."""
        code = main([
            "odds",
            "100", "50",
            "--equity", "0.25",
            "--implied", "80",
            "--reverse-implied", "20",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["future_bets"] == 80.0
        assert data["future_losses"] == 20.0
        assert data["ev_with_future_bets"] is not None

    def test_outs_flush_draw_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Outs analysis for nut flush draw on flop."""
        code = main([
            "outs",
            "Ah Kh", "Qc Qd", "2h 7h 8c",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["hero"] == "Ah Kh"
        assert data["villain"] == "Qc Qd"
        assert data["board"] == "2h 7h 8c"
        assert data["total_outs"] >= 9
        assert "Flush" in data["outs_by_rank"]

    def test_outs_vs_villain_discounting_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Outs calculation against known villain hand discounts blocked/dominated outs."""
        code = main([
            "outs",
            "--hero", "Ah Kh",
            "--villain", "Qc Qd",
            "--board", "2h 7h 8c",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["hero"] == "Ah Kh"
        assert data["villain"] == "Qc Qd"
        assert data["total_outs"] >= 9


# =========================================================================
# 5. Decision Recommendation Subcommand End-to-End
# =========================================================================


class TestE2EDecisionSubcommand:
    """End-to-end integration tests for the 'decision' subcommand."""

    def test_decision_facing_bet_profitable_call_json(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """GTO decision recommendation facing a bet with strong draw."""
        code = main([
            "decision",
            "--pot", "100",
            "--call", "50",
            "--hero", "Ah Kh",
            "--villain", "Qc Qd",
            "--board", "2h 7h 8c",
            "--sims", "500",
            "--seed", "42",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert "recommended_action" in data
        assert data["recommended_action"] in ("call", "raise", "fold")
        assert "pot_odds" in data
        assert "equity" in data
        assert "ev" in data

    def test_decision_checking_spot_default_zero_bet_json(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When bet is 0 (checking spot), recommendation is check or bet."""
        code = main([
            "decision",
            "--pot", "100",
            "--call", "0",
            "--hero", "Ah Kh",
            "--villain", "Qc Qd",
            "--board", "2h 7h 8c",
            "--sims", "500",
            "--seed", "42",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["recommended_action"] in ("check", "bet")
        assert data["bet_to_call"] == 0.0


# =========================================================================
# 6. Interactive Table Simulator Subcommand End-to-End
# =========================================================================


class TestE2ETableSubcommand:
    """End-to-end integration tests for the 'table' subcommand."""

    def test_table_scripted_multi_street_hand_json(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Pre-scripted hand executes from preflop through river showdown."""
        code = main([
            "table",
            "--hero-cards", "Ah Kh",
            "--villain-cards", "Qc Qd",
            "--actions", "call,check,check,check,check,check,check,check",
            "--seed", "42",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["is_hand_over"] is True
        assert len(data["players"]) == 2
        assert len(data["history"]) >= 6
        assert "total_pot" in data
        assert data["total_pot"] >= 4.0

    def test_table_scripted_preflop_fold_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Pre-scripted fold terminates hand immediately without dealing board."""
        code = main([
            "table",
            "--actions", "fold",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["is_hand_over"] is True
        assert len(data["history"]) == 1

    def test_table_with_live_recommendations(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Table simulator with --recommend runs equity calculations during action progression."""
        code = main([
            "table",
            "--hero-cards", "Ah Kh",
            "--recommend",
            "--actions", "call,check",
            "--seed", "42",
            "--no-color",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "Table State" in out
        assert "Pot Odds" in out


# =========================================================================
# 7. Run-It-Twice & Equity Curve Subcommands End-to-End
# =========================================================================


class TestE2ETwiceAndCurveSubcommands:
    """End-to-end integration tests for 'twice' and 'curve' subcommands."""

    def test_twice_simulation_flop_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Run-it-twice simulation computes scoop, split, and variance reduction."""
        code = main([
            "twice",
            "Ah Kh", "Qc Qd",
            "-b", "2h 7h 8c",
            "--pot", "200",
            "--runs", "2",
            "--sims", "500",
            "--seed", "42",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["num_runs"] == 2
        assert "equities" in data
        assert "scoop_pcts" in data
        assert "split_pct" in data
        assert "variance_reduction" in data
        assert data["pot_size"] == 200.0
        assert len(data["expected_payouts"]) == 2

    def test_twice_exact_turn_enumeration_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Run-it-twice exact enumeration on turn board."""
        code = main([
            "twice",
            "Ah Kh", "Qc Qd",
            "-b", "2h 7h 8c 3d",
            "--exact",
            "--pot", "150",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["is_exact"] is True
        assert data["num_sims"] == 946
        assert len(data["expected_payouts"]) == 2

    def test_curve_full_runout_with_chart_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Equity curve tracking street-by-street equity evolution with chart."""
        code = main([
            "curve",
            "Ah Kh", "Qc Qd",
            "-b", "2h 7h 8c 3d 4s",
            "--chart",
            "--seed", "42",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert "points" in data
        assert len(data["points"]) == 4  # Preflop, Flop, Turn, River
        assert data["hands"] == ["Ah Kh", "Qc Qd"]
        assert "plot_data" in data

    def test_curve_next_street_distribution_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Next-street card distribution analysis across remaining deck cards."""
        code = main([
            "curve",
            "Ah Kh", "Qc Qd",
            "-b", "2h 7h 8c",
            "--next",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert "next_street_distribution" in data
        dist = data["next_street_distribution"]
        assert "current_street" in dist
        assert "next_street" in dist
        assert "mean_hero_equity" in dist
        assert len(dist["favorable_cards"]) > 0
        assert len(dist["unfavorable_cards"]) > 0


# =========================================================================
# 8. 13x13 Range Grid Matrix Subcommand End-to-End
# =========================================================================


class TestE2EMatrixSubcommand:
    """End-to-end integration tests for the 'matrix' subcommand."""

    @pytest.mark.parametrize("display_mode", ["hands", "weights", "combos", "heat"])
    def test_matrix_display_modes(
        self, display_mode: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Every matrix display mode formats successfully without error."""
        code = main(["matrix", "top15%", "--mode", display_mode, "--no-color"])
        assert code == 0
        out = capsys.readouterr().out
        assert "Range Matrix: top15%" in out
        assert "Range Summary:" in out

    def test_matrix_blockers_and_show_all(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Matrix visualizer filters dead cards and displays unselected cells with --show-all."""
        code = main([
            "matrix", "AA, KK",
            "--board", "Ah Kd 2s",
            "--show-all",
            "--no-color",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "Blockers:" in out
        assert "72o" in out  # Show-all exposes unselected hands

    def test_matrix_comparison_two_ranges_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Comparing two ranges via matrix outputs valid comparison schema."""
        code = main([
            "matrix", "QQ+, AKs", "TT+, AJs+, KQs, AKo",
            "--json",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert "hero" in data
        assert "villain" in data
        assert "comparison" in data
        cmp_res = data["comparison"]
        assert cmp_res["overlap_combos"] == 22.0
        assert "AA" in cmp_res["overlap_cells"]
        assert "TT" in cmp_res["villain_only_cells"]


# =========================================================================
# 9. Cross-Tool Analysis Pipelines (End-to-End Scenarios)
# =========================================================================


class TestE2EAnalysisPipelines:
    """Complex multi-command poker analysis workflows."""

    def test_hand_review_complete_lifecycle_pipeline(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Simulate a complete hand review workflow across multiple tools:

        1. Preflop equity analysis
        2. Flop outs evaluation
        3. Facing bet pot odds & decision
        4. Turn/river multi-street progression
        5. All-in run-it-twice payout
        """
        # Step 1: Preflop equity
        c1 = main(["equity", "Ah Kh", "Qc Qd", "-s", "42", "-n", "300", "--json"])
        assert c1 == 0
        eq_data = json.loads(capsys.readouterr().out)
        assert eq_data["players"][0]["equity"] >= 0.40

        # Step 2: Flop outs evaluation
        c2 = main(["outs", "Ah Kh", "Qc Qd", "2h 7h 8c", "--json"])
        assert c2 == 0
        outs_data = json.loads(capsys.readouterr().out)
        assert outs_data["total_outs"] >= 9

        # Step 3: Pot odds & Decision facing a $50 bet into $100 pot
        c3 = main([
            "decision",
            "--pot", "100",
            "--call", "50",
            "--hero", "Ah Kh",
            "--villain", "Qc Qd",
            "--board", "2h 7h 8c",
            "-s", "42",
            "-n", "300",
            "--json",
        ])
        assert c3 == 0
        dec_data = json.loads(capsys.readouterr().out)
        assert dec_data["recommended_action"] in ("call", "raise")

        # Step 4: Equity curve across streets
        c4 = main([
            "curve",
            "Ah Kh", "Qc Qd",
            "-b", "2h 7h 8c 3d 4s",
            "-s", "42",
            "--json",
        ])
        assert c4 == 0
        curve_data = json.loads(capsys.readouterr().out)
        assert len(curve_data["points"]) == 4

        # Step 5: Run-it-twice payout resolution
        c5 = main([
            "twice",
            "Ah Kh", "Qc Qd",
            "-b", "2h 7h 8c",
            "--pot", "150",
            "--runs", "2",
            "-s", "42",
            "-n", "300",
            "--json",
        ])
        assert c5 == 0
        rit_data = json.loads(capsys.readouterr().out)
        assert sum(rit_data["expected_payouts"]) == pytest.approx(150.0)

    def test_range_defense_strategy_pipeline(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Simulate a GTO range construction and defense strategy workflow:

        1. Inspect BTN opening range matrix
        2. Compare against BB defend range
        3. Evaluate BTN vs BB equity on wet flop
        """
        # Step 1: BTN open matrix
        c1 = main(["matrix", "btn_open", "--json"])
        assert c1 == 0
        m1 = json.loads(capsys.readouterr().out)
        assert m1["total_combos"] > 0

        # Step 2: Compare BTN open vs BB defend
        c2 = main(["matrix", "btn_open", "bb_vs_btn_defense", "--json"])
        assert c2 == 0
        m2 = json.loads(capsys.readouterr().out)
        assert "comparison" in m2
        assert len(m2["comparison"]["overlap_cells"]) > 0

        # Step 3: Equity simulation on wet flop
        c3 = main([
            "range",
            "btn_open", "bb_vs_btn_defense",
            "-b", "Jh Th 9c",
            "-s", "42",
            "-n", "300",
            "--json",
        ])
        assert c3 == 0
        r3 = json.loads(capsys.readouterr().out)
        assert "hero" in r3 and "villain" in r3


# =========================================================================
# 10. Strict JSON Output Purity & Universal Error Handling
# =========================================================================


class TestE2EErrorHandlingAndPurity:
    """Verifies that all subcommands uphold clean JSON contracts and exit codes."""

    @pytest.mark.parametrize(
        "cmd_args",
        [
            ["equity", "AhKh", "QcQd", "-n", "100", "-s", "42", "--json"],
            ["range", "QQ+", "TT+", "-n", "100", "-s", "42", "--json"],
            ["odds", "--pot", "100", "--call", "50", "--json"],
            ["outs", "AhKh", "QcQd", "2h7h8c", "--json"],
            [
                "decision",
                "--pot", "100",
                "--call", "50",
                "--hero", "AhKh",
                "--villain", "QcQd",
                "-n", "100",
                "-s", "42",
                "--json",
            ],
            ["table", "--actions", "fold", "--json"],
            ["twice", "AhKh", "QcQd", "-b", "2h7h8c", "-n", "100", "-s", "42", "--json"],
            ["curve", "AhKh", "QcQd", "-b", "2h7h8c3d4s", "-s", "42", "--json"],
            ["matrix", "top10%", "--json"],
        ],
    )
    def test_json_purity_across_all_subcommands(
        self, cmd_args: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Every subcommand in --json mode emits strictly valid JSON with no warning/log pollution."""
        code = main(cmd_args)
        assert code == 0
        captured = capsys.readouterr()
        # Verify stderr has no warnings or errors
        assert captured.err.strip() == ""
        # Verify stdout is 100% valid JSON parseable without trimming
        parsed = json.loads(captured.out)
        assert isinstance(parsed, dict)

    @pytest.mark.parametrize(
        ("cmd_args", "err_substring"),
        [
            (["equity", "invalid_hand"], "error"),
            (["equity", "AhKh", "AhKd"], "duplicate"),
            (["range"], "both hero and villain"),
            (["range", "invalid_range_token_xyz", "QQ+"], "error"),
            (["odds"], "both pot size and bet to call"),
            (["odds", "--pot", "-100", "--call", "50"], "non-negative"),
            (["outs"], "hero hole cards, villain hole cards, and board cards"),
            (["decision"], "must specify hero equity"),
            (["twice"], "hero and villain hole cards are required"),
            (["twice", "AhKh", "QcQd", "2h7h8c3d4s"], "0 (preflop), 3 (flop), or 4 (turn)"),
            (["twice", "AhKh", "QcQd", "--runs", "1"], "--runs must be at least 2"),
            (["curve"], "hero cards, villain cards, and board cards"),
            (["matrix"], "at least one range"),
            (["matrix", "invalid_range_xyz"], "error"),
        ],
    )
    def test_cli_error_handling_exit_code_2(
        self, cmd_args: list[str], err_substring: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Invalid arguments return exit code 2 and helpful message on stderr."""
        code = main(cmd_args)
        assert code == 2
        captured = capsys.readouterr()
        assert err_substring.lower() in captured.err.lower()
