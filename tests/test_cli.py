"""Unit and integration tests for the poker-equity CLI module."""

import json
import pytest

from poker_equity import cli_main, create_parser
from poker_equity.cli import Colors, colorize, main, should_enable_color


class TestCliParser:
    """Tests for CLI argument parser configuration."""

    def test_create_parser_structure(self):
        """Parser has required flags and subcommands."""
        parser = create_parser()
        assert parser.prog == "poker-equity"

        # Check subcommands are present
        subparsers_actions = [
            action for action in parser._actions if action.dest == "command"
        ]
        assert len(subparsers_actions) == 1
        choices = subparsers_actions[0].choices
        for expected in ("equity", "range", "odds", "outs", "decision", "table", "curve"):
            assert expected in choices

    def test_main_no_args_prints_help(self, capsys):
        """Invoking without subcommands prints help and exits with 0."""
        code = main([])
        assert code == 0
        captured = capsys.readouterr()
        assert "usage: poker-equity" in captured.out
        assert "subcommands" in captured.out


class TestCliColors:
    """Tests for colorize and terminal formatting helpers."""

    def test_colorize_when_enabled(self):
        """colorize wraps text in ANSI codes."""
        text = colorize("Hero", Colors.GREEN, enable=True)
        assert text == f"{Colors.GREEN}Hero{Colors.RESET}"

    def test_colorize_when_disabled(self):
        """colorize returns plain text when enable=False."""
        text = colorize("Hero", Colors.GREEN, enable=False)
        assert text == "Hero"

    def test_should_enable_color_respects_no_color_flag(self):
        """--no-color flag disables color."""
        assert should_enable_color(no_color_flag=True) is False


class TestCliEquityCommand:
    """Tests for 'equity' subcommand."""

    def test_heads_up_equity(self, capsys):
        """Calculate equity for heads-up matchup."""
        code = main(["equity", "--hero", "Ah As", "--villain", "Kh Ks", "--sims", "500", "--seed", "42"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Poker Monte Carlo Equity Simulation" in captured.out
        assert "Hero" in captured.out
        assert "8" in captured.out  # ~81% equity

    def test_multiway_equity(self, capsys):
        """Calculate equity for 3 players."""
        code = main(["equity", "-p", "Ah As", "Kh Ks", "Qh Qs", "--sims", "500", "--seed", "42"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Player 1" in captured.out
        assert "Player 2" in captured.out
        assert "Player 3" in captured.out

    def test_equity_json_output(self, capsys):
        """--json flag outputs valid structured JSON."""
        code = main(["equity", "--hero", "Ah As", "--villain", "Kh Ks", "--sims", "500", "--seed", "42", "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "players" in data
        assert len(data["players"]) == 2
        assert data["players"][0]["equity"] > 0.70

    def test_missing_players_returns_error(self, capsys):
        """Missing player arguments returns error code 2."""
        code = main(["equity"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Error: Must specify either --hero and --villain or --players" in captured.err


class TestCliOddsCommand:
    """Tests for 'odds' subcommand."""

    def test_pot_odds_basic(self, capsys):
        """Calculate pot odds without equity."""
        code = main(["odds", "--pot", "100", "--call", "50"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Pot Odds & EV Analysis" in captured.out
        assert "33.33%" in captured.out

    def test_pot_odds_with_ev_call(self, capsys):
        """Calculate EV and recommendation when equity provided."""
        code = main(["odds", "--pot", "100", "--call", "50", "--equity", "0.40"])
        assert code == 0
        captured = capsys.readouterr()
        assert "+10.00 chips" in captured.out
        assert "CALL" in captured.out

    def test_odds_json_output(self, capsys):
        """--json output for odds subcommand."""
        code = main(["odds", "--pot", "100", "--call", "50", "--equity", "0.40", "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["pot_size"] == 100.0
        assert data["bet_to_call"] == 50.0
        assert data["ev"] == pytest.approx(10.0)


class TestCliOutsCommand:
    """Tests for 'outs' subcommand."""

    def test_turn_outs_calculation(self, capsys):
        """Calculate 15 outs on the turn."""
        code = main(["outs", "--hero", "Ah Kh", "--villain", "Qc Qd", "--board", "2h 7h 8c 3c"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Drawing Outs Calculation" in captured.out
        assert "Total Outs : 15.0" in captured.out

    def test_outs_json_output(self, capsys):
        """--json output for outs subcommand."""
        code = main(["outs", "--hero", "Ah Kh", "--villain", "Qc Qd", "--board", "2h 7h 8c 3c", "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_outs"] == 15.0
        assert data["win_outs"] == 15
        assert len(data["win_cards"]) == 15


class TestCliDecisionCommand:
    """Tests for 'decision' subcommand."""

    def test_decision_recommendation(self, capsys):
        """Decision recommendation for profitable call."""
        code = main(["decision", "--equity", "0.35", "--pot", "150", "--call", "50"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Recommended Action: CALL" in captured.out
        assert "+20.00 chips" in captured.out

    def test_decision_json_output(self, capsys):
        """--json output for decision subcommand."""
        code = main(["decision", "--equity", "0.35", "--pot", "150", "--call", "50", "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["recommended_action"] == "call"
        assert data["ev"] == pytest.approx(20.0)


class TestCliTableCommand:
    """Tests for 'table' subcommand."""

    def test_table_summary(self, capsys):
        """Display heads-up table summary."""
        code = main(["table", "--blinds", "1/2", "--hero-stack", "150", "--villain-stack", "200"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Table State - PREFLOP" in captured.out
        assert "Hero" in captured.out
        assert "Villain" in captured.out

    def test_table_invalid_blinds_format(self, capsys):
        """Invalid blinds format returns error code 2."""
        code = main(["table", "--blinds", "invalid"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Error: Blinds must be in 'SB/BB' format" in captured.err

    def test_table_json_output(self, capsys):
        """--json output for table subcommand."""
        code = main(["table", "--blinds", "1/2", "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["small_blind"] == 1.0
        assert data["big_blind"] == 2.0
        assert len(data["players"]) == 2


class TestCliCurveCommand:
    """Tests for 'curve' subcommand."""

    def test_curve_summary(self, capsys):
        """Equity curve across full board."""
        code = main([
            "curve",
            "--hero", "Ah As",
            "--villain", "Kh Ks",
            "--board", "2h 7h 8c 3c 4c",
            "--sims", "200",
        ])
        assert code == 0
        captured = capsys.readouterr()
        assert "Equity Curve Analysis" in captured.out
        assert "PREFLOP" in captured.out
        assert "RIVER" in captured.out

    def test_curve_json_output(self, capsys):
        """--json output for curve subcommand."""
        code = main([
            "curve",
            "--hero", "Ah As",
            "--villain", "Kh Ks",
            "--board", "2h 7h 8c 3c 4c",
            "--sims", "200",
            "--json",
        ])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "points" in data
        assert len(data["points"]) == 4


class TestCliExportAndEntry:
    """Tests for cli_main export from top-level package."""

    def test_cli_main_exported(self, capsys):
        """cli_main is callable and functional."""
        code = cli_main(["odds", "--pot", "50", "--call", "25", "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["pot_size"] == 50.0

