"""Unit and integration tests for the poker-equity CLI module."""

import json
import pytest

from poker_equity import Card, Rank, Suit, cli_main, create_parser
from poker_equity.cli import (
    Colors,
    colorize,
    format_card_colored,
    format_cards_colored,
    get_hand_type_description,
    main,
    make_equity_bar,
    pad_visible,
    parse_cards_flexible,
    parse_player_hand_input,
    should_enable_color,
    strip_ansi,
    visible_length,
)


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


class TestCliFormattingHelpers:
    """Tests for visual alignment, card parsing, and table helpers."""

    def test_strip_ansi(self):
        """strip_ansi removes ANSI color escape sequences."""
        colored = f"{Colors.RED}Ace of Hearts{Colors.RESET}"
        assert strip_ansi(colored) == "Ace of Hearts"

    def test_visible_length(self):
        """visible_length ignores ANSI codes when measuring length."""
        colored = f"{Colors.RED}A♥{Colors.RESET}"
        assert visible_length(colored) == 2
        assert visible_length("Hello") == 5

    def test_pad_visible(self):
        """pad_visible correctly aligns text containing ANSI escapes."""
        colored = f"{Colors.GREEN}75.0%{Colors.RESET}"
        # Visible length is 5; padded to width 10
        left = pad_visible(colored, 10, align="left")
        assert visible_length(left) == 10
        assert left.endswith("     ")

        right = pad_visible(colored, 10, align="right")
        assert visible_length(right) == 10
        assert right.startswith("     ")

        center = pad_visible(colored, 11, align="center")
        assert visible_length(center) == 11

    def test_parse_cards_flexible(self):
        """parse_cards_flexible parses various card string formats."""
        assert parse_cards_flexible(None) == []
        assert parse_cards_flexible("") == []
        # Space-separated
        cards = parse_cards_flexible("Ah Kd Qs")
        assert len(cards) == 3
        # Consecutive 2-character tokens
        cards2 = parse_cards_flexible("AhKd")
        assert len(cards2) == 2
        assert cards2[0] == Card.from_str("Ah")
        assert cards2[1] == Card.from_str("Kd")

    def test_parse_player_hand_input(self):
        """parse_player_hand_input normalizes 2-card tuples, ranges, and random."""
        assert parse_player_hand_input(None) is None
        assert parse_player_hand_input("random") is None
        assert parse_player_hand_input("?") is None
        assert parse_player_hand_input("any") is None

        # Two-card combo
        hand = parse_player_hand_input("AhKh")
        assert isinstance(hand, tuple) and len(hand) == 2

        # Range string
        rng = parse_player_hand_input("QQ+")
        assert rng == "QQ+"

    def test_format_card_colored(self):
        """format_card_colored renders card with suit symbol and color."""
        c = Card.from_str("Ah")
        plain = format_card_colored(c, enable=False)
        assert plain == "A♥"

        colored = format_card_colored(c, enable=True)
        assert Colors.HEART in colored
        assert "A♥" in colored

        char_suit = format_card_colored(c, enable=False, use_symbol=False)
        assert char_suit == "Ah"

    def test_format_cards_colored(self):
        """format_cards_colored formats a sequence of cards."""
        text = format_cards_colored("Ah Kd", enable=False)
        assert text == "A♥ K♦"

    def test_make_equity_bar(self):
        """make_equity_bar generates visual progress bar with correct width."""
        bar = make_equity_bar(0.50, width=10, enable=False)
        assert bar == "[█████░░░░░]"
        assert len(bar) == 12

        bar_col = make_equity_bar(0.80, width=10, enable=True)
        assert visible_length(bar_col) == 12

    def test_get_hand_type_description(self):
        """get_hand_type_description provides descriptive names."""
        c_ah = Card.from_str("Ah")
        c_kh = Card.from_str("Kh")
        c_qs = Card.from_str("Qs")
        c_qd = Card.from_str("Qd")
        c_7c = Card.from_str("7c")
        c_2d = Card.from_str("2d")

        assert get_hand_type_description((c_qs, c_qd)) == "Pocket Queens"
        assert get_hand_type_description((c_ah, c_kh)) == "Ace-King suited"
        assert get_hand_type_description((c_7c, c_2d)) == "Seven-Two offsuit"
        assert get_hand_type_description(None) == "Random (100%)"
        assert get_hand_type_description("QQ+") == "Range (QQ+)"

        # Postflop made hand
        board = [Card.from_str("As"), Card.from_str("Ks"), Card.from_str("2c")]
        assert "Two Pair" in get_hand_type_description((c_ah, c_kh), board)


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

    def test_positional_hands_heads_up(self, capsys):
        """Run equity simulation with positional hands syntax."""
        code = main(["equity", "AhKh", "QsQd", "--sims", "500", "--seed", "42"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Poker Monte Carlo Equity Simulation" in captured.out
        assert "Hero" in captured.out
        assert "Villain" in captured.out
        assert "Ace-King suited" in captured.out
        assert "Pocket Queens" in captured.out

    def test_positional_single_hand_random_villain(self, capsys):
        """Single positional hand simulates against random villain."""
        code = main(["equity", "AhKh", "--sims", "500", "--seed", "42"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Random" in captured.out

    def test_named_hero_random_villain(self, capsys):
        """--villain random is handled gracefully."""
        code = main(["equity", "--hero", "AhKh", "--villain", "random", "--sims", "500", "--seed", "42"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Random" in captured.out

    def test_postflop_board_made_hand(self, capsys):
        """Postflop board evaluates and describes made hand."""
        code = main(["equity", "AhKh", "QsQd", "-b", "As 7h 2c", "--sims", "500", "--seed", "42"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Pair of Aces" in captured.out
        assert "Pair of Queens" in captured.out

    def test_convergence_threshold_stops_early(self, capsys):
        """--converge flag halts simulation early once CI narrows."""
        code = main(["equity", "AhKh", "QsQd", "-c", "0.02", "-n", "20000", "-s", "42", "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["converged"] is True
        assert data["sims_run"] < 20000

    def test_parallel_simulation(self, capsys):
        """--parallel runs across worker processes."""
        code = main(["equity", "AhKh", "QsQd", "--parallel", "2", "-n", "500", "-s", "42"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Hero" in captured.out

    def test_progress_flag(self, capsys):
        """--progress executes simulation and outputs progress to stderr."""
        code = main(["equity", "AhKh", "QsQd", "--progress", "-n", "500", "-s", "42"])
        assert code == 0

    def test_duplicate_cards_error(self, capsys):
        """Passing duplicate cards outputs error to stderr and returns code 2."""
        code = main(["equity", "AhKh", "AhKd"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Duplicate card" in captured.err

    def test_invalid_board_error(self, capsys):
        """Invalid board format outputs error to stderr and returns code 2."""
        code = main(["equity", "AhKh", "QsQd", "-b", "invalid_board"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Error parsing board cards" in captured.err


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

