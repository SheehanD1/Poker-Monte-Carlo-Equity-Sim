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
    parse_range_or_preset,
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

    def test_parse_range_or_preset(self):
        """parse_range_or_preset handles hands, ranges, percentages, and presets."""
        hand = parse_range_or_preset("AhKh")
        assert isinstance(hand, tuple) and len(hand) == 2

        r1 = parse_range_or_preset("QQ+, AKs")
        assert len(r1) == 22

        r2 = parse_range_or_preset("15%")
        assert len(r2) > 0

        r3 = parse_range_or_preset("BB_VS_BTN_3BET")
        assert len(r3) > 0

        with pytest.raises(ValueError):
            parse_range_or_preset("")


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


class TestCliRangeCommand:
    """Tests for 'range' subcommand."""

    def test_hand_vs_range_positional(self, capsys):
        """Simulate hand vs range with positional syntax."""
        code = main(["range", "AhKh", "TT+, AJs+, KQs", "-n", "500", "-s", "42"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Range vs Range Equity Simulation" in captured.out
        assert "Hero" in captured.out
        assert "Villain" in captured.out
        assert "TT+, AJs+, KQs" in captured.out

    def test_range_vs_range_named(self, capsys):
        """Simulate range vs range using --hero and --villain."""
        code = main([
            "range",
            "--hero", "QQ+, AKs",
            "--villain", "TT+, AJs+, KQs",
            "-n", "500",
            "-s", "42",
        ])
        assert code == 0
        captured = capsys.readouterr()
        assert "QQ+, AKs" in captured.out
        assert "TT+, AJs+, KQs" in captured.out

    def test_range_with_board(self, capsys):
        """Simulate range equity with flop board."""
        code = main([
            "range",
            "AhKh",
            "TT+, AJs+",
            "-b", "As 7h 2c",
            "-n", "500",
            "-s", "42",
        ])
        assert code == 0
        captured = capsys.readouterr()
        assert "A♠ 7♥ 2♣" in captured.out
        assert "Hero" in captured.out

    def test_range_with_dead_cards(self, capsys):
        """Dead cards reduce live combos and appear in header."""
        code = main([
            "range",
            "AhKh",
            "TT+, AJs+",
            "-d", "As Ks",
            "-n", "500",
            "-s", "42",
        ])
        assert code == 0
        captured = capsys.readouterr()
        assert "Dead" in captured.out
        assert "A♠ K♠" in captured.out

    def test_range_weighted_syntax(self, capsys):
        """Weighted range tokens are handled and simulated."""
        code = main([
            "range",
            "AA:0.8, KK:0.5",
            "QQ+",
            "-n", "500",
            "-s", "42",
        ])
        assert code == 0
        captured = capsys.readouterr()
        assert "AA:0.8, KK:0.5" in captured.out

    def test_range_percentage_preset(self, capsys):
        """Percentage range string (e.g. '20%') is expanded."""
        code = main(["range", "AhKh", "20%", "-n", "500", "-s", "42"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Hero" in captured.out
        assert "Villain" in captured.out

    def test_range_named_preset(self, capsys):
        """Named preset (e.g. 'BB_VS_BTN_3BET') is expanded."""
        code = main(["range", "AhKh", "BB_VS_BTN_3BET", "-n", "500", "-s", "42"])
        assert code == 0
        captured = capsys.readouterr()
        assert "BB_VS_BTN_3BET" in captured.out

    def test_range_combo_breakdown(self, capsys):
        """--breakdown displays top combo equity rankings."""
        code = main(["range", "AhKh", "TT+, AJs+", "-k", "5", "-n", "500", "-s", "42"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Top 5 Combos in Villain Range" in captured.out
        assert "Cell" in captured.out

    def test_range_json_output(self, capsys):
        """--json output produces valid schema with hero and villain metadata."""
        code = main(["range", "AhKh", "TT+, AJs+", "--json", "-n", "500", "-s", "42"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "hero" in data
        assert "villain" in data
        assert data["hero"]["equity"] > 0
        assert data["villain"]["is_range"] is True
        assert data["villain"]["total_combos"] > 0

    def test_range_json_with_breakdown(self, capsys):
        """--json with --breakdown includes combo array."""
        code = main([
            "range",
            "AhKh",
            "TT+, AJs+",
            "--json",
            "-k", "3",
            "-n", "500",
            "-s", "42",
        ])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "breakdown" in data
        assert len(data["breakdown"]) == 3
        assert "combo" in data["breakdown"][0]
        assert "cell" in data["breakdown"][0]
        assert "equity" in data["breakdown"][0]

    def test_range_convergence_threshold(self, capsys):
        """--converge stops early once CI narrows."""
        code = main([
            "range",
            "AhKh",
            "TT+, AJs+",
            "-c", "0.02",
            "-n", "20000",
            "-s", "42",
            "--json",
        ])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["converged"] is True
        assert data["sims_run"] < 20000

    def test_range_parallel_execution(self, capsys):
        """--parallel runs across worker processes."""
        code = main([
            "range",
            "AhKh",
            "TT+, AJs+",
            "--parallel", "2",
            "-n", "500",
            "-s", "42",
        ])
        assert code == 0
        captured = capsys.readouterr()
        assert "Hero" in captured.out

    def test_range_progress_flag(self, capsys):
        """--progress executes simulation with real-time updates."""
        code = main([
            "range",
            "AhKh",
            "TT+, AJs+",
            "--progress",
            "-n", "500",
            "-s", "42",
        ])
        assert code == 0

    def test_range_missing_arguments_error(self, capsys):
        """Missing hero or villain ranges returns exit code 2."""
        code = main(["range"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Error: Must specify both hero and villain ranges" in captured.err

    def test_range_invalid_syntax_error(self, capsys):
        """Invalid range token returns exit code 2."""
        code = main(["range", "invalid_range_xyz", "QQ+"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Error parsing" in captured.err


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

    def test_pot_odds_positional_arguments(self, capsys):
        """Positional <pot> <call> syntax."""
        code = main(["odds", "100", "50"])
        assert code == 0
        captured = capsys.readouterr()
        assert "33.33%" in captured.out

    def test_pot_odds_positional_with_percentage_string(self, capsys):
        """Positional <pot> <call> <equity%> syntax."""
        code = main(["odds", "100", "50", "40%"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Hero Equity     : 40.00%" in captured.out
        assert "+10.00 chips" in captured.out
        assert "CALL" in captured.out

    def test_pot_odds_positional_with_fraction(self, capsys):
        """Positional <pot> <call> <fraction> syntax."""
        code = main(["odds", "100", "50", "0.40"])
        assert code == 0
        captured = capsys.readouterr()
        assert "+10.00 chips" in captured.out

    def test_pot_odds_ev_fold_recommendation(self, capsys):
        """Unprofitable spot recommends FOLD."""
        code = main(["odds", "100", "50", "20%"])
        assert code == 0
        captured = capsys.readouterr()
        assert "-20.00 chips" in captured.out
        assert "FOLD" in captured.out

    def test_pot_odds_with_implied_odds(self, capsys):
        """--future-bets calculates implied odds."""
        code = main(["odds", "100", "50", "--future-bets", "100"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Future Bets     : +100.00 chips anticipated" in captured.out
        assert "Implied Odds    : 20.00% (4.00:1)" in captured.out

    def test_pot_odds_with_reverse_implied_odds(self, capsys):
        """--future-losses calculates reverse implied odds."""
        code = main(["odds", "100", "50", "--future-losses", "50"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Future Losses   : -50.00 chips potential" in captured.out
        assert "Rev Implied Odds: 50.00%" in captured.out

    def test_pot_odds_required_implied_odds_display(self, capsys):
        """Display chips needed on later streets when behind."""
        code = main(["odds", "150", "50", "16.67%"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Req Implied Bets: +" in captured.out

    def test_odds_json_output(self, capsys):
        """--json output for odds subcommand."""
        code = main(["odds", "--pot", "100", "--call", "50", "--equity", "0.40", "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["pot_size"] == 100.0
        assert data["bet_to_call"] == 50.0
        assert data["ev"] == pytest.approx(10.0)

    def test_pot_odds_json_comprehensive(self, capsys):
        """Comprehensive JSON output with future bets and losses."""
        code = main([
            "odds", "100", "50", "40%",
            "-f", "50",
            "-l", "25",
            "--json",
        ])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["pot_size"] == 100.0
        assert data["bet_to_call"] == 50.0
        assert data["pot_odds_pct"] == pytest.approx(1 / 3)
        assert data["pot_odds_ratio"] == [2.0, 1.0]
        assert data["pot_odds_ratio_str"] == "2.00:1"
        assert data["equity"] == 0.4
        assert data["ev"] == pytest.approx(30.0)
        assert data["direct_ev"] == pytest.approx(10.0)
        assert data["is_profitable"] is True
        assert data["recommendation"] == "CALL"
        assert data["future_bets"] == 50.0
        assert data["future_losses"] == 25.0
        assert data["implied_odds_pct"] == pytest.approx(50 / 200)
        assert data["implied_odds_ratio_str"] == "3.00:1"
        assert data["required_future_bets"] == 0.0
        assert data["reverse_implied_odds_pct"] == pytest.approx(75 / 175)

    def test_pot_odds_missing_arguments_error(self, capsys):
        """Missing pot or call returns exit code 2."""
        code = main(["odds"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Error: Must specify both pot size and bet to call" in captured.err

    def test_pot_odds_negative_value_error(self, capsys):
        """Negative pot or call returns exit code 2."""
        code = main(["odds", "-10", "50"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Error: Pot size and bet to call must be non-negative" in captured.err

    def test_pot_odds_invalid_equity_error(self, capsys):
        """Invalid equity returns exit code 2."""
        code = main(["odds", "100", "50", "150%"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Error: Equity must be between 0.0 and 1.0" in captured.err

    def test_pot_odds_interactive_mode(self, monkeypatch, capsys):
        """Interactive input mode prompts for pot, call, and equity."""
        inputs = iter(["100", "50", "35%"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
        code = main(["odds", "--interactive"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Pot Odds & EV Analysis" in captured.out
        assert "Hero Equity     : 35.00%" in captured.out
        assert "+2.50 chips" in captured.out
        assert "CALL" in captured.out


class TestCliOutsCommand:
    """Tests for 'outs' subcommand."""

    def test_turn_outs_calculation(self, capsys):
        """Calculate 15 outs on the turn."""
        code = main(["outs", "--hero", "Ah Kh", "--villain", "Qc Qd", "--board", "2h 7h 8c 3c"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Drawing Outs Calculation" in captured.out
        assert "Total Outs : 15.0" in captured.out

    def test_flop_outs_positional(self, capsys):
        """Positional outs on the flop."""
        code = main(["outs", "AhKh", "QcQd", "2h 7h 8c"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Drawing Outs Calculation" in captured.out
        assert "Total Outs : 15.0" in captured.out
        assert "FLOP, 2 to come" in captured.out
        assert "Flush" in captured.out

    def test_turn_outs_positional_space_separated(self, capsys):
        """Positional 8-card sequence for hero, villain, and turn board."""
        code = main(["outs", "Ah", "Kh", "Qc", "Qd", "2h", "7h", "8c", "3c"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Total Outs : 15.0" in captured.out
        assert "TURN, 1 to come" in captured.out

    def test_outs_named_flags(self, capsys):
        """Named -H, -V, -b flags."""
        code = main(["outs", "-H", "AhKh", "-V", "QcQd", "-b", "2h7h8c"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Total Outs : 15.0" in captured.out

    def test_outs_with_dead_cards(self, capsys):
        """--dead removes known folded cards from outs."""
        code = main(["outs", "AhKh", "QcQd", "2h 7h 8c", "--dead", "Jh Th"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Total Outs : 13.0" in captured.out
        assert "Dead" in captured.out

    def test_outs_methods_rule_and_adjusted(self, capsys):
        """--method rule and --method adjusted_rule on the flop."""
        code = main(["outs", "AhKh", "QcQd", "2h 7h 8c", "-m", "rule"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Equity (rule): 60.00%" in captured.out

        code_adj = main(["outs", "AhKh", "QcQd", "2h 7h 8c", "-m", "adjusted_rule"])
        assert code_adj == 0
        captured_adj = capsys.readouterr()
        assert "Equity (adjusted_rule): 53.00%" in captured_adj.out

    def test_outs_status_ahead_and_trailing(self, capsys):
        """Hero ahead vs Hero trailing status indicator."""
        # Villain has Queens, Hero has Ace-King -> Hero trailing
        code = main(["outs", "AhKh", "QcQd", "2h 7h 8c"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Hero is currently TRAILING" in captured.out

        # Reverse hands: Hero has Queens -> Hero ahead
        code2 = main(["outs", "QcQd", "AhKh", "2h 7h 8c"])
        assert code2 == 0
        captured2 = capsys.readouterr()
        assert "Hero is currently AHEAD" in captured2.out

    def test_outs_json_output(self, capsys):
        """--json output for outs subcommand."""
        code = main(["outs", "--hero", "Ah Kh", "--villain", "Qc Qd", "--board", "2h 7h 8c 3c", "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_outs"] == 15.0
        assert data["win_outs"] == 15
        assert len(data["win_cards"]) == 15

    def test_outs_json_comprehensive(self, capsys):
        """Comprehensive JSON output with dead cards and category breakdown."""
        code = main([
            "outs", "AhKh", "QcQd", "2h 7h 8c",
            "--dead", "Jc Ts",
            "--json",
        ])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["hero"] == "Ah Kh"
        assert data["villain"] == "Qc Qd"
        assert data["board"] == "2h 7h 8c"
        assert data["dead"] == "Jc Ts"
        assert data["street"] == "flop"
        assert data["cards_to_come"] == 2
        assert data["is_hero_ahead"] is False
        assert data["total_outs"] == 15.0
        assert data["win_outs"] == 15
        assert data["tie_outs"] == 0
        assert data["unseen_cards"] == 43
        assert "Flush" in data["outs_by_rank"]
        assert len(data["win_cards"]) == 15
        assert data["rule_of_4_pct"] == 60.0
        assert data["rule_of_4_adjusted_pct"] == 53.0

    def test_outs_missing_arguments_error(self, capsys):
        """Missing card arguments returns exit code 2."""
        code = main(["outs"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Error: Must specify hero hole cards, villain hole cards, and board cards" in captured.err

    def test_outs_duplicate_cards_error(self, capsys):
        """Duplicate cards returns exit code 2."""
        code = main(["outs", "AhKh", "AhQd", "2h 7h 8c"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Duplicate cards detected" in captured.err

    def test_outs_invalid_board_length_error(self, capsys):
        """Board with wrong number of cards returns exit code 2."""
        code = main(["outs", "AhKh", "QcQd", "2h 7h"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Board must contain 3 (flop) or 4 (turn) cards" in captured.err

    def test_outs_interactive_mode(self, monkeypatch, capsys):
        """Interactive input mode prompts for hero, villain, and board."""
        inputs = iter(["Ah Kh", "Qc Qd", "2h 7h 8c"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
        code = main(["outs", "--interactive"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Drawing Outs Calculation" in captured.out
        assert "Total Outs : 15.0" in captured.out


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

    def test_decision_positional_arguments(self, capsys):
        """Positional <equity> <pot> <call> syntax."""
        code = main(["decision", "0.35", "150", "50"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Recommended Action: CALL" in captured.out
        assert "+20.00 chips" in captured.out

    def test_decision_percentage_equity(self, capsys):
        """Equity passed as percentage string."""
        code = main(["decision", "--equity", "35%", "--pot", "150", "--call", "50"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Hero Equity       : 35.0%" in captured.out
        assert "+20.00 chips" in captured.out

    def test_decision_checking_spot_default_call_zero(self, capsys):
        """Checking spot defaults --call to 0.0."""
        code = main(["decision", "--equity", "0.70", "--pot", "100"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Recommended Action: BET" in captured.out
        assert "Free check" in captured.out

    def test_decision_checking_spot_check_recommendation(self, capsys):
        """Low equity in unraised pot recommends CHECK."""
        code = main(["decision", "--equity", "0.20", "--pot", "100"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Recommended Action: CHECK" in captured.out

    def test_decision_fold_recommendation(self, capsys):
        """Negative EV call recommends FOLD."""
        code = main(["decision", "--equity", "0.10", "--pot", "150", "--call", "50"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Recommended Action: FOLD" in captured.out
        assert "0.00 chips" in captured.out

    def test_decision_raise_recommendation_with_fold_equity(self, capsys):
        """High equity with fold equity recommends RAISE."""
        code = main([
            "decision",
            "--equity", "0.60",
            "--pot", "100",
            "--call", "20",
            "--fold-equity", "30%",
        ])
        assert code == 0
        captured = capsys.readouterr()
        assert "Recommended Action: RAISE" in captured.out
        assert "Fold Equity       : 30.0%" in captured.out

    def test_decision_all_in_recommendation_short_stack(self, capsys):
        """Short stack commits all chips for ALL_IN action."""
        code = main([
            "decision",
            "--equity", "0.80",
            "--pot", "100",
            "--call", "20",
            "--stack", "30",
        ])
        assert code == 0
        captured = capsys.readouterr()
        assert "Recommended Action: ALL_IN" in captured.out
        assert "Hero Stack        : 30.00 chips" in captured.out

    def test_decision_hand_simulation_mode(self, capsys):
        """Simulate equity from hero, villain, and board cards."""
        code = main([
            "decision",
            "--hero", "AhKh",
            "--villain", "QcQd",
            "--board", "2h 7h 8c",
            "--pot", "150",
            "--call", "50",
            "--sims", "1000",
            "--seed", "42",
        ])
        assert code == 0
        captured = capsys.readouterr()
        assert "Strategic Decision Recommendation" in captured.out
        assert "Hole Cards" in captured.out
        assert "Recommended Action:" in captured.out

    def test_decision_positional_hands_simulation(self, capsys):
        """Simulate equity with positional hands syntax."""
        code = main([
            "decision", "AhKh", "QcQd", "2h 7h 8c",
            "-p", "150",
            "-c", "50",
            "-n", "1000",
            "--seed", "42",
        ])
        assert code == 0
        captured = capsys.readouterr()
        assert "Recommended Action:" in captured.out
        assert "Hole Cards" in captured.out

    def test_decision_with_future_bets_implied_odds(self, capsys):
        """Implied odds via --future-bets improves call EV."""
        code = main([
            "decision",
            "--equity", "0.20",
            "--pot", "150",
            "--call", "50",
            "--future-bets", "100",
        ])
        assert code == 0
        captured = capsys.readouterr()
        assert "Future Bets (Imp) : +100.00 chips" in captured.out

    def test_decision_ev_comparison_table_display(self, capsys):
        """Verify EV comparison table displays all evaluated actions."""
        code = main(["decision", "--equity", "0.35", "--pot", "150", "--call", "50"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Action Expected Value (EV) Comparison:" in captured.out
        assert "★ Recommended" in captured.out
        assert "CALL" in captured.out
        assert "FOLD" in captured.out

    def test_decision_json_comprehensive(self, capsys):
        """Comprehensive JSON output with stack, fold equity, and SPR."""
        code = main([
            "decision", "0.35", "150", "50", "300",
            "-fe", "25%",
            "-f", "50",
            "--json",
        ])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["recommended_action"] == "raise"
        assert data["equity"] == 0.35
        assert data["pot_odds"] == 0.25
        assert data["pot_size"] == 150.0
        assert data["bet_to_call"] == 50.0
        assert data["stack_size"] == 300.0
        assert data["fold_equity"] == 0.25
        assert data["future_bets"] == 50.0
        assert data["is_profitable"] is True
        assert data["spr"] == 2.0
        assert "fold" in data["ev_by_action"]
        assert "call" in data["ev_by_action"]
        assert "reasoning" in data

    def test_decision_missing_equity_and_hands_error(self, capsys):
        """Missing equity and cards returns exit code 2."""
        code = main(["decision", "--pot", "150"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Error: Must specify hero equity" in captured.err

    def test_decision_missing_pot_error(self, capsys):
        """Missing pot size returns exit code 2."""
        code = main(["decision", "--equity", "0.35"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Error: Must specify pot size" in captured.err

    def test_decision_negative_pot_error(self, capsys):
        """Negative pot returns exit code 2."""
        code = main(["decision", "--equity", "0.35", "--pot", "-100"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Error: Pot size and bet to call must be non-negative" in captured.err

    def test_decision_invalid_equity_error(self, capsys):
        """Invalid equity returns exit code 2."""
        code = main(["decision", "--equity", "120%", "--pot", "100"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Error: Equity must be between 0.0 and 1.0" in captured.err

    def test_decision_interactive_mode(self, monkeypatch, capsys):
        """Interactive input mode prompts for decision parameters."""
        inputs = iter(["150", "50", "35%", "200", "0%"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
        code = main(["decision", "--interactive"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Strategic Decision Recommendation" in captured.out
        assert "Recommended Action: CALL" in captured.out
        assert "+20.00 chips" in captured.out


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

