"""Unit and integration tests for the equity curve and street transitions module."""

import json
import pytest

from poker_equity import (
    Card,
    EquityCurve,
    NextStreetDistribution,
    Street,
    StreetPoint,
    calculate_equity_curve,
    calculate_next_street_distribution,
    calculate_street_trajectory,
    simulate_equity_curve,
)


class TestStreetEnum:
    """Tests for the Street enumeration."""

    def test_enum_values_and_str(self):
        """String conversion and enum values."""
        assert Street.PREFLOP.value == "preflop"
        assert str(Street.PREFLOP) == "PREFLOP"
        assert Street.FLOP.value == "flop"
        assert str(Street.FLOP) == "FLOP"
        assert Street.TURN.value == "turn"
        assert str(Street.TURN) == "TURN"
        assert Street.RIVER.value == "river"
        assert str(Street.RIVER) == "RIVER"

    def test_cards_count(self):
        """Number of community board cards per street."""
        assert Street.PREFLOP.cards_count == 0
        assert Street.FLOP.cards_count == 3
        assert Street.TURN.cards_count == 4
        assert Street.RIVER.cards_count == 5

    def test_next_and_prev_street(self):
        """Bidirectional navigation between consecutive streets."""
        assert Street.PREFLOP.next_street == Street.FLOP
        assert Street.FLOP.next_street == Street.TURN
        assert Street.TURN.next_street == Street.RIVER
        assert Street.RIVER.next_street is None

        assert Street.RIVER.prev_street == Street.TURN
        assert Street.TURN.prev_street == Street.FLOP
        assert Street.FLOP.prev_street == Street.PREFLOP
        assert Street.PREFLOP.prev_street is None

    @pytest.mark.parametrize(
        "length,expected",
        [
            (0, Street.PREFLOP),
            (3, Street.FLOP),
            (4, Street.TURN),
            (5, Street.RIVER),
        ],
    )
    def test_from_board_length_valid(self, length, expected):
        """Mapping valid card counts to streets."""
        assert Street.from_board_length(length) == expected

    @pytest.mark.parametrize("invalid_len", [-1, 1, 2, 6, 7])
    def test_from_board_length_invalid(self, invalid_len):
        """Invalid card counts raise ValueError."""
        with pytest.raises(ValueError, match="Invalid board length"):
            Street.from_board_length(invalid_len)


class TestStreetPointProperties:
    """Tests for StreetPoint dataclass helper properties."""

    def test_properties_and_hand_description(self):
        """hero_equity, villain_equity, deltas, and hand description."""
        pt = StreetPoint(
            street=Street.FLOP,
            board=tuple(),
            equities=(0.70, 0.30),
            hand_results=(None, None),
            delta_equities=(0.15, -0.15),
            leader_idx=0,
            leader_equity=0.70,
        )

        assert pt.hero_equity == 0.70
        assert pt.villain_equity == 0.30
        assert pt.hero_delta == 0.15
        assert pt.villain_delta == -0.15
        assert pt.hand_description(0) == "Hole cards"


class TestCalculateEquityCurveFullRunout:
    """Tests for full 4-street equity curve across Preflop -> Flop -> Turn -> River."""

    def test_aa_vs_kk_dry_runout(self):
        """AA dominates KK throughout on a dry runout."""
        curve = calculate_equity_curve(
            hands=["Ah As", "Kh Ks"],
            board="2h 7c 8d 3s 4c",
            num_sims=1000,
            seed=42,
        )

        assert isinstance(curve, EquityCurve)
        assert len(curve.points) == 4
        assert curve.streets == (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)

        # Equities sum to 1.0 at every street
        for pt in curve.points:
            assert sum(pt.equities) == pytest.approx(1.0)

        # Preflop favorite
        assert curve.hero_equities[0] > 0.75
        # Flop favorite
        assert curve.hero_equities[1] > 0.80
        # Turn favorite
        assert curve.hero_equities[2] > 0.85
        # River showdown: AA wins outright (1.0 vs 0.0)
        assert curve.hero_equities[3] == pytest.approx(1.0)
        assert curve.villain_equities[3] == pytest.approx(0.0)

        assert curve.num_lead_changes == 0
        assert curve.is_bad_beat is False
        assert curve.final_winner_idx == 0

    def test_heads_up_trajectory_convenience(self):
        """calculate_street_trajectory matches calculate_equity_curve."""
        curve = calculate_street_trajectory(
            hero="Ah As",
            villain="Kh Ks",
            board="2h 7c 8d 3s 4c",
            num_sims=500,
            seed=42,
        )
        assert len(curve.points) == 4
        assert curve.final_winner_idx == 0


class TestEquityCurvePartialBoards:
    """Tests for partial board evaluations (preflop-only, flop-only, turn-only)."""

    def test_preflop_only(self):
        """Empty board evaluates only preflop."""
        curve = calculate_equity_curve(hands=["Ah Kh", "Qc Qd"], board=None, num_sims=500, seed=42)
        assert len(curve.points) == 1
        assert curve.streets == (Street.PREFLOP,)
        assert curve.num_lead_changes == 0
        assert curve.max_swing == 0.0
        assert curve.turning_point is None

    def test_flop_only(self):
        """3-card board evaluates Preflop and Flop."""
        curve = calculate_equity_curve(
            hands=["Ah Kh", "Qc Qd"], board="2h 7h 8c", num_sims=500, seed=42
        )
        assert len(curve.points) == 2
        assert curve.streets == (Street.PREFLOP, Street.FLOP)
        assert curve.turning_point is not None
        assert curve.turning_point.street == Street.FLOP

    def test_turn_only(self):
        """4-card board evaluates Preflop, Flop, and Turn."""
        curve = calculate_equity_curve(
            hands=["Ah Kh", "Qc Qd"], board="2h 7h 8c 3c", num_sims=500, seed=42
        )
        assert len(curve.points) == 3
        assert curve.streets == (Street.PREFLOP, Street.FLOP, Street.TURN)


class TestBadBeatDetection:
    """Tests for identifying bad beats and dramatic equity swings."""

    def test_flop_favorite_river_crack(self):
        """Hero flops two pair (>90% favorite), Villain hits set/full house on river."""
        # Hero: Ah Kh, Villain: Qc Qd
        # Board: Ac Kd 2s 2c Qh
        # Flop Ac Kd 2s: Hero has top two pair (AA KK)
        # Turn 2c: Hero still has top two pair with KK kicker
        # River Qh: Villain hits Queen for Queens full of Twos
        curve = calculate_equity_curve(
            hands=["Ah Kh", "Qc Qd"],
            board="Ac Kd 2s 2c Qh",
            num_sims=1000,
            seed=42,
        )

        assert curve.is_bad_beat is True
        assert curve.final_winner_idx == 1  # Villain won
        assert curve.num_lead_changes >= 1
        assert curve.max_swing > 0.80
        assert curve.turning_point is not None
        assert curve.turning_point.street == Street.RIVER

    def test_uncontested_favorite_not_bad_beat(self):
        """Wire-to-wire leader is not a bad beat."""
        curve = calculate_equity_curve(
            hands=["Ah As", "7c 2d"],
            board="Ac Kd 2s 3h 4d",
            num_sims=500,
            seed=42,
        )
        assert curve.is_bad_beat is False
        assert curve.final_winner_idx == 0


class TestMultiWayEquityCurve:
    """Tests for multi-player equity curves."""

    def test_three_players_curve(self):
        """3 players tracked across all 4 streets."""
        curve = calculate_equity_curve(
            hands=["Ah As", "Kh Ks", "Qh Qs"],
            board="2h 7h 8c 3c 4c",
            num_sims=500,
            seed=42,
        )

        assert len(curve.hands) == 3
        for pt in curve.points:
            assert len(pt.equities) == 3
            assert sum(pt.equities) == pytest.approx(1.0)
            assert len(pt.delta_equities) == 3

        # Player equities accessor
        p3_eqs = curve.player_equities(2)
        assert len(p3_eqs) == 4

        summary = curve.summary()
        assert "Equity Curve Analysis" in summary
        assert "Player 1" in summary


class TestSimulateEquityCurve:
    """Tests for simulate_equity_curve completing random runouts."""

    def test_simulate_from_preflop_completes_five_cards(self):
        """Simulate curve with no board generates a full 5-card runout."""
        curve = simulate_equity_curve(
            hands=["Ah As", "Kh Ks"],
            board=None,
            num_sims=500,
            seed=123,
        )

        assert len(curve.points) == 4
        assert len(curve.board) == 5
        assert curve.streets[-1] == Street.RIVER

    def test_simulate_from_flop_preserves_flop_and_adds_two(self):
        """Simulate curve preserving existing flop and dealing turn + river."""
        flop_board = "2h 7h 8c"
        curve = simulate_equity_curve(
            hands=["Ah As", "Kh Ks"],
            board=flop_board,
            num_sims=500,
            seed=123,
        )

        assert len(curve.points) == 4
        assert len(curve.board) == 5
        # The first 3 cards must match the original flop
        assert str(curve.board[0]) == "2h"
        assert str(curve.board[1]) == "7h"
        assert str(curve.board[2]) == "8c"


class TestNextStreetDistribution:
    """Tests for next-street card distribution calculations."""

    def test_turn_to_river_distribution(self):
        """On turn with 15 outs, exactly 15 cards are favorable."""
        dist = calculate_next_street_distribution(
            hero="Ah Kh",
            villain="Qc Qd",
            board="2h 7h 8c 3c",
        )

        assert isinstance(dist, NextStreetDistribution)
        assert dist.current_street == Street.TURN
        assert dist.next_street == Street.RIVER
        assert dist.unseen_cards_count == 44
        assert len(dist.equities_by_card) == 44

        # Hero has 15 outs (flush + overcard Aces and Kings)
        assert dist.favorable_count == 15
        assert dist.unfavorable_count == 29
        assert dist.min_hero_equity == 0.0
        assert dist.max_hero_equity == 1.0
        assert dist.mean_hero_equity == pytest.approx(15.0 / 44.0, abs=1e-4)

        summary = dist.summary()
        assert "Next Street Distribution" in summary
        assert "Unseen cards evaluated : 44" in summary
        assert "Favorable Cards" in summary

    def test_flop_to_turn_distribution(self):
        """On flop, all 45 unseen turn cards are evaluated."""
        dist = calculate_next_street_distribution(
            hero="Ah Kh",
            villain="Qc Qd",
            board="2h 7h 8c",
            num_sims_per_card=200,
            seed=42,
        )

        assert dist.current_street == Street.FLOP
        assert dist.next_street == Street.TURN
        assert dist.unseen_cards_count == 45
        assert len(dist.equities_by_card) == 45
        assert dist.favorable_count + dist.unfavorable_count + dist.neutral_count == 45


class TestEquityCurveFormattingAndSerialization:
    """Tests for summary, to_dict, and plot_data methods."""

    def test_summary_and_serialization(self):
        """summary, to_dict, and plot_data format cleanly."""
        curve = calculate_equity_curve(
            hands=["Ah As", "Kh Ks"],
            board="2h 7h 8c 3c 4c",
            num_sims=500,
            seed=42,
        )

        summary = curve.summary()
        assert "PREFLOP" in summary
        assert "FLOP" in summary
        assert "TURN" in summary
        assert "RIVER" in summary
        assert "Lead Changes" in summary
        assert "Max Swing" in summary

        data_dict = curve.to_dict()
        assert "hands" in data_dict
        assert "points" in data_dict
        assert len(data_dict["points"]) == 4
        # Verify JSON serializability
        json_str = json.dumps(data_dict)
        assert "PREFLOP" not in json_str  # lower-cased in dict
        assert "preflop" in json_str

        plot_dict = curve.plot_data()
        assert plot_dict["labels"] == ["PREFLOP", "FLOP", "TURN", "RIVER"]
        assert len(plot_dict["datasets"]) == 2


class TestEquityCurveValidationErrors:
    """Tests for input validation and exception handling."""

    def test_fewer_than_two_players_raises(self):
        """At least 2 hands required."""
        with pytest.raises(ValueError, match="At least 2 players are required"):
            calculate_equity_curve(hands=["Ah As"])

    def test_invalid_hand_length_raises(self):
        """Hand not having 2 cards raises ValueError."""
        with pytest.raises(ValueError, match="must have exactly 2 cards"):
            calculate_equity_curve(hands=["Ah", "Kh Ks"])

    def test_duplicate_cards_raises(self):
        """Duplicate cards between hand and board raises ValueError."""
        with pytest.raises(ValueError, match="Duplicate card detected"):
            calculate_equity_curve(hands=["Ah As", "Ah Ks"], board="2h 7h 8c")
        with pytest.raises(ValueError, match="Duplicate card detected"):
            calculate_equity_curve(hands=["Ah As", "Kh Ks"], board="Ah 7h 8c")

    def test_invalid_board_length_raises(self):
        """Board with 1, 2, or 6 cards raises ValueError."""
        with pytest.raises(ValueError, match="Board must have 0.*3.*4.*or 5"):
            calculate_equity_curve(hands=["Ah As", "Kh Ks"], board="2h")
        with pytest.raises(ValueError, match="Board must have 0.*3.*4.*or 5"):
            calculate_equity_curve(hands=["Ah As", "Kh Ks"], board="2h 7h")

    def test_next_street_invalid_board_raises(self):
        """calculate_next_street_distribution requires 3 or 4 cards."""
        with pytest.raises(ValueError, match="requires 3.*or 4"):
            calculate_next_street_distribution("Ah As", "Kh Ks", board="")
        with pytest.raises(ValueError, match="requires 3.*or 4"):
            calculate_next_street_distribution("Ah As", "Kh Ks", board="2h 7h 8c 3c 4c")

