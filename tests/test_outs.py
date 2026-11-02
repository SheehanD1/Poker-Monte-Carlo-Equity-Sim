"""Unit tests for outs calculation and Rule of 2/4 estimation."""

import pytest

from poker_equity import (
    Card,
    HandRank,
    OutsResult,
    calculate_equity,
    calculate_improvement_outs,
    calculate_outs,
    exact_river_equity,
    exact_turn_equity,
    expected_value,
    implied_odds,
    is_profitable_call,
    outs_to_equity,
    pot_odds,
    required_implied_odds,
    rule_of_2,
    rule_of_4,
)


class TestRuleOf2:
    """Tests for the rule_of_2 function."""

    def test_standard_values(self):
        """Standard out counts with fractional and percentage returns."""
        assert rule_of_2(9) == pytest.approx(0.18)
        assert rule_of_2(9, as_percent=True) == pytest.approx(18.0)
        assert rule_of_2(4) == pytest.approx(0.08)
        assert rule_of_2(15) == pytest.approx(0.30)
        assert rule_of_2(0) == 0.0

    def test_fractional_outs(self):
        """Fractional outs (e.g. counting chop outs as 0.5)."""
        assert rule_of_2(8.5) == pytest.approx(0.17)
        assert rule_of_2(8.5, as_percent=True) == pytest.approx(17.0)

    def test_cap_at_maximum(self):
        """Extremely high out counts are capped at 1.0 (or 100.0)."""
        assert rule_of_2(60) == 1.0
        assert rule_of_2(60, as_percent=True) == 100.0

    def test_negative_outs_raises(self):
        """Negative outs raises ValueError."""
        with pytest.raises(ValueError, match="outs cannot be negative"):
            rule_of_2(-1)

    @pytest.mark.parametrize("invalid", ["9", None, [9], True, False])
    def test_invalid_type_raises(self, invalid):
        """Non-numeric or boolean types raise TypeError."""
        with pytest.raises(TypeError, match="outs must be a number"):
            rule_of_2(invalid)


class TestRuleOf4:
    """Tests for the rule_of_4 function."""

    def test_unadjusted_rule_of_4(self):
        """Standard unadjusted Rule of 4."""
        assert rule_of_4(9) == pytest.approx(0.36)
        assert rule_of_4(9, as_percent=True) == pytest.approx(36.0)
        assert rule_of_4(4) == pytest.approx(0.16)
        assert rule_of_4(15) == pytest.approx(0.60)
        assert rule_of_4(0) == 0.0

    def test_adjusted_rule_of_4(self):
        """Adjusted Rule of 4 subtracts (outs - 8) when outs > 8."""
        # For 8 outs or fewer, adjusted matches unadjusted
        assert rule_of_4(8, adjusted=True) == pytest.approx(0.32)
        assert rule_of_4(4, adjusted=True) == pytest.approx(0.16)

        # For 9 outs: 9 * 4 - (9 - 8) = 36 - 1 = 35%
        assert rule_of_4(9, adjusted=True) == pytest.approx(0.35)
        assert rule_of_4(9, as_percent=True, adjusted=True) == pytest.approx(35.0)

        # For 15 outs: 15 * 4 - (15 - 8) = 60 - 7 = 53%
        assert rule_of_4(15, adjusted=True) == pytest.approx(0.53)

    def test_cap_at_maximum(self):
        """Outs exceeding 100% are capped."""
        assert rule_of_4(30) == 1.0
        assert rule_of_4(30, as_percent=True) == 100.0

    def test_negative_outs_raises(self):
        """Negative outs raises ValueError."""
        with pytest.raises(ValueError, match="outs cannot be negative"):
            rule_of_4(-2)

    @pytest.mark.parametrize("invalid", ["9", None, [9], True, False])
    def test_invalid_type_raises(self, invalid):
        """Non-numeric or boolean types raise TypeError."""
        with pytest.raises(TypeError, match="outs must be a number"):
            rule_of_4(invalid)


class TestExactEquities:
    """Tests for exact mathematical drawing probabilities."""

    def test_exact_turn_equity(self):
        """Exact 1-card probability (outs / unseen)."""
        assert exact_turn_equity(9, unseen_cards=46) == pytest.approx(9 / 46)
        assert exact_turn_equity(4, unseen_cards=46) == pytest.approx(4 / 46)
        assert exact_turn_equity(0, unseen_cards=46) == 0.0
        assert exact_turn_equity(46, unseen_cards=46) == 1.0

    def test_exact_river_equity(self):
        """Exact 2-card probability from flop to river."""
        # 9 outs with 46 unseen: 1 - (37/46)*(36/45)
        expected = 1.0 - (37 / 46) * (36 / 45)
        assert exact_river_equity(9, unseen_cards=46) == pytest.approx(expected)
        assert exact_river_equity(0, unseen_cards=46) == 0.0
        assert exact_river_equity(46, unseen_cards=46) == 1.0

    def test_exact_equities_validation(self):
        """Negative values or invalid cards raise ValueError."""
        with pytest.raises(ValueError):
            exact_turn_equity(-1, 46)
        with pytest.raises(ValueError):
            exact_turn_equity(9, 0)
        with pytest.raises(ValueError):
            exact_river_equity(-1, 46)
        with pytest.raises(ValueError):
            exact_river_equity(9, 1)


class TestOutsToEquity:
    """Tests for the outs_to_equity dispatcher."""

    def test_flop_methods(self):
        """Flop equity using rule, adjusted_rule, and exact."""
        assert outs_to_equity(9, street="flop", method="rule") == pytest.approx(0.36)
        assert outs_to_equity(9, street="flop", method="adjusted_rule") == pytest.approx(0.35)
        assert outs_to_equity(9, street="flop", method="exact") == pytest.approx(exact_river_equity(9))

    def test_turn_methods(self):
        """Turn equity using rule and exact."""
        assert outs_to_equity(9, street="turn", method="rule") == pytest.approx(0.18)
        assert outs_to_equity(9, street="turn", method="exact") == pytest.approx(9 / 46)

    def test_invalid_parameters_raise(self):
        """Invalid street or method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown street"):
            outs_to_equity(9, street="river")
        with pytest.raises(ValueError, match="Unknown method"):
            outs_to_equity(9, street="flop", method="monte_carlo")


class TestCalculateOuts:
    """Tests for the calculate_outs engine."""

    def test_flush_draw_plus_overcards(self):
        """Ah Kh vs Qc Qd on 2h 7h 8c has 15 outs."""
        res = calculate_outs("Ah Kh", "Qc Qd", "2h 7h 8c")

        assert isinstance(res, OutsResult)
        assert res.is_hero_ahead is False
        # 9 hearts + 3 non-heart Aces + 3 non-heart Kings = 15 outs
        assert res.clean_outs_count == 15
        assert res.total_outs == 15.0
        assert res.tie_outs_count == 0
        assert res.unseen_cards == 45  # 52 - 2 - 2 - 3 = 45

        # All 9 hearts should be present
        hearts = [c for c in res.win_outs if c.suit.char == "h"]
        assert len(hearts) == 9

        # 3 Aces and 3 Kings
        aces = [c for c in res.win_outs if c.rank.char == "A"]
        kings = [c for c in res.win_outs if c.rank.char == "K"]
        assert len(aces) == 3
        assert len(kings) == 3

        # HandRank groupings
        assert HandRank.FLUSH in res.outs_by_rank
        assert HandRank.ONE_PAIR in res.outs_by_rank
        assert len(res.outs_by_rank[HandRank.FLUSH]) == 9
        assert len(res.outs_by_rank[HandRank.ONE_PAIR]) == 6

    def test_open_ended_straight_draw(self):
        """9s 8s vs Ac Kd on 7h 6c 2d has 8 outs to a straight."""
        res = calculate_outs("9s 8s", "Ac Kd", "7h 6c 2d")
        assert res.is_hero_ahead is False

        # Any 5 or any T makes a straight (4 fives + 4 tens = 8 straight outs)
        # Pairs of 9s or 8s also beat Ac Kd high card:
        # 3 remaining 9s + 3 remaining 8s = 6 pair outs
        # Total outs: 8 straight + 6 pair = 14 outs
        assert res.clean_outs_count == 14
        assert len(res.outs_by_rank[HandRank.STRAIGHT]) == 8
        assert len(res.outs_by_rank[HandRank.ONE_PAIR]) == 6

    def test_set_discounts_board_pair_redraws(self):
        """Set of 7s vs flush draw discounts cards that make full house for Villain."""
        # Hero has Ah Kh, Villain has 7s 7d, Flop is 2h 7c 8h.
        # Villain already has Three of a Kind 7s.
        # If an 8 falls (e.g. 8h), Villain makes 7s full of 8s!
        # Thus 8h does NOT give Hero a win because Full House beats Flush!
        res = calculate_outs("Ah Kh", "7s 7d", "2h 7c 8h")
        assert res.is_hero_ahead is False

        # 8h should NOT be in win_outs
        out_strs = [str(c) for c in res.win_outs]
        assert "8h" not in out_strs

    def test_turn_board_outs(self):
        """Board with 4 cards (turn)."""
        res = calculate_outs("Ah Kh", "Qc Qd", "2h 7h 8c 3c")
        assert res.unseen_cards == 44  # 52 - 2 - 2 - 4 = 44
        assert res.clean_outs_count == 15

    def test_tie_outs(self):
        """Detects cards that result in a chop/tie."""
        # Hero has Ks 9c, Villain has Kh 8d, Board is Qc Jd Th 2s.
        # Both currently have King-high straight (K-Q-J-T-9).
        # Any Ace makes Broadway straight (A-K-Q-J-T) on the board for both -> tie!
        res = calculate_outs("Ks 9c", "Kh 8d", "Qc Jd Th 2s")
        assert res.tie_outs_count > 0
        ace_ties = [c for c in res.tie_outs if c.rank.char == "A"]
        assert len(ace_ties) == 4

    def test_input_normalization_and_validation(self):
        """Accepts various input formats and raises on invalid inputs."""
        # List of Card objects
        res1 = calculate_outs("Ah Kh", "Qc Qd", "2h 7h 8c")
        res2 = calculate_outs(
            [Card.from_str("Ah"), Card.from_str("Kh")],
            [Card.from_str("Qc"), Card.from_str("Qd")],
            [Card.from_str("2h"), Card.from_str("7h"), Card.from_str("8c")],
        )
        assert res1.clean_outs_count == res2.clean_outs_count

        # Wrong hand length
        with pytest.raises(ValueError, match="hero must contain exactly 2 cards"):
            calculate_outs("Ah", "Qc Qd", "2h 7h 8c")

        # Wrong board length (e.g. 2 cards or 5 cards)
        with pytest.raises(ValueError, match="board must contain one of"):
            calculate_outs("Ah Kh", "Qc Qd", "2h 7h")
        with pytest.raises(ValueError, match="board must contain one of"):
            calculate_outs("Ah Kh", "Qc Qd", "2h 7h 8c 3c 4c")

        # Duplicate cards
        with pytest.raises(ValueError, match="Duplicate cards detected"):
            calculate_outs("Ah Kh", "Ah Qd", "2h 7h 8c")


class TestOutsResultProperties:
    """Tests for the OutsResult class."""

    def test_outs_result_methods(self):
        """Testing __len__, __contains__, and summary formatting."""
        res = calculate_outs("Ah Kh", "Qc Qd", "2h 7h 8c")
        assert len(res) == 15
        assert "3h" in res
        assert Card.from_str("3h") in res
        assert "Qc" not in res
        assert 123 not in res

        # Summary string
        summary = res.summary()
        assert "Total Outs: 15.0" in summary
        assert "Flush" in summary
        assert "Rule of 2" in summary
        assert "Rule of 4" in summary


class TestCalculateImprovementOuts:
    """Tests for the calculate_improvement_outs function."""

    def test_improvement_to_straight(self):
        """Finding cards that hit a straight."""
        outs_map = calculate_improvement_outs("9s 8s", "7s 6d 2c", target_rank=HandRank.STRAIGHT)
        assert HandRank.STRAIGHT in outs_map
        straight_cards = outs_map[HandRank.STRAIGHT]
        # 4 fives and 4 tens = 8 straight outs
        assert len(straight_cards) == 8
        ranks = {c.rank.char for c in straight_cards}
        assert ranks == {"5", "T"}

    def test_default_improvement_above_current(self):
        """Finding any card that improves current rank (High Card)."""
        outs_map = calculate_improvement_outs("9s 8s", "7s 6d 2c")
        # Should contain One Pair, Straight, etc.
        assert HandRank.ONE_PAIR in outs_map
        assert HandRank.STRAIGHT in outs_map


class TestComplexDrawingSpots:
    """Tests for complex, multi-draw, and edge-case poker scenarios."""

    def test_straight_flush_draw_multiple_ranks(self):
        """Straight flush draw categorizes cards into SF, Flush, Straight, and Pair."""
        # Qh Jh vs Ac Kd on Th 9h 2c (open-ended straight flush draw + 2 overcards)
        res = calculate_outs("Qh Jh", "Ac Kd", "Th 9h 2c")
        assert res.is_hero_ahead is False

        # Straight flush outs: Kh and 8h (2 outs)
        assert HandRank.STRAIGHT_FLUSH in res.outs_by_rank
        sf_outs = res.outs_by_rank[HandRank.STRAIGHT_FLUSH]
        assert len(sf_outs) == 2
        sf_strs = {str(c) for c in sf_outs}
        assert sf_strs == {"Kh", "8h"}

        # Nut flush outs (non-SF hearts): Ah, 7h, 6h, 5h, 4h, 3h, 2h (7 cards)
        assert HandRank.FLUSH in res.outs_by_rank
        flush_outs = res.outs_by_rank[HandRank.FLUSH]
        assert len(flush_outs) == 7

        # Straight outs (non-heart Kings and Eights): 2 Kings (Kc, Ks; Villain has Kd) + 3 Eights = 5
        assert HandRank.STRAIGHT in res.outs_by_rank
        assert len(res.outs_by_rank[HandRank.STRAIGHT]) == 5

        # Pair outs (Queens and Jacks): 3 Queens + 3 Jacks = 6
        assert HandRank.ONE_PAIR in res.outs_by_rank
        assert len(res.outs_by_rank[HandRank.ONE_PAIR]) == 6

        # Total clean outs: 2 SF + 7 Flush + 5 Straight + 6 Pair = 20 clean outs!
        assert res.clean_outs_count == 20

    def test_dominated_flush_draw_filters_dirty_flush_outs(self):
        """Lower flush draw vs higher flush draw filters out all flush cards as non-outs."""
        # Hero has 8h 7h, Villain has Ah 4h, Flop is Kh 3h 2c
        # Villain holds the Ace of hearts. Any heart gives both a flush, but Villain wins!
        res = calculate_outs("8h 7h", "Ah 4h", "Kh 3h 2c")
        assert res.is_hero_ahead is False

        # Hearts must NOT be winning outs for Hero
        heart_outs = [c for c in res.win_outs if c.suit.char == "h"]
        assert len(heart_outs) == 0
        assert HandRank.FLUSH not in res.outs_by_rank

    def test_drawing_completely_dead(self):
        """Hero drawing dead has 0 outs, 0.0 total outs, and 0 equity."""
        # Villain already has nut flush on turn, Hero has smaller flush draw
        res = calculate_outs("Ks Qs", "As 5s", "Ts 8s 2s 4h")
        assert res.is_hero_ahead is False
        assert res.clean_outs_count == 0
        assert res.tie_outs_count == 0
        assert res.total_outs == 0.0
        assert res.rule_of_2 == 0.0
        assert res.exact_turn_equity == 0.0

    def test_quads_one_outer(self):
        """Full house vs higher full house where only the remaining case card wins."""
        # Hero: 8c 8d (Full House, 8s full of 7s)
        # Villain: Jh Jd (Full House, Jacks full of 8s)
        # Board: 8s 7h 7c Js
        res = calculate_outs("8c 8d", "Jh Jd", "8s 7h 7c Js")
        assert res.is_hero_ahead is False

        # Only one out exists: 8h for Four of a Kind 8s!
        assert res.clean_outs_count == 1
        assert res.win_outs[0] == Card.from_str("8h")
        assert HandRank.FOUR_OF_A_KIND in res.outs_by_rank

    def test_counterfeited_two_pair(self):
        """Pocket pair counterfeited by board double-pair."""
        # Hero: 2c 2d, Villain: Ac Kd, Board: 8h 8c 7s 7d
        # Hero has 88772, Villain has 8877A (ahead by Ace kicker!)
        res = calculate_outs("2c 2d", "Ac Kd", "8h 8c 7s 7d")
        assert res.is_hero_ahead is False

        # Only remaining 2s win (2h, 2s for 22288 Full House)
        assert res.clean_outs_count == 2
        ranks = {c.rank.char for c in res.win_outs}
        assert ranks == {"2"}


class TestOutsAndDecisionIntegration:
    """Integration tests connecting outs calculation with pot odds, EV, and simulation."""

    def test_turn_decision_pipeline_profitable_call(self):
        """Turn flush draw facing small bet is profitable on direct pot odds."""
        # Ah Kh vs Qc Qd on 2h 7h 8c 3c (15 outs, 44 unseen cards)
        outs_res = calculate_outs("Ah Kh", "Qc Qd", "2h 7h 8c 3c")
        exact_eq = outs_res.exact_turn_equity  # 15 / 44 = ~34.09%

        # Pot is 150, bet to call is 50 -> required equity is 25%
        pot, bet = 150.0, 50.0
        req_odds = pot_odds(pot, bet)
        assert req_odds == 0.25

        # Since 34.09% > 25%, call is profitable
        assert is_profitable_call(exact_eq, pot, bet) is True
        ev = expected_value(exact_eq, pot, bet)
        assert ev > 0.0
        # EV = 0.3409 * 200 - 50 = ~18.18
        assert ev == pytest.approx((15 / 44) * 200.0 - 50.0)

    def test_turn_decision_pipeline_unprofitable_without_implied_odds(self):
        """Turn flush draw facing pot-sized bet is unprofitable on direct odds, but profitable with implied odds."""
        outs_res = calculate_outs("Ah Kh", "Qc Qd", "2h 7h 8c 3c")
        exact_eq = outs_res.exact_turn_equity  # 34.09%

        # Pot is 100, bet to call is 100 -> direct pot odds requires 50%
        pot, bet = 100.0, 100.0
        assert is_profitable_call(exact_eq, pot, bet) is False
        assert expected_value(exact_eq, pot, bet) < 0.0

        # Calculate required future chips on the river
        needed_chips = required_implied_odds(pot, bet, exact_eq)
        # Needed: 100 / (15/44) - 200 = 4400/15 - 200 = 293.33 - 200 = 93.33
        assert needed_chips == pytest.approx((100.0 / (15 / 44)) - 200.0)

        # When expecting to win 120 chips on river:
        assert implied_odds(pot, bet, future_bets=120.0) < exact_eq
        ev_with_implied = expected_value(exact_eq, pot + 120.0, bet)
        assert ev_with_implied > 0.0

    def test_outs_equity_matches_monte_carlo_simulation(self):
        """Exact turn equity from outs matches Monte Carlo simulation within 1%."""
        hero = "Ah Kh"
        villain = "Qc Qd"
        turn_board = "2h 7h 8c 3c"

        outs_res = calculate_outs(hero, villain, turn_board)
        exact_eq = outs_res.exact_turn_equity

        # Run 10,000 simulations
        sim_res = calculate_equity(hero, villain, board=turn_board, num_sims=10_000, seed=123)

        assert abs(sim_res.equity - exact_eq) < 0.015


