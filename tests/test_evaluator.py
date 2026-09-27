"""Unit tests for the 5-card hand evaluator."""

from __future__ import annotations

import pytest

from poker_equity.card import Card
from poker_equity.evaluator import evaluate_5
from poker_equity.hand_rank import HandRank, HandResult

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

C = Card.from_str  # shorthand for readability


def _eval(*card_strs: str) -> HandResult:
    """Evaluate a hand from string shorthand, e.g. ``_eval("Ah", "Kh", ...)``."""
    return evaluate_5([C(s) for s in card_strs])


# =========================================================================
# Royal Flush
# =========================================================================


class TestRoyalFlush:
    def test_hearts(self) -> None:
        result = _eval("Ah", "Kh", "Qh", "Jh", "Th")
        assert result.rank == HandRank.ROYAL_FLUSH
        assert result.sub_rank == ()
        assert result.description == "Royal Flush"

    def test_spades(self) -> None:
        result = _eval("As", "Ks", "Qs", "Js", "Ts")
        assert result.rank == HandRank.ROYAL_FLUSH

    def test_all_royal_flushes_tie(self) -> None:
        hearts = _eval("Ah", "Kh", "Qh", "Jh", "Th")
        spades = _eval("As", "Ks", "Qs", "Js", "Ts")
        assert hearts == spades


# =========================================================================
# Straight Flush
# =========================================================================


class TestStraightFlush:
    def test_nine_high(self) -> None:
        result = _eval("9s", "8s", "7s", "6s", "5s")
        assert result.rank == HandRank.STRAIGHT_FLUSH
        assert result.sub_rank == (9,)
        assert "Nine-high" in result.description

    def test_six_high(self) -> None:
        result = _eval("6d", "5d", "4d", "3d", "2d")
        assert result.rank == HandRank.STRAIGHT_FLUSH
        assert result.sub_rank == (6,)

    def test_wheel_straight_flush(self) -> None:
        """A-2-3-4-5 suited is the lowest straight flush (5-high)."""
        result = _eval("Ac", "2c", "3c", "4c", "5c")
        assert result.rank == HandRank.STRAIGHT_FLUSH
        assert result.sub_rank == (5,)
        assert "Five-high" in result.description

    def test_ordering_nine_beats_six(self) -> None:
        nine_high = _eval("9s", "8s", "7s", "6s", "5s")
        six_high = _eval("6d", "5d", "4d", "3d", "2d")
        assert nine_high > six_high

    def test_king_high_straight_flush_not_royal(self) -> None:
        result = _eval("Kh", "Qh", "Jh", "Th", "9h")
        assert result.rank == HandRank.STRAIGHT_FLUSH
        assert result.sub_rank == (13,)


# =========================================================================
# Four of a Kind
# =========================================================================


class TestFourOfAKind:
    def test_quad_aces(self) -> None:
        result = _eval("As", "Ah", "Ad", "Ac", "Ks")
        assert result.rank == HandRank.FOUR_OF_A_KIND
        assert result.sub_rank == (14, 13)
        assert "Aces" in result.description

    def test_quad_twos(self) -> None:
        result = _eval("2s", "2h", "2d", "2c", "As")
        assert result.rank == HandRank.FOUR_OF_A_KIND
        assert result.sub_rank == (2, 14)

    def test_quad_aces_beats_quad_kings(self) -> None:
        aces = _eval("As", "Ah", "Ad", "Ac", "2s")
        kings = _eval("Ks", "Kh", "Kd", "Kc", "As")
        assert aces > kings

    def test_same_quads_higher_kicker_wins(self) -> None:
        quads_k_kicker = _eval("As", "Ah", "Ad", "Ac", "Ks")
        quads_q_kicker = _eval("As", "Ah", "Ad", "Ac", "Qs")
        assert quads_k_kicker > quads_q_kicker


# =========================================================================
# Full House
# =========================================================================


class TestFullHouse:
    def test_kings_full_of_queens(self) -> None:
        result = _eval("Ks", "Kh", "Kd", "Qs", "Qh")
        assert result.rank == HandRank.FULL_HOUSE
        assert result.sub_rank == (13, 12)
        assert "Kings" in result.description
        assert "Queens" in result.description

    def test_twos_full_of_threes(self) -> None:
        result = _eval("2s", "2h", "2d", "3s", "3h")
        assert result.rank == HandRank.FULL_HOUSE
        assert result.sub_rank == (2, 3)

    def test_aces_full_beats_kings_full(self) -> None:
        aces_full = _eval("As", "Ah", "Ad", "2s", "2h")
        kings_full = _eval("Ks", "Kh", "Kd", "As", "Ah")
        assert aces_full > kings_full

    def test_same_trips_higher_pair_wins(self) -> None:
        trip_k_pair_q = _eval("Ks", "Kh", "Kd", "Qs", "Qh")
        trip_k_pair_j = _eval("Ks", "Kh", "Kd", "Js", "Jh")
        assert trip_k_pair_q > trip_k_pair_j


# =========================================================================
# Flush
# =========================================================================


class TestFlush:
    def test_ace_high_flush(self) -> None:
        result = _eval("Ah", "Th", "8h", "5h", "2h")
        assert result.rank == HandRank.FLUSH
        assert result.sub_rank == (14, 10, 8, 5, 2)
        assert "Ace-high" in result.description

    def test_king_high_flush(self) -> None:
        result = _eval("Kd", "Jd", "9d", "6d", "3d")
        assert result.rank == HandRank.FLUSH
        assert result.sub_rank == (13, 11, 9, 6, 3)

    def test_ace_high_beats_king_high(self) -> None:
        ace_flush = _eval("Ah", "3h", "4h", "5h", "7h")
        king_flush = _eval("Kd", "Qd", "Jd", "8d", "3d")
        assert ace_flush > king_flush

    def test_same_high_card_second_kicker_decides(self) -> None:
        flush_a = _eval("Ah", "Kh", "Th", "8h", "2h")
        flush_b = _eval("Ad", "Kd", "9d", "8d", "2d")
        assert flush_a > flush_b  # T > 9 in second position

    def test_identical_flush_ranks_tie(self) -> None:
        flush_h = _eval("Ah", "Kh", "Qh", "Jh", "9h")
        flush_d = _eval("Ad", "Kd", "Qd", "Jd", "9d")
        assert flush_h == flush_d


# =========================================================================
# Straight
# =========================================================================


class TestStraight:
    def test_ace_high_straight(self) -> None:
        result = _eval("As", "Kh", "Qd", "Jc", "Ts")
        assert result.rank == HandRank.STRAIGHT
        assert result.sub_rank == (14,)
        assert "Ace-high" in result.description

    def test_nine_high_straight(self) -> None:
        result = _eval("9s", "8h", "7d", "6c", "5s")
        assert result.rank == HandRank.STRAIGHT
        assert result.sub_rank == (9,)

    def test_wheel_straight(self) -> None:
        """A-2-3-4-5 is the lowest straight (5-high)."""
        result = _eval("As", "2h", "3d", "4c", "5s")
        assert result.rank == HandRank.STRAIGHT
        assert result.sub_rank == (5,)
        assert "Five-high" in result.description

    def test_ace_high_beats_king_high(self) -> None:
        ace_str = _eval("As", "Kh", "Qd", "Jc", "Ts")
        king_str = _eval("Ks", "Qh", "Jd", "Tc", "9s")
        assert ace_str > king_str

    def test_wheel_is_lowest_straight(self) -> None:
        wheel = _eval("As", "2h", "3d", "4c", "5s")
        six_high = _eval("6s", "5h", "4d", "3c", "2s")
        assert six_high > wheel

    def test_not_a_wraparound_straight(self) -> None:
        """K-A-2-3-4 is NOT a straight (no wraparound)."""
        result = _eval("Ks", "Ah", "2d", "3c", "4s")
        assert result.rank == HandRank.HIGH_CARD


# =========================================================================
# Three of a Kind
# =========================================================================


class TestThreeOfAKind:
    def test_trip_sevens(self) -> None:
        result = _eval("7s", "7h", "7d", "As", "Kh")
        assert result.rank == HandRank.THREE_OF_A_KIND
        assert result.sub_rank == (7, 14, 13)
        assert "Sevens" in result.description

    def test_trip_aces(self) -> None:
        result = _eval("As", "Ah", "Ad", "Ks", "Qh")
        assert result.rank == HandRank.THREE_OF_A_KIND
        assert result.sub_rank == (14, 13, 12)

    def test_higher_trips_wins(self) -> None:
        trip_a = _eval("As", "Ah", "Ad", "3s", "2h")
        trip_k = _eval("Ks", "Kh", "Kd", "As", "Qh")
        assert trip_a > trip_k

    def test_same_trips_kicker_decides(self) -> None:
        trip_7_ak = _eval("7s", "7h", "7d", "As", "Kh")
        trip_7_aq = _eval("7s", "7h", "7d", "As", "Qh")
        assert trip_7_ak > trip_7_aq


# =========================================================================
# Two Pair
# =========================================================================


class TestTwoPair:
    def test_kings_and_tens(self) -> None:
        result = _eval("Ks", "Kh", "Ts", "Th", "As")
        assert result.rank == HandRank.TWO_PAIR
        assert result.sub_rank == (13, 10, 14)
        assert "Kings" in result.description
        assert "Tens" in result.description

    def test_aces_and_twos(self) -> None:
        result = _eval("As", "Ah", "2s", "2h", "Ks")
        assert result.rank == HandRank.TWO_PAIR
        assert result.sub_rank == (14, 2, 13)

    def test_higher_top_pair_wins(self) -> None:
        aa_22 = _eval("As", "Ah", "2s", "2h", "3s")
        kk_qq = _eval("Ks", "Kh", "Qs", "Qh", "As")
        assert aa_22 > kk_qq

    def test_same_top_pair_higher_bottom_wins(self) -> None:
        aa_kk = _eval("As", "Ah", "Ks", "Kh", "2s")
        aa_qq = _eval("As", "Ah", "Qs", "Qh", "Ks")
        assert aa_kk > aa_qq

    def test_same_two_pair_kicker_decides(self) -> None:
        aa_kk_q = _eval("As", "Ah", "Ks", "Kh", "Qs")
        aa_kk_j = _eval("As", "Ah", "Ks", "Kh", "Js")
        assert aa_kk_q > aa_kk_j


# =========================================================================
# One Pair
# =========================================================================


class TestOnePair:
    def test_pair_of_aces(self) -> None:
        result = _eval("As", "Ah", "Ks", "Qh", "Jd")
        assert result.rank == HandRank.ONE_PAIR
        assert result.sub_rank == (14, 13, 12, 11)
        assert "Aces" in result.description

    def test_pair_of_twos(self) -> None:
        result = _eval("2s", "2h", "As", "Kh", "Qd")
        assert result.rank == HandRank.ONE_PAIR
        assert result.sub_rank == (2, 14, 13, 12)

    def test_higher_pair_wins(self) -> None:
        pair_a = _eval("As", "Ah", "2s", "3h", "4d")
        pair_k = _eval("Ks", "Kh", "As", "Qh", "Jd")
        assert pair_a > pair_k

    def test_same_pair_first_kicker_decides(self) -> None:
        pair_a_kqj = _eval("As", "Ah", "Ks", "Qh", "Jd")
        pair_a_kqt = _eval("As", "Ah", "Ks", "Qh", "Td")
        assert pair_a_kqj > pair_a_kqt

    def test_same_pair_identical_kickers_tie(self) -> None:
        a = _eval("As", "Ah", "Ks", "Qh", "Jd")
        b = _eval("Ad", "Ac", "Kh", "Qd", "Jc")
        assert a == b


# =========================================================================
# High Card
# =========================================================================


class TestHighCard:
    def test_ace_high(self) -> None:
        result = _eval("As", "Kh", "Qd", "Jc", "9s")
        assert result.rank == HandRank.HIGH_CARD
        assert result.sub_rank == (14, 13, 12, 11, 9)
        assert "Ace-high" in result.description

    def test_seven_high(self) -> None:
        result = _eval("7s", "5h", "4d", "3c", "2s")
        assert result.rank == HandRank.HIGH_CARD
        assert result.sub_rank == (7, 5, 4, 3, 2)

    def test_ace_high_beats_king_high(self) -> None:
        ace_high = _eval("As", "2h", "4d", "6c", "8s")
        king_high = _eval("Ks", "Qh", "Jd", "Tc", "8s")
        assert ace_high > king_high

    def test_same_high_card_second_kicker_decides(self) -> None:
        a = _eval("As", "Kh", "Qd", "Jc", "9s")
        b = _eval("As", "Kh", "Qd", "Jc", "8s")
        assert a > b

    def test_identical_ranks_tie(self) -> None:
        a = _eval("As", "Kh", "Qd", "Jc", "9s")
        b = _eval("Ad", "Kc", "Qs", "Jh", "9d")
        assert a == b


# =========================================================================
# Cross-category ordering
# =========================================================================


class TestCrossCategoryOrdering:
    """Verify that all hand categories beat the one below them."""

    def test_royal_flush_beats_straight_flush(self) -> None:
        rf = _eval("Ah", "Kh", "Qh", "Jh", "Th")
        sf = _eval("Ks", "Qs", "Js", "Ts", "9s")
        assert rf > sf

    def test_straight_flush_beats_quads(self) -> None:
        sf = _eval("9s", "8s", "7s", "6s", "5s")
        quads = _eval("As", "Ah", "Ad", "Ac", "Ks")
        assert sf > quads

    def test_quads_beats_full_house(self) -> None:
        quads = _eval("2s", "2h", "2d", "2c", "3s")
        fh = _eval("As", "Ah", "Ad", "Ks", "Kh")
        assert quads > fh

    def test_full_house_beats_flush(self) -> None:
        fh = _eval("2s", "2h", "2d", "3s", "3h")
        flush = _eval("Ah", "Kh", "Qh", "Jh", "9h")
        assert fh > flush

    def test_flush_beats_straight(self) -> None:
        flush = _eval("2h", "4h", "6h", "8h", "Th")
        straight = _eval("As", "Kh", "Qd", "Jc", "Ts")
        assert flush > straight

    def test_straight_beats_trips(self) -> None:
        straight = _eval("5s", "4h", "3d", "2c", "As")
        trips = _eval("As", "Ah", "Ad", "Ks", "Qh")
        assert straight > trips

    def test_trips_beats_two_pair(self) -> None:
        trips = _eval("2s", "2h", "2d", "3s", "4h")
        two_pair = _eval("As", "Ah", "Ks", "Kh", "Qs")
        assert trips > two_pair

    def test_two_pair_beats_one_pair(self) -> None:
        two_pair = _eval("2s", "2h", "3s", "3h", "4d")
        one_pair = _eval("As", "Ah", "Ks", "Qh", "Jd")
        assert two_pair > one_pair

    def test_one_pair_beats_high_card(self) -> None:
        one_pair = _eval("2s", "2h", "3s", "4h", "5d")
        high_card = _eval("As", "Kh", "Qd", "Jc", "9s")
        assert one_pair > high_card


# =========================================================================
# Input validation
# =========================================================================


class TestInputValidation:
    def test_too_few_cards_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly 5"):
            evaluate_5([C("Ah"), C("Kh"), C("Qh")])

    def test_too_many_cards_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly 5"):
            evaluate_5([C("Ah"), C("Kh"), C("Qh"), C("Jh"), C("Th"), C("9h")])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly 5"):
            evaluate_5([])


# =========================================================================
# Full ordering sort test
# =========================================================================


class TestFullOrdering:
    """Sort a collection of different-category hands and verify order."""

    def test_sorted_hands_weakest_to_strongest(self) -> None:
        high_card = _eval("As", "Kh", "Qd", "Jc", "9s")
        one_pair = _eval("As", "Ah", "Ks", "Qh", "Jd")
        two_pair = _eval("As", "Ah", "Ks", "Kh", "Qs")
        trips = _eval("As", "Ah", "Ad", "Ks", "Qh")
        straight = _eval("Ts", "9h", "8d", "7c", "6s")
        flush = _eval("Ah", "Th", "8h", "5h", "2h")
        full_house = _eval("As", "Ah", "Ad", "Ks", "Kh")
        quads = _eval("As", "Ah", "Ad", "Ac", "Ks")
        str_flush = _eval("9s", "8s", "7s", "6s", "5s")
        royal = _eval("Ah", "Kh", "Qh", "Jh", "Th")

        hands = [
            royal, quads, high_card, flush, one_pair,
            str_flush, straight, trips, full_house, two_pair,
        ]
        sorted_hands = sorted(hands)

        assert sorted_hands == [
            high_card, one_pair, two_pair, trips, straight,
            flush, full_house, quads, str_flush, royal,
        ]
