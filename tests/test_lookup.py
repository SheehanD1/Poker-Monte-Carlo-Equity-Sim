"""Unit tests for the lookup table module and score-based evaluation."""

from __future__ import annotations

import pytest

from poker_equity.card import Card, Rank, Suit
from poker_equity.evaluator import (
    evaluate_5,
    evaluate_7,
    evaluate_hand,
    score_5,
    score_7,
    score_hand,
)
from poker_equity.lookup import (
    FLUSH_TABLE,
    MAX_SCORE,
    MIN_SCORE,
    RANK_PRIMES,
    UNSUITED_TABLE,
    is_flush,
    lookup_score,
    prime_product_from_cards,
    rank_bitmask_from_cards,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

C = Card.from_str


# =========================================================================
# Table structure tests
# =========================================================================


class TestTableStructure:
    """Verify the lookup tables have the correct dimensions."""

    def test_flush_table_has_1287_entries(self) -> None:
        """C(13, 5) = 1287 flush rank combinations."""
        assert len(FLUSH_TABLE) == 1287

    def test_unsuited_table_has_6175_entries(self) -> None:
        """6175 distinct non-flush rank multisets."""
        assert len(UNSUITED_TABLE) == 6175

    def test_total_distinct_scores_is_7462(self) -> None:
        """There are exactly 7,462 distinct hand ranks in poker."""
        all_scores = set(FLUSH_TABLE.values()) | set(UNSUITED_TABLE.values())
        assert len(all_scores) == 7462

    def test_max_score_is_7462(self) -> None:
        assert MAX_SCORE == 7462

    def test_min_score_is_1(self) -> None:
        assert MIN_SCORE == 1

    def test_all_scores_are_positive(self) -> None:
        assert all(v >= 1 for v in FLUSH_TABLE.values())
        assert all(v >= 1 for v in UNSUITED_TABLE.values())

    def test_score_range_is_contiguous(self) -> None:
        """Scores should cover 1..7462 with no gaps."""
        all_scores = set(FLUSH_TABLE.values()) | set(UNSUITED_TABLE.values())
        assert all_scores == set(range(1, 7463))


# =========================================================================
# Prime product tests
# =========================================================================


class TestPrimeProduct:
    def test_all_13_ranks_mapped(self) -> None:
        assert len(RANK_PRIMES) == 13
        assert set(RANK_PRIMES.keys()) == set(range(2, 15))

    def test_all_primes_are_unique(self) -> None:
        values = list(RANK_PRIMES.values())
        assert len(values) == len(set(values))

    def test_prime_product_from_cards(self) -> None:
        cards = [C("Ah"), C("Kh"), C("Qh")]
        expected = RANK_PRIMES[14] * RANK_PRIMES[13] * RANK_PRIMES[12]
        assert prime_product_from_cards(cards) == expected

    def test_prime_product_order_independent(self) -> None:
        """Product should be the same regardless of card order."""
        cards_a = [C("Ah"), C("Kh"), C("Qh"), C("Jh"), C("Th")]
        cards_b = [C("Th"), C("Jh"), C("Qh"), C("Kh"), C("Ah")]
        assert prime_product_from_cards(cards_a) == prime_product_from_cards(
            cards_b
        )

    def test_different_ranks_produce_different_products(self) -> None:
        cards_a = [C("Ah"), C("Kh"), C("Qh"), C("Jh"), C("Th")]
        cards_b = [C("Ah"), C("Kh"), C("Qh"), C("Jh"), C("9h")]
        assert prime_product_from_cards(cards_a) != prime_product_from_cards(
            cards_b
        )


# =========================================================================
# Rank bitmask tests
# =========================================================================


class TestRankBitmask:
    def test_single_ace(self) -> None:
        mask = rank_bitmask_from_cards([C("Ah")])
        assert mask == (1 << 14)

    def test_ace_king(self) -> None:
        mask = rank_bitmask_from_cards([C("Ah"), C("Kh")])
        assert mask == (1 << 14) | (1 << 13)

    def test_duplicate_ranks_produce_same_mask(self) -> None:
        """Two aces should produce the same bitmask as one ace."""
        mask_one = rank_bitmask_from_cards([C("Ah")])
        mask_two = rank_bitmask_from_cards([C("Ah"), C("As")])
        assert mask_one == mask_two


# =========================================================================
# Flush detection tests
# =========================================================================


class TestIsFlush:
    def test_flush_detected(self) -> None:
        cards = [C("Ah"), C("Kh"), C("Qh"), C("Jh"), C("9h")]
        assert is_flush(cards)

    def test_non_flush_detected(self) -> None:
        cards = [C("Ah"), C("Kh"), C("Qh"), C("Jh"), C("9s")]
        assert not is_flush(cards)


# =========================================================================
# lookup_score tests
# =========================================================================


class TestLookupScore:
    def test_royal_flush_is_max(self) -> None:
        score = lookup_score([C("Ah"), C("Kh"), C("Qh"), C("Jh"), C("Th")])
        assert score == MAX_SCORE

    def test_worst_high_card_is_min(self) -> None:
        score = lookup_score([C("7s"), C("5h"), C("4d"), C("3c"), C("2s")])
        assert score == MIN_SCORE

    def test_wrong_card_count_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly 5"):
            lookup_score([C("Ah"), C("Kh"), C("Qh")])

    def test_flush_hand_uses_flush_table(self) -> None:
        """A flush hand should get its score from the flush table."""
        cards = [C("Ah"), C("Kh"), C("Qh"), C("Jh"), C("9h")]
        score = lookup_score(cards)
        mask = rank_bitmask_from_cards(cards)
        assert score == FLUSH_TABLE[mask]

    def test_non_flush_hand_uses_unsuited_table(self) -> None:
        """A non-flush hand should get its score from the unsuited table."""
        cards = [C("Ah"), C("Ks"), C("Qh"), C("Jh"), C("9h")]
        score = lookup_score(cards)
        pp = prime_product_from_cards(cards)
        assert score == UNSUITED_TABLE[pp]


# =========================================================================
# score_5 vs evaluate_5 consistency
# =========================================================================


class TestScore5Consistency:
    """Verify that score_5 ordering matches evaluate_5 ordering."""

    HAND_PAIRS: list[tuple[list[str], list[str]]] = [
        # Royal flush > straight flush
        (["Ah", "Kh", "Qh", "Jh", "Th"], ["Ks", "Qs", "Js", "Ts", "9s"]),
        # Straight flush > quads
        (["9s", "8s", "7s", "6s", "5s"], ["As", "Ah", "Ad", "Ac", "Ks"]),
        # Quads > full house
        (["As", "Ah", "Ad", "Ac", "2s"], ["Ks", "Kh", "Kd", "Qs", "Qh"]),
        # Full house > flush
        (["2s", "2h", "2d", "3s", "3h"], ["Ah", "Th", "8h", "5h", "2h"]),
        # Flush > straight
        (["Ah", "Th", "8h", "5h", "3h"], ["As", "Kh", "Qd", "Jc", "Ts"]),
        # Straight > trips
        (["As", "Kh", "Qd", "Jc", "Ts"], ["As", "Ah", "Ad", "Ks", "Qh"]),
        # Trips > two pair
        (["2s", "2h", "2d", "As", "Kh"], ["As", "Ah", "Ks", "Kh", "Qs"]),
        # Two pair > one pair
        (["3s", "3h", "2s", "2h", "As"], ["As", "Ah", "Ks", "Qh", "Jd"]),
        # One pair > high card
        (["2s", "2h", "As", "Kh", "Qd"], ["As", "Kh", "Qd", "Jc", "9s"]),
    ]

    @pytest.mark.parametrize(
        "better_strs,worse_strs",
        HAND_PAIRS,
        ids=[
            "RF>SF", "SF>4K", "4K>FH", "FH>FL", "FL>ST",
            "ST>3K", "3K>2P", "2P>1P", "1P>HC",
        ],
    )
    def test_score_ordering_matches_evaluate(
        self, better_strs: list[str], worse_strs: list[str]
    ) -> None:
        better = [C(s) for s in better_strs]
        worse = [C(s) for s in worse_strs]

        assert score_5(better) > score_5(worse)
        assert evaluate_5(better) > evaluate_5(worse)

    def test_tied_hands_have_equal_scores(self) -> None:
        """Two royal flushes in different suits should have the same score."""
        rf_hearts = [C("Ah"), C("Kh"), C("Qh"), C("Jh"), C("Th")]
        rf_spades = [C("As"), C("Ks"), C("Qs"), C("Js"), C("Ts")]
        assert score_5(rf_hearts) == score_5(rf_spades)

    def test_kicker_ordering_in_scores(self) -> None:
        """Same pair, different kickers — higher kicker should score higher."""
        pair_a_kqj = [C("As"), C("Ah"), C("Ks"), C("Qh"), C("Jd")]
        pair_a_kqt = [C("As"), C("Ah"), C("Ks"), C("Qh"), C("Td")]
        assert score_5(pair_a_kqj) > score_5(pair_a_kqt)


# =========================================================================
# score_7 and score_hand tests
# =========================================================================


class TestScore7:
    def test_finds_best_5_card_hand(self) -> None:
        """7 cards containing a royal flush should score as royal flush."""
        cards = [C("Ah"), C("Kh"), C("Qh"), C("Jh"), C("Th"), C("2c"), C("3d")]
        assert score_7(cards) == MAX_SCORE

    def test_matches_evaluate_7(self) -> None:
        """score_7 and evaluate_7 should agree on ordering."""
        hand_a = [C("As"), C("Ah"), C("Ad"), C("Ks"), C("Kh"), C("2c"), C("3d")]
        hand_b = [C("Ks"), C("Kh"), C("Kd"), C("Qs"), C("Qh"), C("2c"), C("3d")]

        score_cmp = score_7(hand_a) > score_7(hand_b)
        eval_cmp = evaluate_7(hand_a) > evaluate_7(hand_b)
        assert score_cmp == eval_cmp

    def test_wrong_card_count_for_score_hand(self) -> None:
        with pytest.raises(ValueError, match="5, 6, or 7"):
            score_hand([C("Ah"), C("Kh")])


class TestScoreHand:
    def test_5_cards(self) -> None:
        cards = [C("Ah"), C("Kh"), C("Qh"), C("Jh"), C("Th")]
        assert score_hand(cards) == score_5(cards)

    def test_6_cards(self) -> None:
        cards = [C("Ah"), C("Kh"), C("Qh"), C("Jh"), C("Th"), C("2c")]
        assert score_hand(cards) == MAX_SCORE  # royal flush in there

    def test_7_cards(self) -> None:
        cards = [C("Ah"), C("Kh"), C("Qh"), C("Jh"), C("Th"), C("2c"), C("3d")]
        assert score_hand(cards) == score_7(cards)


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    def test_wheel_straight_flush_score(self) -> None:
        """A-2-3-4-5 suited should score lower than 6-high straight flush."""
        wheel_sf = score_5([C("Ac"), C("2c"), C("3c"), C("4c"), C("5c")])
        six_sf = score_5([C("6d"), C("5d"), C("4d"), C("3d"), C("2d")])
        assert six_sf > wheel_sf

    def test_wheel_straight_score(self) -> None:
        """A-2-3-4-5 offsuit should score lower than 6-high straight."""
        wheel = score_5([C("As"), C("2h"), C("3d"), C("4c"), C("5s")])
        six_str = score_5([C("6s"), C("5h"), C("4d"), C("3c"), C("2s")])
        assert six_str > wheel

    def test_all_royal_flushes_score_equal(self) -> None:
        suits = ["h", "d", "c", "s"]
        scores = [
            score_5([C(f"A{s}"), C(f"K{s}"), C(f"Q{s}"), C(f"J{s}"), C(f"T{s}")])
            for s in suits
        ]
        assert len(set(scores)) == 1
        assert scores[0] == MAX_SCORE

    def test_worst_hand_in_each_category_beats_best_of_lower(self) -> None:
        """The worst hand in a category must beat the best in the one below."""
        # Worst pair (22 with 543 kickers) vs best high card (AKQJT unsuited)
        worst_pair = score_5([C("2s"), C("2h"), C("5d"), C("4c"), C("3s")])
        best_hc = score_5([C("As"), C("Kh"), C("Qd"), C("Jc"), C("9s")])
        assert worst_pair > best_hc
