"""Performance benchmarks for the hand evaluator.

These tests verify that the evaluator meets minimum throughput
requirements.  They are marked with ``@pytest.mark.benchmark`` and
``@pytest.mark.slow`` so they can be excluded from fast test runs via
``pytest -m "not slow"``.
"""

from __future__ import annotations

import time

import pytest

from poker_equity.card import Card, Rank, Suit
from poker_equity.deck import Deck
from poker_equity.evaluator import evaluate_5, evaluate_7, score_5, score_7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_random_hands(
    n: int, hand_size: int, seed: int = 42
) -> list[list[Card]]:
    """Generate *n* random hands of *hand_size* cards each."""
    hands: list[list[Card]] = []
    deck = Deck(seed=seed)
    for i in range(n):
        deck.reset(seed=seed + i)
        deck.shuffle()
        hands.append(deck.deal(hand_size))
    return hands


# ---------------------------------------------------------------------------
# 5-card benchmarks
# ---------------------------------------------------------------------------


class TestEvaluator5Performance:
    """Benchmark tests for 5-card evaluation."""

    NUM_HANDS = 100_000

    @pytest.fixture(scope="class")
    def hands_5(self) -> list[list[Card]]:
        return _generate_random_hands(self.NUM_HANDS, 5)

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_evaluate_5_throughput(self, hands_5: list[list[Card]]) -> None:
        """evaluate_5 should process ≥ 20k hands/sec."""
        start = time.perf_counter()
        for hand in hands_5:
            evaluate_5(hand)
        elapsed = time.perf_counter() - start

        throughput = self.NUM_HANDS / elapsed
        print(f"\nevaluate_5: {elapsed:.3f}s — {throughput:,.0f} hands/sec")
        assert throughput >= 20_000, (
            f"evaluate_5 too slow: {throughput:,.0f} hands/sec (need ≥ 20k)"
        )

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_score_5_throughput(self, hands_5: list[list[Card]]) -> None:
        """score_5 (lookup-based) should process ≥ 50k hands/sec."""
        # Warm up the lazy-loaded tables
        score_5(hands_5[0])

        start = time.perf_counter()
        for hand in hands_5:
            score_5(hand)
        elapsed = time.perf_counter() - start

        throughput = self.NUM_HANDS / elapsed
        print(f"\nscore_5:    {elapsed:.3f}s — {throughput:,.0f} hands/sec")
        assert throughput >= 50_000, (
            f"score_5 too slow: {throughput:,.0f} hands/sec (need ≥ 50k)"
        )

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_score_5_faster_than_evaluate_5(
        self, hands_5: list[list[Card]]
    ) -> None:
        """score_5 should be faster than evaluate_5."""
        # Warm up
        score_5(hands_5[0])

        start = time.perf_counter()
        for hand in hands_5:
            score_5(hand)
        score_time = time.perf_counter() - start

        start = time.perf_counter()
        for hand in hands_5:
            evaluate_5(hand)
        eval_time = time.perf_counter() - start

        speedup = eval_time / score_time
        print(f"\nscore_5 speedup vs evaluate_5: {speedup:.1f}x")
        assert speedup >= 1.5, (
            f"score_5 not significantly faster: {speedup:.1f}x (need ≥ 1.5x)"
        )


# ---------------------------------------------------------------------------
# 7-card benchmarks
# ---------------------------------------------------------------------------


class TestEvaluator7Performance:
    """Benchmark tests for 7-card evaluation."""

    NUM_HANDS = 100_000

    @pytest.fixture(scope="class")
    def hands_7(self) -> list[list[Card]]:
        return _generate_random_hands(self.NUM_HANDS, 7)

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_score_7_throughput(self, hands_7: list[list[Card]]) -> None:
        """score_7 should process ≥ 5k hands/sec."""
        # Warm up the lazy-loaded tables
        score_7(hands_7[0])

        start = time.perf_counter()
        for hand in hands_7:
            score_7(hand)
        elapsed = time.perf_counter() - start

        throughput = self.NUM_HANDS / elapsed
        print(f"\nscore_7:    {elapsed:.3f}s — {throughput:,.0f} hands/sec")
        assert throughput >= 5_000, (
            f"score_7 too slow: {throughput:,.0f} hands/sec (need ≥ 5k)"
        )

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_score_7_under_2_seconds(self, hands_7: list[list[Card]]) -> None:
        """100k 7-card hands via score_7 should complete in < 10 seconds."""
        score_7(hands_7[0])  # warm up

        start = time.perf_counter()
        for hand in hands_7:
            score_7(hand)
        elapsed = time.perf_counter() - start

        print(f"\nscore_7 total: {elapsed:.3f}s for {self.NUM_HANDS:,} hands")
        assert elapsed < 10.0, (
            f"score_7 took {elapsed:.3f}s for {self.NUM_HANDS:,} hands "
            f"(need < 10s)"
        )

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_score_7_faster_than_evaluate_7(
        self, hands_7: list[list[Card]]
    ) -> None:
        """score_7 should be ≥ 2x faster than evaluate_7."""
        score_7(hands_7[0])  # warm up

        start = time.perf_counter()
        for hand in hands_7:
            score_7(hand)
        score_time = time.perf_counter() - start

        start = time.perf_counter()
        for hand in hands_7:
            evaluate_7(hand)
        eval_time = time.perf_counter() - start

        speedup = eval_time / score_time
        print(
            f"\nscore_7 vs evaluate_7: {speedup:.1f}x faster "
            f"({score_time:.3f}s vs {eval_time:.3f}s)"
        )
        assert speedup >= 2.0, (
            f"score_7 not fast enough: {speedup:.1f}x (need ≥ 2.0x)"
        )


# ---------------------------------------------------------------------------
# Correctness cross-check
# ---------------------------------------------------------------------------


class TestScoreCorrectnessAtScale:
    """Verify that score-based and HandResult-based evaluators agree."""

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_score_and_evaluate_agree_on_ordering(self) -> None:
        """For 10k random 7-card matchups, score ordering must match
        evaluate ordering."""
        hands = _generate_random_hands(10_000, 7)
        mismatches = 0

        for i in range(0, len(hands) - 1, 2):
            h1, h2 = hands[i], hands[i + 1]

            s1, s2 = score_7(h1), score_7(h2)
            e1, e2 = evaluate_7(h1), evaluate_7(h2)

            score_cmp = (s1 > s2) - (s1 < s2)
            eval_cmp = (e1 > e2) - (e1 < e2)

            if score_cmp != eval_cmp:
                mismatches += 1

        assert mismatches == 0, (
            f"{mismatches} ordering mismatches between score_7 and evaluate_7"
        )
