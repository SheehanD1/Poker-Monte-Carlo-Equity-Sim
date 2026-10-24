"""Unit tests for range vs. range Monte Carlo equity simulation.

Validates known preflop spots, postflop board textures, multi-way range pots,
reproducibility, convergence tracking, and edge-case validation.
"""

from __future__ import annotations

import pytest

from poker_equity.card import Card
from poker_equity.range import Range
from poker_equity.simulation import (
    EquityResult,
    calculate_equities,
    calculate_equity,
    range_vs_range_equity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

C = Card.from_str


# =========================================================================
# Known Preflop Matchup Tests (Theoretical Solved Benchmarks)
# =========================================================================


class TestKnownPreflopSpots:
    def test_aa_vs_any_pocket_pair(self) -> None:
        """AA heavily dominates any lower pocket pair (22-KK): ~81% vs ~19%."""
        results = range_vs_range_equity("AA", "22+", num_sims=5000, seed=101)
        assert len(results) == 2
        assert 0.79 < results[0].equity < 0.83
        assert 0.17 < results[1].equity < 0.21
        assert pytest.approx(sum(r.equity for r in results)) == 1.0

    def test_top_10_percent_vs_top_20_percent(self) -> None:
        """Top 10% range dominates a wider top 20% range: ~62% vs ~38%."""
        top_10 = "88+, AJs+, KQs, AKo"
        top_20 = "66+, A8s+, KTs+, QTs+, JTs, ATo+, KJo+"
        results = range_vs_range_equity(top_10, top_20, num_sims=5000, seed=202)
        assert len(results) == 2
        assert 0.58 < results[0].equity < 0.66
        assert 0.34 < results[1].equity < 0.42
        assert pytest.approx(sum(r.equity for r in results)) == 1.0

    def test_aks_vs_any_pocket_pair(self) -> None:
        """AKs vs 22+ represents a classic coinflip spectrum: ~45-47% for AKs."""
        results = range_vs_range_equity("AKs", "22+", num_sims=5000, seed=303)
        assert len(results) == 2
        assert 0.43 < results[0].equity < 0.48
        assert 0.52 < results[1].equity < 0.57
        assert pytest.approx(sum(r.equity for r in results)) == 1.0

    def test_suited_connectors_vs_overpairs(self) -> None:
        """Middling suited connectors vs premium pairs (QQ+): ~21% vs ~79%."""
        connectors = "T9s, 98s, 87s, 76s"
        overpairs = "QQ+"
        results = range_vs_range_equity(connectors, overpairs, num_sims=5000, seed=404)
        assert len(results) == 2
        assert 0.18 < results[0].equity < 0.24
        assert 0.76 < results[1].equity < 0.82
        assert pytest.approx(sum(r.equity for r in results)) == 1.0

    def test_higher_pairs_dominate_lower_pairs(self) -> None:
        """QQ+ (AA, KK, QQ) heavily crushes lower pairs JJ-22: ~81% vs ~19%."""
        results = range_vs_range_equity("QQ+", "JJ-22", num_sims=5000, seed=505)
        assert len(results) == 2
        assert 0.79 < results[0].equity < 0.84
        assert 0.16 < results[1].equity < 0.21
        assert pytest.approx(sum(r.equity for r in results)) == 1.0

    def test_broadway_kicker_domination(self) -> None:
        """Ace broadways (AKs, AQs, AJs) vs King broadways (KQs, KJs, QJs): ~65% vs ~35%."""
        results = range_vs_range_equity("AKs, AQs, AJs", "KQs, KJs, QJs", num_sims=5000, seed=606)
        assert len(results) == 2
        assert 0.62 < results[0].equity < 0.70
        assert 0.30 < results[1].equity < 0.38
        assert pytest.approx(sum(r.equity for r in results)) == 1.0

    def test_symmetric_range_chop(self) -> None:
        """A range against itself must produce ~50.0% equity for each side."""
        range_str = "TT+, AJs+, KQs"
        results = range_vs_range_equity(range_str, range_str, num_sims=5000, seed=707)
        assert len(results) == 2
        assert 0.48 < results[0].equity < 0.52
        assert 0.48 < results[1].equity < 0.52
        assert pytest.approx(sum(r.equity for r in results)) == 1.0

    def test_random_vs_random_equity(self) -> None:
        """Two random ranges ('random' or '100%') have 50.0% equity each."""
        results = range_vs_range_equity("random", "100%", num_sims=5000, seed=808)
        assert len(results) == 2
        assert 0.48 < results[0].equity < 0.52
        assert 0.48 < results[1].equity < 0.52
        assert pytest.approx(sum(r.equity for r in results)) == 1.0


# =========================================================================
# Postflop Board Texture Interaction Tests
# =========================================================================


class TestPostflopRangeVsRange:
    def test_flop_monotone_flush_advantage(self) -> None:
        """Hero has made flush combos on monotone board Ah Kh Qh vs low pairs."""
        board = "Ah Kh Qh"
        hero_flushes = "JhTh, Jh9h, Th9h"
        villain_pairs = "22, 33, 44"
        results = range_vs_range_equity(hero_flushes, villain_pairs, board=board, num_sims=2000, seed=42)
        assert len(results) == 2
        # Made flush dominates low pairs drawing to full house
        assert results[0].equity > 0.95
        assert results[1].equity < 0.05

    def test_flop_wet_board_straight_domination(self) -> None:
        """Made straights (JTs, 76s) on Th 9d 8c dominate overpairs (AA, KK)."""
        board = "Th 9d 8c"
        results = range_vs_range_equity("JTs, 76s", "AA, KK", board=board, num_sims=2000, seed=42)
        assert len(results) == 2
        # Made straight dominates overpairs
        assert results[0].equity > 0.75
        assert results[1].equity < 0.25

    def test_flop_dry_board_top_pair_advantage(self) -> None:
        """Top pair top kicker (AK) on Kc 7d 2s dominates underpairs (QQ, JJ, TT)."""
        board = "Kc 7d 2s"
        results = range_vs_range_equity("AKs, AKo", "QQ, JJ, TT", board=board, num_sims=2000, seed=42)
        assert len(results) == 2
        assert results[0].equity > 0.85
        assert results[1].equity < 0.15

    def test_turn_set_dominates_two_pair(self) -> None:
        """Set of Aces on As Ks 2d 7c dominates top two pair (AK)."""
        board = "As Ks 2d 7c"
        results = range_vs_range_equity("AA", "AKs", board=board, num_sims=2000, seed=42)
        assert len(results) == 2
        assert results[0].equity > 0.95
        assert results[1].equity < 0.05

    def test_river_showdown_set_beats_overpair(self) -> None:
        """KK makes a set of Kings on Kd 7s 2c 9h 3d, beating AA with 100% equity."""
        board = "Kd 7s 2c 9h 3d"
        results = range_vs_range_equity("AA", "KK", board=board, num_sims=1000, seed=42)
        assert len(results) == 2
        assert results[0].equity == 0.0
        assert results[1].equity == 1.0

    def test_river_board_nut_straight_chop(self) -> None:
        """Board contains Broadway straight (Td Jh Qd Kh Ah): 100% chop."""
        board = "Td Jh Qd Kh Ah"
        results = range_vs_range_equity("22", "33", board=board, num_sims=1000)
        assert len(results) == 2
        assert results[0].wins == 0
        assert results[1].wins == 0
        assert results[0].ties == 1000
        assert results[1].ties == 1000
        assert results[0].equity == 0.5
        assert results[1].equity == 0.5


# =========================================================================
# Mixed Hand vs Range Tests
# =========================================================================


class TestMixedHandAndRangeEquity:
    def test_hero_specific_hand_vs_villain_range(self) -> None:
        """Hero holds pocket Aces [Ah, As] vs villain's QQ+ range (includes AdAc chop)."""
        hero = [C("Ah"), C("As")]
        result = calculate_equity(hero, "QQ+", num_sims=3000, seed=42)
        assert 0.77 < result.equity < 0.82

    def test_hero_range_vs_villain_specific_hand(self) -> None:
        """Hero range TT+ vs villain holding 7c 2d."""
        villain = [C("7c"), C("2d")]
        result = calculate_equity("TT+", villain, num_sims=3000, seed=42)
        assert result.equity > 0.84

    def test_hero_range_vs_villain_random(self) -> None:
        """Hero range QQ+ vs villain with random hand (None)."""
        result = calculate_equity("QQ+", villain=None, num_sims=3000, seed=42)
        assert 0.82 < result.equity < 0.87

    def test_range_instance_vs_string_range(self) -> None:
        """Range object can be paired with string range."""
        r1 = Range.from_str("AA, KK")
        r2 = "QQ, JJ"
        results = range_vs_range_equity(r1, r2, num_sims=2000, seed=42)
        assert len(results) == 2
        assert results[0].equity > 0.78
        assert pytest.approx(sum(r.equity for r in results)) == 1.0


# =========================================================================
# Multi-way Range Simulation Tests
# =========================================================================


class TestMultiWayRangeSimulation:
    def test_three_way_ranges_aa_kk_qq(self) -> None:
        """3-way range pot: AA vs KK vs QQ."""
        hands = ["AA", "KK", "QQ"]
        results = calculate_equities(hands, num_sims=5000, seed=909)
        assert len(results) == 3
        # AA ~65%, KK ~20%, QQ ~15%
        assert 0.62 < results[0].equity < 0.69
        assert 0.17 < results[1].equity < 0.23
        assert 0.12 < results[2].equity < 0.17
        assert pytest.approx(sum(r.equity for r in results)) == 1.0

    def test_three_way_mixed_hand_and_ranges(self) -> None:
        """3-way pot: Hero has specific Aces [Ah, As], villains have KK and QQ ranges."""
        hero = [C("Ah"), C("As")]
        hands = [hero, "KK", "QQ"]
        results = calculate_equities(hands, num_sims=5000, seed=1010)
        assert len(results) == 3
        assert 0.62 < results[0].equity < 0.69
        assert 0.17 < results[1].equity < 0.23
        assert 0.12 < results[2].equity < 0.17
        assert pytest.approx(sum(r.equity for r in results)) == 1.0

    def test_four_way_range_simulation(self) -> None:
        """4-way range pot: AA vs KK vs QQ vs JJ."""
        hands = ["AA", "KK", "QQ", "JJ"]
        results = calculate_equities(hands, num_sims=5000, seed=1111)
        assert len(results) == 4
        assert results[0].equity > results[1].equity > results[2].equity > results[3].equity
        assert pytest.approx(sum(r.equity for r in results)) == 1.0

    def test_three_way_ranges_with_random_player(self) -> None:
        """3-way pot: AA vs KK vs Random player (None)."""
        hands = ["AA", "KK", None]
        results = calculate_equities(hands, num_sims=5000, seed=1212)
        assert len(results) == 3
        assert results[0].equity > 0.60
        assert results[1].equity > 0.15
        assert pytest.approx(sum(r.equity for r in results)) == 1.0


# =========================================================================
# Determinism, Seed Consistency & Convergence Tests
# =========================================================================


class TestRangeDeterminismAndConvergence:
    def test_seed_reproducibility(self) -> None:
        """Identical seeds produce identical simulation outputs."""
        r1 = range_vs_range_equity("AKs", "QQ", num_sims=1000, seed=42)
        r2 = range_vs_range_equity("AKs", "QQ", num_sims=1000, seed=42)
        assert r1[0].wins == r2[0].wins
        assert r1[0].ties == r2[0].ties
        assert r1[0].losses == r2[0].losses
        assert r1[0].equity == r2[0].equity

    def test_different_seeds_produce_variance(self) -> None:
        """Different seeds produce slight Monte Carlo variance."""
        r1 = range_vs_range_equity("AKs", "QQ", num_sims=1000, seed=42)
        r2 = range_vs_range_equity("AKs", "QQ", num_sims=1000, seed=43)
        assert r1[0].equity != r2[0].equity

    def test_convergence_early_stopping(self) -> None:
        """Simulation stops before max iterations when CI threshold is satisfied."""
        results = range_vs_range_equity(
            "AA", "KK",
            num_sims=100_000,
            seed=42,
            convergence_threshold=0.015,
            check_interval=100,
        )
        assert results[0].converged is True
        assert results[0].num_sims < 100_000
        assert results[0].convergence_ci <= 0.015

    def test_progress_callback_frequency(self) -> None:
        """Progress callback is called at the expected intervals."""
        events = []
        results = range_vs_range_equity(
            "AA", "KK",
            num_sims=1200,
            seed=42,
            on_progress=events.append,
            progress_interval=300,
        )
        assert len(events) >= 4
        assert events[-1].sims_completed == 1200
        assert events[-1].pct_complete == 1.0
        assert len(events[-1].current_equities) == 2


# =========================================================================
# Error Handling & Edge Case Tests
# =========================================================================


class TestRangeSimulationValidation:
    def test_empty_range_raises_value_error(self) -> None:
        """Empty range is rejected immediately."""
        with pytest.raises(ValueError, match="range is empty"):
            range_vs_range_equity(Range(), "AA")
        with pytest.raises(ValueError, match="range is empty"):
            range_vs_range_equity("AA", Range())

    def test_dead_cards_completely_block_range(self) -> None:
        """When the board blocks all combos of a range, ValueError is raised."""
        # AhKh has 1 combo; board contains Ah
        with pytest.raises(ValueError, match="no valid combos remaining"):
            range_vs_range_equity("AhKh", "QQ", board="Ah 2c 3d")

    def test_all_combos_conflict_raises_value_error(self) -> None:
        """When two ranges have no mutually exclusive card assignments, raise ValueError."""
        # Both players hold AhKd
        with pytest.raises(ValueError, match="No mutually exclusive combos"):
            range_vs_range_equity("AhKd", "AhKd")

    def test_three_way_impossible_range_assignment(self) -> None:
        """Three players all holding AA is physically impossible (requires 6 Aces)."""
        with pytest.raises(ValueError, match="No mutually exclusive combos"):
            calculate_equities(["AA", "AA", "AA"])

    def test_invalid_num_sims_raises_value_error(self) -> None:
        """Zero or negative num_sims raises ValueError."""
        with pytest.raises(ValueError, match="num_sims must be positive"):
            range_vs_range_equity("AA", "KK", num_sims=0)
        with pytest.raises(ValueError, match="num_sims must be positive"):
            range_vs_range_equity("AA", "KK", num_sims=-100)

    def test_invalid_board_length_raises_value_error(self) -> None:
        """Board with more than 5 cards raises ValueError."""
        with pytest.raises(ValueError, match="Board can have at most 5 cards"):
            range_vs_range_equity("AA", "KK", board="2c 3c 4c 5c 6c 7c")

    def test_duplicate_cards_in_board_raises_value_error(self) -> None:
        """Duplicate cards in board string raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate card: 2c"):
            range_vs_range_equity("AA", "KK", board="2c 2c 3c")
