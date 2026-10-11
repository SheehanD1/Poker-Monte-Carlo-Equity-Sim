"""Unit tests for the parallel Monte Carlo equity simulator."""

from __future__ import annotations

import pytest

from poker_equity.card import Card
from poker_equity.parallel import (
    calculate_equities_parallel,
    calculate_equity_parallel,
)
from poker_equity.simulation import calculate_equity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

C = Card.from_str


# =========================================================================
# Input Validation Tests
# =========================================================================


class TestParallelValidation:
    def test_invalid_num_sims_raises(self) -> None:
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        with pytest.raises(ValueError, match="num_sims must be positive"):
            calculate_equity_parallel(hero, villain, num_sims=0)

        with pytest.raises(ValueError, match="num_sims must be positive"):
            calculate_equity_parallel(hero, villain, num_sims=-100)

    def test_invalid_num_workers_raises(self) -> None:
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        with pytest.raises(ValueError, match="num_workers must be positive"):
            calculate_equity_parallel(hero, villain, num_sims=100, num_workers=0)

        with pytest.raises(ValueError, match="num_workers must be positive"):
            calculate_equity_parallel(hero, villain, num_sims=100, num_workers=-2)

    def test_invalid_hand_size_raises(self) -> None:
        hero = [C("Ah")]
        villain = [C("Kh"), C("Kd")]
        with pytest.raises(ValueError, match="Player 1 must have exactly 2 hole cards"):
            calculate_equity_parallel(hero, villain, num_sims=100)

    def test_duplicate_cards_raises(self) -> None:
        hero = [C("Ah"), C("Kd")]
        villain = [C("Ah"), C("Qc")]
        with pytest.raises(ValueError, match="Duplicate card: Ah"):
            calculate_equity_parallel(hero, villain, num_sims=100)

    def test_board_size_exceeded_raises(self) -> None:
        hero = [C("Ah"), C("Kd")]
        villain = [C("Qh"), C("Qd")]
        board = [C("2c"), C("3d"), C("4h"), C("5s"), C("6c"), C("7d")]
        with pytest.raises(ValueError, match="Board can have at most 5 cards"):
            calculate_equity_parallel(hero, villain, board=board, num_sims=100)


# =========================================================================
# Worker Configuration & Edge Cases
# =========================================================================


class TestWorkerConfigurations:
    def test_fallback_when_num_sims_less_than_workers(self) -> None:
        """When num_sims < num_workers, execution runs synchronously."""
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        res = calculate_equity_parallel(
            hero, villain, num_sims=3, num_workers=8, seed=42
        )
        assert res.num_sims == 3
        assert res.wins + res.ties + res.losses == 3

    def test_single_worker(self) -> None:
        """Runs correctly when explicitly requesting num_workers=1."""
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        res = calculate_equity_parallel(
            hero, villain, num_sims=500, num_workers=1, seed=42
        )
        assert res.num_sims == 500
        assert res.wins + res.ties + res.losses == 500
        assert 0.75 <= res.equity <= 0.90

    def test_two_workers(self) -> None:
        """Runs correctly when num_workers=2."""
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        res = calculate_equity_parallel(
            hero, villain, num_sims=500, num_workers=2, seed=42
        )
        assert res.num_sims == 500
        assert res.wins + res.ties + res.losses == 500

    def test_four_workers(self) -> None:
        """Runs correctly when num_workers=4."""
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        res = calculate_equity_parallel(
            hero, villain, num_sims=500, num_workers=4, seed=42
        )
        assert res.num_sims == 500
        assert res.wins + res.ties + res.losses == 500

    def test_uneven_work_distribution(self) -> None:
        """Total simulations not evenly divisible by num_workers distributes properly."""
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        # 107 sims across 4 workers -> 27, 27, 27, 26
        res = calculate_equity_parallel(
            hero, villain, num_sims=107, num_workers=4, seed=42
        )
        assert res.num_sims == 107
        assert res.wins + res.ties + res.losses == 107

    def test_default_workers_uses_cpu_count(self) -> None:
        """When num_workers is None, uses os.cpu_count() without error."""
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        res = calculate_equity_parallel(
            hero, villain, num_sims=500, num_workers=None, seed=42
        )
        assert res.num_sims == 500
        assert res.wins + res.ties + res.losses == 500


# =========================================================================
# Determinism and Reproducibility
# =========================================================================


class TestParallelDeterminism:
    def test_deterministic_results_with_same_seed_and_workers(self) -> None:
        hero = [C("Ah"), C("Kd")]
        villain = [C("Qs"), C("Qc")]
        r1 = calculate_equity_parallel(
            hero, villain, num_sims=1000, num_workers=2, seed=123
        )
        r2 = calculate_equity_parallel(
            hero, villain, num_sims=1000, num_workers=2, seed=123
        )
        assert r1.wins == r2.wins
        assert r1.ties == r2.ties
        assert r1.losses == r2.losses
        assert r1.equity == pytest.approx(r2.equity)

    def test_different_seeds_produce_different_results(self) -> None:
        hero = [C("Ah"), C("Kd")]
        villain = [C("Qs"), C("Qc")]
        r1 = calculate_equity_parallel(
            hero, villain, num_sims=1000, num_workers=2, seed=100
        )
        r2 = calculate_equity_parallel(
            hero, villain, num_sims=1000, num_workers=2, seed=999
        )
        assert r1.wins != r2.wins or r1.losses != r2.losses


# =========================================================================
# Correctness & Equity Benchmarks
# =========================================================================


class TestParallelCorrectness:
    def test_known_equity_aa_vs_kk(self) -> None:
        """AA vs KK preflop: AA has ~82% equity."""
        hero = [C("Ah"), C("Ac")]
        villain = [C("Kh"), C("Kc")]
        res = calculate_equity_parallel(
            hero, villain, num_sims=3000, num_workers=2, seed=42
        )
        assert res.equity == pytest.approx(0.82, abs=0.03)
        assert res.win_pct + res.tie_pct + res.loss_pct == pytest.approx(1.0)

    def test_parallel_matches_sequential_within_tolerance(self) -> None:
        """Parallel simulation results closely match sequential simulation."""
        hero = [C("Ah"), C("Ks")]
        villain = [C("Qc"), C("Qd")]
        seq_res = calculate_equity(hero, villain, num_sims=4000, seed=42)
        par_res = calculate_equity_parallel(
            hero, villain, num_sims=4000, num_workers=2, seed=999
        )
        assert par_res.equity == pytest.approx(seq_res.equity, abs=0.03)

    def test_river_nut_flush_100_percent(self) -> None:
        """Hero with nuts on river has 100% equity."""
        hero = [C("Ah"), C("Kh")]
        villain = [C("Qs"), C("Qd")]
        board = [C("Th"), C("7h"), C("2h"), C("9c"), C("3d")]
        res = calculate_equity_parallel(
            hero, villain, board=board, num_sims=200, num_workers=2, seed=42
        )
        assert res.equity == 1.0
        assert res.wins == 200
        assert res.losses == 0
        assert res.ties == 0

    def test_river_chop(self) -> None:
        """Board plays best for both players: 50% equity each."""
        hero = [C("2h"), C("3d")]
        villain = [C("4c"), C("5s")]
        board = [C("As"), C("Ks"), C("Qs"), C("Js"), C("Ts")]
        res = calculate_equity_parallel(
            hero, villain, board=board, num_sims=200, num_workers=2, seed=42
        )
        assert res.equity == 0.5
        assert res.ties == 200
        assert res.wins == 0
        assert res.losses == 0

    def test_hero_vs_random_hand(self) -> None:
        """AA vs random hand has ~85% equity."""
        hero = [C("Ah"), C("Ad")]
        res = calculate_equity_parallel(
            hero, villain=None, num_sims=3000, num_workers=2, seed=42
        )
        assert res.equity == pytest.approx(0.85, abs=0.03)
        assert res.win_pct + res.tie_pct + res.loss_pct == pytest.approx(1.0)


# =========================================================================
# Multi-way Parallel Simulations
# =========================================================================


class TestParallelMultiWay:
    def test_three_way_equity_sum(self) -> None:
        """In a 3-way pot, equities must sum to 100%."""
        p1 = [C("Ah"), C("Ac")]
        p2 = [C("Kh"), C("Kc")]
        p3 = [C("Qh"), C("Qc")]
        results = calculate_equities_parallel(
            hands=[p1, p2, p3], num_sims=3000, num_workers=2, seed=42
        )
        assert len(results) == 3
        for res in results:
            assert res.num_sims == 3000
            assert res.wins + res.ties + res.losses == 3000

        total_equity = sum(r.equity for r in results)
        assert total_equity == pytest.approx(1.0, abs=1e-5)

        # AA > KK > QQ
        assert results[0].equity > results[1].equity > results[2].equity

    def test_four_way_known_equities(self) -> None:
        """AA vs KK vs QQ vs JJ 4-way pot equities sum to 100%."""
        p1 = [C("Ah"), C("Ac")]
        p2 = [C("Kh"), C("Kc")]
        p3 = [C("Qh"), C("Qc")]
        p4 = [C("Jh"), C("Jc")]
        results = calculate_equities_parallel(
            hands=[p1, p2, p3, p4], num_sims=3000, num_workers=2, seed=42
        )
        assert len(results) == 4
        total_equity = sum(r.equity for r in results)
        assert total_equity == pytest.approx(1.0, abs=1e-5)

        # Expected ranges: AA ~55%, KK ~18%, QQ ~14%, JJ ~13%
        assert 0.50 <= results[0].equity <= 0.60
        assert 0.14 <= results[1].equity <= 0.22
        assert 0.10 <= results[2].equity <= 0.18
        assert 0.09 <= results[3].equity <= 0.17

    def test_three_way_with_random_hand(self) -> None:
        """Multi-way pot with one or more random hands in parallel."""
        p1 = [C("Ah"), C("Ac")]
        p2 = [C("Kh"), C("Kc")]
        p3 = None
        results = calculate_equities_parallel(
            hands=[p1, p2, p3], num_sims=2000, num_workers=2, seed=42
        )
        assert len(results) == 3
        total_equity = sum(r.equity for r in results)
        assert total_equity == pytest.approx(1.0, abs=1e-5)
        # AA should beat random and KK
        assert results[0].equity > results[1].equity
        assert results[0].equity > results[2].equity

    def test_multi_way_chop(self) -> None:
        """3-way pot where all 3 players tie."""
        p1 = [C("2h"), C("3d")]
        p2 = [C("4c"), C("5s")]
        p3 = [C("6h"), C("7d")]
        board = [C("As"), C("Ks"), C("Qs"), C("Js"), C("Ts")]
        results = calculate_equities_parallel(
            hands=[p1, p2, p3], board=board, num_sims=200, num_workers=2, seed=42
        )
        for res in results:
            assert res.ties == 200
            assert res.wins == 0
            assert res.losses == 0
            assert res.equity == pytest.approx(1.0 / 3.0, abs=1e-5)
