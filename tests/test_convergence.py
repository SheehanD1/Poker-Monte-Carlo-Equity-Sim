"""Unit tests for convergence tracking and early stopping."""

from __future__ import annotations

import math

import pytest

from poker_equity.card import Card
from poker_equity.simulation import (
    EquityResult,
    _build_equity_result,
    _confidence_interval_half_width,
    _MIN_CONVERGENCE_SIMS,
    _Z_95,
    calculate_equities,
    calculate_equity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

C = Card.from_str


# =========================================================================
# _confidence_interval_half_width unit tests
# =========================================================================


class TestConfidenceIntervalHalfWidth:
    def test_returns_inf_for_n_less_than_2(self) -> None:
        """Cannot compute a meaningful CI with fewer than 2 samples."""
        assert _confidence_interval_half_width(0.25, 0) == float("inf")
        assert _confidence_interval_half_width(0.25, 1) == float("inf")

    def test_basic_calculation(self) -> None:
        """Verify the formula: z * sqrt(variance / n)."""
        variance = 0.25
        n = 100
        expected = _Z_95 * math.sqrt(variance / n)
        result = _confidence_interval_half_width(variance, n)
        assert result == pytest.approx(expected)

    def test_ci_decreases_with_more_samples(self) -> None:
        """CI should shrink as sample count grows."""
        variance = 0.25
        ci_100 = _confidence_interval_half_width(variance, 100)
        ci_1000 = _confidence_interval_half_width(variance, 1000)
        ci_10000 = _confidence_interval_half_width(variance, 10000)
        assert ci_100 > ci_1000 > ci_10000

    def test_ci_increases_with_more_variance(self) -> None:
        """Higher variance produces wider confidence intervals."""
        n = 1000
        ci_low = _confidence_interval_half_width(0.01, n)
        ci_high = _confidence_interval_half_width(0.25, n)
        assert ci_high > ci_low

    def test_zero_variance_gives_zero_ci(self) -> None:
        """When all samples are identical, variance is 0 and CI is 0."""
        assert _confidence_interval_half_width(0.0, 100) == 0.0


# =========================================================================
# EquityResult convergence fields
# =========================================================================


class TestEquityResultConvergenceFields:
    def test_default_convergence_fields(self) -> None:
        """EquityResult defaults to converged=False, convergence_ci=0.0."""
        result = _build_equity_result(
            wins=50, ties=10, losses=40, num_sims=100, equity=0.55
        )
        assert result.converged is False
        assert result.convergence_ci == 0.0

    def test_explicit_convergence_fields(self) -> None:
        """EquityResult stores convergence metadata when provided."""
        result = _build_equity_result(
            wins=400, ties=20, losses=80, num_sims=500, equity=0.82,
            converged=True, convergence_ci=0.005,
        )
        assert result.converged is True
        assert result.convergence_ci == pytest.approx(0.005)

    def test_str_includes_converged_tag(self) -> None:
        """String representation includes [converged] when applicable."""
        result = _build_equity_result(
            wins=400, ties=20, losses=80, num_sims=500, equity=0.82,
            converged=True, convergence_ci=0.005,
        )
        s = str(result)
        assert "[converged]" in s

    def test_str_omits_converged_tag_when_false(self) -> None:
        """String representation omits [converged] when not converged."""
        result = _build_equity_result(
            wins=400, ties=20, losses=80, num_sims=500, equity=0.82,
            converged=False, convergence_ci=0.02,
        )
        s = str(result)
        assert "[converged]" not in s

    def test_repr_includes_converged(self) -> None:
        """Repr includes converged status."""
        result = _build_equity_result(
            wins=50, ties=10, losses=40, num_sims=100, equity=0.55,
            converged=True, convergence_ci=0.008,
        )
        r = repr(result)
        assert "converged=True" in r


# =========================================================================
# Input validation for convergence parameters
# =========================================================================


class TestConvergenceValidation:
    def test_negative_convergence_threshold_raises(self) -> None:
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        with pytest.raises(ValueError, match="convergence_threshold must be positive"):
            calculate_equity(
                hero, villain, num_sims=100,
                convergence_threshold=-0.01,
            )

    def test_zero_convergence_threshold_raises(self) -> None:
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        with pytest.raises(ValueError, match="convergence_threshold must be positive"):
            calculate_equity(
                hero, villain, num_sims=100,
                convergence_threshold=0.0,
            )

    def test_negative_check_interval_raises(self) -> None:
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        with pytest.raises(ValueError, match="check_interval must be positive"):
            calculate_equity(
                hero, villain, num_sims=100,
                convergence_threshold=0.01, check_interval=-1,
            )

    def test_zero_check_interval_raises(self) -> None:
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        with pytest.raises(ValueError, match="check_interval must be positive"):
            calculate_equity(
                hero, villain, num_sims=100,
                convergence_threshold=0.01, check_interval=0,
            )


# =========================================================================
# Early stopping behavior
# =========================================================================


class TestEarlyStopping:
    def test_converges_early_with_large_threshold(self) -> None:
        """A very generous threshold (10%) should converge well before max sims."""
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        result = calculate_equity(
            hero, villain,
            num_sims=50_000,
            seed=42,
            convergence_threshold=0.10,
            check_interval=100,
        )
        assert result.converged is True
        assert result.num_sims < 50_000
        # Should still produce a reasonable equity estimate
        assert 0.70 <= result.equity <= 0.90

    def test_does_not_converge_with_tiny_threshold_and_few_sims(self) -> None:
        """An impossibly tight threshold with few sims should not converge."""
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        result = calculate_equity(
            hero, villain,
            num_sims=200,
            seed=42,
            convergence_threshold=0.0001,
            check_interval=100,
        )
        assert result.converged is False
        assert result.num_sims == 200

    def test_convergence_ci_is_reported(self) -> None:
        """When tracking convergence, a non-zero CI is reported."""
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        result = calculate_equity(
            hero, villain,
            num_sims=1000,
            seed=42,
            convergence_threshold=0.05,
            check_interval=100,
        )
        # Whether or not it converged, the CI should be positive
        assert result.convergence_ci > 0.0

    def test_converged_ci_is_below_threshold(self) -> None:
        """When the sim converges, the final CI must be ≤ threshold."""
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        threshold = 0.05
        result = calculate_equity(
            hero, villain,
            num_sims=50_000,
            seed=42,
            convergence_threshold=threshold,
            check_interval=100,
        )
        if result.converged:
            assert result.convergence_ci <= threshold

    def test_early_stop_saves_simulations(self) -> None:
        """A converged run uses fewer sims than a non-converged run."""
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        max_sims = 20_000

        # Without convergence: always runs full count
        result_full = calculate_equity(
            hero, villain, num_sims=max_sims, seed=42,
        )
        assert result_full.num_sims == max_sims
        assert result_full.converged is False

        # With convergence: should stop early
        result_conv = calculate_equity(
            hero, villain, num_sims=max_sims, seed=42,
            convergence_threshold=0.05, check_interval=100,
        )
        assert result_conv.converged is True
        assert result_conv.num_sims < max_sims

    def test_no_convergence_tracking_when_threshold_is_none(self) -> None:
        """Default behavior (threshold=None) disables convergence entirely."""
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        result = calculate_equity(
            hero, villain, num_sims=500, seed=42,
        )
        assert result.converged is False
        assert result.convergence_ci == 0.0
        assert result.num_sims == 500

    def test_never_converges_before_min_sims(self) -> None:
        """Even with a huge threshold, convergence is not checked before
        _MIN_CONVERGENCE_SIMS iterations."""
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        result = calculate_equity(
            hero, villain,
            num_sims=_MIN_CONVERGENCE_SIMS,
            seed=42,
            convergence_threshold=1.0,  # absurdly large
            check_interval=50,
        )
        # With check_interval=50 and max sims=100, the first eligible check
        # is at sim 100. That's also the last sim, so it might converge at
        # exactly _MIN_CONVERGENCE_SIMS. Either way, it ran at least that many.
        assert result.num_sims >= _MIN_CONVERGENCE_SIMS


# =========================================================================
# Check interval behavior
# =========================================================================


class TestCheckInterval:
    def test_smaller_check_interval_detects_convergence_sooner(self) -> None:
        """A smaller check_interval means we catch convergence earlier."""
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        threshold = 0.05

        result_freq = calculate_equity(
            hero, villain, num_sims=50_000, seed=42,
            convergence_threshold=threshold, check_interval=50,
        )
        result_infreq = calculate_equity(
            hero, villain, num_sims=50_000, seed=42,
            convergence_threshold=threshold, check_interval=500,
        )
        # Both should converge, but the frequent checker stops sooner
        # (or at the same time if they happen to land on the same iteration)
        assert result_freq.converged is True
        assert result_infreq.converged is True
        assert result_freq.num_sims <= result_infreq.num_sims

    def test_check_interval_equals_one(self) -> None:
        """check_interval=1 checks every single iteration (no error)."""
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        result = calculate_equity(
            hero, villain, num_sims=500, seed=42,
            convergence_threshold=0.10, check_interval=1,
        )
        # Should work and likely converge early
        assert result.num_sims <= 500


# =========================================================================
# Convergence with multi-way pots
# =========================================================================


class TestMultiWayConvergence:
    def test_three_way_convergence(self) -> None:
        """Convergence works with 3 players."""
        p1 = [C("Ah"), C("Ac")]
        p2 = [C("Kh"), C("Kc")]
        p3 = [C("Qh"), C("Qc")]
        results = calculate_equities(
            hands=[p1, p2, p3],
            num_sims=50_000,
            seed=42,
            convergence_threshold=0.05,
            check_interval=100,
        )
        assert len(results) == 3

        # All players share the same converged flag
        assert all(r.converged for r in results)

        # Equities should sum to ~1.0
        total = sum(r.equity for r in results)
        assert total == pytest.approx(1.0, abs=1e-5)

        # Should have stopped early
        assert results[0].num_sims < 50_000

    def test_multi_way_ci_applies_to_all_players(self) -> None:
        """The convergence CI is the max across all players."""
        p1 = [C("Ah"), C("Ac")]
        p2 = [C("Kh"), C("Kc")]
        p3 = [C("Qh"), C("Qc")]
        threshold = 0.05
        results = calculate_equities(
            hands=[p1, p2, p3],
            num_sims=50_000,
            seed=42,
            convergence_threshold=threshold,
            check_interval=100,
        )
        if results[0].converged:
            # All results share the same CI (the max across players)
            for r in results:
                assert r.convergence_ci <= threshold
                assert r.convergence_ci == results[0].convergence_ci


# =========================================================================
# Convergence with the heads-up convenience wrapper
# =========================================================================


class TestConvenienceWrapperConvergence:
    def test_calculate_equity_passes_convergence_params(self) -> None:
        """calculate_equity properly forwards convergence parameters."""
        hero = [C("Ah"), C("Ad")]
        villain = [C("7s"), C("2d")]
        result = calculate_equity(
            hero, villain,
            num_sims=50_000,
            seed=42,
            convergence_threshold=0.05,
            check_interval=100,
        )
        assert result.converged is True
        assert result.num_sims < 50_000
        assert result.convergence_ci > 0.0
        assert result.convergence_ci <= 0.05


# =========================================================================
# Convergence accuracy
# =========================================================================


class TestConvergenceAccuracy:
    def test_converged_equity_is_accurate(self) -> None:
        """Converged equity should be close to the known true value.
        
        AA vs KK is ~82%. With a 2% convergence threshold, the result
        should be within ~4% of the true value (2% CI + sampling noise).
        """
        hero = [C("Ah"), C("Ad")]
        villain = [C("Kh"), C("Kd")]
        result = calculate_equity(
            hero, villain,
            num_sims=50_000,
            seed=42,
            convergence_threshold=0.02,
            check_interval=100,
        )
        assert result.converged is True
        assert result.equity == pytest.approx(0.82, abs=0.04)

    def test_river_always_converges_immediately(self) -> None:
        """On the river with all 5 cards dealt, there's zero variance.
        
        The outcome is deterministic, so the CI should drop to 0 and 
        convergence should trigger at the first check.
        """
        hero = [C("Ah"), C("Kh")]
        villain = [C("Qs"), C("Qd")]
        board = [C("Th"), C("7h"), C("2h"), C("9c"), C("3d")]
        result = calculate_equity(
            hero, villain, board=board,
            num_sims=10_000,
            seed=42,
            convergence_threshold=0.01,
            check_interval=100,
        )
        # River with nuts should converge quickly
        assert result.converged is True
        assert result.num_sims == _MIN_CONVERGENCE_SIMS
        assert result.equity == 1.0
        assert result.convergence_ci == 0.0
