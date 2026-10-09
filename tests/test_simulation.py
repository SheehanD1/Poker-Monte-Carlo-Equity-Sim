"""Unit tests for the Monte Carlo equity simulator."""

from __future__ import annotations

import pytest

from poker_equity.card import Card
from poker_equity.simulation import EquityResult, _build_equity_result, calculate_equity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

C = Card.from_str


# =========================================================================
# EquityResult Tests
# =========================================================================


class TestEquityResult:
    def test_build_equity_result(self) -> None:
        """Test calculation of percentages and equity."""
        result = _build_equity_result(wins=50, ties=10, losses=40, num_sims=100, equity=0.55)
        assert result.wins == 50
        assert result.ties == 10
        assert result.losses == 40
        assert result.num_sims == 100
        assert result.win_pct == 0.50
        assert result.tie_pct == 0.10
        assert result.loss_pct == 0.40
        assert result.equity == 0.55  # 50 + (10 / 2) / 100

    def test_str_representation(self) -> None:
        """Test the human-readable string representation."""
        result = _build_equity_result(wins=691, ties=5, losses=304, num_sims=1000, equity=0.6935)
        expected = "Equity: 69.3% | Win: 69.1% | Tie: 0.5% | Lose: 30.4% | (1,000 sims)"
        assert str(result) == expected

    def test_repr_representation(self) -> None:
        """Test the developer representation."""
        result = _build_equity_result(wins=50, ties=10, losses=40, num_sims=100, equity=0.55)
        assert repr(result) == (
            "EquityResult(equity=0.5500, win=0.5000, tie=0.1000, loss=0.4000, sims=100)"
        )


# =========================================================================
# Input Validation Tests
# =========================================================================


class TestCalculateEquityValidation:
    def test_hero_must_have_exactly_two_cards(self) -> None:
        hero = [C("Ah")]
        villain = [C("Ks"), C("Kd")]
        with pytest.raises(ValueError, match="Player 1 must have exactly 2 hole cards"):
            calculate_equity(hero, villain, num_sims=10)

    def test_villain_must_have_exactly_two_cards(self) -> None:
        hero = [C("Ah"), C("Kh")]
        villain = [C("Ks"), C("Kd"), C("Kc")]
        with pytest.raises(ValueError, match="Player 2 must have exactly 2 hole cards"):
            calculate_equity(hero, villain, num_sims=10)

    def test_board_cannot_have_more_than_five_cards(self) -> None:
        hero = [C("Ah"), C("Kh")]
        villain = [C("Ks"), C("Kd")]
        board = [C("2c"), C("3d"), C("4h"), C("5s"), C("6c"), C("7d")]
        with pytest.raises(ValueError, match="Board can have at most 5 cards"):
            calculate_equity(hero, villain, board, num_sims=10)

    def test_duplicate_cards_are_rejected(self) -> None:
        hero = [C("Ah"), C("Kh")]
        villain = [C("Ah"), C("Kd")]  # Ah is duplicated
        with pytest.raises(ValueError, match="Duplicate card: Ah"):
            calculate_equity(hero, villain, num_sims=10)

    def test_duplicate_card_in_board(self) -> None:
        hero = [C("Ah"), C("Kh")]
        villain = [C("Qs"), C("Qd")]
        board = [C("Ah"), C("2c"), C("3d")]  # Ah is in hero and board
        with pytest.raises(ValueError, match="Duplicate card: Ah"):
            calculate_equity(hero, villain, board, num_sims=10)

    def test_num_sims_must_be_positive(self) -> None:
        hero = [C("Ah"), C("Kh")]
        villain = [C("Qs"), C("Qd")]
        with pytest.raises(ValueError, match="num_sims must be positive"):
            calculate_equity(hero, villain, num_sims=0)


# =========================================================================
# Core Simulation Tests
# =========================================================================


class TestCalculateEquity:
    def test_deterministic_results_with_seed(self) -> None:
        """Same seed should produce identical results."""
        hero = [C("Ah"), C("Kh")]
        villain = [C("Qs"), C("Qd")]

        res1 = calculate_equity(hero, villain, num_sims=1000, seed=42)
        res2 = calculate_equity(hero, villain, num_sims=1000, seed=42)

        assert res1 == res2
        assert res1.wins == res2.wins
        assert res1.ties == res2.ties
        assert res1.losses == res2.losses

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds should produce slightly different results (usually)."""
        hero = [C("Ah"), C("Kh")]
        villain = [C("Qs"), C("Qd")]

        res1 = calculate_equity(hero, villain, num_sims=1000, seed=42)
        res2 = calculate_equity(hero, villain, num_sims=1000, seed=43)

        # It's highly unlikely (but theoretically possible) for 1000 sims
        # with different seeds to produce the exact same win/tie/loss counts.
        assert res1 != res2

    def test_known_equity_ak_vs_72(self) -> None:
        """AKs should crush 72o (around 67-70% equity)."""
        hero = [C("Ah"), C("Kh")]
        villain = [C("7s"), C("2d")]
        
        # We use a relatively high number of sims for a stable estimate,
        # but mock the RNG by using a seed to ensure the test never flakes.
        res = calculate_equity(hero, villain, num_sims=10_000, seed=123)
        
        assert 0.65 < res.equity < 0.72

    def test_known_equity_aa_vs_kk(self) -> None:
        """AA should crush KK (around 81-83% equity)."""
        hero = [C("As"), C("Ad")]
        villain = [C("Ks"), C("Kd")]
        
        res = calculate_equity(hero, villain, num_sims=10_000, seed=456)
        
        assert 0.80 < res.equity < 0.85

    def test_known_equity_postflop_draw(self) -> None:
        """AK suited vs QQ on a flush + straight draw board."""
        hero = [C("Ah"), C("Kh")]
        villain = [C("Qs"), C("Qd")]
        board = [C("Th"), C("9h"), C("2c")]
        
        # Hero has 15 outs (9 hearts, 3 non-heart Aces, 3 non-heart Kings).
        # Plus backdoor straight outs. Equity should be roughly 53-55%.
        res = calculate_equity(hero, villain, board, num_sims=10_000, seed=789)
        
        assert 0.51 < res.equity < 0.57

    def test_river_situation_100_percent_equity(self) -> None:
        """When there are no cards left to deal and hero has the nuts."""
        hero = [C("Ah"), C("Kh")]
        villain = [C("Qs"), C("Qd")]
        board = [C("Qh"), C("Jh"), C("Th"), C("2c"), C("3d")]  # Hero has Royal Flush
        
        res = calculate_equity(hero, villain, board, num_sims=10)
        
        assert res.wins == 10
        assert res.ties == 0
        assert res.losses == 0
        assert res.equity == 1.0

    def test_river_situation_0_percent_equity(self) -> None:
        """When there are no cards left to deal and hero is drawing dead."""
        hero = [C("Qs"), C("Qd")]
        villain = [C("Ah"), C("Kh")]
        board = [C("Qh"), C("Jh"), C("Th"), C("2c"), C("3d")]  # Villain has Royal Flush
        
        res = calculate_equity(hero, villain, board, num_sims=10)
        
        assert res.wins == 0
        assert res.ties == 0
        assert res.losses == 10
        assert res.equity == 0.0

    def test_river_situation_chop(self) -> None:
        """When the board is the nuts (Royal Flush on board)."""
        hero = [C("2s"), C("3s")]
        villain = [C("4c"), C("5c")]
        board = [C("Ah"), C("Kh"), C("Qh"), C("Jh"), C("Th")]
        
        res = calculate_equity(hero, villain, board, num_sims=10)
        
        assert res.wins == 0
        assert res.ties == 10
        assert res.losses == 0
        assert res.equity == 0.5

    def test_known_equity_aa_vs_random(self) -> None:
        """AA vs random hand should have around 85% equity."""
        hero = [C("As"), C("Ad")]
        
        # Test with villain=None (random hand)
        res = calculate_equity(hero, villain=None, num_sims=10_000, seed=101112)
        
        assert 0.83 < res.equity < 0.87
