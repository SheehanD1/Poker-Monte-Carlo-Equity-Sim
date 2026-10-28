"""Unit tests for pot odds calculations and decision math."""

import math
import pytest

from poker_equity import is_profitable_call, pot_odds, pot_odds_ratio


class TestPotOdds:
    """Tests for the pot_odds function."""

    def test_half_pot_bet(self):
        """Facing a half-pot bet (e.g. pot is $150 after $50 bet, call $50)."""
        # 50 / (150 + 50) = 50 / 200 = 0.25
        assert pot_odds(150.0, 50.0) == pytest.approx(0.25)

    def test_full_pot_bet(self):
        """Facing a pot-sized bet (e.g. pot is $200 after $100 bet, call $100)."""
        # 100 / (200 + 100) = 100 / 300 = 1/3
        assert pot_odds(200.0, 100.0) == pytest.approx(1.0 / 3.0)

    def test_quarter_pot_bet(self):
        """Facing a 1/4 pot bet (e.g. pot $125 after $25 bet, call $25)."""
        # 25 / (125 + 25) = 25 / 150 = 1/6
        assert pot_odds(125.0, 25.0) == pytest.approx(1.0 / 6.0)

    def test_overbet(self):
        """Facing an overbet (e.g. pot $300 after $200 bet into $100 pot, call $200)."""
        # 200 / (300 + 200) = 200 / 500 = 0.40
        assert pot_odds(300.0, 200.0) == pytest.approx(0.40)

    def test_zero_bet_to_call(self):
        """Checking or facing a $0 bet costs nothing, break-even equity is 0."""
        assert pot_odds(100.0, 0.0) == 0.0
        assert pot_odds(0.0, 0.0) == 0.0

    def test_zero_pot_size_with_bet(self):
        """Facing a bet into an empty pot requires 100% equity to break even."""
        assert pot_odds(0.0, 50.0) == pytest.approx(1.0)

    def test_integer_and_float_arguments(self):
        """Both integers and floats should be accepted."""
        assert pot_odds(150, 50) == pytest.approx(0.25)
        assert pot_odds(150.5, 50.5) == pytest.approx(50.5 / 201.0)

    def test_negative_pot_size_raises(self):
        """Negative pot_size should raise ValueError."""
        with pytest.raises(ValueError, match="pot_size cannot be negative"):
            pot_odds(-10.0, 50.0)

    def test_negative_bet_to_call_raises(self):
        """Negative bet_to_call should raise ValueError."""
        with pytest.raises(ValueError, match="bet_to_call cannot be negative"):
            pot_odds(100.0, -5.0)

    @pytest.mark.parametrize("invalid_pot", ["100", None, [100], True, False])
    def test_invalid_pot_size_type_raises(self, invalid_pot):
        """Non-numeric pot_size should raise TypeError."""
        with pytest.raises(TypeError, match="pot_size must be a number"):
            pot_odds(invalid_pot, 50.0)

    @pytest.mark.parametrize("invalid_bet", ["50", None, [50], True, False])
    def test_invalid_bet_to_call_type_raises(self, invalid_bet):
        """Non-numeric bet_to_call should raise TypeError."""
        with pytest.raises(TypeError, match="bet_to_call must be a number"):
            pot_odds(100.0, invalid_bet)


class TestPotOddsRatio:
    """Tests for the pot_odds_ratio function."""

    def test_standard_ratio(self):
        """Pot of 150, call 50 gives 3:1 pot odds."""
        ratio, base = pot_odds_ratio(150.0, 50.0)
        assert ratio == pytest.approx(3.0)
        assert base == 1.0

    def test_even_money_ratio(self):
        """Pot of 100, call 100 gives 1:1 pot odds."""
        ratio, base = pot_odds_ratio(100.0, 100.0)
        assert ratio == pytest.approx(1.0)
        assert base == 1.0

    def test_zero_bet_returns_inf(self):
        """Zero bet to call gives infinite odds."""
        ratio, base = pot_odds_ratio(100.0, 0.0)
        assert math.isinf(ratio)
        assert base == 1.0

    def test_negative_inputs_raise(self):
        """Negative inputs should raise ValueError."""
        with pytest.raises(ValueError):
            pot_odds_ratio(-10.0, 50.0)
        with pytest.raises(ValueError):
            pot_odds_ratio(100.0, -10.0)

    def test_type_error_on_non_numeric(self):
        """Non-numeric inputs should raise TypeError."""
        with pytest.raises(TypeError):
            pot_odds_ratio("100", 50.0)
        with pytest.raises(TypeError):
            pot_odds_ratio(100.0, "50")
        with pytest.raises(TypeError):
            pot_odds_ratio(True, 50.0)


class TestIsProfitableCall:
    """Tests for the is_profitable_call function."""

    def test_profitable_call(self):
        """Equity higher than pot odds requirement is profitable."""
        # 150 pot, 50 call -> requires 25% equity
        assert is_profitable_call(0.30, 150.0, 50.0) is True

    def test_break_even_call(self):
        """Equity exactly equal to pot odds requirement is break-even (True)."""
        assert is_profitable_call(0.25, 150.0, 50.0) is True

    def test_unprofitable_call(self):
        """Equity lower than pot odds requirement is unprofitable."""
        assert is_profitable_call(0.20, 150.0, 50.0) is False

    def test_free_call_always_profitable(self):
        """Any non-negative equity is profitable when calling costs 0."""
        assert is_profitable_call(0.01, 100.0, 0.0) is True
        assert is_profitable_call(0.0, 100.0, 0.0) is True

    def test_equity_out_of_bounds_raises(self):
        """Equity outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="equity must be between 0.0 and 1.0"):
            is_profitable_call(-0.01, 150.0, 50.0)
        with pytest.raises(ValueError, match="equity must be between 0.0 and 1.0"):
            is_profitable_call(1.05, 150.0, 50.0)

    @pytest.mark.parametrize("invalid_eq", ["0.5", None, [0.5], True, False])
    def test_invalid_equity_type_raises(self, invalid_eq):
        """Non-numeric equity should raise TypeError."""
        with pytest.raises(TypeError, match="equity must be a number"):
            is_profitable_call(invalid_eq, 150.0, 50.0)

