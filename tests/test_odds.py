"""Unit tests for pot odds calculations and decision math."""

import math
import pytest

from poker_equity import (
    ev_bet,
    ev_call,
    ev_fold,
    expected_value,
    implied_odds,
    implied_odds_ratio,
    is_profitable_call,
    pot_odds,
    pot_odds_ratio,
    required_implied_odds,
    reverse_implied_odds,
)


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


class TestExpectedValue:
    """Tests for the expected_value and ev_call functions."""

    def test_break_even_call(self):
        """Call with break-even equity has zero EV."""
        # 150 pot, 50 call -> total 200 pot, 25% equity -> 0.25 * 200 - 50 = 0.0
        assert expected_value(0.25, 150.0, 50.0) == pytest.approx(0.0)

    def test_positive_ev_call(self):
        """Call with 50% equity in $150 pot with $50 to call."""
        # 0.50 * 200 - 50 = 50.0
        assert expected_value(0.50, 150.0, 50.0) == pytest.approx(50.0)

    def test_negative_ev_call(self):
        """Call with 10% equity in $150 pot with $50 to call."""
        # 0.10 * 200 - 50 = -30.0
        assert expected_value(0.10, 150.0, 50.0) == pytest.approx(-30.0)

    def test_nut_equity_call(self):
        """Calling with 100% equity wins the original pot."""
        # 1.0 * (100 + 50) - 50 = 100.0
        assert expected_value(1.0, 100.0, 50.0) == pytest.approx(100.0)

    def test_zero_equity_call(self):
        """Calling with 0% equity loses the call amount."""
        # 0.0 * (100 + 50) - 50 = -50.0
        assert expected_value(0.0, 100.0, 50.0) == pytest.approx(-50.0)

    def test_free_check_ev(self):
        """When bet_to_call is 0, EV is simply equity * pot_size."""
        assert expected_value(0.40, 100.0, 0.0) == pytest.approx(40.0)
        assert expected_value(0.0, 100.0, 0.0) == pytest.approx(0.0)

    def test_zero_pot_and_bet(self):
        """Empty pot and zero bet yields 0 EV."""
        assert expected_value(0.50, 0.0, 0.0) == pytest.approx(0.0)

    def test_integer_and_float_arguments(self):
        """Integers and floats should work interchangeably."""
        assert expected_value(0.5, 100, 50) == pytest.approx(25.0)

    def test_ev_call_alias_identical(self):
        """ev_call should return identical values to expected_value."""
        assert ev_call(0.35, 150.0, 50.0) == expected_value(0.35, 150.0, 50.0)

    def test_equity_out_of_bounds_raises(self):
        """Equity outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="equity must be between 0.0 and 1.0"):
            expected_value(-0.1, 100.0, 50.0)
        with pytest.raises(ValueError, match="equity must be between 0.0 and 1.0"):
            expected_value(1.1, 100.0, 50.0)

    def test_negative_pot_size_raises(self):
        """Negative pot_size raises ValueError."""
        with pytest.raises(ValueError, match="pot_size cannot be negative"):
            expected_value(0.5, -10.0, 50.0)

    def test_negative_bet_to_call_raises(self):
        """Negative bet_to_call raises ValueError."""
        with pytest.raises(ValueError, match="bet_to_call cannot be negative"):
            expected_value(0.5, 100.0, -10.0)

    @pytest.mark.parametrize("invalid", ["0.5", None, [0.5], True, False])
    def test_invalid_type_raises(self, invalid):
        """Non-numeric arguments raise TypeError."""
        with pytest.raises(TypeError):
            expected_value(invalid, 100.0, 50.0)
        with pytest.raises(TypeError):
            expected_value(0.5, invalid, 50.0)
        with pytest.raises(TypeError):
            expected_value(0.5, 100.0, invalid)


class TestEVFold:
    """Tests for the ev_fold function."""

    def test_ev_fold_is_zero(self):
        """Folding always has an EV of 0.0."""
        assert ev_fold() == 0.0


class TestEVBet:
    """Tests for the ev_bet function."""

    def test_pure_value_bet_no_folds(self):
        """Betting the nuts with zero fold equity."""
        # Hero bets 50 into 100, villain always calls and hero always wins:
        # 1.0 * (100 + 100) - 50 = +150.0
        assert ev_bet(equity=1.0, pot_size=100.0, bet_size=50.0, fold_equity=0.0) == pytest.approx(150.0)

    def test_pure_bluff_always_folds(self):
        """Betting air with 100% fold equity wins the current pot."""
        # 1.0 * 100.0 = 100.0
        assert ev_bet(equity=0.0, pot_size=100.0, bet_size=50.0, fold_equity=1.0) == pytest.approx(100.0)

    def test_pure_bluff_fifty_percent_folds(self):
        """Bluffing 50 into 100 with 50% fold equity and 0% equity when called."""
        # 0.5 * 100 + 0.5 * (-50) = 50 - 25 = 25.0
        assert ev_bet(equity=0.0, pot_size=100.0, bet_size=50.0, fold_equity=0.5) == pytest.approx(25.0)

    def test_semi_bluff(self):
        """Semi-bluff with draw: fold equity + showdown equity when called."""
        # 100 pot, 50 bet. fold_equity = 0.3, equity_when_called = 0.4
        # pot_if_called = 100 + 100 = 200
        # ev_called = 0.4 * 200 - 50 = 30
        # total_ev = 0.3 * 100 + 0.7 * 30 = 30 + 21 = 51.0
        assert ev_bet(equity=0.4, pot_size=100.0, bet_size=50.0, fold_equity=0.3) == pytest.approx(51.0)

    def test_default_fold_equity_is_zero(self):
        """fold_equity defaults to 0.0."""
        ev_default = ev_bet(equity=0.5, pot_size=100.0, bet_size=50.0)
        ev_explicit = ev_bet(equity=0.5, pot_size=100.0, bet_size=50.0, fold_equity=0.0)
        assert ev_default == ev_explicit

    def test_negative_pot_or_bet_raises(self):
        """Negative pot_size or bet_size raises ValueError."""
        with pytest.raises(ValueError):
            ev_bet(equity=0.5, pot_size=-10.0, bet_size=50.0)
        with pytest.raises(ValueError):
            ev_bet(equity=0.5, pot_size=100.0, bet_size=-50.0)

    def test_out_of_bounds_probabilities_raise(self):
        """Probabilities outside [0, 1] raise ValueError."""
        with pytest.raises(ValueError):
            ev_bet(equity=-0.1, pot_size=100.0, bet_size=50.0)
        with pytest.raises(ValueError):
            ev_bet(equity=1.1, pot_size=100.0, bet_size=50.0)
        with pytest.raises(ValueError):
            ev_bet(equity=0.5, pot_size=100.0, bet_size=50.0, fold_equity=-0.1)
        with pytest.raises(ValueError):
            ev_bet(equity=0.5, pot_size=100.0, bet_size=50.0, fold_equity=1.2)

    def test_invalid_types_raise(self):
        """Non-numeric or boolean values raise TypeError."""
        with pytest.raises(TypeError):
            ev_bet("0.5", 100.0, 50.0)
        with pytest.raises(TypeError):
            ev_bet(True, 100.0, 50.0)
        with pytest.raises(TypeError):
            ev_bet(0.5, 100.0, 50.0, fold_equity=False)


class TestImpliedOdds:
    """Tests for the implied_odds function."""

    def test_implied_odds_zero_future_bets_equals_pot_odds(self):
        """When future bets are 0, implied odds match pot odds."""
        assert implied_odds(150.0, 50.0, 0.0) == pytest.approx(pot_odds(150.0, 50.0))
        assert implied_odds(150.0, 50.0) == pytest.approx(0.25)

    def test_implied_odds_with_future_bets(self):
        """Future bets reduce the required break-even equity."""
        # 50 to call into 150 pot + 100 future bets -> 50 / (150 + 50 + 100) = 50 / 300 = 1/6
        assert implied_odds(150.0, 50.0, 100.0) == pytest.approx(1.0 / 6.0)

    def test_implied_odds_zero_bet_to_call(self):
        """Zero bet to call requires 0 equity."""
        assert implied_odds(100.0, 0.0, 50.0) == 0.0
        assert implied_odds(0.0, 0.0, 0.0) == 0.0

    def test_implied_odds_negative_inputs_raise(self):
        """Negative pot_size, bet_to_call, or future_bets raise ValueError."""
        with pytest.raises(ValueError, match="pot_size cannot be negative"):
            implied_odds(-10.0, 50.0, 50.0)
        with pytest.raises(ValueError, match="bet_to_call cannot be negative"):
            implied_odds(100.0, -10.0, 50.0)
        with pytest.raises(ValueError, match="future_bets cannot be negative"):
            implied_odds(100.0, 50.0, -10.0)

    @pytest.mark.parametrize("invalid", ["50", None, [50], True, False])
    def test_implied_odds_invalid_type_raises(self, invalid):
        """Non-numeric or bool values raise TypeError."""
        with pytest.raises(TypeError):
            implied_odds(invalid, 50.0, 50.0)
        with pytest.raises(TypeError):
            implied_odds(100.0, invalid, 50.0)
        with pytest.raises(TypeError):
            implied_odds(100.0, 50.0, invalid)


class TestRequiredImpliedOdds:
    """Tests for the required_implied_odds function."""

    def test_required_implied_odds_when_behind(self):
        """When equity is below direct pot odds, calculates required future winnings."""
        # Pot 150, call 50 (pot odds = 25%). Hero equity = 1/6 (~16.67%).
        # Needed = 50 / (1/6) - 200 = 300 - 200 = 100.0
        assert required_implied_odds(150.0, 50.0, 1.0 / 6.0) == pytest.approx(100.0)

    def test_required_implied_odds_when_already_profitable(self):
        """When direct equity exceeds pot odds, returns 0.0 additional chips."""
        # Pot 150, call 50 (needs 25%). Hero equity = 30% -> profitable now.
        assert required_implied_odds(150.0, 50.0, 0.30) == 0.0

    def test_required_implied_odds_at_breakeven(self):
        """When direct equity exactly equals pot odds, returns 0.0."""
        assert required_implied_odds(150.0, 50.0, 0.25) == pytest.approx(0.0)

    def test_required_implied_odds_zero_equity(self):
        """Zero equity with positive bet to call returns inf (impossible to break even)."""
        assert required_implied_odds(150.0, 50.0, 0.0) == float("inf")

    def test_required_implied_odds_zero_bet_to_call(self):
        """Zero bet to call costs nothing, requires 0 future winnings."""
        assert required_implied_odds(150.0, 0.0, 0.10) == 0.0
        assert required_implied_odds(150.0, 0.0, 0.0) == 0.0

    def test_required_implied_odds_validation_errors(self):
        """Negative amounts or out of bounds equity raise ValueError."""
        with pytest.raises(ValueError, match="equity must be between 0.0 and 1.0"):
            required_implied_odds(150.0, 50.0, -0.1)
        with pytest.raises(ValueError, match="equity must be between 0.0 and 1.0"):
            required_implied_odds(150.0, 50.0, 1.1)
        with pytest.raises(ValueError, match="pot_size cannot be negative"):
            required_implied_odds(-10.0, 50.0, 0.2)
        with pytest.raises(ValueError, match="bet_to_call cannot be negative"):
            required_implied_odds(100.0, -5.0, 0.2)

    @pytest.mark.parametrize("invalid", ["0.2", None, [0.2], True, False])
    def test_required_implied_odds_invalid_type_raises(self, invalid):
        """Non-numeric or bool values raise TypeError."""
        with pytest.raises(TypeError):
            required_implied_odds(invalid, 50.0, 0.2)
        with pytest.raises(TypeError):
            required_implied_odds(100.0, invalid, 0.2)
        with pytest.raises(TypeError):
            required_implied_odds(100.0, 50.0, invalid)


class TestImpliedOddsRatio:
    """Tests for the implied_odds_ratio function."""

    def test_implied_odds_ratio_standard(self):
        """Pot of 150, call 50, future 100 gives 5:1 odds."""
        ratio, base = implied_odds_ratio(150.0, 50.0, 100.0)
        assert ratio == pytest.approx(5.0)
        assert base == 1.0

    def test_implied_odds_ratio_default_future_bets(self):
        """Default future_bets is 0.0."""
        ratio, base = implied_odds_ratio(150.0, 50.0)
        assert ratio == pytest.approx(3.0)
        assert base == 1.0

    def test_implied_odds_ratio_zero_bet(self):
        """Zero bet to call gives infinite odds."""
        ratio, base = implied_odds_ratio(100.0, 0.0, 50.0)
        assert math.isinf(ratio)
        assert base == 1.0

    def test_implied_odds_ratio_validation_errors(self):
        """Negative inputs raise ValueError."""
        with pytest.raises(ValueError):
            implied_odds_ratio(-10.0, 50.0, 50.0)
        with pytest.raises(ValueError):
            implied_odds_ratio(100.0, -10.0, 50.0)
        with pytest.raises(ValueError):
            implied_odds_ratio(100.0, 50.0, -10.0)

    def test_implied_odds_ratio_type_errors(self):
        """Non-numeric or bool inputs raise TypeError."""
        with pytest.raises(TypeError):
            implied_odds_ratio("100", 50.0, 50.0)
        with pytest.raises(TypeError):
            implied_odds_ratio(100.0, True, 50.0)
        with pytest.raises(TypeError):
            implied_odds_ratio(100.0, 50.0, None)


class TestReverseImpliedOdds:
    """Tests for the reverse_implied_odds function."""

    def test_reverse_implied_odds_zero_losses_equals_pot_odds(self):
        """When future losses are 0, reverse implied odds match pot odds."""
        assert reverse_implied_odds(150.0, 50.0, 0.0) == pytest.approx(pot_odds(150.0, 50.0))
        assert reverse_implied_odds(150.0, 50.0) == pytest.approx(0.25)

    def test_reverse_implied_odds_with_future_losses(self):
        """Future losses increase the required break-even equity."""
        # 50 call + 50 future loss = 100 / (150 + 50 + 50) = 100 / 250 = 0.40
        assert reverse_implied_odds(150.0, 50.0, 50.0) == pytest.approx(0.40)

    def test_reverse_implied_odds_zero_bet_and_loss(self):
        """Zero bet and zero future losses requires 0 equity."""
        assert reverse_implied_odds(100.0, 0.0, 0.0) == 0.0
        assert reverse_implied_odds(0.0, 0.0, 0.0) == 0.0

    def test_reverse_implied_odds_validation_errors(self):
        """Negative inputs raise ValueError."""
        with pytest.raises(ValueError, match="pot_size cannot be negative"):
            reverse_implied_odds(-10.0, 50.0, 20.0)
        with pytest.raises(ValueError, match="bet_to_call cannot be negative"):
            reverse_implied_odds(100.0, -5.0, 20.0)
        with pytest.raises(ValueError, match="future_losses cannot be negative"):
            reverse_implied_odds(100.0, 50.0, -10.0)

    @pytest.mark.parametrize("invalid", ["50", None, [50], True, False])
    def test_reverse_implied_odds_type_errors(self, invalid):
        """Non-numeric or bool inputs raise TypeError."""
        with pytest.raises(TypeError):
            reverse_implied_odds(invalid, 50.0, 20.0)
        with pytest.raises(TypeError):
            reverse_implied_odds(100.0, invalid, 20.0)
        with pytest.raises(TypeError):
            reverse_implied_odds(100.0, 50.0, invalid)



