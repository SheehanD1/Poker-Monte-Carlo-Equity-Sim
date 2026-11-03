"""Unit tests for the decision recommendation engine."""

import pytest

from poker_equity import (
    Action,
    DecisionEngine,
    DecisionResult,
    recommend_decision,
    recommend_drawing_decision,
    recommend_hand_decision,
)


class TestActionEnum:
    """Tests for the Action enum."""

    def test_enum_values_and_str(self):
        """Action values and string representation."""
        assert Action.FOLD.value == "fold"
        assert str(Action.FOLD) == "FOLD"
        assert Action.CALL.value == "call"
        assert str(Action.CALL) == "CALL"
        assert Action.CHECK.value == "check"
        assert str(Action.CHECK) == "CHECK"
        assert Action.BET.value == "bet"
        assert str(Action.BET) == "BET"
        assert Action.RAISE.value == "raise"
        assert str(Action.RAISE) == "RAISE"
        assert Action.ALL_IN.value == "all_in"
        assert str(Action.ALL_IN) == "ALL_IN"

    def test_aggressive_and_passive_properties(self):
        """Categorization of actions into aggressive and passive."""
        assert Action.BET.is_aggressive is True
        assert Action.RAISE.is_aggressive is True
        assert Action.ALL_IN.is_aggressive is True
        assert Action.CHECK.is_aggressive is False
        assert Action.CALL.is_aggressive is False
        assert Action.FOLD.is_aggressive is False

        assert Action.CHECK.is_passive is True
        assert Action.CALL.is_passive is True
        assert Action.BET.is_passive is False
        assert Action.RAISE.is_passive is False


class TestDecisionResult:
    """Tests for the DecisionResult container."""

    def test_properties_and_summary(self):
        """is_profitable, ev_difference, and summary string formatting."""
        evs = {Action.FOLD: 0.0, Action.CALL: 25.0, Action.RAISE: 40.0}
        res = DecisionResult(
            action=Action.RAISE,
            ev=40.0,
            ev_by_action=evs,
            equity=0.60,
            pot_odds=0.25,
            pot_size=150.0,
            bet_to_call=50.0,
            raise_size=150.0,
            fold_equity=0.20,
            reasoning="Raise is highest EV line.",
            alternative_action=Action.CALL,
        )

        assert res.is_profitable is True
        assert res.ev_difference == pytest.approx(15.0)  # 40.0 - 25.0

        summary = res.summary()
        assert "RAISE" in summary
        assert "+40.00 chips" in summary
        assert "60.0%" in summary
        assert "150.00" in summary
        assert "Action EV Comparison:" in summary


class TestDecisionEngineFacingNoBet:
    """Tests for scenarios where bet_to_call is 0 (Checking is free)."""

    def test_strong_hand_bets(self):
        """High equity hand bets into an unraised pot."""
        res = recommend_decision(equity=0.80, pot_size=100.0, bet_to_call=0.0)
        assert res.action in (Action.BET, Action.ALL_IN)
        assert res.ev > 0.0
        assert Action.CHECK in res.ev_by_action

    def test_marginal_hand_checks(self):
        """Weak to marginal hand checks when fold equity is low."""
        # 30% equity with 0 fold equity:
        # Check EV = 0.3 * 100 = 30.0
        # Bet 50 EV = 0.3 * 200 - 50 = 10.0
        res = recommend_decision(equity=0.30, pot_size=100.0, bet_to_call=0.0, fold_equity=0.0)
        assert res.action == Action.CHECK
        assert res.ev == pytest.approx(30.0)

    def test_short_stack_bets_all_in(self):
        """When standard bet exceeds remaining stack, recommend ALL_IN."""
        # Pot is 100, standard half-pot bet is 50, but Hero only has 30 chips
        res = recommend_decision(equity=0.75, pot_size=100.0, bet_to_call=0.0, stack_size=30.0)
        assert res.action == Action.ALL_IN
        assert res.raise_size == 30.0


class TestDecisionEngineFacingBet:
    """Tests for scenarios where Hero faces a bet or raise."""

    def test_profitable_call_over_fold(self):
        """Hero has 35% equity facing $50 into $150 pot (requires 25%). Recommends CALL."""
        res = recommend_decision(
            equity=0.35,
            pot_size=150.0,
            bet_to_call=50.0,
            fold_equity=0.0,
        )
        assert res.action == Action.CALL
        assert res.is_profitable is True
        # EV call = 0.35 * 200 - 50 = 20.0
        assert res.ev == pytest.approx(20.0)

    def test_value_raise_dominates_call(self):
        """Monster equity (85%) with positive fold equity recommends RAISE."""
        res = recommend_decision(
            equity=0.85,
            pot_size=150.0,
            bet_to_call=50.0,
            fold_equity=0.25,
            raise_size=150.0,
        )
        assert res.action == Action.RAISE
        assert res.ev > res.ev_by_action[Action.CALL]
        assert res.raise_size == 150.0

    def test_semi_bluff_raise(self):
        """Draw with decent equity (40%) and high fold equity (50%) prefers RAISE over CALL."""
        res = recommend_decision(
            equity=0.40,
            pot_size=100.0,
            bet_to_call=50.0,
            fold_equity=0.50,
            raise_size=150.0,
        )
        assert res.action == Action.RAISE
        assert res.ev > res.ev_by_action[Action.CALL]

    def test_unprofitable_spot_folds(self):
        """Underdog with 10% equity facing 25% pot odds folds."""
        res = recommend_decision(
            equity=0.10,
            pot_size=150.0,
            bet_to_call=50.0,
            fold_equity=0.0,
        )
        assert res.action == Action.FOLD
        assert res.ev == 0.0
        assert res.ev_by_action[Action.CALL] < 0.0

    def test_all_in_call_when_stack_committed(self):
        """Hero cannot raise when calling puts Hero all-in."""
        res = recommend_decision(
            equity=0.40,
            pot_size=150.0,
            bet_to_call=50.0,
            stack_size=50.0,  # stack equals the bet to call
        )
        assert res.action == Action.CALL
        # Raise should not be in legal actions
        assert Action.RAISE not in res.ev_by_action

    def test_implied_odds_converts_fold_to_call(self):
        """Facing a bet where direct odds are insufficient, implied odds makes CALL profitable."""
        # 15% equity facing 50 into 150 (needs 25% direct pot odds)
        # Direct call EV = 0.15 * 200 - 50 = -20 (Fold recommended)
        res_direct = recommend_decision(
            equity=0.15,
            pot_size=150.0,
            bet_to_call=50.0,
            future_bets=0.0,
        )
        assert res_direct.action == Action.FOLD

        # With 200 expected river chips:
        # Effective pot = 350 + 50 = 400. EV = 0.15 * 400 - 50 = +10.0 (Call recommended)
        res_implied = recommend_decision(
            equity=0.15,
            pot_size=150.0,
            bet_to_call=50.0,
            future_bets=200.0,
        )
        assert res_implied.action == Action.CALL
        assert res_implied.is_profitable is True


class TestHandAndDrawingConvenienceFunctions:
    """Tests for recommend_hand_decision and recommend_drawing_decision."""

    def test_recommend_hand_decision_preflop_monsters(self):
        """AA vs 72o preflop recommends raising."""
        res = recommend_hand_decision(
            hero="Ah As",
            villain="7c 2d",
            board=None,
            pot_size=10.0,
            bet_to_call=2.0,
            num_sims=1000,
            seed=42,
        )
        assert res.action == Action.RAISE
        assert res.is_profitable is True
        assert res.equity > 0.80

    def test_recommend_hand_decision_underdog_folds(self):
        """72o vs AA preflop facing a bet recommends folding."""
        res = recommend_hand_decision(
            hero="7c 2d",
            villain="Ah As",
            board=None,
            pot_size=30.0,
            bet_to_call=15.0,  # requires 15 / 45 = 33.3% equity, 72o has ~12%
            num_sims=1000,
            seed=42,
        )
        assert res.action == Action.FOLD

    def test_recommend_drawing_decision_turn_flush_draw(self):
        """Ah Kh vs Qc Qd on 2h 7h 8c 3c with 15 outs facing small bet calls."""
        res = recommend_drawing_decision(
            hero="Ah Kh",
            villain="Qc Qd",
            board="2h 7h 8c 3c",
            pot_size=150.0,
            bet_to_call=50.0,
            method="exact",
        )
        assert res.action == Action.CALL
        assert res.is_profitable is True


class TestDecisionEngineValidation:
    """Tests for validation errors and constructor parameters."""

    def test_engine_init_validation(self):
        """Validation of DecisionEngine init arguments."""
        with pytest.raises(ValueError):
            DecisionEngine(default_fold_equity=-0.1)
        with pytest.raises(ValueError):
            DecisionEngine(default_raise_multiplier=0.5)
        with pytest.raises(ValueError):
            DecisionEngine(default_bet_fraction=0.0)

    def test_recommend_validation_errors(self):
        """Invalid inputs to recommend raise ValueError or TypeError."""
        engine = DecisionEngine()
        with pytest.raises(ValueError, match="equity must be between"):
            engine.recommend(equity=-0.1, pot_size=100.0)
        with pytest.raises(ValueError, match="pot_size cannot be negative"):
            engine.recommend(equity=0.5, pot_size=-10.0)
        with pytest.raises(ValueError, match="bet_to_call cannot be negative"):
            engine.recommend(equity=0.5, pot_size=100.0, bet_to_call=-5.0)
        with pytest.raises(ValueError, match="stack_size cannot be negative"):
            engine.recommend(equity=0.5, pot_size=100.0, stack_size=-100.0)

    @pytest.mark.parametrize("invalid", ["0.5", None, [0.5], True, False])
    def test_recommend_type_errors(self, invalid):
        """Non-numeric or boolean arguments raise TypeError."""
        engine = DecisionEngine()
        with pytest.raises(TypeError):
            engine.recommend(equity=invalid, pot_size=100.0)
        with pytest.raises(TypeError):
            engine.recommend(equity=0.5, pot_size=invalid)

