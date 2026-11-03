"""Decision recommendation engine for Texas Hold'em.

This module provides strategic decision recommendations (Fold, Check, Call,
Bet, Raise, All-In) based on mathematical expected value (EV), pot odds,
equity, fold equity, stack sizes, and implied odds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from poker_equity.card import Card
from poker_equity.evaluator import evaluate_hand
from poker_equity.odds import ev_bet, ev_call, ev_fold, expected_value, pot_odds
from poker_equity.outs import calculate_outs, outs_to_equity
from poker_equity.simulation import calculate_equity


# =========================================================================
# Action Enumeration
# =========================================================================


class Action(str, Enum):
    """Standard poker player actions."""

    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"

    def __str__(self) -> str:
        return self.value.upper()

    @property
    def is_aggressive(self) -> bool:
        """Whether the action is an aggressive wager (Bet, Raise, All-In)."""
        return self in (Action.BET, Action.RAISE, Action.ALL_IN)

    @property
    def is_passive(self) -> bool:
        """Whether the action is a passive play (Check, Call)."""
        return self in (Action.CHECK, Action.CALL)


# =========================================================================
# Decision Result Container
# =========================================================================


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """The result and strategic analysis of a poker decision recommendation.

    Attributes
    ----------
    action : Action
        The primary recommended action (e.g. ``Action.CALL``, ``Action.RAISE``).
    ev : float
        The expected value of the recommended primary action in chips.
    ev_by_action : dict[Action, float]
        Expected value mapped for each evaluated legal action.
    equity : float
        Hero's equity in the hand as a fraction in ``[0.0, 1.0]``.
    pot_odds : float
        Required break-even equity to call.
    pot_size : float
        The current pot size facing the player.
    bet_to_call : float
        Amount required to stay in the hand.
    raise_size : float | None
        Suggested wager amount if betting or raising; None otherwise.
    fold_equity : float
        Estimated probability that opponent folds to an aggressive wager.
    reasoning : str
        Human-readable explanation of why this action is recommended.
    alternative_action : Action | None
        Secondary viable action if another line has comparable EV; None otherwise.
    """

    action: Action
    ev: float
    ev_by_action: dict[Action, float]
    equity: float
    pot_odds: float
    pot_size: float
    bet_to_call: float
    raise_size: float | None
    fold_equity: float
    reasoning: str
    alternative_action: Action | None = None

    @property
    def is_profitable(self) -> bool:
        """Whether the recommended action has positive or zero EV."""
        return self.ev >= 0.0

    @property
    def ev_difference(self) -> float:
        """Chip difference between the best action and the second-best action."""
        sorted_evs = sorted(self.ev_by_action.values(), reverse=True)
        if len(sorted_evs) >= 2:
            return sorted_evs[0] - sorted_evs[1]
        return 0.0

    def summary(self) -> str:
        """Return a formatted text summary of the decision recommendation."""
        lines = [
            f"Recommended Action: {self.action}",
            f"Expected Value (EV): {self.ev:+.2f} chips",
            f"Hero Equity: {self.equity * 100:.1f}%",
            f"Pot Odds: {self.pot_odds * 100:.1f}% (Pot: {self.pot_size:.2f}, Call: {self.bet_to_call:.2f})",
        ]
        if self.raise_size is not None:
            lines.append(f"Suggested Sizing: {self.raise_size:.2f} chips")
        if self.fold_equity > 0:
            lines.append(f"Fold Equity: {self.fold_equity * 100:.1f}%")

        lines.append("Action EV Comparison:")
        for act, ev_val in sorted(self.ev_by_action.items(), key=lambda item: item[1], reverse=True):
            marker = " *" if act == self.action else ""
            lines.append(f"  {act.value.upper():<7}: {ev_val:+.2f} chips{marker}")

        lines.append(f"Analysis: {self.reasoning}")
        return "\n".join(lines)


# =========================================================================
# Decision Recommendation Engine
# =========================================================================


class DecisionEngine:
    """Configurable decision recommendation engine for Texas Hold'em spots.

    Parameters
    ----------
    default_fold_equity : float, default=0.0
        Default probability that opponent folds to a bet or raise.
    default_raise_multiplier : float, default=3.0
        Standard sizing multiplier for raises when facing a bet (e.g. 3x the bet).
    default_bet_fraction : float, default=0.5
        Standard sizing fraction of the pot when betting into an unraised pot (e.g. 0.5x pot).
    """

    def __init__(
        self,
        default_fold_equity: float = 0.0,
        default_raise_multiplier: float = 3.0,
        default_bet_fraction: float = 0.5,
    ) -> None:
        if not (0.0 <= default_fold_equity <= 1.0):
            raise ValueError(f"default_fold_equity must be in [0, 1], got {default_fold_equity}")
        if default_raise_multiplier < 1.0:
            raise ValueError(f"default_raise_multiplier must be >= 1.0, got {default_raise_multiplier}")
        if default_bet_fraction <= 0.0:
            raise ValueError(f"default_bet_fraction must be > 0.0, got {default_bet_fraction}")

        self.default_fold_equity = default_fold_equity
        self.default_raise_multiplier = default_raise_multiplier
        self.default_bet_fraction = default_bet_fraction

    def recommend(
        self,
        equity: float,
        pot_size: float,
        bet_to_call: float = 0.0,
        stack_size: float | None = None,
        fold_equity: float | None = None,
        raise_size: float | None = None,
        future_bets: float = 0.0,
    ) -> DecisionResult:
        """Analyze a poker spot and recommend the mathematically optimal action.

        Parameters
        ----------
        equity : float
            Hero's winning probability as a fraction in ``[0.0, 1.0]``.
        pot_size : float
            Current pot size before Hero acts.
        bet_to_call : float, default=0.0
            Cost for Hero to continue in the hand.
        stack_size : float or None, default=None
            Remaining chip stack for Hero.  If a bet or raise meets or exceeds
            this amount, ``Action.ALL_IN`` is recommended.
        fold_equity : float or None, default=None
            Estimated fold equity. If None, uses ``self.default_fold_equity``.
        raise_size : float or None, default=None
            Explicit sizing for bet/raise; if None, standard sizing is used.
        future_bets : float, default=0.0
            Anticipated future chips won on later streets (implied odds).

        Returns
        -------
        DecisionResult
            Comprehensive analysis including recommended action and EV breakdown.

        Raises
        ------
        ValueError
            If numerical arguments are out of bounds or negative.
        TypeError
            If inputs are not numeric.
        """
        # Type and value validations
        if isinstance(equity, bool) or not isinstance(equity, (int, float)):
            raise TypeError(f"equity must be a number, got {type(equity).__name__}")
        if isinstance(pot_size, bool) or not isinstance(pot_size, (int, float)):
            raise TypeError(f"pot_size must be a number, got {type(pot_size).__name__}")
        if isinstance(bet_to_call, bool) or not isinstance(bet_to_call, (int, float)):
            raise TypeError(f"bet_to_call must be a number, got {type(bet_to_call).__name__}")
        if isinstance(future_bets, bool) or not isinstance(future_bets, (int, float)):
            raise TypeError(f"future_bets must be a number, got {type(future_bets).__name__}")

        if not (0.0 <= equity <= 1.0):
            raise ValueError(f"equity must be between 0.0 and 1.0, got {equity}")
        if pot_size < 0:
            raise ValueError(f"pot_size cannot be negative, got {pot_size}")
        if bet_to_call < 0:
            raise ValueError(f"bet_to_call cannot be negative, got {bet_to_call}")
        if future_bets < 0:
            raise ValueError(f"future_bets cannot be negative, got {future_bets}")

        fe = self.default_fold_equity if fold_equity is None else fold_equity
        if isinstance(fe, bool) or not isinstance(fe, (int, float)):
            raise TypeError(f"fold_equity must be a number, got {type(fe).__name__}")
        if not (0.0 <= fe <= 1.0):
            raise ValueError(f"fold_equity must be between 0.0 and 1.0, got {fe}")

        if stack_size is not None:
            if isinstance(stack_size, bool) or not isinstance(stack_size, (int, float)):
                raise TypeError(f"stack_size must be a number, got {type(stack_size).__name__}")
            if stack_size < 0:
                raise ValueError(f"stack_size cannot be negative, got {stack_size}")

        if raise_size is not None:
            if isinstance(raise_size, bool) or not isinstance(raise_size, (int, float)):
                raise TypeError(f"raise_size must be a number, got {type(raise_size).__name__}")
            if raise_size < 0:
                raise ValueError(f"raise_size cannot be negative, got {raise_size}")

        # Compute pot odds requirement
        required_pot_odds = pot_odds(pot_size, bet_to_call)

        # -------------------------------------------------------------
        # Scenario A: Facing no bet (Checking is free)
        # -------------------------------------------------------------
        if bet_to_call == 0.0:
            ev_chk = equity * pot_size

            # Determine bet sizing
            if raise_size is not None:
                bet_amt = raise_size
            else:
                bet_amt = max(1.0, pot_size * self.default_bet_fraction)

            if stack_size is not None and bet_amt >= stack_size:
                bet_amt = stack_size
                bet_act = Action.ALL_IN
            else:
                bet_act = Action.BET

            # EV of betting
            ev_b = ev_bet(equity, pot_size, bet_amt, fold_equity=fe)

            ev_by_act = {Action.CHECK: ev_chk, bet_act: ev_b}

            if ev_b > ev_chk:
                rec_action = bet_act
                rec_ev = ev_b
                alt_action = Action.CHECK
                reasoning = (
                    f"Betting ({bet_act.value}) is recommended with EV of {ev_b:+.2f} "
                    f"chips vs Check EV of {ev_chk:+.2f}. "
                    f"Equity is {equity * 100:.1f}% with {fe * 100:.1f}% fold equity."
                )
            else:
                rec_action = Action.CHECK
                rec_ev = ev_chk
                alt_action = bet_act if abs(ev_chk - ev_b) < 1e-4 else None
                reasoning = (
                    f"Checking is recommended with EV of {ev_chk:+.2f} chips. "
                    f"Checking yields higher EV than betting ({ev_b:+.2f})."
                )

            return DecisionResult(
                action=rec_action,
                ev=rec_ev,
                ev_by_action=ev_by_act,
                equity=equity,
                pot_odds=0.0,
                pot_size=pot_size,
                bet_to_call=0.0,
                raise_size=bet_amt if rec_action.is_aggressive else None,
                fold_equity=fe,
                reasoning=reasoning,
                alternative_action=alt_action,
            )

        # -------------------------------------------------------------
        # Scenario B: Facing a bet (Fold, Call, or Raise)
        # -------------------------------------------------------------
        ev_f = ev_fold()  # 0.0
        effective_call_pot = pot_size + future_bets
        ev_c = expected_value(equity, effective_call_pot, bet_to_call)

        # Check if Hero's stack limits calling or raising
        is_all_in_call = stack_size is not None and bet_to_call >= stack_size

        if is_all_in_call:
            # Cannot raise if calling already puts Hero all-in
            ev_by_act = {Action.FOLD: ev_f, Action.CALL: ev_c}
            if ev_c >= ev_f:
                rec_action = Action.CALL
                rec_ev = ev_c
                reasoning = (
                    f"All-in call is profitable ({ev_c:+.2f} EV). "
                    f"Hero equity ({equity * 100:.1f}%) meets/exceeds pot odds "
                    f"requirement ({required_pot_odds * 100:.1f}%)."
                )
            else:
                rec_action = Action.FOLD
                rec_ev = ev_f
                reasoning = (
                    f"Fold is recommended (0.00 EV). Calling is -EV ({ev_c:+.2f}) "
                    f"because equity ({equity * 100:.1f}%) is below pot odds "
                    f"({required_pot_odds * 100:.1f}%)."
                )
            return DecisionResult(
                action=rec_action,
                ev=rec_ev,
                ev_by_action=ev_by_act,
                equity=equity,
                pot_odds=required_pot_odds,
                pot_size=pot_size,
                bet_to_call=bet_to_call,
                raise_size=None,
                fold_equity=fe,
                reasoning=reasoning,
                alternative_action=None,
            )

        # Determine raise sizing and action
        if raise_size is not None:
            raise_total = max(bet_to_call * 2.0, raise_size)
        else:
            raise_total = max(bet_to_call * 2.0, bet_to_call * self.default_raise_multiplier)

        if stack_size is not None and raise_total >= stack_size:
            raise_total = stack_size
            raise_act = Action.ALL_IN
        else:
            raise_act = Action.RAISE

        # Calculate EV of raising facing a bet
        # Hero risks raise_total.
        # With fold equity fe, Hero wins current pot (pot_size + bet_to_call) immediately.
        # When called (1 - fe), opponent calls the raise adding (raise_total - bet_to_call):
        # Pot at showdown = pot_size + bet_to_call + 2 * (raise_total - bet_to_call)
        pot_at_showdown = pot_size - bet_to_call + (2.0 * raise_total)
        ev_when_called = (equity * pot_at_showdown) - raise_total
        ev_r = (fe * (pot_size + bet_to_call)) + ((1.0 - fe) * ev_when_called)

        ev_by_act = {
            Action.FOLD: ev_f,
            Action.CALL: ev_c,
            raise_act: ev_r,
        }

        # Compare actions
        if ev_r > ev_c and ev_r > ev_f:
            rec_action = raise_act
            rec_ev = ev_r
            alt_action = Action.CALL if ev_c >= ev_f else Action.FOLD
            reasoning = (
                f"{raise_act.value.capitalize()} is the highest EV line ({ev_r:+.2f} chips) "
                f"vs Call ({ev_c:+.2f}) and Fold (0.00). "
                f"Equity: {equity * 100:.1f}%, Fold Equity: {fe * 100:.1f}%."
            )
        elif ev_c >= ev_f:
            rec_action = Action.CALL
            rec_ev = ev_c
            alt_action = raise_act if ev_r > ev_f else Action.FOLD
            implied_note = " (including implied odds)" if future_bets > 0 else ""
            reasoning = (
                f"Call is profitable ({ev_c:+.2f} EV){implied_note} and preferred over "
                f"{raise_act.value} ({ev_r:+.2f}). "
                f"Equity ({equity * 100:.1f}%) satisfies pot odds ({required_pot_odds * 100:.1f}%)."
            )
        else:
            rec_action = Action.FOLD
            rec_ev = ev_f
            alt_action = raise_act if ev_r > ev_c else Action.CALL
            reasoning = (
                f"Fold is recommended (0.00 EV). Calling is -EV ({ev_c:+.2f}) and "
                f"{raise_act.value} is -EV ({ev_r:+.2f}) because equity ({equity * 100:.1f}%) "
                f"is insufficient for pot odds ({required_pot_odds * 100:.1f}%)."
            )

        return DecisionResult(
            action=rec_action,
            ev=rec_ev,
            ev_by_action=ev_by_act,
            equity=equity,
            pot_odds=required_pot_odds,
            pot_size=pot_size,
            bet_to_call=bet_to_call,
            raise_size=raise_total if rec_action.is_aggressive else None,
            fold_equity=fe,
            reasoning=reasoning,
            alternative_action=alt_action,
        )


# =========================================================================
# Top-Level Convenience Functions
# =========================================================================

_DEFAULT_ENGINE = DecisionEngine()


def recommend_decision(
    equity: float,
    pot_size: float,
    bet_to_call: float = 0.0,
    stack_size: float | None = None,
    fold_equity: float = 0.0,
    raise_size: float | None = None,
    future_bets: float = 0.0,
) -> DecisionResult:
    """Recommend the optimal poker action based on mathematical parameters.

    Parameters
    ----------
    equity : float
        Hero's equity in the hand as a fraction in ``[0.0, 1.0]``.
    pot_size : float
        Total pot currently in the middle.
    bet_to_call : float, default=0.0
        Amount Hero must call to stay in the hand.
    stack_size : float or None, default=None
        Hero's remaining chip stack.
    fold_equity : float, default=0.0
        Probability that opponent folds to a bet/raise.
    raise_size : float or None, default=None
        Explicit bet or raise size to test.
    future_bets : float, default=0.0
        Additional future chips won when drawing (implied odds).

    Returns
    -------
    DecisionResult
        Recommended action with expected value and analysis.

    Examples
    --------
    >>> from poker_equity.decision import recommend_decision
    >>> # 35% equity facing $50 into $150 pot (requires 25%)
    >>> res = recommend_decision(equity=0.35, pot_size=150.0, bet_to_call=50.0)
    >>> res.action
    <Action.CALL: 'call'>
    >>> res.is_profitable
    True
    """
    return _DEFAULT_ENGINE.recommend(
        equity=equity,
        pot_size=pot_size,
        bet_to_call=bet_to_call,
        stack_size=stack_size,
        fold_equity=fold_equity,
        raise_size=raise_size,
        future_bets=future_bets,
    )


def recommend_hand_decision(
    hero: Sequence[Card | str] | str,
    villain: Sequence[Card | str] | str,
    board: Sequence[Card | str] | str | None,
    pot_size: float,
    bet_to_call: float = 0.0,
    stack_size: float | None = None,
    fold_equity: float = 0.0,
    raise_size: float | None = None,
    future_bets: float = 0.0,
    num_sims: int = 5000,
    seed: int | None = None,
) -> DecisionResult:
    """Recommend a decision by simulating equity between Hero and Villain hands.

    Parameters
    ----------
    hero : Sequence[Card | str] | str
        Hero's 2 hole cards.
    villain : Sequence[Card | str] | str
        Villain's 2 hole cards.
    board : Sequence[Card | str] | str | None
        Community board cards (0 to 5 cards).
    pot_size : float
        Current pot size.
    bet_to_call : float, default=0.0
        Cost to call.
    stack_size : float or None, default=None
        Hero's stack size.
    fold_equity : float, default=0.0
        Opponent folding probability.
    raise_size : float or None, default=None
        Raise size to evaluate.
    future_bets : float, default=0.0
        Implied odds chips.
    num_sims : int, default=5000
        Number of Monte Carlo simulations to estimate equity.
    seed : int or None, default=None
        RNG seed for deterministic evaluation.

    Returns
    -------
    DecisionResult
        Decision recommendation based on simulated equity.
    """
    sim_result = calculate_equity(
        hero=hero,
        villain=villain,
        board=board,
        num_sims=num_sims,
        seed=seed,
    )
    return recommend_decision(
        equity=sim_result.equity,
        pot_size=pot_size,
        bet_to_call=bet_to_call,
        stack_size=stack_size,
        fold_equity=fold_equity,
        raise_size=raise_size,
        future_bets=future_bets,
    )


def recommend_drawing_decision(
    hero: Sequence[Card | str] | str,
    villain: Sequence[Card | str] | str,
    board: Sequence[Card | str] | str,
    pot_size: float,
    bet_to_call: float,
    stack_size: float | None = None,
    fold_equity: float = 0.0,
    raise_size: float | None = None,
    future_bets: float = 0.0,
    method: str = "exact",
) -> DecisionResult:
    """Recommend a decision based on exact outs calculation from a flop/turn draw.

    Parameters
    ----------
    hero : Sequence[Card | str] | str
        Hero's hole cards.
    villain : Sequence[Card | str] | str
        Villain's hole cards.
    board : Sequence[Card | str] | str
        Board cards (3 for flop, 4 for turn).
    pot_size : float
        Current pot.
    bet_to_call : float
        Cost to call.
    stack_size : float or None, default=None
        Hero's stack.
    fold_equity : float, default=0.0
        Fold equity.
    raise_size : float or None, default=None
        Raise size.
    future_bets : float, default=0.0
        Implied odds.
    method : {'exact', 'rule', 'adjusted_rule'}, default='exact'
        Method used to convert outs count to equity.

    Returns
    -------
    DecisionResult
        Strategic decision based on drawing outs and equity.
    """
    outs_result = calculate_outs(hero=hero, villain=villain, board=board)

    # Determine street
    board_len = len(outs_result.hero_current_result.sub_rank)  # or board cards
    # Better: check unseen cards (45 on flop -> flop; 44 on turn -> turn)
    street = "flop" if outs_result.unseen_cards == 45 else "turn"

    drawing_equity = outs_to_equity(
        outs=outs_result.total_outs,
        street=street,
        method=method,
    )

    return recommend_decision(
        equity=drawing_equity,
        pot_size=pot_size,
        bet_to_call=bet_to_call,
        stack_size=stack_size,
        fold_equity=fold_equity,
        raise_size=raise_size,
        future_bets=future_bets,
    )

