"""Outs calculation, drawing odds, and Rule of 2/4 for Texas Hold'em.

This module provides tools for calculating drawing outs, estimating equity
using the Rule of 2 and Rule of 4, calculating exact mathematical drawing
probabilities, and analyzing which specific cards improve a poker hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from poker_equity.card import Card
from poker_equity.constants import FULL_DECK_LIST, cards_to_str, parse_cards
from poker_equity.evaluator import evaluate_hand
from poker_equity.hand_rank import HandRank, HandResult


# =========================================================================
# Rule of 2 and Rule of 4 equity estimators
# =========================================================================


def rule_of_2(outs: float | int, as_percent: bool = False) -> float:
    """Estimate equity on the turn with one card to come using the Rule of 2.

    The Rule of 2 is a standard poker heuristic that estimates the probability
    of hitting a drawing out on the next street by multiplying the number of
    outs by 2.

    Parameters
    ----------
    outs : float or int
        The number of outs (cards that improve the hand to winning).
        Can be fractional if counting tie outs as 0.5.
    as_percent : bool, default=False
        If True, returns the equity as a percentage in ``[0.0, 100.0]``
        (e.g., 18.0 for 9 outs).
        If False, returns the equity as a fraction in ``[0.0, 1.0]``
        (e.g., 0.18 for 9 outs).

    Returns
    -------
    float
        The estimated equity, capped at 1.0 (or 100.0 if ``as_percent=True``).

    Raises
    ------
    ValueError
        If ``outs < 0``.
    TypeError
        If ``outs`` is not a numeric type or is a boolean.

    Examples
    --------
    Flush draw on turn (9 outs):

    >>> from poker_equity.outs import rule_of_2
    >>> rule_of_2(9)
    0.18
    >>> rule_of_2(9, as_percent=True)
    18.0
    """
    if isinstance(outs, bool) or not isinstance(outs, (int, float)):
        raise TypeError(f"outs must be a number, got {type(outs).__name__}")
    if outs < 0:
        raise ValueError(f"outs cannot be negative, got {outs}")

    pct = min(100.0, outs * 2.0)
    return pct if as_percent else pct / 100.0


def rule_of_4(
    outs: float | int,
    as_percent: bool = False,
    adjusted: bool = False,
) -> float:
    """Estimate equity on the flop with two cards to come using the Rule of 4.

    The standard Rule of 4 multiplies the number of outs by 4 to estimate the
    probability of hitting at least one out by the river.

    When ``adjusted=True`` and ``outs > 8``, the standard adjustment
    ``(outs * 4) - (outs - 8)`` is applied to compensate for the linear
    approximation overestimating equity at high out counts.

    Parameters
    ----------
    outs : float or int
        The number of outs.
    as_percent : bool, default=False
        If True, returns the equity as a percentage in ``[0.0, 100.0]``.
        If False, returns the equity as a fraction in ``[0.0, 1.0]``.
    adjusted : bool, default=False
        If True, applies the high-out adjustment when ``outs > 8``.

    Returns
    -------
    float
        The estimated equity, capped at 1.0 (or 100.0 if ``as_percent=True``).

    Raises
    ------
    ValueError
        If ``outs < 0``.
    TypeError
        If ``outs`` is not a numeric type or is a boolean.

    Examples
    --------
    Flush draw on flop (9 outs):

    >>> from poker_equity.outs import rule_of_4
    >>> rule_of_4(9)
    0.36
    >>> rule_of_4(9, adjusted=True)
    0.35
    """
    if isinstance(outs, bool) or not isinstance(outs, (int, float)):
        raise TypeError(f"outs must be a number, got {type(outs).__name__}")
    if outs < 0:
        raise ValueError(f"outs cannot be negative, got {outs}")

    if adjusted and outs > 8:
        pct = (outs * 4.0) - (outs - 8.0)
    else:
        pct = outs * 4.0

    pct = max(0.0, min(100.0, pct))
    return pct if as_percent else pct / 100.0


def exact_turn_equity(outs: float | int, unseen_cards: int = 46) -> float:
    """Calculate the exact mathematical probability of hitting an out on the turn.

    The probability is:

    .. math::
        P(\\text{hit}) = \\frac{\\text{outs}}{\\text{unseen\\_cards}}

    Parameters
    ----------
    outs : float or int
        The number of outs.
    unseen_cards : int, default=46
        The number of unknown cards remaining in the deck.
        Typically 46 on the flop (52 - 2 hero - 2 villain - 3 board = 45, or
        47 if villain hand is unknown).

    Returns
    -------
    float
        Probability as a fraction in ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If ``outs < 0`` or ``unseen_cards <= 0``.
    TypeError
        If arguments are non-numeric.
    """
    if isinstance(outs, bool) or not isinstance(outs, (int, float)):
        raise TypeError(f"outs must be a number, got {type(outs).__name__}")
    if isinstance(unseen_cards, bool) or not isinstance(unseen_cards, int):
        raise TypeError(f"unseen_cards must be an integer, got {type(unseen_cards).__name__}")

    if outs < 0:
        raise ValueError(f"outs cannot be negative, got {outs}")
    if unseen_cards <= 0:
        raise ValueError(f"unseen_cards must be positive, got {unseen_cards}")

    return min(1.0, outs / unseen_cards)


def exact_river_equity(outs: float | int, unseen_cards: int = 46) -> float:
    """Calculate the exact probability of hitting an out by the river (2 cards to come).

    The probability of hitting at least one out across the turn and river is:

    .. math::
        P(\\text{hit by river}) = 1 - \\left( \\frac{\\text{unseen} - \\text{outs}}{\\text{unseen}} \\times \\frac{\\text{unseen} - 1 - \\text{outs}}{\\text{unseen} - 1} \\right)

    Parameters
    ----------
    outs : float or int
        The number of outs.
    unseen_cards : int, default=46
        The number of unknown cards on the flop.

    Returns
    -------
    float
        Probability as a fraction in ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If ``outs < 0`` or ``unseen_cards < 2``.
    TypeError
        If arguments are non-numeric.
    """
    if isinstance(outs, bool) or not isinstance(outs, (int, float)):
        raise TypeError(f"outs must be a number, got {type(outs).__name__}")
    if isinstance(unseen_cards, bool) or not isinstance(unseen_cards, int):
        raise TypeError(f"unseen_cards must be an integer, got {type(unseen_cards).__name__}")

    if outs < 0:
        raise ValueError(f"outs cannot be negative, got {outs}")
    if unseen_cards < 2:
        raise ValueError(f"unseen_cards must be at least 2, got {unseen_cards}")

    if outs >= unseen_cards:
        return 1.0

    miss_turn = (unseen_cards - outs) / unseen_cards
    miss_river = (unseen_cards - 1 - outs) / (unseen_cards - 1)
    return max(0.0, min(1.0, 1.0 - (miss_turn * miss_river)))


def outs_to_equity(
    outs: float | int,
    street: str = "flop",
    method: str = "rule",
) -> float:
    """Convert an out count to estimated or exact equity.

    Parameters
    ----------
    outs : float or int
        The number of outs.
    street : {'flop', 'turn'}, default='flop'
        The current street.  'flop' assumes 2 cards to come; 'turn' assumes
        1 card to come.
    method : {'rule', 'exact', 'adjusted_rule'}, default='rule'
        Calculation method:
        - ``'rule'``: Rule of 4 on flop, Rule of 2 on turn.
        - ``'adjusted_rule'``: Adjusted Rule of 4 on flop, Rule of 2 on turn.
        - ``'exact'``: Exact hypergeometric / probability calculation.

    Returns
    -------
    float
        Equity as a fraction in ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If ``street`` or ``method`` is unrecognized.
    """
    street_clean = street.lower().strip()
    method_clean = method.lower().strip()

    if street_clean == "flop":
        if method_clean == "rule":
            return rule_of_4(outs, adjusted=False)
        elif method_clean == "adjusted_rule":
            return rule_of_4(outs, adjusted=True)
        elif method_clean == "exact":
            return exact_river_equity(outs)
        else:
            raise ValueError(f"Unknown method {method!r}, expected 'rule', 'adjusted_rule', or 'exact'")
    elif street_clean == "turn":
        if method_clean in ("rule", "adjusted_rule"):
            return rule_of_2(outs)
        elif method_clean == "exact":
            return exact_turn_equity(outs, unseen_cards=46)
        else:
            raise ValueError(f"Unknown method {method!r}, expected 'rule', 'adjusted_rule', or 'exact'")
    else:
        raise ValueError(f"Unknown street {street!r}, expected 'flop' or 'turn'")


# =========================================================================
# Outs Result Type
# =========================================================================


@dataclass(frozen=True, slots=True)
class OutsResult:
    """Detailed breakdown and metrics for poker hand outs.

    Attributes
    ----------
    win_outs : tuple[Card, ...]
        Cards that make Hero's hand strictly beat Villain's hand.
    tie_outs : tuple[Card, ...]
        Cards that result in a tie (split pot) between Hero and Villain.
    unseen_cards : int
        Total number of remaining unknown cards in the deck.
    outs_by_rank : dict[HandRank, tuple[Card, ...]]
        Winning cards grouped by the resulting hand rank Hero achieves.
    hero_current_result : HandResult
        Hero's current evaluated hand on the board.
    villain_current_result : HandResult
        Villain's current evaluated hand on the board.
    is_hero_ahead : bool
        Whether Hero is currently winning on the board before the next card.
    """

    win_outs: tuple[Card, ...]
    tie_outs: tuple[Card, ...]
    unseen_cards: int
    outs_by_rank: dict[HandRank, tuple[Card, ...]]
    hero_current_result: HandResult
    villain_current_result: HandResult
    is_hero_ahead: bool

    @property
    def total_outs(self) -> float:
        """Total effective outs, counting tie outs as 0.5."""
        return len(self.win_outs) + 0.5 * len(self.tie_outs)

    @property
    def clean_outs_count(self) -> int:
        """Number of clean winning outs."""
        return len(self.win_outs)

    @property
    def tie_outs_count(self) -> int:
        """Number of tie/split outs."""
        return len(self.tie_outs)

    @property
    def rule_of_2(self) -> float:
        """Rule of 2 equity estimate for 1 card to come."""
        return rule_of_2(self.total_outs)

    @property
    def rule_of_4(self) -> float:
        """Rule of 4 equity estimate for 2 cards to come."""
        return rule_of_4(self.total_outs)

    @property
    def exact_turn_equity(self) -> float:
        """Exact probability of hitting an out on the immediate next card."""
        return exact_turn_equity(self.total_outs, self.unseen_cards)

    @property
    def exact_river_equity(self) -> float:
        """Exact probability of hitting an out by the river."""
        return exact_river_equity(self.total_outs, self.unseen_cards)

    def __len__(self) -> int:
        return len(self.win_outs)

    def __contains__(self, card: Card | str) -> bool:
        if isinstance(card, str):
            c = Card.from_str(card)
        elif isinstance(card, Card):
            c = card
        else:
            return False
        return c in self.win_outs

    def summary(self) -> str:
        """Return a formatted text summary of outs and equity estimates."""
        lines = [
            f"Current Hand: {self.hero_current_result.description}",
            f"Villain Hand: {self.villain_current_result.description}",
            f"Status: {'Ahead' if self.is_hero_ahead else 'Trailing/Tied'}",
            f"Total Outs: {self.total_outs:.1f} ({self.clean_outs_count} win, {self.tie_outs_count} tie)",
            f"Unseen Cards: {self.unseen_cards}",
            f"Rule of 2: {self.rule_of_2 * 100:.1f}%",
            f"Rule of 4: {self.rule_of_4 * 100:.1f}%",
            f"Exact (1 card): {self.exact_turn_equity * 100:.2f}%",
            f"Exact (2 cards): {self.exact_river_equity * 100:.2f}%",
        ]
        if self.outs_by_rank:
            lines.append("Outs by hand rank:")
            for rank in sorted(self.outs_by_rank.keys(), reverse=True):
                cards = self.outs_by_rank[rank]
                card_strs = ", ".join(str(c) for c in cards)
                lines.append(f"  {rank.label} ({len(cards)}): {card_strs}")
        return "\n".join(lines)


# =========================================================================
# Input normalization helpers
# =========================================================================


def _coerce_cards(
    cards_input: Sequence[Card | str] | str,
    expected_len: int | tuple[int, ...] | None = None,
    name: str = "cards",
) -> list[Card]:
    """Parse and normalize card sequences or strings."""
    if isinstance(cards_input, str):
        parsed = parse_cards(cards_input)
    elif isinstance(cards_input, (list, tuple)):
        parsed = [
            Card.from_str(c) if isinstance(c, str) else c
            for c in cards_input
        ]
        for c in parsed:
            if not isinstance(c, Card):
                raise TypeError(f"All elements of {name} must be Card objects or strings, got {type(c).__name__}")
    else:
        raise TypeError(f"{name} must be a card string or sequence of Cards, got {type(cards_input).__name__}")

    if expected_len is not None:
        if isinstance(expected_len, int):
            if len(parsed) != expected_len:
                raise ValueError(f"{name} must contain exactly {expected_len} cards, got {len(parsed)}")
        elif len(parsed) not in expected_len:
            raise ValueError(
                f"{name} must contain one of {expected_len} cards, got {len(parsed)}"
            )

    return parsed


# =========================================================================
# Outs Calculation Engine
# =========================================================================


def calculate_outs(
    hero: Sequence[Card | str] | str,
    villain: Sequence[Card | str] | str,
    board: Sequence[Card | str] | str,
    dead: Sequence[Card | str] | str | None = None,
) -> OutsResult:
    """Calculate the outs for Hero against Villain on a given board.

    Evaluates every card remaining in the deck.  A card is an out if dealing
    it on the next street makes Hero's hand strictly better than Villain's hand
    (or equal for tie outs).

    Parameters
    ----------
    hero : Sequence[Card | str] | str
        Hero's 2 hole cards (e.g. ``"Ah Kh"`` or ``[Card.from_str("Ah"), ...]``).
    villain : Sequence[Card | str] | str
        Villain's 2 hole cards.
    board : Sequence[Card | str] | str
        The community cards (must contain 3 cards for flop or 4 cards for turn).
    dead : Sequence[Card | str] | str | None, default=None
        Known dead or folded cards to exclude from the deck.

    Returns
    -------
    OutsResult
        Detailed breakdown of winning and tie outs, along with equity estimates.

    Raises
    ------
    ValueError
        If hands do not contain 2 cards, board does not contain 3 or 4 cards,
        or duplicate cards are detected.
    TypeError
        If input types are invalid.

    Examples
    --------
    Ah Kh vs Qc Qd on 2h 7h 8c (flush draw + 2 overcards):

    >>> from poker_equity.outs import calculate_outs
    >>> res = calculate_outs("Ah Kh", "Qc Qd", "2h 7h 8c")
    >>> res.total_outs
    15.0
    >>> res.rule_of_4
    0.6
    """
    hero_cards = _coerce_cards(hero, expected_len=2, name="hero")
    villain_cards = _coerce_cards(villain, expected_len=2, name="villain")
    board_cards = _coerce_cards(board, expected_len=(3, 4), name="board")
    dead_input_cards = _coerce_cards(dead, name="dead") if dead else []

    all_cards = hero_cards + villain_cards + board_cards
    if len(set(all_cards)) != len(all_cards):
        raise ValueError(f"Duplicate cards detected in inputs: {cards_to_str(all_cards)}")
    if dead_input_cards:
        overlap = set(all_cards) & set(dead_input_cards)
        if overlap:
            raise ValueError(f"Duplicate cards detected in inputs: {cards_to_str(list(overlap))}")

    # Evaluate current situation
    hero_curr = evaluate_hand(hero_cards + board_cards)
    villain_curr = evaluate_hand(villain_cards + board_cards)
    is_ahead = hero_curr > villain_curr

    dead_cards = set(all_cards) | set(dead_input_cards)
    unseen_cards = [c for c in FULL_DECK_LIST if c not in dead_cards]

    win_outs: list[Card] = []
    tie_outs: list[Card] = []
    outs_by_rank: dict[HandRank, list[Card]] = {}

    for c in unseen_cards:
        next_board = board_cards + [c]
        hero_next = evaluate_hand(hero_cards + next_board)
        villain_next = evaluate_hand(villain_cards + next_board)

        if hero_next > villain_next:
            win_outs.append(c)
            outs_by_rank.setdefault(hero_next.rank, []).append(c)
        elif hero_next == villain_next:
            tie_outs.append(c)

    # Sort cards canonically (rank descending, then suit)
    win_outs.sort(key=lambda card: (card.rank.value, card.suit.value), reverse=True)
    tie_outs.sort(key=lambda card: (card.rank.value, card.suit.value), reverse=True)

    sorted_by_rank = {
        rank: tuple(sorted(cards, key=lambda card: (card.rank.value, card.suit.value), reverse=True))
        for rank, cards in outs_by_rank.items()
    }

    return OutsResult(
        win_outs=tuple(win_outs),
        tie_outs=tuple(tie_outs),
        unseen_cards=len(unseen_cards),
        outs_by_rank=sorted_by_rank,
        hero_current_result=hero_curr,
        villain_current_result=villain_curr,
        is_hero_ahead=is_ahead,
    )


def calculate_improvement_outs(
    hand: Sequence[Card | str] | str,
    board: Sequence[Card | str] | str,
    target_rank: HandRank | None = None,
    dead: Sequence[Card | str] | str | None = None,
) -> dict[HandRank, tuple[Card, ...]]:
    """Find all remaining cards that improve a hand or reach a target hand rank.

    Evaluates every card remaining in the deck.  If ``target_rank`` is None,
    identifies cards that strictly improve the hand's category (e.g. from High
    Card to One Pair, Straight, Flush, etc.).  If ``target_rank`` is specified,
    identifies cards where the resulting hand category equals or exceeds
    ``target_rank``.

    Parameters
    ----------
    hand : Sequence[Card | str] | str
        The 2 hole cards.
    board : Sequence[Card | str] | str
        The community cards (3 or 4 cards).
    target_rank : HandRank or None, default=None
        Minimum target hand rank to achieve.
    dead : Sequence[Card | str] | str | None, default=None
        Known dead or folded cards to exclude from the deck.

    Returns
    -------
    dict[HandRank, tuple[Card, ...]]
        A dictionary mapping achieved HandRank to the tuple of cards that hit it.

    Raises
    ------
    ValueError
        If inputs contain duplicates or invalid card counts.
    """
    hand_cards = _coerce_cards(hand, expected_len=2, name="hand")
    board_cards = _coerce_cards(board, expected_len=(3, 4), name="board")
    dead_input_cards = _coerce_cards(dead, name="dead") if dead else []

    all_cards = hand_cards + board_cards
    if len(set(all_cards)) != len(all_cards):
        raise ValueError(f"Duplicate cards detected: {cards_to_str(all_cards)}")
    if dead_input_cards:
        overlap = set(all_cards) & set(dead_input_cards)
        if overlap:
            raise ValueError(f"Duplicate cards detected: {cards_to_str(list(overlap))}")

    curr_result = evaluate_hand(hand_cards + board_cards)

    dead_cards = set(all_cards) | set(dead_input_cards)
    unseen_cards = [c for c in FULL_DECK_LIST if c not in dead_cards]

    improving_cards: dict[HandRank, list[Card]] = {}

    for c in unseen_cards:
        next_board = board_cards + [c]
        res = evaluate_hand(hand_cards + next_board)

        if target_rank is not None:
            if res.rank >= target_rank:
                improving_cards.setdefault(res.rank, []).append(c)
        else:
            if res.rank > curr_result.rank:
                improving_cards.setdefault(res.rank, []).append(c)

    return {
        rank: tuple(sorted(cards, key=lambda card: (card.rank.value, card.suit.value), reverse=True))
        for rank, cards in improving_cards.items()
    }

