"""Monte Carlo equity simulation for Texas Hold'em.

This module provides the core simulation engine that calculates win / tie /
loss probabilities by running thousands of random board runouts.

The primary entry point is :func:`calculate_equity`, which accepts hero
and villain hole cards, an optional partial board, and a simulation count.

Example
-------
>>> from poker_equity.card import Card
>>> from poker_equity.simulation import calculate_equity
>>> hero = [Card.from_str("Ah"), Card.from_str("Kh")]
>>> villain = [Card.from_str("Qs"), Card.from_str("Qd")]
>>> result = calculate_equity(hero, villain, num_sims=10_000, seed=42)
>>> result.equity  # hero's equity as a float in [0, 1]
0.4...
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from poker_equity.card import Card
from poker_equity.evaluator import score_7


# =========================================================================
# Result type
# =========================================================================


@dataclass(frozen=True, slots=True)
class EquityResult:
    """The result of a Monte Carlo equity simulation.

    All percentages are expressed as floats in ``[0.0, 1.0]``.

    Attributes
    ----------
    wins : int
        Number of simulations where hero won.
    ties : int
        Number of simulations where hero tied with villain.
    losses : int
        Number of simulations where hero lost.
    num_sims : int
        Total number of simulations run.
    equity : float
        Hero's equity: ``wins / num_sims + ties / (2 * num_sims)``.
        Ties count as half a win.
    win_pct : float
        Win percentage: ``wins / num_sims``.
    tie_pct : float
        Tie percentage: ``ties / num_sims``.
    loss_pct : float
        Loss percentage: ``losses / num_sims``.
    """

    wins: int
    ties: int
    losses: int
    num_sims: int
    equity: float
    win_pct: float
    tie_pct: float
    loss_pct: float

    def __str__(self) -> str:
        return (
            f"Equity: {self.equity:.1%} | "
            f"Win: {self.win_pct:.1%} | "
            f"Tie: {self.tie_pct:.1%} | "
            f"Lose: {self.loss_pct:.1%} | "
            f"({self.num_sims:,} sims)"
        )

    def __repr__(self) -> str:
        return (
            f"EquityResult(equity={self.equity:.4f}, "
            f"win={self.win_pct:.4f}, tie={self.tie_pct:.4f}, "
            f"loss={self.loss_pct:.4f}, sims={self.num_sims})"
        )


def _build_equity_result(
    wins: int, ties: int, losses: int, num_sims: int
) -> EquityResult:
    """Construct an EquityResult from raw counts."""
    return EquityResult(
        wins=wins,
        ties=ties,
        losses=losses,
        num_sims=num_sims,
        equity=(wins + ties / 2) / num_sims,
        win_pct=wins / num_sims,
        tie_pct=ties / num_sims,
        loss_pct=losses / num_sims,
    )


# =========================================================================
# Validation helpers
# =========================================================================


def _validate_inputs(
    hero: Sequence[Card],
    villain: Sequence[Card] | None,
    board: Sequence[Card],
) -> None:
    """Validate simulation inputs.

    Raises
    ------
    ValueError
        If any input is invalid (wrong card count, duplicates, etc.).
    """
    if len(hero) != 2:
        msg = f"Hero must have exactly 2 hole cards, got {len(hero)}"
        raise ValueError(msg)

    if villain is not None and len(villain) != 2:
        msg = f"Villain must have exactly 2 hole cards, got {len(villain)}"
        raise ValueError(msg)

    if len(board) > 5:
        msg = f"Board can have at most 5 cards, got {len(board)}"
        raise ValueError(msg)

    # Check for duplicate cards
    all_cards = list(hero) + list(board)
    if villain is not None:
        all_cards += list(villain)
        
    if len(set(all_cards)) != len(all_cards):
        seen: set[Card] = set()
        for card in all_cards:
            if card in seen:
                msg = f"Duplicate card: {card}"
                raise ValueError(msg)
            seen.add(card)


# =========================================================================
# Core simulation
# =========================================================================


def calculate_equity(
    hero: Sequence[Card],
    villain: Sequence[Card] | None = None,
    board: Sequence[Card] | None = None,
    *,
    num_sims: int = 10_000,
    seed: int | None = None,
) -> EquityResult:
    """Calculate hero's equity against villain via Monte Carlo simulation.

    For each simulation, the remaining community cards are dealt randomly
    from the stub deck (all cards minus hero, villain, and known board
    cards).  Both players' hands are scored using the fast lookup-table
    evaluator, and the result is tallied as a win, tie, or loss for hero.

    Parameters
    ----------
    hero:
        Hero's two hole cards.
    villain:
        Villain's two hole cards, or ``None`` to evaluate against a
        random hand (any two unknown cards).
    board:
        Known community cards (0 to 5).  ``None`` or empty list means
        preflop (no board cards dealt yet).
    num_sims:
        Number of Monte Carlo simulations to run.  Higher values give
        more accurate results but take longer.  Default is 10,000.
    seed:
        Optional PRNG seed for reproducible results.

    Returns
    -------
    EquityResult
        The simulation results with win/tie/loss counts and equity.

    Raises
    ------
    ValueError
        If inputs are invalid (wrong card count, duplicates, etc.).

    Examples
    --------
    >>> from poker_equity.card import Card
    >>> hero = [Card.from_str("Ah"), Card.from_str("Kh")]
    >>> villain = [Card.from_str("7s"), Card.from_str("2d")]
    >>> result = calculate_equity(hero, villain, seed=42)
    >>> result.equity > 0.5  # AKs is a big favorite over 72o
    True
    """
    if board is None:
        board = []

    _validate_inputs(hero, villain, board)

    if num_sims <= 0:
        msg = f"num_sims must be positive, got {num_sims}"
        raise ValueError(msg)

    # Build the stub deck: all 52 cards minus known cards
    known_cards = set(hero) | set(board)
    if villain is not None:
        known_cards |= set(villain)
        
    stub = [
        Card(rank, suit)
        for rank in _ALL_RANKS
        for suit in _ALL_SUITS
        if Card(rank, suit) not in known_cards
    ]

    board_list = list(board)
    board_needed = 5 - len(board_list)
    hero_fixed = list(hero) + board_list
    
    rng = random.Random(seed)
    wins = 0
    ties = 0
    losses = 0

    if villain is not None:
        # Fixed villain hand
        villain_fixed = list(villain) + board_list
        for _ in range(num_sims):
            dealt = rng.sample(stub, board_needed)
            hero_hand = hero_fixed + dealt
            villain_hand = villain_fixed + dealt

            hero_score = score_7(hero_hand)
            villain_score = score_7(villain_hand)

            if hero_score > villain_score:
                wins += 1
            elif hero_score == villain_score:
                ties += 1
            else:
                losses += 1
    else:
        # Random villain hand (needs 2 extra cards)
        cards_to_deal = board_needed + 2
        for _ in range(num_sims):
            dealt = rng.sample(stub, cards_to_deal)
            
            # First board_needed cards are the board, last 2 are villain hole cards
            community_dealt = dealt[:board_needed]
            
            hero_hand = hero_fixed + community_dealt
            villain_hand = board_list + dealt  # board_list + community_dealt + villain_hole

            hero_score = score_7(hero_hand)
            villain_score = score_7(villain_hand)

            if hero_score > villain_score:
                wins += 1
            elif hero_score == villain_score:
                ties += 1
            else:
                losses += 1

    return _build_equity_result(wins, ties, losses, num_sims)


# ---------------------------------------------------------------------------
# Internal constants (avoid importing from card at module level for speed)
# ---------------------------------------------------------------------------

from poker_equity.card import Rank, Suit  # noqa: E402

_ALL_RANKS = list(Rank)
_ALL_SUITS = list(Suit)
