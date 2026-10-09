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
        Number of simulations where player won outright.
    ties : int
        Number of simulations where player split the pot.
    losses : int
        Number of simulations where player lost.
    num_sims : int
        Total number of simulations run.
    equity : float
        Player's equity (expected share of the pot).
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
    wins: int, ties: int, losses: int, num_sims: int, equity: float
) -> EquityResult:
    """Construct an EquityResult from raw counts."""
    return EquityResult(
        wins=wins,
        ties=ties,
        losses=losses,
        num_sims=num_sims,
        equity=equity,
        win_pct=wins / num_sims,
        tie_pct=ties / num_sims,
        loss_pct=losses / num_sims,
    )


# =========================================================================
# Validation helpers
# =========================================================================


def _validate_multi_inputs(
    hands: Sequence[Sequence[Card] | None],
    board: Sequence[Card],
) -> None:
    """Validate simulation inputs for multiple players."""
    if not (2 <= len(hands) <= 9):
        msg = f"Must have between 2 and 9 players, got {len(hands)}"
        raise ValueError(msg)

    for i, hand in enumerate(hands):
        if hand is not None and len(hand) != 2:
            msg = f"Player {i+1} must have exactly 2 hole cards or be None, got {len(hand)}"
            raise ValueError(msg)

    if len(board) > 5:
        msg = f"Board can have at most 5 cards, got {len(board)}"
        raise ValueError(msg)

    all_cards = list(board)
    for hand in hands:
        if hand is not None:
            all_cards.extend(hand)
            
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


def calculate_equities(
    hands: Sequence[Sequence[Card] | None],
    board: Sequence[Card] | None = None,
    *,
    num_sims: int = 10_000,
    seed: int | None = None,
) -> list[EquityResult]:
    """Calculate equities for a multi-way pot via Monte Carlo simulation.

    Parameters
    ----------
    hands:
        List of hole cards for each player. Use ``None`` for random hands.
    board:
        Known community cards (0 to 5).
    num_sims:
        Number of Monte Carlo simulations to run.
    seed:
        Optional PRNG seed for reproducible results.

    Returns
    -------
    list[EquityResult]
        The simulation results for each player in the same order as `hands`.
    """
    if board is None:
        board = []

    _validate_multi_inputs(hands, board)

    if num_sims <= 0:
        msg = f"num_sims must be positive, got {num_sims}"
        raise ValueError(msg)

    known_cards = set(board)
    for hand in hands:
        if hand is not None:
            known_cards.update(hand)
            
    stub = [
        Card(rank, suit)
        for rank in _ALL_RANKS
        for suit in _ALL_SUITS
        if Card(rank, suit) not in known_cards
    ]

    board_list = list(board)
    board_needed = 5 - len(board_list)
    random_hands_count = sum(1 for h in hands if h is None)
    cards_to_deal = board_needed + 2 * random_hands_count
    
    num_players = len(hands)
    
    # Pre-build the fixed parts of each player's 7-card hand
    fixed_hands_data = [
        list(h) + board_list if h is not None else [] 
        for h in hands
    ]
    
    rng = random.Random(seed)
    wins = [0] * num_players
    ties = [0] * num_players
    losses = [0] * num_players
    equity_sums = [0.0] * num_players

    for _ in range(num_sims):
        dealt = rng.sample(stub, cards_to_deal)
        community_dealt = dealt[:board_needed]
        
        scores = []
        random_cards_index = board_needed
        
        for i, (hand, fixed) in enumerate(zip(hands, fixed_hands_data)):
            if hand is not None:
                hand_score = score_7(fixed + community_dealt)
            else:
                hole1 = dealt[random_cards_index]
                hole2 = dealt[random_cards_index + 1]
                random_cards_index += 2
                hand_score = score_7(board_list + community_dealt + [hole1, hole2])
            scores.append(hand_score)
            
        max_score = max(scores)
        
        # Determine winners
        winners = []
        for i in range(num_players):
            if scores[i] == max_score:
                winners.append(i)
            else:
                losses[i] += 1
                
        num_winners = len(winners)
        pot_share = 1.0 / num_winners
        
        for i in winners:
            if num_winners == 1:
                wins[i] += 1
            else:
                ties[i] += 1
            equity_sums[i] += pot_share

    return [
        _build_equity_result(
            wins[i], ties[i], losses[i], num_sims, equity_sums[i] / num_sims
        )
        for i in range(num_players)
    ]


def calculate_equity(
    hero: Sequence[Card],
    villain: Sequence[Card] | None = None,
    board: Sequence[Card] | None = None,
    *,
    num_sims: int = 10_000,
    seed: int | None = None,
) -> EquityResult:
    """Calculate hero's equity against villain via Monte Carlo simulation.

    This is a convenience wrapper around :func:`calculate_equities` for 
    heads-up scenarios.

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
    results = calculate_equities(
        hands=[hero, villain],
        board=board,
        num_sims=num_sims,
        seed=seed,
    )
    return results[0]


# ---------------------------------------------------------------------------
# Internal constants (avoid importing from card at module level for speed)
# ---------------------------------------------------------------------------

from poker_equity.card import Rank, Suit  # noqa: E402

_ALL_RANKS = list(Rank)
_ALL_SUITS = list(Suit)
