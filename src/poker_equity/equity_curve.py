"""Street-by-street equity transitions and equity curves for Texas Hold'em.

This module provides tools to track, visualize, and analyze how equity and
hand strengths evolve from Preflop -> Flop -> Turn -> River.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from poker_equity.card import Card
from poker_equity.constants import FULL_DECK_LIST, cards_to_str, parse_cards
from poker_equity.evaluator import evaluate_hand, score_7
from poker_equity.hand_rank import HandResult
from poker_equity.simulation import calculate_equities


# =========================================================================
# Street Enumeration
# =========================================================================


class Street(str, Enum):
    """The betting rounds / streets of Texas Hold'em."""

    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"

    def __str__(self) -> str:
        return self.value.upper()

    @property
    def cards_count(self) -> int:
        """Number of community board cards dealt on this street."""
        if self == Street.PREFLOP:
            return 0
        elif self == Street.FLOP:
            return 3
        elif self == Street.TURN:
            return 4
        else:  # RIVER
            return 5

    @property
    def next_street(self) -> Street | None:
        """The subsequent street, or None if River."""
        if self == Street.PREFLOP:
            return Street.FLOP
        elif self == Street.FLOP:
            return Street.TURN
        elif self == Street.TURN:
            return Street.RIVER
        return None

    @property
    def prev_street(self) -> Street | None:
        """The previous street, or None if Preflop."""
        if self == Street.RIVER:
            return Street.TURN
        elif self == Street.TURN:
            return Street.FLOP
        elif self == Street.FLOP:
            return Street.PREFLOP
        return None

    @classmethod
    def from_board_length(cls, length: int) -> Street:
        """Determine Street from number of community board cards.

        Parameters
        ----------
        length : int
            Number of community board cards (0, 3, 4, or 5).

        Returns
        -------
        Street
            The corresponding street.

        Raises
        ------
        ValueError
            If length is not 0, 3, 4, or 5.
        """
        if length == 0:
            return cls.PREFLOP
        elif length == 3:
            return cls.FLOP
        elif length == 4:
            return cls.TURN
        elif length == 5:
            return cls.RIVER
        raise ValueError(
            f"Invalid board length: {length}. Must be 0 (preflop), 3 (flop), "
            f"4 (turn), or 5 (river)."
        )


# =========================================================================
# StreetPoint and Result Containers
# =========================================================================


@dataclass(frozen=True, slots=True)
class StreetPoint:
    """The state of equity and hand evaluations at a specific street.

    Attributes
    ----------
    street : Street
        The street (PREFLOP, FLOP, TURN, RIVER).
    board : tuple[Card, ...]
        The community cards dealt up to this street.
    equities : tuple[float, ...]
        Equity of each player on this street (sums to 1.0).
    hand_results : tuple[HandResult | None, ...]
        Evaluated hand for each player at this street (None for Preflop).
    delta_equities : tuple[float, ...]
        Change in equity from the previous street (0.0 for Preflop).
    leader_idx : int
        Index of the player holding the highest equity on this street.
    leader_equity : float
        Equity of the leader on this street.
    """

    street: Street
    board: tuple[Card, ...]
    equities: tuple[float, ...]
    hand_results: tuple[HandResult | None, ...]
    delta_equities: tuple[float, ...]
    leader_idx: int
    leader_equity: float

    @property
    def hero_equity(self) -> float:
        """Hero's (player 0) equity at this street."""
        return self.equities[0]

    @property
    def villain_equity(self) -> float:
        """Villain's (player 1) equity at this street in a heads-up pot."""
        return self.equities[1] if len(self.equities) > 1 else 0.0

    @property
    def hero_delta(self) -> float:
        """Hero's equity change from the previous street."""
        return self.delta_equities[0]

    @property
    def villain_delta(self) -> float:
        """Villain's equity change from the previous street in heads-up."""
        return self.delta_equities[1] if len(self.delta_equities) > 1 else 0.0

    def hand_description(self, player_idx: int) -> str:
        """Return a human-readable description of player's hand at this street.

        Parameters
        ----------
        player_idx : int
            Index of the player.

        Returns
        -------
        str
            Hand description (e.g. "Pair of Aces") or "Hole cards" preflop.
        """
        hr = self.hand_results[player_idx]
        if hr is None:
            return "Hole cards"
        return hr.description


@dataclass(frozen=True, slots=True)
class EquityCurve:
    """The street-by-street equity trajectory for a poker hand.

    Attributes
    ----------
    hands : tuple[tuple[Card, Card], ...]
        The players' hole cards.
    points : tuple[StreetPoint, ...]
        Street-by-street evaluation points.
    board : tuple[Card, ...]
        The full community board cards examined.
    """

    hands: tuple[tuple[Card, Card], ...]
    points: tuple[StreetPoint, ...]
    board: tuple[Card, ...]

    @property
    def streets(self) -> tuple[Street, ...]:
        """Sequence of evaluated streets."""
        return tuple(p.street for p in self.points)

    @property
    def hero_equities(self) -> tuple[float, ...]:
        """Hero's equity across all evaluated streets."""
        return tuple(p.hero_equity for p in self.points)

    @property
    def villain_equities(self) -> tuple[float, ...]:
        """Villain's equity across all evaluated streets in heads-up."""
        return tuple(p.villain_equity for p in self.points)

    def player_equities(self, player_idx: int) -> tuple[float, ...]:
        """Equity of a specific player across all evaluated streets.

        Parameters
        ----------
        player_idx : int
            Index of the player.

        Returns
        -------
        tuple[float, ...]
            Sequence of equities across streets.
        """
        return tuple(p.equities[player_idx] for p in self.points)

    @property
    def num_lead_changes(self) -> int:
        """Count of street transitions where the equity leader changed."""
        changes = 0
        for i in range(1, len(self.points)):
            if self.points[i].leader_idx != self.points[i - 1].leader_idx:
                changes += 1
        return changes

    @property
    def max_swing(self) -> float:
        """Maximum single-street equity swing (absolute delta) for Hero."""
        deltas = [abs(p.hero_delta) for p in self.points[1:]]
        return max(deltas) if deltas else 0.0

    @property
    def turning_point(self) -> StreetPoint | None:
        """The street with the largest absolute equity change for Hero."""
        if len(self.points) <= 1:
            return None
        return max(self.points[1:], key=lambda p: abs(p.hero_delta))

    @property
    def is_bad_beat(self) -> bool:
        """Whether a significant favorite (>= 80% on flop or turn) lost on river."""
        if len(self.points) < 4:
            return False  # Needs river evaluation
        river_pt = self.points[-1]
        for pt in self.points[1:3]:  # flop or turn
            if pt.hero_equity >= 0.80 and river_pt.hero_equity < 0.50:
                return True
            if pt.villain_equity >= 0.80 and river_pt.villain_equity < 0.50:
                return True
        return False

    @property
    def final_winner_idx(self) -> int | None:
        """Index of the player with highest equity on final street, or None if tied."""
        final_pt = self.points[-1]
        max_eq = final_pt.leader_equity
        winners = [i for i, eq in enumerate(final_pt.equities) if abs(eq - max_eq) < 1e-6]
        return winners[0] if len(winners) == 1 else None

    def summary(self) -> str:
        """Format an ASCII table report of the equity trajectory across streets.

        Returns
        -------
        str
            Detailed multiline summary report.
        """
        lines = [
            "Equity Curve Analysis",
            "=====================",
        ]

        num_players = len(self.hands)
        is_heads_up = num_players == 2

        if is_heads_up:
            header = (
                f"{'Street':<8} {'Board':<14} {'Hero (P1)':<10} {'P1 Hand':<20} "
                f"{'Villain (P2)':<13} {'P2 Hand':<20} {'Hero Δ':<8} {'Leader'}"
            )
            sep = "-" * len(header)
            lines.extend([header, sep])

            for pt in self.points:
                b_str = cards_to_str(pt.board) if pt.board else "-"
                h_eq_str = f"{pt.hero_equity * 100:.1f}%"
                v_eq_str = f"{pt.villain_equity * 100:.1f}%"
                h_desc = pt.hand_description(0)
                v_desc = pt.hand_description(1)
                delta_str = f"{pt.hero_delta * 100:+.1f}%" if pt.street != Street.PREFLOP else "-"
                leader_str = "Hero" if pt.leader_idx == 0 else "Villain"

                lines.append(
                    f"{str(pt.street):<8} {b_str:<14} {h_eq_str:<10} {h_desc:<20} "
                    f"{v_eq_str:<13} {v_desc:<20} {delta_str:<8} {leader_str}"
                )
            lines.append(sep)
        else:
            header = f"{'Street':<8} {'Board':<16} {'Leader':<10} {'Leader Eq':<12} {'Equities'}"
            sep = "-" * 80
            lines.extend([header, sep])

            for pt in self.points:
                b_str = cards_to_str(pt.board) if pt.board else "-"
                lead_str = f"Player {pt.leader_idx + 1}"
                lead_eq_str = f"{pt.leader_equity * 100:.1f}%"
                eq_strs = [f"P{p+1}: {e * 100:.1f}%" for p, e in enumerate(pt.equities)]
                lines.append(
                    f"{str(pt.street):<8} {b_str:<16} {lead_str:<10} {lead_eq_str:<12} {', '.join(eq_strs)}"
                )
            lines.append(sep)

        tp = self.turning_point
        tp_str = f"{tp.street} ({tp.hero_delta * 100:+.1f}%)" if tp else "None"
        lines.append(
            f"Lead Changes: {self.num_lead_changes} | "
            f"Max Swing: {self.max_swing * 100:.1f}% | "
            f"Turning Point: {tp_str} | "
            f"Bad Beat: {self.is_bad_beat}"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize equity curve data to a dictionary suitable for JSON.

        Returns
        -------
        dict
            Structured representation containing hands, board, and street points.
        """
        return {
            "hands": [cards_to_str(h) for h in self.hands],
            "board": cards_to_str(self.board),
            "num_lead_changes": self.num_lead_changes,
            "max_swing": self.max_swing,
            "turning_point": self.turning_point.street.value if self.turning_point else None,
            "is_bad_beat": self.is_bad_beat,
            "final_winner_idx": self.final_winner_idx,
            "points": [
                {
                    "street": pt.street.value,
                    "board": cards_to_str(pt.board),
                    "equities": list(pt.equities),
                    "delta_equities": list(pt.delta_equities),
                    "leader_idx": pt.leader_idx,
                    "leader_equity": pt.leader_equity,
                    "hand_descriptions": [pt.hand_description(i) for i in range(len(self.hands))],
                }
                for pt in self.points
            ],
        }

    def plot_data(self) -> dict:
        """Format series data suitable for chart plotting libraries (e.g. Chart.js).

        Returns
        -------
        dict
            Dictionary containing labels and dataset series per player.
        """
        labels = [str(pt.street) for pt in self.points]
        datasets = []
        for p_idx, hand in enumerate(self.hands):
            label = "Hero (P1)" if p_idx == 0 else f"Player {p_idx + 1}"
            data = [round(pt.equities[p_idx] * 100.0, 2) for pt in self.points]
            datasets.append({"label": f"{label} [{cards_to_str(hand)}]", "data": data})

        return {
            "labels": labels,
            "datasets": datasets,
        }


@dataclass(frozen=True, slots=True)
class NextStreetDistribution:
    """Distribution of equities across all possible next-street cards.

    Attributes
    ----------
    current_street : Street
        Current street (FLOP or TURN).
    next_street : Street
        Target street (TURN or RIVER).
    unseen_cards_count : int
        Total number of unseen cards evaluated.
    equities_by_card : dict[Card, tuple[float, ...]]
        Map of Card -> tuple of player equities if that card is dealt.
    hero_favorable_cards : tuple[Card, ...]
        Cards that increase Hero's equity.
    hero_unfavorable_cards : tuple[Card, ...]
        Cards that decrease Hero's equity.
    hero_neutral_cards : tuple[Card, ...]
        Cards where Hero's equity remains within +/- 1% of current street.
    min_hero_equity : float
        Minimum Hero equity across all possible next cards.
    max_hero_equity : float
        Maximum Hero equity across all possible next cards.
    mean_hero_equity : float
        Average Hero equity across all possible next cards.
    """

    current_street: Street
    next_street: Street
    unseen_cards_count: int
    equities_by_card: dict[Card, tuple[float, ...]]
    hero_favorable_cards: tuple[Card, ...]
    hero_unfavorable_cards: tuple[Card, ...]
    hero_neutral_cards: tuple[Card, ...]
    min_hero_equity: float
    max_hero_equity: float
    mean_hero_equity: float

    @property
    def favorable_count(self) -> int:
        """Count of cards favorable to Hero."""
        return len(self.hero_favorable_cards)

    @property
    def unfavorable_count(self) -> int:
        """Count of cards unfavorable to Hero."""
        return len(self.hero_unfavorable_cards)

    @property
    def neutral_count(self) -> int:
        """Count of cards neutral to Hero."""
        return len(self.hero_neutral_cards)

    def summary(self) -> str:
        """Format a summary of next-street equity possibilities."""
        lines = [
            f"Next Street Distribution ({self.current_street} -> {self.next_street})",
            "=" * 50,
            f"Unseen cards evaluated : {self.unseen_cards_count}",
            f"Mean Hero Equity       : {self.mean_hero_equity * 100:.1f}%",
            f"Hero Equity Range      : {self.min_hero_equity * 100:.1f}% to {self.max_hero_equity * 100:.1f}%",
            f"Favorable Cards (gain) : {self.favorable_count} ({cards_to_str(self.hero_favorable_cards[:10])}{'...' if self.favorable_count > 10 else ''})",
            f"Unfavorable Cards      : {self.unfavorable_count} ({cards_to_str(self.hero_unfavorable_cards[:10])}{'...' if self.unfavorable_count > 10 else ''})",
            f"Neutral Cards          : {self.neutral_count}",
        ]
        return "\n".join(lines)


# =========================================================================
# Input Parsing and Validation Helpers
# =========================================================================


def _coerce_curve_inputs(
    hands: Sequence[Sequence[Card | str] | str],
    board: Sequence[Card | str] | str | None,
) -> tuple[tuple[tuple[Card, Card], ...], tuple[Card, ...]]:
    """Parse and validate player hands and community board for curve calculation."""
    if len(hands) < 2:
        raise ValueError(f"At least 2 players are required, got {len(hands)}")

    parsed_hands: list[tuple[Card, Card]] = []
    seen_cards: set[Card] = set()

    for p_idx, h in enumerate(hands):
        cards = parse_cards(h) if isinstance(h, str) else [Card.from_str(c) if isinstance(c, str) else c for c in h]
        if len(cards) != 2:
            raise ValueError(f"Player {p_idx + 1} must have exactly 2 cards, got {len(cards)}")
        for c in cards:
            if c in seen_cards:
                raise ValueError(f"Duplicate card detected: {c}")
            seen_cards.add(c)
        parsed_hands.append((cards[0], cards[1]))

    parsed_board: list[Card] = []
    if board is not None:
        cards = parse_cards(board) if isinstance(board, str) else [Card.from_str(c) if isinstance(c, str) else c for c in board]
        if len(cards) not in (0, 3, 4, 5):
            raise ValueError(
                f"Board must have 0 (preflop), 3 (flop), 4 (turn), or 5 (river) cards, got {len(cards)}"
            )
        for c in cards:
            if c in seen_cards:
                raise ValueError(f"Duplicate card detected on board: {c}")
            seen_cards.add(c)
        parsed_board = cards

    return tuple(parsed_hands), tuple(parsed_board)


def _evaluate_showdown_equities(
    hands: tuple[tuple[Card, Card], ...],
    board: tuple[Card, ...],
) -> tuple[float, ...]:
    """Calculate exact showdown equities for 5-card board using 7-card evaluator."""
    board_list = list(board)
    scores = [score_7(list(h) + board_list) for h in hands]
    max_sc = max(scores)
    winners = [i for i, sc in enumerate(scores) if sc == max_sc]
    share = 1.0 / len(winners)
    return tuple(share if i in winners else 0.0 for i in range(len(hands)))


# =========================================================================
# Main Calculation Functions
# =========================================================================


def calculate_equity_curve(
    hands: Sequence[Sequence[Card | str] | str],
    board: Sequence[Card | str] | str | None = None,
    *,
    num_sims: int = 2000,
    seed: int | None = None,
) -> EquityCurve:
    """Calculate the street-by-street equity curve across community cards.

    Evaluates equity and hand rank for each street visible in ``board``:
    - Preflop (0 cards)
    - Flop (first 3 cards of board, if present)
    - Turn (first 4 cards of board, if present)
    - River (all 5 cards of board, if present)

    Parameters
    ----------
    hands : Sequence[Sequence[Card | str] | str]
        Sequence of 2-card player hands (e.g. ``["Ah As", "Kh Ks"]``).
    board : Sequence[Card | str] | str | None, default=None
        Community cards (0, 3, 4, or 5 cards).
    num_sims : int, default=2000
        Number of Monte Carlo simulations per non-showdown street.
    seed : int or None, default=None
        RNG seed for deterministic simulations.

    Returns
    -------
    EquityCurve
        Container containing street points, equity deltas, swings, and analysis.

    Raises
    ------
    ValueError
        If fewer than 2 players, invalid hand or board length, or duplicate cards.

    Examples
    --------
    >>> from poker_equity.equity_curve import calculate_equity_curve
    >>> curve = calculate_equity_curve(["Ah As", "Kh Ks"], board="2h 7h 8c 3c 4c", num_sims=1000, seed=42)
    >>> len(curve.points)
    4
    >>> curve.streets[0].value
    'preflop'
    >>> curve.streets[-1].value
    'river'
    """
    parsed_hands, parsed_board = _coerce_curve_inputs(hands, board)
    board_len = len(parsed_board)

    # Determine streets to evaluate
    streets_to_eval: list[tuple[Street, tuple[Card, ...]]] = [(Street.PREFLOP, ())]
    if board_len >= 3:
        streets_to_eval.append((Street.FLOP, parsed_board[:3]))
    if board_len >= 4:
        streets_to_eval.append((Street.TURN, parsed_board[:4]))
    if board_len == 5:
        streets_to_eval.append((Street.RIVER, parsed_board[:5]))

    points: list[StreetPoint] = []
    prev_equities: tuple[float, ...] = tuple(0.0 for _ in parsed_hands)

    hands_as_cards = [list(h) for h in parsed_hands]

    for step_idx, (street, b_slice) in enumerate(streets_to_eval):
        # 1. Equities
        if street == Street.RIVER:
            equities = _evaluate_showdown_equities(parsed_hands, b_slice)
        else:
            street_seed = (seed + step_idx * 1000) if seed is not None else None
            sim_res = calculate_equities(
                hands=hands_as_cards,
                board=list(b_slice) if b_slice else None,
                num_sims=num_sims,
                seed=street_seed,
            )
            equities = tuple(r.equity for r in sim_res)

        # 2. Hand results
        if street == Street.PREFLOP:
            hand_results = tuple(None for _ in parsed_hands)
        else:
            b_list = list(b_slice)
            hand_results = tuple(evaluate_hand(list(h) + b_list) for h in parsed_hands)

        # 3. Deltas
        if step_idx == 0:
            deltas = tuple(0.0 for _ in parsed_hands)
        else:
            deltas = tuple(eq - prev_eq for eq, prev_eq in zip(equities, prev_equities))

        prev_equities = equities

        # 4. Leader
        max_eq = max(equities)
        leader_idx = equities.index(max_eq)

        points.append(
            StreetPoint(
                street=street,
                board=b_slice,
                equities=equities,
                hand_results=hand_results,
                delta_equities=deltas,
                leader_idx=leader_idx,
                leader_equity=max_eq,
            )
        )

    return EquityCurve(
        hands=parsed_hands,
        points=tuple(points),
        board=parsed_board,
    )


def simulate_equity_curve(
    hands: Sequence[Sequence[Card | str] | str],
    board: Sequence[Card | str] | str | None = None,
    *,
    num_sims: int = 2000,
    seed: int | None = None,
) -> EquityCurve:
    """Simulate a complete 5-card runout and calculate the street-by-street equity curve.

    If ``board`` contains fewer than 5 cards (e.g. Preflop or Flop), the
    remaining cards are dealt randomly from the unseen deck, producing a
    complete Preflop -> Flop -> Turn -> River trajectory.

    Parameters
    ----------
    hands : Sequence[Sequence[Card | str] | str]
        Player hole cards.
    board : Sequence[Card | str] | str | None, default=None
        Existing community cards (0 to 5 cards).
    num_sims : int, default=2000
        Number of simulations per street.
    seed : int or None, default=None
        RNG seed for deterministic runout and simulations.

    Returns
    -------
    EquityCurve
        Complete 4-street equity curve along the completed runout.
    """
    parsed_hands, parsed_board = _coerce_curve_inputs(hands, board)

    if len(parsed_board) < 5:
        # Complete the board from unseen deck
        rng = random.Random(seed)
        known = set(itertools_chain := [c for h in parsed_hands for c in h] + list(parsed_board))
        unseen = [c for c in FULL_DECK_LIST if c not in known]
        needed = 5 - len(parsed_board)
        extra_board = rng.sample(unseen, needed)
        completed_board = list(parsed_board) + extra_board
    else:
        completed_board = list(parsed_board)

    # Calculate full curve along the completed board
    sim_seed = (seed + 42) if seed is not None else None
    return calculate_equity_curve(
        hands=parsed_hands,
        board=completed_board,
        num_sims=num_sims,
        seed=sim_seed,
    )


def calculate_street_trajectory(
    hero: Sequence[Card | str] | str,
    villain: Sequence[Card | str] | str,
    board: Sequence[Card | str] | str | None = None,
    *,
    num_sims: int = 2000,
    seed: int | None = None,
) -> EquityCurve:
    """Heads-up convenience function for calculating street-by-street equity curve.

    Parameters
    ----------
    hero : Sequence[Card | str] | str
        Hero's 2 hole cards.
    villain : Sequence[Card | str] | str
        Villain's 2 hole cards.
    board : Sequence[Card | str] | str | None, default=None
        Community cards (0 to 5 cards).
    num_sims : int, default=2000
        Simulations per street.
    seed : int or None, default=None
        RNG seed.

    Returns
    -------
    EquityCurve
        Street equity curve for Hero vs Villain.
    """
    return calculate_equity_curve(
        hands=[hero, villain],
        board=board,
        num_sims=num_sims,
        seed=seed,
    )


def calculate_next_street_distribution(
    hero: Sequence[Card | str] | str,
    villain: Sequence[Card | str] | str,
    board: Sequence[Card | str] | str,
    *,
    num_sims_per_card: int = 500,
    seed: int | None = None,
) -> NextStreetDistribution:
    """Analyze equity distribution across all possible next-street cards.

    Evaluates:
    - On Flop (3 cards): All 45 unseen Turn cards.
    - On Turn (4 cards): All 44 unseen River cards (exact showdown).

    Parameters
    ----------
    hero : Sequence[Card | str] | str
        Hero's hole cards.
    villain : Sequence[Card | str] | str
        Villain's hole cards.
    board : Sequence[Card | str] | str
        Flop (3 cards) or Turn (4 cards).
    num_sims_per_card : int, default=500
        Simulations per turn card (when evaluating Flop -> Turn).
    seed : int or None, default=None
        RNG seed.

    Returns
    -------
    NextStreetDistribution
        Detailed breakdown of favorable, unfavorable, and neutral cards.

    Raises
    ------
    ValueError
        If board is not 3 (flop) or 4 (turn) cards.
    """
    parsed_hands, parsed_board = _coerce_curve_inputs([hero, villain], board)
    board_len = len(parsed_board)

    if board_len == 3:
        curr_street = Street.FLOP
        next_st = Street.TURN
    elif board_len == 4:
        curr_street = Street.TURN
        next_st = Street.RIVER
    else:
        raise ValueError(
            f"calculate_next_street_distribution requires 3 (flop) or 4 (turn) cards, got {board_len}"
        )

    # Current street equity baseline
    if curr_street == Street.FLOP:
        cur_sim = calculate_equities(
            hands=[list(parsed_hands[0]), list(parsed_hands[1])],
            board=list(parsed_board),
            num_sims=1000,
            seed=seed,
        )
        base_hero_equity = cur_sim[0].equity
    else:  # TURN
        # Exact turn baseline
        known = set(list(parsed_hands[0]) + list(parsed_hands[1]) + list(parsed_board))
        unseen_test = [c for c in FULL_DECK_LIST if c not in known]
        h_scs = [score_7(list(parsed_hands[0]) + list(parsed_board) + [c]) for c in unseen_test]
        v_scs = [score_7(list(parsed_hands[1]) + list(parsed_board) + [c]) for c in unseen_test]
        wins = sum(1.0 if h > v else 0.5 if h == v else 0.0 for h, v in zip(h_scs, v_scs))
        base_hero_equity = wins / len(unseen_test)

    known_cards = set(list(parsed_hands[0]) + list(parsed_hands[1]) + list(parsed_board))
    unseen_cards = [c for c in FULL_DECK_LIST if c not in known_cards]
    unseen_count = len(unseen_cards)

    equities_by_card: dict[Card, tuple[float, ...]] = {}
    favorable: list[Card] = []
    unfavorable: list[Card] = []
    neutral: list[Card] = []

    for c_idx, next_card in enumerate(unseen_cards):
        new_board = list(parsed_board) + [next_card]

        if next_st == Street.RIVER:
            # Exact showdown
            eqs = _evaluate_showdown_equities(parsed_hands, tuple(new_board))
        else:
            card_seed = (seed + c_idx * 100) if seed is not None else None
            sim_res = calculate_equities(
                hands=[list(parsed_hands[0]), list(parsed_hands[1])],
                board=new_board,
                num_sims=num_sims_per_card,
                seed=card_seed,
            )
            eqs = tuple(r.equity for r in sim_res)

        equities_by_card[next_card] = eqs
        h_eq = eqs[0]

        delta = h_eq - base_hero_equity
        if delta > 0.01:
            favorable.append(next_card)
        elif delta < -0.01:
            unfavorable.append(next_card)
        else:
            neutral.append(next_card)

    all_hero_equities = [eqs[0] for eqs in equities_by_card.values()]

    return NextStreetDistribution(
        current_street=curr_street,
        next_street=next_st,
        unseen_cards_count=unseen_count,
        equities_by_card=equities_by_card,
        hero_favorable_cards=tuple(favorable),
        hero_unfavorable_cards=tuple(unfavorable),
        hero_neutral_cards=tuple(neutral),
        min_hero_equity=min(all_hero_equities),
        max_hero_equity=max(all_hero_equities),
        mean_hero_equity=sum(all_hero_equities) / len(all_hero_equities),
    )

