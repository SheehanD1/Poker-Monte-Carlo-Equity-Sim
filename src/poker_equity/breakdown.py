"""Range vs. hand equity breakdown for Texas Hold'em.

This module provides fine-grained per-combo equity analysis when evaluating
a poker hand range against a specific holding.  For every unblocked combo in
the range, the exact or simulated equity against the target hand is computed,
enabling deep strategic analysis (e.g. combo ranking, 13x13 grid cell
aggregation, blocker impact assessment).
"""

from __future__ import annotations

from typing import Mapping, Sequence

from poker_equity.card import Card
from poker_equity.constants import cards_to_str
from poker_equity.range import (
    Combo,
    Range,
    canonical_combo,
    combo_from_str,
    combo_to_cell,
    combo_to_str,
)
from poker_equity.simulation import (
    EquityResult,
    _coerce_board,
    calculate_equity,
)


# =========================================================================
# Input normalization helpers
# =========================================================================


def _coerce_hand(hand: Sequence[Card | str] | str) -> Combo:
    """Parse and normalize a single 2-card hand into a canonical Combo.

    Parameters
    ----------
    hand : Sequence[Card | str] | str
        A 2-card sequence or a 4-character string (e.g. ``"AhKd"`` or ``"Ah Kd"``).

    Returns
    -------
    Combo
        A 2-tuple of Card objects sorted canonically.

    Raises
    ------
    ValueError
        If the hand does not contain exactly two cards or contains duplicates.
    TypeError
        If elements are not Card instances or valid card strings.
    """
    if isinstance(hand, str):
        cleaned = hand.replace(",", " ").strip()
        tokens = cleaned.split()
        if len(tokens) == 2:
            c1, c2 = Card.from_str(tokens[0]), Card.from_str(tokens[1])
        elif len(cleaned) == 4 and not tokens:
            c1, c2 = Card.from_str(cleaned[:2]), Card.from_str(cleaned[2:])
        elif len(tokens) == 1 and len(tokens[0]) == 4:
            c1, c2 = Card.from_str(tokens[0][:2]), Card.from_str(tokens[0][2:])
        else:
            msg = f"Hand must represent exactly 2 cards, got: {hand!r}"
            raise ValueError(msg)
    elif isinstance(hand, (list, tuple)):
        if len(hand) != 2:
            msg = f"Hand must contain exactly 2 cards, got {len(hand)}"
            raise ValueError(msg)
        c1 = Card.from_str(hand[0]) if isinstance(hand[0], str) else hand[0]
        c2 = Card.from_str(hand[1]) if isinstance(hand[1], str) else hand[1]
        if not isinstance(c1, Card) or not isinstance(c2, Card):
            msg = f"Hand elements must be Card instances or card strings, got: {hand!r}"
            raise TypeError(msg)
    else:
        msg = f"Hand must be a 2-card sequence or string, got {type(hand).__name__}"
        raise TypeError(msg)

    if c1 == c2:
        msg = f"A hand cannot contain duplicate cards: {c1}"
        raise ValueError(msg)

    return canonical_combo(c1, c2)


def _coerce_range(range_input: Range | str) -> Range:
    """Normalize range input into a Range instance."""
    if isinstance(range_input, Range):
        r = range_input
    elif isinstance(range_input, str):
        r = Range.from_str(range_input)
    else:
        msg = f"range must be a Range instance or string, got {type(range_input).__name__}"
        raise TypeError(msg)

    if r.is_empty:
        raise ValueError("Range is empty")
    return r


# =========================================================================
# RangeBreakdown container
# =========================================================================


class RangeBreakdown(dict[Combo, EquityResult]):
    """Mapping of each unblocked combo in a range to its EquityResult against a specific hand.

    Inherits from ``dict[Combo, EquityResult]``, providing dictionary behavior
    (indexing, iteration, length) while adding rich analysis properties such as
    overall equity, sorted combo rankings, 13x13 grid cell aggregation, and
    tabular summaries.

    Keys can be accessed via canonical :class:`Combo` tuples or 4-character
    string representations (e.g. ``breakdown["AhKd"]`` or ``breakdown[(c1, c2)]``).
    """

    __slots__ = ("_range", "_hand", "_board", "_as_hero")

    def __init__(
        self,
        items: Mapping[Combo, EquityResult],
        range_: Range,
        hand: Combo,
        board: list[Card],
        *,
        as_hero: bool = False,
    ) -> None:
        super().__init__(items)
        self._range = range_
        self._hand = hand
        self._board = board
        self._as_hero = as_hero

    def __getitem__(self, key: Combo | str) -> EquityResult:
        if isinstance(key, str):
            c = combo_from_str(key)
            return super().__getitem__(c)
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            try:
                c = combo_from_str(key)
                return super().__contains__(c)
            except (ValueError, TypeError):
                return False
        return super().__contains__(key)

    # -- Properties --------------------------------------------------------

    @property
    def range(self) -> Range:
        """The unblocked range evaluated in this breakdown."""
        return self._range

    @property
    def hand(self) -> Combo:
        """The specific opponent hand evaluated against the range."""
        return self._hand

    @property
    def hand_str(self) -> str:
        """String representation of the opponent hand (e.g. ``'AhKd'``)."""
        return combo_to_str(self._hand)

    @property
    def board(self) -> list[Card]:
        """Community board cards."""
        return list(self._board)

    @property
    def num_combos(self) -> int:
        """Total number of live, unblocked combos evaluated."""
        return len(self)

    @property
    def overall_equity(self) -> float:
        """Arithmetic mean equity across all unblocked combos in the breakdown."""
        if not self:
            return 0.0
        return sum(r.equity for r in self.values()) / len(self)

    @property
    def overall_win_pct(self) -> float:
        """Average win percentage across all unblocked combos."""
        if not self:
            return 0.0
        return sum(r.win_pct for r in self.values()) / len(self)

    @property
    def overall_tie_pct(self) -> float:
        """Average tie percentage across all unblocked combos."""
        if not self:
            return 0.0
        return sum(r.tie_pct for r in self.values()) / len(self)

    @property
    def overall_loss_pct(self) -> float:
        """Average loss percentage across all unblocked combos."""
        if not self:
            return 0.0
        return sum(r.loss_pct for r in self.values()) / len(self)

    @property
    def hero_equity(self) -> float:
        """Equity from hero's perspective."""
        if self._as_hero:
            return self.overall_equity
        return 1.0 - self.overall_equity

    @property
    def villain_equity(self) -> float:
        """Equity from villain's perspective."""
        if self._as_hero:
            return 1.0 - self.overall_equity
        return self.overall_equity

    # -- Analysis methods --------------------------------------------------

    def sorted_combos(self, descending: bool = True) -> list[tuple[Combo, EquityResult]]:
        """Return all (combo, EquityResult) pairs sorted by equity.

        Parameters
        ----------
        descending : bool, default True
            If True, highest equity combos appear first.

        Returns
        -------
        list[tuple[Combo, EquityResult]]
            Sorted list of combo-result pairs.
        """
        return sorted(self.items(), key=lambda item: item[1].equity, reverse=descending)

    def best_combos(self, n: int = 5) -> list[tuple[Combo, EquityResult]]:
        """Return the top *n* highest equity combos.

        Parameters
        ----------
        n : int, default 5
            Number of top combos to return.

        Returns
        -------
        list[tuple[Combo, EquityResult]]
            Top *n* combos sorted descending by equity.
        """
        return self.sorted_combos(descending=True)[:n]

    def worst_combos(self, n: int = 5) -> list[tuple[Combo, EquityResult]]:
        """Return the bottom *n* lowest equity combos.

        Parameters
        ----------
        n : int, default 5
            Number of bottom combos to return.

        Returns
        -------
        list[tuple[Combo, EquityResult]]
            Bottom *n* combos sorted ascending by equity.
        """
        return self.sorted_combos(descending=False)[:n]

    def by_cell(self) -> dict[str, float]:
        """Aggregate equities by standard 169-grid cell notation (e.g. 'AKs', 'QQ').

        Returns
        -------
        dict[str, float]
            Dictionary mapping cell notation to its average equity across live combos.
        """
        cell_equities: dict[str, list[float]] = {}
        for combo, result in self.items():
            cell = combo_to_cell(combo)
            cell_equities.setdefault(cell, []).append(result.equity)
        return {
            cell: sum(eqs) / len(eqs)
            for cell, eqs in cell_equities.items()
        }

    def by_cell_counts(self) -> dict[str, int]:
        """Count the number of live combos present in each 169-grid cell.

        Returns
        -------
        dict[str, int]
            Dictionary mapping cell notation to the count of live combos.
        """
        cell_counts: dict[str, int] = {}
        for combo in self.keys():
            cell = combo_to_cell(combo)
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
        return cell_counts

    def summary(self, top_n: int | None = None) -> str:
        """Format a human-readable text table summarizing the breakdown.

        Parameters
        ----------
        top_n : int | None, optional
            If set, displays only the top *top_n* combos followed by a count
            of remaining combos.  If None, displays all combos.

        Returns
        -------
        str
            Formatted summary table.
        """
        board_str = cards_to_str(self._board) if self._board else "Preflop (none)"
        lines = [
            f"Range vs Hand Breakdown: {self._range} vs {self.hand_str}",
            f"Board: {board_str}",
            f"Overall Equity: {self.overall_equity:.1%} "
            f"(Win: {self.overall_win_pct:.1%} | "
            f"Tie: {self.overall_tie_pct:.1%} | "
            f"Lose: {self.overall_loss_pct:.1%})",
            f"Opponent Equity: {1.0 - self.overall_equity:.1%}",
            "-" * 68,
            f"{'Combo':<8} {'Cell':<6} {'Equity':<10} {'Win %':<10} {'Tie %':<10} {'Lose %':<10} {'Sims':<8}",
            "-" * 68,
        ]

        combos_sorted = self.sorted_combos(descending=True)
        display_list = combos_sorted if top_n is None else combos_sorted[:top_n]

        for combo, res in display_list:
            c_str = combo_to_str(combo)
            cell_str = combo_to_cell(combo)
            lines.append(
                f"{c_str:<8} {cell_str:<6} {res.equity:<10.1%} {res.win_pct:<10.1%} "
                f"{res.tie_pct:<10.1%} {res.loss_pct:<10.1%} {res.num_sims:<8,}"
            )

        if top_n is not None and len(combos_sorted) > top_n:
            rem = len(combos_sorted) - top_n
            lines.append(f"... and {rem} more combos")

        lines.append("-" * 68)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"RangeBreakdown(combos={len(self)}, "
            f"overall_equity={self.overall_equity:.4f}, "
            f"hand={self.hand_str!r})"
        )


# =========================================================================
# Core Breakdown Functions
# =========================================================================


def range_vs_hand_breakdown(
    range_: Range | str,
    hand: Sequence[Card | str] | str,
    board: Sequence[Card | str] | str | None = None,
    *,
    num_sims_per_combo: int = 1_000,
    seed: int | None = None,
) -> RangeBreakdown:
    """Calculate the equity breakdown of every unblocked combo in a range against a hand.

    For each combo in *range_* that is not blocked by *hand* or *board*, calculates
    its equity against *hand* via Monte Carlo simulation.

    Parameters
    ----------
    range_ : Range | str
        The range to analyze, as a :class:`Range` instance or range string
        (e.g. ``"QQ+, AKs"``).
    hand : Sequence[Card | str] | str
        Opponent's 2-card hand (e.g. ``[Card.from_str("Ah"), Card.from_str("Kd")]``
        or ``"AhKd"``).
    board : Sequence[Card | str] | str | None, optional
        Community cards (0 to 5 cards).
    num_sims_per_combo : int, default 1,000
        Number of Monte Carlo simulations run for each unblocked combo.
    seed : int | None, optional
        Optional PRNG master seed for deterministic reproducibility.

    Returns
    -------
    RangeBreakdown
        A mapping of each live combo to its :class:`EquityResult`, with
        aggregate properties.

    Raises
    ------
    ValueError
        If inputs are invalid, duplicate cards are detected, or no combos
        in the range remain unblocked.

    Examples
    --------
    >>> from poker_equity.breakdown import range_vs_hand_breakdown
    >>> breakdown = range_vs_hand_breakdown("QQ+, AKs", "AhKd", num_sims_per_combo=1000, seed=42)
    >>> len(breakdown)  # 14 live combos (Ah and Kd are blocked)
    14
    >>> breakdown.overall_equity > 0.60
    True
    """
    if num_sims_per_combo <= 0:
        raise ValueError(f"num_sims_per_combo must be positive, got {num_sims_per_combo}")

    hand_combo = _coerce_hand(hand)
    board_cards = _coerce_board(board)
    range_obj = _coerce_range(range_)

    if len(board_cards) > 5:
        raise ValueError(f"Board can have at most 5 cards, got {len(board_cards)}")

    # Check for duplicate cards within board
    seen_cards: set[Card] = set()
    for c in board_cards:
        if c in seen_cards:
            raise ValueError(f"Duplicate card in board: {c}")
        seen_cards.add(c)

    # Check for duplicate cards between hand and board
    if hand_combo[0] in seen_cards:
        raise ValueError(f"Duplicate card between hand and board: {hand_combo[0]}")
    if hand_combo[1] in seen_cards:
        raise ValueError(f"Duplicate card between hand and board: {hand_combo[1]}")

    dead_cards = seen_cards | {hand_combo[0], hand_combo[1]}
    live_range = range_obj.remove_dead_cards(dead_cards)

    if live_range.is_empty:
        raise ValueError("No combos in range remain after removing dead cards from hand and board")

    results_dict: dict[Combo, EquityResult] = {}
    for i, combo in enumerate(live_range.combos_list):
        combo_seed = seed + i if seed is not None else None
        res = calculate_equity(
            hero=list(combo),
            villain=list(hand_combo),
            board=board_cards,
            num_sims=num_sims_per_combo,
            seed=combo_seed,
        )
        results_dict[combo] = res

    return RangeBreakdown(
        items=results_dict,
        range_=live_range,
        hand=hand_combo,
        board=board_cards,
        as_hero=False,
    )


def hand_vs_range_breakdown(
    hand: Sequence[Card | str] | str,
    range_: Range | str,
    board: Sequence[Card | str] | str | None = None,
    *,
    num_sims_per_combo: int = 1_000,
    seed: int | None = None,
) -> RangeBreakdown:
    """Calculate equity breakdown from hero hand's perspective against each combo in a range.

    Convenience counterpart to :func:`range_vs_hand_breakdown` where Hero holds
    the specific *hand* and each combo's :class:`EquityResult` represents Hero's
    equity against that combo.

    Parameters
    ----------
    hand : Sequence[Card | str] | str
        Hero's 2-card hand.
    range_ : Range | str
        Villain's range.
    board : Sequence[Card | str] | str | None, optional
        Community cards.
    num_sims_per_combo : int, default 1,000
        Simulations per combo.
    seed : int | None, optional
        Optional PRNG seed.

    Returns
    -------
    RangeBreakdown
        Breakdown where each combo maps to Hero's equity against that combo.
    """
    if num_sims_per_combo <= 0:
        raise ValueError(f"num_sims_per_combo must be positive, got {num_sims_per_combo}")

    hand_combo = _coerce_hand(hand)
    board_cards = _coerce_board(board)
    range_obj = _coerce_range(range_)

    if len(board_cards) > 5:
        raise ValueError(f"Board can have at most 5 cards, got {len(board_cards)}")

    seen_cards: set[Card] = set()
    for c in board_cards:
        if c in seen_cards:
            raise ValueError(f"Duplicate card in board: {c}")
        seen_cards.add(c)

    if hand_combo[0] in seen_cards:
        raise ValueError(f"Duplicate card between hand and board: {hand_combo[0]}")
    if hand_combo[1] in seen_cards:
        raise ValueError(f"Duplicate card between hand and board: {hand_combo[1]}")

    dead_cards = seen_cards | {hand_combo[0], hand_combo[1]}
    live_range = range_obj.remove_dead_cards(dead_cards)

    if live_range.is_empty:
        raise ValueError("No combos in range remain after removing dead cards from hand and board")

    results_dict: dict[Combo, EquityResult] = {}
    for i, combo in enumerate(live_range.combos_list):
        combo_seed = seed + i if seed is not None else None
        res = calculate_equity(
            hero=list(hand_combo),
            villain=list(combo),
            board=board_cards,
            num_sims=num_sims_per_combo,
            seed=combo_seed,
        )
        results_dict[combo] = res

    return RangeBreakdown(
        items=results_dict,
        range_=live_range,
        hand=hand_combo,
        board=board_cards,
        as_hero=True,
    )


#: Shorthand alias for range_vs_hand_breakdown
range_breakdown = range_vs_hand_breakdown
