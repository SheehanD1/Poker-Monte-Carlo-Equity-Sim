"""Hand ranking types for the poker-equity evaluator.

This module defines the two core types used to represent evaluated poker
hands:

* :class:`HandRank` — an enum of the ten standard hand categories, from
  High Card (weakest) to Royal Flush (strongest).
* :class:`HandResult` — an immutable value object that fully describes an
  evaluated hand, including its category, a tuple of sub-ranks for
  tie-breaking (kickers), and a human-readable description.

``HandResult`` instances are totally ordered so that any two evaluated hands
can be compared directly with ``<``, ``>``, ``==``, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import total_ordering
from typing import Self


class HandRank(IntEnum):
    """The ten standard poker hand categories, ordered weakest to strongest.

    The integer values provide a natural ordering: a higher value always
    beats a lower value.  Within the same category, hands are further
    distinguished by their *sub-rank* (kicker) tuple — see
    :class:`HandResult`.
    """

    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9

    @property
    def label(self) -> str:
        """Human-readable label, e.g. ``'Three of a Kind'``."""
        return _HAND_RANK_LABELS[self.value]

    def __repr__(self) -> str:
        return f"HandRank.{self.name}"


_HAND_RANK_LABELS: dict[int, str] = {
    0: "High Card",
    1: "One Pair",
    2: "Two Pair",
    3: "Three of a Kind",
    4: "Straight",
    5: "Flush",
    6: "Full House",
    7: "Four of a Kind",
    8: "Straight Flush",
    9: "Royal Flush",
}


@total_ordering
@dataclass(frozen=True, slots=True)
class HandResult:
    """The fully evaluated result of a poker hand.

    Two ``HandResult`` objects can be compared directly: a hand with a
    higher :attr:`rank` always wins.  When ranks are equal, the
    :attr:`sub_rank` tuple is compared element-by-element to determine the
    winner (higher kicker wins).

    Parameters
    ----------
    rank:
        The hand's category (e.g. ``HandRank.FLUSH``).
    sub_rank:
        A tuple of integer values used for tie-breaking within the same
        category.  The exact semantics depend on the category:

        * **High Card / Flush**: the five card ranks in descending order.
        * **One Pair**: ``(pair_rank, kicker1, kicker2, kicker3)``.
        * **Two Pair**: ``(high_pair, low_pair, kicker)``.
        * **Three of a Kind**: ``(trips_rank, kicker1, kicker2)``.
        * **Straight / Straight Flush**: ``(high_card_rank,)``.
        * **Full House**: ``(trips_rank, pair_rank)``.
        * **Four of a Kind**: ``(quads_rank, kicker)``.
        * **Royal Flush**: ``()`` (always ties with another Royal Flush).
    description:
        A human-readable description of the hand,
        e.g. ``"Flush, Ace-high"`` or ``"Two Pair, Kings and Tens"``.

    Examples
    --------
    >>> flush = HandResult(HandRank.FLUSH, (14, 12, 10, 8, 3), "Flush, Ace-high")
    >>> pair = HandResult(HandRank.ONE_PAIR, (10, 14, 9, 5), "Pair of Tens")
    >>> flush > pair
    True
    """

    rank: HandRank
    sub_rank: tuple[int, ...]
    description: str

    # -- comparison --------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandResult):
            return NotImplemented
        return self.rank == other.rank and self.sub_rank == other.sub_rank

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, HandResult):
            return NotImplemented
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.sub_rank < other.sub_rank

    def __hash__(self) -> int:
        return hash((self.rank, self.sub_rank))

    # -- display -----------------------------------------------------------

    def __str__(self) -> str:
        return self.description

    def __repr__(self) -> str:
        return (
            f"HandResult(rank={self.rank!r}, "
            f"sub_rank={self.sub_rank}, "
            f"description={self.description!r})"
        )

    # -- convenience -------------------------------------------------------

    @property
    def category(self) -> str:
        """Alias for ``self.rank.label`` — the category name as a string."""
        return self.rank.label

    def beats(self, other: Self) -> bool:
        """Return ``True`` if this hand strictly beats *other*."""
        return self > other

    def ties_with(self, other: Self) -> bool:
        """Return ``True`` if this hand ties with *other*."""
        return self == other
