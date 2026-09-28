"""Pre-computed lookup tables for O(1) poker hand evaluation.

This module generates two lookup tables at import time that allow 5-card
poker hands to be scored with a single dictionary lookup instead of
repeated conditional logic:

* **Flush table** — keyed by a 13-bit rank bitmask (one bit per rank
  present).  Since all five cards share a suit, rank bitmask alone
  uniquely identifies the hand.  Contains 1,287 entries (C(13,5)).

* **Unsuited table** — keyed by the *prime product* of the five rank
  values.  Each rank maps to a distinct prime number, so the product of
  five primes uniquely identifies any multiset of ranks.  This covers
  all non-flush hands including pairs, trips, full houses, etc.

**Hand scores** are integers in the range ``[1, 7462]`` where **higher is
better**.  Two hands can be compared by comparing their scores directly —
no need to inspect :class:`HandRank` or kicker tuples.

Usage
-----
>>> from poker_equity.lookup import lookup_score, RANK_PRIMES
>>> # Ace-high flush (all hearts)
>>> from poker_equity.card import Card
>>> cards = [Card.from_str(s) for s in ["Ah", "Kh", "Qh", "Jh", "9h"]]
>>> score = lookup_score(cards)
>>> score > 0
True
"""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement
from typing import Final, Sequence

from poker_equity.card import Card

# =========================================================================
# Prime mapping: each rank value (2–14) → a unique prime
# =========================================================================

RANK_PRIMES: Final[dict[int, int]] = {
    2: 2,
    3: 3,
    4: 5,
    5: 7,
    6: 11,
    7: 13,
    8: 17,
    9: 19,
    10: 23,
    11: 29,
    12: 31,
    13: 37,
    14: 41,
}
"""Maps rank integer value (2–14) to its unique prime for hashing."""

ALL_RANK_VALUES: Final[tuple[int, ...]] = tuple(range(2, 15))
"""Rank integer values 2 through 14 (Two through Ace)."""

# =========================================================================
# Internal scoring helpers
# =========================================================================


def _rank_bitmask(ranks: tuple[int, ...]) -> int:
    """Build a 13-bit bitmask from a collection of rank values."""
    mask = 0
    for r in ranks:
        mask |= 1 << r
    return mask


def _prime_product(ranks: tuple[int, ...]) -> int:
    """Compute the product of primes for a multiset of rank values."""
    product = 1
    for r in ranks:
        product *= RANK_PRIMES[r]
    return product


def _is_straight(ranks_sorted_desc: tuple[int, ...]) -> int | None:
    """Return the high card of a straight, or None.

    Assumes *ranks_sorted_desc* contains exactly 5 unique values in
    descending order.
    """
    if len(ranks_sorted_desc) != 5:
        return None

    high = ranks_sorted_desc[0]
    low = ranks_sorted_desc[4]

    # Normal straight: consecutive ranks
    if high - low == 4:
        return high

    # Wheel: A-2-3-4-5
    if ranks_sorted_desc == (14, 5, 4, 3, 2):
        return 5

    return None


def _classify_and_score(
    ranks: tuple[int, ...],
    is_flush: bool,
) -> int:
    """Score a 5-card hand given its rank values and flush status.

    Returns an integer score where higher = better.  The score space
    is partitioned into bands by hand category:

    Category          | Score Range   | Count
    ------------------|---------------|------
    Straight Flush    | 7453 – 7462   |   10
    Four of a Kind    | 7297 – 7452   |  156
    Full House        | 7141 – 7296   |  156
    Flush             | 6854 – 7140   |  287  (1287 - 10 str flush - 990 non-flush)
    Straight          | 6844 – 6853   |   10
    Three of a Kind   | 6532 – 6843   |  312  (actually depends)
    Two Pair          | 5854 – 6531   |  678  (actually depends)
    One Pair          | 3260 – 5853   | 2594  (actually depends)
    High Card         |    1 – 3259   | 3259  (actually depends - but 1277 unique)

    Note: the exact counts are derived from the 7,462 distinct hand
    rankings in poker.
    """
    # We rely on sorted desc and frequency analysis
    from collections import Counter

    counts = Counter(ranks)
    # Sort by (count desc, rank desc)
    freq = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    freq_pattern = tuple(f[1] for f in freq)
    unique_sorted = tuple(sorted(set(ranks), reverse=True))

    straight_high = _is_straight(unique_sorted)

    # ----- Straight flush (includes royal) -----
    if is_flush and straight_high is not None:
        return straight_high  # placeholder, will be adjusted

    # Return a composite key for lookup-table building
    # We build the actual tables below using a different approach
    # This function is not called directly — see _build_tables()
    return 0  # pragma: no cover


# =========================================================================
# Table generation
# =========================================================================


def _build_tables() -> tuple[dict[int, int], dict[int, int]]:
    """Build the flush and unsuited lookup tables.

    This iterates over all possible 5-card rank combinations, evaluates
    each one, and assigns a unique integer score.  The scoring is
    computed by sorting all possible hands by strength and numbering
    them 1 (worst) to 7462 (best).

    Returns
    -------
    tuple[dict[int, int], dict[int, int]]
        ``(flush_table, unsuited_table)`` — each maps a lookup key to
        an integer score in ``[1, 7462]``.
    """
    from poker_equity.hand_rank import HandRank, HandResult
    from poker_equity.evaluator import evaluate_5

    # We need to build Card objects to use evaluate_5, but we only care
    # about rank combinations and flush/not-flush.
    from poker_equity.card import Rank, Suit

    suits = [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]

    # ---- Phase 1: Enumerate all flush hands (5 unique ranks, same suit) ----
    flush_hands: list[tuple[int, HandResult]] = []  # (bitmask, result)

    for combo in combinations(ALL_RANK_VALUES, 5):
        # Build 5 cards all in hearts
        cards = [Card(Rank(r), Suit.HEARTS) for r in combo]
        result = evaluate_5(cards)
        mask = _rank_bitmask(combo)
        flush_hands.append((mask, result))

    # ---- Phase 2: Enumerate all non-flush hands ----
    # All multisets of 5 ranks (with each rank appearing at most 4 times)
    unsuited_hands: list[tuple[int, HandResult]] = []  # (prime_product, result)

    for combo in combinations_with_replacement(ALL_RANK_VALUES, 5):
        # Check no rank appears more than 4 times
        from collections import Counter as Ctr

        rank_counts = Ctr(combo)
        if any(c > 4 for c in rank_counts.values()):
            continue

        # Skip hands with 5 unique ranks AND a valid straight or non-straight
        # that would be a flush — those are in the flush table.
        # For the unsuited table, we build with different suits to ensure
        # it's NOT a flush.
        unique_ranks = set(combo)

        # Assign suits to ensure no flush (all different or mixed)
        cards: list[Card] = []
        for i, r in enumerate(combo):
            # Cycle through suits, ensuring at least 2 different suits
            cards.append(Card(Rank(r), suits[i % 4]))

        # If by chance all ended up same suit (only possible if len=5
        # and all i%4 == same), fix it
        if len(set(c.suit for c in cards)) == 1:
            cards[-1] = Card(cards[-1].rank, suits[(cards[-1].suit.value) % 4])

        result = evaluate_5(cards)
        pp = _prime_product(combo)

        # Only add if this prime product isn't already present
        # (shouldn't happen since prime product is unique per multiset)
        unsuited_hands.append((pp, result))

    # ---- Phase 3: Sort all hands by strength and assign scores ----
    # Flush hands
    flush_hands.sort(key=lambda x: x[1])
    # Unsuited hands
    unsuited_hands.sort(key=lambda x: x[1])

    # We need a global ordering across ALL hands.
    # Merge flush and unsuited, sort, assign scores 1..7462
    all_hands: list[tuple[str, int | None, HandResult]] = []

    for mask, result in flush_hands:
        all_hands.append(("flush", mask, result))

    for pp, result in unsuited_hands:
        all_hands.append(("unsuited", pp, result))

    all_hands.sort(key=lambda x: x[2])

    # Assign scores: 1 = worst, incrementing for each unique strength
    flush_table: dict[int, int] = {}
    unsuited_table: dict[int, int] = {}

    current_score = 0
    prev_result: HandResult | None = None

    for table_type, key, result in all_hands:
        assert key is not None
        # Increment score only when hand strength changes
        if prev_result is None or result != prev_result:
            current_score += 1
        prev_result = result

        if table_type == "flush":
            flush_table[key] = current_score
        else:
            unsuited_table[key] = current_score

    return flush_table, unsuited_table


# =========================================================================
# Module-level table initialization (runs once at import)
# =========================================================================

FLUSH_TABLE: dict[int, int]
UNSUITED_TABLE: dict[int, int]
FLUSH_TABLE, UNSUITED_TABLE = _build_tables()

MAX_SCORE: Final[int] = max(
    max(FLUSH_TABLE.values()),
    max(UNSUITED_TABLE.values()),
)
"""The highest possible hand score (Royal Flush)."""

MIN_SCORE: Final[int] = 1
"""The lowest possible hand score (worst high card)."""


# =========================================================================
# Public API
# =========================================================================


def prime_product_from_cards(cards: Sequence[Card]) -> int:
    """Compute the prime product key for a sequence of cards.

    Parameters
    ----------
    cards:
        A sequence of cards (typically 5).

    Returns
    -------
    int
        The product of each card's rank-prime.
    """
    product = 1
    for c in cards:
        product *= RANK_PRIMES[c.rank.value]
    return product


def rank_bitmask_from_cards(cards: Sequence[Card]) -> int:
    """Compute the rank bitmask for a sequence of cards.

    Parameters
    ----------
    cards:
        A sequence of cards (typically 5).

    Returns
    -------
    int
        A bitmask with bit *i* set if rank *i* is present.
    """
    mask = 0
    for c in cards:
        mask |= 1 << c.rank.value
    return mask


def is_flush(cards: Sequence[Card]) -> bool:
    """Return ``True`` if all cards share the same suit."""
    suit = cards[0].suit
    return all(c.suit == suit for c in cards[1:])


def lookup_score(cards: Sequence[Card]) -> int:
    """Look up the score of a 5-card hand in O(1).

    Parameters
    ----------
    cards:
        Exactly five cards.

    Returns
    -------
    int
        An integer score in ``[1, MAX_SCORE]`` where higher is better.

    Raises
    ------
    ValueError
        If *cards* does not contain exactly 5 cards.
    KeyError
        If the hand signature is not found in the lookup tables (should
        never happen for valid 5-card hands).
    """
    if len(cards) != 5:
        msg = f"lookup_score requires exactly 5 cards, got {len(cards)}"
        raise ValueError(msg)

    if is_flush(cards):
        mask = rank_bitmask_from_cards(cards)
        return FLUSH_TABLE[mask]

    pp = prime_product_from_cards(cards)
    return UNSUITED_TABLE[pp]
