"""Five-, six-, and seven-card poker hand evaluator.

This module provides:

* :func:`evaluate_5` — evaluate exactly five cards.
* :func:`evaluate_7` — evaluate seven cards by picking the best 5-card
  combination out of C(7, 5) = 21 possibilities.
* :func:`evaluate_hand` — convenience dispatcher that accepts 5, 6, or 7
  cards.

The implementation uses **bit-manipulation** for flush and straight
detection:

* **Flush**: all five cards share the same suit — a single equality check.
* **Straight**: each rank maps to a bit position; five consecutive set bits
  form a straight.  The wheel (A-2-3-4-5) is handled as a special case.

Pair / trips / quads detection uses a :class:`~collections.Counter` on the
rank values, then classifies by the resulting frequency pattern.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Sequence

from poker_equity.card import Card, Rank
from poker_equity.hand_rank import HandRank, HandResult

# =========================================================================
# Rank name helpers (for building description strings)
# =========================================================================

_RANK_NAMES: dict[int, str] = {
    2: "Twos", 3: "Threes", 4: "Fours", 5: "Fives",
    6: "Sixes", 7: "Sevens", 8: "Eights", 9: "Nines",
    10: "Tens", 11: "Jacks", 12: "Queens", 13: "Kings", 14: "Aces",
}

_RANK_NAME_SINGULAR: dict[int, str] = {
    2: "Two", 3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 8: "Eight", 9: "Nine",
    10: "Ten", 11: "Jack", 12: "Queen", 13: "King", 14: "Ace",
}

_RANK_CHAR: dict[int, str] = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "T", 11: "J", 12: "Q", 13: "K", 14: "A",
}


def _high_label(rank_val: int) -> str:
    """Return e.g. ``'Ace'`` for 14, ``'King'`` for 13."""
    return _RANK_NAME_SINGULAR[rank_val]


# =========================================================================
# Core evaluation helpers
# =========================================================================


def _is_flush(cards: Sequence[Card]) -> bool:
    """Return ``True`` if all five cards share the same suit."""
    suit = cards[0].suit
    return all(c.suit == suit for c in cards[1:])


def _straight_high(ranks_desc: tuple[int, ...]) -> int | None:
    """Return the high card of the straight, or ``None`` if not a straight.

    Parameters
    ----------
    ranks_desc:
        The five rank values in **descending** order (duplicates already
        removed by the caller — if there are fewer than 5 unique ranks
        this is not a straight).

    Returns
    -------
    int | None
        The rank value of the highest card in the straight, or ``None``.
        For the wheel (A-2-3-4-5), returns ``5``.
    """
    if len(ranks_desc) != 5:
        return None

    # Build a bitmask: bit i is set if rank i is present
    bitmask = 0
    for r in ranks_desc:
        bitmask |= 1 << r

    # Normal straight: five consecutive bits
    high = ranks_desc[0]
    straight_mask = 0
    for i in range(5):
        straight_mask |= 1 << (high - i)
    if bitmask == straight_mask:
        return high

    # Wheel (A-2-3-4-5): Ace acts as 1
    wheel_mask = (1 << 14) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
    if bitmask == wheel_mask:
        return 5  # 5-high straight

    return None


# =========================================================================
# Public API
# =========================================================================


def evaluate_5(cards: Sequence[Card]) -> HandResult:
    """Evaluate a five-card poker hand.

    Parameters
    ----------
    cards:
        Exactly five :class:`Card` objects.

    Returns
    -------
    HandResult
        The evaluated hand with rank, sub-rank (for tie-breaking), and a
        human-readable description.

    Raises
    ------
    ValueError
        If *cards* does not contain exactly five cards.
    """
    if len(cards) != 5:
        msg = f"evaluate_5 requires exactly 5 cards, got {len(cards)}"
        raise ValueError(msg)

    # Extract rank values and sort descending
    rank_values = sorted((c.rank.value for c in cards), reverse=True)
    flush = _is_flush(cards)

    # Count rank frequencies: {rank_value: count}
    counts = Counter(rank_values)
    # Sort by (count desc, rank desc) for classification
    freq = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    freq_pattern = tuple(f[1] for f in freq)  # e.g. (2, 1, 1, 1) for one pair

    unique_desc = tuple(r for r, _ in freq)  # unique ranks, ordered by freq then rank
    ranks_desc = tuple(sorted(set(rank_values), reverse=True))

    # Check for straight
    straight_high = _straight_high(ranks_desc)

    # -----------------------------------------------------------------
    # Classify the hand (from strongest to weakest)
    # -----------------------------------------------------------------

    # --- Royal Flush / Straight Flush ---
    if flush and straight_high is not None:
        if straight_high == 14:  # A-K-Q-J-T
            return HandResult(
                rank=HandRank.ROYAL_FLUSH,
                sub_rank=(),
                description="Royal Flush",
            )
        high_name = _high_label(straight_high)
        return HandResult(
            rank=HandRank.STRAIGHT_FLUSH,
            sub_rank=(straight_high,),
            description=f"Straight Flush, {high_name}-high",
        )

    # --- Four of a Kind ---
    if freq_pattern == (4, 1):
        quads_rank = freq[0][0]
        kicker = freq[1][0]
        return HandResult(
            rank=HandRank.FOUR_OF_A_KIND,
            sub_rank=(quads_rank, kicker),
            description=f"Four of a Kind, {_RANK_NAMES[quads_rank]}",
        )

    # --- Full House ---
    if freq_pattern == (3, 2):
        trips_rank = freq[0][0]
        pair_rank = freq[1][0]
        return HandResult(
            rank=HandRank.FULL_HOUSE,
            sub_rank=(trips_rank, pair_rank),
            description=(
                f"Full House, {_RANK_NAMES[trips_rank]} full of "
                f"{_RANK_NAMES[pair_rank]}"
            ),
        )

    # --- Flush ---
    if flush:
        sub = tuple(rank_values)  # already sorted desc
        return HandResult(
            rank=HandRank.FLUSH,
            sub_rank=sub,
            description=f"Flush, {_high_label(sub[0])}-high",
        )

    # --- Straight ---
    if straight_high is not None:
        high_name = _high_label(straight_high)
        return HandResult(
            rank=HandRank.STRAIGHT,
            sub_rank=(straight_high,),
            description=f"Straight, {high_name}-high",
        )

    # --- Three of a Kind ---
    if freq_pattern == (3, 1, 1):
        trips_rank = freq[0][0]
        kickers = tuple(sorted((freq[1][0], freq[2][0]), reverse=True))
        return HandResult(
            rank=HandRank.THREE_OF_A_KIND,
            sub_rank=(trips_rank, *kickers),
            description=f"Three of a Kind, {_RANK_NAMES[trips_rank]}",
        )

    # --- Two Pair ---
    if freq_pattern == (2, 2, 1):
        high_pair = max(freq[0][0], freq[1][0])
        low_pair = min(freq[0][0], freq[1][0])
        kicker = freq[2][0]
        return HandResult(
            rank=HandRank.TWO_PAIR,
            sub_rank=(high_pair, low_pair, kicker),
            description=(
                f"Two Pair, {_RANK_NAMES[high_pair]} and "
                f"{_RANK_NAMES[low_pair]}"
            ),
        )

    # --- One Pair ---
    if freq_pattern == (2, 1, 1, 1):
        pair_rank = freq[0][0]
        kickers = tuple(sorted(
            (freq[1][0], freq[2][0], freq[3][0]), reverse=True
        ))
        return HandResult(
            rank=HandRank.ONE_PAIR,
            sub_rank=(pair_rank, *kickers),
            description=f"Pair of {_RANK_NAMES[pair_rank]}",
        )

    # --- High Card ---
    sub = tuple(rank_values)
    return HandResult(
        rank=HandRank.HIGH_CARD,
        sub_rank=sub,
        description=f"{_high_label(sub[0])}-high",
    )


def evaluate_7(cards: Sequence[Card]) -> HandResult:
    """Evaluate a seven-card poker hand (e.g. 2 hole + 5 board).

    Iterates over all C(7, 5) = 21 five-card combinations and returns
    the best :class:`HandResult`.

    Parameters
    ----------
    cards:
        Exactly seven :class:`Card` objects.

    Returns
    -------
    HandResult
        The best possible 5-card hand from the seven cards.

    Raises
    ------
    ValueError
        If *cards* does not contain exactly seven cards.
    """
    if len(cards) != 7:
        msg = f"evaluate_7 requires exactly 7 cards, got {len(cards)}"
        raise ValueError(msg)

    best: HandResult | None = None
    for combo in combinations(cards, 5):
        result = evaluate_5(combo)
        if best is None or result > best:
            best = result

    assert best is not None  # guaranteed with 7 cards
    return best


def evaluate_hand(cards: Sequence[Card]) -> HandResult:
    """Evaluate a poker hand of 5, 6, or 7 cards.

    This is a convenience dispatcher:

    * **5 cards** → calls :func:`evaluate_5` directly.
    * **6 cards** → picks the best of C(6, 5) = 6 five-card combos.
    * **7 cards** → calls :func:`evaluate_7` (best of 21 combos).

    Parameters
    ----------
    cards:
        Five, six, or seven :class:`Card` objects.

    Returns
    -------
    HandResult
        The best possible 5-card hand from the given cards.

    Raises
    ------
    ValueError
        If the number of cards is not 5, 6, or 7.
    """
    n = len(cards)

    if n == 5:
        return evaluate_5(cards)

    if n == 7:
        return evaluate_7(cards)

    if n == 6:
        best: HandResult | None = None
        for combo in combinations(cards, 5):
            result = evaluate_5(combo)
            if best is None or result > best:
                best = result
        assert best is not None
        return best

    msg = f"evaluate_hand requires 5, 6, or 7 cards, got {n}"
    raise ValueError(msg)


# =========================================================================
# Fast score-based evaluation (lookup-table hot path)
# =========================================================================
#
# These functions return a single integer score instead of a HandResult.
# Higher score = better hand.  They are the primary evaluation path used
# by the Monte Carlo simulator, where constructing HandResult objects
# (with description strings) for millions of hands would be wasteful.
#
# The lookup tables are imported lazily (on first call) to avoid a
# circular import with lookup.py, which uses evaluate_5 during table
# generation.
# =========================================================================

_FLUSH_TABLE: dict[int, int] | None = None
_UNSUITED_TABLE: dict[int, int] | None = None
_RANK_PRIMES: dict[int, int] | None = None


def _ensure_tables() -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """Lazy-load the lookup tables on first use."""
    global _FLUSH_TABLE, _UNSUITED_TABLE, _RANK_PRIMES  # noqa: PLW0603
    if _FLUSH_TABLE is None:
        from poker_equity.lookup import FLUSH_TABLE, RANK_PRIMES, UNSUITED_TABLE

        _FLUSH_TABLE = FLUSH_TABLE
        _UNSUITED_TABLE = UNSUITED_TABLE
        _RANK_PRIMES = RANK_PRIMES
    assert _UNSUITED_TABLE is not None
    assert _RANK_PRIMES is not None
    return _FLUSH_TABLE, _UNSUITED_TABLE, _RANK_PRIMES


def score_5(cards: Sequence[Card]) -> int:
    """Score a 5-card hand using O(1) lookup tables.

    This is the **fast path** for Monte Carlo simulation — it returns a
    single integer score instead of a full :class:`HandResult`.

    Parameters
    ----------
    cards:
        Exactly five :class:`Card` objects.

    Returns
    -------
    int
        An integer score where **higher is better**.  Scores range from
        1 (worst high card: 7-5-4-3-2) to 7462 (Royal Flush).
    """
    flush_t, unsuited_t, primes = _ensure_tables()

    # Flush check: all five cards same suit
    suit = cards[0].suit
    is_flush = all(c.suit == suit for c in cards[1:])

    if is_flush:
        # Rank bitmask (one bit per unique rank)
        mask = 0
        for c in cards:
            mask |= 1 << c.rank.value
        return flush_t[mask]

    # Prime product of rank values
    product = 1
    for c in cards:
        product *= primes[c.rank.value]
    return unsuited_t[product]


def score_7(cards: Sequence[Card]) -> int:
    """Score a 7-card hand by finding the best 5-card score.

    Iterates over all C(7, 5) = 21 five-card combinations and returns
    the **highest** score.  This is the fast path used by the Monte Carlo
    simulator.

    Parameters
    ----------
    cards:
        Exactly seven :class:`Card` objects.

    Returns
    -------
    int
        The best 5-card score from the seven cards.
    """
    flush_t, unsuited_t, primes = _ensure_tables()
    best = 0

    for combo in combinations(cards, 5):
        # Inline the score_5 logic to avoid function-call overhead
        suit = combo[0].suit
        if all(c.suit == suit for c in combo[1:]):
            mask = 0
            for c in combo:
                mask |= 1 << c.rank.value
            score = flush_t[mask]
        else:
            product = 1
            for c in combo:
                product *= primes[c.rank.value]
            score = unsuited_t[product]

        if score > best:
            best = score

    return best


def score_hand(cards: Sequence[Card]) -> int:
    """Score a poker hand of 5, 6, or 7 cards using lookup tables.

    This is the fast-path equivalent of :func:`evaluate_hand`.

    Parameters
    ----------
    cards:
        Five, six, or seven :class:`Card` objects.

    Returns
    -------
    int
        The best 5-card score from the given cards.

    Raises
    ------
    ValueError
        If the number of cards is not 5, 6, or 7.
    """
    n = len(cards)

    if n == 5:
        return score_5(cards)

    if n == 7:
        return score_7(cards)

    if n == 6:
        flush_t, unsuited_t, primes = _ensure_tables()
        best = 0
        for combo in combinations(cards, 5):
            suit = combo[0].suit
            if all(c.suit == suit for c in combo[1:]):
                mask = 0
                for c in combo:
                    mask |= 1 << c.rank.value
                score = flush_t[mask]
            else:
                product = 1
                for c in combo:
                    product *= primes[c.rank.value]
                score = unsuited_t[product]
            if score > best:
                best = score
        return best

    msg = f"score_hand requires 5, 6, or 7 cards, got {n}"
    raise ValueError(msg)
