"""Shared constants and utility helpers for the poker-equity engine.

This module provides pre-built immutable collections and mapping tables that
are used across the evaluator, simulator, and range parser.  Everything here
is computed once at import time and never mutated.
"""

from __future__ import annotations

from typing import Final

from poker_equity.card import Card, Rank, Suit

# =========================================================================
# Full deck
# =========================================================================

FULL_DECK: Final[frozenset[Card]] = frozenset(
    Card(rank, suit) for rank in Rank for suit in Suit
)
"""Immutable set of all 52 standard playing cards."""

FULL_DECK_LIST: Final[tuple[Card, ...]] = tuple(
    Card(rank, suit) for suit in Suit for rank in Rank
)
"""All 52 cards as an ordered tuple (by suit, then rank)."""

NUM_CARDS: Final[int] = 52

# =========================================================================
# Rank helpers
# =========================================================================

RANKS: Final[tuple[Rank, ...]] = tuple(Rank)
"""All 13 ranks in ascending order (TWO → ACE)."""

RANK_CHARS: Final[str] = "23456789TJQKA"
"""Rank characters in ascending order — useful for range parsing."""

NUM_RANKS: Final[int] = 13

# =========================================================================
# Suit helpers
# =========================================================================

SUITS: Final[tuple[Suit, ...]] = tuple(Suit)
"""All 4 suits in canonical order (CLUBS → SPADES)."""

SUIT_CHARS: Final[str] = "cdhs"
"""Suit characters in canonical order."""

NUM_SUITS: Final[int] = 4

# =========================================================================
# Hand ranking categories
# =========================================================================

HAND_RANKINGS: Final[dict[int, str]] = {
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
"""Maps hand category index → human-readable name (0 = worst, 9 = best)."""

NUM_HAND_CATEGORIES: Final[int] = len(HAND_RANKINGS)

# =========================================================================
# Card utility functions
# =========================================================================


def parse_cards(text: str) -> list[Card]:
    """Parse a whitespace- or comma-separated string of card tokens.

    Parameters
    ----------
    text:
        A string like ``"Ah Kd"`` or ``"Ah,Kd"`` or ``"Ah, Kd Qs"``.

    Returns
    -------
    list[Card]
        The parsed cards in the order they appeared.

    Raises
    ------
    ValueError
        If any token is not a valid two-character card string.

    Examples
    --------
    >>> parse_cards("Ah Kd Qs")
    [Card(A♥), Card(K♦), Card(Q♠)]
    >>> parse_cards("2c,3c,4c")
    [Card(2♣), Card(3♣), Card(4♣)]
    """
    # Normalise separators: replace commas with spaces, then split
    tokens = text.replace(",", " ").split()
    return [Card.from_str(token) for token in tokens]


def validate_no_duplicates(cards: list[Card] | tuple[Card, ...]) -> None:
    """Raise :class:`ValueError` if *cards* contains duplicate entries.

    Parameters
    ----------
    cards:
        A sequence of cards to validate.

    Raises
    ------
    ValueError
        If any card appears more than once, the error message lists the
        duplicates.
    """
    seen: set[Card] = set()
    duplicates: list[Card] = []
    for card in cards:
        if card in seen:
            duplicates.append(card)
        seen.add(card)

    if duplicates:
        dup_strs = ", ".join(repr(c) for c in duplicates)
        msg = f"Duplicate card(s) detected: {dup_strs}"
        raise ValueError(msg)


def cards_to_str(cards: list[Card] | tuple[Card, ...]) -> str:
    """Format a sequence of cards into a compact, human-readable string.

    Parameters
    ----------
    cards:
        The cards to format.

    Returns
    -------
    str
        Space-separated two-character card tokens, e.g. ``"Ah Kd Qs"``.

    Examples
    --------
    >>> cards_to_str([Card.from_str("Ah"), Card.from_str("Kd")])
    'Ah Kd'
    """
    return " ".join(str(c) for c in cards)
