"""Card, Rank, and Suit primitives for the poker-equity engine.

This module defines the foundational data types used throughout the entire
project.  A ``Card`` is an immutable value object identified by its ``Rank``
and ``Suit``.  Cards support comparison (by rank, then suit), hashing (so
they can live in sets / dicts), and convenient construction from the standard
two-character shorthand (e.g. ``"Ah"`` → Ace of Hearts).
"""

from __future__ import annotations

from enum import IntEnum
from functools import total_ordering
from typing import Self


class Rank(IntEnum):
    """Card rank ordered from 2 (lowest) to Ace (highest).

    The integer values mirror the rank's natural ordering, which makes
    hand evaluation arithmetic straightforward.
    """

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def char(self) -> str:
        """Single-character string representation (e.g. ``'A'``, ``'T'``)."""
        return _RANK_TO_CHAR[self.value]

    @classmethod
    def from_char(cls, char: str) -> Rank:
        """Create a ``Rank`` from its single-character string.

        Raises
        ------
        ValueError
            If *char* is not a valid rank character.
        """
        char = char.upper()
        value = _CHAR_TO_RANK.get(char)
        if value is None:
            valid = ", ".join(_CHAR_TO_RANK)
            msg = f"Invalid rank character: {char!r}. Valid characters: {valid}"
            raise ValueError(msg)
        return cls(value)

    def __repr__(self) -> str:
        return f"Rank.{self.name}"


# Rank lookup tables (module-level to avoid IntEnum member conflicts)
_RANK_TO_CHAR: dict[int, str] = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "T", 11: "J", 12: "Q", 13: "K", 14: "A",
}
_CHAR_TO_RANK: dict[str, int] = {v: k for k, v in _RANK_TO_CHAR.items()}


class Suit(IntEnum):
    """Card suit with a conventional ordering: Clubs < Diamonds < Hearts < Spades.

    The ordering is arbitrary but consistent — it is only used for
    tie-breaking display, never for hand evaluation.
    """

    CLUBS = 1
    DIAMONDS = 2
    HEARTS = 3
    SPADES = 4

    @property
    def char(self) -> str:
        """Single lower-case character (``'c'``, ``'d'``, ``'h'``, ``'s'``)."""
        return _SUIT_TO_CHAR[self.value]

    @property
    def symbol(self) -> str:
        """Unicode suit symbol (``'♣'``, ``'♦'``, ``'♥'``, ``'♠'``)."""
        return _SUIT_TO_SYMBOL[self.value]

    @classmethod
    def from_char(cls, char: str) -> Suit:
        """Create a ``Suit`` from its single-character string.

        Raises
        ------
        ValueError
            If *char* is not a valid suit character.
        """
        char = char.lower()
        value = _CHAR_TO_SUIT.get(char)
        if value is None:
            valid = ", ".join(_CHAR_TO_SUIT)
            msg = f"Invalid suit character: {char!r}. Valid characters: {valid}"
            raise ValueError(msg)
        return cls(value)

    def __repr__(self) -> str:
        return f"Suit.{self.name}"


# Suit lookup tables (module-level to avoid IntEnum member conflicts)
_SUIT_TO_CHAR: dict[int, str] = {1: "c", 2: "d", 3: "h", 4: "s"}
_SUIT_TO_SYMBOL: dict[int, str] = {1: "♣", 2: "♦", 3: "♥", 4: "♠"}
_CHAR_TO_SUIT: dict[str, int] = {v: k for k, v in _SUIT_TO_CHAR.items()}


@total_ordering
class Card:
    """An immutable playing card with a :class:`Rank` and :class:`Suit`.

    Cards compare by rank first, then by suit.  They are hashable and can be
    used as dictionary keys and set members.

    Examples
    --------
    >>> ace_of_hearts = Card(Rank.ACE, Suit.HEARTS)
    >>> ace_of_hearts
    Card(A♥)
    >>> Card.from_str("Ah") == ace_of_hearts
    True
    """

    __slots__ = ("_rank", "_suit")

    def __init__(self, rank: Rank, suit: Suit) -> None:
        self._rank = rank
        self._suit = suit

    # -- properties --------------------------------------------------------

    @property
    def rank(self) -> Rank:
        """The card's rank."""
        return self._rank

    @property
    def suit(self) -> Suit:
        """The card's suit."""
        return self._suit

    # -- construction ------------------------------------------------------

    @classmethod
    def from_str(cls, text: str) -> Self:
        """Parse a two-character string like ``"Ah"`` into a :class:`Card`.

        Parameters
        ----------
        text:
            A two-character string where the first character is the rank
            (``2-9``, ``T``, ``J``, ``Q``, ``K``, ``A``) and the second
            character is the suit (``c``, ``d``, ``h``, ``s``).

        Raises
        ------
        ValueError
            If *text* is not exactly two characters or contains invalid
            rank / suit characters.
        """
        text = text.strip()
        if len(text) != 2:
            msg = (
                f"Card string must be exactly 2 characters, got {len(text)}: {text!r}"
            )
            raise ValueError(msg)
        rank = Rank.from_char(text[0])
        suit = Suit.from_char(text[1])
        return cls(rank, suit)

    # -- comparison --------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self._rank == other._rank and self._suit == other._suit

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        if self._rank != other._rank:
            return self._rank < other._rank
        return self._suit < other._suit

    def __hash__(self) -> int:
        return hash((self._rank, self._suit))

    # -- display -----------------------------------------------------------

    def __repr__(self) -> str:
        return f"Card({self._rank.char}{self._suit.symbol})"

    def __str__(self) -> str:
        return f"{self._rank.char}{self._suit.char}"
