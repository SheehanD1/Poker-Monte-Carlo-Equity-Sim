"""Deck management for the poker-equity engine.

A :class:`Deck` represents a standard 52-card deck that supports shuffling,
dealing, and card removal.  It accepts an optional random seed for fully
reproducible simulations — critical for deterministic testing and Monte Carlo
convergence analysis.
"""

from __future__ import annotations

import random
from typing import Sequence

from poker_equity.card import Card, Rank, Suit


def _build_full_deck() -> list[Card]:
    """Return a fresh, ordered list of all 52 cards."""
    return [Card(rank, suit) for suit in Suit for rank in Rank]


class Deck:
    """A standard 52-card deck with shuffle, deal, and removal support.

    Parameters
    ----------
    seed:
        Optional seed for the internal PRNG.  When provided, every call to
        :meth:`shuffle` and :meth:`deal` is fully deterministic.

    Examples
    --------
    >>> deck = Deck(seed=42)
    >>> deck.shuffle()
    >>> hand = deck.deal(2)
    >>> len(hand)
    2
    >>> deck.remaining
    50
    """

    __slots__ = ("_cards", "_rng")

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._cards: list[Card] = _build_full_deck()

    # -- properties --------------------------------------------------------

    @property
    def remaining(self) -> int:
        """Number of cards still in the deck."""
        return len(self._cards)

    @property
    def cards(self) -> list[Card]:
        """A *copy* of the cards currently in the deck (top = index 0)."""
        return list(self._cards)

    # -- mutating operations -----------------------------------------------

    def shuffle(self) -> None:
        """Shuffle the deck in place using the internal PRNG."""
        self._rng.shuffle(self._cards)

    def deal(self, n: int = 1) -> list[Card]:
        """Remove and return *n* cards from the top of the deck.

        Parameters
        ----------
        n:
            Number of cards to deal.  Must be ≥ 1 and ≤ :attr:`remaining`.

        Returns
        -------
        list[Card]
            The dealt cards, in the order they were drawn.

        Raises
        ------
        ValueError
            If *n* < 1 or *n* exceeds the number of remaining cards.
        """
        if n < 1:
            msg = f"Must deal at least 1 card, got {n}"
            raise ValueError(msg)
        if n > len(self._cards):
            msg = (
                f"Cannot deal {n} card(s) from a deck with "
                f"only {len(self._cards)} remaining"
            )
            raise ValueError(msg)
        dealt = self._cards[:n]
        self._cards = self._cards[n:]
        return dealt

    def deal_one(self) -> Card:
        """Convenience method: deal exactly one card.

        Returns
        -------
        Card
            The top card of the deck.

        Raises
        ------
        ValueError
            If the deck is empty.
        """
        return self.deal(1)[0]

    def remove(self, cards: Card | Sequence[Card]) -> None:
        """Remove specific card(s) from the deck (regardless of position).

        This is used to set up known board / hole cards before running a
        simulation — the removed cards will never be dealt randomly.

        Parameters
        ----------
        cards:
            A single :class:`Card` or a sequence of cards to remove.

        Raises
        ------
        ValueError
            If any card in *cards* is not currently in the deck.
        """
        if isinstance(cards, Card):
            cards = [cards]

        # Build a set for O(1) membership checks
        current = set(self._cards)
        for card in cards:
            if card not in current:
                msg = f"{card!r} is not in the deck"
                raise ValueError(msg)
            current.discard(card)

        removed = set(cards)
        self._cards = [c for c in self._cards if c not in removed]

    def reset(self, seed: int | None = None) -> None:
        """Restore the deck to a full, ordered 52-card state.

        Parameters
        ----------
        seed:
            If provided, re-seed the internal PRNG.  If ``None``, the
            existing PRNG state is preserved.
        """
        self._cards = _build_full_deck()
        if seed is not None:
            self._rng = random.Random(seed)

    # -- dunder methods ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._cards)

    def __contains__(self, card: object) -> bool:
        if not isinstance(card, Card):
            return False
        return card in self._cards

    def __repr__(self) -> str:
        return f"Deck(remaining={self.remaining})"
