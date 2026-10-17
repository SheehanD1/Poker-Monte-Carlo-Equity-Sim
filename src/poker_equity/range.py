"""Range representation and combo generation for Texas Hold'em.

This module provides the :class:`Range` class, representing a collection of
possible starting hands as canonical two-card combinations (``(Card, Card)``).
It also includes generator functions for poker hand subsets such as pocket pairs
(6 combos), suited hands (4 combos), offsuit hands (12 combos), unpaired hands
(16 combos), and the complete 1,326-combo Hold'em space.
"""

from __future__ import annotations

import itertools
import random
from typing import Final, Iterable, Iterator, Sequence, TypeAlias

from poker_equity.card import Card, Rank, Suit
from poker_equity.constants import FULL_DECK_LIST, SUITS, parse_cards

# A single Hold'em hand combo is a 2-tuple of distinct Card objects
Combo: TypeAlias = tuple[Card, Card]


# =========================================================================
# Canonical combo normalization & formatting
# =========================================================================


def canonical_combo(card1: Card, card2: Card) -> Combo:
    """Normalize a pair of cards into a canonical combo tuple.

    In Texas Hold'em, hole-card order is irrelevant: ``(Ah, Kd)`` and
    ``(Kd, Ah)`` represent the exact same hand.  Canonical ordering sorts
    descending by card rank, breaking ties by suit (Spades > Hearts > Diamonds
    > Clubs), ensuring every combo has a single unique representation.

    Parameters
    ----------
    card1 : Card
        The first card.
    card2 : Card
        The second card.

    Returns
    -------
    Combo
        A 2-tuple ``(higher_card, lower_card)``.

    Raises
    ------
    TypeError
        If *card1* or *card2* is not an instance of :class:`Card`.
    ValueError
        If *card1* and *card2* are identical cards.

    Examples
    --------
    >>> from poker_equity.card import Card
    >>> c1 = Card.from_str("Kd")
    >>> c2 = Card.from_str("Ah")
    >>> canonical_combo(c1, c2)
    (Card(A♥), Card(K♦))
    """
    if not isinstance(card1, Card) or not isinstance(card2, Card):
        raise TypeError(
            f"Combo cards must be Card instances, got {type(card1).__name__} and {type(card2).__name__}"
        )
    if card1 == card2:
        msg = f"A combo cannot contain two identical cards: {card1!r}"
        raise ValueError(msg)

    if card1 >= card2:
        return (card1, card2)
    return (card2, card1)


def combo_to_str(combo: Combo) -> str:
    """Format a combo as a 4-character string (e.g. ``"AhKd"``).

    Parameters
    ----------
    combo : Combo
        A canonical 2-card combo.

    Returns
    -------
    str
        Four-character string representation.
    """
    return f"{combo[0]}{combo[1]}"


def combo_from_str(text: str) -> Combo:
    """Parse a combo string into a canonical 2-card combo.

    Accepts 4-character strings like ``"AhKd"`` or delimited strings like
    ``"Ah Kd"`` and ``"Ah,Kd"``.

    Parameters
    ----------
    text : str
        String containing two cards.

    Returns
    -------
    Combo
        The normalized canonical combo tuple.

    Raises
    ------
    ValueError
        If the string cannot be parsed into exactly two distinct cards.
    """
    stripped = text.strip()
    if len(stripped) == 4 and " " not in stripped and "," not in stripped:
        c1 = Card.from_str(stripped[:2])
        c2 = Card.from_str(stripped[2:])
        return canonical_combo(c1, c2)

    cards = parse_cards(stripped)
    if len(cards) != 2:
        msg = f"Combo string must contain exactly 2 cards, got {len(cards)}: {text!r}"
        raise ValueError(msg)
    return canonical_combo(cards[0], cards[1])


# =========================================================================
# Rank coercion helper
# =========================================================================


def _coerce_rank(rank: Rank | str) -> Rank:
    """Convert a Rank enum, single-character string, or rank name to Rank.

    Parameters
    ----------
    rank : Rank | str
        A :class:`Rank` enum or string (e.g. ``"A"``, ``"T"``, ``"10"``, ``"ACE"``).

    Returns
    -------
    Rank
        The corresponding :class:`Rank` enum value.

    Raises
    ------
    ValueError
        If *rank* is not a valid rank representation.
    """
    if isinstance(rank, Rank):
        return rank
    if isinstance(rank, str):
        cleaned = rank.strip()
        if len(cleaned) == 1:
            return Rank.from_char(cleaned)
        if cleaned.upper() == "10":
            return Rank.TEN
        upper = cleaned.upper()
        if hasattr(Rank, upper):
            return getattr(Rank, upper)
    raise ValueError(f"Invalid rank: {rank!r}")


# =========================================================================
# Combo generators
# =========================================================================


def generate_pair_combos(rank: Rank | str) -> set[Combo]:
    """Generate all 6 combos for a pocket pair of the given rank.

    For any rank, there are C(4, 2) = 6 suit combinations:
    ``(s, h)``, ``(s, d)``, ``(s, c)``, ``(h, d)``, ``(h, c)``, ``(d, c)``.

    Parameters
    ----------
    rank : Rank | str
        The card rank (e.g. ``Rank.ACE`` or ``"A"``).

    Returns
    -------
    set[Combo]
        Set of 6 canonical combos.
    """
    r = _coerce_rank(rank)
    combos: set[Combo] = set()
    for s1, s2 in itertools.combinations(SUITS, 2):
        combos.add(canonical_combo(Card(r, s1), Card(r, s2)))
    return combos


def generate_suited_combos(rank1: Rank | str, rank2: Rank | str) -> set[Combo]:
    """Generate all 4 suited combos for two distinct ranks.

    There are 4 suits (clubs, diamonds, hearts, spades), yielding exactly
    4 suited combos where both cards share the same suit.

    Parameters
    ----------
    rank1 : Rank | str
        The first rank.
    rank2 : Rank | str
        The second rank (must differ from *rank1*).

    Returns
    -------
    set[Combo]
        Set of 4 canonical suited combos.

    Raises
    ------
    ValueError
        If *rank1* and *rank2* are equal.
    """
    r1 = _coerce_rank(rank1)
    r2 = _coerce_rank(rank2)
    if r1 == r2:
        raise ValueError(
            f"Suited combos require two different ranks, got {r1.name} and {r2.name}"
        )

    combos: set[Combo] = set()
    for s in SUITS:
        combos.add(canonical_combo(Card(r1, s), Card(r2, s)))
    return combos


def generate_offsuit_combos(rank1: Rank | str, rank2: Rank | str) -> set[Combo]:
    """Generate all 12 offsuit combos for two distinct ranks.

    There are 4 * 3 = 12 combinations of distinct suits for two ranks.

    Parameters
    ----------
    rank1 : Rank | str
        The first rank.
    rank2 : Rank | str
        The second rank (must differ from *rank1*).

    Returns
    -------
    set[Combo]
        Set of 12 canonical offsuit combos.

    Raises
    ------
    ValueError
        If *rank1* and *rank2* are equal.
    """
    r1 = _coerce_rank(rank1)
    r2 = _coerce_rank(rank2)
    if r1 == r2:
        raise ValueError(
            f"Offsuit combos require two different ranks, got {r1.name} and {r2.name}"
        )

    combos: set[Combo] = set()
    for s1 in SUITS:
        for s2 in SUITS:
            if s1 != s2:
                combos.add(canonical_combo(Card(r1, s1), Card(r2, s2)))
    return combos


def generate_unpaired_combos(rank1: Rank | str, rank2: Rank | str) -> set[Combo]:
    """Generate all 16 combos (4 suited + 12 offsuit) for two distinct ranks.

    Parameters
    ----------
    rank1 : Rank | str
        The first rank.
    rank2 : Rank | str
        The second rank (must differ from *rank1*).

    Returns
    -------
    set[Combo]
        Set of 16 canonical combos.
    """
    return generate_suited_combos(rank1, rank2) | generate_offsuit_combos(rank1, rank2)


def generate_all_combos() -> set[Combo]:
    """Generate all 1,326 Texas Hold'em starting hand combos.

    C(52, 2) = 1,326 total combinations:
    - 13 pocket pairs * 6 = 78 pair combos
    - 78 suited hand types * 4 = 312 suited combos
    - 78 offsuit hand types * 12 = 936 offsuit combos
    - Total: 78 + 312 + 936 = 1,326 combos.

    Returns
    -------
    set[Combo]
        Set of all 1,326 canonical combos.
    """
    combos: set[Combo] = set()
    for c1, c2 in itertools.combinations(FULL_DECK_LIST, 2):
        combos.add(canonical_combo(c1, c2))
    return combos


TOTAL_HOLDEM_COMBOS: Final[int] = 1326
"""Total number of starting hand combos in Texas Hold'em: C(52, 2) = 1,326."""


# =========================================================================
# Range Class
# =========================================================================


class Range:
    """An immutable collection of Texas Hold'em starting hand combos.

    A ``Range`` stores a set of canonical ``(Card, Card)`` combos.  Combos
    are always normalized via :func:`canonical_combo` so that card order does
    not affect equality, hashing, or membership testing.

    The class supports set operations (union, intersection, difference),
    random sampling, card queries, and factory methods for common hand subsets.

    Examples
    --------
    >>> r_aces = Range.pair("A")
    >>> len(r_aces)
    6
    >>> r_aks = Range.suited("A", "K")
    >>> len(r_aks)
    4
    >>> r_top = r_aces | r_aks
    >>> len(r_top)
    10
    >>> (Card.from_str("Ah"), Card.from_str("Kh")) in r_top
    True
    """

    __slots__ = ("_combos", "_combos_list")

    def __init__(
        self,
        combos: Iterable[tuple[Card, Card] | Sequence[Card]] | None = None,
    ) -> None:
        """Initialize a Range with an optional collection of combos.

        Parameters
        ----------
        combos : Iterable[tuple[Card, Card] | Sequence[Card]] | None
            A collection of 2-card combos, or ``None`` to create an empty range.

        Raises
        ------
        ValueError
            If any combo does not contain exactly two cards, or contains
            identical cards.
        TypeError
            If any combo elements are not :class:`Card` instances.
        """
        if combos is None:
            self._combos: frozenset[Combo] = frozenset()
        elif isinstance(combos, Range):
            self._combos = combos._combos
        else:
            validated: set[Combo] = set()
            for item in combos:
                if not isinstance(item, (tuple, list)) or len(item) != 2:
                    msg = f"Each combo must be a 2-card sequence, got: {item!r}"
                    raise ValueError(msg)
                c1, c2 = item[0], item[1]
                if not isinstance(c1, Card) or not isinstance(c2, Card):
                    msg = (
                        f"Combo elements must be Card instances, got: "
                        f"{type(c1).__name__}, {type(c2).__name__}"
                    )
                    raise TypeError(msg)
                validated.add(canonical_combo(c1, c2))
            self._combos = frozenset(validated)

        self._combos_list: tuple[Combo, ...] | None = None

    # -- Properties --------------------------------------------------------

    @property
    def combos(self) -> frozenset[Combo]:
        """The immutable frozenset of canonical combos in this range."""
        return self._combos

    @property
    def combos_list(self) -> tuple[Combo, ...]:
        """All combos as a sorted tuple in descending order (highest card first)."""
        if self._combos_list is None:
            self._combos_list = tuple(sorted(self._combos, reverse=True))
        return self._combos_list

    @property
    def is_empty(self) -> bool:
        """True if the range contains zero combos."""
        return len(self._combos) == 0

    # -- Factory Constructors ---------------------------------------------

    @classmethod
    def from_combos(
        cls, combos: Iterable[tuple[Card, Card] | Sequence[Card]]
    ) -> Range:
        """Create a Range from an iterable of combos.

        Parameters
        ----------
        combos : Iterable[tuple[Card, Card] | Sequence[Card]]
            A collection of 2-card combos.

        Returns
        -------
        Range
            The constructed Range.
        """
        return cls(combos)

    @classmethod
    def from_cards(cls, card1: Card | str, card2: Card | str) -> Range:
        """Create a single-combo Range from two cards or card strings.

        Parameters
        ----------
        card1 : Card | str
            The first card (e.g. ``Card.from_str("Ah")`` or ``"Ah"``).
        card2 : Card | str
            The second card (e.g. ``Card.from_str("Kd")`` or ``"Kd"``).

        Returns
        -------
        Range
            A Range containing exactly 1 combo.
        """
        c1 = Card.from_str(card1) if isinstance(card1, str) else card1
        c2 = Card.from_str(card2) if isinstance(card2, str) else card2
        return cls([canonical_combo(c1, c2)])

    @classmethod
    def pair(cls, rank: Rank | str) -> Range:
        """Create a Range containing all 6 combos of a pocket pair.

        Parameters
        ----------
        rank : Rank | str
            The pocket pair rank (e.g. ``"A"``, ``"K"``, ``Rank.QUEEN``).

        Returns
        -------
        Range
            Range with 6 combos.
        """
        return cls(generate_pair_combos(rank))

    @classmethod
    def suited(cls, rank1: Rank | str, rank2: Rank | str) -> Range:
        """Create a Range containing all 4 suited combos of two ranks.

        Parameters
        ----------
        rank1 : Rank | str
            The first rank.
        rank2 : Rank | str
            The second rank.

        Returns
        -------
        Range
            Range with 4 combos.
        """
        return cls(generate_suited_combos(rank1, rank2))

    @classmethod
    def offsuit(cls, rank1: Rank | str, rank2: Rank | str) -> Range:
        """Create a Range containing all 12 offsuit combos of two ranks.

        Parameters
        ----------
        rank1 : Rank | str
            The first rank.
        rank2 : Rank | str
            The second rank.

        Returns
        -------
        Range
            Range with 12 combos.
        """
        return cls(generate_offsuit_combos(rank1, rank2))

    @classmethod
    def unpaired(cls, rank1: Rank | str, rank2: Rank | str) -> Range:
        """Create a Range containing all 16 combos (suited + offsuit) of two ranks.

        Parameters
        ----------
        rank1 : Rank | str
            The first rank.
        rank2 : Rank | str
            The second rank.

        Returns
        -------
        Range
            Range with 16 combos.
        """
        return cls(generate_unpaired_combos(rank1, rank2))

    @classmethod
    def all(cls) -> Range:
        """Create a Range containing all 1,326 Texas Hold'em starting hand combos.

        Returns
        -------
        Range
            Range with 1,326 combos (100% of hands).
        """
        return cls(generate_all_combos())

    @classmethod
    def empty(cls) -> Range:
        """Create an empty Range (0 combos).

        Returns
        -------
        Range
            Range with 0 combos.
        """
        return cls()

    # -- Set Operations ---------------------------------------------------

    def union(self, *others: Range) -> Range:
        """Return a new Range with combos present in this range or any of *others*.

        Parameters
        ----------
        *others : Range
            One or more Range instances to union with.

        Returns
        -------
        Range
            The merged Range.
        """
        merged = set(self._combos)
        for other in others:
            if not isinstance(other, Range):
                msg = f"Expected Range instance, got {type(other).__name__}"
                raise TypeError(msg)
            merged.update(other._combos)
        return Range(merged)

    def intersection(self, *others: Range) -> Range:
        """Return a new Range with combos present in this range and all of *others*.

        Parameters
        ----------
        *others : Range
            One or more Range instances to intersect with.

        Returns
        -------
        Range
            The intersection Range.
        """
        res = set(self._combos)
        for other in others:
            if not isinstance(other, Range):
                msg = f"Expected Range instance, got {type(other).__name__}"
                raise TypeError(msg)
            res.intersection_update(other._combos)
        return Range(res)

    def difference(self, *others: Range) -> Range:
        """Return a new Range with combos in this range but not in *others*.

        Parameters
        ----------
        *others : Range
            One or more Range instances to subtract.

        Returns
        -------
        Range
            The difference Range.
        """
        res = set(self._combos)
        for other in others:
            if not isinstance(other, Range):
                msg = f"Expected Range instance, got {type(other).__name__}"
                raise TypeError(msg)
            res.difference_update(other._combos)
        return Range(res)

    def symmetric_difference(self, other: Range) -> Range:
        """Return a new Range with combos in either range, but not both.

        Parameters
        ----------
        other : Range
            Another Range instance.

        Returns
        -------
        Range
            The symmetric difference Range.
        """
        if not isinstance(other, Range):
            msg = f"Expected Range instance, got {type(other).__name__}"
            raise TypeError(msg)
        return Range(self._combos ^ other._combos)

    def __or__(self, other: Range) -> Range:
        if not isinstance(other, Range):
            return NotImplemented
        return self.union(other)

    def __and__(self, other: Range) -> Range:
        if not isinstance(other, Range):
            return NotImplemented
        return self.intersection(other)

    def __sub__(self, other: Range) -> Range:
        if not isinstance(other, Range):
            return NotImplemented
        return self.difference(other)

    def __xor__(self, other: Range) -> Range:
        if not isinstance(other, Range):
            return NotImplemented
        return self.symmetric_difference(other)

    # -- Container / Protocol Methods -------------------------------------

    def __len__(self) -> int:
        """Return the number of unique combos in this range."""
        return len(self._combos)

    def __iter__(self) -> Iterator[Combo]:
        """Iterate over combos in canonical sorted order (highest card first)."""
        return iter(self.combos_list)

    def __bool__(self) -> bool:
        """Return True if the range is non-empty."""
        return bool(self._combos)

    def __contains__(self, item: object) -> bool:
        """Check if a combo or 2-card sequence is contained in this range.

        Accepts a 2-card tuple/list (e.g. ``(c1, c2)``) or a 4-character
        string (e.g. ``"AhKd"``).
        """
        if isinstance(item, (tuple, list)) and len(item) == 2:
            c1, c2 = item[0], item[1]
            if isinstance(c1, Card) and isinstance(c2, Card):
                try:
                    return canonical_combo(c1, c2) in self._combos
                except ValueError:
                    return False
        if isinstance(item, str):
            try:
                return combo_from_str(item) in self._combos
            except (ValueError, TypeError):
                return False
        return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Range):
            return NotImplemented
        return self._combos == other._combos

    def __hash__(self) -> int:
        return hash(self._combos)

    def __repr__(self) -> str:
        count = len(self._combos)
        combo_word = "combo" if count == 1 else "combos"
        return f"Range({count} {combo_word})"

    def __str__(self) -> str:
        count = len(self._combos)
        combo_word = "combo" if count == 1 else "combos"
        return f"Range({count} {combo_word})"

    # -- Sampling & Card Queries ------------------------------------------

    def sample(self, rng: random.Random | None = None) -> Combo:
        """Randomly select one combo uniformly from the range.

        Parameters
        ----------
        rng : random.Random | None
            Optional pseudo-random number generator for reproducible sampling.
            If ``None``, uses Python's standard :mod:`random` module.

        Returns
        -------
        Combo
            A randomly chosen canonical combo.

        Raises
        ------
        ValueError
            If the range is empty.
        """
        if not self._combos:
            raise ValueError("Cannot sample from an empty Range")
        if rng is not None:
            return rng.choice(self.combos_list)
        return random.choice(self.combos_list)

    def contains_card(self, card: Card) -> bool:
        """Check if any combo in the range contains the specified card.

        Parameters
        ----------
        card : Card
            The card to query.

        Returns
        -------
        bool
            True if *card* appears in at least one combo in the range.
        """
        return any(card == c1 or card == c2 for c1, c2 in self._combos)

    def cards(self) -> set[Card]:
        """Return all distinct cards that appear in at least one combo in this range.

        Returns
        -------
        set[Card]
            Set of cards present in this range.
        """
        result: set[Card] = set()
        for c1, c2 in self._combos:
            result.add(c1)
            result.add(c2)
        return result
