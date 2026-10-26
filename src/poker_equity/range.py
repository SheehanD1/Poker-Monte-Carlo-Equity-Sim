"""Range representation and combo generation for Texas Hold'em.

This module provides the :class:`Range` class, representing a collection of
possible starting hands as canonical two-card combinations (``(Card, Card)``).
It also includes generator functions for poker hand subsets such as pocket pairs
(6 combos), suited hands (4 combos), offsuit hands (12 combos), unpaired hands
(16 combos), and the complete 1,326-combo Hold'em space.
"""

from __future__ import annotations

import collections
import itertools
import random
import re
from typing import Final, Iterable, Iterator, Mapping, Sequence, TypeAlias

from poker_equity.card import Card, Rank, Suit
from poker_equity.constants import (
    FULL_DECK_LIST,
    RANK_CHARS,
    SUIT_CHARS,
    SUITS,
    parse_cards,
)

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
# Range String Parsing
# =========================================================================


def _is_card_string(token: str) -> bool:
    """Check if a 2-character string represents a valid Card (e.g. 'Ah')."""
    return (
        len(token) == 2
        and token[0].upper() in RANK_CHARS
        and token[1].lower() in SUIT_CHARS
    )


def _coerce_pair_ranks(s: str) -> tuple[Rank, Rank]:
    """Extract and sort two rank characters descending: (higher_rank, lower_rank)."""
    if len(s) != 2:
        msg = f"Expected exactly 2 rank characters, got: {s!r}"
        raise ValueError(msg)
    r1 = Rank.from_char(s[0])
    r2 = Rank.from_char(s[1])
    if r1 >= r2:
        return (r1, r2)
    return (r2, r1)


def _tokenize_range_string(range_str: str) -> list[str]:
    """Tokenize a range string into individual hand / combo tokens."""
    # Fast path: check if the string parses as two individual cards (e.g. "Ah, Kd" or "Ah Kd")
    try:
        cards = parse_cards(range_str)
        if len(cards) == 2:
            return [f"{cards[0]}{cards[1]}"]
    except ValueError:
        pass

    # Normalize whitespace around weight delimiters so tokens like "AKs : 0.5" stay intact
    norm = re.sub(r"\s*([:@])\s*", r"\1", range_str)
    norm = re.sub(r"\[\s*(.*?)\s*\]", r"[\1]", norm)

    # Replace alternate delimiters with commas
    text = norm.replace(";", ",").replace("\n", ",")
    comma_parts = [p.strip() for p in text.split(",") if p.strip()]
    tokens: list[str] = []
    for part in comma_parts:
        sub = part.split()
        if not sub:
            continue
        # If all sub-elements are 2-char card tokens and count is even (e.g. "Ah Kd Qs Qd"),
        # group them into 4-char specific combo pairs ("AhKd", "QsQd").
        if len(sub) >= 2 and all(_is_card_string(x) for x in sub) and len(sub) % 2 == 0:
            for i in range(0, len(sub), 2):
                tokens.append(f"{sub[i]}{sub[i + 1]}")
        else:
            tokens.extend(sub)
    return tokens


def _extract_token_and_weight(token: str) -> tuple[str, float]:
    """Separate a range token into its base hand specification and frequency weight.

    Supports multiple poker frequency notations:
    - Colon notation: ``"AKs:0.5"``, ``"AKs:50%"``, ``"AKs:50"``
    - At notation: ``"AKs@0.5"``, ``"AKs@50%"``
    - Bracket notation: ``"AKs[0.5]"``, ``"AKs[50%]"``

    Tokens without explicit weights default to a weight of 1.0 (100%).

    Parameters
    ----------
    token : str
        Single range token (e.g. ``"AKs:0.5"``, ``"QQ+"``).

    Returns
    -------
    tuple[str, float]
        Tuple of ``(base_token, weight)`` where ``weight`` is a float in ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If the weight string is malformed or out of range.
    """
    cleaned = token.strip()
    if not cleaned:
        return "", 1.0

    base_token = cleaned
    weight_str: str | None = None

    if cleaned.endswith("]") and "[" in cleaned:
        open_bracket = cleaned.rfind("[")
        base_token = cleaned[:open_bracket].strip()
        weight_str = cleaned[open_bracket + 1 : -1].strip()
    elif "@" in cleaned:
        parts = cleaned.split("@")
        if len(parts) != 2:
            raise ValueError(f"Invalid weight syntax in token: {token!r}")
        base_token = parts[0].strip()
        weight_str = parts[1].strip()
    elif ":" in cleaned:
        parts = cleaned.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid weight syntax in token: {token!r}")
        base_token = parts[0].strip()
        weight_str = parts[1].strip()

    if weight_str is None:
        return base_token, 1.0

    is_percent = False
    if weight_str.endswith("%"):
        is_percent = True
        weight_str = weight_str[:-1].strip()

    try:
        val = float(weight_str)
    except ValueError:
        raise ValueError(f"Invalid weight value in token {token!r}: {weight_str!r}") from None

    if is_percent:
        if not (0.0 <= val <= 100.0):
            raise ValueError(f"Weight percentage must be between 0% and 100%, got {val}%")
        weight = val / 100.0
    else:
        if val < 0.0 or val > 100.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0 (or 0 and 100), got {val}")
        if val > 1.0:
            weight = val / 100.0
        else:
            weight = val

    return base_token, weight


def _parse_dash_token(token: str) -> set[Combo]:
    """Parse a dash-range token (e.g. '88-66', 'T9s-76s', 'A5s-A2s', 'AQo-ATo', 'AQ-AT')."""
    parts = token.split("-")
    if len(parts) != 2:
        msg = f"Invalid range token: {token!r}"
        raise ValueError(msg)
    part1 = parts[0].strip()
    part2 = parts[1].strip()
    if not part1 or not part2:
        msg = f"Invalid range token: {token!r}"
        raise ValueError(msg)

    # 1. Pocket pairs: "88-66" or "66-88"
    if (
        len(part1) == 2
        and len(part2) == 2
        and part1[0].upper() == part1[1].upper()
        and part2[0].upper() == part2[1].upper()
    ):
        r1 = Rank.from_char(part1[0])
        r2 = Rank.from_char(part2[0])
        low_val = min(r1.value, r2.value)
        high_val = max(r1.value, r2.value)
        combos: set[Combo] = set()
        for r_val in range(low_val, high_val + 1):
            combos.update(generate_pair_combos(Rank(r_val)))
        return combos

    # 2. Suited hands: "T9s-76s" or "A5s-A2s"
    if (
        len(part1) == 3
        and len(part2) == 3
        and part1[2].lower() == "s"
        and part2[2].lower() == "s"
    ):
        r1_a, r2_a = _coerce_pair_ranks(part1[:2])
        r1_b, r2_b = _coerce_pair_ranks(part2[:2])
        if r1_a == r2_a or r1_b == r2_b:
            msg = f"Suited hand cannot have identical ranks: {token!r}"
            raise ValueError(msg)

        combos = set()
        # Case A: Same primary rank (kicker range, e.g. "A5s-A2s" or "A2s-A5s")
        if r1_a == r1_b:
            primary = r1_a
            k_low = min(r2_a.value, r2_b.value)
            k_high = max(r2_a.value, r2_b.value)
            for k_val in range(k_low, k_high + 1):
                combos.update(generate_suited_combos(primary, Rank(k_val)))
            return combos

        # Case B: Stepping connectors / gappers (e.g. "T9s-76s" or "76s-T9s")
        gap_a = r1_a.value - r2_a.value
        gap_b = r1_b.value - r2_b.value
        if gap_a != gap_b:
            msg = f"Range endpoints must have matching gap, got {part1} and {part2}"
            raise ValueError(msg)
        gap = gap_a
        top_low = min(r1_a, r1_b)
        top_high = max(r1_a, r1_b)
        for top_val in range(top_low.value, top_high.value + 1):
            bottom_val = top_val - gap
            if bottom_val >= Rank.TWO.value:
                combos.update(generate_suited_combos(Rank(top_val), Rank(bottom_val)))
        return combos

    # 3. Offsuit hands: "AQo-ATo" or "T9o-76o"
    if (
        len(part1) == 3
        and len(part2) == 3
        and part1[2].lower() == "o"
        and part2[2].lower() == "o"
    ):
        r1_a, r2_a = _coerce_pair_ranks(part1[:2])
        r1_b, r2_b = _coerce_pair_ranks(part2[:2])
        if r1_a == r2_a or r1_b == r2_b:
            msg = f"Offsuit hand cannot have identical ranks: {token!r}"
            raise ValueError(msg)

        combos = set()
        if r1_a == r1_b:
            primary = r1_a
            k_low = min(r2_a.value, r2_b.value)
            k_high = max(r2_a.value, r2_b.value)
            for k_val in range(k_low, k_high + 1):
                combos.update(generate_offsuit_combos(primary, Rank(k_val)))
            return combos

        gap_a = r1_a.value - r2_a.value
        gap_b = r1_b.value - r2_b.value
        if gap_a != gap_b:
            msg = f"Range endpoints must have matching gap, got {part1} and {part2}"
            raise ValueError(msg)
        gap = gap_a
        top_low = min(r1_a, r1_b)
        top_high = max(r1_a, r1_b)
        for top_val in range(top_low.value, top_high.value + 1):
            bottom_val = top_val - gap
            if bottom_val >= Rank.TWO.value:
                combos.update(generate_offsuit_combos(Rank(top_val), Rank(bottom_val)))
        return combos

    # 4. Unpaired hands (both suited and offsuit): "AQ-AT" or "T9-76"
    if (
        len(part1) == 2
        and len(part2) == 2
        and part1[0].upper() != part1[1].upper()
        and part2[0].upper() != part2[1].upper()
    ):
        r1_a, r2_a = _coerce_pair_ranks(part1)
        r1_b, r2_b = _coerce_pair_ranks(part2)

        combos = set()
        if r1_a == r1_b:
            primary = r1_a
            k_low = min(r2_a.value, r2_b.value)
            k_high = max(r2_a.value, r2_b.value)
            for k_val in range(k_low, k_high + 1):
                combos.update(generate_unpaired_combos(primary, Rank(k_val)))
            return combos

        gap_a = r1_a.value - r2_a.value
        gap_b = r1_b.value - r2_b.value
        if gap_a != gap_b:
            msg = f"Range endpoints must have matching gap, got {part1} and {part2}"
            raise ValueError(msg)
        gap = gap_a
        top_low = min(r1_a, r1_b)
        top_high = max(r1_a, r1_b)
        for top_val in range(top_low.value, top_high.value + 1):
            bottom_val = top_val - gap
            if bottom_val >= Rank.TWO.value:
                combos.update(generate_unpaired_combos(Rank(top_val), Rank(bottom_val)))
        return combos

    msg = f"Incompatible range endpoints: {token!r}"
    raise ValueError(msg)


def _parse_plus_token(token: str) -> set[Combo]:
    """Parse a plus token (e.g. 'QQ+', 'AKs+', 'A2s+', '76s+', 'AQo+', 'AQ+')."""
    base = token[:-1].strip()

    # 1. Pair plus: "QQ+", "88+", "AA+"
    if len(base) == 2 and base[0].upper() == base[1].upper():
        r = Rank.from_char(base[0])
        combos: set[Combo] = set()
        for r_val in range(r.value, Rank.ACE.value + 1):
            combos.update(generate_pair_combos(Rank(r_val)))
        return combos

    # 2. Suited plus: "A2s+", "KTs+", "76s+", "AKs+"
    if len(base) == 3 and base[2].lower() == "s":
        r1, r2 = _coerce_pair_ranks(base[:2])
        if r1 == r2:
            msg = f"Suited hand cannot have identical ranks: {token!r}"
            raise ValueError(msg)
        gap = r1.value - r2.value
        combos = set()
        # Connectors below King: stepping connectors up to AKs
        if gap == 1 and r1 < Rank.KING:
            for top_val in range(r1.value, Rank.ACE.value + 1):
                combos.update(generate_suited_combos(Rank(top_val), Rank(top_val - 1)))
        else:
            # Kicker moves up to r1 - 1
            for k_val in range(r2.value, r1.value):
                combos.update(generate_suited_combos(r1, Rank(k_val)))
        return combos

    # 3. Offsuit plus: "AQo+", "ATo+", "76o+", "AKo+"
    if len(base) == 3 and base[2].lower() == "o":
        r1, r2 = _coerce_pair_ranks(base[:2])
        if r1 == r2:
            msg = f"Offsuit hand cannot have identical ranks: {token!r}"
            raise ValueError(msg)
        gap = r1.value - r2.value
        combos = set()
        if gap == 1 and r1 < Rank.KING:
            for top_val in range(r1.value, Rank.ACE.value + 1):
                combos.update(generate_offsuit_combos(Rank(top_val), Rank(top_val - 1)))
        else:
            for k_val in range(r2.value, r1.value):
                combos.update(generate_offsuit_combos(r1, Rank(k_val)))
        return combos

    # 4. Unpaired plus: "AQ+", "AT+", "76+", "AK+"
    if len(base) == 2 and base[0].upper() != base[1].upper():
        r1, r2 = _coerce_pair_ranks(base)
        gap = r1.value - r2.value
        combos = set()
        if gap == 1 and r1 < Rank.KING:
            for top_val in range(r1.value, Rank.ACE.value + 1):
                combos.update(generate_unpaired_combos(Rank(top_val), Rank(top_val - 1)))
        else:
            for k_val in range(r2.value, r1.value):
                combos.update(generate_unpaired_combos(r1, Rank(k_val)))
        return combos

    msg = f"Invalid plus token: {token!r}"
    raise ValueError(msg)


def _parse_token(token: str) -> set[Combo]:
    """Parse a single range token into a set of canonical combos."""
    cleaned = token.strip()
    if not cleaned:
        return set()

    lower = cleaned.lower()
    # Special keywords
    if lower in ("100%", "random", "any", "*", "all", "xx"):
        return generate_all_combos()
    if lower in ("none", "empty", "0%"):
        return set()

    # Specific 4-char combo: "AhKd"
    if (
        len(cleaned) == 4
        and _is_card_string(cleaned[:2])
        and _is_card_string(cleaned[2:])
    ):
        c1 = Card.from_str(cleaned[:2])
        c2 = Card.from_str(cleaned[2:])
        return {canonical_combo(c1, c2)}

    # Dash range: "88-66", "T9s-76s", "A5s-A2s"
    if "-" in cleaned:
        return _parse_dash_token(cleaned)

    # Plus range: "QQ+", "AKs+", "A2s+"
    if cleaned.endswith("+"):
        return _parse_plus_token(cleaned)

    # Single pocket pair: "AA", "KK"
    if len(cleaned) == 2 and cleaned[0].upper() == cleaned[1].upper():
        r = Rank.from_char(cleaned[0])
        return generate_pair_combos(r)

    # Single suited hand: "AKs", "QJs"
    if len(cleaned) == 3 and cleaned[2].lower() == "s":
        r1, r2 = _coerce_pair_ranks(cleaned[:2])
        if r1 == r2:
            msg = f"Suited hand cannot have identical ranks: {cleaned!r}"
            raise ValueError(msg)
        return generate_suited_combos(r1, r2)

    # Single offsuit hand: "AKo", "QJo"
    if len(cleaned) == 3 and cleaned[2].lower() == "o":
        r1, r2 = _coerce_pair_ranks(cleaned[:2])
        if r1 == r2:
            msg = f"Offsuit hand cannot have identical ranks: {cleaned!r}"
            raise ValueError(msg)
        return generate_offsuit_combos(r1, r2)

    # Single unpaired hand (suited + offsuit): "AK", "QJ"
    if len(cleaned) == 2 and cleaned[0].upper() != cleaned[1].upper():
        r1, r2 = _coerce_pair_ranks(cleaned)
        return generate_unpaired_combos(r1, r2)

    msg = f"Invalid range token: {cleaned!r}"
    raise ValueError(msg)


def parse_weighted_range_string(range_str: str) -> dict[Combo, float]:
    """Parse a poker range string into a mapping of canonical combos to frequency weights.

    Tokens may specify frequencies/weights using colon, at-sign, or bracket notation:
    - ``"AKs:0.5"`` or ``"AKs:50%"`` or ``"AKs:50"``
    - ``"QQ+@0.75"``
    - ``"T9s-76s[0.25]"``
    - ``"AhKd:0.5"``

    Tokens without explicit weights default to a weight of 1.0.  Later tokens
    in the range string overwrite earlier weights for the same combos.  A weight
    of 0.0 removes previously added combos.

    Parameters
    ----------
    range_str : str
        Standard poker range syntax string with optional weights.

    Returns
    -------
    dict[Combo, float]
        Dictionary mapping each combo to its weight in (0.0, 1.0].

    Raises
    ------
    ValueError
        If any token or weight syntax is invalid.
    """
    tokens = _tokenize_range_string(range_str)
    weights: dict[Combo, float] = {}
    for token in tokens:
        if not token.strip():
            continue
        base_token, weight = _extract_token_and_weight(token)
        if not base_token:
            continue
        combos = _parse_token(base_token)
        if weight <= 0.0:
            for c in combos:
                weights.pop(c, None)
        else:
            for c in combos:
                weights[c] = weight
    return weights


def parse_range_string(range_str: str) -> set[Combo]:
    """Parse a standard poker range string into a set of canonical combos.

    Parameters
    ----------
    range_str : str
        Standard poker range syntax string, e.g. ``"AKs, QQ+, T9s-76s"``.

    Returns
    -------
    set[Combo]
        Set of canonical 2-card combos.

    Raises
    ------
    ValueError
        If any token in *range_str* has invalid poker range syntax.
    """
    return set(parse_weighted_range_string(range_str).keys())


# =========================================================================
# 13x13 Hand Grid Matrix Structure
# =========================================================================

GRID_RANKS: Final[tuple[str, ...]] = (
    "A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"
)
"""The 13 rank characters in standard poker matrix order (Ace to Deuce)."""

NUM_GRID_CELLS: Final[int] = 169
"""The total number of cells in the 13x13 poker hand matrix (13 pairs + 78 suited + 78 offsuit)."""


def _build_hand_grid_cells() -> tuple[str, ...]:
    """Build the canonical 169-cell list in row-major order."""
    cells: list[str] = []
    for i, r1 in enumerate(GRID_RANKS):
        for j, r2 in enumerate(GRID_RANKS):
            if i == j:
                cells.append(f"{r1}{r2}")
            elif i < j:
                cells.append(f"{r1}{r2}s")
            else:
                cells.append(f"{r2}{r1}o")
    return tuple(cells)


HAND_GRID_CELLS: Final[tuple[str, ...]] = _build_hand_grid_cells()
"""Canonical tuple of all 169 hand matrix cells in row-major grid order."""


def _build_cell_combos() -> dict[str, tuple[Combo, ...]]:
    """Precompute the canonical combos for each of the 169 matrix cells."""
    result: dict[str, tuple[Combo, ...]] = {}
    for cell in HAND_GRID_CELLS:
        if len(cell) == 2:
            r = Rank.from_char(cell[0])
            result[cell] = tuple(sorted(generate_pair_combos(r), reverse=True))
        elif cell.endswith("s"):
            r1 = Rank.from_char(cell[0])
            r2 = Rank.from_char(cell[1])
            result[cell] = tuple(sorted(generate_suited_combos(r1, r2), reverse=True))
        elif cell.endswith("o"):
            r1 = Rank.from_char(cell[0])
            r2 = Rank.from_char(cell[1])
            result[cell] = tuple(sorted(generate_offsuit_combos(r1, r2), reverse=True))
    return result


CELL_COMBOS: Final[dict[str, tuple[Combo, ...]]] = _build_cell_combos()
"""Mapping from each 169 hand cell to its canonical tuple of combos."""


def combo_to_cell(combo: Combo) -> str:
    """Return the 13x13 grid cell label for a canonical combo.

    Parameters
    ----------
    combo : Combo
        A 2-card combo.

    Returns
    -------
    str
        Matrix cell string (e.g. ``"AA"``, ``"AKs"``, ``"AKo"``).

    Examples
    --------
    >>> from poker_equity.card import Card
    >>> from poker_equity.range import canonical_combo, combo_to_cell
    >>> combo_to_cell(canonical_combo(Card.from_str("Ah"), Card.from_str("Kh")))
    'AKs'
    >>> combo_to_cell(canonical_combo(Card.from_str("Ah"), Card.from_str("Kd")))
    'AKo'
    >>> combo_to_cell(canonical_combo(Card.from_str("Ah"), Card.from_str("As")))
    'AA'
    """
    c1, c2 = combo
    if c1.rank == c2.rank:
        return f"{c1.rank.char}{c2.rank.char}"
    high, low = (c1, c2) if c1.rank > c2.rank else (c2, c1)
    suffix = "s" if c1.suit == c2.suit else "o"
    return f"{high.rank.char}{low.rank.char}{suffix}"


def _normalize_cell_name(cell: str) -> str:
    """Normalize a hand cell string (e.g. 'aks' -> 'AKs', 'KAs' -> 'AKs', 'aa' -> 'AA')."""
    stripped = cell.strip()
    if len(stripped) == 2:
        return stripped.upper()
    if len(stripped) == 3:
        try:
            r1 = Rank.from_char(stripped[0])
            r2 = Rank.from_char(stripped[1])
            high, low = (r1, r2) if r1 >= r2 else (r2, r1)
            return f"{high.char}{low.char}{stripped[2].lower()}"
        except ValueError:
            return stripped
    return stripped


def _coerce_dead_cards(
    dead_cards: Iterable[Card | str] | Card | str | None,
) -> frozenset[Card]:
    """Convert various dead card inputs into an immutable frozenset of Card objects."""
    if dead_cards is None:
        return frozenset()
    if isinstance(dead_cards, Card):
        return frozenset({dead_cards})
    if isinstance(dead_cards, str):
        stripped = dead_cards.strip()
        if not stripped:
            return frozenset()
        if len(stripped) == 2:
            return frozenset({Card.from_str(stripped)})
        return frozenset(parse_cards(stripped))

    cards: set[Card] = set()
    for item in dead_cards:
        if isinstance(item, Card):
            cards.add(item)
        elif isinstance(item, str):
            stripped = item.strip()
            if not stripped:
                continue
            if len(stripped) == 2:
                cards.add(Card.from_str(stripped))
            else:
                cards.update(parse_cards(stripped))
        else:
            msg = f"Expected Card or card string, got: {type(item).__name__}"
            raise TypeError(msg)
    return frozenset(cards)


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

    __slots__ = (
        "_combos",
        "_combos_list",
        "_weights",
        "_weights_list",
        "_is_weighted",
    )

    def __init__(
        self,
        combos: (
            Iterable[tuple[Card, Card] | Sequence[Card]]
            | Mapping[Combo, float]
            | None
        ) = None,
        weights: Mapping[Combo, float] | Sequence[float] | None = None,
    ) -> None:
        """Initialize a Range with an optional collection of combos and weights.

        Parameters
        ----------
        combos : Iterable[tuple[Card, Card] | Sequence[Card]] | Mapping[Combo, float] | None
            A collection of 2-card combos, a mapping from combos to weights in (0.0, 1.0],
            or ``None`` to create an empty range.
        weights : Mapping[Combo, float] | Sequence[float] | None, optional
            Explicit weights for combos if not provided via a mapping.

        Raises
        ------
        ValueError
            If any combo does not contain exactly two cards, contains identical cards,
            or if any weight is outside [0.0, 1.0].
        TypeError
            If any combo elements are not :class:`Card` instances.
        """
        if combos is None:
            self._combos = frozenset()
            self._weights = {}
            self._is_weighted = False
        elif isinstance(combos, Range):
            self._combos = combos._combos
            self._weights = dict(combos._weights)
            self._is_weighted = combos._is_weighted
            if weights is not None:
                self._apply_weights(weights)
        elif isinstance(combos, collections.abc.Mapping):
            validated_weights: dict[Combo, float] = {}
            validated_combos: set[Combo] = set()
            for item, wt in combos.items():
                c = self._validate_combo(item)
                w = float(wt)
                if not (0.0 <= w <= 1.0):
                    raise ValueError(f"Weight must be in [0.0, 1.0], got {w} for combo {c}")
                if w > 0.0:
                    validated_combos.add(c)
                    validated_weights[c] = w
            self._combos = frozenset(validated_combos)
            self._weights = validated_weights
            self._is_weighted = any(w < 1.0 for w in self._weights.values())
        else:
            validated: set[Combo] = set()
            for item in combos:
                validated.add(self._validate_combo(item))
            self._combos = frozenset(validated)
            self._weights = {c: 1.0 for c in validated}
            self._is_weighted = False
            if weights is not None:
                self._apply_weights(weights)

        self._combos_list: tuple[Combo, ...] | None = None
        self._weights_list: tuple[float, ...] | None = None

    @staticmethod
    def _validate_combo(item: object) -> Combo:
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
        return canonical_combo(c1, c2)

    def _apply_weights(
        self, weights: Mapping[Combo, float] | Sequence[float]
    ) -> None:
        if isinstance(weights, collections.abc.Mapping):
            new_weights: dict[Combo, float] = {}
            new_combos: set[Combo] = set()
            for c in self._combos:
                w = float(weights.get(c, self._weights.get(c, 1.0)))
                if not (0.0 <= w <= 1.0):
                    raise ValueError(f"Weight must be in [0.0, 1.0], got {w}")
                if w > 0.0:
                    new_combos.add(c)
                    new_weights[c] = w
            self._combos = frozenset(new_combos)
            self._weights = new_weights
        elif isinstance(weights, collections.abc.Sequence):
            combos_ordered = sorted(self._combos, reverse=True)
            if len(weights) != len(combos_ordered):
                raise ValueError(
                    f"Length of weights sequence ({len(weights)}) does not match "
                    f"number of combos ({len(combos_ordered)})"
                )
            new_weights = {}
            new_combos = set()
            for c, wt in zip(combos_ordered, weights):
                w = float(wt)
                if not (0.0 <= w <= 1.0):
                    raise ValueError(f"Weight must be in [0.0, 1.0], got {w}")
                if w > 0.0:
                    new_combos.add(c)
                    new_weights[c] = w
            self._combos = frozenset(new_combos)
            self._weights = new_weights
        else:
            raise TypeError(f"weights must be a Mapping or Sequence, got {type(weights).__name__}")
        self._is_weighted = any(w < 1.0 for w in self._weights.values())
        self._combos_list = None
        self._weights_list = None

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
    def weights(self) -> dict[Combo, float]:
        """Mapping of each combo in the range to its frequency weight in (0.0, 1.0]."""
        return dict(self._weights)

    @property
    def is_weighted(self) -> bool:
        """True if any combo in this range has a frequency weight < 1.0."""
        return self._is_weighted

    @property
    def total_combos(self) -> float:
        """The total effective number of combos, summing all combo weights.

        For unweighted ranges, this equals ``len(self)``.
        For weighted ranges, this is the sum of combo frequency weights.
        """
        return sum(self._weights.values())

    @property
    def weights_list(self) -> tuple[float, ...]:
        """Tuple of weights corresponding to combos in combos_list."""
        if self._weights_list is None:
            self._weights_list = tuple(self._weights[c] for c in self.combos_list)
        return self._weights_list

    def get_weight(self, combo: Combo | str) -> float:
        """Return the frequency weight of a combo in this range.

        Parameters
        ----------
        combo : Combo | str
            A canonical combo tuple or 4-character string (e.g. ``"AhKd"``).

        Returns
        -------
        float
            Weight in [0.0, 1.0], or 0.0 if the combo is not in the range.
        """
        if isinstance(combo, str):
            try:
                c = combo_from_str(combo)
            except (ValueError, TypeError):
                return 0.0
        else:
            c = combo
        return self._weights.get(c, 0.0)

    @property
    def is_empty(self) -> bool:
        """True if the range contains zero combos."""
        return len(self._combos) == 0

    # -- Factory Constructors ---------------------------------------------

    @classmethod
    def from_str(cls, range_str: str) -> Range:
        """Parse a standard poker range string into a Range instance.

        Supports pocket pairs (``"QQ+"``, ``"88-66"``), suited hands (``"AKs"``,
        ``"T9s-76s"``, ``"A2s+"``), offsuit hands (``"AKo"``, ``"AQo+"``),
        unpaired hands (``"AK"``, ``"AQ+"``), specific combos (``"AhKd"``),
        weighted tokens (``"AKs:0.5"``, ``"QQ+:0.75"``, ``"AKo:50%"``),
        and special keywords (``"random"``, ``"100%"``).

        Parameters
        ----------
        range_str : str
            A poker range string, e.g. ``"AKs:0.5, QQ+, T9s-76s"`` or ``"AA, KK, AK"``.

        Returns
        -------
        Range
            A new Range containing all parsed combos with their weights.

        Raises
        ------
        ValueError
            If *range_str* contains invalid tokens or syntax.

        Examples
        --------
        >>> r = Range.from_str("QQ+, AKs, T9s-76s")
        >>> len(r)
        38
        """
        weighted_combos = parse_weighted_range_string(range_str)
        return cls(weighted_combos)

    @classmethod
    def from_weighted_matrix(
        cls,
        matrix: Mapping[str, float],
    ) -> Range:
        """Construct a weighted Range from a 13x13 grid mapping of hand weights.

        Unlike :meth:`from_matrix` (which selects a discrete integer count of
        combos), ``from_weighted_matrix`` assigns the continuous float weight
        directly to every combo in that hand cell.

        Parameters
        ----------
        matrix : Mapping[str, float]
            Mapping of hand cell labels (e.g. ``"AA"``, ``"AKs"``, ``"AKo"``)
            to weights in ``[0.0, 1.0]``.

        Returns
        -------
        Range
            A weighted Range instance.

        Raises
        ------
        ValueError
            If any cell name is invalid or weight is out of range.
        """
        combo_weights: dict[Combo, float] = {}
        for cell_raw, weight in matrix.items():
            if weight <= 0.0:
                continue
            if weight > 1.0:
                if weight <= 100.0:
                    weight = weight / 100.0
                else:
                    raise ValueError(f"Weight must be in [0.0, 1.0], got {weight}")

            normalized = _normalize_cell_name(cell_raw)

            # Support unsuited shorthand (e.g. "AK": 0.5 -> AKs + AKo)
            if len(normalized) == 2 and normalized[0] != normalized[1]:
                r1 = Rank.from_char(normalized[0])
                r2 = Rank.from_char(normalized[1])
                high, low = (r1, r2) if r1 >= r2 else (r2, r1)
                for c in CELL_COMBOS[f"{high.char}{low.char}s"]:
                    combo_weights[c] = float(weight)
                for c in CELL_COMBOS[f"{high.char}{low.char}o"]:
                    combo_weights[c] = float(weight)
                continue

            if normalized not in CELL_COMBOS:
                msg = f"Invalid hand matrix cell: {cell_raw!r}"
                raise ValueError(msg)

            for c in CELL_COMBOS[normalized]:
                combo_weights[c] = float(weight)

        return cls(combo_weights)

    @classmethod
    def from_matrix(
        cls,
        matrix: Mapping[str, float],
        threshold: float | None = None,
    ) -> Range:
        """Construct a Range from a 13x13 grid mapping of hand weights.

        Parameters
        ----------
        matrix : Mapping[str, float]
            Mapping of hand cell labels (e.g. ``"AA"``, ``"AKs"``, ``"AKo"``)
            to weights in ``[0.0, 1.0]``.
        threshold : float | None, optional
            If specified, any cell with ``weight >= threshold`` includes 100% of its
            combos, and cells below the threshold are omitted.
            If ``None`` (the default), the number of combos selected for each cell
            is proportional to the weight: ``round(weight * total_combos)``.

        Returns
        -------
        Range
            The constructed Range instance.

        Raises
        ------
        ValueError
            If any cell name in *matrix* is not a valid hand cell.

        Examples
        --------
        >>> r = Range.from_matrix({"AA": 1.0, "AKs": 1.0})
        >>> len(r)
        10
        """
        combos: set[Combo] = set()
        for cell_raw, weight in matrix.items():
            if weight <= 0.0:
                continue

            normalized = _normalize_cell_name(cell_raw)

            # Support unsuited shorthand (e.g. "AK": 1.0 -> AKs + AKo)
            if len(normalized) == 2 and normalized[0] != normalized[1]:
                r1 = Rank.from_char(normalized[0])
                r2 = Rank.from_char(normalized[1])
                high, low = (r1, r2) if r1 >= r2 else (r2, r1)
                suited_key = f"{high.char}{low.char}s"
                offsuit_key = f"{high.char}{low.char}o"
                sub_dict = {suited_key: weight, offsuit_key: weight}
                sub_range = cls.from_matrix(sub_dict, threshold=threshold)
                combos.update(sub_range.combos)
                continue

            if normalized not in CELL_COMBOS:
                msg = f"Invalid hand matrix cell: {cell_raw!r}"
                raise ValueError(msg)

            cell_all = CELL_COMBOS[normalized]
            total_possible = len(cell_all)

            if threshold is not None:
                if weight >= threshold:
                    combos.update(cell_all)
            else:
                num_to_take = min(total_possible, max(0, round(weight * total_possible)))
                combos.update(cell_all[:num_to_take])

        return cls(combos)

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
        merged = dict(self._weights)
        for other in others:
            if not isinstance(other, Range):
                msg = f"Expected Range instance, got {type(other).__name__}"
                raise TypeError(msg)
            for c, w in other._weights.items():
                if c in merged:
                    merged[c] = max(merged[c], w)
                else:
                    merged[c] = w
        return Range(merged)

    def intersection(self, *others: Range) -> Range:
        """Return a new Range with combos present in this range and all of *others*.

        If a combo exists in multiple ranges, the minimum weight is retained.

        Parameters
        ----------
        *others : Range
            One or more Range instances to intersect with.

        Returns
        -------
        Range
            The intersection Range.
        """
        res = dict(self._weights)
        for other in others:
            if not isinstance(other, Range):
                msg = f"Expected Range instance, got {type(other).__name__}"
                raise TypeError(msg)
            res = {
                c: min(res[c], other._weights[c])
                for c in res
                if c in other._weights
            }
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
        res = dict(self._weights)
        for other in others:
            if not isinstance(other, Range):
                msg = f"Expected Range instance, got {type(other).__name__}"
                raise TypeError(msg)
            for c in other._combos:
                res.pop(c, None)
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
        new_weights: dict[Combo, float] = {}
        for c, w in self._weights.items():
            if c not in other._combos:
                new_weights[c] = w
        for c, w in other._weights.items():
            if c not in self._combos:
                new_weights[c] = w
        return Range(new_weights)

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
        return self._combos == other._combos and self._weights == other._weights

    def __hash__(self) -> int:
        if not self._is_weighted:
            return hash(self._combos)
        return hash((self._combos, tuple(sorted(self._weights.items()))))

    def __repr__(self) -> str:
        count = len(self._combos)
        combo_word = "combo" if count == 1 else "combos"
        if self._is_weighted:
            return f"Range({count} {combo_word}, {self.total_combos:.2f} weighted)"
        return f"Range({count} {combo_word})"

    def __str__(self) -> str:
        count = len(self._combos)
        combo_word = "combo" if count == 1 else "combos"
        if self._is_weighted:
            return f"Range({count} {combo_word}, {self.total_combos:.2f} weighted)"
        return f"Range({count} {combo_word})"

    # -- Sampling & Card Queries ------------------------------------------

    def sample(self, rng: random.Random | None = None) -> Combo:
        """Randomly select one combo from the range.

        If the range is weighted, combos are sampled with probability proportional
        to their weights.  If unweighted, combos are sampled uniformly.

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
        if not self._is_weighted:
            if rng is not None:
                return rng.choice(self.combos_list)
            return random.choice(self.combos_list)

        if rng is not None:
            return rng.choices(self.combos_list, weights=self.weights_list, k=1)[0]
        return random.choices(self.combos_list, weights=self.weights_list, k=1)[0]

    def contains_card(self, card: Card | str) -> bool:
        """Check if any combo in the range contains the specified card.

        Parameters
        ----------
        card : Card | str
            The card to query (e.g. ``Card.from_str("Ah")`` or ``"Ah"``).

        Returns
        -------
        bool
            True if *card* appears in at least one combo in the range.
        """
        c = Card.from_str(card) if isinstance(card, str) else card
        return any(c == c1 or c == c2 for c1, c2 in self._combos)

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

    def remove_dead_cards(
        self,
        dead_cards: Iterable[Card | str] | Card | str | None,
    ) -> Range:
        """Filter out combos that contain any of the specified dead cards.

        In poker simulation and equity analysis, cards that are already known
        (such as hero's hole cards or community cards on the board) cannot
        appear in villain's range.  This method returns a new :class:`Range`
        containing only the combos that remain physically possible, preserving
        their frequency weights.

        Parameters
        ----------
        dead_cards : Iterable[Card | str] | Card | str | None
            Cards that are unavailable.  Can be an iterable of :class:`Card`
            objects, an iterable of 2-character strings (e.g. ``["Ah", "Kd"]``),
            a delimited card string (e.g. ``"Ah Kd Qs"``), a single :class:`Card`,
            or ``None``.

        Returns
        -------
        Range
            A new Range containing only the unblocked combos.

        Examples
        --------
        >>> r = Range.from_str("AA")
        >>> len(r)
        6
        >>> r_filtered = r.remove_dead_cards(["As", "Ah"])
        >>> len(r_filtered)  # only AdAc remains
        1
        """
        dead = _coerce_dead_cards(dead_cards)
        if not dead or not self._combos:
            return self
        live_weights = {
            combo: weight
            for combo, weight in self._weights.items()
            if combo[0] not in dead and combo[1] not in dead
        }
        return Range(live_weights)

    def blocked_combos(
        self,
        dead_cards: Iterable[Card | str] | Card | str | None,
    ) -> set[Combo]:
        """Return the set of combos in this range that are blocked by dead cards.

        Parameters
        ----------
        dead_cards : Iterable[Card | str] | Card | str | None
            Dead cards to check.

        Returns
        -------
        set[Combo]
            Set of combos containing at least one dead card.
        """
        dead = _coerce_dead_cards(dead_cards)
        if not dead:
            return set()
        return {
            combo for combo in self._combos
            if combo[0] in dead or combo[1] in dead
        }

    def is_blocked(
        self,
        dead_cards: Iterable[Card | str] | Card | str | None,
    ) -> bool:
        """Return True if at least one combo in this range is blocked by dead cards.

        Parameters
        ----------
        dead_cards : Iterable[Card | str] | Card | str | None
            Dead cards to check.

        Returns
        -------
        bool
            True if any dead card appears in this range.
        """
        dead = _coerce_dead_cards(dead_cards)
        if not dead:
            return False
        return any(combo[0] in dead or combo[1] in dead for combo in self._combos)

    # -- Matrix Representation (13x13 Grid) --------------------------------

    def to_matrix(self, include_zeros: bool = True) -> dict[str, float]:
        """Convert this Range into a 13x13 grid dictionary of hand weights.

        Each key is a standard poker hand label (e.g. ``"AA"``, ``"AKs"``, ``"AKo"``),
        and each value is a float in ``[0.0, 1.0]`` representing the fraction of
        possible combos for that hand that are included in this range, accounting
        for partial combo weights.

        Parameters
        ----------
        include_zeros : bool, default=True
            If True, all 169 matrix cells are included (absent hands have weight 0.0).
            If False, only cells with weight > 0.0 are returned.

        Returns
        -------
        dict[str, float]
            Dictionary mapping hand labels to weights in [0.0, 1.0].

        Examples
        --------
        >>> r = Range.from_str("AA, AKs")
        >>> m = r.to_matrix(include_zeros=False)
        >>> m["AA"]
        1.0
        >>> m["AKs"]
        1.0
        """
        weight_sums: dict[str, float] = collections.defaultdict(float)
        for combo, w in self._weights.items():
            weight_sums[combo_to_cell(combo)] += w

        matrix: dict[str, float] = {}
        for cell in HAND_GRID_CELLS:
            total_possible = len(CELL_COMBOS[cell])
            w_sum = weight_sums.get(cell, 0.0)
            if w_sum > 0.0:
                matrix[cell] = w_sum / total_possible
            elif include_zeros:
                matrix[cell] = 0.0

        return matrix

    def to_grid_2d(self) -> list[list[float]]:
        """Return a 13x13 nested list of weights for UI / matrix visualization.

        Rows and columns correspond to ranks Ace through 2 in descending order.
        The main diagonal contains pocket pairs, the upper-right triangle
        contains suited hands, and the lower-left triangle contains offsuit hands.

        Returns
        -------
        list[list[float]]
            A 13x13 list of floats in [0.0, 1.0].
        """
        matrix = self.to_matrix(include_zeros=True)
        grid: list[list[float]] = []
        idx = 0
        for _ in range(13):
            row: list[float] = []
            for _ in range(13):
                cell = HAND_GRID_CELLS[idx]
                row.append(matrix[cell])
                idx += 1
            grid.append(row)
        return grid

