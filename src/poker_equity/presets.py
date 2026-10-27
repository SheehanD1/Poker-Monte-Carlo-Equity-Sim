"""Predefined poker hand ranges and position-based opening/defense charts.

This module provides standard, GTO-aligned opening and defense ranges for
both 6-max and 9-max Texas Hold'em games across various table positions
(UTG, HJ/MP, CO, BTN, SB, BB).  It also includes short-stack push/fold charts,
standard hand archetypes (pocket pairs, suited connectors), and a canonical
169-hand percentile ranking engine (:func:`top_percent_range`).
"""

from __future__ import annotations

import enum
import functools
import re
from typing import Final

from poker_equity.range import CELL_COMBOS, Combo, Range


# =========================================================================
# Position Enum
# =========================================================================


class Position(str, enum.Enum):
    """Standard poker table positions for 6-max and 9-max games."""

    UTG = "UTG"
    UTG_1 = "UTG+1"
    UTG_2 = "UTG+2"
    LJ = "LJ"
    HJ = "HJ"
    CO = "CO"
    BTN = "BTN"
    SB = "SB"
    BB = "BB"

    def __str__(self) -> str:
        return self.value


# =========================================================================
# Canonical 169 Hand Rank Order (PokerStove / ProPokerTools standard)
# =========================================================================

HAND_RANK_ORDER: Final[tuple[str, ...]] = (
    # Tier 1: Premium pairs & premium broadways
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77",
    "AKs", "AQs", "AJs", "KQs", "ATs", "KJs", "QJs", "JTs",
    "AKo", "66", "AQo", "T9s", "A9s", "KQo", "KTs", "QTs",
    "AJo", "55", "98s", "KJo", "A8s", "87s", "K9s", "Q9s",
    "J9s", "ATo", "44", "76s", "A5s", "A7s", "T8s", "QJo",
    "65s", "K8s", "97s", "33", "A4s", "A6s", "Q8s", "J8s",
    "86s", "KTo", "75s", "54s", "A3s", "22", "K7s", "A2s",
    "QTo", "64s", "96s", "JTo", "85s", "Q7s", "T7s", "J7s",
    "74s", "53s", "K6s", "63s", "K5s", "43s", "84s", "Q6s",
    "T6s", "95s", "K4s", "73s", "J6s", "52s", "62s", "Q5s",
    "42s", "K3s", "83s", "32s", "94s", "K2s", "72s", "J5s",
    "Q4s", "82s", "J4s", "Q3s", "93s", "J3s", "Q2s", "T5s",
    "T4s", "92s", "J2s", "T3s", "T2s",
    # Offsuit hands
    "K9o", "Q9o", "J9o", "T9o", "A9o", "A8o", "K8o", "98o",
    "T8o", "Q8o", "J8o", "A7o", "87o", "K7o", "97o", "76o",
    "Q7o", "T7o", "J7o", "A6o", "86o", "K6o", "65o", "75o",
    "96o", "A5o", "Q6o", "85o", "J6o", "54o", "T6o", "K5o",
    "64o", "A4o", "74o", "Q5o", "95o", "84o", "K4o", "J5o",
    "53o", "A3o", "63o", "T5o", "73o", "Q4o", "43o", "83o",
    "K3o", "94o", "J4o", "A2o", "52o", "62o", "T4o", "Q3o",
    "72o", "82o", "42o", "K2o", "93o", "J3o", "T3o", "32o",
    "Q2o", "92o", "J2o", "T2o",
)
"""Canonical 169-hand rank order from strongest (#1 AA) to weakest (#169 72o)."""


# =========================================================================
# Standard Preset Range String Definitions
# =========================================================================

PRESET_RANGES: Final[dict[str, str]] = {
    # 6-Max Open-Raise (RFI - 100bb Standard)
    "UTG_OPEN": "77+, ATs+, KTs+, QTs+, JTs, T9s, 98s, 87s, AJo+, KQo",
    "HJ_OPEN": "55+, A8s+, K9s+, Q9s+, J9s+, T9s, 98s, 87s, 76s, ATo+, KJo+, QJo",
    "CO_OPEN": "22+, A2s+, K6s+, Q8s+, J8s+, T8s+, 97s+, 86s+, 75s+, 65s, 54s, A9o+, KTo+, QTo+, JTo",
    "BTN_OPEN": "22+, A2s+, K2s+, Q3s+, J5s+, T6s+, 96s+, 85s+, 74s+, 64s+, 53s+, 43s, A2o+, K7o+, Q8o+, J8o+, T8o+, 97o+, 87o, 76o",
    "SB_OPEN": "22+, A2s+, K2s+, Q4s+, J5s+, T6s+, 96s+, 85s+, 75s+, 64s+, 54s, A2o+, K7o+, Q8o+, J8o+, T8o+, 98o, 87o",

    # 9-Max Full Ring Open-Raise (RFI)
    "UTG_9MAX": "88+, ATs+, KJs+, QJs, JTs, AQo+",
    "UTG1_9MAX": "77+, ATs+, KTs+, QJs, JTs, T9s, AJo+, KQo",
    "UTG2_9MAX": "66+, A9s+, KTs+, QTs+, JTs, T9s, 98s, AJo+, KQo",
    "LJ_9MAX": "66+, A9s+, KTs+, QTs+, JTs, T9s, 98s, 87s, ATo+, KJo+, QJo",

    # 3-Bet Ranges
    "BTN_VS_CO_3BET": "TT+, AJs+, KQs, A5s-A4s, AKo",
    "BB_VS_BTN_3BET": "88+, ATs+, KTs+, QJs, A2s-A5s, K9s, Q9s, J9s, T8s, 97s, 86s, 75s, 64s, AJo+, KQo",
    "SB_VS_BTN_3BET": "88+, ATs+, KTs+, QTs+, JTs, A2s-A5s, K9s, AQo+",
    "BB_VS_UTG_3BET": "JJ+, AQs+, AKo",

    # Defense & Calling Ranges
    "BB_VS_BTN_DEFENSE": "22+, A2s+, K2s+, Q2s+, J4s+, T6s+, 96s+, 85s+, 74s+, 63s+, 53s+, 43s, A2o+, K7o+, Q8o+, J8o+, T8o+, 97o+, 87o, 76o",
    "BB_VS_BTN_CALL": "22-77, A2s-A9s, K2s-K9s, Q2s-QTs, J4s-J9s, T6s-T9s, 96s-98s, 85s-87s, 74s-76s, 63s-65s, 53s-54s, 43s, A2o-ATo, K7o-KJo, Q8o-QJo, J8o-JTo, T8o-T9o, 97o+, 87o",
    "BB_VS_UTG_CALL": "22-TT, AJs-ATs, KTs+, QTs+, JTs, T9s, 98s, 87s, AJo, KQo",

    # Push/Fold Short-Stack (Tournament)
    "PUSH_10BB_BTN": "22+, A2+, K2+, Q4o+, Q2s+, J7o+, J4s+, T7o+, T6s+, 97o+, 96s+, 86s+, 75s+, 65s, 54s",
    "PUSH_15BB_BTN": "22+, A2+, K5o+, K2s+, Q8o+, Q5s+, J8o+, J7s+, T8o+, T7s+, 98o, 97s+, 87s, 76s, 65s",
    "CALL_VS_10BB_SHOVE": "55+, A8o+, A2s+, KTo+, K7s+, QTo+, Q9s+, JTs",

    # General Archetypes
    "TIGHT_OPEN": "88+, ATs+, KJs+, QJs, AQo+",
    "STANDARD_OPEN": "55+, A8s+, K9s+, Q9s+, J9s+, T9s, 98s, 87s, 76s, ATo+, KJo+, QJo",
    "LOOSE_OPEN": "22+, A2s+, K2s+, Q3s+, J5s+, T6s+, 96s+, 85s+, 74s+, 64s+, 53s+, 43s, A2o+, K7o+, Q8o+, J8o+, T8o+, 97o+, 87o, 76o",
    "PREMIUM": "TT+, AJs+, KQs, AKo",
    "BROADWAYS": "ATs+, KTs+, QTs+, JTs, ATo+, KTo+, QTo+, JTo",
    "POCKET_PAIRS": "22+",
    "SUITED_CONNECTORS": "AKs, KQs, QJs, JTs, T9s, 98s, 87s, 76s, 65s, 54s, 43s, 32s",
    "SUITED_GAPPERS": "AQs, KJs, QTs, J9s, T8s, 97s, 86s, 75s, 64s, 53s, 42s",
    "SUITED_ACES": "A2s+",
}
"""Dictionary of predefined poker range strings keyed by uppercase canonical identifier."""


PRESET_DESCRIPTIONS: Final[dict[str, str]] = {
    "UTG_OPEN": "6-Max UTG Open-Raise (~11.2%, 148 combos)",
    "HJ_OPEN": "6-Max Hijack / MP Open-Raise (~16.6%, 220 combos)",
    "CO_OPEN": "6-Max Cutoff Open-Raise (~26.7%, 354 combos)",
    "BTN_OPEN": "6-Max Button Open-Raise (~50.8%, 674 combos)",
    "SB_OPEN": "6-Max Small Blind Open-Raise (~47.8%, 634 combos)",
    "UTG_9MAX": "9-Max UTG Open-Raise (~7.4%, 98 combos)",
    "UTG1_9MAX": "9-Max UTG+1 Open-Raise (~10.3%, 136 combos)",
    "UTG2_9MAX": "9-Max UTG+2 Open-Raise (~11.6%, 154 combos)",
    "LJ_9MAX": "9-Max Lojack Open-Raise (~15.0%, 198 combos)",
    "BTN_VS_CO_3BET": "Button vs. Cutoff 3-Bet (~5.0%, 66 combos)",
    "BB_VS_BTN_3BET": "Big Blind vs. Button 3-Bet (~12.8%, 170 combos)",
    "SB_VS_BTN_3BET": "Small Blind vs. Button 3-Bet (~9.5%, 126 combos)",
    "BB_VS_UTG_3BET": "Big Blind vs. UTG 3-Bet (~3.3%, 44 combos)",
    "BB_VS_BTN_DEFENSE": "Big Blind vs. Button Total Defense (Call + 3-Bet, ~52%)",
    "BB_VS_BTN_CALL": "Big Blind vs. Button Flat Call Range",
    "BB_VS_UTG_CALL": "Big Blind vs. UTG Flat Call Range",
    "PUSH_10BB_BTN": "Button 10bb Push/Fold All-In Range (~58.1%, 770 combos)",
    "PUSH_15BB_BTN": "Button 15bb Push/Fold All-In Range (~45.7%, 606 combos)",
    "CALL_VS_10BB_SHOVE": "Calling Range vs. 10bb Button All-In (~21.1%, 280 combos)",
    "TIGHT_OPEN": "Tight / Rock Open-Raise (~7.1%, 94 combos)",
    "STANDARD_OPEN": "Standard Open-Raise (~16.6%, 220 combos)",
    "LOOSE_OPEN": "Loose Open-Raise (~50.8%, 674 combos)",
    "PREMIUM": "Premium Starting Hands (TT+, AJs+, KQs, AKo, ~4.4%)",
    "BROADWAYS": "Broadway Hands (All Ten-to-Ace holdings, ~12.1%)",
    "POCKET_PAIRS": "All Pocket Pairs (22 through AA, 78 combos)",
    "SUITED_CONNECTORS": "All Suited Connectors (AKs down to 32s, 48 combos)",
    "SUITED_GAPPERS": "All One-Gap Suited Hands (AQs down to 42s, 44 combos)",
    "SUITED_ACES": "All Suited Aces (A2s through AKs, 48 combos)",
}
"""Human-readable descriptions for each predefined range."""


# =========================================================================
# Alias Map
# =========================================================================

_ALIASES: Final[dict[str, str]] = {
    # Position shorthands
    "UTG": "UTG_OPEN",
    "UTG1": "UTG1_9MAX",
    "UTG_1": "UTG1_9MAX",
    "UTG+1": "UTG1_9MAX",
    "UTG2": "UTG2_9MAX",
    "UTG_2": "UTG2_9MAX",
    "UTG+2": "UTG2_9MAX",
    "LJ": "LJ_9MAX",
    "LOJACK": "LJ_9MAX",
    "MP": "HJ_OPEN",
    "HJ": "HJ_OPEN",
    "HIJACK": "HJ_OPEN",
    "MIDDLE_POSITION": "HJ_OPEN",
    "CO": "CO_OPEN",
    "CUTOFF": "CO_OPEN",
    "BTN": "BTN_OPEN",
    "BUTTON": "BTN_OPEN",
    "BU": "BTN_OPEN",
    "SB": "SB_OPEN",
    "SMALL_BLIND": "SB_OPEN",
    "BB": "BB_VS_BTN_DEFENSE",
    "BIG_BLIND": "BB_VS_BTN_DEFENSE",

    # 3-bets
    "3BET_BTN": "BTN_VS_CO_3BET",
    "3BET_BB": "BB_VS_BTN_3BET",
    "3BET_SB": "SB_VS_BTN_3BET",

    # Push/Fold
    "PUSH10": "PUSH_10BB_BTN",
    "PUSH_10": "PUSH_10BB_BTN",
    "PUSH15": "PUSH_15BB_BTN",
    "PUSH_15": "PUSH_15BB_BTN",
    "CALL10": "CALL_VS_10BB_SHOVE",
    "CALL_10": "CALL_VS_10BB_SHOVE",

    # Archetypes
    "TIGHT": "TIGHT_OPEN",
    "STANDARD": "STANDARD_OPEN",
    "LOOSE": "LOOSE_OPEN",
    "PAIRS": "POCKET_PAIRS",
    "CONNECTORS": "SUITED_CONNECTORS",
    "GAPPERS": "SUITED_GAPPERS",
    "ACES": "SUITED_ACES",
}


# =========================================================================
# Preset Lookup & Percentile Generation
# =========================================================================


def _normalize_preset_key(name: str | Position) -> str:
    """Normalize input name or Position enum to an uppercase lookup key."""
    if isinstance(name, Position):
        raw = name.value
    else:
        raw = str(name).strip()
    # Replace spaces and hyphens with underscores, strip punctuation
    clean = re.sub(r"[\s\-]+", "_", raw).upper()
    return clean


def top_percent_range(
    percentage: float,
    *,
    weighted_boundary: bool = False,
) -> Range:
    """Generate a Range representing the top *percentage* of Hold'em hands.

    Uses the canonical 169-hand rank ordering (PokerStove / ProPokerTools standard),
    which orders all hand types from #1 (AA) down to #169 (72o).

    Parameters
    ----------
    percentage : float
        A percentage between 0.0 and 100.0 (e.g. ``10.0`` for top 10%).
    weighted_boundary : bool, default=False
        If True, the marginal hand cell on the boundary is assigned a fractional
        frequency weight so that the total weighted combos match the exact percentage.
        If False (default), integer combo counts are selected.

    Returns
    -------
    Range
        A Range instance representing the top starting hands.

    Raises
    ------
    ValueError
        If *percentage* is not between 0.0 and 100.0.

    Examples
    --------
    >>> r10 = top_percent_range(10.0)
    >>> len(r10)
    133
    >>> (Card.from_str("Ah"), Card.from_str("As")) in r10
    True
    """
    if not (0.0 <= percentage <= 100.0):
        msg = f"Percentage must be between 0.0 and 100.0, got: {percentage}"
        raise ValueError(msg)

    if percentage <= 0.0:
        return Range.empty()
    if percentage >= 100.0:
        return Range.all()

    target_combos = (percentage / 100.0) * 1326.0

    if weighted_boundary:
        weights: dict[Combo, float] = {}
        accumulated = 0.0
        for cell in HAND_RANK_ORDER:
            combos = CELL_COMBOS[cell]
            cell_size = len(combos)
            if accumulated + cell_size <= target_combos:
                for c in combos:
                    weights[c] = 1.0
                accumulated += cell_size
            else:
                remaining = target_combos - accumulated
                fraction = max(0.0, min(1.0, remaining / cell_size))
                if fraction > 0.0:
                    for c in combos:
                        weights[c] = fraction
                break
        return Range(weights)

    combos_set: set[Combo] = set()
    accumulated_count = 0
    for cell in HAND_RANK_ORDER:
        cell_combos = CELL_COMBOS[cell]
        cell_size = len(cell_combos)
        if accumulated_count + cell_size <= target_combos:
            combos_set.update(cell_combos)
            accumulated_count += cell_size
        else:
            remaining = round(target_combos - accumulated_count)
            combos_set.update(cell_combos[:remaining])
            break

    return Range(combos_set)


@functools.lru_cache(maxsize=128)
def get_preset_range(name: str | Position) -> Range:
    """Retrieve a pre-built Range for a named preset or position.

    Supports exact preset names (e.g. ``"UTG_OPEN"``), position enums
    (:class:`Position`), position aliases (``"UTG"``, ``"CO"``, ``"BTN"``),
    and percentile shorthands (``"top10"``, ``"top 20%"``).

    Parameters
    ----------
    name : str | Position
        The preset name, position, or alias.

    Returns
    -------
    Range
        A cached Range instance corresponding to the preset.

    Raises
    ------
    KeyError
        If *name* does not match any known preset or alias.

    Examples
    --------
    >>> r = get_preset_range("BTN")
    >>> len(r) > 500
    True
    >>> r_utg = get_preset_range(Position.UTG)
    >>> len(r_utg) == 148
    True
    """
    key = _normalize_preset_key(name)

    # Check for percentile pattern: e.g. "TOP10", "TOP_20", "TOP_15%"
    pct_match = re.match(r"^TOP_?(\d+(?:\.\d+)?)(?:%)?$", key)
    if pct_match:
        pct = float(pct_match.group(1))
        return top_percent_range(pct)

    # Check canonical presets first
    if key in PRESET_RANGES:
        return Range.from_str(PRESET_RANGES[key])

    # Check aliases
    if key in _ALIASES:
        canonical = _ALIASES[key]
        return Range.from_str(PRESET_RANGES[canonical])

    # Not found — build descriptive error
    available = sorted(set(list(PRESET_RANGES.keys()) + list(_ALIASES.keys())))
    msg = f"Unknown preset range {name!r}. Available presets include: {', '.join(available[:10])}..."
    raise KeyError(msg)


def list_presets() -> list[str]:
    """Return a sorted list of all canonical preset range identifiers.

    Returns
    -------
    list[str]
        List of uppercase preset identifiers (e.g. ``["BB_VS_BTN_3BET", ...]``).
    """
    return sorted(PRESET_RANGES.keys())


def get_preset_description(name: str | Position) -> str:
    """Return the human-readable description for a preset range.

    Parameters
    ----------
    name : str | Position
        Preset name or Position enum.

    Returns
    -------
    str
        Summary description of the range holding and expected combo count.
    """
    key = _normalize_preset_key(name)
    if key in _ALIASES:
        key = _ALIASES[key]
    return PRESET_DESCRIPTIONS.get(key, f"Preset range {name}")


# =========================================================================
# Convenience Percentile Range Constants
# =========================================================================

TOP_5_PERCENT: Final[Range] = top_percent_range(5.0)
"""Top 5% of Hold'em hands (~66 combos: AA-TT, AKs, AQs, AKo)."""

TOP_10_PERCENT: Final[Range] = top_percent_range(10.0)
"""Top 10% of Hold'em hands (~133 combos)."""

TOP_15_PERCENT: Final[Range] = top_percent_range(15.0)
"""Top 15% of Hold'em hands (~199 combos: tight-standard opening range)."""

TOP_20_PERCENT: Final[Range] = top_percent_range(20.0)
"""Top 20% of Hold'em hands (~265 combos: MP/HJ opening range)."""

TOP_25_PERCENT: Final[Range] = top_percent_range(25.0)
"""Top 25% of Hold'em hands (~332 combos: Cutoff opening range)."""

TOP_30_PERCENT: Final[Range] = top_percent_range(30.0)
"""Top 30% of Hold'em hands (~398 combos)."""

TOP_40_PERCENT: Final[Range] = top_percent_range(40.0)
"""Top 40% of Hold'em hands (~530 combos: Button/SB opening range)."""

TOP_50_PERCENT: Final[Range] = top_percent_range(50.0)
"""Top 50% of Hold'em hands (~663 combos: wide defense/heads-up range)."""
