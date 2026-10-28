"""poker-equity: A high-performance Monte Carlo poker equity simulator.

Public API
----------

Card primitives::

    from poker_equity import Card, Rank, Suit, Deck

Hand evaluation::

    from poker_equity import (
        HandRank, HandResult,
        evaluate_5, evaluate_7, evaluate_hand,
        score_5, score_7, score_hand,
    )

Constants & utilities::

    from poker_equity import parse_cards, cards_to_str, validate_no_duplicates
"""

from __future__ import annotations

# -- Card primitives -------------------------------------------------------
from poker_equity.card import Card, Rank, Suit

# -- Deck -------------------------------------------------------------------
from poker_equity.deck import Deck

# -- Hand ranking types -----------------------------------------------------
from poker_equity.hand_rank import HandRank, HandResult

# -- Evaluators (rich HandResult API) ---------------------------------------
from poker_equity.evaluator import (
    evaluate_5,
    evaluate_7,
    evaluate_hand,
    score_5,
    score_7,
    score_hand,
)

# -- Constants & utility functions ------------------------------------------
from poker_equity.constants import (
    FULL_DECK,
    FULL_DECK_LIST,
    HAND_RANKINGS,
    RANK_CHARS,
    RANKS,
    SUIT_CHARS,
    SUITS,
    cards_to_str,
    parse_cards,
    validate_no_duplicates,
)

# -- Range & combos --------------------------------------------------------
from poker_equity.range import (
    GRID_RANKS,
    HAND_GRID_CELLS,
    NUM_GRID_CELLS,
    TOTAL_HOLDEM_COMBOS,
    Combo,
    Range,
    canonical_combo,
    combo_from_str,
    combo_to_cell,
    combo_to_str,
    generate_all_combos,
    generate_offsuit_combos,
    generate_pair_combos,
    generate_suited_combos,
    generate_unpaired_combos,
    parse_range_string,
    parse_weighted_range_string,
)

# -- Simulation & Equities --------------------------------------------------
from poker_equity.simulation import (
    EquityResult,
    ProgressCallback,
    ProgressInfo,
    calculate_equities,
    calculate_equity,
    range_vs_range_equity,
)

# -- Range vs Hand Breakdown ------------------------------------------------
from poker_equity.breakdown import (
    RangeBreakdown,
    hand_vs_range_breakdown,
    range_breakdown,
    range_vs_hand_breakdown,
)

# -- Presets ----------------------------------------------------------------
from poker_equity.presets import (
    Position,
    PRESET_DESCRIPTIONS,
    PRESET_RANGES,
    TOP_5_PERCENT,
    TOP_10_PERCENT,
    TOP_15_PERCENT,
    TOP_20_PERCENT,
    TOP_25_PERCENT,
    TOP_30_PERCENT,
    TOP_40_PERCENT,
    TOP_50_PERCENT,
    get_preset_description,
    get_preset_range,
    list_presets,
    top_percent_range,
)

# -- Pot Odds & Decision Math ----------------------------------------------
from poker_equity.odds import (
    is_profitable_call,
    pot_odds,
    pot_odds_ratio,
)

__all__ = [
    # Card primitives
    "Card",
    "Rank",
    "Suit",
    "Deck",
    # Hand ranking
    "HandRank",
    "HandResult",
    # Evaluators (rich)
    "evaluate_5",
    "evaluate_7",
    "evaluate_hand",
    # Evaluators (fast score)
    "score_5",
    "score_7",
    "score_hand",
    # Range & combos
    "Combo",
    "Range",
    "GRID_RANKS",
    "HAND_GRID_CELLS",
    "NUM_GRID_CELLS",
    "TOTAL_HOLDEM_COMBOS",
    "canonical_combo",
    "combo_from_str",
    "combo_to_cell",
    "combo_to_str",
    "generate_all_combos",
    "generate_offsuit_combos",
    "generate_pair_combos",
    "generate_suited_combos",
    "generate_unpaired_combos",
    "parse_range_string",
    "parse_weighted_range_string",
    # Simulation & Equities
    "EquityResult",
    "ProgressCallback",
    "ProgressInfo",
    "calculate_equity",
    "calculate_equities",
    "range_vs_range_equity",
    # Range vs Hand Breakdown
    "RangeBreakdown",
    "hand_vs_range_breakdown",
    "range_breakdown",
    "range_vs_hand_breakdown",
    # Presets
    "Position",
    "PRESET_RANGES",
    "PRESET_DESCRIPTIONS",
    "TOP_5_PERCENT",
    "TOP_10_PERCENT",
    "TOP_15_PERCENT",
    "TOP_20_PERCENT",
    "TOP_25_PERCENT",
    "TOP_30_PERCENT",
    "TOP_40_PERCENT",
    "TOP_50_PERCENT",
    "get_preset_range",
    "list_presets",
    "get_preset_description",
    "top_percent_range",
    # Pot Odds & Decision Math
    "pot_odds",
    "pot_odds_ratio",
    "is_profitable_call",
    # Constants
    "FULL_DECK",
    "FULL_DECK_LIST",
    "RANKS",
    "RANK_CHARS",
    "SUITS",
    "SUIT_CHARS",
    "HAND_RANKINGS",
    # Utilities
    "parse_cards",
    "cards_to_str",
    "validate_no_duplicates",
]


