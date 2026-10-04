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
