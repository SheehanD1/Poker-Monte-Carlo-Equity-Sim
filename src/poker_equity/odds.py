"""Pot odds, expected value (EV), and decision mathematics for Texas Hold'em.

This module provides financial and probabilistic calculations used in poker
decision analysis, including pot odds, break-even equity requirements,
expected value, and implied odds.
"""

from __future__ import annotations


def pot_odds(pot_size: float, bet_to_call: float) -> float:
    """Calculate the required break-even equity given pot size and bet to call.

    Pot odds represent the ratio between the size of the pot and the cost of
    calling.  Expressed as an equity fraction (between 0.0 and 1.0), it indicates
    the minimum winning probability required for calling to be mathematically
    break-even (EV = 0).

    The formula is:

    .. math::
        \\text{required\\_equity} = \\frac{\\text{bet\\_to\\_call}}{\\text{pot\\_size} + \\text{bet\\_to\\_call}}

    Parameters
    ----------
    pot_size : float
        The total amount of chips currently in the pot facing the player.
        This includes all previous bets, blinds, antes, and any bet or raise
        placed by opponents in the current betting round that the player is
        facing.
    bet_to_call : float
        The additional amount of chips the player must wager to stay in the hand.

    Returns
    -------
    float
        The required break-even equity as a fraction between 0.0 and 1.0.
        Returns 0.0 if ``bet_to_call`` is 0 (i.e., checking or a free card).

    Raises
    ------
    ValueError
        If ``pot_size < 0`` or ``bet_to_call < 0``.
    TypeError
        If ``pot_size`` or ``bet_to_call`` is not a numeric type (int or float).

    Examples
    --------
    Facing a half-pot bet ($50 into a $100 pot -> pot facing player is $150,
    to call $50):

    >>> from poker_equity.odds import pot_odds
    >>> pot_odds(pot_size=150.0, bet_to_call=50.0)
    0.25

    Facing a pot-sized bet ($100 into $100 pot -> pot is $200, to call $100):

    >>> pot_odds(pot_size=200.0, bet_to_call=100.0)
    0.3333333333333333

    Facing no bet (free check):

    >>> pot_odds(pot_size=100.0, bet_to_call=0.0)
    0.0
    """
    if isinstance(pot_size, bool) or not isinstance(pot_size, (int, float)):
        raise TypeError(f"pot_size must be a number, got {type(pot_size).__name__}")
    if isinstance(bet_to_call, bool) or not isinstance(bet_to_call, (int, float)):
        raise TypeError(f"bet_to_call must be a number, got {type(bet_to_call).__name__}")

    if pot_size < 0:
        raise ValueError(f"pot_size cannot be negative, got {pot_size}")
    if bet_to_call < 0:
        raise ValueError(f"bet_to_call cannot be negative, got {bet_to_call}")

    total_pot = pot_size + bet_to_call
    if total_pot == 0.0 or bet_to_call == 0.0:
        return 0.0

    return bet_to_call / total_pot


def pot_odds_ratio(pot_size: float, bet_to_call: float) -> tuple[float, float]:
    """Calculate pot odds expressed as a ratio (e.g. (3.0, 1.0) representing 3:1).

    Parameters
    ----------
    pot_size : float
        The total amount of chips currently in the pot facing the player.
    bet_to_call : float
        The additional amount of chips the player must wager to call.

    Returns
    -------
    tuple[float, float]
        A 2-tuple ``(ratio, 1.0)`` representing ``ratio:1``.
        If ``bet_to_call`` is 0, returns ``(float('inf'), 1.0)``.

    Raises
    ------
    ValueError
        If ``pot_size < 0`` or ``bet_to_call < 0``.
    TypeError
        If ``pot_size`` or ``bet_to_call`` is not a numeric type.

    Examples
    --------
    >>> from poker_equity.odds import pot_odds_ratio
    >>> pot_odds_ratio(pot_size=150.0, bet_to_call=50.0)
    (3.0, 1.0)
    """
    if isinstance(pot_size, bool) or not isinstance(pot_size, (int, float)):
        raise TypeError(f"pot_size must be a number, got {type(pot_size).__name__}")
    if isinstance(bet_to_call, bool) or not isinstance(bet_to_call, (int, float)):
        raise TypeError(f"bet_to_call must be a number, got {type(bet_to_call).__name__}")

    if pot_size < 0:
        raise ValueError(f"pot_size cannot be negative, got {pot_size}")
    if bet_to_call < 0:
        raise ValueError(f"bet_to_call cannot be negative, got {bet_to_call}")

    if bet_to_call == 0.0:
        return (float("inf"), 1.0)

    return (pot_size / bet_to_call, 1.0)


def is_profitable_call(equity: float, pot_size: float, bet_to_call: float) -> bool:
    """Determine whether calling is profitable based on equity and pot odds.

    Calling is mathematically profitable (expected value >= 0) if Hero's equity
    is greater than or equal to the break-even pot odds requirement.

    Parameters
    ----------
    equity : float
        Hero's equity in the hand as a fraction between 0.0 and 1.0.
    pot_size : float
        The total amount of chips currently in the pot facing the player.
    bet_to_call : float
        The additional amount of chips the player must wager to call.

    Returns
    -------
    bool
        True if ``equity >= pot_odds(pot_size, bet_to_call)``, False otherwise.

    Raises
    ------
    ValueError
        If ``equity`` is not between 0.0 and 1.0, or if ``pot_size`` or
        ``bet_to_call`` is negative.
    TypeError
        If any argument is not numeric.

    Examples
    --------
    >>> from poker_equity.odds import is_profitable_call
    >>> # 30% equity facing a call that requires 25% equity -> profitable call
    >>> is_profitable_call(equity=0.30, pot_size=150.0, bet_to_call=50.0)
    True
    >>> # 20% equity facing a call that requires 25% equity -> unprofitable call
    >>> is_profitable_call(equity=0.20, pot_size=150.0, bet_to_call=50.0)
    False
    """
    if isinstance(equity, bool) or not isinstance(equity, (int, float)):
        raise TypeError(f"equity must be a number, got {type(equity).__name__}")
    if not (0.0 <= equity <= 1.0):
        raise ValueError(f"equity must be between 0.0 and 1.0, got {equity}")

    required = pot_odds(pot_size, bet_to_call)
    return equity >= required

