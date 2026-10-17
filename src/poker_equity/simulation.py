"""Monte Carlo equity simulation for Texas Hold'em.

This module provides the core simulation engine that calculates win / tie /
loss probabilities by running thousands of random board runouts.

The engine supports **convergence tracking with early stopping**: when the
95% confidence interval for every player's equity shrinks below a
configurable threshold, the simulation halts automatically, saving compute
time without sacrificing accuracy.

An optional **progress callback** can be supplied to receive periodic
updates during the simulation, useful for real-time UI progress bars
or logging.

The primary entry point is :func:`calculate_equity`, which accepts hero
and villain hole cards, an optional partial board, and a simulation count.

Example
-------
>>> from poker_equity.card import Card
>>> from poker_equity.simulation import calculate_equity
>>> hero = [Card.from_str("Ah"), Card.from_str("Kh")]
>>> villain = [Card.from_str("Qs"), Card.from_str("Qd")]
>>> result = calculate_equity(hero, villain, num_sims=10_000, seed=42)
>>> result.equity  # hero's equity as a float in [0, 1]
0.4...
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from poker_equity.card import Card
from poker_equity.evaluator import score_7


# =========================================================================
# Result type
# =========================================================================


@dataclass(frozen=True, slots=True)
class EquityResult:
    """The result of a Monte Carlo equity simulation.

    All percentages are expressed as floats in ``[0.0, 1.0]``.

    Attributes
    ----------
    wins : int
        Number of simulations where player won outright.
    ties : int
        Number of simulations where player split the pot.
    losses : int
        Number of simulations where player lost.
    num_sims : int
        Total number of simulations run.
    equity : float
        Player's equity (expected share of the pot).
    win_pct : float
        Win percentage: ``wins / num_sims``.
    tie_pct : float
        Tie percentage: ``ties / num_sims``.
    loss_pct : float
        Loss percentage: ``losses / num_sims``.
    converged : bool
        ``True`` if the simulation stopped early due to convergence.
    convergence_ci : float
        The 95% confidence interval half-width at the time the
        simulation finished.  ``0.0`` when convergence was not tracked.
    """

    wins: int
    ties: int
    losses: int
    num_sims: int
    equity: float
    win_pct: float
    tie_pct: float
    loss_pct: float
    converged: bool = False
    convergence_ci: float = 0.0

    def __str__(self) -> str:
        parts = [
            f"Equity: {self.equity:.1%}",
            f"Win: {self.win_pct:.1%}",
            f"Tie: {self.tie_pct:.1%}",
            f"Lose: {self.loss_pct:.1%}",
            f"({self.num_sims:,} sims)",
        ]
        if self.converged:
            parts.append("[converged]")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return (
            f"EquityResult(equity={self.equity:.4f}, "
            f"win={self.win_pct:.4f}, tie={self.tie_pct:.4f}, "
            f"loss={self.loss_pct:.4f}, sims={self.num_sims}, "
            f"converged={self.converged})"
        )


def _build_equity_result(
    wins: int,
    ties: int,
    losses: int,
    num_sims: int,
    equity: float,
    *,
    converged: bool = False,
    convergence_ci: float = 0.0,
) -> EquityResult:
    """Construct an EquityResult from raw counts."""
    return EquityResult(
        wins=wins,
        ties=ties,
        losses=losses,
        num_sims=num_sims,
        equity=equity,
        win_pct=wins / num_sims,
        tie_pct=ties / num_sims,
        loss_pct=losses / num_sims,
        converged=converged,
        convergence_ci=convergence_ci,
    )


@dataclass(frozen=True, slots=True)
class ProgressInfo:
    """Snapshot of simulation progress, delivered to ``on_progress`` callbacks.

    Attributes
    ----------
    sims_completed : int
        Number of simulations completed so far.
    sims_total : int
        Maximum number of simulations requested.
    pct_complete : float
        Completion fraction in ``[0.0, 1.0]``.
    current_equities : list[float]
        Running equity estimate for each player at this point.
    elapsed_seconds : float
        Wall-clock seconds elapsed since the simulation started.
    """

    sims_completed: int
    sims_total: int
    pct_complete: float
    current_equities: list[float]
    elapsed_seconds: float

    def __repr__(self) -> str:
        eqs = ", ".join(f"{e:.2%}" for e in self.current_equities)
        return (
            f"ProgressInfo({self.pct_complete:.1%} done, "
            f"sims={self.sims_completed}/{self.sims_total}, "
            f"equities=[{eqs}], "
            f"elapsed={self.elapsed_seconds:.2f}s)"
        )


#: Type alias for progress callbacks.
ProgressCallback = Callable[[ProgressInfo], None]


# =========================================================================
# Convergence tracking (Welford's online algorithm)
# =========================================================================


# 95% confidence z-score
_Z_95 = 1.96

# Minimum number of iterations before we start checking convergence,
# to avoid false positives from tiny sample sizes.
_MIN_CONVERGENCE_SIMS = 100


def _confidence_interval_half_width(
    variance: float, n: int
) -> float:
    """Return the 95% CI half-width given variance and sample count."""
    if n < 2:
        return float("inf")
    return _Z_95 * math.sqrt(variance / n)


# =========================================================================
# Validation helpers
# =========================================================================


def _validate_multi_inputs(
    hands: Sequence[Sequence[Card] | None],
    board: Sequence[Card],
) -> None:
    """Validate simulation inputs for multiple players."""
    if not (2 <= len(hands) <= 9):
        msg = f"Must have between 2 and 9 players, got {len(hands)}"
        raise ValueError(msg)

    for i, hand in enumerate(hands):
        if hand is not None and len(hand) != 2:
            msg = f"Player {i+1} must have exactly 2 hole cards or be None, got {len(hand)}"
            raise ValueError(msg)

    if len(board) > 5:
        msg = f"Board can have at most 5 cards, got {len(board)}"
        raise ValueError(msg)

    all_cards = list(board)
    for hand in hands:
        if hand is not None:
            all_cards.extend(hand)
            
    if len(set(all_cards)) != len(all_cards):
        seen: set[Card] = set()
        for card in all_cards:
            if card in seen:
                msg = f"Duplicate card: {card}"
                raise ValueError(msg)
            seen.add(card)


# =========================================================================
# Core simulation
# =========================================================================


def calculate_equities(
    hands: Sequence[Sequence[Card] | None],
    board: Sequence[Card] | None = None,
    *,
    num_sims: int = 10_000,
    seed: int | None = None,
    convergence_threshold: float | None = None,
    check_interval: int = 100,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 500,
) -> list[EquityResult]:
    """Calculate equities for a multi-way pot via Monte Carlo simulation.

    Parameters
    ----------
    hands:
        List of hole cards for each player. Use ``None`` for random hands.
    board:
        Known community cards (0 to 5).
    num_sims:
        Maximum number of Monte Carlo simulations to run.
    seed:
        Optional PRNG seed for reproducible results.
    convergence_threshold:
        If set, the simulation may stop early when the 95%
        confidence interval half-width for **every** player's equity
        drops below this value.  For example, ``0.01`` means "stop
        once we're within +/-1% of the true equity with 95% confidence".
        ``None`` (the default) disables early stopping.
    check_interval:
        How often (in number of simulations) to check for convergence.
        Smaller values detect convergence sooner but add overhead.
        Default is 100.
    on_progress:
        Optional callback invoked periodically with a :class:`ProgressInfo`
        snapshot.  Useful for driving progress bars or real-time UI
        updates.  ``None`` (the default) disables progress reporting.
    progress_interval:
        How often (in number of simulations) to invoke ``on_progress``.
        Default is 500.  Ignored when ``on_progress`` is ``None``.

    Returns
    -------
    list[EquityResult]
        The simulation results for each player in the same order as `hands`.
    """
    if board is None:
        board = []

    _validate_multi_inputs(hands, board)

    if num_sims <= 0:
        msg = f"num_sims must be positive, got {num_sims}"
        raise ValueError(msg)

    if convergence_threshold is not None and convergence_threshold <= 0:
        msg = f"convergence_threshold must be positive, got {convergence_threshold}"
        raise ValueError(msg)

    if check_interval <= 0:
        msg = f"check_interval must be positive, got {check_interval}"
        raise ValueError(msg)

    if progress_interval <= 0:
        msg = f"progress_interval must be positive, got {progress_interval}"
        raise ValueError(msg)

    known_cards = set(board)
    for hand in hands:
        if hand is not None:
            known_cards.update(hand)
            
    stub = [
        Card(rank, suit)
        for rank in _ALL_RANKS
        for suit in _ALL_SUITS
        if Card(rank, suit) not in known_cards
    ]

    board_list = list(board)
    board_needed = 5 - len(board_list)
    random_hands_count = sum(1 for h in hands if h is None)
    cards_to_deal = board_needed + 2 * random_hands_count
    
    num_players = len(hands)
    
    # Pre-build the fixed parts of each player's 7-card hand
    fixed_hands_data = [
        list(h) + board_list if h is not None else [] 
        for h in hands
    ]
    
    rng = random.Random(seed)
    wins = [0] * num_players
    ties = [0] * num_players
    losses = [0] * num_players
    equity_sums = [0.0] * num_players

    # Welford's online algorithm state for each player
    tracking = convergence_threshold is not None
    welford_mean = [0.0] * num_players
    welford_m2 = [0.0] * num_players
    converged = False
    final_ci = 0.0
    actual_sims = num_sims  # may be reduced by early stopping

    t_start = time.perf_counter()

    for sim_idx in range(1, num_sims + 1):
        dealt = rng.sample(stub, cards_to_deal)
        community_dealt = dealt[:board_needed]
        
        scores = []
        random_cards_index = board_needed
        
        for i, (hand, fixed) in enumerate(zip(hands, fixed_hands_data)):
            if hand is not None:
                hand_score = score_7(fixed + community_dealt)
            else:
                hole1 = dealt[random_cards_index]
                hole2 = dealt[random_cards_index + 1]
                random_cards_index += 2
                hand_score = score_7(board_list + community_dealt + [hole1, hole2])
            scores.append(hand_score)
            
        max_score = max(scores)
        
        # Determine winners
        winners = []
        for i in range(num_players):
            if scores[i] == max_score:
                winners.append(i)
            else:
                losses[i] += 1
                
        num_winners = len(winners)
        pot_share = 1.0 / num_winners
        
        for i in winners:
            if num_winners == 1:
                wins[i] += 1
            else:
                ties[i] += 1
            equity_sums[i] += pot_share

        # Update Welford's running statistics for each player
        if tracking:
            for p in range(num_players):
                # This iteration's outcome: pot share earned
                x = pot_share if p in winners else 0.0
                delta = x - welford_mean[p]
                welford_mean[p] += delta / sim_idx
                delta2 = x - welford_mean[p]
                welford_m2[p] += delta * delta2

            # Periodically check convergence and fire progress callback
            if (
                sim_idx >= _MIN_CONVERGENCE_SIMS
                and sim_idx % check_interval == 0
            ):
                # Fire progress callback at convergence check points
                if on_progress is not None:
                    current_eqs = [
                        equity_sums[p] / sim_idx for p in range(num_players)
                    ]
                    on_progress(ProgressInfo(
                        sims_completed=sim_idx,
                        sims_total=num_sims,
                        pct_complete=sim_idx / num_sims,
                        current_equities=current_eqs,
                        elapsed_seconds=time.perf_counter() - t_start,
                    ))

                all_converged = True
                max_ci = 0.0
                for p in range(num_players):
                    var = welford_m2[p] / (sim_idx - 1)
                    ci = _confidence_interval_half_width(var, sim_idx)
                    if ci > max_ci:
                        max_ci = ci
                    if ci > convergence_threshold:
                        all_converged = False
                        break

                if all_converged:
                    converged = True
                    final_ci = max_ci
                    actual_sims = sim_idx
                    break

        # Fire periodic progress callback
        if (
            on_progress is not None
            and not tracking  # avoid double-firing when convergence also checks
            and sim_idx % progress_interval == 0
        ):
            current_eqs = [
                equity_sums[p] / sim_idx for p in range(num_players)
            ]
            on_progress(ProgressInfo(
                sims_completed=sim_idx,
                sims_total=num_sims,
                pct_complete=sim_idx / num_sims,
                current_equities=current_eqs,
                elapsed_seconds=time.perf_counter() - t_start,
            ))

    # Compute final CI if we tracked but didn't converge early
    if tracking and not converged:
        max_ci = 0.0
        for p in range(num_players):
            if actual_sims >= 2:
                var = welford_m2[p] / (actual_sims - 1)
                ci = _confidence_interval_half_width(var, actual_sims)
                if ci > max_ci:
                    max_ci = ci
        final_ci = max_ci

    return [
        _build_equity_result(
            wins[i],
            ties[i],
            losses[i],
            actual_sims,
            equity_sums[i] / actual_sims,
            converged=converged,
            convergence_ci=final_ci,
        )
        for i in range(num_players)
    ]


def calculate_equity(
    hero: Sequence[Card],
    villain: Sequence[Card] | None = None,
    board: Sequence[Card] | None = None,
    *,
    num_sims: int = 10_000,
    seed: int | None = None,
    convergence_threshold: float | None = None,
    check_interval: int = 100,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 500,
) -> EquityResult:
    """Calculate hero's equity against villain via Monte Carlo simulation.

    This is a convenience wrapper around :func:`calculate_equities` for 
    heads-up scenarios.

    Parameters
    ----------
    hero:
        Hero's two hole cards.
    villain:
        Villain's two hole cards, or ``None`` to evaluate against a
        random hand (any two unknown cards).
    board:
        Known community cards (0 to 5).  ``None`` or empty list means
        preflop (no board cards dealt yet).
    num_sims:
        Maximum number of Monte Carlo simulations to run.  Higher values
        give more accurate results but take longer.  Default is 10,000.
    seed:
        Optional PRNG seed for reproducible results.
    convergence_threshold:
        If set, the simulation may stop early when the 95% CI
        half-width drops below this value for every player.
    check_interval:
        How often (in simulations) to check convergence.
    on_progress:
        Optional callback invoked periodically with a
        :class:`ProgressInfo` snapshot.
    progress_interval:
        How often (in simulations) to invoke ``on_progress``.

    Returns
    -------
    EquityResult
        The simulation results with win/tie/loss counts and equity.

    Raises
    ------
    ValueError
        If inputs are invalid (wrong card count, duplicates, etc.).

    Examples
    --------
    >>> from poker_equity.card import Card
    >>> hero = [Card.from_str("Ah"), Card.from_str("Kh")]
    >>> villain = [Card.from_str("7s"), Card.from_str("2d")]
    >>> result = calculate_equity(hero, villain, seed=42)
    >>> result.equity > 0.5  # AKs is a big favorite over 72o
    True
    """
    results = calculate_equities(
        hands=[hero, villain],
        board=board,
        num_sims=num_sims,
        seed=seed,
        convergence_threshold=convergence_threshold,
        check_interval=check_interval,
        on_progress=on_progress,
        progress_interval=progress_interval,
    )
    return results[0]


# ---------------------------------------------------------------------------
# Internal constants (avoid importing from card at module level for speed)
# ---------------------------------------------------------------------------

from poker_equity.card import Rank, Suit  # noqa: E402

_ALL_RANKS = list(Rank)
_ALL_SUITS = list(Suit)
