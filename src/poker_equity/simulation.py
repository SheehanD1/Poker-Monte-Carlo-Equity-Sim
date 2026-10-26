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

from poker_equity.card import Card, Rank, Suit
from poker_equity.constants import FULL_DECK_LIST
from poker_equity.evaluator import score_7
from poker_equity.range import Combo, Range, canonical_combo

_ALL_RANKS = list(Rank)
_ALL_SUITS = list(Suit)


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


# =========================================================================
# Validation helpers & Input Parsing
# =========================================================================


def _coerce_board(board: Sequence[Card | str] | str | None) -> list[Card]:
    """Parse and normalize community board cards.

    Parameters
    ----------
    board : Sequence[Card | str] | str | None
        Community cards as a sequence of Card objects or 2-character strings,
        a delimited string (e.g. ``"Ah Kd 2s"``), or None.

    Returns
    -------
    list[Card]
        List of Card objects.
    """
    if board is None:
        return []
    if isinstance(board, str):
        board_str = board.strip()
        if not board_str:
            return []
        if " " in board_str or "," in board_str:
            tokens = board_str.replace(",", " ").split()
        elif len(board_str) % 2 == 0:
            tokens = [board_str[i : i + 2] for i in range(0, len(board_str), 2)]
        else:
            tokens = [board_str]
        return [Card.from_str(t) for t in tokens]
    cards: list[Card] = []
    for c in board:
        if isinstance(c, Card):
            cards.append(c)
        elif isinstance(c, str):
            cards.append(Card.from_str(c))
        else:
            msg = f"Invalid card type in board: {type(c).__name__}"
            raise TypeError(msg)
    return cards


def _validate_multi_inputs(
    hands: Sequence[Sequence[Card] | Range | str | None],
    board: Sequence[Card | str] | str | None,
) -> None:
    """Validate simulation inputs for multiple players.

    Ensures player counts, card counts, range non-emptiness, and duplicate
    constraints are satisfied.
    """
    if not (2 <= len(hands) <= 9):
        msg = f"Must have between 2 and 9 players, got {len(hands)}"
        raise ValueError(msg)

    board_cards = _coerce_board(board)
    if len(board_cards) > 5:
        msg = f"Board can have at most 5 cards, got {len(board_cards)}"
        raise ValueError(msg)

    for i, hand in enumerate(hands):
        if hand is None:
            continue
        elif isinstance(hand, Range):
            if hand.is_empty:
                msg = f"Player {i+1} range is empty"
                raise ValueError(msg)
        elif isinstance(hand, str):
            r = Range.from_str(hand)
            if r.is_empty:
                msg = f"Player {i+1} range '{hand}' contains no combos"
                raise ValueError(msg)
        elif isinstance(hand, (list, tuple)):
            if len(hand) != 2:
                msg = f"Player {i+1} must have exactly 2 hole cards or be None, got {len(hand)}"
                raise ValueError(msg)
            c1 = Card.from_str(hand[0]) if isinstance(hand[0], str) else hand[0]
            c2 = Card.from_str(hand[1]) if isinstance(hand[1], str) else hand[1]
            if not isinstance(c1, Card) or not isinstance(c2, Card):
                msg = f"Player {i+1} hole cards must be Card instances or valid card strings"
                raise TypeError(msg)
            if c1 == c2:
                msg = f"Duplicate card: {c1}"
                raise ValueError(msg)
        else:
            msg = (
                f"Player {i+1} hand must be a 2-card sequence, Range, "
                f"range string, or None, got {type(hand).__name__}"
            )
            raise TypeError(msg)

    all_cards = list(board_cards)
    for hand in hands:
        if hand is not None and isinstance(hand, (list, tuple)):
            for c in hand:
                all_cards.append(Card.from_str(c) if isinstance(c, str) else c)

    if len(set(all_cards)) != len(all_cards):
        seen: set[Card] = set()
        for card in all_cards:
            if card in seen:
                msg = f"Duplicate card: {card}"
                raise ValueError(msg)
            seen.add(card)


def _check_combos_mutually_exclusive_2p(
    combos1: Sequence[Combo],
    combos2: Sequence[Combo],
) -> bool:
    """Return True if there is at least one disjoint pair of combos between two players."""
    for c1 in combos1:
        for c2 in combos2:
            if c1[0] != c2[0] and c1[0] != c2[1] and c1[1] != c2[0] and c1[1] != c2[1]:
                return True
    return False


def _check_combos_mutually_exclusive_multi(
    combos_lists: list[Sequence[Combo]],
    player_idx: int = 0,
    used: frozenset[Card] = frozenset(),
) -> bool:
    """Backtracking check to ensure mutually exclusive card assignments exist for all players."""
    if player_idx == len(combos_lists):
        return True
    for c in combos_lists[player_idx]:
        if c[0] not in used and c[1] not in used:
            if _check_combos_mutually_exclusive_multi(
                combos_lists, player_idx + 1, used | {c[0], c[1]}
            ):
                return True
    return False


def _prepare_simulation_inputs(
    hands: Sequence[Sequence[Card] | Range | str | None],
    board: Sequence[Card | str] | str | None,
) -> tuple[
    list[Card],
    list[str],
    list[tuple[Card, Card] | None],
    list[tuple[Combo, ...] | None],
    list[tuple[float, ...] | None],
    list[bool],
    list[Card],
    bool,
]:
    """Parse, validate, and prepare simulation inputs for multi-player simulation."""
    board_cards = _coerce_board(board)
    _validate_multi_inputs(hands, board_cards)

    player_kinds: list[str] = []
    player_fixed: list[tuple[Card, Card] | None] = []
    player_ranges: list[tuple[Combo, ...] | None] = []
    player_weights: list[tuple[float, ...] | None] = []
    player_is_weighted: list[bool] = []
    fixed_cards: list[Card] = list(board_cards)
    has_range = False

    for i, hand in enumerate(hands):
        if hand is None:
            player_kinds.append("random")
            player_fixed.append(None)
            player_ranges.append(None)
            player_weights.append(None)
            player_is_weighted.append(False)
        elif isinstance(hand, Range):
            has_range = True
            player_kinds.append("range")
            player_fixed.append(None)
            player_ranges.append(hand.combos_list)
            player_weights.append(hand.weights_list)
            player_is_weighted.append(hand.is_weighted)
        elif isinstance(hand, str):
            r = Range.from_str(hand)
            has_range = True
            player_kinds.append("range")
            player_fixed.append(None)
            player_ranges.append(r.combos_list)
            player_weights.append(r.weights_list)
            player_is_weighted.append(r.is_weighted)
        elif isinstance(hand, (list, tuple)):
            c1 = Card.from_str(hand[0]) if isinstance(hand[0], str) else hand[0]
            c2 = Card.from_str(hand[1]) if isinstance(hand[1], str) else hand[1]
            fixed = canonical_combo(c1, c2)
            player_kinds.append("fixed")
            player_fixed.append(fixed)
            player_ranges.append(None)
            player_weights.append(None)
            player_is_weighted.append(False)
            fixed_cards.extend(fixed)
        else:
            msg = f"Player {i+1} hand must be a 2-card sequence, Range, range string, or None"
            raise TypeError(msg)

    if has_range:
        dead_set = set(fixed_cards)
        filtered_ranges: list[tuple[Combo, ...] | None] = []
        filtered_weights: list[tuple[float, ...] | None] = []
        filtered_is_weighted: list[bool] = []
        for i, kind in enumerate(player_kinds):
            if kind == "range":
                combos = player_ranges[i]
                weights = player_weights[i]
                assert combos is not None and weights is not None
                live_combos = []
                live_weights = []
                for c, w in zip(combos, weights):
                    if c[0] not in dead_set and c[1] not in dead_set:
                        live_combos.append(c)
                        live_weights.append(w)
                if not live_combos:
                    msg = f"Player {i+1} range has no valid combos remaining after removing dead cards"
                    raise ValueError(msg)
                filtered_ranges.append(tuple(live_combos))
                filtered_weights.append(tuple(live_weights))
                filtered_is_weighted.append(any(w < 1.0 for w in live_weights))
            else:
                filtered_ranges.append(None)
                filtered_weights.append(None)
                filtered_is_weighted.append(False)
        player_ranges = filtered_ranges
        player_weights = filtered_weights
        player_is_weighted = filtered_is_weighted

        # Check mutual exclusivity across range players
        range_indices = [i for i, k in enumerate(player_kinds) if k == "range"]
        if len(range_indices) == 1:
            pass  # 1 range player with live combos; valid deal is guaranteed
        elif len(range_indices) == 2:
            r1 = player_ranges[range_indices[0]]
            r2 = player_ranges[range_indices[1]]
            assert r1 is not None and r2 is not None
            if not _check_combos_mutually_exclusive_2p(r1, r2):
                raise ValueError("No mutually exclusive combos available between players")
        else:
            combos_to_check = [player_ranges[idx] for idx in range_indices]
            assert all(cl is not None for cl in combos_to_check)
            if not _check_combos_mutually_exclusive_multi(combos_to_check):
                raise ValueError("No mutually exclusive combos available between players")

    return (
        board_cards,
        player_kinds,
        player_fixed,
        player_ranges,
        player_weights,
        player_is_weighted,
        fixed_cards,
        has_range,
    )


# =========================================================================
# Core simulation
# =========================================================================


def calculate_equities(
    hands: Sequence[Sequence[Card] | Range | str | None],
    board: Sequence[Card | str] | str | None = None,
    *,
    num_sims: int = 10_000,
    seed: int | None = None,
    convergence_threshold: float | None = None,
    check_interval: int = 100,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 500,
) -> list[EquityResult]:
    """Calculate equities for a multi-way pot via Monte Carlo simulation.

    Supports fixed 2-card hands, :class:`Range` instances, range syntax strings
    (e.g. ``"QQ+, AKs"``), and ``None`` for random hands.

    Parameters
    ----------
    hands:
        List of hole cards or ranges for each player (2 to 9 players).
        Elements can be a 2-card sequence, a :class:`Range` object,
        a range string, or ``None`` (for a random hand).
    board:
        Known community cards (0 to 5), as a sequence of cards, a string,
        or ``None`` (preflop).
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
        Default is 100.
    on_progress:
        Optional callback invoked periodically with a :class:`ProgressInfo`
        snapshot.
    progress_interval:
        How often (in number of simulations) to invoke ``on_progress``.
        Default is 500.

    Returns
    -------
    list[EquityResult]
        The simulation results for each player in the same order as `hands`.
    """
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

    (
        board_cards,
        player_kinds,
        player_fixed,
        player_ranges,
        player_weights,
        player_is_weighted,
        fixed_cards,
        has_range,
    ) = _prepare_simulation_inputs(hands, board)

    num_players = len(hands)
    board_needed = 5 - len(board_cards)
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
    actual_sims = num_sims

    t_start = time.perf_counter()

    if not has_range:
        # -------------------------------------------------------------
        # Fast path: All players are fixed hands or random hands
        # -------------------------------------------------------------
        known_cards = set(fixed_cards)
        stub = [c for c in FULL_DECK_LIST if c not in known_cards]
        random_hands_count = sum(1 for k in player_kinds if k == "random")
        cards_to_deal = board_needed + 2 * random_hands_count

        fixed_hands_data = [
            list(player_fixed[i]) + board_cards if player_fixed[i] is not None else []
            for i in range(num_players)
        ]

        for sim_idx in range(1, num_sims + 1):
            dealt = rng.sample(stub, cards_to_deal)
            community_dealt = dealt[:board_needed]

            scores = []
            random_cards_index = board_needed

            for i in range(num_players):
                if player_fixed[i] is not None:
                    hand_score = score_7(fixed_hands_data[i] + community_dealt)
                else:
                    h1 = dealt[random_cards_index]
                    h2 = dealt[random_cards_index + 1]
                    random_cards_index += 2
                    hand_score = score_7(board_cards + community_dealt + [h1, h2])
                scores.append(hand_score)

            max_score = max(scores)
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

            if tracking:
                for p in range(num_players):
                    x = pot_share if p in winners else 0.0
                    delta = x - welford_mean[p]
                    welford_mean[p] += delta / sim_idx
                    delta2 = x - welford_mean[p]
                    welford_m2[p] += delta * delta2

                if sim_idx >= _MIN_CONVERGENCE_SIMS and sim_idx % check_interval == 0:
                    if on_progress is not None:
                        current_eqs = [
                            equity_sums[p] / sim_idx for p in range(num_players)
                        ]
                        on_progress(
                            ProgressInfo(
                                sims_completed=sim_idx,
                                sims_total=num_sims,
                                pct_complete=sim_idx / num_sims,
                                current_equities=current_eqs,
                                elapsed_seconds=time.perf_counter() - t_start,
                            )
                        )

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

            if (
                on_progress is not None
                and not tracking
                and sim_idx % progress_interval == 0
            ):
                current_eqs = [
                    equity_sums[p] / sim_idx for p in range(num_players)
                ]
                on_progress(
                    ProgressInfo(
                        sims_completed=sim_idx,
                        sims_total=num_sims,
                        pct_complete=sim_idx / num_sims,
                        current_equities=current_eqs,
                        elapsed_seconds=time.perf_counter() - t_start,
                    )
                )

    else:
        # -------------------------------------------------------------
        # Range path: At least one player has a Range
        # -------------------------------------------------------------
        range_indices = [i for i, k in enumerate(player_kinds) if k == "range"]
        num_range_players = len(range_indices)
        random_hands_count = sum(1 for k in player_kinds if k == "random")
        cards_from_stub = board_needed + 2 * random_hands_count
        base_stub = [c for c in FULL_DECK_LIST if c not in fixed_cards]

        if num_range_players == 1:
            r_single_idx = range_indices[0]
            r_combos_single = player_ranges[r_single_idx]
            r_weights_single = player_weights[r_single_idx]
            r_is_w_single = player_is_weighted[r_single_idx]
            assert r_combos_single is not None
        elif num_range_players == 2:
            r_idx_1, r_idx_2 = range_indices[0], range_indices[1]
            r_combos_1 = player_ranges[r_idx_1]
            r_combos_2 = player_ranges[r_idx_2]
            r_weights_1 = player_weights[r_idx_1]
            r_weights_2 = player_weights[r_idx_2]
            r_is_w1 = player_is_weighted[r_idx_1]
            r_is_w2 = player_is_weighted[r_idx_2]
            assert r_combos_1 is not None and r_combos_2 is not None
        else:
            r_combos_multi = [player_ranges[idx] for idx in range_indices]
            r_weights_multi = [player_weights[idx] for idx in range_indices]
            r_is_w_multi = [player_is_weighted[idx] for idx in range_indices]

        sim_hands: list[tuple[Card, Card] | None] = [
            player_fixed[i] for i in range(num_players)
        ]

        for sim_idx in range(1, num_sims + 1):
            if num_range_players == 1:
                if r_is_w_single:
                    c = rng.choices(r_combos_single, weights=r_weights_single, k=1)[0]
                else:
                    c = rng.choice(r_combos_single)
                sim_hands[r_single_idx] = c
                dealt_range_cards = (c[0], c[1])
            elif num_range_players == 2:
                while True:
                    if r_is_w1:
                        c1 = rng.choices(r_combos_1, weights=r_weights_1, k=1)[0]
                    else:
                        c1 = rng.choice(r_combos_1)
                    if r_is_w2:
                        c2 = rng.choices(r_combos_2, weights=r_weights_2, k=1)[0]
                    else:
                        c2 = rng.choice(r_combos_2)
                    if (
                        c1[0] != c2[0]
                        and c1[0] != c2[1]
                        and c1[1] != c2[0]
                        and c1[1] != c2[1]
                    ):
                        break
                sim_hands[r_idx_1] = c1
                sim_hands[r_idx_2] = c2
                dealt_range_cards = (c1[0], c1[1], c2[0], c2[1])
            else:
                while True:
                    sampled = [
                        rng.choices(cl, weights=wl, k=1)[0] if is_w else rng.choice(cl)
                        for cl, wl, is_w in zip(r_combos_multi, r_weights_multi, r_is_w_multi)
                    ]
                    used_set: set[Card] = set()
                    conflict = False
                    for c in sampled:
                        if c[0] in used_set or c[1] in used_set:
                            conflict = True
                            break
                        used_set.add(c[0])
                        used_set.add(c[1])
                    if not conflict:
                        for idx, c in zip(range_indices, sampled):
                            sim_hands[idx] = c
                        dealt_range_cards = tuple(used_set)
                        break

            if cards_from_stub > 0:
                stub = [c for c in base_stub if c not in dealt_range_cards]
                dealt = rng.sample(stub, cards_from_stub)
                community_dealt = dealt[:board_needed]
                rand_idx = board_needed
                for i, k in enumerate(player_kinds):
                    if k == "random":
                        sim_hands[i] = (dealt[rand_idx], dealt[rand_idx + 1])
                        rand_idx += 2
            else:
                community_dealt = []

            full_board = board_cards + community_dealt
            scores = [
                score_7(full_board + [sim_hands[i][0], sim_hands[i][1]])  # type: ignore[index]
                for i in range(num_players)
            ]

            max_score = max(scores)
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

            if tracking:
                for p in range(num_players):
                    x = pot_share if p in winners else 0.0
                    delta = x - welford_mean[p]
                    welford_mean[p] += delta / sim_idx
                    delta2 = x - welford_mean[p]
                    welford_m2[p] += delta * delta2

                if sim_idx >= _MIN_CONVERGENCE_SIMS and sim_idx % check_interval == 0:
                    if on_progress is not None:
                        current_eqs = [
                            equity_sums[p] / sim_idx for p in range(num_players)
                        ]
                        on_progress(
                            ProgressInfo(
                                sims_completed=sim_idx,
                                sims_total=num_sims,
                                pct_complete=sim_idx / num_sims,
                                current_equities=current_eqs,
                                elapsed_seconds=time.perf_counter() - t_start,
                            )
                        )

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

            if (
                on_progress is not None
                and not tracking
                and sim_idx % progress_interval == 0
            ):
                current_eqs = [
                    equity_sums[p] / sim_idx for p in range(num_players)
                ]
                on_progress(
                    ProgressInfo(
                        sims_completed=sim_idx,
                        sims_total=num_sims,
                        pct_complete=sim_idx / num_sims,
                        current_equities=current_eqs,
                        elapsed_seconds=time.perf_counter() - t_start,
                    )
                )

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
    hero: Sequence[Card] | Range | str,
    villain: Sequence[Card] | Range | str | None = None,
    board: Sequence[Card | str] | str | None = None,
    *,
    num_sims: int = 10_000,
    seed: int | None = None,
    convergence_threshold: float | None = None,
    check_interval: int = 100,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 500,
) -> EquityResult:
    """Calculate hero's equity against villain via Monte Carlo simulation.

    Supports specific hands, ranges (as :class:`Range` or range syntax strings),
    and random hands (when villain is ``None``).

    Parameters
    ----------
    hero:
        Hero's hole cards (2-card sequence), :class:`Range`, or range string.
    villain:
        Villain's hole cards, :class:`Range`, range string, or ``None`` to
        evaluate against any two unknown cards.
    board:
        Known community cards (0 to 5), as a sequence of cards, a string,
        or ``None`` (preflop).
    num_sims:
        Maximum number of Monte Carlo simulations to run. Default is 10,000.
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
        The simulation results with win/tie/loss counts and equity for Hero.

    Raises
    ------
    ValueError
        If inputs are invalid (wrong card count, duplicates, empty ranges, etc.).

    Examples
    --------
    >>> from poker_equity.card import Card
    >>> hero = [Card.from_str("Ah"), Card.from_str("Kh")]
    >>> villain = [Card.from_str("7s"), Card.from_str("2d")]
    >>> result = calculate_equity(hero, villain, seed=42)
    >>> result.equity > 0.5  # AKs is a big favorite over 72o
    True
    >>> range_result = calculate_equity(hero="AA", villain="KK", seed=42)
    >>> range_result.equity > 0.80
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


def range_vs_range_equity(
    range1: Range | str | Sequence[Card],
    range2: Range | str | Sequence[Card],
    board: Sequence[Card | str] | str | None = None,
    num_sims: int = 10_000,
    *,
    seed: int | None = None,
    convergence_threshold: float | None = None,
    check_interval: int = 100,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 500,
) -> list[EquityResult]:
    """Calculate equity between two ranges (or hands) via Monte Carlo simulation.

    In Texas Hold'em, evaluating range vs range accounts for card removal
    (blockers), board interaction, and runout distributions across all
    possible combinations.

    Parameters
    ----------
    range1 : Range | str | Sequence[Card]
        Player 1's starting range or specific hand.  Can be a :class:`Range`
        instance, a standard poker range string (e.g. ``"QQ+, AKs"``),
        or a 2-card sequence.
    range2 : Range | str | Sequence[Card]
        Player 2's starting range or specific hand.
    board : Sequence[Card | str] | str | None, optional
        Community cards dealt so far (0 to 5 cards), as a string
        (e.g. ``"Ah Kd 2s"``), a sequence of cards/strings, or ``None`` (preflop).
        If an integer is passed here by positional invocation, it is
        interpreted as *num_sims*.
    num_sims : int, default 10,000
        Maximum number of Monte Carlo simulations to run.
    seed : int | None, optional
        Optional PRNG seed for reproducible results.
    convergence_threshold : float | None, optional
        If set, the simulation halts early once the 95% confidence interval
        half-width for both players drops below this threshold.
    check_interval : int, default 100
        How often (in simulations) to check convergence.
    on_progress : ProgressCallback | None, optional
        Optional callback invoked with a :class:`ProgressInfo` snapshot.
    progress_interval : int, default 500
        How often (in simulations) to invoke *on_progress*.

    Returns
    -------
    list[EquityResult]
        A 2-element list containing ``[range1_result, range2_result]``.

    Examples
    --------
    >>> from poker_equity.simulation import range_vs_range_equity
    >>> r1, r2 = range_vs_range_equity("AA", "KK", num_sims=5000, seed=42)
    >>> r1.equity > 0.80
    True
    >>> r1.equity + r2.equity == 1.0
    True
    """
    if isinstance(board, int):
        num_sims = board
        board = None

    return calculate_equities(
        hands=[range1, range2],
        board=board,
        num_sims=num_sims,
        seed=seed,
        convergence_threshold=convergence_threshold,
        check_interval=check_interval,
        on_progress=on_progress,
        progress_interval=progress_interval,
    )

