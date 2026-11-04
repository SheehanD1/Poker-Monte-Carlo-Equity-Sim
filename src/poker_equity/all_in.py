"""All-in equity and Run-It-Twice (or Run-It-N-Times) simulation for Texas Hold'em.

In all-in situations prior to the river, players frequently agree to "run it twice"
(or N times). The remaining community cards are dealt independently across
multiple boards from the remaining cards in the deck, and the pot is divided
equally among the boards.

Running it multiple times preserves the exact mathematical expected value (EV)
while significantly reducing variance (risk).
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Sequence

from poker_equity.card import Card
from poker_equity.constants import FULL_DECK_LIST, cards_to_str, parse_cards
from poker_equity.evaluator import score_7


# =========================================================================
# Result Data Structures
# =========================================================================


@dataclass(frozen=True, slots=True)
class RunItTimesResult:
    """The result of an all-in Run-It-N-Times simulation or exact calculation.

    Attributes
    ----------
    equities : list[float]
        The overall equity (average pot fraction won) for each player across all runs.
    scoop_pcts : list[float]
        Probability that each player wins (or chops) ALL boards, scooping the whole pot.
    split_pct : float
        Probability that the pot is split (at least two different players win a board).
    board_equities : list[list[float]]
        Per-board equity breakdown: ``board_equities[board_idx][player_idx]``.
    payout_distributions : list[dict[float, float]]
        Frequency of each payout fraction for each player (e.g. {0.0: 0.43, 0.5: 0.46, 1.0: 0.11}).
    variance_single : float
        Sample or exact payout variance when running it once (single board).
    variance_multi : float
        Sample or exact payout variance when running it N times.
    variance_reduction : float
        Percentage variance reduction: ``(1 - variance_multi / variance_single) * 100``.
    num_runs : int
        Number of boards dealt per simulation (e.g. 2 for Run It Twice).
    num_sims : int
        Number of simulations executed (or total combo count if exact).
    is_exact : bool
        Whether this result was calculated by full combinatorial enumeration.
    """

    equities: list[float]
    scoop_pcts: list[float]
    split_pct: float
    board_equities: list[list[float]]
    payout_distributions: list[dict[float, float]]
    variance_single: float
    variance_multi: float
    variance_reduction: float
    num_runs: int
    num_sims: int
    is_exact: bool = False

    @property
    def hero_equity(self) -> float:
        """Hero's (player 0) overall equity."""
        return self.equities[0]

    @property
    def villain_equity(self) -> float:
        """Villain's (player 1) overall equity in a heads-up pot."""
        return self.equities[1] if len(self.equities) > 1 else 0.0

    @property
    def hero_scoop_pct(self) -> float:
        """Probability of Hero scooping all boards."""
        return self.scoop_pcts[0]

    @property
    def villain_scoop_pct(self) -> float:
        """Probability of Villain scooping all boards."""
        return self.scoop_pcts[1] if len(self.scoop_pcts) > 1 else 0.0

    def expected_payouts(self, pot_size: float) -> list[float]:
        """Calculate the expected monetary/chip payout for each player.

        Parameters
        ----------
        pot_size : float
            Total pot size in chips or currency.

        Returns
        -------
        list[float]
            Expected payout for each player.
        """
        if pot_size < 0:
            raise ValueError(f"pot_size cannot be negative, got {pot_size}")
        return [pot_size * eq for eq in self.equities]

    def summary(self) -> str:
        """Format a comprehensive text report of the Run-It-N-Times results."""
        header = f"Run It {self.num_runs} Times Analysis"
        method_str = "Exact Enumeration" if self.is_exact else f"{self.num_sims:,} Simulations"
        lines = [
            f"{header} ({method_str})",
            "=" * len(header),
            "Overall Equities:",
        ]
        for p_idx, eq in enumerate(self.equities):
            label = "Hero (Player 1)" if p_idx == 0 else f"Player {p_idx + 1}"
            lines.append(f"  {label:<16}: {eq * 100:.2f}%")

        lines.append("Outcomes:")
        for p_idx, scoop in enumerate(self.scoop_pcts):
            label = "Hero scoops" if p_idx == 0 else f"Player {p_idx + 1} scoops"
            lines.append(f"  {label:<16}: {scoop * 100:.2f}%")
        lines.append(f"  {'Split pot':<16}: {self.split_pct * 100:.2f}%")

        lines.append("Variance Reduction:")
        lines.append(f"  Single run variance : {self.variance_single:.4f}")
        lines.append(f"  Run It {self.num_runs} variance  : {self.variance_multi:.4f}")
        lines.append(f"  Variance reduction  : {self.variance_reduction:.2f}%")

        if len(self.board_equities) > 1:
            lines.append("Per-Board Breakdown:")
            for b_idx, b_eqs in enumerate(self.board_equities):
                eq_strs = [f"P{p+1}: {e * 100:.1f}%" for p, e in enumerate(b_eqs)]
                lines.append(f"  Board {b_idx + 1}: {', '.join(eq_strs)}")

        return "\n".join(lines)


# =========================================================================
# Input Parsing Helpers
# =========================================================================


def _coerce_hands_and_board(
    hands: Sequence[Sequence[Card | str] | str],
    board: Sequence[Card | str] | str | None,
) -> tuple[list[list[Card]], list[Card]]:
    """Parse, normalize, and validate player hands and community board."""
    if len(hands) < 2:
        raise ValueError(f"At least 2 players are required for all-in simulation, got {len(hands)}")

    parsed_hands: list[list[Card]] = []
    for idx, h in enumerate(hands):
        if isinstance(h, str):
            c_list = parse_cards(h)
        elif isinstance(h, (list, tuple)):
            c_list = [Card.from_str(c) if isinstance(c, str) else c for c in h]
        else:
            raise TypeError(f"Hand at index {idx} must be a card string or sequence of Cards, got {type(h).__name__}")

        if len(c_list) != 2:
            raise ValueError(f"Each hand must contain exactly 2 cards, hand {idx} has {len(c_list)}")
        parsed_hands.append(c_list)

    if board is None:
        board_cards: list[Card] = []
    elif isinstance(board, str):
        board_cards = parse_cards(board)
    elif isinstance(board, (list, tuple)):
        board_cards = [Card.from_str(c) if isinstance(c, str) else c for c in board]
    else:
        raise TypeError(f"Board must be a card string, sequence, or None, got {type(board).__name__}")

    if len(board_cards) not in (0, 3, 4):
        raise ValueError(
            f"All-in board must have 0 (preflop), 3 (flop), or 4 (turn) cards, got {len(board_cards)}"
        )

    all_cards = [c for h in parsed_hands for c in h] + board_cards
    if len(set(all_cards)) != len(all_cards):
        raise ValueError(f"Duplicate cards detected: {cards_to_str(all_cards)}")

    return parsed_hands, board_cards


# =========================================================================
# Core Simulation
# =========================================================================


def simulate_run_it_n(
    hands: Sequence[Sequence[Card | str] | str],
    board: Sequence[Card | str] | str | None = None,
    num_runs: int = 2,
    num_sims: int = 10_000,
    seed: int | None = None,
) -> RunItTimesResult:
    """Simulate an all-in pot run N times using Monte Carlo simulation.

    Parameters
    ----------
    hands : Sequence[Sequence[Card | str] | str]
        Player hole cards (2 or more players, 2 cards each).
    board : Sequence[Card | str] | str | None, default=None
        Existing community board cards (0 for preflop, 3 for flop, 4 for turn).
    num_runs : int, default=2
        Number of boards to run (e.g. 2 for Run It Twice).
    num_sims : int, default=10_000
        Number of Monte Carlo simulations to execute.
    seed : int or None, default=None
        Random seed for deterministic output.

    Returns
    -------
    RunItTimesResult
        Comprehensive statistics on equities, scoop probabilities, split
        probabilities, and variance reduction.

    Raises
    ------
    ValueError
        If ``num_runs < 2``, ``num_sims < 1``, or not enough cards remain in deck.
    """
    if num_runs < 2:
        raise ValueError(f"num_runs must be at least 2, got {num_runs}")
    if num_sims < 1:
        raise ValueError(f"num_sims must be at least 1, got {num_sims}")

    parsed_hands, board_cards = _coerce_hands_and_board(hands, board)
    num_players = len(parsed_hands)

    cards_per_board_needed = 5 - len(board_cards)
    total_cards_needed = num_runs * cards_per_board_needed

    known_cards = {c for h in parsed_hands for c in h} | set(board_cards)
    unseen_deck = [c for c in FULL_DECK_LIST if c not in known_cards]

    if total_cards_needed > len(unseen_deck):
        raise ValueError(
            f"Not enough cards in deck for {num_runs} runs: needed {total_cards_needed}, "
            f"but only {len(unseen_deck)} unseen cards remain"
        )

    rng = random.Random(seed)

    # Accumulators
    # Total payout fraction won across all sims for each player
    player_total_payouts = [0.0] * num_players
    # Count of simulations where each player scooped all runs
    player_scoop_counts = [0] * num_players
    # Count of simulations where the pot was split across players
    split_sims_count = 0
    # Per-board equity sums: board_payout_sums[run_idx][player_idx]
    board_payout_sums = [[0.0] * num_players for _ in range(num_runs)]
    # Per-player payout values across all sims for empirical variance calculation
    player_sim_payouts = [[] for _ in range(num_players)]
    # Board 1 single-run payouts for Player 0 (Hero) to compute baseline single-run variance
    hero_single_run_payouts: list[float] = []

    # Pre-pack hand + base board cards
    base_eval_data = [h + board_cards for h in parsed_hands]

    weight_per_board = 1.0 / num_runs

    for _ in range(num_sims):
        dealt_stub = rng.sample(unseen_deck, total_cards_needed)

        # Track payout fraction won in this simulation for each player
        sim_payouts = [0.0] * num_players
        board_winners: list[list[int]] = []

        card_offset = 0
        for run_idx in range(num_runs):
            run_cards = dealt_stub[card_offset : card_offset + cards_per_board_needed]
            card_offset += cards_per_board_needed

            # Evaluate each player's 7-card hand on this board
            scores = [
                score_7(base_eval_data[p_idx] + run_cards)
                for p_idx in range(num_players)
            ]

            best_score = max(scores)
            winners = [p_idx for p_idx, sc in enumerate(scores) if sc == best_score]
            board_winners.append(winners)

            # Award board share
            board_share = weight_per_board / len(winners)
            for w in winners:
                sim_payouts[w] += board_share
                board_payout_sums[run_idx][w] += 1.0 / len(winners)

            if run_idx == 0:
                hero_single_share = (1.0 / len(winners)) if 0 in winners else 0.0
                hero_single_run_payouts.append(hero_single_share)

        # Accumulate simulation-level metrics
        for p_idx in range(num_players):
            player_total_payouts[p_idx] += sim_payouts[p_idx]
            player_sim_payouts[p_idx].append(sim_payouts[p_idx])

        # Check for scoop vs split
        # A player scoops if they are a winner on EVERY board
        scooper = None
        for p_idx in range(num_players):
            if all(p_idx in b_wins for b_wins in board_winners):
                scooper = p_idx
                player_scoop_counts[p_idx] += 1
                break

        if scooper is None:
            split_sims_count += 1

    # Calculate overall equities
    overall_equities = [payout / num_sims for payout in player_total_payouts]
    scoop_pcts = [cnt / num_sims for cnt in player_scoop_counts]
    split_pct = split_sims_count / num_sims

    per_board_equities = [
        [b_sum / num_sims for b_sum in board_sums]
        for board_sums in board_payout_sums
    ]

    # Calculate payout distributions (rounded to 4 decimal places)
    distributions: list[dict[float, float]] = []
    for p_idx in range(num_players):
        freqs: dict[float, int] = {}
        for val in player_sim_payouts[p_idx]:
            rounded_val = round(val, 4)
            freqs[rounded_val] = freqs.get(rounded_val, 0) + 1
        dist = {k: v / num_sims for k, v in sorted(freqs.items())}
        distributions.append(dist)

    # Calculate variances for Hero (player 0)
    hero_eq = overall_equities[0]

    # Baseline single-run variance: Var(X) = E[X^2] - (E[X])^2
    var_single = (
        sum((x - hero_eq) ** 2 for x in hero_single_run_payouts) / (num_sims - 1)
        if num_sims > 1
        else 0.0
    )

    # Multi-run variance
    var_multi = (
        sum((y - hero_eq) ** 2 for y in player_sim_payouts[0]) / (num_sims - 1)
        if num_sims > 1
        else 0.0
    )

    # Variance reduction percentage
    if var_single > 1e-12:
        var_reduction = max(0.0, (1.0 - (var_multi / var_single)) * 100.0)
    else:
        var_reduction = 0.0

    return RunItTimesResult(
        equities=overall_equities,
        scoop_pcts=scoop_pcts,
        split_pct=split_pct,
        board_equities=per_board_equities,
        payout_distributions=distributions,
        variance_single=var_single,
        variance_multi=var_multi,
        variance_reduction=var_reduction,
        num_runs=num_runs,
        num_sims=num_sims,
        is_exact=False,
    )


def simulate_run_it_twice(
    hero: Sequence[Card | str] | str,
    villain: Sequence[Card | str] | str,
    board: Sequence[Card | str] | str | None = None,
    num_sims: int = 10_000,
    seed: int | None = None,
) -> RunItTimesResult:
    """Simulate heads-up all-in equity running the board twice.

    Parameters
    ----------
    hero : Sequence[Card | str] | str
        Hero's hole cards (e.g. ``"Ah Kh"``).
    villain : Sequence[Card | str] | str
        Villain's hole cards (e.g. ``"Qc Qd"``).
    board : Sequence[Card | str] | str | None, default=None
        Existing community board cards (preflop, flop, or turn).
    num_sims : int, default=10_000
        Number of simulations.
    seed : int or None, default=None
        RNG seed.

    Returns
    -------
    RunItTimesResult
        Run It Twice analysis.

    Examples
    --------
    >>> from poker_equity.all_in import simulate_run_it_twice
    >>> res = simulate_run_it_twice("Ah Kh", "Qc Qd", "2h 7h 8c", num_sims=5000, seed=42)
    >>> res.hero_equity
    0.5...
    >>> res.variance_reduction > 40.0
    True
    """
    return simulate_run_it_n(
        hands=[hero, villain],
        board=board,
        num_runs=2,
        num_sims=num_sims,
        seed=seed,
    )


# =========================================================================
# Exact Enumeration for Turn Run-It-Twice
# =========================================================================


def exact_run_it_twice_turn(
    hero: Sequence[Card | str] | str,
    villain: Sequence[Card | str] | str,
    board: Sequence[Card | str] | str,
) -> RunItTimesResult:
    """Calculate exact mathematical Run-It-Twice equities on the turn.

    On the turn (4 board cards), exactly 1 card is needed per board.
    For two boards, there are :math:`\\binom{44}{2} = 946` unique card pairs.
    This function exhaustively enumerates all 946 river combinations to produce
    exact, error-free results.

    Parameters
    ----------
    hero : Sequence[Card | str] | str
        Hero's 2 hole cards.
    villain : Sequence[Card | str] | str
        Villain's 2 hole cards.
    board : Sequence[Card | str] | str
        The 4 community cards on the turn.

    Returns
    -------
    RunItTimesResult
        Exact mathematical results for Run It Twice.

    Raises
    ------
    ValueError
        If board does not have exactly 4 cards or cards contain duplicates.
    """
    parsed_hands, board_cards = _coerce_hands_and_board([hero, villain], board)
    if len(board_cards) != 4:
        raise ValueError(
            f"exact_run_it_twice_turn requires exactly 4 board cards (turn), got {len(board_cards)}"
        )

    hero_cards, villain_cards = parsed_hands[0], parsed_hands[1]
    known_cards = set(hero_cards + villain_cards + board_cards)
    unseen_deck = [c for c in FULL_DECK_LIST if c not in known_cards]
    assert len(unseen_deck) == 44

    combos = list(itertools.combinations(unseen_deck, 2))
    num_combos = len(combos)  # 946

    # Pre-evaluate all 44 single river cards for Hero and Villain
    single_payouts_hero: list[float] = []
    card_payouts_hero: dict[Card, float] = {}

    for c in unseen_deck:
        h_sc = score_7(hero_cards + board_cards + [c])
        v_sc = score_7(villain_cards + board_cards + [c])
        if h_sc > v_sc:
            pay = 1.0
        elif v_sc > h_sc:
            pay = 0.0
        else:
            pay = 0.5
        single_payouts_hero.append(pay)
        card_payouts_hero[c] = pay

    hero_single_mean = sum(single_payouts_hero) / len(single_payouts_hero)
    var_single = (
        sum((x - hero_single_mean) ** 2 for x in single_payouts_hero)
        / (len(single_payouts_hero) - 1)
    )

    # Outcomes for Hero and Villain across all 946 pairs
    hero_payouts: list[float] = []
    villain_payouts: list[float] = []

    hero_scoop_count = 0
    villain_scoop_count = 0
    split_count = 0

    for c1, c2 in combos:
        p1 = card_payouts_hero[c1]
        p2 = card_payouts_hero[c2]

        h_total = (p1 + p2) / 2.0
        v_total = 1.0 - h_total

        hero_payouts.append(h_total)
        villain_payouts.append(v_total)

        if h_total == 1.0:
            hero_scoop_count += 1
        elif v_total == 1.0:
            villain_scoop_count += 1
        else:
            split_count += 1

    hero_equity = sum(hero_payouts) / num_combos
    villain_equity = sum(villain_payouts) / num_combos

    # Distributions
    hero_freqs: dict[float, int] = {}
    villain_freqs: dict[float, int] = {}
    for h_pay, v_pay in zip(hero_payouts, villain_payouts):
        h_r, v_r = round(h_pay, 4), round(v_pay, 4)
        hero_freqs[h_r] = hero_freqs.get(h_r, 0) + 1
        villain_freqs[v_r] = villain_freqs.get(v_r, 0) + 1

    dists = [
        {k: v / num_combos for k, v in sorted(hero_freqs.items())},
        {k: v / num_combos for k, v in sorted(villain_freqs.items())},
    ]

    # Variances
    var_multi = sum((y - hero_equity) ** 2 for y in hero_payouts) / (num_combos - 1)
    var_reduction = max(0.0, (1.0 - (var_multi / var_single)) * 100.0) if var_single > 1e-12 else 0.0

    b_equities = [
        [hero_equity, villain_equity],
        [hero_equity, villain_equity],
    ]

    return RunItTimesResult(
        equities=[hero_equity, villain_equity],
        scoop_pcts=[hero_scoop_count / num_combos, villain_scoop_count / num_combos],
        split_pct=split_count / num_combos,
        board_equities=b_equities,
        payout_distributions=dists,
        variance_single=var_single,
        variance_multi=var_multi,
        variance_reduction=var_reduction,
        num_runs=2,
        num_sims=num_combos,
        is_exact=True,
    )
