"""Parallel Monte Carlo simulation engine.

This module provides a multiprocessing wrapper around the core
simulation engine to speed up calculations on multi-core systems.
"""

from __future__ import annotations

import multiprocessing
import os
from typing import Sequence

from poker_equity.card import Card
from poker_equity.simulation import (
    EquityResult,
    _build_equity_result,
    _validate_multi_inputs,
    calculate_equities,
)


def _worker(
    hands: Sequence[Sequence[Card] | None],
    board: Sequence[Card] | None,
    num_sims: int,
    seed: int | None,
) -> list[EquityResult]:
    """Worker function for parallel execution."""
    return calculate_equities(
        hands=hands,
        board=board,
        num_sims=num_sims,
        seed=seed,
    )


def calculate_equities_parallel(
    hands: Sequence[Sequence[Card] | None],
    board: Sequence[Card] | None = None,
    *,
    num_sims: int = 10_000,
    num_workers: int | None = None,
    seed: int | None = None,
) -> list[EquityResult]:
    """Calculate equities in parallel across multiple CPU cores.
    
    Splits the total number of simulations across `num_workers` processes.
    
    Parameters
    ----------
    hands:
        List of hole cards for each player. Use ``None`` for random hands.
    board:
        Known community cards (0 to 5).
    num_sims:
        Total number of Monte Carlo simulations to run.
    num_workers:
        Number of parallel worker processes. Defaults to os.cpu_count().
    seed:
        Optional PRNG seed for reproducible results. If provided, each worker
        is seeded deterministically based on this master seed.
        
    Returns
    -------
    list[EquityResult]
        The aggregated simulation results for each player.
    """
    if board is None:
        board = []
        
    _validate_multi_inputs(hands, board)

    if num_sims <= 0:
        raise ValueError(f"num_sims must be positive, got {num_sims}")
        
    if num_workers is None:
        num_workers = os.cpu_count() or 1
        
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")
        
    # If the work is too small, just run synchronously
    if num_sims < num_workers:
        return calculate_equities(hands, board, num_sims=num_sims, seed=seed)
        
    sims_per_worker = num_sims // num_workers
    remainder = num_sims % num_workers
    
    tasks = []
    for i in range(num_workers):
        worker_sims = sims_per_worker + (1 if i < remainder else 0)
        worker_seed = seed + i if seed is not None else None
        tasks.append((hands, board, worker_sims, worker_seed))
        
    # Run tasks in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        worker_results_list = pool.starmap(_worker, tasks)
        
    # Aggregate results
    num_players = len(hands)
    total_wins = [0] * num_players
    total_ties = [0] * num_players
    total_losses = [0] * num_players
    total_equity_sums = [0.0] * num_players
    
    for worker_results in worker_results_list:
        for p in range(num_players):
            res = worker_results[p]
            total_wins[p] += res.wins
            total_ties[p] += res.ties
            total_losses[p] += res.losses
            # Recover the equity sum from the worker's average equity
            total_equity_sums[p] += res.equity * res.num_sims
            
    return [
        _build_equity_result(
            total_wins[p],
            total_ties[p],
            total_losses[p],
            num_sims,
            total_equity_sums[p] / num_sims,
        )
        for p in range(num_players)
    ]


def calculate_equity_parallel(
    hero: Sequence[Card],
    villain: Sequence[Card] | None = None,
    board: Sequence[Card] | None = None,
    *,
    num_sims: int = 10_000,
    num_workers: int | None = None,
    seed: int | None = None,
) -> EquityResult:
    """Calculate hero's equity against villain via parallel Monte Carlo simulation.
    
    This is a convenience wrapper around :func:`calculate_equities_parallel` 
    for heads-up scenarios.
    """
    results = calculate_equities_parallel(
        hands=[hero, villain],
        board=board,
        num_sims=num_sims,
        num_workers=num_workers,
        seed=seed,
    )
    return results[0]
