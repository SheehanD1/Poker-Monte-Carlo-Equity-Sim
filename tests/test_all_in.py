"""Unit tests for all-in equity and Run-It-Twice / Run-It-N-Times calculations."""

import pytest

from poker_equity import (
    Card,
    RunItTimesResult,
    exact_run_it_twice_turn,
    simulate_run_it_n,
    simulate_run_it_twice,
)


class TestRunItTimesResultProperties:
    """Tests for the RunItTimesResult dataclass methods and properties."""

    def test_properties_and_expected_payouts(self):
        """hero_equity, villain_equity, scoops, expected_payouts, and summary."""
        res = RunItTimesResult(
            equities=[0.65, 0.35],
            scoop_pcts=[0.45, 0.15],
            split_pct=0.40,
            board_equities=[[0.65, 0.35], [0.65, 0.35]],
            payout_distributions=[{0.0: 0.15, 0.5: 0.40, 1.0: 0.45}, {0.0: 0.45, 0.5: 0.40, 1.0: 0.15}],
            variance_single=0.2275,
            variance_multi=0.1138,
            variance_reduction=49.98,
            num_runs=2,
            num_sims=5000,
            is_exact=False,
        )

        assert res.hero_equity == 0.65
        assert res.villain_equity == 0.35
        assert res.hero_scoop_pct == 0.45
        assert res.villain_scoop_pct == 0.15

        payouts = res.expected_payouts(pot_size=200.0)
        assert payouts[0] == pytest.approx(130.0)
        assert payouts[1] == pytest.approx(70.0)

        with pytest.raises(ValueError, match="pot_size cannot be negative"):
            res.expected_payouts(-50.0)

        summary = res.summary()
        assert "Run It 2 Times Analysis" in summary
        assert "65.00%" in summary
        assert "49.98%" in summary
        assert "Hero scoops" in summary
        assert "Split pot" in summary


class TestSimulateRunItTwicePreflop:
    """Tests for preflop all-in spots running it twice."""

    def test_aa_vs_kk_preflop(self):
        """AA vs KK preflop running it twice."""
        res = simulate_run_it_twice(
            hero="Ah As",
            villain="Kh Ks",
            board=None,
            num_sims=3000,
            seed=42,
        )

        assert isinstance(res, RunItTimesResult)
        assert res.num_runs == 2
        assert res.is_exact is False

        # Equities must sum to 1.0
        assert sum(res.equities) == pytest.approx(1.0)
        # AA is roughly ~81% favorite vs KK
        assert 0.75 < res.hero_equity < 0.87
        assert 0.13 < res.villain_equity < 0.25

        # AA scoops most of the time, KK scoops rarely
        assert res.hero_scoop_pct > 0.60
        assert res.villain_scoop_pct < 0.10

        # Variance reduction should be substantial (typically 40% - 55%)
        assert res.variance_reduction > 35.0
        assert res.variance_multi < res.variance_single


class TestSimulateRunItTwiceFlop:
    """Tests for flop all-in spots running it twice."""

    def test_flush_draw_vs_overpair_flop(self):
        """Ah Kh (draw) vs Qc Qd (overpair) on 2h 7h 8c."""
        res = simulate_run_it_twice(
            hero="Ah Kh",
            villain="Qc Qd",
            board="2h 7h 8c",
            num_sims=3000,
            seed=42,
        )

        assert sum(res.equities) == pytest.approx(1.0)
        # 15 outs gives Ah Kh ~54% equity
        assert 0.48 < res.hero_equity < 0.60

        # Both players frequently split the pot (1 board each)
        assert res.split_pct > 0.30

        # Payout distribution should contain 0.0, 0.5, and 1.0
        dist = res.payout_distributions[0]
        assert 0.0 in dist
        assert 0.5 in dist
        assert 1.0 in dist
        assert sum(dist.values()) == pytest.approx(1.0)


class TestExactRunItTwiceTurn:
    """Tests for exact enumeration of Run-It-Twice on the turn."""

    def test_exact_turn_enumeration(self):
        """Exact combinatorial calculation on 4-card turn board."""
        hero = "Ah Kh"
        villain = "Qc Qd"
        turn_board = "2h 7h 8c 3c"

        exact_res = exact_run_it_twice_turn(hero, villain, turn_board)

        assert exact_res.is_exact is True
        assert exact_res.num_sims == 946  # C(44, 2) = 946
        assert exact_res.num_runs == 2

        # 15 outs out of 44 cards = 15/44 = 0.340909
        expected_hero_equity = 15.0 / 44.0
        assert exact_res.hero_equity == pytest.approx(expected_hero_equity)
        assert exact_res.villain_equity == pytest.approx(1.0 - expected_hero_equity)

        # Outcomes sum to 1.0
        assert exact_res.hero_scoop_pct + exact_res.villain_scoop_pct + exact_res.split_pct == pytest.approx(1.0)
        assert exact_res.hero_scoop_pct == pytest.approx(105.0 / 946.0)  # C(15, 2) = 105
        assert exact_res.villain_scoop_pct == pytest.approx(406.0 / 946.0)  # C(29, 2) = 406
        assert exact_res.split_pct == pytest.approx(435.0 / 946.0)  # 15 * 29 = 435

        # Significant variance reduction (~52%)
        assert exact_res.variance_reduction == pytest.approx(52.22, abs=1.0)

    def test_simulation_converges_to_exact_turn(self):
        """Monte Carlo simulation closely matches exact combinatorial result."""
        hero = "Ah Kh"
        villain = "Qc Qd"
        turn_board = "2h 7h 8c 3c"

        exact_res = exact_run_it_twice_turn(hero, villain, turn_board)
        sim_res = simulate_run_it_twice(hero, villain, turn_board, num_sims=10_000, seed=123)

        assert abs(sim_res.hero_equity - exact_res.hero_equity) < 0.015
        assert abs(sim_res.split_pct - exact_res.split_pct) < 0.02
        assert abs(sim_res.variance_reduction - exact_res.variance_reduction) < 3.0


class TestSimulateRunItNMultiWay:
    """Tests for multi-way pots and running it 3 times."""

    def test_three_players_run_it_twice(self):
        """3 players running it twice on flop."""
        res = simulate_run_it_n(
            hands=["Ah As", "Kh Ks", "Qh Qs"],
            board="Jh 8h 2c",
            num_runs=2,
            num_sims=2000,
            seed=42,
        )

        assert len(res.equities) == 3
        assert sum(res.equities) == pytest.approx(1.0)
        assert len(res.scoop_pcts) == 3

    def test_heads_up_run_it_three_times(self):
        """Running it 3 times (num_runs=3)."""
        res = simulate_run_it_n(
            hands=["Ah As", "Kh Ks"],
            board="2h 7h 8c",
            num_runs=3,
            num_sims=2000,
            seed=42,
        )

        assert res.num_runs == 3
        assert len(res.board_equities) == 3
        # Running 3 times reduces variance even more than 2 times
        assert res.variance_reduction > 55.0


class TestAllInValidationErrors:
    """Tests for invalid arguments and boundary conditions."""

    def test_fewer_than_two_players_raises(self):
        """Requires at least 2 hands."""
        with pytest.raises(ValueError, match="At least 2 players are required"):
            simulate_run_it_n(hands=["Ah As"])

    def test_invalid_hand_length_raises(self):
        """Hand with 1 or 3 cards raises ValueError."""
        with pytest.raises(ValueError, match="must contain exactly 2 cards"):
            simulate_run_it_n(hands=["Ah", "Kh Ks"])

    def test_invalid_board_length_raises(self):
        """Board with 2 or 5 cards raises ValueError."""
        with pytest.raises(ValueError, match="must have 0.*3.*or 4"):
            simulate_run_it_twice("Ah As", "Kh Ks", board="2h 7h")
        with pytest.raises(ValueError, match="must have 0.*3.*or 4"):
            simulate_run_it_twice("Ah As", "Kh Ks", board="2h 7h 8c 3c 4c")

    def test_exact_turn_wrong_board_raises(self):
        """exact_run_it_twice_turn requires exactly 4 cards."""
        with pytest.raises(ValueError, match="requires exactly 4 board cards"):
            exact_run_it_twice_turn("Ah As", "Kh Ks", board="2h 7h 8c")

    def test_duplicate_cards_raises(self):
        """Duplicate cards across hands or board raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate cards detected"):
            simulate_run_it_twice("Ah As", "Ah Ks", board="2h 7h 8c")

    def test_invalid_runs_or_sims_raises(self):
        """num_runs < 2 or num_sims < 1 raise ValueError."""
        with pytest.raises(ValueError, match="num_runs must be at least 2"):
            simulate_run_it_n(["Ah As", "Kh Ks"], num_runs=1)
        with pytest.raises(ValueError, match="num_sims must be at least 1"):
            simulate_run_it_n(["Ah As", "Kh Ks"], num_sims=0)

    def test_too_many_runs_exceeds_deck_raises(self):
        """Running it 20 times requires 100 cards, exceeding deck size."""
        with pytest.raises(ValueError, match="Not enough cards in deck"):
            simulate_run_it_n(["Ah As", "Kh Ks"], num_runs=20)
