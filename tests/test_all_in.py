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


class TestRunItTimesMathematicalIdentities:
    """Tests verifying fundamental mathematical properties of Run-It-N-Times."""

    @pytest.mark.parametrize(
        "hands,board,runs",
        [
            (["Ah As", "Kh Ks"], None, 2),
            (["Ah Kh", "Qc Qd"], "2h 7h 8c", 2),
            (["Ah As", "Kh Ks", "Qh Qs"], "Jh 8h 2c", 2),
            (["Ah As", "Kh Ks"], "2h 7h 8c", 3),
        ],
    )
    def test_total_equity_and_board_equities_invariance(self, hands, board, runs):
        """Conservation of equity across all boards and players."""
        res = simulate_run_it_n(hands=hands, board=board, num_runs=runs, num_sims=2000, seed=42)

        # 1. Equities must sum to 1.0 across players
        assert sum(res.equities) == pytest.approx(1.0, abs=1e-5)

        # 2. Each individual board's equities must sum to 1.0
        assert len(res.board_equities) == runs
        for b_eqs in res.board_equities:
            assert len(b_eqs) == len(hands)
            assert sum(b_eqs) == pytest.approx(1.0, abs=1e-5)

        # 3. Overall player equity equals average of per-board equities
        for p_idx in range(len(hands)):
            avg_b_eq = sum(res.board_equities[b][p_idx] for b in range(runs)) / runs
            assert avg_b_eq == pytest.approx(res.equities[p_idx], abs=1e-5)

    def test_payout_distribution_first_moment_matches_equity(self):
        """First moment (mean) of discrete payout distribution equals overall equity."""
        res = simulate_run_it_twice("Ah Kh", "Qc Qd", board="2h 7h 8c", num_sims=3000, seed=42)

        for p_idx in range(2):
            eq = res.equities[p_idx]
            dist = res.payout_distributions[p_idx]

            # Probabilities sum to 1.0
            assert sum(dist.values()) == pytest.approx(1.0, abs=1e-5)

            # Expected payout fraction: sum(fraction * probability)
            expected_fraction = sum(frac * prob for frac, prob in dist.items())
            assert expected_fraction == pytest.approx(eq, abs=1e-4)


class TestRunItTwiceChoppedPotAndTies:
    """Tests for symmetrical situations and chopped boards."""

    def test_symmetric_hole_cards_chopped_pot(self):
        """Symmetric hands (Ah Kh vs As Ks on 2c 3c 4d) produce 50/50 equity and split/chop."""
        res = simulate_run_it_twice("Ah Kh", "As Ks", board="2c 3c 4d", num_sims=2500, seed=42)

        # Equities equal
        assert res.hero_equity == pytest.approx(0.50, abs=0.01)
        assert res.villain_equity == pytest.approx(0.50, abs=0.01)

        # Neither player scoops; pot is split or chopped across boards
        assert res.hero_scoop_pct == pytest.approx(0.0, abs=0.02)
        assert res.villain_scoop_pct == pytest.approx(0.0, abs=0.02)
        assert res.split_pct > 0.95

    def test_board_plays_high_chop(self):
        """When board completes a broadway flush or royal that cannot be beaten, pot is 100% chopped."""
        res = exact_run_it_twice_turn("2c 3c", "2d 3d", board="As Ks Qs Js")

        assert res.hero_equity == pytest.approx(0.50)
        assert res.villain_equity == pytest.approx(0.50)
        assert res.split_pct == pytest.approx(1.0)
        assert res.hero_scoop_pct == 0.0
        assert res.villain_scoop_pct == 0.0


class TestRunItNFourWayAllIn:
    """Tests for 4-way all-in spots running it twice and 4 times."""

    @pytest.fixture
    def four_way_flop_setup(self):
        hands = ["Ac Ad", "Kh Qh", "Jc Jd", "Ts 9s"]
        board = "Jh 2h 7c"
        return hands, board

    def test_four_players_run_it_twice_flop(self, four_way_flop_setup):
        """4 players running it twice on flop tracks all 4 distributions."""
        hands, board = four_way_flop_setup
        res = simulate_run_it_n(hands, board=board, num_runs=2, num_sims=1500, seed=42)

        assert len(res.equities) == 4
        assert sum(res.equities) == pytest.approx(1.0)
        assert len(res.scoop_pcts) == 4
        assert len(res.payout_distributions) == 4
        assert len(res.board_equities) == 2

        # Payouts sum to total pot
        pot = 1000.0
        payouts = res.expected_payouts(pot)
        assert sum(payouts) == pytest.approx(pot)

        # Middle set (Player 3) should have highest equity on Jh 2h 7c
        assert res.equities[2] > res.equities[0]  # Jc Jd beats Ac Ad on Jh 2h 7c
        assert res.equities[2] > res.equities[1]  # Jc Jd favorite vs flush draw
        assert res.variance_reduction > 40.0

    def test_four_players_run_it_four_times_variance_drop(self, four_way_flop_setup):
        """Running it 4 times achieves even greater variance reduction (>70%)."""
        hands, board = four_way_flop_setup
        res_2 = simulate_run_it_n(hands, board=board, num_runs=2, num_sims=1000, seed=42)
        res_4 = simulate_run_it_n(hands, board=board, num_runs=4, num_sims=1000, seed=42)

        assert len(res_4.board_equities) == 4
        assert res_4.variance_reduction > res_2.variance_reduction
        assert res_4.variance_reduction > 65.0


class TestExactTurnTheoreticalVarianceIdentity:
    """Mathematical verification of hypergeometric turn variance reduction theorem."""

    @pytest.mark.parametrize(
        "hero,villain,board,expected_outs",
        [
            ("Ah Kh", "Qc Qd", "2h 7h 8c 3c", 15),  # 15 outs (flush + 2 overcards)
            ("Ah 2h", "Ks Kd", "8h 5h Jc 3c", 9),   # 9 flush outs
            ("9h 8h", "As Kd", "2c 6d 7s Qs", 14),  # Gutshot + pair outs
        ],
    )
    def test_hypergeometric_variance_reduction_closed_form(
        self, hero, villain, board, expected_outs
    ):
        """For all clean turn spots, exact sample variance reduction equals 47/90 (52.2222%)."""
        res = exact_run_it_twice_turn(hero, villain, board)

        # Closed form: 47 / 90 * 100% = 52.222222%
        expected_vr = (47.0 / 90.0) * 100.0
        assert res.variance_reduction == pytest.approx(expected_vr, abs=1e-4)


class TestRunItTimesResultFormattingAndEdgeCases:
    """Tests for edge cases, scaling, and summary formatting in RunItTimesResult."""

    def test_summary_multiway_formatting(self):
        """summary() formats 3-way board breakdown and player names correctly."""
        res = simulate_run_it_n(
            hands=["Ah As", "Kh Ks", "Qh Qs"],
            board="Jh 8h 2c",
            num_runs=3,
            num_sims=1000,
            seed=42,
        )
        text = res.summary()
        assert "Run It 3 Times Analysis" in text
        assert "Hero (Player 1)" in text
        assert "Player 2" in text
        assert "Player 3" in text
        assert "Hero scoops" in text
        assert "Player 3 scoops" in text
        assert "Board 1:" in text
        assert "Board 2:" in text
        assert "Board 3:" in text

    def test_expected_payouts_zero_and_scaling(self):
        """expected_payouts handles zero pot and scales strictly linearly."""
        res = simulate_run_it_twice("Ah As", "Kh Ks", num_sims=1000, seed=42)

        # Zero pot
        zero_payouts = res.expected_payouts(0.0)
        assert zero_payouts == [0.0, 0.0]

        # Linear scaling
        p100 = res.expected_payouts(100.0)
        p500 = res.expected_payouts(500.0)
        assert p500[0] == pytest.approx(5.0 * p100[0])
        assert p500[1] == pytest.approx(5.0 * p100[1])

