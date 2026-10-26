"""Unit and integration tests for range weight and frequency support."""

from __future__ import annotations

import random

import pytest

from poker_equity.card import Card, Rank
from poker_equity.constants import parse_cards
from poker_equity.breakdown import range_vs_hand_breakdown
from poker_equity.range import (
    Combo,
    Range,
    canonical_combo,
    combo_from_str,
    parse_range_string,
    parse_weighted_range_string,
)
from poker_equity.simulation import calculate_equity, calculate_equities


# =========================================================================
# TestWeightedRangeParsing
# =========================================================================


class TestWeightedRangeParsing:
    """Tests for range string frequency/weight parsing syntax."""

    def test_colon_float_weight(self) -> None:
        weights = parse_weighted_range_string("AKs:0.5")
        assert len(weights) == 4
        assert all(w == 0.5 for w in weights.values())

    def test_colon_percent_weight(self) -> None:
        weights = parse_weighted_range_string("AKs:50%")
        assert len(weights) == 4
        assert all(w == 0.5 for w in weights.values())

    def test_colon_integer_percent_weight(self) -> None:
        # 50 without % is treated as 50%
        weights = parse_weighted_range_string("AKs:50")
        assert len(weights) == 4
        assert all(w == 0.5 for w in weights.values())

    def test_colon_unit_weight(self) -> None:
        weights = parse_weighted_range_string("AKs:1.0")
        assert len(weights) == 4
        assert all(w == 1.0 for w in weights.values())

    def test_at_syntax_weight(self) -> None:
        weights = parse_weighted_range_string("QQ+@0.75")
        # QQ, KK, AA = 18 combos
        assert len(weights) == 18
        assert all(w == 0.75 for w in weights.values())

    def test_bracket_syntax_weight(self) -> None:
        weights = parse_weighted_range_string("T9s-76s[0.25]")
        # T9s, 98s, 87s, 76s = 16 combos
        assert len(weights) == 16
        assert all(w == 0.25 for w in weights.values())

    def test_whitespace_around_delimiters(self) -> None:
        w1 = parse_weighted_range_string("AKs : 0.5, QQ+ @ 0.75, 88-66 [ 0.25 ]")
        assert len(w1) == 4 + 18 + 18
        r = Range.from_str("AKs : 0.5, QQ+ @ 0.75")
        assert r.get_weight("AhKh") == 0.5
        assert r.get_weight("QhQd") == 0.75

    def test_default_weight_is_one(self) -> None:
        weights = parse_weighted_range_string("AKs:0.5, QQ")
        # AKs should be 0.5, QQ should be 1.0
        aks_combos = Range.suited("A", "K").combos
        qq_combos = Range.pair("Q").combos
        for c in aks_combos:
            assert weights[c] == 0.5
        for c in qq_combos:
            assert weights[c] == 1.0

    def test_later_token_overwrites_weight(self) -> None:
        # QQ+ at 0.5, but AA at 1.0
        weights = parse_weighted_range_string("QQ+:0.5, AA:1.0")
        aa_combos = Range.pair("A").combos
        kk_combos = Range.pair("K").combos
        qq_combos = Range.pair("Q").combos
        for c in aa_combos:
            assert weights[c] == 1.0
        for c in kk_combos:
            assert weights[c] == 0.5
        for c in qq_combos:
            assert weights[c] == 0.5

    def test_zero_weight_removes_combos(self) -> None:
        # QQ+ without KK
        weights = parse_weighted_range_string("QQ+, KK:0.0")
        assert len(weights) == 12  # AA (6) + QQ (6)
        assert not any(c[0].rank == Rank.KING for c in weights)

    def test_specific_combo_with_weight(self) -> None:
        weights = parse_weighted_range_string("AhKd:0.33")
        c = canonical_combo(Card.from_str("Ah"), Card.from_str("Kd"))
        assert len(weights) == 1
        assert weights[c] == 0.33

    def test_parse_range_string_compatibility(self) -> None:
        combos = parse_range_string("AKs:0.5, QQ+:0.75")
        assert len(combos) == 4 + 18

    def test_invalid_weight_syntax_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_weighted_range_string("AKs:abc")

    def test_weight_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_weighted_range_string("AKs:150%")
        with pytest.raises(ValueError):
            parse_weighted_range_string("AKs:-0.5")


# =========================================================================
# TestWeightedRangeClass
# =========================================================================


class TestWeightedRangeClass:
    """Tests for the Range class with weights."""

    def test_init_from_mapping(self) -> None:
        c1 = canonical_combo(Card.from_str("Ah"), Card.from_str("Kh"))
        c2 = canonical_combo(Card.from_str("As"), Card.from_str("Ks"))
        r = Range({c1: 0.5, c2: 0.8})
        assert len(r) == 2
        assert r.is_weighted
        assert r.get_weight(c1) == 0.5
        assert r.get_weight(c2) == 0.8
        assert r.total_combos == pytest.approx(1.3)

    def test_init_with_weights_keyword_dict(self) -> None:
        c1 = canonical_combo(Card.from_str("Ah"), Card.from_str("Kh"))
        c2 = canonical_combo(Card.from_str("As"), Card.from_str("Ks"))
        r = Range([c1, c2], weights={c1: 0.25, c2: 0.75})
        assert r.get_weight(c1) == 0.25
        assert r.get_weight(c2) == 0.75
        assert r.total_combos == pytest.approx(1.0)

    def test_init_with_weights_keyword_sequence(self) -> None:
        r_base = Range.suited("A", "K")
        w_seq = [0.1, 0.2, 0.3, 0.4]
        r = Range(r_base.combos_list, weights=w_seq)
        assert r.is_weighted
        assert r.weights_list == (0.1, 0.2, 0.3, 0.4)
        assert r.total_combos == pytest.approx(1.0)

    def test_unweighted_range_properties(self) -> None:
        r = Range.from_str("AA")
        assert not r.is_weighted
        assert r.total_combos == 6.0
        assert all(w == 1.0 for w in r.weights.values())
        assert r.get_weight("AhAs") == 1.0
        assert r.get_weight("KhKd") == 0.0

    def test_from_str_weighted(self) -> None:
        r = Range.from_str("AKs:0.5, QQ:0.25")
        assert r.is_weighted
        assert len(r) == 10
        assert r.total_combos == pytest.approx(4 * 0.5 + 6 * 0.25)  # 2.0 + 1.5 = 3.5
        assert r.get_weight("AhKh") == 0.5
        assert r.get_weight("QhQd") == 0.25

    def test_from_weighted_matrix(self) -> None:
        m = {"AA": 0.5, "AKs": 0.25, "AKo": 0.1}
        r = Range.from_weighted_matrix(m)
        assert r.is_weighted
        assert r.total_combos == pytest.approx(6 * 0.5 + 4 * 0.25 + 12 * 0.1)  # 3.0 + 1.0 + 1.2 = 5.2
        assert r.get_weight("AhAs") == 0.5
        assert r.get_weight("AhKh") == 0.25
        assert r.get_weight("AhKd") == 0.1

    def test_from_weighted_matrix_unpaired_shorthand(self) -> None:
        # "AK": 0.5 should apply to both AKs and AKo
        r = Range.from_weighted_matrix({"AK": 0.5})
        assert len(r) == 16
        assert r.total_combos == pytest.approx(16 * 0.5)
        assert r.get_weight("AhKh") == 0.5
        assert r.get_weight("AhKd") == 0.5

    def test_to_matrix_weighted(self) -> None:
        r = Range.from_str("AA:0.5, AKs:0.25")
        matrix = r.to_matrix(include_zeros=False)
        assert matrix["AA"] == pytest.approx(0.5)
        assert matrix["AKs"] == pytest.approx(0.25)

    def test_remove_dead_cards_preserves_weights(self) -> None:
        r = Range.from_str("AA:0.5, KK:0.25")
        # Dead As blocks 3 of 6 AA combos, leaving 3 combos at 0.5
        r_dead = r.remove_dead_cards(["As"])
        assert len(r_dead) == 3 + 6
        assert r_dead.total_combos == pytest.approx(3 * 0.5 + 6 * 0.25)
        assert r_dead.get_weight("AhAd") == 0.5
        assert r_dead.get_weight("KhKd") == 0.25

    def test_set_operations_preserve_weights(self) -> None:
        r1 = Range.from_str("AA:0.5, KK:0.3")
        r2 = Range.from_str("AA:0.8, QQ:0.4")

        # Union takes max
        r_union = r1 | r2
        assert r_union.get_weight("AhAs") == pytest.approx(0.8)
        assert r_union.get_weight("KhKs") == pytest.approx(0.3)
        assert r_union.get_weight("QhQs") == pytest.approx(0.4)

        # Intersection takes min
        r_inter = r1 & r2
        assert len(r_inter) == 6
        assert r_inter.get_weight("AhAs") == pytest.approx(0.5)

        # Difference
        r_diff = r1 - r2
        assert set(r_diff.combos) == set(Range.pair("K").combos)
        assert r_diff.get_weight("KhKs") == pytest.approx(0.3)

    def test_repr_and_str(self) -> None:
        r = Range.from_str("AKs:0.5")
        assert "weighted" in repr(r)
        assert "weighted" in str(r)


# =========================================================================
# TestWeightedSampling
# =========================================================================


class TestWeightedSampling:
    """Tests for weighted combo sampling frequency distribution."""

    def test_sampling_proportions(self) -> None:
        # AA at 1.0 (6 combos), 72o at 0.05 (12 combos)
        # Expected weight sum: AA = 6.0, 72o = 0.6. Total = 6.6.
        # AA expected probability = 6.0 / 6.6 ~= 90.9%
        r = Range.from_str("AA:1.0, 72o:0.05")
        rng = random.Random(42)
        n = 5_000
        aa_count = 0
        for _ in range(n):
            combo = r.sample(rng)
            if combo[0].rank == Rank.ACE and combo[1].rank == Rank.ACE:
                aa_count += 1

        freq = aa_count / n
        expected = 6.0 / (6.0 + 12 * 0.05)
        assert abs(freq - expected) < 0.02


# =========================================================================
# TestWeightedSimulation
# =========================================================================


class TestWeightedSimulation:
    """Tests for Monte Carlo simulation with weighted ranges."""

    def test_weighted_range_vs_hand_simulation(self) -> None:
        # Hero has AhKd
        # Villain range 1: pure AA (100% equity for villain ~ 87%)
        res_pure_aa = calculate_equity("AhKd", "AA", num_sims=3_000, seed=123)

        # Villain range 2: mostly 72o, tiny AA
        # AhKd dominates 72o (~67% for hero)
        res_weighted = calculate_equity(
            "AhKd",
            "AA:0.05, 72o:1.0",
            num_sims=3_000,
            seed=123,
        )

        # Hero equity should be much higher against AA:0.05, 72o:1.0 than pure AA
        assert res_weighted.equity > res_pure_aa.equity + 0.35

    def test_range_vs_range_both_weighted(self) -> None:
        # Hero: AKs@0.5, Villain: QQ@0.5
        res = calculate_equities(
            ["AKs:0.5", "QQ:0.5"],
            num_sims=3_000,
            seed=42,
        )
        assert len(res) == 2
        # Classic coinflip spot: ~46% vs ~54%
        assert 0.40 <= res[0].equity <= 0.50
        assert 0.50 <= res[1].equity <= 0.60
        assert res[0].equity + res[1].equity == pytest.approx(1.0, abs=0.01)

    def test_multi_way_with_weighted_range(self) -> None:
        results = calculate_equities(
            ["AA", "KK:0.5", "QQ:0.25"],
            num_sims=2_000,
            seed=42,
        )
        assert len(results) == 3
        # AA should have highest equity
        assert results[0].equity > results[1].equity > results[2].equity


# =========================================================================
# TestWeightedBreakdown
# =========================================================================


class TestWeightedBreakdown:
    """Tests for RangeBreakdown with weighted combos."""

    def test_breakdown_overall_equity_is_weighted(self) -> None:
        # Range with AA at 1.0 and 72o at 0.1 against KK
        # AA vs KK is ~82% for range
        # 72o vs KK is ~13% for range
        # Overall equity must reflect weights: (6 * 0.82 + 12 * 0.1 * 0.13) / (6 + 1.2)
        r = Range.from_str("AA:1.0, 72o:0.1")
        bd = range_vs_hand_breakdown(r, "KhKd", num_sims_per_combo=500, seed=42)

        assert bd.range.is_weighted
        assert bd.total_combos == pytest.approx(6 * 1.0 + 12 * 0.1)

        # AA combos have ~82% equity
        aa_eqs = [res.equity for c, res in bd.items() if c[0].rank == Rank.ACE]
        aa_avg = sum(aa_eqs) / len(aa_eqs)
        assert aa_avg > 0.80

        # Weighted overall equity must be close to AA because 72o has low weight
        assert bd.overall_equity > 0.70

    def test_by_cell_weighted(self) -> None:
        r = Range.from_str("AA:0.5, 72o:0.5")
        bd = range_vs_hand_breakdown(r, "QhQd", num_sims_per_combo=500, seed=42)
        cells = bd.by_cell()
        assert "AA" in cells
        assert "72o" in cells
        assert cells["AA"] > 0.70
        assert cells["72o"] < 0.25
        assert cells["AA"] > cells["72o"]

    def test_by_cell_weights(self) -> None:
        r = Range.from_str("AA:0.5, AKs:0.25")
        bd = range_vs_hand_breakdown(r, "2h2c", num_sims_per_combo=200, seed=42)
        cell_weights = bd.by_cell_weights()
        assert cell_weights["AA"] == pytest.approx(6 * 0.5)
        assert cell_weights["AKs"] == pytest.approx(4 * 0.25)
