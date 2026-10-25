"""Unit tests for range vs. hand equity breakdown."""

from __future__ import annotations

import pytest

from poker_equity.card import Card
from poker_equity.range import Range, canonical_combo
from poker_equity.breakdown import (
    RangeBreakdown,
    hand_vs_range_breakdown,
    range_breakdown,
    range_vs_hand_breakdown,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

C = Card.from_str


# =========================================================================
# RangeBreakdown container structure and dictionary behavior tests
# =========================================================================


class TestRangeBreakdownStructure:
    def test_inherits_from_dict(self) -> None:
        """RangeBreakdown is a subclass of dict and supports dictionary operations."""
        bd = range_vs_hand_breakdown("QQ+, AKs", "AhKd", num_sims_per_combo=500, seed=42)
        assert isinstance(bd, dict)

    def test_correct_unblocked_combo_count(self) -> None:
        """Ah and Kd block 8 combos out of 22 in QQ+, AKs, leaving exactly 14."""
        bd = range_vs_hand_breakdown("QQ+, AKs", "AhKd", num_sims_per_combo=500, seed=42)
        # QQ: 6 combos
        # KK: 3 combos (Kd blocked)
        # AA: 3 combos (Ah blocked)
        # AKs: 2 combos (Ah and Kd blocked: only AsKs, AcKc remain)
        assert len(bd) == 14
        assert bd.num_combos == 14

    def test_getitem_by_combo_tuple(self) -> None:
        """Lookup by canonical Combo tuple."""
        bd = range_vs_hand_breakdown("AA", "KhKd", num_sims_per_combo=500, seed=42)
        combo = canonical_combo(C("As"), C("Ah"))
        assert combo in bd
        res = bd[combo]
        assert res.equity > 0.75

    def test_getitem_by_string(self) -> None:
        """Lookup by 4-char string works with canonical and non-canonical order."""
        bd = range_vs_hand_breakdown("AA", "KhKd", num_sims_per_combo=500, seed=42)
        res1 = bd["AsAh"]
        res2 = bd["AhAs"]
        assert res1 is res2
        assert res1.equity > 0.75

    def test_contains_checks(self) -> None:
        """Membership checks support Combo and string representations."""
        bd = range_vs_hand_breakdown("QQ+, AKs", "AhKd", num_sims_per_combo=500, seed=42)
        assert "AsAd" in bd
        assert "AsKs" in bd
        # AhKh is in AKs but Ah is in opponent's hand, so it's blocked
        assert "AhKh" not in bd
        assert "7c2d" not in bd
        assert "invalid" not in bd

    def test_summary_properties(self) -> None:
        """Verify hand, hand_str, board, and overall equity stats."""
        bd = range_vs_hand_breakdown("AA", "KhKd", num_sims_per_combo=500, seed=42)
        assert bd.hand_str == "KhKd"
        assert bd.hand == canonical_combo(C("Kh"), C("Kd"))
        assert bd.board == []
        assert 0.80 < bd.overall_equity < 0.84
        assert 0.78 < bd.overall_win_pct < 0.84
        assert pytest.approx(bd.overall_win_pct + bd.overall_tie_pct + bd.overall_loss_pct) == 1.0
        assert pytest.approx(bd.hero_equity + bd.villain_equity) == 1.0

    def test_repr(self) -> None:
        """Developer repr format."""
        bd = range_vs_hand_breakdown("AA", "KhKd", num_sims_per_combo=500, seed=42)
        rep = repr(bd)
        assert "RangeBreakdown" in rep
        assert "combos=6" in rep
        assert "hand='KhKd'" in rep


# =========================================================================
# Analysis methods (sorting, best/worst, cell grouping, summary table)
# =========================================================================


class TestRangeBreakdownAnalysis:
    def test_sorted_combos(self) -> None:
        """sorted_combos returns combos in descending or ascending equity order."""
        bd = range_vs_hand_breakdown("QQ+, AKs", "AhKd", num_sims_per_combo=500, seed=42)
        desc = bd.sorted_combos(descending=True)
        assert len(desc) == 14
        for i in range(len(desc) - 1):
            assert desc[i][1].equity >= desc[i + 1][1].equity

        asc = bd.sorted_combos(descending=False)
        assert len(asc) == 14
        for i in range(len(asc) - 1):
            assert asc[i][1].equity <= asc[i + 1][1].equity

    def test_best_and_worst_combos(self) -> None:
        """best_combos and worst_combos return the expected top and bottom slices."""
        bd = range_vs_hand_breakdown("QQ+, AKs", "AhKd", num_sims_per_combo=500, seed=42)
        best3 = bd.best_combos(3)
        worst3 = bd.worst_combos(3)
        assert len(best3) == 3
        assert len(worst3) == 3
        # AA combos have the highest equity against AhKd
        assert all(res.equity > 0.85 for _, res in best3)
        # AKs combos have the lowest equity against AhKd (split or slight underdog)
        assert all(res.equity < 0.65 for _, res in worst3)

    def test_by_cell_aggregation(self) -> None:
        """by_cell groups live combos into 169-grid cell categories and averages equity."""
        bd = range_vs_hand_breakdown("QQ+, AKs", "AhKd", num_sims_per_combo=500, seed=42)
        cells = bd.by_cell()
        assert set(cells.keys()) == {"AA", "KK", "QQ", "AKs"}
        # AA crushes AhKd
        assert cells["AA"] > 0.90
        # KK dominates AhKd (~65-70%)
        assert 0.63 < cells["KK"] < 0.72
        # QQ is a slight favorite over AhKd (~54-58%)
        assert 0.52 < cells["QQ"] < 0.60
        # AKs vs AKo is roughly even (~50-53%)
        assert 0.49 < cells["AKs"] < 0.55

    def test_by_cell_counts(self) -> None:
        """by_cell_counts maps each cell to its live combo count."""
        bd = range_vs_hand_breakdown("QQ+, AKs", "AhKd", num_sims_per_combo=500, seed=42)
        counts = bd.by_cell_counts()
        assert counts == {"AA": 3, "KK": 3, "QQ": 6, "AKs": 2}

    def test_summary_table_formatting(self) -> None:
        """summary produces an ASCII table with headers and data."""
        bd = range_vs_hand_breakdown("AA", "KhKd", num_sims_per_combo=500, seed=42)
        table = bd.summary()
        assert "Range vs Hand Breakdown" in table
        assert "Overall Equity:" in table
        assert "Combo" in table
        assert "AsAh" in table
        assert "Sims" in table

    def test_summary_table_with_top_n(self) -> None:
        """summary with top_n truncates output and notes remaining combos."""
        bd = range_vs_hand_breakdown("QQ+, AKs", "AhKd", num_sims_per_combo=500, seed=42)
        table = bd.summary(top_n=5)
        assert "... and 9 more combos" in table


# =========================================================================
# Spot equity tests across streets (Preflop, Flop, River)
# =========================================================================


class TestBreakdownSpotEquities:
    def test_preflop_range_vs_dominated_hand(self) -> None:
        """Range of broadway pairs JJ+ against 7c2d."""
        bd = range_vs_hand_breakdown("JJ+", "7c2d", num_sims_per_combo=500, seed=42)
        assert bd.overall_equity > 0.80
        assert bd.hero_equity < 0.20

    def test_flop_texture_breakdown(self) -> None:
        """On Kc 7d 2s, AKs combos have high equity while QQ is a second pair."""
        board = "Kc 7d 2s"
        bd = range_vs_hand_breakdown("AKs, QQ", "JhJd", board=board, num_sims_per_combo=500, seed=42)
        cells = bd.by_cell()
        # AKs has top pair Kings on Kc7d2s, dominating Jacks
        assert cells["AKs"] > 0.85
        # QQ is second pair, beating Jacks
        assert cells["QQ"] > 0.70

    def test_river_showdown_exact_breakdown(self) -> None:
        """On river board Kd 7s 2c 9h 3d, KK makes a set (100% equity) while AA loses to KK."""
        board = "Kd 7s 2c 9h 3d"
        bd = range_vs_hand_breakdown("AA, KK", "AhAs", board=board, num_sims_per_combo=100, seed=42)
        # KK makes set of Kings, crushing opponent's pair of Aces
        for combo, res in bd.items():
            if combo[0].rank.char == "K" and combo[1].rank.char == "K":
                assert res.equity == 1.0
                assert res.wins == res.num_sims
            elif combo[0].rank.char == "A" and combo[1].rank.char == "A":
                # Remaining AA combo is AdAc, which ties AhAs
                assert res.equity == 0.5
                assert res.ties == res.num_sims

    def test_range_breakdown_alias(self) -> None:
        """range_breakdown is an alias for range_vs_hand_breakdown."""
        bd = range_breakdown("AA", "KsKh", num_sims_per_combo=500, seed=42)
        assert isinstance(bd, RangeBreakdown)
        assert len(bd) == 6


# =========================================================================
# hand_vs_range_breakdown (Hero perspective)
# =========================================================================


class TestHandVsRangeBreakdown:
    def test_hero_hand_perspective(self) -> None:
        """hand_vs_range_breakdown reports equity from hero hand's perspective."""
        # Hero has AhAs vs villain range KK
        bd = hand_vs_range_breakdown("AhAs", "KK", num_sims_per_combo=500, seed=42)
        assert isinstance(bd, RangeBreakdown)
        assert len(bd) == 6
        # Every KK combo should show Hero (AA) having ~81-82% equity
        for _, res in bd.items():
            assert 0.77 < res.equity < 0.87
        assert bd.hero_equity > 0.80
        assert bd.overall_equity > 0.80

    def test_hero_underdog_perspective(self) -> None:
        """Hero has 7c2d vs villain range AA."""
        bd = hand_vs_range_breakdown("7c2d", "AA", num_sims_per_combo=500, seed=42)
        assert bd.overall_equity < 0.15
        assert bd.hero_equity < 0.15
        assert bd.villain_equity > 0.85


# =========================================================================
# Input validation and error handling tests
# =========================================================================


class TestBreakdownValidationErrors:
    def test_hand_too_few_cards_raises(self) -> None:
        with pytest.raises(ValueError, match="Hand must contain exactly 2 cards"):
            range_vs_hand_breakdown("AA", [C("Ah")])

    def test_hand_too_many_cards_raises(self) -> None:
        with pytest.raises(ValueError, match="Hand must contain exactly 2 cards"):
            range_vs_hand_breakdown("AA", [C("Ah"), C("Kd"), C("Qs")])

    def test_hand_string_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="Hand must represent exactly 2 cards"):
            range_vs_hand_breakdown("AA", "Ah")

    def test_hand_duplicate_cards_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate cards"):
            range_vs_hand_breakdown("AA", [C("Ah"), C("Ah")])

    def test_duplicate_between_hand_and_board_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate card between hand and board"):
            range_vs_hand_breakdown("QQ+", "AhKd", board="Ah 2c 3d")

    def test_duplicate_in_board_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate card in board"):
            range_vs_hand_breakdown("QQ+", "KhKd", board="2c 2c 3d")

    def test_board_too_many_cards_raises(self) -> None:
        with pytest.raises(ValueError, match="Board can have at most 5 cards"):
            range_vs_hand_breakdown("QQ+", "KhKd", board="2c 3c 4c 5c 6c 7c")

    def test_empty_range_raises(self) -> None:
        with pytest.raises(ValueError, match="Range is empty"):
            range_vs_hand_breakdown(Range(), "AhKd")

    def test_completely_blocked_range_raises(self) -> None:
        # AhKh has only 1 combo, and Ah is in opponent hand
        with pytest.raises(ValueError, match="No combos in range remain"):
            range_vs_hand_breakdown("AhKh", "AhKd")

    def test_invalid_num_sims_raises(self) -> None:
        with pytest.raises(ValueError, match="num_sims_per_combo must be positive"):
            range_vs_hand_breakdown("AA", "KhKd", num_sims_per_combo=0)
        with pytest.raises(ValueError, match="num_sims_per_combo must be positive"):
            range_vs_hand_breakdown("AA", "KhKd", num_sims_per_combo=-100)

    def test_seed_determinism(self) -> None:
        """Calling with the same seed yields identical results."""
        bd1 = range_vs_hand_breakdown("AA", "KhKd", num_sims_per_combo=500, seed=123)
        bd2 = range_vs_hand_breakdown("AA", "KhKd", num_sims_per_combo=500, seed=123)
        for c in bd1:
            assert bd1[c].wins == bd2[c].wins
            assert bd1[c].equity == bd2[c].equity
