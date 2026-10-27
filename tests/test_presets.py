"""Unit and integration tests for preset ranges and percentile charts."""

from __future__ import annotations

import pytest

from poker_equity.card import Card, Rank
from poker_equity.presets import (
    Position,
    HAND_RANK_ORDER,
    PRESET_DESCRIPTIONS,
    PRESET_RANGES,
    TOP_5_PERCENT,
    TOP_10_PERCENT,
    TOP_15_PERCENT,
    TOP_20_PERCENT,
    TOP_25_PERCENT,
    TOP_30_PERCENT,
    TOP_40_PERCENT,
    TOP_50_PERCENT,
    get_preset_description,
    get_preset_range,
    list_presets,
    top_percent_range,
)
from poker_equity.range import HAND_GRID_CELLS, Range
from poker_equity.simulation import calculate_equity, calculate_equities


# =========================================================================
# TestPositionEnum
# =========================================================================


class TestPositionEnum:
    """Tests for the Position enum."""

    def test_all_positions_exist(self) -> None:
        expected = {"UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"}
        actual = {pos.value for pos in Position}
        assert actual == expected

    def test_string_representation(self) -> None:
        assert str(Position.UTG) == "UTG"
        assert str(Position.BTN) == "BTN"
        assert str(Position.CO) == "CO"

    def test_equality_with_string(self) -> None:
        assert Position.BTN == "BTN"
        assert Position.UTG == "UTG"


# =========================================================================
# TestHandRankOrder
# =========================================================================


class TestHandRankOrder:
    """Tests for the canonical 169-hand ranking order."""

    def test_contains_all_169_cells(self) -> None:
        assert len(HAND_RANK_ORDER) == 169
        assert len(set(HAND_RANK_ORDER)) == 169
        assert set(HAND_RANK_ORDER) == set(HAND_GRID_CELLS)

    def test_extremes(self) -> None:
        assert HAND_RANK_ORDER[0] == "AA"
        assert HAND_RANK_ORDER[1] == "KK"
        assert HAND_RANK_ORDER[2] == "QQ"
        assert HAND_RANK_ORDER[-1] == "T2o"
        assert "72o" in HAND_RANK_ORDER[-15:]

    def test_type_counts(self) -> None:
        pairs = [c for c in HAND_RANK_ORDER if len(c) == 2]
        suited = [c for c in HAND_RANK_ORDER if c.endswith("s")]
        offsuit = [c for c in HAND_RANK_ORDER if c.endswith("o")]
        assert len(pairs) == 13
        assert len(suited) == 78
        assert len(offsuit) == 78


# =========================================================================
# TestTopPercentRange
# =========================================================================


class TestTopPercentRange:
    """Tests for dynamic top percentile range generation."""

    def test_zero_percent(self) -> None:
        r = top_percent_range(0.0)
        assert len(r) == 0
        assert r.is_empty

    def test_hundred_percent(self) -> None:
        r = top_percent_range(100.0)
        assert len(r) == 1326

    def test_invalid_percentage_raises(self) -> None:
        with pytest.raises(ValueError, match="between 0.0 and 100.0"):
            top_percent_range(-1.0)
        with pytest.raises(ValueError, match="between 0.0 and 100.0"):
            top_percent_range(105.0)

    def test_top_10_percent_discrete(self) -> None:
        r10 = top_percent_range(10.0, weighted_boundary=False)
        # 10% of 1326 is 132.6 -> rounds to 133 combos
        assert abs(len(r10) - 133) <= 2
        assert (Card.from_str("Ah"), Card.from_str("As")) in r10
        assert (Card.from_str("Ah"), Card.from_str("Kh")) in r10
        # 72o should definitely not be in top 10%
        assert (Card.from_str("7s"), Card.from_str("2d")) not in r10

    def test_top_10_percent_weighted(self) -> None:
        r10 = top_percent_range(10.0, weighted_boundary=True)
        assert r10.total_combos == pytest.approx(132.6, abs=0.1)

    def test_percentile_monotonicity(self) -> None:
        r5 = top_percent_range(5.0)
        r15 = top_percent_range(15.0)
        r30 = top_percent_range(30.0)
        r60 = top_percent_range(60.0)

        assert len(r5) < len(r15) < len(r30) < len(r60)
        # A smaller percentile should be a subset of a larger percentile
        assert set(r5.combos).issubset(set(r15.combos))
        assert set(r15.combos).issubset(set(r30.combos))
        assert set(r30.combos).issubset(set(r60.combos))

    def test_constants_definitions(self) -> None:
        assert len(TOP_5_PERCENT) > 0
        assert len(TOP_10_PERCENT) > len(TOP_5_PERCENT)
        assert len(TOP_20_PERCENT) > len(TOP_10_PERCENT)
        assert len(TOP_50_PERCENT) > len(TOP_20_PERCENT)


# =========================================================================
# TestPresetRangesLookup
# =========================================================================


class TestPresetRangesLookup:
    """Tests for get_preset_range and Range.from_preset."""

    def test_get_preset_with_enum(self) -> None:
        r_utg = get_preset_range(Position.UTG)
        assert isinstance(r_utg, Range)
        assert len(r_utg) == 148

    def test_get_preset_case_insensitive(self) -> None:
        r1 = get_preset_range("UTG")
        r2 = get_preset_range("utg")
        r3 = get_preset_range("utg_open")
        r4 = get_preset_range("UTG_OPEN")
        assert r1 == r2 == r3 == r4

    def test_position_opening_hierarchy_6max(self) -> None:
        r_utg = get_preset_range("UTG")
        r_hj = get_preset_range("HJ")
        r_co = get_preset_range("CO")
        r_btn = get_preset_range("BTN")

        # Opening frequencies must strictly increase as position improves
        assert len(r_utg) < len(r_hj) < len(r_co) < len(r_btn)

    def test_range_from_preset_method(self) -> None:
        r1 = Range.from_preset("BTN")
        r2 = get_preset_range("BTN")
        assert r1 == r2

    def test_percentile_preset_aliases(self) -> None:
        r1 = get_preset_range("top10")
        r2 = get_preset_range("TOP_10%")
        r3 = top_percent_range(10.0)
        assert r1 == r2 == r3

    def test_archetype_presets(self) -> None:
        pairs = get_preset_range("POCKET_PAIRS")
        assert len(pairs) == 78  # 13 * 6

        connectors = get_preset_range("SUITED_CONNECTORS")
        assert len(connectors) == 48  # 12 * 4

        aces = get_preset_range("SUITED_ACES")
        assert len(aces) == 48  # 12 * 4

        broadways = get_preset_range("BROADWAYS")
        assert len(broadways) == 160

    def test_push_fold_ranges(self) -> None:
        push10 = get_preset_range("PUSH_10BB_BTN")
        push15 = get_preset_range("PUSH_15BB_BTN")
        # 10bb push range should be wider than 15bb push range
        assert len(push10) > len(push15)

    def test_3bet_ranges(self) -> None:
        r_3bet = get_preset_range("BB_VS_BTN_3BET")
        assert len(r_3bet) > 100
        # Premium hands should be included
        assert (Card.from_str("Ah"), Card.from_str("As")) in r_3bet
        assert (Card.from_str("Kh"), Card.from_str("Ks")) in r_3bet

    def test_unknown_preset_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown preset range"):
            get_preset_range("INVALID_POSITION_XYZ")


# =========================================================================
# TestListPresetsAndDescriptions
# =========================================================================


class TestListPresetsAndDescriptions:
    """Tests for listing presets and fetching descriptions."""

    def test_list_presets(self) -> None:
        presets = list_presets()
        assert isinstance(presets, list)
        assert len(presets) >= 20
        assert "UTG_OPEN" in presets
        assert "BTN_OPEN" in presets
        assert "CO_OPEN" in presets

    def test_every_preset_is_loadable(self) -> None:
        for preset_name in list_presets():
            r = get_preset_range(preset_name)
            assert isinstance(r, Range)
            assert len(r) > 0

    def test_descriptions(self) -> None:
        desc_utg = get_preset_description("UTG")
        assert "UTG" in desc_utg
        desc_btn = get_preset_description(Position.BTN)
        assert "Button" in desc_btn


# =========================================================================
# TestPresetSimulation
# =========================================================================


class TestPresetSimulation:
    """Integration test using preset ranges in Monte Carlo simulation."""

    def test_equity_against_preset_ranges(self) -> None:
        # Hero has AhKd
        # UTG range is tighter and stronger than BTN range
        # AhKd should have less equity against UTG than against BTN
        res_vs_utg = calculate_equity(
            "AhKd",
            get_preset_range("UTG"),
            num_sims=3_000,
            seed=42,
        )
        res_vs_btn = calculate_equity(
            "AhKd",
            get_preset_range("BTN"),
            num_sims=3_000,
            seed=42,
        )

        assert res_vs_btn.equity > res_vs_utg.equity

    def test_preset_vs_preset_simulation(self) -> None:
        # UTG open vs BTN 3-bet
        results = calculate_equities(
            [get_preset_range("UTG_OPEN"), get_preset_range("BTN_VS_CO_3BET")],
            num_sims=2_000,
            seed=42,
        )
        assert len(results) == 2
        assert results[0].equity + results[1].equity == pytest.approx(1.0, abs=0.01)
