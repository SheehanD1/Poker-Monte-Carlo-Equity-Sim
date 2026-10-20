"""Unit tests for the Range class and combo generation functions."""

from __future__ import annotations

import random

import pytest

from poker_equity.card import Card, Rank, Suit
from poker_equity.constants import FULL_DECK
from poker_equity.range import (
    CELL_COMBOS,
    GRID_RANKS,
    HAND_GRID_CELLS,
    NUM_GRID_CELLS,
    TOTAL_HOLDEM_COMBOS,
    Range,
    canonical_combo,
    combo_from_str,
    combo_to_cell,
    combo_to_str,
    generate_all_combos,
    generate_offsuit_combos,
    generate_pair_combos,
    generate_suited_combos,
    generate_unpaired_combos,
    parse_range_string,
)


# =========================================================================
# TestCanonicalCombo
# =========================================================================


class TestCanonicalCombo:
    """Tests for the canonical_combo normalization function."""

    def test_canonical_ordering_different_ranks(self) -> None:
        c1 = Card.from_str("Ah")
        c2 = Card.from_str("Kd")
        # Regardless of argument order, higher rank is always first
        assert canonical_combo(c1, c2) == (c1, c2)
        assert canonical_combo(c2, c1) == (c1, c2)

    def test_canonical_ordering_same_rank_pair(self) -> None:
        c_spades = Card.from_str("As")
        c_hearts = Card.from_str("Ah")
        # Suit ordering: Spades (4) > Hearts (3)
        assert canonical_combo(c_spades, c_hearts) == (c_spades, c_hearts)
        assert canonical_combo(c_hearts, c_spades) == (c_spades, c_hearts)

    def test_identical_cards_raises_value_error(self) -> None:
        c = Card.from_str("Ah")
        with pytest.raises(ValueError, match="cannot contain two identical cards"):
            canonical_combo(c, c)

    def test_invalid_types_raise_type_error(self) -> None:
        c = Card.from_str("Ah")
        with pytest.raises(TypeError, match="must be Card instances"):
            canonical_combo(c, "Kd")  # type: ignore[arg-type]


# =========================================================================
# TestComboStrConversions
# =========================================================================


class TestComboStrConversions:
    """Tests for combo_to_str and combo_from_str."""

    def test_combo_to_str(self) -> None:
        c1 = Card.from_str("Ah")
        c2 = Card.from_str("Kd")
        combo = canonical_combo(c1, c2)
        assert combo_to_str(combo) == "AhKd"

    def test_combo_from_str_compact(self) -> None:
        combo = combo_from_str("AhKd")
        assert combo == (Card.from_str("Ah"), Card.from_str("Kd"))

    def test_combo_from_str_space_delimited(self) -> None:
        combo = combo_from_str("Ah Kd")
        assert combo == (Card.from_str("Ah"), Card.from_str("Kd"))

    def test_combo_from_str_comma_delimited(self) -> None:
        combo = combo_from_str("Ah,Kd")
        assert combo == (Card.from_str("Ah"), Card.from_str("Kd"))

    def test_combo_from_str_normalizes_order(self) -> None:
        combo = combo_from_str("KdAh")
        assert combo == (Card.from_str("Ah"), Card.from_str("Kd"))

    def test_combo_from_str_invalid_length(self) -> None:
        with pytest.raises(ValueError):
            combo_from_str("Ah")

    def test_combo_from_str_duplicate_card(self) -> None:
        with pytest.raises(ValueError):
            combo_from_str("AhAh")


# =========================================================================
# TestComboGenerators
# =========================================================================


class TestComboGenerators:
    """Tests for combo generation helper functions."""

    def test_generate_pair_combos_count_and_types(self) -> None:
        combos = generate_pair_combos("A")
        assert len(combos) == 6
        for c1, c2 in combos:
            assert c1.rank == Rank.ACE
            assert c2.rank == Rank.ACE
            assert c1.suit != c2.suit

    def test_generate_pair_combos_with_rank_enum(self) -> None:
        combos = generate_pair_combos(Rank.KING)
        assert len(combos) == 6
        for c1, c2 in combos:
            assert c1.rank == Rank.KING

    def test_generate_pair_combos_with_ten(self) -> None:
        combos = generate_pair_combos("T")
        assert len(combos) == 6
        combos_10 = generate_pair_combos("10")
        assert combos == combos_10

    def test_generate_suited_combos_count_and_suits(self) -> None:
        combos = generate_suited_combos("A", "K")
        assert len(combos) == 4
        for c1, c2 in combos:
            assert c1.rank == Rank.ACE
            assert c2.rank == Rank.KING
            assert c1.suit == c2.suit

    def test_generate_suited_combos_same_rank_raises(self) -> None:
        with pytest.raises(ValueError, match="two different ranks"):
            generate_suited_combos("A", "A")

    def test_generate_offsuit_combos_count(self) -> None:
        combos = generate_offsuit_combos("A", "K")
        assert len(combos) == 12
        for c1, c2 in combos:
            assert c1.rank == Rank.ACE
            assert c2.rank == Rank.KING
            assert c1.suit != c2.suit

    def test_generate_offsuit_combos_same_rank_raises(self) -> None:
        with pytest.raises(ValueError, match="two different ranks"):
            generate_offsuit_combos("K", "K")

    def test_generate_unpaired_combos_count(self) -> None:
        combos = generate_unpaired_combos("A", "K")
        assert len(combos) == 16
        suited = generate_suited_combos("A", "K")
        offsuit = generate_offsuit_combos("A", "K")
        assert combos == (suited | offsuit)

    def test_generate_all_combos_count(self) -> None:
        combos = generate_all_combos()
        assert len(combos) == TOTAL_HOLDEM_COMBOS
        assert len(combos) == 1326

    def test_all_combos_breakdown(self) -> None:
        combos = generate_all_combos()
        pair_count = sum(1 for c1, c2 in combos if c1.rank == c2.rank)
        suited_count = sum(1 for c1, c2 in combos if c1.rank != c2.rank and c1.suit == c2.suit)
        offsuit_count = sum(1 for c1, c2 in combos if c1.rank != c2.rank and c1.suit != c2.suit)
        assert pair_count == 78       # 13 ranks * 6
        assert suited_count == 312     # 78 pairs of ranks * 4
        assert offsuit_count == 936    # 78 pairs of ranks * 12
        assert pair_count + suited_count + offsuit_count == 1326


# =========================================================================
# TestRangeConstruction
# =========================================================================


class TestRangeConstruction:
    """Tests for Range constructor and classmethod factories."""

    def test_default_empty_range(self) -> None:
        r = Range()
        assert len(r) == 0
        assert r.is_empty
        assert not r
        assert repr(r) == "Range(0 combos)"

    def test_from_combos_iterable(self) -> None:
        c1 = Card.from_str("Ah")
        c2 = Card.from_str("Kd")
        r = Range([(c1, c2)])
        assert len(r) == 1
        assert not r.is_empty
        assert bool(r)
        assert repr(r) == "Range(1 combo)"

    def test_from_range_copy(self) -> None:
        r1 = Range.pair("A")
        r2 = Range(r1)
        assert r1 == r2
        assert r1.combos == r2.combos

    def test_from_cards(self) -> None:
        r = Range.from_cards("Ah", "Kd")
        assert len(r) == 1
        assert (Card.from_str("Ah"), Card.from_str("Kd")) in r

    def test_pair_factory(self) -> None:
        r = Range.pair("Q")
        assert len(r) == 6

    def test_suited_factory(self) -> None:
        r = Range.suited("A", "J")
        assert len(r) == 4

    def test_offsuit_factory(self) -> None:
        r = Range.offsuit("K", "Q")
        assert len(r) == 12

    def test_unpaired_factory(self) -> None:
        r = Range.unpaired("J", "T")
        assert len(r) == 16

    def test_all_factory(self) -> None:
        r = Range.all()
        assert len(r) == 1326

    def test_empty_factory(self) -> None:
        r = Range.empty()
        assert len(r) == 0
        assert r.is_empty

    def test_invalid_combo_length_raises(self) -> None:
        c1 = Card.from_str("Ah")
        with pytest.raises(ValueError, match="2-card sequence"):
            Range([[c1]])  # type: ignore[list-item]

    def test_non_card_in_combo_raises(self) -> None:
        c1 = Card.from_str("Ah")
        with pytest.raises(TypeError, match="Card instances"):
            Range([(c1, "Kd")])  # type: ignore[list-item]


# =========================================================================
# TestRangeProperties
# =========================================================================


class TestRangeProperties:
    """Tests for Range properties and query methods."""

    def test_combos_property_is_frozenset(self) -> None:
        r = Range.pair("A")
        assert isinstance(r.combos, frozenset)
        assert len(r.combos) == 6

    def test_combos_list_is_sorted_descending(self) -> None:
        r = Range.pair("A") | Range.pair("K")
        cl = r.combos_list
        assert isinstance(cl, tuple)
        assert len(cl) == 12
        # First 6 should be AA, next 6 should be KK
        for combo in cl[:6]:
            assert combo[0].rank == Rank.ACE
        for combo in cl[6:]:
            assert combo[0].rank == Rank.KING

    def test_iteration(self) -> None:
        r = Range.pair("A")
        items = list(r)
        assert len(items) == 6
        assert items == list(r.combos_list)

    def test_contains_card(self) -> None:
        r = Range.suited("A", "K")
        assert r.contains_card(Card.from_str("Ah"))
        assert r.contains_card(Card.from_str("Kh"))
        assert not r.contains_card(Card.from_str("Qh"))

    def test_cards_set(self) -> None:
        r = Range.suited("A", "K")
        cards = r.cards()
        assert len(cards) == 8  # 4 Aces + 4 Kings
        assert all(c.rank in (Rank.ACE, Rank.KING) for c in cards)

    def test_all_cards_in_full_range(self) -> None:
        r = Range.all()
        assert r.cards() == set(FULL_DECK)


# =========================================================================
# TestRangeMembership
# =========================================================================


class TestRangeMembership:
    """Tests for membership testing (in operator)."""

    def test_contains_canonical_tuple(self) -> None:
        r = Range.pair("A")
        c1 = Card.from_str("As")
        c2 = Card.from_str("Ah")
        assert (c1, c2) in r

    def test_contains_reversed_tuple(self) -> None:
        r = Range.pair("A")
        c1 = Card.from_str("As")
        c2 = Card.from_str("Ah")
        assert (c2, c1) in r

    def test_contains_string(self) -> None:
        r = Range.pair("A")
        assert "AsAh" in r
        assert "AhAs" in r
        assert "As Ah" in r
        assert "As,Ah" in r
        assert "KsKh" not in r

    def test_contains_invalid_object(self) -> None:
        r = Range.pair("A")
        assert 42 not in r  # type: ignore[operator]
        assert "invalid" not in r


# =========================================================================
# TestRangeSetOperations
# =========================================================================


class TestRangeSetOperations:
    """Tests for union, intersection, difference, and symmetric difference."""

    def test_union(self) -> None:
        r1 = Range.pair("A")  # 6 combos
        r2 = Range.pair("K")  # 6 combos
        merged = r1 | r2
        assert len(merged) == 12
        assert r1.union(r2) == merged

    def test_union_with_overlap(self) -> None:
        r1 = Range.unpaired("A", "K")  # 16 combos
        r2 = Range.suited("A", "K")    # 4 combos (subset of r1)
        merged = r1 | r2
        assert len(merged) == 16
        assert merged == r1

    def test_intersection(self) -> None:
        r1 = Range.unpaired("A", "K")  # 16 combos
        r2 = Range.suited("A", "K")    # 4 combos
        inter = r1 & r2
        assert len(inter) == 4
        assert inter == r2

    def test_intersection_disjoint(self) -> None:
        r1 = Range.pair("A")
        r2 = Range.pair("K")
        assert (r1 & r2).is_empty

    def test_difference(self) -> None:
        r1 = Range.unpaired("A", "K")  # 16 combos
        r2 = Range.suited("A", "K")    # 4 combos
        diff = r1 - r2
        assert len(diff) == 12
        assert diff == Range.offsuit("A", "K")

    def test_symmetric_difference(self) -> None:
        r1 = Range.pair("A") | Range.pair("K")  # 12
        r2 = Range.pair("K") | Range.pair("Q")  # 12
        sym = r1 ^ r2
        assert len(sym) == 12
        assert sym == (Range.pair("A") | Range.pair("Q"))

    def test_set_operations_type_error(self) -> None:
        r = Range.pair("A")
        with pytest.raises(TypeError):
            _ = r | "invalid"  # type: ignore[operator]
        with pytest.raises(TypeError):
            _ = r & [1, 2]  # type: ignore[operator]
        with pytest.raises(TypeError):
            _ = r - 5  # type: ignore[operator]
        with pytest.raises(TypeError):
            _ = r ^ None  # type: ignore[operator]


# =========================================================================
# TestRangeSampling
# =========================================================================


class TestRangeSampling:
    """Tests for Range.sample."""

    def test_sample_returns_valid_combo(self) -> None:
        r = Range.pair("A")
        combo = r.sample()
        assert combo in r
        assert len(combo) == 2

    def test_sample_with_seeded_rng(self) -> None:
        r = Range.all()
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        assert r.sample(rng1) == r.sample(rng2)

    def test_sample_empty_range_raises_value_error(self) -> None:
        r = Range.empty()
        with pytest.raises(ValueError, match="Cannot sample from an empty Range"):
            r.sample()


# =========================================================================
# TestRangeEqualityAndHashing
# =========================================================================


class TestRangeEqualityAndHashing:
    """Tests for equality, hashing, and set/dict usage."""

    def test_equality_different_construction_order(self) -> None:
        c1 = Card.from_str("Ah")
        c2 = Card.from_str("Kd")
        c3 = Card.from_str("Qs")
        c4 = Card.from_str("Jd")

        r1 = Range([(c1, c2), (c3, c4)])
        r2 = Range([(c3, c4), (c2, c1)])  # different order & card reversal
        assert r1 == r2
        assert hash(r1) == hash(r2)

    def test_range_usable_in_set(self) -> None:
        r1 = Range.pair("A")
        r2 = Range.pair("A")
        r3 = Range.pair("K")
        s = {r1, r2, r3}
        assert len(s) == 2

    def test_not_equal_to_non_range(self) -> None:
        r = Range.pair("A")
        assert r != "AA"
        assert r != 42


# =========================================================================
# TestRangeParserPairs
# =========================================================================


class TestRangeParserPairs:
    """Tests for range string parsing of pocket pairs."""

    def test_single_pair_aa(self) -> None:
        r = Range.from_str("AA")
        assert len(r) == 6
        assert r == Range.pair("A")

    def test_single_pair_kk(self) -> None:
        r = Range.from_str("KK")
        assert len(r) == 6
        assert r == Range.pair("K")

    def test_single_pair_22(self) -> None:
        r = Range.from_str("22")
        assert len(r) == 6
        assert r == Range.pair("2")

    def test_plus_notation_qq_plus(self) -> None:
        # QQ+ means QQ, KK, AA -> 3 pairs * 6 combos = 18 combos
        r = Range.from_str("QQ+")
        assert len(r) == 18
        expected = Range.pair("Q") | Range.pair("K") | Range.pair("A")
        assert r == expected

    def test_plus_notation_88_plus(self) -> None:
        # 88+ means 88, 99, TT, JJ, QQ, KK, AA -> 7 pairs * 6 = 42 combos
        r = Range.from_str("88+")
        assert len(r) == 42

    def test_plus_notation_22_plus(self) -> None:
        # 22+ means all 13 pocket pairs -> 78 combos
        r = Range.from_str("22+")
        assert len(r) == 78

    def test_plus_notation_aa_plus(self) -> None:
        # AA+ is just AA -> 6 combos
        r = Range.from_str("AA+")
        assert len(r) == 6
        assert r == Range.pair("A")

    def test_dash_notation_pairs(self) -> None:
        # 88-66 means 88, 77, 66 -> 3 * 6 = 18 combos
        r = Range.from_str("88-66")
        assert len(r) == 18
        expected = Range.pair("8") | Range.pair("7") | Range.pair("6")
        assert r == expected

    def test_dash_notation_pairs_reversed(self) -> None:
        # 66-88 is identical to 88-66
        r = Range.from_str("66-88")
        assert len(r) == 18
        assert r == Range.from_str("88-66")

    def test_dash_notation_same_pair(self) -> None:
        r = Range.from_str("TT-TT")
        assert len(r) == 6
        assert r == Range.pair("T")

    def test_pair_case_insensitivity(self) -> None:
        r1 = Range.from_str("qq+")
        r2 = Range.from_str("QQ+")
        assert r1 == r2


# =========================================================================
# TestRangeParserSuited
# =========================================================================


class TestRangeParserSuited:
    """Tests for range string parsing of suited hands."""

    def test_single_suited_aks(self) -> None:
        r = Range.from_str("AKs")
        assert len(r) == 4
        assert r == Range.suited("A", "K")

    def test_single_suited_qjs(self) -> None:
        r = Range.from_str("QJs")
        assert len(r) == 4
        assert r == Range.suited("Q", "J")

    def test_single_suited_reversed_ranks(self) -> None:
        # KAs normalizes to AKs
        r = Range.from_str("KAs")
        assert len(r) == 4
        assert r == Range.from_str("AKs")

    def test_suited_kicker_plus(self) -> None:
        # A2s+ means A2s through AKs (12 suited hands * 4 = 48 combos)
        r = Range.from_str("A2s+")
        assert len(r) == 48

        # KTs+ means KTs, KJs, KQs (3 hands * 4 = 12 combos)
        r_kt = Range.from_str("KTs+")
        assert len(r_kt) == 12

        # AQs+ means AQs, AKs (2 hands * 4 = 8 combos)
        r_aq = Range.from_str("AQs+")
        assert len(r_aq) == 8

    def test_suited_connector_plus(self) -> None:
        # 76s+ means suited connectors 76s, 87s, 98s, T9s, JTs, QJs, KQs, AKs (8 hands * 4 = 32 combos)
        r = Range.from_str("76s+")
        assert len(r) == 32

        # T9s+ means T9s, JTs, QJs, KQs, AKs (5 hands * 4 = 20 combos)
        r_t9 = Range.from_str("T9s+")
        assert len(r_t9) == 20

    def test_suited_dash_connector_range(self) -> None:
        # T9s-76s means T9s, 98s, 87s, 76s -> 4 * 4 = 16 combos
        r = Range.from_str("T9s-76s")
        assert len(r) == 16
        expected = (
            Range.suited("T", "9")
            | Range.suited("9", "8")
            | Range.suited("8", "7")
            | Range.suited("7", "6")
        )
        assert r == expected

    def test_suited_dash_connector_range_reversed(self) -> None:
        r = Range.from_str("76s-T9s")
        assert len(r) == 16
        assert r == Range.from_str("T9s-76s")

    def test_suited_dash_kicker_range(self) -> None:
        # A5s-A2s means A5s, A4s, A3s, A2s -> 4 * 4 = 16 combos
        r = Range.from_str("A5s-A2s")
        assert len(r) == 16

        # KQs-KTs means KQs, KJs, KTs -> 3 * 4 = 12 combos
        r_kq = Range.from_str("KQs-KTs")
        assert len(r_kq) == 12

    def test_suited_case_insensitivity(self) -> None:
        assert Range.from_str("aks") == Range.from_str("AKs")
        assert Range.from_str("t9s-76s") == Range.from_str("T9s-76s")


# =========================================================================
# TestRangeParserOffsuit
# =========================================================================


class TestRangeParserOffsuit:
    """Tests for range string parsing of offsuit hands."""

    def test_single_offsuit_ako(self) -> None:
        r = Range.from_str("AKo")
        assert len(r) == 12
        assert r == Range.offsuit("A", "K")

    def test_single_offsuit_qjo(self) -> None:
        r = Range.from_str("QJo")
        assert len(r) == 12
        assert r == Range.offsuit("Q", "J")

    def test_single_offsuit_reversed_ranks(self) -> None:
        assert Range.from_str("KAo") == Range.from_str("AKo")

    def test_offsuit_kicker_plus(self) -> None:
        # AQo+ means AQo, AKo -> 2 * 12 = 24 combos
        r = Range.from_str("AQo+")
        assert len(r) == 24
        assert r == (Range.offsuit("A", "Q") | Range.offsuit("A", "K"))

        # ATo+ means ATo, AJo, AQo, AKo -> 4 * 12 = 48 combos
        r_at = Range.from_str("ATo+")
        assert len(r_at) == 48

    def test_offsuit_connector_plus(self) -> None:
        # 76o+ means 76o..AKo (8 hands * 12 = 96 combos)
        r = Range.from_str("76o+")
        assert len(r) == 96

    def test_offsuit_dash_kicker_range(self) -> None:
        # AQo-ATo means AQo, AJo, ATo -> 3 * 12 = 36 combos
        r = Range.from_str("AQo-ATo")
        assert len(r) == 36

    def test_offsuit_dash_connector_range(self) -> None:
        # T9o-76o means T9o, 98o, 87o, 76o -> 4 * 12 = 48 combos
        r = Range.from_str("T9o-76o")
        assert len(r) == 48


# =========================================================================
# TestRangeParserUnpaired
# =========================================================================


class TestRangeParserUnpaired:
    """Tests for range string parsing of hands without suited/offsuit specifier."""

    def test_single_unpaired_ak(self) -> None:
        # AK means both AKs (4) and AKo (12) -> 16 combos
        r = Range.from_str("AK")
        assert len(r) == 16
        assert r == Range.unpaired("A", "K")

    def test_single_unpaired_qj(self) -> None:
        r = Range.from_str("QJ")
        assert len(r) == 16
        assert r == Range.unpaired("Q", "J")

    def test_single_unpaired_reversed_ranks(self) -> None:
        assert Range.from_str("KA") == Range.from_str("AK")

    def test_unpaired_kicker_plus(self) -> None:
        # AQ+ means AQ (16) + AK (16) = 32 combos
        r = Range.from_str("AQ+")
        assert len(r) == 32
        assert r == (Range.unpaired("A", "Q") | Range.unpaired("A", "K"))

        # AT+ means AT, AJ, AQ, AK -> 4 * 16 = 64 combos
        r_at = Range.from_str("AT+")
        assert len(r_at) == 64

    def test_unpaired_connector_plus(self) -> None:
        # 76+ means 8 connector hands * 16 = 128 combos
        r = Range.from_str("76+")
        assert len(r) == 128

    def test_unpaired_dash_kicker_range(self) -> None:
        # AQ-AT means AQ, AJ, AT -> 3 * 16 = 48 combos
        r = Range.from_str("AQ-AT")
        assert len(r) == 48

    def test_unpaired_dash_connector_range(self) -> None:
        # T9-76 means T9, 98, 87, 76 -> 4 * 16 = 64 combos
        r = Range.from_str("T9-76")
        assert len(r) == 64


# =========================================================================
# TestRangeParserSpecificCombos
# =========================================================================


class TestRangeParserSpecificCombos:
    """Tests for parsing specific card combos."""

    def test_specific_combo_4_chars(self) -> None:
        r = Range.from_str("AhKd")
        assert len(r) == 1
        assert (Card.from_str("Ah"), Card.from_str("Kd")) in r

    def test_specific_combo_reversed_order(self) -> None:
        r = Range.from_str("KdAh")
        assert len(r) == 1
        assert (Card.from_str("Ah"), Card.from_str("Kd")) in r

    def test_specific_combos_multiple(self) -> None:
        r = Range.from_str("AhKd, AsKs, 2c2d")
        assert len(r) == 3

    def test_specific_combo_space_separated(self) -> None:
        r = Range.from_str("Ah Kd")
        assert len(r) == 1
        assert (Card.from_str("Ah"), Card.from_str("Kd")) in r

    def test_specific_combo_comma_separated_cards(self) -> None:
        r = Range.from_str("Ah,Kd")
        assert len(r) == 1
        assert (Card.from_str("Ah"), Card.from_str("Kd")) in r

    def test_specific_combos_space_separated_pairs(self) -> None:
        r = Range.from_str("Ah Kd, Qs Qd")
        assert len(r) == 2


# =========================================================================
# TestRangeParserSpecialKeywords
# =========================================================================


class TestRangeParserSpecialKeywords:
    """Tests for special keywords (random, 100%, any, none, empty)."""

    def test_random_keyword(self) -> None:
        r = Range.from_str("random")
        assert len(r) == TOTAL_HOLDEM_COMBOS

    def test_100_percent_keyword(self) -> None:
        r = Range.from_str("100%")
        assert len(r) == TOTAL_HOLDEM_COMBOS

    def test_any_keyword(self) -> None:
        r = Range.from_str("any")
        assert len(r) == TOTAL_HOLDEM_COMBOS

    def test_star_keyword(self) -> None:
        r = Range.from_str("*")
        assert len(r) == TOTAL_HOLDEM_COMBOS

    def test_none_keyword(self) -> None:
        r = Range.from_str("none")
        assert len(r) == 0
        assert r.is_empty

    def test_empty_keyword(self) -> None:
        r = Range.from_str("empty")
        assert len(r) == 0
        assert r.is_empty

    def test_empty_string(self) -> None:
        r = Range.from_str("")
        assert len(r) == 0
        assert r.is_empty

        r_blank = Range.from_str("   ")
        assert len(r_blank) == 0


# =========================================================================
# TestRangeParserCombinedRanges
# =========================================================================


class TestRangeParserCombinedRanges:
    """Tests for composite ranges combining multiple tokens."""

    def test_commit_example_aks_qq_plus_t9s_76s(self) -> None:
        # AKs (4) + QQ+ (18) + T9s-76s (16) = 38 combos
        r = Range.from_str("AKs, QQ+, T9s-76s")
        assert len(r) == 38

    def test_broad_range(self) -> None:
        # AA, KK, QQ, JJ, TT (5*6 = 30) + AKs (4) + AKo (12) = 46 combos
        r = Range.from_str("AA, KK, QQ, JJ, TT, AKs, AKo")
        assert len(r) == 46

    def test_semicolon_delimiter(self) -> None:
        r = Range.from_str("AA; KK; AKs")
        assert len(r) == 16

    def test_space_delimiter(self) -> None:
        r = Range.from_str("AA KK AKs")
        assert len(r) == 16

    def test_mixed_delimiters(self) -> None:
        r = Range.from_str("AA, KK; QQ AKs")
        assert len(r) == 22

    def test_overlapping_tokens_deduplicated(self) -> None:
        # QQ+ includes QQ, KK, AA; adding KK and AA should not create duplicate combos
        r = Range.from_str("QQ+, KK, AA")
        assert len(r) == 18

    def test_redundant_tokens(self) -> None:
        # AK includes AKs and AKo
        r = Range.from_str("AK, AKs, AKo")
        assert len(r) == 16

    def test_extra_commas_and_whitespace(self) -> None:
        r = Range.from_str("  QQ+ , , AKs  , ")
        assert len(r) == 22


# =========================================================================
# TestRangeParserInvalidSyntax
# =========================================================================


class TestRangeParserInvalidSyntax:
    """Tests for invalid range string error handling."""

    def test_invalid_token_string(self) -> None:
        with pytest.raises(ValueError, match="Invalid range token"):
            Range.from_str("XYZ")

    def test_single_char_token(self) -> None:
        with pytest.raises(ValueError, match="Invalid range token"):
            Range.from_str("A")

    def test_three_char_bad_suffix(self) -> None:
        with pytest.raises(ValueError, match="Invalid range token"):
            Range.from_str("AKx")

    def test_pair_with_suited_suffix(self) -> None:
        with pytest.raises(ValueError, match="Suited hand cannot have identical ranks"):
            Range.from_str("AAs")

    def test_pair_with_offsuit_suffix(self) -> None:
        with pytest.raises(ValueError, match="Offsuit hand cannot have identical ranks"):
            Range.from_str("AAo")

    def test_dash_mismatched_gaps(self) -> None:
        with pytest.raises(ValueError, match="matching gap"):
            Range.from_str("T9s-75s")

    def test_dash_pair_and_unpaired(self) -> None:
        with pytest.raises(ValueError, match="Incompatible range endpoints"):
            Range.from_str("99-AK")

    def test_dash_suited_and_offsuit(self) -> None:
        with pytest.raises(ValueError, match="Incompatible range endpoints"):
            Range.from_str("AKs-AQo")

    def test_plus_on_single_char(self) -> None:
        with pytest.raises(ValueError, match="Invalid plus token"):
            Range.from_str("A+")

    def test_invalid_card_in_combo(self) -> None:
        with pytest.raises(ValueError):
            Range.from_str("ZxAh")

    def test_duplicate_card_in_combo(self) -> None:
        with pytest.raises(ValueError, match="identical cards"):
            Range.from_str("AhAh")

    def test_trailing_dash(self) -> None:
        with pytest.raises(ValueError, match="Invalid range token"):
            Range.from_str("AA-")


# =========================================================================
# TestParseRangeStringDirect
# =========================================================================


class TestParseRangeStringDirect:
    """Tests for calling parse_range_string function directly."""

    def test_direct_function_call(self) -> None:
        combos = parse_range_string("QQ+")
        assert isinstance(combos, set)
        assert len(combos) == 18

    def test_direct_matches_from_str(self) -> None:
        combos = parse_range_string("AKs, QQ+, T9s-76s")
        r = Range.from_str("AKs, QQ+, T9s-76s")
        assert combos == r.combos


# =========================================================================
# TestHandGridStructure
# =========================================================================


class TestHandGridStructure:
    """Tests verifying the 13x13 poker matrix constants and layout."""

    def test_grid_ranks_count_and_order(self) -> None:
        assert len(GRID_RANKS) == 13
        assert GRID_RANKS == (
            "A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"
        )

    def test_num_grid_cells_is_169(self) -> None:
        assert NUM_GRID_CELLS == 169
        assert len(HAND_GRID_CELLS) == 169
        # All cell names must be unique
        assert len(set(HAND_GRID_CELLS)) == 169

    def test_diagonal_contains_13_pairs(self) -> None:
        # Indices 0, 14, 28, ..., 168 (i*14) are diagonal elements
        for i, rank in enumerate(GRID_RANKS):
            idx = i * 13 + i
            cell = HAND_GRID_CELLS[idx]
            assert cell == f"{rank}{rank}"

    def test_upper_triangle_is_suited(self) -> None:
        for i in range(13):
            for j in range(i + 1, 13):
                idx = i * 13 + j
                cell = HAND_GRID_CELLS[idx]
                assert cell.endswith("s")
                assert len(cell) == 3
                assert cell[0] == GRID_RANKS[i]
                assert cell[1] == GRID_RANKS[j]

    def test_lower_triangle_is_offsuit(self) -> None:
        for i in range(13):
            for j in range(0, i):
                idx = i * 13 + j
                cell = HAND_GRID_CELLS[idx]
                assert cell.endswith("o")
                assert len(cell) == 3
                assert cell[0] == GRID_RANKS[j]
                assert cell[1] == GRID_RANKS[i]

    def test_cell_combos_mapping_totals(self) -> None:
        assert len(CELL_COMBOS) == 169
        total_combos = 0
        pair_cells = 0
        suited_cells = 0
        offsuit_cells = 0

        for cell, combos in CELL_COMBOS.items():
            total_combos += len(combos)
            if len(cell) == 2:
                pair_cells += 1
                assert len(combos) == 6
            elif cell.endswith("s"):
                suited_cells += 1
                assert len(combos) == 4
            elif cell.endswith("o"):
                offsuit_cells += 1
                assert len(combos) == 12

        assert pair_cells == 13
        assert suited_cells == 78
        assert offsuit_cells == 78
        assert total_combos == TOTAL_HOLDEM_COMBOS
        assert total_combos == 1326

    def test_all_grid_combos_are_unique(self) -> None:
        all_combos: set[Combo] = set()
        for combos in CELL_COMBOS.values():
            all_combos.update(combos)
        assert len(all_combos) == 1326


# =========================================================================
# TestComboToCell
# =========================================================================


class TestComboToCell:
    """Tests for combo_to_cell mapping function."""

    def test_pair_combo_to_cell(self) -> None:
        c1 = Card.from_str("As")
        c2 = Card.from_str("Ah")
        combo = canonical_combo(c1, c2)
        assert combo_to_cell(combo) == "AA"

    def test_suited_combo_to_cell(self) -> None:
        c1 = Card.from_str("Ah")
        c2 = Card.from_str("Kh")
        combo = canonical_combo(c1, c2)
        assert combo_to_cell(combo) == "AKs"

    def test_offsuit_combo_to_cell(self) -> None:
        c1 = Card.from_str("Ah")
        c2 = Card.from_str("Kd")
        combo = canonical_combo(c1, c2)
        assert combo_to_cell(combo) == "AKo"

    def test_low_unpaired_combos(self) -> None:
        c1 = Card.from_str("7c")
        c2 = Card.from_str("2d")
        combo = canonical_combo(c1, c2)
        assert combo_to_cell(combo) == "72o"

        c3 = Card.from_str("3h")
        c4 = Card.from_str("2h")
        combo_suited = canonical_combo(c3, c4)
        assert combo_to_cell(combo_suited) == "32s"


# =========================================================================
# TestRangeToMatrix
# =========================================================================


class TestRangeToMatrix:
    """Tests for Range.to_matrix method."""

    def test_all_range_to_matrix(self) -> None:
        r = Range.all()
        m = r.to_matrix()
        assert len(m) == 169
        assert all(w == 1.0 for w in m.values())

    def test_empty_range_to_matrix(self) -> None:
        r = Range.empty()
        m = r.to_matrix(include_zeros=True)
        assert len(m) == 169
        assert all(w == 0.0 for w in m.values())

        m_sparse = r.to_matrix(include_zeros=False)
        assert m_sparse == {}

    def test_single_pair_range_to_matrix(self) -> None:
        r = Range.from_str("AA")
        m = r.to_matrix(include_zeros=True)
        assert len(m) == 169
        assert m["AA"] == 1.0
        assert m["KK"] == 0.0
        assert m["AKs"] == 0.0

        m_sparse = r.to_matrix(include_zeros=False)
        assert m_sparse == {"AA": 1.0}

    def test_composite_range_to_matrix(self) -> None:
        r = Range.from_str("QQ+, AKs")
        m_sparse = r.to_matrix(include_zeros=False)
        assert set(m_sparse.keys()) == {"QQ", "KK", "AA", "AKs"}
        assert all(w == 1.0 for w in m_sparse.values())

    def test_partial_cell_weight(self) -> None:
        # 1 combo of AKs out of 4 -> weight 0.25
        c1 = Card.from_str("Ah")
        c2 = Card.from_str("Kh")
        r = Range([(c1, c2)])
        m = r.to_matrix(include_zeros=False)
        assert len(m) == 1
        assert m["AKs"] == 0.25

        # 3 combos of AA out of 6 -> weight 0.5
        aa_combos = list(CELL_COMBOS["AA"])[:3]
        r_aa = Range(aa_combos)
        m_aa = r_aa.to_matrix(include_zeros=False)
        assert m_aa["AA"] == 0.5


# =========================================================================
# TestRangeToGrid2D
# =========================================================================


class TestRangeToGrid2D:
    """Tests for Range.to_grid_2d method."""

    def test_grid_dimensions(self) -> None:
        r = Range.from_str("AA, KK, AKs")
        grid = r.to_grid_2d()
        assert len(grid) == 13
        assert all(len(row) == 13 for row in grid)

    def test_all_range_grid_all_ones(self) -> None:
        grid = Range.all().to_grid_2d()
        assert all(all(val == 1.0 for val in row) for row in grid)

    def test_empty_range_grid_all_zeros(self) -> None:
        grid = Range.empty().to_grid_2d()
        assert all(all(val == 0.0 for val in row) for row in grid)

    def test_specific_cell_positions(self) -> None:
        r = Range.from_str("AA, AKs, AKo, 22")
        grid = r.to_grid_2d()
        assert grid[0][0] == 1.0   # AA (row 0, col 0)
        assert grid[0][1] == 1.0   # AKs (row 0, col 1)
        assert grid[1][0] == 1.0   # AKo (row 1, col 0)
        assert grid[12][12] == 1.0 # 22 (row 12, col 12)
        assert grid[0][2] == 0.0   # AQs (row 0, col 2)


# =========================================================================
# TestRangeFromMatrix
# =========================================================================


class TestRangeFromMatrix:
    """Tests for Range.from_matrix constructor."""

    def test_reconstruct_full_range(self) -> None:
        all_matrix = {cell: 1.0 for cell in HAND_GRID_CELLS}
        r = Range.from_matrix(all_matrix)
        assert len(r) == 1326
        assert r == Range.all()

    def test_reconstruct_empty_range(self) -> None:
        r1 = Range.from_matrix({})
        assert r1.is_empty
        assert len(r1) == 0

        zero_matrix = {cell: 0.0 for cell in HAND_GRID_CELLS}
        r2 = Range.from_matrix(zero_matrix)
        assert r2.is_empty
        assert len(r2) == 0

    def test_roundtrip_composite_range(self) -> None:
        original = Range.from_str("QQ+, AKs, T9s-76s")
        matrix = original.to_matrix()
        reconstructed = Range.from_matrix(matrix)
        assert reconstructed == original
        assert len(reconstructed) == len(original)

    def test_partial_weights_proportional_combos(self) -> None:
        # AA has 6 combos. Weight 0.5 -> 3 combos
        r_aa = Range.from_matrix({"AA": 0.5})
        assert len(r_aa) == 3

        # AKs has 4 combos. Weight 0.5 -> 2 combos
        r_aks = Range.from_matrix({"AKs": 0.5})
        assert len(r_aks) == 2

        # AKo has 12 combos. Weight 0.5 -> 6 combos
        r_ako = Range.from_matrix({"AKo": 0.5})
        assert len(r_ako) == 6

    def test_from_matrix_with_threshold(self) -> None:
        # With threshold=0.5:
        # AA at 0.5 >= 0.5 -> includes all 6 combos
        # KK at 0.4 < 0.5 -> includes 0 combos
        r = Range.from_matrix({"AA": 0.5, "KK": 0.4}, threshold=0.5)
        assert len(r) == 6
        assert r == Range.pair("A")

    def test_unsuited_shorthand_key(self) -> None:
        # "AK" key adds both AKs (4) and AKo (12) -> 16 combos
        r = Range.from_matrix({"AK": 1.0})
        assert len(r) == 16
        assert r == Range.unpaired("A", "K")

    def test_case_insensitive_cell_keys(self) -> None:
        r = Range.from_matrix({"aks": 1.0, "  aa  ": 1.0})
        assert len(r) == 10
        assert r == (Range.suited("A", "K") | Range.pair("A"))

    def test_invalid_cell_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid hand matrix cell"):
            Range.from_matrix({"XYZ": 1.0})


