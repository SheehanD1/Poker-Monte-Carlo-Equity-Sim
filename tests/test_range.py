"""Unit tests for the Range class and combo generation functions."""

from __future__ import annotations

import random

import pytest

from poker_equity.card import Card, Rank, Suit
from poker_equity.constants import FULL_DECK
from poker_equity.range import (
    TOTAL_HOLDEM_COMBOS,
    Range,
    canonical_combo,
    combo_from_str,
    combo_to_str,
    generate_all_combos,
    generate_offsuit_combos,
    generate_pair_combos,
    generate_suited_combos,
    generate_unpaired_combos,
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
