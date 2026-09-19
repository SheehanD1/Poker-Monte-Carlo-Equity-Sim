"""Unit tests for the Card, Rank, and Suit primitives."""

from __future__ import annotations

import pytest

from poker_equity.card import Card, Rank, Suit


# =========================================================================
# Rank tests
# =========================================================================


class TestRank:
    """Tests for the Rank enum."""

    def test_values_range_from_two_to_ace(self) -> None:
        assert Rank.TWO == 2
        assert Rank.ACE == 14
        assert len(Rank) == 13

    def test_ordering(self) -> None:
        assert Rank.TWO < Rank.THREE
        assert Rank.JACK < Rank.QUEEN < Rank.KING < Rank.ACE
        assert Rank.ACE > Rank.KING

    @pytest.mark.parametrize(
        ("rank", "expected_char"),
        [
            (Rank.TWO, "2"),
            (Rank.NINE, "9"),
            (Rank.TEN, "T"),
            (Rank.JACK, "J"),
            (Rank.QUEEN, "Q"),
            (Rank.KING, "K"),
            (Rank.ACE, "A"),
        ],
    )
    def test_char_property(self, rank: Rank, expected_char: str) -> None:
        assert rank.char == expected_char

    @pytest.mark.parametrize(
        ("char", "expected_rank"),
        [
            ("2", Rank.TWO),
            ("9", Rank.NINE),
            ("T", Rank.TEN),
            ("t", Rank.TEN),  # case-insensitive
            ("J", Rank.JACK),
            ("Q", Rank.QUEEN),
            ("K", Rank.KING),
            ("A", Rank.ACE),
            ("a", Rank.ACE),  # case-insensitive
        ],
    )
    def test_from_char(self, char: str, expected_rank: Rank) -> None:
        assert Rank.from_char(char) == expected_rank

    @pytest.mark.parametrize("invalid_char", ["0", "1", "X", "Z", "", "10"])
    def test_from_char_invalid_raises(self, invalid_char: str) -> None:
        with pytest.raises(ValueError, match="Invalid rank character"):
            Rank.from_char(invalid_char)

    def test_repr(self) -> None:
        assert repr(Rank.ACE) == "Rank.ACE"
        assert repr(Rank.TWO) == "Rank.TWO"


# =========================================================================
# Suit tests
# =========================================================================


class TestSuit:
    """Tests for the Suit enum."""

    def test_four_suits(self) -> None:
        assert len(Suit) == 4

    def test_ordering(self) -> None:
        assert Suit.CLUBS < Suit.DIAMONDS < Suit.HEARTS < Suit.SPADES

    @pytest.mark.parametrize(
        ("suit", "expected_char", "expected_symbol"),
        [
            (Suit.CLUBS, "c", "♣"),
            (Suit.DIAMONDS, "d", "♦"),
            (Suit.HEARTS, "h", "♥"),
            (Suit.SPADES, "s", "♠"),
        ],
    )
    def test_char_and_symbol(
        self, suit: Suit, expected_char: str, expected_symbol: str
    ) -> None:
        assert suit.char == expected_char
        assert suit.symbol == expected_symbol

    @pytest.mark.parametrize(
        ("char", "expected_suit"),
        [
            ("c", Suit.CLUBS),
            ("C", Suit.CLUBS),  # case-insensitive
            ("d", Suit.DIAMONDS),
            ("h", Suit.HEARTS),
            ("s", Suit.SPADES),
            ("S", Suit.SPADES),  # case-insensitive
        ],
    )
    def test_from_char(self, char: str, expected_suit: Suit) -> None:
        assert Suit.from_char(char) == expected_suit

    @pytest.mark.parametrize("invalid_char", ["x", "1", "♠", ""])
    def test_from_char_invalid_raises(self, invalid_char: str) -> None:
        with pytest.raises(ValueError, match="Invalid suit character"):
            Suit.from_char(invalid_char)

    def test_repr(self) -> None:
        assert repr(Suit.HEARTS) == "Suit.HEARTS"


# =========================================================================
# Card tests
# =========================================================================


class TestCardConstruction:
    """Tests for creating Card instances."""

    def test_direct_construction(self) -> None:
        card = Card(Rank.ACE, Suit.HEARTS)
        assert card.rank == Rank.ACE
        assert card.suit == Suit.HEARTS

    @pytest.mark.parametrize(
        ("text", "expected_rank", "expected_suit"),
        [
            ("Ah", Rank.ACE, Suit.HEARTS),
            ("2c", Rank.TWO, Suit.CLUBS),
            ("Td", Rank.TEN, Suit.DIAMONDS),
            ("Ks", Rank.KING, Suit.SPADES),
            ("9h", Rank.NINE, Suit.HEARTS),
        ],
    )
    def test_from_str(
        self, text: str, expected_rank: Rank, expected_suit: Suit
    ) -> None:
        card = Card.from_str(text)
        assert card.rank == expected_rank
        assert card.suit == expected_suit

    def test_from_str_case_insensitive(self) -> None:
        assert Card.from_str("aH") == Card.from_str("Ah")
        assert Card.from_str("tS") == Card.from_str("Ts")

    def test_from_str_strips_whitespace(self) -> None:
        assert Card.from_str("  Ah  ") == Card.from_str("Ah")

    @pytest.mark.parametrize(
        "invalid_text",
        [
            "",       # empty
            "A",      # too short
            "Ahh",    # too long
            "1h",     # invalid rank
            "Ax",     # invalid suit
            "XX",     # both invalid
        ],
    )
    def test_from_str_invalid_raises(self, invalid_text: str) -> None:
        with pytest.raises(ValueError):
            Card.from_str(invalid_text)

    def test_from_str_empty_string_error_message(self) -> None:
        with pytest.raises(ValueError, match="exactly 2 characters"):
            Card.from_str("")


class TestCardComparison:
    """Tests for Card ordering and equality."""

    def test_equality_same_card(self) -> None:
        assert Card.from_str("Ah") == Card.from_str("Ah")

    def test_inequality_different_rank(self) -> None:
        assert Card.from_str("Ah") != Card.from_str("Kh")

    def test_inequality_different_suit(self) -> None:
        assert Card.from_str("Ah") != Card.from_str("As")

    def test_not_equal_to_non_card(self) -> None:
        assert Card.from_str("Ah") != "Ah"
        assert Card.from_str("Ah") != 14

    def test_less_than_by_rank(self) -> None:
        assert Card.from_str("2s") < Card.from_str("3c")
        assert Card.from_str("Kh") < Card.from_str("As")

    def test_less_than_by_suit_when_rank_equal(self) -> None:
        # Clubs(1) < Diamonds(2) < Hearts(3) < Spades(4)
        assert Card.from_str("Ac") < Card.from_str("Ad")
        assert Card.from_str("Ad") < Card.from_str("Ah")
        assert Card.from_str("Ah") < Card.from_str("As")

    def test_greater_than(self) -> None:
        assert Card.from_str("As") > Card.from_str("Ah")
        assert Card.from_str("Ah") > Card.from_str("Kh")

    def test_less_than_non_card_returns_not_implemented(self) -> None:
        # Should not raise; Python falls back gracefully
        with pytest.raises(TypeError):
            _ = Card.from_str("Ah") < "Ah"  # type: ignore[operator]

    def test_sorting(self) -> None:
        cards = [
            Card.from_str("As"),
            Card.from_str("2c"),
            Card.from_str("Th"),
            Card.from_str("2d"),
            Card.from_str("Ks"),
        ]
        sorted_cards = sorted(cards)
        expected = [
            Card.from_str("2c"),
            Card.from_str("2d"),
            Card.from_str("Th"),
            Card.from_str("Ks"),
            Card.from_str("As"),
        ]
        assert sorted_cards == expected


class TestCardHashing:
    """Tests for Card hashability and set/dict behavior."""

    def test_hash_equal_cards(self) -> None:
        a = Card.from_str("Ah")
        b = Card.from_str("Ah")
        assert hash(a) == hash(b)

    def test_hash_different_cards(self) -> None:
        a = Card.from_str("Ah")
        b = Card.from_str("Kh")
        # Not guaranteed but extremely likely for distinct cards
        assert hash(a) != hash(b)

    def test_set_deduplication(self) -> None:
        cards = {Card.from_str("Ah"), Card.from_str("Ah"), Card.from_str("Kh")}
        assert len(cards) == 2

    def test_dict_key(self) -> None:
        card = Card.from_str("Ah")
        d = {card: "ace of hearts"}
        assert d[Card.from_str("Ah")] == "ace of hearts"


class TestCardDisplay:
    """Tests for Card repr and str."""

    def test_repr(self) -> None:
        card = Card(Rank.ACE, Suit.HEARTS)
        assert repr(card) == "Card(A♥)"

    def test_str(self) -> None:
        card = Card(Rank.ACE, Suit.HEARTS)
        assert str(card) == "Ah"

    @pytest.mark.parametrize(
        ("text", "expected_repr", "expected_str"),
        [
            ("2c", "Card(2♣)", "2c"),
            ("Td", "Card(T♦)", "Td"),
            ("Ks", "Card(K♠)", "Ks"),
        ],
    )
    def test_repr_and_str_parametrized(
        self, text: str, expected_repr: str, expected_str: str
    ) -> None:
        card = Card.from_str(text)
        assert repr(card) == expected_repr
        assert str(card) == expected_str

    def test_all_52_cards_have_unique_str(self) -> None:
        """Every card in the deck should have a unique string representation."""
        all_strs = {str(Card(r, s)) for r in Rank for s in Suit}
        assert len(all_strs) == 52
