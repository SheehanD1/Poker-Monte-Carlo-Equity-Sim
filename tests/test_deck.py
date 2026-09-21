"""Unit tests for the Deck class."""

from __future__ import annotations

import pytest

from poker_equity.card import Card, Rank, Suit
from poker_equity.deck import Deck


# =========================================================================
# Construction & initial state
# =========================================================================


class TestDeckConstruction:
    """Tests for Deck initialization."""

    def test_new_deck_has_52_cards(self) -> None:
        deck = Deck()
        assert deck.remaining == 52
        assert len(deck) == 52

    def test_new_deck_contains_all_52_unique_cards(self) -> None:
        deck = Deck()
        cards = deck.cards
        assert len(set(cards)) == 52

    def test_new_deck_contains_every_rank_suit_combination(self) -> None:
        deck = Deck()
        expected = {Card(r, s) for r in Rank for s in Suit}
        assert set(deck.cards) == expected

    def test_repr(self) -> None:
        deck = Deck()
        assert repr(deck) == "Deck(remaining=52)"

    def test_cards_property_returns_copy(self) -> None:
        """Mutating the returned list must not affect the deck."""
        deck = Deck()
        cards = deck.cards
        cards.clear()
        assert deck.remaining == 52


# =========================================================================
# Shuffle
# =========================================================================


class TestDeckShuffle:
    """Tests for Deck.shuffle()."""

    def test_shuffle_preserves_count(self) -> None:
        deck = Deck(seed=1)
        deck.shuffle()
        assert deck.remaining == 52

    def test_shuffle_preserves_all_cards(self) -> None:
        deck = Deck(seed=1)
        before = set(deck.cards)
        deck.shuffle()
        after = set(deck.cards)
        assert before == after

    def test_shuffle_changes_order(self) -> None:
        deck = Deck(seed=99)
        before = deck.cards.copy()
        deck.shuffle()
        after = deck.cards
        # Astronomically unlikely for a shuffled deck to match original order
        assert before != after

    def test_shuffle_deterministic_with_same_seed(self) -> None:
        deck_a = Deck(seed=42)
        deck_a.shuffle()

        deck_b = Deck(seed=42)
        deck_b.shuffle()

        assert deck_a.cards == deck_b.cards

    def test_shuffle_different_seeds_produce_different_orders(self) -> None:
        deck_a = Deck(seed=1)
        deck_a.shuffle()

        deck_b = Deck(seed=2)
        deck_b.shuffle()

        assert deck_a.cards != deck_b.cards


# =========================================================================
# Deal
# =========================================================================


class TestDeckDeal:
    """Tests for Deck.deal() and Deck.deal_one()."""

    def test_deal_returns_correct_count(self) -> None:
        deck = Deck(seed=42)
        deck.shuffle()
        hand = deck.deal(5)
        assert len(hand) == 5

    def test_deal_reduces_remaining(self) -> None:
        deck = Deck(seed=42)
        deck.shuffle()
        deck.deal(5)
        assert deck.remaining == 47

    def test_deal_returns_cards_from_top(self) -> None:
        deck = Deck(seed=42)
        deck.shuffle()
        top_five = deck.cards[:5]
        dealt = deck.deal(5)
        assert dealt == top_five

    def test_deal_removes_dealt_cards(self) -> None:
        deck = Deck(seed=42)
        deck.shuffle()
        dealt = deck.deal(2)
        for card in dealt:
            assert card not in deck

    def test_deal_one(self) -> None:
        deck = Deck(seed=42)
        deck.shuffle()
        top = deck.cards[0]
        card = deck.deal_one()
        assert card == top
        assert deck.remaining == 51

    def test_deal_entire_deck(self) -> None:
        deck = Deck(seed=42)
        deck.shuffle()
        all_cards = deck.deal(52)
        assert len(all_cards) == 52
        assert deck.remaining == 0

    def test_deal_zero_raises(self) -> None:
        deck = Deck()
        with pytest.raises(ValueError, match="at least 1"):
            deck.deal(0)

    def test_deal_negative_raises(self) -> None:
        deck = Deck()
        with pytest.raises(ValueError, match="at least 1"):
            deck.deal(-1)

    def test_deal_more_than_remaining_raises(self) -> None:
        deck = Deck(seed=42)
        deck.shuffle()
        deck.deal(50)
        with pytest.raises(ValueError, match="Cannot deal 5"):
            deck.deal(5)

    def test_deal_from_empty_deck_raises(self) -> None:
        deck = Deck(seed=42)
        deck.deal(52)
        with pytest.raises(ValueError, match="Cannot deal"):
            deck.deal_one()

    def test_successive_deals_are_consistent(self) -> None:
        """Dealing 2 then 3 should equal dealing 5 from an identical deck."""
        deck_a = Deck(seed=42)
        deck_a.shuffle()
        first_two = deck_a.deal(2)
        next_three = deck_a.deal(3)

        deck_b = Deck(seed=42)
        deck_b.shuffle()
        all_five = deck_b.deal(5)

        assert first_two + next_three == all_five


# =========================================================================
# Remove
# =========================================================================


class TestDeckRemove:
    """Tests for Deck.remove()."""

    def test_remove_single_card(self) -> None:
        deck = Deck()
        ace_spades = Card.from_str("As")
        deck.remove(ace_spades)
        assert deck.remaining == 51
        assert ace_spades not in deck

    def test_remove_multiple_cards(self) -> None:
        deck = Deck()
        cards_to_remove = [Card.from_str("As"), Card.from_str("Kh"), Card.from_str("2c")]
        deck.remove(cards_to_remove)
        assert deck.remaining == 49
        for card in cards_to_remove:
            assert card not in deck

    def test_remove_preserves_other_cards(self) -> None:
        deck = Deck()
        deck.remove(Card.from_str("As"))
        # All other 51 cards should still be present
        remaining = set(deck.cards)
        expected = {Card(r, s) for r in Rank for s in Suit} - {Card.from_str("As")}
        assert remaining == expected

    def test_remove_card_not_in_deck_raises(self) -> None:
        deck = Deck()
        ace_spades = Card.from_str("As")
        deck.remove(ace_spades)
        with pytest.raises(ValueError, match="not in the deck"):
            deck.remove(ace_spades)

    def test_remove_after_deal(self) -> None:
        """Can remove a card that hasn't been dealt yet."""
        deck = Deck(seed=42)
        deck.shuffle()
        deck.deal(10)
        # King of hearts should still be in the remaining 42 cards (or not)
        kh = Card.from_str("Kh")
        if kh in deck:
            deck.remove(kh)
            assert kh not in deck
            assert deck.remaining == 41

    def test_remove_then_deal_excludes_removed(self) -> None:
        deck = Deck(seed=42)
        hero_hand = [Card.from_str("Ah"), Card.from_str("Kh")]
        deck.remove(hero_hand)
        deck.shuffle()
        # Deal remaining 50 cards — none should be the hero's cards
        all_dealt = deck.deal(50)
        for card in hero_hand:
            assert card not in all_dealt


# =========================================================================
# Reset
# =========================================================================


class TestDeckReset:
    """Tests for Deck.reset()."""

    def test_reset_restores_full_deck(self) -> None:
        deck = Deck(seed=42)
        deck.shuffle()
        deck.deal(20)
        deck.reset()
        assert deck.remaining == 52

    def test_reset_restores_all_cards(self) -> None:
        deck = Deck()
        deck.remove([Card.from_str("As"), Card.from_str("Kh")])
        deck.reset()
        expected = {Card(r, s) for r in Rank for s in Suit}
        assert set(deck.cards) == expected

    def test_reset_with_new_seed(self) -> None:
        deck = Deck(seed=1)
        deck.shuffle()
        order_seed_1 = deck.cards.copy()

        deck.reset(seed=2)
        deck.shuffle()
        order_seed_2 = deck.cards.copy()

        assert order_seed_1 != order_seed_2

    def test_reset_preserves_seed_when_none(self) -> None:
        """reset(seed=None) should keep the same PRNG sequence."""
        deck_a = Deck(seed=42)
        deck_a.shuffle()
        deck_a.deal(10)
        deck_a.reset()
        deck_a.shuffle()
        order_a = deck_a.cards.copy()

        # Replay same operations with fresh deck
        deck_b = Deck(seed=42)
        deck_b.shuffle()
        deck_b.deal(10)
        deck_b.reset()
        deck_b.shuffle()
        order_b = deck_b.cards.copy()

        assert order_a == order_b

    def test_repr_after_operations(self) -> None:
        deck = Deck(seed=42)
        deck.shuffle()
        deck.deal(10)
        assert repr(deck) == "Deck(remaining=42)"
        deck.reset()
        assert repr(deck) == "Deck(remaining=52)"


# =========================================================================
# __contains__
# =========================================================================


class TestDeckContains:
    """Tests for the ``in`` operator on Deck."""

    def test_card_in_fresh_deck(self) -> None:
        deck = Deck()
        assert Card.from_str("As") in deck

    def test_card_not_in_deck_after_removal(self) -> None:
        deck = Deck()
        deck.remove(Card.from_str("As"))
        assert Card.from_str("As") not in deck

    def test_non_card_not_in_deck(self) -> None:
        deck = Deck()
        assert "As" not in deck  # type: ignore[operator]
        assert 42 not in deck  # type: ignore[operator]
