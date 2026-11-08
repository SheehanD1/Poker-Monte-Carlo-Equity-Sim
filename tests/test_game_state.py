"""Unit and integration tests for the GameState table container and action engine."""

import json
import pytest

from poker_equity import (
    Action,
    ActionRecord,
    Card,
    GameState,
    PlayerState,
    Position,
    SidePot,
    Street,
)


class TestPlayerState:
    """Tests for the PlayerState container."""

    def test_player_init_and_properties(self):
        """Player properties, has_cards, and reset_street_bet."""
        p = PlayerState(
            player_id="hero",
            name="Hero",
            position=Position.BTN,
            stack=200.0,
            cards=(Card.from_str("Ah"), Card.from_str("Kh")),
        )
        assert p.player_id == "hero"
        assert p.name == "Hero"
        assert p.stack == 200.0
        assert p.current_bet == 0.0
        assert p.total_contributed == 0.0
        assert p.has_cards is True
        assert p.is_active is True
        assert p.is_all_in is False

        p.current_bet = 50.0
        p.reset_street_bet()
        assert p.current_bet == 0.0

    def test_player_validation_errors(self):
        """Negative stacks or bets raise ValueError."""
        with pytest.raises(ValueError, match="stack cannot be negative"):
            PlayerState(player_id="p1", name="P1", position="BTN", stack=-10.0)
        with pytest.raises(ValueError, match="current_bet cannot be negative"):
            PlayerState(player_id="p1", name="P1", position="BTN", stack=10.0, current_bet=-5.0)

    def test_player_to_dict(self):
        """Serialization to dictionary."""
        p = PlayerState(
            player_id="hero",
            name="Hero",
            position=Position.BTN,
            stack=150.0,
            cards=(Card.from_str("Ah"), Card.from_str("As")),
        )
        d = p.to_dict()
        assert d["player_id"] == "hero"
        assert d["position"] == "BTN"
        assert d["stack"] == 150.0
        assert d["cards"] == "Ah As"


class TestSidePotAndActionRecord:
    """Tests for SidePot and ActionRecord data containers."""

    def test_side_pot_properties_and_dict(self):
        """SidePot initialization and dict conversion."""
        sp = SidePot(amount=150.0, eligible_player_ids=["hero", "villain"], is_main=True)
        assert sp.amount == 150.0
        assert sp.is_main is True
        assert "hero" in sp.eligible_player_ids

        d = sp.to_dict()
        assert d["amount"] == 150.0
        assert d["is_main"] is True

    def test_action_record_properties_and_dict(self):
        """ActionRecord logging attributes and dict conversion."""
        rec = ActionRecord(
            street=Street.FLOP,
            player_id="hero",
            action=Action.BET,
            amount=50.0,
            pot_before=100.0,
            pot_after=150.0,
            is_all_in=False,
        )
        assert rec.street == Street.FLOP
        assert rec.action == Action.BET
        assert rec.amount == 50.0
        assert rec.pot_after == 150.0

        d = rec.to_dict()
        assert d["action"] == "bet"
        assert d["amount"] == 50.0


class TestGameStateInitialization:
    """Tests for table creation, validation, and factory methods."""

    def test_create_heads_up(self):
        """Standard Heads-Up table initializes with posted blinds."""
        table = GameState.create_heads_up(
            small_blind=1.0,
            big_blind=2.0,
            hero_stack=100.0,
            villain_stack=100.0,
            hero_cards="Ah As",
            villain_cards="Kh Ks",
            hero_is_btn=True,
        )

        assert table.num_players == 2
        assert table.small_blind == 1.0
        assert table.big_blind == 2.0
        assert table.street == Street.PREFLOP

        hero = table.get_player("hero")
        villain = table.get_player("villain")

        # Hero posted SB ($1), stack is 99
        assert hero.stack == 99.0
        assert hero.current_bet == 1.0
        assert hero.position == Position.BTN

        # Villain posted BB ($2), stack is 98
        assert villain.stack == 98.0
        assert villain.current_bet == 2.0
        assert villain.position == Position.BB

        # Total pot is 3.0, current bet is 2.0
        assert table.total_pot == 3.0
        assert table.current_bet == 2.0

        # Action is on Hero (BTN / SB) first preflop in Heads-Up
        assert table.current_player.player_id == "hero"
        assert table.bet_to_call("hero") == 1.0

    def test_create_6max(self):
        """Standard 6-max table creation."""
        table = GameState.create_6max(
            small_blind=1.0,
            big_blind=2.0,
            stacks=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            button_index=3,
        )

        assert table.num_players == 6
        assert table.button_player.position == Position.BTN
        # In 6-max, action preflop begins on UTG (3 seats after BTN)
        assert table.current_player.position == Position.UTG
        assert table.total_pot == 3.0

    def test_invalid_init_parameters(self):
        """Invalid blinds or button index raise ValueError."""
        with pytest.raises(ValueError, match="small_blind must be positive"):
            GameState(small_blind=-1.0, big_blind=2.0)
        with pytest.raises(ValueError, match="big_blind.*cannot be smaller"):
            GameState(small_blind=5.0, big_blind=2.0)
        with pytest.raises(ValueError, match="ante cannot be negative"):
            GameState(small_blind=1.0, big_blind=2.0, ante=-1.0)
        with pytest.raises(ValueError, match="button_index.*must be between"):
            GameState(
                small_blind=1.0,
                big_blind=2.0,
                players=[PlayerState("p1", "P1", "BTN", 100.0)],
                button_index=5,
            )


class TestBettingAndActionApplication:
    """Tests for poker action processing, chip movements, and legal actions."""

    @pytest.fixture
    def hu_table(self):
        return GameState.create_heads_up(
            small_blind=1.0,
            big_blind=2.0,
            hero_stack=100.0,
            villain_stack=100.0,
            hero_cards="Ah As",
            villain_cards="Kh Ks",
        )

    def test_preflop_raise_and_call(self, hu_table):
        """Hero raises to $6 preflop, Villain calls."""
        # Legal actions for Hero: FOLD, CALL, RAISE, ALL_IN
        assert Action.CALL in hu_table.legal_actions("hero")
        assert Action.RAISE in hu_table.legal_actions("hero")

        # Hero raises to 6.0 total
        rec_raise = hu_table.apply_action(Action.RAISE, amount=6.0)
        assert rec_raise.action == Action.RAISE
        assert rec_raise.amount == 5.0  # 1.0 already posted + 5.0 more = 6.0
        assert hu_table.get_player("hero").stack == 94.0
        assert hu_table.current_bet == 6.0
        assert hu_table.total_pot == 8.0  # 6 from Hero + 2 from Villain

        # Action is now on Villain
        assert hu_table.current_player.player_id == "villain"
        assert hu_table.bet_to_call("villain") == 4.0  # 6 - 2 = 4

        # Villain calls
        rec_call = hu_table.apply_action(Action.CALL)
        assert rec_call.action == Action.CALL
        assert rec_call.amount == 4.0
        assert hu_table.get_player("villain").stack == 94.0
        assert hu_table.total_pot == 12.0

    def test_fold_ends_hand(self, hu_table):
        """Hero raises, Villain folds; hand completes."""
        hu_table.apply_action(Action.RAISE, amount=6.0)
        hu_table.apply_action(Action.FOLD)

        assert hu_table.get_player("villain").is_active is False
        assert hu_table.is_hand_over is True
        assert len(hu_table.active_players) == 1
        assert hu_table.current_player is None

    def test_all_in_action(self, hu_table):
        """Hero shoves all-in preflop."""
        hu_table.apply_action(Action.ALL_IN)

        hero = hu_table.get_player("hero")
        assert hero.stack == 0.0
        assert hero.is_all_in is True
        assert hero.current_bet == 100.0
        assert hu_table.current_bet == 100.0

        # Villain facing all-in call of 98 chips
        assert hu_table.bet_to_call("villain") == 98.0
        assert Action.RAISE not in hu_table.legal_actions("villain")
        assert Action.CALL in hu_table.legal_actions("villain")

    def test_illegal_check_raises_error(self, hu_table):
        """Checking when facing a bet raises ValueError."""
        with pytest.raises(ValueError, match="not legal in this spot"):
            hu_table.apply_action(Action.CHECK)


class TestSidePotCalculations:
    """Tests for multi-stack all-in side pot calculations."""

    def test_three_way_all_in_side_pots(self):
        """3 players with uneven stacks create main and side pots."""
        # Alice has 30, Bob has 70, Charlie has 150
        p1 = PlayerState("p1", "Alice", Position.BTN, 0.0, current_bet=30.0, is_all_in=True)
        p2 = PlayerState("p2", "Bob", Position.SB, 0.0, current_bet=70.0, is_all_in=True)
        p3 = PlayerState("p3", "Charlie", Position.BB, 80.0, current_bet=70.0)

        state = GameState(1.0, 2.0, players=[p1, p2, p3])
        state.collect_bets()

        assert len(state.pots) == 2

        # Main pot: 30 * 3 = 90 chips, all 3 eligible
        main_pot = state.pots[0]
        assert main_pot.is_main is True
        assert main_pot.amount == 90.0
        assert set(main_pot.eligible_player_ids) == {"p1", "p2", "p3"}

        # Side pot: 40 * 2 = 80 chips, Bob and Charlie eligible
        side_pot = state.pots[1]
        assert side_pot.is_main is False
        assert side_pot.amount == 80.0
        assert set(side_pot.eligible_player_ids) == {"p2", "p3"}

        assert state.pot_size == 170.0


class TestStreetProgression:
    """Tests for street transitions (Flop -> Turn -> River)."""

    def test_advance_through_streets(self):
        """Advancing streets updates board and resets bets."""
        table = GameState.create_heads_up(
            small_blind=1.0,
            big_blind=2.0,
            hero_stack=100.0,
            villain_stack=100.0,
        )
        # Preflop action
        table.apply_action(Action.CALL)  # Hero calls 1
        table.apply_action(Action.CHECK)  # Villain checks

        # Flop
        table.deal_flop("2h 7c 8d")
        assert table.street == Street.FLOP
        assert len(table.board) == 3
        assert table.current_bet == 0.0
        assert table.total_pot == 4.0
        # Postflop action starts on Villain (BB)
        assert table.current_player.player_id == "villain"

        # Flop action: check-check
        table.apply_action(Action.CHECK)
        table.apply_action(Action.CHECK)

        # Turn
        table.deal_turn("3c")
        assert table.street == Street.TURN
        assert len(table.board) == 4

        # Turn action: check-check
        table.apply_action(Action.CHECK)
        table.apply_action(Action.CHECK)

        # River
        table.deal_river("4c")
        assert table.street == Street.RIVER
        assert len(table.board) == 5

        # Attempting to advance beyond River raises ValueError
        with pytest.raises(ValueError, match="Already on the River"):
            table.advance_street()


class TestEquityAndDecisionIntegration:
    """Tests for integration with equity calculation and decision engine."""

    def test_calculate_active_equities_and_metrics(self):
        """calculate_active_equities, pot_odds, and spr calculations."""
        table = GameState.create_heads_up(
            small_blind=1.0,
            big_blind=2.0,
            hero_stack=100.0,
            villain_stack=100.0,
            hero_cards="Ah As",
            villain_cards="Kh Ks",
        )
        table.apply_action(Action.RAISE, amount=6.0)
        table.apply_action(Action.CALL)
        table.deal_flop("2h 7c 8d")

        # Flop equity
        eqs = table.calculate_active_equities(num_sims=500, seed=42)
        assert eqs["hero"] > 0.85
        assert eqs["villain"] < 0.15

        # Metrics
        assert table.spr("hero") == pytest.approx(94.0 / 12.0)
        assert table.effective_stack("hero", "villain") == 94.0
        assert table.pot_odds("villain") == 0.0  # no bet facing villain

    def test_recommend_decision_integration(self):
        """recommend_decision returns valid strategic DecisionResult."""
        table = GameState.create_heads_up(
            small_blind=1.0,
            big_blind=2.0,
            hero_stack=100.0,
            villain_stack=100.0,
            hero_cards="Ah As",
            villain_cards="Kh Ks",
        )
        table.apply_action(Action.RAISE, amount=6.0)
        table.apply_action(Action.CALL)
        table.deal_flop("2h 7c 8d")

        # Villain checks
        table.apply_action(Action.CHECK)

        # Hero recommendation: Bet or All-In with overpair
        rec = table.recommend_decision("hero", num_sims=500, seed=42)
        assert rec.action in (Action.BET, Action.ALL_IN)
        assert rec.equity > 0.85
        assert rec.is_profitable is True


class TestTableFormattingAndSerialization:
    """Tests for summary output and JSON serialization."""

    def test_summary_and_to_dict(self):
        """summary contains table info and to_dict is JSON serializable."""
        table = GameState.create_heads_up(
            small_blind=1.0,
            big_blind=2.0,
            hero_stack=100.0,
            villain_stack=100.0,
            hero_cards="Ah As",
            villain_cards="Kh Ks",
        )
        text = table.summary()
        assert "Table State - PREFLOP" in text
        assert "Blinds: 1/2" in text
        assert "Hero" in text
        assert "Villain" in text
        assert "Action on: Hero" in text

        data = table.to_dict()
        assert data["small_blind"] == 1.0
        assert data["big_blind"] == 2.0
        assert len(data["players"]) == 2
        assert len(data["pots"]) == 1

        json_str = json.dumps(data)
        assert "Ah As" in json_str

