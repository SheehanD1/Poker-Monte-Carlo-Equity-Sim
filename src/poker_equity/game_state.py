"""Poker table game state container and tracking engine for Texas Hold'em.

This module models live poker table action, including blinds, player stacks,
positions, pots, side pot calculations, street progression, and action history.
It integrates seamlessly with the equity simulator and decision engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from poker_equity.card import Card
from poker_equity.constants import cards_to_str, parse_cards
from poker_equity.decision import Action, DecisionResult, recommend_decision
from poker_equity.equity_curve import Street
from poker_equity.odds import pot_odds
from poker_equity.presets import Position
from poker_equity.simulation import calculate_equities


# =========================================================================
# Player State
# =========================================================================


@dataclass
class PlayerState:
    """Represents the state of a single seated poker player.

    Attributes
    ----------
    player_id : str | int
        Unique identifier for the player (e.g. "Hero", 0, 1).
    name : str
        Display name of the player.
    position : Position | str
        Table position (e.g. Position.BTN, Position.BB).
    stack : float
        Current chip stack behind.
    current_bet : float, default=0.0
        Chips wagered in the current betting round.
    total_contributed : float, default=0.0
        Total chips contributed to the pot across the entire hand.
    cards : tuple[Card, Card] | None, default=None
        Player hole cards if known.
    range_str : str | None, default=None
        Range string if player is modeled with a range.
    is_active : bool, default=True
        Whether the player is still in the hand (not folded).
    is_all_in : bool, default=False
        Whether the player has committed all their chips.
    is_sitting_out : bool, default=False
        Whether the player is sitting out.
    """

    player_id: str | int
    name: str
    position: Position | str
    stack: float
    current_bet: float = 0.0
    total_contributed: float = 0.0
    cards: tuple[Card, Card] | None = None
    range_str: str | None = None
    is_active: bool = True
    is_all_in: bool = False
    is_sitting_out: bool = False

    def __post_init__(self) -> None:
        if self.stack < 0:
            raise ValueError(f"Player stack cannot be negative, got {self.stack}")
        if self.current_bet < 0:
            raise ValueError(f"Player current_bet cannot be negative, got {self.current_bet}")
        if self.total_contributed < 0:
            raise ValueError(
                f"Player total_contributed cannot be negative, got {self.total_contributed}"
            )

    @property
    def has_cards(self) -> bool:
        """Whether the player has dealt hole cards."""
        return self.cards is not None

    def reset_street_bet(self) -> None:
        """Reset current bet for the start of a new betting round."""
        self.current_bet = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize player state to a dictionary."""
        pos_str = self.position.value if isinstance(self.position, Position) else str(self.position)
        cards_str = cards_to_str(self.cards) if self.cards else None
        return {
            "player_id": self.player_id,
            "name": self.name,
            "position": pos_str,
            "stack": self.stack,
            "current_bet": self.current_bet,
            "total_contributed": self.total_contributed,
            "cards": cards_str,
            "range_str": self.range_str,
            "is_active": self.is_active,
            "is_all_in": self.is_all_in,
            "is_sitting_out": self.is_sitting_out,
        }


# =========================================================================
# Side Pot and Action Record
# =========================================================================


@dataclass
class SidePot:
    """Represents a main or side pot with eligible players.

    Attributes
    ----------
    amount : float
        Chip amount currently in this pot.
    eligible_player_ids : list[str | int]
        List of player IDs who can contest this pot at showdown.
    is_main : bool, default=False
        Whether this is the main pot.
    """

    amount: float
    eligible_player_ids: list[str | int] = field(default_factory=list)
    is_main: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize side pot to a dictionary."""
        return {
            "amount": self.amount,
            "eligible_player_ids": list(self.eligible_player_ids),
            "is_main": self.is_main,
        }


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """A log entry recording a single player action in the hand history.

    Attributes
    ----------
    street : Street
        Street when the action took place.
    player_id : str | int
        ID of the player who acted.
    action : Action
        Action taken (FOLD, CHECK, CALL, BET, RAISE, ALL_IN).
    amount : float
        Chips contributed or wagered by this action.
    pot_before : float
        Total pot size before this action.
    pot_after : float
        Total pot size after this action.
    is_all_in : bool, default=False
        Whether this action placed the player all-in.
    """

    street: Street
    player_id: str | int
    action: Action
    amount: float
    pot_before: float
    pot_after: float
    is_all_in: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize action record to a dictionary."""
        return {
            "street": self.street.value,
            "player_id": self.player_id,
            "action": self.action.value,
            "amount": self.amount,
            "pot_before": self.pot_before,
            "pot_after": self.pot_after,
            "is_all_in": self.is_all_in,
        }


# =========================================================================
# GameState Container
# =========================================================================


class GameState:
    """Comprehensive representation of Texas Hold'em active table state.

    Manages blinds, seated players, community cards, street transitions,
    pots/side pots, chip movements, legal actions, and action history.

    Parameters
    ----------
    small_blind : float
        Small blind amount in chips.
    big_blind : float
        Big blind amount in chips.
    ante : float, default=0.0
        Per-player ante amount in chips.
    players : Sequence[PlayerState] | None, default=None
        Seated players at the table.
    board : Sequence[Card | str] | str | None, default=None
        Current community board cards.
    street : Street, default=Street.PREFLOP
        Current street.
    button_index : int, default=0
        Index of the dealer button seat in ``players``.

    Raises
    ------
    ValueError
        If blinds are negative or button index is invalid.
    """

    def __init__(
        self,
        small_blind: float,
        big_blind: float,
        ante: float = 0.0,
        players: Sequence[PlayerState] | None = None,
        board: Sequence[Card | str] | str | None = None,
        street: Street = Street.PREFLOP,
        button_index: int = 0,
    ) -> None:
        if small_blind <= 0:
            raise ValueError(f"small_blind must be positive, got {small_blind}")
        if big_blind <= 0:
            raise ValueError(f"big_blind must be positive, got {big_blind}")
        if big_blind < small_blind:
            raise ValueError(
                f"big_blind ({big_blind}) cannot be smaller than small_blind ({small_blind})"
            )
        if ante < 0:
            raise ValueError(f"ante cannot be negative, got {ante}")

        self.small_blind = float(small_blind)
        self.big_blind = float(big_blind)
        self.ante = float(ante)
        self.players: list[PlayerState] = list(players) if players is not None else []
        self.street = street
        self.button_index = button_index

        if self.players and not (0 <= button_index < len(self.players)):
            raise ValueError(
                f"button_index ({button_index}) must be between 0 and {len(self.players) - 1}"
            )

        # Parse board
        if board is None:
            self.board: list[Card] = []
        elif isinstance(board, str):
            self.board = parse_cards(board)
        else:
            self.board = [Card.from_str(c) if isinstance(c, str) else c for c in board]

        self.current_bet: float = 0.0
        self.min_raise: float = self.big_blind
        self.action_index: int = 0
        self.pots: list[SidePot] = [SidePot(amount=0.0, eligible_player_ids=[], is_main=True)]
        self.history: list[ActionRecord] = []

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def num_players(self) -> int:
        """Total number of seated players."""
        return len(self.players)

    @property
    def active_players(self) -> list[PlayerState]:
        """Players who have not folded."""
        return [p for p in self.players if p.is_active and not p.is_sitting_out]

    @property
    def all_in_players(self) -> list[PlayerState]:
        """Players who are all-in."""
        return [p for p in self.players if p.is_active and p.is_all_in]

    @property
    def folded_players(self) -> list[PlayerState]:
        """Players who have folded."""
        return [p for p in self.players if not p.is_active]

    @property
    def pot_size(self) -> float:
        """Total chips in collected pots (main + side pots)."""
        return sum(p.amount for p in self.pots)

    @property
    def uncollected_bets(self) -> float:
        """Chips currently wagered on the table in front of players."""
        return sum(p.current_bet for p in self.players)

    @property
    def total_pot(self) -> float:
        """Total pot including collected pots and uncollected current bets."""
        return self.pot_size + self.uncollected_bets

    @property
    def is_hand_over(self) -> bool:
        """Whether the hand is complete (1 player left or action complete on river)."""
        if len(self.active_players) <= 1:
            return True
        if self.street == Street.RIVER and len(self.board) == 5:
            # If all active players have matched current_bet or are all-in
            active_not_all_in = [p for p in self.active_players if not p.is_all_in]
            if len(active_not_all_in) <= 1 and len(self.history) > 0:
                return True
        return False

    @property
    def current_player(self) -> PlayerState | None:
        """The player whose turn it is to act, or None if hand is over."""
        if self.is_hand_over or not self.players:
            return None
        return self.players[self.action_index]

    @property
    def button_player(self) -> PlayerState:
        """The player holding the dealer button."""
        return self.players[self.button_index]

    # -------------------------------------------------------------------------
    # Player Lookups and Poker Metrics
    # -------------------------------------------------------------------------

    def get_player(self, player_id: str | int) -> PlayerState:
        """Retrieve a player by their ID."""
        for p in self.players:
            if p.player_id == player_id:
                return p
        raise KeyError(f"Player '{player_id}' not found at table")

    def get_player_index(self, player_id: str | int) -> int:
        """Retrieve a player's seat index by their ID."""
        for idx, p in enumerate(self.players):
            if p.player_id == player_id:
                return idx
        raise KeyError(f"Player '{player_id}' not found at table")

    def bet_to_call(self, player_id: str | int | None = None) -> float:
        """Calculate amount required for a player to call the current wager.

        Parameters
        ----------
        player_id : str | int | None, default=None
            Player ID, or current player to act if None.

        Returns
        -------
        float
            Chip cost to call.
        """
        p = self.get_player(player_id) if player_id is not None else self.current_player
        if p is None or not p.is_active or p.is_all_in:
            return 0.0
        call_amt = max(0.0, self.current_bet - p.current_bet)
        return min(p.stack, call_amt)

    def pot_odds(self, player_id: str | int | None = None) -> float:
        """Calculate direct break-even pot odds for a player facing a bet.

        Parameters
        ----------
        player_id : str | int | None, default=None
            Player ID, or current player to act if None.

        Returns
        -------
        float
            Required break-even equity fraction in ``[0.0, 1.0]``.
        """
        call_cost = self.bet_to_call(player_id)
        if call_cost <= 0:
            return 0.0
        return pot_odds(self.total_pot, call_cost)

    def spr(self, player_id: str | int | None = None) -> float:
        """Calculate Stack-to-Pot Ratio (SPR) for a player.

        Parameters
        ----------
        player_id : str | int | None, default=None
            Player ID, or current player to act if None.

        Returns
        -------
        float
            SPR ratio (stack / total_pot), or 0.0 if pot is zero.
        """
        p = self.get_player(player_id) if player_id is not None else self.current_player
        if p is None or self.total_pot == 0:
            return 0.0
        return p.stack / self.total_pot

    def effective_stack(self, p1_id: str | int, p2_id: str | int) -> float:
        """Calculate effective stack between two players in chips."""
        p1 = self.get_player(p1_id)
        p2 = self.get_player(p2_id)
        return min(p1.stack + p1.current_bet, p2.stack + p2.current_bet)

    def legal_actions(self, player_id: str | int | None = None) -> list[Action]:
        """List all legal actions available to a player in this spot.

        Parameters
        ----------
        player_id : str | int | None, default=None
            Player ID, or current player if None.

        Returns
        -------
        list[Action]
            List of legal Action enums.
        """
        p = self.get_player(player_id) if player_id is not None else self.current_player
        if p is None or not p.is_active or p.is_all_in:
            return []

        call_cost = self.bet_to_call(p.player_id)
        actions = []

        if call_cost == 0.0:
            actions.append(Action.CHECK)
            if p.stack > 0:
                actions.append(Action.BET)
                actions.append(Action.ALL_IN)
        else:
            actions.append(Action.FOLD)
            actions.append(Action.CALL)
            # Raise is legal if player has more chips than the call cost
            if p.stack > call_cost:
                actions.append(Action.RAISE)
                actions.append(Action.ALL_IN)

        return actions

    def min_legal_raise(self, player_id: str | int | None = None) -> float:
        """Calculate the minimum total wager required for a legal raise.

        Returns
        -------
        float
            Minimum total wager (including player's current bet).
        """
        p = self.get_player(player_id) if player_id is not None else self.current_player
        if p is None:
            return 0.0
        min_total = self.current_bet + self.min_raise
        max_possible = p.stack + p.current_bet
        return min(min_total, max_possible)

    # -------------------------------------------------------------------------
    # Game Engine & State Transitions
    # -------------------------------------------------------------------------

    def post_blinds(self) -> None:
        """Post small blind, big blind, and antes for all active players."""
        if len(self.players) < 2:
            raise ValueError("At least 2 players are required to post blinds")

        # 1. Post antes
        if self.ante > 0:
            for p in self.players:
                if not p.is_sitting_out:
                    ante_amt = min(p.stack, self.ante)
                    p.stack -= ante_amt
                    p.total_contributed += ante_amt
                    self.pots[0].amount += ante_amt
                    if p.stack == 0:
                        p.is_all_in = True

        # 2. Determine blind seats
        n = len(self.players)
        if n == 2:
            # Heads-up: Button is SB, other seat is BB
            sb_idx = self.button_index
            bb_idx = (self.button_index + 1) % n
            first_to_act = sb_idx
        else:
            sb_idx = (self.button_index + 1) % n
            bb_idx = (self.button_index + 2) % n
            first_to_act = (self.button_index + 3) % n

        sb_player = self.players[sb_idx]
        bb_player = self.players[bb_idx]

        # Post SB
        sb_amt = min(sb_player.stack, self.small_blind)
        sb_player.stack -= sb_amt
        sb_player.current_bet += sb_amt
        sb_player.total_contributed += sb_amt
        if sb_player.stack == 0:
            sb_player.is_all_in = True

        # Post BB
        bb_amt = min(bb_player.stack, self.big_blind)
        bb_player.stack -= bb_amt
        bb_player.current_bet += bb_amt
        bb_player.total_contributed += bb_amt
        if bb_player.stack == 0:
            bb_player.is_all_in = True

        self.current_bet = max(sb_player.current_bet, bb_player.current_bet)
        self.min_raise = self.big_blind
        self.action_index = first_to_act

    def apply_action(
        self,
        action: Action,
        amount: float | None = None,
        player_id: str | int | None = None,
    ) -> ActionRecord:
        """Apply a player action, update stacks and bets, and advance turn.

        Parameters
        ----------
        action : Action
            Action to perform.
        amount : float | None, default=None
            Wager or raise amount (total chips in front of player).
        player_id : str | int | None, default=None
            Acting player ID, or current player if None.

        Returns
        -------
        ActionRecord
            Recorded hand action.

        Raises
        ------
        ValueError
            If action is illegal or amount is out of range.
        """
        p = self.get_player(player_id) if player_id is not None else self.current_player
        if p is None:
            raise ValueError("No active player to act")

        legal = self.legal_actions(p.player_id)
        if action not in legal:
            raise ValueError(f"Action '{action}' is not legal in this spot. Legal: {legal}")

        pot_before = self.total_pot
        chips_added = 0.0
        is_all_in = False

        if action == Action.FOLD:
            p.is_active = False

        elif action == Action.CHECK:
            pass

        elif action == Action.CALL:
            cost = self.bet_to_call(p.player_id)
            chips_added = cost
            p.stack -= cost
            p.current_bet += cost
            p.total_contributed += cost
            if p.stack == 0:
                p.is_all_in = True
                is_all_in = True

        elif action in (Action.BET, Action.RAISE):
            min_tot = self.min_legal_raise(p.player_id)
            max_tot = p.stack + p.current_bet

            if amount is None:
                target_total = min_tot
            else:
                target_total = float(amount)

            if target_total < min_tot and target_total < max_tot:
                raise ValueError(
                    f"Raise amount {target_total:.2f} is below minimum legal raise of {min_tot:.2f}"
                )
            if target_total > max_tot:
                target_total = max_tot

            cost = target_total - p.current_bet
            chips_added = cost
            p.stack -= cost
            p.current_bet = target_total
            p.total_contributed += cost

            raise_delta = target_total - self.current_bet
            if raise_delta >= self.min_raise:
                self.min_raise = raise_delta

            self.current_bet = target_total
            if p.stack == 0:
                p.is_all_in = True
                is_all_in = True

        elif action == Action.ALL_IN:
            chips_added = p.stack
            target_total = p.current_bet + p.stack
            p.stack = 0.0
            p.current_bet = target_total
            p.total_contributed += chips_added
            p.is_all_in = True
            is_all_in = True

            if target_total > self.current_bet:
                raise_delta = target_total - self.current_bet
                if raise_delta >= self.min_raise:
                    self.min_raise = raise_delta
                self.current_bet = target_total

        pot_after = self.total_pot
        record = ActionRecord(
            street=self.street,
            player_id=p.player_id,
            action=action,
            amount=chips_added,
            pot_before=pot_before,
            pot_after=pot_after,
            is_all_in=is_all_in,
        )
        self.history.append(record)

        # Advance to next active player
        self._advance_action_index()
        return record

    def _advance_action_index(self) -> None:
        """Move action index to the next eligible active player."""
        n = len(self.players)
        for step in range(1, n + 1):
            next_idx = (self.action_index + step) % n
            cand = self.players[next_idx]
            if cand.is_active and not cand.is_all_in:
                self.action_index = next_idx
                return

    def collect_bets(self) -> None:
        """Collect current bets into main pot and side pots."""
        bets = {p.player_id: p.current_bet for p in self.players if p.current_bet > 0}
        if not bets:
            return

        all_in_bets = sorted(
            {p.current_bet for p in self.players if p.is_all_in and p.current_bet > 0}
        )

        prev_level = 0.0
        for level in all_in_bets:
            tier_amount = level - prev_level
            if tier_amount <= 0:
                continue

            pot_tier_chips = 0.0
            eligible_ids = []

            for p in self.players:
                if p.current_bet > prev_level:
                    contrib = min(p.current_bet - prev_level, tier_amount)
                    pot_tier_chips += contrib
                    if p.is_active:
                        eligible_ids.append(p.player_id)

            if pot_tier_chips > 0:
                if not self.pots or (len(self.pots) == 1 and self.pots[0].amount == 0.0):
                    self.pots = [SidePot(amount=pot_tier_chips, eligible_player_ids=eligible_ids, is_main=True)]
                else:
                    self.pots.append(SidePot(amount=pot_tier_chips, eligible_player_ids=eligible_ids, is_main=False))

            prev_level = level

        # Remaining non-all-in chips
        remaining_tier_chips = 0.0
        remaining_eligible_ids = []
        for p in self.players:
            if p.current_bet > prev_level:
                rem = p.current_bet - prev_level
                remaining_tier_chips += rem
                if p.is_active:
                    remaining_eligible_ids.append(p.player_id)

        if remaining_tier_chips > 0:
            if not self.pots or (len(self.pots) == 1 and self.pots[0].amount == 0.0):
                self.pots = [SidePot(amount=remaining_tier_chips, eligible_player_ids=remaining_eligible_ids, is_main=True)]
            else:
                # Add to existing top pot or create side pot
                last_pot = self.pots[-1]
                if set(last_pot.eligible_player_ids) == set(remaining_eligible_ids):
                    last_pot.amount += remaining_tier_chips
                else:
                    self.pots.append(SidePot(amount=remaining_tier_chips, eligible_player_ids=remaining_eligible_ids, is_main=False))

        # Reset player current bets
        for p in self.players:
            p.reset_street_bet()

        self.current_bet = 0.0
        self.min_raise = self.big_blind

    def advance_street(
        self,
        new_cards: Sequence[Card | str] | str | None = None,
    ) -> Street:
        """Advance game state to the next street and update board cards.

        Parameters
        ----------
        new_cards : Sequence[Card | str] | str | None, default=None
            New community card(s) dealt on this street:
            - 3 cards when moving from PREFLOP to FLOP
            - 1 card when moving from FLOP to TURN
            - 1 card when moving from TURN to RIVER

        Returns
        -------
        Street
            The newly active street.

        Raises
        ------
        ValueError
            If already on River or invalid card count provided.
        """
        nxt = self.street.next_street
        if nxt is None:
            raise ValueError("Already on the River; cannot advance street further")

        self.collect_bets()

        if new_cards is not None:
            cards = (
                parse_cards(new_cards)
                if isinstance(new_cards, str)
                else [Card.from_str(c) if isinstance(c, str) else c for c in new_cards]
            )
            expected_new = 3 if nxt == Street.FLOP else 1
            if len(cards) != expected_new:
                raise ValueError(
                    f"Advancing to {nxt} requires {expected_new} new cards, got {len(cards)}"
                )
            self.board.extend(cards)

        self.street = nxt
        self.current_bet = 0.0
        self.min_raise = self.big_blind

        # Reset postflop action index to first active player after BTN
        n = len(self.players)
        first_postflop = (self.button_index + 1) % n
        for step in range(n):
            cand_idx = (first_postflop + step) % n
            cand = self.players[cand_idx]
            if cand.is_active and not cand.is_all_in:
                self.action_index = cand_idx
                break

        return self.street

    def deal_flop(self, cards: Sequence[Card | str] | str) -> Street:
        """Advance to Flop with 3 community cards."""
        if self.street != Street.PREFLOP:
            raise ValueError(f"Cannot deal flop from street {self.street}")
        return self.advance_street(cards)

    def deal_turn(self, card: Card | str) -> Street:
        """Advance to Turn with 1 community card."""
        if self.street != Street.FLOP:
            raise ValueError(f"Cannot deal turn from street {self.street}")
        return self.advance_street([card])

    def deal_river(self, card: Card | str) -> Street:
        """Advance to River with 1 community card."""
        if self.street != Street.TURN:
            raise ValueError(f"Cannot deal river from street {self.street}")
        return self.advance_street([card])

    # -------------------------------------------------------------------------
    # Equity Integration & Decision Recommendation
    # -------------------------------------------------------------------------

    def calculate_active_equities(
        self,
        num_sims: int = 2000,
        seed: int | None = None,
    ) -> dict[str | int, float]:
        """Calculate current equity for all active players holding known cards.

        Parameters
        ----------
        num_sims : int, default=2000
            Number of Monte Carlo simulations.
        seed : int or None, default=None
            RNG seed.

        Returns
        -------
        dict[str | int, float]
            Map of player_id -> equity fraction in ``[0.0, 1.0]``.
        """
        active_with_cards = [p for p in self.active_players if p.has_cards]
        if len(active_with_cards) < 2:
            return {p.player_id: 1.0 for p in active_with_cards}

        hands = [list(p.cards) for p in active_with_cards]  # type: ignore[arg-type]
        sim_results = calculate_equities(
            hands=hands,
            board=self.board if self.board else None,
            num_sims=num_sims,
            seed=seed,
        )

        return {
            p.player_id: res.equity
            for p, res in zip(active_with_cards, sim_results)
        }

    def recommend_decision(
        self,
        player_id: str | int | None = None,
        fold_equity: float = 0.0,
        future_bets: float = 0.0,
        num_sims: int = 2000,
        seed: int | None = None,
    ) -> DecisionResult:
        """Recommend mathematically optimal action for a player in this spot.

        Parameters
        ----------
        player_id : str | int | None, default=None
            Acting player ID, or current player if None.
        fold_equity : float, default=0.0
            Estimated probability that opponents fold to a bet/raise.
        future_bets : float, default=0.0
            Anticipated future chips won (implied odds).
        num_sims : int, default=2000
            Simulations to calculate equity.
        seed : int or None, default=None
            RNG seed.

        Returns
        -------
        DecisionResult
            Recommended action with EV comparison and strategic reasoning.
        """
        p = self.get_player(player_id) if player_id is not None else self.current_player
        if p is None:
            raise ValueError("No active player to evaluate")

        eq_map = self.calculate_active_equities(num_sims=num_sims, seed=seed)
        player_equity = eq_map.get(p.player_id, 0.5)

        return recommend_decision(
            equity=player_equity,
            pot_size=self.total_pot,
            bet_to_call=self.bet_to_call(p.player_id),
            stack_size=p.stack,
            fold_equity=fold_equity,
            future_bets=future_bets,
        )

    # -------------------------------------------------------------------------
    # Formatting and Serialization
    # -------------------------------------------------------------------------

    def summary(self) -> str:
        """Format a human-readable ASCII summary of the poker table state."""
        b_str = cards_to_str(self.board) if self.board else "None"
        curr_p = self.current_player
        curr_str = f"{curr_p.name} ({curr_p.position})" if curr_p else "Hand Complete"

        lines = [
            f"Table State - {self.street} | Blinds: {self.small_blind:.0f}/{self.big_blind:.0f} (Ante: {self.ante:.0f})",
            f"Board: [{b_str}] | Total Pot: {self.total_pot:.2f} chips (Current Bet: {self.current_bet:.2f})",
            "=" * 75,
            f"{'Seat':<6} {'Name':<12} {'Pos':<6} {'Stack':<10} {'Bet':<8} {'Cards':<12} {'Status'}",
            "-" * 75,
        ]

        for idx, p in enumerate(self.players):
            btn = " (BTN)" if idx == self.button_index else ""
            pos_label = f"{p.position}{btn}"
            cards_label = cards_to_str(p.cards) if p.cards else "-"
            status = "All-In" if p.is_all_in else ("Active" if p.is_active else "Folded")
            lines.append(
                f"{idx + 1:<6} {p.name:<12} {pos_label:<6} {p.stack:<10.2f} "
                f"{p.current_bet:<8.2f} {cards_label:<12} {status}"
            )

        lines.append("-" * 75)
        lines.append(f"Action on: {curr_str}")
        if curr_p and not self.is_hand_over:
            call_amt = self.bet_to_call(curr_p.player_id)
            p_odds = self.pot_odds(curr_p.player_id)
            lines.append(f"Cost to call: {call_amt:.2f} chips | Pot Odds: {p_odds * 100:.1f}% | SPR: {self.spr(curr_p.player_id):.2f}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete game state to a dictionary suitable for JSON."""
        return {
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "ante": self.ante,
            "street": self.street.value,
            "button_index": self.button_index,
            "action_index": self.action_index,
            "current_bet": self.current_bet,
            "min_raise": self.min_raise,
            "pot_size": self.pot_size,
            "total_pot": self.total_pot,
            "board": cards_to_str(self.board),
            "is_hand_over": self.is_hand_over,
            "players": [p.to_dict() for p in self.players],
            "pots": [pot.to_dict() for pot in self.pots],
            "history": [rec.to_dict() for rec in self.history],
        }

    # -------------------------------------------------------------------------
    # Factory Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def create_heads_up(
        cls,
        small_blind: float,
        big_blind: float,
        hero_stack: float,
        villain_stack: float,
        hero_cards: Sequence[Card | str] | str | None = None,
        villain_cards: Sequence[Card | str] | str | None = None,
        hero_is_btn: bool = True,
        hero_name: str = "Hero",
        villain_name: str = "Villain",
        ante: float = 0.0,
    ) -> GameState:
        """Create a standard Heads-Up table game state.

        Parameters
        ----------
        small_blind : float
            Small blind amount.
        big_blind : float
            Big blind amount.
        hero_stack : float
            Hero's chip stack.
        villain_stack : float
            Villain's chip stack.
        hero_cards : Sequence[Card | str] | str | None, default=None
            Hero's hole cards.
        villain_cards : Sequence[Card | str] | str | None, default=None
            Villain's hole cards.
        hero_is_btn : bool, default=True
            Whether Hero holds the button (SB in heads-up).
        hero_name : str, default="Hero"
            Hero's display name.
        villain_name : str, default="Villain"
            Villain's display name.
        ante : float, default=0.0
            Ante per player.

        Returns
        -------
        GameState
            Initialized heads-up table.
        """
        h_cards = (
            tuple(parse_cards(hero_cards))  # type: ignore[assignment]
            if hero_cards is not None
            else None
        )
        v_cards = (
            tuple(parse_cards(villain_cards))  # type: ignore[assignment]
            if villain_cards is not None
            else None
        )

        if hero_is_btn:
            p1 = PlayerState(
                player_id="hero",
                name=hero_name,
                position=Position.BTN,
                stack=hero_stack,
                cards=h_cards,
            )
            p2 = PlayerState(
                player_id="villain",
                name=villain_name,
                position=Position.BB,
                stack=villain_stack,
                cards=v_cards,
            )
            btn_idx = 0
        else:
            p1 = PlayerState(
                player_id="villain",
                name=villain_name,
                position=Position.BTN,
                stack=villain_stack,
                cards=v_cards,
            )
            p2 = PlayerState(
                player_id="hero",
                name=hero_name,
                position=Position.BB,
                stack=hero_stack,
                cards=h_cards,
            )
            btn_idx = 0

        state = cls(
            small_blind=small_blind,
            big_blind=big_blind,
            ante=ante,
            players=[p1, p2],
            button_index=btn_idx,
        )
        state.post_blinds()
        return state

    @classmethod
    def create_6max(
        cls,
        small_blind: float,
        big_blind: float,
        stacks: Sequence[float],
        names: Sequence[str] | None = None,
        cards: Sequence[Sequence[Card | str] | str | None] | None = None,
        button_index: int = 0,
        ante: float = 0.0,
    ) -> GameState:
        """Create a standard 6-max table game state with position assignments.

        Positions assigned in order: UTG, HJ, CO, BTN, SB, BB (relative to button).

        Parameters
        ----------
        small_blind : float
            Small blind.
        big_blind : float
            Big blind.
        stacks : Sequence[float]
            Chip stacks for up to 6 players.
        names : Sequence[str] | None, default=None
            Player names.
        cards : Sequence[Sequence[Card | str] | str | None] | None, default=None
            Hole cards per player.
        button_index : int, default=0
            Index of button seat.
        ante : float, default=0.0
            Ante per player.

        Returns
        -------
        GameState
            Initialized 6-max table.
        """
        n = len(stacks)
        if not (2 <= n <= 6):
            raise ValueError(f"6-max table requires 2 to 6 players, got {n}")

        positions_6max = [
            Position.UTG,
            Position.HJ,
            Position.CO,
            Position.BTN,
            Position.SB,
            Position.BB,
        ]
        # Align positions so button_index has Position.BTN
        btn_pos_idx = positions_6max.index(Position.BTN)
        offset = btn_pos_idx - button_index

        player_states = []
        for i, stack in enumerate(stacks):
            name = names[i] if names and i < len(names) else f"Player {i+1}"
            pos = positions_6max[(i + offset) % len(positions_6max)]
            c_raw = cards[i] if cards and i < len(cards) else None
            p_cards = (
                tuple(parse_cards(c_raw))  # type: ignore[assignment]
                if c_raw is not None
                else None
            )

            p = PlayerState(
                player_id=f"p{i+1}",
                name=name,
                position=pos,
                stack=stack,
                cards=p_cards,
            )
            player_states.append(p)

        state = cls(
            small_blind=small_blind,
            big_blind=big_blind,
            ante=ante,
            players=player_states,
            button_index=button_index,
        )
        state.post_blinds()
        return state

