"""Command-line interface (CLI) for the poker-equity simulator.

Provides subcommands for equity calculation, range analysis, pot odds,
outs, decision recommendations, table tracking, and equity curves.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Any, Callable, Sequence

import poker_equity
import poker_equity.decision
from poker_equity.all_in import (
    RunItTimesResult,
    exact_run_it_twice_turn,
    simulate_run_it_n,
    simulate_run_it_twice,
)
from poker_equity.breakdown import range_vs_hand_breakdown
from poker_equity.card import Card, Rank, Suit
from poker_equity.constants import cards_to_str, parse_cards
from poker_equity.deck import Deck
from poker_equity.equity_curve import (
    EquityCurve,
    NextStreetDistribution,
    Street,
    calculate_equity_curve,
    calculate_next_street_distribution,
)
from poker_equity.evaluator import evaluate_hand
from poker_equity.parallel import calculate_equities_parallel
from poker_equity.presets import PRESET_RANGES, get_preset_range, top_percent_range
from poker_equity.range import Range, combo_to_cell, combo_to_str
from poker_equity.simulation import (
    ProgressCallback,
    ProgressInfo,
    calculate_equities,
)


# =========================================================================
# Terminal Color Formatting
# =========================================================================


class Colors:
    """ANSI color codes for terminal highlighting."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Standard colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Poker suit colors
    HEART = "\033[91m"     # Bright red
    DIAMOND = "\033[94m"   # Bright blue (or red)
    CLUB = "\033[92m"      # Bright green
    SPADE = "\033[97m"     # Bright white/black


def colorize(text: str, color_code: str, enable: bool = True) -> str:
    """Apply an ANSI color to text if coloring is enabled."""
    if not enable:
        return text
    return f"{color_code}{text}{Colors.RESET}"


def should_enable_color(no_color_flag: bool = False) -> bool:
    """Determine whether terminal coloring should be enabled."""
    if no_color_flag:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    # Check if stdout is a TTY
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# =========================================================================
# Terminal Formatting & Card Helpers
# =========================================================================

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text.

    Parameters
    ----------
    text : str
        Input string possibly containing ANSI codes.

    Returns
    -------
    str
        Clean string with all escape sequences removed.
    """
    return _ANSI_RE.sub("", text)


def visible_length(text: str) -> int:
    """Return visible width of string ignoring ANSI codes.

    Parameters
    ----------
    text : str
        String to measure.

    Returns
    -------
    int
        Visible character length.
    """
    return len(strip_ansi(text))


def pad_visible(text: str, width: int, align: str = "left") -> str:
    """Pad a string containing ANSI escape codes to a target visible width.

    Parameters
    ----------
    text : str
        String to pad.
    width : int
        Target visible width.
    align : str
        Alignment mode: ``"left"``, ``"right"``, or ``"center"``.

    Returns
    -------
    str
        Padded string.
    """
    vlen = visible_length(text)
    pad = max(0, width - vlen)
    if align == "right":
        return (" " * pad) + text
    elif align == "center":
        left = pad // 2
        return (" " * left) + text + (" " * (pad - left))
    return text + (" " * pad)


def parse_cards_flexible(text: str | None) -> list[Card]:
    """Parse cards from flexible string format (e.g. 'Ah Kh', 'Ah,Kh', or 'AhKh').

    Parameters
    ----------
    text : str | None
        String containing card tokens.

    Returns
    -------
    list[Card]
        List of parsed Card objects.
    """
    if not text:
        return []
    s = text.strip()
    if not s:
        return []
    if " " in s or "," in s:
        tokens = s.replace(",", " ").split()
    elif len(s) % 2 == 0:
        tokens = [s[i : i + 2] for i in range(0, len(s), 2)]
    else:
        tokens = [s]
    return [Card.from_str(t) for t in tokens]


def parse_player_hand_input(
    text: str | tuple[Card, Card] | list[Card] | None,
) -> tuple[Card, Card] | str | None:
    """Parse player hand input into 2-card tuple, range string, or None (random).

    Parameters
    ----------
    text : str | tuple[Card, Card] | list[Card] | None
        Input string or card sequence representing hole cards, a range, or random.

    Returns
    -------
    tuple[Card, Card] | str | None
        Normalized hand representation.
    """
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        if len(text) == 2 and isinstance(text[0], Card) and isinstance(text[1], Card):
            return (text[0], text[1])
        s = " ".join(str(c) for c in text)
    else:
        s = str(text).strip()

    if not s or s.lower() in ("random", "?", "any", "100%"):
        return None

    try:
        cards = parse_cards_flexible(s)
        if len(cards) == 2:
            return (cards[0], cards[1])
    except Exception:
        pass

    return s


def parse_range_or_preset(spec: str) -> Range | tuple[Card, Card]:
    """Parse range specification into a Range instance or 2-card tuple.

    Supports:
    - 2-card holdings: e.g. ``'AhKh'`` or ``'Ah Kh'``
    - Standard poker range syntax: e.g. ``'QQ+, AKs'``
    - Weighted range syntax: e.g. ``'AA:0.8, KK:0.5'``
    - Percentage ranges: e.g. ``'20%'`` or ``'top 15%'``
    - Preset range names: e.g. ``'BB_VS_BTN_3BET'``

    Parameters
    ----------
    spec : str
        String representing a range, percentage, preset, or 2-card hand.

    Returns
    -------
    Range | tuple[Card, Card]
        Parsed Range instance or canonical 2-card tuple.

    Raises
    ------
    ValueError
        If the input string cannot be parsed as a valid hand, range, or preset.
    """
    s = spec.strip()
    if not s:
        raise ValueError("Range specification cannot be empty")

    # Check 2-card hand first
    try:
        cards = parse_cards_flexible(s)
        if len(cards) == 2:
            return (cards[0], cards[1])
    except Exception:
        pass

    # Check percentage string (e.g. "20%", "top 15%")
    if s.endswith("%"):
        num_str = s.rstrip("%").lower().replace("top", "").strip()
        try:
            pct = float(num_str)
            return top_percent_range(pct)
        except ValueError:
            pass

    # Check preset name
    s_upper = s.upper().replace("-", "_").replace(" ", "_")
    if s_upper in PRESET_RANGES:
        return get_preset_range(s_upper)

    # Standard or weighted range string
    return Range.from_str(s)


def parse_equity_value(val: str | float | int) -> float:
    """Parse equity value from fraction (0.35) or percentage string ('35%', '35').

    Parameters
    ----------
    val : str | float | int
        Equity representation.

    Returns
    -------
    float
        Parsed equity in range [0.0, 1.0].

    Raises
    ------
    ValueError
        If the value cannot be parsed as a float.
    """
    if isinstance(val, (int, float)):
        f = float(val)
        return f / 100.0 if f > 1.0 else f
    s = str(val).strip()
    if s.endswith("%"):
        s = s[:-1].strip()
        f = float(s) / 100.0
        return f
    f = float(s)
    return f / 100.0 if f > 1.0 else f


def format_card_colored(
    card: Card | str | None,
    enable: bool = True,
    use_symbol: bool = True,
) -> str:
    """Format a single card with ANSI suit color and suit symbol.

    Parameters
    ----------
    card : Card | str | None
        Card object or 2-char string.
    enable : bool
        Whether to apply ANSI color.
    use_symbol : bool
        Whether to use Unicode suit symbol (e.g. '♥') or suit char ('h').

    Returns
    -------
    str
        Formatted card string.
    """
    if card is None:
        return "??"
    if isinstance(card, str):
        try:
            c = Card.from_str(card)
        except Exception:
            return card
    else:
        c = card

    suit_char = c.suit.symbol if use_symbol else c.suit.char
    suit_colors = {
        Suit.HEARTS: Colors.HEART,
        Suit.DIAMONDS: Colors.DIAMOND,
        Suit.CLUBS: Colors.CLUB,
        Suit.SPADES: Colors.SPADE,
    }
    color = suit_colors.get(c.suit, Colors.RESET)
    card_str = f"{c.rank.char}{suit_char}"
    return colorize(card_str, color, enable)


def format_cards_colored(
    cards: Sequence[Card | str] | str | None,
    enable: bool = True,
    use_symbol: bool = True,
) -> str:
    """Format a sequence of cards with ANSI suit colors separated by spaces.

    Parameters
    ----------
    cards : Sequence[Card | str] | str | None
        Collection of cards or card string.
    enable : bool
        Whether to apply ANSI color.
    use_symbol : bool
        Whether to use Unicode suit symbols.

    Returns
    -------
    str
        Space-separated formatted cards.
    """
    if cards is None:
        return ""
    if isinstance(cards, str):
        parsed = parse_cards_flexible(cards)
    else:
        parsed = [c if isinstance(c, Card) else Card.from_str(c) for c in cards]
    return " ".join(
        format_card_colored(c, enable=enable, use_symbol=use_symbol)
        for c in parsed
    )


def make_equity_bar(
    equity: float,
    width: int = 10,
    enable: bool = True,
    use_color: bool | None = None,
) -> str:
    """Construct an ASCII/Unicode progress bar showing equity fraction.

    Parameters
    ----------
    equity : float
        Equity fraction between 0.0 and 1.0.
    width : int
        Number of block characters inside the bar.
    enable : bool
        Whether to apply ANSI color codes.
    use_color : bool | None, default=None
        Alias for ``enable`` to support flexible keyword calling.

    Returns
    -------
    str
        Formatted equity bar e.g. ``'[██████░░░░]'``.
    """
    if use_color is not None:
        enable = use_color

    eq_clamped = max(0.0, min(1.0, equity))
    filled = int(round(eq_clamped * width))
    empty = max(0, width - filled)

    if not enable:
        return f"[{'█' * filled}{'░' * empty}]"

    color = (
        Colors.GREEN
        if equity >= 0.55
        else (Colors.YELLOW if equity >= 0.40 else Colors.RED)
    )
    bar_filled = colorize("█" * filled, color, enable=True)
    bar_empty = colorize("░" * empty, Colors.DIM, enable=True)
    return f"[{bar_filled}{bar_empty}]"


_RANK_NAMES: dict[int, str] = {
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
    10: "Ten",
    11: "Jack",
    12: "Queen",
    13: "King",
    14: "Ace",
}

_RANK_PLURALS: dict[int, str] = {
    2: "Deuces",
    3: "Treys",
    4: "Fours",
    5: "Fives",
    6: "Sixes",
    7: "Sevens",
    8: "Eights",
    9: "Nines",
    10: "Tens",
    11: "Jacks",
    12: "Queens",
    13: "Kings",
    14: "Aces",
}


def get_hand_type_description(
    hand: tuple[Card, Card] | str | None,
    board: Sequence[Card] | None = None,
) -> str:
    """Return a descriptive name for a hand (preflop texture or postflop made hand).

    Parameters
    ----------
    hand : tuple[Card, Card] | str | None
        Hole cards tuple, range string, or None.
    board : Sequence[Card] | None
        Community board cards, if any.

    Returns
    -------
    str
        Human-readable description.
    """
    if hand is None:
        return "Random (100%)"
    if isinstance(hand, str):
        return f"Range ({hand})"

    c1, c2 = hand
    board_cards = list(board) if board else []

    # If postflop board exists (3+ cards), evaluate made hand
    if len(board_cards) >= 3:
        try:
            res = evaluate_hand(board_cards + [c1, c2])
            return res.description
        except Exception:
            pass

    # Preflop hand description
    if c1.rank == c2.rank:
        plural = _RANK_PLURALS.get(c1.rank.value, f"{c1.rank.char}s")
        return f"Pocket {plural}"

    high_c = max(c1, c2, key=lambda c: c.rank)
    low_c = min(c1, c2, key=lambda c: c.rank)
    high_name = _RANK_NAMES.get(high_c.rank.value, high_c.rank.char)
    low_name = _RANK_NAMES.get(low_c.rank.value, low_c.rank.char)

    if c1.suit == c2.suit:
        return f"{high_name}-{low_name} suited"
    return f"{high_name}-{low_name} offsuit"


def make_progress_callback() -> tuple[ProgressCallback, Callable[[], None]]:
    """Create a real-time progress callback for terminal stderr updates.

    Returns
    -------
    tuple[ProgressCallback, Callable[[], None]]
        Tuple of (on_progress callback, cleanup finish function).
    """
    last_update = [0.0]

    def on_progress(info: ProgressInfo) -> None:
        now = time.perf_counter()
        if now - last_update[0] < 0.05 and info.pct_complete < 1.0:
            return  # Throttle updates to ~20 fps
        last_update[0] = now
        bar_len = 20
        filled = int(info.pct_complete * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stderr.write(
            f"\rSimulating: [{bar}] {info.pct_complete * 100:5.1f}% "
            f"({info.sims_completed:,}/{info.sims_total:,}) "
        )
        sys.stderr.flush()

    def finish() -> None:
        sys.stderr.write("\r" + " " * 65 + "\r")
        sys.stderr.flush()

    return on_progress, finish


# =========================================================================
# Argument Parser Construction
# =========================================================================


def create_parser() -> argparse.ArgumentParser:
    """Construct the main argument parser with all subcommands.

    Returns
    -------
    argparse.ArgumentParser
        Configured CLI parser with subcommands and arguments.
    """
    version_str = f"poker-equity {getattr(poker_equity, '__version__', '0.1.0')}"

    # Common options available both globally and on subcommands
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored ANSI terminal output.",
    )
    common_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in structured JSON format.",
    )

    parser = argparse.ArgumentParser(
        prog="poker-equity",
        description="High-performance Monte Carlo poker equity simulator & decision engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common_parser],
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=version_str,
        help="Show program version and exit.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="subcommands",
        description="Available operations",
        help="Run 'poker-equity <command> --help' for command-specific arguments.",
    )

    # ---------------------------------------------------------------------
    # Subcommand: equity
    # ---------------------------------------------------------------------
    p_eq = subparsers.add_parser(
        "equity",
        parents=[common_parser],
        help="Calculate Monte Carlo equity between hands or players.",
        description="Simulate win/tie/loss probabilities for Texas Hold'em hands.",
    )
    p_eq.add_argument(
        "hands",
        nargs="*",
        help="Player hands as positional arguments (e.g. 'AhKh' 'QsQd').",
    )
    p_eq.add_argument(
        "--hero",
        "-H",
        type=str,
        help="Hero hole cards (e.g. 'Ah Kh' or 'AhKh').",
    )
    p_eq.add_argument(
        "--villain",
        "-V",
        type=str,
        help="Villain hole cards (e.g. 'Qc Qd' or 'random').",
    )
    p_eq.add_argument(
        "--players",
        "-p",
        nargs="+",
        help="Multi-player hands (e.g. -p 'Ah As' 'Kh Ks' 'Qh Qs').",
    )
    p_eq.add_argument(
        "--board",
        "-b",
        type=str,
        default=None,
        help="Community cards (e.g. '2h 7h 8c' or '2h7h8c3c4c').",
    )
    p_eq.add_argument(
        "--dead",
        "-d",
        type=str,
        default=None,
        help="Dead / folded cards to exclude from the deck (e.g. 'Jc Ts').",
    )
    p_eq.add_argument(
        "--sims",
        "-n",
        type=int,
        default=10_000,
        help="Number of Monte Carlo simulations (default: 10,000).",
    )
    p_eq.add_argument(
        "--converge",
        "-c",
        type=float,
        default=None,
        metavar="EPS",
        help="Stop early when 95%% CI half-width is below EPS (e.g. 0.005 for ±0.5%%).",
    )
    p_eq.add_argument(
        "--parallel",
        "-j",
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="N",
        help="Run across N worker processes (default: all CPU cores if -j given without value).",
    )
    p_eq.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="RNG seed for deterministic evaluation.",
    )
    p_eq.add_argument(
        "--progress",
        action="store_true",
        help="Display real-time simulation progress in terminal.",
    )

    # ---------------------------------------------------------------------
    # Subcommand: range
    # ---------------------------------------------------------------------
    p_rng = subparsers.add_parser(
        "range",
        parents=[common_parser],
        help="Calculate range-vs-range or hand-vs-range equity.",
        description="Evaluate equity against weighted or unweighted hand ranges.",
    )
    p_rng.add_argument(
        "ranges",
        nargs="*",
        help="Hero and villain ranges as positional arguments (e.g. 'AhKh' 'TT+, AJs+').",
    )
    p_rng.add_argument(
        "--hero",
        "-H",
        type=str,
        default=None,
        help="Hero hand or range string (e.g. 'Ah Kh', 'QQ+', 'TT+:0.5').",
    )
    p_rng.add_argument(
        "--villain",
        "-V",
        type=str,
        default=None,
        help="Villain range string (e.g. 'TT+, AJs+, KQs', '25%').",
    )
    p_rng.add_argument(
        "--board",
        "-b",
        type=str,
        default=None,
        help="Community cards (e.g. '2h 7h 8c').",
    )
    p_rng.add_argument(
        "--dead",
        "-d",
        type=str,
        default=None,
        help="Dead / folded cards to exclude from deck and ranges (e.g. 'Jc Ts').",
    )
    p_rng.add_argument(
        "--sims",
        "-n",
        type=int,
        default=10_000,
        help="Number of simulations (default: 10,000).",
    )
    p_rng.add_argument(
        "--converge",
        "-c",
        type=float,
        default=None,
        metavar="EPS",
        help="Stop early when 95%% CI half-width is below EPS (e.g. 0.005 for ±0.5%%).",
    )
    p_rng.add_argument(
        "--parallel",
        "-j",
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="N",
        help="Run across N worker processes (default: all CPU cores if -j given without value).",
    )
    p_rng.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="RNG seed for deterministic evaluation.",
    )
    p_rng.add_argument(
        "--progress",
        action="store_true",
        help="Display real-time simulation progress in terminal.",
    )
    p_rng.add_argument(
        "--breakdown",
        "-k",
        type=int,
        nargs="?",
        const=10,
        default=None,
        metavar="N",
        help="Show equity breakdown for top N combos in the range (for hand vs range).",
    )

    # ---------------------------------------------------------------------
    # Subcommand: odds
    # ---------------------------------------------------------------------
    p_odds = subparsers.add_parser(
        "odds",
        parents=[common_parser],
        help="Calculate pot odds, expected value (EV), and implied odds.",
        description="Compute pot odds requirement, EV of calling, and future bets needed.",
    )
    p_odds.add_argument(
        "numbers",
        nargs="*",
        help="Positional numbers: <pot> <call> [<equity>] (e.g. '100 50' or '100 50 35%%').",
    )
    p_odds.add_argument(
        "--pot",
        "-p",
        type=float,
        default=None,
        help="Current pot size before calling.",
    )
    p_odds.add_argument(
        "--call",
        "-c",
        type=float,
        default=None,
        help="Amount to call.",
    )
    p_odds.add_argument(
        "--equity",
        "-e",
        type=str,
        default=None,
        help="Hero equity as a fraction (0.0 to 1.0) or percentage (e.g. '35%%' or '35').",
    )
    p_odds.add_argument(
        "--future-bets",
        "-f",
        type=float,
        default=0.0,
        help="Expected future chips won on later streets (implied odds).",
    )
    p_odds.add_argument(
        "--future-losses",
        "-l",
        type=float,
        default=0.0,
        help="Potential future chips lost on later streets if outdrawn (reverse implied odds).",
    )
    p_odds.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Prompt interactively for pot, call, and equity values.",
    )

    # ---------------------------------------------------------------------
    # Subcommand: outs
    # ---------------------------------------------------------------------
    p_outs = subparsers.add_parser(
        "outs",
        parents=[common_parser],
        help="Calculate drawing outs and rule-of-2/4 equities.",
        description="Identify winning outs on flop or turn and convert to equity.",
    )
    p_outs.add_argument(
        "cards",
        nargs="*",
        help="Positional cards: <hero> <villain> <board> (e.g. 'AhKh' 'QcQd' '2h 7h 8c').",
    )
    p_outs.add_argument(
        "--hero",
        "-H",
        type=str,
        default=None,
        help="Hero hole cards (e.g. 'Ah Kh').",
    )
    p_outs.add_argument(
        "--villain",
        "-V",
        type=str,
        default=None,
        help="Villain hole cards (e.g. 'Qc Qd').",
    )
    p_outs.add_argument(
        "--board",
        "-b",
        type=str,
        default=None,
        help="Flop (3 cards) or Turn (4 cards) (e.g. '2h 7h 8c').",
    )
    p_outs.add_argument(
        "--dead",
        "-d",
        type=str,
        default=None,
        help="Known dead or folded cards to exclude from the deck (e.g. 'Jc Ts').",
    )
    p_outs.add_argument(
        "--method",
        "-m",
        choices=["exact", "rule", "adjusted_rule"],
        default="exact",
        help="Method to calculate equity from outs (default: exact).",
    )
    p_outs.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Prompt interactively for hero, villain, and board cards.",
    )

    # ---------------------------------------------------------------------
    # Subcommand: decision
    # ---------------------------------------------------------------------
    p_dec = subparsers.add_parser(
        "decision",
        parents=[common_parser],
        help="Strategic decision recommendation (Fold, Check, Call, Bet, Raise, All-In).",
        description="Recommends the highest EV action based on equity, pot odds, fold equity, and stack size.",
    )
    p_dec.add_argument(
        "numbers",
        nargs="*",
        help="Positional arguments: <equity> <pot> [<call>] [<stack>] OR <hero> <villain> [<board>].",
    )
    p_dec.add_argument(
        "--equity",
        "-e",
        type=str,
        default=None,
        help="Hero equity as a fraction (0.0 to 1.0) or percentage (e.g. '35%%' or '0.35').",
    )
    p_dec.add_argument(
        "--pot",
        "-p",
        type=float,
        default=None,
        help="Current pot size before calling.",
    )
    p_dec.add_argument(
        "--call",
        "-c",
        type=float,
        default=0.0,
        help="Bet to call (default: 0.0 for checking spots).",
    )
    p_dec.add_argument(
        "--stack",
        "-s",
        type=float,
        default=None,
        help="Hero remaining stack size.",
    )
    p_dec.add_argument(
        "--fold-equity",
        "-fe",
        type=str,
        default=None,
        help="Estimated fold equity (0.0 to 1.0 or percentage, e.g. '20%%').",
    )
    p_dec.add_argument(
        "--raise-size",
        "-r",
        type=float,
        default=None,
        help="Explicit bet or raise size in chips.",
    )
    p_dec.add_argument(
        "--future-bets",
        "-f",
        type=float,
        default=0.0,
        help="Anticipated future chips won on later streets (implied odds).",
    )
    p_dec.add_argument(
        "--hero",
        "-H",
        type=str,
        default=None,
        help="Hero hole cards if simulating equity directly (e.g. 'Ah Kh').",
    )
    p_dec.add_argument(
        "--villain",
        "-V",
        type=str,
        default=None,
        help="Villain hole cards if simulating equity directly (e.g. 'Qc Qd').",
    )
    p_dec.add_argument(
        "--board",
        "-b",
        type=str,
        default=None,
        help="Community cards if simulating equity directly (e.g. '2h 7h 8c').",
    )
    p_dec.add_argument(
        "--sims",
        "-n",
        type=int,
        default=5000,
        help="Simulations count when estimating equity from hands (default: 5000).",
    )
    p_dec.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for deterministic equity simulation.",
    )
    p_dec.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Prompt interactively for decision parameters.",
    )

    # ---------------------------------------------------------------------
    # Subcommand: table
    # ---------------------------------------------------------------------
    p_tbl = subparsers.add_parser(
        "table",
        parents=[common_parser],
        help="Create and simulate an interactive poker table.",
        description=(
            "Interactive table simulator with betting rounds, side pots, "
            "and live decision recommendations. Use --interactive (-i) "
            "to step through betting actions in a REPL loop."
        ),
    )
    p_tbl.add_argument(
        "--blinds",
        type=str,
        default="1/2",
        help="Small and big blind format 'SB/BB' (default: '1/2').",
    )
    p_tbl.add_argument(
        "--hero-stack",
        type=float,
        default=100.0,
        help="Hero chip stack (default: 100).",
    )
    p_tbl.add_argument(
        "--villain-stack",
        type=float,
        default=100.0,
        help="Villain chip stack (default: 100).",
    )
    p_tbl.add_argument(
        "--hero-cards",
        type=str,
        default=None,
        help="Hero hole cards (e.g. 'Ah As').",
    )
    p_tbl.add_argument(
        "--villain-cards",
        type=str,
        default=None,
        help="Villain hole cards (e.g. 'Kh Ks').",
    )
    p_tbl.add_argument(
        "--board",
        type=str,
        default=None,
        help="Pre-set community cards (e.g. 'Ah Kh Qh' for flop).",
    )
    p_tbl.add_argument(
        "--ante",
        type=float,
        default=0.0,
        help="Ante per player (default: 0).",
    )
    p_tbl.add_argument(
        "--hero-is-bb",
        action="store_true",
        default=False,
        help="Place Hero in the BB (default: Hero is BTN/SB).",
    )
    p_tbl.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        default=False,
        help="Enter interactive REPL mode to play betting actions.",
    )
    p_tbl.add_argument(
        "--actions",
        type=str,
        default=None,
        help=(
            "Pre-scripted action sequence separated by commas "
            "(e.g. 'call,check,check,bet 6,call'). "
            "Applied before entering interactive mode or displaying state."
        ),
    )
    p_tbl.add_argument(
        "--recommend",
        action="store_true",
        default=False,
        help="Show live decision recommendations at each action point.",
    )
    p_tbl.add_argument(
        "--sims",
        type=int,
        default=2000,
        help="Monte Carlo simulations for equity/recommendation (default: 2000).",
    )
    p_tbl.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="RNG seed for deterministic deck dealing and equity simulation.",
    )

    # ---------------------------------------------------------------------
    # Subcommand: twice (run-it-twice)
    # ---------------------------------------------------------------------
    p_tw = subparsers.add_parser(
        "twice",
        aliases=["rit", "run-it-twice"],
        parents=[common_parser],
        help="Simulate all-in Run-It-Twice (or N-times) board runs.",
        description=(
            "Run-It-Twice (or Run-It-N-Times) all-in simulation for heads-up pots. "
            "Evaluates multi-board equities, scoop probabilities, split probabilities, "
            "pot payouts, and variance reduction."
        ),
    )
    p_tw.add_argument(
        "cards",
        nargs="*",
        help="Positional cards: <hero> <villain> [<board>] (e.g. 'Ah Kh' 'Qc Qd' '2h 7h 8c').",
    )
    p_tw.add_argument(
        "--hero",
        "-H",
        type=str,
        default=None,
        help="Hero hole cards (e.g. 'Ah Kh').",
    )
    p_tw.add_argument(
        "--villain",
        "-V",
        type=str,
        default=None,
        help="Villain hole cards (e.g. 'Qc Qd').",
    )
    p_tw.add_argument(
        "--board",
        "-b",
        type=str,
        default=None,
        help="Existing board cards: 0 (preflop), 3 (flop), or 4 (turn).",
    )
    p_tw.add_argument(
        "--runs",
        "-r",
        type=int,
        default=2,
        help="Number of boards to run (default: 2 for Run-It-Twice).",
    )
    p_tw.add_argument(
        "--pot",
        "-p",
        type=float,
        default=None,
        help="Total pot size for expected chip/dollar payouts.",
    )
    p_tw.add_argument(
        "--exact",
        "-e",
        action="store_true",
        default=False,
        help="Use exact combinatorial enumeration (turn boards only).",
    )
    p_tw.add_argument(
        "--sims",
        "-n",
        type=int,
        default=10_000,
        help="Number of Monte Carlo simulations (default: 10,000).",
    )
    p_tw.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="RNG seed for deterministic evaluation.",
    )

    # ---------------------------------------------------------------------
    # Subcommand: curve
    # ---------------------------------------------------------------------
    p_crv = subparsers.add_parser(
        "curve",
        parents=[common_parser],
        help="Track street-by-street equity transitions across board cards.",
        description=(
            "Analyzes equity curve from Preflop -> Flop -> Turn -> River, "
            "identifying lead changes, equity swings, turning points, and bad beats."
        ),
    )
    p_crv.add_argument(
        "cards",
        nargs="*",
        help="Positional cards: <hero> <villain> [<board>] (e.g. 'Ah As' 'Kh Ks' '2h 7h 8c 3c 4c').",
    )
    p_crv.add_argument(
        "--hero",
        "-H",
        type=str,
        default=None,
        help="Hero hole cards (e.g. 'Ah As').",
    )
    p_crv.add_argument(
        "--villain",
        "-V",
        type=str,
        default=None,
        help="Villain hole cards (e.g. 'Kh Ks').",
    )
    p_crv.add_argument(
        "--board",
        "-b",
        type=str,
        default=None,
        help="Board cards (3 to 5 cards, e.g. '2h 7h 8c 3c 4c').",
    )
    p_crv.add_argument(
        "--sims",
        "-n",
        type=int,
        default=2000,
        help="Simulations per street (default: 2000).",
    )
    p_crv.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="RNG seed for deterministic evaluation.",
    )
    p_crv.add_argument(
        "--chart",
        "--ascii",
        action="store_true",
        default=False,
        help="Display an ASCII trajectory visualization of equity across streets.",
    )
    p_crv.add_argument(
        "--next",
        action="store_true",
        default=False,
        help="Analyze equity distribution across all possible next-street cards (flop or turn).",
    )

    return parser


# =========================================================================
# Subcommand Execution Handlers
# =========================================================================


def handle_equity_command(args: argparse.Namespace, use_color: bool = True) -> int:
    """Execute the 'equity' subcommand."""
    # Determine player hands from positional hands, --players, or --hero/--villain
    pos_hands = getattr(args, "hands", None) or []
    if len(pos_hands) >= 2:
        raw_hands = list(pos_hands)
    elif len(pos_hands) == 1:
        villain = args.villain if args.villain is not None else "random"
        raw_hands = [pos_hands[0], villain]
    elif args.players:
        raw_hands = list(args.players)
    elif args.hero:
        villain = args.villain if args.villain is not None else "random"
        raw_hands = [args.hero, villain]
    else:
        sys.stderr.write(
            "Error: Must specify either --hero and --villain or --players (or positional hands).\n"
        )
        return 2

    # Parse board cards
    try:
        board_cards = parse_cards_flexible(args.board) if args.board else []
    except Exception as exc:
        sys.stderr.write(f"Error parsing board cards '{args.board}': {exc}\n")
        return 2

    # Parse dead cards
    try:
        dead_cards = (
            parse_cards_flexible(getattr(args, "dead", None))
            if getattr(args, "dead", None)
            else []
        )
    except Exception as exc:
        sys.stderr.write(f"Error parsing dead cards '{args.dead}': {exc}\n")
        return 2

    # Parse player hands
    try:
        parsed_hands = [parse_player_hand_input(h) for h in raw_hands]
    except Exception as exc:
        sys.stderr.write(f"Error parsing player hands: {exc}\n")
        return 2

    sims = max(1, args.sims)

    # Progress callback setup
    on_prog: ProgressCallback | None = None
    finish_prog: Callable[[], None] = lambda: None
    if getattr(args, "progress", False) and not args.json:
        on_prog, finish_prog = make_progress_callback()

    # Simulation execution
    t0 = time.perf_counter()
    try:
        if getattr(args, "parallel", None) is not None:
            num_workers = (
                os.cpu_count() or 1
                if args.parallel == 0
                else max(1, args.parallel)
            )
            results = calculate_equities_parallel(
                hands=parsed_hands,
                board=board_cards,
                dead=dead_cards,
                num_sims=sims,
                num_workers=num_workers,
                seed=args.seed,
            )
        else:
            results = calculate_equities(
                hands=parsed_hands,
                board=board_cards,
                dead=dead_cards,
                num_sims=sims,
                seed=args.seed,
                convergence_threshold=getattr(args, "converge", None),
                on_progress=on_prog,
            )
    except Exception as exc:
        finish_prog()
        sys.stderr.write(f"Simulation error: {exc}\n")
        return 2
    finally:
        finish_prog()

    elapsed = time.perf_counter() - t0
    actual_sims = results[0].num_sims if results else sims
    sims_per_sec = actual_sims / elapsed if elapsed > 0 else 0.0

    # Determine display labels
    if len(parsed_hands) == 2 and (
        getattr(args, "hero", None) or not getattr(args, "players", None)
    ):
        labels = ["Hero", "Villain"]
    else:
        labels = [f"Player {i + 1}" for i in range(len(parsed_hands))]

    # JSON output
    if args.json:
        data = {
            "board": [str(c) for c in board_cards] if board_cards else None,
            "board_str": cards_to_str(board_cards) if board_cards else None,
            "dead": [str(c) for c in dead_cards] if dead_cards else None,
            "dead_str": cards_to_str(dead_cards) if dead_cards else None,
            "sims_requested": sims,
            "sims_run": actual_sims,
            "converged": any(r.converged for r in results),
            "elapsed_seconds": round(elapsed, 4),
            "sims_per_second": round(sims_per_sec, 1),
            "players": [],
        }
        for idx, (lbl, h_raw, h_parsed, r) in enumerate(
            zip(labels, raw_hands, parsed_hands, results)
        ):
            ci_half = (
                r.convergence_ci
                if r.convergence_ci > 0
                else (
                    1.96
                    * math.sqrt(
                        r.equity * (1.0 - r.equity) / max(1, r.num_sims)
                    )
                )
            )
            hand_desc = get_hand_type_description(h_parsed, board_cards)
            h_str = str(h_raw) if h_raw is not None else "random"
            data["players"].append({
                "name": lbl,
                "hand": h_str,
                "hand_type": hand_desc,
                "equity": r.equity,
                "win_pct": r.win_pct,
                "tie_pct": r.tie_pct,
                "loss_pct": r.loss_pct,
                "wins": r.wins,
                "ties": r.ties,
                "losses": r.losses,
                "ci_95_half_width": round(ci_half, 4),
                "ci_95_low": max(0.0, round(r.equity - ci_half, 4)),
                "ci_95_high": min(1.0, round(r.equity + ci_half, 4)),
            })
        print(json.dumps(data, indent=2))
        return 0

    # Human-readable formatted terminal output
    if elapsed < 0.001:
        time_str = f"{elapsed * 1_000_000:.0f} µs"
    elif elapsed < 1.0:
        time_str = f"{elapsed * 1000:.1f} ms"
    else:
        time_str = f"{elapsed:.2f} s"

    converged_note = " [converged]" if any(r.converged for r in results) else ""
    b_display = (
        format_cards_colored(board_cards, enable=use_color)
        if board_cards
        else "None (Preflop)"
    )

    print(colorize("Poker Monte Carlo Equity Simulation", Colors.BOLD, use_color))
    print("=" * 79)
    print(f"Board : {b_display}")
    if dead_cards:
        print(f"Dead  : {format_cards_colored(dead_cards, enable=use_color)}")
    print(
        f"Sims  : {actual_sims:,}{converged_note}  |  "
        f"Time: {time_str}  |  Speed: {sims_per_sec:,.0f} sims/sec"
    )
    print("-" * 79)

    col_w = {
        "player": 7,
        "hand": 6,
        "type": 15,
        "equity": 8,
        "win": 7,
        "tie": 7,
        "bar": 12,
    }
    hdr_line = (
        f"│ {pad_visible('Player', col_w['player'])} "
        f"│ {pad_visible('Hand', col_w['hand'])} "
        f"│ {pad_visible('Type', col_w['type'])} "
        f"│ {pad_visible('Equity', col_w['equity'], 'right')} "
        f"│ {pad_visible('Win %', col_w['win'], 'right')} "
        f"│ {pad_visible('Tie %', col_w['tie'], 'right')} "
        f"│ {pad_visible('Visual Bar', col_w['bar'], 'center')} │"
    )
    sep_line = (
        f"├─{'─' * col_w['player']}─"
        f"┼─{'─' * col_w['hand']}─"
        f"┼─{'─' * col_w['type']}─"
        f"┼─{'─' * col_w['equity']}─"
        f"┼─{'─' * col_w['win']}─"
        f"┼─{'─' * col_w['tie']}─"
        f"┼─{'─' * col_w['bar']}─┤"
    )
    top_line = (
        f"┌─{'─' * col_w['player']}─"
        f"┬─{'─' * col_w['hand']}─"
        f"┬─{'─' * col_w['type']}─"
        f"┬─{'─' * col_w['equity']}─"
        f"┬─{'─' * col_w['win']}─"
        f"┬─{'─' * col_w['tie']}─"
        f"┬─{'─' * col_w['bar']}─┐"
    )
    bot_line = (
        f"└─{'─' * col_w['player']}─"
        f"┴─{'─' * col_w['hand']}─"
        f"┴─{'─' * col_w['type']}─"
        f"┴─{'─' * col_w['equity']}─"
        f"┴─{'─' * col_w['win']}─"
        f"┴─{'─' * col_w['tie']}─"
        f"┴─{'─' * col_w['bar']}─┘"
    )

    print(top_line)
    print(hdr_line)
    print(sep_line)

    for idx, (lbl, h_raw, h_parsed, r) in enumerate(
        zip(labels, raw_hands, parsed_hands, results)
    ):
        if h_parsed is None:
            hand_str = "Random"
        elif isinstance(h_parsed, tuple):
            hand_str = format_cards_colored(h_parsed, enable=use_color)
        else:
            hand_str = str(h_parsed)

        hand_desc = get_hand_type_description(h_parsed, board_cards)
        if len(hand_desc) > col_w["type"]:
            hand_desc = hand_desc[: col_w["type"] - 1] + "…"

        eq_pct_str = f"{r.equity * 100:.2f}%"
        eq_col = (
            Colors.GREEN
            if r.equity >= 0.55
            else (Colors.YELLOW if r.equity >= 0.40 else Colors.RED)
        )
        eq_colored = colorize(eq_pct_str, eq_col, use_color)

        bar_str = make_equity_bar(
            r.equity, width=col_w["bar"] - 2, enable=use_color
        )

        row_str = (
            f"│ {pad_visible(lbl, col_w['player'])} "
            f"│ {pad_visible(hand_str, col_w['hand'])} "
            f"│ {pad_visible(hand_desc, col_w['type'])} "
            f"│ {pad_visible(eq_colored, col_w['equity'], 'right')} "
            f"│ {pad_visible(f'{r.win_pct * 100:.1f}%', col_w['win'], 'right')} "
            f"│ {pad_visible(f'{r.tie_pct * 100:.1f}%', col_w['tie'], 'right')} "
            f"│ {pad_visible(bar_str, col_w['bar'], 'center')} │"
        )
        print(row_str)

    print(bot_line)

    for idx, (lbl, h_raw, h_parsed, r) in enumerate(
        zip(labels, raw_hands, parsed_hands, results)
    ):
        ci_half = (
            r.convergence_ci
            if r.convergence_ci > 0
            else (
                1.96
                * math.sqrt(
                    r.equity * (1.0 - r.equity) / max(1, r.num_sims)
                )
            )
        )
        print(
            f"  {lbl:7s}: Equity {r.equity * 100:5.2f}% ± {ci_half * 100:.2f}% (95% CI) "
            f"[Win: {r.win_pct * 100:.1f}%, Tie: {r.tie_pct * 100:.1f}%, Loss: {r.loss_pct * 100:.1f}%]"
        )

    return 0


def handle_range_command(args: argparse.Namespace, use_color: bool = True) -> int:
    """Execute the 'range' subcommand."""
    # Resolve hero and villain input
    pos_ranges = getattr(args, "ranges", None) or []
    if len(pos_ranges) >= 2:
        hero_raw, villain_raw = pos_ranges[0], pos_ranges[1]
    elif getattr(args, "hero", None) and getattr(args, "villain", None):
        hero_raw, villain_raw = args.hero, args.villain
    elif len(pos_ranges) == 1 and getattr(args, "villain", None):
        hero_raw, villain_raw = pos_ranges[0], args.villain
    elif getattr(args, "hero", None) and len(pos_ranges) == 1:
        hero_raw, villain_raw = args.hero, pos_ranges[0]
    else:
        sys.stderr.write(
            "Error: Must specify both hero and villain ranges (via --hero and --villain or positional arguments).\n"
        )
        return 2

    # Parse board cards
    try:
        board_cards = parse_cards_flexible(args.board) if args.board else []
    except Exception as exc:
        sys.stderr.write(f"Error parsing board cards '{args.board}': {exc}\n")
        return 2

    # Parse dead cards
    try:
        dead_cards = (
            parse_cards_flexible(getattr(args, "dead", None))
            if getattr(args, "dead", None)
            else []
        )
    except Exception as exc:
        sys.stderr.write(f"Error parsing dead cards '{args.dead}': {exc}\n")
        return 2

    # Parse hero and villain
    try:
        hero_item = parse_range_or_preset(hero_raw)
    except Exception as exc:
        sys.stderr.write(f"Error parsing hero range '{hero_raw}': {exc}\n")
        return 2

    try:
        villain_item = parse_range_or_preset(villain_raw)
    except Exception as exc:
        sys.stderr.write(f"Error parsing villain range '{villain_raw}': {exc}\n")
        return 2

    # Metadata extraction
    def get_range_meta(
        item: Range | tuple[Card, Card],
        opp_item: Range | tuple[Card, Card],
        raw_spec: str,
    ) -> dict[str, Any]:
        all_dead = list(board_cards) + list(dead_cards)
        if isinstance(opp_item, tuple):
            all_dead.extend(opp_item)

        if isinstance(item, tuple):
            return {
                "is_range": False,
                "spec": raw_spec,
                "display": format_cards_colored(item, enable=use_color),
                "type_desc": get_hand_type_description(item, board_cards),
                "total_combos": 1,
                "weighted_combos": 1.0,
                "live_combos": 1,
                "live_weighted": 1.0,
                "is_weighted": False,
                "pct_of_hands": 1 / 1326,
            }
        else:
            live_r = item.remove_dead_cards(all_dead)
            return {
                "is_range": True,
                "spec": raw_spec,
                "display": raw_spec,
                "type_desc": f"{len(item)} combos"
                + (
                    f" ({item.total_combos:.1f} weighted)"
                    if item.is_weighted
                    else ""
                ),
                "total_combos": len(item),
                "weighted_combos": item.total_combos,
                "live_combos": len(live_r),
                "live_weighted": live_r.total_combos,
                "is_weighted": item.is_weighted,
                "pct_of_hands": len(live_r) / 1326,
            }

    hero_meta = get_range_meta(hero_item, villain_item, hero_raw)
    villain_meta = get_range_meta(villain_item, hero_item, villain_raw)

    sims = max(1, args.sims)

    # Progress callback setup
    on_prog: ProgressCallback | None = None
    finish_prog: Callable[[], None] = lambda: None
    if getattr(args, "progress", False) and not args.json:
        on_prog, finish_prog = make_progress_callback()

    # Simulation execution
    t0 = time.perf_counter()
    try:
        if getattr(args, "parallel", None) is not None:
            num_workers = (
                os.cpu_count() or 1
                if args.parallel == 0
                else max(1, args.parallel)
            )
            results = calculate_equities_parallel(
                hands=[hero_item, villain_item],
                board=board_cards,
                dead=dead_cards,
                num_sims=sims,
                num_workers=num_workers,
                seed=args.seed,
            )
        else:
            results = calculate_equities(
                hands=[hero_item, villain_item],
                board=board_cards,
                dead=dead_cards,
                num_sims=sims,
                seed=args.seed,
                convergence_threshold=getattr(args, "converge", None),
                on_progress=on_prog,
            )
    except Exception as exc:
        finish_prog()
        sys.stderr.write(f"Simulation error: {exc}\n")
        return 2
    finally:
        finish_prog()

    elapsed = time.perf_counter() - t0
    actual_sims = results[0].num_sims if results else sims
    sims_per_sec = actual_sims / elapsed if elapsed > 0 else 0.0

    hero_res = results[0]
    villain_res = results[1]

    hero_ci = (
        hero_res.convergence_ci
        if hero_res.convergence_ci > 0
        else (
            1.96
            * math.sqrt(
                hero_res.equity
                * (1.0 - hero_res.equity)
                / max(1, hero_res.num_sims)
            )
        )
    )
    villain_ci = (
        villain_res.convergence_ci
        if villain_res.convergence_ci > 0
        else (
            1.96
            * math.sqrt(
                villain_res.equity
                * (1.0 - villain_res.equity)
                / max(1, villain_res.num_sims)
            )
        )
    )

    # Optional combo breakdown calculation
    breakdown_list = []
    breakdown_target_name = None
    if getattr(args, "breakdown", None) is not None:
        top_n = max(1, args.breakdown)
        if isinstance(hero_item, tuple) and isinstance(villain_item, Range):
            breakdown_target_name = "Villain"
            dead_for_breakdown = (
                list(board_cards) + list(dead_cards) + list(hero_item)
            )
            live_target = villain_item.remove_dead_cards(dead_for_breakdown)
            if not live_target.is_empty:
                bd = range_vs_hand_breakdown(
                    live_target,
                    hero_item,
                    board=board_cards,
                    num_sims_per_combo=min(
                        1000, max(200, sims // max(1, len(live_target)))
                    ),
                    seed=args.seed,
                )
                breakdown_list = [
                    {
                        "combo": combo_to_str(c),
                        "cell": combo_to_cell(c),
                        "equity": r.equity,
                        "win_pct": r.win_pct,
                        "tie_pct": r.tie_pct,
                        "loss_pct": r.loss_pct,
                        "weight": live_target.get_weight(c),
                    }
                    for c, r in bd.sorted_combos(descending=True)[:top_n]
                ]
        elif isinstance(hero_item, Range) and isinstance(villain_item, tuple):
            breakdown_target_name = "Hero"
            dead_for_breakdown = (
                list(board_cards) + list(dead_cards) + list(villain_item)
            )
            live_target = hero_item.remove_dead_cards(dead_for_breakdown)
            if not live_target.is_empty:
                bd = range_vs_hand_breakdown(
                    live_target,
                    villain_item,
                    board=board_cards,
                    num_sims_per_combo=min(
                        1000, max(200, sims // max(1, len(live_target)))
                    ),
                    seed=args.seed,
                )
                breakdown_list = [
                    {
                        "combo": combo_to_str(c),
                        "cell": combo_to_cell(c),
                        "equity": r.equity,
                        "win_pct": r.win_pct,
                        "tie_pct": r.tie_pct,
                        "loss_pct": r.loss_pct,
                        "weight": live_target.get_weight(c),
                    }
                    for c, r in bd.sorted_combos(descending=True)[:top_n]
                ]

    # JSON output
    if args.json:
        data = {
            "board": [str(c) for c in board_cards] if board_cards else None,
            "board_str": cards_to_str(board_cards) if board_cards else None,
            "dead": [str(c) for c in dead_cards] if dead_cards else None,
            "dead_str": cards_to_str(dead_cards) if dead_cards else None,
            "sims_requested": sims,
            "sims_run": actual_sims,
            "converged": any(r.converged for r in results),
            "elapsed_seconds": round(elapsed, 4),
            "sims_per_second": round(sims_per_sec, 1),
            "hero": {
                "spec": hero_raw,
                "is_range": hero_meta["is_range"],
                "total_combos": hero_meta["total_combos"],
                "weighted_combos": round(hero_meta["weighted_combos"], 2),
                "live_combos": hero_meta["live_combos"],
                "live_weighted_combos": round(hero_meta["live_weighted"], 2),
                "pct_of_hands": round(hero_meta["pct_of_hands"] * 100, 2),
                "equity": hero_res.equity,
                "win_pct": hero_res.win_pct,
                "tie_pct": hero_res.tie_pct,
                "loss_pct": hero_res.loss_pct,
                "wins": hero_res.wins,
                "ties": hero_res.ties,
                "losses": hero_res.losses,
                "ci_95_half_width": round(hero_ci, 4),
                "ci_95_low": max(0.0, round(hero_res.equity - hero_ci, 4)),
                "ci_95_high": min(1.0, round(hero_res.equity + hero_ci, 4)),
            },
            "villain": {
                "spec": villain_raw,
                "is_range": villain_meta["is_range"],
                "total_combos": villain_meta["total_combos"],
                "weighted_combos": round(villain_meta["weighted_combos"], 2),
                "live_combos": villain_meta["live_combos"],
                "live_weighted_combos": round(
                    villain_meta["live_weighted"], 2
                ),
                "pct_of_hands": round(villain_meta["pct_of_hands"] * 100, 2),
                "equity": villain_res.equity,
                "win_pct": villain_res.win_pct,
                "tie_pct": villain_res.tie_pct,
                "loss_pct": villain_res.loss_pct,
                "wins": villain_res.wins,
                "ties": villain_res.ties,
                "losses": villain_res.losses,
                "ci_95_half_width": round(villain_ci, 4),
                "ci_95_low": max(0.0, round(villain_res.equity - villain_ci, 4)),
                "ci_95_high": min(1.0, round(villain_res.equity + villain_ci, 4)),
            },
        }
        if breakdown_list:
            data["breakdown"] = breakdown_list
        print(json.dumps(data, indent=2))
        return 0

    # Human-readable formatted terminal output
    if elapsed < 0.001:
        time_str = f"{elapsed * 1_000_000:.0f} µs"
    elif elapsed < 1.0:
        time_str = f"{elapsed * 1000:.1f} ms"
    else:
        time_str = f"{elapsed:.2f} s"

    converged_note = " [converged]" if any(r.converged for r in results) else ""
    b_display = (
        format_cards_colored(board_cards, enable=use_color)
        if board_cards
        else "None (Preflop)"
    )

    print(colorize("Range vs Range Equity Simulation", Colors.BOLD, use_color))
    print("=" * 79)
    print(f"Board : {b_display}")
    if dead_cards:
        print(f"Dead  : {format_cards_colored(dead_cards, enable=use_color)}")
    print(
        f"Sims  : {actual_sims:,}{converged_note}  |  "
        f"Time: {time_str}  |  Speed: {sims_per_sec:,.0f} sims/sec"
    )
    print("-" * 79)

    h_desc = (
        f"{hero_meta['total_combos']} combos, {hero_meta['live_combos']} live "
        f"[{hero_meta['pct_of_hands'] * 100:.1f}%]"
    ) if hero_meta["is_range"] else hero_meta["type_desc"]
    v_desc = (
        f"{villain_meta['total_combos']} combos, {villain_meta['live_combos']} live "
        f"[{villain_meta['pct_of_hands'] * 100:.1f}%]"
    ) if villain_meta["is_range"] else villain_meta["type_desc"]

    print(f"Hero   : {hero_meta['display']} ({h_desc})")
    print(f"Villain: {villain_meta['display']} ({v_desc})")
    print("-" * 79)

    col_w = {
        "player": 7,
        "range": 22,
        "equity": 8,
        "win": 7,
        "tie": 7,
        "bar": 12,
    }
    top_line = (
        f"┌─{'─' * col_w['player']}─"
        f"┬─{'─' * col_w['range']}─"
        f"┬─{'─' * col_w['equity']}─"
        f"┬─{'─' * col_w['win']}─"
        f"┬─{'─' * col_w['tie']}─"
        f"┬─{'─' * col_w['bar']}─┐"
    )
    hdr_line = (
        f"│ {pad_visible('Player', col_w['player'])} "
        f"│ {pad_visible('Range / Hand', col_w['range'])} "
        f"│ {pad_visible('Equity', col_w['equity'], 'right')} "
        f"│ {pad_visible('Win %', col_w['win'], 'right')} "
        f"│ {pad_visible('Tie %', col_w['tie'], 'right')} "
        f"│ {pad_visible('Visual Bar', col_w['bar'], 'center')} │"
    )
    sep_line = (
        f"├─{'─' * col_w['player']}─"
        f"┼─{'─' * col_w['range']}─"
        f"┼─{'─' * col_w['equity']}─"
        f"┼─{'─' * col_w['win']}─"
        f"┼─{'─' * col_w['tie']}─"
        f"┼─{'─' * col_w['bar']}─┤"
    )
    bot_line = (
        f"└─{'─' * col_w['player']}─"
        f"┴─{'─' * col_w['range']}─"
        f"┴─{'─' * col_w['equity']}─"
        f"┴─{'─' * col_w['win']}─"
        f"┴─{'─' * col_w['tie']}─"
        f"┴─{'─' * col_w['bar']}─┘"
    )

    print(top_line)
    print(hdr_line)
    print(sep_line)

    for p_name, meta, res in [
        ("Hero", hero_meta, hero_res),
        ("Villain", villain_meta, villain_res),
    ]:
        eq_pct_str = f"{res.equity * 100:.2f}%"
        eq_col = (
            Colors.GREEN
            if res.equity >= 0.55
            else (Colors.YELLOW if res.equity >= 0.40 else Colors.RED)
        )
        eq_colored = colorize(eq_pct_str, eq_col, use_color)
        bar_str = make_equity_bar(
            res.equity, width=col_w["bar"] - 2, enable=use_color
        )

        label_display = meta["display"]
        if len(label_display) > col_w["range"]:
            label_display = label_display[: col_w["range"] - 1] + "…"

        row_str = (
            f"│ {pad_visible(p_name, col_w['player'])} "
            f"│ {pad_visible(label_display, col_w['range'])} "
            f"│ {pad_visible(eq_colored, col_w['equity'], 'right')} "
            f"│ {pad_visible(f'{res.win_pct * 100:.1f}%', col_w['win'], 'right')} "
            f"│ {pad_visible(f'{res.tie_pct * 100:.1f}%', col_w['tie'], 'right')} "
            f"│ {pad_visible(bar_str, col_w['bar'], 'center')} │"
        )
        print(row_str)

    print(bot_line)

    print(
        f"  Hero   : Equity {hero_res.equity * 100:5.2f}% ± {hero_ci * 100:.2f}% (95% CI) "
        f"[Win: {hero_res.win_pct * 100:.1f}%, Tie: {hero_res.tie_pct * 100:.1f}%, Loss: {hero_res.loss_pct * 100:.1f}%]"
    )
    print(
        f"  Villain: Equity {villain_res.equity * 100:5.2f}% ± {villain_ci * 100:.2f}% (95% CI) "
        f"[Win: {villain_res.win_pct * 100:.1f}%, Tie: {villain_res.tie_pct * 100:.1f}%, Loss: {villain_res.loss_pct * 100:.1f}%]"
    )

    if breakdown_list and breakdown_target_name:
        print(
            "\n"
            + colorize(
                f"Top {len(breakdown_list)} Combos in {breakdown_target_name} Range",
                Colors.BOLD,
                use_color,
            )
        )
        print("-" * 65)
        print(
            f"{'Combo':<8} {'Cell':<6} {'Equity':<10} {'Win %':<10} {'Tie %':<10} {'Weight':<8}"
        )
        print("-" * 65)
        for item in breakdown_list:
            c_str = item["combo"]
            c_colored = format_cards_colored(c_str, enable=use_color)
            eq_str = f"{item['equity'] * 100:.1f}%"
            print(
                f"{pad_visible(c_colored, 8)} {item['cell']:<6} {eq_str:<10} "
                f"{item['win_pct'] * 100:.1f}%     {item['tie_pct'] * 100:.1f}%     {item['weight']:.2f}"
            )
        print("-" * 65)

    return 0


def handle_odds_command(args: argparse.Namespace, use_color: bool = True) -> int:
    """Execute the 'odds' subcommand for pot odds, EV, and implied odds analysis.

    Supports positional arguments (e.g. ``odds 100 50 35%``), flags (``--pot``,
    ``--call``, ``--equity``, ``--future-bets``, ``--future-losses``), and
    interactive prompting mode.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    use_color : bool, default=True
        Whether ANSI color formatting is enabled.

    Returns
    -------
    int
        Exit status code (0 for success, non-zero for error).

    Examples
    --------
    >>> import argparse
    >>> from poker_equity.cli import handle_odds_command
    >>> args = argparse.Namespace(
    ...     numbers=["100", "50"], pot=None, call=None, equity="0.40",
    ...     future_bets=0.0, future_losses=0.0, interactive=False, json=True
    ... )
    >>> handle_odds_command(args, use_color=False)
    0
    """
    pot = args.pot
    call = args.call
    raw_equity = args.equity
    numbers = getattr(args, "numbers", []) or []

    # Interactive prompt if requested or in interactive tty with no arguments
    if getattr(args, "interactive", False) or (
        pot is None and call is None and not numbers and sys.stdin.isatty()
    ):
        try:
            if pot is None:
                val = input("Enter pot size: ").strip()
                pot = float(val)
            if call is None:
                val = input("Enter bet to call: ").strip()
                call = float(val)
            if raw_equity is None:
                val = input("Enter hero equity (optional, e.g. '35%' or '0.35', or Enter to skip): ").strip()
                if val:
                    raw_equity = val
        except (EOFError, KeyboardInterrupt):
            sys.stderr.write("\nOdds calculation cancelled.\n")
            return 1
        except Exception as exc:
            sys.stderr.write(f"Error parsing interactive inputs: {exc}\n")
            return 2

    # Parse positional numbers if pot / call / equity not set via flags
    if numbers:
        if pot is None and len(numbers) >= 1:
            try:
                pot = float(numbers[0])
            except ValueError:
                sys.stderr.write(f"Error: Invalid pot size '{numbers[0]}'. Must be a number.\n")
                return 2
        if call is None and len(numbers) >= 2:
            try:
                call = float(numbers[1])
            except ValueError:
                sys.stderr.write(f"Error: Invalid bet to call '{numbers[1]}'. Must be a number.\n")
                return 2
        if raw_equity is None and len(numbers) >= 3:
            raw_equity = numbers[2]

    if pot is None or call is None:
        sys.stderr.write(
            "Error: Must specify both pot size and bet to call (e.g. 'poker-equity odds 100 50' or '--pot 100 --call 50').\n"
        )
        return 2

    if pot < 0 or call < 0:
        sys.stderr.write("Error: Pot size and bet to call must be non-negative.\n")
        return 2

    # Parse equity if provided
    equity: float | None = None
    if raw_equity is not None:
        try:
            equity = parse_equity_value(raw_equity)
            if not (0.0 <= equity <= 1.0):
                sys.stderr.write(
                    f"Error: Equity must be between 0.0 and 1.0 (or 0% and 100%), got {raw_equity}.\n"
                )
                return 2
        except Exception as exc:
            sys.stderr.write(f"Error parsing equity '{raw_equity}': {exc}\n")
            return 2

    future_bets = getattr(args, "future_bets", 0.0) or 0.0
    future_losses = getattr(args, "future_losses", 0.0) or 0.0

    req_odds = poker_equity.pot_odds(pot, call)
    ratio = poker_equity.pot_odds_ratio(pot, call)
    ratio_str = f"{ratio[0]:.2f}:1" if ratio[0] != float("inf") else "inf:1"

    ev_val: float | None = None
    direct_ev: float | None = None
    ev_with_future: float | None = None
    is_profitable: bool | None = None
    req_future: float | None = None

    if equity is not None:
        direct_ev = poker_equity.expected_value(equity, pot, call)
        ev_val = poker_equity.expected_value(equity, pot + future_bets, call)
        ev_with_future = ev_val
        is_profitable = poker_equity.is_profitable_call(equity, pot, call)
        req_future = poker_equity.required_implied_odds(pot, call, equity)

    imp_odds: float | None = None
    imp_ratio: tuple[float, float] | None = None
    imp_ratio_str: str | None = None
    if future_bets > 0:
        imp_odds = poker_equity.implied_odds(pot, call, future_bets)
        imp_ratio = poker_equity.implied_odds_ratio(pot, call, future_bets)
        imp_ratio_str = f"{imp_ratio[0]:.2f}:1" if imp_ratio[0] != float("inf") else "inf:1"

    rev_imp_odds: float | None = None
    if future_losses > 0:
        rev_imp_odds = poker_equity.reverse_implied_odds(pot, call, future_losses)

    if args.json:
        data = {
            "pot_size": pot,
            "bet_to_call": call,
            "pot_odds_pct": req_odds,
            "pot_odds_ratio": list(ratio),
            "pot_odds_ratio_str": ratio_str,
            "equity": equity,
            "ev": ev_val,
            "direct_ev": direct_ev,
            "is_profitable": is_profitable,
            "recommendation": ("CALL" if ev_val >= 0 else "FOLD") if ev_val is not None else None,
            "future_bets": future_bets,
            "future_losses": future_losses,
            "implied_odds_pct": imp_odds,
            "implied_odds_ratio": list(imp_ratio) if imp_ratio else None,
            "implied_odds_ratio_str": imp_ratio_str,
            "required_future_bets": req_future,
            "ev_with_future_bets": ev_with_future if future_bets > 0 else None,
            "reverse_implied_odds_pct": rev_imp_odds,
        }
        print(json.dumps(data, indent=2))
        return 0

    print(colorize("Pot Odds & EV Analysis", Colors.BOLD, use_color))
    print("=" * 48)
    print(f"Pot Size        : {pot:.2f} chips")
    print(f"Cost to Call    : {call:.2f} chips")
    print(f"Total Pot       : {pot + call:.2f} chips")
    print(f"Pot Odds        : {colorize(f'{req_odds * 100:.2f}%', Colors.CYAN, use_color)} ({ratio})")

    if future_bets > 0 and imp_odds is not None:
        print(f"Future Bets     : +{future_bets:.2f} chips anticipated")
        print(f"Implied Odds    : {colorize(f'{imp_odds * 100:.2f}%', Colors.CYAN, use_color)} ({imp_ratio_str})")

    if future_losses > 0 and rev_imp_odds is not None:
        print(f"Future Losses   : -{future_losses:.2f} chips potential")
        print(f"Rev Implied Odds: {colorize(f'{rev_imp_odds * 100:.2f}%', Colors.YELLOW, use_color)}")

    if equity is not None:
        print("-" * 48)
        print(f"Hero Equity     : {colorize(f'{equity * 100:.2f}%', Colors.BOLD, use_color)}")
        ev_col = Colors.GREEN if ev_val >= 0 else Colors.RED
        print(f"Expected Value  : {colorize(f'{ev_val:+.2f} chips', ev_col, use_color)}")

        if future_bets > 0 and direct_ev is not None:
            dir_col = Colors.GREEN if direct_ev >= 0 else Colors.RED
            print(f"Direct EV       : {colorize(f'{direct_ev:+.2f} chips', dir_col, use_color)}")

        if req_future is not None:
            if req_future > 0:
                print(f"Req Implied Bets: +{req_future:.2f} chips needed on later streets")
            else:
                print("Req Implied Bets: 0.00 chips (Direct pot odds sufficient)")

        rec = "CALL" if ev_val >= 0 else "FOLD"
        print(f"Recommendation  : {colorize(rec, ev_col, use_color)}")

    return 0


def handle_outs_command(args: argparse.Namespace, use_color: bool = True) -> int:
    """Execute the 'outs' subcommand for drawing outs and equity calculation.

    Identifies winning and tie outs on the flop or turn, evaluates made hand
    categories, computes Rule of 2/4 and exact mathematical equities, and
    displays cards grouped by resulting made hand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    use_color : bool, default=True
        Whether ANSI color formatting is enabled.

    Returns
    -------
    int
        Exit status code (0 for success, non-zero for error).

    Examples
    --------
    >>> import argparse
    >>> from poker_equity.cli import handle_outs_command
    >>> args = argparse.Namespace(
    ...     cards=["AhKh", "QcQd", "2h 7h 8c"], hero=None, villain=None,
    ...     board=None, dead=None, method="exact", interactive=False, json=True
    ... )
    >>> handle_outs_command(args, use_color=False)
    0
    """
    hero_raw = args.hero
    villain_raw = args.villain
    board_raw = args.board
    cards_pos = getattr(args, "cards", []) or []

    # Interactive prompt if requested or in interactive tty with no arguments
    if getattr(args, "interactive", False) or (
        hero_raw is None and villain_raw is None and board_raw is None and not cards_pos and sys.stdin.isatty()
    ):
        try:
            if hero_raw is None:
                hero_raw = input("Enter Hero cards (e.g. 'Ah Kh'): ").strip()
            if villain_raw is None:
                villain_raw = input("Enter Villain cards (e.g. 'Qc Qd'): ").strip()
            if board_raw is None:
                board_raw = input("Enter Board cards (3 or 4 cards, e.g. '2h 7h 8c'): ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.stderr.write("\nOuts calculation cancelled.\n")
            return 1
        except Exception as exc:
            sys.stderr.write(f"Error parsing interactive inputs: {exc}\n")
            return 2

    # Positional resolution
    if cards_pos:
        if len(cards_pos) == 3:
            hero_raw = hero_raw or cards_pos[0]
            villain_raw = villain_raw or cards_pos[1]
            board_raw = board_raw or cards_pos[2]
        elif len(cards_pos) >= 7 and (hero_raw is None or villain_raw is None or board_raw is None):
            try:
                all_pos_cards = parse_cards_flexible(" ".join(cards_pos))
                if len(all_pos_cards) in (7, 8):
                    if hero_raw is None:
                        hero_raw = cards_to_str(all_pos_cards[0:2])
                    if villain_raw is None:
                        villain_raw = cards_to_str(all_pos_cards[2:4])
                    if board_raw is None:
                        board_raw = cards_to_str(all_pos_cards[4:])
            except Exception:
                pass
        elif hero_raw is not None and villain_raw is not None and board_raw is None:
            board_raw = " ".join(cards_pos)

    if not hero_raw or not villain_raw or not board_raw:
        sys.stderr.write(
            "Error: Must specify hero hole cards, villain hole cards, and board cards (3 or 4 cards) "
            "(e.g. 'poker-equity outs AhKh QcQd \"2h 7h 8c\"' or '--hero AhKh --villain QcQd --board \"2h 7h 8c\"').\n"
        )
        return 2

    try:
        hero_cards = parse_cards_flexible(hero_raw)
        if len(hero_cards) != 2:
            sys.stderr.write(f"Error: Hero must have exactly 2 cards, got {len(hero_cards)}.\n")
            return 2
    except Exception as exc:
        sys.stderr.write(f"Error parsing hero cards '{hero_raw}': {exc}\n")
        return 2

    try:
        villain_cards = parse_cards_flexible(villain_raw)
        if len(villain_cards) != 2:
            sys.stderr.write(f"Error: Villain must have exactly 2 cards, got {len(villain_cards)}.\n")
            return 2
    except Exception as exc:
        sys.stderr.write(f"Error parsing villain cards '{villain_raw}': {exc}\n")
        return 2

    try:
        board_cards = parse_cards_flexible(board_raw)
        if len(board_cards) not in (3, 4):
            sys.stderr.write(f"Error: Board must contain 3 (flop) or 4 (turn) cards, got {len(board_cards)}.\n")
            return 2
    except Exception as exc:
        sys.stderr.write(f"Error parsing board cards '{board_raw}': {exc}\n")
        return 2

    dead_cards = []
    if getattr(args, "dead", None):
        try:
            dead_cards = parse_cards_flexible(args.dead)
        except Exception as exc:
            sys.stderr.write(f"Error parsing dead cards '{args.dead}': {exc}\n")
            return 2

    try:
        res = poker_equity.calculate_outs(
            hero=hero_cards,
            villain=villain_cards,
            board=board_cards,
            dead=dead_cards,
        )
    except Exception as exc:
        sys.stderr.write(f"Error calculating outs: {exc}\n")
        return 2

    street = "flop" if len(board_cards) == 3 else "turn"
    cards_to_come = 2 if street == "flop" else 1
    equity = poker_equity.outs_to_equity(res.total_outs, street=street, method=args.method)
    rule_2 = res.rule_of_2
    rule_4 = res.rule_of_4
    rule_4_adj = poker_equity.rule_of_4(res.total_outs, adjusted=True)
    exact_1 = res.exact_turn_equity
    exact_2 = res.exact_river_equity

    if args.json:
        data = {
            "hero": cards_to_str(hero_cards),
            "villain": cards_to_str(villain_cards),
            "board": cards_to_str(board_cards),
            "dead": cards_to_str(dead_cards) if dead_cards else None,
            "street": street,
            "cards_to_come": cards_to_come,
            "hero_hand": res.hero_current_result.description,
            "villain_hand": res.villain_current_result.description,
            "is_hero_ahead": res.is_hero_ahead,
            "total_outs": res.total_outs,
            "win_outs": len(res.win_outs),
            "tie_outs": len(res.tie_outs),
            "unseen_cards": res.unseen_cards,
            "drawing_equity": equity,
            "method": args.method,
            "rule_of_2_pct": round(rule_2 * 100, 2),
            "rule_of_4_pct": round(rule_4 * 100, 2),
            "rule_of_4_adjusted_pct": round(rule_4_adj * 100, 2),
            "exact_1_card_pct": round(exact_1 * 100, 2),
            "exact_2_card_pct": round(exact_2 * 100, 2),
            "win_cards": [str(c) for c in res.win_outs],
            "tie_cards": [str(c) for c in res.tie_outs],
            "outs_by_rank": {
                rank.label: [str(c) for c in cards]
                for rank, cards in res.outs_by_rank.items()
            },
        }
        print(json.dumps(data, indent=2))
        return 0

    print(colorize("Drawing Outs Calculation", Colors.BOLD, use_color))
    print("=" * 55)
    b_colored = format_cards_colored(board_cards, enable=use_color)
    h_colored = format_cards_colored(hero_cards, enable=use_color)
    v_colored = format_cards_colored(villain_cards, enable=use_color)

    print(f"Hero   : {args.hero or cards_to_str(hero_cards)} ({h_colored}) - {res.hero_current_result.description}")
    print(f"Villain: {args.villain or cards_to_str(villain_cards)} ({v_colored}) - {res.villain_current_result.description}")
    print(f"Board  : {args.board or cards_to_str(board_cards)} ({b_colored}) [{street.upper()}, {cards_to_come} to come]")
    if dead_cards:
        print(f"Dead   : {format_cards_colored(dead_cards, enable=use_color)}")

    if res.is_hero_ahead:
        status_str = colorize("Hero is currently AHEAD", Colors.GREEN, use_color)
    elif res.hero_current_result == res.villain_current_result:
        status_str = colorize("Hands are currently TIED", Colors.YELLOW, use_color)
    else:
        status_str = colorize("Hero is currently TRAILING", Colors.RED, use_color)
    print(f"Status : {status_str}")
    print("-" * 55)

    print(f"Total Outs : {colorize(str(res.total_outs), Colors.GREEN, use_color)} ({len(res.win_outs)} win, {len(res.tie_outs)} tie)")
    print(f"Equity ({args.method}): {colorize(f'{equity * 100:.2f}%', Colors.CYAN, use_color)}")

    if res.win_outs:
        outs_str = cards_to_str(res.win_outs[:12])
        ellipsis = "..." if len(res.win_outs) > 12 else ""
        print(f"Outs Cards : {outs_str}{ellipsis}")

    print("-" * 55)
    print(colorize("Equity Projections:", Colors.BOLD, use_color))
    if street == "flop":
        print(f"  Exact (Turn or River) : {exact_2 * 100:5.2f}%")
        print(f"  Exact (Turn only)     : {exact_1 * 100:5.2f}%")
        print(f"  Rule of 4 (Standard)  : {rule_4 * 100:5.1f}%")
        if res.total_outs > 8:
            print(f"  Rule of 4 (Adjusted)  : {rule_4_adj * 100:5.1f}%")
    else:
        print(f"  Exact (River)         : {exact_1 * 100:5.2f}%")
        print(f"  Rule of 2 (Standard)  : {rule_2 * 100:5.1f}%")

    if res.outs_by_rank:
        print("-" * 55)
        print(colorize("Winning Outs by Made Hand:", Colors.BOLD, use_color))
        for rank in sorted(res.outs_by_rank.keys(), reverse=True):
            cards = res.outs_by_rank[rank]
            c_str = format_cards_colored(cards, enable=use_color)
            print(f"  {rank.label:<16} ({len(cards):2d} out{'s' if len(cards) > 1 else ''}): {c_str}")

    if res.tie_outs:
        c_str = format_cards_colored(res.tie_outs, enable=use_color)
        print(f"  Tie/Split Outs   ({len(res.tie_outs):2d} out{'s' if len(res.tie_outs) > 1 else ''}): {c_str}")

    return 0


def handle_decision_command(args: argparse.Namespace, use_color: bool = True) -> int:
    """Execute the 'decision' subcommand for strategic action recommendations.

    Recommends the mathematically optimal action (Fold, Check, Call, Bet, Raise,
    or All-In) based on expected value (EV), pot odds, fold equity, stack size,
    and implied odds.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    use_color : bool, default=True
        Whether ANSI color formatting is enabled.

    Returns
    -------
    int
        Exit status code (0 for success, non-zero for error).

    Examples
    --------
    >>> import argparse
    >>> from poker_equity.cli import handle_decision_command
    >>> args = argparse.Namespace(
    ...     numbers=["0.35", "150", "50"], equity=None, pot=None, call=0.0,
    ...     stack=None, fold_equity=None, raise_size=None, future_bets=0.0,
    ...     hero=None, villain=None, board=None, sims=5000, seed=None,
    ...     interactive=False, json=True
    ... )
    >>> handle_decision_command(args, use_color=False)
    0
    """
    raw_equity = args.equity
    pot = args.pot
    call = args.call if args.call is not None else 0.0
    stack = args.stack
    raw_fe = args.fold_equity
    raise_size = args.raise_size
    future_bets = getattr(args, "future_bets", 0.0) or 0.0
    hero_raw = args.hero
    villain_raw = args.villain
    board_raw = args.board
    numbers = getattr(args, "numbers", []) or []

    # Positional argument resolution
    if numbers:
        # Check if first positional argument is a hand rather than a number
        first_token = numbers[0].strip()
        is_first_num = False
        try:
            parse_equity_value(first_token)
            is_first_num = True
        except ValueError:
            is_first_num = False

        if not is_first_num:
            # Positional cards mode: <hero> <villain> [<board>]
            if hero_raw is None and len(numbers) >= 1:
                hero_raw = numbers[0]
            if villain_raw is None and len(numbers) >= 2:
                villain_raw = numbers[1]
            if board_raw is None and len(numbers) >= 3:
                board_raw = numbers[2]
        else:
            # Positional math mode: <equity> <pot> [<call>] [<stack>]
            if raw_equity is None and len(numbers) >= 1:
                raw_equity = numbers[0]
            if pot is None and len(numbers) >= 2:
                try:
                    pot = float(numbers[1])
                except ValueError:
                    sys.stderr.write(f"Error: Invalid pot size '{numbers[1]}'. Must be a number.\n")
                    return 2
            if call == 0.0 and len(numbers) >= 3:
                try:
                    call = float(numbers[2])
                except ValueError:
                    sys.stderr.write(f"Error: Invalid bet to call '{numbers[2]}'. Must be a number.\n")
                    return 2
            if stack is None and len(numbers) >= 4:
                try:
                    stack = float(numbers[3])
                except ValueError:
                    sys.stderr.write(f"Error: Invalid stack size '{numbers[3]}'. Must be a number.\n")
                    return 2

    # Interactive prompt if requested or in interactive tty with no arguments
    if getattr(args, "interactive", False) or (
        raw_equity is None and pot is None and hero_raw is None and not numbers and sys.stdin.isatty()
    ):
        try:
            pot_str = input("Current pot size: ").strip()
            pot = float(pot_str)
            call_str = input("Bet to call (Enter for 0.0 / Check): ").strip()
            call = float(call_str) if call_str else 0.0
            eq_str = input("Hero equity (e.g. '35%' or '0.35', or Enter to specify hands): ").strip()
            if eq_str:
                raw_equity = eq_str
            else:
                hero_raw = input("Hero cards (e.g. 'Ah Kh'): ").strip()
                villain_raw = input("Villain cards (e.g. 'Qc Qd'): ").strip()
                board_raw = input("Board cards (optional, e.g. '2h 7h 8c'): ").strip() or None
            stack_str = input("Hero stack size (optional, Enter to skip): ").strip()
            if stack_str:
                stack = float(stack_str)
            fe_str = input("Fold equity estimate (optional, e.g. '20%', Enter for 0%): ").strip()
            if fe_str:
                raw_fe = fe_str
        except (EOFError, KeyboardInterrupt):
            sys.stderr.write("\nDecision calculation cancelled.\n")
            return 1
        except Exception as exc:
            sys.stderr.write(f"Error reading inputs: {exc}\n")
            return 2

    # Parse or simulate equity
    equity: float | None = None
    hero_cards = None
    villain_cards = None
    board_cards = None

    if raw_equity is not None:
        try:
            equity = parse_equity_value(raw_equity)
            if not (0.0 <= equity <= 1.0):
                sys.stderr.write(f"Error: Equity must be between 0.0 and 1.0 (or 0% and 100%), got {raw_equity}.\n")
                return 2
        except Exception as exc:
            sys.stderr.write(f"Error parsing equity '{raw_equity}': {exc}\n")
            return 2
    elif hero_raw and villain_raw:
        try:
            hero_cards = parse_cards_flexible(hero_raw)
            if len(hero_cards) != 2:
                sys.stderr.write(f"Error: Hero must have exactly 2 cards, got {len(hero_cards)}.\n")
                return 2
            villain_cards = parse_cards_flexible(villain_raw)
            if len(villain_cards) != 2:
                sys.stderr.write(f"Error: Villain must have exactly 2 cards, got {len(villain_cards)}.\n")
                return 2
            board_cards = parse_cards_flexible(board_raw) if board_raw else []
            sim_res = poker_equity.calculate_equity(
                hero=hero_cards,
                villain=villain_cards,
                board=board_cards,
                num_sims=args.sims,
                seed=args.seed,
            )
            equity = sim_res.equity
        except Exception as exc:
            sys.stderr.write(f"Error calculating hand equity: {exc}\n")
            return 2
    else:
        sys.stderr.write(
            "Error: Must specify hero equity (e.g. '--equity 35%' or '0.35 150 50') "
            "or hole cards (e.g. '--hero AhKh --villain QcQd').\n"
        )
        return 2

    if pot is None:
        sys.stderr.write("Error: Must specify pot size (e.g. '--pot 150' or '0.35 150 50').\n")
        return 2

    if pot < 0 or call < 0:
        sys.stderr.write("Error: Pot size and bet to call must be non-negative.\n")
        return 2

    if stack is not None and stack < 0:
        sys.stderr.write("Error: Stack size cannot be negative.\n")
        return 2

    # Fold equity parsing
    fold_equity = 0.0
    if raw_fe is not None:
        try:
            fold_equity = parse_equity_value(raw_fe)
            if not (0.0 <= fold_equity <= 1.0):
                sys.stderr.write(f"Error: Fold equity must be between 0.0 and 1.0, got {raw_fe}.\n")
                return 2
        except Exception as exc:
            sys.stderr.write(f"Error parsing fold equity '{raw_fe}': {exc}\n")
            return 2

    res = poker_equity.recommend_decision(
        equity=equity,
        pot_size=pot,
        bet_to_call=call,
        stack_size=stack,
        fold_equity=fold_equity,
        raise_size=raise_size,
        future_bets=future_bets,
    )

    if args.json:
        data = {
            "recommended_action": res.action.value,
            "ev": res.ev,
            "equity": res.equity,
            "pot_odds": res.pot_odds,
            "pot_size": res.pot_size,
            "bet_to_call": res.bet_to_call,
            "stack_size": stack,
            "raise_size": res.raise_size,
            "fold_equity": res.fold_equity,
            "future_bets": future_bets,
            "is_profitable": res.is_profitable,
            "ev_difference": res.ev_difference,
            "alternative_action": res.alternative_action.value if res.alternative_action else None,
            "ev_by_action": {k.value: v for k, v in res.ev_by_action.items()},
            "reasoning": res.reasoning,
            "spr": round(stack / pot, 2) if (stack is not None and pot > 0) else None,
        }
        if hero_cards and villain_cards:
            data["hero"] = cards_to_str(hero_cards)
            data["villain"] = cards_to_str(villain_cards)
            data["board"] = cards_to_str(board_cards) if board_cards else None
        print(json.dumps(data, indent=2))
        return 0

    col = Colors.GREEN if res.is_profitable else Colors.YELLOW
    print(colorize("Strategic Decision Recommendation", Colors.BOLD, use_color))
    print("=" * 64)
    print(colorize(f"Recommended Action: {res.action}", col, use_color))
    print(f"Expected Value    : {colorize(f'{res.ev:+.2f} chips', col, use_color)}")
    print(f"Hero Equity       : {res.equity * 100:.1f}%")
    pot_odds_str = f"{res.pot_odds * 100:.1f}%" if res.bet_to_call > 0 else "0.0% (Free check)"
    print(f"Pot Odds          : {pot_odds_str}")
    print(f"Pot Size          : {res.pot_size:.2f} chips (Cost to Call: {res.bet_to_call:.2f})")
    if stack is not None:
        spr_str = f"{stack / pot:.2f}" if pot > 0 else "N/A"
        print(f"Hero Stack        : {stack:.2f} chips (SPR: {spr_str})")
    if res.raise_size is not None:
        print(f"Suggested Sizing  : {res.raise_size:.2f} chips")
    if res.fold_equity > 0:
        print(f"Fold Equity       : {res.fold_equity * 100:.1f}%")
    if future_bets > 0:
        print(f"Future Bets (Imp) : +{future_bets:.2f} chips")
    if hero_cards and villain_cards:
        h_disp = format_cards_colored(hero_cards, enable=use_color)
        v_disp = format_cards_colored(villain_cards, enable=use_color)
        b_disp = format_cards_colored(board_cards, enable=use_color) if board_cards else "None (Preflop)"
        print(f"Hole Cards        : Hero {h_disp} vs Villain {v_disp} | Board: {b_disp}")
    print("-" * 64)

    # EV Comparison Table
    print(colorize("Action Expected Value (EV) Comparison:", Colors.BOLD, use_color))
    print(f"{'Action':<10} {'EV (chips)':<14} {'Delta vs Best':<16} {'Status':<16}")
    print("-" * 64)
    sorted_actions = sorted(res.ev_by_action.items(), key=lambda item: item[1], reverse=True)
    best_ev = res.ev
    for act, act_ev in sorted_actions:
        delta = act_ev - best_ev
        delta_str = "  0.00" if abs(delta) < 1e-4 else f"{delta:+6.2f}"
        if act == res.action:
            status = colorize("★ Recommended", col, use_color)
            act_label = colorize(act.value.upper(), col, use_color)
        elif res.alternative_action == act:
            status = colorize("◆ Viable Line", Colors.CYAN, use_color)
            act_label = act.value.upper()
        else:
            status = "Suboptimal"
            act_label = act.value.upper()
        print(f"{pad_visible(act_label, 10)} {act_ev:+10.2f} chips   {delta_str:<16} {status}")
    print("-" * 64)
    print(f"Analysis: {res.reasoning}")
    return 0


def handle_table_command(
    args: argparse.Namespace,
    use_color: bool = True,
    input_fn: Callable[[str], str] | None = None,
) -> int:
    """Execute the 'table' subcommand with optional interactive simulation.

    Supports static table display, pre-scripted action sequences, and
    fully interactive REPL mode with live decision recommendations,
    side pot tracking, community card dealing, and hand history display.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments including blinds, stacks, cards, board,
        interactive flag, actions, seed, and recommendation settings.
    use_color : bool, default=True
        Whether to use ANSI terminal colors in output.
    input_fn : Callable[[str], str] | None, default=None
        Custom input callable for testing or headless execution.

    Returns
    -------
    int
        Exit status code (0 for success, non-zero for error).
    """
    blinds_parts = args.blinds.split("/")
    if len(blinds_parts) != 2:
        sys.stderr.write("Error: Blinds must be in 'SB/BB' format (e.g. '1/2').\n")
        return 2

    try:
        sb, bb = float(blinds_parts[0]), float(blinds_parts[1])
    except ValueError:
        sys.stderr.write("Error: Blinds must be in 'SB/BB' format (e.g. '1/2').\n")
        return 2

    table = poker_equity.GameState.create_heads_up(
        small_blind=sb,
        big_blind=bb,
        hero_stack=args.hero_stack,
        villain_stack=args.villain_stack,
        hero_cards=args.hero_cards,
        villain_cards=args.villain_cards,
        hero_is_btn=not args.hero_is_bb,
        ante=args.ante,
    )

    # Pre-set board cards if provided
    if args.board:
        board_cards = parse_cards_flexible(args.board)
        if len(board_cards) == 3:
            table.deal_flop(board_cards)
        elif len(board_cards) == 4:
            table.deal_flop(board_cards[:3])
            table.deal_turn(board_cards[3])
        elif len(board_cards) == 5:
            table.deal_flop(board_cards[:3])
            table.deal_turn(board_cards[3])
            table.deal_river(board_cards[4])
        elif len(board_cards) > 0:
            sys.stderr.write(
                f"Error: Board must have 3, 4, or 5 cards, got {len(board_cards)}.\n"
            )
            return 2

    seed = getattr(args, "seed", None)
    deck = _init_deck(table, seed=seed)

    # Apply pre-scripted actions if provided
    if args.actions:
        result = _apply_scripted_actions(
            table, args.actions, use_color, deck=deck, quiet=args.json,
        )
        if result != 0:
            return result

    # JSON output mode (non-interactive)
    if args.json:
        print(json.dumps(table.to_dict(), indent=2))
        return 0

    # Interactive REPL mode
    if args.interactive:
        return _table_interactive_loop(
            table,
            use_color=use_color,
            show_recommend=args.recommend,
            num_sims=args.sims,
            seed=seed,
            deck=deck,
            input_fn=input_fn,
        )

    # Static display mode
    _print_table_state(table, use_color=use_color)
    if args.recommend and not table.is_hand_over:
        _print_table_recommendation(
            table, use_color=use_color, num_sims=args.sims, seed=seed,
        )
    if table.is_hand_over:
        _resolve_showdown(table, use_color=use_color)
    return 0


def _init_deck(table: "poker_equity.GameState", seed: int | None = None) -> Deck:
    """Initialize a deck for the table with known cards removed and shuffled.

    Parameters
    ----------
    table : GameState
        Table containing players and board cards.
    seed : int | None, default=None
        RNG seed for deterministic dealing.

    Returns
    -------
    Deck
        Prepared and shuffled Deck instance.
    """
    deck = Deck(seed=seed)
    known = []
    for p in table.players:
        if p.cards:
            known.extend(p.cards)
    if table.board:
        known.extend(table.board)
    if known:
        deck.remove(known)
    deck.shuffle()
    return deck


def _parse_action_input(text: str) -> tuple[str, float | None]:
    """Parse an action string into (action_name, optional_amount).

    Parameters
    ----------
    text : str
        User input like ``"call"``, ``"bet 10"``, ``"raise 25"``, or ``"all-in"``.

    Returns
    -------
    tuple[str, float | None]
        Normalized action name and optional chip amount.

    Examples
    --------
    >>> _parse_action_input("bet 10")
    ('bet', 10.0)
    >>> _parse_action_input("call")
    ('call', None)
    >>> _parse_action_input("all-in")
    ('all_in', None)
    """
    parts = text.strip().lower().split(None, 1)
    if not parts:
        return ("", None)
    action_str = parts[0].replace("-", "_")
    amount = None
    if len(parts) > 1:
        try:
            amount = float(parts[1])
        except ValueError:
            pass
    return (action_str, amount)


def _resolve_action(
    action_str: str, amount: float | None, table: "poker_equity.GameState",
) -> tuple["poker_equity.decision.Action", float | None]:
    """Resolve a text action string to an Action enum and validated amount.

    Parameters
    ----------
    action_str : str
        Normalized action name (e.g. ``"fold"``, ``"bet"``, ``"all_in"``).
    amount : float | None
        Optional chip amount for bet/raise actions.
    table : GameState
        Current game state for validation context.

    Returns
    -------
    tuple[Action, float | None]
        Resolved Action enum and optional amount.

    Raises
    ------
    ValueError
        If the action string is unrecognized or not legal in the current spot.
    """
    action_map = {
        "fold": poker_equity.decision.Action.FOLD,
        "f": poker_equity.decision.Action.FOLD,
        "check": poker_equity.decision.Action.CHECK,
        "x": poker_equity.decision.Action.CHECK,
        "call": poker_equity.decision.Action.CALL,
        "c": poker_equity.decision.Action.CALL,
        "bet": poker_equity.decision.Action.BET,
        "b": poker_equity.decision.Action.BET,
        "raise": poker_equity.decision.Action.RAISE,
        "r": poker_equity.decision.Action.RAISE,
        "all_in": poker_equity.decision.Action.ALL_IN,
        "allin": poker_equity.decision.Action.ALL_IN,
        "ai": poker_equity.decision.Action.ALL_IN,
        "jam": poker_equity.decision.Action.ALL_IN,
        "shove": poker_equity.decision.Action.ALL_IN,
    }

    if action_str not in action_map:
        raise ValueError(
            f"Unknown action '{action_str}'. "
            f"Valid: fold/f, check/x, call/c, bet/b <amount>, "
            f"raise/r <amount>, all-in/ai/jam/shove"
        )

    action = action_map[action_str]
    legal = table.legal_actions()
    if action not in legal:
        legal_strs = [a.value for a in legal]
        raise ValueError(
            f"Action '{action.value}' is not legal here. Legal: {', '.join(legal_strs)}"
        )

    return (action, amount)


def _apply_scripted_actions(
    table: "poker_equity.GameState",
    actions_str: str,
    use_color: bool = True,
    deck: Deck | None = None,
    quiet: bool = False,
) -> int:
    """Apply a comma-separated sequence of pre-scripted actions to the table.

    Parameters
    ----------
    table : GameState
        Mutable game state to apply actions to.
    actions_str : str
        Comma-separated action strings (e.g. ``"call,check,bet 10"``).
    use_color : bool, default=True
        Whether ANSI colors are enabled.
    deck : Deck | None, default=None
        Deck for auto-dealing cards on street transition.
    quiet : bool, default=False
        Whether to suppress intermediate print statements (e.g. for JSON output).

    Returns
    -------
    int
        0 on success, 2 on action parsing/validation error.
    """
    action_tokens = [a.strip() for a in actions_str.split(",") if a.strip()]
    for token in action_tokens:
        if table.is_hand_over:
            break
        action_str, amount = _parse_action_input(token)
        try:
            action, amt = _resolve_action(action_str, amount, table)
            table.apply_action(action, amount=amt)
            _check_street_transition(
                table, use_color=use_color, deck=deck, quiet=quiet,
            )
        except ValueError as exc:
            sys.stderr.write(f"Error applying action '{token}': {exc}\n")
            return 2
    return 0


def _print_table_state(
    table: "poker_equity.GameState",
    use_color: bool = True,
) -> None:
    """Print a formatted table state display to stdout.

    Parameters
    ----------
    table : GameState
        Current game state to display.
    use_color : bool, default=True
        Whether ANSI colors are enabled.
    """
    header = colorize(
        f"=== Table: {table.street.value} | "
        f"Blinds {table.small_blind:.0f}/{table.big_blind:.0f} ===",
        Colors.BOLD + Colors.CYAN, use_color,
    )
    print(header)
    print(table.summary())

    # Show side pots if present
    if len(table.pots) > 1:
        print(colorize("\nSide Pots:", Colors.BOLD + Colors.YELLOW, use_color))
        for idx, pot in enumerate(table.pots):
            label = "Main Pot" if pot.is_main else f"Side Pot {idx}"
            eligible = ", ".join(str(pid) for pid in pot.eligible_player_ids)
            print(f"  {label}: {pot.amount:.2f} chips (eligible: {eligible})")

    # Show hand history if any
    if table.history:
        print(colorize("\nHand History:", Colors.BOLD + Colors.DIM, use_color))
        for rec in table.history:
            amt_str = f" {rec.amount:.2f}" if rec.amount > 0 else ""
            ai_str = " [ALL-IN]" if rec.is_all_in else ""
            print(
                f"  [{rec.street.value}] {rec.player_id}: "
                f"{rec.action.value.upper()}{amt_str}{ai_str}"
            )
    print()


def _print_table_recommendation(
    table: "poker_equity.GameState",
    use_color: bool = True,
    num_sims: int = 2000,
    seed: int | None = None,
) -> None:
    """Print a live decision recommendation for the current player.

    Parameters
    ----------
    table : GameState
        Current game state.
    use_color : bool, default=True
        Whether ANSI colors are enabled.
    num_sims : int, default=2000
        Monte Carlo simulations for equity calculation.
    seed : int | None, default=None
        RNG seed for equity calculation.
    """
    curr = table.current_player
    if curr is None or not curr.has_cards:
        return

    active_with_cards = [p for p in table.active_players if p.has_cards]
    if len(active_with_cards) < 2:
        print(colorize(
            "  (Recommendation requires both players to have known cards)",
            Colors.DIM, use_color,
        ))
        return

    try:
        result = table.recommend_decision(num_sims=num_sims, seed=seed)
        rec_color = Colors.GREEN if result.is_profitable else Colors.RED
        print(colorize(
            f"  Recommendation for {curr.name}: {result.action.value.upper()} "
            f"(Equity: {result.equity:.1%}, EV: {result.ev:+.2f})",
            rec_color, use_color,
        ))
    except Exception as exc:
        print(colorize(
            f"  (Recommendation unavailable: {exc})",
            Colors.DIM, use_color,
        ))


def _format_action_prompt(
    table: "poker_equity.GameState", use_color: bool = True,
) -> str:
    """Build the interactive action prompt string showing legal actions.

    Parameters
    ----------
    table : GameState
        Current game state.
    use_color : bool, default=True
        Whether ANSI colors are enabled.

    Returns
    -------
    str
        Prompt string listing legal actions for the current player.
    """
    curr = table.current_player
    if curr is None:
        return ""
    legal = table.legal_actions()
    action_labels = []
    for a in legal:
        if a.value in ("bet", "raise"):
            min_r = table.min_legal_raise()
            action_labels.append(f"{a.value} <amount> (min {min_r:.0f})")
        else:
            action_labels.append(a.value)

    call_cost = table.bet_to_call()
    cost_str = f" | Cost to call: {call_cost:.2f}" if call_cost > 0 else ""
    player_label = colorize(
        f"{curr.name} ({curr.position})", Colors.BOLD + Colors.GREEN, use_color,
    )
    actions_str = colorize(
        " | ".join(action_labels), Colors.YELLOW, use_color,
    )
    return f"{player_label}{cost_str}\n  Actions: [{actions_str}]\n> "


def _table_interactive_loop(
    table: "poker_equity.GameState",
    use_color: bool = True,
    show_recommend: bool = False,
    num_sims: int = 2000,
    seed: int | None = None,
    deck: Deck | None = None,
    input_fn: Callable[[str], str] | None = None,
) -> int:
    """Run the interactive table REPL loop.

    Prompts the user for actions at each decision point, advances streets
    automatically when betting rounds complete, deals community cards,
    and displays updated table state after each action.

    Parameters
    ----------
    table : GameState
        Mutable game state.
    use_color : bool, default=True
        Whether ANSI colors are enabled.
    show_recommend : bool, default=False
        Whether to auto-display recommendations at each action point.
    num_sims : int, default=2000
        Monte Carlo simulations for equity/recommendation.
    seed : int | None, default=None
        RNG seed for simulation repeatability.
    deck : Deck | None, default=None
        Deck to draw community cards from.
    input_fn : Callable[[str], str] | None, default=None
        Custom input function for testing. Defaults to ``builtins.input``.

    Returns
    -------
    int
        Exit status code (0 for normal exit).
    """
    if input_fn is None:
        input_fn = input

    print(colorize(
        "=== Interactive Table Simulator ===",
        Colors.BOLD + Colors.CYAN, use_color,
    ))
    print(colorize(
        "Type 'help' for commands, 'quit' to exit.\n",
        Colors.DIM, use_color,
    ))

    _print_table_state(table, use_color=use_color)

    while not table.is_hand_over:
        if show_recommend:
            _print_table_recommendation(
                table, use_color=use_color, num_sims=num_sims, seed=seed,
            )

        prompt = _format_action_prompt(table, use_color=use_color)
        try:
            raw = input_fn(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting table simulator.")
            return 0

        raw = raw.strip()
        if not raw:
            continue

        cmd_lower = raw.lower()
        if cmd_lower in ("quit", "q", "exit"):
            print("Exiting table simulator.")
            return 0
        elif cmd_lower in ("help", "h", "?"):
            _print_interactive_help(use_color=use_color)
            continue
        elif cmd_lower in ("state", "s"):
            _print_table_state(table, use_color=use_color)
            continue
        elif cmd_lower in ("history", "hist"):
            _print_hand_history(table, use_color=use_color)
            continue
        elif cmd_lower in ("recommend", "rec"):
            _print_table_recommendation(
                table, use_color=use_color, num_sims=num_sims, seed=seed,
            )
            continue
        elif cmd_lower.startswith("deal ") or cmd_lower.startswith("board "):
            card_str = raw.split(None, 1)[1]
            try:
                new_c = parse_cards_flexible(card_str)
                table.board.extend(new_c)
                if deck:
                    deck.remove(new_c)
                print(colorize(f"  Added board cards: {cards_to_str(new_c)}", Colors.GREEN, use_color))
            except Exception as exc:
                print(colorize(f"  Error parsing cards: {exc}", Colors.RED, use_color))
            continue

        # Parse and apply action
        action_str, amount = _parse_action_input(raw)
        try:
            action, amt = _resolve_action(action_str, amount, table)
            record = table.apply_action(action, amount=amt)
            _print_action_record(record, use_color=use_color)
        except ValueError as exc:
            print(colorize(f"  Error: {exc}", Colors.RED, use_color))
            continue

        # Check for street transition
        _check_street_transition(table, use_color=use_color, deck=deck)

    # Hand complete
    print(colorize(
        "\n=== Hand Complete ===", Colors.BOLD + Colors.CYAN, use_color,
    ))
    _print_table_state(table, use_color=use_color)
    _resolve_showdown(table, use_color=use_color)
    return 0


def _print_action_record(
    record: "poker_equity.game_state.ActionRecord",
    use_color: bool = True,
) -> None:
    """Print a single action record after it's applied.

    Parameters
    ----------
    record : ActionRecord
        The action that was just applied.
    use_color : bool, default=True
        Whether ANSI colors are enabled.
    """
    amt_str = f" {record.amount:.2f}" if record.amount > 0 else ""
    ai_str = colorize(" [ALL-IN]", Colors.RED + Colors.BOLD, use_color) if record.is_all_in else ""
    action_label = record.action.value.upper()
    if record.action.is_aggressive:
        action_label = colorize(action_label, Colors.RED, use_color)
    elif record.action == poker_equity.decision.Action.FOLD:
        action_label = colorize(action_label, Colors.DIM, use_color)
    else:
        action_label = colorize(action_label, Colors.GREEN, use_color)

    print(f"  -> {record.player_id}: {action_label}{amt_str}{ai_str}  (Pot: {record.pot_after:.2f})")


def _check_street_transition(
    table: "poker_equity.GameState",
    use_color: bool = True,
    deck: Deck | None = None,
    quiet: bool = False,
) -> bool:
    """Check if the current betting round is complete and advance streets.

    Handles all-in board runouts, standard street advances (dealing new
    community cards from the deck if undealt), and showdown triggering.

    Parameters
    ----------
    table : GameState
        Mutable game state.
    use_color : bool, default=True
        Whether ANSI colors are enabled.
    deck : Deck | None, default=None
        Deck to draw community cards from.
    quiet : bool, default=False
        Whether to suppress print statements (e.g. for JSON output).

    Returns
    -------
    bool
        True if a transition or runout occurred, False otherwise.
    """
    if table.is_hand_over:
        return False

    active_not_allin = [p for p in table.active_players if not p.is_all_in]

    # 1. All-in runout check: 0 non-all-in, or 1 non-all-in who matched current bet
    is_all_in_runout = False
    if len(active_not_allin) == 0:
        is_all_in_runout = True
    elif len(active_not_allin) == 1:
        if active_not_allin[0].current_bet == table.current_bet:
            street_actions = [r for r in table.history if r.street == table.street]
            if street_actions:
                is_all_in_runout = True

    if is_all_in_runout:
        while table.street.next_street is not None:
            next_st = table.street.next_street
            needed = 3 if next_st == poker_equity.equity_curve.Street.FLOP else 1
            target_board_len = 3 if next_st == poker_equity.equity_curve.Street.FLOP else (4 if next_st == poker_equity.equity_curve.Street.TURN else 5)
            if len(table.board) < target_board_len and deck and deck.remaining >= needed:
                new_cards = deck.deal(needed)
            else:
                new_cards = []
            if not quiet:
                print(colorize(
                    f"\n--- Dealing {next_st.name}{(': ' + cards_to_str(new_cards)) if new_cards else ''} ---",
                    Colors.BOLD + Colors.MAGENTA, use_color,
                ))
            table.advance_street(new_cards=new_cards if new_cards else None)
            if not quiet:
                _print_table_state(table, use_color=use_color)

        table.collect_bets()
        return True

    # 2. Standard street progression
    all_matched = all(p.current_bet == table.current_bet for p in active_not_allin)
    if not all_matched:
        return False

    street_actions = [r for r in table.history if r.street == table.street]
    if not street_actions:
        return False

    acted_players = {r.player_id for r in street_actions}
    all_acted = all(p.player_id in acted_players for p in active_not_allin)
    if not all_acted:
        return False

    if table.street.next_street is not None:
        next_st = table.street.next_street
        needed = 3 if next_st == poker_equity.equity_curve.Street.FLOP else 1
        target_board_len = 3 if next_st == poker_equity.equity_curve.Street.FLOP else (4 if next_st == poker_equity.equity_curve.Street.TURN else 5)
        if len(table.board) < target_board_len and deck and deck.remaining >= needed:
            new_cards = deck.deal(needed)
        else:
            new_cards = []
        if not quiet:
            print(colorize(
                f"\n--- Advancing to {next_st.name}{(': ' + cards_to_str(new_cards)) if new_cards else ''} ---",
                Colors.BOLD + Colors.MAGENTA, use_color,
            ))
        table.advance_street(new_cards=new_cards if new_cards else None)
        if not quiet:
            _print_table_state(table, use_color=use_color)
        return True
    else:
        table.collect_bets()
        return True


def _resolve_showdown(
    table: "poker_equity.GameState",
    use_color: bool = True,
) -> None:
    """Resolve and print the showdown outcome or uncalled pot award.

    Parameters
    ----------
    table : GameState
        Completed game state.
    use_color : bool, default=True
        Whether ANSI colors are enabled.
    """
    if len(table.active_players) == 1:
        winner = table.active_players[0]
        w_name = colorize(f"{winner.name} ({winner.player_id})", Colors.BOLD + Colors.GREEN, use_color)
        print(colorize("\nHand Result:", Colors.BOLD, use_color))
        print(f"  {w_name} wins {table.total_pot:.2f} chips (uncontested)!")
        return

    print(colorize("\nShowdown:", Colors.BOLD + Colors.CYAN, use_color))
    evaluated_hands = {}
    for p in table.active_players:
        if p.has_cards and len(table.board) >= 5:
            full_cards = list(p.cards) + list(table.board)
            res = evaluate_hand(full_cards)
            evaluated_hands[p.player_id] = res
            cards_str = cards_to_str(p.cards)
            print(f"  {p.name} [{cards_str}]: {res.description} ({res.rank.label})")
        elif p.has_cards:
            cards_str = cards_to_str(p.cards)
            print(f"  {p.name} [{cards_str}]")
        else:
            print(f"  {p.name} [?? ??]")

    if not evaluated_hands:
        return

    for idx, pot in enumerate(table.pots):
        label = "Main Pot" if pot.is_main else f"Side Pot {idx}"
        contenders = [
            pid for pid in pot.eligible_player_ids
            if pid in evaluated_hands
        ]
        if not contenders:
            continue

        best_score = max(evaluated_hands[pid] for pid in contenders)
        winners = [pid for pid in contenders if evaluated_hands[pid] == best_score]

        if len(winners) == 1:
            win_p = table.get_player(winners[0])
            w_label = colorize(f"{win_p.name}", Colors.BOLD + Colors.GREEN, use_color)
            print(f"  -> {label} ({pot.amount:.2f} chips): won by {w_label} ({best_score.description})")
        else:
            win_names = [table.get_player(pid).name for pid in winners]
            split_amt = pot.amount / len(winners)
            names_str = colorize(", ".join(win_names), Colors.BOLD + Colors.YELLOW, use_color)
            print(f"  -> {label} ({pot.amount:.2f} chips): chopped between {names_str} ({split_amt:.2f} each)")


def _print_interactive_help(use_color: bool = True) -> None:
    """Print help text for interactive table commands.

    Parameters
    ----------
    use_color : bool, default=True
        Whether ANSI colors are enabled.
    """
    title = colorize("Interactive Table Commands:", Colors.BOLD, use_color)
    print(f"\n{title}")
    commands = [
        ("fold / f", "Fold your hand"),
        ("check / x", "Check (when no bet to call)"),
        ("call / c", "Call the current bet"),
        ("bet <amt> / b <amt>", "Open with a bet"),
        ("raise <amt> / r <amt>", "Raise to <amt> total"),
        ("all-in / ai / jam", "Go all-in"),
        ("deal <cards>", "Manually deal/add board cards"),
        ("state / s", "Redisplay table state"),
        ("history / hist", "Show hand history"),
        ("recommend / rec", "Get decision recommendation"),
        ("help / h / ?", "Show this help"),
        ("quit / q / exit", "Exit simulator"),
    ]
    for cmd, desc in commands:
        cmd_fmt = colorize(f"  {cmd:<26}", Colors.YELLOW, use_color)
        print(f"{cmd_fmt} {desc}")
    print()


def _print_hand_history(
    table: "poker_equity.GameState",
    use_color: bool = True,
) -> None:
    """Print the full hand history for the current game.

    Parameters
    ----------
    table : GameState
        Current game state.
    use_color : bool, default=True
        Whether ANSI colors are enabled.
    """
    if not table.history:
        print(colorize("  No actions recorded yet.", Colors.DIM, use_color))
        return
    print(colorize("Hand History:", Colors.BOLD, use_color))
    for rec in table.history:
        amt_str = f" {rec.amount:.2f}" if rec.amount > 0 else ""
        ai_str = " [ALL-IN]" if rec.is_all_in else ""
        print(
            f"  [{rec.street.value}] {rec.player_id}: "
            f"{rec.action.value.upper()}{amt_str}{ai_str} "
            f"(pot: {rec.pot_before:.2f} -> {rec.pot_after:.2f})"
        )
    print()


def format_ascii_equity_curve(curve: EquityCurve, use_color: bool = True) -> str:
    """Format an ASCII equity trajectory chart across streets.

    Parameters
    ----------
    curve : EquityCurve
        Calculated equity curve containing street points.
    use_color : bool, default=True
        Whether ANSI color codes are enabled.

    Returns
    -------
    str
        Formatted multiline ASCII trajectory representation.
    """
    lines = []
    lines.append(colorize("Street-by-Street Equity Trajectory:", Colors.BOLD, use_color))
    lines.append("=" * 68)

    turning_pt = curve.turning_point

    for pt in curve.points:
        st_name = pt.street.name
        h_eq = pt.hero_equity
        v_eq = pt.villain_equity
        h_bar = make_equity_bar(h_eq, width=20, use_color=use_color)
        v_bar = make_equity_bar(v_eq, width=20, use_color=use_color)

        delta_tag = ""
        if pt.street != Street.PREFLOP:
            d_val = pt.hero_delta
            d_col = Colors.GREEN if d_val > 0 else (Colors.RED if d_val < 0 else Colors.DIM)
            delta_tag = f" (Hero Δ: {colorize(f'{d_val * 100:+.1f}%', d_col, use_color)})"

        if turning_pt and pt.street == turning_pt.street:
            delta_tag += " " + colorize("[TURNING POINT]", Colors.BOLD + Colors.YELLOW, use_color)

        if pt.street == Street.RIVER and curve.is_bad_beat:
            delta_tag += " " + colorize("[BAD BEAT!]", Colors.BOLD + Colors.RED, use_color)

        b_cards = cards_to_str(pt.board) if pt.board else "None"

        lines.append(
            f"  {colorize(f'[{st_name:<7}]', Colors.BOLD + Colors.CYAN, use_color)} "
            f"Board: [{b_cards:<14}]{delta_tag}"
        )
        lines.append(
            f"    Hero   : {h_eq * 100:5.1f}% {h_bar}  [{pt.hand_description(0)}]"
        )
        lines.append(
            f"    Villain: {v_eq * 100:5.1f}% {v_bar}  [{pt.hand_description(1)}]"
        )
        lines.append("")

    # Trajectory Spark Chart
    lines.append(colorize("Equity Spark Graph (Hero):", Colors.BOLD + Colors.CYAN, use_color))
    spark_levels = ["  100% |", "   80% |", "   60% |", "   40% |", "   20% |", "    0% |"]
    grid_rows = [[" " for _ in range(len(curve.points) * 10)] for _ in range(5)]

    for col_idx, pt in enumerate(curve.points):
        x = col_idx * 10 + 3
        eq = pt.hero_equity
        row = min(4, max(0, 4 - int(round(eq * 4))))
        marker = colorize("●", Colors.GREEN if eq >= 0.5 else Colors.RED, use_color)
        grid_rows[row][x] = marker
        pct_label = f"{eq * 100:.0f}%"
        for i, ch in enumerate(pct_label):
            if x + 1 + i < len(grid_rows[row]):
                grid_rows[row][x + 1 + i] = ch

    for r_idx, row in enumerate(grid_rows):
        lines.append(f"{spark_levels[r_idx]} {''.join(row)}")
    lines.append("       +" + "-" * (len(curve.points) * 10 + 4))

    st_labels = "       " + "".join(f"{pt.street.name:<10}" for pt in curve.points)
    lines.append(colorize(st_labels, Colors.DIM, use_color))

    return "\n".join(lines)


def handle_curve_command(args: argparse.Namespace, use_color: bool = True) -> int:
    """Execute the 'curve' subcommand for street-by-street equity analysis."""
    hero_raw = args.hero
    villain_raw = args.villain
    board_raw = args.board
    cards_pos = getattr(args, "cards", []) or []

    # Positional resolution
    if cards_pos:
        if len(cards_pos) == 3:
            hero_raw = hero_raw or cards_pos[0]
            villain_raw = villain_raw or cards_pos[1]
            board_raw = board_raw or cards_pos[2]
        elif len(cards_pos) >= 5 and (hero_raw is None or villain_raw is None or board_raw is None):
            try:
                all_pos = parse_cards_flexible(" ".join(cards_pos))
                if len(all_pos) in (7, 8, 9):
                    if hero_raw is None:
                        hero_raw = cards_to_str(all_pos[0:2])
                    if villain_raw is None:
                        villain_raw = cards_to_str(all_pos[2:4])
                    if board_raw is None:
                        board_raw = cards_to_str(all_pos[4:])
            except Exception:
                pass
        elif hero_raw is not None and villain_raw is not None and board_raw is None:
            board_raw = " ".join(cards_pos)

    if not hero_raw or not villain_raw or not board_raw:
        sys.stderr.write(
            "Error: Hero cards, villain cards, and board cards (3, 4, or 5 cards) are required "
            "(e.g. 'poker-equity curve AhAs KhKs \"2h 7h 8c 3c 4c\"' or '--hero AhAs --villain KhKs --board \"2h 7h 8c 3c 4c\"').\n"
        )
        return 2

    try:
        h_cards = parse_cards_flexible(hero_raw)
        v_cards = parse_cards_flexible(villain_raw)
        b_cards = parse_cards_flexible(board_raw)
    except Exception as exc:
        sys.stderr.write(f"Error parsing cards: {exc}\n")
        return 2

    if len(h_cards) != 2:
        sys.stderr.write(f"Error: Hero must have exactly 2 cards, got {len(h_cards)}.\n")
        return 2
    if len(v_cards) != 2:
        sys.stderr.write(f"Error: Villain must have exactly 2 cards, got {len(v_cards)}.\n")
        return 2
    if len(b_cards) not in (3, 4, 5):
        sys.stderr.write(
            f"Error: Board must have 3 (flop), 4 (turn), or 5 (river) cards, got {len(b_cards)}.\n"
        )
        return 2

    all_cards = h_cards + v_cards + b_cards
    if len(set(all_cards)) != len(all_cards):
        sys.stderr.write(f"Error: Duplicate cards detected: {cards_to_str(all_cards)}\n")
        return 2

    seed = getattr(args, "seed", None)
    try:
        curve = calculate_equity_curve(
            hands=[h_cards, v_cards],
            board=b_cards,
            num_sims=args.sims,
            seed=seed,
        )
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    next_dist = None
    if getattr(args, "next", False):
        if len(b_cards) not in (3, 4):
            sys.stderr.write(
                f"Error: --next is only supported on flop (3 cards) or turn (4 cards), got {len(b_cards)}.\n"
            )
            return 2
        try:
            next_dist = calculate_next_street_distribution(
                hero=h_cards,
                villain=v_cards,
                board=b_cards,
                seed=seed,
            )
        except Exception as exc:
            sys.stderr.write(f"Error analyzing next street: {exc}\n")
            return 1

    if args.json:
        data = curve.to_dict()
        data["plot_data"] = curve.plot_data()
        if next_dist is not None:
            data["next_street_distribution"] = next_dist.to_dict()
        print(json.dumps(data, indent=2))
        return 0

    print(curve.summary())

    if getattr(args, "chart", False) or getattr(args, "ascii", False):
        print()
        print(format_ascii_equity_curve(curve, use_color=use_color))

    if next_dist is not None:
        print()
        print(next_dist.summary())

    return 0


def format_run_it_twice_summary(
    res: RunItTimesResult,
    pot: float | None = None,
    board: Sequence[Card] | None = None,
    use_color: bool = True,
) -> str:
    """Format terminal report for Run-It-Twice (or Run-It-N-Times) simulation.

    Parameters
    ----------
    res : RunItTimesResult
        Simulation or exact enumeration result.
    pot : float | None, default=None
        Optional total pot size in chips or currency.
    board : Sequence[Card] | None, default=None
        Existing board cards if any.
    use_color : bool, default=True
        Whether ANSI color formatting is enabled.

    Returns
    -------
    str
        Formatted multiline terminal output.
    """
    method_str = "Exact Enumeration" if res.is_exact else f"{res.num_sims:,} Simulations"
    header = f"=== Run It {res.num_runs} Times Analysis ({method_str}) ==="
    lines = [
        colorize(header, Colors.BOLD + Colors.CYAN, use_color),
    ]
    if board:
        st_name = "Preflop" if len(board) == 0 else ("Flop" if len(board) == 3 else "Turn")
        lines.append(f"Board: [{colorize(cards_to_str(board), Colors.BOLD, use_color)}] ({st_name})")
    lines.append("=" * len(strip_ansi(header)))

    lines.append(colorize("Overall Equities:", Colors.BOLD, use_color))
    for p_idx, eq in enumerate(res.equities):
        label = "Hero (Player 1)" if p_idx == 0 else ("Villain (Player 2)" if (p_idx == 1 and len(res.equities) == 2) else f"Player {p_idx + 1}")
        col = Colors.GREEN if eq >= 0.5 else Colors.RED
        bar = make_equity_bar(eq, width=22, use_color=use_color)
        lines.append(f"  {label:<18}: {colorize(f'{eq * 100:6.2f}%', col, use_color)}  {bar}")

    lines.append("")
    lines.append(colorize(f"Multi-Board Outcomes ({res.num_runs} Boards):", Colors.BOLD, use_color))
    for p_idx, scoop in enumerate(res.scoop_pcts):
        label = f"Hero scoops all {res.num_runs}" if p_idx == 0 else (f"Villain scoops all {res.num_runs}" if (p_idx == 1 and len(res.scoop_pcts) == 2) else f"Player {p_idx + 1} scoops all {res.num_runs}")
        lines.append(f"  {label:<28}: {scoop * 100:6.2f}%")
    lines.append(f"  {'Split pot (split boards)':<28}: {res.split_pct * 100:6.2f}%")

    if pot is not None and pot > 0:
        lines.append("")
        lines.append(colorize("Expected Payouts:", Colors.BOLD, use_color))
        lines.append(f"  Total Pot Size             : {pot:.2f} chips")
        payouts = res.expected_payouts(pot)
        for p_idx, pay in enumerate(payouts):
            label = "Hero expected payout" if p_idx == 0 else ("Villain expected payout" if (p_idx == 1 and len(payouts) == 2) else f"Player {p_idx + 1} expected payout")
            lines.append(f"  {label:<28}: {pay:.2f} chips ({res.equities[p_idx] * 100:.1f}%)")

    lines.append("")
    lines.append(colorize("Variance Reduction:", Colors.BOLD, use_color))
    lines.append(f"  Single-board variance      : {res.variance_single:.4f}")
    lines.append(f"  Run It {res.num_runs} variance         : {res.variance_multi:.4f}")
    vr_col = Colors.BOLD + Colors.GREEN if res.variance_reduction > 0 else Colors.YELLOW
    lines.append(
        f"  Variance reduction         : {colorize(f'{res.variance_reduction:.2f}%', vr_col, use_color)} "
        f"{colorize('(Risk reduction)', Colors.DIM, use_color)}"
    )

    if len(res.board_equities) > 1:
        lines.append("")
        lines.append(colorize("Per-Board Breakdown:", Colors.BOLD, use_color))
        for b_idx, b_eqs in enumerate(res.board_equities):
            eq_strs = [f"P{p+1}: {e * 100:.1f}%" for p, e in enumerate(b_eqs)]
            lines.append(f"  Board {b_idx + 1}: {', '.join(eq_strs)}")

    return "\n".join(lines)


def handle_twice_command(args: argparse.Namespace, use_color: bool = True) -> int:
    """Execute the 'twice' subcommand for Run-It-Twice (and Run-It-N-Times) simulation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    use_color : bool, default=True
        Whether ANSI color formatting is enabled.

    Returns
    -------
    int
        Exit status code (0 for success, non-zero for error).
    """
    hero_raw = args.hero
    villain_raw = args.villain
    board_raw = args.board
    cards_pos = getattr(args, "cards", []) or []

    # Positional resolution
    if cards_pos:
        if len(cards_pos) == 2:
            hero_raw = hero_raw or cards_pos[0]
            villain_raw = villain_raw or cards_pos[1]
        elif len(cards_pos) == 3:
            hero_raw = hero_raw or cards_pos[0]
            villain_raw = villain_raw or cards_pos[1]
            board_raw = board_raw or cards_pos[2]
        elif len(cards_pos) >= 4 and (hero_raw is None or villain_raw is None):
            try:
                all_pos = parse_cards_flexible(" ".join(cards_pos))
                if len(all_pos) >= 4:
                    if hero_raw is None:
                        hero_raw = cards_to_str(all_pos[0:2])
                    if villain_raw is None:
                        villain_raw = cards_to_str(all_pos[2:4])
                    if board_raw is None and len(all_pos) > 4:
                        board_raw = cards_to_str(all_pos[4:])
            except Exception:
                pass
        elif hero_raw is not None and villain_raw is not None and board_raw is None:
            board_raw = " ".join(cards_pos)

    if not hero_raw or not villain_raw:
        sys.stderr.write(
            "Error: Both hero and villain hole cards are required for Run It Twice "
            "(e.g. 'poker-equity twice AhKh QcQd' or '--hero AhKh --villain QcQd').\n"
        )
        return 2

    try:
        h_cards = parse_cards_flexible(hero_raw)
        v_cards = parse_cards_flexible(villain_raw)
        b_cards = parse_cards_flexible(board_raw) if board_raw else []
    except Exception as exc:
        sys.stderr.write(f"Error parsing cards: {exc}\n")
        return 2

    if len(h_cards) != 2:
        sys.stderr.write(f"Error: Hero must have exactly 2 cards, got {len(h_cards)}.\n")
        return 2
    if len(v_cards) != 2:
        sys.stderr.write(f"Error: Villain must have exactly 2 cards, got {len(v_cards)}.\n")
        return 2
    if len(b_cards) not in (0, 3, 4):
        sys.stderr.write(
            f"Error: Board must have 0 (preflop), 3 (flop), or 4 (turn) cards, got {len(b_cards)}.\n"
        )
        return 2

    all_cards = h_cards + v_cards + b_cards
    if len(set(all_cards)) != len(all_cards):
        sys.stderr.write(f"Error: Duplicate cards detected: {cards_to_str(all_cards)}\n")
        return 2

    if args.exact:
        if len(b_cards) != 4:
            sys.stderr.write(
                f"Error: --exact is only supported on the turn (4 board cards), got {len(b_cards)}.\n"
            )
            return 2
        try:
            res = exact_run_it_twice_turn(h_cards, v_cards, b_cards)
        except Exception as exc:
            sys.stderr.write(f"Error: {exc}\n")
            return 1
    else:
        num_runs = getattr(args, "runs", 2) or 2
        if num_runs < 2:
            sys.stderr.write(f"Error: --runs must be at least 2, got {num_runs}.\n")
            return 2
        try:
            if num_runs == 2:
                res = simulate_run_it_twice(
                    hero=h_cards,
                    villain=v_cards,
                    board=b_cards if b_cards else None,
                    num_sims=args.sims,
                    seed=args.seed,
                )
            else:
                res = simulate_run_it_n(
                    hands=[h_cards, v_cards],
                    board=b_cards if b_cards else None,
                    num_runs=num_runs,
                    num_sims=args.sims,
                    seed=args.seed,
                )
        except Exception as exc:
            sys.stderr.write(f"Error: {exc}\n")
            return 1

    if args.json:
        data = res.to_dict()
        data["hero"] = cards_to_str(h_cards)
        data["villain"] = cards_to_str(v_cards)
        data["board"] = cards_to_str(b_cards)
        if args.pot is not None:
            data["pot_size"] = args.pot
            data["expected_payouts"] = res.expected_payouts(args.pot)
        print(json.dumps(data, indent=2))
        return 0

    print(format_run_it_twice_summary(res, pot=args.pot, board=b_cards, use_color=use_color))
    return 0


# =========================================================================
# Main Entry Point
# =========================================================================


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entry point function.

    Parameters
    ----------
    argv : Sequence[str] | None, default=None
        Command-line arguments. If None, uses ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status code (0 for success, non-zero for error).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    use_color = should_enable_color(no_color_flag=args.no_color)

    try:
        if args.command == "equity":
            return handle_equity_command(args, use_color=use_color)
        elif args.command == "odds":
            return handle_odds_command(args, use_color=use_color)
        elif args.command == "outs":
            return handle_outs_command(args, use_color=use_color)
        elif args.command == "decision":
            return handle_decision_command(args, use_color=use_color)
        elif args.command == "table":
            return handle_table_command(args, use_color=use_color)
        elif args.command == "curve":
            return handle_curve_command(args, use_color=use_color)
        elif args.command == "range":
            return handle_range_command(args, use_color=use_color)
        elif args.command in ("twice", "rit", "run-it-twice"):
            return handle_twice_command(args, use_color=use_color)
        else:
            parser.print_help()
            return 1
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
