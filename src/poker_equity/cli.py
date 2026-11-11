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
from poker_equity.breakdown import range_vs_hand_breakdown
from poker_equity.card import Card, Rank, Suit
from poker_equity.constants import cards_to_str, parse_cards
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

    Returns
    -------
    str
        Formatted equity bar e.g. ``'[██████░░░░]'``.
    """
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
        "--pot",
        "-p",
        type=float,
        required=True,
        help="Current pot size before calling.",
    )
    p_odds.add_argument(
        "--call",
        "-c",
        type=float,
        required=True,
        help="Amount to call.",
    )
    p_odds.add_argument(
        "--equity",
        "-e",
        type=float,
        default=None,
        help="Hero equity as a fraction (0.0 to 1.0) or percentage (0 to 100).",
    )
    p_odds.add_argument(
        "--future-bets",
        "-f",
        type=float,
        default=0.0,
        help="Expected future chips won on later streets (implied odds).",
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
        "--hero",
        "-H",
        type=str,
        required=True,
        help="Hero hole cards (e.g. 'Ah Kh').",
    )
    p_outs.add_argument(
        "--villain",
        "-V",
        type=str,
        required=True,
        help="Villain hole cards (e.g. 'Qc Qd').",
    )
    p_outs.add_argument(
        "--board",
        "-b",
        type=str,
        required=True,
        help="Flop (3 cards) or Turn (4 cards).",
    )
    p_outs.add_argument(
        "--method",
        "-m",
        choices=["exact", "rule", "adjusted_rule"],
        default="exact",
        help="Method to calculate equity from outs (default: exact).",
    )

    # ---------------------------------------------------------------------
    # Subcommand: decision
    # ---------------------------------------------------------------------
    p_dec = subparsers.add_parser(
        "decision",
        parents=[common_parser],
        help="Strategic decision recommendation (Fold, Call, Raise).",
        description="Recommends the highest EV action based on equity, pot odds, and fold equity.",
    )
    p_dec.add_argument(
        "--equity",
        "-e",
        type=float,
        required=True,
        help="Hero equity (0.0 to 1.0 or percentage).",
    )
    p_dec.add_argument(
        "--pot",
        "-p",
        type=float,
        required=True,
        help="Current pot size.",
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
        type=float,
        default=0.0,
        help="Estimated fold equity (0.0 to 1.0).",
    )

    # ---------------------------------------------------------------------
    # Subcommand: table
    # ---------------------------------------------------------------------
    p_tbl = subparsers.add_parser(
        "table",
        parents=[common_parser],
        help="Create and display a poker table game state.",
        description="Inspect table state, blinds, stacks, positions, and SPR.",
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

    # ---------------------------------------------------------------------
    # Subcommand: curve
    # ---------------------------------------------------------------------
    p_crv = subparsers.add_parser(
        "curve",
        parents=[common_parser],
        help="Track street-by-street equity transitions across board cards.",
        description="Analyzes equity curve from Preflop -> Flop -> Turn -> River.",
    )
    p_crv.add_argument(
        "--hero",
        "-H",
        type=str,
        required=True,
        help="Hero hole cards (e.g. 'Ah As').",
    )
    p_crv.add_argument(
        "--villain",
        "-V",
        type=str,
        required=True,
        help="Villain hole cards (e.g. 'Kh Ks').",
    )
    p_crv.add_argument(
        "--board",
        "-b",
        type=str,
        required=True,
        help="Board cards (3 to 5 cards, e.g. '2h 7h 8c 3c 4c').",
    )
    p_crv.add_argument(
        "--sims",
        "-n",
        type=int,
        default=2000,
        help="Simulations per street (default: 2000).",
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
    """Execute the 'odds' subcommand."""
    pot = args.pot
    call = args.call
    req_odds = poker_equity.pot_odds(pot, call)
    ratio = poker_equity.pot_odds_ratio(pot, call)

    equity = args.equity
    if equity is not None and equity > 1.0:
        equity = equity / 100.0  # Allow user to pass percentage (e.g. 35 -> 0.35)

    ev_val = None
    if equity is not None:
        ev_val = poker_equity.expected_value(equity, pot + args.future_bets, call)

    if args.json:
        data = {
            "pot_size": pot,
            "bet_to_call": call,
            "pot_odds_pct": req_odds,
            "pot_odds_ratio": ratio,
            "equity": equity,
            "ev": ev_val,
        }
        print(json.dumps(data, indent=2))
        return 0

    print(colorize("Pot Odds & EV Analysis", Colors.BOLD, use_color))
    print("=" * 40)
    print(f"Pot Size        : {pot:.2f} chips")
    print(f"Cost to Call    : {call:.2f} chips")
    print(f"Pot Odds        : {colorize(f'{req_odds * 100:.2f}%', Colors.CYAN, use_color)} ({ratio})")

    if equity is not None:
        ev_col = Colors.GREEN if ev_val >= 0 else Colors.RED
        print(f"Hero Equity     : {equity * 100:.2f}%")
        print(f"Expected Value  : {colorize(f'{ev_val:+.2f} chips', ev_col, use_color)}")
        rec = "CALL" if ev_val >= 0 else "FOLD"
        print(f"Recommendation  : {colorize(rec, ev_col, use_color)}")

    return 0


def handle_outs_command(args: argparse.Namespace, use_color: bool = True) -> int:
    """Execute the 'outs' subcommand."""
    res = poker_equity.calculate_outs(
        hero=args.hero,
        villain=args.villain,
        board=args.board,
    )

    board_cards = parse_cards(args.board)
    street = "flop" if len(board_cards) == 3 else "turn"
    equity = poker_equity.outs_to_equity(res.total_outs, street=street, method=args.method)

    if args.json:
        data = {
            "total_outs": res.total_outs,
            "win_outs": len(res.win_outs),
            "tie_outs": len(res.tie_outs),
            "drawing_equity": equity,
            "method": args.method,
            "win_cards": [str(c) for c in res.win_outs],
        }
        print(json.dumps(data, indent=2))
        return 0

    print(colorize("Drawing Outs Calculation", Colors.BOLD, use_color))
    print("=" * 40)
    print(f"Hero   : {args.hero}")
    print(f"Villain: {args.villain}")
    print(f"Board  : {args.board}")
    print("-" * 40)
    print(f"Total Outs : {colorize(str(res.total_outs), Colors.GREEN, use_color)}")
    print(f"Equity ({args.method}): {colorize(f'{equity * 100:.2f}%', Colors.CYAN, use_color)}")
    if res.win_outs:
        outs_str = cards_to_str(res.win_outs[:12])
        ellipsis = "..." if len(res.win_outs) > 12 else ""
        print(f"Outs Cards : {outs_str}{ellipsis}")

    return 0


def handle_decision_command(args: argparse.Namespace, use_color: bool = True) -> int:
    """Execute the 'decision' subcommand."""
    eq = args.equity
    if eq > 1.0:
        eq = eq / 100.0

    res = poker_equity.recommend_decision(
        equity=eq,
        pot_size=args.pot,
        bet_to_call=args.call,
        stack_size=args.stack,
        fold_equity=args.fold_equity,
    )

    if args.json:
        data = {
            "recommended_action": res.action.value,
            "ev": res.ev,
            "equity": res.equity,
            "pot_odds": res.pot_odds,
            "ev_by_action": {k.value: v for k, v in res.ev_by_action.items()},
            "reasoning": res.reasoning,
        }
        print(json.dumps(data, indent=2))
        return 0

    col = Colors.GREEN if res.is_profitable else Colors.YELLOW
    print(colorize(f"Recommended Action: {res.action}", col, use_color))
    print(f"Expected Value    : {res.ev:+.2f} chips")
    print(f"Hero Equity       : {res.equity * 100:.1f}%")
    print(f"Pot Odds          : {res.pot_odds * 100:.1f}%")
    print("-" * 40)
    print(f"Analysis: {res.reasoning}")
    return 0


def handle_table_command(args: argparse.Namespace, use_color: bool = True) -> int:
    """Execute the 'table' subcommand."""
    blinds_parts = args.blinds.split("/")
    if len(blinds_parts) != 2:
        sys.stderr.write("Error: Blinds must be in 'SB/BB' format (e.g. '1/2').\n")
        return 2

    sb, bb = float(blinds_parts[0]), float(blinds_parts[1])
    table = poker_equity.GameState.create_heads_up(
        small_blind=sb,
        big_blind=bb,
        hero_stack=args.hero_stack,
        villain_stack=args.villain_stack,
        hero_cards=args.hero_cards,
        villain_cards=args.villain_cards,
    )

    if args.json:
        print(json.dumps(table.to_dict(), indent=2))
        return 0

    print(table.summary())
    return 0


def handle_curve_command(args: argparse.Namespace, use_color: bool = True) -> int:
    """Execute the 'curve' subcommand."""
    curve = poker_equity.calculate_equity_curve(
        hands=[args.hero, args.villain],
        board=args.board,
        num_sims=args.sims,
    )

    if args.json:
        print(json.dumps(curve.to_dict(), indent=2))
        return 0

    print(curve.summary())
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
        else:
            parser.print_help()
            return 1
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
