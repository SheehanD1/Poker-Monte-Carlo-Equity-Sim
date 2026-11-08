"""Command-line interface (CLI) for the poker-equity simulator.

Provides subcommands for equity calculation, range analysis, pot odds,
outs, decision recommendations, table tracking, and equity curves.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence

import poker_equity
from poker_equity.card import Card
from poker_equity.constants import cards_to_str, parse_cards


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
        "--sims",
        "-n",
        type=int,
        default=10_000,
        help="Number of Monte Carlo simulations (default: 10,000).",
    )
    p_eq.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="RNG seed for deterministic evaluation.",
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
        "--hero",
        "-H",
        type=str,
        required=True,
        help="Hero hand or range string (e.g. 'Ah Kh' or 'QQ+, AKs').",
    )
    p_rng.add_argument(
        "--villain",
        "-V",
        type=str,
        required=True,
        help="Villain range string (e.g. 'TT+, AJs+, KQs').",
    )
    p_rng.add_argument(
        "--board",
        "-b",
        type=str,
        default=None,
        help="Community cards (e.g. '2h 7h 8c').",
    )
    p_rng.add_argument(
        "--sims",
        "-n",
        type=int,
        default=10_000,
        help="Number of simulations (default: 10,000).",
    )
    p_rng.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="RNG seed.",
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
    # Determine players
    if args.players:
        hands = args.players
    elif args.hero and args.villain:
        hands = [args.hero, args.villain]
    else:
        sys.stderr.write(
            "Error: Must specify either --hero and --villain or --players.\n"
        )
        return 2

    sims = max(1, args.sims)
    results = poker_equity.calculate_equities(
        hands=hands,
        board=args.board,
        num_sims=sims,
        seed=args.seed,
    )

    if args.json:
        data = {
            "board": args.board,
            "sims": sims,
            "players": [
                {
                    "hand": str(hands[idx]),
                    "equity": r.equity,
                    "win_pct": r.win_pct,
                    "tie_pct": r.tie_pct,
                    "loss_pct": r.loss_pct,
                }
                for idx, r in enumerate(results)
            ],
        }
        print(json.dumps(data, indent=2))
        return 0

    # Formatted terminal output
    print(colorize("Poker Monte Carlo Equity Simulation", Colors.BOLD, use_color))
    print("=" * 45)
    b_str = args.board if args.board else "None (Preflop)"
    print(f"Board : {b_str}")
    print(f"Sims  : {sims:,}")
    print("-" * 45)

    for idx, r in enumerate(results):
        hand_label = str(hands[idx])
        p_label = "Hero   " if idx == 0 and len(hands) == 2 else f"Player {idx + 1}"
        eq_str = f"{r.equity * 100:.2f}%"
        colored_eq = colorize(eq_str, Colors.GREEN if r.equity >= 0.5 else Colors.YELLOW, use_color)
        print(f"{p_label} [{hand_label}]: Equity {colored_eq} (Win: {r.win_pct * 100:.1f}%, Tie: {r.tie_pct * 100:.1f}%)")

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
            # Scaffold for range command (completed in Commit 53)
            print("Range-vs-range CLI command scaffolded.")
            return 0
        else:
            parser.print_help()
            return 1
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
