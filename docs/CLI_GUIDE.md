# ♠️ Poker Equity CLI User Guide & Reference Manual

The `poker-equity` command-line interface provides high-performance Monte Carlo simulation, combinatorial hand analysis, pot odds math, GTO decision modeling, interactive table tracking, and terminal visualizations directly in your shell.

---

## Table of Contents

- [1. Installation & Execution](#1-installation--execution)
- [2. Global Options](#2-global-options)
- [3. Card, Hand & Range Syntax](#3-card-hand--range-syntax)
- [4. Subcommand Reference](#4-subcommand-reference)
  - [4.1 `equity` — Monte Carlo Equity Calculator](#41-equity--monte-carlo-equity-calculator)
  - [4.2 `range` — Range vs Range Equity Engine](#42-range--range-vs-range-equity-engine)
  - [4.3 `odds` — Pot Odds, EV & Implied Odds](#43-odds--pot-odds-ev--implied-odds)
  - [4.4 `outs` — Drawing Outs & Rule of 2/4](#44-outs--drawing-outs--rule-of-24)
  - [4.5 `decision` — Strategic Decision Engine](#45-decision--strategic-decision-engine)
  - [4.6 `table` — Interactive Table Simulator](#46-table--interactive-table-simulator)
  - [4.7 `twice` — Run-It-Twice & Variance Reduction](#47-twice--run-it-twice--variance-reduction)
  - [4.8 `curve` — Street-by-Street Equity Trajectory](#48-curve--street-by-street-equity-trajectory)
  - [4.9 `matrix` — 13×13 Range Grid Matrix Visualizer](#49-matrix--1313-range-grid-matrix-visualizer)
- [5. Terminal Visualizations & Display Modes](#5-terminal-visualizations--display-modes)
- [6. JSON Purity & Shell Automation](#6-json-purity--shell-automation)
- [7. Complete Real-World Analysis Pipelines](#7-complete-real-world-analysis-pipelines)

---

## 1. Installation & Execution

### Binary Invocation
Once installed via `pip install -e .` or binary package:

```bash
poker-equity <command> [options]
```

### Direct Python Module Invocation
You can execute the CLI without installation using Python's `-m` flag:

```bash
python -m poker_equity <command> [options]
```

---

## 2. Global Options

The following flags apply across all subcommands:

| Flag | Short | Description |
|------|-------|-------------|
| `--json` | - | Output structured JSON to `stdout` (guaranteed clean, pure JSON; logs and progress go to `stderr`). |
| `--no-color` | - | Disable ANSI terminal escape codes and colored card suits (also respects `NO_COLOR=1` environment variable). |
| `--help` | `-h` | Display command-specific help, argument signatures, and usage examples. |
| `--version` | `-V` | Print current library version and exit. |

---

## 3. Card, Hand & Range Syntax

### Card Syntax
Cards are written as 2 characters: `<Rank><Suit>`:
- **Ranks**: `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `T` (or `10`), `J`, `Q`, `K`, `A` (case-insensitive).
- **Suits**: `c` (♣ Clubs), `d` (♦ Diamonds), `h` (♥ Hearts), `s` (♠ Spades).
- *Examples*: `Ah`, `Kd`, `2c`, `Ts`.

### Hole Cards & Boards
- **Hole Cards**: 2 cards, separated by spaces or contiguous: `"Ah Kh"`, `"AhKh"`.
- **Board Cards**: 3 cards (flop), 4 cards (turn), or 5 cards (river): `"2h 7h 8c"`, `"2h7h8c3d4s"`.
- **Dead Cards**: Cards known to be folded or removed from the deck: `-d "9h Th"`.

### Range Syntax
Ranges define sets of starting combinations:
- **Pocket Pairs**:
  - Exact pair: `AA`, `KK`, `22` (6 combos each).
  - Plus notation: `TT+` (TT, JJ, QQ, KK, AA — 30 combos).
  - Dash interval: `99-77` (99, 88, 77 — 18 combos).
- **Suited Hands**:
  - Exact suited combo: `AKs`, `T9s` (4 combos each).
  - Plus notation: `AJs+` (AJs, AQs, AKs — 12 combos).
  - Dash interval: `T9s-76s` (T9s, 98s, 87s, 76s — 16 combos).
- **Offsuit Hands**:
  - Exact offsuit combo: `AKo`, `QJo` (12 combos each).
  - Plus notation: `AQo+` (AQo, AKo — 24 combos).
  - Dash interval: `JTo-87o` (JTo, T9o, 98o, 87o — 48 combos).
- **Unpaired General**:
  - Suited and offsuit combined: `AK` (16 combos), `QJ+` (QJ, KQ, AQ, AK — 64 combos).
- **Weighted Combos**:
  - Specify weights between 0.0 and 1.0 using `:`: `AA:0.8, KK:0.5, AKs:1.0`.
- **Percentile Ranges**:
  - Top percentage of hands: `top10%`, `top15%`, `25%`.
- **Pre-Built Named Presets**:
  - Position opening ranges: `UTG_OPEN`, `HJ_OPEN`, `CO_OPEN`, `BTN_OPEN`, `SB_OPEN`.
  - 3-Bet & Defense ranges: `BB_VS_BTN_DEFENSE`, `BB_VS_BTN_3BET`, `BB_VS_BTN_CALL`, `BTN_VS_CO_3BET`.
  - Hand archetypes: `PREMIUM`, `BROADWAYS`, `POCKET_PAIRS`, `SUITED_CONNECTORS`, `SUITED_ACES`.

---

## 4. Subcommand Reference

### 4.1 `equity` — Monte Carlo Equity Calculator

Simulates win, tie, and loss probabilities between two or more players with optional community board cards, dead cards, parallel processing, and early convergence stopping.

#### Syntax
```bash
poker-equity equity [hands...] [-b BOARD] [-d DEAD] [-n SIMS] [-s SEED] [-c CONVERGE] [-j WORKERS] [--progress] [--json]
```

#### Key Arguments
- `hands`: Positional player hands (e.g. `"Ah Kh"` `"Qc Qd"`).
- `--hero` / `-H`, `--villain` / `-V`: Alternative named arguments for heads-up spots.
- `--players` / `-p`: Multiway player hands (e.g. `-p "Ah As" "Kh Ks" "Qh Qs"`).
- `--board` / `-b`: Community cards (3, 4, or 5 cards).
- `--dead` / `-d`: Folded or burnt cards excluded from the remaining deck.
- `--sims` / `-n`: Number of Monte Carlo simulations (default: 10,000).
- `--seed` / `-s`: Fixed RNG seed for deterministic reproduction.
- `--converge` / `-c`: Stop simulation early once 95% confidence interval half-width drops below this threshold (e.g. `0.005` for $\pm 0.5\%$).
- `--parallel` / `-j`: Number of parallel worker processes.
- `--progress`: Display real-time progress indicator on stderr.

#### Example
```bash
poker-equity equity "Ah Kh" "Qc Qd" -b "2h 7h 8c" -n 10000 -s 42 --no-color
```

#### Output
```text
Poker Monte Carlo Equity Simulation
===============================================================================
Board : 2♥ 7♥ 8♣
Sims  : 10,000  |  Time: 1.58 s  |  Speed: 6,323 sims/sec
-------------------------------------------------------------------------------
┌─────────┬────────┬─────────────────┬──────────┬─────────┬─────────┬──────────────┐
│ Player  │ Hand   │ Type            │   Equity │   Win % │   Tie % │  Visual Bar  │
├─────────┼────────┼─────────────────┼──────────┼─────────┼─────────┼──────────────┤
│ Hero    │ A♥ K♥  │ Ace-high        │   54.61% │   54.6% │    0.0% │ [█████░░░░░] │
│ Villain │ Q♣ Q♦  │ Pair of Queens  │   45.39% │   45.4% │    0.0% │ [█████░░░░░] │
└─────────┴────────┴─────────────────┴──────────┴─────────┴─────────┴──────────────┘
  Hero   : Equity 54.61% ± 0.98% (95% CI) [Win: 54.6%, Tie: 0.0%, Loss: 45.4%]
  Villain: Equity 45.39% ± 0.98% (95% CI) [Win: 45.4%, Tie: 0.0%, Loss: 54.6%]
```

---

### 4.2 `range` — Range vs Range Equity Engine

Calculates equity when one or both players hold an entire range of hands. Accurately models card removal, blockers, and combo weighting.

#### Syntax
```bash
poker-equity range <hero_range> <villain_range> [-b BOARD] [-d DEAD] [-n SIMS] [-k BREAKDOWN] [--matrix] [--json]
```

#### Key Arguments
- `hero_range`, `villain_range`: Range strings, hand notations, percentage shorthands (`top15%`), or preset names (`BTN_OPEN`).
- `--breakdown` / `-k`: Show detailed per-combo equity and win rate breakdown for top $N$ combos in range.
- `--matrix` / `--grid`: Display the 13×13 range matrix in addition to equity calculations.
- `--board` / `-b`, `--dead` / `-d`: Community cards and dead cards.

#### Example
```bash
poker-equity range "Ah Kd" "TT+, AJs+, KQs" -b "Kh 7d 2c" -n 2000 -s 42 --breakdown 5 --no-color
```

#### Output
```text
Range vs Range Equity Simulation
===============================================================================
Board : K♥ 7♦ 2♣
Sims  : 2,000  |  Time: 495.4 ms  |  Speed: 4,037 sims/sec
-------------------------------------------------------------------------------
Hero   : A♥ K♦ (Pair of Kings)
Villain: TT+, AJs+, KQs (46 combos, 32 live [2.4%])
-------------------------------------------------------------------------------
┌─────────┬────────────────────────┬──────────┬─────────┬─────────┬──────────────┐
│ Player  │ Range / Hand           │   Equity │   Win % │   Tie % │  Visual Bar  │
├─────────┼────────────────────────┼──────────┼─────────┼─────────┼──────────────┤
│ Hero    │ A♥ K♦                  │   78.72% │   75.4% │    6.7% │ [████████░░] │
│ Villain │ TT+, AJs+, KQs         │   21.27% │   17.9% │    6.7% │ [██░░░░░░░░] │
└─────────┴────────────────────────┴──────────┴─────────┴─────────┴──────────────┘
  Hero   : Equity 78.72% ± 1.79% (95% CI) [Win: 75.4%, Tie: 6.7%, Loss: 17.9%]
  Villain: Equity 21.27% ± 1.79% (95% CI) [Win: 17.9%, Tie: 6.7%, Loss: 75.4%]

Top 5 Combos in Villain Range
-----------------------------------------------------------------
Combo    Cell   Equity     Win %      Tie %      Weight  
-----------------------------------------------------------------
K♠ K♣    KK     100.0%     100.0%     0.0%     1.00
A♦ A♣    AA     95.0%      95.0%     0.0%     1.00
A♠ A♦    AA     91.5%      91.5%     0.0%     1.00
A♠ A♣    AA     90.0%      90.0%     0.0%     1.00
A♣ K♣    AKs    53.0%      6.0%     94.0%     1.00
-----------------------------------------------------------------
```

---

### 4.3 `odds` — Pot Odds, EV & Implied Odds

Computes the mathematical break-even threshold to call, the expected value (EV) in chips, direct vs future EV, and the required future bets (implied odds).

#### Syntax
```bash
poker-equity odds <pot> <call> [<equity>] [--implied CHIPS] [--reverse-implied CHIPS] [-i] [--json]
```

#### Key Arguments
- `pot`, `call`: Current pot size and cost to call.
- `--equity` / `-e`: Hero equity as percentage (`35%`, `35`) or fraction (`0.35`).
- `--implied` / `--future-bets` / `-f`: Additional chips expected to be won on future streets if Hero hits their draw.
- `--reverse-implied` / `--future-losses` / `-l`: Chips Hero risks losing on later streets if outdrawn.
- `--interactive` / `-i`: Interactive prompting mode for all inputs.

#### Example
```bash
poker-equity odds 100 50 --equity 0.35 --implied 60 --no-color
```

#### Output
```text
Pot Odds & EV Analysis
================================================
Pot Size        : 100.00 chips
Cost to Call    : 50.00 chips
Total Pot       : 150.00 chips
Pot Odds        : 33.33% (2.00:1)
Future Bets     : +60.00 chips anticipated
Implied Odds    : 23.81% (3.20:1)
------------------------------------------------
Hero Equity     : 35.00%
Expected Value  : +23.50 chips
Direct EV       : +2.50 chips
Req Implied Bets: 0.00 chips (Direct pot odds sufficient)
Recommendation  : CALL
```

---

### 4.4 `outs` — Drawing Outs & Rule of 2/4

Calculates unseen deck cards that improve Hero's hand to win against Villain, discounting cards that also improve Villain. Displays exact probabilities alongside the Rule of 2 and Rule of 4.

#### Syntax
```bash
poker-equity outs <hero> <villain> <board> [-d DEAD] [--method exact|rule|adjusted_rule] [-i] [--json]
```

#### Key Arguments
- `hero`, `villain`: Player hole cards (2 cards each).
- `board`: Community cards (flop with 3 cards or turn with 4 cards).
- `--dead` / `-d`: Dead cards excluded from outs calculations.
- `--interactive` / `-i`: Interactive prompting mode.

#### Example
```bash
poker-equity outs "Ah Kh" "Qc Qd" "2h 7h 8c" --no-color
```

#### Output
```text
Drawing Outs Calculation
=======================================================
Hero   : Ah Kh (A♥ K♥) - Ace-high
Villain: Qc Qd (Q♣ Q♦) - Pair of Queens
Board  : 2h 7h 8c (2♥ 7♥ 8♣) [FLOP, 2 to come]
Status : Hero is currently TRAILING
-------------------------------------------------------
Total Outs : 15.0 (15 win, 0 tie)
Equity (exact): 55.07%
Outs Cards : As Ad Ac Ks Kd Kc Qh Jh Th 9h 8h 6h...
-------------------------------------------------------
Equity Projections:
  Exact (Turn or River) : 56.06%
  Exact (Turn only)     : 33.33%
  Rule of 4 (Standard)  :  60.0%
  Rule of 4 (Adjusted)  :  53.0%
-------------------------------------------------------
Winning Outs by Made Hand:
  Flush            ( 9 outs): Q♥ J♥ T♥ 9♥ 8♥ 6♥ 5♥ 4♥ 3♥
  One Pair         ( 6 outs): A♠ A♦ A♣ K♠ K♦ K♣
```

---

### 4.5 `decision` — Strategic Decision Engine

Recommends the game-theoretically optimal action (`FOLD`, `CHECK`, `CALL`, `RAISE`, `ALL_IN`) by comparing expected values across all legal options.

#### Syntax
```bash
poker-equity decision --hero <hero> --villain <villain> --board <board> --pot <pot> --call <call> [options]
# Or direct mathematical mode:
poker-equity decision <equity> <pot> [<call>] [<stack>] [options]
```

#### Key Arguments
- `--equity` / `-e`: Hero equity (bypasses simulation).
- `--hero` / `-H`, `--villain` / `-V`, `--board` / `-b`: Hole cards and board for live simulation.
- `--pot` / `-p`: Total pot size.
- `--call` / `-c`: Bet to call (`0` for checking spots).
- `--stack` / `-k`: Hero's remaining chip stack.
- `--fold-equity` / `-fe`: Probability Villain folds when Hero raises ($0.0$ to $1.0$).
- `--raise-size` / `-r`: Test raise size for EV evaluation.

#### Example
```bash
poker-equity decision --hero "Ah Kh" --villain "Qc Qd" --board "2h 7h 8c" --pot 100 --call 50 -n 2000 -s 42 --no-color
```

#### Output
```text
Strategic Decision Recommendation
================================================================
Recommended Action: CALL
Expected Value    : +28.45 chips
Hero Equity       : 52.3%
Pot Odds          : 33.3%
Pot Size          : 100.00 chips (Cost to Call: 50.00)
Hero Stack        : 42.00 chips (SPR: 0.42)
Hole Cards        : Hero A♥ K♥ vs Villain Q♣ Q♦ | Board: 2♥ 7♥ 8♣
----------------------------------------------------------------
Action Expected Value (EV) Comparison:
Action     EV (chips)     Delta vs Best    Status          
----------------------------------------------------------------
CALL           +28.45 chips     0.00           ★ Recommended
FOLD            +0.00 chips   -28.45           Suboptimal
----------------------------------------------------------------
Analysis: All-in call is profitable (+28.45 EV). Hero equity (52.3%) meets/exceeds pot odds requirement (33.3%).
```

---

### 4.6 `table` — Interactive Table Simulator

Full poker game state engine modeling betting rounds, blinds, player stacks, side pots, community card dealing, live equity recommendations, and hand histories.

#### Syntax
```bash
poker-equity table [--blinds SB/BB] [--hero-cards CARDS] [--villain-cards CARDS] [--actions ACTIONS] [-i] [--recommend] [--json]
```

#### Key Arguments
- `--blinds`: Blinds string (e.g. `1/2` or `5/10`, default `1/2`).
- `--hero-stack`, `--villain-stack`: Starting chip stacks (default 100 BBs).
- `--hero-cards`, `--villain-cards`: Hole cards.
- `--actions`: Comma-separated pre-scripted action tokens (e.g. `"call,check,bet 10,call"`).
- `--interactive` / `-i`: Launch interactive REPL loop.
- `--recommend`: Enable live Monte Carlo decision support and equity calculation at each action.

#### Interactive REPL Meta-Commands
When running with `-i`:
- `call`, `check`, `fold`, `bet <amt>`, `raise <amt>`, `all_in`: Perform betting action.
- `recommend`: Run live equity and GTO recommendation for current player.
- `state`: Print current board, pot size, player stacks, and active bets.
- `history`: Display complete hand history of all actions taken so far.
- `quit` / `exit`: Exit the simulation.

#### Example (Pre-Scripted Actions)
```bash
poker-equity table --hero-cards "Ah Kh" --villain-cards "Qc Qd" --actions "call,check,bet 10,call" --seed 42 --no-color
```

---

### 4.7 `twice` — Run-It-Twice & Variance Reduction

Simulates Run-It-Twice (or Run-It-$N$-Times) for all-in scenarios. Computes scoop probabilities, split pot rates, expected chip payouts, and risk/variance reduction percentages. Also supports exact combinatorial enumeration on turn boards.

#### Syntax
```bash
poker-equity twice <hero> <villain> [-b BOARD] [--runs N] [--pot POT] [--exact] [-n SIMS] [-s SEED] [--json]
```
*(Aliases: `poker-equity rit`, `poker-equity run-it-twice`)*

#### Key Arguments
- `hero`, `villain`: Hole cards for both players.
- `--board` / `-b`: Flop (3 cards) or turn (4 cards).
- `--runs`: Number of board runouts (default: 2).
- `--pot`: Total pot size in chips for expected payout calculation.
- `--exact`: Perform exact combinatorial enumeration over all $\binom{44}{2} = 946$ remaining card pairs (turn boards only).

#### Example
```bash
poker-equity twice "Ah Kh" "Qc Qd" -b "2h 7h 8c 3d" --exact --pot 150 --no-color
```

#### Output
```text
=== Run It 2 Times Analysis (Exact Enumeration) ===
Board: [2h 7h 8c 3d] (Turn)
===================================================
Overall Equities:
  Hero (Player 1)   :  34.09%  [███████░░░░░░░░░░░░░░░]
  Villain (Player 2):  65.91%  [██████████████░░░░░░░░]

Multi-Board Outcomes (2 Boards):
  Hero scoops all 2           :  11.10%
  Villain scoops all 2        :  42.92%
  Split pot (split boards)    :  45.98%

Expected Payouts:
  Total Pot Size             : 150.00 chips
  Hero expected payout        : 51.14 chips (34.1%)
  Villain expected payout     : 98.86 chips (65.9%)

Variance Reduction:
  Single-board variance      : 0.2299
  Run It 2 variance         : 0.1098
  Variance reduction         : 52.22% (Risk reduction)

Per-Board Breakdown:
  Board 1: P1: 34.1%, P2: 65.9%
  Board 2: P1: 34.1%, P2: 65.9%
```

---

### 4.8 `curve` — Street-by-Street Equity Trajectory

Tracks equity transitions across Preflop $\to$ Flop $\to$ Turn $\to$ River. Detects turning points, lead changes, bad beats, and generates terminal ASCII sparkline charts.

#### Syntax
```bash
poker-equity curve <hero> <villain> -b <board> [--chart] [--next] [-s SEED] [--json]
```

#### Key Arguments
- `hero`, `villain`: Hole cards for both players.
- `--board` / `-b`: Full 5-card board (or 3-4 cards).
- `--chart` / `--ascii`: Display ASCII sparkline chart and street-by-street visual bars.
- `--next`: Analyze equity distribution across all possible next-street cards (Flop $\to$ Turn or Turn $\to$ River) categorizing cards into Favorable, Unfavorable, and Neutral.

#### Example
```bash
poker-equity curve "Ah Kh" "Qc Qd" -b "2h 7h 8c 3d 4s" -s 42 --chart --no-color
```

#### Output
```text
Equity Curve Analysis
=====================
Street   Board          Hero (P1)  P1 Hand              Villain (P2)  P2 Hand              Hero Δ   Leader
----------------------------------------------------------------------------------------------------------
PREFLOP  -              46.2%      Hole cards           53.8%         Hole cards           -        Villain
FLOP     2h 7h 8c       53.2%      Ace-high             46.8%         Pair of Queens       +7.0%    Hero
TURN     2h 7h 8c 3d    33.2%      Ace-high             66.8%         Pair of Queens       -20.0%   Villain
RIVER    2h 7h 8c 3d 4s 0.0%       Ace-high             100.0%        Pair of Queens       -33.2%   Villain
----------------------------------------------------------------------------------------------------------
Lead Changes: 2 | Max Swing: 33.2% | Turning Point: RIVER (-33.2%) | Bad Beat: False

Street-by-Street Equity Trajectory:
====================================================================
  [PREFLOP] Board: [None          ]
    Hero   :  46.2% [█████████░░░░░░░░░░░]  [Hole cards]
    Villain:  53.8% [███████████░░░░░░░░░]  [Hole cards]

  [FLOP   ] Board: [2h 7h 8c      ] (Hero Δ: +7.0%)
    Hero   :  53.2% [███████████░░░░░░░░░]  [Ace-high]
    Villain:  46.8% [█████████░░░░░░░░░░░]  [Pair of Queens]

  [TURN   ] Board: [2h 7h 8c 3d   ] (Hero Δ: -20.0%)
    Hero   :  33.2% [███████░░░░░░░░░░░░░]  [Ace-high]
    Villain:  66.8% [█████████████░░░░░░░]  [Pair of Queens]

  [RIVER  ] Board: [2h 7h 8c 3d 4s] (Hero Δ: -33.2%) [TURNING POINT]
    Hero   :   0.0% [░░░░░░░░░░░░░░░░░░░░]  [Ace-high]
    Villain: 100.0% [████████████████████]  [Pair of Queens]

Equity Spark Graph (Hero):
  100% |                                         
   80% |                                         
   60% |    ●46%      ●53%                       
   40% |                        ●33%             
   20% |                                  ●0%    
       +--------------------------------------------
       PREFLOP   FLOP      TURN      RIVER     
```

---

### 4.9 `matrix` — 13×13 Range Grid Matrix Visualizer

Visualizes Texas Hold'em starting hand ranges in a canonical 13×13 matrix (Pocket pairs on diagonal, suited hands top-right, offsuit bottom-left).

#### Syntax
```bash
poker-equity matrix [ranges...] [-b BOARD] [-d DEAD] [--mode hands|weights|combos|heat] [--show-all] [--json]
```
*(Aliases: `poker-equity grid`, `poker-equity matrix13`)*

#### Key Arguments
- `ranges`: 1 range to visualize, or 2 ranges (`hero` and `villain`) to compare.
- `--mode` / `-m`: Cell display format:
  - `hands`: Hand label (e.g. `AA`, `AKs`, `AKo`) or dot for excluded.
  - `weights`: Percentage weight integer (e.g. `100`, `50`, `25`).
  - `combos`: Active combo count after blockers (e.g. `6`, `3`, `12`).
  - `heat`: ANSI color block shading representing density (`██`, `▓▓`, `▒▒`, `░░`).
- `--show-all` / `-a`: Show dim labels for excluded hands instead of dots.
- `--board` / `-b`, `--dead` / `-d`: Cards treated as blockers, automatically deducting combos and marking partial weights (`*`).

#### Example
```bash
poker-equity matrix "QQ+, AKs" --board "Ah Kd 2s" --no-color
```

#### Output
```text
Range Matrix: QQ+, AKs
      A    K    Q    J    T    9    8    7    6    5    4    3    2  
   ┌──────────────────────────────────────────────────────────────────┐
 A │ AA*  AKs*  .    .    .    .    .    .    .    .    .    .    .   │
 K │  .   KK*   .    .    .    .    .    .    .    .    .    .    .   │
 Q │  .    .    QQ   .    .    .    .    .    .    .    .    .    .   │
 J │  .    .    .    .    .    .    .    .    .    .    .    .    .   │
 T │  .    .    .    .    .    .    .    .    .    .    .    .    .   │
 9 │  .    .    .    .    .    .    .    .    .    .    .    .    .   │
 8 │  .    .    .    .    .    .    .    .    .    .    .    .    .   │
 7 │  .    .    .    .    .    .    .    .    .    .    .    .    .   │
 6 │  .    .    .    .    .    .    .    .    .    .    .    .    .   │
 5 │  .    .    .    .    .    .    .    .    .    .    .    .    .   │
 4 │  .    .    .    .    .    .    .    .    .    .    .    .    .   │
 3 │  .    .    .    .    .    .    .    .    .    .    .    .    .   │
 2 │  .    .    .    .    .    .    .    .    .    .    .    .    .   │
   └──────────────────────────────────────────────────────────────────┘

  Range Summary: 14.0 / 1326 combos (1.06%) across 4 / 169 hands
    Pairs:    12.0 / 78 combos ( 15.4%) in  3/13 hands
    Suited:    2.0 / 312 combos (  0.6%) in  1/78 hands
    Offsuit:   0.0 / 936 combos (  0.0%) in  0/78 hands
    Blockers: [A♥ K♦ 2♠] (8.0 combos removed by blockers)
  Legend: Hands in range shown; [*] Partial weight; [.] Excluded
```

---

## 5. Terminal Visualizations & Display Modes

The CLI includes built-in ANSI color-coded visual formatters:
- **Suit Symbols & Colors**:
  - Spades (♠): Bright Cyan / White
  - Hearts (♥): Bright Red
  - Diamonds (♦): Bright Blue
  - Clubs (♣): Bright Green
- **Visual Equity Bars**: High-density horizontal percentage bars: `[████████░░░░░░░░░░]`.
- **Confidence Intervals**: Explicit 95% Wald confidence intervals: `54.61% ± 0.98%`.
- **Comparison Matrix**: When comparing two ranges, cells display:
  - `=` Overlap (in both ranges)
  - `+` Hero Only
  - `-` Villain Only
  - `.` Neither

---

## 6. JSON Purity & Shell Automation

Every subcommand accepts `--json`. The CLI guarantees **JSON purity**:
- Only strictly valid JSON is written to `stdout`.
- All progress bars, diagnostics, and warnings are directed to `stderr`.
- You can safely pipe CLI output directly to `jq`, Python, or shell scripts.

### Examples

#### Extracting Hero Equity with `jq`
```bash
poker-equity equity "Ah Kh" "Qc Qd" --json | jq '.players[0].equity'
# Output: 0.4619
```

#### Filtering Overlapping Combos from Range Comparison
```bash
poker-equity matrix "btn_open" "bb_vs_btn_defense" --json | jq '.comparison.overlap_combos'
# Output: 674.0
```

#### Automating Facing-Bet Decision in Bash
```bash
RECOMMENDED=$(poker-equity decision --pot 120 --call 40 --equity 0.35 --json | jq -r '.recommended_action')
echo "Action to take: $RECOMMENDED"
# Output: Action to take: call
```

---

## 7. Complete Real-World Analysis Pipelines

### Pipeline 1: Postflop Hand Review
Analyze a complete hand from preflop to showdown:

```bash
# Step 1: Preflop equity
poker-equity equity "Ah Kh" "Qc Qd" -n 50000

# Step 2: Flop outs evaluation after 2h 7h 8c
poker-equity outs "Ah Kh" "Qc Qd" "2h 7h 8c"

# Step 3: Pot odds facing a $50 bet into $100 pot
poker-equity odds 100 50 --equity 0.54 --implied 80

# Step 4: Decision recommendation
poker-equity decision --hero "Ah Kh" --villain "Qc Qd" --board "2h 7h 8c" --pot 100 --call 50

# Step 5: Run-it-twice payout resolution when players get all-in on turn
poker-equity twice "Ah Kh" "Qc Qd" -b "2h 7h 8c 3d" --exact --pot 300
```

### Pipeline 2: Preflop GTO Defense Strategy
Construct and evaluate opening vs defending ranges:

```bash
# Step 1: Inspect BTN opening range
poker-equity matrix "btn_open" --mode combos

# Step 2: Compare BTN open vs BB defend range
poker-equity matrix "btn_open" "bb_vs_btn_defense"

# Step 3: Simulate range equity on wet flop
poker-equity range "btn_open" "bb_vs_btn_defense" -b "Jh Th 9c" -n 10000 --breakdown 10
```
