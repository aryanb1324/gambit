#!/usr/bin/env python3
"""CLI: interactive play against Gambit (or Gambit vs Stockfish).

Usage
-----
    # Play against Gambit in the terminal (you are White)
    python scripts/play.py --checkpoint checkpoints/selfplay_latest.pt

    # Watch Gambit vs Stockfish at ELO 2000
    python scripts/play.py --checkpoint checkpoints/selfplay_latest.pt \\
        --vs-stockfish --elo 2000

Input format: standard UCI move strings (e.g. ``e2e4``, ``g1f3``, ``e7e8q``).
Type ``quit`` or Ctrl-C to exit.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import chess.engine

from gambit.mcts.mcts import MCTS
from gambit.network.resnet import GambitNet, load_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_board(board: chess.Board) -> None:
    print()
    print(board)
    print()


def human_move(board: chess.Board) -> chess.Move:
    while True:
        raw = input("Your move (UCI, e.g. e2e4): ").strip()
        if raw.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            sys.exit(0)
        try:
            move = chess.Move.from_uci(raw)
            if move in board.legal_moves:
                return move
            print(f"  Illegal move: {raw}")
        except ValueError:
            print(f"  Invalid move format: {raw}")


# ---------------------------------------------------------------------------
# Game modes
# ---------------------------------------------------------------------------

def play_vs_human(model: GambitNet, mcts: MCTS, simulations: int) -> None:
    """Interactive game: human (White) vs Gambit (Black)."""
    board = chess.Board()
    mcts.num_simulations = simulations

    print("Gambit Chess Engine — You are WHITE")
    print_board(board)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = human_move(board)
        else:
            print("Gambit is thinking...")
            move = mcts.select_move(board, temperature=0)
            print(f"Gambit plays: {move.uci()}")

        board.push(move)
        print_board(board)

    print(f"Game over! Result: {board.result()}")


def play_vs_stockfish(
    model:          GambitNet,
    mcts:           MCTS,
    stockfish_path: str,
    stockfish_elo:  int,
    simulations:    int,
) -> None:
    """Gambit (White) vs Stockfish — non-interactive, prints moves."""
    from gambit.arena.vs_stockfish import play_game_vs_stockfish

    mcts.num_simulations = simulations
    print(f"Gambit (White) vs Stockfish ELO {stockfish_elo}")
    result = play_game_vs_stockfish(
        model, mcts,
        stockfish_path=stockfish_path,
        stockfish_elo=stockfish_elo,
        gambit_color=chess.WHITE,
        num_simulations=simulations,
    )
    print(f"\nResult: Gambit {result.upper()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Play against Gambit")
    parser.add_argument(
        "--checkpoint", "-c",
        default=None,
        help="Path to .pt checkpoint (random weights if omitted)"
    )
    parser.add_argument(
        "--vs-stockfish", action="store_true",
        help="Watch Gambit play Stockfish instead of interactive mode"
    )
    parser.add_argument(
        "--stockfish-path", default="stockfish",
        help="Path to Stockfish binary"
    )
    parser.add_argument("--elo",         type=int, default=2000,
                        help="Stockfish ELO (used with --vs-stockfish)")
    parser.add_argument("--simulations", type=int, default=400,
                        help="MCTS simulations per move")
    parser.add_argument("--device",      default="cpu")

    args = parser.parse_args()

    if args.checkpoint:
        model, _, _ = load_checkpoint(args.checkpoint, device=args.device)
    else:
        print("[play] No checkpoint specified — using random weights")
        model = GambitNet()

    model.eval()
    mcts = MCTS(model=model, num_simulations=args.simulations, device=args.device)

    if args.vs_stockfish:
        play_vs_stockfish(model, mcts, args.stockfish_path, args.elo, args.simulations)
    else:
        play_vs_human(model, mcts, args.simulations)


if __name__ == "__main__":
    main()
