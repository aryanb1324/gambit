"""Arena: Gambit vs Stockfish evaluation.

Plays a series of games between Gambit (MCTS + GambitNet) and Stockfish at a
configurable ELO level, returning win/draw/loss tallies and a rough Elo
estimate for Gambit.

Example
-------
>>> from gambit.arena.vs_stockfish import run_arena
>>> from gambit.network.resnet import GambitNet
>>> from gambit.mcts.mcts import MCTS
>>> model = GambitNet(); mcts = MCTS(model)
>>> results = run_arena(model, mcts, num_games=20, stockfish_elo=2000)
>>> print(results)
"""

from __future__ import annotations

import math
from typing import Optional

import chess
import chess.engine
from tqdm import tqdm

from gambit.mcts.mcts import MCTS
from gambit.network.resnet import GambitNet


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------


def play_game_vs_stockfish(
    model:           GambitNet,
    mcts:            MCTS,
    stockfish_path:  str         = "stockfish",
    stockfish_elo:   int         = 2000,
    gambit_color:    chess.Color = chess.WHITE,
    time_limit:      float       = 1.0,
    num_simulations: int         = 800,
) -> str:
    """Play one full game between Gambit and Stockfish.

    Parameters
    ----------
    model:
        GambitNet (weights already loaded).
    mcts:
        Configured MCTS instance.
    stockfish_path:
        Path to the Stockfish binary.
    stockfish_elo:
        Stockfish skill level (UCI_LimitStrength / UCI_Elo).
    gambit_color:
        The color Gambit plays.
    time_limit:
        Time Stockfish is allowed per move (seconds).
    num_simulations:
        MCTS simulations for Gambit's moves.

    Returns
    -------
    str
        ``"win"`` / ``"draw"`` / ``"loss"`` from Gambit's perspective.
    """
    board = chess.Board()

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as sf:
        # Set Stockfish ELO
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})

        while not board.is_game_over():
            if board.turn == gambit_color:
                # Gambit's move
                mcts.num_simulations = num_simulations
                move = mcts.select_move(board, temperature=0)
            else:
                # Stockfish's move
                result = sf.play(board, chess.engine.Limit(time=time_limit))
                move   = result.move

            board.push(move)

    return _board_result_to_outcome(board.result(), gambit_color)


# ---------------------------------------------------------------------------
# Arena (multiple games)
# ---------------------------------------------------------------------------


def run_arena(
    model:          GambitNet,
    mcts:           MCTS,
    num_games:      int   = 20,
    stockfish_path: str   = "stockfish",
    stockfish_elo:  int   = 2000,
    **kwargs,
) -> dict:
    """Play *num_games* against Stockfish (half as White, half as Black).

    Parameters
    ----------
    model:
        GambitNet.
    mcts:
        MCTS instance.
    num_games:
        Total number of games (rounded down to nearest even).
    stockfish_path:
        Path to the Stockfish binary.
    stockfish_elo:
        Stockfish ELO limit.
    **kwargs:
        Forwarded to :func:`play_game_vs_stockfish`.

    Returns
    -------
    dict
        ``{"wins": int, "draws": int, "losses": int, "win_rate": float, "elo_estimate": float}``
    """
    wins   = 0
    draws  = 0
    losses = 0

    half    = num_games // 2
    colors  = [chess.WHITE] * half + [chess.BLACK] * (num_games - half)

    for color in tqdm(colors, desc="arena games"):
        result = play_game_vs_stockfish(
            model, mcts,
            stockfish_path=stockfish_path,
            stockfish_elo=stockfish_elo,
            gambit_color=color,
            **kwargs,
        )
        if result == "win":
            wins   += 1
        elif result == "draw":
            draws  += 1
        else:
            losses += 1

    total    = wins + draws + losses
    win_rate = (wins + 0.5 * draws) / max(total, 1)
    elo_est  = estimate_elo(win_rate, stockfish_elo)

    return {
        "wins":         wins,
        "draws":        draws,
        "losses":       losses,
        "win_rate":     round(win_rate, 4),
        "elo_estimate": round(elo_est, 1),
    }


# ---------------------------------------------------------------------------
# Elo helpers
# ---------------------------------------------------------------------------


def estimate_elo(win_rate: float, opponent_elo: int) -> float:
    """Estimate Gambit's Elo given *win_rate* against *opponent_elo*.

    Uses the standard Elo formula::

        E = opponent_elo + 400 × log10(win_rate / (1 − win_rate))

    Parameters
    ----------
    win_rate:
        Fraction of points scored (win=1, draw=0.5, loss=0) in (0, 1).
    opponent_elo:
        Known Elo of the opponent.

    Returns
    -------
    float
        Estimated Elo.
    """
    # Clamp to avoid log domain error
    win_rate = max(1e-6, min(1.0 - 1e-6, win_rate))
    return opponent_elo + 400 * math.log10(win_rate / (1.0 - win_rate))


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _board_result_to_outcome(result: str, gambit_color: chess.Color) -> str:
    if result == "1-0":
        return "win"  if gambit_color == chess.WHITE else "loss"
    if result == "0-1":
        return "win"  if gambit_color == chess.BLACK else "loss"
    return "draw"
