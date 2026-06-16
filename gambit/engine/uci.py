"""UCI protocol interface for Gambit.

Implements the Universal Chess Interface (UCI) so Gambit can be plugged into
any chess GUI or arena tool (Arena, Cute Chess, En Croissant, etc.).

Supported commands
------------------
uci          → identify engine, list options, "uciok"
isready      → "readyok"
ucinewgame   → reset board
position     → set board from FEN or startpos + optional move list
go           → start search, respond "bestmove <move>"
stop         → stop search (interrupt)
quit         → exit

Usage
-----
Run as a subprocess::

    python -m gambit.engine.uci --checkpoint checkpoints/selfplay_latest.pt

Or register with a GUI as a UCI engine pointing to this script.
"""

from __future__ import annotations

import argparse
import sys
import threading
from typing import Optional

import chess

from gambit.mcts.mcts import MCTS
from gambit.network.resnet import GambitNet, load_checkpoint

_DEFAULT_SIMULATIONS = 800


class UCIEngine:
    """Read UCI commands from stdin, write responses to stdout.

    Parameters
    ----------
    model:
        The :class:`~gambit.network.resnet.GambitNet` to use for search.
    mcts:
        Pre-configured :class:`~gambit.mcts.mcts.MCTS` instance.
    """

    def __init__(self, model: GambitNet, mcts: MCTS) -> None:
        self.model  = model
        self.mcts   = mcts
        self.board  = chess.Board()
        self._stop  = threading.Event()
        self._search_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Block on stdin, dispatch UCI commands until ``quit``."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            self._dispatch(line)

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, line: str) -> None:
        tokens = line.split()
        cmd    = tokens[0] if tokens else ""

        if cmd == "uci":
            self._cmd_uci()
        elif cmd == "isready":
            self._cmd_isready()
        elif cmd == "ucinewgame":
            self._cmd_ucinewgame()
        elif cmd == "position":
            self._cmd_position(tokens[1:])
        elif cmd == "go":
            self._cmd_go(tokens[1:])
        elif cmd == "stop":
            self._cmd_stop()
        elif cmd == "quit":
            sys.exit(0)
        # Unknown commands are silently ignored (UCI spec)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _cmd_uci(self) -> None:
        self._send("id name Gambit")
        self._send("id author Gambit Team")
        self._send("option name Simulations type spin default 800 min 1 max 100000")
        self._send("uciok")

    def _cmd_isready(self) -> None:
        self._send("readyok")

    def _cmd_ucinewgame(self) -> None:
        self.board = chess.Board()

    def _cmd_position(self, args: list[str]) -> None:
        """Parse ``position [startpos|fen <FEN>] [moves <move>...]``."""
        idx = 0
        if not args:
            return

        if args[idx] == "startpos":
            self.board = chess.Board()
            idx += 1
        elif args[idx] == "fen":
            idx += 1
            fen_parts = []
            while idx < len(args) and args[idx] != "moves":
                fen_parts.append(args[idx])
                idx += 1
            self.board = chess.Board(" ".join(fen_parts))

        if idx < len(args) and args[idx] == "moves":
            idx += 1
            while idx < len(args):
                move = chess.Move.from_uci(args[idx])
                self.board.push(move)
                idx += 1

    def _cmd_go(self, args: list[str]) -> None:
        """Start MCTS search in a background thread."""
        self._stop.clear()

        def search_and_respond() -> None:
            move = self.mcts.select_move(self.board, temperature=0)
            if not self._stop.is_set():
                self._send(f"bestmove {move.uci()}")

        self._search_thread = threading.Thread(target=search_and_respond, daemon=True)
        self._search_thread.start()

    def _cmd_stop(self) -> None:
        """Signal the search thread to stop and wait for it."""
        self._stop.set()
        if self._search_thread and self._search_thread.is_alive():
            self._search_thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # I/O helper
    # ------------------------------------------------------------------

    @staticmethod
    def _send(msg: str) -> None:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point: load checkpoint and start UCI loop."""
    parser = argparse.ArgumentParser(description="Gambit UCI engine")
    parser.add_argument(
        "--checkpoint", "-c",
        default=None,
        help="Path to a .pt checkpoint (uses random weights if omitted)",
    )
    parser.add_argument(
        "--simulations", "-n",
        type=int,
        default=_DEFAULT_SIMULATIONS,
        help="MCTS simulations per move",
    )
    parser.add_argument(
        "--device", "-d",
        default="cpu",
        help="Torch device (cpu / cuda)",
    )
    args = parser.parse_args(argv)

    if args.checkpoint:
        model, _, _ = load_checkpoint(args.checkpoint, device=args.device)
    else:
        model = GambitNet()
        model.eval()

    mcts   = MCTS(model=model, num_simulations=args.simulations, device=args.device)
    engine = UCIEngine(model=model, mcts=mcts)
    engine.run()


if __name__ == "__main__":
    main()
