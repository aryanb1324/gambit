#!/usr/bin/env python3
"""CLI: self-play reinforcement learning.

Usage
-----
    python scripts/train_selfplay.py \\
        --checkpoint checkpoints/supervised_final.pt \\
        --output checkpoints/ \\
        --iterations 100
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gambit.train.selfplay import train_selfplay


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-play RL training for GambitNet"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Starting checkpoint (supervised_final.pt or previous selfplay ckpt)"
    )
    parser.add_argument(
        "--output", default="checkpoints/",
        help="Directory to save self-play checkpoints"
    )
    parser.add_argument("--iterations",    type=int, default=100)
    parser.add_argument("--games",         type=int, default=25,
                        help="Games to generate per iteration")
    parser.add_argument("--train-steps",   type=int, default=200,
                        help="Training steps per iteration")
    parser.add_argument("--batch-size",    type=int, default=512)
    parser.add_argument("--simulations",   type=int, default=400,
                        help="MCTS simulations per move during self-play")
    parser.add_argument("--device",        default=None)

    args = parser.parse_args()

    kwargs = dict(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        num_iterations=args.iterations,
        games_per_iter=args.games,
        train_steps_per_iter=args.train_steps,
        batch_size=args.batch_size,
        num_simulations=args.simulations,
    )
    if args.device:
        kwargs["device"] = args.device

    train_selfplay(**kwargs)


if __name__ == "__main__":
    main()
