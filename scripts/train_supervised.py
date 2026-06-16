#!/usr/bin/env python3
"""CLI: supervised pretraining.

Usage
-----
    python scripts/train_supervised.py \\
        --data data/lichess_2024-01.h5 \\
        --epochs 10 \\
        --checkpoint-dir checkpoints/
"""

import argparse
import sys
import os

# Allow running from repo root without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gambit.train.supervised import train_supervised


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GambitNet on supervised Lichess data"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to HDF5 training file (output of parse_pgn_to_tensors)"
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints/",
        help="Directory to save checkpoints (default: checkpoints/)"
    )
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--batch-size",    type=int,   default=512)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight-decay",  type=float, default=1e-4)
    parser.add_argument("--res-blocks",    type=int,   default=10)
    parser.add_argument("--filters",       type=int,   default=128)
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from an existing checkpoint .pt file"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (cpu / cuda). Auto-detected if omitted."
    )

    args = parser.parse_args()

    kwargs = dict(
        data_path=args.data,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_res_blocks=args.res_blocks,
        num_filters=args.filters,
        resume_from=args.resume,
    )
    if args.device:
        kwargs["device"] = args.device

    train_supervised(**kwargs)


if __name__ == "__main__":
    main()
