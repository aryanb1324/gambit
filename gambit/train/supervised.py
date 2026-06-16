from __future__ import annotations
"""Phase 1: supervised pretraining on Lichess game data.

Trains GambitNet to predict the move played in positions from strong games
(ELO ≥ 2200) and to estimate the game outcome.

Loss
----
    total_loss = cross_entropy(policy_logits, target_move)
               + mse_loss(value, target_value)

Training logs (every 100 steps)::

    step=100  policy_loss=5.423  value_loss=0.812  total=6.235

Checkpoints are saved every 1 000 steps and at the end of training.

Example
-------
>>> from gambit.train.supervised import train_supervised
>>> train_supervised(
...     data_path="data/lichess_2024-01.h5",
...     checkpoint_dir="checkpoints/",
...     num_epochs=10,
... )
"""


import os
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from gambit.data.parser import ChessDataset
from gambit.network.resnet import GambitNet, load_checkpoint, save_checkpoint

# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_supervised(
    data_path:      str,
    checkpoint_dir: str,
    num_epochs:     int   = 10,
    batch_size:     int   = 512,
    lr:             float = 1e-3,
    weight_decay:   float = 1e-4,
    num_res_blocks: int   = 10,
    num_filters:    int   = 128,
    resume_from:    Optional[str] = None,
    device:         str   = "cuda" if torch.cuda.is_available() else "cpu",
) -> GambitNet:
    """Train GambitNet on supervised data from *data_path*.

    Parameters
    ----------
    data_path:
        HDF5 file produced by :func:`~gambit.data.parser.parse_pgn_to_tensors`.
    checkpoint_dir:
        Directory to save checkpoints.
    num_epochs:
        Number of passes over the dataset.
    batch_size:
        Training batch size.
    lr:
        Learning rate for AdamW.
    weight_decay:
        L2 regularisation for AdamW.
    num_res_blocks:
        ResNet depth (ignored when resuming).
    num_filters:
        Number of convolutional filters (ignored when resuming).
    resume_from:
        Optional path to an existing checkpoint to resume from.
    device:
        Torch device string.

    Returns
    -------
    GambitNet
        The trained model.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Build or load model
    if resume_from:
        model, optimizer, start_step = load_checkpoint(resume_from, device=device)
        print(f"[supervised] Resuming from {resume_from} at step {start_step}")
    else:
        model     = GambitNet(num_res_blocks=num_res_blocks, num_filters=num_filters).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        start_step = 0

    dataset    = ChessDataset(data_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.startswith("cuda")),
        drop_last=True,
    )

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn  = nn.MSELoss()

    global_step = start_step

    for epoch in range(num_epochs):
        model.train()
        epoch_policy_loss = 0.0
        epoch_value_loss  = 0.0
        n_batches         = 0

        pbar = tqdm(dataloader, desc=f"epoch {epoch + 1}/{num_epochs}")
        for boards, target_moves, target_values in pbar:
            boards         = boards.to(device)
            target_moves   = target_moves.to(device)
            target_values  = target_values.to(device).unsqueeze(1)

            optimizer.zero_grad()
            policy_logits, value_pred = model(boards)

            ploss = policy_loss_fn(policy_logits, target_moves)
            vloss = value_loss_fn(value_pred, target_values)
            loss  = ploss + vloss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step       += 1
            epoch_policy_loss += ploss.item()
            epoch_value_loss  += vloss.item()
            n_batches         += 1

            # Log every 100 steps
            if global_step % 100 == 0:
                pbar.set_postfix({
                    "p_loss": f"{ploss.item():.3f}",
                    "v_loss": f"{vloss.item():.3f}",
                    "step":   global_step,
                })

            # Checkpoint every 1000 steps
            if global_step % 1000 == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"supervised_step{global_step}.pt")
                save_checkpoint(model, optimizer, global_step, ckpt_path)

        avg_p = epoch_policy_loss / max(n_batches, 1)
        avg_v = epoch_value_loss  / max(n_batches, 1)
        print(
            f"[epoch {epoch + 1}] avg_policy_loss={avg_p:.4f}  "
            f"avg_value_loss={avg_v:.4f}  steps={global_step}"
        )

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "supervised_final.pt")
    save_checkpoint(model, optimizer, global_step, final_path)
    print(f"[supervised] Training complete — saved to {final_path}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(
    model:       GambitNet,
    data_loader: torch.utils.data.DataLoader,
    device:      str,
) -> dict[str, float]:
    """Evaluate *model* on *data_loader*.

    Parameters
    ----------
    model:
        Model to evaluate.
    data_loader:
        DataLoader over a :class:`~gambit.data.parser.ChessDataset`.
    device:
        Torch device string.

    Returns
    -------
    dict[str, float]
        ``{"policy_acc": float, "value_mae": float}``
    """
    model.eval()
    correct = 0
    total   = 0
    mae_sum = 0.0

    for boards, target_moves, target_values in data_loader:
        boards        = boards.to(device)
        target_moves  = target_moves.to(device)
        target_values = target_values.to(device)

        policy_logits, value_pred = model(boards)

        preds    = policy_logits.argmax(dim=1)
        correct += (preds == target_moves).sum().item()
        total   += len(target_moves)
        mae_sum += (value_pred.squeeze(1) - target_values).abs().sum().item()

    policy_acc = correct / max(total, 1)
    value_mae  = mae_sum / max(total, 1)
    return {"policy_acc": policy_acc, "value_mae": value_mae}
