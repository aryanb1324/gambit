from __future__ import annotations
"""GambitNet — ResNet with policy and value heads.

Architecture
------------
Input  : (batch, 17, 8, 8)
Body   : Conv(num_filters, 3×3, pad=1) → BN → ReLU → N × ResBlock
Policy : Conv(2, 1×1) → BN → ReLU → flatten → Linear(128, action_size)
Value  : Conv(1, 1×1) → BN → ReLU → flatten → Linear(64, 64) → ReLU → Linear(64, 1) → Tanh
"""


import os
from typing import Optional

import chess
import numpy as np
import torch
import torch.nn as nn

from gambit.board.encoder import encode_board, get_legal_move_indices

ACTION_SIZE = 4096


class ResBlock(nn.Module):
    """A single residual block: two Conv3×3 layers with BN, ReLU, and a skip connection."""

    def __init__(self, num_filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(num_filters)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class GambitNet(nn.Module):
    """AlphaZero-style ResNet with a shared body and separate policy / value heads.

    Parameters
    ----------
    num_res_blocks:
        Number of residual blocks in the body (default 10).
    num_filters:
        Number of convolutional filters (default 128).
    action_size:
        Size of the policy head output (default 4096).
    """

    def __init__(
        self,
        num_res_blocks: int = 10,
        num_filters: int = 128,
        action_size: int = ACTION_SIZE,
    ) -> None:
        super().__init__()
        self.action_size = action_size

        # Initial convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(17, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.res_blocks = nn.Sequential(*[ResBlock(num_filters) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * 64, action_size)

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass.

        Parameters
        ----------
        x:\n            Board tensor of shape ``(batch, 17, 8, 8)``.\n\n        Returns\n        -------\n        tuple[torch.Tensor, torch.Tensor]\n            ``(policy_logits, value)`` where ``policy_logits`` has shape
            ``(batch, action_size)`` and ``value`` has shape ``(batch, 1)``.
        """
        body = self.res_blocks(self.input_conv(x))

        # Policy head
        p = self.policy_conv(body)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = self.value_conv(body)
        v = v.view(v.size(0), -1)
        value = self.value_fc(v)

        return policy_logits, value

    @torch.no_grad()
    def predict(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        """Encode *board* and return masked policy probabilities and a value estimate.

        Parameters
        ----------
        board:
            Current chess position.

        Returns
        -------
        tuple[dict[chess.Move, float], float]
            ``(policy_dict, value)`` where ``policy_dict`` maps each legal move to
            its probability (sums to ≈ 1.0) and ``value`` is a scalar in ``[-1, 1]``
            from the current player's perspective.
        """
        device = next(self.parameters()).device
        tensor = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)

        self.eval()
        logits, value_t = self(tensor)

        logits_np = logits.squeeze(0).cpu().numpy()  # (action_size,)

        legal_indices = get_legal_move_indices(board)
        legal_moves   = list(board.legal_moves)

        if not legal_indices:
            return {}, float(value_t.item())

        # Mask: keep only legal move logits
        masked = np.full(len(logits_np), -1e9, dtype=np.float32)
        for i in legal_indices:
            if i < len(logits_np):
                masked[i] = logits_np[i]

        # Softmax over legal subset
        legal_logits = np.array([masked[i] for i in legal_indices], dtype=np.float32)
        legal_logits -= legal_logits.max()
        exp_logits   = np.exp(legal_logits)
        probs        = exp_logits / exp_logits.sum()

        policy_dict: dict[chess.Move, float] = {}
        for move, prob in zip(legal_moves, probs):
            policy_dict[move] = float(prob)

        return policy_dict, float(value_t.item())


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: GambitNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
) -> None:
    """Save model + optimizer state to *path*.

    Parameters
    ----------
    model:
        The :class:`GambitNet` to save.
    optimizer:
        Current optimizer (state preserved for resumption).
    step:
        Global training step counter.
    path:
        Destination ``.pt`` file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "step":          step,
            "model_state":   model.state_dict(),
            "optim_state":   optimizer.state_dict(),
            "model_kwargs":  {
                "num_res_blocks": len(model.res_blocks),
                "num_filters":    model.policy_fc.in_features // 2 // 64
                                  if hasattr(model, "policy_fc") else 128,
                "action_size":    model.action_size,
            },
        },
        path,
    )


def load_checkpoint(
    path: str,
    device: str = "cpu",
) -> tuple[GambitNet, torch.optim.Optimizer, int]:
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Parameters
    ----------
    path:
        Path to the ``.pt`` file.
    device:
        Torch device string (e.g. ``"cuda"`` or ``"cpu"``).

    Returns
    -------
    tuple[GambitNet, torch.optim.Optimizer, int]
        ``(model, optimizer, step)``.
    """
    ckpt      = torch.load(path, map_location=device)
    kwargs    = ckpt.get("model_kwargs", {})
    model     = GambitNet(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])

    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(ckpt["optim_state"])

    step = ckpt.get("step", 0)
    return model, optimizer, step
