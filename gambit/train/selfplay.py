"""Phase 3: self-play reinforcement learning loop.

Each iteration:
1. Generate ``games_per_iter`` complete games using MCTS.
2. Add positions to the replay buffer.
3. Train the model for ``train_steps_per_iter`` steps on random buffer samples.
4. Save a checkpoint.
5. Log iteration stats and repeat.

The MCTS policy targets (visit-count distributions) are used as the supervised
signal for the policy head (cross-entropy), while the game outcome provides
the target for the value head (MSE).

Replay buffer
-------------
Stores up to ``max_size`` tuples::

    (board_tensor: np.ndarray[17,8,8],
     mcts_policy:  np.ndarray[4096],   ← visit-count distribution over action space
     outcome:      float)              ← +1 / 0 / -1 from current player's PoV
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Optional

import chess
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from gambit.board.encoder import encode_board, move_to_index
from gambit.mcts.mcts import MCTS
from gambit.network.resnet import ACTION_SIZE, GambitNet, load_checkpoint, save_checkpoint

# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

Example = tuple[np.ndarray, np.ndarray, float]


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer.

    Parameters
    ----------
    max_size:
        Maximum number of examples stored before old entries are evicted.
    """

    def __init__(self, max_size: int = 500_000) -> None:
        self._buffer: deque[Example] = deque(maxlen=max_size)

    def push(self, examples: list[Example]) -> None:
        """Add a list of examples to the buffer."""
        self._buffer.extend(examples)

    def sample(self, batch_size: int) -> list[Example]:
        """Sample *batch_size* examples uniformly at random."""
        return random.sample(self._buffer, min(batch_size, len(self._buffer)))

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Self-play game generation
# ---------------------------------------------------------------------------


def run_selfplay_game(
    model:          GambitNet,
    mcts:           MCTS,
    temp_threshold: int = 30,
) -> list[Example]:
    """Play one full game using MCTS and collect training examples.

    Parameters
    ----------
    model:
        Current model (used inside MCTS for neural evaluations).
    mcts:
        Pre-configured MCTS instance.
    temp_threshold:
        Move number up to which temperature=1.0 is used; afterwards
        temperature=0 (greedy) for sharper play.

    Returns
    -------
    list[Example]
        One ``(board_tensor, mcts_policy_vector, outcome)`` per position.
        Outcome is from the perspective of the player to move at that position.
    """
    board    = chess.Board()
    examples: list[tuple[np.ndarray, dict[chess.Move, float], chess.Color]] = []
    move_num = 0

    while not board.is_game_over():
        temperature = 1.0 if move_num < temp_threshold else 0.0
        policy_dict = mcts.search(board)

        # Store raw policy dict with current turn
        examples.append((encode_board(board), policy_dict, board.turn))

        move = _sample_move(policy_dict, temperature)
        board.push(move)
        move_num += 1

    # Determine outcome
    result   = board.result()
    outcome  = _result_to_outcome(result)  # from White's perspective

    # Convert to training format
    training_examples: list[Example] = []
    for board_tensor, policy_dict, turn in examples:
        # Outcome from this player's perspective
        if turn == chess.WHITE:
            player_outcome = outcome
        else:
            player_outcome = -outcome

        # Convert policy dict to dense vector
        policy_vec = np.zeros(ACTION_SIZE, dtype=np.float32)
        for move, prob in policy_dict.items():
            idx = move_to_index(move)
            if 0 <= idx < ACTION_SIZE:
                policy_vec[idx] = prob

        training_examples.append((board_tensor, policy_vec, player_outcome))

    return training_examples


def _sample_move(
    policy_dict: dict[chess.Move, float],
    temperature: float,
) -> chess.Move:
    moves  = list(policy_dict.keys())
    counts = np.array(list(policy_dict.values()), dtype=np.float64)

    if temperature == 0:
        return moves[int(np.argmax(counts))]

    counts_t = counts ** (1.0 / max(temperature, 1e-8))
    total    = counts_t.sum()
    probs    = counts_t / total if total > 0 else np.ones(len(moves)) / len(moves)
    idx      = np.random.choice(len(moves), p=probs)
    return moves[idx]


def _result_to_outcome(result: str) -> float:
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    return 0.0


# ---------------------------------------------------------------------------
# Self-play training loop
# ---------------------------------------------------------------------------


def train_selfplay(
    checkpoint_path:        str,
    output_dir:             str,
    num_iterations:         int   = 100,
    games_per_iter:         int   = 25,
    train_steps_per_iter:   int   = 200,
    batch_size:             int   = 512,
    num_simulations:        int   = 400,
    device:                 str   = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Main self-play reinforcement learning loop.

    Parameters
    ----------
    checkpoint_path:
        Starting checkpoint (typically the supervised-pretrained model).
    output_dir:
        Directory for saving iteration checkpoints.
    num_iterations:
        Number of generate-then-train cycles.
    games_per_iter:
        Games to generate per iteration.
    train_steps_per_iter:
        Training mini-batch steps per iteration.
    batch_size:
        Training batch size.
    num_simulations:
        MCTS simulations per move during self-play.
    device:
        Torch device string.
    """
    os.makedirs(output_dir, exist_ok=True)

    model, optimizer, start_step = load_checkpoint(checkpoint_path, device=device)
    model.to(device)

    mcts          = MCTS(model=model, num_simulations=num_simulations, device=device)
    buffer        = ReplayBuffer(max_size=500_000)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn  = nn.MSELoss()
    global_step    = start_step

    for iteration in range(1, num_iterations + 1):
        # ── 1. Generate games ────────────────────────────────────────────
        model.eval()
        all_examples: list[Example] = []
        game_lengths: list[int]     = []

        for g in tqdm(range(games_per_iter), desc=f"iter {iteration} games"):
            examples = run_selfplay_game(model, mcts)
            all_examples.extend(examples)
            game_lengths.append(len(examples))

        buffer.push(all_examples)
        avg_len = sum(game_lengths) / max(len(game_lengths), 1)

        # ── 2. Train ──────────────────────────────────────────────────────
        if len(buffer) < batch_size:
            print(f"[selfplay iter {iteration}] buffer too small ({len(buffer)}), skipping train")
            continue

        model.train()
        p_losses: list[float] = []
        v_losses: list[float] = []

        for _ in range(train_steps_per_iter):
            batch = buffer.sample(batch_size)

            boards_np   = np.stack([e[0] for e in batch], axis=0)
            policies_np = np.stack([e[1] for e in batch], axis=0)
            values_np   = np.array([e[2] for e in batch], dtype=np.float32)

            boards_t   = torch.from_numpy(boards_np).to(device)
            policies_t = torch.from_numpy(policies_np).to(device)
            values_t   = torch.from_numpy(values_np).unsqueeze(1).to(device)

            optimizer.zero_grad()
            policy_logits, value_pred = model(boards_t)

            ploss = -(policies_t * torch.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            vloss = value_loss_fn(value_pred, values_t)
            loss  = ploss + vloss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            p_losses.append(ploss.item())
            v_losses.append(vloss.item())

        avg_p = sum(p_losses) / max(len(p_losses), 1)
        avg_v = sum(v_losses) / max(len(v_losses), 1)

        print(
            f"[selfplay iter {iteration:3d}] "
            f"avg_game_len={avg_len:.1f}  "
            f"policy_loss={avg_p:.4f}  "
            f"value_loss={avg_v:.4f}  "
            f"buffer_size={len(buffer)}"
        )

        # ── 3. Checkpoint ─────────────────────────────────────────────────
        ckpt_path = os.path.join(output_dir, f"selfplay_iter{iteration:04d}.pt")
        save_checkpoint(model, optimizer, global_step, ckpt_path)

        # Always overwrite "latest"
        save_checkpoint(
            model, optimizer, global_step,
            os.path.join(output_dir, "selfplay_latest.pt"),
        )

    print("[selfplay] Training complete.")
