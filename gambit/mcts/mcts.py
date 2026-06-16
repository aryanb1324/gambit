from __future__ import annotations
"""Monte Carlo Tree Search (MCTS) for Gambit.

Implements AlphaZero-style PUCT search:
    U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

where
    Q = mean action value  (W / N)
    P = prior from the policy head
    N = visit count

Dirichlet noise (alpha=0.3, epsilon=0.25) is mixed into the root priors at
expansion time to encourage exploration during self-play.
"""


import math
from typing import Optional

import chess
import numpy as np

from gambit.board.encoder import get_legal_move_indices, move_to_index


class MCTSNode:
    """A node in the MCTS tree, corresponding to a board position.

    Parameters
    ----------
    board:
        The position at this node (copied from parent).
    parent:
        Parent node, or ``None`` for the root.
    move:
        The move that led to this node from the parent.
    prior:
        Prior probability assigned by the policy network.
    """

    __slots__ = (
        "board",
        "parent",
        "move",
        "children",
        "visit_count",
        "value_sum",
        "prior",
        "is_expanded",
    )

    def __init__(
        self,
        board: chess.Board,
        parent: Optional["MCTSNode"] = None,
        move: Optional[chess.Move] = None,
        prior: float = 0.0,
    ) -> None:
        self.board       = board
        self.parent      = parent
        self.move        = move
        self.children: dict[chess.Move, MCTSNode] = {}
        self.visit_count  = 0
        self.value_sum    = 0.0
        self.prior        = prior
        self.is_expanded  = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def Q(self) -> float:
        """Mean action value W/N (returns 0 when unvisited)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_terminal(self) -> bool:
        """True when the game is over at this node."""
        return self.board.is_game_over()

    def __repr__(self) -> str:  # pragma: no cover
        move_str = self.move.uci() if self.move else "root"
        return (
            f"MCTSNode(move={move_str}, N={self.visit_count}, "
            f"Q={self.Q:.3f}, P={self.prior:.3f})"
        )


class MCTS:
    """AlphaZero-style MCTS driven by a :class:`~gambit.network.resnet.GambitNet`.

    Parameters
    ----------
    model:
        The neural network used for policy and value evaluation.
    c_puct:
        Exploration constant in the PUCT formula (default 1.0).
    num_simulations:
        Number of simulations per search call (default 800).
    device:
        Torch device string (default ``"cpu"``).
    """

    def __init__(
        self,
        model: "GambitNet",  # type: ignore[name-defined]  # noqa: F821
        c_puct: float = 1.0,
        num_simulations: int = 800,
        device: str = "cpu",
    ) -> None:
        self.model           = model
        self.c_puct          = c_puct
        self.num_simulations = num_simulations
        self.device          = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        board: chess.Board,
        add_noise: bool = True,
    ) -> dict[chess.Move, float]:
        """Run ``num_simulations`` simulations from *board*.

        Parameters
        ----------
        board:
            Root position to search from.
        add_noise:
            If ``True`` (default), mix Dirichlet noise into root priors for
            exploration (self-play).  Set to ``False`` for greedy / deterministic
            inference.

        Returns
        -------
        dict[chess.Move, float]
            MCTS policy: ``{move: visit_count / total_visits}``.
        """
        root = MCTSNode(board=board.copy())
        # Initial expansion — only add Dirichlet noise in training / self-play
        self._expand(root, add_noise=add_noise)

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not node.is_terminal and not node.is_expanded:
                value = self._expand(node, add_noise=False)
            else:
                value = self._terminal_value(node)
            self._backup(node, value)

        total = sum(c.visit_count for c in root.children.values())
        if total == 0:
            # Fallback: uniform over legal moves
            legal = list(board.legal_moves)
            return {m: 1.0 / len(legal) for m in legal} if legal else {}

        return {
            move: child.visit_count / total
            for move, child in root.children.items()
        }

    def select_move(self, board: chess.Board, temperature: float = 1.0) -> chess.Move:
        """Run search then sample a move.

        Parameters
        ----------
        board:
            Current position.
        temperature:
            Sampling temperature.  ``0`` → greedy (no Dirichlet noise);
            ``1`` → proportional to visit counts (with noise).

        Returns
        -------
        chess.Move
            Selected move.
        """
        # Greedy mode: disable Dirichlet noise for deterministic inference
        add_noise = temperature != 0
        policy = self.search(board, add_noise=add_noise)
        moves  = list(policy.keys())
        counts = np.array([policy[m] for m in moves], dtype=np.float64)

        if temperature == 0:
            return moves[int(np.argmax(counts))]

        counts_t = counts ** (1.0 / temperature)
        probs    = counts_t / counts_t.sum()
        idx      = np.random.choice(len(moves), p=probs)
        return moves[idx]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Descend the tree using PUCT until we reach an unexpanded or terminal node."""
        while node.is_expanded and not node.is_terminal:
            node = self._best_child(node)
        return node

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with the highest PUCT score."""
        sqrt_n = math.sqrt(max(node.visit_count, 1))
        best_score = -float("inf")
        best_child: Optional[MCTSNode] = None

        for child in node.children.values():
            puct = child.Q + self.c_puct * child.prior * sqrt_n / (1 + child.visit_count)
            if puct > best_score:
                best_score = puct
                best_child = child

        assert best_child is not None, "Node has no children but is marked expanded"
        return best_child

    def _expand(self, node: MCTSNode, *, add_noise: bool = False) -> float:
        """Expand *node*: call the network, create child nodes, return value.

        Parameters
        ----------
        node:
            The node to expand.
        add_noise:
            If ``True``, mix Dirichlet noise into root priors (self-play exploration).

        Returns
        -------
        float
            Value estimate from the network's value head.
        """
        policy_dict, value = self.model.predict(node.board)

        if not policy_dict:
            # Terminal (no legal moves)
            node.is_expanded = True
            return value

        priors = list(policy_dict.values())
        moves  = list(policy_dict.keys())

        if add_noise and len(priors) > 0:
            alpha  = 0.3
            eps    = 0.25
            noise  = np.random.dirichlet([alpha] * len(priors))
            priors = [(1 - eps) * p + eps * n for p, n in zip(priors, noise)]

        for move, prior in zip(moves, priors):
            child_board = node.board.copy()
            child_board.push(move)
            node.children[move] = MCTSNode(
                board=child_board,
                parent=node,
                move=move,
                prior=prior,
            )

        node.is_expanded = True
        return value

    def _backup(self, node: MCTSNode, value: float) -> None:
        """Walk up the tree, updating visit counts and value sums (negamax)."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum   += value
            value                = -value
            current              = current.parent

    @staticmethod
    def _terminal_value(node: MCTSNode) -> float:
        """Return the game outcome value for a terminal node."""
        result = node.board.result()
        if result == "1-0":
            # White wins; value from the *current* player's perspective
            return 1.0 if node.board.turn == chess.BLACK else -1.0
        if result == "0-1":
            return 1.0 if node.board.turn == chess.WHITE else -1.0
        return 0.0  # draw
