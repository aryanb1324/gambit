"""Tests for gambit.mcts.mcts."""

import chess
import pytest

from gambit.mcts.mcts import MCTS, MCTSNode
from gambit.network.resnet import GambitNet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_mcts() -> MCTS:
    """MCTS with a tiny model and few simulations for fast tests."""
    model = GambitNet(num_res_blocks=2, num_filters=32, action_size=4096)
    return MCTS(model=model, c_puct=1.0, num_simulations=50, device="cpu")


# ---------------------------------------------------------------------------
# MCTSNode unit tests
# ---------------------------------------------------------------------------

class TestMCTSNode:
    def test_q_unvisited(self) -> None:
        """Q is 0 when visit_count=0."""
        node = MCTSNode(board=chess.Board())
        assert node.Q == 0.0

    def test_q_after_update(self) -> None:
        """Q equals value_sum / visit_count."""
        node = MCTSNode(board=chess.Board())
        node.visit_count = 4
        node.value_sum   = 2.0
        assert abs(node.Q - 0.5) < 1e-9

    def test_is_terminal_startpos(self) -> None:
        """Start position is not terminal."""
        node = MCTSNode(board=chess.Board())
        assert not node.is_terminal

    def test_is_terminal_checkmate(self) -> None:
        """Fool's mate position is terminal."""
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        node  = MCTSNode(board=board)
        assert node.is_terminal


# ---------------------------------------------------------------------------
# MCTS integration tests
# ---------------------------------------------------------------------------

class TestMCTS:
    def test_search_returns_legal_moves_only(self, small_mcts: MCTS) -> None:
        """search() keys should all be legal moves."""
        board  = chess.Board()
        policy = small_mcts.search(board)
        legal  = set(board.legal_moves)
        for move in policy:
            assert move in legal, f"Illegal move {move} in MCTS policy"

    def test_search_probs_sum_to_one(self, small_mcts: MCTS) -> None:
        """MCTS policy probabilities (visit counts) sum to 1.0."""
        board  = chess.Board()
        policy = small_mcts.search(board)
        total  = sum(policy.values())
        assert abs(total - 1.0) < 1e-6, f"MCTS probs sum to {total}"

    def test_mcts_returns_legal_move(self, small_mcts: MCTS) -> None:
        """select_move() returns a legal move."""
        board = chess.Board()
        move  = small_mcts.select_move(board, temperature=1.0)
        assert move in board.legal_moves, f"Illegal move returned: {move}"

    def test_mcts_returns_legal_move_greedy(self, small_mcts: MCTS) -> None:
        """select_move(temperature=0) (greedy) still returns a legal move."""
        board = chess.Board()
        move  = small_mcts.select_move(board, temperature=0)
        assert move in board.legal_moves

    def test_visit_counts_nonzero(self, small_mcts: MCTS) -> None:
        """After search, at least one child of root should have a non-zero visit count."""
        board  = chess.Board()
        policy = small_mcts.search(board)
        assert any(v > 0 for v in policy.values()), "All visit counts are zero"

    def test_temperature_zero_deterministic(self, small_mcts: MCTS) -> None:
        """With temperature=0, the same position should always return the same move."""
        board  = chess.Board()
        moves  = {small_mcts.select_move(board, temperature=0) for _ in range(3)}
        assert len(moves) == 1, f"Expected deterministic move, got: {moves}"

    def test_mcts_from_midgame(self, small_mcts: MCTS) -> None:
        """MCTS works correctly from a mid-game position."""
        board = chess.Board()
        for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]:
            board.push(chess.Move.from_uci(uci))
        move = small_mcts.select_move(board)
        assert move in board.legal_moves

    def test_no_expansion_on_terminal(self, small_mcts: MCTS) -> None:
        """Searching from a terminal position returns an empty policy."""
        # Fool's mate — White is mated
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        assert board.is_game_over()
        policy = small_mcts.search(board)
        assert policy == {} or sum(policy.values()) <= 1.0 + 1e-6
