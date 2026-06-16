"""Tests for gambit.board.encoder."""

import chess
import numpy as np
import pytest

from gambit.board.encoder import (
    encode_board,
    get_legal_move_indices,
    index_to_move,
    move_to_index,
)


class TestEncodeBoard:
    def test_encode_startpos_shape(self) -> None:
        """Encoded starting position has shape (17, 8, 8)."""
        board  = chess.Board()
        tensor = encode_board(board)
        assert tensor.shape == (17, 8, 8), f"Expected (17,8,8), got {tensor.shape}"
        assert tensor.dtype == np.float32

    def test_encode_startpos_piece_counts(self) -> None:
        """Piece planes have the right number of 1s at the starting position."""
        board  = chess.Board()
        tensor = encode_board(board)

        # White pawns (plane 0): 8 pawns on rank 2
        assert tensor[0].sum() == 8, "White pawn plane should have 8 pieces"
        # Black pawns (plane 6): 8 pawns on rank 7
        assert tensor[6].sum() == 8, "Black pawn plane should have 8 pieces"
        # White king (plane 5): 1 king
        assert tensor[5].sum() == 1, "White king plane should have 1 piece"
        # Black king (plane 11): 1 king
        assert tensor[11].sum() == 1, "Black king plane should have 1 piece"

    def test_encode_startpos_side_to_move(self) -> None:
        """Plane 12 is all-ones when White to move."""
        board  = chess.Board()
        tensor = encode_board(board)
        assert tensor[12].min() == 1.0, "Side-to-move plane should be 1.0 for White"

    def test_encode_black_to_move(self) -> None:
        """Plane 12 is all-zeros when Black to move."""
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        tensor = encode_board(board)
        assert tensor[12].max() == 0.0, "Side-to-move plane should be 0.0 for Black"

    def test_encode_castling_planes(self) -> None:
        """Starting position has all four castling rights set (planes 13–16)."""
        board  = chess.Board()
        tensor = encode_board(board)
        for plane in [13, 14, 15, 16]:
            assert tensor[plane].min() == 1.0, f"Castling plane {plane} should be 1.0"

    def test_encode_values_are_binary(self) -> None:
        """All values in the output tensor are 0 or 1."""
        board  = chess.Board()
        tensor = encode_board(board)
        unique = set(np.unique(tensor))
        assert unique <= {0.0, 1.0}, f"Expected only 0/1 values, got: {unique}"


class TestMoveEncoding:
    def test_move_to_index_range(self) -> None:
        """move_to_index returns an int in [0, 4159]."""
        board = chess.Board()
        for move in board.legal_moves:
            idx = move_to_index(move)
            assert 0 <= idx <= 4159, f"Index {idx} out of range for move {move}"

    def test_move_encoding_roundtrip(self) -> None:
        """Encoding then decoding a move returns the original move."""
        board = chess.Board()
        for move in board.legal_moves:
            idx        = move_to_index(move)
            recovered  = index_to_move(idx, board)
            assert recovered == move, f"Roundtrip failed: {move} → {idx} → {recovered}"

    def test_underpromotion_offset(self) -> None:
        """Under-promotions use offsets starting at 4096."""
        # Craft a board with a pawn on the 7th rank ready to promote
        board = chess.Board("8/P7/8/8/8/8/8/4K1k1 w - - 0 1")
        for move in board.legal_moves:
            if move.promotion == chess.KNIGHT:
                idx = move_to_index(move)
                assert idx >= 4096, f"Knight promotion index {idx} < 4096"


class TestLegalMoveIndices:
    def test_startpos_count(self) -> None:
        """Starting position has 20 legal moves."""
        board   = chess.Board()
        indices = get_legal_move_indices(board)
        assert len(indices) == 20, f"Expected 20 legal moves, got {len(indices)}"

    def test_indices_correspond_to_legal_moves(self) -> None:
        """Every index returned corresponds to an actual legal move."""
        board   = chess.Board()
        indices = set(get_legal_move_indices(board))
        for move in board.legal_moves:
            assert move_to_index(move) in indices

    def test_no_duplicates(self) -> None:
        """Returned indices are unique."""
        board   = chess.Board()
        indices = get_legal_move_indices(board)
        assert len(indices) == len(set(indices))
