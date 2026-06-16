from __future__ import annotations
"""Board encoder — converts a chess.Board into a float32 tensor of shape (17, 8, 8).

Plane layout
------------
 0– 5 : white pieces  (P, N, B, R, Q, K)
 6–11 : black pieces  (p, n, b, r, q, k)
12    : side to move  (1.0 everywhere = white, 0.0 = black)
13    : white kingside castling right
14    : white queenside castling right
15    : black kingside castling right
16    : black queenside castling right

Action space
------------
Base moves:  from_sq * 64 + to_sq  → [0, 4095]
Promotions to non-queen pieces add offsets starting at 4096.
Total action size used throughout the project: 4096 (queen-promotion shares the
base move index; under-promotions are rare and handled via offset mapping).
"""


from typing import Optional

import chess
import numpy as np

# Piece type order used for plane indexing (white planes 0-5, black planes 6-11)
_PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]

# Under-promotion offsets (starting at 4096)
# knight=0, bishop=1, rook=2  (queen is implicit in base move)
_UNDERPROMO_OFFSET: dict[chess.PieceType, int] = {
    chess.KNIGHT: 4096,
    chess.BISHOP: 4096 + 64,
    chess.ROOK:   4096 + 128,
}


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode *board* into a float32 array of shape ``(17, 8, 8)``.

    Parameters
    ----------
    board:
        Position to encode.

    Returns
    -------
    np.ndarray
        Shape ``(17, 8, 8)``, dtype ``float32``.
    """
    planes = np.zeros((17, 8, 8), dtype=np.float32)

    # Piece planes
    for plane_idx, piece_type in enumerate(_PIECE_TYPES):
        # White pieces
        for sq in board.pieces(piece_type, chess.WHITE):
            r = chess.square_rank(sq)
            f = chess.square_file(sq)
            planes[plane_idx, r, f] = 1.0
        # Black pieces
        for sq in board.pieces(piece_type, chess.BLACK):
            r = chess.square_rank(sq)
            f = chess.square_file(sq)
            planes[plane_idx + 6, r, f] = 1.0

    # Side to move
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0

    return planes


def move_to_index(move: chess.Move) -> int:
    """Map a UCI move to an integer in ``[0, 4159]``.

    Base moves use ``from_sq * 64 + to_sq`` → ``[0, 4095]``.
    Under-promotions (non-queen) get offsets starting at 4096.

    Parameters
    ----------
    move:
        A :class:`chess.Move`.

    Returns
    -------
    int
        Action index.
    """
    if move.promotion is not None and move.promotion != chess.QUEEN:
        offset = _UNDERPROMO_OFFSET[move.promotion]
        return offset + move.to_square
    return move.from_square * 64 + move.to_square


def index_to_move(idx: int, board: chess.Board) -> Optional[chess.Move]:
    """Reverse mapping: action index → legal :class:`chess.Move` (or ``None``).

    Parameters
    ----------
    idx:
        Action index produced by :func:`move_to_index`.
    board:
        Current position (used to resolve legal moves).

    Returns
    -------
    chess.Move or None
        The matching legal move, or ``None`` if no legal move corresponds to *idx*.
    """
    for move in board.legal_moves:
        if move_to_index(move) == idx:
            return move
    return None


def get_legal_move_indices(board: chess.Board) -> list[int]:
    """Return a list of action indices for all legal moves in *board*.

    Parameters
    ----------
    board:
        Current position.

    Returns
    -------
    list[int]
        Action indices suitable for masking the policy head output.
    """
    return [move_to_index(m) for m in board.legal_moves]
