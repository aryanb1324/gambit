"""Tests for gambit.network.resnet."""

import chess
import numpy as np
import pytest
import torch

from gambit.network.resnet import GambitNet, ResBlock


class TestResBlock:
    def test_output_shape_unchanged(self) -> None:
        """ResBlock preserves the spatial dimensions and filter count."""
        block = ResBlock(num_filters=64)
        x     = torch.randn(2, 64, 8, 8)
        out   = block(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


class TestGambitNetForward:
    @pytest.fixture
    def small_model(self) -> GambitNet:
        """Tiny model (2 res-blocks, 32 filters) for fast testing."""
        return GambitNet(num_res_blocks=2, num_filters=32, action_size=4096)

    def test_forward_pass_shapes(self, small_model: GambitNet) -> None:
        """forward() returns (policy_logits, value) with correct shapes."""
        x              = torch.randn(1, 17, 8, 8)
        policy_logits, value = small_model(x)
        assert policy_logits.shape == (1, 4096), \
            f"Policy shape: expected (1, 4096), got {policy_logits.shape}"
        assert value.shape == (1, 1), \
            f"Value shape: expected (1, 1), got {value.shape}"

    def test_forward_batch(self, small_model: GambitNet) -> None:
        """forward() works with a batch size > 1."""
        x              = torch.randn(4, 17, 8, 8)
        policy_logits, value = small_model(x)
        assert policy_logits.shape == (4, 4096)
        assert value.shape == (4, 1)

    def test_value_range(self, small_model: GambitNet) -> None:
        """Value output should lie in [-1, 1] (tanh)."""
        x = torch.randn(8, 17, 8, 8)
        _, value = small_model(x)
        assert value.min().item() >= -1.0 - 1e-6, "Value below -1"
        assert value.max().item() <=  1.0 + 1e-6, "Value above +1"

    def test_predict_returns_dict_and_float(self, small_model: GambitNet) -> None:
        """predict() returns (dict[Move, float], float)."""
        board = chess.Board()
        policy_dict, value = small_model.predict(board)

        assert isinstance(policy_dict, dict), "Policy should be a dict"
        assert isinstance(value, float),      "Value should be a float"
        assert -1.0 - 1e-6 <= value <= 1.0 + 1e-6, f"Value out of range: {value}"

    def test_predict_probs_sum_to_one(self, small_model: GambitNet) -> None:
        """Policy probabilities sum to approximately 1.0."""
        board       = chess.Board()
        policy_dict, _ = small_model.predict(board)
        total       = sum(policy_dict.values())
        assert abs(total - 1.0) < 1e-4, f"Probs sum to {total}, expected ~1.0"

    def test_predict_only_legal_moves(self, small_model: GambitNet) -> None:
        """predict() only returns legal moves as keys."""
        board       = chess.Board()
        policy_dict, _ = small_model.predict(board)
        legal       = set(board.legal_moves)
        for move in policy_dict:
            assert move in legal, f"Illegal move {move} in policy dict"

    def test_predict_all_legal_moves_present(self, small_model: GambitNet) -> None:
        """All legal moves should appear in the policy dict."""
        board       = chess.Board()
        policy_dict, _ = small_model.predict(board)
        legal       = set(board.legal_moves)
        assert set(policy_dict.keys()) == legal

    def test_gradient_flows(self, small_model: GambitNet) -> None:
        """Backward pass should produce non-None gradients."""
        small_model.train()
        x = torch.randn(2, 17, 8, 8, requires_grad=False)
        policy_logits, value = small_model(x)
        loss = policy_logits.sum() + value.sum()
        loss.backward()
        for name, param in small_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
