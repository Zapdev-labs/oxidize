from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dflash.config import (
    DFlashTrainConfig,
    build_target_layer_ids,
    positional_loss_weights,
)


class DFlashConfigTests(unittest.TestCase):
    def test_qwen38_target_layers_match_oxidize_c_defaults(self) -> None:
        ids = build_target_layer_ids(5, 64)
        self.assertEqual(ids, [1, 16, 31, 46, 61])

    def test_n_feat_is_concatenated_hidden(self) -> None:
        cfg = DFlashTrainConfig()
        self.assertEqual(cfg.n_feat, 5120 * 5)
        self.assertEqual(cfg.target_layer_ids, [1, 16, 31, 46, 61])

    def test_loss_weights_are_normalized_and_decay(self) -> None:
        w = positional_loss_weights(15, 7.0)
        self.assertEqual(len(w), 15)
        self.assertAlmostEqual(sum(w) / len(w), 1.0, places=6)
        self.assertGreater(w[0], w[-1])

    def test_loss_weights_reject_non_positive_gamma(self) -> None:
        with self.assertRaises(ValueError):
            positional_loss_weights(15, 0.0)
        with self.assertRaises(ValueError):
            positional_loss_weights(15, -1.0)

    def test_target_layer_ids_are_distinct_or_rejected(self) -> None:
        ids = build_target_layer_ids(3, 8)
        self.assertEqual(len(set(ids)), len(ids))
        with self.assertRaises(ValueError):
            build_target_layer_ids(5, 4)
        with self.assertRaises(ValueError):
            DFlashTrainConfig(num_target_layers=5, target_n_layers=4)

    def test_duplicate_explicit_target_layer_ids_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DFlashTrainConfig(target_layer_ids=[1, 1, 2])


class DFlashAnchorTests(unittest.TestCase):
    def test_final_anchor_is_included(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from scripts.dflash.model import sample_anchors

        seq_len, block = 10, 4
        gen = torch.Generator().manual_seed(0)
        anchors = sample_anchors(seq_len, block, max_anchors=100, generator=gen)
        self.assertEqual(sorted(anchors.tolist()), list(range(1, seq_len - block + 1)))
        self.assertIn(seq_len - block, anchors.tolist())

    def test_single_anchor_when_seq_is_block_plus_one(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from scripts.dflash.model import sample_anchors

        block = 4
        gen = torch.Generator().manual_seed(0)
        anchors = sample_anchors(block + 1, block, max_anchors=8, generator=gen)
        self.assertEqual(anchors.tolist(), [1])
        self.assertEqual(sample_anchors(block, block, max_anchors=8, generator=gen).numel(), 0)


class DFlashMaskTests(unittest.TestCase):
    def test_block_attention_keeps_prefix_and_full_block(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from scripts.dflash.model import block_attention_bias

        keep = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
        bias = block_attention_bias(keep, block_size=4, dtype=torch.float32)
        self.assertEqual(tuple(bias.shape), (2, 1, 4, 9))
        self.assertTrue(torch.isfinite(bias[0, 0, 0, :3]).all())
        self.assertTrue(torch.isinf(bias[0, 0, 0, 3:5]).all())
        self.assertTrue(torch.isfinite(bias[0, 0, 2, 5:]).all())


class DFlashTinyTrainTests(unittest.TestCase):
    def test_tiny_draft_backward(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from scripts.dflash.model import DFlashDraftModel, block_attention_bias

        cfg = DFlashTrainConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            intermediate_size=64,
            vocab_size=128,
            block_size=4,
            num_target_layers=2,
            target_n_layers=8,
            max_anchors=2,
        )
        draft = DFlashDraftModel(cfg)
        bsz, seq, blk = 2, 8, cfg.block_size
        noise = torch.randn(bsz, blk, cfg.hidden_size)
        target = torch.randn(bsz, seq, cfg.n_feat)
        noise_pos = torch.arange(3, 3 + blk).unsqueeze(0).expand(bsz, -1)
        context_pos = torch.arange(seq).unsqueeze(0).expand(bsz, -1)
        keep = torch.zeros(bsz, seq, dtype=torch.bool)
        keep[:, :4] = True
        bias = block_attention_bias(keep, blk, noise.dtype)
        hidden = draft(noise, target, noise_pos, context_pos, bias)
        loss = hidden.pow(2).mean()
        loss.backward()
        grads = [p.grad.abs().sum().item() for p in draft.parameters() if p.grad is not None]
        self.assertTrue(any(g > 0.0 for g in grads))
        self.assertEqual(hidden.shape, (bsz, blk, cfg.hidden_size))
