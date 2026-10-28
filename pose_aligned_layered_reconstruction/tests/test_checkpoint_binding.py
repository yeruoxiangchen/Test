from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from pose_aligned_layered_reconstruction.checkpoints import (
    load_layered_ss_checkpoint,
    load_target_lari_checkpoint,
    save_layered_ss_checkpoint,
    save_target_lari_checkpoint,
)
from pose_aligned_layered_reconstruction.native_ss import LayeredNativeSSFlow
from pose_aligned_layered_reconstruction.target_lari import TargetConditionedLaRI
from pose_aligned_layered_reconstruction.tests.test_native_ss_conditions import DummySSFlow


class CheckpointBindingTests(unittest.TestCase):
    def test_layer_head_checkpoint_roundtrip_and_architecture_rejection(self) -> None:
        model = TargetConditionedLaRI(input_channels=8, hidden_channels=16, decoder_blocks=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lari.pt"
            save_target_lari_checkpoint(
                path, model=model, step=2, optimizer=None, data_identity={}, history=[]
            )
            recovered = TargetConditionedLaRI(input_channels=8, hidden_channels=16, decoder_blocks=1)
            binding = load_target_lari_checkpoint(recovered, path)
            self.assertEqual(binding["step"], 2)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, recovered.state_dict()[name], rtol=0, atol=0)
            incompatible = TargetConditionedLaRI(input_channels=8, hidden_channels=16, decoder_blocks=2)
            with self.assertRaisesRegex(ValueError, "architecture differs"):
                load_target_lari_checkpoint(incompatible, path)

    def test_ss_adapter_is_bound_to_parent_and_exact_layer_predictor(self) -> None:
        model = LayeredNativeSSFlow(DummySSFlow(), layer_channels=8, lari_hidden_channels=16)
        allowed = ("layer_encoder.", "view_gate.", "layer_block_condition.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(allowed))
        optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
        parent = {"sha256": "parent-a", "format": "test", "step": 30000, "weights": "ema"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ss.pt"
            save_layered_ss_checkpoint(
                path, model=model, step=1, parent_binding=parent,
                optimizer=optimizer,
                data_identity={"layer_source": "predicted", "target_lari": {"sha256": "lari-a"}},
                history=[],
            )
            kwargs = {"parent_binding": parent, "target_lari_binding": {"sha256": "lari-a"}, "layer_source": "predicted"}
            self.assertEqual(load_layered_ss_checkpoint(model, path, **kwargs)["step"], 1)
            with self.assertRaisesRegex(ValueError, "exact Target-LaRI"):
                load_layered_ss_checkpoint(model, path, **{**kwargs, "target_lari_binding": {"sha256": "lari-b"}})
            with self.assertRaisesRegex(RuntimeError, "parent binding differs"):
                load_layered_ss_checkpoint(model, path, **{**kwargs, "parent_binding": {**parent, "sha256": "parent-b"}})
            with self.assertRaisesRegex(ValueError, "adapter trained with"):
                load_layered_ss_checkpoint(model, path, **{**kwargs, "layer_source": "oracle"})


if __name__ == "__main__":
    unittest.main()
