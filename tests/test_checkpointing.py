from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from glm_dflash2.checkpointing import (
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
)


class CheckpointingTest(unittest.TestCase):
    def test_exact_single_rank_state_and_rng_round_trip(self):
        torch.manual_seed(3)
        random.seed(3)
        np.random.seed(3)
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))
        anchor = torch.Generator().manual_seed(99)
        loss = model(torch.ones(2, 3)).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        expected_weights = {k: v.clone() for k, v in model.state_dict().items()}
        expected_next_anchor = torch.rand(3, generator=anchor)
        anchor.manual_seed(99)
        progress = TrainingProgress(global_step=1, micro_step=2, epoch=0, sample_cursor=4)

        with tempfile.TemporaryDirectory() as tmp:
            save_training_checkpoint(
                tmp,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                progress=progress,
                anchor_generator=anchor,
                at_optimizer_boundary=True,
                semantic_config={"cache": "abc", "grad_accum": 2},
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10)
            torch.rand(10)
            random.random()
            np.random.rand()
            anchor.manual_seed(7)
            restored = load_training_checkpoint(
                tmp,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                anchor_generator=anchor,
                expected_semantic_config={"cache": "abc", "grad_accum": 2},
            )
            self.assertEqual(restored, progress)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected_weights[key])
            torch.testing.assert_close(torch.rand(3, generator=anchor), expected_next_anchor)
            self.assertEqual(scheduler.last_epoch, 1)
            self.assertTrue((Path(tmp) / "COMPLETE").is_file())

    def test_rejects_resume_with_different_semantic_config(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            save_training_checkpoint(
                tmp,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                progress=TrainingProgress(0, 0, 0, 0),
                anchor_generator=torch.Generator(),
                at_optimizer_boundary=True,
                semantic_config={"mask_token_id": 7},
            )
            with self.assertRaisesRegex(ValueError, "semantic configuration"):
                load_training_checkpoint(
                    tmp,
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    anchor_generator=torch.Generator(),
                    expected_semantic_config={"mask_token_id": 8},
                )

    def test_rejects_mid_accumulation_checkpoint(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "optimizer-step boundary"):
                save_training_checkpoint(
                    tmp,
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    progress=TrainingProgress(0, 1, 0, 1),
                    anchor_generator=torch.Generator(),
                    at_optimizer_boundary=False,
                )

    def test_checkpoint_fsyncs_payload_metadata_and_complete_marker(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "glm_dflash2.checkpointing.os.fsync"
        ) as fsync:
            save_training_checkpoint(
                tmp,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                progress=TrainingProgress(0, 0, 0, 0),
                anchor_generator=torch.Generator(),
                at_optimizer_boundary=True,
            )
            self.assertGreaterEqual(fsync.call_count, 4)


if __name__ == "__main__":
    unittest.main()
