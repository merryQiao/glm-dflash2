from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.fsdp import fully_shard

from glm_dflash2.checkpointing import (
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
)
from glm_dflash2.distributed import configure_accumulation


def _distributed_resume_worker(rank: int, rendezvous: str, checkpoint: str, reports: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        torch.manual_seed(22)
        model = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 2))
        fully_shard(model[0])
        fully_shard(model[2])
        fully_shard(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
        optimizer.zero_grad(set_to_none=True)
        for micro in range(2):
            synchronize = micro == 1
            configure_accumulation(model, synchronize=synchronize)
            value = torch.full((3, 4), float(rank + micro + 1))
            model(value).square().mean().backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(2)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        expected = model(torch.ones(2, 4)).detach().cpu()
        anchor = torch.Generator().manual_seed(55 + rank)
        anchor_state = anchor.get_state()
        expected_anchor = torch.rand(4, generator=anchor)
        anchor.set_state(anchor_state)
        progress = TrainingProgress(1, 2, 0, 2)
        save_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=progress,
            anchor_generator=anchor,
            at_optimizer_boundary=True,
        )
        for parameter in model.parameters():
            with torch.no_grad():
                parameter.add_(1)
        restored = load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            anchor_generator=anchor,
        )
        actual = model(torch.ones(2, 4)).detach().cpu()
        passed = bool(
            torch.allclose(actual, expected)
            and torch.equal(torch.rand(4, generator=anchor), expected_anchor)
            and restored == progress
            and scheduler.last_epoch == 1
        )
        Path(reports, f"rank-{rank}.json").write_text(json.dumps({"passed": passed}))
    finally:
        dist.destroy_process_group()


class DistributedCheckpointingTest(unittest.TestCase):
    def test_two_rank_fsdp2_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "reports"
            reports.mkdir()
            mp.spawn(
                _distributed_resume_worker,
                args=(str(root / "rdzv"), str(root / "checkpoint"), str(reports)),
                nprocs=2,
                join=True,
            )
            for rank in range(2):
                self.assertTrue(json.loads((reports / f"rank-{rank}.json").read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
