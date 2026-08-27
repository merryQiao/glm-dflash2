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
from torch.distributed.device_mesh import init_device_mesh

from glm_dflash2.dflash2_model import Qwen3DFlash2DraftModel, build_dflash2_config
from glm_dflash2.distributed import configure_accumulation, distributed_any
from glm_dflash2.offline_trainer import OfflineDFlash2Trainer
from glm_dflash2.target_io import FrozenTargetIO
from tools.train_drafter_offline import _sample_or_dummy_anchors


def _worker(rank: int, rendezvous: str, reports: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(4)
        config = build_dflash2_config(
            vocab_size=13, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=4, target_layer_ids=[0, 1], num_target_layers=2,
            block_size=4, mask_token_id=12, conv_group_size=4,
            selector_rank=4, selector_top_k=4, sliding_window=None,
        )
        draft = Qwen3DFlash2DraftModel(config)
        mesh = init_device_mesh("cpu", (2,))
        fully_shard(draft.layers[0], mesh=mesh)
        fully_shard(draft.candidate_selector, mesh=mesh)
        fully_shard(draft, mesh=mesh)
        embed = nn.Embedding(13, 8)
        head = nn.Linear(8, 13, bias=False)
        embed.weight.requires_grad_(False)
        head.weight.requires_grad_(False)
        trainer = OfflineDFlash2Trainer(
            draft,
            FrozenTargetIO(embed, head, {
                "schema": "glm-drafter-target-io-v2",
                "source_model_fingerprint": "tiny", "model_revision": "revision",
                "tokenizer_fingerprint": "tokenizer", "hidden_size": 8, "vocab_size": 13,
                "source_dtypes": {"embed_tokens": "torch.bfloat16", "lm_head": "torch.bfloat16"},
                "logit_transform": "identity", "lm_head_bias": False,
            }),
            cache_manifest={
                "spec": {"layer_ids": [0, 1], "hidden_size": 8,
                         "dtype": "bfloat16", "mask_semantics": "dflash_target_token",
                         "schema_version": 2,
                         "final_hidden_semantics": "post_final_norm_lm_head_input"},
                "provenance": {"model_fingerprint": "tiny", "model_revision": "revision",
                               "tokenizer_fingerprint": "tokenizer", "vocab_size": 13,
                               "target_hidden_dtype": "bfloat16", "logical_layer_ids": [0, 1]},
            },
            num_anchors=1, token_chunk_size=2, vocab_chunk_size=5,
        )
        batch = {
            "sample_ids": [f"rank-{rank}"],
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
            "attention_mask": torch.ones(1, 6, dtype=torch.bool),
            "loss_mask": torch.ones(1, 6, dtype=torch.bool) if rank == 0
                         else torch.zeros(1, 6, dtype=torch.bool),
            "hidden_states": torch.randn(1, 6, 16),
        }
        anchors, keep, local_has = _sample_or_dummy_anchors(batch, trainer, epoch=0)
        global_has = distributed_any(local_has, torch.device("cpu"))
        configure_accumulation(draft, synchronize=True)
        output = trainer(batch, anchor_positions=anchors, block_keep_mask=keep)
        output.loss.backward()
        optimizer = torch.optim.SGD(trainer.parameters(), lr=1e-3)
        optimizer.step()
        dist.barrier()
        Path(reports, f"rank-{rank}.json").write_text(json.dumps({
            "global_has": global_has,
            "local_has": local_has,
            "finite": bool(torch.isfinite(output.loss)),
        }))
    finally:
        dist.destroy_process_group()


class EmptyAnchorDistributedTest(unittest.TestCase):
    def test_one_empty_rank_still_joins_fsdp_forward_and_backward(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "reports"
            reports.mkdir()
            mp.spawn(_worker, args=(str(root / "rdzv"), str(reports)), nprocs=2, join=True)
            values = [json.loads((reports / f"rank-{rank}.json").read_text()) for rank in range(2)]
            self.assertEqual([value["local_has"] for value in values], [True, False])
            self.assertTrue(all(value["global_has"] and value["finite"] for value in values))


if __name__ == "__main__":
    unittest.main()
