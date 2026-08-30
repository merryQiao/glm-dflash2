from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from omni_stage_c.dflash_model import DFlashModel
from omni_stage_c.dflash2_model import DFlash2Model
from omni_stage_c.dspark_model import DSparkModel
from omni_stage_c.modeling_common import DraftModelConfig
from omni_stage_c.offline_trainer import OfflineMethodTrainer, TrainingBatch


@dataclass(frozen=True)
class TrainingRecipe:
    method: str
    block_size: int
    anchors_per_sample: int = 512
    seed: int = 42
    max_window_tokens: int = 4096
    epochs: int = 3
    learning_rate: float = 6e-4
    warmup_steps: int = 1000
    gradient_accumulation: int = 8


def recipe_for(method: str, block_size: int) -> TrainingRecipe:
    from omni_stage_c.contracts import validate_method_block

    validate_method_block(method, block_size)
    return TrainingRecipe(method=method, block_size=block_size)


def _model(method: str, config: DraftModelConfig) -> nn.Module:
    if method == "dflash":
        return DFlashModel(config)
    if method == "dflash2":
        return DFlash2Model(config)
    if method == "dspark":
        return DSparkModel(config)
    raise ValueError(method)


def run_tiny_smoke(*, method: str, block_size: int) -> dict[str, float | int | bool]:
    """One real forward/backward/optimizer step for each supported route."""

    torch.manual_seed(17)
    recipe_for(method, block_size)
    config = DraftModelConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        num_aux_layers=5, vocab_size=32, mrope_section=(1, 1, 0),
    )
    model = _model(method, config)
    embedding = nn.Embedding(config.vocab_size, config.hidden_size)
    head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    trainer = OfflineMethodTrainer(
        method=method, block_size=block_size, model=model,
        target_embedding=embedding, target_lm_head=head, vocab_chunk_size=11,
    )
    batch_size, anchors, context = 1, 1, 3
    target_ids = torch.randint(0, config.vocab_size - 1, (batch_size, anchors, block_size))
    input_ids = torch.full_like(target_ids, config.vocab_size - 1)
    input_ids[..., 0] = target_ids[..., 0]
    total = context + anchors * block_size
    position_ids = torch.arange(total).view(1, 1, total).expand(3, batch_size, -1).clone()
    batch = TrainingBatch(
        input_ids=input_ids,
        target_ids=target_ids,
        position_ids=position_ids,
        auxiliary_hidden=torch.randn(batch_size, context, 5, config.hidden_size),
        target_final_hidden=torch.randn(batch_size, anchors, block_size, config.hidden_size),
        keep_mask=torch.ones(batch_size, anchors, dtype=torch.bool),
        attention_mask=torch.ones(
            batch_size, anchors, block_size, context + block_size, dtype=torch.bool
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    result = trainer.compute_loss(batch)
    result.loss.backward()
    optimizer.step()
    return {
        "finite_loss": bool(torch.isfinite(result.loss.detach()).item()),
        "loss": float(result.loss.detach()),
        "optimizer_steps": 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-Omni Thinker offline Stage C trainer")
    parser.add_argument("--method", choices=("dflash", "dflash2", "dspark"), required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--hidden-cache-dir", type=Path)
    parser.add_argument("--target-io-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--backend", default="hccl")
    parser.add_argument("--strategy", choices=("single", "fsdp2"), default="fsdp2")
    parser.add_argument("--tiny-smoke", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--mask-token-id", type=int, default=152063)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tiny_smoke:
        print(run_tiny_smoke(method=args.method, block_size=args.block_size))
        return
    missing = [name for name in ("hidden_cache_dir", "target_io_dir", "output_dir")
               if getattr(args, name) is None]
    if missing:
        raise SystemExit("production training requires: " + ", ".join(missing))
    # The production iterator/checkpoint implementation is imported lazily so
    # CPU contract tests never require torch-npu.
    from omni_stage_c.training_loop import run_training

    run_training(args)


if __name__ == "__main__":
    main()
