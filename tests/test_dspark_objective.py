from __future__ import annotations

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from glm_dflash2.dspark_model import LowRankMarkovHead
from glm_dflash2.method_objectives import compute_dspark_loss, reconstruct_target_logits


class DSparkObjectiveTest(unittest.TestCase):
    def _fixture(self):
        draft = torch.tensor(
            [[[[1.0, -0.5, 0.25, 0.75], [0.0, 0.5, -1.0, 1.0]]]],
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        teacher = torch.tensor(
            [[[[0.5, 0.25, -0.5, 1.0], [1.0, -0.25, 0.5, 0.0]]]],
            dtype=torch.bfloat16,
        )
        weight = torch.tensor(
            [
                [0.5, 0.0, 0.5, -0.5],
                [0.0, 1.0, -0.5, 0.25],
                [1.0, -0.5, 0.0, 0.5],
                [-0.5, 0.5, 1.0, 0.0],
                [0.25, -1.0, 0.5, 1.0],
            ],
            dtype=torch.bfloat16,
        )
        markov = LowRankMarkovHead(vocab_size=5, hidden_size=4, rank=2).to(
            torch.bfloat16
        )
        with torch.no_grad():
            markov.predecessor_codebook.zero_()
            markov.successor_codebook.zero_()
            markov.hidden_projection.weight.zero_()
            markov.predecessor_codebook[1] = torch.tensor([1.0, 0.5])
            markov.predecessor_codebook[2] = torch.tensor([0.25, 1.0])
            markov.successor_codebook[3] = torch.tensor([0.5, -0.25])
            markov.hidden_projection.weight[:, :2] = torch.eye(2)
        return draft, teacher, weight, markov

    def test_reconstructs_teacher_logits_with_bf16_matmul(self):
        _, teacher, weight, _ = self._fixture()
        flat = teacher.reshape(-1, 4)
        with mock.patch("torch.mm", wraps=torch.mm) as matrix_multiply:
            actual = reconstruct_target_logits(flat, weight, vocab_chunk_size=2)
        expected = torch.mm(flat, weight.T).float()
        torch.testing.assert_close(actual, expected)
        self.assertTrue(matrix_multiply.called)
        self.assertTrue(
            all(call.args[0].dtype == call.args[1].dtype == torch.bfloat16 for call in matrix_multiply.call_args_list)
        )

    def test_full_vocab_l1_confidence_and_total_match_dense_reference(self):
        draft, teacher, weight, markov = self._fixture()
        target_ids = torch.tensor([[[2, 4]]])
        predecessors = torch.tensor([[[1, 2]]])
        confidence = torch.tensor([[[0.2, -0.4]]], requires_grad=True)
        mask = torch.ones_like(target_ids, dtype=torch.bool)
        result = compute_dspark_loss(
            draft_hidden=draft,
            target_hidden=teacher,
            target_ids=target_ids,
            predecessor_ids=predecessors,
            confidence_logits=confidence,
            lm_head_weight=weight,
            markov_head=markov,
            token_mask=mask,
            gamma=7.0,
            vocab_chunk_size=2,
        )

        flat_draft = draft.reshape(-1, 4)
        flat_teacher = teacher.reshape(-1, 4)
        dense_draft = torch.mm(flat_draft, weight.T).float()
        dense_draft = dense_draft + markov.score_chunk(
            flat_draft, predecessors.reshape(-1), 0, 5
        ).float()
        dense_teacher = torch.mm(flat_teacher, weight.T).float()
        q = dense_draft.softmax(-1)
        p = dense_teacher.softmax(-1)
        expected_l1 = (q - p).abs().sum(-1).reshape_as(target_ids)
        expected_confidence_target = (1.0 - 0.5 * expected_l1).clamp(0.0, 1.0)
        expected_ce = F.cross_entropy(
            dense_draft, target_ids.reshape(-1), reduction="none"
        ).reshape_as(target_ids)
        weights = torch.exp(-torch.arange(2, dtype=torch.float32) / 7.0).reshape(1, 1, 2)
        weighted = lambda value: (value * weights).sum() / weights.sum()
        expected_confidence = F.binary_cross_entropy_with_logits(
            confidence, expected_confidence_target, reduction="none"
        )
        expected_total = (
            0.1 * weighted(expected_ce)
            + 0.9 * weighted(expected_l1)
            + weighted(expected_confidence)
        )
        torch.testing.assert_close(result.l1_per_token, expected_l1, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            result.confidence_target, expected_confidence_target, atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(result.local_total, expected_total, atol=1e-6, rtol=1e-6)
        self.assertFalse(result.confidence_target.requires_grad)
        result.local_total.backward()
        self.assertIsNotNone(draft.grad)
        self.assertIsNotNone(confidence.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in markov.parameters()))

    def test_chunked_objective_invokes_markov_through_module_forward(self):
        draft, teacher, weight, markov = self._fixture()
        target_ids = torch.tensor([[[2, 4]]])
        predecessors = torch.tensor([[[1, 2]]])
        confidence = torch.zeros_like(target_ids, dtype=torch.float32)
        mask = torch.ones_like(target_ids, dtype=torch.bool)
        calls = []

        original = markov.forward

        def remember(*args, **kwargs):
            calls.append((args[2], args[3]))
            return original(*args, **kwargs)

        with mock.patch.object(markov, "forward", side_effect=remember):
            compute_dspark_loss(
                draft_hidden=draft,
                target_hidden=teacher,
                target_ids=target_ids,
                predecessor_ids=predecessors,
                confidence_logits=confidence,
                lm_head_weight=weight,
                markov_head=markov,
                token_mask=mask,
                gamma=7.0,
                vocab_chunk_size=2,
            )
        self.assertGreater(len(calls), 0)


if __name__ == "__main__":
    unittest.main()
