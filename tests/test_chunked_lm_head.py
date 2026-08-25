from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from glm_dflash2.chunked_lm_head import chunked_lm_projection


class ChunkedLmHeadTest(unittest.TestCase):
    def _compare(self, token_chunk: int, vocab_chunk: int) -> None:
        torch.manual_seed(4)
        hidden = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(19, 7, dtype=torch.float64)
        targets = torch.randint(0, 19, (2, 3))
        mask = torch.tensor([[True, False, True], [True, True, False]])

        dense_hidden = hidden.detach().clone().requires_grad_(True)
        dense_logits = dense_hidden @ weight.T
        dense_nll = F.cross_entropy(
            dense_logits.reshape(-1, 19), targets.reshape(-1), reduction="none"
        ).reshape_as(targets)
        dense_values, dense_ids = dense_logits.topk(5, dim=-1)
        dense_loss = dense_nll[mask].sum()
        dense_loss.backward()

        result = chunked_lm_projection(
            hidden,
            targets,
            weight,
            top_k=5,
            token_chunk_size=token_chunk,
            vocab_chunk_size=vocab_chunk,
            token_mask=mask,
        )
        result.nll.sum().backward()
        torch.testing.assert_close(result.nll[mask], dense_nll[mask], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            result.topk_scores[mask], dense_values[mask], rtol=1e-12, atol=1e-12
        )
        self.assertTrue(torch.equal(result.topk_ids[mask], dense_ids[mask]))
        self.assertTrue(torch.equal(result.nll[~mask], torch.zeros_like(result.nll[~mask])))
        self.assertTrue(torch.isneginf(result.topk_scores[~mask]).all())
        torch.testing.assert_close(hidden.grad, dense_hidden.grad, rtol=1e-12, atol=1e-12)

    def test_matches_dense_across_chunk_shapes(self):
        for token_chunk, vocab_chunk in ((1, 1), (2, 4), (7, 8), (64, 64)):
            with self.subTest(token_chunk=token_chunk, vocab_chunk=vocab_chunk):
                self._compare(token_chunk, vocab_chunk)

    def test_bfloat16_values_match_dense_fp32_reduction(self):
        torch.manual_seed(9)
        hidden = torch.randn(5, 8, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(23, 8, dtype=torch.bfloat16)
        targets = torch.tensor([0, 5, 9, 12, 22])
        result = chunked_lm_projection(
            hidden,
            targets,
            weight,
            top_k=4,
            token_chunk_size=2,
            vocab_chunk_size=7,
        )
        logits = (hidden @ weight.T).float()
        expected = F.cross_entropy(logits, targets, reduction="none")
        torch.testing.assert_close(result.nll, expected, rtol=3e-3, atol=3e-3)
        torch.testing.assert_close(result.topk_scores, logits.topk(4, -1).values, rtol=0, atol=0)
        self.assertTrue(torch.equal(result.topk_ids, logits.topk(4, -1).indices))

    def test_ties_preserve_exact_scores_and_valid_ids(self):
        hidden = torch.tensor([[1.0, 0.0]])
        weight = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.5, 0.0], [0.5, 0.0]])
        result = chunked_lm_projection(
            hidden,
            torch.tensor([0]),
            weight,
            top_k=3,
            token_chunk_size=1,
            vocab_chunk_size=2,
        )
        self.assertEqual(result.topk_scores.tolist(), [[1.0, 1.0, 0.5]])
        self.assertEqual(set(result.topk_ids[0, :2].tolist()), {0, 1})
        self.assertIn(result.topk_ids[0, 2].item(), {2, 3})

    def test_rejects_invalid_arguments(self):
        hidden = torch.zeros(2, 3)
        weight = torch.zeros(4, 3)
        with self.assertRaises(ValueError):
            chunked_lm_projection(hidden, torch.zeros(2, dtype=torch.long), weight, top_k=5)
        with self.assertRaises(ValueError):
            chunked_lm_projection(hidden, torch.zeros(3, dtype=torch.long), weight, top_k=2)


if __name__ == "__main__":
    unittest.main()
