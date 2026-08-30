from __future__ import annotations

import torch

from omni_sd.thinker_mrope import extend_response_position_ids, validate_position_ids


def test_text_response_positions_continue_after_multimodal_maximum():
    prompt = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 1, 9, 10],
            [0, 1, 5, 6],
        ],
        dtype=torch.int64,
    )
    positions = extend_response_position_ids(prompt, response_tokens=3)
    assert positions.shape == (7, 3)
    assert positions[:4].T.tolist() == prompt.tolist()
    assert positions[4:].tolist() == [[11, 11, 11], [12, 12, 12], [13, 13, 13]]


def test_position_contract_rejects_wrong_axis_or_dtype():
    validate_position_ids(torch.zeros(4, 3, dtype=torch.int64), tokens=4)
    for invalid in (
        torch.zeros(3, 4, dtype=torch.int64),
        torch.zeros(4, 3, dtype=torch.int32),
    ):
        try:
            validate_position_ids(invalid, tokens=4)
        except ValueError as exc:
            assert "position" in str(exc)
        else:
            raise AssertionError("invalid mRoPE positions were accepted")
