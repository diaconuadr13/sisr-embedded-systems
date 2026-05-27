import torch

from utils.metrics import calculate_temporal_consistency_error


def test_temporal_consistency_is_zero_for_identical_sequences() -> None:
    frames = torch.rand(3, 1, 8, 8)

    assert calculate_temporal_consistency_error(frames, frames.clone()) == 0.0


def test_temporal_consistency_measures_delta_error() -> None:
    hr = torch.zeros(3, 1, 2, 2)
    sr = hr.clone()
    sr[2] = 0.5

    assert calculate_temporal_consistency_error(sr, hr) == 0.25
