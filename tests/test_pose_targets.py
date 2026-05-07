from __future__ import annotations

import torch

from pose_targets import build_paf_targets, build_pcm_targets, decode_pcm_argmax


def test_build_pcm_targets_places_peak_near_keypoint() -> None:
    keypoints = torch.zeros(1, 17, 2)
    keypoints[0, 0] = torch.tensor([0.5, 0.5])

    pcm = build_pcm_targets(keypoints, heatmap_size=5, sigma=1.0)

    peak = pcm[0, 0].argmax()
    y = int(peak // 5)
    x = int(peak % 5)
    assert (x, y) == (2, 2)
    assert pcm.shape == (1, 17, 5, 5)


def test_build_paf_targets_encodes_horizontal_direction() -> None:
    keypoints = torch.zeros(1, 17, 2)
    keypoints[0, 5] = torch.tensor([0.0, 0.5])
    keypoints[0, 6] = torch.tensor([1.0, 0.5])

    paf = build_paf_targets(keypoints, heatmap_size=5, width=1.0, edges=((5, 6),))

    assert paf.shape == (1, 2, 5, 5)
    assert torch.isclose(paf[0, 0, 2, 2], torch.tensor(1.0))
    assert torch.isclose(paf[0, 1, 2, 2], torch.tensor(0.0))


def test_decode_pcm_argmax_returns_normalized_coordinates() -> None:
    pcm = torch.zeros(1, 17, 5, 5)
    pcm[0, 0, 3, 1] = 10.0

    keypoints = decode_pcm_argmax(pcm)

    assert keypoints.shape == (1, 17, 2)
    assert torch.allclose(keypoints[0, 0], torch.tensor([0.25, 0.75]))
