import torch
import pytest

from data.data_utils import segmask_to_multilabel_torch


# Test cases
@pytest.mark.parametrize(
    "batch_size,num_classes,segmask,expected",
    [
        (1, 3, torch.tensor([[[[0, 1, 2]]]]), torch.tensor([[1, 1, 1]])),
        (
            2,
            5,
            torch.tensor([[[[0, 0, 0]], [[1, 1, 1]]], [[[2, 2, 2]], [[3, 3, 3]]]]),
            torch.tensor([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0]]),
        ),
        (
            3,
            4,
            torch.tensor([[[[0]]], [[[1]]], [[[2]]]]),
            torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
        ),
        (
            1,
            2,
            torch.tensor([[[[0, 0, 0]]]]),
            torch.tensor([[1, 0]]),
        ),  # Edge case: Only one class present
        (
            1,
            2,
            torch.tensor([[[[1, 1, 1]]]]),
            torch.tensor([[0, 1]]),
        ),  # Edge case: Only one class present (not class 0)
    ],
)
def test_segmask_to_multilabel_torch(batch_size, num_classes, segmask, expected):
    result = segmask_to_multilabel_torch(segmask, num_classes)
    assert torch.equal(
        result, expected
    ), f"Failed for batch_size={batch_size}, num_classes={num_classes}"
