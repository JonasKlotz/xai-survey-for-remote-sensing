import numpy as np
import torch
import pytest

from training.augmentations import CutMix_segmentations
from src.training.dataset_sanity_checker import calculate_segmentation_label_mse_loss


def create_mock_batch(batch_size, channels, height, width, num_classes):
    features = torch.rand(batch_size, channels, height, width)
    # Ensure at least one label per batch item
    targets = torch.zeros(batch_size, num_classes).long()
    targets[torch.arange(batch_size), torch.randint(0, num_classes, (batch_size,))] = 1
    targets += (
        torch.rand(batch_size, num_classes) < 0.3
    ).long()  # Add more labels randomly
    # clip at 1,
    targets = torch.clamp(targets, 0, 1)

    segmentations = torch.zeros((batch_size, height, width), dtype=torch.long)
    for i in range(batch_size):
        class_indices = targets[i].nonzero(as_tuple=False).squeeze(1)
        pixels_per_class = (height * width) // class_indices.size(0)
        remaining_pixels = (height * width) % class_indices.size(0)

        pixel_assignments = torch.cat(
            [
                torch.full((pixels_per_class,), class_index, dtype=torch.long)
                for class_index in class_indices
            ]
        )
        if remaining_pixels > 0:
            pixel_assignments = torch.cat(
                (pixel_assignments, class_indices[:remaining_pixels])
            )

        segmentations[i] = pixel_assignments[torch.randperm(height * width)].view(
            height, width
        )

    return {"features": features, "segmentations": segmentations, "targets": targets}


@pytest.fixture
def mock_batch():
    return create_mock_batch(
        batch_size=2, channels=3, height=10, width=10, num_classes=6
    )


def test_CutMix_segmentations(mock_batch):
    # Control randomness for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Call the function under test
    augmented_batch = CutMix_segmentations(
        mock_batch, max_aug_area=0.5, min_aug_area=0.1, aug_p=1.0, overhead=10
    )

    # Assertions to verify the expected behavior
    # Ensure the shapes of the augmented features and segmentations remain unchanged
    assert augmented_batch["features"].shape == mock_batch["features"].shape
    assert augmented_batch["segmentations"].shape == mock_batch["segmentations"].shape

    # Additional checks can include verifying that certain areas of the features/segmentations are modified as expected
    # and that the targets are correctly updated based on the augmentation. This may require a more detailed setup
    # of mock return values and understanding of the expected outcome based on those mocks.


def test_check_labels_and_segments(mock_batch):
    # Create a mock batch and labels tensor
    segmentations = mock_batch["segmentations"]
    targets = mock_batch["targets"]

    # Check if the function correctly identifies that the labels are the same in the segmentation mask
    x = calculate_segmentation_label_mse_loss(segmentations, targets)
    assert x == 0.0  # Since the labels are the same as the segmentation mask


if __name__ == "__main__":
    pytest.main()
