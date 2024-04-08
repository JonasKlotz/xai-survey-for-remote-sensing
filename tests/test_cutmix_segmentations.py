import numpy as np
import torch
import pytest

from training.augmentations import CutMix_segmentations


# A helper function to create a mock batch for testing
def create_mock_batch(batch_size, channels, height, width, num_classes):
    features = torch.rand(batch_size, channels, height, width)
    segmentations = torch.randint(0, num_classes - 1, (batch_size, 1, height, width))

    targets = torch.randint(0, 2, (batch_size, num_classes))
    return {"features": features, "segmentations": segmentations, "targets": targets}


@pytest.fixture
def mock_batch():
    return create_mock_batch(
        batch_size=2, channels=3, height=224, width=224, num_classes=6
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


# Run the test (this line is for illustration; typically, you'd run tests via the command line with pytest)
if __name__ == "__main__":
    pytest.main()
