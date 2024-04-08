import tqdm
import torch
import torch.nn.functional as F

from data.data_utils import get_dataloader_from_cfg, parse_batch  # noqa: E402
from utility.cluster_logging import logger  # noqa: E402


def sanity_check_labels_and_segmasks(cfg, loader_name="train"):
    logger.debug("Calculating threshold for cutmix and segmentation")
    logger.debug(f"Using CUDA: {torch.cuda.is_available()}")

    cfg, data_loader = get_dataloader_from_cfg(cfg, loader_name=loader_name)
    total_loss = 0
    pbar = tqdm.tqdm(total=len(data_loader), desc="Initial Loss Calculation")
    first_batch = True
    for batch in data_loader:
        (
            features,
            target,
            _,
            segments,
            idx,
            _,
        ) = parse_batch(batch)
        if first_batch:
            logger.debug(f"Features shape: {features.shape}")
            logger.debug(f"Target shape: {target.shape}")
            logger.debug(f"Segments shape: {segments.shape}")
            first_batch = False

        seg_label_mse = calculate_segmentation_label_mse_loss(segments, target)

        total_loss += seg_label_mse
        current_avg_loss = total_loss / (pbar.n + 1)

        # Update tqdm description with the current average loss
        pbar.set_description(f"Average MSE Loss: {current_avg_loss:.4f}")
        pbar.update(1)  # Manually update the progress bar by one step

    pbar.close()  # Ensure to close the progress bar to clean up properly

    threshold = total_loss / len(data_loader)
    logger.debug(f"Error for labels and segmentation: {threshold}")


def calculate_segmentation_label_mse_loss(
    segmentations: torch.Tensor, labels: torch.Tensor
) -> float:
    """
    Optimized calculation of the Mean Squared Error (MSE) loss between transformed segmentations and labels,
    removing the explicit for loops for efficiency.

    Parameters
    ----------
    segmentations : torch.Tensor
        Segmentation mask with shape (batch_size, h, w), where each pixel value indicates the class index.
    labels : torch.Tensor
        Binary multi-label tensor with shape (batch_size, num_classes), indicating the presence or absence
        of each class in the corresponding image.

    Returns
    -------
    float
        The MSE loss indicating the discrepancy between the transformed segmentations and the actual labels.
    """

    batch_size, num_classes = labels.shape

    # Create a range tensor of class indices [0, num_classes-1]
    class_indices = torch.arange(num_classes, device=segmentations.device).view(
        1, num_classes, 1, 1
    )

    # Add a singleton channel dimension to segmentations to match class_indices dimensions for broadcasting
    segmentations_with_channel = segmentations.unsqueeze(1)

    # Expand segmentations to have the same shape as class_indices for comparison
    expanded_segmentations = segmentations_with_channel.expand(-1, num_classes, -1, -1)

    # Check class presence in segmentations; result is a binary tensor of shape (batch_size, num_classes)
    class_presence = (expanded_segmentations == class_indices).any(dim=(2, 3)).float()

    # Calculate the MSE loss between the binary representation of the segmentations and the labels
    mse_loss = F.mse_loss(class_presence, labels.float())

    return mse_loss.item()
