# import pytest
import torch


def test_segmentations_to_relevancy_map():
    # generate a batch of segmentations
    batch_size = 1
    num_classes = 5
    s_batch = torch.randint(0, num_classes, (batch_size, 1, 10, 10))

    # generate corresponding labels only if the segmentation contains that class
    y_batch = torch.zeros((batch_size, num_classes))

    for i in range(batch_size):
        for j in range(num_classes):
            if torch.any(s_batch[i] == j):
                y_batch[i, j] = 1

    print(s_batch.shape)
    print(y_batch)
