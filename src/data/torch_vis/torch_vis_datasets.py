from typing import Union, List

import torchvision.datasets as datasets
import torch
import torch.nn.functional as f


class Caltech101Dataset:
    def __init__(
        self,
        dataset_path,
        target_type: Union[List[str], str] = None,
        transform=None,
        target_transform=None,
        download=True,
    ):
        self.root = dataset_path
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if target_type is None:
            target_type = ["category"]  # , 'annotation']
            self.return_several_targets = False

        self.target_type = target_type
        self.dataset = datasets.Caltech101(
            self.root, self.target_type, download=self.download
        )

    def __getitem__(self, index):
        """Get item at position idx of Dataset.

        Parameters
        ----------
        index

        Returns
        -------
        tuple
            (image, *target) where target is a tuple of the target_type. Depending on the target_type,
            the tuple can be of different length.
        """

        features, targets = self.dataset[index]
        # some images (e.g. class car side) are grayscale, convert them to RGB
        if features.mode != "RGB":
            features = features.convert("RGB")

        if self.transform:
            features = self.transform(features)
        if self.return_several_targets:
            # todo: one hot enc necessary?
            targets = (
                f.one_hot(torch.tensor(targets[0]), num_classes=101),
                torch.tensor(targets[1]),
            )
        else:
            targets = torch.tensor(targets)

        # todo: handle annotations
        return features, targets, index

    def __len__(self):
        return len(self.dataset)
