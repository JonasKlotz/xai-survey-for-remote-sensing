import torchvision.datasets as datasets
import torch
from PIL import ImageDraw
import os
import os.path
from typing import Any, List, Tuple, Union

from PIL import Image
from torchvision.transforms import transforms


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
            target_type = ["category", "annotation"]
            self.return_several_targets = True

        self.target_type = target_type
        self.dataset = datasets.Caltech101(
            self.root, self.target_type, download=self.download
        )
        # monkey patch the __getitem__ method to add the bbox to the target tuple
        self.dataset.__getitem__ = caltech_getitem

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

        features, target_tuple = caltech_getitem(self.dataset, index)
        label = torch.tensor(target_tuple[0])
        contour = target_tuple[1]
        bbox = target_tuple[2]

        segmentation_mask = generate_segmentation_mask_caltech(features, contour, bbox)

        # some images (e.g. class car side) are grayscale, convert them to RGB
        if features.mode != "RGB":
            features = features.convert("RGB")

        if self.transform:
            features = self.transform(features)

        if self.target_transform:
            segmentation_mask = self.target_transform(segmentation_mask)

        else:
            target_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((224, 224))]
            )
            segmentation_mask = target_transforms(segmentation_mask)

        return {
            "features": features,
            "targets": label,
            "index": index,
            "segmentations": segmentation_mask,
        }

    def __len__(self):
        return len(self.dataset)


def caltech_getitem(self, index: int) -> Tuple[Any, Any]:
    """Monkey patch for the __getitem__ method of the Caltech101 dataset.

    The caltech annotations usually contain bbox and object contour information, where the object contour is relative
    to the bbox. However, torchvision only provides the object contour, which is useless without the bbox.
     This method adds the bbox to the target tuple.

     See https://github.com/pytorch/vision/issues/7748

    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where the type of target specified by target_type.
    """
    import scipy.io

    img = Image.open(
        os.path.join(
            self.root,
            "101_ObjectCategories",
            self.categories[self.y[index]],
            f"image_{self.index[index]:04d}.jpg",
        )
    )

    target: Any = []
    for t in self.target_type:
        if t == "category":
            target.append(self.y[index])
        elif t == "annotation":
            data = scipy.io.loadmat(
                os.path.join(
                    self.root,
                    "Annotations",
                    self.annotation_categories[self.y[index]],
                    f"annotation_{self.index[index]:04d}.mat",
                )
            )
            target.append(data["obj_contour"])
            target.append(data["box_coord"])
    target = tuple(target) if len(target) > 1 else target[0]

    if self.transform is not None:
        img = self.transform(img)

    if self.target_transform is not None:
        target = self.target_transform(target)

    return img, target


def generate_segmentation_mask_caltech(image, contour, bbox):
    """Generate a segmentation mask from the object contour and bbox."""
    contour += bbox.T[[2, 0], :]
    # create new black image with the same size ( greyscale)
    greyscale_image = Image.new("L", image.size, 0)

    draw = ImageDraw.Draw(greyscale_image)
    draw.polygon(
        list(map(tuple, contour.T.astype("int64").tolist())),
        outline="white",
        fill="white",
    )
    return greyscale_image
