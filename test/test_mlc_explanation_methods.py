import unittest

import numpy as np
import torch

from src.xai.xai_methods.gradcam_impl import GradCamImpl
from src.xai.xai_methods.ig_impl import IntegratedGradientsImpl
from src.xai.xai_methods.lime_impl import LimeImpl
from src.xai.xai_methods.lrp_impl import LRPImpl
from xai.explanation_main import create_model

# fix seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


def generate_pseudo_data(h: int, w: int, c: int, num_samples: int, ) -> torch.Tensor:
    return torch.from_numpy(np.random.rand(num_samples, c, h, w)).float()


def generate_pseudo_labels(num_samples: int, num_classes: int, num_dims: int = 1) -> torch.Tensor:
    return torch.from_numpy(np.random.randint(0, num_classes, size=(num_samples, num_dims))).long()


def generate_pseudo_segmentation(h: int, w: int, num_classes: int, num_samples: int) -> torch.Tensor:
    return torch.from_numpy(np.random.randint(0, num_classes, size=(num_samples, h, w))).long()


def save_np_array(array: np.ndarray, path: str, name: str):
    np.save(f"{path}/{name}.npy", array)


def load_np_array(path: str, name: str) -> np.ndarray:
    return np.load(f"{path}/{name}.npy")


def save_tensor_as_np(tensor: torch.Tensor, path: str, name: str):
    save_np_array(tensor.cpu().detach().numpy(), path, name)


def load_tensor_from_np(path: str, name: str) -> torch.Tensor:
    return torch.from_numpy(load_np_array(path, name)).float()


class MockModel(torch.nn.Module):

    def __init__(self, num_classes: int):
        super(MockModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(100, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, num_classes),

        )
        # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, inp):
        x = self.layers(inp)
        # x = self.softmax(x)
        return x


class TestExplanationMethods(unittest.TestCase):

    def setUp(self):
        self.num_classes = 10
        self.num_samples = 2
        self.h = 100
        self.w = 100
        self.c = 1

        self.model = create_model()  # MockModel(num_classes=self.num_classes)

        self.Lime = LimeImpl(self.model)
        # todo gradcam has problems as no layer is specified
        # self.GradCam = GradCamImpl(self.model)
        self.LRP = LRPImpl(self.model)
        self.IG = IntegratedGradientsImpl(self.model)

        self.image_tensor = generate_pseudo_data(h=self.h, w=self.w, c=self.c, num_samples=self.num_samples)
        self.labels = generate_pseudo_labels(num_samples=self.num_samples, num_classes=self.num_classes)
        self.segmentation = generate_pseudo_segmentation(h=self.h, w=self.w, num_classes=self.num_classes,
                                                         num_samples=self.num_samples)

    def test_slc_explanations(self):
        # for sample in range(self.num_samples):
        # self.Lime.explain(self.image_tensor[sample], target=self.labels[sample])
        # self.GradCam.explain(self.image_tensor[sample], target=self.labels[sample])
        self.LRP.explain(self.image_tensor, target=self.labels)
        self.IG.explain(self.image_tensor, target=self.labels)
