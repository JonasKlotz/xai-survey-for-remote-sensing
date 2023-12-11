import yaml
from PIL import Image
from torchvision import transforms

from data.data_utils import get_loader_for_datamodule
from models.get_models import get_model
from src.data.get_data_modules import load_data_module
from src.xai.xai_methods.explanation_manager import ExplanationsManager
from utility.cluster_logging import logger


def load_test_imagenet_image(idx_to_labels, image_tensor):
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # inv_normalize = transforms.Normalize(
    #     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
    #     std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
    # )
    img = Image.open("/home/jonasklotz/Studys/MASTERS/XAI_PLAYGROUND/img/swan.jpeg")
    transformed_img = transform(img)
    image_tensor = transform_normalize(transformed_img)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def generate_explanations(cfg: dict):
    logger.debug("Generating explanations")
    # load datamodule
    data_module = load_data_module(cfg)
    test_loader = get_loader_for_datamodule(data_module)

    # load model
    model = get_model(
        cfg,
        num_classes=data_module.num_classes,
        input_channels=data_module.dims[0],
        pretrained=True,
    )

    batch = next(iter(test_loader))

    explanation_manager = ExplanationsManager(cfg, model)
    explanation_manager.explain_batch(batch)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/explanations_config.yml",
    )

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generate_explanations(config)
