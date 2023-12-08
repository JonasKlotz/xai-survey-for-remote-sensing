import os

import torch
import yaml

from data.get_data_modules import load_data_module
from data.zarr_handler import load_most_recent_batches
from models.lightningresnet import LightningResnet
from xai.generate_explanations import generate_explanations
from xai.metrics.metrics_manager import MetricsManager
from xai.xai_methods.deeplift_impl import DeepLiftImpl

device_string = "gpu" if torch.cuda.is_available() else "cpu"


def evaluate_explanation_methods(
    explanations_config: dict, load_precomputed: bool = True
):
    """
    Evaluate Explanation Methods

    """
    if not load_precomputed:
        generate_explanations(explanations_config)

    all_zarrs = load_most_recent_batches(results_dir=explanations_config["results_dir"])

    x_batch = all_zarrs["x_batch"]
    y_batch = all_zarrs["y_batch"]
    # test batch
    a_batch = all_zarrs["a_batch_deeplift"]

    # convert zarr to numpy
    x_batch = x_batch[:]

    y_batch = y_batch[:]
    # convert to int
    y_batch = y_batch.astype(int)
    a_batch = a_batch[:]
    # generate s batch from x_batch by thresholding everything <=0 to 0 and everything >0 to 1
    s_batch = x_batch.copy()
    s_batch[s_batch <= 0] = 0
    s_batch[s_batch > 0] = 1

    print(
        f"x_batch shape: {x_batch.shape} \n"
        f"y_batch shape: {y_batch.shape}\n"
        f"a_batch shape: {a_batch.shape}"
    )

    data_module = load_data_module(explanations_config["dataset_name"])
    # load model
    # todo: get model function must be improved
    model = LightningResnet(
        num_classes=data_module.num_classes, input_channels=data_module.dims[0]
    )

    model_name= f"resnet18_{explanations_config['dataset_name']}.pt"
    model_path = os.path.join(explanations_config["model_dir"], model_name)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    explanation = DeepLiftImpl(model)
    metrics_manager = MetricsManager(
        model=model,
        explanation=explanation,
        aggregate=True,
        device_string=device_string,
        log=True,
        log_dir=explanations_config["log_dir"],
    )

    all_results = metrics_manager.evaluate_batch(
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        s_batch=s_batch,
    )

    print(all_results)



def main():
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

    evaluate_explanation_methods(config["evaluations"])


if __name__ == "__main__":
    main()