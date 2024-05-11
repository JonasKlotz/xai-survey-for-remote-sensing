import torch

import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, f1_score
from data.data_utils import (
    get_dataloader_from_cfg,
    parse_batch,
    load_data_module,
)
from models.get_models import get_model
from utility.cluster_logging import logger
from visualization.dataset_visualizations import plot_distribution
from visualization.plot_metrics.plot_helpers import save_fig


def calculate_classification_thresholds(cfg: dict):
    logger.debug("Generating explanations")
    logger.debug(f"Using CUDA: {torch.cuda.is_available()}")
    logger.debug(f"Using device: {cfg['device']}")
    cfg["method"] = "train"  # to enable logging
    cfg["data"]["num_workers"] = 0  # for debugging

    cfg["results_path"] = f"/media/storagecube/jonasklotz/results/{cfg["dataset_name"]}"
    logger.debug(f"Updated for Results path: {cfg['results_path']}")

    cfg, data_loader = get_dataloader_from_cfg(cfg, loader_name="train")
    plots_dir = os.path.join(cfg["results_path"], cfg["experiment_name"], "plots")
    # load model
    model = get_model(cfg, self_trained=True).to(cfg["device"])
    model.eval()
    t_path = os.path.join(
        "/media/storagecube/jonasklotz/results/ben/ben_vgg_2024-05-11_14-04-30/plots",
        "ben_train_optimal_thresholds.txt",
    )

    # calculate_optimal_classification_threshold(cfg, data_loader, model, plots_dir, t_path)

    data_module, cfg = load_data_module(cfg)

    from pytorch_lightning import Trainer

    # init trainer
    trainer = Trainer(
        default_root_dir=cfg["training_root_path"],
        max_epochs=cfg["max_epochs"],
        accelerator="auto",
        gradient_clip_val=1,
        # devices=[gpu],
        inference_mode=True,  # we always want gradients for RRR
        # log_every_n_steps=20,
    )
    trainer.test(model, datamodule=data_module)

    # read the optimal thresholds
    optimal_thresholds_loaded = np.loadtxt(t_path)
    model.set_classification_thresholds(optimal_thresholds_loaded)
    trainer.test(model, datamodule=data_module)


def calculate_optimal_classification_threshold(
    cfg, data_loader, model, plots_dir, t_path="optimal_thresholds.txt"
):
    # Initialize lists to store predictions and targets
    all_predictions = []
    all_targets = []
    i = 0
    # Iterate over data loader
    for batch in tqdm.tqdm(data_loader):
        (
            features,
            target,
            _,
            segments,
            idx,
            _,
        ) = parse_batch(batch)

        features = features.to(model.device, dtype=model.dtype)

        # Ensure that the model and the input are on the same device
        predictions, logits = model.prediction_step(features)

        # Store predictions and targets
        all_predictions.append(logits.numpy(force=True))
        all_targets.append(target.numpy(force=True))
        # if i > 10:
        #     break
        i += 1
    # Convert lists to numpy arrays
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    # sum all targets along dim 0
    all_targets_sum = np.sum(all_targets, axis=0)
    fig = plot_distribution(cfg, torch.from_numpy(all_targets_sum))
    # save plotly figure
    save_fig(fig, "Ben Elbow", plots_dir)
    # Calculate optimal thresholds and save plots
    optimal_thresholds = calculate_optimal_thresholds_and_save_plots(
        all_predictions, all_targets, plot_dir=plots_dir
    )
    logger.debug(f"Optimal: {optimal_thresholds}")
    # Write thresholds to a file
    np.savetxt(
        t_path,
        optimal_thresholds,
        fmt="%0.4f",
        header="Optimal thresholds for each class",
    )
    logger.debug(
        f"Optimal thresholds saved to '{plots_dir}/optimal_thresholds.txt' and plots saved to '{plots_dir}' directory."
    )


def calculate_optimal_thresholds_and_save_plots(predictions, targets, plot_dir="plots"):
    num_classes = targets.shape[1]
    optimal_thresholds = np.zeros(num_classes)

    # Create directory for plots if it doesn't exist
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for i in range(num_classes):
        # Skip classes with no positive examples
        if np.sum(targets[:, i]) == 0:
            logger.warn(
                f"No positive samples found for class {i + 1}. Skipping threshold calculation."
            )
            optimal_thresholds[i] = 0.5  # or set a default value
            continue
        thresholds = np.linspace(0, 1, 100)
        f1_scores = [
            f1_score(
                targets[:, i],
                (predictions[:, i] > threshold).astype(int),
                average="weighted",
            )
            for threshold in thresholds
        ]
        optimal_index = np.argmax(f1_scores)
        optimal_thresholds[i] = thresholds[optimal_index]

        # Save the plot to a file
        plot_elbow(f1_scores, i, optimal_index, plot_dir, thresholds)

    return optimal_thresholds


def plot_elbow(
    ap_scores, i, optimal_index, plot_dir, thresholds, metric_name="Average Precision"
):
    plt.figure()
    plt.plot(thresholds, ap_scores, label=f"Class {i + 1}")
    plt.scatter(
        thresholds[optimal_index], ap_scores[optimal_index], color="red"
    )  # mark the optimal point
    plt.title(f"{metric_name} Curve for Class {i + 1}")
    plt.xlabel("Threshold")
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(f"{plot_dir}/AP_Class_{i + 1}.png")
    # also save the data to a file
    np.savetxt(
        f"{plot_dir}/AP_Class_{i + 1}.txt",
        np.vstack((thresholds, ap_scores)).T,
        fmt="%0.4f",
        header=f"Threshold, {metric_name}",
    )
    plt.close()
