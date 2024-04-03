import torch
import tqdm


from data.data_utils import get_dataloader_from_cfg, parse_batch  # noqa: E402
from training.augmentations import derive_labels
from utility.cluster_logging import logger  # noqa: E402
import optuna
from functools import partial


def calculate_threshold_for_xai_masks(cfg, num_trials=100):
    logger.debug("Calculating threshold for cutmix and segmentation")
    logger.debug(f"Using CUDA: {torch.cuda.is_available()}")

    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Using device: {cfg['device']}")

    cfg["method"] = "explain"
    cfg["data"]["num_workers"] = 1
    cfg["data"]["batch_size"] = 2

    assert cfg["mode"] == "cutmix", f"Mode is not cutmix, but {cfg['mode']}"
    assert cfg["explanation_methods"], "No explanation method provided"
    assert (
        len(cfg["explanation_methods"]) == 1
    ), "Only one explanation method is supported"

    cfg, data_loader = get_dataloader_from_cfg(cfg, loader_name="train")

    # Use functools.partial to create a new function that has cfg and additional_data fixed
    objective_with_config = partial(objective, cfg=cfg, data_loader=data_loader)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_with_config, n_trials=num_trials)

    print("Best trial:", study.best_trial.params)
    # write to a yaml file
    with open("cutmix_threshold.yaml", "w") as f:
        f.write(f"cutmix_threshold: {study.best_trial.params['cutmix_threshold']}\n")
        f.write(
            f"segmentation_threshold: {study.best_trial.params['segmentation_threshold']}\n"
        )


def objective(trial, cfg, data_loader):
    # Suggest values for the cutmix_threshold and segmentation_threshold
    cutmix_threshold = trial.suggest_int("cutmix_threshold", 5, 30)
    segmentation_threshold = trial.suggest_float("segmentation_threshold", 0.01, 0.3)

    # Setup the loss based on your cfg
    if cfg["task"] == "multilabel":
        loss = torch.nn.BCELoss()
    else:
        loss = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    for batch in tqdm.tqdm(data_loader):
        (
            features,
            target,
            _,
            segments,
            idx,
            _,
        ) = parse_batch(batch)
        segments = segments.squeeze()
        segmentations_mask = (segments > segmentation_threshold).bool()
        new_targets = derive_labels(
            old_labels=target,
            seg_masks=segmentations_mask,
            not_changed=[],
            threshold=cutmix_threshold,
        )

        if cfg["task"] == "multilabel":
            new_targets = new_targets.float()
        target = target.float()

        loss_value = loss(target, new_targets)
        total_loss += loss_value.item()

    average_loss = total_loss / len(data_loader)
    return average_loss
