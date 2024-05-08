import pickle

import lmdb
import numpy as np
import torch
import tqdm
import os

from config_utils import parse_config
from data.lmdb_handler import LMDBDataHandler


from data.data_utils import get_dataloader_from_cfg, parse_batch  # noqa: E402
from models.get_models import get_model  # noqa: E402
from utility.cluster_logging import logger  # noqa: E402
from xai.explanations.explanation_manager import ExplanationsManager  # noqa: E402


def main(
    config_path: str,
):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, config_path)

    general_config = parse_config(config_path, project_root, True)
    general_config["debug"] = True

    generate_xai_masks(general_config)


def generate_xai_masks(cfg):
    logger.debug("Generating explanations")
    logger.debug(f"Using CUDA: {torch.cuda.is_available()}")
    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Using device: {cfg['device']}")
    cfg["method"] = "explain"
    cfg["data"]["num_workers"] = 0
    cfg["data"]["batch_size"] = 1

    cfg, data_loader = get_dataloader_from_cfg(cfg, loader_name="train")

    # load model
    model = get_model(cfg, self_trained=True).to(cfg["device"])
    model.eval()

    explanation_manager = ExplanationsManager(cfg, model, save=False)
    segmentation_handler_dict = _create_lmdb_handlers(cfg, explanation_manager)
    for batch in tqdm.tqdm(data_loader):
        (
            features,
            target,
            _,
            segments,
            idx,
            _,
        ) = parse_batch(batch)

        batch_dict = explanation_manager.explain_batch(batch, explain_all=False)
        _save_segmentations_to_lmdb(
            data_loader,
            batch_dict,
            segmentation_handler_dict,
            explanation_manager.explanations.keys(),
        )


def save_outputs(seg_lmdb_path: str, outputs, threshold: float) -> None:
    path = seg_lmdb_path + f"_{threshold}"
    seg_lmdb = lmdb.open(path, lock=False, map_size=int(1e12))

    with seg_lmdb.begin(write=True) as txn:
        for batch in outputs:
            for key, xai_mask in zip(batch[0], batch[1]):
                txn.put(key.encode(), pickle.dumps(xai_mask))


def post_process_output(output: torch.Tensor, batch_y):
    output = output.squeeze()
    batch_y = batch_y.squeeze()
    output = output.numpy(force=True)
    cl_sel = np.invert(batch_y.numpy(force=True).astype(bool))
    output[cl_sel, :, :] = 0
    # output = (output > threshold).astype(bool)
    # expand dimension to (batch, class_maps, h, w)
    output = np.expand_dims(output, axis=0)

    # move cam dimension to last dimension -> (batch, h, w, class_maps)
    output = np.moveaxis(output, 1, -1)

    return output


def _create_lmdb_handlers(cfg, explanation_manager):
    segmentation_handler_dict = {}
    base_path = f"/media/storagecube/jonasklotz/{cfg['experiment_name']}"
    os.makedirs(base_path, exist_ok=True)
    for explanation_method_name in explanation_manager.explanations.keys():
        lmdb_path = f"{base_path}/{explanation_method_name}.lmdb"
        logger.debug(f"Creating LMDB for {explanation_method_name} at {lmdb_path}")
        segmentation_handler_dict[explanation_method_name] = LMDBDataHandler(
            path=lmdb_path, write_only=True
        )
    return segmentation_handler_dict


def _save_segmentations_to_lmdb(
    data_loader, batch_dict, segmentation_handler_dict, explanation_method_names, dataset_name="BEN"
):
    for explanation_method_name in explanation_method_names:
        attribution_maps = batch_dict[f"a_{explanation_method_name}_data"]
        index = batch_dict["index_data"]
        batch_y = batch_dict["y_data"]
        for idx, write_index in enumerate(index):
            attr = attribution_maps[idx]
            label = batch_y[idx]
            if dataset_name == "BEN":
                patch_name = write_index
            else:
                write_index = write_index.item()
                patch_name = data_loader.dataset.get_patch_name(write_index)

            write_attribution_map = post_process_output(attr, batch_y=label)

            lmdb_handler = segmentation_handler_dict[explanation_method_name]

            with lmdb_handler.env.begin(write=True) as txn:
                txn.put(patch_name.encode("utf-8"), pickle.dumps(write_attribution_map))


if __name__ == "__main__":
    main(
        config_path="/home/jonasklotz/Studys/MASTERS/XAI/config/deepglobe_vgg_config.yml"
    )
