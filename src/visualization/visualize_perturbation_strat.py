import torch

from config_utils import parse_config

from visualization.explanation_visualizer import ExplanationVisualizer
from data.data_utils import get_index_to_name
from models.get_models import get_model
import pytorch_lightning

pytorch_lightning.seed_everything(42)


def main():
    project_root = "/home/jonasklotz/Studys/MASTERS/XAI"

    config_path = "/home/jonasklotz/Studys/MASTERS/XAI/config/deepglobe_vgg_config.yml"

    cfg = parse_config(config_path, project_root)
    cfg["data"]["batch_size"] = 1
    cfg["data"]["num_workers"] = 0
    cfg["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #
    # data_loader = get_zarr_dataloader(
    #     cfg,
    # )
    # _ = data_loader.zarr_keys

    model = get_model(cfg, self_trained=True)
    # model = model.double()
    model.eval()
    index2name = get_index_to_name(cfg)
    explanation_visualizer = ExplanationVisualizer(cfg, index2name)

    explanation_visualizer.output_path = (
        "/home/jonasklotz/Studys/MASTERS/results_22_4_final/perturbation"
    )
    save_path = "/home/jonasklotz/Studys/MASTERS/results_22_4_final/perturbation"

    methods = ["GradCAM", "DeepLIFT"]
    # samples = ["2039", "2094"]
    samples = ["2094"]
    show = False
    for method in methods:
        for sample_name in samples:
            explanation_visualizer.output_path = f"/home/jonasklotz/Studys/MASTERS/results_22_4_final/perturbation/{sample_name}"

            batch_path = save_path + f"/batch_dict_sample_{sample_name}.pt"

            (
                attributions_dict,
                image_tensor,
                index_tensor,
                true_labels,
                segments_tensor,
            ) = load_batch_dict(batch_path, method)

            k_list = (
                [
                    0.05,
                    0.1,
                    0.2,
                    0.5,
                    0.9,
                ]
                if method == "GradCAM"
                else [0.05, 0.1, 0.2, 0.5, 0.9, 1.0]
            )
            plot_all(
                attributions_dict,
                explanation_visualizer,
                image_tensor,
                index_tensor,
                k_list,
                method,
                model,
                segments_tensor,
                show,
                true_labels,
            )


def plot_all(
    attributions_dict,
    explanation_visualizer,
    image_tensor,
    index_tensor,
    k_list,
    method,
    model,
    segments_tensor,
    show,
    true_labels,
):
    explanation_visualizer.visualize_top_k_attributions_with_predictions(
        attribution_dict=attributions_dict,
        image_tensor=image_tensor,
        label_tensor=true_labels,
        segmentation_tensor=segments_tensor,
        predictions_tensor=true_labels,
        show=show,
        k_list=k_list,
        removal_strategy="MORF",
        model=model,
        title=f"Top k attributions removed from the Image (MORF)({method})",
        save_name=f"Sample {index_tensor}: ",
    )
    explanation_visualizer.visualize_top_k_attributions_with_predictions(
        attribution_dict=attributions_dict,
        image_tensor=image_tensor,
        label_tensor=true_labels,
        segmentation_tensor=segments_tensor,
        predictions_tensor=true_labels,
        show=show,
        k_list=k_list,
        removal_strategy="MORF_BASELINE",
        model=model,
        title=f"Only Top k attributions present in the Image (MORF with baseline) ({method})",
        save_name=f"Sample {index_tensor}: ",
    )
    explanation_visualizer.visualize_top_k_attributions_with_predictions(
        attribution_dict=attributions_dict,
        image_tensor=image_tensor,
        label_tensor=true_labels,
        segmentation_tensor=segments_tensor,
        predictions_tensor=true_labels,
        show=show,
        k_list=k_list,
        removal_strategy="LERF",
        model=model,
        title=f"Bottom k attributions removed from the Image (LERF) ({method})",
        save_name=f"Sample {index_tensor}: ",
    )
    explanation_visualizer.visualize_top_k_predictions_function(
        attribution_dict=attributions_dict,
        image_tensor=image_tensor,
        label_tensor=true_labels,
        k_list=k_list,
        removal_strategy="LERF",
        model=model,
        title=f"Prediction Curve: Bottom k attributions removed from the Image (LERF) ({method})",
        save_name=f"Sample {index_tensor}: ",
        show=show,
    )
    explanation_visualizer.visualize_top_k_predictions_function(
        attribution_dict=attributions_dict,
        image_tensor=image_tensor,
        label_tensor=true_labels,
        k_list=k_list,
        removal_strategy="MORF",
        model=model,
        title=f"Prediction Curve: Top k attributions removed from the Image (MORF) ({method})",
        save_name=f"Sample {index_tensor}: ",
        show=show,
    )
    explanation_visualizer.visualize_top_k_predictions_function(
        attribution_dict=attributions_dict,
        image_tensor=image_tensor,
        label_tensor=true_labels,
        k_list=k_list,
        removal_strategy="MORF_BASELINE",
        model=model,
        title=f"Prediction Curve: Only Top k attributions present in the Image (MORF with baseline) ({method})",
        save_name=f"Sample {index_tensor}: ",
        show=show,
    )


def load_batch_dict(batch_path, method):
    batch_dict = torch.load(batch_path, map_location=torch.device("cpu"))
    batch_dict = {k: torch.tensor(v) for k, v in batch_dict.items()}
    image_tensor = batch_dict.pop("x_data")[0]
    true_labels = batch_dict.pop("y_data")[0]
    # predicted_label_tensor = batch_dict.pop("y_pred_data")[0]
    segments_tensor = batch_dict.pop("s_data")[0]
    index_tensor = batch_dict.pop("index_data")[0]
    attributions_dict = batch_dict
    # only take a_gradcam_data from attribution dict
    # attributions_dict = {
    #     k: v[0] for k, v in attributions_dict.items() if k == "a_deeplift_data"
    # }
    # #filter the attribution by only taking the predicted label
    # filter the attribution by only taking the predicted label
    # method = "DeepLIFT"
    key = f"a_{method.lower()}_data"
    attributions_dict = {k: v[0] for k, v in attributions_dict.items() if k == key}
    return attributions_dict, image_tensor, index_tensor, true_labels, segments_tensor


if __name__ == "__main__":
    main()
