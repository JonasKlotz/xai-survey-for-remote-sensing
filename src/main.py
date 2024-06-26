import os
import sys

import typer
from typing_extensions import Annotated

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(project_root)
print(f"Added {project_root} to path.")
quantus_path = os.path.join(project_root, "src/xai/metrics/Quantus")
sys.path.append(quantus_path)
print(f"Added {quantus_path} to path.")

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def run_training(
    config_path: str,
    random_seed: Annotated[int, typer.Option()] = 42,
    debug: Annotated[bool, typer.Option()] = False,
    explanation_method: Annotated[str, typer.Option()] = None,
    gpu: Annotated[int, typer.Option()] = 3,
    mode: Annotated[str, typer.Option()] = "normal",
    model_name: Annotated[str, typer.Option()] = "vgg",
    tune: Annotated[bool, typer.Option()] = False,
    normal_segmentations: Annotated[bool, typer.Option()] = False,
    rrr_lambda: Annotated[float, typer.Option()] = 1.0,
    rrr_distance: Annotated[str, typer.Option()] = "elementwise",
    epochs: Annotated[int, typer.Option()] = 20,
    min_aug_area: Annotated[float, typer.Option()] = 0.1,
    max_aug_area: Annotated[float, typer.Option()] = 0.5,
    visualize_after_training: Annotated[bool, typer.Option()] = False,
):
    from config_utils import (
        setup_everything,
        create_training_experiment_and_group_names,
    )

    general_config = setup_everything(
        config_path=config_path,
        random_seed=random_seed,
        project_root=project_root,
        debug=debug,
        explanation_method=explanation_method,
        gpu=gpu,
        model_name=model_name,
        max_epochs=epochs,
        max_aug_area=max_aug_area,
        min_aug_area=min_aug_area,
        rrr_lambda=rrr_lambda,
        mode=mode,
        rrr_distance=rrr_distance,
        normal_segmentations=normal_segmentations,
        visualize_after_training=visualize_after_training,
    )
    assert mode in ["normal", "cutmix", "rrr"], f"Mode {mode} not supported."

    general_config = create_training_experiment_and_group_names(general_config)

    from training.train import train

    train(general_config, tune=tune)


@app.command()
def run_generate_explanations(
    config_path: str,
    random_seed: Annotated[int, typer.Option()] = 42,
    debug: Annotated[bool, typer.Option()] = False,
    explanation_method: Annotated[str, typer.Option()] = None,
    gpu: Annotated[int, typer.Option()] = 3,
    mode: Annotated[str, typer.Option()] = "normal",
):
    from config_utils import setup_everything

    general_config = setup_everything(
        config_path=config_path,
        random_seed=random_seed,
        project_root=project_root,
        debug=debug,
        explanation_method=explanation_method,
        gpu=gpu,
    )

    if mode == "normal":
        from xai.explanations.generate_explanations import generate_explanations

        generate_explanations(general_config)

    elif mode == "cutmix":
        from cutmix.create_xai_masks import generate_xai_masks

        generate_xai_masks(general_config)

    else:
        raise ValueError(f"Mode {mode} not supported.")


@app.command()
def run_calculate_classification_thresholds(
    config_path: str,
    random_seed: Annotated[int, typer.Option()] = 42,
    debug: Annotated[bool, typer.Option()] = False,
    explanation_method: Annotated[str, typer.Option()] = None,
    gpu: Annotated[int, typer.Option()] = 3,
):
    from config_utils import setup_everything

    general_config = setup_everything(
        config_path=config_path,
        random_seed=random_seed,
        project_root=project_root,
        debug=debug,
        explanation_method=explanation_method,
        gpu=gpu,
    )

    from data.calculate_classification_thresholds import (
        calculate_classification_thresholds,
    )

    calculate_classification_thresholds(cfg=general_config)


@app.command()
def run_evaluate_explanations(
    config_path: str,
    metrics_config_path: str,
    random_seed: Annotated[int, typer.Option()] = 42,
    debug: Annotated[bool, typer.Option()] = False,
    explanation_method: Annotated[str, typer.Option()] = None,
    gpu: Annotated[int, typer.Option()] = 3,
    save_data: Annotated[bool, typer.Option()] = False,
    results_path: Annotated[str, typer.Option()] = None,
):
    from config_utils import setup_everything

    general_config = setup_everything(
        config_path=config_path,
        random_seed=random_seed,
        project_root=project_root,
        debug=debug,
        explanation_method=explanation_method,
        gpu=gpu,
        results_path=results_path,
    )

    from config_utils import load_yaml

    # if metrics_config_path not startwwitch project root
    if not metrics_config_path.startswith(project_root):
        metrics_config_path = os.path.join(project_root, metrics_config_path)
    metrics_config = load_yaml(metrics_config_path)

    from xai.metrics.evaluate_explanation_methods import evaluate_explanation_methods

    evaluate_explanation_methods(general_config, metrics_config, save_data=save_data)


@app.command()
def run_visualize(
    config_path: str,
    random_seed: Annotated[int, typer.Option()] = 42,
    debug: Annotated[bool, typer.Option()] = False,
    explanation_method: Annotated[str, typer.Option()] = None,
    gpu: Annotated[int, typer.Option()] = 3,
    generate_explanations: Annotated[bool, typer.Option()] = False,
):
    from config_utils import setup_everything

    general_config = setup_everything(
        config_path=config_path,
        random_seed=random_seed,
        project_root=project_root,
        debug=debug,
        explanation_method=explanation_method,
        gpu=gpu,
        generate_explanations=generate_explanations,
    )

    from visualization.visualize import visualize

    visualize(general_config)


@app.command()
def run_grid_search(
    config_path: str,
    random_seed: Annotated[int, typer.Option()] = 42,
    debug: Annotated[bool, typer.Option()] = False,
    explanation_method: Annotated[str, typer.Option()] = None,
    gpu: Annotated[int, typer.Option()] = 3,
    mode: Annotated[str, typer.Option()] = "normal",
    num_trials: Annotated[int, typer.Option()] = 100,
):
    from config_utils import setup_everything

    general_config = setup_everything(
        config_path=config_path,
        random_seed=random_seed,
        project_root=project_root,
        debug=debug,
        explanation_method=explanation_method,
        gpu=gpu,
    )
    general_config["mode"] = mode

    from cutmix.calculate_threshold import calculate_threshold_for_xai_masks

    calculate_threshold_for_xai_masks(general_config, num_trials=num_trials)


@app.command()
def run_sanity_checks(
    config_path: str,
    random_seed: Annotated[int, typer.Option()] = 42,
    debug: Annotated[bool, typer.Option()] = False,
    explanation_method: Annotated[str, typer.Option()] = None,
    gpu: Annotated[int, typer.Option()] = 3,
    loader_name: Annotated[str, typer.Option()] = "train",
):
    from config_utils import setup_everything

    general_config = setup_everything(
        config_path=config_path,
        random_seed=random_seed,
        project_root=project_root,
        debug=debug,
        explanation_method=explanation_method,
        gpu=gpu,
    )

    from training.dataset_sanity_checker import sanity_check_labels_and_segmasks

    sanity_check_labels_and_segmasks(general_config, loader_name=loader_name)


@app.command()
def run_sanity_check_augmentation_thresholds(
    config_path: str,
    random_seed: Annotated[int, typer.Option()] = 42,
    debug: Annotated[bool, typer.Option()] = False,
    explanation_method: Annotated[str, typer.Option()] = None,
    gpu: Annotated[int, typer.Option()] = 3,
):
    from config_utils import setup_everything

    general_config = setup_everything(
        config_path=config_path,
        random_seed=random_seed,
        project_root=project_root,
        debug=debug,
        explanation_method=explanation_method,
        gpu=gpu,
    )

    from cutmix.sanity_check_augmentations import sanity_check_augmentation_thresholds

    sanity_check_augmentation_thresholds(general_config)


@app.command()
def run_dataset_distribution(
    config_path: str,
    random_seed: Annotated[int, typer.Option()] = 42,
    debug: Annotated[bool, typer.Option()] = False,
    gpu: Annotated[int, typer.Option()] = 3,
):
    from config_utils import setup_everything

    general_config = setup_everything(
        config_path=config_path,
        random_seed=random_seed,
        project_root=project_root,
        debug=debug,
        explanation_method=None,
        gpu=gpu,
    )

    from visualization.dataset_visualizations import plot_datamodule_distribution

    plot_datamodule_distribution(general_config)


if __name__ == "__main__":
    app()
