import os
import sys

import typer
from typing_extensions import Annotated


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

    if explanation_method and mode == "cutmix":
        # Set the path to the lmdb file on the cluster
        general_config["data"][
            "segmentations_lmdb_path"
        ] = f"/media/storagecube/jonasklotz/deepglobe_vgg_lmdbs_new/{explanation_method}.lmdb"

    if explanation_method and mode == "rrr":
        general_config["rrr_explanation"] = explanation_method

    if mode != "normal":
        general_config["experiment_name"] += f"_{mode}"

    general_config["mode"] = mode
    from training.train import train

    train(general_config)


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
def run_evaluate_explanations(
    config_path: str,
    metrics_config_path: str,
    random_seed: Annotated[int, typer.Option()] = 42,
    debug: Annotated[bool, typer.Option()] = False,
    explanation_method: Annotated[str, typer.Option()] = None,
    gpu: Annotated[int, typer.Option()] = 3,
    save_data: Annotated[bool, typer.Option()] = False,
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

    from config_utils import load_yaml

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


if __name__ == "__main__":
    app()
