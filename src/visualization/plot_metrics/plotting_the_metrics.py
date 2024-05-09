import os

import pandas as pd
import typer
from typing_extensions import Annotated

from visualization.plot_cutmix_thresh_matrices import plot_cutmix_thresh_matrices
from visualization.plot_metrics.parse_data import (
    recalculate_score_direction,
    scale_df,
    log_some_cols,
    _load_df,
    read_all_csvs,
    read_into_dfs,
)
from visualization.plot_metrics.plot_helpers import (
    plot_matrix,
    plot_best_overall_method,
    plot_bar_metric_categories,
    plot_best_metric_per_category,
    get_metrics_categories,
    plot_bar_metric_comparison,
    plot_bar_double_metric,
    plot_metrics_comparison_scatter,
    plot_bar_single_metric,
    plot_time_matrix,
)
from visualization.plot_metrics.stats_analysis import calc_and_plot_correlation

rename_dict = {
    "gradcam": "GradCAM",
    "guided": "Guided GradCAM",
    "lime": "LIME",
    "deeplift": "DeepLift",
    "integrated": "Integrated Gradients",
    "lrp": "LRP",
    "occlusion": "Occlusion",
}

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main_multilabel(
    csv_dir: Annotated[str, typer.Option()] = None,
):
    # csv_dir = "/home/jonasklotz/Studys/MASTERS/Thesis/Final_Results/deepglobe/metrics"
    dataset_name = "deepglobe"
    metric_to_plot = "IROF"
    metric_1 = "Region Segmentation LERF"
    metric_2 = "Region Segmentation MORF"

    if not csv_dir:
        csv_dir = "/home/jonasklotz/Studys/MASTERS/results_22_4_final/deepglobe/metrics"

    visualization_save_dir = f"{csv_dir}/visualizations"
    os.makedirs(visualization_save_dir, exist_ok=True)

    # metric_to_plot = "IROF"
    # metric_1 = "Region Segmentation LERF"
    # metric_2 = "Region Segmentation MORF"
    df_full, time_df_long = _load_df(csv_dir, visualization_save_dir)
    time_df_long["Method"] = time_df_long["Method"].replace(rename_dict)
    df_full["Method"] = df_full["Method"].replace(rename_dict)

    # plot_result_distribution(df_full, dataset_name, visualization_save_dir)

    plot_matrix(
        df_full,
        visualization_save_dir=visualization_save_dir,
        title_text=f"{dataset_name}: Unprocessed Metric Matrix",
    )
    df_preprocessed = preprocess_metrics(df_full)

    plot_matrix(
        df_preprocessed,
        visualization_save_dir=visualization_save_dir,
        title_text=f"{dataset_name}: Metric Matrix",
    )
    (
        df_only_correct,
        df_only_correct_grouped,
        df_only_false,
        df_only_false_grouped,
    ) = extract_correct_and_false_preds(df_preprocessed)

    categories = get_metrics_categories(df_preprocessed["Metric"].unique())

    rrr_df_path = (
        "/home/jonasklotz/Studys/MASTERS/Thesis/Final_Results/deepglobe/rrr/rrr_all.csv"
    )
    rrr_df = pd.read_csv(rrr_df_path, sep=",", index_col=None, header=0)
    # Rename the Methods
    rrr_df["Method"] = rrr_df["Method"].replace(rename_dict)
    for col in rrr_df.columns:
        if "Method" in col:
            continue

        calc_and_plot_correlation(
            df_preprocessed,
            rrr_df,
            metric_to_correlate=col,
            visualization_save_dir=visualization_save_dir,
            title_prefix=f"{dataset_name}: ",
        )

    plot_bar_metric_categories(
        df_preprocessed,
        categories,
        visualization_save_dir=visualization_save_dir,
        title_prefix=f"{dataset_name}: ",
    )
    plot_best_metric_per_category(
        df_preprocessed,
        categories,
        visualization_save_dir=visualization_save_dir,
        title_text=f"{dataset_name}: Best Method per Category ",
    )

    plot_best_overall_method(
        df_preprocessed,
        all_methods=df_preprocessed["Method"].unique(),
        visualization_save_dir=visualization_save_dir,
        title_text=f"{dataset_name}: Best Overall Method",
    )

    # Merge the two dataframes on 'Method' and 'Metric' to align them
    df_difference = pd.merge(
        df_only_false_grouped,
        df_only_correct_grouped,
        on=["Method", "Metric"],
        suffixes=("_false", "_correct"),
        how="inner",
        validate="many_to_many",
    )
    df_difference["Value_diff"] = (
        df_difference["Value_false"] - df_difference["Value_correct"]
    )

    df_difference["Value"] = (
        df_difference["Value_diff"] / df_difference["Value_correct"]
    ) * 100

    plot_bar_metric_comparison(
        df_difference,
        title_text=f"{dataset_name}: Performance Drop in %: Correct vs. False Predictions",
        visualization_save_dir=visualization_save_dir,
        y_axis_unit="%",
    )

    plot_best_overall_method(
        df_preprocessed,
        all_methods=df_preprocessed["Method"].unique(),
        visualization_save_dir=visualization_save_dir,
        title_text=f"{dataset_name}: Best Overall Method",
    )

    # df without correct prediction column and sample index
    df = df_preprocessed.drop(columns=["CorrectPrediction", "SampleIndex"])

    # Plotting all metrics
    plot_bar_metric_comparison(
        df,
        title_text=f"{dataset_name}: Comparison of Methods Across all Metrics",
        visualization_save_dir=visualization_save_dir,
    )

    # Plotting all metrics for only correct predictions

    plot_bar_metric_comparison(
        df_only_correct,
        title_text=f"{dataset_name}: Comparison of Methods Across all Metrics (Only Correct Predictions)",
        visualization_save_dir=visualization_save_dir,
    )

    # Plotting all metrics for only false predictions

    plot_bar_metric_comparison(
        df_only_false,
        title_text=f"{dataset_name}: Comparison of Methods Across all Metrics (Only False Predictions)",
        visualization_save_dir=visualization_save_dir,
    )

    plot_bar_single_metric(
        df,
        metric_to_plot,
        title_text=f"{dataset_name}: Comparison of Methods for {metric_to_plot}",
        visualization_save_dir=visualization_save_dir,
    )

    plot_metrics_comparison_scatter(
        df,
        ["Method", "Class"],
        "Metric",
        "Value",
        metric_1,
        metric_2,
        visualization_save_dir=visualization_save_dir,
    )

    plot_bar_double_metric(
        df,
        [metric_1, metric_2],
        title_text=f"DG Comparison of Methods for {metric_to_plot}",
        visualization_save_dir=visualization_save_dir,
    )


def preprocess_metrics(df_full):
    metrics_to_apply_log = [
        "Relative Input Stability",
        "Relative Output Stability",
        # "Effective Complexity",
    ]
    df_full = log_some_cols(df_full, metrics_to_apply_log=metrics_to_apply_log, base=2)
    df_full = scale_df(df_full, scale="robust")
    df_full = recalculate_score_direction(df_full)
    # clip data to 0-1
    df_full["Value"] = df_full["Value"].clip(0, 1)
    df_full["Method"] = df_full["Method"].replace(rename_dict)
    # round to 2 decimals
    df_full["Value"] = df_full["Value"].round(2)
    return df_full


def extract_correct_and_false_preds(df_full):
    df_only_correct = df_full[df_full["CorrectPrediction"] == True]  # noqa: E712 necessary for filtering
    df_only_correct = df_only_correct.drop(columns=["CorrectPrediction", "SampleIndex"])
    df_only_false = df_full[df_full["CorrectPrediction"] == False]  # noqa: E712 necessary for filtering
    df_only_false = df_only_false.drop(columns=["CorrectPrediction", "SampleIndex"])
    df_only_false_grouped = (
        df_only_false.groupby(["Method", "Metric"])["Value"].mean().reset_index()
    )
    df_only_correct_grouped = (
        df_only_correct.groupby(["Method", "Metric"])["Value"].mean().reset_index()
    )
    return (
        df_only_correct,
        df_only_correct_grouped,
        df_only_false,
        df_only_false_grouped,
    )


@app.command()
def main_singlelabel(
    csv_dir: Annotated[str, typer.Option()] = None,
):
    dataset_name = "caltech101"
    if not csv_dir:
        csv_dir = "/home/jonasklotz/Studys/MASTERS/results_22_4_final/caltech/metrics"

    visualization_save_dir = f"{csv_dir}/visualizations"
    os.makedirs(visualization_save_dir, exist_ok=True)

    metrics_csvs, time_csvs, labels_csvs = read_all_csvs(csv_dir)
    labels_dfs, metrics_dfs, time_dfs = read_into_dfs(
        labels_csvs, metrics_csvs, time_csvs
    )

    # metric_to_plot = "IROF"
    # metric_1 = "Region Segmentation LERF"
    # metric_2 = "Region Segmentation MORF"
    save_path = f"{visualization_save_dir}/df_full.csv"
    save_path = os.path.abspath(save_path)
    # if file does not exist read df

    df_full, time_df_long = _load_df(
        csv_dir, visualization_save_dir, task="singlelabel"
    )
    time_df_long["Method"] = time_df_long["Method"].replace(rename_dict)

    plot_time_matrix(
        time_df_long,
        visualization_save_dir=visualization_save_dir,
        title_text=f"{dataset_name}: Seconds spent per sample for each method and metric",
    )
    df_full = preprocess_metrics(df_full)

    plot_matrix(
        df_full,
        visualization_save_dir=visualization_save_dir,
        title_text=f"{dataset_name}: Metric Matrix",
    )

    # rrr_df_path = (
    #     "/home/jonasklotz/Studys/MASTERS/Final_Results/caltech101/rrr/rrr_all.csv"
    # )
    # rrr_df = pd.read_csv(rrr_df_path, sep=",", index_col=None, header=0)
    # # Rename the Methods
    # rrr_df["Method"] = rrr_df["Method"].replace(rename_dict)
    # for col in rrr_df.columns:
    #     if "Method" in col:
    #         continue
    #
    #     calc_and_plot_correlation(
    #         df_full,
    #         rrr_df,
    #         metric_to_correlate=col,
    #         visualization_save_dir=visualization_save_dir,
    #         title_prefix=f"{dataset_name}: ",
    #     )

    plot_best_overall_method(
        df_full,
        all_methods=df_full["Method"].unique(),
        visualization_save_dir=visualization_save_dir,
        title_text=f"{dataset_name}: Best Overall Method",
    )
    categories = get_metrics_categories(df_full["Metric"].unique())

    plot_best_metric_per_category(
        df_full,
        categories,
        visualization_save_dir=visualization_save_dir,
        title_text=f"{dataset_name}: Best Metric per Category ",
    )

    plot_bar_metric_comparison(
        df_full,
        title_text=f"{dataset_name}: Comparison of Methods Across all Metrics",
        visualization_save_dir=visualization_save_dir,
    )

    plot_bar_metric_categories(
        df_full, categories, visualization_save_dir=visualization_save_dir
    )

    # df without correct prediction column and sample index
    # Plotting all metrics
    plot_bar_metric_comparison(
        df_full,
        title_text=f"{dataset_name}: Comparison of Methods Across all Metrics",
        visualization_save_dir=visualization_save_dir,
    )


@app.command()
def run_plot_cutmix_thresh_matrices(csv_dir: Annotated[str, typer.Option()] = None):
    if not csv_dir:
        csv_dir = "/home/jonasklotz/Studys/MASTERS/results_22_4_final/deepglobe/threshold_matrices"
    save_dir = f"{csv_dir}/visualizations"
    os.makedirs(save_dir, exist_ok=True)

    csv_files = os.listdir(csv_dir)
    csv_files = [file for file in csv_files if file.endswith(".csv")]

    acc_csvs = [file for file in csv_files if "acc" in file]
    # f1_csvs = [file for file in csv_files if "f1" in file]

    acc_dfs = [
        pd.read_csv(os.path.join(csv_dir, file), index_col=0) for file in acc_csvs
    ]
    acc_filenames = [file.split(".")[0] for file in acc_csvs]

    plot_cutmix_thresh_matrices(
        acc_dfs,
        acc_filenames,
        save_dir,
        title="DeepGlobe: Accuracy for different CutMix and Segmentation Thresholds",
    )


if __name__ == "__main__":
    app()
