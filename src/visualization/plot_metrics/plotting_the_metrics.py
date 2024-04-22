import os

import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

from visualization.plot_metrics.parse_data import (
    parse_data_single_label,
    recalculate_score_direction,
    scale_df,
)
from visualization.plot_metrics.plot_helpers import (
    plot_matrix,
    plot_best_overall_method,
    plot_bar_metric_categories,
    plot_best_metric_per_category,
    get_metrics_categories,
    plot_bar_double_metric,
    plot_metrics_comparison_scatter,
    plot_bar_single_metric,
    plot_bar_metric_comparison,
    plot_result_distribution,
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

app = typer.Typer()


@app.command()
def main_multilabel():
    # csv_dir = "/home/jonasklotz/Studys/MASTERS/Thesis/Final_Results/deepglobe/metrics"
    dataset_name = "Deepglobe"
    metric_to_plot = "IROF"
    metric_1 = "Region Segmentation LERF"
    metric_2 = "Region Segmentation MORF"
    save_path = "/home/jonasklotz/Studys/MASTERS/Thesis/Final_Results/deepglobe/metrics/df_test.csv"

    visualization_save_dir = (
        "/home/jonasklotz/Studys/MASTERS/Final_Results/deepglobe/temporary"
    )

    # df_full = parse_data(csv_dir)
    # # # save the df to a csv
    # df_full.to_csv(save_path, index=False, sep=';')

    dtypes = {
        "Method": str,
        "Metric": str,
        "Class": str,
        "Value": float,
        "SampleIndex": int,
        "CorrectPrediction": bool,
    }
    # read the df from the csv
    df_full = pd.read_csv(save_path, sep=";", index_col=None, header=0, dtype=dtypes)
    # Rename the Methods

    df_full["Method"] = df_full["Method"].replace(rename_dict)
    df_full = recalculate_score_direction(df_full)
    df_full = df_full[df_full["Metric"] != "Monotonicity-Arya"]

    df = df_full.drop(columns=["CorrectPrediction", "SampleIndex"])

    # drop all entries where Metric is "Monotonicity Arya"
    df_full = df_full[df_full["Metric"] != "Monotonicity-Arya"]

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

    categories = get_metrics_categories(df_full["Metric"].unique())

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
            df_full,
            rrr_df,
            metric_to_correlate=col,
            visualization_save_dir=visualization_save_dir,
            title_prefix=f"{dataset_name}: ",
        )

    plot_matrix(
        df_full,
        visualization_save_dir=visualization_save_dir,
        title_text=f"{dataset_name}: Metric Matrix",
    )

    plot_bar_metric_categories(
        df_full,
        categories,
        visualization_save_dir=visualization_save_dir,
        title_prefix=f"{dataset_name}: ",
    )
    plot_best_metric_per_category(
        df_full,
        categories,
        visualization_save_dir=visualization_save_dir,
        title_text=f"{dataset_name}: Best Method per Category ",
    )

    plot_best_overall_method(
        df_full,
        all_methods=df_full["Method"].unique(),
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
        df_full,
        all_methods=df_full["Method"].unique(),
        visualization_save_dir=visualization_save_dir,
        title_text=f"{dataset_name}: Best Overall Method",
    )

    # df without correct prediction column and sample index
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


@app.command()
def main_singlelabel(
    csv_dir: Annotated[str, typer.Option()] = None,
):
    dataset_name = "caltech101"
    if not csv_dir:
        csv_dir = "/home/jonasklotz/Studys/MASTERS/results_22_4_final/caltech/metrics"

    visualization_save_dir = f"{csv_dir}/visualizations"
    os.makedirs(visualization_save_dir, exist_ok=True)

    # metric_to_plot = "IROF"
    # metric_1 = "Region Segmentation LERF"
    # metric_2 = "Region Segmentation MORF"
    save_path = f"{visualization_save_dir}/df_full.csv"
    save_path = os.path.abspath(save_path)
    # if file does not exist read df
    if os.path.isfile(save_path):
        dtypes = {
            "SampleIndex": int,
            "Method": str,
            "Metric": str,
            "Value": float,
            "CorrectPrediction": bool,
        }
        # read the df from the csv
        df_full = pd.read_csv(
            save_path, sep=";", index_col=None, header=0, dtype=dtypes
        )
    else:
        # we parse the data and save it
        df_full = parse_data_single_label(csv_dir)
        # # save the df to a csv
        df_full.to_csv(save_path, index=False, sep=";")

    plot_result_distribution(df_full, dataset_name, visualization_save_dir)
    return
    df_full = scale_df(df_full)

    # df_full = log_some_cols(df_full)

    df_full = recalculate_score_direction(df_full)  # todo: redo this
    # df = df_full.drop(columns=["SampleIndex"])
    df_full["Method"] = df_full["Method"].replace(rename_dict)
    df_full = df_full[df_full["Metric"] != "Monotonicity-Arya"]

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
        title_text=f"{dataset_name}: Best Metric per Category",
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


def log_some_cols(df):
    metrics_to_apply_log = [
        "Relative Input Stability",
        "Relative Output Stability",
        "Complexity",
        "Infidelity",
    ]
    for metric in metrics_to_apply_log:
        # apply logarithm to all Values where the Column MEtric is metric
        df.loc[df["Metric"] == metric, "Value"] = df.loc[
            df["Metric"] == metric, "Value"
        ].apply(np.log2)

    return df


if __name__ == "__main__":
    app()
