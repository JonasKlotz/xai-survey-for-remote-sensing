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
)
from visualization.plot_metrics.plot_helpers import (
    plot_matrix,
    plot_best_overall_method,
    plot_bar_metric_categories,
    plot_best_metric_per_category,
    get_metrics_categories,
    plot_bar_metric_comparison,
    plot_time_matrix,
    plot_with_correlation,
)

rename_dict = {
    "gradcam": "GradCAM",
    "guided": "Guided GradCAM",
    "lime": "LIME",
    "deeplift": "DeepLift",
    "integrated": "Integrated Gradients",
    "lrp": "LRP",
    "occlusion": "Occlusion",
}
rp_renaming_dict = {
    "Region Segmentation LERF": "Region Perturbation LERF",
    "Region Segmentation MORF": "Region Perturbation MORF",
}
dataset_rename_dict = {
    "deepglobe": "DeepGlobe",
    "caltech": "Caltech101",
    "ben": "BEN",
}

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main_multilabel(
    csv_dir: Annotated[str, typer.Option()] = None, dataset_name="deepglobe"
):
    # csv_dir = "/home/jonasklotz/Studys/MASTERS/Thesis/Final_Results/deepglobe/metrics"

    # metric_to_plot = "IROF"
    # metric_1 = "Region Segmentation LERF"
    # metric_2 = "Region Segmentation MORF"
    # region the region segmentation metrics to region perturbation metrics

    if not csv_dir:
        csv_dir = (
            f"/home/jonasklotz/Studys/MASTERS/results_22_4_final/{dataset_name}/metrics"
        )

    dataset_name = dataset_rename_dict[dataset_name]

    visualization_save_dir = f"{csv_dir}/visualizations"
    os.makedirs(visualization_save_dir, exist_ok=True)

    # metric_to_plot = "IROF"
    # metric_1 = "Region Segmentation LERF"
    # metric_2 = "Region Segmentation MORF"
    df_full, time_df_long = _load_df(csv_dir, visualization_save_dir)

    time_df_long["Method"] = time_df_long["Method"].replace(rename_dict)
    time_df_long["Metric"] = time_df_long["Metric"].replace(rp_renaming_dict)

    df_full["Method"] = df_full["Method"].replace(rename_dict)
    df_full["Metric"] = df_full["Metric"].replace(rp_renaming_dict)

    # plot_time_matrix(
    #     time_df_long,
    #     visualization_save_dir=visualization_save_dir,
    #     title_text=f"{dataset_name}: Seconds spent per sample for each method and metric",
    # )

    # plot_result_distribution(df_full, dataset_name, visualization_save_dir)

    plot_matrix(
        df_full,
        visualization_save_dir=visualization_save_dir,
        title_text=f"{dataset_name}: Unprocessed Metric Matrix",
    )
    df_preprocessed = preprocess_metrics(df_full)
    # plot_result_distribution(df_full, dataset_name, visualization_save_dir)

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
    if dataset_name == "deepglobe":
        # BEN HAS NO CORRECT PREDS
        plot_matrix(
            df_only_correct,
            visualization_save_dir=visualization_save_dir,
            title_text=f"{dataset_name}: Metric Matrix for correct predictions",
        )
        plot_matrix(
            df_only_false,
            visualization_save_dir=visualization_save_dir,
            title_text=f"{dataset_name}: Metric Matrix for wrong predictions",
        )
        df_difference = pd.merge(
            df_only_false_grouped,
            df_only_correct_grouped,
            on=["Method", "Metric"],
            suffixes=("_false", "_correct"),
            how="inner",
            validate="many_to_many",
        )
        df_difference["Value"] = (
            df_difference["Value_false"] - df_difference["Value_correct"]
        )
        plot_matrix(
            df_difference,
            visualization_save_dir=visualization_save_dir,
            title_text=f"{dataset_name}: Difference of Metric Matrices for for correct and wrong predictions",
        )

    categories = get_metrics_categories(df_preprocessed["Metric"].unique())

    # plot_bar_metric_categories(
    #     df_preprocessed,
    #     categories,
    #     visualization_save_dir=visualization_save_dir,
    #     title_prefix=f"{dataset_name}: ",
    # )
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

    # plot_bar_metric_comparison(
    #     df_difference,
    #     title_text=f"{dataset_name}: Performance Drop in %: Correct vs. False Predictions",
    #     visualization_save_dir=visualization_save_dir,
    #     y_axis_unit="%",
    # )
    #
    # plot_best_overall_method(
    #     df_preprocessed,
    #     all_methods=df_preprocessed["Method"].unique(),
    #     visualization_save_dir=visualization_save_dir,
    #     title_text=f"{dataset_name}: Best Overall Method",
    # )
    #
    # # df without correct prediction column and sample index
    # df = df_preprocessed.drop(columns=["CorrectPrediction", "SampleIndex"])
    #
    # # Plotting all metrics
    # plot_bar_metric_comparison(
    #     df,
    #     title_text=f"{dataset_name}: Comparison of Methods Across all Metrics",
    #     visualization_save_dir=visualization_save_dir,
    # )
    #
    # # Plotting all metrics for only correct predictions
    #
    # plot_bar_metric_comparison(
    #     df_only_correct,
    #     title_text=f"{dataset_name}: Comparison of Methods Across all Metrics (Only Correct Predictions)",
    #     visualization_save_dir=visualization_save_dir,
    # )
    #
    # # Plotting all metrics for only false predictions
    #
    # plot_bar_metric_comparison(
    #     df_only_false,
    #     title_text=f"{dataset_name}: Comparison of Methods Across all Metrics (Only False Predictions)",
    #     visualization_save_dir=visualization_save_dir,
    # )
    #
    # plot_bar_single_metric(
    #     df,
    #     metric_to_plot,
    #     title_text=f"{dataset_name}: Comparison of Methods for {metric_to_plot}",
    #     visualization_save_dir=visualization_save_dir,
    # )

    # plot_metrics_comparison_scatter(
    #     df,
    #     ["Method", "Class"],
    #     "Metric",
    #     "Value",
    #     metric_1,
    #     metric_2,
    #     visualization_save_dir=visualization_save_dir,
    # )
    #
    # plot_bar_double_metric(
    #     df,
    #     [metric_1, metric_2],
    #     title_text=f"DG Comparison of Methods for {metric_to_plot}",
    #     visualization_save_dir=visualization_save_dir,
    # )
    # df_difference["Value_diff"] = (
    #     df_difference["Value_false"] - df_difference["Value_correct"]
    # )


def preprocess_metrics(df_full):
    metrics_to_apply_log = [
        "Relative Input Stability",
        "Relative Output Stability",
        # "Effective Complexity",
    ]
    df_full = log_some_cols(df_full, metrics_to_apply_log=metrics_to_apply_log, base=2)
    df_full = scale_df(df_full, scale="robust")

    df_full["Metric"] = df_full["Metric"].replace(rp_renaming_dict)
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
    dataset_name = "caltech"
    if not csv_dir:
        csv_dir = "/home/jonasklotz/Studys/MASTERS/results_22_4_final/caltech/metrics"
        # csv_dir= "/home/jonasklotz/Studys/MASTERS/results_22_4_final/caltech/metrics/region_pertur_metrics"

    visualization_save_dir = f"{csv_dir}/visualizations"
    os.makedirs(visualization_save_dir, exist_ok=True)

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

    df_full["Method"] = df_full["Method"].replace(rename_dict)
    df_full["Metric"] = df_full["Metric"].replace(rp_renaming_dict)
    plot_matrix(
        df_full,
        visualization_save_dir=visualization_save_dir,
        title_text=f"{dataset_name}: Unprocessed Metric Matrix",
    )

    df_full = preprocess_metrics(df_full)
    dataset_name = dataset_rename_dict[dataset_name]

    plot_matrix(
        df_full,
        visualization_save_dir=visualization_save_dir,
        title_text=f"{dataset_name}: Metric Matrix",
    )
    # plot_matrix(
    #     df_full,
    #     visualization_save_dir=visualization_save_dir,
    #     title_text=f"{dataset_name}: Localisation Metric Matrix for RRR GradCAM trained model",
    # )

    plot_time_matrix(
        time_df_long,
        visualization_save_dir=visualization_save_dir,
        title_text=f"{dataset_name}: Seconds spent per sample for each method and metric",
    )

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
        title="DeepGlobe: Accuracy for different t_map and t_cam Thresholds",
    )


@app.command()
def rrr_singlelabel():
    csv_dir = "/home/jonasklotz/Studys/MASTERS/results_22_4_final/caltech/metrics"
    # dataset_name = "caltech101"
    visualization_save_dir = f"{csv_dir}/visualizations"
    os.makedirs(visualization_save_dir, exist_ok=True)
    rrr_df_path = "/home/jonasklotz/Studys/MASTERS/results_22_4_final/caltech/rrr/explanations/caltech_rrr_cleaned.csv"
    result_df = get_rrr_df(rrr_df_path)

    df_full, _ = _load_df(csv_dir, visualization_save_dir, task="singlelabel")
    df_full = preprocess_metrics(df_full)
    df_full["Method"] = df_full["Method"].replace(rename_dict)
    # get categories for the metrics
    # add the categories to the
    # average
    # merge on the method column
    df_full = pd.merge(
        df_full,
        result_df,
        on="Method",
        how="inner",
        validate="many_to_one",
    )
    categories = get_metrics_categories(df_full["Metric"].unique())
    # add the categories to the df
    df_full["Category"] = df_full["Metric"].apply(lambda x: categories[x])

    df = (
        df_full.groupby(
            ["Method", "Metric", "Category", "Parameter", "Parameter_Value"]
        )["Value"]
        .mean()
        .reset_index()
    )

    # remove all rows where Method==  GradCAM and Parameter "1.0_distanceelementwise",
    # as it is a hard outllier
    df = df[
        ~((df["Method"] == "GradCAM") & (df["Parameter"] == "10.0_distanceelementwise"))
    ]
    df = df[~((df["Method"] == "GradCAM") & (df["Parameter"] == "1.0_distancemse"))]

    plot_with_correlation(
        df,
        "Metric",
        "Parameter_Value",
        "Parameter",
        "Caltech101: Correlational Analysis of xAI Metrics with Test Accuracy in RRR Training",
        visualization_save_dir,
    )
    # plot_with_correlation(
    #     df,
    #     "Category",
    #     "Parameter_Value",
    #     "Parameter",
    #     "Caltech101: Correlational Analysis of xAI Metric Categories with Test Accuracy in RRR Training",
    #     visualization_save_dir,
    # )


@app.command()
def rrr_multilabel(dataset_name: str = "DeepGlobe"):
    csv_dir = (
        f"/home/jonasklotz/Studys/MASTERS/results_22_4_final/{dataset_name}/metrics"
    )
    visualization_save_dir = f"{csv_dir}/visualizations"
    os.makedirs(visualization_save_dir, exist_ok=True)
    rrr_df_path = f"/home/jonasklotz/Studys/MASTERS/results_22_4_final/{dataset_name}/rrr/{dataset_name}_rrr_cleaned.csv"
    result_df = get_rrr_df(rrr_df_path)

    df_full, _ = _load_df(csv_dir, visualization_save_dir, task="multilabel")
    df_full = preprocess_metrics(df_full)
    df_full["Method"] = df_full["Method"].replace(rename_dict)
    # get categories for the metrics
    # add the categories to the
    # average
    # merge on the method column
    df_full = pd.merge(
        df_full,
        result_df,
        on="Method",
        how="inner",
        validate="many_to_one",
    )
    categories = get_metrics_categories(df_full["Metric"].unique())
    # add the categories to the df
    df_full["Category"] = df_full["Metric"].apply(lambda x: categories[x])
    # parameter_columns = [
    #     "1.0_distancemse",
    #     "10.0_distancemse",
    #     "1.0_distanceelementwise",
    #     "10.0_distanceelementwise",
    # ]

    # id_vars = [col for col in df_full.columns if col not in parameter_columns]
    #
    # # Use melt to stack the parameter columns into a single column
    # df_long = df_full.melt(
    #     id_vars=id_vars,
    #     value_vars=parameter_columns,
    #     var_name="Parameter",
    #     value_name="Parameter_Value",
    # )

    df = (
        df_full.groupby(
            ["Method", "Metric", "Category", "Parameter", "Parameter_Value"]
        )["Value"]
        .mean()
        .reset_index()
    )

    # remove all rows where Method==  GradCAM and Parameter "1.0_distanceelementwise",
    # as it is a hard outllier
    # df = df[
    #     ~((df["Method"] == "GradCAM") & (df["Parameter"] == "10.0_distanceelementwise"))
    # ]
    # df = df[~((df["Method"] == "GradCAM") & (df["Parameter"] == "1.0_distancemse"))]

    plot_with_correlation(
        df,
        "Metric",
        "Parameter_Value",
        "Parameter",
        "DeepGlobe: Correlational Analysis of xAI Metrics with Test mAP in RRR Training",
        visualization_save_dir,
    )
    # plot_with_correlation(
    #     df,
    #     "Category",
    #     "Parameter_Value",
    #     "Parameter",
    #     "DeepGlobe: Correlational Analysis of xAI Metric Categories with Test mAP in RRR Training",
    #     visualization_save_dir,
    # )


def get_rrr_df(rrr_df_path):
    rrr_df = pd.read_csv(rrr_df_path, sep=",", index_col=None, header=0)

    # Function to get max value and corresponding column name for each method
    def get_max_value_and_column(row):
        max_value = row[1:].max()
        max_column = row[1:].idxmax()
        return pd.Series(
            [max_value, max_column], index=["Parameter_Value", "Parameter"]
        )

    # Apply the function to each row
    max_values_df = rrr_df.apply(get_max_value_and_column, axis=1)
    # Combine the original 'Method' column with the new DataFrame
    result_df = pd.concat([rrr_df["Method"], max_values_df], axis=1)
    rename_parameter_dict = {
        "1.0_distancemse": "MSE Distance with  λ = 1",
        "10.0_distancemse": "MSE Distance with  λ = 10",
        "1.0_distanceelementwise": "Elementwise Distance with  λ = 1",
        "10.0_distanceelementwise": "Elementwise Distance with  λ = 10",
    }
    result_df["Parameter"] = result_df["Parameter"].replace(rename_parameter_dict)
    return result_df


def get_cutmix_df(rrr_df_path):
    df = pd.read_csv(rrr_df_path, sep=",", index_col=None, header=0)

    # Function to get max value and corresponding column name for each row
    def get_max_value_and_column(row):
        max_value = row[["0.1-0.5", "0.3-0.7"]].max()
        max_column = row[["0.1-0.5", "0.3-0.7"]].idxmax()
        return pd.Series(
            [max_value, max_column], index=["Parameter_Value", "Parameter"]
        )

    # Apply the function to each row and concatenate the result to the original dataframe
    df = df.join(df.apply(get_max_value_and_column, axis=1))
    # drop [['0.1-0.5', '0.3-0.7']
    df = df.drop(columns=["0.1-0.5", "0.3-0.7"])
    rename_parameter_dict = {"0.1-0.5": "Boxsize 0.1-0.5", "0.3-0.7": "Boxsize 0.3-0.7"}
    df["Parameter"] = df["Parameter"].replace(rename_parameter_dict)
    df["Method"] = df["Method"].replace(rename_dict)

    return df


@app.command()
def compare_single_and_multi_label(mlc_dataset="deepglobe", slc_dataset="ben"):
    slc_metrics_df, slc_time_df = get_dataframes(
        slc_dataset, slc=slc_dataset == "caltech"
    )
    mlc_metrics_df, mlc_time_df = get_dataframes(
        mlc_dataset, slc=mlc_dataset == "caltech"
    )

    grouped_slc = (
        slc_metrics_df.groupby(["Method", "Metric"])["Value"].mean().reset_index()
    )
    grouped_mlc = (
        mlc_metrics_df.groupby(["Method", "Metric"])["Value"].mean().reset_index()
    )

    diffed = pd.merge(
        grouped_slc,
        grouped_mlc,
        on=["Method", "Metric"],
        suffixes=("_slc", "_mlc"),
        how="inner",
        validate="many_to_many",
    )
    diffed["Value"] = diffed["Value_slc"] - diffed["Value_mlc"]
    diffed["Value"] = diffed["Value"].round(2)

    visualization_save_dir = f"/home/jonasklotz/Studys/MASTERS/results_22_4_final/comparison/{slc_dataset}_{mlc_dataset}/visualizations"
    os.makedirs(visualization_save_dir, exist_ok=True)

    plot_matrix(
        diffed,
        visualization_save_dir=visualization_save_dir,
        title_text=f"{slc_dataset} vs. {mlc_dataset}: Difference of Metric Matrices",
    )


@app.command()
def cutmix_multilabel(dataset_name: str = "deepglobe"):
    csv_dir = (
        f"/home/jonasklotz/Studys/MASTERS/results_22_4_final/{dataset_name}/metrics"
    )
    visualization_save_dir = f"/home/jonasklotz/Studys/MASTERS/results_22_4_final/{dataset_name}/cutmix/visualizations"
    os.makedirs(visualization_save_dir, exist_ok=True)
    cutmix_csv_path = f"/home/jonasklotz/Studys/MASTERS/results_22_4_final/{dataset_name}/cutmix/{dataset_name}_cutmix_cleaned.csv"
    result_df = get_cutmix_df(cutmix_csv_path)
    # splitdf on model name
    # Creating separate DataFrames for resnet and vgg
    df_resnet = result_df[result_df["Model"] == "resnet"].reset_index(drop=True)
    df_vgg = result_df[result_df["Model"] == "vgg"].reset_index(drop=True)

    df_full, _ = _load_df(csv_dir, visualization_save_dir, task="multilabel")
    df_full = preprocess_metrics(df_full)
    df_full["Method"] = df_full["Method"].replace(rename_dict)

    dataset_name = dataset_rename_dict[dataset_name]
    process_and_plot_cutmix(
        df_full,
        df_resnet,
        visualization_save_dir,
        model_name="ResNET",
        dataset_name=dataset_name,
    )
    process_and_plot_cutmix(
        df_full,
        df_vgg,
        visualization_save_dir,
        model_name="VGG",
        dataset_name=dataset_name,
    )


def process_and_plot_cutmix(
    df_full,
    df_resnet,
    visualization_save_dir,
    model_name="resnet",
    dataset_name="DeepGlobe",
):
    df_full = pd.merge(
        df_full,
        df_resnet,
        on="Method",
        how="inner",
        validate="many_to_one",
    )
    categories = get_metrics_categories(df_full["Metric"].unique())
    # add the categories to the df
    df_full["Category"] = df_full["Metric"].apply(lambda x: categories[x])
    df = (
        df_full.groupby(
            ["Method", "Metric", "Category", "Parameter", "Parameter_Value"]
        )["Value"]
        .mean()
        .reset_index()
    )
    plot_with_correlation(
        df,
        "Metric",
        "Parameter_Value",
        "Parameter",
        f"{dataset_name}: Correlational Analysis of xAI Metrics with Test mAP in CutMix Training for {model_name}",
        visualization_save_dir,
    )
    # plot_with_correlation(
    #     df,
    #     "Category",
    #     "Parameter_Value",
    #     "Parameter",
    #     f"{dataset_name}: Correlational Analysis of xAI Metric Categories with Test mAP in CutMix Training for {model_name}",
    #     visualization_save_dir,
    # )


def get_dataframes(dataset_name, slc=False):
    csv_dir = (
        f"/home/jonasklotz/Studys/MASTERS/results_22_4_final/{dataset_name}/metrics"
    )
    visualization_save_dir = f"{csv_dir}/visualizations"
    os.makedirs(visualization_save_dir, exist_ok=True)
    # metric_to_plot = "IROF"
    # metric_1 = "Region Segmentation LERF"
    # metric_2 = "Region Segmentation MORF"
    save_path = f"{visualization_save_dir}/df_full.csv"
    save_path = os.path.abspath(save_path)
    # if file does not exist read df
    df_full, time_df_long = _load_df(
        csv_dir, visualization_save_dir, task="singlelabel" if slc else "multilabel"
    )
    time_df_long["Method"] = time_df_long["Method"].replace(rename_dict)

    df_full["Metric"] = df_full["Metric"].replace(rp_renaming_dict)
    time_df_long["Metric"] = time_df_long["Metric"].replace(rp_renaming_dict)
    df_full = preprocess_metrics(df_full)
    return df_full, time_df_long


if __name__ == "__main__":
    app()
