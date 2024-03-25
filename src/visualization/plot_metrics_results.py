import os
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import shapely

from quantus.helpers.constants import AVAILABLE_METRICS
import plotly.graph_objects as go

# Constants
UP_ARROW = "\u2191"  # up arrow
DOWN_ARROW = "\u2193"  # down arrow


# RESULTS_DIR_PATH = "/home/jonasklotz/Studys/MASTERS/XAI/results/metrics/caltech101_vgg_good_old"
RESULTS_DIR_PATH = "/home/jonasklotz/Studys/MASTERS/XAI/results/caltech101_vgg_final_results_with_metrics"

# var = {'region_perturb':"Re", 'selectivity', 'infidelity',
#        'relative_input_stability', 'relative_output_stability', 'sparseness',
#        'complexity', 'effective_complexity', 'random_logits', 'completeness',
#        'non_sensitivity', 'pointing_game', 'attribution_localisation',
#        'top_k_intersection', 'relevance_rank_accuracy',
#        'relevance_mass_accuracy', 'auc'}


def read_all_csvs(results_dir_path):
    metrics_csvs = []
    time_csvs = []
    labels_csvs = []
    for file in glob.glob(results_dir_path + "/*.csv"):
        if file.endswith("time.csv"):
            time_csvs.append(file)
        elif file.endswith("metrics.csv"):
            metrics_csvs.append(file)
        elif file.endswith("labels.csv"):
            labels_csvs.append(file)
    return metrics_csvs, time_csvs, labels_csvs


# def create_single_label_mean_df(csv_list):
#     mean_df = pd.DataFrame()
#     for tmp_metric_csv_path in csv_list:
#         filename = os.path.basename(tmp_metric_csv_path).replace(".csv", "")
#         tmp_df = pd.read_csv(tmp_metric_csv_path, sep=";", index_col=None, header=0)
#         # the DF contains only elements in lists e.g [1] as string, it is a single element list
#         # we need to convert it to a float
#         for col in tmp_df.columns:
#             # test if the columns contain a list, if column contains strings
#             column = tmp_df[col]
#             if pd.api.types.is_string_dtype(column.dtype):
#                 if column.str.contains(r"[\[](\d+.\d+)[\]]").any():
#                     # extract the element from the list and convert it to a float
#                     tmp_df[col] = (
#                         tmp_df[col]
#                         .str.extract(r"[\[](\d+.\d+)[\]]", expand=False)
#                         .astype(float)
#                     )
#                 elif column.str.contains(r"(\d+.\d+)").any():
#                     tmp_df[col] = tmp_df[col].astype(float)
#                 else:
#                     print(f"Column {col} in {filename} does not contain float values")
#             elif pd.api.types.is_numeric_dtype(column.dtype):
#                 tmp_df[col] = tmp_df[col].astype(float)
#
#         # create a column with the filename as index and the mean of the metrics as columns in all_metrics_df
#         mean_df[filename] = tmp_df.mean()
#     return mean_df


def create_single_label_mean_df(csv_list):
    mean_df = pd.DataFrame()
    for tmp_metric_csv_path in csv_list:
        filename = os.path.basename(tmp_metric_csv_path).replace(".csv", "")
        tmp_df = pd.read_csv(tmp_metric_csv_path, sep=";", index_col=None, header=0)

        for col in tmp_df.columns:
            column = tmp_df[col]
            if pd.api.types.is_string_dtype(column.dtype):
                # Extract float from string format "[float]"
                tmp_df[col] = (
                    tmp_df[col]
                    .str.extract(r"\[?(\d+\.\d+)\]?", expand=False)
                    .astype(float)
                )
            elif pd.api.types.is_numeric_dtype(column.dtype):
                # Ensure the column is treated as float
                tmp_df[col] = tmp_df[col].astype(float)

        mean_df[filename] = tmp_df.mean()
    return mean_df


def create_multi_label_mean_df(data_csv_list, label_csv_list):
    pass


def main():
    metrics_csvs, time_csvs, labels_csvs = read_all_csvs(RESULTS_DIR_PATH)
    mean_metrics_df = create_single_label_mean_df(metrics_csvs).T

    metrics_in_df = get_metric_objects(mean_metrics_df)

    columns_to_apply_log = [
        "Relative Input Stability",
        "Relative Output Stability",
        "Complexity",
        "Infidelity",
    ]
    for col in columns_to_apply_log:
        new_col_name = col
        mean_metrics_df[new_col_name] = mean_metrics_df[col].apply(np.log)
        # mean_metrics_df.drop(columns=col, inplace=True)

    category_df_dict = defaultdict(pd.DataFrame)

    for column_name in mean_metrics_df.columns:
        metric = metrics_in_df[column_name]
        if metric.score_direction.value == "higher":
            new_column_name = column_name + " " + UP_ARROW
        else:
            new_column_name = column_name + " " + DOWN_ARROW

        mean_metrics_df[new_column_name] = mean_metrics_df[column_name]
        # remove the old column
        mean_metrics_df.drop(columns=column_name, inplace=True)

        # add the column to the category_df_dict
        category_df_dict[metric.evaluation_category.value][
            new_column_name
        ] = mean_metrics_df[new_column_name]

    localisation_df = category_df_dict["Localisation"]
    faithfulness_df = category_df_dict["Faithfulness"]

    categories = [
        "Region Segmentation ↓",
        "Selectivity ↓",
        "Random Logit ↓",
        "Effective Complexity ↓",
        "Pointing Game ↑",
        "Top-K Intersection ↑",
        "Relevance Mass Accuracy ↑",
        "Relevance Rank Accuracy ↑",
        "Attribution Localisation ↑",
        "AUC ↑",
    ]
    spiderplot_df(mean_metrics_df, categories=categories, title="All Metrics")

    # plot the localisation metrics
    spiderplot_df(localisation_df, title="Localisation Metrics")
    # plot the faithfulness metrics
    spiderplot_df(faithfulness_df, title="Faithfulness Metrics")


def spiderplot_df(input_df, categories=None, title="Spiderplot"):
    fig = go.Figure()
    if categories is None:
        categories = input_df.columns
    else:
        input_df = input_df[categories]

    for index, row in input_df.iterrows():
        fig.add_trace(
            go.Scatterpolar(r=row, theta=categories, fill="toself", name=index)
        )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                # range=[0, 1]
            )
        ),
        title=title,
        showlegend=True,
    )

    # get data back out of figure
    df = pd.concat(
        [
            pd.DataFrame(
                {"r": t.r, "theta": t.theta, "trace": np.full(len(t.r), t.name)}
            )
            for t in fig.data
        ]
    )
    # convert theta to be in radians
    df["theta_n"] = pd.factorize(df["theta"])[0]
    df["theta_radian"] = (df["theta_n"] / (df["theta_n"].max() + 1)) * 2 * np.pi
    # work out x,y co-ordinates
    df["x"] = np.cos(df["theta_radian"]) * df["r"]
    df["y"] = np.sin(df["theta_radian"]) * df["r"]

    # now generate a polygon from co-ordinates using shapely
    # & getting the area of the polygon
    df_a = df.groupby("trace").apply(
        lambda d: shapely.geometry.MultiPoint(
            list(zip(d["x"], d["y"]))
        ).convex_hull.area
    )

    # let's use the areas in the name of the traces
    fig.for_each_trace(lambda t: t.update(name=f"{t.name} {df_a.loc[t.name]:.1f}"))

    fig.show()


def get_metric_objects(mean_metrics_df):
    metrics_in_df = {}
    # Map colnames to AVAILABLE_METRICS
    for col in mean_metrics_df.columns:
        for category, metric_names in AVAILABLE_METRICS.items():
            for metric_name, metric in metric_names.items():
                if metric_name == col:
                    metrics_in_df[col] = metric
    return metrics_in_df


if __name__ == "__main__":
    main()
