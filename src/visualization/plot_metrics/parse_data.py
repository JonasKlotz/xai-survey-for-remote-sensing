import ast
import glob
import os

import numpy as np
import pandas as pd

from quantus import AVAILABLE_METRICS
from utility.cluster_logging import logger


def generate_test_df(n_metrics=2, n_methods=3, n_classes=4):
    # Generate all combinations of Method, Metric, and Class
    methods = ["Method " + str(i) for i in range(1, n_methods + 1)]
    metrics = ["Metric " + str(i) for i in range(1, n_metrics + 1)]
    classes = ["Class " + str(i) for i in range(1, n_classes + 1)]

    # Create a list for each column
    data = {"Method": [], "Metric": [], "Class": []}
    for method in methods:
        for metric in metrics:
            for class_ in classes:
                data["Method"].append(method)
                data["Metric"].append(metric)
                data["Class"].append(class_)

    # Convert to DataFrame
    tst_df = pd.DataFrame(data)

    # Add random values for the metrics
    tst_df["Value"] = np.random.randint(50, 100, size=len(tst_df))
    return tst_df


def parse_data_df(df):
    for col in df.columns:
        # Check if value is a string, then try to convert string representations of lists to actual lists
        def parse_string(x):
            try:
                return ast.literal_eval(x.strip())
            except ValueError:
                return x  # Return the original value if parsing fails

        df[col] = df[col].apply(lambda x: parse_string(x) if isinstance(x, str) else x)
        # Flatten the lists and convert to floats, handling nan values
        df[col] = df[col].apply(
            lambda x: [
                float(item) if item == item else np.nan
                for sublist in x
                for item in sublist
            ]
            if isinstance(x, list)
            else x
        )
    return df


def parse_labels_df(df):
    true_labels = []
    predicted_labels = []

    for index, row in df.iterrows():
        # Parse true labels
        true_label = row["True Label"]
        true_label = true_label.strip("[]").split(". ")
        true_label = [int(float(x)) for x in true_label]
        true_labels.append(true_label)

        # Parse predicted labels
        predicted_label = row["Predicted Label"]
        predicted_label = predicted_label.strip("[array()]").split(",")
        predicted_label = [
            int(float(x.strip())) for x in predicted_label if x.strip().isdigit()
        ]
        predicted_label_onehot = [
            1 if i in predicted_label else 0 for i in range(len(true_label))
        ]
        predicted_labels.append(predicted_label_onehot)

    true_labels_df = pd.DataFrame(
        true_labels, columns=[f"true_class{i + 1}" for i in range(len(true_labels[0]))]
    )
    predicted_labels_df = pd.DataFrame(
        predicted_labels,
        columns=[f"predicted_class{i + 1}" for i in range(len(predicted_labels[0]))],
    )
    # concatenate the two dataframes
    # all_labels_df = pd.concat([true_labels_df, predicted_labels_df], axis=1)

    return true_labels_df, predicted_labels_df


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


def safe_eval_list(list_str):
    try:
        # Pre-process the list string to handle 'nan' properly
        list_str_processed = list_str.replace("nan", "np.nan")
        # Safely evaluate the modified string into a list
        return eval(list_str_processed)
    except:  # noqa: E722
        # Return an empty list or other appropriate error handling
        return []


def expand_metrics_df(df):
    """Expand the df to a multi index df.

    Each entry in the df is a list of values, each column is representing a metric, each value in the list is a value for
    a class. This function creates a multi index df with the first level being the class and the second level being the
    metric.
    E.G.:
    Before:
    | Metric 1 | Metric 2 |
    | [1, 2]   | [3, 4]   |
    | [1, 2]   | [3, 4]   |
    After:
    | Metric 1 | Metric 2 | Metric 1 | Metric 2 |
    | Class 1  | Class 1  | Class 2  | Class 2  |
    | 1        | 3        | 2        | 4        |
    | 1        | 3        | 2        | 4        |
    """
    # Check if DataFrame is empty
    if df.empty:
        return pd.DataFrame()
    # map safe_eval_list to each cell in the DataFrame
    df = df.map(safe_eval_list)

    # Assuming each list in df columns has the same length
    class_count = len(
        df.iloc[0, 0]
    )  # Number of classes based on the length of the list in the first cell

    # Creating class names
    class_names = [f"Class_{i}" for i in range(1, class_count + 1)]

    dataframes = []

    for metric in df.columns:
        for i, class_name in enumerate(class_names):
            temp_df = pd.DataFrame(
                {
                    "SampleIndex": df.index,
                    "Metric": metric,
                    "Class": class_name,
                    "Value": df[metric].apply(lambda x: x[i]),
                }
            )
            dataframes.append(temp_df)

    expanded_df = pd.concat(dataframes, ignore_index=True)

    # round to 2 decimal places
    expanded_df = expanded_df.round(2)

    return expanded_df


def parse_data(csv_dir, n_classes=6):
    """

    Returns df like:
        df = pd.DataFrame({
        'Method': ['Method 1', 'Method 1', 'Method 2', 'Method 2'],
        'Metric': ['Metric 1', 'Metric 2', 'Metric 1', 'Metric 2'],
        'Value': [100, 200, 150, 250]
    })

    """
    metrics_csvs, time_csvs, labels_csvs = read_all_csvs(csv_dir)
    labels_dfs, metrics_dfs, time_dfs = read_into_dfs(
        labels_csvs, metrics_csvs, time_csvs
    )

    # get minimal length to cut all dfs to the same length
    min_length = min([len(df) for df in metrics_dfs])

    # cut all dfs to the same length
    metrics_dfs = [df.iloc[:min_length] for df in metrics_dfs]
    time_dfs = [df.iloc[:min_length] for df in time_dfs]
    labels_dfs = [df.iloc[:min_length] for df in labels_dfs]

    label_dfs = [parse_labels_df(df) for df in labels_dfs]

    label_df = label_dfs[0]
    correct_predictions_data = (
        np.sum(label_df[0].values == label_df[1].values, axis=1) == n_classes
    )
    correct_predictions = pd.Series(correct_predictions_data, name="CorrectPrediction")
    metrics_dfs = [expand_metrics_df(df) for df in metrics_dfs]

    # take csv as key
    keys = [file.split("/")[-1].split("_")[0] for file in metrics_csvs]
    # add key as method column to each df
    for key, df in zip(keys, metrics_dfs):
        df["Method"] = key

    # concatenate all dfs
    df_long = pd.concat(metrics_dfs, ignore_index=True)

    # remove nan values
    df_long = df_long.dropna()

    # join correct predictions to df_long using index and SampleIndex
    df_long = df_long.merge(
        correct_predictions, how="inner", left_on="SampleIndex", right_index=True
    )

    dtypes = {
        "Method": str,
        "Metric": str,
        "Class": str,
        "Value": float,
        "SampleIndex": int,
        "CorrectPrediction": bool,
    }
    df_long = df_long.astype(dtypes)

    # round to 2 decimal places
    df_long["Value"] = df_long["Value"].round(2)
    return df_long


def parse_data_df_single_label(df):
    for col in df:
        # if the column is a string, then try to convert string representations of lists to actual lists
        df[col] = df[col].apply(
            lambda x: safe_eval_list(x) if isinstance(x, str) else x
        )
        # Check if value is a list with a single element, then extract the element
        df[col] = df[col].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x
        )
    # infer the correct dtypes
    df = df.infer_objects()
    # round to 2 decimal places
    df = df.round(2)

    # # aggregate the mean for each column
    # df = df.mean().reset_index()

    df_list = []
    for metric in df.columns:
        tmp_df = pd.DataFrame(
            {
                "SampleIndex": df.index,
                "Metric": metric,
                "Value": df[metric],
            }
        )
        df_list.append(tmp_df)
    df = pd.concat(df_list, ignore_index=True)
    return df


def parse_time_df_single_label(df):
    # infer the correct dtypes
    df = df.infer_objects()

    df_list = []
    for metric in df.columns:
        tmp_df = pd.DataFrame(
            {
                "SampleIndex": df.index,
                "Metric": metric,
                "Value": df[metric],
            }
        )
        df_list.append(tmp_df)
    df = pd.concat(df_list, ignore_index=True)
    return df


def preprocess_dataframes(dfs):
    min_length = min(len(df) for df in dfs)
    max_length = max(len(df) for df in dfs)
    if min_length != max_length:
        logger.warning("The length of the dfs is not the same")
    return [df.iloc[:min_length] for df in dfs]


def calculate_correct_predictions_single_label(label_df):
    correct_predictions_data = label_df.iloc[:, 0].values == label_df.iloc[:, 1].values
    return pd.Series(correct_predictions_data, name="CorrectPrediction")


def expand_and_annotate_single_label_dfs(
    dfs,
    csv_files,
):
    expanded_dfs = [parse_data_df_single_label(df) for df in dfs]
    keys = [file.split("/")[-1].split("_")[0] for file in csv_files]
    for key, df in zip(keys, expanded_dfs):
        df["Method"] = key
    return expanded_dfs


def merge_dataframes_single_label(dfs, correct_predictions):
    df_long = pd.concat(dfs, ignore_index=True)
    return df_long.merge(
        correct_predictions,
        how="inner",
        left_on="SampleIndex",
        right_index=True,
        validate="many_to_many",
    )


def parse_data_single_label(csv_dir):
    metrics_csvs, time_csvs, labels_csvs = read_all_csvs(csv_dir)
    labels_dfs, metrics_dfs, time_dfs = read_into_dfs(
        labels_csvs, metrics_csvs, time_csvs
    )

    # Preprocess DataFrames: trim all to minimum length
    metrics_dfs = preprocess_dataframes(metrics_dfs)
    time_dfs = preprocess_dataframes(time_dfs)
    labels_dfs = preprocess_dataframes(labels_dfs)

    # Calculate correct predictions for single labels
    correct_predictions = calculate_correct_predictions_single_label(labels_dfs[0])

    # Expand and annotate DataFrames
    metrics_dfs = expand_and_annotate_single_label_dfs(metrics_dfs, metrics_csvs)

    time_dfs = [parse_time_df_single_label(df) for df in time_dfs]
    keys = [file.split("/")[-1].split("_")[0] for file in time_csvs]
    for key, df in zip(keys, time_dfs):
        df["Method"] = key

    # Merge DataFrames with predictions
    metrics_df_long = merge_dataframes_single_label(metrics_dfs, correct_predictions)
    time_df_long = merge_dataframes_single_label(time_dfs, correct_predictions)

    return metrics_df_long, time_df_long


def read_into_dfs(labels_csvs, metrics_csvs, time_csvs):
    metrics_dfs = [
        pd.read_csv(file, sep=";", index_col=None, header=0) for file in metrics_csvs
    ]
    time_dfs = [
        pd.read_csv(file, sep=";", index_col=None, header=0) for file in time_csvs
    ]
    labels_dfs = [
        pd.read_csv(file, sep=";", index_col=None, header=0) for file in labels_csvs
    ]
    return labels_dfs, metrics_dfs, time_dfs


def recalculate_score_direction(df_test):
    # parse the data
    metric_objects = get_metric_objects(df_test["Metric"].unique())
    # iterate over the metric objects and apply the score direction
    # we want to multiply -1 to the values of the metrics that are lower is better
    for metric_name, metric in metric_objects.items():
        if metric.score_direction.value == "lower":
            df_test.loc[df_test["Metric"] == metric_name, "Value"] = (
                1 - df_test.loc[df_test["Metric"] == metric_name, "Value"]
            )
    return df_test


def _z_score_scale(df, group_col, to_scale_metrics, value_col):
    for group_col_entry in to_scale_metrics:
        # get the indices of the group
        group_indices = df[df[group_col] == group_col_entry].index
        # calculate mean and standard deviation for the group
        mean_val = df.loc[group_indices, value_col].mean()
        std_val = df.loc[group_indices, value_col].std()
        # scale the values
        df.loc[group_indices, value_col] = (
            df.loc[group_indices, value_col] - mean_val
        ) / std_val
    return df


def _robust_scale(
    df, group_col, to_scale_metrics, value_col, quantile_range=(0.25, 0.75)
):
    from sklearn.preprocessing import RobustScaler, MinMaxScaler

    # Iterate over each unique entry in to_scale_metrics to apply scaling
    for metric in to_scale_metrics:
        # Initialize scalers inside the loop to ensure fresh scaling each time
        robust_scaler = RobustScaler(quantile_range=quantile_range)
        min_max_scaler = MinMaxScaler()

        # Filter the DataFrame for the current metric
        mask = df[group_col] == metric
        values = df.loc[mask, value_col].to_numpy().reshape(-1, 1)

        # Apply robust scaling
        values = robust_scaler.fit_transform(values)

        # Apply min-max scaling
        values = min_max_scaler.fit_transform(values)

        # Assign scaled values back to the DataFrame
        df.loc[mask, value_col] = values.flatten()

    return df


def scale_df(df, group_col: str = "Metric", value_col: str = "Value", scale="min_max"):
    """
    Scale the values of the df to a 0-1 scale for each group in the group_col, except for the metrics in
    not_to_scale_metrics. The scaling is done by min-max scaling.

    Parameters
    ----------
    df: pd.DataFrame
        Columns: SampleIndex, Metric, Value, Method, CorrectPrediction

    group_col
    value_col
    scale : str
        The scaling method to use. Options: min_max, z_score, robust


    Returns
    -------

    """
    assert group_col in df.columns, f"Group column {group_col} not in df columns"
    assert value_col in df.columns, f"Value column {value_col} not in df columns"

    # 0-1 scaled columns. These columns are scaled between 0 and 1 by definition.
    not_to_scale_metrics = [
        "Attribution Localization",
        "Monotonicity-Arya",
        "Pointing Game",
        "Top-K Intersection",
        "Relevance Mass Accuracy",
        "Relevance Rank Accuracy",
        "Attribution Localisation",
    ]
    all_group_col_entries = df[group_col].unique()
    to_scale_metrics = [
        metric for metric in all_group_col_entries if metric not in not_to_scale_metrics
    ]

    if scale == "min_max":
        df = _min_max_scale(df, group_col, to_scale_metrics, value_col)
    elif scale == "z_score":
        df = _z_score_scale(df, group_col, to_scale_metrics, value_col)
    elif scale == "robust":
        df = _robust_scale(df, group_col, to_scale_metrics, value_col)
    return df


def remove_outliers(df_full, group_col: str = "Metric", value_col: str = "Value"):
    """
    Remove outliers from the df_full by replacing the values that are more than 3 standard deviations away from the
    median with np.nan.

    Parameters
    ----------
    df_full : pd.DataFrame
        Columns: SampleIndex, Metric, Value, Method, CorrectPrediction
    group_col
        Column to group by
    value_col
        Column to remove outliers from
    Returns
    -------
    pd.DataFrame
        DataFrame without outliers
    """
    df_full = df_full.copy()
    medians = df_full.groupby(group_col)[value_col].median()
    stds = df_full.groupby(group_col)[value_col].std()
    print(medians)
    for group_col_entry in df_full[group_col].unique():
        group_indices = df_full[df_full[group_col] == group_col_entry].index
        median = medians[group_col_entry]
        std = stds[group_col_entry]
        df_full.loc[group_indices, value_col] = df_full.loc[
            group_indices, value_col
        ].apply(lambda x: x if abs(x - median) < 3 * std else np.nan)
    return df_full.dropna()


def _min_max_scale(df, group_col, to_scale_metrics, value_col):
    # get the min and max values for each group
    min_values = df.groupby(group_col)[value_col].min()
    max_values = df.groupby(group_col)[value_col].max()
    # iterate over the groups and scale the values
    for group_col_entry in to_scale_metrics:
        # get the min and max value for the group
        min_val = min_values[group_col_entry]
        max_val = max_values[group_col_entry]
        # get the indices of the group
        group_indices = df[df[group_col] == group_col_entry].index
        # scale the values
        df.loc[group_indices, value_col] = (
            df.loc[group_indices, value_col] - min_val
        ) / (max_val - min_val) + 0.00000001
    return df


def get_metric_objects(metrics):
    metrics_dict = {k: None for k in metrics}
    # Map metrics to AVAILABLE_METRICS
    for m_name in metrics:
        for category, metric_names in AVAILABLE_METRICS.items():
            for metric_name, metric in metric_names.items():
                if metric_name in m_name:
                    metrics_dict[m_name] = metric
    return metrics_dict


def log_some_cols(df, metrics_to_apply_log=None, base=2):
    """
    Applies logarithmic transformation to specified metrics within the 'Value' column of a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    metrics_to_apply_log (list of str): List of metric names to apply the logarithmic transformation to.
        Defaults to ["Relative Input Stability", "Relative Output Stability", "Effective Complexity"].
    base (int): Base of the logarithm. Default is 2 for log base 2.

    Returns:
    pd.DataFrame: DataFrame with the logarithmic transformation applied to specified metrics.

    Raises:
    ValueError: If there are non-positive values in the columns to which the logarithm is applied.
    """
    if metrics_to_apply_log is None:
        metrics_to_apply_log = [
            "Relative Input Stability",
            "Relative Output Stability",
            "Effective Complexity",
        ]

    for metric in metrics_to_apply_log:
        # Select the values to be transformed
        mask = df["Metric"] == metric
        values = df.loc[mask, "Value"]

        # Check for non-positive values to avoid math domain error
        if (values <= 0).any():
            # clip the values to 0.0001
            values = values.clip(0.0000001)
            #
            # raise ValueError(f"Non-positive values encountered in metric '{metric}'. Logarithm cannot be applied.")

        # Apply logarithmic transformation directly with numpy for efficiency
        df.loc[mask, "Value"] = np.log(values) / np.log(base)

    return df


def _load_df(csv_dir, visualization_save_dir, task="multilabel"):
    data_save_path = f"{visualization_save_dir}/df_full_new.csv"
    data_save_path = os.path.abspath(data_save_path)
    time_save_path = f"{visualization_save_dir}/time_df_full_new.csv"
    time_save_path = os.path.abspath(time_save_path)

    # if file does not exist read df
    if os.path.isfile(data_save_path):
        dtypes = {
            "SampleIndex": int,
            "Method": str,
            "Metric": str,
            "Value": float,
            "CorrectPrediction": bool,
        }
        # read the df from the csv
        metrics_df_long = pd.read_csv(
            data_save_path, sep=";", index_col=None, header=0, dtype=dtypes
        )
        time_df_long = pd.read_csv(
            time_save_path,
            sep=";",
            index_col=None,
            header=0,
        )
    else:
        # we parse the data and save it
        if task == "multilabel":
            metrics_df_long = parse_data(csv_dir)
            metrics_df_long.to_csv(data_save_path, index=False, sep=";")

            time_df_long = None
        elif task == "singlelabel":
            metrics_df_long, time_df_long = parse_data_single_label(csv_dir)
            metrics_df_long.to_csv(data_save_path, index=False, sep=";")
            time_df_long.to_csv(time_save_path, index=False, sep=";")
        else:
            raise ValueError(f"Task {task} not supported")
        # # save the df to a csv
    return metrics_df_long, time_df_long
