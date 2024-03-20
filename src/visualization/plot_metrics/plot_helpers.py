import os

import pandas as pd
from plotly import express as px, graph_objects as go
from plotly.subplots import make_subplots

from quantus import AVAILABLE_METRICS


def get_method_colors(methods):
    """Generates a dictionary mapping methods to colors."""
    # order methods alphabetically and map them to colors
    methods = sorted(methods)
    return {
        method: color for method, color in zip(methods, px.colors.qualitative.Plotly)
    }


def create_fig(subplot_titles, n_rows, n_cols, title_text, y_axis_order):
    """Creates and returns a figure with subplots and updates the layout."""
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )
    fig.update_layout(
        height=500 * n_rows,
        width=500 * n_cols,
        title_text=title_text,
        showlegend=False,
        font=dict(size=20),
    )
    fig.for_each_annotation(lambda a: a.update(font_size=20))
    fig.update_yaxes(categoryorder="array", categoryarray=y_axis_order)
    return fig


def add_traces(fig, df, methods, method_colors, row, col):
    """Adds bar traces to the figure."""
    for method in methods:
        df_method = df[df["Method"] == method]
        fig.add_trace(
            go.Bar(
                x=df_method["Method"],
                y=df_method["Value"],
                name=method,
                marker_color=method_colors[method],
            ),
            row=row,
            col=col,
        )
    # if metric is in the df
    if "Metric" in df.columns:
        metric_name = df["Metric"].unique()[0]
        categories = get_metrics_categories([metric_name])
        if categories[metric_name] == "Localisation":
            y_axis = list(fig.select_yaxes(row=row, col=col))[0]
            # set range for y axis between min and 1
            df_min = df["Value"].min()
            df_min = min(df_min, 0)
            y_axis.update(range=[df_min, 1])


def plot_generalized(
    df_grouped,
    subplot_col,
    title_text,
    n_plots_per_row=4,
    visualization_save_dir=None,
    y_axis_unit=None,
):
    """Generalized plotting function that expects pre-grouped data."""
    subplot_items = df_grouped[subplot_col].unique()
    methods = df_grouped["Method"].unique()

    if len(subplot_items) == 0:
        print("No data available for the selected metric.")
        return

    n_items = len(subplot_items)
    method_colors = get_method_colors(methods)

    n_rows = max(
        1, (n_items + n_plots_per_row - 1) // n_plots_per_row
    )  # Ensure at least 1 row
    n_cols = min(n_items, n_plots_per_row)

    fig = create_fig(subplot_items, n_rows, n_cols, title_text, methods)

    for i, item in enumerate(subplot_items, start=1):
        row = (i - 1) // n_cols + 1
        col = (i - 1) % n_cols + 1
        df_filtered = df_grouped[df_grouped[subplot_col] == item]

        add_traces(fig, df_filtered, methods, method_colors, row, col)

    # Update layout
    if y_axis_unit is not None:
        fig.update_yaxes(title_text=y_axis_unit)
    fig.show()
    save_fig(fig, title_text, visualization_save_dir)
    return fig


def save_fig(fig, title_text, visualization_save_dir):
    if visualization_save_dir is None:
        return

    os.makedirs(visualization_save_dir, exist_ok=True)

    save_path = visualization_save_dir + "/" + title_text + ".svg"
    # fig.write_image(save_path)
    jpg_save_path = save_path.replace(".svg", ".jpg")
    fig.write_image(jpg_save_path)
    print(f"Figure saved to {jpg_save_path}")


def plot_bar_metric_comparison(
    input_df, title_text=None, visualization_save_dir=None, y_axis_unit=None
):
    if title_text is None:
        title_text = "Comparison of Methods Across all Metrics"

    df_grouped = input_df.groupby(["Method", "Metric"])["Value"].mean().reset_index()

    plot_generalized(
        df_grouped,
        "Metric",
        title_text,
        visualization_save_dir=visualization_save_dir,
        y_axis_unit=y_axis_unit,
        n_plots_per_row=5,
    )


def plot_bar_single_metric(
    input_df, metric_to_plot, title_text=None, visualization_save_dir=None
):
    if title_text is None:
        title_text = f"Comparison of Methods for {metric_to_plot}"

    df_filtered = input_df[input_df["Metric"] == metric_to_plot]
    df_grouped = df_filtered.groupby(["Method", "Class"])["Value"].mean().reset_index()

    plot_generalized(
        df_grouped, "Class", title_text, visualization_save_dir=visualization_save_dir
    )


def plot_metrics_comparison_scatter(
    df,
    index_cols,
    columns_col,
    values_col,
    metric_x,
    metric_y,
    visualization_save_dir=None,
):
    """
    Generates a single plot with multiple subplots, each comparing two specified metrics across different methods
    by first pivoting the DataFrame.

    Parameters:
    - df: DataFrame, original DataFrame with methods, classes, metrics, and values.
    - index_cols: list, column names to set as index in the pivot table (e.g., ['Method', 'Class']).
    - columns_col: str, column name to pivot into columns.
    - values_col: str, column name to use for values in the pivot table.
    - metric_x: str, name of the first metric to compare.
    - metric_y: str, name of the second metric to compare.
    - height: int, optional, height of the figure.
    - width: int, optional, width of the figure.
    """
    # Pivot the DataFrame
    df_pivoted = df.pivot_table(
        index=index_cols,
        columns=columns_col,
        values=values_col,
        aggfunc="mean",
        fill_value=0,
    ).reset_index()

    # Unique methods for subplot titles
    methods = df_pivoted[
        index_cols[0]
    ].unique()  # Assuming the first index_col is 'Method'
    rows = 1
    cols = len(methods)
    # Setup subplots
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=methods)

    for i, method in enumerate(methods, start=1):
        df_filtered = df_pivoted[df_pivoted[index_cols[0]] == method]
        fig.add_trace(
            go.Scatter(
                x=df_filtered[metric_x],
                y=df_filtered[metric_y],
                mode="markers",
                name=method,
                marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
                text=df_filtered[index_cols[1]],
            ),  # Assuming the second index_col is 'Class'
            row=1,
            col=i,
        )

    # Update layout
    height = 600  # 400 * rows
    width = 400 * cols
    title_text = f"Comparison: {metric_x} vs. {metric_y} Across Methods"

    fig.update_layout(height=height, width=width, title_text=title_text)
    fig.update_xaxes(title_text=metric_x)
    fig.update_yaxes(title_text=metric_y)

    fig.show()
    save_fig(fig, title_text, visualization_save_dir)
    return fig


def plot_best_overall_method(
    df_test, all_methods, visualization_save_dir=None, title_text=None, show=True
):
    # calculate mean over all metrics
    df_mean = df_test.groupby(["Method", "Metric"])["Value"].mean().reset_index()
    # count which methods have the highest mean value for each metric
    idx = df_mean.groupby("Metric")["Value"].idxmax()
    # Select the rows that correspond to the max values
    max_value_methods = df_mean.loc[idx]
    # Assuming 'max_value_methods' is your DataFrame with the methods that achieved the max values for each metric
    # First, get the count of metrics where each method achieved the max value
    method_performance_count = max_value_methods["Method"].value_counts().reset_index()
    method_performance_count.columns = ["Method", "Count"]
    # Next, create a list of metrics for each method where it achieved the max value, for the hover data
    metrics_per_method = (
        max_value_methods.groupby("Method")["Metric"].apply(list).reset_index()
    )
    metrics_per_method.columns = ["Method", "Metrics"]
    # Merge to get a single DataFrame with method, count of max performance, and the metrics names
    plot_data = pd.merge(method_performance_count, metrics_per_method, on="Method")

    # Add methods that did not achieve the max value for any metric
    missing_methods = list(set(all_methods) - set(plot_data["Method"]))
    missing_methods_df = pd.DataFrame(
        {"Method": missing_methods, "Count": 0, "Metrics": ""}
    )
    plot_data = pd.concat([plot_data, missing_methods_df], ignore_index=True)

    # Generate colors for methods
    method_colors = get_method_colors(plot_data["Method"])

    # Map methods to their colors for the plot
    plot_data["Color"] = plot_data["Method"].map(method_colors)

    # Create the bar plot with method-specific colors
    fig = px.bar(
        plot_data,
        x="Method",
        y="Count",
        text="Count",
        hover_data=["Metrics"],
        title=title_text,
        color="Method",
        color_discrete_map=method_colors,
        height=800,
        width=800,
    )  # Apply colors here

    if title_text is None:
        title_text = "Number of top scored Metrics per Method"
    # Customize y axis text
    fig.update_yaxes(title_text="Number of top scored Metrics")
    # Customize hover text to display metrics names
    fig.update_traces(
        hovertemplate="<br>".join(
            ["Method: %{x}", "Max Count: %{y}", "Metrics: %{customdata[0]}"]
        )
    )
    # increase fontsize
    fig.update_layout(font=dict(size=20))
    if show:
        fig.show()
    save_fig(fig, title_text, visualization_save_dir)
    return fig


def get_metrics_categories(metrics):
    metrics_dict = {k: None for k in metrics}
    # Map metrics to AVAILABLE_METRICS
    for m_name in metrics:
        for category, metric_names in AVAILABLE_METRICS.items():
            # this is a bit hacky, but it works. Handles EG Region Segmentation both orders
            for metric_name in metric_names:
                if metric_name in m_name:
                    metrics_dict[m_name] = category
    return metrics_dict


def plot_bar_metric_categories(df_full, categories, visualization_save_dir):
    categorie_names = list(set(categories.values()))
    # iterate over the categories and plot the metrics that belong to the category
    for category in categorie_names:
        # get the metrics that belong to the category
        metrics = [k for k, v in categories.items() if v == category]
        # filter the df to only contain the metrics that belong to the category
        df_filtered = df_full[df_full["Metric"].isin(metrics)]
        # plot the metrics
        plot_bar_metric_comparison(
            df_filtered,
            title_text=f"Comparison of Methods for {category}",
            visualization_save_dir=visualization_save_dir,
        )


def plot_best_metric_per_category(
    df_avg, categories, visualization_save_dir, title_text=None
):
    categorie_names = list(set(categories.values()))
    # iterate over the categories and plot the metrics that belong to the category
    for category in categorie_names:
        # get the metrics that belong to the category
        metrics = [k for k, v in categories.items() if v == category]
        # filter the df to only contain the metrics that belong to the category
        df_filtered = df_avg[df_avg["Metric"].isin(metrics)]
        # plot the metrics
        plot_best_overall_method(
            df_filtered,
            all_methods=df_filtered["Method"].unique(),
            visualization_save_dir=visualization_save_dir,
            title_text=f"{title_text}{category}",
        )


def plot_bar_double_metric(
    input_df, metrics_to_plot, title_text=None, visualization_save_dir=None, show=False
):
    if title_text is None:
        title_text = f"Comparison of Methods for {metrics_to_plot}"

    df_filtered = input_df[input_df["Metric"].isin(metrics_to_plot)]

    df_grouped = df_filtered.groupby(["Method", "Metric"])["Value"].mean().reset_index()

    plot_generalized(
        df_grouped, "Metric", title_text, visualization_save_dir=visualization_save_dir
    )
