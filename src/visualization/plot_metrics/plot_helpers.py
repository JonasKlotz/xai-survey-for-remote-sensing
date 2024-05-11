import os
from typing import Optional

import pandas as pd
from plotly import express as px, graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

from quantus import AVAILABLE_METRICS
from scipy.stats import pearsonr

COLORSCALE = "RdBu_r"


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

    # Replace problematic characters in filenames
    filename = title_text.replace(" ", "_").replace("/", "_").replace("\\", "_")
    save_path = os.path.join(visualization_save_dir, filename + ".png")

    # fig.update_layout(
    #     autosize=False,
    #     width=1920,
    #     height=1080)
    # Save as JPEG, specify the dimensions directly within write_image if needed
    fig.write_image(save_path)
    print(f"Figure saved to {save_path}")


def plot_bar_metric_comparison(
    input_df, title_text=None, visualization_save_dir=None, y_axis_unit=None
):
    if title_text is None:
        title_text = "Comparison of Methods Across all Metrics"

    df_mean = input_df.groupby(["Method", "Metric"])["Value"].mean().reset_index()
    df_mean["Value"] = df_mean["Value"].round(2)

    plot_generalized(
        df_mean,
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
    if title_text is None:
        title_text = "Number of top scored Metrics per Method"

    # calculate mean over all metrics
    df_mean = df_test.groupby(["Method", "Metric"])["Value"].mean().reset_index()
    # round to 2 decimal places
    df_mean["Value"] = df_mean["Value"].round(2)

    # count which methods have the highest mean value for each metric
    # idx = df_mean.groupby("Metric")["Value"].idxmax()
    # Select the rows that correspond to the max values
    # max_value_methods = df_mean.loc[idx]
    max_indices = df_mean.groupby("Metric")["Value"].transform(max) == df_mean["Value"]
    max_value_methods = df_mean[max_indices]
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
    plot_data = pd.merge(
        method_performance_count,
        metrics_per_method,
        on="Method",
        how="inner",
        validate="many_to_many",
    )

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


def plot_result_distribution(df, dataset_name, visualization_save_dir):
    metrics = df["Metric"].unique()
    for metric in metrics:
        fig = px.violin(
            df[df["Metric"] == metric],
            x="Value",
            color="Method",
            title=f"{dataset_name}: Distribution of {metric} Results",
            labels={"Value": metric},
        )
        save_fig(
            fig,
            f"Distribution of {dataset_name} Results for {metric}",
            visualization_save_dir,
        )

        fig.show()


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


def plot_bar_metric_categories(
    df_full, categories, visualization_save_dir, title_prefix=""
):
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
            title_text=f"{title_prefix}Comparison of Methods for {category}",
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

    df_mean = df_filtered.groupby(["Method", "Metric"])["Value"].mean().reset_index()
    df_mean["Value"] = df_mean["Value"].round(2)

    plot_generalized(
        df_mean, "Metric", title_text, visualization_save_dir=visualization_save_dir
    )


def plot_correlation(results, visualization_save_dir=None, title=None):
    fig = go.Figure()
    # Add bars for correlation coefficients
    fig.add_trace(
        go.Bar(
            x=results.index,
            y=results["Correlation"],
            marker_color=results["Correlation"],  # Use correlation value for color
            marker=dict(colorscale=COLORSCALE),  # Specify your desired colorscale here
            text=results["P-value Text"],  # Use p-value text for hoverinfo
            hoverinfo="text+y",
        )
    )
    # Highlight the 0 line
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    # Annotations for statistical significance
    for i, row in results.iterrows():
        if row["Significant"]:  # If significant, add an annotation
            fig.add_annotation(
                x=i,
                y=row["Correlation"],
                text="*",
                showarrow=False,
                yshift=10,
                font=dict(color="red", size=20),
            )
    # Add general annotation for statistical relevance
    fig.add_annotation(
        text="* p < 0.05 is statistically significant",
        xref="paper",
        yref="paper",
        x=1,
        y=1.02,
        align="right",
        showarrow=False,
        font=dict(size=12),
        xanchor="right",
        yanchor="bottom",
    )

    if title is None:
        title = "Correlation Coefficients with Statistical Significance"

    categories = get_metrics_categories(results.index.unique())

    # Add annotations for each category
    for i, metric in enumerate(results.index):
        # Assuming you have a function or a way to get the category of a metric
        category = categories[
            metric
        ]  # Assuming 'categories' dictionary maps metrics to their categories
        fig.add_annotation(
            x=i,
            y=1,
            xref="x",
            yref="paper",
            text=category,
            showarrow=False,
            font=dict(family="Arial", size=12, color="black"),
            align="center",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Correlation Coefficient",
        xaxis_tickangle=-45,
    )

    # set size
    fig.update_layout(
        autosize=False,
        width=2500,
        height=1000,
    )

    save_fig(fig, title, visualization_save_dir)
    fig.show()


def standart_scale_df(df):
    scaler = StandardScaler()
    df_scaled = df.copy()
    scaled_data = scaler.fit_transform(df_scaled)
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

    return df_scaled


def barplot_metrics_time(df, visualization_save_dir=None, title_text=None):
    if df["Value"].dtype == object:
        df["Value"] = pd.to_timedelta(df["Value"])

    avg_times = df.groupby("Metric")["Value"].mean().reset_index()
    avg_times["AvgTimeSeconds"] = avg_times["Value"].dt.total_seconds()

    # Sort by average time
    avg_times = avg_times.sort_values(by="AvgTimeSeconds", ascending=False)

    # Creating a bar plot with sorted values and uniform color
    fig = px.bar(
        avg_times,
        x="Metric",
        y="AvgTimeSeconds",
        labels={"AvgTimeSeconds": "Average Time (seconds)"},
        title=title_text if title_text else "Average Time by Metric",
        color_discrete_sequence=["grey"],
    )
    fig.update_layout(
        yaxis_type="log",  # Set y-axis to logarithmic scale
        title_font_size=20,
        font=dict(size=16, color="black"),
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
    )
    fig.show()

    save_fig(fig, title_text, visualization_save_dir)


def plot_time_matrix(df_full, visualization_save_dir=None, title_text=None):
    df_full["Value"] = pd.to_timedelta(df_full["Value"]).dt.total_seconds()
    df_mean = df_full.groupby(["Method", "Metric"])["Value"].mean().reset_index()

    # reorder the df that the columns are the metrics and the rows are the methods
    df_mean = df_mean.pivot(index="Method", columns="Metric", values="Value")
    # get categories
    categories = get_metrics_categories(df_full["Metric"].unique())
    # reorder the columns using the categories dict
    category_names = sorted(set(categories.values()))
    categories_lists = {k: [] for k in category_names}
    for metric, category in categories.items():
        categories_lists[category].append(metric)
    # reorder the columns
    new_columns = [li for sublist in categories_lists.values() for li in sublist]
    df_mean = df_mean[new_columns]

    row_order = [
        "Guided GradCAM",
        "LIME",
        "DeepLift",
        "Integrated Gradients",
        "LRP",
        "GradCAM",
        "Occlusion",
    ]

    # row_order = [rename_dict[row] for row in row_order]
    # reorder the rows
    df_mean = df_mean.loc[row_order]

    df_scaled = df_mean  # standart_scale_df(df_grouped)
    # Convert the DataFrame to a 2D array
    matrix = df_scaled.values

    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=df_scaled.columns,  # Metrics as x
            y=df_scaled.index,  # Methods as y
            hoverongaps=False,  # Don't allow hovering over gaps
            colorscale="OrRd",
            text=matrix,  # Set text to the same z-values
            texttemplate="%{text:.2f}",
        ),
    )
    if title_text is None:
        title_text = "Standard Scaled Metrics"

    # Update the layout if needed
    fig.update_layout(
        title=title_text,
        xaxis_title="Metrics",
        yaxis_title="Methods",
        xaxis={"side": "bottom"},
    )  # Ensure the x-axis (metrics) is at the top

    categories = get_metrics_categories(df_full["Metric"].unique())

    # Add annotations for each category
    for i, metric in enumerate(df_scaled.columns):
        # Assuming you have a function or a way to get the category of a metric
        category = categories[
            metric
        ]  # Assuming 'categories' dictionary maps metrics to their categories
        fig.add_annotation(
            x=i,
            y=1.05,
            xref="x",
            yref="paper",
            text=category,
            showarrow=False,
            font=dict(family="Arial", size=20, color="black"),
            align="center",
        )

    # Adjust layout to ensure annotations are visible
    fig.update_layout(margin=dict(t=100))

    # set figure size
    fig.update_layout(
        autosize=False,
        width=2500,
        height=1000,
    )

    # increase fontsize
    fig.update_layout(font=dict(size=20))

    save_fig(fig, title_text, visualization_save_dir)

    # Show the figure
    fig.show()


def plot_matrix(df_full, visualization_save_dir=None, title_text=None):
    df_mean = df_full.groupby(["Method", "Metric"])["Value"].mean().reset_index()
    df_mean["Value"] = df_mean["Value"].round(2)

    # reorder the df that the columns are the metrics and the rows are the methods
    df_mean = df_mean.pivot(index="Method", columns="Metric", values="Value")
    # get categories
    categories = get_metrics_categories(df_full["Metric"].unique())
    # reorder the columns using the categories dict
    category_names = sorted(set(categories.values()))
    categories_lists = {k: [] for k in category_names}
    for metric, category in categories.items():
        categories_lists[category].append(metric)
    # reorder the columns
    new_columns = [li for sublist in categories_lists.values() for li in sublist]
    df_mean = df_mean[new_columns]

    row_order = [
        "Guided GradCAM",
        "LIME",
        "DeepLift",
        "Integrated Gradients",
        "LRP",
        "GradCAM",
        "Occlusion",
    ]

    # row_order = [rename_dict[row] for row in row_order]
    # reorder the rows
    df_mean = df_mean.loc[row_order]

    df_scaled = df_mean  # standart_scale_df(df_grouped)
    # Convert the DataFrame to a 2D array
    matrix = df_scaled.values

    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=df_scaled.columns,  # Metrics as x
            y=df_scaled.index,  # Methods as y
            hoverongaps=False,  # Don't allow hovering over gaps
            colorscale="OrRd",
            text=matrix,  # Set text to the same z-values
            texttemplate="%{text:.2f}",
            zmin=0,
            zmax=1,
        ),
    )
    if title_text is None:
        title_text = "Standard Scaled Metrics"

    # Update the layout if needed
    fig.update_layout(
        title=title_text,
        xaxis_title="Metrics",
        yaxis_title="Methods",
        xaxis={"side": "bottom"},
    )  # Ensure the x-axis (metrics) is at the top

    categories = get_metrics_categories(df_full["Metric"].unique())

    # Add annotations for each category
    for i, metric in enumerate(df_scaled.columns):
        # Assuming you have a function or a way to get the category of a metric
        category = categories[
            metric
        ]  # Assuming 'categories' dictionary maps metrics to their categories
        fig.add_annotation(
            x=i,
            y=1.05,
            xref="x",
            yref="paper",
            text=category,
            showarrow=False,
            font=dict(family="Arial", size=20, color="black"),
            align="center",
        )

    # Adjust layout to ensure annotations are visible
    fig.update_layout(margin=dict(t=100))

    # set figure size
    fig.update_layout(
        autosize=False,
        width=2500,
        height=1000,
    )

    # increase fontsize
    fig.update_layout(font=dict(size=20))

    save_fig(fig, title_text, visualization_save_dir)

    # Show the figure
    fig.show()


def plot_with_correlation(
    df: pd.DataFrame,
    grouping_column: str,
    col: str,
    parameter_column: str,
    title: str,
    visualization_save_dir: Optional[str] = None,
):
    """
    Creates a faceted scatter plot with correlations annotated in the subplot titles,
    where plots are ordered by the calculated correlation values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be plotted. Must include the specified columns.
    grouping_column : str
        The name of the column in `df` used to group data and create facets.
    col : str
        The column name representing the numerical data to be used on the x-axis and to calculate correlation against 'Value'.
    parameter_column : str
        The column name used to symbolize different markers in the scatter plot, adding another dimension to the facets.
    title : str
        The title for the scatter plot.
    visualization_save_dir : Optional[str], default None
        The directory where the plot will be saved. If None, the plot will not be saved.

    Returns
    -------
    None
        The function generates a plot and shows it using `fig.show()`. Optionally, saves the plot if a directory is provided.

    """
    if grouping_column == "Metric":
        plots_per_row = 3
        plot_height = 200
    else:
        plots_per_row = 2
        plot_height = 400
    # Calculate correlation for the specified grouping
    results = df.groupby(grouping_column).apply(
        calculate_correlation_and_significance, col=col
    )
    corr_dict = results["Correlation"].to_dict()
    p_value_dict = results["P-value"].to_dict()
    # Sort the correlations, and extract the index (group names) in the order of correlation
    sorted_groups = results["Correlation"].sort_values(ascending=False).index.tolist()

    # Determine the number of facets
    num_facets = len(sorted_groups)

    # Add a new column to df for correlation annotations in the plot
    df["Correlation"] = (
        df[grouping_column].map(corr_dict).apply(lambda x: f"Correlation: {x:.2f}")
    )

    # Adjust the height based on the number of facets
    height = max(
        plot_height, plot_height * ((num_facets + 1) // 2)
    )  # Adjust the multiplier for height as needed

    # Creating a faceted scatter plot with correlation in titles and ordering plots by correlation
    fig = px.scatter(
        df,
        x=col,
        y="Value",
        color="Method",
        symbol=parameter_column,
        title=title,
        labels={"Value": "Metric Performance", col: "Training Accuracy"},
        facet_col=grouping_column,
        facet_col_wrap=plots_per_row,
        category_orders={
            grouping_column: sorted_groups
        },  # Use sorted order for plotting
        height=height,
        width=height,
    )  # Dynamic height based on the number of plots

    # Update subplot titles with correlation values and adjust layout for better readability
    for key, corr in corr_dict.items():
        fig.for_each_annotation(
            lambda a: a.update(
                text=a.text.split("=")[-1]
                + f", Corr: {corr:.2f}, p-value: {p_value_dict[key]:.2f}"
            )
            if key in a.text
            else ()
        )

    # Update layout settings to improve readability and spacing
    fig.update_layout(
        title_font_size=18,
        font_size=12,
        legend_title_font_size=12,
        legend_font_size=10,
        # Adjusting margins and spacing between plots
        margin=dict(
            l=50, r=50, t=80, b=50
        ),  # Adjust left, right, top, bottom margins as needed
        grid={
            "rows": (num_facets + 1) // 2,
            "columns": 2,
            "pattern": "independent",
            "xgap": 1,
            "ygap": 1,
        },  # Increased gaps
    )
    # Ensure x-axis and y-axis tick labels are visible on all subplots
    fig.update_xaxes(tickmode="auto", showticklabels=True)
    fig.update_yaxes(tickmode="auto", showticklabels=True)

    fig.show()
    save_fig(fig, title, visualization_save_dir)


def calculate_correlation_and_significance(x, col):
    # Calculate Pearson's correlation coefficient and the p-value
    correlation, p_value = pearsonr(x[col], x["Value"])
    return pd.Series({"Correlation": correlation, "P-value": p_value})
