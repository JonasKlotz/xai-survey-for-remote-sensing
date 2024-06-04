import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from visualization.plot_metrics.plot_helpers import get_metrics_categories
import numpy as np
import os


def barplot_csv(save_path, filename, csv_path):
    df = pd.read_csv(csv_path + "/" + filename)
    categories = get_metrics_categories(df["Metric"].unique())
    # add the categories to the df
    df["Category"] = df["Metric"].apply(lambda x: categories[x])
    metric_rename_dict = {
        "Region Perturbation MORF": "RP MORF",
        "Region Perturbation LERF": "RP LERF",
        "Relevance Mass Accuracy": "RMA",
        "Relevance Rank Accuracy": "RRA",
        "Relative Input Stability": "RIS",
        "Relative Output Stability": "ROS",
        "Effective Complexity": "ECO",
        "Pointing Game": "PG",
        "Top-K Intersection": "TKI",
        "Attribution Localisation": "AL",
        "Max-Sensitivity": "MS",
    }
    df["Metric"] = df["Metric"].apply(
        lambda x: metric_rename_dict[x] if x in metric_rename_dict else x
    )
    # drop nan values
    df = df.dropna()

    save_path = (
        save_path
        + "/metric_barplot_"
        + filename.replace(".csv", ".pdf").replace(" ", "_").replace(":", "_")
    )
    barplot_correlations(df, save_path)


def barplot_correlations(df, save_path):
    # Sort the dataframe based on the Category column alphabetically
    df = df.sort_values("Category")

    # Global min and max for the correlation values to fix the color scale
    global_min = df["Correlation"].min()
    global_max = df["Correlation"].max()

    # Create a list of unique categories
    categories = sorted(df["Category"].unique())
    # total_categories = len(categories)

    # Determine the number of rows needed for three columns per row
    rows = 2  # Add 2 for ceiling without using math.ceil
    cols = 3

    # Create subplots with multiple columns
    subplot_titles = [
        f"{i}) {category}" for i, category in enumerate(categories, start=1)
    ]
    fig = make_subplots(
        rows=rows,
        cols=3,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    # Plot each category in a different subplot
    subplot_counter = 1
    for i, category in enumerate(categories, start=1):
        # Calculate the position of the subplot
        row = (subplot_counter - 1) // cols + 1
        col = (subplot_counter - 1) % cols + 1

        # Filter the data for the current category
        filtered_data = df[df["Category"] == category]

        # Extract correlations and metrics, then sort by correlations
        sorted_indices = np.argsort(-filtered_data["Correlation"].values)
        sorted_correlations = filtered_data.iloc[sorted_indices]["Correlation"]
        sorted_metrics = filtered_data.iloc[sorted_indices]["Metric"]

        # Create a bar chart for the sorted data
        fig.add_trace(
            go.Bar(
                x=sorted_metrics,
                y=sorted_correlations,
                text=sorted_correlations.apply(lambda x: f"{x:.2f}"),
                textposition="auto",
                marker=dict(
                    color=sorted_correlations,
                    colorscale="RdBu_r",
                    cmin=global_min,
                    cmax=global_max,
                ),
            ),
            row=row,
            col=col,
        )

        # Update y-axis for each subplot
        axis_name = f"yaxis{subplot_counter}" if subplot_counter > 1 else "yaxis"
        fig.update_layout(
            **{
                axis_name: {
                    "tickmode": "auto",  # Let Plotly decide the best tickmode
                    "nticks": 10,  # Maximum number of tick marks
                    "range": [
                        round(1.1 * global_min, 1),
                        round(1.1 * global_max, 1),
                    ],  # Set the range a bit beyond the min and max for clarity
                    "tickformat": ".1f",  # Formatting tick labels to two decimal places
                }
            }
        )
        fig.update_xaxes(title_text="xAI Metrics", row=row, col=col)
        fig.update_yaxes(title_text="Correlation Coefficient", row=row, col=col)

        subplot_counter += 1

    # Adjusting the font size for subplot titles
    for i, annot in enumerate(fig.layout.annotations):
        annot.update(font=dict(size=30))

    # Update layout to make the plot look nice
    fig.update_layout(
        height=700 * rows,
        width=800 * 3,
        title_text="",
        showlegend=False,
        font=dict(size=25),
    )

    # Save the figure if save_path is not None
    if save_path:
        fig.write_image(save_path)
        print(f"Saved the plot to {save_path}")

    # fig.show()


def main():
    main_dir = "/home/jonasklotz/Studys/MASTERS/results_22_4_final/new_correlations"
    csv_dir = f"{main_dir}/data"
    save_dir = f"{main_dir}/plots/metric_barplots"
    # read all csv files in the directory
    for filename in os.listdir(csv_dir):
        if filename.endswith(".csv"):
            barplot_csv(save_dir, filename, csv_dir)


if __name__ == "__main__":
    main()
