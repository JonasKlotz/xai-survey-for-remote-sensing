import os
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualization.plot_metrics.plot_helpers import get_metrics_categories


def create_correlation_bar_plot(correlation_data, title):
    # Extract and sort data for the plot
    categories = list(correlation_data.keys())
    correlations = [correlation_data[cat] for cat in categories]
    # round to 2 decimals
    correlations = [round(cor, 2) for cor in correlations]
    # Sort data by correlation values
    sorted_data = sorted(
        zip(categories, correlations), key=lambda x: x[1], reverse=True
    )
    categories, correlations = zip(*sorted_data)

    # Create the bar plot for correlations with color scale and annotations
    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=correlations,
                text=correlations,
                textposition="auto",
                marker=dict(color=correlations, cmin=-1, cmax=1, colorscale="RdBu_r"),
            )
        ]
    )

    for i, annot in enumerate(fig.layout.annotations):
        annot.update(font=dict(size=30))

    # Update the layout of the plot with annotations and font size

    fig.update_layout(
        title=title,
        xaxis_title="xAI Metric Categories",
        yaxis_title="Correlation Coefficient",
        font=dict(size=25),
        width=1600,
        height=800,
    )
    # safe as pdf
    fig.write_image(f"{title.replace(" ", "_")}_correlation_plot.pdf")

    return fig


def create_combined_correlation_bar_plot(
    correlation_data1,
    correlation_data2,
    data_dir,
    data_set_name1="BEN",
    data_set_name2="BEN",
    method_name="RRR",
    model_name1="ResNet",
    model_name2="VGG",
):
    # Create individual plots
    fig1 = create_correlation_bar_plot(correlation_data1, "")

    fig2 = create_correlation_bar_plot(correlation_data2, "")

    # Create subplot
    title1 = f"1) {model_name1} trained on {data_set_name1} using {method_name}"
    title2 = f"2) {model_name2} trained on {data_set_name2} using {method_name}"
    fig = make_subplots(rows=1, cols=2, subplot_titles=(title1, title2))

    # Add traces from individual plots
    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in fig2.data:
        fig.add_trace(trace, row=1, col=2)
    # Add annotations from individual plots
    for annotation in fig1.layout.annotations:
        fig.add_annotation(annotation, row=1, col=1)
    for annotation in fig2.layout.annotations:
        fig.add_annotation(annotation, row=1, col=2)

    fig.update_layout(
        width=1600,
        height=800,
        showlegend=False,  # Remove the legend
        font=dict(size=22),
    )

    # Update x-axis and y-axis titles for all subplots
    fig.update_xaxes(title_text="xAI Metric Categories", row=1, col=1)
    fig.update_yaxes(
        title_text="Fisher-Z Mean of Correlation Coefficients", row=1, col=1
    )
    fig.update_xaxes(title_text="xAI Metric Categories", row=1, col=2)
    fig.update_yaxes(
        title_text="Fisher-Z Mean of Correlation Coefficients", row=1, col=2
    )
    # increase subtitle font size
    for i, annot in enumerate(fig.layout.annotations):
        annot.update(font=dict(size=30))

    # Save the plot as a PDF
    fig.write_image(
        f"{data_dir}/corrplot_{method_name}_{model_name1}_{model_name2}_{data_set_name1}.pdf"
    )

    return fig


def average_correlation(coefficients):
    # Transform each correlation coefficient to Fisher Z-value
    z_values = np.arctanh(coefficients)

    # Calculate the mean of Z-values
    mean_z = np.mean(z_values)

    # Convert the average Z-value back to a correlation coefficient
    average_r = np.tanh(mean_z)

    return average_r


def get_data(data_dir):
    csv_names = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            csv_names.append(filename)
    csv_dict = {}
    # Read the data from the csv files
    for csv in csv_names:
        df = pd.read_csv(f"{data_dir}/{csv}")
        categories = get_metrics_categories(df["Metric"].unique())
        df["Category"] = df["Metric"].apply(lambda x: categories[x])
        # dropna
        df = df.dropna()
        correlation_averages = {}
        for category in df["Category"].unique():
            category_data = df[df["Category"] == category]
            coefficients = category_data["Correlation"].values
            correlation_averages[category] = average_correlation(coefficients)
        csv_dict[csv] = correlation_averages

    # extract the rrr and cutmix csvs from the csv_dict
    rrr_keys = [csv for csv in csv_dict if "RRR" in csv]
    cutmix_data = [csv for csv in csv_dict if "CutMix" in csv]
    ben_cutmix_keys = [csv for csv in cutmix_data if "BEN" in csv]
    deepglobe_cutmix_keys = [csv for csv in cutmix_data if "DeepGlobe" in csv]
    return csv_dict, rrr_keys, ben_cutmix_keys, deepglobe_cutmix_keys


def plot_all(ben_cutmix_keys, csv_dict, deepglobe_cutmix_keys, rrr_keys, data_dir):
    # Caltech101 RRR vgg
    caltech_rrr_key = [key for key in rrr_keys if "Caltech101" in key][0]
    caltech_rrr_data = csv_dict[caltech_rrr_key]
    # DeepGlobe RRR vgg
    dg_rrr_key = [key for key in rrr_keys if "DeepGlobe" in key][0]
    dg_rrr_data = csv_dict[dg_rrr_key]
    # Generate combined plot
    combined_plot = create_combined_correlation_bar_plot(
        caltech_rrr_data,
        dg_rrr_data,
        data_dir,
        data_set_name1="Caltech101",
        data_set_name2="DeepGlobe",
        method_name="RRR",
        model_name1="VGG",
        model_name2="VGG",
    )
    combined_plot.show()
    # DeepGlobe CutMix for ResNet and VGG
    dg_resnet_cutmix_data = [
        csv_dict[key] for key in deepglobe_cutmix_keys if "ResNET" in key
    ][0]
    dg_vgg_cutmix_data = [
        csv_dict[key] for key in deepglobe_cutmix_keys if "VGG" in key
    ][0]
    combined_plot = create_combined_correlation_bar_plot(
        dg_resnet_cutmix_data,
        dg_vgg_cutmix_data,
        data_dir,
        data_set_name1="DeepGlobe",
        data_set_name2="DeepGlobe",
        method_name="CutMix",
        model_name1="ResNET",
        model_name2="VGG",
    )
    combined_plot.show()
    # BEN CutMix for ResNET
    ben_cutmix_resnet_key = [key for key in ben_cutmix_keys if "ResNET" in key][0]
    ben_cutmix_resnet_data = csv_dict[ben_cutmix_resnet_key]
    # BEN CutMix for VGG
    ben_cutmix_vgg_key = [key for key in ben_cutmix_keys if "VGG" in key][0]
    ben_cutmix_vgg_data = csv_dict[ben_cutmix_vgg_key]
    combined_plot = create_combined_correlation_bar_plot(
        ben_cutmix_resnet_data,
        ben_cutmix_vgg_data,
        data_dir,
        data_set_name1="BEN",
        data_set_name2="BEN",
        method_name="CutMix",
        model_name1="ResNET",
        model_name2="VGG",
    )
    # combined_plot.show()


def main():
    main_dir = "/home/jonasklotz/Studys/MASTERS/results_22_4_final/new_correlations"
    csv_dir = f"{main_dir}/data"
    save_dir = f"{main_dir}/plots/summary_barplots"
    csv_dict, rrr_keys, ben_cutmix_keys, deepglobe_cutmix_keys = get_data(csv_dir)
    plot_all(ben_cutmix_keys, csv_dict, deepglobe_cutmix_keys, rrr_keys, save_dir)

    # iterate over the whole csv_dict and get the average correlation for each category
    category_correlations = defaultdict(list)

    for value in csv_dict.values():
        for category, correlation in value.items():
            category_correlations[category].append(correlation)
    # calulate the average correlation for each category
    for category, correlations in category_correlations.items():
        category_correlations[category] = average_correlation(correlations)
    title = "Average Correlation of xAI Metric Categories with Improvement in Model Performance"
    fig = create_correlation_bar_plot(category_correlations, title)
    # fig.show()
    save_path = f"{save_dir}/{title.replace(" ", "_")}_correlation_plot.pdf"
    fig.write_image(save_path)


if __name__ == "__main__":
    main()
