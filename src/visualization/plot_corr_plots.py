import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_correlation_bar_plot(
    correlation_data, data_set_name="BEN", method_name="RRR", model_name="ResNet"
):
    # Extract and sort data for the plot
    categories = list(correlation_data.keys())
    correlations = [correlation_data[cat]["Correlation"] for cat in categories]
    p_values = [correlation_data[cat]["P-value"] for cat in categories]

    # Sort data by correlation values
    sorted_data = sorted(
        zip(categories, correlations, p_values), key=lambda x: x[1], reverse=True
    )
    categories, correlations, p_values = zip(*sorted_data)

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

    # Adjusting annotations to change direction based on the correlation value
    annotations = []
    for cat, corr, p_val in zip(categories, correlations, p_values):
        # Determine the direction of the arrow based on the correlation value
        arrow_y = -40 if corr >= 0 else 40
        annotations.append(
            dict(
                x=cat,
                y=corr,
                text=f"P-value: {p_val:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=arrow_y,
            )
        )

    # Update the layout of the plot with annotations and font size
    title = f"{data_set_name}: Correlation of xAI Metric Categories with Improvement in Model Performance for {method_name}"
    if model_name:
        title += f" ({model_name})"
    fig.update_layout(
        title=title,
        xaxis_title="xAI Metric Categories",
        yaxis_title="Correlation Coefficient",
        annotations=annotations,
        font=dict(size=25),
        width=800,
        height=800,
    )
    # safe as pdf
    fig.write_image(f"{data_set_name}_{method_name}_correlation_plot.pdf")

    return fig


def create_combined_correlation_bar_plot(
    correlation_data1,
    correlation_data2,
    data_set_name1="BEN",
    data_set_name2="BEN",
    method_name="RRR",
    model_name1="ResNet",
    model_name2="VGG",
):
    # Create individual plots
    fig1 = create_correlation_bar_plot(
        correlation_data1, data_set_name1, method_name, model_name1
    )
    fig2 = create_correlation_bar_plot(
        correlation_data2, data_set_name2, method_name, model_name2
    )

    # Create subplot
    title1 = f"{model_name1} trained on {data_set_name1 } using {method_name}"
    title2 = f"{model_name2} trained on {data_set_name2 } using {method_name}"
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
    )
    # Update x-axis and y-axis titles for all subplots
    fig.update_xaxes(title_text="xAI Metric Categories", row=1, col=1)
    fig.update_yaxes(title_text="Correlation Coefficient", row=1, col=1)
    fig.update_xaxes(title_text="xAI Metric Categories", row=1, col=2)
    fig.update_yaxes(title_text="Correlation Coefficient", row=1, col=2)

    # Save the plot as a PDF
    fig.write_image(
        f"new_plots/corrplot_{method_name}_{model_name1}_{model_name2}_{data_set_name1}.pdf"
    )

    return fig


def main():
    # Caltech101 RRR vgg
    ct_rrr_correlation_data = {
        "Randomisation": {"Correlation": 0.38, "P-value": 0.45},
        "Localisation": {"Correlation": 0.26, "P-value": 0.17},
        "Faithfulness": {"Correlation": 0.18, "P-value": 0.47},
        "Robustness": {"Correlation": -0.05, "P-value": 0.85},
        "Complexity": {"Correlation": -0.28, "P-value": 0.37},
    }

    # DeepGlobe RRR vgg
    dg_rrr_correlation_data = {
        "Randomisation": {"Correlation": 0.84, "P-value": 0.04},
        "Localisation": {"Correlation": 0.19, "P-value": 0.33},
        "Robustness": {"Correlation": 0.15, "P-value": 0.54},
        "Faithfulness": {"Correlation": 0.04, "P-value": 0.83},
        "Complexity": {"Correlation": -0.78, "P-value": 0.00},
    }

    # Generate combined plot
    combined_plot = create_combined_correlation_bar_plot(
        ct_rrr_correlation_data,
        dg_rrr_correlation_data,
        data_set_name1="Caltech101",
        data_set_name2="DeepGlobe",
        method_name="RRR",
        model_name1="VGG",
        model_name2="VGG",
    )
    combined_plot.show()

    # Deepglobe CutMix VGG
    dg_cutmix_vgg_correlation_data = {
        "Randomisation": {"Correlation": 0.41, "P-value": 0.42},
        "Localisation": {"Correlation": 0.13, "P-value": 0.48},
        "Robustness": {"Correlation": -0.07, "P-value": 0.79},
        "Faithfulness": {"Correlation": 0.00, "P-value": 0.99},
        "Complexity": {"Correlation": -0.14, "P-value": 0.67},
    }

    # DeepGlobe CutMix for ResNET
    dg_cutmix_resnet_correlation_data = {
        "Localisation": {"Correlation": 0.09, "P-value": 0.61},
        "Randomisation": {"Correlation": 0.08, "P-value": 0.86},
        "Robustness": {"Correlation": -0.01, "P-value": 0.98},
        "Faithfulness": {"Correlation": -0.02, "P-value": 0.89},
        "Complexity": {"Correlation": -0.09, "P-value": 0.76},
    }
    combined_plot = create_combined_correlation_bar_plot(
        dg_cutmix_resnet_correlation_data,
        dg_cutmix_vgg_correlation_data,
        data_set_name1="DeepGlobe",
        data_set_name2="DeepGlobe",
        method_name="CutMix",
        model_name1="ResNet",
        model_name2="VGG",
    )

    combined_plot.show()

    # BEN CutMix for ResNET
    ben_cutmix_resnet_correlation_data = {
        "Randomisation": {"Correlation": 0.80, "P-value": 0.03},
        "Faithfulness": {"Correlation": 0.17, "P-value": 0.38},
        "Robustness": {"Correlation": 0.02, "P-value": 0.94},
        "Complexity": {"Correlation": -0.48, "P-value": 0.08},
    }

    # BEN CutMix for VGG
    ben_cutmix_vgg_correlation_data = {
        "Randomisation": {"Correlation": 0.25, "P-value": 0.58},
        "Faithfulness": {"Correlation": 0.13, "P-value": 0.50},
        "Robustness": {"Correlation": 0.01, "P-value": 0.97},
        "Complexity": {"Correlation": -0.20, "P-value": 0.50},
    }
    combined_plot = create_combined_correlation_bar_plot(
        ben_cutmix_resnet_correlation_data,
        ben_cutmix_vgg_correlation_data,
        data_set_name1="BEN",
        data_set_name2="BEN",
        method_name="CutMix",
        model_name1="ResNet",
        model_name2="VGG",
    )

    combined_plot.show()


if __name__ == "__main__":
    main()
